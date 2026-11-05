use axon_uic::specialist::{
    Abstention, CapsuleId, EmpiricalMetrics, FEATURES, FoundryConfig, Observation, Promotion,
    RuleForm, SpecialistCapsule, train_specialist,
};
use std::{fs::OpenOptions, io::Write as _, path::PathBuf, time::Instant};

const TRAIN: usize = 1024;
const SELECTION: usize = 512;
const AUDIT: usize = 512;
const HOLDOUT: usize = 1024;

#[derive(Clone, Copy)]
enum Scenario {
    CleanThreshold,
    NoisyThreshold,
    NoSignal,
    Xor,
    Parity3,
}

impl Scenario {
    const ALL: [Self; 5] = [
        Self::CleanThreshold,
        Self::NoisyThreshold,
        Self::NoSignal,
        Self::Xor,
        Self::Parity3,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::CleanThreshold => "clean_threshold",
            Self::NoisyThreshold => "noisy_threshold_10pct",
            Self::NoSignal => "no_signal",
            Self::Xor => "xor_compositional",
            Self::Parity3 => "parity3_unrepresentable",
        }
    }
}

#[derive(Default)]
struct Scores {
    attempts: usize,
    split_rules: usize,
    promoted: usize,
    promotions: [usize; 4],
    forms: [usize; 3],
    audit_samples: usize,
    audit_accepted: usize,
    audit_errors: usize,
    audit_brier_sum: f64,
    audit_baseline_brier_sum: f64,
    audit_ece_sum: f64,
    candidate_holdout_samples: usize,
    candidate_holdout_errors: usize,
    candidate_holdout_brier_sum: f64,
    candidate_holdout_bins: [(usize, f64, f64); 10],
    holdout_samples: usize,
    holdout_accepted: usize,
    holdout_errors: usize,
    holdout_brier_sum: f64,
    holdout_ece_sum: f64,
}

impl Scores {
    fn record_audit(&mut self, promotion: Promotion, form: RuleForm, metrics: EmpiricalMetrics) {
        self.attempts += 1;
        self.split_rules += usize::from(form != RuleForm::Constant);
        self.promotions[promotion_index(promotion)] += 1;
        self.forms[form_index(form)] += 1;
        self.audit_samples += metrics.samples;
        self.audit_accepted += metrics.accepted;
        self.audit_errors += metrics.errors;
        self.audit_brier_sum += metrics.brier * metrics.samples as f64;
        self.audit_baseline_brier_sum += metrics.baseline_brier * metrics.samples as f64;
        self.audit_ece_sum += metrics.ece * metrics.samples as f64;
    }

    fn record_holdout(&mut self, metrics: EmpiricalMetrics) {
        self.promoted += 1;
        self.holdout_samples += metrics.samples;
        self.holdout_accepted += metrics.accepted;
        self.holdout_errors += metrics.errors;
        self.holdout_brier_sum += metrics.brier * metrics.samples as f64;
        self.holdout_ece_sum += metrics.ece * metrics.samples as f64;
    }

    fn record_candidate_holdout(
        &mut self,
        rule: axon_uic::specialist::BinaryRule,
        samples: &[Observation],
    ) {
        self.candidate_holdout_samples += samples.len();
        for sample in samples {
            let probability = rule.estimate(&sample.features);
            let correct = probability.label() == sample.label;
            self.candidate_holdout_errors += usize::from(!correct);
            self.candidate_holdout_brier_sum +=
                (probability.get() - f64::from(sample.label)).powi(2);
            let bin = &mut self.candidate_holdout_bins
                [((probability.confidence() * 10.0) as usize).min(9)];
            bin.0 += 1;
            bin.1 += probability.confidence();
            bin.2 += f64::from(correct);
        }
    }

    fn json(&self, name: &str) -> String {
        let audit_coverage = ratio(self.audit_accepted, self.audit_samples);
        let audit_error = optional_ratio_json(self.audit_errors, self.audit_accepted);
        let candidate_holdout_error = ratio(
            self.candidate_holdout_errors,
            self.candidate_holdout_samples,
        );
        let holdout_coverage = ratio(self.holdout_accepted, self.holdout_samples);
        let holdout_error = optional_ratio_json(self.holdout_errors, self.holdout_accepted);
        let audit_brier = self.audit_brier_sum / self.audit_samples as f64;
        let audit_baseline_brier = self.audit_baseline_brier_sum / self.audit_samples as f64;
        let audit_ece = self.audit_ece_sum / self.audit_samples as f64;
        let candidate_holdout_brier =
            self.candidate_holdout_brier_sum / self.candidate_holdout_samples as f64;
        let candidate_holdout_ece = self
            .candidate_holdout_bins
            .iter()
            .map(|bin| (bin.1 - bin.2).abs())
            .sum::<f64>()
            / self.candidate_holdout_samples as f64;
        let holdout_brier = if self.holdout_samples == 0 {
            0.0
        } else {
            self.holdout_brier_sum / self.holdout_samples as f64
        };
        let holdout_ece = if self.holdout_samples == 0 {
            0.0
        } else {
            self.holdout_ece_sum / self.holdout_samples as f64
        };
        format!(
            "{{\"scenario\":\"{name}\",\"attempts\":{},\"split_rules\":{},\"promoted\":{},\"form_counts\":{{\"constant\":{},\"threshold\":{},\"pair_table\":{}}},\"promotion_counts\":{{\"accepted\":{},\"no_gain\":{},\"insufficient_coverage\":{},\"excessive_error\":{}}},\"audit\":{{\"samples\":{},\"accepted\":{},\"errors\":{},\"coverage\":{audit_coverage:.9},\"selective_error\":{audit_error},\"brier\":{audit_brier:.9},\"baseline_brier\":{audit_baseline_brier:.9},\"ece\":{audit_ece:.9}}},\"candidate_holdout\":{{\"samples\":{},\"errors\":{},\"error\":{candidate_holdout_error:.9},\"brier\":{candidate_holdout_brier:.9},\"ece\":{candidate_holdout_ece:.9}}},\"deployed_holdout\":{{\"samples\":{},\"accepted\":{},\"errors\":{},\"coverage\":{holdout_coverage:.9},\"selective_error\":{holdout_error},\"brier\":{holdout_brier:.9},\"ece\":{holdout_ece:.9}}}}}",
            self.attempts,
            self.split_rules,
            self.promoted,
            self.forms[0],
            self.forms[1],
            self.forms[2],
            self.promotions[0],
            self.promotions[1],
            self.promotions[2],
            self.promotions[3],
            self.audit_samples,
            self.audit_accepted,
            self.audit_errors,
            self.candidate_holdout_samples,
            self.candidate_holdout_errors,
            self.holdout_samples,
            self.holdout_accepted,
            self.holdout_errors,
        )
    }
}

#[derive(Default)]
struct RuntimeChecks {
    ood_checks: usize,
    ood_abstentions: usize,
    drift_runs: usize,
    drift_events: usize,
    drift_min: usize,
    drift_max: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("specialist lab: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut seeds = 128_usize;
    let mut base_seed = 20260922_u64;
    let mut output: Option<PathBuf> = None;
    let mut checkpoint: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "axon-uic-specialist-lab [--seeds 1..4096] [--seed N] [--output report.json] [--checkpoint new.capsule]"
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing value")?;
        match flag.as_str() {
            "--seeds" => {
                seeds = value.parse().map_err(|_| "invalid seed count")?;
                if !(1..=4096).contains(&seeds) {
                    return Err("seeds must be in 1..=4096".into());
                }
            }
            "--seed" => base_seed = value.parse().map_err(|_| "invalid base seed")?,
            "--output" => output = Some(value.into()),
            "--checkpoint" => checkpoint = Some(value.into()),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    for path in [output.as_ref(), checkpoint.as_ref()].into_iter().flatten() {
        if path.exists() {
            return Err(format!("refusing to replace {}", path.display()));
        }
    }
    if output.is_some() && output == checkpoint {
        return Err("report and checkpoint paths must differ".into());
    }
    let started = Instant::now();
    let mut scores: [Scores; 5] = std::array::from_fn(|_| Scores::default());
    let mut runtime = RuntimeChecks {
        drift_min: usize::MAX,
        ..RuntimeChecks::default()
    };
    let config = FoundryConfig {
        min_confidence: 0.75,
        ..FoundryConfig::default()
    };
    let mut checkpoint_saved = false;
    for seed_index in 0..seeds {
        let seed = base_seed.wrapping_add(seed_index as u64);
        for (scenario_index, scenario) in Scenario::ALL.into_iter().enumerate() {
            let threshold = (seed % 31) as i64 - 15;
            let train = generate(scenario, seed ^ 0x243f6a8885a308d3, 0, TRAIN, threshold);
            let selection = generate(scenario, seed ^ 0x13198a2e03707344, 1, SELECTION, threshold);
            let audit = generate(scenario, seed ^ 0xa4093822299f31d0, 2, AUDIT, threshold);
            let outcome = train_specialist(
                CapsuleId {
                    context: seed,
                    version: 1,
                },
                config,
                &train,
                &selection,
                &audit,
            )
            .map_err(|e| {
                format!(
                    "seed={seed} scenario={} training failed: {e:?}",
                    scenario.name()
                )
            })?;
            scores[scenario_index].record_audit(
                outcome.evidence.promotion,
                outcome.evidence.rule.form(),
                outcome.evidence.audit,
            );
            let holdout = generate(scenario, seed ^ 0x082efa98ec4e6c89, 3, HOLDOUT, threshold);
            scores[scenario_index].record_candidate_holdout(outcome.evidence.rule, &holdout);
            if let Some(mut capsule) = outcome.capsule {
                let metrics = capsule
                    .evaluate(&holdout)
                    .map_err(|e| format!("holdout: {e:?}"))?;
                scores[scenario_index].record_holdout(metrics);
                check_out_of_domain(&capsule, seed, &mut runtime)?;
                if matches!(scenario, Scenario::CleanThreshold) {
                    if seed_index == 0
                        && let Some(path) = &checkpoint
                    {
                        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                        }
                        capsule.save(path).map_err(|e| e.to_string())?;
                        let loaded = SpecialistCapsule::load(
                            path,
                            CapsuleId {
                                context: seed,
                                version: 1,
                            },
                        )
                        .map_err(|e| e.to_string())?;
                        if loaded != capsule {
                            return Err("checkpoint roundtrip changed capsule".into());
                        }
                        checkpoint_saved = true;
                    }
                    run_drift(&mut capsule, seed, threshold, &mut runtime)?;
                }
            }
        }
    }
    let elapsed_ms = started.elapsed().as_millis();
    let scenario_json: Vec<_> = Scenario::ALL
        .iter()
        .enumerate()
        .map(|(i, scenario)| scores[i].json(scenario.name()))
        .collect();
    let drift_mean = ratio(runtime.drift_events, runtime.drift_runs);
    let report = format!(
        "{{\"schema\":\"axon-specialist-lab-v4\",\"base_seed\":{base_seed},\"seeds\":{seeds},\"partitions\":{{\"train\":{TRAIN},\"selection\":{SELECTION},\"audit\":{AUDIT},\"frozen_holdout\":{HOLDOUT}}},\"family\":\"constant-threshold-or-two-predicate-table\",\"capsule_memory_bytes\":{},\"checkpoint_bytes\":{},\"checkpoint_saved\":{checkpoint_saved},\"checkpoint_id\":{{\"context\":{base_seed},\"version\":1}},\"config\":{{\"min_leaf\":{},\"min_audit_accepted\":{},\"max_predicates\":{},\"min_confidence\":{},\"min_coverage\":{},\"max_error\":{},\"min_brier_gain\":{},\"complexity_penalty\":{},\"drift_window\":{},\"max_window_error\":{}}},\"runtime_checks\":{{\"ood_checks\":{},\"ood_abstentions\":{},\"drift_runs\":{},\"drift_events_mean\":{drift_mean:.3},\"drift_events_min\":{},\"drift_events_max\":{}}},\"elapsed_ms\":{elapsed_ms},\"scenarios\":[{}]}}\n",
        std::mem::size_of::<SpecialistCapsule>(),
        SpecialistCapsule::checkpoint_bytes(),
        config.min_leaf,
        config.min_audit_accepted,
        config.max_predicates,
        config.min_confidence,
        config.min_coverage,
        config.max_error,
        config.min_brier_gain,
        config.complexity_penalty,
        config.drift_window,
        config.max_window_error,
        runtime.ood_checks,
        runtime.ood_abstentions,
        runtime.drift_runs,
        if runtime.drift_runs == 0 {
            0
        } else {
            runtime.drift_min
        },
        runtime.drift_max,
        scenario_json.join(",")
    );
    for (scenario, score) in Scenario::ALL.iter().zip(&scores) {
        println!(
            "{}: promoted {}/{}; candidate holdout error {:.2}%; deployed error {} at {:.2}% coverage",
            scenario.name(),
            score.promoted,
            score.attempts,
            100.0
                * ratio(
                    score.candidate_holdout_errors,
                    score.candidate_holdout_samples
                ),
            percent(score.holdout_errors, score.holdout_accepted),
            100.0 * ratio(score.holdout_accepted, score.holdout_samples),
        );
    }
    println!(
        "{} OOD abstentions in {} checks; drift suspension after {:.1} events on average; {}-byte capsule; {elapsed_ms} ms",
        runtime.ood_abstentions,
        runtime.ood_checks,
        drift_mean,
        std::mem::size_of::<SpecialistCapsule>(),
    );
    if let Some(path) = output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| e.to_string())?;
        file.write_all(report.as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|e| e.to_string())?;
        println!("report={}", path.display());
    }
    if checkpoint.is_some() && !checkpoint_saved {
        return Err("no clean specialist was available for checkpointing".into());
    }
    Ok(())
}

fn promotion_index(promotion: Promotion) -> usize {
    match promotion {
        Promotion::Accepted => 0,
        Promotion::NoGain => 1,
        Promotion::InsufficientCoverage => 2,
        Promotion::ExcessiveError => 3,
    }
}

fn form_index(form: RuleForm) -> usize {
    match form {
        RuleForm::Constant => 0,
        RuleForm::Threshold => 1,
        RuleForm::PairTable => 2,
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn optional_ratio_json(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "null".into()
    } else {
        format!("{:.9}", ratio(numerator, denominator))
    }
}

fn percent(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "n/a".into()
    } else {
        format!("{:.2}%", 100.0 * ratio(numerator, denominator))
    }
}

fn generate(
    scenario: Scenario,
    seed: u64,
    partition: u64,
    count: usize,
    threshold: i64,
) -> Vec<Observation> {
    let mut rng = Rng(seed);
    let mut rows = vec![[0_i64; FEATURES]; count];
    for feature in 0..FEATURES {
        let mut values: Vec<_> = (0..count).map(|index| (index % 128) as i64 - 64).collect();
        for index in (1..values.len()).rev() {
            let other = (rng.next() % (index as u64 + 1)) as usize;
            values.swap(index, other);
        }
        for (row, value) in rows.iter_mut().zip(values) {
            row[feature] = value;
        }
    }
    rows.into_iter()
        .enumerate()
        .map(|(index, features)| {
            let mut label = match scenario {
                Scenario::CleanThreshold | Scenario::NoisyThreshold => features[2] > threshold,
                Scenario::NoSignal => rng.next() & 1 == 1,
                Scenario::Xor => (features[0] >= 0) ^ (features[1] >= 0),
                Scenario::Parity3 => (features[0] >= 0) ^ (features[1] >= 0) ^ (features[2] >= 0),
            };
            if matches!(scenario, Scenario::NoisyThreshold) && rng.next().is_multiple_of(10) {
                label = !label;
            }
            Observation {
                id: partition * 1_000_000 + index as u64,
                features,
                label,
            }
        })
        .collect()
}

fn check_out_of_domain(
    capsule: &SpecialistCapsule,
    context: u64,
    checks: &mut RuntimeChecks,
) -> Result<(), String> {
    let domain = capsule.domain();
    for feature in 0..FEATURES {
        for high in [false, true] {
            let mut values = std::array::from_fn(|f| domain[f].0);
            values[feature] = if high {
                domain[feature]
                    .1
                    .checked_add(1)
                    .ok_or("domain high overflow")?
            } else {
                domain[feature]
                    .0
                    .checked_sub(1)
                    .ok_or("domain low overflow")?
            };
            checks.ood_checks += 1;
            checks.ood_abstentions +=
                usize::from(capsule.predict(context, &values) == Err(Abstention::OutsideDomain));
        }
    }
    Ok(())
}

fn run_drift(
    capsule: &mut SpecialistCapsule,
    context: u64,
    threshold: i64,
    checks: &mut RuntimeChecks,
) -> Result<(), String> {
    let mut event = 0_u64;
    while !capsule.suspended() && event < 1024 {
        event += 1;
        let x = (event as i64 * 29).rem_euclid(128) - 64;
        let features = [x, -x.saturating_add(1), x, x.rem_euclid(7)];
        capsule
            .observe_labeled(event, context, &features, !(x > threshold))
            .map_err(|e| format!("drift feedback failed: {e:?}"))?;
    }
    if !capsule.suspended() {
        return Err("drift monitor did not suspend".into());
    }
    let events = event as usize;
    checks.drift_runs += 1;
    checks.drift_events += events;
    checks.drift_min = checks.drift_min.min(events);
    checks.drift_max = checks.drift_max.max(events);
    Ok(())
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}
