use std::{
    collections::BTreeSet, env, fs::OpenOptions, hint::black_box, io::Write, mem::size_of,
    path::PathBuf, time::Instant,
};

use axon_uic::{
    DeltaForge, DerivedArtifact, DerivedVariancePlan, ExactPopulationVariance, FoldSpec, Replace,
    ReplaceDelta, VarianceCache,
};

const DEFAULT_LENGTH: usize = 1_048_576;
const DEFAULT_STEPS: usize = 256;
const DEFAULT_UPDATES: usize = 32;
const DEFAULT_SEED: u64 = 20_261_201;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Options {
    length: usize,
    steps: usize,
    updates: usize,
    seed: u64,
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    LowMagnitude,
    HighOffsetTightSpread,
    BroadU32,
}

impl Scenario {
    const ALL: [Self; 3] = [
        Self::LowMagnitude,
        Self::HighOffsetTightSpread,
        Self::BroadU32,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::LowMagnitude => "low_magnitude",
            Self::HighOffsetTightSpread => "high_offset_tight_spread",
            Self::BroadU32 => "broad_u32",
        }
    }

    const fn value(self, random: u64) -> u64 {
        match self {
            Self::LowMagnitude => random & 0x3FF,
            Self::HighOffsetTightSpread => 1_000_000_000 + (random & 0x3FF),
            Self::BroadU32 => random & u32::MAX as u64,
        }
    }
}

#[derive(Debug)]
struct ScenarioResult {
    scenario: Scenario,
    initialization_ns: u64,
    full_ns: Vec<u64>,
    candidate_ns: Vec<u64>,
    candidate_faster: usize,
    full_faster: usize,
    ties: usize,
    replacements: usize,
    final_variance: ExactPopulationVariance,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let options = parse_options(env::args().skip(1))?;
    let synthesis_started = Instant::now();
    let DerivedArtifact::PopulationVariance(plan) =
        DeltaForge::synthesize_capability(FoldSpec::PopulationVarianceExactU64)
            .map_err(|error| error.to_string())?
    else {
        return Err("population variance capability derived a different artifact".to_owned());
    };
    let synthesis_ns = nanos(synthesis_started);

    let mut results = Vec::with_capacity(Scenario::ALL.len());
    for (index, scenario) in Scenario::ALL.into_iter().enumerate() {
        results.push(run_scenario(
            plan,
            scenario,
            &options,
            options.seed ^ (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        )?);
    }

    println!(
        "DeltaForge exact population variance: {} values, {} events, {} replacements/event, {}-byte cache",
        options.length,
        options.steps,
        options.updates,
        size_of::<VarianceCache>()
    );
    for result in &results {
        let full_total = sum_ns(&result.full_ns);
        let candidate_total = sum_ns(&result.candidate_ns);
        let installed_total = candidate_total + u128::from(result.initialization_ns);
        println!(
            "{}: full {:.3} ms, moments {:.3} ms (+ {:.3} ms init), steady {:.3}x, installed {:.3}x, exact {}/{}",
            result.scenario.name(),
            full_total as f64 / 1_000_000.0,
            candidate_total as f64 / 1_000_000.0,
            result.initialization_ns as f64 / 1_000_000.0,
            full_total as f64 / candidate_total.max(1) as f64,
            full_total as f64 / installed_total.max(1) as f64,
            result.final_variance.numerator(),
            result.final_variance.denominator(),
        );
        println!(
            "  paired events: moments faster {}, full faster {}, ties {}",
            result.candidate_faster, result.full_faster, result.ties
        );
    }

    if let Some(path) = &options.output {
        let report = report_json(&options, synthesis_ns, &results);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("cannot create `{}`: {error}", path.display()))?;
        file.write_all(report.as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("cannot persist `{}`: {error}", path.display()))?;
        println!("report: {}", path.display());
    }
    Ok(())
}

fn parse_options(arguments: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        length: DEFAULT_LENGTH,
        steps: DEFAULT_STEPS,
        updates: DEFAULT_UPDATES,
        seed: DEFAULT_SEED,
        output: None,
    };
    let mut arguments = arguments;
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--length" => options.length = bounded(&value, 1, 16_777_216, &flag)?,
            "--steps" => options.steps = bounded(&value, 1, 1_000_000, &flag)?,
            "--updates" => options.updates = bounded(&value, 1, 1_000_000, &flag)?,
            "--seed" => {
                options.seed = value
                    .parse()
                    .map_err(|_| format!("invalid value for {flag}: {value}"))?;
            }
            "--output" => options.output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    if options.updates > options.length {
        return Err("--updates must not exceed --length".to_owned());
    }
    Ok(options)
}

fn bounded(value: &str, minimum: usize, maximum: usize, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|value| (minimum..=maximum).contains(value))
        .ok_or_else(|| format!("{flag} must be between {minimum} and {maximum}"))
}

fn run_scenario(
    plan: DerivedVariancePlan,
    scenario: Scenario,
    options: &Options,
    seed: u64,
) -> Result<ScenarioResult, String> {
    let mut rng = Lcg(seed);
    let mut full_values = Vec::new();
    full_values
        .try_reserve_exact(options.length)
        .map_err(|error| format!("cannot allocate benchmark source: {error}"))?;
    full_values.extend((0..options.length).map(|_| scenario.value(rng.next())));
    let mut candidate_values = full_values.clone();
    let initialization_started = Instant::now();
    let (_, mut cache) = plan
        .full(&candidate_values)
        .map_err(|error| format!("initial variance: {error}"))?;
    let initialization_ns = nanos(initialization_started);
    let mut full_ns = Vec::with_capacity(options.steps);
    let mut candidate_ns = Vec::with_capacity(options.steps);
    let mut candidate_faster = 0;
    let mut full_faster = 0;
    let mut ties = 0;

    for step in 0..options.steps {
        let change = generate_change(&full_values, scenario, options.updates, &mut rng)?;
        let (full_elapsed, full_variance, candidate_elapsed, candidate_variance, next_cache) =
            if step % 2 == 0 {
                let candidate = execute_candidate(plan, &mut candidate_values, &change, cache)?;
                let full = execute_full(plan, &mut full_values, &change)?;
                (full.0, full.1, candidate.0, candidate.1, candidate.2)
            } else {
                let full = execute_full(plan, &mut full_values, &change)?;
                let candidate = execute_candidate(plan, &mut candidate_values, &change, cache)?;
                (full.0, full.1, candidate.0, candidate.1, candidate.2)
            };
        cache = next_cache;
        if full_variance != candidate_variance {
            return Err(format!(
                "variance divergence in {} at event {step}",
                scenario.name()
            ));
        }
        match candidate_elapsed.cmp(&full_elapsed) {
            std::cmp::Ordering::Less => candidate_faster += 1,
            std::cmp::Ordering::Greater => full_faster += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }
        full_ns.push(full_elapsed);
        candidate_ns.push(candidate_elapsed);
        black_box((full_variance, candidate_variance));
    }
    if full_values != candidate_values {
        return Err(format!("source divergence in {}", scenario.name()));
    }
    let (final_variance, final_cache) = plan
        .full(&full_values)
        .map_err(|error| format!("final variance: {error}"))?;
    if cache != final_cache {
        return Err(format!("final cache divergence in {}", scenario.name()));
    }
    Ok(ScenarioResult {
        scenario,
        initialization_ns,
        full_ns,
        candidate_ns,
        candidate_faster,
        full_faster,
        ties,
        replacements: options.steps * options.updates,
        final_variance,
    })
}

fn generate_change(
    values: &[u64],
    scenario: Scenario,
    updates: usize,
    rng: &mut Lcg,
) -> Result<ReplaceDelta, String> {
    let mut indices = BTreeSet::new();
    while indices.len() < updates {
        indices.insert(rng.next() as usize % values.len());
    }
    let changes = indices
        .into_iter()
        .map(|index| Replace::new(index, values[index], scenario.value(rng.next())))
        .collect();
    ReplaceDelta::try_new(changes).map_err(|error| format!("invalid generated change: {error:?}"))
}

fn execute_full(
    plan: DerivedVariancePlan,
    values: &mut [u64],
    change: &ReplaceDelta,
) -> Result<(u64, ExactPopulationVariance), String> {
    let started = Instant::now();
    apply_in_place(values, change)?;
    let variance = plan
        .full(values)
        .map_err(|error| format!("full variance: {error}"))?
        .0;
    let elapsed = nanos(started);
    Ok((elapsed, black_box(variance)))
}

fn execute_candidate(
    plan: DerivedVariancePlan,
    values: &mut [u64],
    change: &ReplaceDelta,
    cache: VarianceCache,
) -> Result<(u64, ExactPopulationVariance, VarianceCache), String> {
    let started = Instant::now();
    let (variance, next_cache) = plan
        .delta(change, &cache)
        .map_err(|error| format!("incremental variance: {error}"))?;
    apply_in_place(values, change)?;
    let elapsed = nanos(started);
    Ok((elapsed, black_box(variance), next_cache))
}

fn apply_in_place(values: &mut [u64], change: &ReplaceDelta) -> Result<(), String> {
    for replacement in change.changes() {
        let current = values
            .get(replacement.index())
            .ok_or_else(|| format!("index {} is out of bounds", replacement.index()))?;
        if *current != replacement.old() {
            return Err(format!("stale old value at index {}", replacement.index()));
        }
    }
    for replacement in change.changes() {
        values[replacement.index()] = replacement.new_value();
    }
    Ok(())
}

fn nanos(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn sum_ns(samples: &[u64]) -> u128 {
    samples.iter().map(|&sample| u128::from(sample)).sum()
}

fn percentile(samples: &[u64], numerator: usize, denominator: usize) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() - 1) * numerator / denominator;
    sorted[rank]
}

fn report_json(options: &Options, synthesis_ns: u64, results: &[ScenarioResult]) -> String {
    let scenarios = results
        .iter()
        .map(|result| {
            let full_total = sum_ns(&result.full_ns);
            let candidate_total = sum_ns(&result.candidate_ns);
            let candidate_installed = candidate_total + u128::from(result.initialization_ns);
            format!(
                "{{\"scenario\":\"{}\",\"events\":{},\"replacements\":{},\"initialization_ns\":{},\"full_total_ns\":{},\"candidate_steady_ns\":{},\"candidate_with_initialization_ns\":{},\"full_p50_ns\":{},\"full_p95_ns\":{},\"candidate_p50_ns\":{},\"candidate_p95_ns\":{},\"full_over_candidate_steady\":{:.9},\"full_over_candidate_with_initialization\":{:.9},\"candidate_faster_events\":{},\"full_faster_events\":{},\"tied_events\":{},\"full_samples_ns\":{},\"candidate_samples_ns\":{},\"final_numerator\":\"{}\",\"final_denominator\":\"{}\",\"exact\":true}}",
                result.scenario.name(),
                options.steps,
                result.replacements,
                result.initialization_ns,
                full_total,
                candidate_total,
                candidate_installed,
                percentile(&result.full_ns, 50, 100),
                percentile(&result.full_ns, 95, 100),
                percentile(&result.candidate_ns, 50, 100),
                percentile(&result.candidate_ns, 95, 100),
                full_total as f64 / candidate_total.max(1) as f64,
                full_total as f64 / candidate_installed.max(1) as f64,
                result.candidate_faster,
                result.full_faster,
                result.ties,
                samples_json(&result.full_ns),
                samples_json(&result.candidate_ns),
                result.final_variance.numerator(),
                result.final_variance.denominator(),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"schema\":\"axon-deltaforge-population-variance-v1\",\"length\":{},\"source_bytes\":{},\"steps\":{},\"updates_per_step\":{},\"seed\":{},\"synthesis_ns\":{},\"cache_bytes\":{},\"logical_words\":3,\"arithmetic\":\"checked_u128_exact_or_refuse\",\"scenarios\":[{}]}}\n",
        options.length,
        options.length * size_of::<u64>(),
        options.steps,
        options.updates,
        options.seed,
        synthesis_ns,
        size_of::<VarianceCache>(),
        scenarios,
    )
}

fn samples_json(samples: &[u64]) -> String {
    let body = samples
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_rejects_more_updates_than_values() {
        assert!(
            parse_options(
                ["--length", "64", "--updates", "65"]
                    .map(str::to_owned)
                    .into_iter()
            )
            .is_err()
        );
    }

    #[test]
    fn every_distribution_keeps_exact_paired_results() {
        let options = Options {
            length: 256,
            steps: 32,
            updates: 8,
            seed: 7,
            output: None,
        };
        let DerivedArtifact::PopulationVariance(plan) =
            DeltaForge::synthesize_capability(FoldSpec::PopulationVarianceExactU64).unwrap()
        else {
            panic!("variance spec must derive a variance artifact");
        };

        for (index, scenario) in Scenario::ALL.into_iter().enumerate() {
            let result = run_scenario(plan, scenario, &options, index as u64 + 1).unwrap();
            assert_eq!(result.full_ns.len(), options.steps);
            assert_eq!(result.candidate_ns.len(), options.steps);
            assert_eq!(
                result.candidate_faster + result.full_faster + result.ties,
                options.steps
            );
            let report = report_json(&options, 0, &[result]);
            assert!(report.contains("\"exact\":true"));
            assert!(report.contains("\"full_samples_ns\":["));
        }
    }
}
