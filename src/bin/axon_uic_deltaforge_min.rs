use std::{
    collections::BTreeSet, env, fs::OpenOptions, hint::black_box, io::Write, mem::size_of,
    path::PathBuf, time::Instant,
};

use axon_uic::{
    DeltaForge, DerivedArtifact, DerivedMinimumPlan, FoldSpec, MinimumCache, MinimumUpdate,
    Replace, ReplaceDelta,
};

const DEFAULT_LENGTH: usize = 1_048_576;
const DEFAULT_STEPS: usize = 1_024;
const DEFAULT_UPDATES: usize = 32;
const DEFAULT_SEED: u64 = 20_260_922;

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
    Stable,
    Mixed,
    Churn,
}

impl Scenario {
    const ALL: [Self; 3] = [Self::Stable, Self::Mixed, Self::Churn];

    const fn name(self) -> &'static str {
        match self {
            Self::Stable => "stable_witness",
            Self::Mixed => "mixed_witness",
            Self::Churn => "exhausted_every_step",
        }
    }

    const fn exhausted_witness(self, step: usize) -> Option<usize> {
        match self {
            Self::Stable => None,
            Self::Mixed if (step + 1).is_multiple_of(64) => Some(step / 64),
            Self::Mixed => None,
            Self::Churn => Some(step),
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
    rebuilds: usize,
    updates: usize,
    final_minimum: u64,
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
    let DerivedArtifact::Minimum(plan) =
        DeltaForge::synthesize_capability(FoldSpec::MinU64).map_err(|error| error.to_string())?
    else {
        return Err("MIN capability derived a different artifact".to_owned());
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
        "DeltaForge MIN: {} values, {} events, {} replacements/event, {}-byte witness",
        options.length,
        options.steps,
        options.updates,
        size_of::<MinimumCache>()
    );
    for result in &results {
        let full_total = sum_ns(&result.full_ns);
        let candidate_total = sum_ns(&result.candidate_ns);
        let installed_total = candidate_total + u128::from(result.initialization_ns);
        println!(
            "{}: rebuilds {}/{}, full {:.3} ms, witness {:.3} ms (+ {:.3} ms init), steady {:.3}x, installed {:.3}x, exact min {}",
            result.scenario.name(),
            result.rebuilds,
            options.steps,
            full_total as f64 / 1_000_000.0,
            candidate_total as f64 / 1_000_000.0,
            result.initialization_ns as f64 / 1_000_000.0,
            full_total as f64 / candidate_total.max(1) as f64,
            full_total as f64 / installed_total.max(1) as f64,
            result.final_minimum
        );
        println!(
            "  paired events: witness faster {}, full faster {}, ties {}",
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
            "--length" => options.length = bounded(&value, 64, 16_777_216, &flag)?,
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
    if options.steps > options.length / 2 {
        return Err("--steps must not exceed half of --length".to_owned());
    }
    let tail = options.length - options.length / 2;
    if options.updates > tail {
        return Err("--updates exceeds the stable tail available per event".to_owned());
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
    plan: DerivedMinimumPlan,
    scenario: Scenario,
    options: &Options,
    seed: u64,
) -> Result<ScenarioResult, String> {
    let mut full_values: Vec<_> = (0..options.length as u64).collect();
    let mut candidate_values = full_values.clone();
    let initialization_started = Instant::now();
    let (_, mut cache) = plan
        .full(&candidate_values)
        .map_err(|error| error.to_string())?;
    let initialization_ns = nanos(initialization_started);
    let mut full_ns = Vec::with_capacity(options.steps);
    let mut candidate_ns = Vec::with_capacity(options.steps);
    let mut rebuilds = 0;
    let mut candidate_faster = 0;
    let mut full_faster = 0;
    let mut ties = 0;
    let mut rng = Lcg(seed);

    for step in 0..options.steps {
        let change = generate_change(&full_values, scenario, step, options, &mut rng)?;
        let (full_elapsed, full_minimum, candidate_elapsed, candidate_minimum, rebuilt) =
            if step % 2 == 0 {
                let (candidate_elapsed, candidate_minimum, rebuilt, next_cache) =
                    execute_candidate(plan, &mut candidate_values, &change, cache)?;
                cache = next_cache;
                let (full_elapsed, full_minimum) = execute_full(&mut full_values, &change)?;
                (
                    full_elapsed,
                    full_minimum,
                    candidate_elapsed,
                    candidate_minimum,
                    rebuilt,
                )
            } else {
                let (full_elapsed, full_minimum) = execute_full(&mut full_values, &change)?;
                let (candidate_elapsed, candidate_minimum, rebuilt, next_cache) =
                    execute_candidate(plan, &mut candidate_values, &change, cache)?;
                cache = next_cache;
                (
                    full_elapsed,
                    full_minimum,
                    candidate_elapsed,
                    candidate_minimum,
                    rebuilt,
                )
            };
        if full_minimum != candidate_minimum {
            return Err(format!(
                "minimum divergence in {} at event {step}",
                scenario.name()
            ));
        }
        rebuilds += usize::from(rebuilt);
        match candidate_elapsed.cmp(&full_elapsed) {
            std::cmp::Ordering::Less => candidate_faster += 1,
            std::cmp::Ordering::Greater => full_faster += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }
        full_ns.push(full_elapsed);
        candidate_ns.push(candidate_elapsed);
        black_box((full_minimum, candidate_minimum));
    }
    if full_values != candidate_values {
        return Err(format!("source divergence in {}", scenario.name()));
    }
    let final_minimum = *full_values
        .iter()
        .min()
        .ok_or_else(|| "empty benchmark source".to_owned())?;
    Ok(ScenarioResult {
        scenario,
        initialization_ns,
        full_ns,
        candidate_ns,
        candidate_faster,
        full_faster,
        ties,
        rebuilds,
        updates: options.steps * options.updates,
        final_minimum,
    })
}

fn generate_change(
    values: &[u64],
    scenario: Scenario,
    step: usize,
    options: &Options,
    rng: &mut Lcg,
) -> Result<ReplaceDelta, String> {
    let mut indices = BTreeSet::new();
    if let Some(witness) = scenario.exhausted_witness(step) {
        indices.insert(witness);
    }
    let tail_start = options.length / 2;
    while indices.len() < options.updates {
        indices.insert(tail_start + rng.next() as usize % (options.length - tail_start));
    }
    let changes = indices
        .into_iter()
        .enumerate()
        .map(|(ordinal, index)| {
            let new_value = options.length as u64
                + (step * options.updates + ordinal) as u64
                + 1
                + (rng.next() & 0xFFFF);
            Replace::new(index, values[index], new_value)
        })
        .collect();
    ReplaceDelta::try_new(changes).map_err(|error| format!("invalid generated change: {error:?}"))
}

fn execute_full(values: &mut [u64], change: &ReplaceDelta) -> Result<(u64, u64), String> {
    let started = Instant::now();
    apply_in_place(values, change)?;
    let minimum = values
        .iter()
        .copied()
        .min()
        .ok_or_else(|| "empty full source".to_owned())?;
    let elapsed = nanos(started);
    Ok((elapsed, black_box(minimum)))
}

fn execute_candidate(
    plan: DerivedMinimumPlan,
    values: &mut [u64],
    change: &ReplaceDelta,
    cache: MinimumCache,
) -> Result<(u64, u64, bool, MinimumCache), String> {
    let started = Instant::now();
    let update = plan
        .delta(change, &cache)
        .map_err(|error| error.to_string())?;
    apply_in_place(values, change)?;
    let (minimum, next_cache, rebuilt) = match update {
        MinimumUpdate::Incremental { minimum, cache } => (minimum, cache, false),
        MinimumUpdate::RebuildRequired => {
            let (minimum, cache) = plan.full(values).map_err(|error| error.to_string())?;
            (minimum, cache, true)
        }
    };
    let elapsed = nanos(started);
    Ok((elapsed, black_box(minimum), rebuilt, next_cache))
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
                "{{\"scenario\":\"{}\",\"events\":{},\"replacements\":{},\"rebuilds\":{},\"rebuild_rate\":{:.9},\"source_scans_avoided\":{},\"initialization_ns\":{},\"full_total_ns\":{},\"candidate_steady_ns\":{},\"candidate_with_initialization_ns\":{},\"full_p50_ns\":{},\"full_p95_ns\":{},\"candidate_p50_ns\":{},\"candidate_p95_ns\":{},\"full_over_candidate_steady\":{:.9},\"full_over_candidate_with_initialization\":{:.9},\"candidate_faster_events\":{},\"full_faster_events\":{},\"tied_events\":{},\"full_samples_ns\":{},\"candidate_samples_ns\":{},\"final_minimum\":{},\"exact\":true}}",
                result.scenario.name(),
                options.steps,
                result.updates,
                result.rebuilds,
                result.rebuilds as f64 / options.steps as f64,
                options.steps - result.rebuilds,
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
                result.final_minimum,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"schema\":\"axon-deltaforge-min-v3\",\"length\":{},\"source_bytes\":{},\"steps\":{},\"updates_per_step\":{},\"seed\":{},\"synthesis_ns\":{},\"witness_bytes\":{},\"witness_words\":3,\"fallback\":\"exact_full_rebuild_when_last_minimum_witness_is_removed\",\"scenarios\":[{}]}}\n",
        options.length,
        options.length * size_of::<u64>(),
        options.steps,
        options.updates,
        options.seed,
        synthesis_ns,
        size_of::<MinimumCache>(),
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
    fn parser_rejects_shapes_that_cannot_preserve_the_churn_fixture() {
        assert!(
            parse_options(
                ["--length", "64", "--steps", "33"]
                    .map(str::to_owned)
                    .into_iter()
            )
            .is_err()
        );
        assert!(
            parse_options(
                ["--length", "64", "--updates", "33"]
                    .map(str::to_owned)
                    .into_iter()
            )
            .is_err()
        );
    }

    #[test]
    fn all_regimes_keep_exact_results_and_expose_the_expected_fallbacks() {
        let options = Options {
            length: 256,
            steps: 128,
            updates: 8,
            seed: 7,
            output: None,
        };
        let DerivedArtifact::Minimum(plan) =
            DeltaForge::synthesize_capability(FoldSpec::MinU64).unwrap()
        else {
            panic!("minimum spec must derive a minimum artifact");
        };
        let stable = run_scenario(plan, Scenario::Stable, &options, 1).unwrap();
        let mixed = run_scenario(plan, Scenario::Mixed, &options, 2).unwrap();
        let churn = run_scenario(plan, Scenario::Churn, &options, 3).unwrap();

        assert_eq!(stable.rebuilds, 0);
        assert_eq!(mixed.rebuilds, 2);
        assert_eq!(churn.rebuilds, options.steps);
        for result in [stable, mixed, churn] {
            assert_eq!(
                result.candidate_faster + result.full_faster + result.ties,
                options.steps
            );
        }
    }
}
