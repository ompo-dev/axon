use std::env;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axon_uic::{
    ArtifactLifetime, ArtifactStatus, ArtifactStore, AxonTask, BenchContract, BenchPhase,
    BreakEven, ChangeStructure, DerivedArtifact, DerivedAveragePlan, ExactAverage, Replace,
    ReplaceDelta, VectorU64,
};

const DEFAULT_MIB: usize = 64;
const DEFAULT_RUNS: usize = 5;
const MAX_MIB: usize = 256;
const MAX_RUNS: usize = 30;
const POINTS: [usize; 3] = [1_024, 1_000_000, 4_000_000];

#[derive(Clone, Copy)]
struct Config {
    mib: usize,
    runs: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mib: DEFAULT_MIB,
            runs: DEFAULT_RUNS,
        }
    }
}

#[derive(Clone, Copy)]
struct Measurement {
    contract: BenchContract,
    average: ExactAverage,
}

#[derive(Clone, Copy)]
struct Summary {
    hot: Duration,
    lifecycle: Duration,
    phases: [Duration; 12],
    representative: BenchContract,
}

#[derive(Clone, Copy)]
struct PointResult {
    updates: usize,
    full: Summary,
    reuse: Summary,
    creation: Summary,
    break_even: BreakEven,
    checksum: ExactAverage,
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(message) => {
            eprintln!("deltaforge AVG sweep failed: {message}");
            std::process::exit(2);
        }
    }
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
    let mut arguments = env::args().skip(1);
    while let Some(flag) = arguments.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            return Err("usage: axon-uic-deltaforge-avg [--mib 1..256] [--runs 1..30]".to_owned());
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--mib" => config.mib = parse_bounded(&value, 1, MAX_MIB, "--mib")?,
            "--runs" => config.runs = parse_bounded(&value, 1, MAX_RUNS, "--runs")?,
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    Ok(config)
}

fn parse_bounded(value: &str, minimum: usize, maximum: usize, flag: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|_| format!("invalid integer for {flag}: {value}"))?;
    (minimum..=maximum)
        .contains(&value)
        .then_some(value)
        .ok_or_else(|| format!("{flag} must be between {minimum} and {maximum}"))
}

fn run(config: Config) -> Result<(), String> {
    let words = config
        .mib
        .checked_mul(1024 * 1024 / size_of::<u64>())
        .ok_or_else(|| "vector size overflow".to_owned())?;
    let seed = build_data(words)?;
    let points: Vec<_> = POINTS.into_iter().filter(|point| *point <= words).collect();
    if points.is_empty() {
        return Err("vector too small for every AVG point".to_owned());
    }

    println!("# AXON-UIC DeltaForge-AVG physical sweep");
    println!(
        "host {} {}, {} MiB vector, {} paired runs per point",
        env::consts::OS,
        env::consts::ARCH,
        config.mib,
        config.runs
    );
    println!(
        "input FoldSpec::AverageExactU64; derived cache is exact (sum: u128, count: usize). HOT is execution only; LIFECYCLE includes input generation, allocation, initialization, validation and certificate check."
    );

    let mut results = Vec::with_capacity(points.len());
    for updates in points {
        let result = measure_point(&seed, updates, config.runs)?;
        println!(
            "point {updates:>7}: full HOT/LIFECYCLE {:>8.3}/{:>8.3} ms; reuse {:>8.3}/{:>8.3} ms; creation p50 {:>8.6} ms; break-even {:?}",
            milliseconds(result.full.hot),
            milliseconds(result.full.lifecycle),
            milliseconds(result.reuse.hot),
            milliseconds(result.reuse.lifecycle),
            milliseconds(result.creation.lifecycle),
            result.break_even,
        );
        results.push(result);
    }

    println!(
        "\n| Final writes | Full HOT p50 ms | Full LIFECYCLE p50 ms | Reuse HOT p50 ms | Reuse LIFECYCLE p50 ms | Artifact creation p50 ms | Measured break-even |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---|");
    for result in &results {
        println!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.6} | {:?} |",
            result.updates,
            milliseconds(result.full.hot),
            milliseconds(result.full.lifecycle),
            milliseconds(result.reuse.hot),
            milliseconds(result.reuse.lifecycle),
            milliseconds(result.creation.lifecycle),
            result.break_even,
        );
    }
    let largest = results.last().expect("at least one point");
    println!("\nLargest-point phase p50 ms:");
    println!("| Phase | Full | Reused artifact | Artifact creation |");
    println!("|---|---:|---:|---:|");
    for (index, phase) in BenchPhase::ALL.iter().enumerate() {
        println!(
            "| {} | {:.6} | {:.6} | {:.6} |",
            phase.as_str(),
            milliseconds(largest.full.phases[index]),
            milliseconds(largest.reuse.phases[index]),
            milliseconds(largest.creation.phases[index]),
        );
    }
    println!(
        "all paired rounds exact parity true; largest exact average {}/{}.",
        largest.checksum.numerator(),
        largest.checksum.denominator()
    );
    println!(
        "limit: this measures a declared exact AVG artifact and its certificate check. It does not establish general program synthesis, learned discovery, or an optimization promotion."
    );
    Ok(())
}

fn measure_point(seed: &[u64], updates: usize, runs: usize) -> Result<PointResult, String> {
    let mut full_samples = Vec::with_capacity(runs);
    let mut reuse_samples = Vec::with_capacity(runs);
    let mut creation_samples = Vec::with_capacity(runs);
    let mut checksum = None;

    for round in 0..runs {
        let root = create_temporary_artifact_root(updates, round)?;
        let store = ArtifactStore::open(&root);
        let task = task_for_artifact(updates, round)?;
        let (created_plan, creation) = install_average(&store, &task, ArtifactStatus::Created)?;
        let (plan, artifact_load) = install_average(&store, &task, ArtifactStatus::Reused)?;
        let (first, second) = if round % 2 == 0 {
            (
                run_full(seed, updates, &created_plan)?,
                run_reuse(seed, updates, &plan, artifact_load)?,
            )
        } else {
            (
                run_reuse(seed, updates, &plan, artifact_load)?,
                run_full(seed, updates, &created_plan)?,
            )
        };
        let (full, reuse) = if round % 2 == 0 {
            (first, second)
        } else {
            (second, first)
        };
        if full.average != reuse.average {
            return Err(format!(
                "exact AVG parity failure at {updates} writes, round {}",
                round + 1
            ));
        }
        let full_lifecycle = full.contract.lifecycle().map_err(contract_error)?;
        let reuse_lifecycle = reuse.contract.lifecycle().map_err(contract_error)?;
        println!(
            "  run {:02}: full HOT/LIFECYCLE {:>8.3}/{:>8.3} ms; reuse {:>8.3}/{:>8.3} ms; parity true",
            round + 1,
            milliseconds(full.contract.hot()),
            milliseconds(full_lifecycle),
            milliseconds(reuse.contract.hot()),
            milliseconds(reuse_lifecycle),
        );
        checksum = Some(full.average);
        full_samples.push(full);
        reuse_samples.push(reuse);
        creation_samples.push(Measurement {
            contract: creation,
            average: full.average,
        });
        std::fs::remove_dir_all(&root)
            .map_err(|error| format!("cannot remove temporary artifact root: {error}"))?;
    }

    let full = summarize(&full_samples)?;
    let reuse = summarize(&reuse_samples)?;
    let creation = summarize(&creation_samples)?;
    let baseline = ArtifactLifetime::try_new(
        BenchContract::empty(),
        full_samples.iter().map(|sample| sample.contract).collect(),
        BenchContract::empty(),
    )
    .map_err(|error| format!("full lifetime: {error:?}"))?;
    let candidate = ArtifactLifetime::try_new(
        creation.representative,
        reuse_samples.iter().map(|sample| sample.contract).collect(),
        BenchContract::empty(),
    )
    .map_err(|error| format!("artifact lifetime: {error:?}"))?;
    let break_even = candidate
        .first_break_even_against(&baseline, runs)
        .map_err(|error| format!("break-even: {error:?}"))?;

    Ok(PointResult {
        updates,
        full,
        reuse,
        creation,
        break_even,
        checksum: checksum.expect("runs is nonzero"),
    })
}

fn install_average(
    store: &ArtifactStore,
    task: &AxonTask,
    expected_status: ArtifactStatus,
) -> Result<(DerivedAveragePlan, BenchContract), String> {
    let started = Instant::now();
    let installed = black_box(
        store
            .install(task)
            .map_err(|error| format!("artifact store: {error}"))?,
    );
    if installed.status() != expected_status {
        return Err(format!(
            "expected artifact status {}, got {}",
            expected_status.as_str(),
            installed.status().as_str()
        ));
    }
    let plan = match installed.artifact() {
        DerivedArtifact::Average(plan) => plan,
        DerivedArtifact::Sum(_) => return Err("AVG capability derived a SUM artifact".to_owned()),
    };
    let phase = match expected_status {
        ArtifactStatus::Created => BenchPhase::ArtifactPersist,
        ArtifactStatus::Reused => BenchPhase::ArtifactLoad,
    };
    Ok((
        plan,
        BenchContract::empty().with_phase(phase, started.elapsed()),
    ))
}

fn task_for_artifact(updates: usize, round: usize) -> Result<AxonTask, String> {
    AxonTask::parse(&format!(
        "task avg_{updates}_{round} {{\n  data numbers: Vec<u64> = [1]\n  goal derive IncrementalArtifact<AverageExactU64>\n}}\n"
    ))
    .map_err(|error| format!("benchmark task: {error}"))
}

fn create_temporary_artifact_root(
    updates: usize,
    round: usize,
) -> Result<std::path::PathBuf, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock before epoch: {error}"))?
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "axon-uic-avg-{}-{updates}-{round}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&root)
        .map_err(|error| format!("cannot create temporary artifact root: {error}"))?;
    Ok(root)
}

fn run_full(
    seed: &[u64],
    updates: usize,
    plan: &DerivedAveragePlan,
) -> Result<Measurement, String> {
    let (changes, mut contract) = generate_changes(seed, updates)?;
    let (before, setup) = instantiate_state(seed)?;
    contract = add_setup(contract, setup)?;
    let started = Instant::now();
    let next = VectorU64
        .apply(&before, &changes)
        .map_err(|error| format!("full apply: {error:?}"))?;
    let (average, _) = plan
        .full(&next)
        .map_err(|error| format!("full AVG: {error}"))?;
    black_box(average);
    contract = contract.with_phase(BenchPhase::Execution, started.elapsed());
    Ok(Measurement { contract, average })
}

fn run_reuse(
    seed: &[u64],
    updates: usize,
    plan: &DerivedAveragePlan,
    artifact_load: BenchContract,
) -> Result<Measurement, String> {
    let (changes, mut contract) = generate_changes(seed, updates)?;
    let (before, setup) = instantiate_state(seed)?;
    contract = add_setup(contract, setup)?;
    contract = contract
        .with_added_phase(
            BenchPhase::ArtifactLoad,
            artifact_load.phase(BenchPhase::ArtifactLoad),
        )
        .map_err(contract_error)?;
    let initialization_started = Instant::now();
    let (_, cache) = plan
        .full(&before)
        .map_err(|error| format!("initialize AVG cache: {error}"))?;
    contract = contract
        .with_added_phase(BenchPhase::Initialization, initialization_started.elapsed())
        .map_err(contract_error)?;
    let started = Instant::now();
    let next = VectorU64
        .apply(&before, &changes)
        .map_err(|error| format!("reused apply: {error:?}"))?;
    let (average, next_cache) = plan
        .delta(&changes, &cache)
        .map_err(|error| format!("reused AVG delta: {error}"))?;
    black_box(average);
    contract = contract.with_phase(BenchPhase::Execution, started.elapsed());
    let validation_started = Instant::now();
    let (exact, exact_cache) = plan
        .full(&next)
        .map_err(|error| format!("validate AVG: {error}"))?;
    if average != exact || next_cache != exact_cache {
        return Err("reused AVG delta diverged from exact full fold".to_owned());
    }
    contract = contract.with_phase(BenchPhase::ResultValidation, validation_started.elapsed());
    let verification_started = Instant::now();
    plan.check(&before, &changes)
        .map_err(|error| format!("AVG certificate failed: {error}"))?;
    contract = contract.with_phase(BenchPhase::Verification, verification_started.elapsed());
    Ok(Measurement { contract, average })
}

fn generate_changes(seed: &[u64], updates: usize) -> Result<(ReplaceDelta, BenchContract), String> {
    let started = Instant::now();
    let changes = build_replacements(seed, updates)?;
    Ok((
        changes,
        BenchContract::empty().with_phase(BenchPhase::InputGeneration, started.elapsed()),
    ))
}

fn instantiate_state(seed: &[u64]) -> Result<(Vec<u64>, BenchContract), String> {
    let allocation_started = Instant::now();
    let mut state = Vec::new();
    state
        .try_reserve_exact(seed.len())
        .map_err(|error| format!("cannot allocate transaction state: {error}"))?;
    let allocation = allocation_started.elapsed();
    let initialization_started = Instant::now();
    state.extend_from_slice(seed);
    let initialization = initialization_started.elapsed();
    Ok((
        state,
        BenchContract::empty()
            .with_phase(BenchPhase::Allocation, allocation)
            .with_phase(BenchPhase::Initialization, initialization),
    ))
}

fn add_setup(mut contract: BenchContract, setup: BenchContract) -> Result<BenchContract, String> {
    for phase in [BenchPhase::Allocation, BenchPhase::Initialization] {
        contract = contract
            .with_added_phase(phase, setup.phase(phase))
            .map_err(contract_error)?;
    }
    Ok(contract)
}

fn summarize(samples: &[Measurement]) -> Result<Summary, String> {
    if samples.is_empty() {
        return Err("cannot summarize zero samples".to_owned());
    }
    let hot = percentile(samples.iter().map(|sample| sample.contract.hot()).collect());
    let lifecycle = percentile(
        samples
            .iter()
            .map(|sample| sample.contract.lifecycle().map_err(contract_error))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let mut phases = [Duration::ZERO; 12];
    for (index, phase) in BenchPhase::ALL.iter().enumerate() {
        phases[index] = percentile(
            samples
                .iter()
                .map(|sample| sample.contract.phase(*phase))
                .collect(),
        );
    }
    Ok(Summary {
        hot,
        lifecycle,
        phases,
        representative: representative_contract(samples)?,
    })
}

fn representative_contract(samples: &[Measurement]) -> Result<BenchContract, String> {
    let mut contracts = samples
        .iter()
        .map(|sample| {
            sample
                .contract
                .lifecycle()
                .map(|lifecycle| (lifecycle, sample.contract))
                .map_err(contract_error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    contracts.sort_unstable_by_key(|(lifecycle, _)| *lifecycle);
    Ok(contracts[(contracts.len() / 2).min(contracts.len() - 1)].1)
}

fn contract_error(error: axon_uic::BenchContractError) -> String {
    format!("benchmark contract: {error:?}")
}

fn build_data(words: usize) -> Result<Vec<u64>, String> {
    let mut data = Vec::new();
    data.try_reserve_exact(words)
        .map_err(|error| format!("cannot allocate benchmark data: {error}"))?;
    let mut state = 0x0A66_1234_u64;
    for _ in 0..words {
        state = xorshift(state);
        data.push(state);
    }
    Ok(data)
}

fn build_replacements(seed: &[u64], updates: usize) -> Result<ReplaceDelta, String> {
    if updates > seed.len() {
        return Err("write count exceeds vector length".to_owned());
    }
    let changes: Vec<_> = seed
        .iter()
        .enumerate()
        .take(updates)
        .map(|(index, old)| Replace::new(index, *old, xorshift(index as u64).wrapping_add(0xA66)))
        .collect();
    ReplaceDelta::try_new(changes).map_err(|error| format!("invalid replacement stream: {error:?}"))
}

fn percentile(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[(values.len() / 2).min(values.len() - 1)]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn xorshift(mut value: u64) -> u64 {
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avg_paths_keep_exact_parity_and_charge_lifecycle_phases() {
        let seed = build_data(128).unwrap();
        let root = create_temporary_artifact_root(16, 0).unwrap();
        let store = ArtifactStore::open(&root);
        let task = task_for_artifact(16, 0).unwrap();
        let plan = install_average(&store, &task, ArtifactStatus::Created)
            .unwrap()
            .0;
        let full = run_full(&seed, 16, &plan).unwrap();
        let (reused, load) = install_average(&store, &task, ArtifactStatus::Reused).unwrap();
        let reuse = run_reuse(&seed, 16, &reused, load).unwrap();

        assert_eq!(full.average, reuse.average);
        assert!(full.contract.lifecycle().unwrap() >= full.contract.hot());
        assert!(reuse.contract.lifecycle().unwrap() >= reuse.contract.hot());
        assert!(reuse.contract.phase(BenchPhase::Verification) > Duration::ZERO);
        assert!(reuse.contract.phase(BenchPhase::ArtifactLoad) > Duration::ZERO);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parser_rejects_out_of_bounds_values() {
        assert!(parse_bounded("0", 1, 2, "--runs").is_err());
        assert!(parse_bounded("3", 1, 2, "--runs").is_err());
    }
}
