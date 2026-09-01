use std::env;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use axon_uic::{
    BenchContract, BenchPhase, ChangeStructure, DeltaForge, DerivedSumPlan, FoldSpec, Replace,
    ReplaceDelta, StrategyEvidence, StrategyKey, StrategyMetric, StrategyStatus, UpdateLayout,
    VectorU64, WorkloadSignature,
};
use axon_uic::{MeasurementContext, ObservationFrontier, OperatorKind};

const DEFAULT_MIB: usize = 64;
const DEFAULT_RUNS: usize = 5;
const MAX_MIB: usize = 256;
const MAX_RUNS: usize = 30;
const POINTS: [usize; 3] = [1_024, 1_000_000, 4_000_000];

#[derive(Clone)]
struct Config {
    mib: usize,
    runs: usize,
    hardware_id: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mib: DEFAULT_MIB,
            runs: DEFAULT_RUNS,
            hardware_id: "unprofiled".to_owned(),
        }
    }
}

#[derive(Clone, Copy)]
struct Measurement {
    contract: BenchContract,
    total: u64,
}

#[derive(Clone, Copy)]
struct PointResult {
    updates: usize,
    full: PathResult,
    raw_delta: PathResult,
    forged_delta: PathResult,
    forged_evidence_status: StrategyStatus,
    forged_headroom_basis_points: i64,
    checksum: u64,
}

#[derive(Clone, Copy)]
struct PathResult {
    hot: Duration,
    lifecycle: Duration,
    phases: [Duration; 10],
}

#[derive(Clone, Copy)]
enum Path {
    Full,
    RawDelta,
    ForgedDelta,
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(message) => {
            eprintln!("deltaforge sweep failed: {message}");
            std::process::exit(2);
        }
    }
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            return Err(
                "usage: axon-uic-deltaforge-sum [--mib 1..256] [--runs 1..30] [--hardware-id id]"
                    .to_owned(),
            );
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--mib" => config.mib = parse_bounded(&value, 1, MAX_MIB, "--mib")?,
            "--runs" => config.runs = parse_bounded(&value, 1, MAX_RUNS, "--runs")?,
            "--hardware-id" if !value.trim().is_empty() => config.hardware_id = value,
            "--hardware-id" => return Err("--hardware-id cannot be empty".to_owned()),
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
    let derive_started = Instant::now();
    let plan = black_box(
        DeltaForge::synthesize(black_box(FoldSpec::AddModU64))
            .map_err(|error| format!("forge: {error:?}"))?,
    );
    let derive_duration = derive_started.elapsed();
    let points: Vec<_> = POINTS.into_iter().filter(|&point| point <= words).collect();
    if points.is_empty() {
        return Err("vector too small for every DeltaForge point".to_owned());
    }

    println!("# AXON-UIC DeltaForge-SUM physical sweep");
    println!(
        "host {} {}, {} MiB vector, {} runs per point",
        env::consts::OS,
        env::consts::ARCH,
        config.mib,
        config.runs
    );
    println!(
        "input FoldSpec::AddModU64; reference certificate {:?}, {:?}, {:?}; reference setup {:.6} ms",
        plan.certificate().algebra(),
        plan.certificate().maintenance_state(),
        plan.certificate().update_rule(),
        milliseconds(derive_duration)
    );
    println!(
        "protocol: all paths generate the same deterministic checked ReplaceDelta per round. HOT is execution only; LIFECYCLE includes every named phase. Exact checks are reported separately, never folded into HOT."
    );

    let mut results = Vec::with_capacity(points.len());
    for updates in points {
        let result = measure_point(&seed, &plan, updates, config.runs, &config.hardware_id)?;
        println!(
            "point {updates:>7}: full HOT/LIFECYCLE {:>8.3}/{:>8.3} ms; raw {:>8.3}/{:>8.3} ms; forged {:>8.3}/{:>8.3} ms; parity true",
            milliseconds(result.full.hot),
            milliseconds(result.full.lifecycle),
            milliseconds(result.raw_delta.hot),
            milliseconds(result.raw_delta.lifecycle),
            milliseconds(result.forged_delta.hot),
            milliseconds(result.forged_delta.lifecycle)
        );
        println!(
            "  paired Raw×Forge HOT evidence: {:?}; headroom {} bp",
            result.forged_evidence_status, result.forged_headroom_basis_points
        );
        results.push(result);
    }

    println!(
        "\n| Final writes | Full HOT p50 ms | Full LIFECYCLE p50 ms | Raw HOT p50 ms | Raw LIFECYCLE p50 ms | Forge HOT p50 ms | Forge LIFECYCLE p50 ms | Forge / Raw HOT |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
    for result in &results {
        println!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2}x |",
            result.updates,
            milliseconds(result.full.hot),
            milliseconds(result.full.lifecycle),
            milliseconds(result.raw_delta.hot),
            milliseconds(result.raw_delta.lifecycle),
            milliseconds(result.forged_delta.hot),
            milliseconds(result.forged_delta.lifecycle),
            duration_ratio(result.raw_delta.hot, result.forged_delta.hot),
        );
    }
    let largest = results.last().expect("nonempty points");
    println!("\nLargest-point phase p50 ms; zero means not applicable in this batch protocol:");
    println!("| Phase | Full | Raw Delta | Forged Delta |");
    println!("|---|---:|---:|---:|");
    for (index, phase) in BenchPhase::ALL.iter().enumerate() {
        println!(
            "| {} | {:.6} | {:.6} | {:.6} |",
            phase.as_str(),
            milliseconds(largest.full.phases[index]),
            milliseconds(largest.raw_delta.phases[index]),
            milliseconds(largest.forged_delta.phases[index]),
        );
    }
    println!(
        "all points exact parity true; largest-point checksum {:016X}.",
        largest.checksum
    );
    println!(
        "largest-point paired Raw×Forge HOT evidence: {:?}; headroom {} bp.",
        largest.forged_evidence_status, largest.forged_headroom_basis_points
    );
    println!(
        "limit: grammar recognizes only declared modular addition. This is derived-program synthesis with an executable checker, not blind program synthesis, formal proof or learning."
    );
    Ok(())
}

fn measure_point(
    seed: &[u64],
    plan: &DerivedSumPlan,
    updates: usize,
    runs: usize,
    hardware_id: &str,
) -> Result<PointResult, String> {
    let mut full_samples = Vec::with_capacity(runs);
    let mut raw_samples = Vec::with_capacity(runs);
    let mut forged_samples = Vec::with_capacity(runs);
    let mut checksum = 0_u64;

    for round in 0..runs {
        let mut full = None;
        let mut raw = None;
        let mut forged = None;
        for path in path_order(round) {
            match path {
                Path::Full => full = Some(run_full(seed, updates, plan)?),
                Path::RawDelta => raw = Some(run_raw_delta(seed, updates, plan)?),
                Path::ForgedDelta => forged = Some(run_forged_delta(seed, updates)?),
            }
        }
        let full = full.expect("every order includes Full");
        let raw = raw.expect("every order includes Raw Delta");
        let forged = forged.expect("every order includes Forged Delta");
        if full.total != raw.total || full.total != forged.total {
            return Err(format!(
                "exact parity failure at {updates} writes, round {}",
                round + 1
            ));
        }
        let full_lifecycle = full.contract.lifecycle().map_err(contract_error)?;
        let raw_lifecycle = raw.contract.lifecycle().map_err(contract_error)?;
        let forged_lifecycle = forged.contract.lifecycle().map_err(contract_error)?;
        println!(
            "  run {:02}: full HOT/LIFECYCLE {:>8.3}/{:>8.3} ms; raw {:>8.3}/{:>8.3} ms; forged {:>8.3}/{:>8.3} ms; parity true",
            round + 1,
            milliseconds(full.contract.hot()),
            milliseconds(full_lifecycle),
            milliseconds(raw.contract.hot()),
            milliseconds(raw_lifecycle),
            milliseconds(forged.contract.hot()),
            milliseconds(forged_lifecycle),
        );
        checksum = full.total;
        full_samples.push(full);
        raw_samples.push(raw);
        forged_samples.push(forged);
    }

    let forged_evidence = forged_hot_evidence(
        seed.len(),
        updates,
        hardware_id,
        &raw_samples,
        &forged_samples,
    )?;

    Ok(PointResult {
        updates,
        full: aggregate_samples(full_samples)?,
        raw_delta: aggregate_samples(raw_samples)?,
        forged_delta: aggregate_samples(forged_samples)?,
        forged_evidence_status: forged_evidence.status(),
        forged_headroom_basis_points: forged_evidence.oracle_headroom_basis_points(),
        checksum,
    })
}

fn path_order(round: usize) -> [Path; 3] {
    const ORDERS: [[Path; 3]; 6] = [
        [Path::Full, Path::RawDelta, Path::ForgedDelta],
        [Path::Full, Path::ForgedDelta, Path::RawDelta],
        [Path::RawDelta, Path::Full, Path::ForgedDelta],
        [Path::RawDelta, Path::ForgedDelta, Path::Full],
        [Path::ForgedDelta, Path::Full, Path::RawDelta],
        [Path::ForgedDelta, Path::RawDelta, Path::Full],
    ];
    ORDERS[(round.wrapping_mul(5).wrapping_add(1)) % ORDERS.len()]
}

fn run_full(seed: &[u64], updates: usize, plan: &DerivedSumPlan) -> Result<Measurement, String> {
    let (changes, mut contract) = generate_changes(seed, updates)?;
    let (before, setup) = instantiate_state(seed)?;
    contract = contract_with_setup(contract, setup)?;
    let started = Instant::now();
    let next = VectorU64
        .apply(&before, &changes)
        .map_err(|error| format!("full apply: {error:?}"))?;
    let (total, _) = plan.full(&next);
    black_box(total);
    contract = contract.with_phase(BenchPhase::Execution, started.elapsed());
    let teardown_started = Instant::now();
    drop(next);
    drop(before);
    drop(changes);
    contract = contract.with_phase(BenchPhase::Teardown, teardown_started.elapsed());
    Ok(Measurement { contract, total })
}

fn run_raw_delta(
    seed: &[u64],
    updates: usize,
    plan: &DerivedSumPlan,
) -> Result<Measurement, String> {
    let (changes, mut contract) = generate_changes(seed, updates)?;
    let (before, setup) = instantiate_state(seed)?;
    contract = contract_with_setup(contract, setup)?;
    let initialization_started = Instant::now();
    let (initial_total, _) = plan.full(&before);
    contract = contract
        .with_added_phase(BenchPhase::Initialization, initialization_started.elapsed())
        .map_err(contract_error)?;
    let started = Instant::now();
    let next = VectorU64
        .apply(&before, &changes)
        .map_err(|error| format!("raw apply: {error:?}"))?;
    let total = changes
        .changes()
        .iter()
        .fold(initial_total, |total, change| {
            total
                .wrapping_sub(change.old())
                .wrapping_add(change.new_value())
        });
    black_box(total);
    contract = contract.with_phase(BenchPhase::Execution, started.elapsed());
    let validation_started = Instant::now();
    let (exact_total, _) = plan.full(&next);
    (total == exact_total)
        .then_some(())
        .ok_or_else(|| "raw delta diverged from exact fold".to_owned())?;
    contract = contract.with_phase(BenchPhase::ResultValidation, validation_started.elapsed());
    let teardown_started = Instant::now();
    drop(next);
    drop(before);
    drop(changes);
    contract = contract.with_phase(BenchPhase::Teardown, teardown_started.elapsed());
    Ok(Measurement { contract, total })
}

fn run_forged_delta(seed: &[u64], updates: usize) -> Result<Measurement, String> {
    let (changes, mut contract) = generate_changes(seed, updates)?;
    let (before, setup) = instantiate_state(seed)?;
    contract = contract_with_setup(contract, setup)?;
    let synthesis_started = Instant::now();
    let plan = black_box(
        DeltaForge::synthesize(black_box(FoldSpec::AddModU64))
            .map_err(|error| format!("forge: {error:?}"))?,
    );
    contract = contract.with_phase(BenchPhase::Synthesis, synthesis_started.elapsed());
    let initialization_started = Instant::now();
    let (initial_total, cache) = plan.full(&before);
    contract = contract
        .with_added_phase(BenchPhase::Initialization, initialization_started.elapsed())
        .map_err(contract_error)?;
    let started = Instant::now();
    let next = VectorU64
        .apply(&before, &changes)
        .map_err(|error| format!("forged apply: {error:?}"))?;
    let (output_delta, next_cache) = plan
        .delta(&changes, &cache)
        .map_err(|error| format!("forged delta: {error:?}"))?;
    let total = plan.apply_output_delta(initial_total, output_delta);
    black_box(total);
    contract = contract.with_phase(BenchPhase::Execution, started.elapsed());
    let validation_started = Instant::now();
    let (exact_total, exact_cache) = plan.full(&next);
    if total != exact_total || next_cache != exact_cache {
        return Err("forged delta diverged from exact fold".to_owned());
    }
    contract = contract.with_phase(BenchPhase::ResultValidation, validation_started.elapsed());
    let verification_started = Instant::now();
    plan.check(&before, &changes)
        .map_err(|error| format!("forged certificate failed: {error:?}"))?;
    contract = contract.with_phase(BenchPhase::Verification, verification_started.elapsed());
    let teardown_started = Instant::now();
    drop(next);
    drop(before);
    drop(changes);
    contract = contract.with_phase(BenchPhase::Teardown, teardown_started.elapsed());
    Ok(Measurement { contract, total })
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

fn contract_with_setup(
    mut contract: BenchContract,
    setup: BenchContract,
) -> Result<BenchContract, String> {
    for phase in [BenchPhase::Allocation, BenchPhase::Initialization] {
        contract = contract
            .with_added_phase(phase, setup.phase(phase))
            .map_err(contract_error)?;
    }
    Ok(contract)
}

fn contract_error(error: axon_uic::BenchContractError) -> String {
    format!("benchmark contract: {error:?}")
}

fn aggregate_samples(samples: Vec<Measurement>) -> Result<PathResult, String> {
    if samples.is_empty() {
        return Err("cannot aggregate zero benchmark samples".to_owned());
    }
    let hot = percentile(
        samples.iter().map(|sample| sample.contract.hot()).collect(),
        50,
    );
    let lifecycle = percentile(
        samples
            .iter()
            .map(|sample| sample.contract.lifecycle().map_err(contract_error))
            .collect::<Result<Vec<_>, _>>()?,
        50,
    );
    let mut phases = [Duration::ZERO; 10];
    for (index, phase) in BenchPhase::ALL.iter().enumerate() {
        phases[index] = percentile(
            samples
                .iter()
                .map(|sample| sample.contract.phase(*phase))
                .collect(),
            50,
        );
    }
    Ok(PathResult {
        hot,
        lifecycle,
        phases,
    })
}

fn forged_hot_evidence(
    words: usize,
    updates: usize,
    hardware_id: &str,
    raw_samples: &[Measurement],
    forged_samples: &[Measurement],
) -> Result<StrategyEvidence, String> {
    let context = MeasurementContext::new(
        hardware_id,
        UpdateLayout::CanonicalShardOrdered,
        1,
        StrategyMetric::Latency,
        2,
    )
    .map_err(|error| format!("benchmark context: {error:?}"))?;
    let workload = WorkloadSignature::new(
        OperatorKind::Sum,
        words,
        1,
        updates,
        updates,
        ObservationFrontier::FinalStateOnly,
        context,
    )
    .map_err(|error| format!("benchmark workload: {error:?}"))?;
    let raw_hot: Vec<_> = raw_samples
        .iter()
        .map(|sample| duration_nanos(sample.contract.hot()))
        .collect();
    let forged_hot: Vec<_> = forged_samples
        .iter()
        .map(|sample| duration_nanos(sample.contract.hot()))
        .collect();
    StrategyEvidence::from_paired_samples(
        workload,
        StrategyKey::RawDelta,
        &raw_hot,
        StrategyKey::ForgedDelta,
        &forged_hot,
    )
    .map_err(|error| format!("paired Raw×Forge evidence: {error:?}"))
}

fn duration_nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn build_data(words: usize) -> Result<Vec<u64>, String> {
    let mut data = Vec::new();
    data.try_reserve_exact(words)
        .map_err(|error| format!("cannot allocate benchmark data: {error}"))?;
    let mut state = 0x0D31_7AF0_u64;
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
    let mut changes = Vec::new();
    changes
        .try_reserve_exact(updates)
        .map_err(|error| format!("cannot allocate replacement stream: {error}"))?;
    for (index, &old) in seed.iter().enumerate().take(updates) {
        changes.push(Replace::new(
            index,
            old,
            xorshift(index as u64).wrapping_add(0x000F_026E),
        ));
    }
    ReplaceDelta::try_new(changes).map_err(|error| format!("invalid replacement stream: {error:?}"))
}

fn percentile(mut values: Vec<Duration>, percentile: usize) -> Duration {
    values.sort_unstable();
    values[(values.len() * percentile).div_ceil(100).saturating_sub(1)]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn duration_ratio(numerator: Duration, denominator: Duration) -> f64 {
    numerator.as_nanos() as f64 / denominator.as_nanos().max(1) as f64
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
    fn full_raw_and_forged_paths_keep_exact_parity() {
        let seed = build_data(128).unwrap();
        let plan = DeltaForge::synthesize(FoldSpec::AddModU64).unwrap();
        let full = run_full(&seed, 16, &plan).unwrap();
        let raw = run_raw_delta(&seed, 16, &plan).unwrap();
        let forged = run_forged_delta(&seed, 16).unwrap();

        assert_eq!(full.total, raw.total);
        assert_eq!(full.total, forged.total);
        assert!(full.contract.lifecycle().unwrap() >= full.contract.hot());
        assert!(raw.contract.lifecycle().unwrap() >= raw.contract.hot());
        assert!(forged.contract.lifecycle().unwrap() >= forged.contract.hot());
    }

    #[test]
    fn parser_and_permuted_path_order_cover_bounds() {
        assert!(parse_bounded("0", 1, 2, "--runs").is_err());
        assert!(parse_bounded("3", 1, 2, "--runs").is_err());
        assert!(
            path_order(0)
                .iter()
                .any(|path| matches!(path, Path::ForgedDelta))
        );
    }

    #[test]
    fn paired_raw_and_forge_crossings_do_not_promote_a_strategy() {
        let raw = [measurement_with_hot(10), measurement_with_hot(20)];
        let forged = [measurement_with_hot(12), measurement_with_hot(18)];

        let evidence = forged_hot_evidence(16, 2, "test-cpu", &raw, &forged).unwrap();

        assert_eq!(evidence.status(), StrategyStatus::Inconclusive);
    }

    fn measurement_with_hot(milliseconds: u64) -> Measurement {
        Measurement {
            contract: BenchContract::empty()
                .with_phase(BenchPhase::Execution, Duration::from_millis(milliseconds)),
            total: 0,
        }
    }
}
