use std::env;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use axon_uic::{
    ChangeStructure, DeltaForge, DerivedSumPlan, FoldSpec, Replace, ReplaceDelta, VectorU64,
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
    execution: Duration,
    verification: Duration,
    total: u64,
}

#[derive(Clone, Copy)]
struct PointResult {
    updates: usize,
    full: Duration,
    raw_delta: Duration,
    forged_delta: Duration,
    raw_verification: Duration,
    forged_verification: Duration,
    checksum: u64,
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
            return Err("usage: axon-uic-deltaforge-sum [--mib 1..256] [--runs 1..30]".to_owned());
        }
        let value = args
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
    let derive_started = Instant::now();
    let plan =
        DeltaForge::synthesize(FoldSpec::AddModU64).map_err(|error| format!("forge: {error:?}"))?;
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
        "input FoldSpec::AddModU64; derived certificate {:?}, {:?}, {:?}; one-time synthesis {:.6} ms outside samples",
        plan.certificate().algebra(),
        plan.certificate().maintenance_state(),
        plan.certificate().update_rule(),
        milliseconds(derive_duration)
    );
    println!(
        "protocol: all paths materialize the same checked ReplaceDelta; Full recomputes total, Raw Delta is manual control, Forged Delta uses derived plan. Exact checks stay outside execution timers."
    );

    let mut results = Vec::with_capacity(points.len());
    for updates in points {
        let result = measure_point(&seed, &plan, updates, config.runs)?;
        println!(
            "point {updates:>7}: full {:>8.3} ms; raw {:>8.3} ms; forged {:>8.3} ms; parity true",
            milliseconds(result.full),
            milliseconds(result.raw_delta),
            milliseconds(result.forged_delta)
        );
        results.push(result);
    }

    println!(
        "\n| Final writes | Full p50 ms | Raw Delta p50 ms | Forged Delta p50 ms | Forge / Raw | Raw verify p50 ms | Forge verify p50 ms |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for result in &results {
        println!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.2}x | {:.3} | {:.3} |",
            result.updates,
            milliseconds(result.full),
            milliseconds(result.raw_delta),
            milliseconds(result.forged_delta),
            duration_ratio(result.raw_delta, result.forged_delta),
            milliseconds(result.raw_verification),
            milliseconds(result.forged_verification),
        );
    }
    println!(
        "all points exact parity true; largest-point checksum {:016X}.",
        results.last().expect("nonempty points").checksum
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
) -> Result<PointResult, String> {
    let changes = build_replacements(seed, updates)?;
    let mut full_samples = Vec::with_capacity(runs);
    let mut raw_samples = Vec::with_capacity(runs);
    let mut forged_samples = Vec::with_capacity(runs);
    let mut raw_verification_samples = Vec::with_capacity(runs);
    let mut forged_verification_samples = Vec::with_capacity(runs);
    let mut checksum = 0_u64;

    for round in 0..runs {
        let mut full = None;
        let mut raw = None;
        let mut forged = None;
        for path in path_order(round) {
            match path {
                Path::Full => full = Some(run_full(seed, &changes, plan)?),
                Path::RawDelta => raw = Some(run_raw_delta(seed, &changes, plan)?),
                Path::ForgedDelta => forged = Some(run_forged_delta(seed, &changes, plan)?),
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
        println!(
            "  run {:02}: full {:>8.3} ms; raw {:>8.3} ms; forged {:>8.3} ms; parity true",
            round + 1,
            milliseconds(full.execution),
            milliseconds(raw.execution),
            milliseconds(forged.execution)
        );
        checksum = full.total;
        full_samples.push(full.execution);
        raw_samples.push(raw.execution);
        forged_samples.push(forged.execution);
        raw_verification_samples.push(raw.verification);
        forged_verification_samples.push(forged.verification);
    }

    Ok(PointResult {
        updates,
        full: percentile(full_samples, 50),
        raw_delta: percentile(raw_samples, 50),
        forged_delta: percentile(forged_samples, 50),
        raw_verification: percentile(raw_verification_samples, 50),
        forged_verification: percentile(forged_verification_samples, 50),
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

fn run_full(
    seed: &[u64],
    changes: &ReplaceDelta,
    plan: &DerivedSumPlan,
) -> Result<Measurement, String> {
    let before = seed.to_vec();
    let started = Instant::now();
    let next = VectorU64
        .apply(&before, changes)
        .map_err(|error| format!("full apply: {error:?}"))?;
    let (total, _) = plan.full(&next);
    black_box(total);
    Ok(Measurement {
        execution: started.elapsed(),
        verification: Duration::ZERO,
        total,
    })
}

fn run_raw_delta(
    seed: &[u64],
    changes: &ReplaceDelta,
    plan: &DerivedSumPlan,
) -> Result<Measurement, String> {
    let before = seed.to_vec();
    let (initial_total, _) = plan.full(&before);
    let started = Instant::now();
    let next = VectorU64
        .apply(&before, changes)
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
    let execution = started.elapsed();
    let verification_started = Instant::now();
    let (exact_total, _) = plan.full(&next);
    (total == exact_total)
        .then_some(())
        .ok_or_else(|| "raw delta diverged from exact fold".to_owned())?;
    Ok(Measurement {
        execution,
        verification: verification_started.elapsed(),
        total,
    })
}

fn run_forged_delta(
    seed: &[u64],
    changes: &ReplaceDelta,
    plan: &DerivedSumPlan,
) -> Result<Measurement, String> {
    let before = seed.to_vec();
    let (initial_total, cache) = plan.full(&before);
    let started = Instant::now();
    let next = VectorU64
        .apply(&before, changes)
        .map_err(|error| format!("forged apply: {error:?}"))?;
    let (output_delta, next_cache) = plan
        .delta(changes, &cache)
        .map_err(|error| format!("forged delta: {error:?}"))?;
    let total = plan.apply_output_delta(initial_total, output_delta);
    black_box(total);
    let execution = started.elapsed();
    let verification_started = Instant::now();
    let (exact_total, exact_cache) = plan.full(&next);
    if total != exact_total || next_cache != exact_cache {
        return Err("forged delta diverged from exact fold".to_owned());
    }
    plan.check(&before, changes)
        .map_err(|error| format!("forged certificate failed: {error:?}"))?;
    Ok(Measurement {
        execution,
        verification: verification_started.elapsed(),
        total,
    })
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
        let changes = build_replacements(&seed, 16).unwrap();
        let full = run_full(&seed, &changes, &plan).unwrap();
        let raw = run_raw_delta(&seed, &changes, &plan).unwrap();
        let forged = run_forged_delta(&seed, &changes, &plan).unwrap();

        assert_eq!(full.total, raw.total);
        assert_eq!(full.total, forged.total);
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
}
