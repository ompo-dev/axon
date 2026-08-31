use std::env;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use axon_uic::{
    ChangeSupport, CostEstimate, ExecutionStrategy, IncrementalizabilityAnalyzer, OperatorKind,
    PointUpdate, coalesce_adjacent_last_writes,
};

const DEFAULT_MIB: usize = 64;
const DEFAULT_RUNS: usize = 3;
const DEFAULT_MAX_UPDATES: usize = 65_536;
const MAX_MIB: usize = 256;
const MAX_RUNS: usize = 10;
const MAX_UPDATES: usize = 8_000_000;
const CALIBRATION_EVENTS: usize = 1_000_000;
const EVENT_RUN_LENGTH: usize = 4;
const SWEEP_POINTS: [usize; 10] = [
    1, 16, 256, 4_096, 65_536, 262_144, 1_000_000, 2_000_000, 4_000_000, 8_000_000,
];

#[derive(Clone, Copy)]
struct Config {
    mib: usize,
    runs: usize,
    max_updates: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mib: DEFAULT_MIB,
            runs: DEFAULT_RUNS,
            max_updates: DEFAULT_MAX_UPDATES,
        }
    }
}

#[derive(Clone, Copy)]
struct Measurement {
    duration: Duration,
    first_total: u64,
    total: u64,
}

#[derive(Clone, Copy)]
struct PointResult {
    events: usize,
    coalesced_events: usize,
    batches: usize,
    full: Duration,
    coalesced_full: Duration,
    delta: Duration,
    coalesced: Duration,
    checksum: u64,
}

fn main() {
    match parse_args() {
        Ok(config) => {
            if let Err(message) = run(config) {
                eprintln!("sweep failed: {message}");
                std::process::exit(1);
            }
        }
        Err(message) => {
            eprintln!("{message}");
            eprintln!(
                "usage: axon-uic-delta-sweep [--mib 1..256] [--runs 1..10] [--max-updates 1..8000000]"
            );
            std::process::exit(2);
        }
    }
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            return Err("help requested".to_owned());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--mib" => config.mib = parse_bounded(&value, 1, MAX_MIB, "--mib")?,
            "--runs" => config.runs = parse_bounded(&value, 1, MAX_RUNS, "--runs")?,
            "--max-updates" => {
                config.max_updates = parse_bounded(&value, 1, MAX_UPDATES, "--max-updates")?
            }
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

fn validate_stream_capacity(words: usize, events: usize) -> Result<(), String> {
    let keys = events.div_ceil(EVENT_RUN_LENGTH);
    (keys <= words).then_some(()).ok_or_else(|| {
        format!(
            "largest sweep point needs {keys} distinct keys, but --mib provides only {words}; increase --mib or lower --max-updates"
        )
    })
}

fn run(config: Config) -> Result<(), String> {
    let words = config.mib * 1024 * 1024 / size_of::<u64>();
    let seed = build_data(words)?;
    let contract = IncrementalizabilityAnalyzer::analyze(OperatorKind::Sum);
    let global = IncrementalizabilityAnalyzer::analyze(OperatorKind::Sort);
    let points: Vec<_> = SWEEP_POINTS
        .into_iter()
        .filter(|&events| events <= config.max_updates)
        .collect();
    if points.is_empty() {
        return Err("--max-updates excludes every sweep point".to_owned());
    }
    validate_stream_capacity(words, *points.last().expect("nonempty points"))?;

    println!("# AXON-UIC Delta Algebra physical sweep");
    println!(
        "host {} {}, {} MiB vector, {} runs per point, calibration target {} events",
        env::consts::OS,
        env::consts::ARCH,
        config.mib,
        config.runs,
        CALIBRATION_EVENTS
    );
    println!(
        "operator SUM: class {:?}, exact delta {}, coalescing {}; SORT strategy {:?}",
        contract.class(),
        contract.exact(),
        contract.supports_coalescing(),
        global.select(
            ChangeSupport::new(1, words as u64).unwrap(),
            CostEstimate::new(1, 0, 0, 0)
        )
    );
    println!(
        "event stream: {} adjacent writes per key; full, delta and coalesced outputs must match exactly.",
        EVENT_RUN_LENGTH
    );

    let mut results = Vec::with_capacity(points.len());
    for events in points {
        let result = measure_point(&seed, events, config.runs)?;
        println!(
            "events {:>7}, final {:>7}, batches {:>7}: full {:>8.3} ms; full+coal {:>8.3} ms; delta {:>8.6} ms; delta+coal {:>8.6} ms",
            result.events,
            result.coalesced_events,
            result.batches,
            milliseconds(result.full),
            milliseconds(result.coalesced_full),
            milliseconds(result.delta),
            milliseconds(result.coalesced)
        );
        results.push(result);
    }

    println!(
        "\n| Events | Unique final writes | Support | Full p50 ms | Full+coalesce p50 ms | Delta p50 ms normalized | Delta+coalesce p50 ms normalized | Chosen |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---|");
    for result in &results {
        let strategy = choose_strategy(result, words);
        println!(
            "| {} | {} | {:.6}% | {:.3} | {:.3} | {:.6} | {:.6} | {:?} |",
            result.events,
            result.coalesced_events,
            100.0 * result.coalesced_events as f64 / words as f64,
            milliseconds(result.full),
            milliseconds(result.coalesced_full),
            milliseconds(result.delta),
            milliseconds(result.coalesced),
            strategy
        );
    }
    let latest = results.last().expect("nonempty points");
    let full_reads = words as u128 * size_of::<u64>() as u128;
    let coalesced_reads = latest.coalesced_events as u128 * 2 * size_of::<u64>() as u128;
    let structural_elimination = 100.0 * (1.0 - coalesced_reads as f64 / full_reads as f64);
    println!(
        "\nLargest point: full logical reads {full_reads} bytes; coalesced delta {coalesced_reads} bytes; elimination {structural_elimination:.8}%."
    );
    println!(
        "All points exact parity true; largest-point final checksum {:016X}.",
        latest.checksum
    );
    println!(
        "SUM has no global crossover claim here: this sweep records its response curve only. SORT is classified Full by contract."
    );
    println!(
        "Limit: exact SUM plus adjacent last-write coalescing on local CPU/RAM; no claim about arbitrary programs, energy, DRAM traffic or AGI."
    );
    Ok(())
}

fn choose_strategy(result: &PointResult, words: usize) -> ExecutionStrategy {
    let contract = IncrementalizabilityAnalyzer::analyze(OperatorKind::Sum);
    let best_full = result.full.min(result.coalesced_full);
    let (best_delta, support) = if result.delta <= result.coalesced {
        (result.delta, result.events)
    } else {
        (result.coalesced, result.coalesced_events)
    };
    let full_ns = best_full.as_nanos().min(u64::MAX as u128) as u64;
    let delta_ns = best_delta.as_nanos().min(u64::MAX as u128) as u64;
    contract.select(
        ChangeSupport::new(support as u64, words as u64).expect("validated support"),
        CostEstimate::new(full_ns, delta_ns, 0, 0),
    )
}

fn measure_point(seed: &[u64], events: usize, runs: usize) -> Result<PointResult, String> {
    let updates = build_updates(events, seed.len())?;
    let coalesced_events = coalesce_adjacent_last_writes(&updates).len();
    let batches = CALIBRATION_EVENTS.div_ceil(events).max(1);
    let mut full_samples = Vec::with_capacity(runs);
    let mut coalesced_full_samples = Vec::with_capacity(runs);
    let mut delta_samples = Vec::with_capacity(runs);
    let mut coalesced_samples = Vec::with_capacity(runs);
    let mut checksum = 0_u64;

    for run in 0..runs {
        let full = run_full(seed, &updates)?;
        let coalesced_full = run_coalesced_full(seed, &updates)?;
        let delta = run_delta(seed, &updates, batches)?;
        let coalesced = run_coalesced(seed, &updates, batches)?;
        if full.total != coalesced_full.total
            || full.total != delta.first_total
            || full.total != coalesced.first_total
            || delta.total != coalesced.total
        {
            return Err(format!(
                "parity failure at {events} events, run {}",
                run + 1
            ));
        }
        checksum = coalesced.total;
        full_samples.push(full.duration);
        coalesced_full_samples.push(coalesced_full.duration);
        delta_samples.push(delta.duration);
        coalesced_samples.push(coalesced.duration);
    }

    Ok(PointResult {
        events,
        coalesced_events,
        batches,
        full: percentile(full_samples, 50),
        coalesced_full: percentile(coalesced_full_samples, 50),
        delta: percentile(delta_samples, 50),
        coalesced: percentile(coalesced_samples, 50),
        checksum,
    })
}

fn run_full(seed: &[u64], updates: &[PointUpdate]) -> Result<Measurement, String> {
    let mut data = clone_data(seed)?;
    let started = Instant::now();
    apply_updates(&mut data, updates)?;
    let total = checksum(&data);
    black_box(total);
    Ok(Measurement {
        duration: started.elapsed(),
        first_total: total,
        total,
    })
}

fn run_coalesced_full(seed: &[u64], updates: &[PointUpdate]) -> Result<Measurement, String> {
    let mut data = clone_data(seed)?;
    let started = Instant::now();
    let coalesced = coalesce_adjacent_last_writes(updates);
    apply_updates(&mut data, &coalesced)?;
    let total = checksum(&data);
    black_box(total);
    Ok(Measurement {
        duration: started.elapsed(),
        first_total: total,
        total,
    })
}

fn run_delta(seed: &[u64], updates: &[PointUpdate], batches: usize) -> Result<Measurement, String> {
    let mut data = clone_data(seed)?;
    let mut total = checksum(&data);
    let started = Instant::now();
    let mut trace = 0_u64;
    let mut first_total = total;
    for batch in 0..batches {
        total = apply_delta(&mut data, updates, total, batch as u64)?;
        if batch == 0 {
            first_total = total;
        }
        trace ^= total.rotate_left((batch % 64) as u32);
    }
    black_box(trace);
    let elapsed = started.elapsed();
    verify_total(&data, total)?;
    Ok(Measurement {
        duration: elapsed.div_f64(batches as f64),
        first_total,
        total,
    })
}

fn run_coalesced(
    seed: &[u64],
    updates: &[PointUpdate],
    batches: usize,
) -> Result<Measurement, String> {
    let mut data = clone_data(seed)?;
    let mut total = checksum(&data);
    let started = Instant::now();
    let mut trace = 0_u64;
    let mut first_total = total;
    for batch in 0..batches {
        let coalesced = coalesce_adjacent_last_writes(updates);
        total = apply_delta(&mut data, &coalesced, total, batch as u64)?;
        if batch == 0 {
            first_total = total;
        }
        trace ^= total.rotate_left((batch % 64) as u32);
    }
    black_box(trace);
    let elapsed = started.elapsed();
    verify_total(&data, total)?;
    Ok(Measurement {
        duration: elapsed.div_f64(batches as f64),
        first_total,
        total,
    })
}

fn build_data(words: usize) -> Result<Vec<u64>, String> {
    let mut data = Vec::new();
    data.try_reserve_exact(words)
        .map_err(|error| format!("cannot allocate benchmark data: {error}"))?;
    let mut state = 0xD3_17_AA_u64;
    for _ in 0..words {
        state = xorshift(state);
        data.push(state);
    }
    Ok(data)
}

fn build_updates(events: usize, length: usize) -> Result<Vec<PointUpdate>, String> {
    validate_stream_capacity(length, events)?;
    let mut updates = Vec::new();
    updates
        .try_reserve_exact(events)
        .map_err(|error| format!("cannot allocate update batch: {error}"))?;
    for event in 0..events {
        let group = event / EVENT_RUN_LENGTH;
        let index = group.wrapping_mul(1_048_583).wrapping_add(17) % length;
        updates.push(PointUpdate::new(index, xorshift(event as u64 + 1)));
    }
    Ok(updates)
}

fn clone_data(source: &[u64]) -> Result<Vec<u64>, String> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(source.len())
        .map_err(|error| format!("cannot clone benchmark data: {error}"))?;
    copy.extend_from_slice(source);
    Ok(copy)
}

fn apply_updates(data: &mut [u64], updates: &[PointUpdate]) -> Result<(), String> {
    for update in updates {
        let slot = data
            .get_mut(update.index())
            .ok_or_else(|| format!("update index out of bounds: {}", update.index()))?;
        *slot = update.value();
    }
    Ok(())
}

fn apply_delta(
    data: &mut [u64],
    updates: &[PointUpdate],
    total: u64,
    epoch: u64,
) -> Result<u64, String> {
    let mut total = total;
    for update in updates {
        let old = *data
            .get(update.index())
            .ok_or_else(|| format!("update index out of bounds: {}", update.index()))?;
        let value = update.value().wrapping_add(epoch);
        data[update.index()] = value;
        total = total.wrapping_sub(old).wrapping_add(value);
    }
    Ok(total)
}

fn verify_total(data: &[u64], total: u64) -> Result<(), String> {
    (checksum(data) == total)
        .then_some(())
        .ok_or_else(|| "incremental sum diverged from exact checksum".to_owned())
}

fn checksum(data: &[u64]) -> u64 {
    data.iter()
        .copied()
        .fold(0_u64, |sum, value| sum.wrapping_add(value))
}

fn percentile(mut values: Vec<Duration>, percentile: usize) -> Duration {
    values.sort_unstable();
    values[(values.len() * percentile).div_ceil(100).saturating_sub(1)]
}

fn xorshift(mut value: u64) -> u64 {
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_three_paths_have_identical_final_sum() {
        let seed = build_data(128).unwrap();
        let updates = build_updates(16, seed.len()).unwrap();
        let full = run_full(&seed, &updates).unwrap();
        let coalesced_full = run_coalesced_full(&seed, &updates).unwrap();
        let delta = run_delta(&seed, &updates, 2).unwrap();
        let coalesced = run_coalesced(&seed, &updates, 2).unwrap();

        assert_eq!(full.total, coalesced_full.total);
        assert_eq!(full.total, delta.first_total);
        assert_eq!(full.total, coalesced.first_total);
        assert_eq!(delta.total, coalesced.total);
        assert!(coalesce_adjacent_last_writes(&updates).len() < updates.len());
    }

    #[test]
    fn parser_rejects_out_of_range_values() {
        assert!(parse_bounded("0", 1, 10, "--runs").is_err());
        assert!(parse_bounded("11", 1, 10, "--runs").is_err());
    }

    #[test]
    fn rejects_stream_that_exceeds_vector_key_space() {
        assert!(validate_stream_capacity(128, 513).is_err());
    }

    #[test]
    fn selector_uses_best_measured_full_path() {
        let result = PointResult {
            events: 8,
            coalesced_events: 2,
            batches: 1,
            full: Duration::from_millis(10),
            coalesced_full: Duration::from_millis(20),
            delta: Duration::from_millis(30),
            coalesced: Duration::from_millis(15),
            checksum: 0,
        };

        assert_eq!(choose_strategy(&result, 16), ExecutionStrategy::Full);
    }

    #[test]
    fn selector_preserves_sub_nanosecond_average_delta_advantage() {
        let result = PointResult {
            events: 2_000_000,
            coalesced_events: 500_000,
            batches: 1,
            full: Duration::from_nanos(7_614_000),
            coalesced_full: Duration::from_nanos(9_757_000),
            delta: Duration::from_nanos(7_466_600),
            coalesced: Duration::from_nanos(7_759_600),
            checksum: 0,
        };

        assert_eq!(
            choose_strategy(&result, 8_388_608),
            ExecutionStrategy::Delta
        );
    }
}
