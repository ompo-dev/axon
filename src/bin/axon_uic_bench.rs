use std::env;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

const DEFAULT_MIB: usize = 64;
const DEFAULT_QUERIES: usize = 20;
const DEFAULT_RUNS: usize = 5;
const WARMUPS: usize = 2;
const DELTA_CALIBRATION_BATCHES: usize = 10_000;
const MAX_MIB: usize = 256;
const MAX_QUERIES: usize = 200;
const MAX_RUNS: usize = 10;

#[derive(Clone, Copy)]
struct Config {
    mib: usize,
    queries: usize,
    runs: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mib: DEFAULT_MIB,
            queries: DEFAULT_QUERIES,
            runs: DEFAULT_RUNS,
        }
    }
}

#[derive(Clone, Copy)]
struct Round {
    full: Duration,
    delta: Duration,
    checksum: u64,
}

fn main() {
    match parse_args() {
        Ok(config) => {
            if let Err(message) = run(config) {
                eprintln!("benchmark failed: {message}");
                std::process::exit(1);
            }
        }
        Err(message) => {
            eprintln!("{message}");
            eprintln!("usage: axon-uic-bench [--mib 1..256] [--queries 1..200] [--runs 1..10]");
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
            "--queries" => config.queries = parse_bounded(&value, 1, MAX_QUERIES, "--queries")?,
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
    let words = config.mib * 1024 * 1024 / size_of::<u64>();
    let seed = build_data(words)?;
    let physical_bytes = words * size_of::<u64>();
    println!("# AXON-UIC Demand×Delta physical sweep");
    println!(
        "host {} {}, data {} MiB per engine, queries {}, measured runs {}, warmups {}",
        env::consts::OS,
        env::consts::ARCH,
        config.mib,
        config.queries,
        config.runs,
        WARMUPS
    );
    println!(
        "protocol: same deterministic point updates; exact checksum parity is mandatory; delta is normalized from {DELTA_CALIBRATION_BATCHES} batches."
    );

    for warmup in 0..WARMUPS {
        black_box(run_round(&seed, config.queries, warmup % 2 == 1)?);
    }

    let mut rounds = Vec::with_capacity(config.runs);
    for index in 0..config.runs {
        let round = run_round(&seed, config.queries, index % 2 == 1)?;
        println!(
            "run {:02}: full batch {:>9.3} ms; delta batch {:>9.6} ms; first-batch parity true",
            index + 1,
            milliseconds(round.full),
            milliseconds(round.delta)
        );
        rounds.push(round);
    }

    let full: Vec<_> = rounds.iter().map(|round| round.full).collect();
    let delta: Vec<_> = rounds.iter().map(|round| round.delta).collect();
    let full_p50 = percentile(full, 50);
    let full_p95 = percentile(rounds.iter().map(|round| round.full).collect(), 95);
    let delta_p50 = percentile(delta, 50);
    let delta_p95 = percentile(rounds.iter().map(|round| round.delta).collect(), 95);
    let full_per_query_ns = per_query_nanoseconds(full_p50, config.queries);
    let delta_per_query_ns = per_query_nanoseconds(delta_p50, config.queries);
    let batch_speedup = speedup(full_p50, delta_p50);
    let per_query_speedup = full_per_query_ns / delta_per_query_ns;
    let full_bytes = physical_bytes as u128 * config.queries as u128;
    let delta_bytes = 2_u128 * size_of::<u64>() as u128 * config.queries as u128;
    let checksum = rounds.first().map(|round| round.checksum).unwrap_or(0);

    println!("\n| Metric | Result |");
    println!("|---|---:|");
    println!(
        "| Full batch p50 / p95 | {:.3} / {:.3} ms |",
        milliseconds(full_p50),
        milliseconds(full_p95)
    );
    println!(
        "| Delta batch p50 / p95 | {:.6} / {:.6} ms |",
        milliseconds(delta_p50),
        milliseconds(delta_p95)
    );
    println!(
        "| Full per query p50 (derived from {} queries) | {:.3} ns |",
        config.queries, full_per_query_ns
    );
    println!(
        "| Delta per query p50 (derived from {} queries) | {:.3} ns |",
        config.queries, delta_per_query_ns
    );
    println!("| Observed batch speedup p50 | {:.2}x |", batch_speedup);
    println!(
        "| Observed per-query speedup p50 | {:.2}x |",
        per_query_speedup
    );
    println!(
        "| Logical reads per run, full / delta | {} / {} bytes |",
        full_bytes, delta_bytes
    );
    println!("| Final checksum | {checksum:016X} |");
    println!("| Exact parity, first batch | true |");
    println!("| Exact accumulator validation, all batches | true |");
    println!(
        "\nLimit: measures this exact sum-under-point-update workload on local CPU/RAM; no energy or AGI claim."
    );
    Ok(())
}

fn build_data(words: usize) -> Result<Vec<u64>, String> {
    let mut data = Vec::new();
    data.try_reserve_exact(words)
        .map_err(|error| format!("cannot allocate benchmark data: {error}"))?;
    let mut state = 0xA8_0A_1C_u64;
    for _ in 0..words {
        state = xorshift(state);
        data.push(state);
    }
    Ok(data)
}

fn run_round(seed: &[u64], queries: usize, delta_first: bool) -> Result<Round, String> {
    let mut full_data = clone_data(seed)?;
    let mut delta_data = clone_data(seed)?;
    let (full, full_trace, full_total, delta, delta_trace, delta_total, delta_final) =
        if delta_first {
            let (delta, delta_trace, delta_total, delta_final) =
                run_delta(&mut delta_data, queries)?;
            let (full, full_trace, full_total) = run_full(&mut full_data, queries);
            (
                full,
                full_trace,
                full_total,
                delta,
                delta_trace,
                delta_total,
                delta_final,
            )
        } else {
            let (full, full_trace, full_total) = run_full(&mut full_data, queries);
            let (delta, delta_trace, delta_total, delta_final) =
                run_delta(&mut delta_data, queries)?;
            (
                full,
                full_trace,
                full_total,
                delta,
                delta_trace,
                delta_total,
                delta_final,
            )
        };
    if full_total != delta_total {
        return Err("full and delta final total diverged".to_owned());
    }
    if full_trace != delta_trace {
        return Err("full and delta output trace diverged".to_owned());
    }
    Ok(Round {
        full,
        delta,
        checksum: delta_final,
    })
}

fn clone_data(source: &[u64]) -> Result<Vec<u64>, String> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(source.len())
        .map_err(|error| format!("cannot clone benchmark data: {error}"))?;
    copy.extend_from_slice(source);
    Ok(copy)
}

fn run_full(data: &mut [u64], queries: usize) -> (Duration, u64, u64) {
    let started = Instant::now();
    let mut trace = 0_u64;
    let mut total = 0_u64;
    for query in 0..queries {
        data[update_index(query, data.len())] = update_value(query);
        total = checksum(data);
        trace ^= total.rotate_left((query % 64) as u32);
    }
    (started.elapsed(), trace, total)
}

fn run_delta(data: &mut [u64], queries: usize) -> Result<(Duration, u64, u64, u64), String> {
    let mut total = checksum(data);
    let started = Instant::now();
    let mut first_trace = 0_u64;
    let mut first_total = total;
    for batch in 0..DELTA_CALIBRATION_BATCHES {
        let mut trace = 0_u64;
        for query in 0..queries {
            let update = batch * queries + query;
            let index = update_index(query, data.len());
            let old = data[index];
            let new = update_value(update);
            data[index] = new;
            total = total.wrapping_sub(old).wrapping_add(new);
            trace ^= total.rotate_left((query % 64) as u32);
        }
        if batch == 0 {
            first_trace = trace;
            first_total = total;
        }
    }
    black_box(total);
    let elapsed = started.elapsed();
    let exact_final = checksum(data);
    if total != exact_final {
        return Err("delta accumulator diverged from exact final checksum".to_owned());
    }
    let normalized = elapsed.div_f64(DELTA_CALIBRATION_BATCHES as f64);
    Ok((normalized, first_trace, first_total, exact_final))
}

fn update_index(query: usize, length: usize) -> usize {
    query.wrapping_mul(1_048_583).wrapping_add(17) % length
}

fn update_value(query: usize) -> u64 {
    xorshift(query as u64 + 0x9E37_79B9_7F4A_7C15)
}

fn xorshift(mut value: u64) -> u64 {
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

fn checksum(data: &[u64]) -> u64 {
    data.iter()
        .copied()
        .fold(0_u64, |sum, value| sum.wrapping_add(value))
}

fn percentile(mut values: Vec<Duration>, percentile: usize) -> Duration {
    values.sort_unstable();
    let index = (values.len() * percentile).div_ceil(100).saturating_sub(1);
    values[index]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn per_query_nanoseconds(duration: Duration, queries: usize) -> f64 {
    duration.as_secs_f64() * 1_000_000_000.0 / queries as f64
}

fn speedup(full: Duration, delta: Duration) -> f64 {
    full.as_nanos() as f64 / delta.as_nanos().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_and_delta_paths_keep_exact_parity() {
        let seed = build_data(128).unwrap();
        let round = run_round(&seed, 3, false).unwrap();

        assert!(!round.full.is_zero());
        assert!(!round.delta.is_zero());
    }

    #[test]
    fn percentile_selects_requested_rank() {
        let values = vec![
            Duration::from_nanos(1),
            Duration::from_nanos(2),
            Duration::from_nanos(3),
        ];

        assert_eq!(percentile(values, 50), Duration::from_nanos(2));
    }

    #[test]
    fn batch_and_per_query_speedups_are_equivalent() {
        let full = Duration::from_nanos(2_000);
        let delta = Duration::from_nanos(20);
        let full_per_query = per_query_nanoseconds(full, 20);
        let delta_per_query = per_query_nanoseconds(delta, 20);

        assert_eq!(full_per_query, 100.0);
        assert_eq!(delta_per_query, 1.0);
        assert_eq!(speedup(full, delta), full_per_query / delta_per_query);
    }
}
