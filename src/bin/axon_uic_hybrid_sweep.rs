use std::env;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use axon_uic::{ObservationFrontier, PointUpdate, coalesce_adjacent_at_frontier};

const DEFAULT_MIB: usize = 64;
const DEFAULT_RUNS: usize = 5;
const MAX_MIB: usize = 256;
const MAX_RUNS: usize = 10;
const SHARD_COUNT: usize = 64;
const DENSE_SHARDS: usize = 8;
const DUPLICATE_SHARD: usize = DENSE_SHARDS;
const SPARSE_SHARD: usize = DUPLICATE_SHARD + 1;
const DUPLICATE_RUN: usize = 4;
const SPARSE_UPDATES: usize = 1_024;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShardStrategy {
    Skip,
    RawDelta,
    CoalescedDelta,
    FullLocal,
}

#[derive(Clone, Copy, Default)]
struct PlanCounts {
    skip: usize,
    raw_delta: usize,
    coalesced_delta: usize,
    full_local: usize,
}

impl PlanCounts {
    fn record(&mut self, strategy: ShardStrategy) {
        match strategy {
            ShardStrategy::Skip => self.skip += 1,
            ShardStrategy::RawDelta => self.raw_delta += 1,
            ShardStrategy::CoalescedDelta => self.coalesced_delta += 1,
            ShardStrategy::FullLocal => self.full_local += 1,
        }
    }
}

enum ChangeStorage<'a> {
    Borrowed(&'a [PointUpdate]),
    Owned(Vec<PointUpdate>),
}

impl ChangeStorage<'_> {
    fn as_slice(&self) -> &[PointUpdate] {
        match self {
            Self::Borrowed(updates) => updates,
            Self::Owned(updates) => updates,
        }
    }
}

struct CompiledShard<'a> {
    strategy: ShardStrategy,
    updates: ChangeStorage<'a>,
}

struct CompiledChanges<'a> {
    shards: Vec<CompiledShard<'a>>,
    counts: PlanCounts,
}

#[derive(Clone, Copy)]
struct Measurement {
    duration: Duration,
    total: u64,
}

#[derive(Clone, Copy)]
struct HybridMeasurement {
    duration: Duration,
    compile: Duration,
    total: u64,
}

fn main() {
    match parse_args() {
        Ok(config) => {
            if let Err(message) = run(config) {
                eprintln!("hybrid sweep failed: {message}");
                std::process::exit(1);
            }
        }
        Err(message) => {
            eprintln!("{message}");
            eprintln!("usage: axon-uic-hybrid-sweep [--mib 1..256] [--runs 1..10]");
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
    if !words.is_multiple_of(SHARD_COUNT) {
        return Err("vector length must divide into fixed shards".to_owned());
    }
    let shard_words = words / SHARD_COUNT;
    let seed = build_data(words)?;
    let updates = build_mixed_updates(shard_words)?;
    let final_updates =
        coalesce_adjacent_at_frontier(&updates, ObservationFrontier::FinalStateOnly)
            .map_err(|error| format!("cannot coalesce final-state workload: {error:?}"))?
            .len();
    let template = compile_changes(&updates, words, shard_words)?;

    println!("# AXON-UIC Hybrid Recompute physical sweep");
    println!(
        "host {} {}, {} MiB vector, {} shards of {} KiB, {} runs",
        env::consts::OS,
        env::consts::ARCH,
        config.mib,
        SHARD_COUNT,
        shard_words * size_of::<u64>() / 1024,
        config.runs
    );
    println!(
        "mixed changes: {} raw events, {} final writes; plan SKIP {}, RAW_DELTA {}, COALESCED_DELTA {}, FULL_LOCAL {}",
        updates.len(),
        final_updates,
        template.counts.skip,
        template.counts.raw_delta,
        template.counts.coalesced_delta,
        template.counts.full_local
    );
    println!(
        "protocol: hybrid timing includes validation, run indexing, coalescing, classification and local execution; clones and maintained initial sums stay outside every timed path."
    );

    let mut full_samples = Vec::with_capacity(config.runs);
    let mut delta_samples = Vec::with_capacity(config.runs);
    let mut coalesced_samples = Vec::with_capacity(config.runs);
    let mut hybrid_samples = Vec::with_capacity(config.runs);
    let mut compile_samples = Vec::with_capacity(config.runs);
    let mut checksum = 0_u64;

    for round in 0..config.runs {
        let (full, delta, coalesced, hybrid) = if round % 2 == 0 {
            let full = run_full_global(&seed, &updates)?;
            let delta = run_delta_global(&seed, &updates)?;
            let coalesced = run_coalesced_global(&seed, &updates)?;
            let hybrid = run_hybrid(&seed, &updates, shard_words)?;
            (full, delta, coalesced, hybrid)
        } else {
            let hybrid = run_hybrid(&seed, &updates, shard_words)?;
            let coalesced = run_coalesced_global(&seed, &updates)?;
            let delta = run_delta_global(&seed, &updates)?;
            let full = run_full_global(&seed, &updates)?;
            (full, delta, coalesced, hybrid)
        };
        if full.total != delta.total || full.total != coalesced.total || full.total != hybrid.total
        {
            return Err(format!("exact parity failure in round {}", round + 1));
        }
        println!(
            "run {:02}: full {:>8.3} ms; delta {:>8.3} ms; coal {:>8.3} ms; hybrid {:>8.3} ms (compile {:>8.3} ms); parity true",
            round + 1,
            milliseconds(full.duration),
            milliseconds(delta.duration),
            milliseconds(coalesced.duration),
            milliseconds(hybrid.duration),
            milliseconds(hybrid.compile)
        );
        checksum = full.total;
        full_samples.push(full.duration);
        delta_samples.push(delta.duration);
        coalesced_samples.push(coalesced.duration);
        hybrid_samples.push(hybrid.duration);
        compile_samples.push(hybrid.compile);
    }

    let full_p50 = percentile(full_samples, 50);
    let delta_p50 = percentile(delta_samples, 50);
    let coalesced_p50 = percentile(coalesced_samples, 50);
    let hybrid_p50 = percentile(hybrid_samples, 50);
    let compile_p50 = percentile(compile_samples, 50);
    let best_global = full_p50.min(delta_p50).min(coalesced_p50);
    println!("\n| Path | p50 ms |");
    println!("|---|---:|");
    println!("| Full global | {:.3} |", milliseconds(full_p50));
    println!("| Raw Delta global | {:.3} |", milliseconds(delta_p50));
    println!(
        "| Coalesced Delta global | {:.3} |",
        milliseconds(coalesced_p50)
    );
    println!(
        "| Hybrid per shard, end-to-end | {:.3} |",
        milliseconds(hybrid_p50)
    );
    println!(
        "| Hybrid compiler portion | {:.3} |",
        milliseconds(compile_p50)
    );
    println!(
        "| Hybrid vs best global | {:.2}x |",
        speedup(best_global, hybrid_p50)
    );
    println!("| Exact final checksum | {checksum:016X} |");
    println!(
        "\nLimit: one SUM workload with fixed 64-way shards and a deterministic morphology rule. This does not learn thresholds or synthesize maintenance state."
    );
    Ok(())
}

fn build_data(words: usize) -> Result<Vec<u64>, String> {
    let mut data = Vec::new();
    data.try_reserve_exact(words)
        .map_err(|error| format!("cannot allocate benchmark data: {error}"))?;
    let mut state = 0x48_59_42_52_49_44_u64;
    for _ in 0..words {
        state = xorshift(state);
        data.push(state);
    }
    Ok(data)
}

fn build_mixed_updates(shard_words: usize) -> Result<Vec<PointUpdate>, String> {
    let duplicate_keys = (shard_words / 8).max(1);
    let sparse = (shard_words / 16).clamp(1, SPARSE_UPDATES);
    let capacity = DENSE_SHARDS
        .checked_mul(shard_words)
        .and_then(|count| count.checked_add(duplicate_keys * DUPLICATE_RUN))
        .and_then(|count| count.checked_add(sparse))
        .ok_or_else(|| "update count overflow".to_owned())?;
    let mut updates = Vec::new();
    updates
        .try_reserve_exact(capacity)
        .map_err(|error| format!("cannot allocate update stream: {error}"))?;
    let mut event = 0_u64;
    for shard in 0..DENSE_SHARDS {
        let start = shard * shard_words;
        for offset in 0..shard_words {
            updates.push(PointUpdate::new(start + offset, update_value(event)));
            event = event.wrapping_add(1);
        }
    }
    let duplicate_start = DUPLICATE_SHARD * shard_words;
    for offset in 0..duplicate_keys {
        for _ in 0..DUPLICATE_RUN {
            updates.push(PointUpdate::new(
                duplicate_start + offset,
                update_value(event),
            ));
            event = event.wrapping_add(1);
        }
    }
    let sparse_start = SPARSE_SHARD * shard_words;
    for offset in 0..sparse {
        let index = sparse_start + offset.wrapping_mul(97) % shard_words;
        updates.push(PointUpdate::new(index, update_value(event)));
        event = event.wrapping_add(1);
    }
    Ok(updates)
}

fn compile_changes<'a>(
    updates: &'a [PointUpdate],
    words: usize,
    shard_words: usize,
) -> Result<CompiledChanges<'a>, String> {
    let mut counts = PlanCounts::default();
    let mut shards = Vec::with_capacity(SHARD_COUNT);
    let mut cursor = 0;
    for shard in 0..SHARD_COUNT {
        let start = cursor;
        while let Some(update) = updates.get(cursor) {
            if update.index() >= words {
                return Err(format!("update index out of bounds: {}", update.index()));
            }
            let update_shard = update.index() / shard_words;
            if update_shard < shard {
                return Err("shard-indexed change set must be ordered by shard".to_owned());
            }
            if update_shard != shard {
                break;
            }
            cursor += 1;
        }
        let raw = &updates[start..cursor];
        let has_adjacent_duplicates = raw
            .windows(2)
            .any(|pair| pair[0].index() == pair[1].index());
        let storage = if has_adjacent_duplicates {
            ChangeStorage::Owned(
                coalesce_adjacent_at_frontier(raw, ObservationFrontier::FinalStateOnly)
                    .map_err(|error| format!("cannot coalesce final-state shard: {error:?}"))?,
            )
        } else {
            ChangeStorage::Borrowed(raw)
        };
        let strategy = if storage.as_slice().is_empty() {
            ShardStrategy::Skip
        } else if storage.as_slice().len().saturating_mul(2) >= shard_words {
            ShardStrategy::FullLocal
        } else if has_adjacent_duplicates {
            ShardStrategy::CoalescedDelta
        } else {
            ShardStrategy::RawDelta
        };
        counts.record(strategy);
        shards.push(CompiledShard {
            strategy,
            updates: storage,
        });
    }
    if cursor != updates.len() {
        return Err("shard-indexed change set must be ordered by shard".to_owned());
    }
    Ok(CompiledChanges { shards, counts })
}

fn run_full_global(seed: &[u64], updates: &[PointUpdate]) -> Result<Measurement, String> {
    let mut data = clone_data(seed)?;
    let started = Instant::now();
    apply_updates(&mut data, updates)?;
    let total = checksum(&data);
    black_box(total);
    Ok(Measurement {
        duration: started.elapsed(),
        total,
    })
}

fn run_delta_global(seed: &[u64], updates: &[PointUpdate]) -> Result<Measurement, String> {
    let mut data = clone_data(seed)?;
    let mut total = checksum(&data);
    let started = Instant::now();
    total = apply_delta(&mut data, updates, total)?;
    black_box(total);
    let duration = started.elapsed();
    verify_total(&data, total)?;
    Ok(Measurement { duration, total })
}

fn run_coalesced_global(seed: &[u64], updates: &[PointUpdate]) -> Result<Measurement, String> {
    let mut data = clone_data(seed)?;
    let mut total = checksum(&data);
    let started = Instant::now();
    let coalesced = coalesce_adjacent_at_frontier(updates, ObservationFrontier::FinalStateOnly)
        .map_err(|error| format!("cannot coalesce final-state updates: {error:?}"))?;
    total = apply_delta(&mut data, &coalesced, total)?;
    black_box(total);
    let duration = started.elapsed();
    verify_total(&data, total)?;
    Ok(Measurement { duration, total })
}

fn run_hybrid(
    seed: &[u64],
    updates: &[PointUpdate],
    shard_words: usize,
) -> Result<HybridMeasurement, String> {
    let mut data = clone_data(seed)?;
    let mut shard_totals: Vec<_> = data.chunks_exact(shard_words).map(checksum).collect();
    let started = Instant::now();
    let compiled = compile_changes(updates, data.len(), shard_words)?;
    let compile = started.elapsed();
    for (shard, change) in compiled.shards.iter().enumerate() {
        let start = shard * shard_words;
        let end = start + shard_words;
        match change.strategy {
            ShardStrategy::Skip => {}
            ShardStrategy::RawDelta | ShardStrategy::CoalescedDelta => {
                shard_totals[shard] =
                    apply_delta(&mut data, change.updates.as_slice(), shard_totals[shard])?;
            }
            ShardStrategy::FullLocal => {
                apply_updates(&mut data, change.updates.as_slice())?;
                shard_totals[shard] = checksum(&data[start..end]);
            }
        }
    }
    let total = shard_totals
        .iter()
        .copied()
        .fold(0_u64, |sum, value| sum.wrapping_add(value));
    black_box(total);
    let duration = started.elapsed();
    verify_total(&data, total)?;
    Ok(HybridMeasurement {
        duration,
        compile,
        total,
    })
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

fn apply_delta(data: &mut [u64], updates: &[PointUpdate], total: u64) -> Result<u64, String> {
    let mut total = total;
    for update in updates {
        let old = *data
            .get(update.index())
            .ok_or_else(|| format!("update index out of bounds: {}", update.index()))?;
        data[update.index()] = update.value();
        total = total.wrapping_sub(old).wrapping_add(update.value());
    }
    Ok(total)
}

fn verify_total(data: &[u64], total: u64) -> Result<(), String> {
    (checksum(data) == total)
        .then_some(())
        .ok_or_else(|| "incremental total diverged from exact checksum".to_owned())
}

fn checksum(data: &[u64]) -> u64 {
    data.iter()
        .copied()
        .fold(0_u64, |sum, value| sum.wrapping_add(value))
}

fn update_value(event: u64) -> u64 {
    xorshift(event.wrapping_add(0xC0_A1_E5_C0_A1_E5))
}

fn xorshift(mut value: u64) -> u64 {
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}

fn percentile(mut values: Vec<Duration>, percentile: usize) -> Duration {
    values.sort_unstable();
    values[(values.len() * percentile).div_ceil(100).saturating_sub(1)]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn speedup(baseline: Duration, candidate: Duration) -> f64 {
    baseline.as_nanos() as f64 / candidate.as_nanos().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_selects_all_four_local_strategies() {
        let shard_words = 128;
        let updates = build_mixed_updates(shard_words).unwrap();
        let compiled = compile_changes(&updates, shard_words * SHARD_COUNT, shard_words).unwrap();

        assert_eq!(compiled.counts.full_local, DENSE_SHARDS);
        assert_eq!(compiled.counts.coalesced_delta, 1);
        assert_eq!(compiled.counts.raw_delta, 1);
        assert_eq!(compiled.counts.skip, SHARD_COUNT - DENSE_SHARDS - 2);
    }

    #[test]
    fn hybrid_matches_all_global_paths() {
        let shard_words = 128;
        let seed = build_data(shard_words * SHARD_COUNT).unwrap();
        let updates = build_mixed_updates(shard_words).unwrap();
        let full = run_full_global(&seed, &updates).unwrap();
        let delta = run_delta_global(&seed, &updates).unwrap();
        let coalesced = run_coalesced_global(&seed, &updates).unwrap();
        let hybrid = run_hybrid(&seed, &updates, shard_words).unwrap();

        assert_eq!(full.total, delta.total);
        assert_eq!(full.total, coalesced.total);
        assert_eq!(full.total, hybrid.total);
    }

    #[test]
    fn compiler_rejects_out_of_bounds_updates() {
        let update = PointUpdate::new(128, 1);

        assert!(compile_changes(&[update], 128, 2).is_err());
    }

    #[test]
    fn compiler_rejects_unordered_shards_and_cli_rejects_bad_bounds() {
        let updates = [PointUpdate::new(4, 1), PointUpdate::new(2, 2)];

        assert!(compile_changes(&updates, 128, 2).is_err());
        assert!(parse_bounded("0", 1, 10, "--runs").is_err());
        assert!(parse_bounded("11", 1, 10, "--runs").is_err());
    }
}
