mod live_support;

use axon_uic::{AverageStrategy, LiveAverage, LiveAverageStore, LiveBenchContract};
use live_support::*;
use std::{hint::black_box, path::PathBuf, time::Instant};

struct Options {
    sizes: Vec<usize>,
    batches: usize,
    updates: usize,
    paired: usize,
    seed: u64,
    output: Option<PathBuf>,
    durable: Option<PathBuf>,
    spread: bool,
    checkpoint_every: usize,
    durable_batches: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("live bench: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = Options {
        sizes: vec![1, 16, 64, 256],
        batches: 10_000,
        updates: 1024,
        paired: 128,
        seed: 20260913,
        output: None,
        durable: None,
        spread: true,
        checkpoint_every: 0,
        durable_batches: 256,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "axon-uic-live-avg [--mib 1,16,64,256,1024] [--batches 10000] [--updates 1024] [--paired-batches 128] [--seed N] [--layout spread|contiguous] [--output report.json] [--durable new-file] [--durable-batches 256] [--checkpoint-every 0]"
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing option value")?;
        match flag.as_str() {
            "--mib" => {
                options.sizes = value
                    .split(',')
                    .map(|v| bounded(v, 1, 1024))
                    .collect::<Result<_, _>>()?
            }
            "--batches" => options.batches = bounded(&value, 1, 1_000_000)?,
            "--updates" => options.updates = bounded(&value, 0, 8_388_608)?,
            "--paired-batches" => options.paired = bounded(&value, 1, 10_000)?,
            "--seed" => options.seed = value.parse()?,
            "--durable-batches" => options.durable_batches = bounded(&value, 1, 10000)?,
            "--checkpoint-every" => options.checkpoint_every = bounded(&value, 0, 10000)?,
            "--layout" => {
                options.spread = match value.as_str() {
                    "spread" => true,
                    "contiguous" => false,
                    _ => return Err("unknown layout".into()),
                }
            }
            "--output" => options.output = Some(value.into()),
            "--durable" => options.durable = Some(value.into()),
            _ => return Err(format!("unknown option: {flag}").into()),
        }
    }
    if options
        .sizes
        .iter()
        .any(|mib| options.updates > mib * 131072)
    {
        return Err("updates exceed smallest source".into());
    }
    println!(
        "Live AVG; seed {}; batches {}; updates {}; layout {}; paired prefix {}",
        options.seed,
        options.batches,
        options.updates,
        if options.spread {
            "spread"
        } else {
            "contiguous"
        },
        options.paired.min(options.batches)
    );
    println!("RAM experiment: generation and audit measured separately; no disk writes per batch.");
    println!(
        "| MiB | Materialize ms | Apply+query p50 us | p95 us | Full paired p50 us | Measured break-even batch | Exact |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---|");
    let mut records = Vec::new();
    for &mib in &options.sizes {
        records.push(measure(mib, &options)?);
    }
    let durable_record = if let Some(path) = &options.durable {
        Some(measure_durable(path, &options)?)
    } else {
        None
    };
    if let Some(output) = &options.output {
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let json = format!(
            "{{\n\"schema\":\"axon-live-avg-v1\",\"seed\":{},\"batches\":{},\"updates\":{},\"spread\":{},\"records\":[{}],\"durable\":{}\n}}\n",
            options.seed,
            options.batches,
            options.updates,
            options.spread,
            records.join(",\n"),
            durable_record.unwrap_or_else(|| "null".into())
        );
        std::fs::write(output, json)?;
    }
    Ok(())
}

fn measure(mib: usize, options: &Options) -> Result<String, Box<dyn std::error::Error>> {
    let mut rng = Rng(options.seed);
    let start = Instant::now();
    let values: Vec<u64> = (0..mib * 131072).map(|_| rng.next()).collect();
    let generation_ns = start.elapsed().as_nanos();
    let full_values = values.clone();
    let artifact_start = Instant::now();
    let artifact = semantic();
    let artifact_ns = artifact_start.elapsed().as_nanos();
    let start = Instant::now();
    let mut live = LiveAverage::materialize(options.seed, values, artifact)?;
    let materialize_ns = start.elapsed().as_nanos() as u64;
    let mut full = LiveAverage::materialize(options.seed, full_values, artifact)?;
    let mut deltas = Vec::with_capacity(options.batches);
    let mut full_times = Vec::new();
    let mut paired_delta_ns = 0_u128;
    let mut full_total_ns = 0_u128;
    let mut delta_generation_ns = 0_u128;
    let mut break_even = None;
    for index in 0..options.batches {
        let start = Instant::now();
        let delta = batch(live.values(), options.updates, &mut rng, options.spread);
        delta_generation_ns += start.elapsed().as_nanos();
        let delta_ns;
        if index < options.paired {
            let full_ns;
            if index % 2 == 0 {
                full_ns = timed_apply(&mut full, &delta, AverageStrategy::Full);
                delta_ns = timed_apply(&mut live, &delta, AverageStrategy::Incremental);
            } else {
                delta_ns = timed_apply(&mut live, &delta, AverageStrategy::Incremental);
                full_ns = timed_apply(&mut full, &delta, AverageStrategy::Full);
            }
            assert_eq!(full.query(), live.query());
            full_times.push(full_ns);
            paired_delta_ns += delta_ns as u128;
            full_total_ns += full_ns as u128;
            if break_even.is_none()
                && materialize_ns as u128 + artifact_ns + paired_delta_ns < full_total_ns
            {
                break_even = Some(index + 1);
            }
        } else {
            delta_ns = timed_apply(&mut live, &delta, AverageStrategy::Incremental);
        }
        deltas.push(delta_ns);
    }
    let start = Instant::now();
    audit(&live);
    let audit_ns = start.elapsed().as_nanos();
    // Grouped reads keep timer resolution from dominating the O(1) query measurement.
    let query_reads = 100_000;
    let start = Instant::now();
    for _ in 0..query_reads {
        black_box(black_box(&live).query());
    }
    let query_ns_per_read = start.elapsed().as_nanos() as f64 / query_reads as f64;
    let p50 = percentile(&deltas, 50);
    let p95 = percentile(&deltas, 95);
    let full_p50 = percentile(&full_times, 50);
    let steady_ns: u128 = deltas.iter().map(|v| *v as u128).sum();
    let contract = LiveBenchContract {
        setup: std::time::Duration::from_nanos(materialize_ns + artifact_ns as u64),
        steady: std::time::Duration::from_nanos(steady_ns as u64),
        audit: std::time::Duration::from_nanos(audit_ns as u64),
        ..LiveBenchContract::default()
    };
    let production_ns = contract
        .production()
        .expect("bounded experiment duration")
        .as_nanos();
    let break_even_text = break_even.map_or("null".into(), |v| v.to_string());
    println!(
        "| {mib} | {:.3} | {:.3} | {:.3} | {:.3} | {} | true |",
        materialize_ns as f64 / 1e6,
        p50 as f64 / 1000.0,
        p95 as f64 / 1000.0,
        full_p50 as f64 / 1000.0,
        break_even_text
    );
    Ok(format!(
        "{{\"mib\":{mib},\"source_generation_ns\":{generation_ns},\"artifact_ns\":{artifact_ns},\"materialize_ns\":{materialize_ns},\"delta_generation_ns\":{delta_generation_ns},\"production_ram_ns\":{production_ns},\"steady_total_ns\":{steady_ns},\"steady_p50_ns\":{p50},\"steady_p95_ns\":{p95},\"full_p50_ns\":{full_p50},\"paired_batches\":{},\"paired_delta_ns\":{paired_delta_ns},\"paired_full_ns\":{full_total_ns},\"break_even\":{break_even_text},\"query_ns_per_read\":{query_ns_per_read:.3},\"audit_ns\":{audit_ns},\"exact\":true,\"state_bytes\":{},\"steady_samples_ns\":{:?},\"full_samples_ns\":{:?}}}",
        full_times.len(),
        std::mem::size_of::<LiveAverage>(),
        deltas,
        full_times
    ))
}

fn measure_durable(
    path: &PathBuf,
    options: &Options,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let source = materialize(131072, options.seed);
    let start = Instant::now();
    let mut store = LiveAverageStore::create(path, source)?;
    let create_ns = start.elapsed().as_nanos();
    let mut rng = Rng(options.seed ^ 77);
    let mut times = Vec::new();
    let mut recovery_ns = 0_u128;
    let batches = options.durable_batches;
    let updates = options.updates.min(131072);
    let mut restarts = 0;
    let mut checkpoints = 0;
    let mut checkpoint_ns = 0_u128;
    let mut bytes_reclaimed = 0_u64;
    let mut peak_file_bytes = std::fs::metadata(path)?.len();
    for index in 0..batches {
        let state = store.state()?;
        let delta = batch(
            state.values(),
            options.updates.min(state.values().len()),
            &mut rng,
            options.spread,
        );
        let start = Instant::now();
        store.apply(options.seed, index as u64, index as u64 + 1, &delta)?;
        black_box(store.state()?.query());
        times.push(start.elapsed().as_nanos() as u64);
        peak_file_bytes = peak_file_bytes.max(std::fs::metadata(path)?.len());
        if options.checkpoint_every != 0 && (index + 1).is_multiple_of(options.checkpoint_every) {
            let start = Instant::now();
            let report = store.compact()?;
            checkpoint_ns += start.elapsed().as_nanos();
            bytes_reclaimed += report.bytes_before.saturating_sub(report.bytes_after);
            checkpoints += 1;
        }
        if rng.next().is_multiple_of(13) || index + 1 == batches {
            let expected = store.state()?.clone();
            drop(store);
            let start = Instant::now();
            let (recovered, _) = LiveAverageStore::open(path)?;
            recovery_ns += start.elapsed().as_nanos();
            assert_eq!(recovered.state()?, &expected);
            store = recovered;
            restarts += 1;
        }
    }
    audit(store.state()?);
    let p50 = percentile(&times, 50);
    let final_file_bytes = std::fs::metadata(path)?.len();
    let durable_total_ns: u128 = times.iter().map(|ns| *ns as u128).sum();
    println!(
        "Durable 1 MiB: {batches} synced batches, p50 {:.3} ms, {restarts} reopen/replay checks, exact=true",
        p50 as f64 / 1e6
    );
    println!(
        "Checkpoints: {checkpoints}; total {:.3} ms; final file {final_file_bytes} bytes; peak active file {peak_file_bytes} bytes; recovery total {:.3} ms",
        checkpoint_ns as f64 / 1e6,
        recovery_ns as f64 / 1e6
    );
    Ok(format!(
        "{{\"mib\":1,\"batches\":{batches},\"updates\":{updates},\"create_ns\":{create_ns},\"synced_p50_ns\":{p50},\"durable_total_ns\":{durable_total_ns},\"restarts\":{restarts},\"recovery_total_ns\":{recovery_ns},\"checkpoint_every\":{},\"checkpoints\":{checkpoints},\"checkpoint_total_ns\":{checkpoint_ns},\"bytes_reclaimed\":{bytes_reclaimed},\"final_file_bytes\":{final_file_bytes},\"peak_active_file_bytes\":{peak_file_bytes},\"exact\":true}}",
        options.checkpoint_every
    ))
}
