mod live_support;

use axon_uic::{AverageStrategy, LiveAverage, OnlineAveragePolicy, Replace, ReplaceDelta};
use live_support::*;
use std::{hint::black_box, path::PathBuf, time::Instant};

struct Options {
    epochs: usize,
    seed: u64,
    output: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    resume: Option<PathBuf>,
    context: String,
    spread: bool,
    selected_feedback: bool,
}

#[derive(Default)]
struct Scores {
    cases: u64,
    decisive: u64,
    correct: u64,
    full_ns: u128,
    incremental_ns: u128,
    selected_ns: u128,
    oracle_ns: u128,
    selection_ns: u128,
    incremental_choices: u64,
}

impl Scores {
    fn record(&mut self, choice: AverageStrategy, incremental: u64, full: u64, selection: u64) {
        self.cases += 1;
        self.full_ns += full as u128;
        self.incremental_ns += incremental as u128;
        self.oracle_ns += incremental.min(full) as u128;
        self.selection_ns += selection as u128;
        self.selected_ns += selection as u128
            + match choice {
                AverageStrategy::Incremental => {
                    self.incremental_choices += 1;
                    incremental as u128
                }
                AverageStrategy::Full => full as u128,
            };
        if incremental.abs_diff(full) as u128 * 10 >= incremental.min(full) as u128 {
            self.decisive += 1;
            if (incremental < full) == (choice == AverageStrategy::Incremental) {
                self.correct += 1;
            }
        }
    }

    fn json(&self) -> String {
        format!(
            "{{\"cases\":{},\"decisive_cases\":{},\"correct_decisions\":{},\"incremental_choices\":{},\"always_full_ns\":{},\"always_incremental_ns\":{},\"selected_with_routing_ns\":{},\"oracle_ns\":{},\"routing_ns\":{}}}",
            self.cases,
            self.decisive,
            self.correct,
            self.incremental_choices,
            self.full_ns,
            self.incremental_ns,
            self.selected_ns,
            self.oracle_ns,
            self.selection_ns
        )
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("online training: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let hardware =
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| std::env::consts::ARCH.into());
    let mut options = Options {
        epochs: 32,
        seed: 20260913,
        output: None,
        checkpoint: None,
        resume: None,
        context: hardware,
        spread: true,
        selected_feedback: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            println!(
                "axon-uic-online-train [--epochs 0..1000] [--seed N] [--context hardware-id] [--layout spread|contiguous] [--feedback paired|selected] [--checkpoint new-file] [--resume file] [--output report.json]"
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing option value")?;
        match flag.as_str() {
            "--epochs" => options.epochs = bounded(&value, 0, 1000)?,
            "--seed" => options.seed = value.parse()?,
            "--context" => options.context = value,
            "--feedback" => {
                options.selected_feedback = match value.as_str() {
                    "paired" => false,
                    "selected" => true,
                    _ => return Err("unknown feedback mode".into()),
                }
            }
            "--layout" => {
                options.spread = match value.as_str() {
                    "spread" => true,
                    "contiguous" => false,
                    _ => return Err("unknown layout".into()),
                }
            }
            "--checkpoint" => options.checkpoint = Some(value.into()),
            "--resume" => options.resume = Some(value.into()),
            "--output" => options.output = Some(value.into()),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let context = format!(
        "{}/avg-paired-v1/{}/{}",
        options.context,
        if options.spread {
            "spread"
        } else {
            "contiguous"
        },
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    let mut policy = if let Some(path) = &options.resume {
        OnlineAveragePolicy::load(path, &context)?
    } else {
        OnlineAveragePolicy::new(&context)
    };
    let start = Instant::now();
    let mut rng = Rng(options.seed);
    let mut progressive = Scores::default();
    let mut training_records = Vec::new();
    let training_sizes = [256, 4096, 65536, 262144];
    let mut sources: Vec<_> = training_sizes
        .iter()
        .map(|n| {
            let live = materialize(*n, rng.next());
            (live.clone(), live)
        })
        .collect();
    for epoch in 0..options.epochs {
        // Rotate context order; data and update positions change independently of labels.
        for offset in 0..16 {
            let cell = (offset + epoch * 7) % 16;
            let n = training_sizes[cell / 4];
            let k = updates(n, cell % 4);
            let (live, full) = &mut sources[cell / 4];
            let delta = batch(live.values(), k, &mut rng, options.spread);
            let (choice, routing) = choose(&policy, n, k, options.selected_feedback);
            let (incremental, recompute) = measure_pair(live, full, &delta, epoch + offset);
            progressive.record(choice, incremental, recompute, routing);
            // Prediction precedes both observations; holdout below never calls observe.
            if !options.selected_feedback || choice == AverageStrategy::Incremental {
                policy.observe(n, k, AverageStrategy::Incremental, incremental);
            }
            if !options.selected_feedback || choice == AverageStrategy::Full {
                policy.observe(n, k, AverageStrategy::Full, recompute);
            }
            training_records.push(format!(
                "[{epoch},{n},{k},{incremental},{recompute},{routing},{}]",
                usize::from(choice == AverageStrategy::Incremental)
            ));
        }
    }
    let training_ns = start.elapsed().as_nanos();
    if let Some(checkpoint) = &options.checkpoint {
        make_parent(checkpoint)?;
        policy.save(checkpoint)?;
        assert_eq!(OnlineAveragePolicy::load(checkpoint, &context)?, policy);
    }
    let frozen = policy.clone();
    let mut heldout = Scores::default();
    let mut holdout_rng = Rng(options.seed ^ 0xD1B54A32D192ED03);
    let mut records = Vec::new();
    let mut executed_policy_ns = 0_u128;
    for n in [384, 6144, 98304, 393216] {
        let mut live = materialize(n, holdout_rng.next());
        let mut full = live.clone();
        let mut selected = live.clone();
        for density in 0..4 {
            let k = updates(n, density);
            for trial in 0..16 {
                let delta = batch(live.values(), k, &mut holdout_rng, options.spread);
                let (choice, routing) = choose(&policy, n, k, false);
                let mut selected_ns = 0;
                if trial % 3 == 0 {
                    selected_ns = probe_policy(&mut selected, &delta, &policy);
                }
                let (incremental, recompute) = measure_pair(&mut live, &mut full, &delta, trial);
                if trial % 3 != 0 {
                    selected_ns = probe_policy(&mut selected, &delta, &policy);
                }
                assert_eq!(selected.query(), live.query());
                assert_eq!(selected.values(), live.values());
                executed_policy_ns += selected_ns as u128;
                heldout.record(choice, incremental, recompute, routing);
                records.push(format!(
                    "[{n},{k},{incremental},{recompute},{routing},{},{selected_ns}]",
                    usize::from(choice == AverageStrategy::Incremental)
                ));
            }
        }
    }
    assert_eq!(policy, frozen, "heldout evaluation must not train");
    println!(
        "Online AVG policy; seed {}; epochs {}; measured observations {}; checkpoint {} bytes",
        options.seed,
        options.epochs,
        policy.observations(),
        OnlineAveragePolicy::checkpoint_bytes()
    );
    println!(
        "Training: paired real executions, predict before observe; four known size bands and four delta densities."
    );
    println!(
        "Learning feedback: {}; paired reference measurements remain experiment costs.",
        if options.selected_feedback {
            "selected action only, exploration every 16 observations"
        } else {
            "both strategies"
        }
    );
    println!(
        "Holdout: 256 new workloads, different sizes/data/positions, same AVG family and layout; frozen policy."
    );
    println!("| Metric | Progressive training | Frozen holdout |");
    println!("|---|---:|---:|");
    println!(
        "| Correct decisive choices | {}/{} | {}/{} |",
        progressive.correct, progressive.decisive, heldout.correct, heldout.decisive
    );
    println!(
        "| Incremental choices | {}/{} | {}/{} |",
        progressive.incremental_choices,
        progressive.cases,
        heldout.incremental_choices,
        heldout.cases
    );
    println!(
        "| Always Full (sum of per-batch costs), ms | {:.3} | {:.3} |",
        progressive.full_ns as f64 / 1e6,
        heldout.full_ns as f64 / 1e6
    );
    println!(
        "| Always Incremental, ms | {:.3} | {:.3} |",
        progressive.incremental_ns as f64 / 1e6,
        heldout.incremental_ns as f64 / 1e6
    );
    println!(
        "| Estimated chosen cost including routing, ms | {:.3} | {:.3} |",
        progressive.selected_ns as f64 / 1e6,
        heldout.selected_ns as f64 / 1e6
    );
    println!(
        "| Executed policy including routing, ms | not measured | {:.3} |",
        executed_policy_ns as f64 / 1e6
    );
    println!(
        "| Oracle choice excluding routing, ms | {:.3} | {:.3} |",
        progressive.oracle_ns as f64 / 1e6,
        heldout.oracle_ns as f64 / 1e6
    );
    println!(
        "Training wall time {:.3} s; exact parity true; learned/fixed incremental ratio {:.3}",
        training_ns as f64 / 1e9,
        executed_policy_ns as f64 / heldout.incremental_ns as f64
    );
    println!(
        "Scope: empirical strategy selection. Two preimplemented algorithms; no new algorithms, language, perception, or AGI demonstrated."
    );
    if let Some(output) = &options.output {
        make_parent(output)?;
        let json = format!(
            "{{\"schema\":\"axon-online-avg-v2\",\"seed\":{},\"epochs\":{},\"training_wall_ns\":{training_ns},\"executed_policy_ns\":{executed_policy_ns},\"checkpoint_bytes\":{},\"observations\":{},\"policy_bytes_in_memory\":{},\"spread\":{},\"selected_feedback\":{},\"training\":{},\"holdout\":{},\"training_columns\":[\"epoch\",\"values\",\"updates\",\"incremental_ns\",\"full_ns\",\"routing_ns\",\"chose_incremental\"],\"training_records\":[{}],\"holdout_columns\":[\"values\",\"updates\",\"incremental_ns\",\"full_ns\",\"routing_ns\",\"chose_incremental\",\"executed_policy_ns\"],\"holdout_records\":[{}],\"exact\":true}}\n",
            options.seed,
            options.epochs,
            OnlineAveragePolicy::checkpoint_bytes(),
            policy.observations(),
            std::mem::size_of::<OnlineAveragePolicy>(),
            options.spread,
            options.selected_feedback,
            progressive.json(),
            heldout.json(),
            training_records.join(","),
            records.join(",")
        );
        std::fs::write(output, json)?;
    }
    Ok(())
}

fn make_parent(path: &std::path::Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn updates(n: usize, density: usize) -> usize {
    match density {
        0 => n / 128,
        1 => n / 8,
        2 => n / 2,
        _ => n,
    }
}

fn choose(
    policy: &OnlineAveragePolicy,
    n: usize,
    k: usize,
    explore: bool,
) -> (AverageStrategy, u64) {
    let start = Instant::now();
    let choice = if explore {
        black_box(policy).explore(black_box(n), black_box(k))
    } else {
        black_box(policy).choose(black_box(n), black_box(k))
    };
    black_box(choice);
    (choice, start.elapsed().as_nanos() as u64)
}

fn measure_pair(
    live: &mut LiveAverage,
    full: &mut LiveAverage,
    delta: &ReplaceDelta,
    order: usize,
) -> (u64, u64) {
    let reverse = ReplaceDelta::try_new(
        delta
            .changes()
            .iter()
            .map(|r| Replace::new(r.index(), r.new_value(), r.old()))
            .collect(),
    )
    .unwrap();
    let incremental;
    let recompute;
    if order.is_multiple_of(2) {
        incremental = probe(live, delta, &reverse, AverageStrategy::Incremental);
        recompute = probe(full, delta, &reverse, AverageStrategy::Full);
    } else {
        recompute = probe(full, delta, &reverse, AverageStrategy::Full);
        incremental = probe(live, delta, &reverse, AverageStrategy::Incremental);
    }
    // Advance the stream after timing. Audit the changed state, not just the
    // round-trip endpoint where compensating errors could cancel.
    live.apply_with_strategy(live.last_event() + 1, delta, AverageStrategy::Incremental)
        .unwrap();
    full.apply_with_strategy(full.last_event() + 1, delta, AverageStrategy::Full)
        .unwrap();
    assert_eq!(live.query(), full.query());
    assert_eq!(live.values(), full.values());
    audit(live);
    (incremental, recompute)
}

fn probe(
    live: &mut LiveAverage,
    delta: &ReplaceDelta,
    reverse: &ReplaceDelta,
    strategy: AverageStrategy,
) -> u64 {
    let repeats = if live.values().len() <= 8192 { 32 } else { 4 };
    let start = Instant::now();
    for _ in 0..repeats {
        for changes in [delta, reverse] {
            let event = live.last_event() + 1;
            black_box(
                live.apply_with_strategy(event, black_box(changes), strategy)
                    .unwrap(),
            );
            black_box(live.query());
        }
    }
    (start.elapsed().as_nanos() / (repeats * 2)).max(1) as u64
}

fn probe_policy(live: &mut LiveAverage, delta: &ReplaceDelta, policy: &OnlineAveragePolicy) -> u64 {
    let reverse = ReplaceDelta::try_new(
        delta
            .changes()
            .iter()
            .map(|r| Replace::new(r.index(), r.new_value(), r.old()))
            .collect(),
    )
    .unwrap();
    let repeats = if live.values().len() <= 8192 { 32 } else { 4 };
    let start = Instant::now();
    for _ in 0..repeats {
        for changes in [delta, &reverse] {
            let strategy = black_box(policy).choose(
                black_box(live.values().len()),
                black_box(changes.changes().len()),
            );
            let event = live.last_event() + 1;
            black_box(
                live.apply_with_strategy(event, black_box(changes), strategy)
                    .unwrap(),
            );
            black_box(live.query());
        }
    }
    let elapsed = (start.elapsed().as_nanos() / (repeats * 2)).max(1) as u64;
    let strategy = policy.choose(live.values().len(), delta.changes().len());
    live.apply_with_strategy(live.last_event() + 1, delta, strategy)
        .unwrap();
    elapsed
}
