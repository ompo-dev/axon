mod live_support;

use axon_uic::{AverageStrategy, LiveAverage, OnlineAveragePolicy, Replace, ReplaceDelta};
use live_support::{Rng, batch, bounded, materialize, percentile};
use std::{fmt::Write as _, hint::black_box, path::PathBuf, time::Instant};

const PHASES: [&str; 8] = [
    "sparse_spread",
    "dense_spread",
    "sparse_contiguous",
    "dense_contiguous",
    "empty",
    "mixed",
    "boundary_density",
    "hotspot",
];
const TRAIN_SIZES: [usize; 4] = [384, 6144, 98304, 393216];
const HOLDOUT_SIZES: [usize; 4] = [511, 8191, 131071, 524287];
const CONTEXT: &str = "axon-online-stream-v2/mixed-layout/selected-only";

#[derive(Default)]
struct Scores {
    samples: [Vec<u64>; 4],
    choices: [usize; 2],
}

impl Scores {
    fn record(&mut self, elapsed: [u64; 4], strategy: AverageStrategy) {
        for (samples, elapsed) in self.samples.iter_mut().zip(elapsed) {
            samples.push(elapsed);
        }
        self.choices[usize::from(strategy == AverageStrategy::Full)] += 1;
    }
    fn totals(&self) -> [u128; 4] {
        self.samples
            .each_ref()
            .map(|samples| samples.iter().map(|n| u128::from(*n)).sum())
    }
    fn json(&self) -> String {
        let totals = self.totals();
        let p50 = self.samples.each_ref().map(|s| percentile(s, 50));
        let p95 = self.samples.each_ref().map(|s| percentile(s, 95));
        format!(
            "{{\"cases\":{},\"totals_ns\":{totals:?},\"p50_ns\":{p50:?},\"p95_ns\":{p95:?},\"policy_choices_incremental_full\":{:?}}}",
            self.samples[0].len(),
            self.choices
        )
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("stream benchmark: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut seed = 20260913_u64;
    let mut steps = 256;
    let mut output: Option<PathBuf> = None;
    let mut checkpoint: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "axon-uic-online-stream [--seed N] [--steps 4..4096 (multiple of 4)] [--output report.json] [--checkpoint new.policy]"
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing value")?;
        match flag.as_str() {
            "--seed" => seed = value.parse()?,
            "--steps" => steps = bounded(&value, 4, 4096)?,
            "--output" => output = Some(value.into()),
            "--checkpoint" => checkpoint = Some(value.into()),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    if !steps.is_multiple_of(4) {
        return Err("steps must be a multiple of 4 for balanced execution order".into());
    }
    let hardware =
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| std::env::consts::ARCH.into());
    let context = format!(
        "{CONTEXT}/{hardware}/{}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    let mut policy = OnlineAveragePolicy::new(&context);
    let start = Instant::now();
    let mut records = String::new();
    let (training, training_phases) =
        evaluate(&mut policy, TRAIN_SIZES, steps, seed, true, &mut records);
    let training_wall_ns = start.elapsed().as_nanos();
    let frozen = policy.clone();
    let start = Instant::now();
    let (holdout, holdout_phases) = evaluate(
        &mut policy,
        HOLDOUT_SIZES,
        steps,
        seed ^ 0xa0761d6478bd642f,
        false,
        &mut records,
    );
    let holdout_wall_ns = start.elapsed().as_nanos();
    assert_eq!(policy, frozen, "holdout must not update cost estimates");
    assert_eq!(policy.observations(), (steps * 32) as u64);
    if let Some(path) = checkpoint {
        make_parent(&path)?;
        policy.save(&path)?;
        assert_eq!(policy, OnlineAveragePolicy::load(path, &context)?);
    }
    for (name, scores) in [("online_training", &training), ("frozen_holdout", &holdout)] {
        let [delta, full, heuristic, learned] = scores.totals();
        println!(
            "{name}: {} batches; Delta={:.3}ms Full={:.3}ms heuristic={:.3}ms learned={:.3}ms; learned/heuristic={:.4}",
            scores.samples[0].len(),
            delta as f64 / 1e6,
            full as f64 / 1e6,
            heuristic as f64 / 1e6,
            learned as f64 / 1e6,
            learned as f64 / heuristic as f64
        );
    }
    println!(
        "{} selected-only observations; {}-byte policy; exact oracle/source parity for all four candidates; balanced order in every phase and size",
        policy.observations(),
        OnlineAveragePolicy::checkpoint_bytes()
    );
    if let Some(path) = output {
        make_parent(&path)?;
        let exact_checks = steps * 32 * 2 * 4;
        let json = format!(
            "{{\"schema\":\"axon-online-stream-v2\",\"seed\":{seed},\"steps_per_size_phase\":{steps},\"training_wall_ns\":{training_wall_ns},\"holdout_wall_ns\":{holdout_wall_ns},\"observations\":{},\"policy_memory_bytes\":{},\"checkpoint_bytes\":{},\"exact_checks\":{exact_checks},\"exact\":true,\"candidate_order\":[\"incremental\",\"full\",\"density_heuristic\",\"learned\"],\"order_offsets\":[0,1,3,2],\"heuristic\":\"incremental if k <= floor(n/4), else full\",\"training\":{},\"holdout\":{},\"training_phases\":[{}],\"holdout_phases\":[{}],\"columns\":[\"training\",\"phase\",\"n\",\"k\",\"first_candidate\",\"incremental_ns\",\"full_ns\",\"heuristic_ns\",\"learned_ns\",\"learned_full\"],\"records\":[{records}]}}\n",
            policy.observations(),
            std::mem::size_of::<OnlineAveragePolicy>(),
            OnlineAveragePolicy::checkpoint_bytes(),
            training.json(),
            holdout.json(),
            training_phases.join(","),
            holdout_phases.join(",")
        );
        std::fs::write(path, json)?;
    }
    Ok(())
}

fn make_parent(path: &std::path::Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn evaluate(
    policy: &mut OnlineAveragePolicy,
    sizes: [usize; 4],
    steps: usize,
    seed: u64,
    training: bool,
    records: &mut String,
) -> (Scores, Vec<String>) {
    let mut rng = Rng(seed);
    let mut sources: Vec<([LiveAverage; 4], Vec<u64>)> = sizes
        .iter()
        .map(|n| {
            let live = materialize(*n, rng.next());
            (
                std::array::from_fn(|_| live.clone()),
                live.values().to_vec(),
            )
        })
        .collect();
    let mut total = Scores::default();
    let mut phases = Vec::new();
    for (phase, name) in PHASES.iter().enumerate() {
        let mut scores = Scores::default();
        for step in 0..steps {
            for (size, (states, oracle)) in sources.iter_mut().enumerate() {
                let n = oracle.len();
                let delta = workload(phase, oracle, &mut rng);
                let k = delta.changes().len();
                let event = states[0].last_event() + 1;
                let first = (step + size + phase + seed as usize % 4) % 4;
                let mut elapsed = [0_u64; 4];
                let mut learned_strategy = AverageStrategy::Full;
                // Balance positions and immediate predecessor pairs in each block.
                for candidate in execution_order(first) {
                    let state = &mut states[candidate];
                    let start = Instant::now();
                    let average = if candidate == 3 && training {
                        let execution = policy
                            .execute_and_learn(state, event, black_box(&delta))
                            .unwrap();
                        learned_strategy = execution.strategy;
                        execution.average
                    } else {
                        let strategy = match candidate {
                            0 => AverageStrategy::Incremental,
                            1 => AverageStrategy::Full,
                            2 => heuristic(black_box(n), black_box(k)),
                            _ => {
                                learned_strategy =
                                    black_box(&*policy).choose(black_box(n), black_box(k));
                                learned_strategy
                            }
                        };
                        state
                            .apply_with_strategy(event, black_box(&delta), strategy)
                            .unwrap()
                    };
                    black_box(average);
                    elapsed[candidate] =
                        start.elapsed().as_nanos().clamp(1, u64::MAX as u128) as u64;
                }
                // Source generation, independent reference update and full audit are
                // outside candidate timers, but included in experiment wall time.
                for change in delta.changes() {
                    oracle[change.index()] = change.new_value();
                }
                let sum = oracle.iter().map(|v| u128::from(*v)).sum::<u128>();
                for state in states {
                    assert_eq!(state.values(), oracle);
                    assert_eq!(state.query().numerator(), sum);
                    assert_eq!(state.query().denominator(), n);
                    assert_eq!(state.version(), event);
                }
                total.record(elapsed, learned_strategy);
                scores.record(elapsed, learned_strategy);
                if !records.is_empty() {
                    records.push(',');
                }
                write!(
                    records,
                    "[{},{phase},{n},{k},{first},{},{},{},{},{}]",
                    usize::from(training),
                    elapsed[0],
                    elapsed[1],
                    elapsed[2],
                    elapsed[3],
                    usize::from(learned_strategy == AverageStrategy::Full)
                )
                .unwrap();
            }
        }
        phases.push(format!(
            "{{\"phase\":\"{name}\",\"scores\":{}}}",
            scores.json()
        ));
    }
    (total, phases)
}

fn heuristic(n: usize, k: usize) -> AverageStrategy {
    if k <= n / 4 {
        AverageStrategy::Incremental
    } else {
        AverageStrategy::Full
    }
}

fn execution_order(first: usize) -> [usize; 4] {
    [0, 1, 3, 2].map(|offset| (first + offset) % 4)
}

fn workload(phase: usize, values: &[u64], rng: &mut Rng) -> ReplaceDelta {
    let n = values.len();
    let k = match phase {
        0 | 2 => 1 + rng.next() as usize % (n / 64).max(1),
        1 | 3 => n / 2 + rng.next() as usize % (n - n / 2 + 1),
        4 => 0,
        5 => [0, 1, n / 128, n / 8, n / 2, n][rng.next() as usize % 6],
        6 => [n / 4 - 1, n / 4, n / 4 + 1, n / 4 * 3, n / 4 * 3 + 1][rng.next() as usize % 5],
        _ => (n / 128).max(1),
    };
    if phase == 7 {
        return ReplaceDelta::try_new(
            (0..k)
                .map(|index| Replace::new(index, values[index], rng.next()))
                .collect(),
        )
        .unwrap();
    }
    let spread = match phase {
        2 | 3 => false,
        5 | 6 => rng.next().is_multiple_of(2),
        _ => true,
    };
    batch(values, k, rng, spread)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generated_workloads_are_canonical_and_bounded() {
        let mut rng = Rng(42);
        for n in [384, 511, 8191] {
            let live = materialize(n, rng.next());
            for phase in 0..8 {
                for _ in 0..16 {
                    let delta = workload(phase, live.values(), &mut rng);
                    assert!(delta.changes().len() <= n);
                    let mut changed = live.clone();
                    changed
                        .apply_with_strategy(1, &delta, AverageStrategy::Full)
                        .unwrap();
                    live_support::audit(&changed);
                }
            }
        }
    }

    #[test]
    fn streaming_protocol_learns_only_during_training_and_balances_order() {
        let mut policy = OnlineAveragePolicy::new(CONTEXT);
        let mut records = String::new();
        let (scores, _) = evaluate(&mut policy, [16, 17, 31, 32], 8, 42, true, &mut records);
        assert_eq!(policy.observations(), 256);
        assert_eq!(scores.samples[0].len(), 256);
        let frozen = policy.clone();
        evaluate(
            &mut policy,
            [19, 23, 29, 37],
            8,
            43,
            false,
            &mut String::new(),
        );
        assert_eq!(policy, frozen);
        for phase in 0..8 {
            for size in 0..4 {
                let mut positions = [[0; 4]; 4];
                let mut predecessors = [[0; 4]; 4];
                for step in 0..8 {
                    let order = execution_order((step + size + phase + 42 % 4) % 4);
                    for (position, candidate) in order.iter().enumerate() {
                        positions[*candidate][position] += 1;
                    }
                    for pair in order.windows(2) {
                        predecessors[pair[0]][pair[1]] += 1;
                    }
                }
                assert_eq!(positions, [[2; 4]; 4]);
                for (from, row) in predecessors.iter().enumerate() {
                    for (to, count) in row.iter().enumerate() {
                        assert_eq!(*count, if from == to { 0 } else { 2 });
                    }
                }
            }
        }
    }
}
