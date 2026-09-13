mod live_support;

use axon_uic::{
    AverageStrategy, ExactAverage, LiveAverage, OnlineAveragePolicy, Replace, ReplaceDelta,
};
use live_support::{Rng, bounded, semantic};
use std::{path::PathBuf, time::Instant};

#[derive(Default, Debug)]
struct Report {
    steps: u64,
    accepted: u64,
    rejected: u64,
    replacements: u64,
    exact_checks: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("stress failure: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut seeds = 64;
    let mut steps = 8192;
    let mut base_seed = 20260916;
    let mut output: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "axon-uic-stress [--seeds 1..1024] [--steps 1..1000000] [--seed N] [--output report.json]"
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing value")?;
        match flag.as_str() {
            "--seeds" => seeds = bounded(&value, 1, 1024)?,
            "--steps" => steps = bounded(&value, 1, 1_000_000)?,
            "--seed" => base_seed = value.parse::<u64>().map_err(|e| e.to_string())?,
            "--output" => output = Some(value.into()),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    let start = Instant::now();
    let mut total = Report::default();
    for index in 0..seeds {
        let seed = base_seed.wrapping_add(index as u64);
        let report = differential(seed, steps)?;
        total.steps += report.steps;
        total.accepted += report.accepted;
        total.rejected += report.rejected;
        total.replacements += report.replacements;
        total.exact_checks += report.exact_checks;
    }
    let elapsed_ms = start.elapsed().as_millis();
    println!(
        "{seeds} seeds; {} steps; {} accepted events; {} invalid events rejected; {} replacements; {} independent exact checks; {elapsed_ms} ms",
        total.steps, total.accepted, total.rejected, total.replacements, total.exact_checks
    );
    if let Some(path) = output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let json = format!(
            "{{\"schema\":\"axon-differential-stress-v1\",\"base_seed\":{base_seed},\"seeds\":{seeds},\"steps_per_seed\":{steps},\"steps\":{},\"accepted\":{},\"rejected\":{},\"replacements\":{},\"exact_checks\":{},\"elapsed_ms\":{elapsed_ms},\"exact\":true}}\n",
            total.steps, total.accepted, total.rejected, total.replacements, total.exact_checks
        );
        std::fs::write(path, json).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn differential(seed: u64, steps: usize) -> Result<Report, String> {
    let n = [1, 2, 3, 7, 16, 63, 257, 1024, 4096][seed as usize % 9];
    let mut rng = Rng(seed);
    let mut oracle: Vec<_> = (0..n).map(|_| extreme(&mut rng)).collect();
    let mut delta_state =
        LiveAverage::materialize(seed, oracle.clone(), semantic()).map_err(|e| e.to_string())?;
    let mut full_state = delta_state.clone();
    let mut learned_state = delta_state.clone();
    let mut policy = OnlineAveragePolicy::new("differential-correctness");
    let mut report = Report::default();
    for step in 0..steps {
        let mode = rng.next() % 12;
        let k = match mode {
            0 => 0,
            2 => n,
            _ => 1 + rng.next() as usize % n.min(32),
        };
        let mut changes = Vec::with_capacity(k + 1);
        for i in 0..k {
            let index = i * n / k;
            let new = if mode == 3 {
                oracle[index]
            } else {
                extreme(&mut rng)
            };
            changes.push(Replace::new(index, oracle[index], new));
        }
        if mode == 4 {
            let last = changes.last_mut().expect("nonempty invalid batch");
            *last = Replace::new(last.index(), last.old().wrapping_add(1), last.new_value());
        }
        if mode == 8 {
            changes.push(Replace::new(n, 0, 1));
        }
        let delta = ReplaceDelta::try_new(changes).map_err(|e| format!("generator: {e:?}"))?;
        let version = delta_state.version();
        let event = version + 1;
        let invalid = (4..=8).contains(&mode);
        let result = delta_state.apply(
            if mode == 5 {
                seed.wrapping_add(1)
            } else {
                seed
            },
            if mode == 6 { version + 1 } else { version },
            if mode == 7 { version } else { event },
            &delta,
        );
        let context = || format!("seed={seed}, step={step}, mode={mode}, n={n}");
        if invalid {
            if result.is_ok() {
                return Err(format!("accepted invalid event: {}", context()));
            }
            if delta_state != full_state {
                return Err(format!("partial mutation after rejection: {}", context()));
            }
            if matches!(mode, 4 | 7 | 8) {
                let observations = policy.observations();
                if policy
                    .execute_and_learn(
                        &mut learned_state,
                        if mode == 7 { version } else { event },
                        &delta,
                    )
                    .is_ok()
                    || observations != policy.observations()
                    || learned_state != full_state
                {
                    return Err(format!("invalid event changed learning: {}", context()));
                }
            }
            report.rejected += 1;
        } else {
            result.map_err(|e| format!("{}: {e}", context()))?;
            full_state
                .apply_with_strategy(event, &delta, AverageStrategy::Full)
                .map_err(|e| e.to_string())?;
            policy
                .execute_and_learn(&mut learned_state, event, &delta)
                .map_err(|e| e.to_string())?;
            for change in delta.changes() {
                oracle[change.index()] = change.new_value();
            }
            let numerator = oracle.iter().map(|v| u128::from(*v)).sum();
            let expected = ExactAverage::new(numerator, n).map_err(|e| e.to_string())?;
            for state in [&delta_state, &full_state, &learned_state] {
                if state.query() != expected || state.values() != oracle || state.version() != event
                {
                    return Err(format!("oracle mismatch: {}", context()));
                }
                report.exact_checks += 1;
            }
            report.accepted += 1;
            report.replacements += k as u64;
            if policy.observations() != report.accepted {
                return Err(format!("observation count mismatch: {}", context()));
            }
        }
        report.steps += 1;
    }
    Ok(report)
}

fn extreme(rng: &mut Rng) -> u64 {
    match rng.next() % 5 {
        0 => 0,
        1 => u64::MAX,
        2 => 1,
        3 => 1_u64 << 63,
        _ => rng.next(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_histories_match_independent_oracle_with_rollback() {
        for seed in 0..18 {
            let r = differential(seed, 1024).unwrap();
            assert!(r.accepted > 0 && r.rejected > 0);
        }
    }

    #[test]
    fn disjoint_updates_commute_and_inverse_restores_the_exact_source() {
        let initial = LiveAverage::materialize(1, vec![u64::MAX, 0, 3], semantic()).unwrap();
        let a = ReplaceDelta::try_new(vec![Replace::new(0, u64::MAX, 1)]).unwrap();
        let b = ReplaceDelta::try_new(vec![Replace::new(1, 0, u64::MAX)]).unwrap();
        let mut left = initial.clone();
        let mut right = initial.clone();
        left.apply(1, 0, 1, &a).unwrap();
        left.apply(1, 1, 2, &b).unwrap();
        right.apply(1, 0, 1, &b).unwrap();
        right.apply(1, 1, 2, &a).unwrap();
        assert_eq!(left, right);
        let inverse = ReplaceDelta::try_new(vec![
            Replace::new(0, 1, u64::MAX),
            Replace::new(1, u64::MAX, 0),
        ])
        .unwrap();
        left.apply(1, 2, 3, &inverse).unwrap();
        assert_eq!(left.values(), initial.values());
        assert_eq!(left.query(), initial.query());
    }
}
