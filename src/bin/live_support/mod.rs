#![allow(dead_code)]

use axon_uic::{AverageStrategy, FoldSpec, LiveAverage, Replace, ReplaceDelta, SemanticArtifact};
use std::{hint::black_box, time::Instant};

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

pub fn materialize(n: usize, seed: u64) -> LiveAverage {
    let mut rng = Rng(seed);
    let values = (0..n).map(|_| rng.next()).collect();
    LiveAverage::materialize(seed, values, semantic()).expect("nonempty generated source")
}

pub fn semantic() -> SemanticArtifact {
    SemanticArtifact::synthesize(FoldSpec::AverageExactU64).expect("supported semantics")
}

pub fn batch(values: &[u64], updates: usize, rng: &mut Rng, spread: bool) -> ReplaceDelta {
    let n = values.len();
    let start = (rng.next() as usize) % (n - updates + 1);
    let changes = (0..updates)
        .map(|i| {
            let index = if spread {
                let lower = i * n / updates;
                let upper = (i + 1) * n / updates;
                lower + rng.next() as usize % (upper - lower)
            } else {
                start + i
            };
            Replace::new(index, values[index], rng.next())
        })
        .collect();
    ReplaceDelta::try_new(changes).expect("generated indices are canonical")
}

pub fn timed_apply(live: &mut LiveAverage, delta: &ReplaceDelta, strategy: AverageStrategy) -> u64 {
    let event = live.last_event() + 1;
    let start = Instant::now();
    let result = live
        .apply_with_strategy(event, black_box(delta), strategy)
        .expect("valid batch");
    black_box(result);
    black_box(live.query());
    start.elapsed().as_nanos().max(1) as u64
}

pub fn percentile(samples: &[u64], percent: usize) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() - 1) * percent / 100]
}

pub fn audit(live: &LiveAverage) {
    let total: u128 = live.values().iter().map(|v| *v as u128).sum();
    assert_eq!(live.query().numerator(), total, "independent full audit");
    assert_eq!(live.query().denominator(), live.values().len());
}

pub fn bounded(value: &str, min: usize, max: usize) -> Result<usize, String> {
    let n = value
        .parse::<usize>()
        .map_err(|_| format!("invalid integer: {value}"))?;
    if !(min..=max).contains(&n) {
        return Err(format!("expected {min}..{max}: {n}"));
    }
    Ok(n)
}
