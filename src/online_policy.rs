use std::{
    fs::OpenOptions,
    io::{self, Read, Write},
    path::Path,
};

use crate::{
    AverageStrategy, ExactAverage, LiveAverage, LiveError, ReplaceDelta, live_store::checksum,
};

const BUCKETS: usize = 16;
const MIN_SAMPLES: u64 = 4;
const MAGIC: &[u8; 8] = b"AXPOL001";
const NORMALIZED_MAGIC: &[u8; 8] = b"AXPOL002";
const RATE_SCALE: u128 = 1024;
const RECORD_BYTES: usize = 24 + BUCKETS * 2 * 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Cost {
    samples: u64,
    // Raw nanoseconds, or fixed-point nanoseconds per work unit.
    mean_ns: u64,
}

impl Cost {
    fn observe(&mut self, elapsed_ns: u64) {
        self.samples = self.samples.saturating_add(1);
        let divisor = self.samples.min(8);
        // Equivalent to signed truncating division, without 128-bit division.
        if elapsed_ns >= self.mean_ns {
            self.mean_ns += (elapsed_ns - self.mean_ns) / divisor;
        } else {
            self.mean_ns -= (self.mean_ns - elapsed_ns) / divisor;
        }
    }
}

/// Bounded empirical cost memory for the two implemented exact AVG strategies.
/// Layout, hardware and timing protocol belong in the caller's context identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OnlineAveragePolicy {
    context_hash: u64,
    costs: [[Cost; 2]; BUCKETS],
    normalized: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnlineExecution {
    pub strategy: AverageStrategy,
    pub elapsed_ns: u64,
    pub average: ExactAverage,
}

impl OnlineAveragePolicy {
    pub fn new(context: &str) -> Self {
        Self {
            context_hash: checksum(context.as_bytes()),
            costs: [[Cost::default(); 2]; BUCKETS],
            normalized: false,
        }
    }

    /// Estimate cost per changed value (Delta) or source value (Full), so unequal
    /// workloads in one bucket do not directly compare their absolute durations.
    /// This is a local cost approximation, not a correctness or speed guarantee.
    pub fn normalized(context: &str) -> Self {
        Self {
            normalized: true,
            ..Self::new(context)
        }
    }

    pub fn choose(&self, values: usize, updates: usize) -> AverageStrategy {
        if values > 0 && updates == 0 {
            return AverageStrategy::Incremental;
        }
        let Some(bucket) = bucket(values, updates) else {
            return AverageStrategy::Full;
        };
        self.preferred(self.costs[bucket], values, updates)
    }

    fn preferred(
        &self,
        [incremental, full]: [Cost; 2],
        values: usize,
        updates: usize,
    ) -> AverageStrategy {
        let (incremental_units, full_units) = if self.normalized {
            (updates as u128, values as u128)
        } else {
            (1, 1)
        };
        if incremental.samples >= MIN_SAMPLES
            && full.samples >= MIN_SAMPLES
            && incremental.mean_ns as u128 * incremental_units < full.mean_ns as u128 * full_units
        {
            AverageStrategy::Incremental
        } else {
            AverageStrategy::Full
        }
    }

    /// Balance initial samples, then explore the other strategy every 16 observations.
    /// The caller can keep using choose() when an immutable evaluation is required.
    pub fn explore(&self, values: usize, updates: usize) -> AverageStrategy {
        if values > 0 && updates == 0 {
            return AverageStrategy::Incremental;
        }
        let Some(bucket) = bucket(values, updates) else {
            return AverageStrategy::Full;
        };
        self.exploration(self.costs[bucket], values, updates)
    }

    fn exploration(
        &self,
        [incremental, full]: [Cost; 2],
        values: usize,
        updates: usize,
    ) -> AverageStrategy {
        if incremental.samples < MIN_SAMPLES || full.samples < MIN_SAMPLES {
            return if incremental.samples <= full.samples {
                AverageStrategy::Incremental
            } else {
                AverageStrategy::Full
            };
        }
        let preferred = self.preferred([incremental, full], values, updates);
        if incremental
            .samples
            .saturating_add(full.samples)
            .is_multiple_of(16)
        {
            match preferred {
                AverageStrategy::Incremental => AverageStrategy::Full,
                AverageStrategy::Full => AverageStrategy::Incremental,
            }
        } else {
            preferred
        }
    }

    /// Execute one real event and learn only its observed cost. No oracle or second
    /// execution is required. Rejected and empty events never train. Empty events
    /// still validate and advance the source version, but have no strategy evidence.
    pub fn execute_and_learn(
        &mut self,
        live: &mut LiveAverage,
        event: u64,
        delta: &ReplaceDelta,
    ) -> Result<OnlineExecution, LiveError> {
        let n = live.values().len();
        let k = delta.changes().len();
        let bucket = bucket(n, k);
        let strategy = if k == 0 {
            AverageStrategy::Incremental
        } else {
            bucket
                .map(|index| self.exploration(self.costs[index], n, k))
                .unwrap_or(AverageStrategy::Full)
        };
        let start = std::time::Instant::now();
        let average = live.apply_with_strategy(event, delta, strategy)?;
        let elapsed_ns = start.elapsed().as_nanos().clamp(1, u64::MAX as u128) as u64;
        if k != 0
            && let Some(bucket) = bucket
        {
            let sample = self.cost_sample(n, k, strategy, elapsed_ns);
            self.costs[bucket][usize::from(strategy == AverageStrategy::Full)].observe(sample);
        }
        Ok(OnlineExecution {
            strategy,
            elapsed_ns,
            average,
        })
    }

    /// Update after the corresponding execution has completed and passed parity checks.
    /// A bounded exponential mean (1/8 after warmup) can follow changing costs.
    /// Empty events are excluded because both strategies use the same empty path.
    pub fn observe(
        &mut self,
        values: usize,
        updates: usize,
        strategy: AverageStrategy,
        elapsed_ns: u64,
    ) -> bool {
        let Some(bucket) = bucket(values, updates) else {
            return false;
        };
        if elapsed_ns == 0 || updates == 0 {
            return false;
        }
        let index = usize::from(strategy == AverageStrategy::Full);
        let sample = self.cost_sample(values, updates, strategy, elapsed_ns);
        self.costs[bucket][index].observe(sample);
        true
    }

    fn cost_sample(
        &self,
        values: usize,
        updates: usize,
        strategy: AverageStrategy,
        elapsed_ns: u64,
    ) -> u64 {
        if !self.normalized {
            return elapsed_ns;
        }
        let units = match strategy {
            AverageStrategy::Incremental => updates,
            AverageStrategy::Full => values,
        };
        (elapsed_ns as u128 * RATE_SCALE / units as u128).clamp(1, u64::MAX as u128) as u64
    }

    pub fn observations(&self) -> u64 {
        self.costs
            .iter()
            .flatten()
            .fold(0_u64, |sum, cost| sum.saturating_add(cost.samples))
    }

    pub const fn checkpoint_bytes() -> usize {
        RECORD_BYTES
    }

    /// Exclusive creation makes each saved training result immutable.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut bytes = Vec::with_capacity(RECORD_BYTES);
        bytes.extend_from_slice(if self.normalized {
            NORMALIZED_MAGIC
        } else {
            MAGIC
        });
        bytes.extend_from_slice(&self.context_hash.to_le_bytes());
        for cost in self.costs.iter().flatten() {
            bytes.extend_from_slice(&cost.samples.to_le_bytes());
            bytes.extend_from_slice(&cost.mean_ns.to_le_bytes());
        }
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&bytes)?;
        file.sync_all()
    }

    pub fn load(path: impl AsRef<Path>, context: &str) -> io::Result<Self> {
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid policy or context mismatch",
            )
        };
        let mut bytes = Vec::with_capacity(RECORD_BYTES + 1);
        std::fs::File::open(path)?
            .take((RECORD_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() != RECORD_BYTES || (&bytes[..8] != MAGIC && &bytes[..8] != NORMALIZED_MAGIC)
        {
            return Err(invalid());
        }
        let number = |offset: usize| {
            u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("bounded field"))
        };
        if number(8) != checksum(context.as_bytes())
            || number(RECORD_BYTES - 8) != checksum(&bytes[..RECORD_BYTES - 8])
        {
            return Err(invalid());
        }
        let mut policy = Self::new(context);
        policy.normalized = &bytes[..8] == NORMALIZED_MAGIC;
        for (index, cost) in policy.costs.iter_mut().flatten().enumerate() {
            cost.samples = number(16 + index * 16);
            cost.mean_ns = number(24 + index * 16);
            if (cost.samples == 0) != (cost.mean_ns == 0) {
                return Err(invalid());
            }
        }
        Ok(policy)
    }
}

fn bucket(values: usize, updates: usize) -> Option<usize> {
    if values == 0 || updates > values {
        return None;
    }
    let size = match values {
        1..=512 => 0,
        513..=8192 => 1,
        8193..=131072 => 2,
        _ => 3,
    };
    let density = if updates <= values / 64 {
        0
    } else if updates <= values / 4 {
        1
    } else if updates <= values / 4 * 3 {
        2
    } else {
        3
    };
    Some(size * 4 + density)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_cost_update_matches_signed_reference_at_extremes_and_with_noise() {
        let mut random = 20260922_u64;
        for samples in [0, 1, 2, 6, 7, 8, 100, u64::MAX - 1, u64::MAX] {
            for mean_ns in [0, 1, 7, 500, u64::MAX / 2, u64::MAX] {
                let mut cost = Cost { samples, mean_ns };
                for step in 0..1024 {
                    random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let elapsed = match step % 4 {
                        0 => 1,
                        1 => u64::MAX,
                        _ => random.max(1),
                    };
                    let next_samples = cost.samples.saturating_add(1);
                    let expected = cost.mean_ns as i128
                        + (elapsed as i128 - cost.mean_ns as i128) / next_samples.min(8) as i128;
                    cost.observe(elapsed);
                    assert_eq!(cost.mean_ns, expected as u64);
                    assert_eq!(cost.samples, next_samples);
                }
            }
        }
    }
}
