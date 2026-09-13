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
const RECORD_BYTES: usize = 24 + BUCKETS * 2 * 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Cost {
    samples: u64,
    mean_ns: u64,
}

/// Bounded empirical cost memory for the two implemented exact AVG strategies.
/// Layout, hardware and timing protocol belong in the caller's context identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OnlineAveragePolicy {
    context_hash: u64,
    costs: [[Cost; 2]; BUCKETS],
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
        }
    }

    pub fn choose(&self, values: usize, updates: usize) -> AverageStrategy {
        let Some(bucket) = bucket(values, updates) else {
            return AverageStrategy::Full;
        };
        let [incremental, full] = self.costs[bucket];
        if incremental.samples >= MIN_SAMPLES
            && full.samples >= MIN_SAMPLES
            && incremental.mean_ns < full.mean_ns
        {
            AverageStrategy::Incremental
        } else {
            AverageStrategy::Full
        }
    }

    /// Balance initial samples, then explore the other strategy every 16 observations.
    /// The caller can keep using choose() when an immutable evaluation is required.
    pub fn explore(&self, values: usize, updates: usize) -> AverageStrategy {
        let Some(bucket) = bucket(values, updates) else {
            return AverageStrategy::Full;
        };
        let [incremental, full] = self.costs[bucket];
        if incremental.samples < MIN_SAMPLES || full.samples < MIN_SAMPLES {
            return if incremental.samples <= full.samples {
                AverageStrategy::Incremental
            } else {
                AverageStrategy::Full
            };
        }
        let preferred = self.choose(values, updates);
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
    /// execution is required. Rejected events never become learning observations.
    pub fn execute_and_learn(
        &mut self,
        live: &mut LiveAverage,
        event: u64,
        delta: &ReplaceDelta,
    ) -> Result<OnlineExecution, LiveError> {
        let n = live.values().len();
        let k = delta.changes().len();
        let strategy = self.explore(n, k);
        let start = std::time::Instant::now();
        let average = live.apply_with_strategy(event, delta, strategy)?;
        let elapsed_ns = start.elapsed().as_nanos().clamp(1, u64::MAX as u128) as u64;
        self.observe(n, k, strategy, elapsed_ns);
        Ok(OnlineExecution {
            strategy,
            elapsed_ns,
            average,
        })
    }

    /// Update after the corresponding execution has completed and passed parity checks.
    /// A bounded exponential mean (1/8 after warmup) can follow changing costs.
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
        if elapsed_ns == 0 {
            return false;
        }
        let index = usize::from(strategy == AverageStrategy::Full);
        let cost = &mut self.costs[bucket][index];
        cost.samples = cost.samples.saturating_add(1);
        let divisor = cost.samples.min(8) as i128;
        let next = cost.mean_ns as i128 + (elapsed_ns as i128 - cost.mean_ns as i128) / divisor;
        cost.mean_ns = next as u64;
        true
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
        bytes.extend_from_slice(MAGIC);
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
        if bytes.len() != RECORD_BYTES || &bytes[..8] != MAGIC {
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
