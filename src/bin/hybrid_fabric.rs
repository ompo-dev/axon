use std::hint::black_box;
use std::time::{Duration, Instant};

use axon_uic::{ObservationFrontier, PointUpdate};

use super::{
    PlanCounts, SHARD_COUNT, ShardStrategy, apply_delta, apply_updates, checksum, clone_data,
    select_strategy, verify_total,
};

struct FabricShard {
    strategy: ShardStrategy,
    updates: Vec<PointUpdate>,
    coalesced: bool,
}

pub(super) struct ChangeFabric {
    shards: Vec<FabricShard>,
    pub(super) counts: PlanCounts,
    words: usize,
    shard_words: usize,
    last_shard: Option<usize>,
}

#[derive(Clone, Copy)]
pub(super) struct FabricMeasurement {
    pub(super) duration: Duration,
    pub(super) ingest: Duration,
    pub(super) execution: Duration,
    pub(super) verification: Duration,
    pub(super) total: u64,
}

impl ChangeFabric {
    fn try_new(words: usize, shard_words: usize) -> Result<Self, String> {
        if shard_words == 0
            || shard_words
                .checked_mul(SHARD_COUNT)
                .filter(|&expected_words| expected_words == words)
                .is_none()
        {
            return Err("change fabric requires an exact fixed-shard shape".to_owned());
        }
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(FabricShard {
                strategy: ShardStrategy::Skip,
                updates: Vec::new(),
                coalesced: false,
            });
        }
        Ok(Self {
            shards,
            counts: PlanCounts {
                skip: SHARD_COUNT,
                raw_delta: 0,
                coalesced_delta: 0,
                full_local: 0,
            },
            words,
            shard_words,
            last_shard: None,
        })
    }

    fn ingest(&mut self, update: PointUpdate, frontier: ObservationFrontier) -> Result<(), String> {
        if update.index() >= self.words {
            return Err(format!("update index out of bounds: {}", update.index()));
        }
        let shard = update.index() / self.shard_words;
        if self.last_shard.is_some_and(|last_shard| shard < last_shard) {
            return Err("change fabric requires updates ordered by shard".to_owned());
        }
        self.last_shard = Some(shard);
        let region = self
            .shards
            .get_mut(shard)
            .ok_or_else(|| "change fabric shard index out of bounds".to_owned())?;
        let previous_strategy = region.strategy;
        let adjacent_duplicate = region
            .updates
            .last()
            .is_some_and(|previous| previous.index() == update.index());
        if adjacent_duplicate {
            if frontier == ObservationFrontier::IntermediateObserved {
                return Err("cannot coalesce with intermediate observations".to_owned());
            }
            *region
                .updates
                .last_mut()
                .ok_or_else(|| "change fabric lost adjacent update".to_owned())? = update;
            region.coalesced = true;
        } else {
            if region.updates.len() == region.updates.capacity() {
                region
                    .updates
                    .try_reserve(1)
                    .map_err(|error| format!("cannot allocate change fabric shard: {error}"))?;
            }
            region.updates.push(update);
        }
        region.strategy = select_strategy(&region.updates, region.coalesced, self.shard_words);
        if region.strategy != previous_strategy {
            self.counts.remove(previous_strategy);
            self.counts.record(region.strategy);
        }
        Ok(())
    }
}

pub(super) fn build_change_fabric(
    updates: &[PointUpdate],
    words: usize,
    shard_words: usize,
    frontier: ObservationFrontier,
) -> Result<ChangeFabric, String> {
    let mut fabric = ChangeFabric::try_new(words, shard_words)?;
    for update in updates {
        fabric.ingest(*update, frontier)?;
    }
    Ok(fabric)
}

pub(super) fn run_fabric_hybrid(
    seed: &[u64],
    updates: &[PointUpdate],
    shard_words: usize,
) -> Result<FabricMeasurement, String> {
    let mut data = clone_data(seed)?;
    let mut shard_totals: Vec<_> = data.chunks_exact(shard_words).map(checksum).collect();
    let started = Instant::now();
    let ingest_started = Instant::now();
    let fabric = build_change_fabric(
        updates,
        data.len(),
        shard_words,
        ObservationFrontier::FinalStateOnly,
    )?;
    let ingest = ingest_started.elapsed();
    let execution_started = Instant::now();
    let total = execute_fabric_hybrid(&mut data, &mut shard_totals, &fabric)?;
    black_box(total);
    let execution = execution_started.elapsed();
    let duration = started.elapsed();
    let verification_started = Instant::now();
    verify_total(&data, total)?;
    let verification = verification_started.elapsed();
    Ok(FabricMeasurement {
        duration,
        ingest,
        execution,
        verification,
        total,
    })
}

pub(super) fn execute_fabric_hybrid(
    data: &mut [u64],
    shard_totals: &mut [u64],
    fabric: &ChangeFabric,
) -> Result<u64, String> {
    let expected_words = fabric
        .shard_words
        .checked_mul(fabric.shards.len())
        .ok_or_else(|| "change fabric shape overflow".to_owned())?;
    if fabric.words != expected_words
        || data.len() != fabric.words
        || shard_totals.len() != fabric.shards.len()
    {
        return Err("change fabric shape does not match execution state".to_owned());
    }
    for (shard, region) in fabric.shards.iter().enumerate() {
        let start = shard * fabric.shard_words;
        let end = start + fabric.shard_words;
        match region.strategy {
            ShardStrategy::Skip => {}
            ShardStrategy::RawDelta | ShardStrategy::CoalescedDelta => {
                shard_totals[shard] = apply_delta(data, &region.updates, shard_totals[shard])?;
            }
            ShardStrategy::FullLocal => {
                apply_updates(data, &region.updates)?;
                shard_totals[shard] = checksum(&data[start..end]);
            }
        }
    }
    Ok(shard_totals
        .iter()
        .copied()
        .fold(0_u64, |sum, value| sum.wrapping_add(value)))
}
