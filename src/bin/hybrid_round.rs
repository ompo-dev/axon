use super::hybrid_fabric::{FabricMeasurement, run_fabric_hybrid};
use super::{
    CompiledChanges, HybridMeasurement, Measurement, OracleMeasurement, PointUpdate,
    run_coalesced_global, run_delta_global, run_full_global, run_hybrid, run_hybrid_oracle,
};

#[derive(Clone, Copy)]
enum Path {
    Full,
    RawDelta,
    CoalescedDelta,
    Hybrid,
    Oracle,
    Fabric,
}

pub(super) fn run_round(
    round: usize,
    seed: &[u64],
    updates: &[PointUpdate],
    shard_words: usize,
    template: &CompiledChanges<'_>,
) -> Result<
    (
        Measurement,
        Measurement,
        Measurement,
        HybridMeasurement,
        OracleMeasurement,
        FabricMeasurement,
    ),
    String,
> {
    let mut full = None;
    let mut raw_delta = None;
    let mut coalesced_delta = None;
    let mut hybrid = None;
    let mut oracle = None;
    let mut fabric = None;
    for path in path_order(round) {
        match path {
            Path::Full => full = Some(run_full_global(seed, updates)?),
            Path::RawDelta => raw_delta = Some(run_delta_global(seed, updates)?),
            Path::CoalescedDelta => coalesced_delta = Some(run_coalesced_global(seed, updates)?),
            Path::Hybrid => hybrid = Some(run_hybrid(seed, updates, shard_words)?),
            Path::Oracle => oracle = Some(run_hybrid_oracle(seed, template)?),
            Path::Fabric => fabric = Some(run_fabric_hybrid(seed, updates, shard_words)?),
        }
    }
    Ok((
        full.ok_or_else(|| "round omitted Full".to_owned())?,
        raw_delta.ok_or_else(|| "round omitted Raw Delta".to_owned())?,
        coalesced_delta.ok_or_else(|| "round omitted Coalesced Delta".to_owned())?,
        hybrid.ok_or_else(|| "round omitted Hybrid".to_owned())?,
        oracle.ok_or_else(|| "round omitted Oracle".to_owned())?,
        fabric.ok_or_else(|| "round omitted Fabric".to_owned())?,
    ))
}

fn path_order(round: usize) -> [Path; 6] {
    const ORDERS: [[Path; 6]; 6] = [
        [
            Path::Full,
            Path::RawDelta,
            Path::CoalescedDelta,
            Path::Hybrid,
            Path::Oracle,
            Path::Fabric,
        ],
        [
            Path::Full,
            Path::CoalescedDelta,
            Path::Oracle,
            Path::Fabric,
            Path::Hybrid,
            Path::RawDelta,
        ],
        [
            Path::RawDelta,
            Path::Hybrid,
            Path::Full,
            Path::Fabric,
            Path::CoalescedDelta,
            Path::Oracle,
        ],
        [
            Path::CoalescedDelta,
            Path::Fabric,
            Path::RawDelta,
            Path::Oracle,
            Path::Full,
            Path::Hybrid,
        ],
        [
            Path::Hybrid,
            Path::Oracle,
            Path::Fabric,
            Path::Full,
            Path::RawDelta,
            Path::CoalescedDelta,
        ],
        [
            Path::Fabric,
            Path::Hybrid,
            Path::CoalescedDelta,
            Path::RawDelta,
            Path::Oracle,
            Path::Full,
        ],
    ];
    ORDERS[xorshift(round as u64 + 1) as usize % ORDERS.len()]
}

fn xorshift(mut value: u64) -> u64 {
    value ^= value << 13;
    value ^= value >> 7;
    value ^ (value << 17)
}
