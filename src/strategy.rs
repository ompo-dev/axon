use crate::{ObservationFrontier, OperatorKind};

/// A stable identifier for a known physical execution family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrategyKey {
    RawDelta,
    FullRecompute,
    CoalescedDelta,
    HybridShard,
    ChangeFabric,
}

/// Closed latency interval in nanoseconds from paired physical samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostInterval {
    p50: u64,
    lower: u64,
    upper: u64,
    samples: usize,
}

impl CostInterval {
    pub fn new(lower: u64, upper: u64) -> Result<Self, StrategyError> {
        if lower > upper {
            return Err(StrategyError::InvertedCostInterval);
        }
        Ok(Self {
            p50: lower.saturating_add((upper - lower) / 2),
            lower,
            upper,
            samples: 0,
        })
    }

    pub fn from_samples(samples: &[u64]) -> Result<Self, StrategyError> {
        let (&first, rest) = samples.split_first().ok_or(StrategyError::EmptySamples)?;
        let mut ordered = Vec::with_capacity(samples.len());
        ordered.push(first);
        ordered.extend_from_slice(rest);
        ordered.sort_unstable();
        Ok(Self {
            p50: ordered[(ordered.len() - 1) / 2],
            lower: first.min(*ordered.first().expect("nonempty samples")),
            upper: *ordered.last().expect("nonempty samples"),
            samples: ordered.len(),
        })
    }

    pub const fn lower(self) -> u64 {
        self.lower
    }

    pub const fn upper(self) -> u64 {
        self.upper
    }

    pub const fn p50(self) -> u64 {
        self.p50
    }

    pub const fn samples(self) -> usize {
        self.samples
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateLayout {
    CanonicalShardOrdered,
    AdjacentRuns,
    Arbitrary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrategyMetric {
    Latency,
    Throughput,
    Energy,
}

/// Physical dimensions that must match before a cached decision can run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementContext {
    hardware_id: String,
    update_layout: UpdateLayout,
    queries_per_ingest: u32,
    metric: StrategyMetric,
    protocol_version: u16,
}

impl MeasurementContext {
    pub fn new(
        hardware_id: impl Into<String>,
        update_layout: UpdateLayout,
        queries_per_ingest: u32,
        metric: StrategyMetric,
        protocol_version: u16,
    ) -> Result<Self, StrategyError> {
        let hardware_id = hardware_id.into();
        if hardware_id.trim().is_empty() || queries_per_ingest == 0 || protocol_version == 0 {
            return Err(StrategyError::InvalidMeasurementContext);
        }
        Ok(Self {
            hardware_id,
            update_layout,
            queries_per_ingest,
            metric,
            protocol_version,
        })
    }

    pub fn hardware_id(&self) -> &str {
        &self.hardware_id
    }

    pub const fn update_layout(&self) -> UpdateLayout {
        self.update_layout
    }

    pub const fn queries_per_ingest(&self) -> u32 {
        self.queries_per_ingest
    }

    pub const fn metric(&self) -> StrategyMetric {
        self.metric
    }

    pub const fn protocol_version(&self) -> u16 {
        self.protocol_version
    }
}

/// Workload dimensions actually covered by an evidence record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkloadSignature {
    operator: OperatorKind,
    state_words: usize,
    shard_count: usize,
    raw_events: usize,
    final_writes: usize,
    frontier: ObservationFrontier,
    context: MeasurementContext,
}

impl WorkloadSignature {
    pub fn new(
        operator: OperatorKind,
        state_words: usize,
        shard_count: usize,
        raw_events: usize,
        final_writes: usize,
        frontier: ObservationFrontier,
        context: MeasurementContext,
    ) -> Result<Self, StrategyError> {
        if state_words == 0
            || shard_count == 0
            || final_writes > raw_events
            || final_writes > state_words
        {
            return Err(StrategyError::InvalidWorkloadSignature);
        }
        Ok(Self {
            operator,
            state_words,
            shard_count,
            raw_events,
            final_writes,
            frontier,
            context,
        })
    }

    pub const fn operator(&self) -> OperatorKind {
        self.operator
    }

    pub const fn state_words(&self) -> usize {
        self.state_words
    }

    pub const fn shard_count(&self) -> usize {
        self.shard_count
    }

    pub const fn raw_events(&self) -> usize {
        self.raw_events
    }

    pub const fn frontier(&self) -> ObservationFrontier {
        self.frontier
    }

    pub fn context(&self) -> &MeasurementContext {
        &self.context
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReopenCondition {
    OperatorChanged,
    UpdateLocalityChanged,
    HardwareChanged,
    MemoryHierarchyChanged,
    QueryUpdateRatioChanged,
    MetricChanged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrategyStatus {
    Supported,
    LatencyDominated,
    Inconclusive,
}

/// A failed family remains evidence, scoped to its measured workload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StrategyRefutation {
    domain: WorkloadSignature,
    baseline: StrategyKey,
    candidate: StrategyKey,
    oracle_headroom_basis_points: i64,
    reopen_if: Vec<ReopenCondition>,
}

impl StrategyRefutation {
    pub fn domain(&self) -> &WorkloadSignature {
        &self.domain
    }

    pub const fn baseline(&self) -> StrategyKey {
        self.baseline
    }

    pub const fn candidate(&self) -> StrategyKey {
        self.candidate
    }

    pub const fn oracle_headroom_basis_points(&self) -> i64 {
        self.oracle_headroom_basis_points
    }

    pub fn reopen_if(&self) -> &[ReopenCondition] {
        &self.reopen_if
    }
}

/// Oracle comparison: promotion only follows non-overlapping physical bounds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StrategyEvidence {
    domain: WorkloadSignature,
    baseline: StrategyKey,
    baseline_cost: CostInterval,
    candidate: StrategyKey,
    oracle_cost: CostInterval,
    status: StrategyStatus,
    oracle_headroom_basis_points: i64,
    refutation: Option<StrategyRefutation>,
}

impl StrategyEvidence {
    pub fn from_oracle(
        domain: WorkloadSignature,
        baseline: StrategyKey,
        baseline_cost: CostInterval,
        candidate: StrategyKey,
        oracle_cost: CostInterval,
    ) -> Self {
        let status = if oracle_cost.lower() > baseline_cost.upper() {
            StrategyStatus::LatencyDominated
        } else if oracle_cost.upper() < baseline_cost.lower() {
            StrategyStatus::Supported
        } else {
            StrategyStatus::Inconclusive
        };
        let oracle_headroom_basis_points = headroom_basis_points(baseline_cost, oracle_cost);
        let refutation = (status == StrategyStatus::LatencyDominated).then(|| StrategyRefutation {
            domain: domain.clone(),
            baseline,
            candidate,
            oracle_headroom_basis_points,
            reopen_if: vec![
                ReopenCondition::OperatorChanged,
                ReopenCondition::UpdateLocalityChanged,
                ReopenCondition::HardwareChanged,
                ReopenCondition::MemoryHierarchyChanged,
                ReopenCondition::QueryUpdateRatioChanged,
                ReopenCondition::MetricChanged,
            ],
        });
        Self {
            domain,
            baseline,
            baseline_cost,
            candidate,
            oracle_cost,
            status,
            oracle_headroom_basis_points,
            refutation,
        }
    }

    /// Builds evidence from paired rounds. A reflex requires every observed pair to agree.
    pub fn from_paired_samples(
        domain: WorkloadSignature,
        baseline: StrategyKey,
        baseline_samples: &[u64],
        candidate: StrategyKey,
        candidate_samples: &[u64],
    ) -> Result<Self, StrategyError> {
        if baseline_samples.len() != candidate_samples.len() {
            return Err(StrategyError::MismatchedSampleCounts);
        }
        let baseline_cost = CostInterval::from_samples(baseline_samples)?;
        let oracle_cost = CostInterval::from_samples(candidate_samples)?;
        let mut evidence =
            Self::from_oracle(domain, baseline, baseline_cost, candidate, oracle_cost);
        evidence.status = if candidate_samples
            .iter()
            .zip(baseline_samples)
            .all(|(candidate, baseline)| candidate < baseline)
        {
            StrategyStatus::Supported
        } else if candidate_samples
            .iter()
            .zip(baseline_samples)
            .all(|(candidate, baseline)| candidate > baseline)
        {
            StrategyStatus::LatencyDominated
        } else {
            StrategyStatus::Inconclusive
        };
        evidence.refutation =
            (evidence.status == StrategyStatus::LatencyDominated).then(|| StrategyRefutation {
                domain: evidence.domain.clone(),
                baseline: evidence.baseline,
                candidate: evidence.candidate,
                oracle_headroom_basis_points: evidence.oracle_headroom_basis_points,
                reopen_if: vec![
                    ReopenCondition::OperatorChanged,
                    ReopenCondition::UpdateLocalityChanged,
                    ReopenCondition::HardwareChanged,
                    ReopenCondition::MemoryHierarchyChanged,
                    ReopenCondition::QueryUpdateRatioChanged,
                    ReopenCondition::MetricChanged,
                ],
            });
        Ok(evidence)
    }

    pub fn domain(&self) -> &WorkloadSignature {
        &self.domain
    }

    pub const fn baseline(&self) -> StrategyKey {
        self.baseline
    }

    pub const fn baseline_cost(&self) -> CostInterval {
        self.baseline_cost
    }

    pub const fn candidate(&self) -> StrategyKey {
        self.candidate
    }

    pub const fn oracle_cost(&self) -> CostInterval {
        self.oracle_cost
    }

    pub const fn status(&self) -> StrategyStatus {
        self.status
    }

    pub const fn oracle_headroom_basis_points(&self) -> i64 {
        self.oracle_headroom_basis_points
    }

    pub fn refutation(&self) -> Option<&StrategyRefutation> {
        self.refutation.as_ref()
    }

    fn winner(&self) -> Option<StrategyKey> {
        match self.status {
            StrategyStatus::Supported => Some(self.candidate),
            StrategyStatus::LatencyDominated => Some(self.baseline),
            StrategyStatus::Inconclusive => None,
        }
    }
}

/// Meta-JIT turns a settled strategy decision into an exact workload guard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetaJit {
    guard: WorkloadSignature,
    strategy: StrategyKey,
}

impl MetaJit {
    pub fn from_evidence(evidence: &StrategyEvidence) -> Option<Self> {
        evidence.winner().map(|strategy| Self {
            guard: evidence.domain.clone(),
            strategy,
        })
    }

    pub fn select(&self, workload: &WorkloadSignature) -> Option<StrategyKey> {
        (self.guard == *workload).then_some(self.strategy)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrategyError {
    InvertedCostInterval,
    InvalidWorkloadSignature,
    InvalidMeasurementContext,
    EmptySamples,
    MismatchedSampleCounts,
}

fn headroom_basis_points(baseline: CostInterval, oracle: CostInterval) -> i64 {
    let baseline = baseline.p50();
    if baseline == 0 {
        return 0;
    }
    let difference = i128::from(baseline) - i128::from(oracle.p50());
    (difference.saturating_mul(10_000) / i128::from(baseline))
        .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}
