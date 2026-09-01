use std::collections::BTreeMap;

/// Structural classes tell the runtime whether an exact change morphism exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeltaClass {
    Constant,
    Logarithmic,
    Affected,
    Global,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperatorKind {
    Sum,
    Count,
    Xor,
    Sort,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionStrategy {
    Delta,
    Full,
}

/// Coalescing is valid only when no observer can distinguish overwritten events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationFrontier {
    FinalStateOnly,
    IntermediateObserved,
}

/// Cost model learned from a response curve or supplied by a physical backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostEstimate {
    full: u64,
    delta_fixed: u64,
    delta_per_change: u64,
    validation: u64,
}

impl CostEstimate {
    pub const fn new(full: u64, delta_fixed: u64, delta_per_change: u64, validation: u64) -> Self {
        Self {
            full,
            delta_fixed,
            delta_per_change,
            validation,
        }
    }

    pub fn delta_cost(self, changed: u64) -> u64 {
        self.delta_fixed
            .saturating_add(self.delta_per_change.saturating_mul(changed))
            .saturating_add(self.validation)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChangeSupport {
    changed: u64,
    total: u64,
}

impl ChangeSupport {
    pub fn new(changed: u64, total: u64) -> Result<Self, DeltaError> {
        if changed > total {
            return Err(DeltaError::InvalidSupport);
        }
        Ok(Self { changed, total })
    }

    pub const fn changed(self) -> u64 {
        self.changed
    }

    pub const fn total(self) -> u64 {
        self.total
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IncrementalContract {
    kind: OperatorKind,
    class: DeltaClass,
    exact: bool,
    supports_coalescing: bool,
}

impl IncrementalContract {
    pub const fn class(self) -> DeltaClass {
        self.class
    }

    pub const fn exact(self) -> bool {
        self.exact
    }

    pub const fn supports_coalescing(self) -> bool {
        self.supports_coalescing
    }

    /// Full is mandatory when no exact delta morphism is known.
    pub fn select(self, support: ChangeSupport, costs: CostEstimate) -> ExecutionStrategy {
        if !self.exact || costs.delta_cost(support.changed()) >= costs.full {
            ExecutionStrategy::Full
        } else {
            ExecutionStrategy::Delta
        }
    }

    /// Largest change support for which Delta strictly beats Full under this cost model.
    pub fn crossover_support(self, total: u64, costs: CostEstimate) -> Option<u64> {
        if !self.exact {
            return None;
        }
        let fixed = costs.delta_fixed.saturating_add(costs.validation);
        if fixed >= costs.full {
            return None;
        }
        if costs.delta_per_change == 0 {
            return Some(total);
        }
        Some(
            costs
                .full
                .saturating_sub(fixed)
                .saturating_sub(1)
                .checked_div(costs.delta_per_change)
                .unwrap_or(0)
                .min(total),
        )
    }

    pub const fn kind(self) -> OperatorKind {
        self.kind
    }
}

pub struct IncrementalizabilityAnalyzer;

impl IncrementalizabilityAnalyzer {
    pub const fn analyze(kind: OperatorKind) -> IncrementalContract {
        match kind {
            OperatorKind::Sum | OperatorKind::Count | OperatorKind::Xor => IncrementalContract {
                kind,
                class: DeltaClass::Constant,
                exact: true,
                supports_coalescing: true,
            },
            OperatorKind::Sort => IncrementalContract {
                kind,
                class: DeltaClass::Global,
                exact: false,
                supports_coalescing: false,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PointUpdate {
    index: usize,
    value: u64,
}

/// Linear-time coalescing for bursty streams. Non-adjacent events retain order.
pub fn coalesce_adjacent_last_writes(updates: &[PointUpdate]) -> Vec<PointUpdate> {
    let mut output: Vec<PointUpdate> = Vec::with_capacity(updates.len());
    for update in updates {
        if let Some(previous) = output.last_mut()
            && previous.index() == update.index()
        {
            *previous = *update;
            continue;
        }
        output.push(*update);
    }
    output
}

/// Coalesces adjacent last writes only after the observation frontier is closed.
pub fn coalesce_adjacent_at_frontier(
    updates: &[PointUpdate],
    frontier: ObservationFrontier,
) -> Result<Vec<PointUpdate>, DeltaError> {
    match frontier {
        ObservationFrontier::FinalStateOnly => Ok(coalesce_adjacent_last_writes(updates)),
        ObservationFrontier::IntermediateObserved => Err(DeltaError::ObservationNotClosed),
    }
}

impl PointUpdate {
    pub const fn new(index: usize, value: u64) -> Self {
        Self { index, value }
    }

    pub const fn index(self) -> usize {
        self.index
    }

    pub const fn value(self) -> u64 {
        self.value
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeltaError {
    InvalidSupport,
    IndexOutOfBounds(usize),
    ObservationNotClosed,
}

/// Exact modular-`u64` sum state. Updates return fresh state, preserving prior versions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SumState {
    values: Vec<u64>,
    total: u64,
}

impl SumState {
    pub fn try_from_values(values: Vec<u64>) -> Result<Self, DeltaError> {
        let total = sum_values(&values);
        Ok(Self { values, total })
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn full_after(&self, updates: &[PointUpdate]) -> Result<Self, DeltaError> {
        let values = self.apply_values(updates)?;
        Self::try_from_values(values)
    }

    pub fn apply_delta(&self, updates: &[PointUpdate]) -> Result<Self, DeltaError> {
        let mut values = self.values.clone();
        let mut total = self.total;
        for update in updates {
            let old = *values
                .get(update.index())
                .ok_or(DeltaError::IndexOutOfBounds(update.index()))?;
            values[update.index()] = update.value();
            total = total.wrapping_sub(old).wrapping_add(update.value());
        }
        Ok(Self { values, total })
    }

    /// Last-write coalescing is exact for final-state queries: overwritten updates disappear.
    pub fn apply_coalesced(
        &self,
        updates: &[PointUpdate],
        frontier: ObservationFrontier,
    ) -> Result<(Self, usize), DeltaError> {
        if frontier == ObservationFrontier::IntermediateObserved {
            return Err(DeltaError::ObservationNotClosed);
        }
        let mut latest = BTreeMap::new();
        for update in updates {
            if update.index() >= self.values.len() {
                return Err(DeltaError::IndexOutOfBounds(update.index()));
            }
            latest.insert(update.index(), update.value());
        }
        let coalesced: Vec<_> = latest
            .into_iter()
            .map(|(index, value)| PointUpdate::new(index, value))
            .collect();
        let applied = coalesced.len();
        Ok((self.apply_delta(&coalesced)?, applied))
    }

    fn apply_values(&self, updates: &[PointUpdate]) -> Result<Vec<u64>, DeltaError> {
        let mut values = self.values.clone();
        for update in updates {
            let slot = values
                .get_mut(update.index())
                .ok_or(DeltaError::IndexOutOfBounds(update.index()))?;
            *slot = update.value();
        }
        Ok(values)
    }
}

fn sum_values(values: &[u64]) -> u64 {
    values
        .iter()
        .copied()
        .fold(0_u64, |sum, value| sum.wrapping_add(value))
}
