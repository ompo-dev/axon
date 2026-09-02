use std::time::Duration;

/// A named portion of one benchmarked workload lifecycle.
///
/// `Duration::ZERO` means that the phase was intentionally not measured or is
/// not applicable to this workload. It must not be interpreted as a hidden cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchPhase {
    InputGeneration,
    Initialization,
    Synthesis,
    Verification,
    Allocation,
    Ingestion,
    Planning,
    ArtifactLoad,
    ArtifactPersist,
    Execution,
    ResultValidation,
    Teardown,
}

impl BenchPhase {
    pub const ALL: [Self; 12] = [
        Self::InputGeneration,
        Self::Initialization,
        Self::Synthesis,
        Self::Verification,
        Self::Allocation,
        Self::Ingestion,
        Self::Planning,
        Self::ArtifactLoad,
        Self::ArtifactPersist,
        Self::Execution,
        Self::ResultValidation,
        Self::Teardown,
    ];

    const fn index(self) -> usize {
        match self {
            Self::InputGeneration => 0,
            Self::Initialization => 1,
            Self::Synthesis => 2,
            Self::Verification => 3,
            Self::Allocation => 4,
            Self::Ingestion => 5,
            Self::Planning => 6,
            Self::ArtifactLoad => 7,
            Self::ArtifactPersist => 8,
            Self::Execution => 9,
            Self::ResultValidation => 10,
            Self::Teardown => 11,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InputGeneration => "input_generation",
            Self::Initialization => "initialization",
            Self::Synthesis => "synthesis",
            Self::Verification => "verification",
            Self::Allocation => "allocation",
            Self::Ingestion => "ingestion",
            Self::Planning => "planning",
            Self::ArtifactLoad => "artifact_load",
            Self::ArtifactPersist => "artifact_persist",
            Self::Execution => "execution",
            Self::ResultValidation => "result_validation",
            Self::Teardown => "teardown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchContractError {
    DurationOverflow,
}

/// Explicit costs of a capability that can be installed and used repeatedly.
///
/// Creation is charged exactly once. Each entry in `steady_state` is one
/// independently measured reuse of that installed artifact. Invalidation is
/// explicit so it cannot disappear from an amortized claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactLifetime {
    cold_create: BenchContract,
    steady_state: Vec<BenchContract>,
    invalidation: BenchContract,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactLifetimeError {
    EmptySteadyState,
    ZeroUses,
    UseUnavailable(usize),
    DurationOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreakEven {
    AtUses(usize),
    NotReached,
}

impl ArtifactLifetime {
    pub fn try_new(
        cold_create: BenchContract,
        steady_state: Vec<BenchContract>,
        invalidation: BenchContract,
    ) -> Result<Self, ArtifactLifetimeError> {
        if steady_state.is_empty() {
            return Err(ArtifactLifetimeError::EmptySteadyState);
        }
        Ok(Self {
            cold_create,
            steady_state,
            invalidation,
        })
    }

    pub fn cold_create(&self) -> Result<Duration, ArtifactLifetimeError> {
        self.cold_create.lifecycle().map_err(map_contract_error)
    }

    pub fn hot(&self, use_index: usize) -> Result<Duration, ArtifactLifetimeError> {
        self.steady_state_contract(use_index)
            .map(BenchContract::hot)
    }

    pub fn lifecycle_one(&self, use_index: usize) -> Result<Duration, ArtifactLifetimeError> {
        let uses = use_index
            .checked_add(1)
            .ok_or(ArtifactLifetimeError::UseUnavailable(use_index))?;
        self.total_for_uses(uses)
    }

    /// The cost after installation, excluding creation and invalidation.
    pub fn steady_state(&self, uses: usize) -> Result<Duration, ArtifactLifetimeError> {
        self.total_steady_state(uses)
    }

    /// Creation once, measured uses, and invalidation once.
    pub fn amortized(&self, uses: usize) -> Result<Duration, ArtifactLifetimeError> {
        self.total_for_uses(uses)
    }

    pub fn invalidation(&self) -> Result<Duration, ArtifactLifetimeError> {
        self.invalidation.lifecycle().map_err(map_contract_error)
    }

    /// Finds the first measured use count where the candidate is strictly
    /// cheaper. It is a deterministic cost calculation, not statistical
    /// promotion; use `StrategyEvidence` for paired physical claims.
    pub fn first_break_even_against(
        &self,
        baseline: &Self,
        max_uses: usize,
    ) -> Result<BreakEven, ArtifactLifetimeError> {
        if max_uses == 0 {
            return Err(ArtifactLifetimeError::ZeroUses);
        }
        self.require_uses(max_uses)?;
        baseline.require_uses(max_uses)?;
        for uses in 1..=max_uses {
            if self.total_for_uses(uses)? < baseline.total_for_uses(uses)? {
                return Ok(BreakEven::AtUses(uses));
            }
        }
        Ok(BreakEven::NotReached)
    }

    fn total_for_uses(&self, uses: usize) -> Result<Duration, ArtifactLifetimeError> {
        let cold_create = self.cold_create()?;
        let steady_state = self.total_steady_state(uses)?;
        let invalidation = self.invalidation()?;
        add_duration(add_duration(cold_create, steady_state)?, invalidation)
    }

    fn total_steady_state(&self, uses: usize) -> Result<Duration, ArtifactLifetimeError> {
        self.require_uses(uses)?;
        self.steady_state
            .iter()
            .take(uses)
            .try_fold(Duration::ZERO, |total, contract| {
                add_duration(total, contract.lifecycle().map_err(map_contract_error)?)
            })
    }

    fn require_uses(&self, uses: usize) -> Result<(), ArtifactLifetimeError> {
        if uses == 0 {
            return Err(ArtifactLifetimeError::ZeroUses);
        }
        if uses > self.steady_state.len() {
            return Err(ArtifactLifetimeError::UseUnavailable(uses));
        }
        Ok(())
    }

    fn steady_state_contract(
        &self,
        use_index: usize,
    ) -> Result<&BenchContract, ArtifactLifetimeError> {
        self.steady_state
            .get(use_index)
            .ok_or(ArtifactLifetimeError::UseUnavailable(use_index + 1))
    }
}

fn add_duration(left: Duration, right: Duration) -> Result<Duration, ArtifactLifetimeError> {
    left.checked_add(right)
        .ok_or(ArtifactLifetimeError::DurationOverflow)
}

const fn map_contract_error(_: BenchContractError) -> ArtifactLifetimeError {
    ArtifactLifetimeError::DurationOverflow
}

/// A comparable timing record with a fixed boundary vocabulary.
///
/// `HOT` is exactly the `Execution` phase. `LIFECYCLE` is the checked sum of
/// every phase in [`BenchPhase::ALL`]. Neither metric may substitute for the
/// other in a comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchContract {
    durations: [Duration; 12],
}

impl BenchContract {
    pub const fn empty() -> Self {
        Self {
            durations: [Duration::ZERO; 12],
        }
    }

    /// Returns a new record with `phase` replaced, preserving the original.
    pub fn with_phase(mut self, phase: BenchPhase, duration: Duration) -> Self {
        self.durations[phase.index()] = duration;
        self
    }

    pub fn with_added_phase(
        mut self,
        phase: BenchPhase,
        duration: Duration,
    ) -> Result<Self, BenchContractError> {
        self.durations[phase.index()] = self.durations[phase.index()]
            .checked_add(duration)
            .ok_or(BenchContractError::DurationOverflow)?;
        Ok(self)
    }

    pub const fn phase(&self, phase: BenchPhase) -> Duration {
        self.durations[phase.index()]
    }

    pub const fn hot(&self) -> Duration {
        self.phase(BenchPhase::Execution)
    }

    pub fn lifecycle(&self) -> Result<Duration, BenchContractError> {
        BenchPhase::ALL
            .iter()
            .try_fold(Duration::ZERO, |total, phase| {
                total
                    .checked_add(self.phase(*phase))
                    .ok_or(BenchContractError::DurationOverflow)
            })
    }
}

impl Default for BenchContract {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn hot_contains_only_execution() {
        let contract = BenchContract::empty()
            .with_phase(BenchPhase::Initialization, Duration::from_millis(3))
            .with_phase(BenchPhase::Execution, Duration::from_millis(5))
            .with_phase(BenchPhase::ResultValidation, Duration::from_millis(7));

        assert_eq!(contract.hot(), Duration::from_millis(5));
    }

    #[test]
    fn lifecycle_sums_every_reported_phase() {
        let contract = BenchContract::empty()
            .with_phase(BenchPhase::InputGeneration, Duration::from_millis(1))
            .with_phase(BenchPhase::Allocation, Duration::from_millis(2))
            .with_phase(BenchPhase::Execution, Duration::from_millis(3))
            .with_phase(BenchPhase::Teardown, Duration::from_millis(4));

        assert_eq!(contract.lifecycle().unwrap(), Duration::from_millis(10));
    }

    #[test]
    fn adding_a_phase_returns_a_new_contract() {
        let original = BenchContract::empty();
        let amended = original.with_phase(BenchPhase::Planning, Duration::from_millis(2));

        assert_eq!(original.phase(BenchPhase::Planning), Duration::ZERO);
        assert_eq!(
            amended.phase(BenchPhase::Planning),
            Duration::from_millis(2)
        );
    }

    #[test]
    fn repeated_phase_costs_are_added_explicitly() {
        let contract = BenchContract::empty()
            .with_added_phase(BenchPhase::Initialization, Duration::from_millis(2))
            .unwrap()
            .with_added_phase(BenchPhase::Initialization, Duration::from_millis(3))
            .unwrap();

        assert_eq!(
            contract.phase(BenchPhase::Initialization),
            Duration::from_millis(5)
        );
    }

    #[test]
    fn lifecycle_rejects_duration_overflow() {
        let contract = BenchContract::empty()
            .with_phase(BenchPhase::Initialization, Duration::MAX)
            .with_phase(BenchPhase::Execution, Duration::from_nanos(1));

        assert_eq!(
            contract.lifecycle(),
            Err(BenchContractError::DurationOverflow)
        );
    }
}
