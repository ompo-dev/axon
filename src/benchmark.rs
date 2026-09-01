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
    Execution,
    ResultValidation,
    Teardown,
}

impl BenchPhase {
    pub const ALL: [Self; 10] = [
        Self::InputGeneration,
        Self::Initialization,
        Self::Synthesis,
        Self::Verification,
        Self::Allocation,
        Self::Ingestion,
        Self::Planning,
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
            Self::Execution => 7,
            Self::ResultValidation => 8,
            Self::Teardown => 9,
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

/// A comparable timing record with a fixed boundary vocabulary.
///
/// `HOT` is exactly the `Execution` phase. `LIFECYCLE` is the checked sum of
/// every phase in [`BenchPhase::ALL`]. Neither metric may substitute for the
/// other in a comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchContract {
    durations: [Duration; 10],
}

impl BenchContract {
    pub const fn empty() -> Self {
        Self {
            durations: [Duration::ZERO; 10],
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
