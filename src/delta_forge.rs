use std::fmt;

use crate::{ChangeError, ChangeStructure, IncrementalOp, ReplaceDelta, SumFold, VectorU64};

/// Input grammar is intentionally small: an operator declaration, never an updater callback.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FoldSpec {
    AddModU64,
    AverageExactU64,
    MinU64,
}

impl FoldSpec {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AddModU64 => "AddModU64",
            Self::AverageExactU64 => "AverageExactU64",
            Self::MinU64 => "MinU64",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "AddModU64" => Some(Self::AddModU64),
            "AverageExactU64" => Some(Self::AverageExactU64),
            "MinU64" => Some(Self::MinU64),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgebraicClass {
    CommutativeGroup,
    CommutativeGroupWithInvariantCardinality,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaintenanceState {
    ModularTotal,
    ExactSumAndCount,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateRule {
    SubtractOldThenAddNew,
    SubtractOldThenAddNewPreserveCount,
}

/// Small declarative certificate checked against concrete state and changes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeltaCertificate {
    fold: FoldSpec,
    algebra: AlgebraicClass,
    maintenance_state: MaintenanceState,
    update_rule: UpdateRule,
}

impl DeltaCertificate {
    pub const fn fold(self) -> FoldSpec {
        self.fold
    }

    pub const fn algebra(self) -> AlgebraicClass {
        self.algebra
    }

    pub const fn maintenance_state(self) -> MaintenanceState {
        self.maintenance_state
    }

    pub const fn update_rule(self) -> UpdateRule {
        self.update_rule
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForgeError {
    UnsupportedFold(FoldSpec),
    InvalidChange(ChangeError),
    EmptyInput,
    ArithmeticOverflow,
    InvalidCache,
    CertificateViolation,
}

impl fmt::Display for ForgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFold(spec) => write!(formatter, "unsupported fold {}", spec.as_str()),
            Self::InvalidChange(error) => write!(formatter, "invalid change: {error:?}"),
            Self::EmptyInput => formatter.write_str("average requires at least one value"),
            Self::ArithmeticOverflow => formatter.write_str("artifact arithmetic overflowed"),
            Self::InvalidCache => {
                formatter.write_str("artifact cache is inconsistent with the change")
            }
            Self::CertificateViolation => formatter.write_str("artifact certificate was violated"),
        }
    }
}

impl std::error::Error for ForgeError {}

/// A derived program for `fold(Add mod u64)`. It owns no mutable state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DerivedSumPlan {
    certificate: DeltaCertificate,
    fold: SumFold,
}

impl DerivedSumPlan {
    pub const fn certificate(&self) -> DeltaCertificate {
        self.certificate
    }

    pub fn full(&self, input: &Vec<u64>) -> (u64, u64) {
        self.fold.full(input)
    }

    pub fn delta(&self, change: &ReplaceDelta, cache: &u64) -> Result<(u64, u64), ForgeError> {
        self.fold
            .delta(change, cache)
            .map_err(ForgeError::InvalidChange)
    }

    pub fn apply_output_delta(&self, total: u64, delta: u64) -> u64 {
        total.wrapping_add(delta)
    }

    /// Checks the concrete proof obligation before a derived plan can be promoted.
    pub fn check(&self, before: &Vec<u64>, change: &ReplaceDelta) -> Result<(), ForgeError> {
        let after = VectorU64
            .apply(before, change)
            .map_err(ForgeError::InvalidChange)?;
        let (old_total, cache) = self.full(before);
        let (output_delta, next_cache) = self.delta(change, &cache)?;
        let (expected_total, expected_cache) = self.full(&after);
        (self.apply_output_delta(old_total, output_delta) == expected_total
            && next_cache == expected_cache)
            .then_some(())
            .ok_or(ForgeError::CertificateViolation)
    }
}

/// An average represented as an exact fraction; formatting and rounding are deliberately a
/// caller concern, so the artifact never loses information during maintenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactAverage {
    numerator: u128,
    denominator: usize,
}

impl ExactAverage {
    pub fn new(numerator: u128, denominator: usize) -> Result<Self, ForgeError> {
        (denominator > 0)
            .then_some(Self {
                numerator,
                denominator,
            })
            .ok_or(ForgeError::EmptyInput)
    }

    pub const fn numerator(self) -> u128 {
        self.numerator
    }

    pub const fn denominator(self) -> usize {
        self.denominator
    }
}

/// The auxiliary state is derived by the declared average capability, never supplied by a
/// caller. It makes the exact replacement rule explicit and independently checkable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AverageCache {
    total: u128,
    count: usize,
}

impl AverageCache {
    pub const fn total(self) -> u128 {
        self.total
    }

    pub const fn count(self) -> usize {
        self.count
    }
}

/// A derived program for `average(u64)` with an exact `(sum, count)` cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DerivedAveragePlan {
    certificate: DeltaCertificate,
}

impl DerivedAveragePlan {
    pub const fn certificate(&self) -> DeltaCertificate {
        self.certificate
    }

    pub fn full(&self, input: &[u64]) -> Result<(ExactAverage, AverageCache), ForgeError> {
        let count = input.len();
        let total = input.iter().try_fold(0_u128, |total, value| {
            total
                .checked_add(u128::from(*value))
                .ok_or(ForgeError::ArithmeticOverflow)
        })?;
        let average = ExactAverage::new(total, count)?;
        Ok((average, AverageCache { total, count }))
    }

    pub fn delta(
        &self,
        change: &ReplaceDelta,
        cache: &AverageCache,
    ) -> Result<(ExactAverage, AverageCache), ForgeError> {
        let total = change
            .changes()
            .iter()
            .try_fold(cache.total, |total, change| {
                total
                    .checked_sub(u128::from(change.old()))
                    .ok_or(ForgeError::InvalidCache)?
                    .checked_add(u128::from(change.new_value()))
                    .ok_or(ForgeError::ArithmeticOverflow)
            })?;
        let next_cache = AverageCache {
            total,
            count: cache.count,
        };
        Ok((ExactAverage::new(total, cache.count)?, next_cache))
    }

    /// Checks the concrete proof obligation before an average artifact can be promoted.
    pub fn check(&self, before: &[u64], change: &ReplaceDelta) -> Result<(), ForgeError> {
        let before = before.to_vec();
        let after = VectorU64
            .apply(&before, change)
            .map_err(ForgeError::InvalidChange)?;
        let (old_average, cache) = self.full(&before)?;
        let (next_average, next_cache) = self.delta(change, &cache)?;
        let (expected_average, expected_cache) = self.full(&after)?;
        (old_average.denominator() == next_average.denominator()
            && next_average == expected_average
            && next_cache == expected_cache)
            .then_some(())
            .ok_or(ForgeError::CertificateViolation)
    }
}

/// A capability-derived artifact. Every variant carries the program and certificate needed to
/// execute and validate that capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivedArtifact {
    Sum(DerivedSumPlan),
    Average(DerivedAveragePlan),
}

pub struct DeltaForge;

impl DeltaForge {
    pub const fn synthesize(spec: FoldSpec) -> Result<DerivedSumPlan, ForgeError> {
        match spec {
            FoldSpec::AddModU64 => Ok(DerivedSumPlan {
                certificate: DeltaCertificate {
                    fold: FoldSpec::AddModU64,
                    algebra: AlgebraicClass::CommutativeGroup,
                    maintenance_state: MaintenanceState::ModularTotal,
                    update_rule: UpdateRule::SubtractOldThenAddNew,
                },
                fold: SumFold,
            }),
            FoldSpec::AverageExactU64 | FoldSpec::MinU64 => Err(ForgeError::UnsupportedFold(spec)),
        }
    }

    /// Derives the complete artifact, including required auxiliary state, from a capability
    /// declaration. This is intentionally separate from the legacy SUM-only convenience API.
    pub const fn synthesize_capability(spec: FoldSpec) -> Result<DerivedArtifact, ForgeError> {
        match spec {
            FoldSpec::AddModU64 => match Self::synthesize(FoldSpec::AddModU64) {
                Ok(plan) => Ok(DerivedArtifact::Sum(plan)),
                Err(error) => Err(error),
            },
            FoldSpec::AverageExactU64 => Ok(DerivedArtifact::Average(DerivedAveragePlan {
                certificate: DeltaCertificate {
                    fold: FoldSpec::AverageExactU64,
                    algebra: AlgebraicClass::CommutativeGroupWithInvariantCardinality,
                    maintenance_state: MaintenanceState::ExactSumAndCount,
                    update_rule: UpdateRule::SubtractOldThenAddNewPreserveCount,
                },
            })),
            FoldSpec::MinU64 => Err(ForgeError::UnsupportedFold(FoldSpec::MinU64)),
        }
    }
}
