use crate::{ChangeError, ChangeStructure, IncrementalOp, ReplaceDelta, SumFold, VectorU64};

/// Input grammar is intentionally small: an operator declaration, never an updater callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FoldSpec {
    AddModU64,
    MinU64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgebraicClass {
    CommutativeGroup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaintenanceState {
    ModularTotal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateRule {
    SubtractOldThenAddNew,
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
    CertificateViolation,
}

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
            FoldSpec::MinU64 => Err(ForgeError::UnsupportedFold(FoldSpec::MinU64)),
        }
    }
}
