//! Custos com proveniência explícita para a V5/Ω.
//!
//! `Declared` representa um modelo ou orçamento. `Measured` só deve ser usado
//! quando a instrumentação de execução efetivamente o produz.

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostOrigin {
    Declared,
    Measured,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostVector {
    pub compute_ops: u64,
    pub bytes_moved: u64,
    pub bytes_stored: u64,
    pub bytes_erased: u64,
    pub elapsed_ns: u64,
    pub energy_nj: Option<u64>,
    pub origin: CostOrigin,
}

impl CostVector {
    pub const fn declared(
        compute_ops: u64,
        bytes_moved: u64,
        bytes_stored: u64,
        bytes_erased: u64,
        elapsed_ns: u64,
    ) -> Self {
        Self {
            compute_ops,
            bytes_moved,
            bytes_stored,
            bytes_erased,
            elapsed_ns,
            energy_nj: None,
            origin: CostOrigin::Declared,
        }
    }

    pub const fn measured(
        compute_ops: u64,
        bytes_moved: u64,
        bytes_stored: u64,
        bytes_erased: u64,
        elapsed_ns: u64,
        energy_nj: u64,
    ) -> Self {
        Self {
            compute_ops,
            bytes_moved,
            bytes_stored,
            bytes_erased,
            elapsed_ns,
            energy_nj: Some(energy_nj),
            origin: CostOrigin::Measured,
        }
    }

    pub fn ensure_comparable_to(self, other: Self) -> Result<(), CostError> {
        if self.origin == other.origin {
            Ok(())
        } else {
            Err(CostError::MixedOrigins)
        }
    }

    pub fn weighted_total(self, weights: CostWeights) -> f64 {
        weights.compute * self.compute_ops as f64
            + weights.move_data * self.bytes_moved as f64
            + weights.store * self.bytes_stored as f64
            + weights.erase * self.bytes_erased as f64
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostWeights {
    pub compute: f64,
    pub move_data: f64,
    pub store: f64,
    pub erase: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            compute: 1.0,
            move_data: 4.0,
            store: 1.5,
            erase: 8.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostError {
    MixedOrigins,
}

impl Display for CostError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MixedOrigins => write!(
                f,
                "declared and measured costs cannot support the same efficiency comparison"
            ),
        }
    }
}

impl Error for CostError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_cost_cannot_be_compared_to_measured_cost() {
        let declared = CostVector::declared(1, 8, 0, 0, 1);
        let measured = CostVector::measured(1, 8, 0, 0, 1, 1);

        assert_eq!(
            declared.ensure_comparable_to(measured),
            Err(CostError::MixedOrigins)
        );
    }

    #[test]
    fn movement_and_erasure_are_more_expensive_than_equal_compute_by_default() {
        let compute = CostVector::declared(10, 0, 0, 0, 0);
        let movement = CostVector::declared(0, 10, 0, 0, 0);
        let erase = CostVector::declared(0, 0, 0, 10, 0);

        assert!(
            movement.weighted_total(CostWeights::default())
                > compute.weighted_total(CostWeights::default())
        );
        assert!(
            erase.weighted_total(CostWeights::default())
                > movement.weighted_total(CostWeights::default())
        );
    }
}
