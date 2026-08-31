//! Álgebra de custos AXON-Λ.
//!
//! Os valores desta camada são unidades comparáveis declaradas pelo chamador.
//! Não representam watts, joules ou telemetria física sem uma calibração externa.

use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostVector {
    pub energy_units: u64,
    pub bytes_moved: u64,
    pub latency_units: u64,
    pub memory_bytes: u64,
    pub risk_milliunits: u64,
}

impl CostVector {
    pub const fn new(
        energy_units: u64,
        bytes_moved: u64,
        latency_units: u64,
        memory_bytes: u64,
        risk_milliunits: u64,
    ) -> Self {
        Self {
            energy_units,
            bytes_moved,
            latency_units,
            memory_bytes,
            risk_milliunits,
        }
    }

    /// Composição sequencial no semiring de custos: cada componente se soma.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            energy_units: self.energy_units.checked_add(other.energy_units)?,
            bytes_moved: self.bytes_moved.checked_add(other.bytes_moved)?,
            latency_units: self.latency_units.checked_add(other.latency_units)?,
            memory_bytes: self.memory_bytes.checked_add(other.memory_bytes)?,
            risk_milliunits: self.risk_milliunits.checked_add(other.risk_milliunits)?,
        })
    }

    /// Verdadeiro quando este custo é não-pior em todas as dimensões e melhor
    /// em pelo menos uma; custos iguais permanecem como uma única opção.
    pub fn dominates(self, other: Self) -> bool {
        let left = [
            self.energy_units,
            self.bytes_moved,
            self.latency_units,
            self.memory_bytes,
            self.risk_milliunits,
        ];
        let right = [
            other.energy_units,
            other.bytes_moved,
            other.latency_units,
            other.memory_bytes,
            other.risk_milliunits,
        ];
        left.iter().zip(right).all(|(a, b)| a <= &b) && left.iter().zip(right).any(|(a, b)| a < &b)
    }

    pub fn weighted_score(self, weights: CostWeights) -> u128 {
        let costs = [
            self.energy_units,
            self.bytes_moved,
            self.latency_units,
            self.memory_bytes,
            self.risk_milliunits,
        ];
        let weights = [
            weights.energy,
            weights.bytes,
            weights.latency,
            weights.memory,
            weights.risk,
        ];
        costs
            .into_iter()
            .zip(weights)
            .fold(0_u128, |score, (cost, weight)| {
                score.saturating_add(u128::from(cost).saturating_mul(u128::from(weight)))
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostWeights {
    pub energy: u64,
    pub bytes: u64,
    pub latency: u64,
    pub memory: u64,
    pub risk: u64,
}

impl CostWeights {
    pub const fn new(energy: u64, bytes: u64, latency: u64, memory: u64, risk: u64) -> Self {
        Self {
            energy,
            bytes,
            latency,
            memory,
            risk,
        }
    }

    pub const fn latency_only() -> Self {
        Self::new(0, 0, 1, 0, 0)
    }

    pub const fn memory_only() -> Self {
        Self::new(0, 0, 0, 1, 0)
    }
}

/// Conjunto mínimo de custos não dominados. A seleção por preço físico vem
/// depois, para que um backend não imponha sua preferência ao contrato semântico.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParetoFrontier {
    options: Vec<CostVector>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostError {
    EmptyFrontier,
}

impl Display for CostError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFrontier => write!(formatter, "Pareto frontier requires one cost option"),
        }
    }
}

impl Error for CostError {}

impl ParetoFrontier {
    pub fn new(mut costs: Vec<CostVector>) -> Result<Self, CostError> {
        if costs.is_empty() {
            return Err(CostError::EmptyFrontier);
        }
        costs.sort_by_key(|cost| {
            (
                cost.energy_units,
                cost.bytes_moved,
                cost.latency_units,
                cost.memory_bytes,
                cost.risk_milliunits,
            )
        });
        costs.dedup();
        let options = costs
            .iter()
            .copied()
            .filter(|candidate| {
                !costs
                    .iter()
                    .copied()
                    .any(|other| other != *candidate && other.dominates(*candidate))
            })
            .collect();
        Ok(Self { options })
    }

    pub fn options(&self) -> &[CostVector] {
        &self.options
    }

    pub fn select(&self, weights: CostWeights) -> CostVector {
        self.options
            .iter()
            .copied()
            .min_by_key(|cost| cost.weighted_score(weights))
            .expect("ParetoFrontier construction rejects an empty cost set")
    }
}
