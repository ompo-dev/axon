//! Factor Fabric determinístico para o experimento Demand × Delta.
//!
//! O grafo de referência é uma coleção de cadeias lineares independentes
//! realmente alocadas em memória. A topologia restrita permite provar paridade
//! exata com recomputação total sem apresentá-la como uma Factor Fabric geral.

use std::error::Error;
use std::fmt::{Display, Formatter};

use super::cost::{CostVector, CostWeights};

const RULE_MULTIPLIER: u64 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Demand {
    pub goal: usize,
    pub max_error_milliunits: u64,
}

impl Demand {
    pub const fn exact(goal: usize) -> Self {
        Self {
            goal,
            max_error_milliunits: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvidenceDelta {
    pub factor: usize,
    pub replacement_value: u64,
}

impl EvidenceDelta {
    pub const fn new(factor: usize, replacement_value: u64) -> Self {
        Self {
            factor,
            replacement_value,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CognitiveSlice {
    pub demanded_factors: usize,
    pub changed_factors: usize,
    pub active_factors: usize,
}

/// Nó materializado do subconjunto executável de AXON-Λ. A regra afim permite
/// uma derivada incremental exata, mas os links continuam explícitos para que
/// um milhão de Factors não seja reduzido a uma fórmula analítica no sweep.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorNode {
    predecessor: Option<usize>,
    successor: Option<usize>,
    root_value: u64,
    additive_constant: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeltaOverlay {
    pub changed_values: Vec<(usize, u64)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveMode {
    Reuse,
    DeltaPropagation,
    FullRecompute,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryResult {
    pub value: u64,
    pub mode: AdaptiveMode,
    pub slice: CognitiveSlice,
    /// Custo declarado do plano lógico, separado do tempo físico do sweep.
    pub estimated_cost: CostVector,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FabricError {
    InvalidShape,
    InvalidFactor,
}

impl Display for FabricError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape => write!(
                formatter,
                "factor count must be positive and divisible by chain length"
            ),
            Self::InvalidFactor => {
                write!(formatter, "demand or evidence references an unknown factor")
            }
        }
    }
}

impl Error for FabricError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainFabric {
    nodes: Vec<FactorNode>,
    base_values: Vec<u64>,
}

impl ChainFabric {
    pub fn new(node_count: usize, chain_len: usize) -> Result<Self, FabricError> {
        if node_count == 0 || chain_len == 0 || !node_count.is_multiple_of(chain_len) {
            return Err(FabricError::InvalidShape);
        }
        let mut nodes = Vec::with_capacity(node_count);
        for factor in 0..node_count {
            let starts_chain = factor % chain_len == 0;
            let ends_chain = (factor + 1) % chain_len == 0;
            nodes.push(FactorNode {
                predecessor: if starts_chain { None } else { Some(factor - 1) },
                successor: if ends_chain { None } else { Some(factor + 1) },
                root_value: root_value(factor / chain_len),
                additive_constant: (factor as u64 + 1).wrapping_mul(17),
            });
        }
        let mut base_values = Vec::with_capacity(node_count);
        for factor in 0..node_count {
            base_values.push(Self::evaluate_node(&nodes, &base_values, factor));
        }
        Ok(Self { nodes, base_values })
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Limite de bytes para uma construção, sem contar buffers temporários de
    /// recomputação. Usado para recusar sweeps que excederiam a memória segura.
    pub fn estimated_storage_bytes(node_count: usize) -> Option<u64> {
        let per_factor =
            std::mem::size_of::<FactorNode>().checked_add(std::mem::size_of::<u64>())?;
        u64::try_from(node_count)
            .ok()?
            .checked_mul(u64::try_from(per_factor).ok()?)
    }

    pub fn base_value(&self, factor: usize) -> Result<u64, FabricError> {
        self.base_values
            .get(factor)
            .copied()
            .ok_or(FabricError::InvalidFactor)
    }

    pub fn slice(
        &self,
        demand: Demand,
        change: EvidenceDelta,
    ) -> Result<CognitiveSlice, FabricError> {
        self.validate(demand, change)?;
        let demanded_factors = self.count_backward(demand.goal);
        let changed_factors = self.count_forward(change.factor);
        let active_factors = self.path_to_goal(change.factor, demand.goal).unwrap_or(0);
        Ok(CognitiveSlice {
            demanded_factors,
            changed_factors,
            active_factors,
        })
    }

    /// Baseline de referência: recalcula todas as cadeias e materializa o
    /// estado completo, mesmo quando apenas uma decisão local é pedida.
    pub fn full_query(
        &self,
        demand: Demand,
        change: EvidenceDelta,
    ) -> Result<QueryResult, FabricError> {
        let slice = self.slice(demand, change)?;
        let values = self.full_values_after(change);
        Ok(QueryResult {
            value: values[demand.goal],
            mode: AdaptiveMode::FullRecompute,
            slice,
            estimated_cost: self.full_cost(),
        })
    }

    /// Propaga uma mudança exata por `DF(x, Δx) = 3 * Δx`; somente a parte que
    /// alcança a decisão demandada é executada. É válido porque cada link
    /// materializado usa uma regra afim no anel de inteiros com overflow modular.
    pub fn query(
        &self,
        demand: Demand,
        change: EvidenceDelta,
        weights: CostWeights,
    ) -> Result<QueryResult, FabricError> {
        let slice = self.slice(demand, change)?;
        if slice.active_factors == 0 {
            return Ok(QueryResult {
                value: self.base_values[demand.goal],
                mode: AdaptiveMode::Reuse,
                slice,
                estimated_cost: CostVector::new(0, 0, 0, 0, 0),
            });
        }
        let delta_cost = self.delta_cost(slice.active_factors);
        if self.full_cost().weighted_score(weights) <= delta_cost.weighted_score(weights) {
            return self.full_query(demand, change);
        }

        let mut delta = change
            .replacement_value
            .wrapping_sub(self.base_values[change.factor]);
        for _ in 1..slice.active_factors {
            delta = delta.wrapping_mul(RULE_MULTIPLIER);
        }
        Ok(QueryResult {
            value: self.base_values[demand.goal].wrapping_add(delta),
            mode: AdaptiveMode::DeltaPropagation,
            slice,
            estimated_cost: delta_cost,
        })
    }

    fn validate(&self, demand: Demand, change: EvidenceDelta) -> Result<(), FabricError> {
        if demand.goal >= self.nodes.len() || change.factor >= self.nodes.len() {
            Err(FabricError::InvalidFactor)
        } else {
            Ok(())
        }
    }

    fn full_cost(&self) -> CostVector {
        let nodes = self.nodes.len() as u64;
        CostVector::new(
            nodes.saturating_mul(2),
            nodes.saturating_mul(8),
            nodes,
            nodes.saturating_mul(8),
            0,
        )
    }

    fn delta_cost(&self, active_factors: usize) -> CostVector {
        let active = active_factors as u64;
        CostVector::new(
            active.saturating_mul(2).saturating_add(4),
            active.saturating_mul(8).saturating_add(32),
            active.saturating_add(4),
            active.saturating_mul(8).saturating_add(32),
            0,
        )
    }

    /// Overlay esparso que representa a propagação em todo o cone de mudança.
    /// Ele serve para verificação de paridade de estado fora da janela física de
    /// benchmark; a consulta sob demanda continua a executar só a interseção.
    pub fn delta_overlay(&self, change: EvidenceDelta) -> Result<DeltaOverlay, FabricError> {
        if change.factor >= self.nodes.len() {
            return Err(FabricError::InvalidFactor);
        }
        let mut changed_values = Vec::new();
        let mut factor = change.factor;
        let mut value = change.replacement_value;
        loop {
            changed_values.push((factor, value));
            let Some(successor) = self.nodes[factor].successor else {
                break;
            };
            value = apply_rule(value, self.nodes[successor].additive_constant);
            factor = successor;
        }
        Ok(DeltaOverlay { changed_values })
    }

    /// Verifica cada valor da recomputação contra `base + overlay`, não apenas a
    /// decisão final. Esta é a barreira contra benchmarks que aceleram a query,
    /// mas deixam o estado incremental semanticamente incorreto.
    pub fn delta_overlay_matches_full(&self, change: EvidenceDelta) -> Result<bool, FabricError> {
        let full = self.full_values_after(change);
        let overlay = self.delta_overlay(change)?;
        let mut overlay_values = vec![None; self.nodes.len()];
        for (factor, value) in overlay.changed_values {
            overlay_values[factor] = Some(value);
        }
        Ok(full.iter().enumerate().all(|(factor, full_value)| {
            overlay_values[factor].unwrap_or(self.base_values[factor]) == *full_value
        }))
    }

    fn full_values_after(&self, change: EvidenceDelta) -> Vec<u64> {
        let mut values = Vec::with_capacity(self.nodes.len());
        for factor in 0..self.nodes.len() {
            let computed = Self::evaluate_node(&self.nodes, &values, factor);
            values.push(if factor == change.factor {
                change.replacement_value
            } else {
                computed
            });
        }
        values
    }

    fn evaluate_node(nodes: &[FactorNode], values: &[u64], factor: usize) -> u64 {
        let node = &nodes[factor];
        node.predecessor.map_or(node.root_value, |predecessor| {
            apply_rule(values[predecessor], node.additive_constant)
        })
    }

    fn count_backward(&self, start: usize) -> usize {
        let mut count = 0;
        let mut current = Some(start);
        while let Some(factor) = current {
            count += 1;
            current = self.nodes[factor].predecessor;
        }
        count
    }

    fn count_forward(&self, start: usize) -> usize {
        let mut count = 0;
        let mut current = Some(start);
        while let Some(factor) = current {
            count += 1;
            current = self.nodes[factor].successor;
        }
        count
    }

    fn path_to_goal(&self, start: usize, goal: usize) -> Option<usize> {
        let mut count = 0;
        let mut current = Some(start);
        while let Some(factor) = current {
            count += 1;
            if factor == goal {
                return Some(count);
            }
            current = self.nodes[factor].successor;
        }
        None
    }
}

fn root_value(group: usize) -> u64 {
    let mut state = group as u64 + 0x9E37_79B9_7F4A_7C15;
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^ (state >> 31)
}

fn apply_rule(previous: u64, additive_constant: u64) -> u64 {
    previous
        .wrapping_mul(RULE_MULTIPLIER)
        .wrapping_add(additive_constant)
}
