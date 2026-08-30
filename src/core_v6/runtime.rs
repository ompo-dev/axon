//! Runtime determinístico e single-threaded para o Factor Fabric.

use std::collections::BTreeSet;

use crate::core_v5::{CostOrigin, CostVector, CostWeights};

use super::factor::FactorGraph;
use super::ids::FactorId;
use super::message::{CognitiveMessage, MessageDisposition, MessageScheduler};
use super::metrics::CognitiveMetrics;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CognitiveBudget {
    pub max_operations: u64,
    pub max_bytes_moved: u64,
    pub max_weighted_cost: u64,
    pub origin: CostOrigin,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcessingOutcome {
    Suppressed,
    Processed { target: FactorId },
    BudgetExhausted,
}

#[derive(Clone, Debug)]
pub struct CognitiveRuntime {
    graph: FactorGraph,
    scheduler: MessageScheduler,
    metrics: CognitiveMetrics,
    budget: CognitiveBudget,
    used_operations: u64,
    used_bytes: u64,
    used_weighted_cost: f64,
    active: BTreeSet<FactorId>,
}

impl CognitiveRuntime {
    pub fn new(graph: FactorGraph, budget: CognitiveBudget) -> Self {
        Self {
            graph,
            scheduler: MessageScheduler::default(),
            metrics: CognitiveMetrics::default(),
            budget,
            used_operations: 0,
            used_bytes: 0,
            used_weighted_cost: 0.0,
            active: BTreeSet::new(),
        }
    }

    pub fn submit(&mut self, message: CognitiveMessage) -> ProcessingOutcome {
        match self.scheduler.submit(message, &mut self.metrics) {
            MessageDisposition::Suppressed => ProcessingOutcome::Suppressed,
            MessageDisposition::Enqueued(_) => self.process_next(),
        }
    }

    pub fn process_next(&mut self) -> ProcessingOutcome {
        let Some(candidate) = self.scheduler.peek() else {
            return ProcessingOutcome::Suppressed;
        };
        if candidate.cost.origin != self.budget.origin || !self.fits(candidate.cost) {
            return ProcessingOutcome::BudgetExhausted;
        }
        if self.graph.factor(candidate.to).is_none() {
            let _ = self.scheduler.next(&mut self.metrics);
            return ProcessingOutcome::Suppressed;
        }
        let message = self
            .scheduler
            .next(&mut self.metrics)
            .expect("candidate was peeked before processing");
        self.used_operations = self
            .used_operations
            .saturating_add(message.cost.compute_ops);
        self.used_bytes = self.used_bytes.saturating_add(message.cost.bytes_moved);
        self.used_weighted_cost += message.cost.weighted_total(CostWeights::default());
        self.active.insert(message.to);
        self.metrics.factor_visits = self.metrics.factor_visits.saturating_add(1);
        self.metrics.active_factors_peak = self
            .metrics
            .active_factors_peak
            .max(self.active.len() as u64);
        ProcessingOutcome::Processed { target: message.to }
    }

    pub fn metrics(&self) -> CognitiveMetrics {
        self.metrics
    }

    pub fn active(&self) -> &BTreeSet<FactorId> {
        &self.active
    }

    fn fits(&self, cost: CostVector) -> bool {
        self.used_operations.saturating_add(cost.compute_ops) <= self.budget.max_operations
            && self.used_bytes.saturating_add(cost.bytes_moved) <= self.budget.max_bytes_moved
            && self.used_weighted_cost + cost.weighted_total(CostWeights::default())
                <= self.budget.max_weighted_cost as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_v6::{
        Factor, FactorKind, MessagePayload, RelationKind, RepresentationKind, RevisionId,
    };

    fn graph() -> FactorGraph {
        FactorGraph::default()
            .insert(
                Factor::try_new(
                    FactorId(1),
                    FactorKind::Semantic,
                    RepresentationKind::Graph,
                    "a",
                    0.1,
                    RevisionId(1),
                )
                .unwrap(),
            )
            .unwrap()
            .insert(
                Factor::try_new(
                    FactorId(2),
                    FactorKind::Semantic,
                    RepresentationKind::Graph,
                    "b",
                    0.1,
                    RevisionId(1),
                )
                .unwrap(),
            )
            .unwrap()
            .connect(crate::core_v6::FactorEdge {
                from: FactorId(1),
                to: FactorId(2),
                relation: RelationKind::Causes,
            })
            .unwrap()
    }

    fn message(cost: CostVector) -> CognitiveMessage {
        CognitiveMessage {
            from: FactorId(1),
            to: FactorId(2),
            payload: MessagePayload::Residual("relevant".to_string()),
            residual_milliunits: 1_000,
            goal_milliunits: 0,
            uncertainty_milliunits: 0,
            information_gain_milliunits: 0,
            timestamp: 1,
            provenance: vec!["test".to_string()],
            cost,
        }
    }

    #[test]
    fn runtime_touches_only_target_factor_and_stops_before_exceeding_budget() {
        let budget = CognitiveBudget {
            max_operations: 2,
            max_bytes_moved: 8,
            max_weighted_cost: 100,
            origin: CostOrigin::Declared,
        };
        let mut runtime = CognitiveRuntime::new(graph(), budget);

        assert_eq!(
            runtime.submit(message(CostVector::declared(1, 4, 0, 0, 1))),
            ProcessingOutcome::Processed {
                target: FactorId(2)
            }
        );
        assert_eq!(runtime.active(), &BTreeSet::from([FactorId(2)]));
        assert_eq!(runtime.metrics().factor_visits, 1);
        assert_eq!(
            runtime.submit(message(CostVector::declared(2, 8, 0, 0, 1))),
            ProcessingOutcome::BudgetExhausted
        );
        assert_eq!(runtime.metrics().factor_visits, 1);
        assert_eq!(runtime.metrics().messages_processed, 1);
    }
}
