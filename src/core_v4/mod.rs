//! V4 experimental: plano de controle para cognição esparsa e progressiva.
//!
//! Este módulo decide qual estratégia pode receber orçamento. Ele não executa
//! codecs, simulações ou modelos de mundo e não altera o runtime existente.

mod firewall;
mod procedural;
mod scheduler;

pub use firewall::{
    FirewallDecision, KnowledgeMutation, KnowledgeTier, MemoryFirewall, ReversibleJournal,
};
pub use procedural::{
    ProceduralCircuit, ProceduralDispatch, ProceduralError, ProceduralFabric, ProceduralStep,
};
pub use scheduler::{
    CandidateStrategy, Capability, CognitiveLevel, CognitiveMode, CognitiveRequest,
    CognitiveScheduler, CognitiveSignals, ComputeBudget, ReframeEvidence, ResourceUse,
    ScheduledPlan, SchedulerError, StrategyOutcome,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn signals(goal: f32, uncertainty: f32, risk: f32) -> CognitiveSignals {
        CognitiveSignals::try_new(0.1, goal, uncertainty, risk, 0.1, 0.1).unwrap()
    }

    fn budget() -> ComputeBudget {
        ComputeBudget::new(4, 512, 100).unwrap()
    }

    #[test]
    fn scheduler_prefers_an_affordable_coarse_lookup_when_its_confidence_is_sufficient() {
        let request = CognitiveRequest::new(
            "capital-question",
            signals(0.8, 0.1, 0.1),
            BTreeSet::from([
                Capability::SemanticLookup,
                Capability::DeliberativeReasoning,
            ]),
        );
        let candidates = [
            CandidateStrategy::try_new(
                CognitiveMode::Retrieve,
                CognitiveLevel::L0,
                0.96,
                0.2,
                ResourceUse::new(1, 64, 4),
                BTreeSet::from([Capability::SemanticLookup]),
            )
            .unwrap(),
            CandidateStrategy::try_new(
                CognitiveMode::Think,
                CognitiveLevel::L2,
                0.99,
                0.3,
                ResourceUse::new(3, 400, 70),
                BTreeSet::from([Capability::DeliberativeReasoning]),
            )
            .unwrap(),
        ];

        let plan = CognitiveScheduler::new(budget())
            .plan(&request, &candidates)
            .unwrap();

        assert_eq!(plan.mode, CognitiveMode::Retrieve);
        assert_eq!(plan.level, CognitiveLevel::L0);
    }

    #[test]
    fn scheduler_never_selects_a_strategy_without_its_required_capability() {
        let request = CognitiveRequest::new(
            "unavailable-simulation",
            signals(1.0, 0.8, 0.8),
            BTreeSet::from([Capability::SemanticLookup]),
        );
        let candidates = [
            CandidateStrategy::try_new(
                CognitiveMode::Simulate,
                CognitiveLevel::L4,
                0.99,
                0.9,
                ResourceUse::new(3, 400, 70),
                BTreeSet::from([Capability::Simulation]),
            )
            .unwrap(),
            CandidateStrategy::try_new(
                CognitiveMode::Ask,
                CognitiveLevel::L0,
                0.80,
                0.1,
                ResourceUse::new(1, 16, 2),
                BTreeSet::new(),
            )
            .unwrap(),
        ];

        let plan = CognitiveScheduler::new(budget())
            .plan(&request, &candidates)
            .unwrap();

        assert_eq!(plan.mode, CognitiveMode::Ask);
    }

    #[test]
    fn verified_low_cost_outcomes_can_promote_a_cheaper_strategy_for_the_same_task() {
        let request = CognitiveRequest::new(
            "known-procedure",
            signals(0.8, 0.1, 0.1),
            BTreeSet::from([Capability::ProceduralCircuit, Capability::SemanticLookup]),
        );
        let candidates = [
            CandidateStrategy::try_new(
                CognitiveMode::Reflex,
                CognitiveLevel::L0,
                0.70,
                0.1,
                ResourceUse::new(1, 64, 4),
                BTreeSet::from([Capability::ProceduralCircuit]),
            )
            .unwrap(),
            CandidateStrategy::try_new(
                CognitiveMode::Retrieve,
                CognitiveLevel::L0,
                0.85,
                0.1,
                ResourceUse::new(1, 64, 4),
                BTreeSet::from([Capability::SemanticLookup]),
            )
            .unwrap(),
        ];
        let scheduler = CognitiveScheduler::new(budget());
        assert_eq!(
            scheduler.plan(&request, &candidates).unwrap().mode,
            CognitiveMode::Retrieve
        );

        let learned = (0..3).fold(scheduler, |scheduler, _| {
            scheduler.record_outcome(
                "known-procedure",
                CognitiveMode::Reflex,
                StrategyOutcome {
                    verified: true,
                    actual_use: ResourceUse::new(1, 64, 4),
                },
            )
        });

        assert_eq!(
            learned.plan(&request, &candidates).unwrap().mode,
            CognitiveMode::Reflex
        );
    }

    #[test]
    fn structured_persistent_residual_gates_reframe_over_parameter_adaptation() {
        let request = CognitiveRequest::new(
            "causal-anomaly",
            signals(1.0, 0.8, 0.8),
            BTreeSet::from([Capability::DeliberativeReasoning, Capability::Reframe]),
        )
        .with_reframe_evidence(ReframeEvidence::try_new(0.9, 0.9, 0.9, 0.05).unwrap());
        let candidates = [
            CandidateStrategy::try_new(
                CognitiveMode::Think,
                CognitiveLevel::L2,
                0.90,
                0.4,
                ResourceUse::new(2, 160, 20),
                BTreeSet::from([Capability::DeliberativeReasoning]),
            )
            .unwrap(),
            CandidateStrategy::try_new(
                CognitiveMode::Reframe,
                CognitiveLevel::L5,
                0.85,
                0.9,
                ResourceUse::new(4, 480, 90),
                BTreeSet::from([Capability::Reframe]),
            )
            .unwrap(),
        ];

        let plan = CognitiveScheduler::new(budget())
            .plan(&request, &candidates)
            .unwrap();

        assert_eq!(plan.mode, CognitiveMode::Reframe);
    }

    #[test]
    fn reframe_pressure_does_not_silently_degrade_to_asking_when_a_reframe_is_available() {
        let request = CognitiveRequest::new(
            "structural-anomaly",
            signals(1.0, 0.9, 0.9),
            BTreeSet::from([Capability::Reframe]),
        )
        .with_reframe_evidence(ReframeEvidence::try_new(0.9, 0.9, 0.9, 0.05).unwrap());
        let candidates = [
            CandidateStrategy::try_new(
                CognitiveMode::Ask,
                CognitiveLevel::L0,
                0.99,
                0.9,
                ResourceUse::new(1, 8, 1),
                BTreeSet::new(),
            )
            .unwrap(),
            CandidateStrategy::try_new(
                CognitiveMode::Reframe,
                CognitiveLevel::L5,
                0.85,
                0.8,
                ResourceUse::new(4, 480, 90),
                BTreeSet::from([Capability::Reframe]),
            )
            .unwrap(),
        ];

        let plan = CognitiveScheduler::new(budget())
            .plan(&request, &candidates)
            .unwrap();

        assert_eq!(plan.mode, CognitiveMode::Reframe);
    }

    #[test]
    fn procedural_fabric_compiles_only_verified_repetition_and_deoptimizes_on_guard_failure() {
        let context = BTreeSet::from(["polynomial".to_string(), "integer-exponent".to_string()]);
        let first = ProceduralFabric::new(2)
            .unwrap()
            .record_verified_success("differentiate-polynomial", &[2, 5, 9], &context)
            .unwrap();
        assert!(first.compiled.is_none());
        let second = first
            .fabric
            .record_verified_success("differentiate-polynomial", &[2, 5, 9], &context)
            .unwrap();
        let circuit = second.compiled.unwrap();

        assert_eq!(
            second.fabric.dispatch(
                &circuit,
                &BTreeSet::from(["fractional-exponent".to_string()])
            ),
            ProceduralDispatch::Deoptimized {
                circuit_id: circuit.id
            }
        );
    }

    #[test]
    fn protected_knowledge_forks_a_candidate_and_the_journal_rolls_back_exactly() {
        let mutation = KnowledgeMutation::new("power-rule", "n*x^(n-1)", "incorrect-rule");
        let firewall = MemoryFirewall;

        let decision = firewall.propose(KnowledgeTier::Protected, mutation.clone());
        assert_eq!(
            decision,
            FirewallDecision::ForkCandidate {
                mutation: mutation.clone()
            }
        );

        let journal = ReversibleJournal::default().append(mutation.clone());
        assert_eq!(journal.rollback(mutation.id), Some(mutation));
    }
}
