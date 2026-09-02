//! AXON-UIC: small, executable primitives for certified adaptive cognition.

pub mod artifact_store;
pub mod benchmark;
pub mod capability;
pub mod change;
pub mod cli;
pub mod delta;
pub mod delta_forge;
pub mod morphology;
pub mod refinement;
pub mod runtime;
pub mod solver;
pub mod strategy;
pub mod structure;
pub mod task;

pub use artifact_store::{ArtifactStatus, ArtifactStore, ArtifactStoreError, InstalledArtifact};

pub use benchmark::{
    ArtifactLifetime, ArtifactLifetimeError, BenchContract, BenchContractError, BenchPhase,
    BreakEven,
};
pub use capability::{Authority, Capability, CapabilityGate, Effect, Feasibility, GateFailure};
pub use change::{
    ChangeError, ChangeStructure, IncrementalOp, ModularU64, Replace, ReplaceDelta, SumFold,
    VectorU64,
};
pub use delta::{
    ChangeSupport, CostEstimate, DeltaClass, ExecutionStrategy, IncrementalizabilityAnalyzer,
    ObservationFrontier, OperatorKind, PointUpdate, SumState, coalesce_adjacent_at_frontier,
    coalesce_adjacent_last_writes,
};
pub use delta_forge::{
    AlgebraicClass, AverageCache, DeltaCertificate, DeltaForge, DerivedArtifact,
    DerivedAveragePlan, DerivedSumPlan, ExactAverage, FoldSpec, ForgeError, MaintenanceState,
    UpdateRule,
};
pub use morphology::{Morphology, MorphologyError, Region, RemorphPolicy, SemanticContract};
pub use refinement::{
    CostPrices, DecisionCertificate, DecisionError, Interval, PhysicalCost, Refinement,
    RefinementSet, select_refinement,
};
pub use runtime::{CheckedExecution, ExecutionMode, OptimizationFailure, run_checked};
pub use solver::{ArtifactOutcome, SolveError, SolveReceipt, solve_task};
pub use strategy::{
    CostInterval, MeasurementContext, MetaJit, StrategyEvidence, StrategyKey, StrategyMetric,
    StrategyStatus, UpdateLayout, WorkloadSignature,
};
pub use structure::{AbstractionContract, ExecutionSlice, LiftCertificate};
pub use task::{AxonTask, TaskError};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn interval(lower: i64, upper: i64) -> Interval {
        Interval::try_new(lower, upper).expect("test interval is valid")
    }

    fn decision<const N: usize>(utilities: [(&str, Interval); N]) -> DecisionCertificate {
        DecisionCertificate::try_from_utilities(utilities).expect("test action names are unique")
    }

    fn context() -> MeasurementContext {
        MeasurementContext::new(
            "i7-13650HX",
            UpdateLayout::CanonicalShardOrdered,
            1,
            StrategyMetric::Latency,
            1,
        )
        .unwrap()
    }

    #[test]
    fn higher_budget_contracts_result_set() {
        let coarse = RefinementSet::new(interval(10, 90), 1);
        let refined = coarse.refine(interval(40, 60), 2).unwrap();

        assert!(refined.is_subset_of(&coarse));
        assert_eq!(refined.budget(), 3);
    }

    #[test]
    fn analyzer_classifies_exact_delta_and_global_fallback() {
        let sum = IncrementalizabilityAnalyzer::analyze(OperatorKind::Sum);
        let sort = IncrementalizabilityAnalyzer::analyze(OperatorKind::Sort);

        assert_eq!(sum.class(), DeltaClass::Constant);
        assert!(sum.exact());
        assert!(sum.supports_coalescing());
        assert_eq!(sort.class(), DeltaClass::Global);
        assert!(!sort.exact());
        assert_eq!(
            sort.select(
                ChangeSupport::new(1, 1).unwrap(),
                CostEstimate::new(u64::MAX, 0, 0, 0)
            ),
            ExecutionStrategy::Full
        );
    }

    #[test]
    fn selector_uses_delta_only_below_measured_crossover() {
        let contract = IncrementalizabilityAnalyzer::analyze(OperatorKind::Sum);
        let costs = CostEstimate::new(100, 10, 1, 0);

        assert_eq!(
            contract.select(ChangeSupport::new(1, 100).unwrap(), costs),
            ExecutionStrategy::Delta
        );
        assert_eq!(
            contract.select(ChangeSupport::new(90, 100).unwrap(), costs),
            ExecutionStrategy::Full
        );
        assert_eq!(contract.crossover_support(100, costs), Some(89));
    }

    #[test]
    fn sum_delta_and_coalesced_delta_match_full_recomputation() {
        let state = SumState::try_from_values(vec![4, 8, 15, 16]).unwrap();
        let updates = [
            PointUpdate::new(1, 12),
            PointUpdate::new(1, 20),
            PointUpdate::new(3, 23),
        ];

        let full = state.full_after(&updates).unwrap();
        let delta = state.apply_delta(&updates).unwrap();
        let (coalesced, applied) = state
            .apply_coalesced(&updates, ObservationFrontier::FinalStateOnly)
            .unwrap();

        assert_eq!(full, delta);
        assert_eq!(full, coalesced);
        assert_eq!(applied, 2);
    }

    #[test]
    fn adjacent_event_coalescing_keeps_only_last_write_in_each_run() {
        let events = [
            PointUpdate::new(1, 2),
            PointUpdate::new(1, 3),
            PointUpdate::new(2, 4),
            PointUpdate::new(1, 5),
        ];

        assert_eq!(
            coalesce_adjacent_last_writes(&events),
            vec![
                PointUpdate::new(1, 3),
                PointUpdate::new(2, 4),
                PointUpdate::new(1, 5),
            ]
        );
    }

    #[test]
    fn coalescing_refuses_visible_intermediate_events() {
        let updates = [PointUpdate::new(1, 2), PointUpdate::new(1, 3)];
        let state = SumState::try_from_values(vec![0, 0]).unwrap();

        assert!(
            coalesce_adjacent_at_frontier(&updates, ObservationFrontier::FinalStateOnly).is_ok()
        );
        assert!(
            coalesce_adjacent_at_frontier(&updates, ObservationFrontier::IntermediateObserved)
                .is_err()
        );
        assert!(
            state
                .apply_coalesced(&updates, ObservationFrontier::IntermediateObserved)
                .is_err()
        );
    }

    #[test]
    fn delta_rejects_invalid_change_support_and_index() {
        assert!(ChangeSupport::new(5, 4).is_err());
        let state = SumState::try_from_values(vec![1]).unwrap();

        assert!(state.apply_delta(&[PointUpdate::new(1, 2)]).is_err());
    }

    #[test]
    fn certifies_only_a_decision_with_separated_bounds() {
        let decision = decision([("act", interval(81, 92)), ("wait", interval(43, 70))]);

        assert_eq!(decision.certified_action(), Some("act"));
    }

    #[test]
    fn refuses_an_ambiguous_decision_until_refined() {
        let decision = decision([("act", interval(50, 90)), ("wait", interval(60, 80))]);

        assert_eq!(decision.certified_action(), None);
        assert_eq!(decision.ambiguity(), 30);
    }

    #[test]
    fn chooses_refinement_by_ambiguity_reduction_per_cost() {
        let choices = [
            Refinement::new("read sensor", 20, PhysicalCost::new(5, 4, 1)),
            Refinement::new("exact proof", 30, PhysicalCost::new(20, 2, 1)),
        ];

        assert_eq!(
            select_refinement(&choices, CostPrices::unit())
                .unwrap()
                .name(),
            "read sensor"
        );
    }

    #[test]
    fn decision_certificate_rejects_duplicate_actions() {
        assert_eq!(
            DecisionCertificate::try_from_utilities([
                ("act", interval(81, 92)),
                ("act", interval(0, 100)),
            ]),
            Err(DecisionError::DuplicateAction("act".to_owned()))
        );
    }

    #[test]
    fn gate_rejects_missing_authority_before_realization() {
        let gate = CapabilityGate::new(Feasibility::ready(), Authority::from([Effect::ReadData]));
        let capability = Capability::new("send report", [Effect::Network]);

        assert_eq!(
            gate.evaluate(&capability),
            Err(GateFailure::Unauthorized(Effect::Network))
        );
    }

    #[test]
    fn gate_rejects_impossible_goal_even_with_authority() {
        let gate = CapabilityGate::new(
            Feasibility::new(true, false, true, true, true),
            Authority::from([Effect::Actuator]),
        );
        let capability = Capability::new("move arm", [Effect::Actuator]);

        assert_eq!(
            gate.evaluate(&capability),
            Err(GateFailure::NotIdentifiable)
        );
    }

    #[test]
    fn morphology_protects_base_and_allocates_highest_marginal_value() {
        let plan = Morphology::allocate(
            100,
            [
                Region::new("trusted", 40, [(0, 0)]),
                Region::new("memory", 0, [(20, 50), (40, 70)]),
                Region::new("programs", 0, [(20, 60), (40, 65)]),
            ],
        )
        .expect("budget covers protected base");

        assert_eq!(plan.bytes_for("trusted"), Some(40));
        assert_eq!(plan.bytes_for("memory"), Some(40));
        assert_eq!(plan.bytes_for("programs"), Some(20));
    }

    #[test]
    fn remorph_requires_amortization_and_semantic_contract() {
        let contract = SemanticContract::new("memory", 7, 0xA11CE);
        let current = Morphology::allocate(100, [Region::new("memory", 20, [(0, 0)])])
            .unwrap()
            .with_contract(contract.clone())
            .unwrap();
        let candidate = Morphology::allocate(100, [Region::new("memory", 20, [(40, 100)])])
            .unwrap()
            .with_contract(contract)
            .unwrap();

        assert!(!RemorphPolicy::new(10, 5, 2).accepts(&current, &candidate, 20, 1, 2));
        assert!(RemorphPolicy::new(30, 5, 2).accepts(&current, &candidate, 20, 1, 2));
    }

    #[test]
    fn remorph_rejects_migrated_regions_without_matching_contracts() {
        let current = Morphology::allocate(100, [Region::new("memory", 20, [(0, 0)])]).unwrap();
        let candidate =
            Morphology::allocate(100, [Region::new("memory", 20, [(40, 100)])]).unwrap();

        assert!(!RemorphPolicy::new(30, 5, 2).accepts(&current, &candidate, 20, 1, 2));
    }

    #[test]
    fn remorph_rejects_contract_change_even_when_that_region_keeps_its_bytes() {
        let memory = SemanticContract::new("memory", 1, 10);
        let programs_v1 = SemanticContract::new("programs", 1, 11);
        let current = Morphology::allocate(
            100,
            [
                Region::new("memory", 20, [(0, 0)]),
                Region::new("programs", 20, [(0, 0)]),
            ],
        )
        .unwrap()
        .with_contract(memory.clone())
        .unwrap()
        .with_contract(programs_v1)
        .unwrap();
        let candidate = Morphology::allocate(
            100,
            [
                Region::new("memory", 20, [(40, 100)]),
                Region::new("programs", 20, [(0, 0)]),
            ],
        )
        .unwrap()
        .with_contract(memory)
        .unwrap()
        .with_contract(SemanticContract::new("programs", 2, 11))
        .unwrap();

        assert!(!RemorphPolicy::new(30, 5, 2).accepts(&current, &candidate, 20, 1, 2));
    }

    #[test]
    fn morphology_rejects_overflowing_tier_size() {
        assert_eq!(
            Morphology::allocate(u64::MAX, [Region::new("memory", u64::MAX, [(1, 1)])]),
            Err(MorphologyError::SizeOverflow("memory".to_owned()))
        );
    }

    #[test]
    fn demand_delta_excludes_irrelevant_changes() {
        let dependencies = BTreeMap::from([
            ("goal".to_owned(), vec!["sum".to_owned()]),
            (
                "sum".to_owned(),
                vec!["left".to_owned(), "right".to_owned()],
            ),
            ("left".to_owned(), vec!["sensor".to_owned()]),
            ("right".to_owned(), vec!["history".to_owned()]),
            ("noise".to_owned(), Vec::new()),
        ]);

        let slice = ExecutionSlice::build("goal", ["sensor", "noise"], &dependencies);

        assert_eq!(
            slice.nodes(),
            &BTreeSet::from([
                "sensor".to_owned(),
                "left".to_owned(),
                "sum".to_owned(),
                "goal".to_owned(),
            ])
        );
    }

    #[test]
    fn lift_certificate_matches_exact_fallback() {
        let values = [41, 41, 41, 41];
        let certificate = LiftCertificate::from_identical(&values).unwrap();

        assert_eq!(certificate.members(), 4);
        assert!(certificate.matches_exact_max(&values));
        assert!(LiftCertificate::from_identical(&[1, 2]).is_none());
    }

    #[test]
    fn abstraction_contract_rejects_excess_error() {
        let contract = AbstractionContract::new(2);

        assert!(contract.preserves(10, 12));
        assert!(!contract.preserves(10, 13));
    }

    #[test]
    fn optimized_failure_and_mismatch_use_exact_fallback() {
        let failed = run_checked::<i64>(Err(OptimizationFailure::Unavailable), || 42);
        let mismatched = run_checked(Ok(41_i64), || 42);

        assert_eq!(failed.mode(), ExecutionMode::ExactFallback);
        assert_eq!(failed.value(), &42);
        assert_eq!(mismatched.mode(), ExecutionMode::ExactFallback);
        assert_eq!(mismatched.value(), &42);
    }

    #[test]
    fn oracle_dominance_refutes_hybrid_only_inside_its_measured_domain() {
        let workload = WorkloadSignature::new(
            OperatorKind::Sum,
            8_388_608,
            64,
            1_115_136,
            1_065_984,
            ObservationFrontier::FinalStateOnly,
            context(),
        )
        .unwrap();
        let evidence = StrategyEvidence::from_oracle(
            workload.clone(),
            StrategyKey::RawDelta,
            CostInterval::new(1_490, 1_500).unwrap(),
            StrategyKey::HybridShard,
            CostInterval::new(1_590, 1_600).unwrap(),
        );

        assert_eq!(evidence.status(), StrategyStatus::LatencyDominated);
        assert!(evidence.oracle_headroom_basis_points() < 0);

        let jit = MetaJit::from_evidence(&evidence).unwrap();
        assert_eq!(jit.select(&workload), Some(StrategyKey::RawDelta));
        assert_eq!(
            jit.select(
                &WorkloadSignature::new(
                    OperatorKind::Sum,
                    8_388_608,
                    64,
                    1_115_136,
                    1_065_984,
                    ObservationFrontier::IntermediateObserved,
                    context(),
                )
                .unwrap()
            ),
            None
        );
    }

    #[test]
    fn overlapping_cost_intervals_do_not_create_a_meta_jit_reflex() {
        let workload = WorkloadSignature::new(
            OperatorKind::Sum,
            16,
            1,
            4,
            4,
            ObservationFrontier::FinalStateOnly,
            context(),
        )
        .unwrap();
        let evidence = StrategyEvidence::from_oracle(
            workload,
            StrategyKey::RawDelta,
            CostInterval::new(10, 20).unwrap(),
            StrategyKey::HybridShard,
            CostInterval::new(15, 25).unwrap(),
        );

        assert_eq!(evidence.status(), StrategyStatus::Inconclusive);
        assert!(MetaJit::from_evidence(&evidence).is_none());
    }

    #[test]
    fn meta_jit_reopens_when_measurement_context_changes() {
        let context = context();
        let workload = WorkloadSignature::new(
            OperatorKind::Sum,
            16,
            1,
            4,
            4,
            ObservationFrontier::FinalStateOnly,
            context,
        )
        .unwrap();
        let evidence = StrategyEvidence::from_paired_samples(
            workload.clone(),
            StrategyKey::RawDelta,
            &[10, 11, 12],
            StrategyKey::HybridShard,
            &[20, 21, 22],
        )
        .unwrap();
        let jit = MetaJit::from_evidence(&evidence).unwrap();
        let changed_hardware = WorkloadSignature::new(
            OperatorKind::Sum,
            16,
            1,
            4,
            4,
            ObservationFrontier::FinalStateOnly,
            MeasurementContext::new(
                "other-cpu",
                UpdateLayout::CanonicalShardOrdered,
                1,
                StrategyMetric::Latency,
                1,
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(jit.select(&changed_hardware), None);
        let changed_layout = WorkloadSignature::new(
            OperatorKind::Sum,
            16,
            1,
            4,
            4,
            ObservationFrontier::FinalStateOnly,
            MeasurementContext::new(
                "i7-13650HX",
                UpdateLayout::Arbitrary,
                1,
                StrategyMetric::Latency,
                1,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(jit.select(&changed_layout), None);
        assert!(CostInterval::from_samples(&[]).is_err());
        assert!(
            StrategyEvidence::from_paired_samples(
                workload,
                StrategyKey::RawDelta,
                &[10],
                StrategyKey::HybridShard,
                &[20, 21],
            )
            .is_err()
        );
    }
}
