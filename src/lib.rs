//! AXON-UIC: small, executable primitives for certified adaptive cognition.

pub mod capability;
pub mod morphology;
pub mod refinement;
pub mod runtime;
pub mod structure;

pub use capability::{Authority, Capability, CapabilityGate, Effect, Feasibility, GateFailure};
pub use morphology::{Morphology, MorphologyError, Region, RemorphPolicy, SemanticContract};
pub use refinement::{
    CostPrices, DecisionCertificate, DecisionError, Interval, PhysicalCost, Refinement,
    RefinementSet, select_refinement,
};
pub use runtime::{CheckedExecution, ExecutionMode, OptimizationFailure, run_checked};
pub use structure::{AbstractionContract, ExecutionSlice, LiftCertificate};

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

    #[test]
    fn higher_budget_contracts_result_set() {
        let coarse = RefinementSet::new(interval(10, 90), 1);
        let refined = coarse.refine(interval(40, 60), 2).unwrap();

        assert!(refined.is_subset_of(&coarse));
        assert_eq!(refined.budget(), 3);
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
}
