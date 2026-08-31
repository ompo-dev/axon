//! AXON-Λ: kernel matemático independente do backend para contratos, deltas e quotients.

mod autolift;
mod contract;
mod cost;
mod fabric;
mod general;
mod quotient;

pub use autolift::{
    AutoLiftError, CertifiedAutoLift, LiftCertificate, LiftedClass as AutoLiftedClass, LocalUnlift,
};

pub use contract::{
    ContractedMorphism, DecisionCertificate, MorphismImplementation, RealizationError,
    RealizationPlan, SemanticAbi, VerificationStrength,
};
pub use cost::{CostError, CostVector, CostWeights, ParetoFrontier};
pub use fabric::{
    AdaptiveMode, ChainFabric, CognitiveSlice, Demand, EvidenceDelta, FabricError, QueryResult,
};
pub use general::{
    DependencyFingerprint, FixpointCertificate, GeneralFactor, GeneralGraph, GeneralRule,
    GraphDelta, GraphError, GraphEvaluation, GraphQueryResult, GraphSlice, StructuralMode,
    VersionedDependency,
};
pub use quotient::{LiftedClass, LiftedPopulation, QuotientError};

/// Journal canônico pequeno usado na conformance entre realizações AXON-Λ.
pub fn canonical_conformance_journal() -> String {
    use std::collections::BTreeSet;

    let claims = |values: &[&str]| -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    };
    let required = ContractedMorphism::new(
        SemanticAbi::new("affine/chain", "u64", "u64", 0xA11F_1A00),
        1,
        claims(&["u64 modular arithmetic"]),
        claims(&["exact affine result"]),
        0,
        VerificationStrength::Exhaustive,
    );
    let candidate = ContractedMorphism::new(
        SemanticAbi::new("affine/chain", "u64", "u64", 0xA11F_1A00),
        1,
        BTreeSet::new(),
        claims(&["exact affine result", "stable journal"]),
        0,
        VerificationStrength::Exhaustive,
    );
    let local = ChainFabric::new(32, 8).expect("fixed conformance fabric");
    let local_result = local
        .query(
            Demand::exact(15),
            EvidenceDelta::new(12, 777),
            CostWeights::latency_only(),
        )
        .expect("fixed local query");
    let global = ChainFabric::new(32, 32).expect("fixed conformance fabric");
    let global_result = global
        .query(
            Demand::exact(31),
            EvidenceDelta::new(0, 777),
            CostWeights::latency_only(),
        )
        .expect("fixed global query");
    let frontier = ParetoFrontier::new(vec![
        CostVector::new(2, 2, 20, 2, 0),
        CostVector::new(20, 20, 2, 20, 0),
        CostVector::new(30, 30, 30, 30, 1),
    ])
    .expect("fixed conformance costs");
    let lifted = LiftedPopulation::from_values(&[7, 7, 3, 7, 3, 11]);
    format!(
        "AXON-LAMBDA/1\nrefinement={}\nlocal=value:{};mode:{:?};B:{};F:{};A:{}\nglobal=value:{};mode:{:?};A:{}\npareto=options:{};latency:{};memory:{}\nlift=sum:{};classes:{};unlift:{}\n",
        candidate.refines(&required),
        local_result.value,
        local_result.mode,
        local_result.slice.demanded_factors,
        local_result.slice.changed_factors,
        local_result.slice.active_factors,
        global_result.value,
        global_result.mode,
        global_result.slice.active_factors,
        frontier.options().len(),
        frontier.select(CostWeights::latency_only()).latency_units,
        frontier.select(CostWeights::memory_only()).memory_bytes,
        lifted.lifted_sum(),
        lifted.classes().len(),
        lifted.unlift_value(1, 99).expect("fixed member"),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::process::Command;

    use super::*;

    fn claims(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn contract(
        preconditions: &[&str],
        guarantees: &[&str],
        error: u64,
        verification: VerificationStrength,
    ) -> ContractedMorphism {
        ContractedMorphism::new(
            SemanticAbi::new("arith/add", "opaque", "opaque", 0xADD0_0001),
            1,
            claims(preconditions),
            claims(guarantees),
            error,
            verification,
        )
    }

    #[test]
    fn refinement_preserves_semantics_without_strengthening_preconditions() {
        let required = contract(
            &["integer inputs", "no overflow"],
            &["sum"],
            0,
            VerificationStrength::Exhaustive,
        );
        let compiled = contract(
            &["integer inputs"],
            &["sum", "checksum"],
            0,
            VerificationStrength::Exhaustive,
        );
        let stricter = contract(
            &["integer inputs", "no overflow", "aligned memory"],
            &["sum"],
            0,
            VerificationStrength::Exhaustive,
        );
        let changed_rule = compiled.clone().with_abi(SemanticAbi::new(
            "arith/add",
            "opaque",
            "opaque",
            0xDEAD_BEEF,
        ));

        assert!(compiled.refines(&required));
        assert!(!stricter.refines(&required));
        assert!(!changed_rule.refines(&required));
    }

    #[test]
    fn realization_rejects_a_cheap_implementation_that_breaks_the_certificate() {
        let required = contract(
            &["integer inputs"],
            &["sum"],
            450,
            VerificationStrength::Sampled,
        );
        let factor = MorphismImplementation::new(
            "ADD",
            required.clone(),
            DecisionCertificate::new(900, 100),
            vec![
                (
                    "approx",
                    contract(
                        &["integer inputs"],
                        &["sum"],
                        450,
                        VerificationStrength::Sampled,
                    ),
                    CostVector::new(1, 1, 1, 1, 1),
                ),
                (
                    "compiled",
                    contract(
                        &["integer inputs"],
                        &["sum"],
                        0,
                        VerificationStrength::Exhaustive,
                    ),
                    CostVector::new(4, 4, 4, 4, 0),
                ),
                (
                    "exact",
                    contract(
                        &["integer inputs"],
                        &["sum"],
                        0,
                        VerificationStrength::Exhaustive,
                    ),
                    CostVector::new(9, 9, 9, 9, 0),
                ),
            ],
        );

        let plan = factor.realize(CostWeights::latency_only()).unwrap();
        assert_eq!(plan.name, "compiled");
        assert!(plan.certificate_preserved);
    }

    #[test]
    fn pareto_frontier_keeps_incomparable_costs_and_removes_dominated_costs() {
        let frontier = ParetoFrontier::new(vec![
            CostVector::new(2, 2, 20, 2, 0),
            CostVector::new(20, 20, 2, 20, 0),
            CostVector::new(30, 30, 30, 30, 1),
        ])
        .unwrap();

        assert_eq!(frontier.options().len(), 2);
        assert_eq!(
            frontier.select(CostWeights::latency_only()),
            CostVector::new(20, 20, 2, 20, 0)
        );
        assert_eq!(
            frontier.select(CostWeights::memory_only()),
            CostVector::new(2, 2, 20, 2, 0)
        );
    }

    #[test]
    fn sequential_cost_composition_is_checked_and_preserves_each_dimension() {
        assert_eq!(
            CostVector::new(1, 2, 3, 4, 5).checked_add(CostVector::new(10, 20, 30, 40, 50)),
            Some(CostVector::new(11, 22, 33, 44, 55)),
        );
        assert_eq!(
            CostVector::new(u64::MAX, 0, 0, 0, 0).checked_add(CostVector::new(1, 0, 0, 0, 0)),
            None,
        );
    }

    #[test]
    fn empty_pareto_frontier_is_a_typed_error_not_a_panic() {
        assert_eq!(
            ParetoFrontier::new(Vec::new()),
            Err(CostError::EmptyFrontier)
        );
    }

    #[test]
    fn demand_delta_matches_full_recompute_for_the_local_light_cone() {
        let fabric = ChainFabric::new(128, 8).unwrap();
        let demand = Demand::exact(15);
        let change = EvidenceDelta::new(8, 777);

        let full = fabric.full_query(demand, change).unwrap();
        let adaptive = fabric
            .query(demand, change, CostWeights::latency_only())
            .unwrap();

        assert_eq!(full.value, adaptive.value);
        assert_eq!(adaptive.mode, AdaptiveMode::DeltaPropagation);
        assert_eq!(adaptive.slice.demanded_factors, 8);
        assert_eq!(adaptive.slice.changed_factors, 8);
        assert_eq!(adaptive.slice.active_factors, 8);
    }

    #[test]
    fn irrelevant_evidence_reuses_the_certified_value_without_waking_the_slice() {
        let fabric = ChainFabric::new(128, 8).unwrap();
        let outcome = fabric
            .query(
                Demand::exact(15),
                EvidenceDelta::new(24, 777),
                CostWeights::latency_only(),
            )
            .unwrap();

        assert_eq!(outcome.mode, AdaptiveMode::Reuse);
        assert_eq!(outcome.slice.active_factors, 0);
        assert_eq!(outcome.value, fabric.base_value(15).unwrap());
    }

    #[test]
    fn adaptive_runtime_falls_back_to_full_recompute_for_a_global_cascade() {
        let fabric = ChainFabric::new(128, 128).unwrap();
        let demand = Demand::exact(127);
        let change = EvidenceDelta::new(0, 777);

        let full = fabric.full_query(demand, change).unwrap();
        let adaptive = fabric
            .query(demand, change, CostWeights::latency_only())
            .unwrap();

        assert_eq!(full.value, adaptive.value);
        assert_eq!(adaptive.mode, AdaptiveMode::FullRecompute);
        assert_eq!(adaptive.slice.active_factors, 128);
    }

    #[test]
    fn sparse_delta_overlay_matches_every_value_of_a_full_recomputation() {
        let fabric = ChainFabric::new(128, 8).unwrap();

        assert!(
            fabric
                .delta_overlay_matches_full(EvidenceDelta::new(12, 777))
                .unwrap()
        );
    }

    #[test]
    fn lift_preserves_an_exchangeable_aggregate_and_unlift_is_local() {
        let population = LiftedPopulation::from_values(&[7, 7, 3, 7, 3, 11]);

        assert_eq!(population.exact_sum(), 38);
        assert_eq!(population.lifted_sum(), 38);
        assert_eq!(population.classes().len(), 3);
        assert_eq!(population.unlift_value(1, 99).unwrap(), 130);
    }

    #[test]
    fn python_and_rust_implementations_emit_the_same_canonical_journal() {
        let script = format!(
            "{}/tools/axon_lambda_conformance.py",
            env!("CARGO_MANIFEST_DIR").replace('\\', "/")
        );
        let output = Command::new("python").arg(script).output().unwrap();

        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout)
                .unwrap()
                .replace("\r\n", "\n"),
            canonical_conformance_journal()
        );
    }

    #[test]
    fn lambda_squared_delta_matches_full_for_a_dag_and_carries_dependencies() {
        let graph = general::GeneralGraph::new(vec![
            general::GeneralFactor::source(2),
            general::GeneralFactor::affine(0, 3, 1),
            general::GeneralFactor::affine(1, 5, 2),
            general::GeneralFactor::max(vec![1, 2], -10),
            general::GeneralFactor::source(88),
        ])
        .unwrap();

        let full = graph
            .full_query(3, general::GraphDelta::replace_source(0, 7))
            .unwrap();
        let incremental = graph
            .query(3, general::GraphDelta::replace_source(0, 7))
            .unwrap();

        assert_eq!(incremental.mode, general::StructuralMode::DeltaPropagation);
        assert_eq!(incremental.value, full.value);
        assert!(
            graph
                .delta_overlay_matches_full(general::GraphDelta::replace_source(0, 7))
                .unwrap()
        );
        assert!(
            incremental.dependency.validates(
                graph.graph_digest(),
                &graph
                    .revisions_after(general::GraphDelta::replace_source(0, 7))
                    .unwrap()
            )
        );
    }

    #[test]
    fn lambda_squared_classifies_monotone_and_contractive_sccs_and_falls_back_safely() {
        let monotone = general::GeneralGraph::new(vec![
            general::GeneralFactor::source(5),
            general::GeneralFactor::max(vec![0, 2], 0),
            general::GeneralFactor::max(vec![1], 0),
        ])
        .unwrap();
        let monotone_result = monotone
            .query(2, general::GraphDelta::replace_source(0, 9))
            .unwrap();
        assert_eq!(
            monotone_result.mode,
            general::StructuralMode::MonotoneFixpoint
        );
        assert_eq!(monotone_result.value, 9);
        assert!(
            monotone_result
                .fixpoints
                .iter()
                .any(|certificate| certificate.residual_max == 0)
        );

        let contractive = general::GeneralGraph::new(vec![
            general::GeneralFactor::source(0),
            general::GeneralFactor::contractive_half(vec![0, 2], 64),
            general::GeneralFactor::contractive_half(vec![1], 64),
        ])
        .unwrap();
        let contractive_result = contractive
            .query(2, general::GraphDelta::replace_source(0, 32))
            .unwrap();
        assert_eq!(
            contractive_result.mode,
            general::StructuralMode::ContractiveFixpoint
        );
        assert_eq!(contractive_result.value, 63);
        assert!(contractive_result.fixpoints.iter().any(|certificate| {
            certificate.lipschitz_numerator == Some(1)
                && certificate.lipschitz_denominator == Some(2)
                && certificate.residual_max == 0
        }));

        let negative_contractive = general::GeneralGraph::new(vec![
            general::GeneralFactor::source(-5),
            general::GeneralFactor::contractive_half(vec![0, 2], -4),
            general::GeneralFactor::contractive_half(vec![1], -4),
        ])
        .unwrap();
        let negative_result = negative_contractive
            .query(2, general::GraphDelta::replace_source(0, -5))
            .unwrap();
        assert_eq!(
            negative_result.mode,
            general::StructuralMode::ContractiveFixpoint
        );
        assert_eq!(negative_result.value, -4);
        assert!(
            negative_result
                .fixpoints
                .iter()
                .any(|certificate| certificate.residual_max == 0)
        );

        let opaque_cycle = general::GeneralGraph::new(vec![
            general::GeneralFactor::opaque_constant(vec![1], 17),
            general::GeneralFactor::opaque_constant(vec![0], 23),
        ])
        .unwrap();
        let fallback = opaque_cycle.evaluate().unwrap();
        assert_eq!(fallback.mode, general::StructuralMode::FullFallback);
        assert_eq!(fallback.values, vec![17, 23]);
    }

    #[test]
    fn lambda_squared_auto_lift_requires_a_certificate_and_unlifts_only_the_changed_member() {
        let graph = general::GeneralGraph::new(vec![
            general::GeneralFactor::source(7),
            general::GeneralFactor::source(7),
            general::GeneralFactor::source(7),
            general::GeneralFactor::source(7),
            general::GeneralFactor::source(7),
            general::GeneralFactor::source(7),
            general::GeneralFactor::source(2),
            general::GeneralFactor::max(vec![0, 1, 2, 3, 4, 5, 6], -10),
        ])
        .unwrap();
        let lift = autolift::CertifiedAutoLift::discover(&graph).unwrap();

        assert_eq!(lift.classes().len(), 1);
        assert_eq!(lift.classes()[0].members.len(), 6);
        assert!(lift.verify(&graph));
        assert_eq!(
            lift.lifted_max(&graph, 7).unwrap(),
            graph.base_value(7).unwrap()
        );

        let unlift = lift.unlift(2, 99).unwrap();
        let full = graph
            .full_query(7, general::GraphDelta::replace_source(2, 99))
            .unwrap();
        assert_eq!(unlift.lifted_max(&lift, &graph, 7).unwrap(), full.value);
        assert_eq!(unlift.specialized_members(), 1);
        assert_eq!(unlift.remaining_members(), 5);
        assert_eq!(unlift.member(), 2);

        let non_commutative = general::GeneralGraph::new(vec![
            general::GeneralFactor::source(7),
            general::GeneralFactor::source(7),
            general::GeneralFactor::affine(0, 1, 0),
            general::GeneralFactor::affine(1, 1, 0),
        ])
        .unwrap();
        assert!(
            autolift::CertifiedAutoLift::discover(&non_commutative)
                .unwrap()
                .classes()
                .is_empty()
        );
    }
}
