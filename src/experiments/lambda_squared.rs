//! Relatório determinístico do corte AXON-Λ².

use crate::core_lambda::{
    CertifiedAutoLift, GeneralFactor, GeneralGraph, GraphDelta, StructuralMode,
};

#[derive(Clone, Debug, PartialEq)]
pub struct LambdaSquaredReport {
    pub dag_delta_matches_full: bool,
    pub dag_dependency_certificate_valid: bool,
    pub monotone_scc_certified: bool,
    pub contractive_scc_certified: bool,
    pub opaque_cycle_falls_back_to_full: bool,
    pub auto_lift_classes: usize,
    pub auto_lift_verified: bool,
    pub unlift_matches_full: bool,
    pub unlift_remaining_members: usize,
}

pub(super) fn run() -> LambdaSquaredReport {
    let dag = GeneralGraph::new(vec![
        GeneralFactor::source(2),
        GeneralFactor::affine(0, 3, 1),
        GeneralFactor::affine(1, 5, 2),
        GeneralFactor::max(vec![1, 2], -10),
        GeneralFactor::source(88),
    ])
    .expect("fixed DAG");
    let delta = GraphDelta::replace_source(0, 7);
    let full = dag.full_query(3, delta).expect("fixed full DAG");
    let incremental = dag.query(3, delta).expect("fixed delta DAG");

    let monotone = GeneralGraph::new(vec![
        GeneralFactor::source(5),
        GeneralFactor::max(vec![0, 2], 0),
        GeneralFactor::max(vec![1], 0),
    ])
    .expect("fixed monotone SCC");
    let monotone_result = monotone
        .query(2, GraphDelta::replace_source(0, 9))
        .expect("fixed monotone query");

    let contractive = GeneralGraph::new(vec![
        GeneralFactor::source(0),
        GeneralFactor::contractive_half(vec![0, 2], 64),
        GeneralFactor::contractive_half(vec![1], 64),
    ])
    .expect("fixed contractive SCC");
    let contractive_result = contractive
        .query(2, GraphDelta::replace_source(0, 32))
        .expect("fixed contractive query");

    let opaque = GeneralGraph::new(vec![
        GeneralFactor::opaque_constant(vec![1], 17),
        GeneralFactor::opaque_constant(vec![0], 23),
    ])
    .expect("fixed opaque SCC");

    let lifted_graph = GeneralGraph::new(vec![
        GeneralFactor::source(7),
        GeneralFactor::source(7),
        GeneralFactor::source(7),
        GeneralFactor::source(7),
        GeneralFactor::source(7),
        GeneralFactor::source(7),
        GeneralFactor::source(2),
        GeneralFactor::max(vec![0, 1, 2, 3, 4, 5, 6], -10),
    ])
    .expect("fixed exchangeable graph");
    let lift = CertifiedAutoLift::discover(&lifted_graph).expect("fixed Auto-LIFT");
    let unlift = lift.unlift(2, 99).expect("fixed local UNLIFT");
    let full_unlift = lifted_graph
        .full_query(7, GraphDelta::replace_source(2, 99))
        .expect("fixed full UNLIFT");

    LambdaSquaredReport {
        dag_delta_matches_full: incremental.mode == StructuralMode::DeltaPropagation
            && incremental.value == full.value
            && dag
                .delta_overlay_matches_full(delta)
                .expect("fixed overlay"),
        dag_dependency_certificate_valid: incremental.dependency.validates(
            dag.graph_digest(),
            &dag.revisions_after(delta).expect("fixed revisions"),
        ),
        monotone_scc_certified: monotone_result.mode == StructuralMode::MonotoneFixpoint
            && monotone_result.value == 9
            && monotone_result
                .fixpoints
                .iter()
                .all(|certificate| certificate.residual_max == 0),
        contractive_scc_certified: contractive_result.mode == StructuralMode::ContractiveFixpoint
            && contractive_result.fixpoints.iter().any(|certificate| {
                certificate.lipschitz_numerator == Some(1)
                    && certificate.lipschitz_denominator == Some(2)
                    && certificate.residual_max == 0
            }),
        opaque_cycle_falls_back_to_full: opaque.evaluate().expect("fixed opaque eval").mode
            == StructuralMode::FullFallback,
        auto_lift_classes: lift.classes().len(),
        auto_lift_verified: lift.verify(&lifted_graph),
        unlift_matches_full: unlift
            .lifted_max(&lift, &lifted_graph, 7)
            .expect("fixed lifted max")
            == full_unlift.value,
        unlift_remaining_members: unlift.remaining_members(),
    }
}

impl LambdaSquaredReport {
    pub fn to_markdown(&self) -> String {
        format!(
            r#"# AXON-Λ² — General Graph Calculus

| Invariante | Resultado |
|---|---:|
| DAG: delta = full, incluindo overlay | {} |
| Fingerprint de dependências válido | {} |
| SCC monotônica chega a fixpoint certificado | {} |
| SCC contractiva chega a fixpoint com L=1/2 | {} |
| Ciclo opaco usa fallback full exato | {} |
| Classes Auto-LIFT certificadas | {} |
| Certificados Auto-LIFT verificados | {} |
| UNLIFT local = recomputação total | {} |
| Membros restantes na classe após UNLIFT | {} |

O escopo é restrito: `max` comutativo e Sources indistinguíveis; igualdade de
cor é somente candidata e não autoriza compressão sem certificado.
"#,
            self.dag_delta_matches_full,
            self.dag_dependency_certificate_valid,
            self.monotone_scc_certified,
            self.contractive_scc_certified,
            self.opaque_cycle_falls_back_to_full,
            self.auto_lift_classes,
            self.auto_lift_verified,
            self.unlift_matches_full,
            self.unlift_remaining_members,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn lambda_squared_report_is_a_safe_generalization_of_the_initial_kernel() {
        let report = run();
        assert!(report.dag_delta_matches_full);
        assert!(report.dag_dependency_certificate_valid);
        assert!(report.monotone_scc_certified);
        assert!(report.contractive_scc_certified);
        assert!(report.opaque_cycle_falls_back_to_full);
        assert_eq!(report.auto_lift_classes, 1);
        assert!(report.auto_lift_verified);
        assert!(report.unlift_matches_full);
        assert_eq!(report.unlift_remaining_members, 5);
    }
}
