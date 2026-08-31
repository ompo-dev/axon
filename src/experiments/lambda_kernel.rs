//! Experimento determinístico do recorte AXON-Λ: refinamento e light cone.

use std::collections::BTreeSet;

use crate::core_lambda::{
    AdaptiveMode, ChainFabric, ContractedMorphism, CostVector, CostWeights, DecisionCertificate,
    Demand, EvidenceDelta, LiftedPopulation, MorphismImplementation, SemanticAbi,
    VerificationStrength,
};

#[derive(Clone, Debug, PartialEq)]
pub struct LambdaKernelReport {
    pub contract_refinement_preserved: bool,
    pub unsafe_approximate_rejected: bool,
    pub local_demanded_factors: usize,
    pub local_changed_factors: usize,
    pub local_active_factors: usize,
    pub local_matches_full: bool,
    pub local_delta_selected: bool,
    pub global_active_factors: usize,
    pub global_matches_full: bool,
    pub global_full_selected: bool,
    pub overlay_matches_full_state: bool,
    pub lifted_sum_matches_individual_sum: bool,
    pub lifted_classes: usize,
    /// O lab só confirma que o journal Rust está disponível; a igualdade com
    /// Python é verificada pelo teste de conformance do core.
    pub canonical_journal_available: bool,
}

pub(super) fn run() -> LambdaKernelReport {
    let required = contract(450, VerificationStrength::Sampled);
    let compiled = contract(0, VerificationStrength::Exhaustive);
    let factor = MorphismImplementation::new(
        "AFFINE",
        required.clone(),
        DecisionCertificate::new(900, 100),
        vec![
            ("approx", required, CostVector::new(1, 1, 1, 1, 0)),
            ("compiled", compiled.clone(), CostVector::new(4, 4, 4, 4, 0)),
            ("exact", compiled.clone(), CostVector::new(9, 9, 9, 9, 0)),
        ],
    );
    let plan = factor.realize(CostWeights::latency_only()).unwrap();

    let local = ChainFabric::new(1_000, 1_000).unwrap();
    let local_demand = Demand::exact(999);
    let local_change = EvidenceDelta::new(500, 777);
    let local_full = local.full_query(local_demand, local_change).unwrap();
    let local_delta = local
        .query(local_demand, local_change, CostWeights::latency_only())
        .unwrap();

    let global = ChainFabric::new(1_000, 1_000).unwrap();
    let global_demand = Demand::exact(999);
    let global_change = EvidenceDelta::new(0, 777);
    let global_full = global.full_query(global_demand, global_change).unwrap();
    let global_adaptive = global
        .query(global_demand, global_change, CostWeights::latency_only())
        .unwrap();
    let population = LiftedPopulation::from_values(&[7, 7, 3, 7, 3, 11]);

    LambdaKernelReport {
        contract_refinement_preserved: compiled
            .refines(&contract(450, VerificationStrength::Sampled)),
        unsafe_approximate_rejected: plan.name == "compiled",
        local_demanded_factors: local_delta.slice.demanded_factors,
        local_changed_factors: local_delta.slice.changed_factors,
        local_active_factors: local_delta.slice.active_factors,
        local_matches_full: local_delta.value == local_full.value,
        local_delta_selected: local_delta.mode == AdaptiveMode::DeltaPropagation,
        global_active_factors: global_adaptive.slice.active_factors,
        global_matches_full: global_adaptive.value == global_full.value,
        global_full_selected: global_adaptive.mode == AdaptiveMode::FullRecompute,
        overlay_matches_full_state: local.delta_overlay_matches_full(local_change).unwrap()
            && global.delta_overlay_matches_full(global_change).unwrap(),
        lifted_sum_matches_individual_sum: population.exact_sum() == population.lifted_sum(),
        lifted_classes: population.classes().len(),
        canonical_journal_available: crate::core_lambda::canonical_conformance_journal()
            .starts_with("AXON-LAMBDA/1\n"),
    }
}

impl LambdaKernelReport {
    pub fn to_markdown(&self) -> String {
        format!(
            r#"# AXON-Λ — Kernel determinístico

| Invariante | Resultado |
|---|---:|
| Refinamento de contrato preservado | {} |
| Aproximado barato recusado pelo certificado | {} |
| Cone local B / F / A | {} / {} / {} |
| Local: paridade / delta selecionado | {} / {} |
| Cascata global A / paridade / full selecionado | {} / {} / {} |
| Estado completo = base + overlay | {} |
| LIFT exato / classes | {} / {} |
| Journal AXON-Λ disponível para conformance | {} |

O domínio é uma cadeia de Factors afins com aritmética modular. O relatório é
determinístico; custos são declarados e não constituem benchmark físico.
"#,
            self.contract_refinement_preserved,
            self.unsafe_approximate_rejected,
            self.local_demanded_factors,
            self.local_changed_factors,
            self.local_active_factors,
            self.local_matches_full,
            self.local_delta_selected,
            self.global_active_factors,
            self.global_matches_full,
            self.global_full_selected,
            self.overlay_matches_full_state,
            self.lifted_sum_matches_individual_sum,
            self.lifted_classes,
            self.canonical_journal_available,
        )
    }
}

fn contract(error: u64, verification: VerificationStrength) -> ContractedMorphism {
    ContractedMorphism::new(
        SemanticAbi::new("affine/chain", "u64", "u64", 0xA11F_1A00),
        1,
        ["u64 modular arithmetic"]
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>(),
        ["exact affine result"]
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>(),
        error,
        verification,
    )
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn lambda_report_preserves_contracts_and_rejects_global_incrementalism() {
        let report = run();
        assert!(report.contract_refinement_preserved);
        assert!(report.unsafe_approximate_rejected);
        assert_eq!(report.local_demanded_factors, 1_000);
        assert_eq!(report.local_changed_factors, 500);
        assert_eq!(report.local_active_factors, 500);
        assert!(report.local_matches_full);
        assert!(report.local_delta_selected);
        assert_eq!(report.global_active_factors, 1_000);
        assert!(report.global_matches_full);
        assert!(report.global_full_selected);
        assert!(report.overlay_matches_full_state);
        assert!(report.lifted_sum_matches_individual_sum);
    }
}
