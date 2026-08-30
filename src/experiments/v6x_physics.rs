//! Laboratório V6-X para a camada Ψ-IR; todos os perfis são declarados.

use crate::core_v5::CostOrigin;
use crate::core_v6::{
    CleanupAction, CleanupOption, PhysicalBackend, PhysicalCompiler, PhysicalCost,
    PhysicalOperation, PhysicalOperationKind, PhysicalProfile, PhysicalStateKind,
    PrecisionRequirement, RealizationError,
};

/// Resultados determinísticos do contrato Physics-First, sem hardware conectado.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V6XPhysicsReport {
    pub exact_verification_uses_digital: bool,
    pub boundary_tax_rejects_false_pbit_win: bool,
    pub pbit_sampling_selected_when_colocated: bool,
    pub mixed_cost_provenance_rejected: bool,
    pub uncompute_selected_when_valid_and_cheaper: bool,
    pub erase_selected_when_uncompute_is_costlier: bool,
    pub all_selected_costs_are_declared: bool,
}

impl V6XPhysicsReport {
    pub fn to_markdown(&self) -> String {
        format!(
            "# Axon V6-X / Ψ-IR Lab\n\
\n- Prova exata roteada para digital: {}.\n\
- P-bit barato isoladamente rejeitado pelo Boundary Tax: {}.\n\
- P-bit co-localizado escolhido para sampling aproximado: {}.\n\
- Custos declared/metered mistos recusados: {}.\n\
- UNCOMPUTE escolhido apenas se válido e mais barato: {}.\n\
- ERASE escolhido quando UNCOMPUTE custa mais: {}.\n\
- Planos selecionados têm origem declarada: {}.\n\
- Resultado: valida a decisão do Ψ-IR com custos abstratos declarados; não\n\
  executa p-bit, analógico, fotônica, reversível ou quantum hardware.\n",
            self.exact_verification_uses_digital,
            self.boundary_tax_rejects_false_pbit_win,
            self.pbit_sampling_selected_when_colocated,
            self.mixed_cost_provenance_rejected,
            self.uncompute_selected_when_valid_and_cheaper,
            self.erase_selected_when_uncompute_is_costlier,
            self.all_selected_costs_are_declared,
        )
    }
}

pub(super) fn run() -> V6XPhysicsReport {
    let compiler = PhysicalCompiler;
    let exact = compiler
        .realize(
            PhysicalOperation {
                kind: PhysicalOperationKind::ExactVerification,
                precision: PrecisionRequirement::Exact,
                latency_target_ns: None,
            },
            &[
                profile(
                    PhysicalBackend::PBitArray,
                    PhysicalStateKind::Probabilistic,
                    false,
                    PhysicalOperationKind::ExactVerification,
                    cost(1, 0, 20, CostOrigin::Declared),
                ),
                profile(
                    PhysicalBackend::CpuExact,
                    PhysicalStateKind::Digital,
                    true,
                    PhysicalOperationKind::ExactVerification,
                    cost(10, 0, 0, CostOrigin::Declared),
                ),
            ],
        )
        .unwrap();
    let boundary = compiler
        .realize(
            sampling_operation(),
            &[
                profile(
                    PhysicalBackend::CpuExact,
                    PhysicalStateKind::Digital,
                    true,
                    PhysicalOperationKind::PopulationSampling,
                    cost(100, 0, 0, CostOrigin::Declared),
                ),
                profile(
                    PhysicalBackend::PBitArray,
                    PhysicalStateKind::Probabilistic,
                    false,
                    PhysicalOperationKind::PopulationSampling,
                    cost(1, 150, 30, CostOrigin::Declared),
                ),
            ],
        )
        .unwrap();
    let colocated = compiler
        .realize(
            sampling_operation(),
            &[
                profile(
                    PhysicalBackend::CpuExact,
                    PhysicalStateKind::Digital,
                    true,
                    PhysicalOperationKind::PopulationSampling,
                    cost(100, 0, 0, CostOrigin::Declared),
                ),
                profile(
                    PhysicalBackend::PBitArray,
                    PhysicalStateKind::Probabilistic,
                    false,
                    PhysicalOperationKind::PopulationSampling,
                    cost(10, 5, 30, CostOrigin::Declared),
                ),
            ],
        )
        .unwrap();
    let mixed_cost_provenance_rejected = compiler.realize(
        sampling_operation(),
        &[
            profile(
                PhysicalBackend::CpuExact,
                PhysicalStateKind::Digital,
                true,
                PhysicalOperationKind::PopulationSampling,
                cost(10, 0, 0, CostOrigin::Declared),
            ),
            profile(
                PhysicalBackend::PBitArray,
                PhysicalStateKind::Probabilistic,
                false,
                PhysicalOperationKind::PopulationSampling,
                cost(1, 0, 30, CostOrigin::Measured),
            ),
        ],
    ) == Err(RealizationError::IncomparableCostProvenance);

    let erase = CleanupOption {
        action: CleanupAction::Erase,
        cost: cost(20, 0, 0, CostOrigin::Declared),
    };
    let cheap_uncompute = CleanupOption {
        action: CleanupAction::Uncompute,
        cost: cost(5, 0, 0, CostOrigin::Declared),
    };
    let expensive_uncompute = CleanupOption {
        action: CleanupAction::Uncompute,
        cost: cost(30, 0, 0, CostOrigin::Declared),
    };
    let uncompute_selected_when_valid_and_cheaper = compiler
        .select_cleanup(true, true, &[erase, cheap_uncompute])
        .is_ok_and(|plan| plan.action == CleanupAction::Uncompute);
    let erase_selected_when_uncompute_is_costlier = compiler
        .select_cleanup(true, true, &[erase, expensive_uncompute])
        .is_ok_and(|plan| plan.action == CleanupAction::Erase);

    V6XPhysicsReport {
        exact_verification_uses_digital: exact.backend == PhysicalBackend::CpuExact,
        boundary_tax_rejects_false_pbit_win: boundary.backend == PhysicalBackend::CpuExact,
        pbit_sampling_selected_when_colocated: colocated.backend == PhysicalBackend::PBitArray,
        mixed_cost_provenance_rejected,
        uncompute_selected_when_valid_and_cheaper,
        erase_selected_when_uncompute_is_costlier,
        all_selected_costs_are_declared: exact.cost.origin == CostOrigin::Declared
            && boundary.cost.origin == CostOrigin::Declared
            && colocated.cost.origin == CostOrigin::Declared,
    }
}

fn sampling_operation() -> PhysicalOperation {
    PhysicalOperation {
        kind: PhysicalOperationKind::PopulationSampling,
        precision: PrecisionRequirement::BoundedError {
            max_error_milliunits: 50,
        },
        latency_target_ns: None,
    }
}

fn profile(
    backend: PhysicalBackend,
    state: PhysicalStateKind,
    supports_exact: bool,
    operation: PhysicalOperationKind,
    cost: PhysicalCost,
) -> PhysicalProfile {
    PhysicalProfile {
        backend,
        state,
        supports_exact,
        operations: vec![operation],
        cost,
    }
}

fn cost(
    compute_units: u64,
    boundary_units: u64,
    error_milliunits: u32,
    origin: CostOrigin,
) -> PhysicalCost {
    PhysicalCost {
        encode_units: boundary_units,
        move_units: 0,
        compute_units,
        decode_units: 0,
        verify_units: 0,
        cooling_units: 0,
        calibration_units: 0,
        wear_units: 0,
        latency_ns: 10,
        error_milliunits,
        origin,
        unit: crate::core_v6::PhysicalCostUnit::AbstractScore,
        source_id: 1,
        calibration_id: 1,
    }
}
