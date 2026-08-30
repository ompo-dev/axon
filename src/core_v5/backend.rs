//! Compilador físico declarativo para alvos heterogêneos futuros.
//!
//! Nenhum backend aqui executa hardware. A seleção somente preserva contratos
//! de exatidão e registra custos ainda declarados.

use super::cost::{CostVector, CostWeights};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CognitiveOperation {
    ExactVerification,
    SymbolicLogic,
    SimilaritySearch,
    TemporalPattern,
}

impl CognitiveOperation {
    fn requires_exact(self) -> bool {
        matches!(self, Self::ExactVerification | Self::SymbolicLogic)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PhysicalBackend {
    CpuExact,
    HdcApprox,
    ReservoirApprox,
    MemristorApprox,
    PhotonicInterconnect,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendProfile {
    pub backend: PhysicalBackend,
    pub supports_exact: bool,
    pub supported_operations: Vec<CognitiveOperation>,
    pub estimated_cost: CostVector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendPlan {
    pub operation: CognitiveOperation,
    pub backend: PhysicalBackend,
    pub estimated_cost: CostVector,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhysicalCognitiveCompiler;

impl PhysicalCognitiveCompiler {
    pub fn select(
        &self,
        operation: CognitiveOperation,
        profiles: &[BackendProfile],
    ) -> Option<BackendPlan> {
        let eligible = profiles
            .iter()
            .filter(|profile| profile.supported_operations.contains(&operation))
            .filter(|profile| !operation.requires_exact() || profile.supports_exact)
            .collect::<Vec<_>>();
        let origin = eligible.first()?.estimated_cost.origin;
        if eligible
            .iter()
            .any(|profile| profile.estimated_cost.origin != origin)
        {
            return None;
        }
        eligible
            .into_iter()
            .min_by(|left, right| {
                left.estimated_cost
                    .weighted_total(CostWeights::default())
                    .total_cmp(&right.estimated_cost.weighted_total(CostWeights::default()))
                    .then_with(|| left.backend.cmp(&right.backend))
            })
            .map(|profile| BackendPlan {
                operation,
                backend: profile.backend,
                estimated_cost: profile.estimated_cost,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(
        backend: PhysicalBackend,
        exact: bool,
        operations: &[CognitiveOperation],
        cost: u64,
    ) -> BackendProfile {
        BackendProfile {
            backend,
            supports_exact: exact,
            supported_operations: operations.to_vec(),
            estimated_cost: CostVector::declared(cost, 0, 0, 0, 1),
        }
    }

    #[test]
    fn exact_verification_never_routes_to_an_approximate_backend() {
        let plan = PhysicalCognitiveCompiler.select(
            CognitiveOperation::ExactVerification,
            &[
                profile(
                    PhysicalBackend::HdcApprox,
                    false,
                    &[CognitiveOperation::ExactVerification],
                    1,
                ),
                profile(
                    PhysicalBackend::CpuExact,
                    true,
                    &[CognitiveOperation::ExactVerification],
                    10,
                ),
            ],
        );

        assert_eq!(plan.unwrap().backend, PhysicalBackend::CpuExact);
    }

    #[test]
    fn approximate_similarity_can_choose_a_cheaper_declared_accelerator() {
        let plan = PhysicalCognitiveCompiler.select(
            CognitiveOperation::SimilaritySearch,
            &[
                profile(
                    PhysicalBackend::CpuExact,
                    true,
                    &[CognitiveOperation::SimilaritySearch],
                    10,
                ),
                profile(
                    PhysicalBackend::HdcApprox,
                    false,
                    &[CognitiveOperation::SimilaritySearch],
                    1,
                ),
            ],
        );

        assert_eq!(plan.unwrap().backend, PhysicalBackend::HdcApprox);
    }

    #[test]
    fn compiler_accepts_calibrated_profiles_but_rejects_mixed_provenance() {
        let calibrated = BackendProfile {
            backend: PhysicalBackend::CpuExact,
            supports_exact: true,
            supported_operations: vec![CognitiveOperation::SymbolicLogic],
            estimated_cost: CostVector::measured(3, 0, 0, 0, 1, 1),
        };
        assert_eq!(
            PhysicalCognitiveCompiler
                .select(
                    CognitiveOperation::SymbolicLogic,
                    std::slice::from_ref(&calibrated)
                )
                .unwrap()
                .estimated_cost
                .origin,
            crate::core_v5::CostOrigin::Measured
        );
        assert!(
            PhysicalCognitiveCompiler
                .select(
                    CognitiveOperation::SymbolicLogic,
                    &[
                        calibrated,
                        profile(
                            PhysicalBackend::PhotonicInterconnect,
                            true,
                            &[CognitiveOperation::SymbolicLogic],
                            1,
                        ),
                    ],
                )
                .is_none()
        );
    }
}
