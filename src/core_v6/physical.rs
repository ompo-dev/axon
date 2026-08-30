//! Ψ-IR experimental: descreve e seleciona realizações físicas sem dirigir hardware.
//!
//! Perfis `Declared` são hipóteses de engenharia. Um plano só pode ser chamado
//! medido quando todos os custos comparados foram instrumentados com a mesma
//! proveniência. Nenhuma variante deste módulo executa dispositivos físicos.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::core_v5::CostOrigin;

/// Forma física em que um backend representa o estado durante uma operação.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PhysicalStateKind {
    Digital,
    Binary,
    Ternary,
    Analog,
    Probabilistic,
    Oscillatory,
    Photonic,
    Reservoir,
    Quantum,
    Reversible,
}

/// Transformação lógica que pode receber uma realização física diferente.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PhysicalOperationKind {
    ExactVerification,
    SymbolicSearch,
    AssociativeSimilarity,
    PopulationSampling,
    TemporalDynamics,
    DenseMatrixBlock,
    ReversibleScratch,
}

impl PhysicalOperationKind {
    fn inherently_exact(self) -> bool {
        matches!(self, Self::ExactVerification | Self::SymbolicSearch)
    }
}

/// Precisão exigida pela Ω-IR; o backend só recebe a operação se puder cumpri-la.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrecisionRequirement {
    Exact,
    BoundedError { max_error_milliunits: u32 },
}

impl PrecisionRequirement {
    fn accepts(self, error_milliunits: u32) -> bool {
        match self {
            Self::Exact => error_milliunits == 0,
            Self::BoundedError {
                max_error_milliunits,
            } => error_milliunits <= max_error_milliunits,
        }
    }
}

/// Pedido de `REALIZE` produzido pela camada Ω-IR.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalOperation {
    pub kind: PhysicalOperationKind,
    pub precision: PrecisionRequirement,
    pub latency_target_ns: Option<u64>,
}

impl PhysicalOperation {
    fn requires_exact(self) -> bool {
        self.kind.inherently_exact()
            || matches!(
                self.precision,
                PrecisionRequirement::Exact
                    | PrecisionRequirement::BoundedError {
                        max_error_milliunits: 0
                    }
            )
    }
}

/// Alvos possíveis. Variantes não digitais continuam sendo apenas perfis nesta fase.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PhysicalBackend {
    CpuExact,
    SimdExact,
    HdcAssociative,
    AnalogCrossbar,
    PBitArray,
    OscillatorNetwork,
    PhotonicReservoir,
    QuantumReservoir,
    QuantumCircuit,
    ReversibleCircuit,
}

/// Custo de ponta a ponta: conversões e controle não podem desaparecer da decisão.
///
/// A unidade é deliberadamente abstrata enquanto `origin == Declared`. Somente
/// um perfil instrumentado pode usar `Measured`; isso não transforma por si só
/// a seleção em uma alegação de eficiência de hardware geral.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PhysicalCostUnit {
    AbstractScore,
    NanojouleEquivalent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalCost {
    pub encode_units: u64,
    pub move_units: u64,
    pub compute_units: u64,
    pub decode_units: u64,
    pub verify_units: u64,
    pub cooling_units: u64,
    pub calibration_units: u64,
    pub wear_units: u64,
    pub latency_ns: u64,
    pub error_milliunits: u32,
    pub origin: CostOrigin,
    pub unit: PhysicalCostUnit,
    /// Identifica a instrumentação ou hipótese de engenharia que produziu o custo.
    pub source_id: u64,
    /// Versão da calibração aplicada à fonte; revisões diferentes não competem.
    pub calibration_id: u64,
}

impl PhysicalCost {
    pub fn comparability_key(self) -> (CostOrigin, PhysicalCostUnit, u64, u64) {
        (self.origin, self.unit, self.source_id, self.calibration_id)
    }

    pub fn checked_total(self) -> Option<u64> {
        [
            self.encode_units,
            self.move_units,
            self.compute_units,
            self.decode_units,
            self.verify_units,
            self.cooling_units,
            self.calibration_units,
            self.wear_units,
        ]
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
    }
}

/// Característica declarada ou medida de um backend para uma classe de operações.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalProfile {
    pub backend: PhysicalBackend,
    pub state: PhysicalStateKind,
    pub supports_exact: bool,
    pub operations: Vec<PhysicalOperationKind>,
    pub cost: PhysicalCost,
}

/// Plano auditável retornado por `REALIZE`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalPlan {
    pub operation: PhysicalOperation,
    pub backend: PhysicalBackend,
    pub state: PhysicalStateKind,
    pub total_boundary_units: u64,
    pub cost: PhysicalCost,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhysicalCompiler;

impl PhysicalCompiler {
    /// Escolhe a realização mais barata que satisfaz precisão, latência e origem.
    pub fn realize(
        &self,
        operation: PhysicalOperation,
        profiles: &[PhysicalProfile],
    ) -> Result<PhysicalPlan, RealizationError> {
        let eligible = profiles
            .iter()
            .filter(|profile| profile.operations.contains(&operation.kind))
            .filter(|profile| !operation.requires_exact() || profile.supports_exact)
            .filter(|profile| operation.precision.accepts(profile.cost.error_milliunits))
            .filter(|profile| {
                operation
                    .latency_target_ns
                    .is_none_or(|target| profile.cost.latency_ns <= target)
            })
            .collect::<Vec<_>>();
        let comparability_key = eligible
            .first()
            .ok_or(RealizationError::NoEligibleBackend)?
            .cost
            .comparability_key();
        if eligible
            .iter()
            .any(|profile| profile.cost.comparability_key() != comparability_key)
        {
            return Err(RealizationError::IncomparableCostProvenance);
        }
        if eligible
            .iter()
            .any(|profile| profile.cost.checked_total().is_none())
        {
            return Err(RealizationError::InvalidCostOverflow);
        }
        let selected = eligible
            .into_iter()
            .min_by(|left, right| {
                left.cost
                    .checked_total()
                    .expect("eligible cost was validated")
                    .cmp(
                        &right
                            .cost
                            .checked_total()
                            .expect("eligible cost was validated"),
                    )
                    .then_with(|| left.cost.error_milliunits.cmp(&right.cost.error_milliunits))
                    .then_with(|| left.backend.cmp(&right.backend))
            })
            .expect("eligible is nonempty after origin check");
        Ok(PhysicalPlan {
            operation,
            backend: selected.backend,
            state: selected.state,
            total_boundary_units: selected
                .cost
                .checked_total()
                .expect("selected cost was validated"),
            cost: selected.cost,
        })
    }

    /// Decide como liberar scratch sem tornar reversibilidade uma regra universal.
    pub fn select_cleanup(
        &self,
        reversible: bool,
        result_committed: bool,
        candidates: &[CleanupOption],
    ) -> Result<CleanupPlan, RealizationError> {
        let eligible = candidates
            .iter()
            .filter(|candidate| {
                candidate.action != CleanupAction::Uncompute || reversible && result_committed
            })
            .collect::<Vec<_>>();
        let comparability_key = eligible
            .first()
            .ok_or(RealizationError::NoEligibleCleanup)?
            .cost
            .comparability_key();
        if eligible
            .iter()
            .any(|candidate| candidate.cost.comparability_key() != comparability_key)
        {
            return Err(RealizationError::IncomparableCostProvenance);
        }
        if eligible
            .iter()
            .any(|candidate| candidate.cost.checked_total().is_none())
        {
            return Err(RealizationError::InvalidCostOverflow);
        }
        let selected = eligible
            .into_iter()
            .min_by(|left, right| {
                left.cost
                    .checked_total()
                    .expect("eligible cost was validated")
                    .cmp(
                        &right
                            .cost
                            .checked_total()
                            .expect("eligible cost was validated"),
                    )
                    .then_with(|| left.action.cmp(&right.action))
            })
            .expect("eligible is nonempty after origin check");
        Ok(CleanupPlan {
            action: selected.action,
            total_boundary_units: selected
                .cost
                .checked_total()
                .expect("selected cost was validated"),
            cost: selected.cost,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealizationError {
    NoEligibleBackend,
    NoEligibleCleanup,
    IncomparableCostProvenance,
    InvalidCostOverflow,
}

impl Display for RealizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoEligibleBackend => write!(f, "no physical profile satisfies this operation"),
            Self::NoEligibleCleanup => write!(f, "no scratch cleanup policy is valid"),
            Self::IncomparableCostProvenance => write!(
                f,
                "physical costs must share origin, unit, source and calibration before comparison"
            ),
            Self::InvalidCostOverflow => write!(f, "physical cost total overflows u64"),
        }
    }
}

impl Error for RealizationError {}

/// Política de limpeza de estado temporário; o compilador pode comparar as quatro.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CleanupAction {
    Erase,
    Checkpoint,
    Recompute,
    Uncompute,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CleanupOption {
    pub action: CleanupAction,
    pub cost: PhysicalCost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CleanupPlan {
    pub action: CleanupAction,
    pub total_boundary_units: u64,
    pub cost: PhysicalCost,
}

/// Scratch puramente lógico cujo resultado é comprometido antes da reversão.
///
/// `uncompute` restaura o scratch inicial sem apagar o resultado já extraído.
/// É uma semântica testável em software, não uma afirmação de reversibilidade
/// termodinâmica de uma CPU convencional.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReversibleScratch<T> {
    initial: T,
    working: T,
    committed_result: Option<T>,
    forward_steps: Vec<String>,
}

impl<T: Clone> ReversibleScratch<T> {
    pub fn new(initial: T) -> Self {
        Self {
            working: initial.clone(),
            initial,
            committed_result: None,
            forward_steps: Vec::new(),
        }
    }

    pub fn compute(&self, step: impl Into<String>, working: T) -> Self {
        let mut next = self.clone();
        next.working = working;
        next.forward_steps.push(step.into());
        next
    }

    pub fn commit_result(&self, result: T) -> Self {
        Self {
            committed_result: Some(result),
            ..self.clone()
        }
    }

    pub fn uncompute(&self) -> Option<Self> {
        self.committed_result.as_ref().map(|result| Self {
            initial: self.initial.clone(),
            working: self.initial.clone(),
            committed_result: Some(result.clone()),
            forward_steps: Vec::new(),
        })
    }

    pub fn working(&self) -> &T {
        &self.working
    }

    pub fn committed_result(&self) -> Option<&T> {
        self.committed_result.as_ref()
    }

    pub fn pending_steps(&self) -> usize {
        self.forward_steps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(total_compute: u64, boundary: u64, error: u32, origin: CostOrigin) -> PhysicalCost {
        PhysicalCost {
            encode_units: boundary,
            move_units: 0,
            compute_units: total_compute,
            decode_units: 0,
            verify_units: 0,
            cooling_units: 0,
            calibration_units: 0,
            wear_units: 0,
            latency_ns: 10,
            error_milliunits: error,
            origin,
            unit: PhysicalCostUnit::AbstractScore,
            source_id: 1,
            calibration_id: 1,
        }
    }

    fn profile(
        backend: PhysicalBackend,
        state: PhysicalStateKind,
        exact: bool,
        operation: PhysicalOperationKind,
        cost: PhysicalCost,
    ) -> PhysicalProfile {
        PhysicalProfile {
            backend,
            state,
            supports_exact: exact,
            operations: vec![operation],
            cost,
        }
    }

    #[test]
    fn exact_operation_never_routes_to_an_approximate_physical_backend() {
        let operation = PhysicalOperation {
            kind: PhysicalOperationKind::ExactVerification,
            precision: PrecisionRequirement::Exact,
            latency_target_ns: None,
        };
        let plan = PhysicalCompiler
            .realize(
                operation,
                &[
                    profile(
                        PhysicalBackend::PBitArray,
                        PhysicalStateKind::Probabilistic,
                        false,
                        PhysicalOperationKind::ExactVerification,
                        cost(1, 0, 10, CostOrigin::Declared),
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

        assert_eq!(plan.backend, PhysicalBackend::CpuExact);
    }

    #[test]
    fn boundary_tax_can_make_a_cheap_pbit_compute_path_more_expensive_end_to_end() {
        let operation = PhysicalOperation {
            kind: PhysicalOperationKind::PopulationSampling,
            precision: PrecisionRequirement::BoundedError {
                max_error_milliunits: 50,
            },
            latency_target_ns: None,
        };
        let plan = PhysicalCompiler
            .realize(
                operation,
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

        assert_eq!(plan.backend, PhysicalBackend::CpuExact);
        assert_eq!(plan.total_boundary_units, 100);
    }

    #[test]
    fn colocated_approximate_backend_can_win_when_full_boundary_cost_is_lower() {
        let operation = PhysicalOperation {
            kind: PhysicalOperationKind::AssociativeSimilarity,
            precision: PrecisionRequirement::BoundedError {
                max_error_milliunits: 30,
            },
            latency_target_ns: Some(20),
        };
        let plan = PhysicalCompiler
            .realize(
                operation,
                &[
                    profile(
                        PhysicalBackend::CpuExact,
                        PhysicalStateKind::Digital,
                        true,
                        PhysicalOperationKind::AssociativeSimilarity,
                        cost(80, 0, 0, CostOrigin::Declared),
                    ),
                    profile(
                        PhysicalBackend::HdcAssociative,
                        PhysicalStateKind::Ternary,
                        false,
                        PhysicalOperationKind::AssociativeSimilarity,
                        cost(10, 5, 20, CostOrigin::Declared),
                    ),
                ],
            )
            .unwrap();

        assert_eq!(plan.backend, PhysicalBackend::HdcAssociative);
        assert_eq!(plan.total_boundary_units, 15);
    }

    #[test]
    fn realization_refuses_to_rank_declared_against_measured_costs() {
        let operation = PhysicalOperation {
            kind: PhysicalOperationKind::PopulationSampling,
            precision: PrecisionRequirement::BoundedError {
                max_error_milliunits: 100,
            },
            latency_target_ns: None,
        };
        let result = PhysicalCompiler.realize(
            operation,
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
                    cost(1, 0, 50, CostOrigin::Measured),
                ),
            ],
        );

        assert_eq!(result, Err(RealizationError::IncomparableCostProvenance));
    }

    #[test]
    fn zero_error_bound_requires_an_exact_backend() {
        let operation = PhysicalOperation {
            kind: PhysicalOperationKind::AssociativeSimilarity,
            precision: PrecisionRequirement::BoundedError {
                max_error_milliunits: 0,
            },
            latency_target_ns: None,
        };
        let plan = PhysicalCompiler
            .realize(
                operation,
                &[
                    profile(
                        PhysicalBackend::PBitArray,
                        PhysicalStateKind::Probabilistic,
                        false,
                        PhysicalOperationKind::AssociativeSimilarity,
                        cost(1, 0, 0, CostOrigin::Declared),
                    ),
                    profile(
                        PhysicalBackend::CpuExact,
                        PhysicalStateKind::Digital,
                        true,
                        PhysicalOperationKind::AssociativeSimilarity,
                        cost(10, 0, 0, CostOrigin::Declared),
                    ),
                ],
            )
            .unwrap();

        assert_eq!(plan.backend, PhysicalBackend::CpuExact);
    }

    #[test]
    fn realization_refuses_costs_from_different_calibrations_and_overflow() {
        let operation = PhysicalOperation {
            kind: PhysicalOperationKind::PopulationSampling,
            precision: PrecisionRequirement::BoundedError {
                max_error_milliunits: 100,
            },
            latency_target_ns: None,
        };
        let first = profile(
            PhysicalBackend::CpuExact,
            PhysicalStateKind::Digital,
            true,
            PhysicalOperationKind::PopulationSampling,
            cost(10, 0, 0, CostOrigin::Measured),
        );
        let mut different_calibration = profile(
            PhysicalBackend::PBitArray,
            PhysicalStateKind::Probabilistic,
            false,
            PhysicalOperationKind::PopulationSampling,
            cost(1, 0, 30, CostOrigin::Measured),
        );
        different_calibration.cost.calibration_id = 2;
        assert_eq!(
            PhysicalCompiler.realize(operation, &[first, different_calibration]),
            Err(RealizationError::IncomparableCostProvenance)
        );
        assert_eq!(
            PhysicalCompiler.realize(
                operation,
                &[profile(
                    PhysicalBackend::CpuExact,
                    PhysicalStateKind::Digital,
                    true,
                    PhysicalOperationKind::PopulationSampling,
                    cost(u64::MAX, 1, 0, CostOrigin::Declared),
                )],
            ),
            Err(RealizationError::InvalidCostOverflow)
        );
    }

    #[test]
    fn uncompute_restores_scratch_only_after_preserving_the_result() {
        let scratch = ReversibleScratch::new("initial".to_string())
            .compute("derive temporary witness", "temporary".to_string());
        assert!(scratch.uncompute().is_none());

        let cleaned = scratch
            .commit_result("proof".to_string())
            .uncompute()
            .unwrap();
        assert_eq!(cleaned.working(), "initial");
        assert_eq!(cleaned.committed_result(), Some(&"proof".to_string()));
        assert_eq!(cleaned.pending_steps(), 0);
    }

    #[test]
    fn cleanup_uses_uncompute_only_when_valid_and_cheaper_than_erasure() {
        let erase = CleanupOption {
            action: CleanupAction::Erase,
            cost: cost(20, 0, 0, CostOrigin::Declared),
        };
        let uncompute = CleanupOption {
            action: CleanupAction::Uncompute,
            cost: cost(5, 0, 0, CostOrigin::Declared),
        };

        assert_eq!(
            PhysicalCompiler
                .select_cleanup(false, true, &[erase, uncompute])
                .unwrap()
                .action,
            CleanupAction::Erase
        );
        assert_eq!(
            PhysicalCompiler
                .select_cleanup(true, true, &[erase, uncompute])
                .unwrap()
                .action,
            CleanupAction::Uncompute
        );
        let expensive_uncompute = CleanupOption {
            action: CleanupAction::Uncompute,
            cost: cost(30, 0, 0, CostOrigin::Declared),
        };
        assert_eq!(
            PhysicalCompiler
                .select_cleanup(true, true, &[erase, expensive_uncompute])
                .unwrap()
                .action,
            CleanupAction::Erase
        );
    }
}
