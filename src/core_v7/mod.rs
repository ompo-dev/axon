//! AXON V7: compilacao morfogenica sob restricoes de recursos.
//!
//! Esta camada ainda e experimental. Ela nao executa cognicao geral; ela
//! transforma um orcamento fisico em um plano cognitivo auditavel.

use std::error::Error;
use std::fmt::{Display, Formatter};

const MIB: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CognitiveRegion {
    Kernel,
    WorkingState,
    SemanticCodes,
    RetrievalIndex,
    ProgramCache,
    EpisodicMemory,
    CandidateWorlds,
}

impl CognitiveRegion {
    pub const fn all() -> [Self; 7] {
        [
            Self::Kernel,
            Self::WorkingState,
            Self::SemanticCodes,
            Self::RetrievalIndex,
            Self::ProgramCache,
            Self::EpisodicMemory,
            Self::CandidateWorlds,
        ]
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::WorkingState => "working-state",
            Self::SemanticCodes => "semantic-codes",
            Self::RetrievalIndex => "retrieval-index",
            Self::ProgramCache => "program-cache",
            Self::EpisodicMemory => "episodic-memory",
            Self::CandidateWorlds => "candidate-worlds",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResourceBudget {
    pub memory_bytes: u64,
}

impl ResourceBudget {
    pub const fn memory_only(memory_bytes: u64) -> Self {
        Self { memory_bytes }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkloadProfile {
    pub semantic: f64,
    pub retrieval: f64,
    pub programs: f64,
    pub episodic: f64,
    pub exploration: f64,
}

impl WorkloadProfile {
    pub const fn balanced() -> Self {
        Self {
            semantic: 1.0,
            retrieval: 1.0,
            programs: 1.0,
            episodic: 1.0,
            exploration: 1.0,
        }
    }

    pub const fn research() -> Self {
        Self {
            semantic: 1.15,
            retrieval: 1.05,
            programs: 1.20,
            episodic: 0.85,
            exploration: 1.35,
        }
    }

    pub fn is_valid(self) -> bool {
        let weights = [
            self.semantic,
            self.retrieval,
            self.programs,
            self.episodic,
            self.exploration,
        ];
        weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
            && weights.iter().any(|weight| *weight > 0.0)
    }

    fn weight(self, region: CognitiveRegion) -> f64 {
        match region {
            CognitiveRegion::Kernel | CognitiveRegion::WorkingState => 1.0,
            CognitiveRegion::SemanticCodes => self.semantic,
            CognitiveRegion::RetrievalIndex => self.retrieval,
            CognitiveRegion::ProgramCache => self.programs,
            CognitiveRegion::EpisodicMemory => self.episodic,
            CognitiveRegion::CandidateWorlds => self.exploration,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionTier {
    Critical,
    Compressed,
    Balanced,
    Expanded,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionAllocation {
    pub region: CognitiveRegion,
    pub bytes: u64,
    pub minimum_bytes: u64,
    pub desired_bytes: u64,
    pub utility: f64,
    pub tier: RegionTier,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CognitiveBodyPlan {
    pub budget: ResourceBudget,
    pub allocations: Vec<RegionAllocation>,
    pub active_bytes: u64,
    pub archived_bytes: u64,
    pub quality: f64,
    pub shadow_memory_price: f64,
}

impl CognitiveBodyPlan {
    pub fn allocation(&self, region: CognitiveRegion) -> Option<&RegionAllocation> {
        self.allocations
            .iter()
            .find(|allocation| allocation.region == region)
    }
}

/// Estado econômico imutável que decide se uma nova morfologia paga a migração.
///
/// A decisão é deliberadamente conservadora: uma mudança só é aplicada quando
/// o ganho esperado no workload futuro é maior que o custo explícito de mover
/// a memória ativa. Ela não inventa watts, joules ou economia física medida.
#[derive(Clone, Debug, PartialEq)]
pub struct CognitiveMetabolism {
    compiler: MorphogenicCompiler,
    plan: CognitiveBodyPlan,
    migration_price_per_byte: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RemorphReport {
    pub decision: RemorphDecision,
    pub forced_by_budget: bool,
    pub migration_bytes: u64,
    pub current_utility: f64,
    pub candidate_utility: f64,
    pub expected_future_gain: f64,
    pub migration_cost: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemorphDecision {
    Deferred,
    Applied,
}

impl CognitiveMetabolism {
    pub fn new(
        compiler: MorphogenicCompiler,
        budget: ResourceBudget,
        workload: WorkloadProfile,
        migration_price_per_byte: f64,
    ) -> Result<Self, MorphogenicError> {
        Ok(Self {
            compiler,
            plan: compiler.compile(budget, workload)?,
            migration_price_per_byte: migration_price_per_byte.max(0.0),
        })
    }

    pub fn plan(&self) -> &CognitiveBodyPlan {
        &self.plan
    }

    pub fn observe(
        self,
        budget: ResourceBudget,
        workload: WorkloadProfile,
        expected_cycles: u64,
    ) -> Result<(Self, RemorphReport), MorphogenicError> {
        let candidate = self.compiler.compile(budget, workload)?;
        let current_utility = utility_for_plan(&self.plan, workload);
        let candidate_utility = utility_for_plan(&candidate, workload);
        let migration_bytes = migration_bytes(&self.plan, &candidate);
        let forced_by_budget = self.plan.active_bytes > budget.memory_bytes;
        let expected_future_gain =
            (candidate_utility - current_utility).max(0.0) * expected_cycles as f64;
        let migration_cost = migration_bytes as f64 * self.migration_price_per_byte;
        let decision = if forced_by_budget
            || (candidate_utility > current_utility && expected_future_gain > migration_cost)
        {
            RemorphDecision::Applied
        } else {
            RemorphDecision::Deferred
        };
        let next = if decision == RemorphDecision::Applied {
            Self {
                compiler: self.compiler,
                plan: candidate,
                migration_price_per_byte: self.migration_price_per_byte,
            }
        } else {
            self
        };
        Ok((
            next,
            RemorphReport {
                decision,
                forced_by_budget,
                migration_bytes,
                current_utility,
                candidate_utility,
                expected_future_gain,
                migration_cost,
            },
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MorphogenicError {
    InsufficientKernelBudget {
        required_bytes: u64,
        available_bytes: u64,
    },
    InvalidWorkloadProfile,
}

impl Display for MorphogenicError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientKernelBudget {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "memory budget is below mandatory seed: required {required_bytes} bytes, got {available_bytes}"
            ),
            Self::InvalidWorkloadProfile => {
                write!(
                    f,
                    "workload weights must be finite, non-negative, and non-empty"
                )
            }
        }
    }
}

impl Error for MorphogenicError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MorphogenicCompiler {
    quantum_bytes: u64,
}

impl Default for MorphogenicCompiler {
    fn default() -> Self {
        Self { quantum_bytes: MIB }
    }
}

impl MorphogenicCompiler {
    pub const fn new(quantum_bytes: u64) -> Self {
        Self { quantum_bytes }
    }

    pub fn compile(
        self,
        budget: ResourceBudget,
        workload: WorkloadProfile,
    ) -> Result<CognitiveBodyPlan, MorphogenicError> {
        if !workload.is_valid() {
            return Err(MorphogenicError::InvalidWorkloadProfile);
        }
        let policies = default_policies();
        let required_bytes = policies.iter().map(|policy| policy.minimum_bytes).sum();
        if budget.memory_bytes < required_bytes {
            return Err(MorphogenicError::InsufficientKernelBudget {
                required_bytes,
                available_bytes: budget.memory_bytes,
            });
        }

        let mut bytes = policies
            .iter()
            .map(|policy| policy.minimum_bytes)
            .collect::<Vec<_>>();
        let mut remaining = budget.memory_bytes - required_bytes;
        let mut shadow_memory_price = 0.0;
        let quantum = self.quantum_bytes.max(1);

        while remaining > 0 {
            let Some((index, price)) = best_region(&policies, &bytes, workload, remaining, quantum)
            else {
                break;
            };
            let available = policies[index].desired_bytes - bytes[index];
            let step = remaining.min(quantum).min(available);
            bytes[index] += step;
            remaining -= step;
            shadow_memory_price = price;
        }

        let allocations = policies
            .iter()
            .zip(bytes)
            .map(|(policy, bytes)| allocation(*policy, bytes, workload))
            .collect::<Vec<_>>();
        let active_bytes = allocations.iter().map(|allocation| allocation.bytes).sum();
        let archived_bytes = allocations
            .iter()
            .map(|allocation| allocation.desired_bytes - allocation.bytes)
            .sum();
        let quality = total_utility(&allocations)
            / policies
                .iter()
                .map(|policy| utility(*policy, policy.desired_bytes, workload))
                .sum::<f64>();

        Ok(CognitiveBodyPlan {
            budget,
            allocations,
            active_bytes,
            archived_bytes,
            quality,
            shadow_memory_price,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct RegionPolicy {
    region: CognitiveRegion,
    minimum_bytes: u64,
    desired_bytes: u64,
    saturation_bytes: u64,
    base_utility: f64,
}

fn default_policies() -> [RegionPolicy; 7] {
    [
        policy(CognitiveRegion::Kernel, 24, 96, 24, 3.0),
        policy(CognitiveRegion::WorkingState, 16, 512, 96, 2.2),
        policy(CognitiveRegion::SemanticCodes, 8, 2_048, 256, 2.8),
        policy(CognitiveRegion::RetrievalIndex, 4, 1_024, 192, 2.1),
        policy(CognitiveRegion::ProgramCache, 4, 2_048, 256, 2.4),
        policy(CognitiveRegion::EpisodicMemory, 4, 8_192, 768, 1.9),
        policy(CognitiveRegion::CandidateWorlds, 4, 2_048, 384, 1.7),
    ]
}

fn policy(
    region: CognitiveRegion,
    minimum_mib: u64,
    desired_mib: u64,
    saturation_mib: u64,
    base_utility: f64,
) -> RegionPolicy {
    RegionPolicy {
        region,
        minimum_bytes: minimum_mib * MIB,
        desired_bytes: desired_mib * MIB,
        saturation_bytes: saturation_mib * MIB,
        base_utility,
    }
}

fn best_region(
    policies: &[RegionPolicy],
    bytes: &[u64],
    workload: WorkloadProfile,
    remaining: u64,
    quantum: u64,
) -> Option<(usize, f64)> {
    policies
        .iter()
        .enumerate()
        .filter(|(index, policy)| bytes[*index] < policy.desired_bytes)
        .map(|(index, policy)| {
            let step = remaining
                .min(quantum)
                .min(policy.desired_bytes - bytes[index]);
            let before = utility(*policy, bytes[index], workload);
            let after = utility(*policy, bytes[index] + step, workload);
            (index, (after - before) / step as f64)
        })
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
}

fn allocation(policy: RegionPolicy, bytes: u64, workload: WorkloadProfile) -> RegionAllocation {
    RegionAllocation {
        region: policy.region,
        bytes,
        minimum_bytes: policy.minimum_bytes,
        desired_bytes: policy.desired_bytes,
        utility: utility(policy, bytes, workload),
        tier: tier(policy, bytes),
    }
}

fn utility(policy: RegionPolicy, bytes: u64, workload: WorkloadProfile) -> f64 {
    let normalized = (1.0 + bytes as f64 / policy.saturation_bytes as f64).ln()
        / (1.0 + policy.desired_bytes as f64 / policy.saturation_bytes as f64).ln();
    policy.base_utility * workload.weight(policy.region) * normalized
}

fn tier(policy: RegionPolicy, bytes: u64) -> RegionTier {
    if bytes <= policy.minimum_bytes {
        RegionTier::Critical
    } else {
        let span = policy.desired_bytes - policy.minimum_bytes;
        let filled = bytes - policy.minimum_bytes;
        let ratio = filled as f64 / span.max(1) as f64;
        if ratio < 0.25 {
            RegionTier::Compressed
        } else if ratio < 0.75 {
            RegionTier::Balanced
        } else {
            RegionTier::Expanded
        }
    }
}

fn total_utility(allocations: &[RegionAllocation]) -> f64 {
    allocations
        .iter()
        .map(|allocation| allocation.utility)
        .sum::<f64>()
}

fn utility_for_plan(plan: &CognitiveBodyPlan, workload: WorkloadProfile) -> f64 {
    default_policies()
        .iter()
        .map(|policy| {
            let bytes = plan
                .allocation(policy.region)
                .map_or(0, |allocation| allocation.bytes);
            utility(*policy, bytes, workload)
        })
        .sum()
}

fn migration_bytes(current: &CognitiveBodyPlan, candidate: &CognitiveBodyPlan) -> u64 {
    CognitiveRegion::all()
        .into_iter()
        .map(|region| {
            let current_bytes = current.allocation(region).map_or(0, |entry| entry.bytes);
            let candidate_bytes = candidate.allocation(region).map_or(0, |entry| entry.bytes);
            current_bytes.abs_diff(candidate_bytes)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_and_budget_produce_the_same_body_plan() {
        let compiler = MorphogenicCompiler::default();
        let budget = ResourceBudget::memory_only(512 * MIB);

        assert_eq!(
            compiler.compile(budget, WorkloadProfile::research()),
            compiler.compile(budget, WorkloadProfile::research())
        );
    }

    #[test]
    fn quality_is_monotonic_as_memory_budget_grows() {
        let compiler = MorphogenicCompiler::default();
        let budgets = [64, 128, 256, 512, 1_024, 4_096, 16_384];
        let mut previous = 0.0;

        for budget in budgets {
            let plan = compiler
                .compile(
                    ResourceBudget::memory_only(budget * MIB),
                    WorkloadProfile::research(),
                )
                .unwrap();
            assert!(plan.quality >= previous);
            previous = plan.quality;
        }
    }

    #[test]
    fn constrained_memory_keeps_the_seed_and_archives_large_regions() {
        let plan = MorphogenicCompiler::default()
            .compile(
                ResourceBudget::memory_only(64 * MIB),
                WorkloadProfile::balanced(),
            )
            .unwrap();

        assert_eq!(
            plan.allocation(CognitiveRegion::Kernel).unwrap().tier,
            RegionTier::Critical
        );
        assert_eq!(
            plan.allocation(CognitiveRegion::EpisodicMemory)
                .unwrap()
                .tier,
            RegionTier::Critical
        );
        assert!(plan.archived_bytes > plan.active_bytes);
    }

    #[test]
    fn abundant_memory_expands_high_value_regions_without_exceeding_the_budget() {
        let plan = MorphogenicCompiler::default()
            .compile(
                ResourceBudget::memory_only(16_384 * MIB),
                WorkloadProfile::research(),
            )
            .unwrap();

        assert!(plan.active_bytes <= plan.budget.memory_bytes);
        assert_eq!(plan.archived_bytes, 0);
        assert_eq!(
            plan.allocation(CognitiveRegion::ProgramCache).unwrap().tier,
            RegionTier::Expanded
        );
        assert_eq!(
            plan.allocation(CognitiveRegion::CandidateWorlds)
                .unwrap()
                .tier,
            RegionTier::Expanded
        );
    }

    #[test]
    fn budget_below_mandatory_seed_is_rejected() {
        assert_eq!(
            MorphogenicCompiler::default().compile(
                ResourceBudget::memory_only(63 * MIB),
                WorkloadProfile::balanced(),
            ),
            Err(MorphogenicError::InsufficientKernelBudget {
                required_bytes: 64 * MIB,
                available_bytes: 63 * MIB,
            })
        );
    }

    #[test]
    fn metabolism_only_remorphs_when_future_value_pays_migration() {
        let compiler = MorphogenicCompiler::default();
        let exploration_spike = WorkloadProfile {
            semantic: 0.5,
            retrieval: 0.5,
            programs: 0.5,
            episodic: 0.5,
            exploration: 10.0,
        };
        let metabolism = CognitiveMetabolism::new(
            compiler,
            ResourceBudget::memory_only(128 * MIB),
            WorkloadProfile::balanced(),
            0.000_000_001,
        )
        .unwrap();
        let original = metabolism.plan().clone();
        let (metabolism, deferred) = metabolism
            .observe(ResourceBudget::memory_only(128 * MIB), exploration_spike, 0)
            .unwrap();
        assert_eq!(deferred.decision, RemorphDecision::Deferred);
        assert_eq!(metabolism.plan(), &original);

        let (remorphed, applied) = metabolism
            .observe(
                ResourceBudget::memory_only(128 * MIB),
                exploration_spike,
                1_000_000,
            )
            .unwrap();
        assert_eq!(applied.decision, RemorphDecision::Applied);
        assert_ne!(remorphed.plan(), &original);
        assert!(applied.expected_future_gain > applied.migration_cost);
    }

    #[test]
    fn metabolism_must_remorph_when_the_budget_shrinks() {
        let metabolism = CognitiveMetabolism::new(
            MorphogenicCompiler::default(),
            ResourceBudget::memory_only(512 * MIB),
            WorkloadProfile::balanced(),
            1_000_000.0,
        )
        .unwrap();

        let (remorphed, report) = metabolism
            .observe(
                ResourceBudget::memory_only(128 * MIB),
                WorkloadProfile::balanced(),
                0,
            )
            .unwrap();

        assert_eq!(report.decision, RemorphDecision::Applied);
        assert!(report.forced_by_budget);
        assert!(remorphed.plan().active_bytes <= 128 * MIB);
    }

    #[test]
    fn invalid_workload_weights_are_rejected() {
        let invalid = WorkloadProfile {
            semantic: f64::NAN,
            ..WorkloadProfile::balanced()
        };

        assert_eq!(
            MorphogenicCompiler::default().compile(ResourceBudget::memory_only(64 * MIB), invalid),
            Err(MorphogenicError::InvalidWorkloadProfile)
        );

        let empty = WorkloadProfile {
            semantic: 0.0,
            retrieval: 0.0,
            programs: 0.0,
            episodic: 0.0,
            exploration: 0.0,
        };
        assert_eq!(
            MorphogenicCompiler::default().compile(ResourceBudget::memory_only(64 * MIB), empty),
            Err(MorphogenicError::InvalidWorkloadProfile)
        );
    }
}
