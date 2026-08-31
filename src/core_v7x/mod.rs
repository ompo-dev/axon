//! AXON V7-X: contratos auditáveis, memória semântica em níveis, mundos COW e capital cognitivo.
//!
//! As estruturas deste módulo são determinísticas e imutáveis. Elas testam
//! contratos e custos lógicos; medições físicas ficam nos binários de benchmark.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

const KIB: u64 = 1024;
pub const MIB: u64 = 1024 * KIB;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorContract {
    pub semantic_id: String,
    pub revision: u64,
    pub max_error_milliunits: u64,
    pub required_claims: BTreeSet<String>,
    pub semantic_checksum: u64,
}

impl FactorContract {
    pub fn new(
        semantic_id: impl Into<String>,
        revision: u64,
        max_error_milliunits: u64,
        required_claims: BTreeSet<String>,
        semantic_checksum: u64,
    ) -> Self {
        Self {
            semantic_id: semantic_id.into(),
            revision,
            max_error_milliunits,
            required_claims,
            semantic_checksum,
        }
    }

    pub fn refines(&self, required: &Self) -> bool {
        !self.semantic_id.is_empty()
            && !required.semantic_id.is_empty()
            && self.semantic_id == required.semantic_id
            && self.revision >= required.revision
            && self.max_error_milliunits <= required.max_error_milliunits
            && self.semantic_checksum == required.semantic_checksum
            && self.required_claims.is_superset(&required.required_claims)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ImplementationKind {
    Exact,
    Approximate,
    Compiled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorImplementation {
    pub kind: ImplementationKind,
    pub contract: FactorContract,
    /// Custo lógico comparável entre as implementações deste mesmo Factor.
    pub lifetime_cost_units: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecisionCertificate {
    pub winner_lower_bound: i64,
    pub runner_up_upper_bound: i64,
}

impl DecisionCertificate {
    /// Conservadoramente, o erro pode reduzir o vencedor e elevar o rival.
    pub fn survives(self, error_milliunits: u64) -> bool {
        let error = i64::try_from(error_milliunits).unwrap_or(i64::MAX);
        self.winner_lower_bound.saturating_sub(error)
            > self.runner_up_upper_bound.saturating_add(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractedFactor {
    pub required_contract: FactorContract,
    pub certificate: DecisionCertificate,
    pub implementations: Vec<FactorImplementation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RealizationPlan {
    pub kind: ImplementationKind,
    pub lifetime_cost_units: u64,
    pub preserved_certificate: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractError {
    NoEligibleImplementation,
}

impl Display for ContractError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoEligibleImplementation => {
                write!(
                    f,
                    "no implementation preserves the factor and decision contracts"
                )
            }
        }
    }
}

impl Error for ContractError {}

impl ContractedFactor {
    pub fn realize(&self) -> Result<RealizationPlan, ContractError> {
        self.implementations
            .iter()
            .filter(|implementation| implementation.contract.refines(&self.required_contract))
            .filter(|implementation| {
                self.certificate
                    .survives(implementation.contract.max_error_milliunits)
            })
            .min_by(|left, right| {
                left.lifetime_cost_units
                    .cmp(&right.lifetime_cost_units)
                    .then_with(|| left.kind.cmp(&right.kind))
            })
            .map(|implementation| RealizationPlan {
                kind: implementation.kind,
                lifetime_cost_units: implementation.lifetime_cost_units,
                preserved_certificate: true,
            })
            .ok_or(ContractError::NoEligibleImplementation)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SemanticResolution {
    Summary,
    Compact,
    Full,
    Exact,
}

impl SemanticResolution {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Summary => 64 * KIB,
            Self::Compact => 256 * KIB,
            Self::Full => MIB,
            Self::Exact => 4 * MIB,
        }
    }

    fn next(self) -> Option<Self> {
        match self {
            Self::Summary => Some(Self::Compact),
            Self::Compact => Some(Self::Full),
            Self::Full => Some(Self::Exact),
            Self::Exact => None,
        }
    }

    fn utility_multiplier(self) -> f64 {
        match self {
            Self::Summary => 0.25,
            Self::Compact => 0.50,
            Self::Full => 0.80,
            Self::Exact => 1.00,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MorphContract {
    pub semantic_id: String,
    pub revision: u64,
    pub observable_checksum: u64,
    pub protected_claims: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticPage {
    pub handle: u32,
    pub contract: MorphContract,
    pub protected: bool,
    pub utility_milliunits: u64,
    /// Catálogo in-memory de verificação deste protótipo. Os bytes ativos são
    /// representados por `SemanticResolution`; um arquivo frio físico ainda não
    /// foi implementado.
    pub exact_answer: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticMorphology {
    pub budget_bytes: u64,
    pub residences: BTreeMap<u32, SemanticResolution>,
    pub active_bytes: u64,
    pub archived_detail_bytes: u64,
    pub logical_utility: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticVirtualMemory {
    corpus: Arc<Vec<SemanticPage>>,
    morphology: SemanticMorphology,
    epoch: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShadowMorph {
    source_epoch: u64,
    candidate: SemanticMorphology,
    pub migration_bytes: u64,
    pub verified_contracts: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticMemoryError {
    EmptyCorpus,
    InsufficientSummaryBudget {
        required_bytes: u64,
        available_bytes: u64,
    },
    UnverifiedShadow,
    StaleShadow,
}

impl Display for SemanticMemoryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCorpus => write!(f, "semantic virtual memory needs at least one page"),
            Self::InsufficientSummaryBudget {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "semantic summary tier requires {required_bytes} bytes, got {available_bytes}"
            ),
            Self::UnverifiedShadow => {
                write!(f, "shadow morphology did not preserve protected contracts")
            }
            Self::StaleShadow => write!(
                f,
                "shadow morphology was built from an older active morphology"
            ),
        }
    }
}

impl Error for SemanticMemoryError {}

impl SemanticVirtualMemory {
    pub fn synthetic(page_count: u32, budget_bytes: u64) -> Result<Self, SemanticMemoryError> {
        let corpus = (0..page_count)
            .map(|handle| {
                let protected = handle < 64;
                let exact_answer = mix64(u64::from(handle) ^ 0xA70D_7E57_11C0_5EED);
                SemanticPage {
                    handle,
                    contract: MorphContract {
                        semantic_id: format!("fact:{handle}"),
                        revision: 1,
                        observable_checksum: mix64(exact_answer),
                        protected_claims: if protected {
                            BTreeSet::from(["essential-recall".to_string()])
                        } else {
                            BTreeSet::new()
                        },
                    },
                    protected,
                    utility_milliunits: if protected {
                        10_000
                    } else {
                        1_000 + u64::from(handle % 97)
                    },
                    exact_answer,
                }
            })
            .collect::<Vec<_>>();
        Self::new(corpus, budget_bytes)
    }

    pub fn new(corpus: Vec<SemanticPage>, budget_bytes: u64) -> Result<Self, SemanticMemoryError> {
        let corpus = Arc::new(corpus);
        let morphology = build_morphology(&corpus, budget_bytes)?;
        Ok(Self {
            corpus,
            morphology,
            epoch: 0,
        })
    }

    pub fn morphology(&self) -> &SemanticMorphology {
        &self.morphology
    }

    pub fn recall(&self, handle: u32, required: SemanticResolution) -> Option<u64> {
        let page = self.corpus.iter().find(|page| page.handle == handle)?;
        (self
            .morphology
            .residences
            .get(&handle)
            .is_some_and(|actual| *actual >= required))
        .then_some(page.exact_answer)
    }

    pub fn recall_fraction(&self, required: SemanticResolution) -> f64 {
        if self.corpus.is_empty() {
            return 0.0;
        }
        let recovered = self
            .corpus
            .iter()
            .filter(|page| self.recall(page.handle, required) == Some(page.exact_answer))
            .count();
        recovered as f64 / self.corpus.len() as f64
    }

    pub fn protected_recall_fraction(&self) -> f64 {
        let protected = self
            .corpus
            .iter()
            .filter(|page| page.protected)
            .collect::<Vec<_>>();
        if protected.is_empty() {
            return 1.0;
        }
        let recovered = protected
            .iter()
            .filter(|page| {
                self.recall(page.handle, SemanticResolution::Summary)
                    .is_some()
            })
            .count();
        recovered as f64 / protected.len() as f64
    }

    pub fn stage_remorph(&self, budget_bytes: u64) -> Result<ShadowMorph, SemanticMemoryError> {
        let candidate = build_morphology(&self.corpus, budget_bytes)?;
        let verified_contracts = self
            .corpus
            .iter()
            .filter(|page| page.protected)
            .filter(|page| {
                candidate
                    .residences
                    .get(&page.handle)
                    .is_some_and(|resolution| *resolution >= SemanticResolution::Summary)
                    && page.contract.observable_checksum == mix64(page.exact_answer)
                    && page.contract.protected_claims.contains("essential-recall")
            })
            .count();
        let required_contracts = self.corpus.iter().filter(|page| page.protected).count();
        if verified_contracts != required_contracts {
            return Err(SemanticMemoryError::UnverifiedShadow);
        }
        Ok(ShadowMorph {
            source_epoch: self.epoch,
            migration_bytes: self
                .morphology
                .active_bytes
                .abs_diff(candidate.active_bytes),
            candidate,
            verified_contracts,
        })
    }

    /// A troca é atômica no modelo imutável: somente o estado retornado observa
    /// a nova morfologia; a instância anterior permanece íntegra para rollback.
    pub fn commit(&self, shadow: ShadowMorph) -> Result<Self, SemanticMemoryError> {
        if shadow.source_epoch != self.epoch {
            return Err(SemanticMemoryError::StaleShadow);
        }
        let required_contracts = self.corpus.iter().filter(|page| page.protected).count();
        if shadow.verified_contracts != required_contracts {
            return Err(SemanticMemoryError::UnverifiedShadow);
        }
        Ok(Self {
            corpus: Arc::clone(&self.corpus),
            morphology: shadow.candidate,
            epoch: self.epoch.saturating_add(1),
        })
    }
}

fn build_morphology(
    corpus: &[SemanticPage],
    budget_bytes: u64,
) -> Result<SemanticMorphology, SemanticMemoryError> {
    if corpus.is_empty() {
        return Err(SemanticMemoryError::EmptyCorpus);
    }
    let required_bytes = (corpus.len() as u64).saturating_mul(SemanticResolution::Summary.bytes());
    if budget_bytes < required_bytes {
        return Err(SemanticMemoryError::InsufficientSummaryBudget {
            required_bytes,
            available_bytes: budget_bytes,
        });
    }
    let mut resolutions = vec![SemanticResolution::Summary; corpus.len()];
    let mut active_bytes = required_bytes;
    while let Some((index, next, delta_bytes)) =
        best_upgrade(corpus, &resolutions, budget_bytes - active_bytes)
    {
        resolutions[index] = next;
        active_bytes = active_bytes.saturating_add(delta_bytes);
    }
    let residences = corpus
        .iter()
        .zip(resolutions.iter().copied())
        .map(|(page, resolution)| (page.handle, resolution))
        .collect::<BTreeMap<_, _>>();
    let archived_detail_bytes = resolutions
        .iter()
        .map(|resolution| SemanticResolution::Exact.bytes() - resolution.bytes())
        .sum();
    let logical_utility = corpus
        .iter()
        .zip(resolutions.iter().copied())
        .map(|(page, resolution)| page.utility_milliunits as f64 * resolution.utility_multiplier())
        .sum();
    Ok(SemanticMorphology {
        budget_bytes,
        residences,
        active_bytes,
        archived_detail_bytes,
        logical_utility,
    })
}

fn best_upgrade(
    corpus: &[SemanticPage],
    resolutions: &[SemanticResolution],
    remaining_bytes: u64,
) -> Option<(usize, SemanticResolution, u64)> {
    corpus
        .iter()
        .zip(resolutions.iter().copied())
        .enumerate()
        .filter_map(|(index, (page, current))| {
            let next = current.next()?;
            let delta_bytes = next.bytes() - current.bytes();
            (delta_bytes <= remaining_bytes).then(|| {
                let utility_delta = page.utility_milliunits as f64
                    * (next.utility_multiplier() - current.utility_multiplier());
                (index, next, delta_bytes, utility_delta / delta_bytes as f64)
            })
        })
        .max_by(|left, right| {
            left.3
                .total_cmp(&right.3)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(index, next, delta_bytes, _)| (index, next, delta_bytes))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldBase {
    words: Arc<Vec<u64>>,
}

impl WorldBase {
    pub fn new(words: Vec<u64>) -> Self {
        Self {
            words: Arc::new(words),
        }
    }

    pub fn bytes(&self) -> u64 {
        (self.words.len() as u64).saturating_mul(std::mem::size_of::<u64>() as u64)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VersionedWorld {
    base: Arc<WorldBase>,
    deltas: BTreeMap<usize, u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldError {
    UnknownWord,
}

impl Display for WorldError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownWord => write!(f, "world delta references a word outside the shared base"),
        }
    }
}

impl Error for WorldError {}

impl VersionedWorld {
    pub fn from_base(base: WorldBase) -> Self {
        Self {
            base: Arc::new(base),
            deltas: BTreeMap::new(),
        }
    }

    pub fn fork(&self, changes: &[(usize, u64)]) -> Result<Self, WorldError> {
        if changes
            .iter()
            .any(|(index, _)| *index >= self.base.words.len())
        {
            return Err(WorldError::UnknownWord);
        }
        let mut deltas = self.deltas.clone();
        for (index, value) in changes {
            deltas.insert(*index, *value);
        }
        Ok(Self {
            base: Arc::clone(&self.base),
            deltas,
        })
    }

    pub fn value(&self, index: usize) -> Option<u64> {
        self.deltas
            .get(&index)
            .copied()
            .or_else(|| self.base.words.get(index).copied())
    }

    pub fn checksum(&self) -> u64 {
        self.base
            .words
            .iter()
            .enumerate()
            .fold(0x51A8_0C0D_u64, |state, (index, value)| {
                let value = self.deltas.get(&index).unwrap_or(value);
                state.rotate_left(7) ^ mix64(*value ^ index as u64)
            })
    }

    pub fn delta_bytes(&self) -> u64 {
        (self.deltas.len() as u64).saturating_mul(2 * std::mem::size_of::<u64>() as u64)
    }

    pub fn base_bytes(&self) -> u64 {
        self.base.bytes()
    }

    pub fn shared_footprint_bytes(worlds: &[Self]) -> u64 {
        let mut bases = BTreeSet::new();
        let mut bytes = 0_u64;
        for world in worlds {
            let address = Arc::as_ptr(&world.base) as usize;
            if bases.insert(address) {
                bytes = bytes.saturating_add(world.base_bytes());
            }
            bytes = bytes.saturating_add(world.delta_bytes());
        }
        bytes
    }

    pub fn full_copy_footprint_bytes(world_count: usize, base_bytes: u64) -> u64 {
        (world_count as u64).saturating_mul(base_bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapitalDispatch {
    Interpreted,
    Compiled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapitalOutcome {
    pub answer: u64,
    pub dispatch: CapitalDispatch,
    pub primitive_cost_units: u64,
    pub verified: bool,
    pub capital_delta: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CognitiveCapitalRuntime {
    compile_after: u32,
    verified_runs: BTreeMap<String, u32>,
    compiled_families: BTreeSet<String>,
    total_cost_units: u64,
    verified_capital: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapitalError {
    InvalidFamily,
    InputTooLarge,
    ArithmeticOverflow,
}

impl Display for CapitalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFamily => write!(f, "task family must not be empty"),
            Self::InputTooLarge => write!(f, "task input exceeds the bounded interpreter limit"),
            Self::ArithmeticOverflow => write!(f, "task result overflows u64"),
        }
    }
}

impl Error for CapitalError {}

impl CognitiveCapitalRuntime {
    pub fn new(compile_after: u32) -> Option<Self> {
        (compile_after > 0).then_some(Self {
            compile_after,
            verified_runs: BTreeMap::new(),
            compiled_families: BTreeSet::new(),
            total_cost_units: 0,
            verified_capital: 0,
        })
    }

    pub fn solve(&self, family: &str, input: u64) -> Result<(Self, CapitalOutcome), CapitalError> {
        if family.is_empty() {
            return Err(CapitalError::InvalidFamily);
        }
        if input > 1_000_000 {
            return Err(CapitalError::InputTooLarge);
        }
        let compiled = self.compiled_families.contains(family);
        let answer = triangular_formula(input)?;
        let (dispatch, primitive_cost_units) = if compiled {
            (CapitalDispatch::Compiled, 1)
        } else {
            (CapitalDispatch::Interpreted, input.max(1))
        };
        let verified = if compiled {
            true
        } else {
            triangular_interpreted(input)? == answer
        };
        let mut next = self.clone();
        let mut capital_delta = 0;
        if verified && !compiled {
            let runs = next.verified_runs.entry(family.to_string()).or_default();
            *runs = runs.saturating_add(1);
            if *runs >= next.compile_after && next.compiled_families.insert(family.to_string()) {
                next.verified_capital = next.verified_capital.saturating_add(1);
                capital_delta = 1;
            }
        }
        next.total_cost_units = next.total_cost_units.saturating_add(primitive_cost_units);
        Ok((
            next,
            CapitalOutcome {
                answer,
                dispatch,
                primitive_cost_units,
                verified,
                capital_delta,
            },
        ))
    }

    pub fn verified_capital(&self) -> u64 {
        self.verified_capital
    }

    pub fn total_cost_units(&self) -> u64 {
        self.total_cost_units
    }

    pub fn cognitive_capital_efficiency(&self) -> f64 {
        if self.total_cost_units == 0 {
            0.0
        } else {
            self.verified_capital as f64 / self.total_cost_units as f64
        }
    }
}

fn triangular_interpreted(input: u64) -> Result<u64, CapitalError> {
    (1..=input).try_fold(0_u64, |sum, value| {
        sum.checked_add(value)
            .ok_or(CapitalError::ArithmeticOverflow)
    })
}

fn triangular_formula(input: u64) -> Result<u64, CapitalError> {
    let (left, right) = if input.is_multiple_of(2) {
        (input / 2, input.saturating_add(1))
    } else {
        (input, input.saturating_add(1) / 2)
    };
    left.checked_mul(right)
        .ok_or(CapitalError::ArithmeticOverflow)
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(error: u64, claims: &[&str]) -> FactorContract {
        FactorContract::new(
            "factor:route",
            1,
            error,
            claims.iter().map(|claim| (*claim).to_string()).collect(),
            0xC0DE,
        )
    }

    #[test]
    fn realization_rejects_the_cheapest_implementation_when_it_breaks_the_decision_contract() {
        let factor = ContractedFactor {
            required_contract: contract(10, &["same-route"]),
            certificate: DecisionCertificate {
                winner_lower_bound: 110,
                runner_up_upper_bound: 100,
            },
            implementations: vec![
                FactorImplementation {
                    kind: ImplementationKind::Exact,
                    contract: contract(0, &["same-route"]),
                    lifetime_cost_units: 100,
                },
                FactorImplementation {
                    kind: ImplementationKind::Approximate,
                    contract: contract(6, &["same-route"]),
                    lifetime_cost_units: 1,
                },
                FactorImplementation {
                    kind: ImplementationKind::Compiled,
                    contract: contract(0, &["same-route", "compiled-guard"]),
                    lifetime_cost_units: 10,
                },
            ],
        };

        assert_eq!(
            factor.realize().unwrap(),
            RealizationPlan {
                kind: ImplementationKind::Compiled,
                lifetime_cost_units: 10,
                preserved_certificate: true,
            }
        );
    }

    #[test]
    fn realization_refuses_to_trade_away_an_uncertified_decision() {
        let factor = ContractedFactor {
            required_contract: contract(10, &["same-route"]),
            certificate: DecisionCertificate {
                winner_lower_bound: 110,
                runner_up_upper_bound: 100,
            },
            implementations: vec![FactorImplementation {
                kind: ImplementationKind::Approximate,
                contract: contract(6, &["same-route"]),
                lifetime_cost_units: 1,
            }],
        };

        assert_eq!(
            factor.realize(),
            Err(ContractError::NoEligibleImplementation)
        );
    }

    #[test]
    fn realization_rejects_a_contract_without_semantic_identity() {
        let factor = ContractedFactor {
            required_contract: FactorContract::new("", 1, 0, BTreeSet::new(), 7),
            certificate: DecisionCertificate {
                winner_lower_bound: 2,
                runner_up_upper_bound: 0,
            },
            implementations: vec![FactorImplementation {
                kind: ImplementationKind::Exact,
                contract: FactorContract::new("", 1, 0, BTreeSet::new(), 7),
                lifetime_cost_units: 1,
            }],
        };

        assert_eq!(
            factor.realize(),
            Err(ContractError::NoEligibleImplementation)
        );
    }

    #[test]
    fn semantic_remorph_commits_only_a_verified_shadow_and_keeps_protected_facts() {
        let memory = SemanticVirtualMemory::synthetic(1_024, 64 * MIB).unwrap();
        assert_eq!(memory.protected_recall_fraction(), 1.0);
        assert_eq!(memory.recall_fraction(SemanticResolution::Exact), 0.0);

        let shadow = memory.stage_remorph(512 * MIB).unwrap();
        assert_eq!(shadow.verified_contracts, 64);
        let remorphed = memory.commit(shadow).unwrap();

        assert_eq!(remorphed.protected_recall_fraction(), 1.0);
        assert!(
            remorphed.recall_fraction(SemanticResolution::Exact)
                > memory.recall_fraction(SemanticResolution::Exact)
        );
        assert_eq!(memory.morphology().budget_bytes, 64 * MIB);
        assert_eq!(remorphed.morphology().budget_bytes, 512 * MIB);
    }

    #[test]
    fn semantic_memory_rejects_a_budget_below_the_summary_contract() {
        assert_eq!(
            SemanticVirtualMemory::synthetic(1_024, 64 * MIB - 1),
            Err(SemanticMemoryError::InsufficientSummaryBudget {
                required_bytes: 64 * MIB,
                available_bytes: 64 * MIB - 1,
            })
        );
    }

    #[test]
    fn semantic_memory_rejects_an_empty_corpus() {
        assert_eq!(
            SemanticVirtualMemory::synthetic(0, 64 * MIB),
            Err(SemanticMemoryError::EmptyCorpus)
        );
    }

    #[test]
    fn semantic_memory_rejects_a_shadow_from_an_older_epoch() {
        let memory = SemanticVirtualMemory::synthetic(1_024, 64 * MIB).unwrap();
        let shadow = memory.stage_remorph(128 * MIB).unwrap();
        let remorphed = memory.commit(shadow.clone()).unwrap();

        assert_eq!(
            remorphed.commit(shadow),
            Err(SemanticMemoryError::StaleShadow)
        );
    }

    #[test]
    fn versioned_worlds_share_the_base_and_preserve_each_branch_value() {
        let world = VersionedWorld::from_base(WorldBase::new((0..64).collect()));
        let left = world.fork(&[(7, 700)]).unwrap();
        let right = world.fork(&[(7, 900), (11, 1_100)]).unwrap();

        assert_eq!(world.value(7), Some(7));
        assert_eq!(left.value(7), Some(700));
        assert_eq!(right.value(7), Some(900));
        assert_eq!(right.value(11), Some(1_100));

        let branches = (0..1_000)
            .map(|index| world.fork(&[(index % 64, index as u64)]).unwrap())
            .collect::<Vec<_>>();
        assert!(
            VersionedWorld::shared_footprint_bytes(&branches)
                < VersionedWorld::full_copy_footprint_bytes(branches.len(), world.base_bytes())
        );
    }

    #[test]
    fn versioned_world_rejects_a_delta_outside_the_shared_base() {
        let world = VersionedWorld::from_base(WorldBase::new(vec![1, 2]));

        assert_eq!(world.fork(&[(2, 3)]), Err(WorldError::UnknownWord));
    }

    #[test]
    fn verified_repetition_builds_capital_and_reduces_the_familiar_task_cost() {
        let mut runtime = CognitiveCapitalRuntime::new(3).unwrap();
        let mut first_cost = 0;
        for call in 0..3 {
            let (next, outcome) = runtime.solve("triangular", 128).unwrap();
            if call == 0 {
                first_cost = outcome.primitive_cost_units;
            }
            assert_eq!(outcome.answer, 8_256);
            assert!(outcome.verified);
            runtime = next;
        }
        let (runtime, familiar) = runtime.solve("triangular", 128).unwrap();

        assert_eq!(familiar.dispatch, CapitalDispatch::Compiled);
        assert_eq!(runtime.verified_capital(), 1);
        assert!(familiar.primitive_cost_units < first_cost);
        assert!(runtime.cognitive_capital_efficiency() > 0.0);
    }

    #[test]
    fn cognitive_capital_bounds_interpreter_work() {
        assert_eq!(
            CognitiveCapitalRuntime::new(3)
                .unwrap()
                .solve("triangular", 1_000_001),
            Err(CapitalError::InputTooLarge)
        );
    }
}
