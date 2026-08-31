use std::collections::{BTreeMap, BTreeSet};

const MAX_FRONTIER_STATES: usize = 100_000;

/// Stable behaviour required to survive a morphology swap.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct SemanticContract {
    semantic_id: String,
    revision: u64,
    digest: u64,
}

impl SemanticContract {
    pub fn new(semantic_id: impl Into<String>, revision: u64, digest: u64) -> Self {
        Self {
            semantic_id: semantic_id.into(),
            revision,
            digest,
        }
    }

    pub fn semantic_id(&self) -> &str {
        &self.semantic_id
    }
}

/// A resource region has protected base bytes and discrete task-morphology tiers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Region {
    name: String,
    protected_bytes: u64,
    tiers: Vec<(u64, i64)>,
}

impl Region {
    pub fn new<const N: usize>(
        name: impl Into<String>,
        protected_bytes: u64,
        tiers: [(u64, i64); N],
    ) -> Self {
        Self {
            name: name.into(),
            protected_bytes,
            tiers: tiers.into_iter().collect(),
        }
    }

    fn choices(&self) -> Result<Vec<(u64, i64)>, MorphologyError> {
        let mut choices = BTreeMap::from([(0_u64, 0_i64)]);
        for &(extra, utility) in &self.tiers {
            if utility < 0 {
                return Err(MorphologyError::NegativeUtility(self.name.clone()));
            }
            choices
                .entry(extra)
                .and_modify(|current| *current = (*current).max(utility))
                .or_insert(utility);
        }
        Ok(choices.into_iter().collect())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MorphologyError {
    ProtectedBudgetExceeded,
    NegativeUtility(String),
    DuplicateRegion(String),
    SizeOverflow(String),
    UnknownContract(String),
    SearchSpaceExceeded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AllocationState {
    bytes: u64,
    utility: i64,
    regions: BTreeMap<String, u64>,
}

/// Base morphology plus a budget-selected task delta. Construction does not mutate input regions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Morphology {
    budget: u64,
    utility: i64,
    regions: BTreeMap<String, u64>,
    contracts: BTreeMap<String, SemanticContract>,
}

impl Morphology {
    pub fn allocate<const N: usize>(
        budget: u64,
        regions: [Region; N],
    ) -> Result<Self, MorphologyError> {
        let mut names = BTreeSet::new();
        let mut states = vec![AllocationState {
            bytes: 0,
            utility: 0,
            regions: BTreeMap::new(),
        }];

        for region in regions {
            if !names.insert(region.name.clone()) {
                return Err(MorphologyError::DuplicateRegion(region.name));
            }
            let choices = region.choices()?;
            let mut next = Vec::new();
            for state in &states {
                for &(extra, utility) in &choices {
                    let region_bytes = region
                        .protected_bytes
                        .checked_add(extra)
                        .ok_or_else(|| MorphologyError::SizeOverflow(region.name.clone()))?;
                    let bytes = state
                        .bytes
                        .checked_add(region_bytes)
                        .ok_or_else(|| MorphologyError::SizeOverflow(region.name.clone()))?;
                    if bytes > budget {
                        continue;
                    }
                    let mut allocations = state.regions.clone();
                    allocations.insert(region.name.clone(), region_bytes);
                    next.push(AllocationState {
                        bytes,
                        utility: state.utility.saturating_add(utility),
                        regions: allocations,
                    });
                    if next.len() > MAX_FRONTIER_STATES {
                        return Err(MorphologyError::SearchSpaceExceeded);
                    }
                }
            }
            if next.is_empty() {
                return Err(MorphologyError::ProtectedBudgetExceeded);
            }
            states = pareto_prune(next)?;
        }

        let best = states
            .into_iter()
            .max_by_key(|state| (state.utility, std::cmp::Reverse(state.bytes)))
            .expect("a nonempty region list leaves at least one state");
        Ok(Self {
            budget,
            utility: best.utility,
            regions: best.regions,
            contracts: BTreeMap::new(),
        })
    }

    pub fn with_contract(mut self, contract: SemanticContract) -> Result<Self, MorphologyError> {
        if !self.regions.contains_key(contract.semantic_id()) {
            return Err(MorphologyError::UnknownContract(
                contract.semantic_id().to_owned(),
            ));
        }
        self.contracts
            .insert(contract.semantic_id().to_owned(), contract);
        Ok(self)
    }

    pub fn bytes_for(&self, region: &str) -> Option<u64> {
        self.regions.get(region).copied()
    }

    pub fn utility(&self) -> i64 {
        self.utility
    }

    fn preserves_migrated_contracts(&self, other: &Self) -> bool {
        if self.contracts != other.contracts {
            return false;
        }
        self.regions.iter().all(|(name, current_bytes)| {
            let candidate_bytes = other.regions.get(name);
            if candidate_bytes == Some(current_bytes) {
                return true;
            }
            self.contracts.contains_key(name)
        }) && other
            .regions
            .keys()
            .all(|name| self.regions.contains_key(name))
    }
}

fn pareto_prune(mut states: Vec<AllocationState>) -> Result<Vec<AllocationState>, MorphologyError> {
    states.sort_by_key(|state| (state.bytes, std::cmp::Reverse(state.utility)));
    let mut best_utility = i64::MIN;
    states.retain(|state| {
        let useful = state.utility > best_utility;
        best_utility = best_utility.max(state.utility);
        useful
    });
    if states.len() > MAX_FRONTIER_STATES {
        return Err(MorphologyError::SearchSpaceExceeded);
    }
    Ok(states)
}

/// Migration is permitted only after dwell time and amortized benefit clear the hysteresis margin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RemorphPolicy {
    expected_queries: u64,
    hysteresis: u64,
    minimum_dwell: u64,
}

impl RemorphPolicy {
    pub const fn new(expected_queries: u64, hysteresis: u64, minimum_dwell: u64) -> Self {
        Self {
            expected_queries,
            hysteresis,
            minimum_dwell,
        }
    }

    pub fn accepts(
        self,
        current: &Morphology,
        candidate: &Morphology,
        migration_cost: u64,
        saved_per_query: u64,
        dwell_queries: u64,
    ) -> bool {
        if dwell_queries < self.minimum_dwell
            || !current.preserves_migrated_contracts(candidate)
            || candidate.utility() < current.utility()
        {
            return false;
        }
        self.expected_queries.saturating_mul(saved_per_query)
            > migration_cost.saturating_add(self.hysteresis)
    }
}
