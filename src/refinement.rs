use std::collections::BTreeSet;

/// Closed sound bounds. `try_new` rejects invalid input rather than widening it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Interval {
    lower: i64,
    upper: i64,
}

impl Interval {
    pub fn try_new(lower: i64, upper: i64) -> Result<Self, IntervalError> {
        if lower > upper {
            return Err(IntervalError::InvertedBounds);
        }
        Ok(Self { lower, upper })
    }

    pub fn lower(self) -> i64 {
        self.lower
    }

    pub fn upper(self) -> i64 {
        self.upper
    }

    pub fn contains(self, other: Self) -> bool {
        self.lower <= other.lower && other.upper <= self.upper
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntervalError {
    InvertedBounds,
}

/// A result-set whose permitted values can only contract with additional budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RefinementSet {
    bounds: Interval,
    budget: u64,
}

impl RefinementSet {
    pub fn new(bounds: Interval, budget: u64) -> Self {
        Self { bounds, budget }
    }

    pub fn refine(self, bounds: Interval, additional_budget: u64) -> Result<Self, RefinementError> {
        if !self.bounds.contains(bounds) {
            return Err(RefinementError::NotMonotone);
        }
        Ok(Self {
            bounds,
            budget: self.budget.saturating_add(additional_budget),
        })
    }

    pub fn is_subset_of(self, other: &Self) -> bool {
        other.bounds.contains(self.bounds)
    }

    pub fn budget(self) -> u64 {
        self.budget
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefinementError {
    NotMonotone,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ActionBound {
    action: String,
    utility: Interval,
}

/// A certificate exists only when one action's lower bound beats every rival's upper bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionCertificate {
    actions: Vec<ActionBound>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecisionError {
    DuplicateAction(String),
}

impl DecisionCertificate {
    pub fn try_from_utilities<const N: usize>(
        utilities: [(&str, Interval); N],
    ) -> Result<Self, DecisionError> {
        let mut names = BTreeSet::new();
        let mut actions = Vec::with_capacity(N);
        for (action, utility) in utilities {
            if !names.insert(action) {
                return Err(DecisionError::DuplicateAction(action.to_owned()));
            }
            actions.push(ActionBound {
                action: action.to_owned(),
                utility,
            });
        }
        Ok(Self { actions })
    }

    pub fn certified_action(&self) -> Option<&str> {
        self.actions.iter().find_map(|candidate| {
            self.actions
                .iter()
                .filter(|other| other.action != candidate.action)
                .all(|other| candidate.utility.lower() > other.utility.upper())
                .then_some(candidate.action.as_str())
        })
    }

    pub fn ambiguity(&self) -> i64 {
        let Some(best) = self
            .actions
            .iter()
            .max_by_key(|entry| entry.utility.lower())
        else {
            return 0;
        };
        self.actions
            .iter()
            .filter(|entry| entry.action != best.action)
            .map(|entry| entry.utility.upper().saturating_sub(best.utility.lower()))
            .max()
            .unwrap_or(0)
            .max(0)
    }
}

/// Physical cost remains a vector until a current hardware policy assigns prices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalCost {
    pub latency: u64,
    pub bytes_moved: u64,
    pub energy: u64,
}

impl PhysicalCost {
    pub const fn new(latency: u64, bytes_moved: u64, energy: u64) -> Self {
        Self {
            latency,
            bytes_moved,
            energy,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostPrices {
    latency: u64,
    bytes_moved: u64,
    energy: u64,
}

impl CostPrices {
    pub const fn new(latency: u64, bytes_moved: u64, energy: u64) -> Self {
        Self {
            latency,
            bytes_moved,
            energy,
        }
    }

    pub const fn unit() -> Self {
        Self::new(1, 1, 1)
    }

    fn price(self, cost: PhysicalCost) -> u64 {
        cost.latency
            .saturating_mul(self.latency)
            .saturating_add(cost.bytes_moved.saturating_mul(self.bytes_moved))
            .saturating_add(cost.energy.saturating_mul(self.energy))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Refinement {
    name: &'static str,
    expected_overlap_reduction: u64,
    cost: PhysicalCost,
}

impl Refinement {
    pub const fn new(
        name: &'static str,
        expected_overlap_reduction: u64,
        cost: PhysicalCost,
    ) -> Self {
        Self {
            name,
            expected_overlap_reduction,
            cost,
        }
    }

    pub const fn name(self) -> &'static str {
        self.name
    }
}

/// Select expected decision ambiguity reduction per currently priced physical cost.
pub fn select_refinement(choices: &[Refinement], prices: CostPrices) -> Option<&Refinement> {
    choices.iter().max_by(|left, right| {
        let left_cost = prices.price(left.cost).max(1) as u128;
        let right_cost = prices.price(right.cost).max(1) as u128;
        let left_score = (left.expected_overlap_reduction as u128).saturating_mul(right_cost);
        let right_score = (right.expected_overlap_reduction as u128).saturating_mul(left_cost);
        left_score
            .cmp(&right_score)
            .then_with(|| left.name.cmp(right.name).reverse())
    })
}
