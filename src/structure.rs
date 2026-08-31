use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Exact execution cone: nodes both needed by the goal and affected by a delta.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionSlice {
    nodes: BTreeSet<String>,
}

impl ExecutionSlice {
    pub fn build<const N: usize>(
        goal: &str,
        changed: [&str; N],
        dependencies: &BTreeMap<String, Vec<String>>,
    ) -> Self {
        let needed = backward_cone(goal, dependencies);
        let impacted: BTreeSet<_> = changed
            .into_iter()
            .filter(|node| needed.contains(*node))
            .map(str::to_owned)
            .collect();
        Self {
            nodes: forward_cone(&impacted, &needed, dependencies),
        }
    }

    pub fn nodes(&self) -> &BTreeSet<String> {
        &self.nodes
    }
}

fn backward_cone(goal: &str, dependencies: &BTreeMap<String, Vec<String>>) -> BTreeSet<String> {
    let mut needed = BTreeSet::new();
    let mut work = VecDeque::from([goal.to_owned()]);
    while let Some(node) = work.pop_front() {
        if !needed.insert(node.clone()) {
            continue;
        }
        if let Some(inputs) = dependencies.get(&node) {
            work.extend(inputs.iter().cloned());
        }
    }
    needed
}

fn forward_cone(
    impacted: &BTreeSet<String>,
    needed: &BTreeSet<String>,
    dependencies: &BTreeMap<String, Vec<String>>,
) -> BTreeSet<String> {
    let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (node, inputs) in dependencies {
        for input in inputs {
            dependents.entry(input).or_default().push(node);
        }
    }
    let mut result = BTreeSet::new();
    let mut work: VecDeque<_> = impacted.iter().cloned().collect();
    while let Some(node) = work.pop_front() {
        if !needed.contains(&node) || !result.insert(node.clone()) {
            continue;
        }
        if let Some(outputs) = dependents.get(node.as_str()) {
            work.extend(outputs.iter().map(|output| (*output).to_owned()));
        }
    }
    result
}

/// Restricted exact LIFT certificate: a class of equal source values can share one representative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LiftCertificate {
    representative: i64,
    members: usize,
}

impl LiftCertificate {
    pub fn from_identical(values: &[i64]) -> Option<Self> {
        let (&representative, rest) = values.split_first()?;
        rest.iter()
            .all(|&value| value == representative)
            .then_some(Self {
                representative,
                members: values.len(),
            })
    }

    pub const fn members(self) -> usize {
        self.members
    }

    pub fn matches_exact_max(self, values: &[i64]) -> bool {
        values.iter().copied().max() == Some(self.representative) && values.len() == self.members
    }
}

/// Approximate abstraction must remain within its declared observable error bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AbstractionContract {
    error_bound: u64,
}

impl AbstractionContract {
    pub const fn new(error_bound: u64) -> Self {
        Self { error_bound }
    }

    pub fn preserves(self, concrete_transition: i64, abstract_transition: i64) -> bool {
        concrete_transition.abs_diff(abstract_transition) <= self.error_bound
    }
}
