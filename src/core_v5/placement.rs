//! Plasticidade de localização lógica: coativação reduz custo de comunicação.

use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocationPlasticity {
    locations: BTreeMap<String, i32>,
    coactivations: BTreeMap<(String, String), u32>,
    relocate_after: u32,
}

impl LocationPlasticity {
    pub fn new(relocate_after: u32) -> Self {
        Self {
            locations: BTreeMap::new(),
            coactivations: BTreeMap::new(),
            relocate_after: relocate_after.max(1),
        }
    }

    pub fn place(mut self, concept: impl Into<String>, region: i32) -> Self {
        self.locations.insert(concept.into(), region);
        self
    }

    pub fn route_cost(&self, left: &str, right: &str) -> Option<u32> {
        Some(
            self.locations
                .get(left)?
                .abs_diff(*self.locations.get(right)?),
        )
    }

    pub fn observe_joint_use(&self, left: &str, right: &str) -> Self {
        let (first, second) = ordered_pair(left, right);
        let mut next = self.clone();
        let count = next
            .coactivations
            .entry((first.clone(), second.clone()))
            .or_insert(0);
        *count = count.saturating_add(1);
        if *count >= next.relocate_after
            && let Some(region) = next.locations.get(&first).copied()
        {
            next.locations.insert(second, region);
        }
        next
    }
}

impl Default for LocationPlasticity {
    fn default() -> Self {
        Self::new(3)
    }
}

fn ordered_pair(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_joint_use_reduces_logical_route_distance() {
        let initial = LocationPlasticity::new(2).place("a", 1).place("b", 9);
        let learned = initial
            .observe_joint_use("a", "b")
            .observe_joint_use("a", "b");

        assert_eq!(initial.route_cost("a", "b"), Some(8));
        assert_eq!(learned.route_cost("a", "b"), Some(0));
    }
}
