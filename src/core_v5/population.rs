//! Busca abductiva por população de mundos e seleção ativa de intervenções.

use std::collections::{BTreeMap, BTreeSet};

use super::cost::{CostVector, CostWeights};
use super::world::StructuralOperator;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WorldFamily {
    Direct,
    Reverse,
    Latent,
    Alternative,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldFitness {
    pub prediction: f32,
    pub generalization: f32,
    pub simplicity: f32,
    pub novelty: f32,
    pub falsifiability: f32,
}

impl WorldFitness {
    pub fn utility(self) -> f32 {
        0.30 * self.prediction
            + 0.25 * self.generalization
            + 0.20 * self.simplicity
            + 0.10 * self.novelty
            + 0.15 * self.falsifiability
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorldModel {
    pub id: String,
    pub parent: Option<String>,
    pub family: WorldFamily,
    pub assumptions: BTreeSet<String>,
    pub transformations: Vec<StructuralOperator>,
    pub fitness: WorldFitness,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PopulationOfWorlds {
    candidates: BTreeMap<String, WorldModel>,
    retired: Vec<WorldModel>,
}

impl PopulationOfWorlds {
    pub fn admit(&self, world: WorldModel) -> Self {
        let mut next = self.clone();
        next.candidates.insert(world.id.clone(), world);
        next
    }

    pub fn evolve(
        &self,
        parent_id: &str,
        id: impl Into<String>,
        family: WorldFamily,
        mutation: StructuralOperator,
        fitness: WorldFitness,
    ) -> Option<Self> {
        let parent = self.candidates.get(parent_id)?;
        let id = id.into();
        let child = WorldModel {
            id: id.clone(),
            parent: Some(parent.id.clone()),
            family,
            assumptions: parent.assumptions.clone(),
            transformations: parent
                .transformations
                .iter()
                .cloned()
                .chain([mutation])
                .collect(),
            fitness,
        };
        let mut next = self.clone();
        next.candidates.insert(id, child);
        Some(next)
    }

    pub fn best(&self) -> Option<&WorldModel> {
        self.candidates.values().max_by(|left, right| {
            left.fitness
                .utility()
                .total_cmp(&right.fitness.utility())
                .then_with(|| right.id.cmp(&left.id))
        })
    }

    /// Retém pelo menos o melhor candidato de cada família antes de preencher
    /// as vagas restantes por fitness. Isso evita colapso prematuro de hipótese.
    pub fn select_diverse(&self, limit: usize) -> Vec<WorldModel> {
        let mut best_by_family = BTreeMap::<WorldFamily, &WorldModel>::new();
        for world in self.candidates.values() {
            let replace = best_by_family
                .get(&world.family)
                .is_none_or(|current| world.fitness.utility() > current.fitness.utility());
            if replace {
                best_by_family.insert(world.family, world);
            }
        }
        let mut selected = best_by_family.into_values().cloned().collect::<Vec<_>>();
        selected.sort_by(|left, right| {
            right
                .fitness
                .utility()
                .total_cmp(&left.fitness.utility())
                .then_with(|| left.id.cmp(&right.id))
        });
        let selected_ids = selected
            .iter()
            .map(|world| world.id.clone())
            .collect::<BTreeSet<_>>();
        let mut remaining = self
            .candidates
            .values()
            .filter(|world| !selected_ids.contains(&world.id))
            .cloned()
            .collect::<Vec<_>>();
        remaining.sort_by(|left, right| {
            right
                .fitness
                .utility()
                .total_cmp(&left.fitness.utility())
                .then_with(|| left.id.cmp(&right.id))
        });
        selected.extend(remaining);
        selected.truncate(limit);
        selected
    }

    pub fn retire(&self, id: &str) -> Option<Self> {
        let mut next = self.clone();
        let retired = next.candidates.remove(id)?;
        next.retired.push(retired);
        Some(next)
    }

    pub fn retired_len(&self) -> usize {
        self.retired.len()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Intervention {
    pub id: String,
    /// Uma predição por hipótese. O planejador nunca recebe o mundo verdadeiro.
    pub predicted_outcomes: BTreeMap<String, bool>,
    pub cost: CostVector,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentChoice {
    pub id: String,
    pub information_value_per_cost: f64,
    pub cost: CostVector,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExperimentPlan {
    Run(ExperimentChoice),
    NoExperiment,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ActiveExperimentPlanner;

impl ActiveExperimentPlanner {
    pub fn choose(&self, interventions: &[Intervention]) -> ExperimentPlan {
        let weights = CostWeights::default();
        let Some(origin) = interventions
            .first()
            .map(|intervention| intervention.cost.origin)
        else {
            return ExperimentPlan::NoExperiment;
        };
        if interventions
            .iter()
            .any(|intervention| intervention.cost.origin != origin)
        {
            return ExperimentPlan::NoExperiment;
        }
        interventions
            .iter()
            .filter_map(|intervention| {
                let outcomes = intervention
                    .predicted_outcomes
                    .values()
                    .copied()
                    .collect::<BTreeSet<_>>();
                let information_value = u32::from(outcomes.len() > 1) as f64;
                let cost = intervention.cost.weighted_total(weights);
                (cost > 0.0 && information_value > 0.0).then(|| ExperimentChoice {
                    id: intervention.id.clone(),
                    information_value_per_cost: information_value / cost,
                    cost: intervention.cost,
                })
            })
            .max_by(|left, right| {
                left.information_value_per_cost
                    .total_cmp(&right.information_value_per_cost)
                    .then_with(|| right.id.cmp(&left.id))
            })
            .map_or(ExperimentPlan::NoExperiment, ExperimentPlan::Run)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fitness(value: f32) -> WorldFitness {
        WorldFitness {
            prediction: value,
            generalization: value,
            simplicity: value,
            novelty: value,
            falsifiability: value,
        }
    }

    fn world(id: &str, family: WorldFamily, value: f32) -> WorldModel {
        WorldModel {
            id: id.to_string(),
            parent: None,
            family,
            assumptions: BTreeSet::new(),
            transformations: Vec::new(),
            fitness: fitness(value),
        }
    }

    #[test]
    fn population_preserves_structural_diversity() {
        let population = PopulationOfWorlds::default()
            .admit(world("direct", WorldFamily::Direct, 0.9))
            .admit(world("reverse", WorldFamily::Reverse, 0.8))
            .admit(world("latent", WorldFamily::Latent, 0.7));

        let families = population
            .select_diverse(3)
            .into_iter()
            .map(|world| world.family)
            .collect::<BTreeSet<_>>();

        assert_eq!(families.len(), 3);
        assert_eq!(population.best().unwrap().id, "direct");
    }

    #[test]
    fn active_planner_selects_a_discriminative_intervention_without_oracle_label() {
        let common_cost = CostVector::declared(1, 4, 0, 0, 1);
        let plan = ActiveExperimentPlanner.choose(&[
            Intervention {
                id: "correlation-only".to_string(),
                predicted_outcomes: BTreeMap::from([
                    ("direct".to_string(), true),
                    ("reverse".to_string(), true),
                ]),
                cost: common_cost,
            },
            Intervention {
                id: "intervene-a".to_string(),
                predicted_outcomes: BTreeMap::from([
                    ("direct".to_string(), true),
                    ("reverse".to_string(), false),
                ]),
                cost: common_cost,
            },
        ]);

        assert!(
            matches!(plan, ExperimentPlan::Run(ExperimentChoice { id, .. }) if id == "intervene-a")
        );
    }

    #[test]
    fn active_planner_accepts_consistent_measured_costs_but_rejects_mixed_origins() {
        let measured = Intervention {
            id: "measured-intervention".to_string(),
            predicted_outcomes: BTreeMap::from([
                ("first".to_string(), true),
                ("second".to_string(), false),
            ]),
            cost: CostVector::measured(1, 1, 0, 0, 1, 1),
        };
        assert!(matches!(
            ActiveExperimentPlanner.choose(std::slice::from_ref(&measured)),
            ExperimentPlan::Run(_)
        ));
        let declared = Intervention {
            id: "declared-intervention".to_string(),
            predicted_outcomes: measured.predicted_outcomes.clone(),
            cost: CostVector::declared(1, 1, 0, 0, 1),
        };
        assert_eq!(
            ActiveExperimentPlanner.choose(&[measured, declared]),
            ExperimentPlan::NoExperiment
        );
    }
}
