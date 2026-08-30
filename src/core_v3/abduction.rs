use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalModel {
    direct_causes: BTreeSet<(String, String)>,
    latent_causes: BTreeMap<String, BTreeSet<String>>,
    relations: BTreeSet<(String, String)>,
    operators: BTreeSet<String>,
    split_concepts: BTreeSet<String>,
    fused_concepts: BTreeSet<(String, String)>,
    pub dimension: usize,
}

impl CausalModel {
    pub fn with_direct_cause(
        cause: impl Into<String>,
        effect: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self {
            direct_causes: BTreeSet::from([(cause.into(), effect.into())]),
            latent_causes: BTreeMap::new(),
            relations: BTreeSet::new(),
            operators: BTreeSet::new(),
            split_concepts: BTreeSet::new(),
            fused_concepts: BTreeSet::new(),
            dimension: dimension.max(1),
        }
    }

    pub fn has_direct_cause(&self, cause: &str, effect: &str) -> bool {
        self.direct_causes
            .contains(&(cause.to_string(), effect.to_string()))
    }

    pub fn has_latent_cause_for(&self, left: &str, right: &str) -> bool {
        self.latent_causes
            .values()
            .any(|effects| effects.contains(left) && effects.contains(right))
    }

    pub fn counterfactual_loss(&self, contradiction: &Contradiction) -> u32 {
        calculate_counterfactual_loss(self, contradiction)
    }

    fn shared_observation_support(&self, source: &str, target: &str, observations: u32) -> u32 {
        if self.has_latent_cause_for(source, target)
            || self.has_path(source, target)
            || self.has_path(target, source)
        {
            observations
        } else {
            0
        }
    }

    fn context_key(&self) -> String {
        format!("{self:?}")
    }

    fn without_direct_cause(&self, cause: &str, effect: &str) -> Self {
        let mut next = self.clone();
        next.direct_causes
            .remove(&(cause.to_string(), effect.to_string()));
        next
    }

    fn with_added_cause(&self, cause: impl Into<String>, effect: impl Into<String>) -> Self {
        let mut next = self.clone();
        next.direct_causes.insert((cause.into(), effect.into()));
        next
    }

    fn with_latent_cause(&self, latent: String, effects: [&str; 2]) -> Self {
        let mut next = self.clone();
        next.latent_causes
            .insert(latent, effects.into_iter().map(str::to_string).collect());
        next
    }

    fn has_path(&self, source: &str, target: &str) -> bool {
        let mut queue = VecDeque::from([source.to_string()]);
        let mut visited = BTreeSet::new();
        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.clone()) {
                continue;
            }
            for (cause, effect) in &self.direct_causes {
                if cause == &current {
                    if effect == target {
                        return true;
                    }
                    queue.push_back(effect.clone());
                }
            }
        }
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Counterfactual {
    pub intervention: String,
    pub target: String,
    pub expected_effect: bool,
}

impl Counterfactual {
    pub fn expect(
        intervention: impl Into<String>,
        target: impl Into<String>,
        expected_effect: bool,
    ) -> Self {
        Self {
            intervention: intervention.into(),
            target: target.into(),
            expected_effect,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Contradiction {
    pub source: String,
    pub target: String,
    pub shared_observations: u32,
    pub counterfactuals: Vec<Counterfactual>,
}

impl Contradiction {
    pub fn between(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            shared_observations: 0,
            counterfactuals: Vec::new(),
        }
    }

    pub fn with_shared_observations(mut self, shared_observations: u32) -> Self {
        self.shared_observations = shared_observations;
        self
    }

    pub fn with_counterfactual(mut self, counterfactual: Counterfactual) -> Self {
        self.counterfactuals.push(counterfactual);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ReframeKind {
    RemovePremise,
    ReverseCausality,
    IntroduceMediator,
    IntroduceLatentCause,
    FuseConcepts,
    SplitConcept,
    CreateRelation,
    CreateOperator,
    ExpandDimension,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReframeHypothesis {
    pub kind: ReframeKind,
    pub model: CausalModel,
    pub score: f32,
    pub counterfactual_loss: u32,
}

/// Independent candidate produced before cross-island ranking, avoiding premature convergence.
#[derive(Clone, Debug, PartialEq)]
pub struct HypothesisIsland {
    pub id: u32,
    pub proposal: ReframeHypothesis,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NegativeArchive {
    failed: BTreeSet<(ReframeKind, String, String, String)>,
}

impl NegativeArchive {
    pub fn record(
        &self,
        model: &CausalModel,
        kind: ReframeKind,
        source: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        let mut failed = self.failed.clone();
        failed.insert((kind, source.into(), target.into(), model.context_key()));
        Self { failed }
    }

    fn contains(&self, model: &CausalModel, kind: ReframeKind, source: &str, target: &str) -> bool {
        self.failed.contains(&(
            kind,
            source.to_string(),
            target.to_string(),
            model.context_key(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct AbductiveEngine {
    dimension_growth: usize,
}

impl Default for AbductiveEngine {
    fn default() -> Self {
        Self {
            dimension_growth: 2,
        }
    }
}

impl AbductiveEngine {
    /// Produces isolated structural alternatives; the source model remains unchanged.
    pub fn reframe(
        &self,
        model: &CausalModel,
        contradiction: &Contradiction,
        negative_archive: &NegativeArchive,
    ) -> Vec<HypothesisIsland> {
        let source = contradiction.source.as_str();
        let target = contradiction.target.as_str();
        let removed = model.without_direct_cause(source, target);
        let mediator = format!("mediator:{source}:{target}");
        let latent = format!("latent:{source}:{target}");
        let candidates = vec![
            (ReframeKind::RemovePremise, removed.clone(), 0.05),
            (
                ReframeKind::ReverseCausality,
                removed.with_added_cause(target, source),
                0.10,
            ),
            (
                ReframeKind::IntroduceMediator,
                removed
                    .with_added_cause(source, mediator.clone())
                    .with_added_cause(mediator, target),
                0.30,
            ),
            (
                ReframeKind::IntroduceLatentCause,
                removed.with_latent_cause(latent, [source, target]),
                0.30,
            ),
            (
                ReframeKind::FuseConcepts,
                fuse(&removed, source, target),
                0.20,
            ),
            (ReframeKind::SplitConcept, split(&removed, source), 0.20),
            (
                ReframeKind::CreateRelation,
                relation(&removed, source, target),
                0.20,
            ),
            (
                ReframeKind::CreateOperator,
                operator(&removed, source, target),
                0.25,
            ),
            (
                ReframeKind::ExpandDimension,
                expand_dimension(&removed, self.dimension_growth),
                0.15,
            ),
        ];

        candidates
            .into_iter()
            .filter(|(kind, _, _)| !negative_archive.contains(model, *kind, source, target))
            .enumerate()
            .map(|(id, (kind, candidate, complexity))| {
                let counterfactual_loss = candidate.counterfactual_loss(contradiction);
                let observational_support = candidate.shared_observation_support(
                    source,
                    target,
                    contradiction.shared_observations,
                );
                HypothesisIsland {
                    id: id as u32,
                    proposal: ReframeHypothesis {
                        kind,
                        model: candidate,
                        score: observational_support as f32
                            - counterfactual_loss as f32 * 10.0
                            - complexity,
                        counterfactual_loss,
                    },
                }
            })
            .collect()
    }

    pub fn best<'a>(&self, islands: &'a [HypothesisIsland]) -> Option<&'a HypothesisIsland> {
        islands
            .iter()
            .max_by(|left, right| left.proposal.score.total_cmp(&right.proposal.score))
    }
}

fn calculate_counterfactual_loss(model: &CausalModel, contradiction: &Contradiction) -> u32 {
    contradiction
        .counterfactuals
        .iter()
        .filter(|probe| model.has_path(&probe.intervention, &probe.target) != probe.expected_effect)
        .count() as u32
}

fn fuse(model: &CausalModel, source: &str, target: &str) -> CausalModel {
    let mut next = model.clone();
    next.fused_concepts
        .insert((source.to_string(), target.to_string()));
    next
}

fn split(model: &CausalModel, source: &str) -> CausalModel {
    let mut next = model.clone();
    next.split_concepts.insert(source.to_string());
    next
}

fn relation(model: &CausalModel, source: &str, target: &str) -> CausalModel {
    let mut next = model.clone();
    next.relations
        .insert((source.to_string(), target.to_string()));
    next
}

fn operator(model: &CausalModel, source: &str, target: &str) -> CausalModel {
    let mut next = model.clone();
    next.operators.insert(format!("operator:{source}:{target}"));
    next
}

fn expand_dimension(model: &CausalModel, factor: usize) -> CausalModel {
    let mut next = model.clone();
    next.dimension = next.dimension.saturating_mul(factor).max(1);
    next
}
