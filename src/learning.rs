use std::cmp::Ordering;

const FEATURE_COUNT: usize = 8;
const LEARNING_RATE: i32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateKind {
    FullRecompute,
    GroupDelta,
    SumCountAverage,
    VarianceMoments,
    VarianceWelford,
    OrderedIndex,
}

impl CandidateKind {
    pub const ALL: [Self; 6] = [
        Self::FullRecompute,
        Self::GroupDelta,
        Self::SumCountAverage,
        Self::VarianceMoments,
        Self::VarianceWelford,
        Self::OrderedIndex,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullRecompute => "full",
            Self::GroupDelta => "group_delta",
            Self::SumCountAverage => "sum_count_average",
            Self::VarianceMoments => "variance_moments",
            Self::VarianceWelford => "variance_welford",
            Self::OrderedIndex => "ordered_index",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::FullRecompute => 0,
            Self::GroupDelta => 1,
            Self::SumCountAverage => 2,
            Self::VarianceMoments => 3,
            Self::VarianceWelford => 4,
            Self::OrderedIndex => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProblemFeatures {
    fold: bool,
    invertible: bool,
    shape_preserving_replace: bool,
    nested_aggregate: bool,
    requires_order: bool,
    numeric_exact: bool,
    numeric_stability_sensitive: bool,
}

impl ProblemFeatures {
    pub const fn sum_replace() -> Self {
        Self {
            fold: true,
            invertible: true,
            shape_preserving_replace: true,
            nested_aggregate: false,
            requires_order: false,
            numeric_exact: true,
            numeric_stability_sensitive: false,
        }
    }

    pub const fn average_replace() -> Self {
        Self {
            fold: true,
            invertible: true,
            shape_preserving_replace: true,
            nested_aggregate: true,
            requires_order: false,
            numeric_exact: true,
            numeric_stability_sensitive: false,
        }
    }

    pub const fn variance_replace() -> Self {
        Self {
            fold: true,
            invertible: true,
            shape_preserving_replace: true,
            nested_aggregate: true,
            requires_order: false,
            numeric_exact: false,
            numeric_stability_sensitive: true,
        }
    }

    pub const fn min_replace() -> Self {
        Self {
            fold: true,
            invertible: false,
            shape_preserving_replace: true,
            nested_aggregate: false,
            requires_order: true,
            numeric_exact: true,
            numeric_stability_sensitive: false,
        }
    }

    const fn vector(self) -> [i32; FEATURE_COUNT] {
        [
            1,
            self.fold as i32,
            self.invertible as i32,
            self.shape_preserving_replace as i32,
            self.nested_aggregate as i32,
            self.requires_order as i32,
            self.numeric_stability_sensitive as i32,
            self.numeric_exact as i32,
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LearningProblem {
    name: &'static str,
    features: ProblemFeatures,
    accepted: CandidateKind,
}

impl LearningProblem {
    pub const fn new(
        name: &'static str,
        features: ProblemFeatures,
        accepted: CandidateKind,
    ) -> Self {
        Self {
            name,
            features,
            accepted,
        }
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const fn features(self) -> ProblemFeatures {
        self.features
    }

    pub const fn accepted(self) -> CandidateKind {
        self.accepted
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LearningTrial {
    problem: LearningProblem,
    attempts: usize,
}

impl LearningTrial {
    pub const fn problem(self) -> LearningProblem {
        self.problem
    }

    pub const fn attempts(self) -> usize {
        self.attempts
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LearningSummary {
    tasks: usize,
    attempts_before: usize,
    attempts_after: usize,
}

impl LearningSummary {
    pub const fn tasks(self) -> usize {
        self.tasks
    }

    pub const fn attempts_before(self) -> usize {
        self.attempts_before
    }

    pub const fn attempts_after(self) -> usize {
        self.attempts_after
    }

    pub fn search_reduction_basis_points(self) -> i64 {
        if self.attempts_before == 0 {
            return 0;
        }
        let saved = self.attempts_before as i128 - self.attempts_after as i128;
        (saved * 10_000 / self.attempts_before as i128).clamp(i64::MIN as i128, i64::MAX as i128)
            as i64
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructurePrior {
    weights: [[i32; FEATURE_COUNT]; 6],
}

impl StructurePrior {
    pub const fn new() -> Self {
        Self {
            weights: [[0; FEATURE_COUNT]; 6],
        }
    }

    pub fn rank(&self, features: ProblemFeatures) -> Vec<CandidateKind> {
        let mut candidates = CandidateKind::ALL;
        candidates.sort_by(|left, right| {
            self.score(*right, features)
                .cmp(&self.score(*left, features))
                .then_with(|| tie_breaker(*left, *right))
        });
        candidates.to_vec()
    }

    pub fn attempts_for(&self, problem: LearningProblem) -> usize {
        self.rank(problem.features())
            .into_iter()
            .position(|candidate| candidate == problem.accepted())
            .map(|index| index + 1)
            .expect("candidate set contains every synthetic answer")
    }

    pub fn learn(&mut self, problem: LearningProblem) -> LearningTrial {
        let ranking = self.rank(problem.features());
        let attempts = ranking
            .iter()
            .position(|candidate| *candidate == problem.accepted())
            .map(|index| index + 1)
            .expect("candidate set contains every synthetic answer");
        let features = problem.features().vector();
        if attempts > 1 {
            reinforce(&mut self.weights[problem.accepted().index()], features);
            for rejected in ranking.into_iter().take(attempts - 1) {
                penalize(&mut self.weights[rejected.index()], features);
            }
        }
        LearningTrial { problem, attempts }
    }

    fn score(&self, candidate: CandidateKind, features: ProblemFeatures) -> i64 {
        self.weights[candidate.index()]
            .iter()
            .zip(features.vector())
            .map(|(weight, feature)| i64::from(*weight) * i64::from(feature))
            .sum()
    }
}

impl Default for StructurePrior {
    fn default() -> Self {
        Self::new()
    }
}

pub fn train_structure_prior(
    prior: &mut StructurePrior,
    train: &[LearningProblem],
    evaluation: &[LearningProblem],
) -> LearningSummary {
    let attempts_before = evaluation
        .iter()
        .copied()
        .map(|problem| prior.attempts_for(problem))
        .sum();
    for problem in train.iter().copied() {
        prior.learn(problem);
    }
    let attempts_after = evaluation
        .iter()
        .copied()
        .map(|problem| prior.attempts_for(problem))
        .sum();
    LearningSummary {
        tasks: evaluation.len(),
        attempts_before,
        attempts_after,
    }
}

pub fn default_learning_curriculum() -> Vec<LearningProblem> {
    vec![
        LearningProblem::new(
            "sum/modular-u64/replace",
            ProblemFeatures::sum_replace(),
            CandidateKind::GroupDelta,
        ),
        LearningProblem::new(
            "avg/exact-u64/replace",
            ProblemFeatures::average_replace(),
            CandidateKind::SumCountAverage,
        ),
        LearningProblem::new(
            "variance/stable/replace",
            ProblemFeatures::variance_replace(),
            CandidateKind::VarianceWelford,
        ),
        LearningProblem::new(
            "min/u64/replace",
            ProblemFeatures::min_replace(),
            CandidateKind::OrderedIndex,
        ),
    ]
}

fn reinforce(weights: &mut [i32; FEATURE_COUNT], features: [i32; FEATURE_COUNT]) {
    for (weight, feature) in weights.iter_mut().zip(features) {
        *weight = weight.saturating_add(feature.saturating_mul(LEARNING_RATE));
    }
}

fn penalize(weights: &mut [i32; FEATURE_COUNT], features: [i32; FEATURE_COUNT]) {
    for (weight, feature) in weights.iter_mut().zip(features) {
        *weight = weight.saturating_sub(feature.saturating_mul(LEARNING_RATE));
    }
}

fn tie_breaker(left: CandidateKind, right: CandidateKind) -> Ordering {
    left.index().cmp(&right.index())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regressions_are_reported_as_negative_improvements() {
        let summary = LearningSummary {
            tasks: 1,
            attempts_before: 1,
            attempts_after: 3,
        };
        assert_eq!(summary.search_reduction_basis_points(), -20_000);
    }

    #[test]
    fn correct_ranking_does_not_accumulate_unbounded_reinforcement() {
        let mut prior = StructurePrior::new();
        let problem = default_learning_curriculum()[0];
        prior.learn(problem);
        let learned = prior.clone();
        for _ in 0..10000 {
            assert_eq!(prior.learn(problem).attempts(), 1);
        }
        assert_eq!(prior, learned);
    }

    #[test]
    fn exact_numeric_contract_is_an_observable_feature() {
        let exact = ProblemFeatures::average_replace();
        let mut approximate = exact;
        approximate.numeric_exact = false;
        assert_ne!(exact.vector(), approximate.vector());
    }

    #[test]
    fn saturated_weights_keep_scores_ordered_without_overflow() {
        let mut prior = StructurePrior::new();
        prior.weights[0] = [i32::MAX; FEATURE_COUNT];
        prior.weights[1] = [i32::MAX - 1; FEATURE_COUNT];
        assert_eq!(
            prior.rank(ProblemFeatures::average_replace())[0],
            CandidateKind::FullRecompute
        );
    }

    #[test]
    fn blank_prior_starts_with_deterministic_search_order() {
        let prior = StructurePrior::new();
        let problem = LearningProblem::new(
            "variance/stable/replace",
            ProblemFeatures::variance_replace(),
            CandidateKind::VarianceWelford,
        );

        assert_eq!(prior.attempts_for(problem), 5);
    }

    #[test]
    fn experience_reduces_candidate_search_without_changing_the_answer() {
        let mut prior = StructurePrior::new();
        let curriculum = default_learning_curriculum();

        let before: usize = curriculum
            .iter()
            .copied()
            .map(|problem| prior.attempts_for(problem))
            .sum();
        for _ in 0..6 {
            for problem in curriculum.iter().copied() {
                prior.learn(problem);
            }
        }
        let after: usize = curriculum
            .iter()
            .copied()
            .map(|problem| prior.attempts_for(problem))
            .sum();

        assert!(after < before);
        for problem in curriculum {
            assert_eq!(prior.rank(problem.features())[0], problem.accepted());
        }
    }

    #[test]
    fn training_summary_reports_basis_point_reduction() {
        let mut prior = StructurePrior::new();
        let curriculum = default_learning_curriculum();
        let summary = train_structure_prior(&mut prior, &curriculum, &curriculum);

        assert_eq!(summary.tasks(), curriculum.len());
        assert!(summary.attempts_after() < summary.attempts_before());
        assert!(summary.search_reduction_basis_points() > 0);
    }
}
