use axon_uic::{
    CandidateKind, LearningProblem, ProblemFeatures, StructurePrior, default_learning_curriculum,
    train_structure_prior,
};

#[test]
fn learning_moves_repeated_structure_to_the_front() {
    let mut prior = StructurePrior::new();
    let variance = LearningProblem::new(
        "variance/stable/replace",
        ProblemFeatures::variance_replace(),
        CandidateKind::VarianceWelford,
    );

    assert!(prior.attempts_for(variance) > 1);
    for _ in 0..4 {
        prior.learn(variance);
    }

    assert_eq!(
        prior.rank(variance.features())[0],
        CandidateKind::VarianceWelford
    );
    assert_eq!(prior.attempts_for(variance), 1);
}

#[test]
fn learn_bench_has_a_measurable_search_reduction_contract() {
    let mut prior = StructurePrior::new();
    let curriculum = default_learning_curriculum();
    let mut train = Vec::new();
    for _ in 0..8 {
        train.extend(curriculum.iter().copied());
    }
    let summary = train_structure_prior(&mut prior, &train, &curriculum);

    assert!(summary.attempts_before() > summary.attempts_after());
    assert!(summary.search_reduction_basis_points() >= 5_000);
}
