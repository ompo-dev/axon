use axon_uic::{
    AverageStrategy::{Full, Incremental},
    OnlineAveragePolicy,
};

#[test]
fn empty_observations_cannot_overwrite_evidence_about_real_updates() {
    let mut policy = OnlineAveragePolicy::new("empty-is-not-a-workload-sample");
    for _ in 0..8 {
        assert!(policy.observe(4096, 16, Incremental, 20));
        assert!(policy.observe(4096, 16, Full, 200));
    }
    let before = policy.clone();
    for _ in 0..1000 {
        assert!(!policy.observe(4096, 0, Full, 1));
        assert!(!policy.observe(4096, 0, Incremental, 1));
    }
    assert_eq!(policy, before);
    assert_eq!(policy.choose(4096, 16), Incremental);
    assert_eq!(policy.choose(4096, 0), Incremental);
    assert_eq!(policy.explore(4096, 0), Incremental);
}

#[test]
fn empty_execution_advances_the_event_without_training_or_skipping_guards() {
    use axon_uic::{FoldSpec, LiveAverage, ReplaceDelta, SemanticArtifact};
    let semantic = SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap();
    let mut state = LiveAverage::materialize(42, vec![u64::MAX, 1], semantic).unwrap();
    let mut policy = OnlineAveragePolicy::new("empty-execution");
    let empty = ReplaceDelta::try_new(vec![]).unwrap();
    let frozen = policy.clone();
    for event in 1..=256 {
        let receipt = policy.execute_and_learn(&mut state, event, &empty).unwrap();
        assert_eq!(receipt.strategy, Incremental);
        assert_eq!(receipt.average.numerator(), u64::MAX as u128 + 1);
        assert!(receipt.elapsed_ns > 0);
        assert_eq!(state.version(), event);
        assert_eq!(policy, frozen);
    }
    let before = state.clone();
    assert!(policy.execute_and_learn(&mut state, 256, &empty).is_err());
    assert!(policy.execute_and_learn(&mut state, 258, &empty).is_err());
    assert_eq!(state, before);
    assert_eq!(policy, frozen);
}

#[test]
fn normalized_costs_transfer_across_work_amounts_inside_the_same_bucket() {
    let mut raw = OnlineAveragePolicy::new("raw-control");
    let mut normalized = OnlineAveragePolicy::normalized("normalized-control");
    // Both sources and all update counts below share the same cost bucket.
    for _ in 0..8 {
        for policy in [&mut raw, &mut normalized] {
            policy.observe(512, 4, Incremental, 40);
            policy.observe(512, 4, Full, 20);
        }
    }
    assert_eq!(raw.choose(512, 1), Full);
    assert_eq!(normalized.choose(512, 1), Incremental);
    assert_eq!(normalized.choose(512, 8), Full);
    assert_eq!(normalized.choose(256, 1), Full);
    let frozen = normalized.clone();
    assert!(!normalized.observe(512, 0, Full, 1));
    assert_eq!(normalized, frozen);
}

#[test]
fn normalized_policy_persistence_retains_mode_and_handles_extreme_costs() {
    let path = std::env::temp_dir().join(format!(
        "axon-normalized-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut policy = OnlineAveragePolicy::normalized("normalized-extremes");
    for _ in 0..8 {
        policy.observe(512, 4, Incremental, 40);
        policy.observe(512, 4, Full, 20);
        policy.observe(usize::MAX, usize::MAX, Incremental, u64::MAX);
        policy.observe(usize::MAX, usize::MAX, Full, 1);
    }
    assert_eq!(policy.choose(512, 1), Incremental);
    assert_eq!(policy.choose(usize::MAX, usize::MAX), Full);
    policy.save(&path).unwrap();
    let loaded = OnlineAveragePolicy::load(&path, "normalized-extremes").unwrap();
    assert_eq!(loaded, policy);
    assert_eq!(loaded.choose(512, 1), Incremental);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len() as usize,
        OnlineAveragePolicy::checkpoint_bytes()
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn measured_costs_change_decisions_and_follow_a_cost_reversal() {
    let mut policy = OnlineAveragePolicy::new("test-cpu/protocol1");
    assert_eq!(policy.choose(4096, 16), Full);
    for _ in 0..8 {
        policy.observe(4096, 16, Incremental, 20);
        policy.observe(4096, 16, Full, 200);
    }
    assert_eq!(policy.choose(6000, 24), Incremental);
    assert_eq!(policy.choose(4096, 4000), Full);
    for _ in 0..40 {
        policy.observe(4096, 16, Incremental, 500);
        policy.observe(4096, 16, Full, 10);
    }
    assert_eq!(policy.choose(4096, 16), Full);
    assert!(!policy.observe(0, 0, Full, 10));
    assert!(!policy.observe(10, 11, Full, 10));
    assert!(!policy.observe(10, 1, Full, 0));
}

#[test]
fn policy_checkpoint_roundtrip_detects_corruption_and_context_change() {
    let root = std::env::temp_dir().join(format!(
        "axon-policy-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut policy = OnlineAveragePolicy::new("cpu-a");
    policy.observe(4096, 1, Incremental, u64::MAX);
    policy.observe(4096, 1, Incremental, 1);
    policy.save(&root).unwrap();
    assert_eq!(
        std::fs::metadata(&root).unwrap().len() as usize,
        OnlineAveragePolicy::checkpoint_bytes()
    );
    assert_eq!(OnlineAveragePolicy::load(&root, "cpu-a").unwrap(), policy);
    assert!(OnlineAveragePolicy::load(&root, "cpu-b").is_err());
    assert!(policy.save(&root).is_err());
    let mut bytes = std::fs::read(&root).unwrap();
    assert_eq!(
        &bytes[..8],
        b"AXPOL001",
        "raw policy format stays compatible"
    );
    bytes[31] ^= 1;
    std::fs::write(&root, bytes).unwrap();
    assert!(OnlineAveragePolicy::load(&root, "cpu-a").is_err());
    std::fs::remove_file(root).unwrap();
}

#[test]
fn exploration_revisits_alternatives_and_adapts_with_only_chosen_feedback() {
    let mut policy = OnlineAveragePolicy::new("synthetic-cost-drift");
    let mut choices = [0; 2];
    for _ in 0..128 {
        let strategy = policy.explore(4096, 16);
        choices[usize::from(strategy == Full)] += 1;
        policy.observe(4096, 16, strategy, if strategy == Full { 200 } else { 20 });
    }
    assert_eq!(policy.choose(4096, 16), Incremental);
    assert!(choices[0] > choices[1] && choices[1] > 4);
    for _ in 0..256 {
        let strategy = policy.explore(4096, 16);
        policy.observe(4096, 16, strategy, if strategy == Full { 20 } else { 500 });
    }
    assert_eq!(policy.choose(4096, 16), Full);
    assert_eq!(policy.observations(), 384);
}

#[test]
fn real_event_learning_preserves_correctness_and_ignores_rejected_events() {
    use axon_uic::{FoldSpec, LiveAverage, Replace, ReplaceDelta, SemanticArtifact};
    let semantic = SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap();
    let mut state = LiveAverage::materialize(42, vec![1, 2, 3, 4], semantic).unwrap();
    let mut policy = OnlineAveragePolicy::new("actual-execution");
    for event in 1..=256 {
        let old = state.values()[0];
        let delta = ReplaceDelta::try_new(vec![Replace::new(0, old, event + 100)]).unwrap();
        let receipt = policy.execute_and_learn(&mut state, event, &delta).unwrap();
        assert_eq!(receipt.average.numerator(), event as u128 + 109);
        assert_eq!(receipt.average.denominator(), 4);
        assert!(receipt.elapsed_ns > 0);
        assert_eq!(policy.observations(), event);
    }
    let before = state.clone();
    let learning_before = policy.clone();
    let stale = ReplaceDelta::try_new(vec![Replace::new(0, 1, 999)]).unwrap();
    assert!(policy.execute_and_learn(&mut state, 257, &stale).is_err());
    assert_eq!(state, before);
    assert_eq!(policy, learning_before);
}
