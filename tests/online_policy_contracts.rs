use axon_uic::{
    AverageStrategy::{Full, Incremental},
    OnlineAveragePolicy,
};

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
