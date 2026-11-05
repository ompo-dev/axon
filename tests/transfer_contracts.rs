use axon_uic::{
    Authority, Effect, EpisodeStore,
    behavior::*,
    behavior_lab::{CachedOracle, Rng, audit_machine},
    transfer::*,
    transfer_lab::*,
};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
fn sources() -> Vec<BehaviorModel> {
    [Family::Grid, Family::Protocol, Family::Pipeline]
        .into_iter()
        .map(|family| {
            let extent = if family == Family::Protocol { 3 } else { 2 };
            let env = Environment::new(family, extent, 0);
            let mut oracle = CachedOracle::new(env.clone(), BehaviorBudget::default());
            let model =
                discover(4, &format!("test/source/{}", family.name()), &mut oracle).unwrap();
            assert!(
                audit_machine(&model, 4, 0, |s, a| env.transition(s, a))
                    .unwrap()
                    .counterexample
                    .is_none()
            );
            model
        })
        .collect()
}
fn target(family: Family, seed: u64) -> Rewired {
    let extent = if family == Family::Protocol { 3 } else { 2 };
    let mut rng = Rng::new(seed);
    let env = Environment::new(family, extent, 0);
    let initial = rng.pick(env.states());
    Rewired::random(Environment::new(family, extent, initial), &mut rng)
}
fn assert_exact(model: &BehaviorModel, env: &Rewired) {
    let audit = audit_machine(model, 4, env.environment.initial(), |s, a| {
        env.transition(s, a)
    })
    .unwrap();
    assert!(audit.counterexample.is_none(), "{:?}", audit.counterexample);
}
#[test]
fn transfers_across_three_interfaces_without_given_wiring_or_initial_state() {
    let source = sources();
    let refs = source.iter().collect::<Vec<_>>();
    for family in [Family::Grid, Family::Protocol, Family::Pipeline] {
        for seed in 100..108 {
            let env = target(family, seed);
            let mut oracle = CachedOracle::new(env.clone(), BehaviorBudget::default());
            let mut learner =
                TransferLearner::new(&refs, 4, "target", TransferBudget::default()).unwrap();
            let result = learner.learn(&mut oracle).unwrap();
            assert_exact(&result.model, &env);
            assert!(result.stats.queries <= 128);
            assert!(result.stats.initial_hypotheses > result.stats.surviving_hypotheses);
            assert!(!result.source_fingerprints.is_empty());
        }
    }
}
#[test]
fn no_model_is_promoted_before_observations_or_with_partial_output_bindings() {
    let source = sources();
    let mut learner = TransferLearner::new(
        &source.iter().collect::<Vec<_>>(),
        4,
        "target",
        TransferBudget::default(),
    )
    .unwrap();
    assert!(learner.compile().unwrap().is_none());
    learner.observe(&[0], &[65535]).unwrap();
    assert!(learner.compile().unwrap().is_none());
}
#[test]
fn contradictions_suspend_instead_of_changing_the_prior() {
    let source = sources();
    let mut learner = TransferLearner::new(
        &source.iter().collect::<Vec<_>>(),
        4,
        "target",
        TransferBudget::default(),
    )
    .unwrap();
    learner.observe(&[0], &[17]).unwrap();
    assert!(matches!(
        learner.observe(&[0, 1], &[18, 99]),
        Err(TransferError::Refuted)
    ));
    assert!(learner.suspended());
    assert!(matches!(
        learner.choose_probe(),
        Err(TransferError::Suspended)
    ));
}
#[test]
fn malformed_replies_suspend_and_duplicate_probes_are_not_counted_twice() {
    let source = sources();
    let mut learner = TransferLearner::new(
        &source.iter().collect::<Vec<_>>(),
        4,
        "target",
        TransferBudget::default(),
    )
    .unwrap();
    learner.observe(&[0], &[17]).unwrap();
    assert!(learner.observe(&[0], &[17]).is_err());
    assert_eq!(learner.stats().queries, 1);
    assert!(learner.observe(&[1, 2], &[17]).is_err());
    assert!(learner.suspended());
}
#[test]
fn empty_invalid_or_too_long_probes_cannot_update_hypotheses() {
    let source = sources();
    let learner = TransferLearner::new(
        &source.iter().collect::<Vec<_>>(),
        4,
        "target",
        TransferBudget::default(),
    )
    .unwrap();
    for word in [Vec::new(), vec![4], vec![0; 49]] {
        assert!(learner.preflight(&word).is_err());
    }
    assert_eq!(learner.stats().queries, 0);
}
struct Spy {
    calls: usize,
}
impl SequenceOracle for Spy {
    fn query(&mut self, w: &[u8]) -> Result<Reply, BehaviorError> {
        self.calls += 1;
        Ok(vec![1; w.len()])
    }
}
#[test]
fn invalid_configuration_is_not_silently_replaced_by_an_unbounded_fallback() {
    let source = sources();
    let mut spy = Spy { calls: 0 };
    let budget = TransferBudget {
        max_hypotheses: 0,
        ..TransferBudget::default()
    };
    assert!(matches!(
        adapt(
            &source.iter().collect::<Vec<_>>(),
            4,
            "target",
            budget,
            &mut spy
        ),
        Err(TransferError::Invalid(_))
    ));
    assert_eq!(spy.calls, 0);
}
#[test]
fn work_and_hypothesis_limits_refuse_before_querying_environment() {
    let source = sources();
    let refs = source.iter().collect::<Vec<_>>();
    assert!(matches!(
        TransferLearner::new(
            &refs,
            4,
            "target",
            TransferBudget {
                max_hypotheses: 1,
                ..TransferBudget::default()
            }
        ),
        Err(TransferError::Limit(_))
    ));
    let mut learner = TransferLearner::new(
        &refs,
        4,
        "target",
        TransferBudget {
            max_work: 1,
            ..TransferBudget::default()
        },
    )
    .unwrap();
    let mut spy = Spy { calls: 0 };
    assert!(matches!(
        learner.learn(&mut spy),
        Err(TransferError::Limit(_))
    ));
    assert_eq!(spy.calls, 0);
}
#[test]
fn unknown_nine_state_body_refutes_old_atlas_and_is_learned_without_erasing_it() {
    let source = sources();
    let before = source
        .iter()
        .map(BehaviorModel::fingerprint)
        .collect::<Vec<_>>();
    let mut rng = Rng::new(91917);
    let env = Rewired::random(Environment::new(Family::Grid, 3, 8), &mut rng);
    let mut oracle = CachedOracle::new(env.clone(), BehaviorBudget::default());
    let result = adapt(
        &source.iter().collect::<Vec<_>>(),
        4,
        "larger-body",
        TransferBudget::default(),
        &mut oracle,
    )
    .unwrap();
    assert_eq!(result.mode, AdaptationMode::Discovery);
    assert_eq!(result.model.states(), 9);
    assert_exact(&result.model, &env);
    assert_eq!(
        before,
        source
            .iter()
            .map(BehaviorModel::fingerprint)
            .collect::<Vec<_>>()
    );
    let mut atlas = source;
    atlas.push(result.model);
    let env = Rewired::random(Environment::new(Family::Grid, 3, 4), &mut rng);
    let mut oracle = CachedOracle::new(env.clone(), BehaviorBudget::default());
    let result = adapt(
        &atlas.iter().collect::<Vec<_>>(),
        4,
        "another-large-body",
        TransferBudget::default(),
        &mut oracle,
    )
    .unwrap();
    assert_eq!(result.mode, AdaptationMode::StructuralTransfer);
    assert_exact(&result.model, &env);
}
#[test]
fn remapped_model_plans_new_goals_and_preserves_bounded_history() {
    let source = sources();
    let env = target(Family::Grid, 7209);
    let mut oracle = CachedOracle::new(env.clone(), BehaviorBudget::default());
    let result = adapt(
        &source.iter().collect::<Vec<_>>(),
        4,
        "new-body",
        TransferBudget::default(),
        &mut oracle,
    )
    .unwrap();
    for source_output in env.environment.symbols() {
        let goal = env.output_for(source_output);
        let plan = result
            .model
            .plan_output(&[], goal, &BTreeSet::new())
            .unwrap()
            .unwrap();
        let reply = oracle.query(&plan).unwrap();
        assert_eq!(reply.last(), Some(&goal));
    }
    assert!(
        result
            .model
            .plan_output(&[], 65535, &BTreeSet::from([65535]))
            .is_err()
    );
}
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "axon-transfer-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn durable_transfer_replays_without_new_resets_or_authority() {
    let t = Temp::new();
    let source = sources();
    let refs = source.iter().collect::<Vec<_>>();
    let env = target(Family::Protocol, 89193);
    let path = t.0.join("experience.axep");
    let artifact = t.0.join("target.axbehavior");
    let mut store = EpisodeStore::open(&path).unwrap().0;
    let before;
    {
        let mut memory = BehaviorMemory::restore(
            &mut store,
            "target",
            4,
            BehaviorBudget::default(),
            Authority::from([Effect::Simulate, Effect::WriteData]),
            env.clone(),
        )
        .unwrap();
        let result = adapt(&refs, 4, "target", TransferBudget::default(), &mut memory).unwrap();
        result.model.save_new(&artifact).unwrap();
        before = memory.queries();
    }
    drop(store);
    let mut store = EpisodeStore::open(&path).unwrap().0;
    let mut memory = BehaviorMemory::restore(
        &mut store,
        "target",
        4,
        BehaviorBudget::default(),
        Authority::from([]),
        env,
    )
    .unwrap();
    let result = adapt(&refs, 4, "target", TransferBudget::default(), &mut memory).unwrap();
    assert_eq!(before, memory.queries());
    assert_eq!(result.model, BehaviorModel::load(&artifact).unwrap());
}
#[test]
fn missing_authority_blocks_even_the_first_reset() {
    let t = Temp::new();
    let source = sources();
    let mut store = EpisodeStore::open(t.0.join("experience.axep")).unwrap().0;
    let mut memory = BehaviorMemory::restore(
        &mut store,
        "target",
        4,
        BehaviorBudget::default(),
        Authority::from([Effect::WriteData]),
        target(Family::Grid, 991),
    )
    .unwrap();
    assert!(matches!(
        adapt(
            &source.iter().collect::<Vec<_>>(),
            4,
            "target",
            TransferBudget::default(),
            &mut memory
        ),
        Err(TransferError::Oracle(BehaviorError::Unauthorized(
            Effect::Simulate
        )))
    ));
    assert_eq!(memory.queries(), 0);
}
#[test]
fn artifacts_reopen_in_the_existing_demand_runtime_and_never_clobber() {
    let t = Temp::new();
    let source = sources();
    let env = target(Family::Grid, 8762);
    let mut oracle = CachedOracle::new(env.clone(), BehaviorBudget::default());
    let result = adapt(
        &source.iter().collect::<Vec<_>>(),
        4,
        "target",
        TransferBudget::default(),
        &mut oracle,
    )
    .unwrap();
    let path = t.0.join("target.axbehavior");
    result.model.save_new(&path).unwrap();
    assert!(result.model.save_new(&path).is_err());
    let authority = Authority::from([Effect::ReadData, Effect::WriteData]);
    let mut catalog = axon_uic::demand::Catalog::new(&t.0);
    catalog
        .register(
            axon_uic::demand::ModuleSpec {
                id: "adapted",
                kind: axon_uic::demand::ModuleKind::Behavior,
                file: "target.axbehavior",
                input: "event",
                output: "prediction",
                cost: 1,
            },
            &authority,
        )
        .unwrap();
    let mut runtime =
        axon_uic::demand::DemandRuntime::new(catalog, axon_uic::demand::Seed::default()).unwrap();
    runtime
        .with_module("adapted", "target", &authority, |m| {
            let axon_uic::demand::Module::Behavior(model) = m else {
                panic!()
            };
            assert_eq!(model, &result.model);
            Ok(())
        })
        .unwrap();
    runtime.unload_all();
    assert_eq!(runtime.stats().active_bytes, 0);
}

#[test]
fn public_demo_publishes_payload_and_replay_executes_no_new_environment_queries() {
    let t = Temp::new();
    let dir = t.0.join("demo");
    let payload = t.0.join("input.md");
    let body = "# A new document\n\nAn actual artifact from the remapped interface.\n";
    fs::write(&payload, body).unwrap();
    let report = axon_uic::cli::command(&[
        "transfer-demo".into(),
        dir.display().to_string(),
        "--payload".into(),
        payload.display().to_string(),
    ])
    .unwrap();
    assert!(report.contains("mode=StructuralTransfer"));
    assert!(report.contains("new_environment_queries=0"));
    assert_eq!(fs::read_to_string(dir.join("published.md")).unwrap(), body);
    assert_eq!(fs::read_to_string(&payload).unwrap(), body);
    let replay =
        axon_uic::cli::command(&["transfer-replay".into(), dir.display().to_string()]).unwrap();
    assert!(replay.contains("replay_model_matches=true"));
    assert!(axon_uic::cli::command(&["transfer-demo".into(), dir.display().to_string()]).is_err());
    assert_eq!(fs::read_to_string(dir.join("published.md")).unwrap(), body);
}
#[test]
fn invalid_payload_and_ambiguous_cli_do_not_create_outputs() {
    let t = Temp::new();
    let dir = t.0.join("demo");
    let payload = t.0.join("input.md");
    fs::write(&payload, "invalid").unwrap();
    assert!(
        axon_uic::cli::command(&[
            "transfer-demo".into(),
            dir.display().to_string(),
            "--payload".into(),
            payload.display().to_string()
        ])
        .is_err()
    );
    assert!(!dir.exists());
    assert!(
        axon_uic::cli::command(&[
            "transfer-demo".into(),
            dir.display().to_string(),
            "--unknown".into(),
            "x".into()
        ])
        .is_err()
    );
    assert!(!dir.exists());
}
#[test]
fn changed_atlas_is_refused_before_replay_even_if_other_models_are_valid() {
    let t = Temp::new();
    let dir = t.0.join("demo");
    axon_uic::cli::command(&["transfer-demo".into(), dir.display().to_string()]).unwrap();
    let target = dir.join("source-control.axbehavior");
    let replacement = fs::read(dir.join("source-protocol.axbehavior")).unwrap();
    fs::write(&target, &replacement).unwrap();
    assert!(
        axon_uic::cli::command(&["transfer-replay".into(), dir.display().to_string()]).is_err()
    );
    assert_eq!(fs::read(&target).unwrap(), replacement);
}
#[test]
fn output_bindings_accept_extreme_tokens_without_sentinel_collisions() {
    let source = sources();
    let refs = source.iter().collect::<Vec<_>>();
    let mut learner = TransferLearner::new(&refs, 4, "target", TransferBudget::default()).unwrap();
    learner.observe(&[0, 1], &[0, 65535]).unwrap();
    assert!(learner.stats().surviving_hypotheses > 0);
}

#[test]
fn compact_selection_matches_every_reference_probe_and_consensus_model() {
    let source = sources();
    let refs = source.iter().collect::<Vec<_>>();
    for family in [Family::Grid, Family::Protocol, Family::Pipeline] {
        for seed in 1830..1838 {
            let env = target(family, seed);
            let mut oracle = CachedOracle::new(env, BehaviorBudget::default());
            let mut fast =
                TransferLearner::new(&refs, 4, "parity", TransferBudget::default()).unwrap();
            let mut slow =
                TransferLearner::new(&refs, 4, "parity", TransferBudget::default()).unwrap();
            loop {
                let compiled = fast.compile().unwrap();
                let reference = slow.compile().unwrap();
                if let Some(compiled) = compiled {
                    assert_eq!(compiled.model, reference.unwrap().model);
                    break;
                }
                assert!(reference.is_none());
                let a = fast.choose_probe().unwrap().unwrap();
                let b = slow.choose_probe_reference().unwrap().unwrap();
                assert_eq!(a, b);
                let reply = oracle.query(&a.word).unwrap();
                fast.observe(&a.word, &reply).unwrap();
                slow.observe(&b.word, &reply).unwrap();
            }
        }
    }
}
