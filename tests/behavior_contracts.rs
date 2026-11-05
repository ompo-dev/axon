use axon_uic::behavior::*;
use axon_uic::behavior_lab::*;
use axon_uic::{Authority, Effect, EpisodeLimits, EpisodeStore};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
static NONCE: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "axon-behavior-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn journal(&self) -> PathBuf {
        self.0.join("experience.axep")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn auth() -> Authority {
    Authority::from([Effect::Simulate, Effect::WriteData])
}
fn payload() -> Vec<u8> {
    b"# AXON learned workflow\nA bounded local publication, not a physical release.\n".to_vec()
}

#[test]
fn discovers_memory_states_with_identical_visible_replies() {
    let budget = BehaviorBudget::default();
    let mut oracle = CachedOracle::new(DocumentWorkflow::new(payload(), None), budget);
    let result = discover(WORKFLOW_ACTIONS, "test", budget, &mut oracle);
    assert!(
        result.is_ok(),
        "result={result:?} queries={} steps={}",
        oracle.raw.queries,
        oracle.raw.steps
    );
    let (model, revisions, suffixes) = result.unwrap();
    assert!(revisions.len() > 1);
    assert!(model.states() > 8);
    assert!(suffixes > WORKFLOW_ACTIONS as usize);
    let audit = audit_machine(&model, WORKFLOW_ACTIONS, 0, |state, action| {
        let mut sut = DocumentWorkflow::new(payload(), None);
        sut.set_audit_state(state as u16);
        let out = sut.step(action)?;
        Ok((sut.hidden_state() as usize, out))
    })
    .unwrap();
    assert!(
        audit.counterexample.is_none(),
        "counterexample {:?}, revisions {:?}",
        audit.counterexample,
        revisions
    );
    // Same last observation, different latent states and future consequences.
    assert_eq!(
        model.predict(&[0, 1, 2]).unwrap().last(),
        model.predict(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap().last()
    );
    assert_ne!(
        model.state_after(&[0, 1, 2]).unwrap(),
        model.state_after(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap()
    );
    assert_eq!(model.predict(&[0, 1, 2, 8]).unwrap().last(), Some(&DENIED));
    assert_eq!(
        model.predict(&[0, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap().last(),
        Some(&PUBLISHED)
    );
}
#[test]
fn random_hidden_systems_match_independent_exhaustive_product_audit() {
    let mut rng = Rng::new(817031);
    for _ in 0..12 {
        let hidden = HiddenMachine::generate(&mut rng);
        let actions = hidden.actions;
        let budget = BehaviorBudget::default();
        let mut oracle = CachedOracle::new(hidden.clone(), budget);
        let (model, _, _) = discover(actions, "random", budget, &mut oracle).unwrap();
        let audit = audit_machine(&model, actions, 0, |s, a| Ok(hidden.transition(s, a))).unwrap();
        assert!(audit.counterexample.is_none());
        assert_eq!(model.states(), hidden.minimal_states());
        assert!(oracle.trie_nodes() > model.states());
    }
}
#[test]
fn memory_restores_learning_without_contacting_environment_and_publishes_real_file() {
    let temp = Temp::new();
    let budget = BehaviorBudget::default();
    let model;
    let queries;
    {
        let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
        let mut memory = BehaviorMemory::restore(
            &mut store,
            "test",
            WORKFLOW_ACTIONS,
            budget,
            auth(),
            DocumentWorkflow::new(payload(), None),
        )
        .unwrap();
        model = discover(WORKFLOW_ACTIONS, "test", budget, &mut memory)
            .unwrap()
            .0;
        queries = memory.queries();
        model.save_new(temp.0.join("model.axbehavior")).unwrap();
    }
    let loaded = BehaviorModel::load(temp.0.join("model.axbehavior")).unwrap();
    assert_eq!(loaded, model);
    let calls = Arc::new(AtomicUsize::new(0));
    struct Never(Arc<AtomicUsize>);
    impl Resettable for Never {
        fn reset(&mut self) -> Result<(), BehaviorError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            panic!("replay must not reset")
        }
        fn step(&mut self, _: u8) -> Result<u16, BehaviorError> {
            panic!()
        }
    }
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let mut memory = BehaviorMemory::restore(
        &mut store,
        "test",
        WORKFLOW_ACTIONS,
        budget,
        Authority::from([]),
        Never(calls.clone()),
    )
    .unwrap();
    assert_eq!(
        discover(WORKFLOW_ACTIONS, "test", budget, &mut memory)
            .unwrap()
            .0,
        model
    );
    assert_eq!(memory.queries(), queries);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    drop(memory);
    let published = temp.0.join("published.md");
    let mut memory = BehaviorMemory::restore(
        &mut store,
        "test",
        WORKFLOW_ACTIONS,
        budget,
        auth(),
        DocumentWorkflow::new(payload(), Some(published.clone())),
    )
    .unwrap();
    let plan = loaded
        .plan_output(&[], PUBLISHED, &BTreeSet::from([401, DENIED]))
        .unwrap()
        .unwrap();
    assert_eq!(plan.len(), 9);
    assert_eq!(
        memory
            .execute_goal(&loaded, &[], PUBLISHED, &BTreeSet::from([401, DENIED]))
            .unwrap()
            .unwrap()
            .last(),
        Some(&PUBLISHED)
    );
    assert_eq!(fs::read(published).unwrap(), payload());
}
#[test]
fn learned_model_finds_edit_after_signature_bug() {
    let budget = BehaviorBudget::default();
    let mut oracle = CachedOracle::new(DocumentWorkflow::new(payload(), None), budget);
    let (model, _, _) = discover(WORKFLOW_ACTIONS, "test", budget, &mut oracle).unwrap();
    let witness = [0, 1, 2, 3, 4, 5, 6, 7, 2, 8];
    assert_eq!(model.predict(&witness).unwrap().last(), Some(&DENIED));
    let mut mutant = DocumentWorkflow::new(payload(), None);
    mutant.mutant = true;
    let mut bad = RawOracle::new(mutant);
    let generated = regression_probe(&model, &mut bad, 65_536).unwrap().unwrap();
    assert_ne!(
        model.predict(&generated).unwrap(),
        bad.query(&generated).unwrap()
    );
    assert_eq!(bad.query(&witness).unwrap().last(), Some(&PUBLISHED));
    let audit = audit_machine(&model, WORKFLOW_ACTIONS, 0, |s, a| {
        let mut sut = DocumentWorkflow::new(payload(), None);
        sut.mutant = true;
        sut.set_audit_state(s as u16);
        let out = sut.step(a)?;
        Ok((sut.hidden_state() as usize, out))
    })
    .unwrap();
    assert!(audit.counterexample.is_some());
}
#[test]
fn authority_and_journal_quota_prevent_even_reset() {
    struct Counter(Arc<AtomicUsize>);
    impl Resettable for Counter {
        fn reset(&mut self) -> Result<(), BehaviorError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn step(&mut self, _: u8) -> Result<u16, BehaviorError> {
            Ok(0)
        }
    }
    let temp = Temp::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let mut memory = BehaviorMemory::restore(
        &mut store,
        "test",
        2,
        BehaviorBudget::default(),
        Authority::from([]),
        Counter(calls.clone()),
    )
    .unwrap();
    assert!(memory.query(&[0]).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    drop(memory);
    drop(store);
    let mut store = EpisodeStore::open_with_limits(
        temp.0.join("tiny"),
        EpisodeLimits {
            max_file_bytes: 16,
            ..EpisodeLimits::default()
        },
    )
    .unwrap()
    .0;
    let mut memory = BehaviorMemory::restore(
        &mut store,
        "test",
        2,
        BehaviorBudget::default(),
        auth(),
        Counter(calls.clone()),
    )
    .unwrap();
    assert!(memory.query(&[0]).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}
#[test]
fn drift_stops_at_first_step_and_suspension_survives_reopen() {
    let temp = Temp::new();
    let budget = BehaviorBudget::default();
    let calls = Arc::new(AtomicUsize::new(0));
    struct Toggle {
        calls: Arc<AtomicUsize>,
        output: u16,
    }
    impl Resettable for Toggle {
        fn reset(&mut self) -> Result<(), BehaviorError> {
            Ok(())
        }
        fn step(&mut self, _: u8) -> Result<u16, BehaviorError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.output)
        }
    }
    {
        let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
        let mut memory = BehaviorMemory::restore(
            &mut store,
            "test",
            1,
            budget,
            auth(),
            Toggle {
                calls: calls.clone(),
                output: 1,
            },
        )
        .unwrap();
        memory.query(&[0, 0]).unwrap();
        drop(memory);
        let mut memory = BehaviorMemory::restore(
            &mut store,
            "test",
            1,
            budget,
            auth(),
            Toggle {
                calls: calls.clone(),
                output: 2,
            },
        )
        .unwrap();
        let before = calls.load(Ordering::Relaxed);
        assert!(matches!(
            memory.fresh(&[0, 0, 0]),
            Err(BehaviorError::QueryOccurred { .. })
        ));
        assert_eq!(calls.load(Ordering::Relaxed) - before, 1);
        assert!(memory.suspended());
        assert!(memory.query(&[0]).is_err());
    }
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let memory = BehaviorMemory::restore(
        &mut store,
        "test",
        1,
        budget,
        auth(),
        Toggle { calls, output: 1 },
    )
    .unwrap();
    assert!(memory.suspended());
}

#[test]
fn shallow_conformance_is_not_an_equivalence_certificate() {
    let budget = BehaviorBudget::default();
    let mut oracle = CachedOracle::new(DocumentWorkflow::new(payload(), None), budget);
    let learner_context = "shallow";
    let mut learner = BehaviorLearner::new(WORKFLOW_ACTIONS, learner_context, budget).unwrap();
    let early = learner.hypothesize(&mut oracle).unwrap();
    assert!(
        learner
            .counterexample(&early, &mut oracle, 1, 65_536)
            .unwrap()
            .is_none()
    );
    let audit = audit_machine(&early, WORKFLOW_ACTIONS, 0, |s, a| {
        let mut sut = DocumentWorkflow::new(payload(), None);
        sut.set_audit_state(s as u16);
        let out = sut.step(a)?;
        Ok((sut.hidden_state() as usize, out))
    })
    .unwrap();
    assert!(audit.counterexample.is_some());
    assert!(
        early
            .plan_output(&[], PUBLISHED, &BTreeSet::new())
            .unwrap()
            .is_none()
    );
}
#[test]
fn a_counterexample_splits_state_and_preserves_the_old_hypothesis() {
    let budget = BehaviorBudget::default();
    let mut oracle = CachedOracle::new(DocumentWorkflow::new(payload(), None), budget);
    let mut learner = BehaviorLearner::new(WORKFLOW_ACTIONS, "revision", budget).unwrap();
    let old = learner.hypothesize(&mut oracle).unwrap();
    let hash = old.fingerprint();
    assert!(matches!(
        learner.refine(&old, &[0], &mut oracle),
        Err(BehaviorError::NotCounterexample)
    ));
    learner
        .refine(&old, &[0, 1, 2, 3, 4, 5, 6, 7, 8], &mut oracle)
        .unwrap();
    let new = learner.hypothesize(&mut oracle).unwrap();
    assert!(new.states() > old.states());
    assert_eq!(old.fingerprint(), hash);
    assert_eq!(
        new.predict(&[0, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap().last(),
        Some(&PUBLISHED)
    );
}
#[test]
fn model_storage_refuses_truncation_corruption_and_clobber() {
    let temp = Temp::new();
    let mut oracle = CachedOracle::new(
        DocumentWorkflow::new(payload(), None),
        BehaviorBudget::default(),
    );
    let model = BehaviorLearner::new(WORKFLOW_ACTIONS, "storage", BehaviorBudget::default())
        .unwrap()
        .hypothesize(&mut oracle)
        .unwrap();
    let path = temp.0.join("model");
    model.save_new(&path).unwrap();
    assert!(model.save_new(&path).is_err());
    let bytes = fs::read(&path).unwrap();
    for n in 0..bytes.len() {
        let file = temp.0.join(format!("cut-{n}"));
        fs::write(&file, &bytes[..n]).unwrap();
        assert!(BehaviorModel::load(file).is_err());
    }
    let mut corrupted = bytes;
    corrupted[13] ^= 1;
    let file = temp.0.join("corrupt");
    fs::write(&file, corrupted).unwrap();
    assert!(BehaviorModel::load(file).is_err());
}
#[test]
fn valid_checksum_does_not_bypass_target_or_dimension_validation() {
    let temp = Temp::new();
    let mut oracle = CachedOracle::new(
        DocumentWorkflow::new(payload(), None),
        BehaviorBudget::default(),
    );
    let model = BehaviorLearner::new(WORKFLOW_ACTIONS, "storage", BehaviorBudget::default())
        .unwrap()
        .hypothesize(&mut oracle)
        .unwrap();
    let path = temp.0.join("model");
    model.save_new(&path).unwrap();
    let bytes = fs::read(path).unwrap();
    let start = 13 + "storage".len();
    for index in [8, 9, start] {
        let mut forged = bytes.clone();
        forged[index] = 255;
        let p = forged.len() - 8;
        let hash = forged[..p].iter().fold(0xcbf29ce484222325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        forged[p..].copy_from_slice(&hash.to_le_bytes());
        let path = temp.0.join(format!("forged-{index}"));
        fs::write(&path, forged).unwrap();
        assert!(BehaviorModel::load(path).is_err());
    }
}
#[test]
fn bounds_are_refusals_not_silently_partial_models() {
    assert!(BehaviorLearner::new(17, "test", BehaviorBudget::default()).is_err());
    assert!(BehaviorLearner::new(2, " ", BehaviorBudget::default()).is_err());
    let budget = BehaviorBudget {
        max_states: 1,
        ..BehaviorBudget::default()
    };
    let mut oracle = CachedOracle::new(DocumentWorkflow::new(payload(), None), budget);
    let mut learner = BehaviorLearner::new(WORKFLOW_ACTIONS, "bound", budget).unwrap();
    assert!(matches!(
        learner.hypothesize(&mut oracle),
        Err(BehaviorError::Limit("latent states"))
    ));
    let budget = BehaviorBudget::default();
    let mut oracle = CachedOracle::new(DocumentWorkflow::new(payload(), None), budget);
    let mut learner = BehaviorLearner::new(WORKFLOW_ACTIONS, "bound", budget).unwrap();
    let model = learner.hypothesize(&mut oracle).unwrap();
    assert!(matches!(
        learner.counterexample(&model, &mut oracle, 1, 1),
        Err(BehaviorError::Limit("conformance tests"))
    ));
    assert!(model.predict(&[11]).is_err());
    assert!(model.predict(&[0; 49]).is_err());
}
#[test]
fn malformed_oracle_replies_never_become_hypotheses() {
    struct Wrong;
    impl SequenceOracle for Wrong {
        fn query(&mut self, _: &[u8]) -> Result<Reply, BehaviorError> {
            Ok(Vec::new())
        }
    }
    let mut learner = BehaviorLearner::new(1, "test", BehaviorBudget::default()).unwrap();
    assert!(matches!(
        learner.hypothesize(&mut Wrong),
        Err(BehaviorError::Invalid("oracle reply length"))
    ));
}
#[test]
fn step_failure_is_durable_and_prevents_further_actions() {
    struct Fails;
    impl Resettable for Fails {
        fn reset(&mut self) -> Result<(), BehaviorError> {
            Ok(())
        }
        fn step(&mut self, _: u8) -> Result<u16, BehaviorError> {
            Err(BehaviorError::Invalid("adapter failure"))
        }
    }
    let temp = Temp::new();
    {
        let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
        let mut memory = BehaviorMemory::restore(
            &mut store,
            "failure",
            1,
            BehaviorBudget::default(),
            auth(),
            Fails,
        )
        .unwrap();
        assert!(matches!(
            memory.query(&[0, 0]),
            Err(BehaviorError::QueryOccurred { .. })
        ));
        assert!(memory.suspended());
        assert!(memory.query(&[0]).is_err());
    }
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    assert!(
        BehaviorMemory::restore(
            &mut store,
            "failure",
            1,
            BehaviorBudget::default(),
            auth(),
            Fails
        )
        .unwrap()
        .suspended()
    );
}
#[test]
fn execution_checks_each_step_instead_of_running_the_entire_bad_plan() {
    let budget = BehaviorBudget::default();
    let mut oracle = CachedOracle::new(DocumentWorkflow::new(payload(), None), budget);
    let model = discover(WORKFLOW_ACTIONS, "drift", budget, &mut oracle)
        .unwrap()
        .0;
    struct Bad(Arc<AtomicUsize>);
    impl Resettable for Bad {
        fn reset(&mut self) -> Result<(), BehaviorError> {
            Ok(())
        }
        fn step(&mut self, _: u8) -> Result<u16, BehaviorError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(999)
        }
    }
    let temp = Temp::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let mut memory = BehaviorMemory::restore(
        &mut store,
        "drift",
        WORKFLOW_ACTIONS,
        budget,
        auth(),
        Bad(calls.clone()),
    )
    .unwrap();
    assert!(
        matches!(memory.execute_goal(&model,&[],PUBLISHED,&BTreeSet::new()),Err(BehaviorError::QueryOccurred{reply,..}) if reply==vec![999])
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}
#[test]
fn forbidden_goal_cannot_trigger_adapter_effects() {
    let budget = BehaviorBudget::default();
    let mut oracle = CachedOracle::new(DocumentWorkflow::new(payload(), None), budget);
    let model = BehaviorLearner::new(WORKFLOW_ACTIONS, "safe", budget)
        .unwrap()
        .hypothesize(&mut oracle)
        .unwrap();
    let temp = Temp::new();
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let mut memory = BehaviorMemory::restore(
        &mut store,
        "safe",
        WORKFLOW_ACTIONS,
        budget,
        Authority::from([]),
        DocumentWorkflow::new(payload(), None),
    )
    .unwrap();
    assert!(
        memory
            .execute_goal(&model, &[], ACK, &BTreeSet::from([ACK]))
            .is_err()
    );
    assert_eq!(memory.queries(), 0);
}
#[test]
fn public_cli_publishes_and_preserves_payload_and_previous_outputs() {
    use std::process::Command;
    let temp = Temp::new();
    let body = temp.0.join("source.md");
    fs::write(&body, payload()).unwrap();
    let output = temp.0.join("demo");
    let bin = env!("CARGO_BIN_EXE_axon");
    let result = Command::new(bin)
        .arg("behavior-demo")
        .arg(&output)
        .arg("--payload")
        .arg(&body)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fs::read(&body).unwrap(), payload());
    assert_eq!(fs::read(output.join("published.md")).unwrap(), payload());
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .contains("replay_new_queries=0")
    );
    assert!(
        !Command::new(bin)
            .arg("behavior-demo")
            .arg(&output)
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(
        Command::new(bin)
            .arg("behavior-inspect")
            .arg(output.join("document.axbehavior"))
            .output()
            .unwrap()
            .status
            .success()
    );
}

#[test]
fn restored_shared_prefix_conflicts_suspend_without_reinterpreting_the_evidence() {
    use axon_uic::{EpisodeData, EpisodeStamp, EpisodeValue};
    use std::collections::BTreeMap;
    let temp = Temp::new();
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    for (event, input, output) in [("one", "0001", "00010002"), ("two", "00", "0000")] {
        let fields = BTreeMap::from([
            ("actions".into(), EpisodeValue::Integer(2)),
            ("input".into(), EpisodeValue::Text(input.into())),
            ("output".into(), EpisodeValue::Text(output.into())),
        ]);
        store
            .record(
                EpisodeStamp::new("axon.behavior/v1", event, 1),
                EpisodeData::Trace {
                    context: "conflict".into(),
                    channel: "behavior.query/v1".into(),
                    fields,
                },
            )
            .unwrap();
    }
    let memory = BehaviorMemory::restore(
        &mut store,
        "conflict",
        2,
        BehaviorBudget::default(),
        Authority::from([]),
        DocumentWorkflow::new(payload(), None),
    )
    .unwrap();
    assert!(memory.suspended());
}

#[test]
fn learned_tracker_handles_a_million_unreset_events_in_fixed_memory_and_refuses_drift() {
    let mut rng = Rng::new(713010);
    let mut hidden = HiddenMachine::generate(&mut rng);
    let actions = hidden.actions;
    let mut oracle = CachedOracle::new(hidden.clone(), BehaviorBudget::default());
    let model = discover(actions, "stream", BehaviorBudget::default(), &mut oracle)
        .unwrap()
        .0;
    let mut tracker = BehaviorTracker::new(&model, "stream").unwrap();
    let bytes = std::mem::size_of_val(&tracker);
    assert!(BehaviorTracker::new(&model, "other").is_err());
    for _ in 0..1_000_000 {
        let action = rng.pick(actions as usize) as u8;
        let predicted = tracker.predict_next(action).unwrap();
        let actual = hidden.step(action).unwrap();
        assert_eq!(predicted, actual);
        tracker.observe(action, actual).unwrap();
    }
    assert_eq!(tracker.observations(), 1_000_000);
    assert_eq!(std::mem::size_of_val(&tracker), bytes);
    assert!(bytes <= 32);
    let predicted = tracker.predict_next(0).unwrap();
    assert!(matches!(
        tracker.observe(0, predicted.wrapping_add(1)),
        Err(BehaviorError::UnexpectedOutput { .. })
    ));
    assert!(tracker.suspended());
    assert!(tracker.plan(0, &BTreeSet::new()).is_err());
}
