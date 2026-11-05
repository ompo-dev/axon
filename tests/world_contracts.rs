use axon_uic::world::*;
use axon_uic::{
    Authority, Effect, EpisodeData, EpisodeLimits, EpisodeStamp, EpisodeStore, EpisodeValue,
};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

struct Temp(PathBuf);
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "axon-world-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn journal(&self) -> PathBuf {
        self.0.join("events.axep")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn authority() -> Authority {
    Authority::from([Effect::Simulate, Effect::WriteData])
}
// Deliberately independent of TransitionRule::apply.
fn oracle(state: u8, action: u8) -> u8 {
    match action {
        0 if state & 1 != 0 => state | 2,
        1 if state & 1 != 0 && state & 4 == 0 => state ^ 8,
        2 => state ^ 1,
        3 if state & 8 == 0 => state & !16,
        4 if state & 2 != 0 => state ^ 4,
        _ => state,
    }
}
fn trained() -> WorldLearner {
    let shape = WorldShape::new(5, 5).unwrap();
    let mut learner = WorldLearner::new(shape, "test").unwrap();
    let scope = ProbeScope::unrestricted(shape, 512);
    while let ProbeChoice::Experiment(p) = learner.choose_probe(&scope).unwrap() {
        learner
            .observe(Transition {
                before: p.state,
                action: p.action,
                after: oracle(p.state, p.action),
                kind: ObservationKind::Intervention,
            })
            .unwrap();
    }
    assert!(learner.identified());
    learner
}

#[test]
fn active_learning_identifies_hidden_guards_and_effects_not_given_to_learner() {
    let learner = trained();
    assert!(learner.interventions() < 160);
    let model = learner.compile().unwrap();
    for s in 0..32 {
        for a in 0..5 {
            assert_eq!(model.predict(s, a).unwrap(), oracle(s, a));
        }
    }
}
#[test]
fn passive_correlations_do_not_narrow_interventional_hypotheses() {
    let mut learner = WorldLearner::new(WorldShape::new(3, 2).unwrap(), "test").unwrap();
    let counts = learner.candidate_counts();
    for state in 0..8 {
        learner
            .observe(Transition {
                before: state,
                action: 0,
                after: state ^ 7,
                kind: ObservationKind::Passive,
            })
            .unwrap();
    }
    assert_eq!(learner.candidate_counts(), counts);
    assert_eq!(learner.interventions(), 0);
    assert!(!learner.identified());
}
#[test]
fn conservative_planning_distinguishes_uncertainty_from_unreachability() {
    let shape = WorldShape::new(2, 1).unwrap();
    let learner = WorldLearner::new(shape, "test").unwrap();
    assert_eq!(
        learner
            .plan(0, Goal { on: 1, off: 0 }, shape.all_states())
            .unwrap(),
        PlanResult::NeedsExperiment
    );
    let mut noop = learner;
    for state in 0..4 {
        noop.observe(Transition {
            before: state,
            action: 0,
            after: state,
            kind: ObservationKind::Intervention,
        })
        .unwrap();
    }
    assert!(noop.identified());
    assert_eq!(
        noop.plan(0, Goal { on: 1, off: 0 }, shape.all_states())
            .unwrap(),
        PlanResult::Unreachable
    );
}
#[test]
fn novel_goal_requires_ordered_plan_and_matches_independent_oracle() {
    let model = trained().compile().unwrap();
    let PlanResult::Found(plan) = model
        .plan(20, Goal { on: 11, off: 20 }, u32::MAX.into())
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(plan.steps.len(), 5);
    let mut state = 20;
    for step in plan.steps {
        assert_eq!(state, step.before);
        state = oracle(state, step.action);
        assert_eq!(state, step.after);
    }
    assert!(Goal { on: 11, off: 20 }.matches(state));
}
#[test]
fn safety_is_over_every_surviving_hypothesis_not_only_selected_candidate() {
    let shape = WorldShape::new(2, 1).unwrap();
    let learner = WorldLearner::new(shape, "test").unwrap();
    let mut scope = ProbeScope::unrestricted(shape, 100);
    scope.starts = 1;
    scope.outcomes = 1;
    assert_eq!(
        learner.choose_probe(&scope).unwrap(),
        ProbeChoice::NoSafeAffordableExperiment
    );
    scope.outcomes = shape.all_states();
    scope.remaining = 0;
    assert_eq!(
        learner.choose_probe(&scope).unwrap(),
        ProbeChoice::NoSafeAffordableExperiment
    );
}
#[test]
fn cost_weighting_changes_probe_selection() {
    let shape = WorldShape::new(3, 2).unwrap();
    let learner = WorldLearner::new(shape, "test").unwrap();
    let mut scope = ProbeScope::unrestricted(shape, 100);
    scope.action_costs = vec![100, 1];
    let ProbeChoice::Experiment(p) = learner.choose_probe(&scope).unwrap() else {
        panic!()
    };
    assert_eq!(p.action, 1);
}
#[test]
fn out_of_family_behavior_is_refuted_instead_of_silently_retrained() {
    let mut learner = WorldLearner::new(WorldShape::new(3, 1).unwrap(), "test").unwrap();
    assert!(matches!(
        learner.observe(Transition {
            before: 0,
            action: 0,
            after: 7,
            kind: ObservationKind::Intervention
        }),
        Err(WorldError::Contradiction)
    ));
    assert!(learner.suspended());
    assert!(learner.compile().is_err());
    assert!(learner.predictions(0, 0).is_err());
}
#[test]
fn authorization_and_quota_fail_before_any_callback() {
    let temp = Temp::new();
    let shape = WorldShape::new(2, 1).unwrap();
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let mut session = WorldSession::restore(&mut store, shape, "test").unwrap();
    let mut scope = ProbeScope::unrestricted(shape, 100);
    let mut calls = 0;
    for auth in [
        Authority::from([]),
        Authority::from([Effect::Simulate]),
        Authority::from([Effect::Actuator, Effect::WriteData]),
    ] {
        assert!(
            session
                .intervene(
                    Probe {
                        state: 0,
                        action: 0,
                        cost: 1
                    },
                    &mut scope,
                    &auth,
                    |s, _| {
                        calls += 1;
                        s
                    }
                )
                .is_err()
        );
    }
    assert_eq!(calls, 0);
    assert_eq!(scope.remaining, 100);
    drop(session);
    drop(store);
    let mut store = EpisodeStore::open_with_limits(
        temp.0.join("tiny.axep"),
        EpisodeLimits {
            max_file_bytes: 16,
            ..EpisodeLimits::default()
        },
    )
    .unwrap()
    .0;
    let mut session = WorldSession::restore(&mut store, shape, "test").unwrap();
    assert!(
        session
            .intervene(
                Probe {
                    state: 0,
                    action: 0,
                    cost: 1
                },
                &mut scope,
                &authority(),
                |s, _| {
                    calls += 1;
                    s
                }
            )
            .is_err()
    );
    assert_eq!(calls, 0);
}
#[test]
fn denied_outcome_scope_is_checked_before_callback() {
    let temp = Temp::new();
    let shape = WorldShape::new(2, 1).unwrap();
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let mut session = WorldSession::restore(&mut store, shape, "test").unwrap();
    let mut scope = ProbeScope::unrestricted(shape, 100);
    scope.outcomes = 1;
    let mut calls = 0;
    assert!(
        session
            .intervene(
                Probe {
                    state: 0,
                    action: 0,
                    cost: 1
                },
                &mut scope,
                &authority(),
                |s, _| {
                    calls += 1;
                    s
                }
            )
            .is_err()
    );
    assert_eq!(calls, 0);
}
#[test]
fn durable_replay_recovers_model_without_repeating_experiments() {
    let temp = Temp::new();
    let shape = WorldShape::new(5, 5).unwrap();
    let expected;
    {
        let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
        let mut session = WorldSession::restore(&mut store, shape, "test").unwrap();
        session.record_passive(0, 0, 31).unwrap();
        let mut scope = ProbeScope::unrestricted(shape, 512);
        while let ProbeChoice::Experiment(p) = session.learner().choose_probe(&scope).unwrap() {
            session
                .intervene(p, &mut scope, &authority(), oracle)
                .unwrap();
        }
        expected = session.learner().compile().unwrap();
        expected.save_new(temp.0.join("model.axworld")).unwrap();
    }
    let (store, recovered) = EpisodeStore::open(temp.journal()).unwrap();
    assert_eq!(
        recovered.replayed_episodes,
        usize::from(expected.interventions()) + 1
    );
    assert_eq!(
        WorldLearner::restore(&store, shape, "test")
            .unwrap()
            .compile()
            .unwrap(),
        expected
    );
    assert_eq!(
        WorldModel::load(temp.0.join("model.axworld")).unwrap(),
        expected
    );
    assert!(
        !WorldLearner::restore(&store, shape, "other")
            .unwrap()
            .identified()
    );
    assert!(WorldLearner::restore(&store, WorldShape::new(4, 5).unwrap(), "test").is_err());
}
#[test]
fn execution_stops_on_first_drift_and_old_checkpoint_is_unchanged() {
    let temp = Temp::new();
    let shape = WorldShape::new(5, 5).unwrap();
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let mut session = WorldSession::restore(&mut store, shape, "test").unwrap();
    let mut scope = ProbeScope::unrestricted(shape, 512);
    while let ProbeChoice::Experiment(p) = session.learner().choose_probe(&scope).unwrap() {
        session
            .intervene(p, &mut scope, &authority(), oracle)
            .unwrap();
    }
    let model = session.learner().compile().unwrap();
    let fingerprint = model.fingerprint();
    let PlanResult::Found(plan) = model
        .plan(20, Goal { on: 11, off: 20 }, shape.all_states())
        .unwrap()
    else {
        panic!()
    };
    let mut calls = 0;
    assert!(
        session
            .execute(&model, &plan, &mut scope, &authority(), |s, _| {
                calls += 1;
                s
            })
            .is_err()
    );
    assert_eq!(calls, 1);
    assert!(session.learner().suspended());
    assert_eq!(model.fingerprint(), fingerprint);
    assert!(
        session
            .intervene(
                Probe {
                    state: 0,
                    action: 0,
                    cost: 1
                },
                &mut scope,
                &authority(),
                |s, _| {
                    calls += 1;
                    s
                }
            )
            .is_err()
    );
    assert_eq!(calls, 1);
    drop(session);
    assert!(
        WorldLearner::restore(&store, shape, "test")
            .unwrap()
            .suspended()
    );
}
#[test]
fn tampered_or_wrong_goal_plan_does_not_act() {
    let temp = Temp::new();
    let shape = WorldShape::new(5, 5).unwrap();
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let mut session = WorldSession::restore(&mut store, shape, "test").unwrap();
    let mut scope = ProbeScope::unrestricted(shape, 512);
    while let ProbeChoice::Experiment(p) = session.learner().choose_probe(&scope).unwrap() {
        session
            .intervene(p, &mut scope, &authority(), oracle)
            .unwrap();
    }
    let model = session.learner().compile().unwrap();
    let PlanResult::Found(mut plan) = model
        .plan(20, Goal { on: 11, off: 20 }, shape.all_states())
        .unwrap()
    else {
        panic!()
    };
    plan.steps[0].after ^= 1;
    let mut calls = 0;
    assert!(
        session
            .execute(&model, &plan, &mut scope, &authority(), |s, _| {
                calls += 1;
                s
            })
            .is_err()
    );
    assert_eq!(calls, 0);
}
#[test]
fn artifacts_are_no_clobber_bounded_and_checked() {
    let temp = Temp::new();
    let path = temp.0.join("model.axworld");
    let model = trained().compile().unwrap();
    model.save_new(&path).unwrap();
    assert!(model.save_new(&path).is_err());
    let bytes = fs::read(&path).unwrap();
    for length in 0..bytes.len() {
        let p = temp.0.join(format!("truncated-{length}"));
        fs::write(&p, &bytes[..length]).unwrap();
        assert!(WorldModel::load(p).is_err());
    }
    let p = temp.0.join("corrupt");
    let mut corrupt = bytes;
    corrupt[13] ^= 1;
    fs::write(&p, corrupt).unwrap();
    assert!(WorldModel::load(&p).is_err());
    fs::write(&p, vec![0; 513]).unwrap();
    assert!(WorldModel::load(p).is_err());
}
#[test]
fn trace_roundtrip_indexes_values_and_enforces_field_bounds() {
    let temp = Temp::new();
    let fields = BTreeMap::from([
        ("text".to_owned(), EpisodeValue::Text("ok".to_owned())),
        ("int".to_owned(), EpisodeValue::Integer(-7)),
        ("bool".to_owned(), EpisodeValue::Boolean(true)),
    ]);
    let data = EpisodeData::Trace {
        context: "lab".to_owned(),
        channel: "sample/v1".to_owned(),
        fields,
    };
    {
        let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
        let stamp = EpisodeStamp::new("test", "one", 1);
        store.record(stamp.clone(), data.clone()).unwrap();
        assert!(store.record(stamp, data.clone()).unwrap().duplicate);
        let fields = (0..17)
            .map(|i| (format!("f{i}"), EpisodeValue::Integer(i)))
            .collect();
        assert!(
            store
                .record(
                    EpisodeStamp::new("test", "bad", 2),
                    EpisodeData::Trace {
                        context: "lab".into(),
                        channel: "sample/v1".into(),
                        fields
                    }
                )
                .is_err()
        );
        assert_eq!(store.len(), 1);
    }
    let store = EpisodeStore::open(temp.journal()).unwrap().0;
    assert_eq!(store.traces("lab", "sample/v1").unwrap()[0].data(), &data);
    assert!(store.traces("other", "sample/v1").unwrap().is_empty());
}
#[test]
fn shape_context_state_scope_and_goal_are_bounded() {
    assert!(WorldShape::new(7, 1).is_err());
    assert!(WorldShape::new(1, 9).is_err());
    let shape = WorldShape::new(6, 8).unwrap();
    assert_eq!(shape.all_states(), u64::MAX);
    assert!(WorldLearner::new(shape, " ").is_err());
    let learner = WorldLearner::new(shape, "test").unwrap();
    assert!(learner.predictions(64, 0).is_err());
    assert!(learner.predictions(0, 8).is_err());
    assert!(learner.plan(0, Goal { on: 1, off: 1 }, u64::MAX).is_err());
    let mut scope = ProbeScope::unrestricted(shape, 100);
    scope.action_costs[0] = 0;
    assert!(learner.choose_probe(&scope).is_err());
}

#[test]
fn three_literal_guard_and_stochastic_transitions_are_negative_controls() {
    let shape = WorldShape::new(4, 1).unwrap();
    let mut learner = WorldLearner::new(shape, "three-guard").unwrap();
    let mut refuted = false;
    for state in 0..16 {
        let after = if state & 7 == 7 { state | 8 } else { state };
        if matches!(
            learner.observe(Transition {
                before: state,
                action: 0,
                after,
                kind: ObservationKind::Intervention
            }),
            Err(WorldError::Contradiction)
        ) {
            refuted = true;
            break;
        }
    }
    assert!(refuted);
    assert!(learner.suspended());
    let mut learner = WorldLearner::new(shape, "stochastic").unwrap();
    learner
        .observe(Transition {
            before: 0,
            action: 0,
            after: 1,
            kind: ObservationKind::Intervention,
        })
        .unwrap();
    assert!(matches!(
        learner.observe(Transition {
            before: 0,
            action: 0,
            after: 0,
            kind: ObservationKind::Intervention
        }),
        Err(WorldError::Contradiction)
    ));
}

#[test]
fn record_limit_does_not_commit_or_silently_drop_observations() {
    let shape = WorldShape::new(2, 1).unwrap();
    let mut learner = WorldLearner::new(shape, "limit").unwrap();
    let passive = Transition {
        before: 0,
        action: 0,
        after: 0,
        kind: ObservationKind::Passive,
    };
    for _ in 0..512 {
        learner.observe(passive).unwrap();
    }
    let counts = learner.candidate_counts();
    assert!(matches!(
        learner.observe(passive),
        Err(WorldError::Exhausted)
    ));
    assert_eq!(learner.records(), 512);
    assert_eq!(learner.candidate_counts(), counts);
    assert!(matches!(
        learner.choose_probe(&ProbeScope::unrestricted(shape, 100)),
        Err(WorldError::Exhausted)
    ));
}

#[test]
fn journal_schema_cannot_be_silently_reinterpreted_as_training() {
    let temp = Temp::new();
    let shape = WorldShape::new(2, 1).unwrap();
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let fields = BTreeMap::from([
        ("bits".into(), EpisodeValue::Integer(2)),
        ("actions".into(), EpisodeValue::Integer(1)),
        ("before".into(), EpisodeValue::Integer(0)),
        ("action".into(), EpisodeValue::Integer(0)),
        ("after".into(), EpisodeValue::Integer(1)),
        ("intervention".into(), EpisodeValue::Boolean(true)),
    ]);
    store
        .record(
            EpisodeStamp::new("other", "one", 1),
            EpisodeData::Trace {
                context: "test".into(),
                channel: "world.transition/v1".into(),
                fields,
            },
        )
        .unwrap();
    assert!(matches!(
        WorldLearner::restore(&store, shape, "test"),
        Err(WorldError::Invalid("trace source"))
    ));
}

#[test]
fn public_cli_demo_reloads_plans_and_refuses_existing_output() {
    use std::process::Command;
    let temp = Temp::new();
    let root = temp.0.join("demo");
    let bin = env!("CARGO_BIN_EXE_axon");
    let result = Command::new(bin)
        .arg("world-demo")
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout).unwrap();
    assert!(output.contains("replay_exact=true audit=160/160"));
    assert!(output.contains("final_state=11 reached=true"));
    let model = root.join("workcell.axworld");
    let previous = fs::read(&model).unwrap();
    assert!(
        !Command::new(bin)
            .arg("world-demo")
            .arg(&root)
            .output()
            .unwrap()
            .status
            .success()
    );
    assert_eq!(fs::read(&model).unwrap(), previous);
    let result = Command::new(bin)
        .arg("world-plan")
        .arg(&model)
        .args([
            "--context",
            "demo/workcell-v1",
            "--start",
            "20",
            "--on",
            "11",
            "--off",
            "20",
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert!(String::from_utf8(result.stdout).unwrap().contains("Found"));
    assert!(
        !Command::new(bin)
            .arg("world-plan")
            .arg(&model)
            .args([
                "--context",
                "other",
                "--start",
                "20",
                "--on",
                "11",
                "--off",
                "20"
            ])
            .output()
            .unwrap()
            .status
            .success()
    );
    let rebuilt = root.join("rebuilt.axworld");
    assert!(
        Command::new(bin)
            .arg("world-rebuild")
            .arg(root.join("workcell.axep"))
            .args([
                "--bits",
                "5",
                "--actions",
                "5",
                "--context",
                "demo/workcell-v1",
                "--model"
            ])
            .arg(&rebuilt)
            .output()
            .unwrap()
            .status
            .success()
    );
    let original = WorldModel::load(&model).unwrap();
    let restored = WorldModel::load(rebuilt).unwrap();
    assert_eq!(original.rules(), restored.rules());
    assert_eq!(restored.interventions(), original.interventions() + 5);
    let missing = root.join("missing.axep");
    assert!(
        !Command::new(bin)
            .arg("world-rebuild")
            .arg(&missing)
            .args([
                "--bits",
                "5",
                "--actions",
                "5",
                "--context",
                "demo/workcell-v1",
                "--model"
            ])
            .arg(root.join("bad.axworld"))
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(!missing.exists());
}

#[test]
fn partial_models_can_plan_only_on_consensus_edges() {
    let shape = WorldShape::new(3, 2).unwrap();
    let mut learner = WorldLearner::new(shape, "partial").unwrap();
    learner
        .observe(Transition {
            before: 0,
            action: 0,
            after: 1,
            kind: ObservationKind::Intervention,
        })
        .unwrap();
    assert!(!learner.identified());
    assert!(learner.compile().is_err());
    let PlanResult::Found(plan) = learner
        .plan(0, Goal { on: 1, off: 6 }, shape.all_states())
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(
        plan.steps,
        vec![PlanStep {
            before: 0,
            action: 0,
            after: 1
        }]
    );
    assert_eq!(
        learner
            .plan(0, Goal { on: 4, off: 0 }, shape.all_states())
            .unwrap(),
        PlanResult::NeedsExperiment
    );
}

#[test]
fn valid_checksums_do_not_bypass_model_rule_and_count_validation() {
    let temp = Temp::new();
    let path = temp.0.join("original");
    let model = trained().compile().unwrap();
    model.save_new(&path).unwrap();
    let bytes = fs::read(path).unwrap();
    let context_size = u16::from_le_bytes(bytes[10..12].try_into().unwrap()) as usize;
    let counts = 12 + context_size;
    let rules = counts + 12;
    for (index, value) in [(8, 7), (counts + 1, 255), (rules, 128), (rules + 3, 3)] {
        let mut forged = bytes.clone();
        forged[index] = value;
        let payload = forged.len() - 8;
        let hash = forged[..payload]
            .iter()
            .fold(0xcbf29ce484222325_u64, |h, &b| {
                (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
            });
        forged[payload..].copy_from_slice(&hash.to_le_bytes());
        let file = temp.0.join(format!("forged-{index}"));
        fs::write(&file, forged).unwrap();
        assert!(WorldModel::load(file).is_err());
    }
}
