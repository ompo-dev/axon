use axon_uic::{
    Authority, Effect,
    behavior::*,
    factor::*,
    factor_lab::{self, Environment, Kind},
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
fn cache(env: Environment) -> FrameCache<Environment> {
    let (a, d) = (env.actions(), env.dimensions());
    FrameCache::new(
        env,
        a,
        d,
        FrameBudget::default(),
        Authority::from([Effect::Simulate]),
    )
    .unwrap()
}
fn fit(env: &Environment, context: &str) -> FactorModel {
    learn(
        env.actions(),
        env.dimensions(),
        context,
        &mut cache(env.clone()),
    )
    .unwrap()
}
#[test]
fn projections_learn_a_joint_system_larger_than_the_monolithic_state_limit() {
    let env = Environment::random(Kind::Workflow, 4, 531719)
        .unwrap()
        .zero_initial();
    let model = fit(&env, "four-documents");
    let audit = factor_lab::audit(&model, &env, 65536).unwrap();
    assert!(audit.counterexample.is_none(), "{:?}", audit.counterexample);
    assert_eq!(audit.reachable_pairs, 256);
    assert_eq!(model.factor_states(), 16);
    assert_eq!(model.product_states(), 256);
    let goal = model.goal(&env.goal_tokens()).unwrap();
    let plan = model
        .plan(&[], &goal, PlanBudget::default())
        .unwrap()
        .unwrap();
    assert_eq!(plan.len(), 12);
    let result = cache(env).query(&plan).unwrap();
    for (i, token) in model
        .predict(&plan)
        .unwrap()
        .last()
        .unwrap()
        .values()
        .iter()
        .enumerate()
    {
        assert_eq!(*token, result.last().unwrap().values()[i]);
    }
}
#[test]
fn parallel_and_coupled_interfaces_match_the_whole_product_audit() {
    for seed in 831..837 {
        for (kind, n) in [(Kind::Cycles, 3), (Kind::Workflow, 3), (Kind::Coupled, 2)] {
            let env = Environment::random(kind, n, seed).unwrap();
            let model = fit(&env, "hidden");
            let audit = factor_lab::audit(&model, &env, 65536).unwrap();
            assert!(audit.counterexample.is_none());
            if kind == Kind::Workflow {
                assert_eq!(audit.reachable_pairs as u64, model.product_states());
            } else {
                assert_eq!(audit.reachable_pairs, env.states());
            }
            if kind == Kind::Coupled {
                assert_eq!(model.factor_states(), 12);
                assert_eq!(model.product_states(), 27);
                assert_eq!(audit.reachable_pairs, 9);
            }
        }
    }
}
struct Spy {
    resets: usize,
    steps: usize,
    wrong_shape: bool,
}
impl FrameResettable for Spy {
    fn reset(&mut self) -> Result<(), BehaviorError> {
        self.resets += 1;
        Ok(())
    }
    fn step(&mut self, _: u8) -> Result<Frame, BehaviorError> {
        self.steps += 1;
        Frame::new(if self.wrong_shape { &[1] } else { &[1, 2] })
    }
}
#[test]
fn frame_queries_preflight_authority_shape_word_and_all_quotas() {
    for (budget, authority, word) in [
        (FrameBudget::default(), Authority::from([]), vec![0]),
        (
            FrameBudget::default(),
            Authority::from([Effect::Simulate]),
            vec![2],
        ),
        (
            FrameBudget {
                max_steps: 1,
                ..FrameBudget::default()
            },
            Authority::from([Effect::Simulate]),
            vec![0, 0],
        ),
        (
            FrameBudget {
                max_cache_nodes: 1,
                ..FrameBudget::default()
            },
            Authority::from([Effect::Simulate]),
            vec![0, 0],
        ),
    ] {
        let mut c = FrameCache::new(
            Spy {
                resets: 0,
                steps: 0,
                wrong_shape: false,
            },
            2,
            2,
            budget,
            authority,
        )
        .unwrap();
        assert!(c.query(&word).is_err());
        assert_eq!(c.queries(), 0);
        assert_eq!(c.steps(), 0);
    }
    let mut c = FrameCache::new(
        Spy {
            resets: 0,
            steps: 0,
            wrong_shape: true,
        },
        2,
        2,
        FrameBudget::default(),
        Authority::from([Effect::Simulate]),
    )
    .unwrap();
    assert!(c.query(&[0, 1]).is_err());
    assert_eq!(c.steps(), 1);
    assert!(c.suspended());
    assert!(matches!(c.query(&[0]), Err(BehaviorError::Suspended)));
}
struct Mutating {
    rounds: u16,
}
impl FrameResettable for Mutating {
    fn reset(&mut self) -> Result<(), BehaviorError> {
        self.rounds += 1;
        Ok(())
    }
    fn step(&mut self, _: u8) -> Result<Frame, BehaviorError> {
        Frame::new(&[self.rounds, 2])
    }
}
#[test]
fn shared_cache_checks_fresh_replies_and_stops_at_the_first_inconsistent_prefix() {
    let mut c = FrameCache::new(
        Mutating { rounds: 0 },
        2,
        2,
        FrameBudget::default(),
        Authority::from([Effect::Simulate]),
    )
    .unwrap();
    c.query(&[0]).unwrap();
    c.query(&[0]).unwrap();
    assert_eq!(c.queries(), 1);
    assert!(matches!(
        c.fresh(&[0, 1]),
        Err(BehaviorError::NonDeterministic)
    ));
    assert_eq!(c.steps(), 2);
    assert!(c.suspended());
}
#[test]
fn invalid_projection_configuration_never_calls_the_oracle() {
    struct NoContact;
    impl FrameOracle for NoContact {
        fn query(&mut self, _: &[u8]) -> Result<FrameReply, BehaviorError> {
            panic!("contact");
        }
    }
    for (a, n, context) in [
        (0, 2, "ok"),
        (17, 2, "ok"),
        (2, 9, "ok"),
        (2, 0, "ok"),
        (2, 2, " bad"),
    ] {
        assert!(learn(a, n, context, &mut NoContact).is_err());
    }
    assert!(Frame::new(&[]).is_err());
    assert!(Frame::new(&[0; 9]).is_err());
}
#[test]
fn planner_distinguishes_impossible_goals_from_resource_exhaustion_and_identity_change() {
    let env = Environment::random(Kind::Cycles, 3, 719).unwrap();
    let model = fit(&env, "planning");
    let goal = model.goal(&env.goal_tokens()).unwrap();
    assert!(model.goal(&[]).is_err());
    assert!(model.goal(&[(0, 1), (0, 1)]).is_err());
    assert!(model.goal(&[(3, 1)]).is_err());
    let mut absent = 0;
    while model.factors()[0].access_sequences().iter().any(|prefix| {
        (0..model.actions()).any(|a| {
            model.factors()[0]
                .edge(model.factors()[0].state_after(prefix).unwrap(), a)
                .unwrap()
                .output
                == absent
        })
    }) {
        absent += 1;
    }
    let no_goal = model.goal(&[(0, absent)]).unwrap();
    assert!(
        model
            .plan(&[], &no_goal, PlanBudget::default())
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        model.plan(
            &[],
            &no_goal,
            PlanBudget {
                max_nodes: 1,
                max_depth: 48
            }
        ),
        Err(FactorError::Limit(_))
    ));
    assert!(matches!(
        model.plan(
            &[],
            &no_goal,
            PlanBudget {
                max_nodes: 65536,
                max_depth: 1
            }
        ),
        Err(FactorError::Limit(_))
    ));
    assert!(
        fit(&env, "other")
            .plan(&[], &goal, PlanBudget::default())
            .is_err()
    );
    let history = model
        .plan(&[], &goal, PlanBudget::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        model.plan(&history, &goal, PlanBudget::default()).unwrap(),
        Some(vec![])
    );
}
#[test]
fn tracker_checks_all_signals_transactionally_and_suspends_on_drift() {
    let env = Environment::random(Kind::Cycles, 3, 617).unwrap();
    let model = fit(&env, "track");
    let mut tracker = FactorTracker::new(&model, "track").unwrap();
    let expected = tracker.predict_next(0).unwrap();
    let mut bad = expected.values().to_vec();
    bad[1] ^= 1;
    assert!(matches!(
        tracker.observe(0, Frame::new(&bad).unwrap()),
        Err(FactorError::Drift { dimension: 1, .. })
    ));
    assert_eq!(tracker.states(), &[0; MAX_FACTORS]);
    assert_eq!(tracker.observations(), 0);
    assert!(tracker.suspended());
    assert!(tracker.predict_next(0).is_err());
    assert!(FactorTracker::new(&model, "other").is_err());
}
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "axon-factor-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn bundle_reuses_existing_catalog_integrity_context_and_no_clobber_contracts() {
    let t = Temp::new();
    let env = Environment::random(Kind::Cycles, 3, 319).unwrap();
    let model = fit(&env, "stored");
    let authority = Authority::from([Effect::ReadData, Effect::WriteData]);
    let bundle = t.0.join("bundle");
    assert!(
        model
            .save_bundle_new(&bundle, &Authority::from([]))
            .is_err()
    );
    assert!(!bundle.exists());
    model.save_bundle_new(&bundle, &authority).unwrap();
    let path = bundle.join("factors.axseed");
    assert_eq!(
        FactorModel::load_bundle(&path, "stored", &authority).unwrap(),
        model
    );
    assert!(FactorModel::load_bundle(&path, "other", &authority).is_err());
    assert!(FactorModel::load_bundle(&path, "stored", &Authority::from([])).is_err());
    assert!(model.save_bundle_new(&bundle, &authority).is_err());
    let file = bundle.join("factor-0.axbehavior");
    let mut b = fs::read(&file).unwrap();
    b[15] ^= 1;
    fs::write(&file, &b).unwrap();
    assert!(FactorModel::load_bundle(&path, "stored", &authority).is_err());
    assert_eq!(fs::read(file).unwrap(), b);
}
#[test]
fn execution_preplans_before_effects_and_publishes_four_actual_documents() {
    let t = Temp::new();
    let env = Environment::random(Kind::Workflow, 4, 71931)
        .unwrap()
        .zero_initial();
    let model = fit(&env, "execute");
    let goal = model.goal(&env.goal_tokens()).unwrap();
    let mut spy = Spy {
        resets: 0,
        steps: 0,
        wrong_shape: false,
    };
    assert!(
        model
            .execute_goal(
                &[],
                &goal,
                &mut spy,
                &Authority::from([Effect::Simulate]),
                PlanBudget::default()
            )
            .is_err()
    );
    assert_eq!(spy.resets, 0);
    let files = (0..4)
        .map(|i| {
            (
                format!("# Document {i}\n\nPublished from a joint learned plan.\n").into_bytes(),
                t.0.join(format!("doc-{i}.md")),
            )
        })
        .collect::<Vec<_>>();
    let mut physical = env.publishing(files.clone()).unwrap();
    let plan = model
        .execute_goal(
            &[],
            &goal,
            &mut physical,
            &Authority::from([Effect::Simulate, Effect::WriteData]),
            PlanBudget::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(plan.len(), 12);
    for (bytes, path) in files {
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
}
#[test]
fn execution_drift_does_not_execute_the_remainder_of_the_plan() {
    let env = Environment::random(Kind::Workflow, 2, 613719)
        .unwrap()
        .zero_initial();
    let model = fit(&env, "execute");
    let goal = model.goal(&env.goal_tokens()).unwrap();
    let mut spy = Spy {
        resets: 0,
        steps: 0,
        wrong_shape: false,
    };
    assert!(matches!(
        model.execute_goal(
            &[],
            &goal,
            &mut spy,
            &Authority::from([Effect::Simulate, Effect::WriteData]),
            PlanBudget::default()
        ),
        Err(FactorError::Drift { .. })
    ));
    assert_eq!(spy.steps, 1);
    assert_eq!(spy.resets, 1);
}

#[test]
fn public_cli_reloads_the_controller_and_publishes_new_payloads_without_relearning() {
    let t = Temp::new();
    let input = t.0.join("input");
    fs::create_dir(&input).unwrap();
    for i in 0..4 {
        fs::write(
            input.join(format!("{i}.md")),
            format!("# Original document {i}\n\nIndependent new content.\n"),
        )
        .unwrap();
    }
    let demo = t.0.join("demo");
    let report = axon_uic::cli::command(&[
        "factor-demo".into(),
        demo.display().to_string(),
        "--inputs".into(),
        input.display().to_string(),
    ])
    .unwrap();
    assert!(report.contains("audited_reachable_states=256"));
    assert!(report.contains("execution_actions=12"));
    for i in 0..4 {
        assert_eq!(
            fs::read(input.join(format!("{i}.md"))).unwrap(),
            fs::read(demo.join(format!("published/document-{i}.md"))).unwrap()
        );
    }
    for i in 0..4 {
        fs::write(
            input.join(format!("{i}.md")),
            format!("# Revised document {i}\n\nEntirely different content for reuse.\n"),
        )
        .unwrap();
    }
    let output = t.0.join("reused");
    let args = [
        "factor-publish".into(),
        demo.join("models/factors.axseed").display().to_string(),
        output.display().to_string(),
        "--inputs".into(),
        input.display().to_string(),
    ];
    let report = axon_uic::cli::command(&args).unwrap();
    assert!(report.contains("new_learning_queries=0"));
    for i in 0..4 {
        assert_eq!(
            fs::read(input.join(format!("{i}.md"))).unwrap(),
            fs::read(output.join(format!("document-{i}.md"))).unwrap()
        );
    }
    assert!(axon_uic::cli::command(&args).is_err());
}
#[test]
fn cli_rejects_invalid_inputs_and_mismatched_dimensions_before_creating_output() {
    let t = Temp::new();
    let input = t.0.join("input");
    fs::create_dir(&input).unwrap();
    for i in 0..2 {
        fs::write(input.join(format!("{i}.md")), "invalid").unwrap();
    }
    let demo = t.0.join("demo");
    assert!(
        axon_uic::cli::command(&[
            "factor-demo".into(),
            demo.display().to_string(),
            "--inputs".into(),
            input.display().to_string()
        ])
        .is_err()
    );
    assert!(!demo.exists());
    axon_uic::cli::command(&["factor-demo".into(), demo.display().to_string()]).unwrap();
    for i in 0..2 {
        fs::write(
            input.join(format!("{i}.md")),
            "# Valid input\n\nA properly titled document.\n",
        )
        .unwrap();
    }
    let output = t.0.join("reused");
    assert!(
        axon_uic::cli::command(&[
            "factor-publish".into(),
            demo.join("models/factors.axseed").display().to_string(),
            output.display().to_string(),
            "--inputs".into(),
            input.display().to_string()
        ])
        .is_err()
    );
    assert!(!output.exists());
}
