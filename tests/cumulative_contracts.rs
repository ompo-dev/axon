use axon_uic::{
    Authority, Effect, EpisodeStamp,
    cumulative::*,
    demand::{controller::*, genome::*, *},
    grounded_language::LanguageFrame,
    numeric::*,
};
use std::{
    fs,
    path::PathBuf,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "axon-cumulative-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
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
fn auth() -> Authority {
    Authority::from([Effect::ReadData, Effect::WriteData])
}
fn sources() -> &'static Vec<NumericSkill> {
    static S: OnceLock<Vec<NumericSkill>> = OnceLock::new();
    S.get_or_init(|| {
        (0..3)
            .map(|case| {
                NumericSkill::learn(
                    "opaque",
                    format!("past/{case}"),
                    [(2., 3.), (5., 7.), (-3., 4.), (0., 9.), (1.25, 8.)]
                        .into_iter()
                        .map(|(a, b)| {
                            NumericExample::new(
                                vec![a, b],
                                match case {
                                    0 => a + b,
                                    1 => a * b,
                                    _ => a - b,
                                },
                            )
                        })
                        .collect(),
                    &[],
                    NumericBudget::default(),
                    NumericTolerance::default(),
                )
                .unwrap()
            })
            .collect()
    })
}
fn frame() -> LanguageFrame {
    LanguageFrame::learn(&[
        (
            "pedido alpha 20 kits cada kit usa 3 pecas limite 30000 centavos".into(),
            "alpha".into(),
            "20,3,30000".into(),
        ),
        (
            "pedido beta 7 kits cada kit usa 5 pecas limite 15000 centavos".into(),
            "beta".into(),
            "7,5,15000".into(),
        ),
        (
            "pedido gamma 11 kits cada kit usa 2 pecas limite 40000 centavos".into(),
            "gamma".into(),
            "11,2,40000".into(),
        ),
    ])
    .unwrap()
}
fn seed() -> StudySeed {
    StudySeed::new(
        "study/v1",
        sources().clone(),
        vec![frame()],
        StudyPolicy::default(),
    )
    .unwrap()
}
#[test]
fn frames_and_study_seed_are_portable_bounded_and_reject_every_truncation() {
    let original = seed();
    let bytes = original.to_bytes().unwrap();
    let reopened = StudySeed::from_bytes(&bytes).unwrap();
    assert_eq!(original.fingerprint(), reopened.fingerprint());
    assert_eq!(
        reopened.frames()[0]
            .read_values("pedido delta 30 kits cada kit usa 4 pecas limite 55000 centavos")
            .unwrap(),
        vec![30., 4., 55000.]
    );
    for n in 0..bytes.len() {
        assert!(StudySeed::from_bytes(&bytes[..n]).is_err(), "prefix {n}");
    }
    for n in (0..bytes.len()).step_by(17) {
        let mut corrupt = bytes.clone();
        corrupt[n] ^= 1;
        assert!(StudySeed::from_bytes(&corrupt).is_err());
    }
    let child = original.offspring(5314).unwrap();
    child.check_descendant(&original).unwrap();
    assert_eq!(child.skills().len(), original.skills().len());
    assert_eq!(child.parent(), original.fingerprint());
    assert_ne!(child.fingerprint(), original.fingerprint());
    let t = Temp::new();
    assert!(
        original
            .save_new(
                &t.0.join("seed.axstudy"),
                &Authority::from([Effect::ReadData])
            )
            .is_err()
    );
    assert!(!t.0.join("seed.axstudy").exists());
}
#[test]
fn active_question_discriminates_and_restart_preserves_observations_without_oracle_calls() {
    let t = Temp::new();
    let seed = seed();
    let journal = t.0.join("individual.axep");
    let a = auth();
    let mut learner = Investigator::open(&seed, &journal, "task/opaque", 2, true, &a).unwrap();
    learner
        .observe(
            EpisodeStamp::new("teacher", "zero", 1),
            vec![0., 0.],
            Some(0.),
            &a,
        )
        .unwrap();
    assert_eq!(learner.survivors(), 3);
    let probe = learner
        .next_probe(&[vec![0., 0.], vec![1., 0.], vec![2., 3.]])
        .unwrap()
        .unwrap();
    assert_eq!(probe.input, vec![2., 3.]);
    assert!(probe.disagreement > 0);
    let stamp = EpisodeStamp::new("teacher", "next", 2);
    learner
        .observe(stamp.clone(), probe.input.clone(), Some(6.), &a)
        .unwrap();
    let before = learner.journal_bytes().unwrap();
    assert!(
        learner
            .observe(stamp, probe.input, Some(6.), &a)
            .unwrap()
            .duplicate
    );
    assert_eq!(learner.journal_bytes().unwrap(), before);
    assert_eq!(learner.observations(), 2);
    let acquired = learner.acquire().unwrap();
    assert!(acquired.borrowed);
    assert_eq!(
        acquired
            .skill
            .predict(acquired.skill.context(), &[7., 9.])
            .unwrap(),
        63.
    );
    drop(learner);
    let reopened = Investigator::open(&seed, &journal, "task/opaque", 2, true, &a).unwrap();
    assert_eq!(reopened.observations(), 2);
    assert_eq!(reopened.survivors(), 1);
    assert_eq!(
        reopened.acquire().unwrap().skill.fingerprint(),
        sources()[1].fingerprint()
    );
    drop(reopened);
    let wrong = seed.offspring(1).unwrap();
    assert!(Investigator::open(&wrong, &journal, "task/opaque", 2, true, &a).is_err());
}
#[test]
fn contradictory_readings_are_preserved_and_no_longer_produce_a_prediction() {
    let t = Temp::new();
    let seed = seed();
    let a = auth();
    let mut learner = Investigator::open(&seed, &t.0.join("m"), "task", 2, true, &a).unwrap();
    learner
        .observe(EpisodeStamp::new("s", "one", 1), vec![2., 3.], Some(6.), &a)
        .unwrap();
    learner
        .observe(EpisodeStamp::new("s", "two", 2), vec![2., 3.], Some(8.), &a)
        .unwrap();
    assert!(learner.conflicted());
    assert!(matches!(learner.acquire(), Err(StudyError::Conflict)));
    assert_eq!(learner.observations(), 2);
}
#[test]
fn operational_genome_inherits_language_programs_policy_and_checked_descendants() {
    let t = Temp::new();
    let a = auth();
    let parent = seed();
    parent.save_new(&t.0.join("parent.axstudy"), &a).unwrap();
    let mut catalog = Catalog::new(&t.0);
    catalog
        .register(
            ModuleSpec {
                id: "learning",
                kind: ModuleKind::Study,
                file: "parent.axstudy",
                input: "study",
                output: "study",
                cost: 1,
            },
            &a,
        )
        .unwrap();
    let mut runtime = DemandRuntime::new(
        catalog,
        Seed {
            active_bytes: 262144,
            active_modules: 2,
            ..Seed::default()
        },
    )
    .unwrap();
    let genome = Genome::capture(
        "whole/v1",
        &mut runtime,
        CompositionBudget::default(),
        &[],
        262144,
        &a,
    )
    .unwrap();
    let child = parent.offspring(9123).unwrap();
    child.save_new(&t.0.join("child.axstudy"), &a).unwrap();
    let mut child_catalog = Catalog::new(&t.0);
    child_catalog
        .register(
            ModuleSpec {
                id: "learning",
                kind: ModuleKind::Study,
                file: "child.axstudy",
                input: "study",
                output: "study",
                cost: 1,
            },
            &a,
        )
        .unwrap();
    let mut child_runtime = DemandRuntime::new(child_catalog, runtime.seed()).unwrap();
    let descendant = genome.inherit_runtime(&mut child_runtime, &[], &a).unwrap();
    assert_eq!(descendant.parent(), genome.fingerprint());
    let number = descendant.number();
    let decoded = Genome::from_number(&number).unwrap();
    let mut born = decoded
        .birth(&t.0.join("birth"), BirthLimits::default(), &a)
        .unwrap();
    fs::rename(t.0.join("parent.axstudy"), t.0.join("parent-removed")).unwrap();
    fs::rename(t.0.join("child.axstudy"), t.0.join("child-removed")).unwrap();
    let loaded = born
        .controller
        .runtime_mut()
        .with_module("learning", "study/v1", &a, |m| match m {
            Module::Study(s) => Ok(s.clone()),
            _ => panic!(),
        })
        .unwrap();
    assert_eq!(loaded.fingerprint(), child.fingerprint());
    assert_eq!(
        loaded.frames()[0]
            .read_values("pedido new 13 kits cada kit usa 4 pecas limite 1800 centavos")
            .unwrap(),
        vec![13., 4., 1800.]
    );
}

#[test]
fn inherited_composition_survives_birth_and_transfers_under_the_same_cold_search_ceiling() {
    let t = Temp::new();
    let mut rng = axon_uic::behavior_lab::Rng::new(13719483);
    let rows = |n: usize, rng: &mut axon_uic::behavior_lab::Rng| {
        (0..12)
            .map(|_| {
                (0..n)
                    .map(|_| (rng.pick(201) as f64 - 100.) / 17.)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let two = NumericSkill::learn(
        "opaque",
        "past/two",
        rows(2, &mut rng)
            .into_iter()
            .map(|x| {
                let y = x[0].hypot(x[1]);
                NumericExample::new(x, y)
            })
            .collect(),
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap();
    let three = NumericSkill::learn(
        "opaque",
        "past/three",
        rows(3, &mut rng)
            .into_iter()
            .map(|x| {
                let y = x[0].hypot(x[1]).hypot(x[2]);
                NumericExample::new(x, y)
            })
            .collect(),
        &[&two],
        NumericBudget {
            max_cost: 5,
            ..NumericBudget::default()
        },
        NumericTolerance::default(),
    )
    .unwrap();
    let policy = StudyPolicy {
        search: NumericBudget {
            max_cost: 6,
            ..StudyPolicy::default().search
        },
        ..StudyPolicy::default()
    };
    let initial = StudySeed::new("study/transfer", vec![two], vec![frame()], policy).unwrap();
    let grown = initial.consolidate(&[three]).unwrap();
    grown.check_descendant(&initial).unwrap();
    let grown = StudySeed::from_bytes(&grown.to_bytes().unwrap()).unwrap();
    let teacher = rows(4, &mut rng);
    for inherit in [false, true] {
        let mut agent = Investigator::open(
            &grown,
            &t.0.join(format!("{inherit}.axep")),
            "unknown/new",
            4,
            inherit,
            &auth(),
        )
        .unwrap();
        for (i, x) in teacher.iter().enumerate().take(8) {
            let y = x[0].hypot(x[1]).hypot(x[2]) / x[3];
            agent
                .observe(
                    EpisodeStamp::new("t", format!("q-{i}"), i as u64),
                    x.clone(),
                    Some(y),
                    &auth(),
                )
                .unwrap();
        }
        let result = agent.acquire();
        if !inherit {
            assert!(result.is_err());
            continue;
        }
        let result = result.unwrap();
        assert!(!result.borrowed);
        assert!(result.search_work <= policy.total_work);
        for _ in 0..1000 {
            let mut x = (0..4)
                .map(|_| (rng.pick(20001) as f64 - 10000.) / 31.)
                .collect::<Vec<_>>();
            if x[3] == 0. {
                x[3] = 0.25;
            }
            let expected = x[0].hypot(x[1]).hypot(x[2]) / x[3];
            assert!(NumericTolerance::default().agrees(
                result.skill.predict(result.skill.context(), &x).unwrap(),
                expected
            ));
        }
        assert!(
            result
                .skill
                .predict(result.skill.context(), &[1., 2., 3., 0.])
                .is_err()
        );
    }
}

#[test]
fn duplicate_at_question_limit_is_idempotent_and_conflict_survives_reopening() {
    let t = Temp::new();
    let policy = StudyPolicy {
        questions: 1,
        ..StudyPolicy::default()
    };
    let seed = seed().with_policy(policy).unwrap();
    let stamp = EpisodeStamp::new("t", "only", 1);
    let mut agent =
        Investigator::open(&seed, &t.0.join("one.axep"), "one", 2, true, &auth()).unwrap();
    agent
        .observe(stamp.clone(), vec![2., 3.], Some(6.), &auth())
        .unwrap();
    let bytes = agent.journal_bytes().unwrap();
    assert!(
        agent
            .observe(stamp, vec![2., 3.], Some(6.), &auth())
            .unwrap()
            .duplicate
    );
    assert_eq!(agent.journal_bytes().unwrap(), bytes);
    assert!(matches!(
        agent.observe(
            EpisodeStamp::new("t", "extra", 2),
            vec![1., 4.],
            Some(4.),
            &auth()
        ),
        Err(StudyError::Budget(_))
    ));
    assert!(
        Investigator::open(
            &seed,
            &t.0.join("denied.axep"),
            "one",
            2,
            true,
            &Authority::from([Effect::ReadData])
        )
        .is_err()
    );
    assert!(!t.0.join("denied.axep").exists());
    drop(agent);
    let seed = seed.with_policy(StudyPolicy::default()).unwrap();
    let path = t.0.join("conflict.axep");
    let mut agent = Investigator::open(&seed, &path, "changed", 2, true, &auth()).unwrap();
    agent
        .observe(
            EpisodeStamp::new("t", "before", 1),
            vec![2., 3.],
            Some(6.),
            &auth(),
        )
        .unwrap();
    agent
        .observe(
            EpisodeStamp::new("t", "after", 2),
            vec![2., 3.],
            Some(7.),
            &auth(),
        )
        .unwrap();
    drop(agent);
    let restored = Investigator::open(&seed, &path, "changed", 2, true, &auth()).unwrap();
    assert!(restored.conflicted());
    assert!(matches!(restored.acquire(), Err(StudyError::Conflict)));
}

#[test]
fn unrelated_or_forgetting_seed_cannot_replace_a_learning_module() {
    let parent = seed();
    let unrelated = StudySeed::new(
        parent.context(),
        sources()[..2].to_vec(),
        vec![frame()],
        StudyPolicy::default(),
    )
    .unwrap();
    assert!(unrelated.check_descendant(&parent).is_err());
    let wrong_frame = LanguageFrame::learn(&[
        ("alpha com 2 e 3".into(), "alpha".into(), "2,3".into()),
        ("beta com 5 e 7".into(), "beta".into(), "5,7".into()),
        ("gamma com 9 e 4".into(), "gamma".into(), "9,4".into()),
    ])
    .unwrap();
    let forged = StudySeed::new(
        parent.context(),
        sources().clone(),
        vec![wrong_frame],
        StudyPolicy::default(),
    )
    .unwrap()
    .offspring(8)
    .unwrap();
    assert!(forged.check_descendant(&parent).is_err());
}

#[test]
fn synthesized_program_must_respect_domain_observations_not_only_finite_examples() {
    let t = Temp::new();
    let seed = seed();
    for inherit in [false, true] {
        let mut learner = Investigator::open(
            &seed,
            &t.0.join(format!("domain-{inherit}.axep")),
            "new/domain",
            2,
            inherit,
            &auth(),
        )
        .unwrap();
        for (i, (input, actual)) in [
            (vec![2., 3.], Some(6.)),
            (vec![4., 5.], Some(20.)),
            (vec![-3., 4.], Some(-12.)),
            (vec![0., 9.], None),
        ]
        .into_iter()
        .enumerate()
        {
            learner
                .observe(
                    EpisodeStamp::new("teacher", format!("q-{i}"), i as u64),
                    input,
                    actual,
                    &auth(),
                )
                .unwrap();
        }
        match learner.acquire() {
            Err(StudyError::Evidence { search_work }) => assert!(search_work > 0),
            _ => panic!(
                "product fits positive examples but violates the explicit domain observation"
            ),
        }
    }
}
