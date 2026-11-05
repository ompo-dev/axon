use axon_uic::reactive::{agent::*, *};
use std::{
    fs,
    path::PathBuf,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

fn traces<F: FnMut(i64) -> i64>(x: &[i64], mut f: F) -> Vec<ReactiveTrace> {
    vec![ReactiveTrace::new(
        x.iter()
            .map(|x| ReactiveEvent::new(vec![*x], f(*x)))
            .collect(),
    )]
}
fn streak() -> &'static ReactiveSkill {
    static SKILL: OnceLock<ReactiveSkill> = OnceLock::new();
    SKILL.get_or_init(|| {
        let mut state = 0;
        ReactiveSkill::learn(
            "source",
            "signal/streak",
            traces(
                &[0, 1, 2, 0, 1, 1, 0, 2, -1, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0],
                |x| {
                    state = if x > 0 { state + 1 } else { 0 };
                    state
                },
            ),
            &[],
            ReactiveBudget::default(),
        )
        .unwrap()
    })
}
fn peak() -> ReactiveSkill {
    let mut current = 0;
    let mut best = 0;
    ReactiveSkill::learn(
        "composed",
        "signal/peak",
        traces(
            &[0, 1, 2, 0, 1, 1, 0, 2, -1, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0],
            |x| {
                current = if x > 0 { current + 1 } else { 0 };
                best = best.max(current);
                best
            },
        ),
        &[streak()],
        ReactiveBudget::default(),
    )
    .unwrap()
}
fn replay<F: FnMut(i64) -> i64>(
    skill: &ReactiveSkill,
    x: impl IntoIterator<Item = i64>,
    mut truth: F,
) {
    let circuit = skill.compile().unwrap();
    let mut cursor = circuit.cursor();
    for x in x {
        assert_eq!(
            circuit.step(&mut cursor, skill.context(), &[x]).unwrap(),
            truth(x),
            "{} at {x}",
            skill.description()
        );
    }
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "axon-reactive-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        if let (Ok(p), Ok(root)) = (self.0.canonicalize(), std::env::temp_dir().canonicalize()) {
            assert!(p.starts_with(root));
            let _ = fs::remove_dir_all(p);
        }
    }
}
#[test]
fn conditional_boundary_and_branches_are_induced_not_selected_by_task_name() {
    let x = [-5, -2, 0, 1, 2, 3, 4, 7, -1, 6];
    let examples = traces(&x, |x| if x < 2 { x * x } else { x + x });
    let a = ReactiveSkill::learn(
        "unknown-a",
        "case/a",
        examples.clone(),
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    let b = ReactiveSkill::learn(
        "unrelated-name",
        "case/b",
        examples,
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    assert_eq!(a.description(), b.description());
    assert_eq!(a.memory_registers(), 0);
    replay(&a, -25..=25, |x| if x < 2 { x * x } else { x + x });
}
#[test]
fn an_unobserved_register_and_readout_phase_are_discovered_for_delay() {
    let mut before = 0;
    let a = ReactiveSkill::learn(
        "delay",
        "case/delay",
        traces(&[1, 1, 0, 2, 1, 0, 0, 3, -1, 2, 0, 1], |x| {
            let y = before;
            before = x;
            y
        }),
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    assert_eq!(a.memory_registers(), 1);
    let mut before = 0;
    replay(&a, (0..1000).map(|i| i * 13 % 31 - 15), |x| {
        let y = before;
        before = x;
        y
    });
}
#[test]
fn conditional_recurrence_generalizes_to_new_values_and_much_longer_streams() {
    let mut state = 0;
    replay(streak(), (0..20000).map(|i| i * 37 % 101 - 40), |x| {
        state = if x > 0 { state + 1 } else { 0 };
        state
    });
    assert_eq!(streak().memory_registers(), 1);
}
#[test]
fn a_new_procedure_composes_acquired_memory_with_another_register() {
    let a = peak();
    assert_eq!(a.memory_registers(), 2);
    assert_eq!(a.cells(), 2);
    assert_eq!(a.dependencies(), vec![streak().fingerprint()]);
    let mut state = 0;
    let mut best = 0;
    replay(
        &a,
        (0..20000).map(|i| if i % 113 < 90 { i % 7 + 1 } else { -1 }),
        |x| {
            state = if x > 0 { state + 1 } else { 0 };
            best = best.max(state);
            best
        },
    );
    assert!(a.to_bytes().len() < 16384);
}
#[test]
fn demonstration_resets_are_preserved_instead_of_concatenating_histories() {
    let source = streak();
    let mut examples = source.traces().to_vec();
    examples.extend(source.traces().iter().cloned());
    let a = ReactiveSkill::learn(
        "reset",
        "case/reset",
        examples,
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    let mut state = 0;
    replay(&a, [2, 2, 0, 1, -2, 1], |x| {
        state = if x > 0 { state + 1 } else { 0 };
        state
    });
}
#[test]
fn overflowing_steps_do_not_commit_even_a_partially_updated_circuit() {
    let mut total = 0;
    let a = ReactiveSkill::learn(
        "sum",
        "case/overflow",
        traces(&[1, 1, 2, -1, 0, 3, -2, 1], |x| {
            total += x;
            total
        }),
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    let circuit = a.compile().unwrap();
    let mut cursor = circuit.cursor();
    assert_eq!(
        circuit.step(&mut cursor, a.context(), &[i64::MAX]).unwrap(),
        i64::MAX
    );
    let before = cursor.clone();
    assert!(circuit.step(&mut cursor, a.context(), &[1]).is_err());
    assert_eq!(cursor, before);
}
#[test]
fn untaken_branches_are_not_evaluated() {
    let a = ReactiveSkill::learn(
        "lazy",
        "case/lazy",
        traces(&[-3, -2, -1, 0, 1, 2, 3], |x| if x < 0 { x * x } else { x }),
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    replay(&a, [i64::MAX], |x| x);
}
#[test]
fn invalid_schema_context_and_cursor_identity_never_change_state() {
    let a = streak();
    let c = a.compile().unwrap();
    let mut s = c.cursor();
    let before = s.clone();
    assert!(c.step(&mut s, "wrong", &[1]).is_err());
    assert!(c.step(&mut s, a.context(), &[1, 2]).is_err());
    assert_eq!(s, before);
    let other = peak().compile().unwrap();
    assert!(other.step(&mut s, "signal/peak", &[1]).is_err());
    assert_eq!(s, before);
    assert!(ReactiveCursor::from_bytes(&s.to_bytes(), &other, "signal/peak").is_err());
}
#[test]
fn capsules_and_cursors_reopen_with_the_same_future_behavior_and_no_clobber() {
    let temp = Temp::new();
    let a = peak();
    let model = temp.0.join("skill.axreact");
    a.save_new(&model).unwrap();
    let bytes = fs::read(&model).unwrap();
    assert!(a.save_new(&model).is_err());
    assert_eq!(fs::read(&model).unwrap(), bytes);
    let c = a.compile().unwrap();
    let mut s = c.cursor();
    for x in [1, 1, 1, 0, 1] {
        c.step(&mut s, a.context(), &[x]).unwrap();
    }
    let path = temp.0.join("state.axrc");
    s.save_new(&path).unwrap();
    assert!(s.save_new(&path).is_err());
    let b = ReactiveSkill::open(&model, a.context()).unwrap();
    let d = b.compile().unwrap();
    let mut t = ReactiveCursor::open(&path, &d, b.context()).unwrap();
    for x in [1, 1, 1, 1, 0, 2, 2, -1] {
        assert_eq!(
            c.step(&mut s, a.context(), &[x]).unwrap(),
            d.step(&mut t, b.context(), &[x]).unwrap()
        );
    }
}
#[test]
fn every_truncation_and_byte_corruption_is_refused() {
    let a = streak();
    let bytes = a.to_bytes();
    for i in 0..bytes.len() {
        assert!(ReactiveSkill::from_bytes(&bytes[..i]).is_err());
        let mut bad = bytes.clone();
        bad[i] ^= 1;
        assert!(ReactiveSkill::from_bytes(&bad).is_err());
    }
    let c = a.compile().unwrap();
    let bytes = c.cursor().to_bytes();
    for i in 0..bytes.len() {
        assert!(ReactiveCursor::from_bytes(&bytes[..i], &c, a.context()).is_err());
    }
}
#[test]
fn search_quotas_stop_without_returning_a_fitted_but_unverified_program() {
    let data = traces(&[0, 1, 0, 1, 0, 1, 0, 1], |x| x + 1);
    assert!(matches!(
        ReactiveSkill::learn(
            "budget",
            "case/budget",
            data,
            &[],
            ReactiveBudget {
                max_work: 1,
                ..ReactiveBudget::default()
            }
        ),
        Err(ReactiveError::Budget(_))
    ));
    assert!(
        ReactiveSkill::learn(
            "bad",
            "case/bad",
            traces(&[1, 2], |x| x),
            &[],
            ReactiveBudget::default()
        )
        .is_err()
    );
    let bad = vec![ReactiveTrace::new(vec![
        ReactiveEvent::new(vec![1, 2, 3], 4);
        8
    ])];
    assert!(ReactiveSkill::learn("bad", "case/bad", bad, &[], ReactiveBudget::default()).is_err());
}
#[test]
fn a_refuted_cursor_suspends_instead_of_silently_overwriting_its_theory() {
    let a = streak();
    let c = a.compile().unwrap();
    let mut s = c.cursor();
    let p = c.step(&mut s, a.context(), &[1]).unwrap();
    assert!(matches!(
        c.feedback(&mut s, p, p + 1),
        Err(ReactiveError::Drift { .. })
    ));
    assert!(s.suspended());
    let before = s.clone();
    assert!(c.step(&mut s, a.context(), &[0]).is_err());
    assert_eq!(s, before);
}
fn warm(a: &mut ReactiveAgent) -> u64 {
    let mut state = 0;
    for id in 1..=64 {
        let x = a
            .suggest_probe()
            .map(|x| x[0])
            .unwrap_or([0, 1, 2, 0, 1, 1, 0, 2][(id as usize - 1) % 8]);
        state = if x > 0 { state + 1 } else { 0 };
        let p = a.observe(id, &[x]).unwrap();
        if a.phase() != ReactivePhase::Active {
            assert_eq!(p.value, None);
        }
        a.feedback(id, state).unwrap();
        if a.phase() == ReactivePhase::Active {
            return id;
        }
    }
    panic!("did not admit")
}
#[test]
fn autonomous_acquisition_requires_future_predictions_and_labels() {
    let mut a = ReactiveAgent::new("agent/live", 1, Vec::new(), ReactiveBudget::default()).unwrap();
    let admitted = warm(&mut a);
    assert!(admitted >= 16);
    assert_eq!(a.statistics().promotions, 1);
    assert_eq!(a.skill().unwrap().memory_registers(), 1);
    let id = admitted + 1;
    a.observe(id, &[0]).unwrap();
    assert_eq!(a.feedback(id, 0).unwrap(), ReactiveFeedback::Accepted);
    assert_eq!(a.statistics().scored, 1);
    assert_eq!(a.statistics().correct, 1);
    assert_eq!(a.feedback(id, 0).unwrap(), ReactiveFeedback::Duplicate);
    a.observe(id + 1, &[1]).unwrap();
    assert_eq!(a.feedback(id + 1, 55).unwrap(), ReactiveFeedback::Drift);
    assert_eq!(a.phase(), ReactivePhase::Suspended);
}
#[test]
fn future_counterexamples_refute_a_shadow_candidate_without_training_on_its_prediction() {
    let mut a =
        ReactiveAgent::new("agent/refute", 1, Vec::new(), ReactiveBudget::default()).unwrap();
    for id in 1..=8 {
        a.observe(id, &[id as i64]).unwrap();
        a.feedback(id, id as i64 + 1).unwrap();
    }
    assert_eq!(a.phase(), ReactivePhase::Validating);
    a.observe(9, &[9]).unwrap();
    assert_eq!(a.feedback(9, 300).unwrap(), ReactiveFeedback::Rejected);
    assert_eq!(a.statistics().promotions, 0);
}
#[test]
fn agent_checkpoint_preserves_identity_arbitrary_event_ids_scores_and_acknowledgement() {
    let temp = Temp::new();
    let path = temp.0.join("agent.axra");
    let mut a =
        ReactiveAgent::new("agent/resume", 1, Vec::new(), ReactiveBudget::default()).unwrap();
    let admitted = warm(&mut a);
    a.observe(1000, &[0]).unwrap();
    assert!(a.checkpoint_new(&path).is_err());
    a.feedback(1000, 0).unwrap();
    a.checkpoint_new(&path).unwrap();
    let bytes = fs::read(&path).unwrap();
    assert!(a.checkpoint_new(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    let mut b = ReactiveAgent::resume("agent/resume", &path).unwrap();
    assert_eq!(a.statistics(), b.statistics());
    assert_eq!(b.cursor().unwrap().ticks(), admitted + 1);
    assert_eq!(b.feedback(1000, 0).unwrap(), ReactiveFeedback::Duplicate);
    assert_eq!(
        a.observe(1001, &[1]).unwrap(),
        b.observe(1001, &[1]).unwrap()
    );
    a.feedback(1001, 1).unwrap();
    b.feedback(1001, 1).unwrap();
    assert_eq!(a.statistics(), b.statistics());
}
#[test]
fn missing_feedback_and_input_errors_cannot_trigger_learning() {
    let mut a =
        ReactiveAgent::new("agent/errors", 1, Vec::new(), ReactiveBudget::default()).unwrap();
    assert!(a.observe(1, &[1, 2]).is_err());
    assert!(a.feedback(1, 1).is_err());
    assert_eq!(a.statistics().observations, 0);
    a.observe(1, &[1]).unwrap();
    let before = a.statistics();
    assert!(a.observe(2, &[2]).is_err());
    assert!(a.feedback(2, 2).is_err());
    assert_eq!(a.statistics(), before);
    assert_eq!(before.searches, 0);
}

#[test]
fn shared_acquired_cells_execute_once_in_a_composed_circuit() {
    let peak = peak();
    let mut duration = 0;
    let mut max = 0;
    let skill = ReactiveSkill::learn(
        "gap",
        "signal/gap",
        traces(
            &[0, 1, 2, 0, 1, 1, 0, 2, -1, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0],
            |x| {
                duration = if x > 0 { duration + 1 } else { 0 };
                max = max.max(duration);
                max - duration
            },
        ),
        &[streak(), &peak],
        ReactiveBudget::default(),
    )
    .unwrap();
    let circuit = skill.compile().unwrap();
    assert_eq!(circuit.memory_registers(), 2);
    assert_eq!(circuit.cells(), 3);
    let mut duration = 0;
    let mut max = 0;
    replay(
        &skill,
        (0..1000).map(|i| if i % 47 < 30 { 1 } else { 0 }),
        |x| {
            duration = if x > 0 { duration + 1 } else { 0 };
            max = max.max(duration);
            max - duration
        },
    );
}

#[test]
fn repetitive_future_validation_can_still_admit_a_wrong_hypothesis_and_drift_is_visible() {
    let mut a =
        ReactiveAgent::new("agent/ambiguity", 1, Vec::new(), ReactiveBudget::default()).unwrap();
    let mut duration = 0;
    let mut id = 0;
    while a.phase() != ReactivePhase::Active {
        id += 1;
        assert!(id <= 64);
        let x = [0, 1, 2, 0, 1, 1, 0, 2][(id as usize - 1) % 8];
        duration = if x > 0 { duration + 1 } else { 0 };
        a.observe(id, &[x]).unwrap();
        a.feedback(id, duration).unwrap();
    }
    a.observe(id + 1, &[0]).unwrap();
    assert_eq!(a.feedback(id + 1, 0).unwrap(), ReactiveFeedback::Drift);
    assert_eq!(a.statistics().scored - a.statistics().correct, 1);
}

fn resign(bytes: &mut [u8]) {
    let end = bytes.len() - 8;
    let hash = bytes[..end].iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    bytes[end..].copy_from_slice(&hash.to_le_bytes());
}
#[test]
fn checksums_do_not_replace_search_cursor_or_admission_semantic_validation() {
    let mut bytes = streak().to_bytes();
    let end = bytes.len() - 8;
    bytes[end - 40..end - 32].copy_from_slice(&0u64.to_le_bytes());
    resign(&mut bytes);
    assert!(ReactiveSkill::from_bytes(&bytes).is_err());
    let circuit = streak().compile().unwrap();
    let mut bytes = circuit.cursor().to_bytes();
    bytes[41..49].copy_from_slice(&1i64.to_le_bytes());
    resign(&mut bytes);
    assert!(ReactiveCursor::from_bytes(&bytes, &circuit, streak().context()).is_err());
    let temp = Temp::new();
    let path = temp.0.join("proof.axra");
    let mut a = ReactiveAgent::new("proof", 1, Vec::new(), ReactiveBudget::default()).unwrap();
    warm(&mut a);
    a.checkpoint_new(&path).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"AXRA0002");
    bytes[136..144].copy_from_slice(&777i64.to_le_bytes());
    resign(&mut bytes);
    fs::write(&path, bytes).unwrap();
    assert!(ReactiveAgent::resume("proof", &path).is_err());
}
#[test]
fn a_parent_guard_failure_rolls_back_already_evaluated_acquired_child_state() {
    let x = [1, 2, -1, 0, 3, 1, -2, 1];
    let mut sum = 0;
    let base = ReactiveSkill::learn(
        "base",
        "atomic/base",
        traces(&x, |x| {
            sum += x;
            sum
        }),
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    let mut sum = 0;
    let parent = ReactiveSkill::learn(
        "parent",
        "atomic/parent",
        traces(&x, |x| {
            sum += x;
            sum + sum
        }),
        &[&base],
        ReactiveBudget::default(),
    )
    .unwrap();
    assert_eq!(parent.dependencies(), vec![base.fingerprint()]);
    let circuit = parent.compile().unwrap();
    let mut cursor = circuit.cursor();
    let before = cursor.clone();
    assert!(
        circuit
            .step(&mut cursor, parent.context(), &[i64::MAX])
            .is_err()
    );
    assert_eq!(cursor, before);
}

#[test]
fn authorized_probes_explore_new_values_without_creating_observations_or_authority() {
    let mut a = ReactiveAgent::new("probes", 1, Vec::new(), ReactiveBudget::default()).unwrap();
    assert!(a.suggest_probe_with_bounds(&[]).is_err());
    assert!(a.suggest_probe_with_bounds(&[(2, 1)]).is_err());
    assert_eq!(a.suggest_probe_with_bounds(&[(-32, 32)]).unwrap(), None);
    for (i, x) in [0, 1, 2, 3, -1, 0, 1, 4].into_iter().enumerate() {
        a.observe(i as u64 + 1, &[x]).unwrap();
        a.feedback(i as u64 + 1, x).unwrap();
    }
    let before = a.statistics();
    assert_eq!(
        a.suggest_probe_with_bounds(&[(-32, 32)]).unwrap(),
        Some(vec![-32])
    );
    assert_eq!(
        a.suggest_probe_with_bounds(&[(7, 8)]).unwrap(),
        Some(vec![7])
    );
    for _ in 0..10 {
        let x = a
            .suggest_probe_with_bounds(&[(i64::MIN, i64::MAX)])
            .unwrap()
            .unwrap();
        assert_eq!(x.len(), 1);
        assert_eq!(a.statistics(), before);
    }
    a.observe(9, &[5]).unwrap();
    assert_eq!(a.suggest_probe_with_bounds(&[(-32, 32)]).unwrap(), None);
}

#[test]
fn distracting_acquired_tools_are_refuted_and_pruned_by_discriminating_experience() {
    let inputs = [0, 1, 2, 3, -1, 0, 1, 4, -3, 2, 1, 0, 2, 2, -2, 3, 1, 0];
    let conditional = ReactiveSkill::learn(
        "a",
        "pool/a",
        traces(&inputs, |x| if x < 2 { x * x } else { x + x }),
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    let mut previous = 0;
    let delay = ReactiveSkill::learn(
        "b",
        "pool/b",
        traces(&inputs, |x| {
            let y = previous;
            previous = x;
            y
        }),
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    let mut total = 0;
    let sum = ReactiveSkill::learn(
        "c",
        "pool/c",
        traces(&inputs, |x| {
            total += x;
            total
        }),
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    let mut a = ReactiveAgent::new(
        "pool/target",
        1,
        vec![conditional, delay, sum, streak().clone()],
        ReactiveBudget::default(),
    )
    .unwrap();
    let mut duration = 0;
    let mut best = 0;
    for id in 1..=64 {
        let x = if id <= 8 {
            inputs[id - 1]
        } else {
            a.suggest_probe_with_bounds(&[(-32, 32)]).unwrap().unwrap()[0]
        };
        duration = if x > 0 { duration + 1 } else { 0 };
        best = best.max(duration);
        a.observe(id as u64, &[x]).unwrap();
        a.feedback(id as u64, best).unwrap();
        if a.phase() == ReactivePhase::Active {
            break;
        }
    }
    assert_eq!(a.phase(), ReactivePhase::Active);
    assert!(a.statistics().rejected_candidates > 0);
    assert_eq!(
        a.skill().unwrap().dependencies(),
        vec![streak().fingerprint()]
    );
    let mut duration = 0;
    let mut best = 0;
    replay(
        a.skill().unwrap(),
        (0..10000).map(|i| if i % 127 < 110 { 33 + i % 71 } else { -i % 31 }),
        |x| {
            duration = if x > 0 { duration + 1 } else { 0 };
            best = best.max(duration);
            best
        },
    );
}

#[test]
fn two_input_predicates_do_not_require_a_named_task_or_single_channel_schema() {
    let events = (-4..=4)
        .flat_map(|x| {
            (-4..=4).map(move |y| ReactiveEvent::new(vec![x, y], if x < y { x + y } else { x * y }))
        })
        .collect();
    let skill = ReactiveSkill::learn(
        "unknown",
        "two/channels",
        vec![ReactiveTrace::new(events)],
        &[],
        ReactiveBudget {
            max_ticks: 64,
            ..ReactiveBudget::default()
        },
    );
    // One trace cannot silently exceed its quota.
    assert!(skill.is_err());
    let events = (-3..=3)
        .flat_map(|x| {
            (-3..=3).map(move |y| ReactiveEvent::new(vec![x, y], if x < y { x + y } else { x * y }))
        })
        .collect();
    let skill = ReactiveSkill::learn(
        "unknown",
        "two/channels",
        vec![ReactiveTrace::new(events)],
        &[],
        ReactiveBudget::default(),
    )
    .unwrap();
    let c = skill.compile().unwrap();
    let mut s = c.cursor();
    for x in -20..=20 {
        for y in -20..=20 {
            assert_eq!(
                c.step(&mut s, skill.context(), &[x, y]).unwrap(),
                if x < y { x + y } else { x * y }
            );
        }
    }
    let mut a = ReactiveAgent::new("two/probes", 2, Vec::new(), ReactiveBudget::default()).unwrap();
    for i in 1..=8 {
        a.observe(i, &[i as i64, -(i as i64)]).unwrap();
        a.feedback(i, 0).unwrap();
    }
    let before = a.statistics();
    let p = a
        .suggest_probe_with_bounds(&[(0, 3), (-3, 0)])
        .unwrap()
        .unwrap();
    assert!((0..=3).contains(&p[0]) && (-3..=0).contains(&p[1]));
    assert_eq!(a.statistics(), before);
}
