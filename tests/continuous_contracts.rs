use axon_uic::numeric::{continuous::*, *};
fn agent() -> ContinuousAgent {
    ContinuousAgent::new("sensor/test", 1, None, ContinuousPolicy::default()).unwrap()
}
fn input(id: u64) -> f64 {
    ((id * 17) % 97) as f64 / 7.0 - 7.0
}
fn step(a: &mut ContinuousAgent, id: u64, f: impl Fn(f64) -> f64) -> FeedbackReceipt {
    let x = input(id);
    a.observe(id, &[x]).unwrap();
    let r = a.feedback(id, f(x)).unwrap();
    if let Some(job) = a.take_learning_job() {
        a.submit_learning_result(job.run(None));
    }
    r
}
fn warm(a: &mut ContinuousAgent) {
    for id in 1..=30 {
        step(a, id, |x| 2.0 * x + 1.0);
    }
    assert_eq!(a.phase(), ContinuousPhase::Active);
}

#[test]
fn promotion_requires_future_labels_and_does_not_wait_for_the_end_of_a_stream() {
    let mut a = agent();
    for id in 1..=8 {
        step(&mut a, id, |x| 2.0 * x + 1.0);
    }
    assert_eq!(a.phase(), ContinuousPhase::Validating);
    assert_eq!(a.candidate_progress(), Some((0, 0)));
    assert_eq!(a.statistics().predictions, 0);
    for id in 9..=13 {
        step(&mut a, id, |x| 2.0 * x + 1.0);
        assert_ne!(a.phase(), ContinuousPhase::Active);
    }
    assert_eq!(
        step(&mut a, 14, |x| 2.0 * x + 1.0).status,
        FeedbackStatus::Promoted
    );
    let p = a.observe(15, &[25.0]).unwrap();
    assert_eq!(p.value, Some(51.0));
    assert!(a.feedback(15, 51.0).unwrap().correct);
}
#[test]
fn delayed_feedback_is_scored_against_the_original_prediction_not_a_refit() {
    let mut a = agent();
    warm(&mut a);
    let p = a.observe(31, &[4.0]).unwrap();
    assert_eq!(p.value, Some(9.0));
    a.observe(32, &[5.0]).unwrap();
    a.feedback(32, 11.0).unwrap();
    let before = a.statistics();
    let r = a.feedback(31, 17.0).unwrap();
    assert_eq!(r.status, FeedbackStatus::Historical);
    assert!(r.scored && !r.correct);
    assert_eq!(a.statistics().scored, before.scored + 1);
    assert_eq!(a.statistics().correct, before.correct);
    assert_eq!(a.statistics().drifts, 0);
}
#[test]
fn abrupt_drift_suspends_then_adapts_and_preserves_immutable_lineage() {
    let mut a = agent();
    warm(&mut a);
    let first = a.model_identity().unwrap();
    for id in 31..=140 {
        step(&mut a, id, |x| -1.75 * x + 4.25);
    }
    assert_eq!(a.phase(), ContinuousPhase::Active);
    assert_eq!(a.statistics().drifts, 1);
    assert_eq!(a.archived_models(), 2);
    assert_eq!(a.active_skill().unwrap().revision(), 2);
    assert_eq!(a.active_skill().unwrap().parent(), Some(first));
    assert!(a.policy().tolerance.agrees(
        a.observe(141, &[37.0]).unwrap().value.unwrap(),
        -1.75 * 37.0 + 4.25
    ));
}
#[test]
fn recurring_regimes_reactivate_a_known_program_without_new_synthesis() {
    let mut a = agent();
    warm(&mut a);
    let first = a.model_identity();
    for id in 31..=120 {
        step(&mut a, id, |x| -1.75 * x + 4.25);
    }
    let work = a.statistics().search_work;
    for id in 121..=220 {
        step(&mut a, id, |x| 2.0 * x + 1.0);
    }
    assert_eq!(a.phase(), ContinuousPhase::Active);
    assert_eq!(a.model_identity(), first);
    assert_eq!(a.archived_models(), 2);
    assert_eq!(a.statistics().reuses, 1);
    assert_eq!(a.statistics().search_work, work);
    assert!(a.statistics().archive_work > 0);
}
#[test]
fn isolated_outliers_are_errors_not_automatically_new_regimes() {
    let mut a = agent();
    warm(&mut a);
    for id in 31..=160 {
        step(&mut a, id, |x| {
            2.0 * x + 1.0 + if id == 50 || id == 100 { 100.0 } else { 0.0 }
        });
    }
    assert_eq!(a.phase(), ContinuousPhase::Active);
    assert_eq!(a.statistics().drifts, 0);
    assert_eq!(a.statistics().scored - a.statistics().correct, 2);
}
#[test]
fn a_job_that_finishes_after_the_world_changes_cannot_promote_stale_learning() {
    let mut a = agent();
    for id in 1..=8 {
        let x = input(id);
        a.observe(id, &[x]).unwrap();
        a.feedback(id, 2.0 * x + 1.0).unwrap();
    }
    let job = a.take_learning_job().unwrap();
    for id in 9..=24 {
        let x = input(id);
        a.observe(id, &[x]).unwrap();
        a.feedback(id, -1.75 * x + 4.25).unwrap();
    }
    assert_eq!(a.submit_learning_result(job.run(None)), JobStatus::Stale);
    assert_eq!(a.statistics().promotions, 0);
    assert!(a.statistics().search_work > 0);
    for id in 25..=100 {
        step(&mut a, id, |x| -1.75 * x + 4.25);
    }
    assert_eq!(a.phase(), ContinuousPhase::Active);
}
#[test]
fn a_candidate_is_refuted_by_future_data_even_if_all_training_examples_fit() {
    let mut a = agent();
    for id in 1..=8 {
        step(&mut a, id, |x| 2.0 * x + 1.0);
    }
    assert_eq!(a.phase(), ContinuousPhase::Validating);
    assert_eq!(
        step(&mut a, 9, |x| -1.75 * x + 4.25).status,
        FeedbackStatus::CandidateRejected
    );
    assert_eq!(a.statistics().promotions, 0);
}
#[test]
fn no_feedback_never_fakes_learning_and_retained_state_is_bounded() {
    let mut a = agent();
    for id in 1..=200 {
        a.observe(id, &[input(id)]).unwrap();
    }
    let bytes = a.retained_bytes();
    for id in 201..=10200 {
        a.observe(id, &[input(id)]).unwrap();
    }
    assert_eq!(a.retained_bytes(), bytes);
    assert!(a.statistics().expired > 10000);
    assert!(a.take_learning_job().is_none());
    assert_eq!(a.statistics().promotions, 0);
}
#[test]
fn duplicate_events_unknown_feedback_and_backpressure_do_not_mutate_learning() {
    let mut a = ContinuousAgent::new(
        "small",
        1,
        None,
        ContinuousPolicy {
            pending_capacity: 1,
            max_label_delay: 96,
            ..ContinuousPolicy::default()
        },
    )
    .unwrap();
    let p = a.observe(1, &[2.0]).unwrap();
    assert_eq!(
        a.observe(1, &[2.0]).unwrap(),
        Prediction {
            duplicate: true,
            ..p
        }
    );
    let before = a.to_bytes();
    assert!(a.observe(1, &[3.0]).is_err());
    assert!(a.observe(2, &[3.0]).is_err());
    assert!(a.feedback(7, 5.0).is_err());
    assert!(a.feedback(1, f64::NAN).is_err());
    assert_eq!(a.to_bytes(), before);
    a.feedback(1, 5.0).unwrap();
    let before = a.to_bytes();
    assert_eq!(
        a.feedback(1, 5.0).unwrap().status,
        FeedbackStatus::Duplicate
    );
    assert!(a.feedback(1, 6.0).is_err());
    assert_eq!(a.to_bytes(), before);
}
#[test]
fn checkpoints_preserve_pending_predictions_and_historical_models_without_clobber() {
    let mut a = agent();
    warm(&mut a);
    a.observe(31, &[4.0]).unwrap();
    let bytes = a.to_bytes();
    let mut restored = ContinuousAgent::from_bytes(&bytes).unwrap();
    assert_eq!(restored.to_bytes(), bytes);
    assert!(restored.feedback(31, 9.0).unwrap().correct);
    assert_eq!(restored.model_identity(), a.model_identity());
    for id in 32..=140 {
        step(&mut restored, id, |x| -1.75 * x + 4.25);
    }
    let reopened = ContinuousAgent::from_bytes(&restored.to_bytes()).unwrap();
    assert_eq!(reopened.archived_models(), 2);
    assert_eq!(reopened.phase(), ContinuousPhase::Active);
}
#[test]
fn checkpoint_corruption_truncation_and_forged_counters_are_rejected() {
    let mut a = agent();
    warm(&mut a);
    let bytes = a.to_bytes();
    for i in 0..bytes.len() {
        assert!(ContinuousAgent::from_bytes(&bytes[..i]).is_err());
    }
    let mut changed = bytes.clone();
    let middle = changed.len() / 2;
    changed[middle] ^= 1;
    assert!(ContinuousAgent::from_bytes(&changed).is_err());
    let mut changed = bytes.clone();
    changed[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    let end = changed.len() - 8;
    let hash = changed[..end].iter().fold(0xcbf29ce484222325_u64, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x100000001b3)
    });
    changed[end..].copy_from_slice(&hash.to_le_bytes());
    assert!(ContinuousAgent::from_bytes(&changed).is_err());
}
#[test]
fn in_flight_jobs_are_invalidated_by_restart_instead_of_replaying_old_commits() {
    let mut a = agent();
    for id in 1..=8 {
        let x = input(id);
        a.observe(id, &[x]).unwrap();
        a.feedback(id, 2.0 * x + 1.0).unwrap();
    }
    let old = a.take_learning_job().unwrap();
    let mut restored = ContinuousAgent::from_bytes(&a.to_bytes()).unwrap();
    assert_eq!(
        restored.submit_learning_result(old.run(None)),
        JobStatus::Stale
    );
    let new = restored.take_learning_job().unwrap();
    assert_eq!(
        restored.submit_learning_result(new.run(None)),
        JobStatus::Candidate
    );
}
#[test]
fn unsupported_changes_keep_abstaining_with_bounded_retry_frequency() {
    let mut a = ContinuousAgent::new(
        "unsupported",
        1,
        None,
        ContinuousPolicy {
            search: NumericBudget {
                max_cost: 5,
                max_terms: 4000,
                max_work: 50000,
            },
            ..ContinuousPolicy::default()
        },
    )
    .unwrap();
    for id in 1..=600 {
        step(&mut a, id, |x| x.sin());
    }
    assert_eq!(a.statistics().promotions, 0);
    assert!(a.statistics().failed_jobs >= 3);
    assert!(a.statistics().jobs < 30);
}
#[test]
fn archive_limits_stop_admission_without_erasing_previous_knowledge() {
    let mut a = ContinuousAgent::new(
        "limited",
        1,
        None,
        ContinuousPolicy {
            archive_models: 1,
            ..ContinuousPolicy::default()
        },
    )
    .unwrap();
    warm(&mut a);
    let identity = a.model_identity();
    let mut full = false;
    for id in 31..=150 {
        full |= step(&mut a, id, |x| -1.75 * x + 4.25).status == FeedbackStatus::ArchiveFull;
    }
    assert!(full);
    assert_eq!(a.archived_models(), 1);
    assert_eq!(a.model_identity(), identity);
    assert_ne!(a.phase(), ContinuousPhase::Active);
}

#[test]
fn raw_frames_generate_their_own_delayed_targets_and_learn_new_dynamics_by_composition() {
    let mut a = ContinuousAgent::new("raw/dynamics", 2, None, ContinuousPolicy::default()).unwrap();
    let mut state = 13.0;
    for id in 1..=400 {
        let ambient = input(id) * 3.0;
        let r = a.advance_frame(id, &[state, ambient], 0).unwrap();
        if let Some(job) = a.take_learning_job() {
            a.submit_learning_result(job.run(None));
        }
        assert_eq!(r.previous.is_some(), id > 1);
        state = (state + ambient) / 2.0;
    }
    assert_eq!(a.phase(), ContinuousPhase::Active);
    let first = a.model_identity().unwrap();
    let initial_work = a.statistics().search_work;
    for id in 401..=800 {
        let ambient = input(id) * 3.0;
        a.advance_frame(id, &[state, ambient], 0).unwrap();
        if let Some(job) = a.take_learning_job() {
            a.submit_learning_result(job.run(None));
        }
        state = ((state + ambient) / 2.0 + ambient) / 2.0;
    }
    assert_eq!(a.phase(), ContinuousPhase::Active);
    assert!(a.statistics().search_work > initial_work);
    assert!(a.active_skill().unwrap().dependencies().contains(&first));
    let bytes = a.to_bytes();
    a = ContinuousAgent::from_bytes(&bytes).unwrap();
    for id in 801..=1200 {
        let ambient = input(id) * 3.0;
        a.advance_frame(id, &[state, ambient], 0).unwrap();
        if let Some(job) = a.take_learning_job() {
            a.submit_learning_result(job.run(None));
        }
        state = (state + ambient) / 2.0;
    }
    assert_eq!(a.model_identity(), Some(first));
    assert!(a.statistics().reuses > 0);
    assert_eq!(a.statistics().feedback, a.statistics().observations - 1);
    assert_eq!(a.pending_count(), 1);
}
#[test]
fn temporal_duplicates_and_mode_changes_never_relabel_an_old_experience() {
    let mut a = ContinuousAgent::new("raw", 2, None, ContinuousPolicy::default()).unwrap();
    a.advance_frame(1, &[3.0, 4.0], 0).unwrap();
    let before = a.to_bytes();
    assert!(
        a.advance_frame(1, &[3.0, 4.0], 0)
            .unwrap()
            .prediction
            .duplicate
    );
    assert!(a.advance_frame(1, &[3.0, 5.0], 0).is_err());
    assert!(a.advance_frame(2, &[5.0, 4.0], 1).is_err());
    assert!(a.feedback(1, 5.0).is_err());
    assert!(a.observe(2, &[5.0, 4.0]).is_err());
    assert_eq!(a.to_bytes(), before);
    assert_eq!(
        a.advance_frame(2, &[5.0, 4.0], 0)
            .unwrap()
            .previous
            .unwrap()
            .status,
        FeedbackStatus::Accepted
    );
}

#[test]
fn impossible_training_diversity_is_rejected_before_allocating_an_agent() {
    let policy = ContinuousPolicy {
        training_window: 2,
        min_training: 2,
        validation_distinct: 4,
        ..ContinuousPolicy::default()
    };
    assert!(ContinuousAgent::new("bad/diversity", 1, None, policy).is_err());
    assert!(
        ContinuousAgent::new(
            "small/diversity",
            1,
            None,
            ContinuousPolicy {
                validation_distinct: 2,
                ..policy
            }
        )
        .is_ok()
    );
}
