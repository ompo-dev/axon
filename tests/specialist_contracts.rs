use axon_uic::specialist::*;
use axon_uic::{Authority, Capability, CapabilityGate, Effect, Feasibility, GateFailure};

fn id() -> CapsuleId {
    CapsuleId {
        context: 42,
        version: 1,
    }
}

fn data(start: u64, n: usize) -> Vec<Observation> {
    (0..n)
        .map(|i| {
            let x = (i % 128) as i64 - 64;
            Observation {
                id: start + i as u64,
                features: [x.rem_euclid(2), 0, x, x.rem_euclid(5)],
                label: x >= 11,
            }
        })
        .collect()
}

fn trained(config: FoundryConfig) -> SpecialistCapsule {
    train_specialist(
        id(),
        config,
        &data(0, 1024),
        &data(1024, 1024),
        &data(2048, 1024),
    )
    .unwrap()
    .capsule
    .unwrap()
}

#[test]
fn probability_rejects_nonfinite_and_out_of_range_values() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
        assert!(Probability::new(invalid).is_none());
    }
    for valid in [0.0, 0.5, 1.0] {
        assert_eq!(Probability::new(valid).unwrap().get(), valid);
    }
    assert!(Probability::new(0.5).unwrap().label());
}

#[test]
fn learns_feature_and_threshold_instead_of_receiving_a_strategy_label() {
    let capsule = trained(FoundryConfig::default());
    assert_eq!(capsule.evidence().rule.split(), Some((2, 10)));
    assert_eq!(capsule.evidence().rule.form(), RuleForm::Threshold);
    assert_eq!(capsule.evidence().rule.second_split(), None);
    assert_eq!(capsule.evidence().audit.errors, 0);
    assert_eq!(capsule.evidence().audit.coverage(), 1.0);
    for row in data(9999, 128) {
        assert_eq!(
            capsule
                .predict(42, &row.features)
                .unwrap()
                .probability
                .label(),
            row.label
        );
    }
}

#[test]
fn models_can_learn_reversed_rules() {
    let mut splits = [data(0, 1024), data(1024, 1024), data(2048, 1024)];
    for row in splits.iter_mut().flatten() {
        row.label = !row.label;
    }
    let output = train_specialist(
        id(),
        FoundryConfig::default(),
        &splits[0],
        &splits[1],
        &splits[2],
    )
    .unwrap();
    assert_eq!(output.evidence.rule.split(), Some((2, 10)));
    assert!(
        output
            .capsule
            .unwrap()
            .predict(42, &[0, 0, -60, 0])
            .unwrap()
            .probability
            .label()
    );
}

#[test]
fn constant_and_indistinguishable_inputs_cannot_fake_a_gain() {
    for constant_label in [false, true] {
        let mut splits = [data(0, 1024), data(1024, 1024), data(2048, 1024)];
        for row in splits.iter_mut().flatten() {
            row.label = constant_label;
        }
        let out = train_specialist(
            id(),
            FoundryConfig::default(),
            &splits[0],
            &splits[1],
            &splits[2],
        )
        .unwrap();
        assert_eq!(out.evidence.promotion, Promotion::NoGain);
        assert!(out.capsule.is_none());
    }
    let mut splits = [data(0, 1024), data(1024, 1024), data(2048, 1024)];
    for row in splits.iter_mut().flatten() {
        row.features = [i64::MAX; FEATURES];
    }
    let out = train_specialist(
        id(),
        FoundryConfig::default(),
        &splits[0],
        &splits[1],
        &splits[2],
    )
    .unwrap();
    assert_eq!(out.evidence.promotion, Promotion::NoGain);
    assert_eq!(out.evidence.rule.form(), RuleForm::Constant);
}

#[test]
fn composition_learns_xor_while_one_predicate_refuses_it() {
    let splits: Vec<Vec<_>> = (0..3)
        .map(|split| {
            (0..1024)
                .map(|i| Observation {
                    id: split * 1024 + i,
                    features: [(i % 2) as i64, ((i / 2) % 2) as i64, 0, 0],
                    label: (i % 2 == 1) ^ ((i / 2) % 2 == 1),
                })
                .collect()
        })
        .collect();
    let out = train_specialist(
        id(),
        FoundryConfig::default(),
        &splits[0],
        &splits[1],
        &splits[2],
    )
    .unwrap();
    assert_eq!(out.evidence.promotion, Promotion::Accepted);
    assert_eq!(out.evidence.rule.form(), RuleForm::PairTable);
    assert!(out.evidence.rule.second_split().is_some());
    assert_eq!(out.evidence.audit.errors, 0);
    assert!(out.capsule.is_some());

    let one_predicate = FoundryConfig {
        max_predicates: 1,
        ..FoundryConfig::default()
    };
    let out = train_specialist(id(), one_predicate, &splits[0], &splits[1], &splits[2]).unwrap();
    assert_eq!(out.evidence.promotion, Promotion::NoGain);
    assert_eq!(out.evidence.rule.form(), RuleForm::Constant);
    assert!(out.capsule.is_none());
}

#[test]
fn three_bit_parity_is_an_explicit_negative_control_for_two_predicates() {
    let splits: Vec<Vec<_>> = (0..3)
        .map(|split| {
            (0..1024)
                .map(|i| {
                    let a = i % 2 != 0;
                    let b = (i / 2) % 2 != 0;
                    let c = (i / 4) % 2 != 0;
                    Observation {
                        id: split * 1024 + i,
                        features: [i64::from(a), i64::from(b), i64::from(c), 0],
                        label: a ^ b ^ c,
                    }
                })
                .collect()
        })
        .collect();
    let out = train_specialist(
        id(),
        FoundryConfig::default(),
        &splits[0],
        &splits[1],
        &splits[2],
    )
    .unwrap();
    assert_eq!(out.evidence.promotion, Promotion::NoGain);
    assert_eq!(out.evidence.audit.brier, 0.25);
    assert_eq!(out.evidence.audit.selective_error(), None);
    assert!(out.capsule.is_none());
}

#[test]
fn audit_labels_cannot_change_the_selected_rule() {
    let train = data(0, 1024);
    let selection = data(1024, 1024);
    let mut audit = data(2048, 1024);
    let good =
        train_specialist(id(), FoundryConfig::default(), &train, &selection, &audit).unwrap();
    for row in &mut audit {
        row.label = !row.label;
    }
    let bad = train_specialist(id(), FoundryConfig::default(), &train, &selection, &audit).unwrap();
    assert_eq!(good.evidence.rule, bad.evidence.rule);
    assert_eq!(good.evidence.selection, bad.evidence.selection);
    assert!(good.capsule.is_some());
    assert!(bad.capsule.is_none());
}

#[test]
fn selection_can_reject_a_training_only_pattern() {
    let mut selection = data(1024, 1024);
    for row in &mut selection {
        row.label = !row.label;
    }
    let out = train_specialist(
        id(),
        FoundryConfig::default(),
        &data(0, 1024),
        &selection,
        &data(2048, 1024),
    )
    .unwrap();
    assert!(out.evidence.rule.split().is_none());
    assert_eq!(out.evidence.promotion, Promotion::NoGain);
}

#[test]
fn duplicate_ids_within_or_across_partitions_are_rejected() {
    for partition in 0..3 {
        let mut splits = [data(0, 1024), data(1024, 1024), data(2048, 1024)];
        splits[partition][100].id = 0;
        assert_eq!(
            train_specialist(
                id(),
                FoundryConfig::default(),
                &splits[0],
                &splits[1],
                &splits[2]
            )
            .unwrap_err(),
            FoundryError::DuplicateObservation(0)
        );
    }
}

#[test]
fn configuration_and_resource_limits_fail_closed() {
    let rows = data(0, 1024);
    let defaults = FoundryConfig::default();
    for config in [
        FoundryConfig {
            min_leaf: 0,
            ..defaults
        },
        FoundryConfig {
            min_leaf: usize::MAX,
            ..defaults
        },
        FoundryConfig {
            min_audit_accepted: 0,
            ..defaults
        },
        FoundryConfig {
            max_predicates: 0,
            ..defaults
        },
        FoundryConfig {
            max_predicates: 3,
            ..defaults
        },
        FoundryConfig {
            min_confidence: f64::NAN,
            ..defaults
        },
        FoundryConfig {
            min_confidence: 0.49,
            ..defaults
        },
        FoundryConfig {
            max_error: f64::INFINITY,
            ..defaults
        },
        FoundryConfig {
            complexity_penalty: -1.0,
            ..defaults
        },
        FoundryConfig {
            min_brier_gain: 0.0,
            ..defaults
        },
        FoundryConfig {
            drift_window: MAX_DRIFT_WINDOW + 1,
            ..defaults
        },
        FoundryConfig {
            drift_window: 0,
            ..defaults
        },
    ] {
        assert_eq!(
            train_specialist(id(), config, &rows, &rows, &rows).unwrap_err(),
            FoundryError::InvalidConfig
        );
    }
    assert_eq!(
        train_specialist(
            CapsuleId { version: 0, ..id() },
            defaults,
            &rows,
            &rows,
            &rows
        )
        .unwrap_err(),
        FoundryError::InvalidVersion
    );
    assert_eq!(
        train_specialist(id(), defaults, &[], &rows, &rows).unwrap_err(),
        FoundryError::TooFewSamples
    );
    let oversized = vec![rows[0]; MAX_SPLIT_SAMPLES + 1];
    assert_eq!(
        train_specialist(id(), defaults, &oversized, &rows, &rows).unwrap_err(),
        FoundryError::TooManySamples
    );
}

#[test]
fn context_range_and_confidence_guards_abstain() {
    let capsule = trained(FoundryConfig::default());
    assert_eq!(
        capsule.predict(43, &[0, 0, 0, 0]),
        Err(Abstention::ContextMismatch)
    );
    assert_eq!(
        capsule.predict(42, &[0, 0, 64, 0]),
        Err(Abstention::OutsideDomain)
    );
    assert_eq!(
        capsule.predict(42, &[0, 0, -65, 0]),
        Err(Abstention::OutsideDomain)
    );
    let mut splits = [data(0, 1024), data(1024, 1024), data(2048, 1024)];
    for row in splits.iter_mut().flatten() {
        if row.features[2] < 0 {
            row.label = row.id % 256 >= 128;
        }
    }
    let config = FoundryConfig {
        min_coverage: 0.35,
        ..FoundryConfig::default()
    };
    let capsule = train_specialist(id(), config, &splits[0], &splits[1], &splits[2])
        .unwrap()
        .capsule
        .unwrap();
    assert_eq!(
        capsule.predict(42, &[0, 0, -64, 1]),
        Err(Abstention::LowConfidence)
    );
}

#[test]
fn coverage_and_error_are_separate_promotion_conditions() {
    let train = data(0, 1024);
    let selection = data(1024, 1024);
    let mut audit = data(2048, 1024);
    for row in &mut audit {
        row.features[1] = 1;
    }
    let out = train_specialist(id(), FoundryConfig::default(), &train, &selection, &audit).unwrap();
    assert_eq!(out.evidence.promotion, Promotion::InsufficientCoverage);
    assert_eq!(out.evidence.audit.selective_error(), None);
    let mut audit = data(2048, 1024);
    for row in audit.iter_mut().step_by(5) {
        row.label = !row.label;
    }
    let out = train_specialist(id(), FoundryConfig::default(), &train, &selection, &audit).unwrap();
    assert_eq!(out.evidence.promotion, Promotion::ExcessiveError);
}

#[test]
fn confidence_never_grants_network_authority() {
    let capsule = trained(FoundryConfig::default());
    let gate = CapabilityGate::new(Feasibility::ready(), Authority::from([Effect::ReadData]));
    let capability = Capability::new("network-action", [Effect::Network]);
    assert_eq!(
        capsule.decide(42, &[0, 0, 50, 0], &capability, &gate),
        Err(DecisionFailure::Gate(GateFailure::Unauthorized(
            Effect::Network
        )))
    );
    let authorized = CapabilityGate::new(Feasibility::ready(), Authority::from([Effect::Network]));
    assert!(
        capsule
            .decide(42, &[0, 0, 50, 0], &capability, &authorized)
            .is_ok()
    );
}

#[test]
fn invalid_feedback_is_transactional_and_duplicate_feedback_is_rejected() {
    let mut capsule = trained(FoundryConfig::default());
    let before = capsule.clone();
    assert_eq!(
        capsule.observe_labeled(2, 42, &[0; FEATURES], false),
        Err(FeedbackError::OutOfOrder)
    );
    assert_eq!(
        capsule.observe_labeled(1, 43, &[0; FEATURES], false),
        Err(FeedbackError::Abstain(Abstention::ContextMismatch))
    );
    assert_eq!(
        capsule.observe_labeled(1, 42, &[i64::MAX; FEATURES], false),
        Err(FeedbackError::Abstain(Abstention::OutsideDomain))
    );
    assert_eq!(capsule, before);
    capsule
        .observe_labeled(1, 42, &[0; FEATURES], false)
        .unwrap();
    let before = capsule.clone();
    assert_eq!(
        capsule.observe_labeled(1, 42, &[0; FEATURES], true),
        Err(FeedbackError::OutOfOrder)
    );
    assert_eq!(capsule, before);
}

#[test]
fn drift_suspends_without_changing_rule_or_rewriting_evidence() {
    let mut capsule = trained(FoundryConfig::default());
    let evidence = capsule.evidence();
    for event in 1..=1000 {
        capsule
            .observe_labeled(event, 42, &[0; FEATURES], false)
            .unwrap();
    }
    for event in 1001..=1017 {
        capsule
            .observe_labeled(event, 42, &[0; FEATURES], true)
            .unwrap();
    }
    assert!(capsule.suspended());
    assert_eq!(capsule.evidence(), evidence);
    assert_eq!(
        capsule.predict(42, &[0; FEATURES]),
        Err(Abstention::Suspended)
    );
    let before = capsule.clone();
    assert_eq!(
        capsule.observe_labeled(1018, 42, &[0; FEATURES], false),
        Err(FeedbackError::Abstain(Abstention::Suspended))
    );
    assert_eq!(capsule, before);
}

#[test]
fn drift_window_forgets_old_errors_and_supports_boundary_sizes() {
    for window in [1, 64, MAX_DRIFT_WINDOW] {
        let config = FoundryConfig {
            drift_window: window,
            max_window_error: 1.0,
            ..FoundryConfig::default()
        };
        let mut capsule = trained(config);
        for event in 1..=window {
            capsule
                .observe_labeled(event as u64, 42, &[0; FEATURES], true)
                .unwrap();
        }
        for i in 1..=window {
            let feedback = capsule
                .observe_labeled((window + i) as u64, 42, &[0; FEATURES], false)
                .unwrap();
            assert_eq!(feedback.window_errors, window - i);
        }
        assert!(!capsule.suspended());
    }
}

#[test]
fn threshold_search_handles_extreme_integers_without_midpoint_overflow() {
    let splits: Vec<Vec<_>> = (0..3)
        .map(|split| {
            (0..128)
                .map(|i| Observation {
                    id: split * 128 + i,
                    features: [if i % 2 == 0 { i64::MIN } else { i64::MAX }, 0, 0, 0],
                    label: i % 2 != 0,
                })
                .collect()
        })
        .collect();
    let out = train_specialist(
        id(),
        FoundryConfig::default(),
        &splits[0],
        &splits[1],
        &splits[2],
    )
    .unwrap();
    assert_eq!(out.evidence.rule.split(), Some((0, i64::MIN)));
    assert_eq!(out.evidence.audit.errors, 0);
}

#[test]
fn brier_and_calibration_metrics_match_an_independent_calculation() {
    let capsule = trained(FoundryConfig::default());
    let samples = data(4096, 1024);
    let metrics = capsule.evaluate(&samples).unwrap();
    let probabilities: Vec<_> = samples
        .iter()
        .map(|s| capsule.predict(42, &s.features).unwrap().probability)
        .collect();
    let expected_brier = probabilities
        .iter()
        .zip(&samples)
        .map(|(p, s)| (p.get() - f64::from(s.label)).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    let expected_ece = probabilities
        .iter()
        .map(|p| 1.0 - p.confidence())
        .sum::<f64>()
        / samples.len() as f64;
    assert!((metrics.brier - expected_brier).abs() < 1e-12);
    assert!((metrics.ece - expected_ece).abs() < 1e-12);
    assert_eq!(capsule.evaluate(&[]), Err(FoundryError::TooFewSamples));
}

#[test]
fn fitted_loss_matches_exhaustive_reference_search_on_random_small_datasets() {
    let config = FoundryConfig {
        min_leaf: 2,
        min_audit_accepted: 1,
        max_predicates: 1,
        complexity_penalty: 0.0,
        ..FoundryConfig::default()
    };
    let mut rng = 77_u64;
    for _ in 0..64 {
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let train: Vec<_> = (0..32)
            .map(|id| Observation {
                id,
                features: std::array::from_fn(|_| (next() % 8) as i64),
                label: next() % 2 != 0,
            })
            .collect();
        let mut selection = train.clone();
        let mut audit = train.clone();
        for row in &mut selection {
            row.id += 32;
        }
        for row in &mut audit {
            row.id += 64;
        }
        let out = train_specialist(id(), config, &train, &selection, &audit).unwrap();
        let prior = (train.iter().filter(|s| s.label).count() as f64 + 1.0) / 34.0;
        let mut expected = train
            .iter()
            .map(|s| (prior - f64::from(s.label)).powi(2))
            .sum::<f64>();
        for feature in 0..FEATURES {
            for threshold in 0..8 {
                let left: Vec<_> = train
                    .iter()
                    .filter(|s| s.features[feature] <= threshold)
                    .collect();
                let right: Vec<_> = train
                    .iter()
                    .filter(|s| s.features[feature] > threshold)
                    .collect();
                if left.len() < config.min_leaf || right.len() < config.min_leaf {
                    continue;
                }
                let score = [&left, &right]
                    .iter()
                    .map(|group| {
                        let p = (group.iter().filter(|s| s.label).count() as f64 + 1.0)
                            / (group.len() as f64 + 2.0);
                        group
                            .iter()
                            .map(|s| (p - f64::from(s.label)).powi(2))
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                expected = expected.min(score);
            }
        }
        assert!((out.evidence.selection.brier - expected / 32.0).abs() < 1e-12);
    }
}

#[test]
fn checkpoint_roundtrip_preserves_monitor_and_rejects_corruption() {
    let mut capsule = trained(FoundryConfig::default());
    capsule
        .observe_labeled(1, 42, &[0; FEATURES], false)
        .unwrap();
    capsule
        .observe_labeled(2, 42, &[0; FEATURES], true)
        .unwrap();
    assert!(std::mem::size_of::<SpecialistCapsule>() <= 1024);
    let path = std::env::temp_dir().join(format!(
        "axon-specialist-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    capsule.save(&path).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len() as usize,
        SpecialistCapsule::checkpoint_bytes()
    );
    assert_eq!(SpecialistCapsule::load(&path, id()).unwrap(), capsule);
    assert_eq!(
        capsule.save(&path).unwrap_err().kind(),
        std::io::ErrorKind::AlreadyExists
    );
    for wrong in [
        CapsuleId {
            context: 43,
            ..id()
        },
        CapsuleId { version: 2, ..id() },
    ] {
        assert_eq!(
            SpecialistCapsule::load(&path, wrong).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }
    let original = std::fs::read(&path).unwrap();
    for index in 0..original.len() {
        let mut corrupt = original.clone();
        corrupt[index] ^= 1;
        std::fs::write(&path, corrupt).unwrap();
        assert_eq!(
            SpecialistCapsule::load(&path, id()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData,
            "accepted corruption at byte {index}"
        );
    }
    for length in 0..original.len() {
        std::fs::write(&path, &original[..length]).unwrap();
        assert_eq!(
            SpecialistCapsule::load(&path, id()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData,
            "accepted truncation at length {length}"
        );
    }
    let mut oversized = original.clone();
    oversized.push(0);
    std::fs::write(&path, oversized).unwrap();
    assert_eq!(
        SpecialistCapsule::load(&path, id()).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );

    // The checksum catches accidents, while semantic validation handles rewritten records.
    let mut forged = original.clone();
    forged[178] = 1; // Promotion::NoGain cannot back a live capsule (v2 layout).
    let checksum = forged[..forged.len() - 8]
        .iter()
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
    let checksum_offset = forged.len() - 8;
    forged[checksum_offset..].copy_from_slice(&checksum.to_le_bytes());
    std::fs::write(&path, forged).unwrap();
    assert_eq!(
        SpecialistCapsule::load(&path, id()).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
    std::fs::write(&path, original).unwrap();
    assert_eq!(SpecialistCapsule::load(&path, id()).unwrap(), capsule);
    std::fs::remove_file(path).unwrap();
}
