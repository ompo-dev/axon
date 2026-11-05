use axon_uic::{
    CandidateDecision, ChangeStructure, DeltaForge, DerivedArtifact, ExactAverage,
    ExactPopulationVariance, FoldSpec, ForgeError, MaintenanceState, MinimumUpdate, RebuildPolicy,
    SynthesisBudget, UpdateRule, VectorU64,
};

#[test]
fn forge_derives_modular_sum_without_an_updater_input() {
    let plan = DeltaForge::synthesize(FoldSpec::AddModU64).unwrap();
    let vector = VectorU64;
    let before = vec![u64::MAX, 4, 8, 15];
    let after = vec![1, 4, 23, 15];
    let change = vector.diff(&after, &before).unwrap();
    let (old_total, cache) = plan.full(&before);
    let (output_delta, next_cache) = plan.delta(&change, &cache).unwrap();

    assert_eq!(
        plan.certificate().maintenance_state(),
        MaintenanceState::ModularTotal
    );
    assert_eq!(
        plan.certificate().update_rule(),
        UpdateRule::SubtractOldThenAddNew
    );
    assert_eq!(
        plan.apply_output_delta(old_total, output_delta),
        plan.full(&after).0
    );
    assert_eq!(next_cache, plan.full(&after).1);
    assert!(plan.check(&before, &change).is_ok());
}

#[test]
fn forge_searches_bounded_state_and_refuses_an_unsafe_minimum_budget() {
    let trace = DeltaForge::analyze(FoldSpec::MinU64, SynthesisBudget::default());

    assert_eq!(
        trace.selected(),
        Some(MaintenanceState::MinimumAndMultiplicity)
    );
    assert_eq!(
        trace.attempts().map(|attempt| attempt.decision()),
        [
            CandidateDecision::SemanticMismatch,
            CandidateDecision::SemanticMismatch,
            CandidateDecision::Selected,
            CandidateDecision::SemanticMismatch,
        ]
    );
    assert_eq!(
        trace.attempts()[2].rebuild_policy(),
        RebuildPolicy::WhenWitnessExhausted
    );
    assert_eq!(
        DeltaForge::synthesize_with_budget(FoldSpec::MinU64, SynthesisBudget::new(2, true)),
        Err(ForgeError::StateBudgetExceeded {
            required_words: 3,
            available_words: 2,
        })
    );
    assert_eq!(
        DeltaForge::synthesize_with_budget(FoldSpec::MinU64, SynthesisBudget::new(3, false)),
        Err(ForgeError::ExactRebuildDisallowed(FoldSpec::MinU64))
    );
}

#[test]
fn minimum_witness_updates_locally_and_requests_rebuild_only_when_information_is_missing() {
    let artifact = DeltaForge::synthesize_capability(FoldSpec::MinU64).unwrap();
    let DerivedArtifact::Minimum(plan) = artifact else {
        panic!("minimum spec must derive a minimum artifact");
    };
    let vector = VectorU64;
    let before = vec![1, 1, 5, 9];
    let after_one = vec![4, 1, 5, 9];
    let first_change = vector.diff(&after_one, &before).unwrap();
    let (_, cache) = plan.full(&before).unwrap();

    let MinimumUpdate::Incremental {
        minimum,
        cache: next_cache,
    } = plan.delta(&first_change, &cache).unwrap()
    else {
        panic!("one remaining witness proves the minimum");
    };
    assert_eq!(minimum, 1);
    assert_eq!(next_cache.multiplicity(), 1);

    let after_two = vec![4, 6, 5, 9];
    let second_change = vector.diff(&after_two, &after_one).unwrap();
    assert_eq!(
        plan.delta(&second_change, &next_cache).unwrap(),
        MinimumUpdate::RebuildRequired
    );
    assert_eq!(plan.full(&after_two).unwrap().0, 4);
    assert!(plan.check(&after_one, &second_change).is_ok());

    let lower = vec![4, 0, 5, 9];
    let lower_change = vector.diff(&lower, &after_two).unwrap();
    let (_, rebuilt_cache) = plan.full(&after_two).unwrap();
    assert_eq!(
        plan.delta(&lower_change, &rebuilt_cache).unwrap().minimum(),
        Some(0)
    );

    let all_before = vec![3, 4];
    let all_after = vec![8, 7];
    let all_change = vector.diff(&all_after, &all_before).unwrap();
    let (_, all_cache) = plan.full(&all_before).unwrap();
    assert_eq!(
        plan.delta(&all_change, &all_cache).unwrap().minimum(),
        Some(7)
    );
    assert_eq!(plan.full(&[]), Err(ForgeError::EmptyInput));
}

#[test]
fn minimum_witness_matches_full_recomputation_across_random_histories() {
    let DerivedArtifact::Minimum(plan) =
        DeltaForge::synthesize_capability(FoldSpec::MinU64).unwrap()
    else {
        panic!("minimum spec must derive a minimum artifact");
    };
    let vector = VectorU64;
    let mut rng = 0xA6A6_51D3_91E7_204B_u64;

    for length in 1..=64 {
        let mut values: Vec<_> = (0..length)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                rng % 17
            })
            .collect();
        let (_, mut cache) = plan.full(&values).unwrap();
        for _ in 0..256 {
            let mut after = values.clone();
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let index = rng as usize % length;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            after[index] = rng % 17;
            let change = vector.diff(&after, &values).unwrap();
            let update = plan.delta(&change, &cache).unwrap();
            let expected = plan.full(&after).unwrap();
            match update {
                MinimumUpdate::Incremental {
                    minimum,
                    cache: next_cache,
                } => {
                    assert_eq!(minimum, expected.0);
                    assert_eq!(next_cache, expected.1);
                    cache = next_cache;
                }
                MinimumUpdate::RebuildRequired => cache = expected.1,
            }
            assert!(plan.check(&values, &change).is_ok());
            values = after;
        }
    }
}

#[test]
fn minimum_witness_is_exact_for_every_small_state_transition() {
    let DerivedArtifact::Minimum(plan) =
        DeltaForge::synthesize_capability(FoldSpec::MinU64).unwrap()
    else {
        panic!("minimum spec must derive a minimum artifact");
    };
    let vector = VectorU64;
    let mut incremental = 0_usize;
    let mut rebuilt = 0_usize;

    for length in 1..=5 {
        let states = 3_usize.pow(length as u32);
        for before_code in 0..states {
            let before = ternary_vector(before_code, length);
            let (_, cache) = plan.full(&before).unwrap();
            for after_code in 0..states {
                let after = ternary_vector(after_code, length);
                let change = vector.diff(&after, &before).unwrap();
                let expected = plan.full(&after).unwrap();
                match plan.delta(&change, &cache).unwrap() {
                    MinimumUpdate::Incremental {
                        minimum,
                        cache: next_cache,
                    } => {
                        incremental += 1;
                        assert_eq!((minimum, next_cache), expected);
                    }
                    MinimumUpdate::RebuildRequired => rebuilt += 1,
                }
            }
        }
    }

    assert!(incremental > 0);
    assert!(rebuilt > 0);
}

fn ternary_vector(mut code: usize, length: usize) -> Vec<u64> {
    (0..length)
        .map(|_| {
            let value = (code % 3) as u64;
            code /= 3;
            value
        })
        .collect()
}

#[test]
fn forge_derives_exact_average_state_without_being_given_sum_and_count() {
    let artifact = DeltaForge::synthesize_capability(FoldSpec::AverageExactU64).unwrap();
    let vector = VectorU64;
    let before = vec![2, 4, 9];
    let after = vec![2, 10, 9];
    let change = vector.diff(&after, &before).unwrap();

    let DerivedArtifact::Average(plan) = artifact else {
        panic!("average spec must derive an average artifact");
    };
    let (old_average, cache) = plan.full(&before).unwrap();
    let (next_average, next_cache) = plan.delta(&change, &cache).unwrap();

    assert_eq!(old_average, ExactAverage::new(15, 3).unwrap());
    assert_eq!(next_average, ExactAverage::new(21, 3).unwrap());
    assert_eq!(next_cache, plan.full(&after).unwrap().1);
    assert_eq!(
        plan.certificate().maintenance_state(),
        MaintenanceState::ExactSumAndCount
    );
    assert_eq!(
        plan.certificate().update_rule(),
        UpdateRule::SubtractOldThenAddNewPreserveCount
    );
    assert!(plan.check(&before, &change).is_ok());
}

#[test]
fn exact_average_never_reuses_modular_sum_semantics() {
    let artifact = DeltaForge::synthesize_capability(FoldSpec::AverageExactU64).unwrap();
    let DerivedArtifact::Average(plan) = artifact else {
        panic!("average spec must derive an average artifact");
    };

    let (average, _) = plan.full(&[u64::MAX, u64::MAX]).unwrap();

    assert_eq!(
        average,
        ExactAverage::new(u128::from(u64::MAX) * 2, 2).unwrap()
    );
}

#[test]
fn forge_derives_exact_population_variance_from_three_moments() {
    let trace = DeltaForge::analyze(
        FoldSpec::PopulationVarianceExactU64,
        SynthesisBudget::default(),
    );
    assert_eq!(
        trace.selected(),
        Some(MaintenanceState::ExactFirstAndSecondMoments)
    );
    assert_eq!(
        trace.attempts().map(|attempt| attempt.decision()),
        [
            CandidateDecision::SemanticMismatch,
            CandidateDecision::SemanticMismatch,
            CandidateDecision::SemanticMismatch,
            CandidateDecision::Selected,
        ]
    );
    assert_eq!(
        DeltaForge::synthesize_with_budget(
            FoldSpec::PopulationVarianceExactU64,
            SynthesisBudget::new(2, true),
        ),
        Err(ForgeError::StateBudgetExceeded {
            required_words: 3,
            available_words: 2,
        })
    );

    let DerivedArtifact::PopulationVariance(plan) =
        DeltaForge::synthesize_capability(FoldSpec::PopulationVarianceExactU64).unwrap()
    else {
        panic!("variance spec must derive a population-variance artifact");
    };
    let before = vec![1, 2, 3];
    let after = vec![1, 4, 3];
    let change = VectorU64.diff(&after, &before).unwrap();
    let (variance, cache) = plan.full(&before).unwrap();
    let (next_variance, next_cache) = plan.delta(&change, &cache).unwrap();

    assert_eq!(variance, ExactPopulationVariance::new(6, 9).unwrap());
    assert_eq!(next_variance, ExactPopulationVariance::new(14, 9).unwrap());
    assert_eq!(next_cache, plan.full(&after).unwrap().1);
    assert_eq!(
        plan.certificate().maintenance_state(),
        MaintenanceState::ExactFirstAndSecondMoments
    );
    assert_eq!(
        plan.certificate().update_rule(),
        UpdateRule::ReplaceExactMomentsPreserveCount
    );
    assert!(plan.check(&before, &change).is_ok());
}

#[test]
fn exact_population_variance_matches_every_small_state_transition() {
    let DerivedArtifact::PopulationVariance(plan) =
        DeltaForge::synthesize_capability(FoldSpec::PopulationVarianceExactU64).unwrap()
    else {
        panic!("variance spec must derive a population-variance artifact");
    };

    for length in 1..=5 {
        let states = 3_usize.pow(length as u32);
        for before_code in 0..states {
            let before = ternary_vector(before_code, length);
            let (_, cache) = plan.full(&before).unwrap();
            for after_code in 0..states {
                let after = ternary_vector(after_code, length);
                let change = VectorU64.diff(&after, &before).unwrap();
                let (variance, next_cache) = plan.delta(&change, &cache).unwrap();
                let expected_cache = plan.full(&after).unwrap().1;
                let (expected_numerator, expected_denominator) = reference_variance(&after);

                assert_eq!(variance.numerator(), expected_numerator);
                assert_eq!(variance.denominator(), expected_denominator);
                assert_eq!(next_cache, expected_cache);
            }
        }
    }
}

#[test]
fn exact_population_variance_refuses_unrepresentable_moments() {
    let DerivedArtifact::PopulationVariance(plan) =
        DeltaForge::synthesize_capability(FoldSpec::PopulationVarianceExactU64).unwrap()
    else {
        panic!("variance spec must derive a population-variance artifact");
    };

    let (single, cache) = plan.full(&[u64::MAX]).unwrap();
    assert_eq!(single, ExactPopulationVariance::new(0, 1).unwrap());
    assert_eq!(cache.total(), u128::from(u64::MAX));
    assert_eq!(
        plan.full(&[u64::MAX, u64::MAX]),
        Err(ForgeError::ArithmeticOverflow)
    );
    assert_eq!(
        plan.full(&[u64::MAX, 0]),
        Err(ForgeError::ArithmeticOverflow)
    );

    let before = vec![0, 0];
    let after = vec![u64::MAX, u64::MAX];
    let change = VectorU64.diff(&after, &before).unwrap();
    let (_, cache) = plan.full(&before).unwrap();
    assert_eq!(
        plan.delta(&change, &cache),
        Err(ForgeError::ArithmeticOverflow)
    );
    assert_eq!(cache, plan.full(&before).unwrap().1);
    assert_eq!(plan.full(&[]), Err(ForgeError::EmptyInput));
}

#[test]
fn exact_population_variance_matches_random_supported_histories() {
    let DerivedArtifact::PopulationVariance(plan) =
        DeltaForge::synthesize_capability(FoldSpec::PopulationVarianceExactU64).unwrap()
    else {
        panic!("variance spec must derive a population-variance artifact");
    };
    let mut rng = 0x91C4_77A6_D39E_205B_u64;

    for length in 1..=64 {
        let mut values: Vec<_> = (0..length)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                1_000_000_000 + rng % 10_000
            })
            .collect();
        let (_, mut cache) = plan.full(&values).unwrap();
        for step in 0..256 {
            let mut after = values.clone();
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let index = rng as usize % length;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            after[index] = if step % 2 == 0 {
                rng % 1_000_000
            } else {
                1_000_000_000 + rng % 10_000
            };
            let change = VectorU64.diff(&after, &values).unwrap();
            let (variance, next_cache) = plan.delta(&change, &cache).unwrap();
            let expected = plan.full(&after).unwrap();
            let reference = reference_variance(&after);

            assert_eq!(variance, expected.0);
            assert_eq!(variance.numerator(), reference.0);
            assert_eq!(variance.denominator(), reference.1);
            assert_eq!(next_cache, expected.1);
            values = after;
            cache = next_cache;
        }
    }
}

fn reference_variance(values: &[u64]) -> (u128, u128) {
    let count = values.len() as u128;
    let total: u128 = values.iter().map(|value| u128::from(*value)).sum();
    let squared_total: u128 = values
        .iter()
        .map(|value| u128::from(*value) * u128::from(*value))
        .sum();
    (count * squared_total - total * total, count * count)
}
