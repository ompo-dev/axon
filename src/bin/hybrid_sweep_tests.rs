use super::*;

fn compile_final<'a>(
    updates: &'a [PointUpdate],
    words: usize,
    shard_words: usize,
) -> Result<CompiledChanges<'a>, String> {
    compile_changes_timed(updates, words, shard_words).map(|(compiled, _)| compiled)
}

#[test]
fn compiler_selects_all_four_local_strategies() {
    let shard_words = 128;
    let updates = build_mixed_updates(shard_words).unwrap();
    let compiled = compile_final(&updates, shard_words * SHARD_COUNT, shard_words).unwrap();

    assert_eq!(compiled.counts.full_local, DENSE_SHARDS);
    assert_eq!(compiled.counts.coalesced_delta, 1);
    assert_eq!(compiled.counts.raw_delta, 1);
    assert_eq!(compiled.counts.skip, SHARD_COUNT - DENSE_SHARDS - 2);
}

#[test]
fn hybrid_matches_all_global_paths() {
    let shard_words = 128;
    let seed = build_data(shard_words * SHARD_COUNT).unwrap();
    let updates = build_mixed_updates(shard_words).unwrap();
    let full = run_full_global(&seed, &updates).unwrap();
    let delta = run_delta_global(&seed, &updates).unwrap();
    let coalesced = run_coalesced_global(&seed, &updates).unwrap();
    let hybrid = run_hybrid(&seed, &updates, shard_words).unwrap();

    assert_eq!(full.total, delta.total);
    assert_eq!(full.total, coalesced.total);
    assert_eq!(full.total, hybrid.total);
}

#[test]
fn permuted_round_measures_every_path_with_exact_parity() {
    let shard_words = 128;
    let seed = build_data(shard_words * SHARD_COUNT).unwrap();
    let updates = build_mixed_updates(shard_words).unwrap();
    let template = compile_final(&updates, seed.len(), shard_words).unwrap();
    let (full, delta, coalesced, hybrid, oracle, fabric) =
        run_round(3, &seed, &updates, shard_words, &template).unwrap();

    assert_eq!(full.total, delta.total);
    assert_eq!(full.total, coalesced.total);
    assert_eq!(full.total, hybrid.total);
    assert_eq!(full.total, oracle.total);
    assert_eq!(full.total, fabric.total);
}

#[test]
fn oracle_executor_matches_all_global_paths() {
    let shard_words = 128;
    let seed = build_data(shard_words * SHARD_COUNT).unwrap();
    let updates = build_mixed_updates(shard_words).unwrap();
    let plan = compile_final(&updates, seed.len(), shard_words).unwrap();
    let full = run_full_global(&seed, &updates).unwrap();
    let delta = run_delta_global(&seed, &updates).unwrap();
    let coalesced = run_coalesced_global(&seed, &updates).unwrap();
    let oracle = run_hybrid_oracle(&seed, &plan).unwrap();

    assert_eq!(full.total, delta.total);
    assert_eq!(full.total, coalesced.total);
    assert_eq!(full.total, oracle.total);
}

#[test]
fn oracle_rejects_seed_shape_mismatch() {
    let shard_words = 128;
    let updates = build_mixed_updates(shard_words).unwrap();
    let plan = compile_final(&updates, shard_words * SHARD_COUNT, shard_words).unwrap();
    let wrong_shape = build_data(shard_words * (SHARD_COUNT - 1)).unwrap();

    assert!(run_hybrid_oracle(&wrong_shape, &plan).is_err());
}

#[test]
fn timed_compiler_preserves_partition_and_accounts_for_its_phases() {
    let shard_words = 128;
    let updates = build_mixed_updates(shard_words).unwrap();
    let (compiled, timing) =
        compile_changes_timed(&updates, shard_words * SHARD_COUNT, shard_words).unwrap();

    assert_eq!(
        compiled.counts.skip
            + compiled.counts.raw_delta
            + compiled.counts.coalesced_delta
            + compiled.counts.full_local,
        SHARD_COUNT
    );
    assert!(timing.total >= timing.validate_and_index);
    assert!(timing.total >= timing.classify_and_materialize);
    for (shard, change) in compiled.shards.iter().enumerate() {
        let start = shard * shard_words;
        let end = start + shard_words;
        assert!(
            change
                .updates
                .as_slice()
                .iter()
                .all(|update| (start..end).contains(&update.index()))
        );
    }
}

#[test]
fn compiler_refuses_to_coalesce_when_intermediate_events_are_observed() {
    let updates = [PointUpdate::new(0, 1), PointUpdate::new(0, 2)];

    assert!(
        compile_changes_timed_at_frontier(
            &updates,
            128,
            2,
            ObservationFrontier::IntermediateObserved,
        )
        .is_err()
    );
}

#[test]
fn compiler_selects_full_at_half_support_and_delta_below_it() {
    let shard_words = 128;
    let half: Vec<_> = (0..64).map(|index| PointUpdate::new(index, 1)).collect();
    let below_half: Vec<_> = (0..63).map(|index| PointUpdate::new(index, 1)).collect();

    let at_half = compile_final(&half, shard_words * SHARD_COUNT, shard_words).unwrap();
    let below = compile_final(&below_half, shard_words * SHARD_COUNT, shard_words).unwrap();

    assert_eq!(at_half.shards[0].strategy, ShardStrategy::FullLocal);
    assert_eq!(below.shards[0].strategy, ShardStrategy::RawDelta);
}

#[test]
fn change_fabric_keeps_plan_incremental_and_matches_full() {
    let shard_words = 128;
    let seed = build_data(shard_words * SHARD_COUNT).unwrap();
    let updates = build_mixed_updates(shard_words).unwrap();
    let fabric = build_change_fabric(
        &updates,
        seed.len(),
        shard_words,
        ObservationFrontier::FinalStateOnly,
    )
    .unwrap();
    let full = run_full_global(&seed, &updates).unwrap();
    let mut data = clone_data(&seed).unwrap();
    let mut shard_totals: Vec<_> = data.chunks_exact(shard_words).map(checksum).collect();
    let fabric_total = execute_fabric_hybrid(&mut data, &mut shard_totals, &fabric).unwrap();

    assert_eq!(full.total, fabric_total);
    assert_eq!(fabric.counts.full_local, DENSE_SHARDS);
    assert_eq!(fabric.counts.coalesced_delta, 1);
    assert_eq!(fabric.counts.raw_delta, 1);
    assert_eq!(fabric.counts.skip, SHARD_COUNT - DENSE_SHARDS - 2);
}

#[test]
fn change_fabric_counts_ingest_and_refuses_observable_coalescing() {
    let updates = [PointUpdate::new(0, 1), PointUpdate::new(0, 2)];

    assert!(
        build_change_fabric(&updates, 128, 2, ObservationFrontier::IntermediateObserved,).is_err()
    );
}

#[test]
fn compiler_rejects_out_of_bounds_updates() {
    let update = PointUpdate::new(128, 1);

    assert!(compile_final(&[update], 128, 2).is_err());
}

#[test]
fn compiler_rejects_unordered_shards_and_cli_rejects_bad_bounds() {
    let updates = [PointUpdate::new(4, 1), PointUpdate::new(2, 2)];

    assert!(compile_final(&updates, 128, 2).is_err());
    assert!(parse_bounded("0", 1, 10, "--runs").is_err());
    assert!(parse_bounded("11", 1, 10, "--runs").is_err());
}
