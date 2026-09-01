use axon_uic::{
    ChangeStructure, IncrementalOp, ModularU64, Replace, ReplaceDelta, SumFold, VectorU64,
};

#[test]
fn change_structures_preserve_zero_diff_and_incremental_sum_laws() {
    let scalar = ModularU64;
    let vector = VectorU64;
    let fold = SumFold;

    for seed in 0_u64..1_000 {
        let before = vec![seed, seed.wrapping_mul(3), !seed, seed.rotate_left(17)];
        let after: Vec<_> = before
            .iter()
            .enumerate()
            .map(|(index, value)| value.wrapping_add(seed).wrapping_add(index as u64))
            .collect();
        let input_delta = vector.diff(&after, &before).unwrap();
        let applied = vector.apply(&before, &input_delta).unwrap();
        let (old_total, cache) = fold.full(&before);
        let (output_delta, next_cache) = fold.delta(&input_delta, &cache).unwrap();
        let (new_total, new_cache) = fold.full(&applied);

        assert_eq!(applied, after);
        assert_eq!(scalar.apply(&before[0], &scalar.zero()).unwrap(), before[0]);
        assert_eq!(
            scalar
                .apply(&before[0], &scalar.diff(&after[0], &before[0]).unwrap())
                .unwrap(),
            after[0]
        );
        assert_eq!(scalar.apply(&old_total, &output_delta).unwrap(), new_total);
        assert_eq!(next_cache, new_cache);
    }
}

#[test]
fn replace_delta_is_canonical_and_incompatible_application_is_transactional() {
    let vector = VectorU64;
    let before = vec![4, 8, 15];
    let invalid =
        ReplaceDelta::try_new(vec![Replace::new(0, 4, 16), Replace::new(3, 0, 23)]).unwrap();

    assert_eq!(
        vector.apply(&before, &invalid),
        Err(axon_uic::ChangeError::IndexOutOfBounds(3))
    );
    assert_eq!(before, vec![4, 8, 15]);
    let stale = ReplaceDelta::try_new(vec![Replace::new(0, 5, 16)]).unwrap();
    assert!(vector.apply(&before, &stale).is_err());
    assert_eq!(before, vec![4, 8, 15]);
    assert!(ReplaceDelta::try_new(vec![Replace::new(1, 8, 12), Replace::new(1, 12, 20)]).is_err());
}
