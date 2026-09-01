use axon_uic::{
    ChangeStructure, DeltaForge, FoldSpec, ForgeError, MaintenanceState, UpdateRule, VectorU64,
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
fn forge_refuses_minimum_until_auxiliary_state_is_derived() {
    assert_eq!(
        DeltaForge::synthesize(FoldSpec::MinU64),
        Err(ForgeError::UnsupportedFold(FoldSpec::MinU64))
    );
}
