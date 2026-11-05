#[path = "../src/bin/controller_support.rs"]
mod support;
use axon_uic::{
    Authority, Effect,
    behavior::Resettable,
    demand::{controller::*, *},
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{collections::BTreeSet, fs, path::PathBuf};

#[test]
fn delayed_feedback_has_a_hard_pending_limit_and_admission_cannot_choose_a_replacement() {
    let root = Temp::new();
    let auth = Authority::from([Effect::ReadData, Effect::WriteData, Effect::Simulate]);
    support::bootstrap(&root.0, support::DEV_SEED, &auth).unwrap();
    let mut runtime = support::runtime(support::catalog(&root.0, 0, &auth).unwrap(), 5).unwrap();
    let mut admission = support::traces(0, 37918713, 2);
    admission[0][0].output = 9;
    let refused = Procedure::discover(
        support::GOALS[0],
        support::traces(0, 7139161, 3),
        admission,
        &mut runtime,
        &auth,
        CompositionBudget::default(),
    )
    .unwrap();
    assert_eq!(refused.stop, CompositionStop::Refuted);
    assert!(refused.procedure.is_none());
    let admitted = Procedure::discover(
        support::GOALS[0],
        support::traces(0, 7139161, 3),
        support::traces(0, 37918713, 2),
        &mut runtime,
        &auth,
        CompositionBudget::default(),
    )
    .unwrap()
    .procedure
    .unwrap();
    let mut controller = GoalController::new(runtime);
    controller.install(admitted, &auth).unwrap();
    for id in 1..=64 {
        controller
            .propose(support::GOALS[0], id, "packet|7,8", &auth)
            .unwrap();
    }
    assert_eq!(controller.pending_count(), 64);
    assert!(
        controller
            .propose(support::GOALS[0], 65, "packet|7,8", &auth)
            .is_err()
    );
    assert!(controller.feedback(support::GOALS[0], 64, 0).unwrap());
    controller
        .propose(support::GOALS[0], 65, "packet|7,8", &auth)
        .unwrap();
    assert_eq!(controller.pending_count(), 64);
    assert!(!controller.feedback(support::GOALS[0], 1, 1).unwrap());
    assert_eq!(controller.pending_count(), 0);
    assert!(
        controller
            .propose(support::GOALS[0], 66, "packet|7,8", &auth)
            .is_err()
    );
}

#[test]
fn uncertain_effects_are_receipted_suspended_and_never_automatically_retried() {
    use axon_uic::behavior::BehaviorError;
    struct FailAfterEffect {
        calls: usize,
    }
    impl Resettable for FailAfterEffect {
        fn reset(&mut self) -> Result<(), BehaviorError> {
            Ok(())
        }
        fn step(&mut self, _action: u8) -> Result<u16, BehaviorError> {
            self.calls += 1;
            Err(BehaviorError::Invalid("external failure after effect"))
        }
    }
    let root = Temp::new();
    let auth = Authority::from([Effect::ReadData, Effect::WriteData, Effect::Simulate]);
    support::bootstrap(&root.0, support::DEV_SEED, &auth).unwrap();
    let mut runtime = support::runtime(support::catalog(&root.0, 0, &auth).unwrap(), 5).unwrap();
    let procedure = Procedure::discover(
        support::GOALS[0],
        support::traces(0, 7139161, 3),
        support::traces(0, 37918713, 2),
        &mut runtime,
        &auth,
        CompositionBudget::default(),
    )
    .unwrap()
    .procedure
    .unwrap();
    let mut controller = GoalController::new(runtime);
    controller.install(procedure, &auth).unwrap();
    controller
        .propose(support::GOALS[0], 1, "packet|7,8", &auth)
        .unwrap();
    let mut cursor = controller
        .runtime()
        .behavior_cursor("device", support::DEVICE)
        .unwrap();
    let mut adapter = FailAfterEffect { calls: 0 };
    let receipt = controller
        .dispatch(
            1,
            ("device", support::DEVICE),
            &mut cursor,
            &mut adapter,
            &BTreeSet::new(),
            &auth,
        )
        .unwrap();
    assert_eq!(receipt.status, DispatchStatus::Uncertain);
    assert_eq!(receipt.attempted.len(), 1);
    assert!(receipt.observations.is_empty() && cursor.suspended());
    assert!(
        controller
            .dispatch(
                1,
                ("device", support::DEVICE),
                &mut cursor,
                &mut adapter,
                &BTreeSet::new(),
                &auth
            )
            .is_err()
    );
    assert_eq!(adapter.calls, 1);
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "axon-goals-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!root.exists());
        Self(root)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        if let Ok(root) = self.0.canonicalize() {
            assert!(root.starts_with(std::env::temp_dir().canonicalize().unwrap()));
            fs::remove_dir_all(root).unwrap();
        }
    }
}
#[test]
fn learned_composition_is_durable_scoped_stateful_and_safe_under_changed_objectives() {
    let root = Temp::new();
    assert!(!support::RESERVES.contains(&support::DEV_SEED));
    let authority = Authority::from([Effect::ReadData, Effect::WriteData, Effect::Simulate]);
    let stats = support::bootstrap(&root.0, support::DEV_SEED, &authority).unwrap();
    assert!(
        stats.numeric_work > 0
            && stats.reactive_work > 0
            && stats.device_queries > 0
            && stats.device_steps > 0
    );
    let mut runtime = support::runtime(
        Catalog::load(root.0.join("modules.axseed"), &authority).unwrap(),
        1,
    )
    .unwrap();
    let mut procedures = Vec::new();
    for (goal, context) in support::GOALS.iter().enumerate() {
        let training = support::traces(goal, 7139161, 3);
        let admission = support::traces(goal, 37918713, 2);
        assert!(
            Procedure::discover(
                context,
                vec![],
                admission.clone(),
                &mut runtime,
                &authority,
                CompositionBudget::default()
            )
            .is_err()
        );
        let outcome = Procedure::discover(
            context,
            training,
            admission,
            &mut runtime,
            &authority,
            CompositionBudget::default(),
        )
        .unwrap();
        assert_eq!(outcome.stop, CompositionStop::Admitted);
        let p = outcome.procedure.unwrap();
        assert!(p.steps().len() >= 5);
        let path = root.0.join(format!("goal-{goal}.axgoal"));
        p.save_new(&path, &authority).unwrap();
        assert!(p.save_new(&path, &authority).is_err());
        assert!(Procedure::open(&path, "foreign", &mut runtime, &authority).is_err());
        let bytes = p.to_bytes();
        for n in 0..bytes.len() {
            assert!(Procedure::from_bytes(&bytes[..n], runtime.catalog()).is_err());
        }
        let mut broken = bytes.clone();
        broken[12] ^= 1;
        assert!(Procedure::from_bytes(&broken, runtime.catalog()).is_err());
        let reopened = Procedure::open(&path, context, &mut runtime, &authority).unwrap();
        assert_eq!(p.fingerprint(), reopened.fingerprint());
        let mut cursor = reopened.cursor();
        let before = cursor.clone();
        assert!(
            reopened
                .predict(context, &mut runtime, &mut cursor, "bad-input", &authority)
                .is_err()
        );
        assert_eq!(cursor, before);
        let mut resealed = bytes.clone();
        resealed[11 + context.len()] = 255;
        let end = resealed.len() - 8;
        let hash = resealed[..end].iter().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
        resealed[end..].copy_from_slice(&hash.to_le_bytes());
        assert!(Procedure::from_bytes(&resealed, runtime.catalog()).is_err());
        procedures.push(reopened);
    }
    let mut controller = GoalController::new(runtime);
    for p in procedures {
        controller.install(p, &authority).unwrap();
    }
    let mut previous = 0;
    let mut device = support::Device::new(support::DEV_SEED);
    let mut cursor = controller
        .runtime()
        .behavior_cursor("device", support::DEVICE)
        .unwrap();
    for i in 1..=80u64 {
        let goal = (i % 2) as usize;
        let values = [(i % 11) as i64, (i % 13) as i64];
        let expected = support::expectation(goal, values, &mut previous);
        let decision = controller
            .propose(
                support::GOALS[goal],
                i,
                &support::wire(i, values),
                &authority,
            )
            .unwrap();
        assert_eq!(decision.value, expected);
        assert!(
            controller
                .propose(
                    support::GOALS[goal],
                    i,
                    &support::wire(i, values),
                    &authority
                )
                .is_err()
        );
        let effects = device.effects();
        assert!(
            controller
                .dispatch(
                    i,
                    ("device", support::DEVICE),
                    &mut cursor,
                    &mut device,
                    &BTreeSet::new(),
                    &Authority::from([Effect::ReadData])
                )
                .is_err()
        );
        assert_eq!(device.effects(), effects);
        let receipt = controller
            .dispatch(
                i,
                ("device", support::DEVICE),
                &mut cursor,
                &mut device,
                &BTreeSet::new(),
                &authority,
            )
            .unwrap();
        assert_eq!(receipt.status, DispatchStatus::Reached);
        assert_eq!(device.state(), expected as u16);
        let effects = device.effects();
        assert!(
            controller
                .dispatch(
                    i,
                    ("device", support::DEVICE),
                    &mut cursor,
                    &mut device,
                    &BTreeSet::new(),
                    &authority
                )
                .is_err()
        );
        assert_eq!(device.effects(), effects);
        assert!(controller.feedback("foreign", i, expected).is_err());
        assert!(
            controller
                .feedback(support::GOALS[goal], i, expected)
                .unwrap()
        );
        controller.runtime_mut().unload_all();
    }
    assert_eq!(controller.runtime().stats().peak_active_modules, 1);
    assert!(controller.runtime().stats().evictions > 0);
    device.reset().unwrap();
    cursor = controller
        .runtime()
        .behavior_cursor("device", support::DEVICE)
        .unwrap();
    let decision = controller
        .propose(support::GOALS[0], 81, "node|7,8", &authority)
        .unwrap();
    device.change();
    let receipt = controller
        .dispatch(
            81,
            ("device", support::DEVICE),
            &mut cursor,
            &mut device,
            &BTreeSet::new(),
            &authority,
        )
        .unwrap();
    assert_eq!(receipt.status, DispatchStatus::Refuted);
    assert!(cursor.suspended());
    assert!(
        controller
            .feedback(support::GOALS[0], 81, decision.value)
            .unwrap()
    );
    support::learn_device(&root.0, 1, &device, &authority).unwrap();
    device.reset().unwrap();
    controller
        .rebind(
            support::runtime(support::catalog(&root.0, 1, &authority).unwrap(), 1).unwrap(),
            &authority,
        )
        .unwrap();
    cursor = controller
        .runtime()
        .behavior_cursor("device", support::DEVICE)
        .unwrap();
    let d = controller
        .propose(support::GOALS[0], 82, "new-source|7,8", &authority)
        .unwrap();
    assert_eq!(
        controller
            .dispatch(
                82,
                ("device", support::DEVICE),
                &mut cursor,
                &mut device,
                &BTreeSet::new(),
                &authority
            )
            .unwrap()
            .status,
        DispatchStatus::Reached
    );
    assert!(
        !controller
            .feedback(support::GOALS[0], 82, 1 - d.value)
            .unwrap()
    );
    assert!(
        controller
            .propose(support::GOALS[0], 83, "new-source|7,8", &authority)
            .is_err()
    );
    assert!(
        controller
            .propose(support::GOALS[1], 83, "new-source|7,8", &authority)
            .is_ok()
    );
}
