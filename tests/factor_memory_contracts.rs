use axon_uic::{
    Authority, Effect,
    behavior_lab::Rng,
    factor::*,
    factor_lab::{self, Environment, Kind},
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
struct Meter;
static MEASURING: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicUsize = AtomicUsize::new(0);
// One test per executable isolates the allocator window.
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() && MEASURING.load(Ordering::Relaxed) {
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, n: usize) -> *mut u8 {
        let p = unsafe { System.realloc(p, layout, n) };
        if !p.is_null() && MEASURING.load(Ordering::Relaxed) {
            BYTES.fetch_add(n, Ordering::Relaxed);
        }
        p
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;
#[test]
fn million_joint_observations_allocate_nothing_after_learning() {
    let mut env = Environment::random(Kind::Cycles, 6, 371831).unwrap();
    let mut oracle = FrameCache::new(
        env.clone(),
        env.actions(),
        env.dimensions(),
        FrameBudget::default(),
        Authority::from([Effect::Simulate]),
    )
    .unwrap();
    let model = learn(env.actions(), env.dimensions(), "hot-frame", &mut oracle).unwrap();
    assert!(
        factor_lab::audit(&model, &env, 65536)
            .unwrap()
            .counterexample
            .is_none()
    );
    let mut tracker = FactorTracker::new(&model, "hot-frame").unwrap();
    let mut rng = Rng::new(51911);
    BYTES.store(0, Ordering::SeqCst);
    MEASURING.store(true, Ordering::SeqCst);
    for _ in 0..1_000_000 {
        let action = rng.pick(env.actions() as usize) as u8;
        let expected = tracker.predict_next(action).unwrap();
        let actual = env.step(action).unwrap();
        assert_eq!(expected, actual);
        tracker.observe(action, actual).unwrap();
    }
    MEASURING.store(false, Ordering::SeqCst);
    let allocated = BYTES.load(Ordering::SeqCst);
    assert_eq!(allocated, 0);
    println!(
        "events={} tracker_bytes={} model_bytes={} factor_states={} product_upper={} hot_allocated_bytes={allocated}",
        tracker.observations(),
        std::mem::size_of_val(&tracker),
        model.encoded_model_bytes(),
        model.factor_states(),
        model.product_states()
    );
}
