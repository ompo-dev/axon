use axon_uic::{behavior::*, behavior_lab::*};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
struct Meter;
static MEASURING: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicUsize = AtomicUsize::new(0);
// A single test in this executable isolates the allocator window.
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
        let next = unsafe { System.realloc(p, layout, n) };
        if !next.is_null() && MEASURING.load(Ordering::Relaxed) {
            BYTES.fetch_add(n, Ordering::Relaxed);
        }
        next
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;
#[test]
fn million_observations_allocate_nothing_in_the_learned_monitor() {
    let mut rng = Rng::new(713010);
    let mut hidden = HiddenMachine::generate(&mut rng);
    let actions = hidden.actions;
    let mut oracle = CachedOracle::new(hidden.clone(), BehaviorBudget::default());
    let model = discover(actions, "monitor", BehaviorBudget::default(), &mut oracle)
        .unwrap()
        .0;
    assert!(
        audit_machine(&model, actions, 0, |s, a| Ok(hidden.transition(s, a)))
            .unwrap()
            .counterexample
            .is_none()
    );
    let mut tracker = BehaviorTracker::new(&model, "monitor").unwrap();
    BYTES.store(0, Ordering::SeqCst);
    MEASURING.store(true, Ordering::SeqCst);
    for _ in 0..1_000_000 {
        let action = rng.pick(actions as usize) as u8;
        let expected = tracker.predict_next(action).unwrap();
        let observed = hidden.step(action).unwrap();
        assert_eq!(expected, observed);
        tracker.observe(action, observed).unwrap();
    }
    MEASURING.store(false, Ordering::SeqCst);
    let allocated = BYTES.load(Ordering::SeqCst);
    assert_eq!(allocated, 0);
    let monitor = std::mem::size_of_val(&tracker);
    println!(
        "events={} monitor_bytes={monitor} model_encoded_bytes={} hot_allocated_bytes={allocated}",
        tracker.observations(),
        model.encoded_bytes()
    );
}
