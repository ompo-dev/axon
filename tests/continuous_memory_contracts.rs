use axon_uic::numeric::continuous::*;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};
struct Count;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Count = Count;
#[test]
fn a_million_live_predictions_and_labels_use_fixed_memory_after_admission() {
    let mut agent =
        ContinuousAgent::new("sensor/hot", 1, None, ContinuousPolicy::default()).unwrap();
    for id in 1..=200 {
        let x = id as f64 / 7.0;
        agent.observe(id, &[x]).unwrap();
        agent.feedback(id, 2.0 * x + 1.0).unwrap();
        if let Some(job) = agent.take_learning_job() {
            agent.submit_learning_result(job.run(None));
        }
    }
    assert_eq!(agent.phase(), ContinuousPhase::Active);
    let bytes = agent.retained_bytes();
    let before = ALLOCS.load(Ordering::Relaxed);
    TRACK.store(true, Ordering::SeqCst);
    for id in 201..=1000200 {
        let x = black_box(id as f64 / 37.0);
        black_box(agent.observe(id, &[x]).unwrap());
        black_box(agent.feedback(id, 2.0 * x + 1.0).unwrap());
        assert!(agent.take_learning_job().is_none());
    }
    TRACK.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCS.load(Ordering::Relaxed) - before, 0);
    assert_eq!(agent.retained_bytes(), bytes);
    assert_eq!(agent.statistics().drifts, 0);
    assert!(agent.pending_count() == 0 && agent.window_count() <= 16);
}
