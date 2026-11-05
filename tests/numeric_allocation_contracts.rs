use axon_uic::numeric::*;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};
struct Count;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Count = Count;
#[test]
fn inference_and_feedback_use_no_heap_allocations_after_learning() {
    let skill = NumericSkill::learn(
        "calibration",
        "sensor/hot",
        [-7.0, 0.0, 3.0, 16.0]
            .into_iter()
            .map(|x| NumericExample::new(vec![x], 2.0 * x + 1.0))
            .collect(),
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap();
    let mut monitor = NumericMonitor::new(&skill);
    let execution = skill.program().compile_execution();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    TRACK.store(true, Ordering::SeqCst);
    for i in 0..100_000 {
        let x = black_box(i as f64 / 37.0);
        let result = monitor.predict(&[x]).unwrap();
        black_box(result);
        black_box(execution.evaluate(&[x]).unwrap());
        monitor.observe(&[x], 2.0 * x + 1.0).unwrap();
    }
    TRACK.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed) - before, 0);
    assert_eq!(monitor.observations(), 100_000);
    assert_eq!(skill.program().scratch_bytes(), 512);
}
