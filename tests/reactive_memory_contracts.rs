use axon_uic::reactive::{agent::*, *};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};
struct Count;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static ALLOCATOR: Count = Count;
#[test]
fn a_million_admitted_reactive_steps_and_feedbacks_allocate_nothing_new() {
    let mut a =
        ReactiveAgent::new("memory/stream", 1, Vec::new(), ReactiveBudget::default()).unwrap();
    let mut value = 0;
    let mut id = 0;
    while a.phase() != ReactivePhase::Active {
        id += 1;
        assert!(id <= 64);
        let x = a
            .suggest_probe()
            .map(|x| x[0])
            .unwrap_or([0, 1, 2, 0, 1, 1, 0, 2][(id as usize - 1) % 8]);
        value = if x > 0 { value + 1 } else { 0 };
        a.observe(id, &[x]).unwrap();
        a.feedback(id, value).unwrap();
    }
    id += 1;
    a.observe(id, &[0]).unwrap();
    a.feedback(id, 0).unwrap();
    value = 0;
    let bytes = a.retained_bytes();
    let before = ALLOCS.load(Ordering::Relaxed);
    TRACK.store(true, Ordering::SeqCst);
    for tick in 1..=1_000_000 {
        let x = black_box(if tick % 257 == 0 { 0 } else { 1 });
        value = if x > 0 { value + 1 } else { 0 };
        id += 1;
        let p = a.observe(id, &[x]).unwrap();
        assert_eq!(p.value, Some(value));
        a.feedback(id, value).unwrap();
        assert!(a.suggest_probe().is_none());
        assert!(a.suggest_probe_with_bounds(&[(-32, 32)]).unwrap().is_none());
    }
    TRACK.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCS.load(Ordering::Relaxed) - before, 0);
    assert_eq!(a.retained_bytes(), bytes);
    assert_eq!(a.statistics().drifts, 0);
}
