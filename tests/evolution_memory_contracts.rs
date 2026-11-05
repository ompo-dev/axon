use axon_uic::numeric::{evolution::*, *};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};
struct Counter;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counter {
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
static ALLOCATOR: Counter = Counter;
#[test]
fn evolved_program_execution_does_not_run_the_population_or_allocate_on_every_input() {
    let data = |xs: &[f64]| {
        xs.iter()
            .map(|x| NumericExample::new(vec![*x], x * x))
            .collect()
    };
    let s = EvolvedSkill::evolve(
        "memory",
        "memory",
        data(&[-3.0, -2.0, 0.0, 1.0, 2.0, 3.0]),
        data(&[-4.0, 4.0, 5.0]),
        &[],
        91,
        EvolutionBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap()
    .skill
    .unwrap();
    let execution = s.program().compile_execution();
    let before = ALLOCS.load(Ordering::Relaxed);
    TRACK.store(true, Ordering::SeqCst);
    for i in 0..1_000_000 {
        let x = black_box(i as f64 / 13.0);
        assert_eq!(s.predict("memory", &[x]).unwrap(), x * x);
        assert_eq!(execution.evaluate(&[x]).unwrap(), x * x);
    }
    TRACK.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCS.load(Ordering::Relaxed) - before, 0);
}
