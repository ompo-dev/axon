use axon_uic::{
    AverageStrategy, FoldSpec, LiveAverage, LiveAverageStore, OnlineAveragePolicy, Replace,
    ReplaceDelta, SemanticArtifact,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct Meter;
static MEASURING: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

// One test in this executable isolates metering from other test workloads.
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && MEASURING.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(pointer, layout, new_size) };
        if !next.is_null() && MEASURING.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
        }
        next
    }
}

#[global_allocator]
static ALLOCATOR: Meter = Meter;

#[test]
fn incremental_execution_allocates_nothing_and_recovery_does_not_duplicate_source() {
    let semantic = SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap();
    let mut live = LiveAverage::materialize(7, vec![1; 524288], semantic).unwrap();
    let mut policy = OnlineAveragePolicy::new("allocation-test");
    let delta = ReplaceDelta::try_new(vec![Replace::new(0, 1, 2)]).unwrap();
    let inverse = ReplaceDelta::try_new(vec![Replace::new(0, 2, 1)]).unwrap();
    let empty = ReplaceDelta::try_new(vec![]).unwrap();
    ALLOCATED.store(0, Ordering::SeqCst);
    MEASURING.store(true, Ordering::SeqCst);
    for _ in 0..1000 {
        live.apply(7, live.version(), live.version() + 1, &delta)
            .unwrap();
        let event = live.version() + 1;
        policy
            .execute_and_learn(&mut live, event, &inverse)
            .unwrap();
    }
    MEASURING.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCATED.load(Ordering::SeqCst), 0);
    live.apply_with_strategy(live.version() + 1, &empty, AverageStrategy::Full)
        .unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("axon-memory-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let path = root.join("state.bin");
    drop(LiveAverageStore::create(&path, live).unwrap());
    ALLOCATED.store(0, Ordering::SeqCst);
    MEASURING.store(true, Ordering::SeqCst);
    let (store, report) = LiveAverageStore::open(&path).unwrap();
    MEASURING.store(false, Ordering::SeqCst);
    let bytes = ALLOCATED.load(Ordering::SeqCst);
    assert!(
        bytes <= 4 * 1024 * 1024 + 65536,
        "recovery allocated {bytes} bytes for a 4 MiB source"
    );
    assert_eq!(report.replayed_events, 0);
    assert_eq!(store.state().unwrap().query().numerator(), 524288);
    println!(
        "recovery_source_bytes=4194304 recovery_allocated_bytes={bytes} steady_allocated_bytes=0"
    );
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}
