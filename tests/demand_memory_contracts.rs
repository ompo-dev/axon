use axon_uic::{Authority, Effect, TextBudget, TextExample, TextSkill, demand::*};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
};
struct Meter;
static LIVE: AtomicI64 = AtomicI64::new(0);
static BASE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);
static MEASURE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
fn add(n: i64) {
    let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    if MEASURE.load(Ordering::Relaxed) {
        PEAK.fetch_max(live - BASE.load(Ordering::Relaxed), Ordering::Relaxed);
        if n > 0 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
    }
}
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            add(l.size() as i64);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        add(-(l.size() as i64));
        unsafe { System.dealloc(p, l) };
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            add(n as i64 - l.size() as i64);
        }
        q
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;

// Only one test in this executable: compare payload accounting with actual Rust allocations.
#[test]
fn demand_loading_releases_payload_and_hot_access_allocates_nothing() {
    let root = std::env::temp_dir().join(format!(
        "axon-demand-meter-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let authority = Authority::from([Effect::ReadData, Effect::WriteData]);
    let mut c = Catalog::new(&root);
    for i in 0..32 {
        let id = format!("skill-{i}");
        let file = format!("skill-{i}.axtext");
        TextSkill::learn(
            &id,
            &id,
            vec![
                TextExample::new("Ana Maria|Silva", "Silva, Ana Maria"),
                TextExample::new("Bruno|Costa", "Costa, Bruno"),
            ],
            TextBudget::default(),
        )
        .unwrap()
        .save_new(root.join(&file))
        .unwrap();
        c.register(
            ModuleSpec {
                id: &id,
                kind: ModuleKind::Text,
                file: &file,
                input: "raw",
                output: "label",
                cost: 1,
            },
            &authority,
        )
        .unwrap();
    }
    let mut r = DemandRuntime::new(c, Seed::default()).unwrap();
    let before = LIVE.load(Ordering::SeqCst);
    BASE.store(before, Ordering::SeqCst);
    PEAK.store(0, Ordering::SeqCst);
    MEASURE.store(true, Ordering::SeqCst);
    r.with_module("skill-0", "skill-0", &authority, |_| Ok(()))
        .unwrap();
    MEASURE.store(false, Ordering::SeqCst);
    let retained = LIVE.load(Ordering::SeqCst) - before;
    let cold_peak = PEAK.load(Ordering::SeqCst);
    assert_eq!(retained, r.stats().active_bytes as i64);
    ALLOCATIONS.store(0, Ordering::SeqCst);
    MEASURE.store(true, Ordering::SeqCst);
    for _ in 0..100_000 {
        r.with_module("skill-0", "skill-0", &authority, |_| Ok(()))
            .unwrap();
    }
    MEASURE.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
    r.unload_all();
    assert_eq!(LIVE.load(Ordering::SeqCst), before);
    println!(
        "catalog_modules=32 retained_module_bytes={retained} cold_peak_additional_rust_bytes={cold_peak} hot_accesses=100000 hot_allocations=0 metadata_bytes={}",
        r.metadata_bytes()
    );
    drop(r);
    std::fs::remove_dir_all(root).unwrap();
}
