use axon_uic::{
    FoldSpec, LiveAverage, LiveAverageStore, LiveError, Replace, ReplaceDelta, SemanticArtifact,
    StoreLimits,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static ID: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "axon-adversarial-{}-{nonce}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> PathBuf {
        self.0.join("state.bin")
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // This directory is exclusively owned by this test fixture.
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn state() -> LiveAverage {
    LiveAverage::materialize(
        7,
        vec![1, 2, u64::MAX],
        SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap(),
    )
    .unwrap()
}

fn delta(changes: &[(usize, u64, u64)]) -> ReplaceDelta {
    ReplaceDelta::try_new(
        changes
            .iter()
            .map(|&(i, o, n)| Replace::new(i, o, n))
            .collect(),
    )
    .unwrap()
}

fn history(fixture: &Fixture) -> (Vec<u8>, Vec<(usize, LiveAverage)>) {
    let path = fixture.path();
    let mut store = LiveAverageStore::create(&path, state()).unwrap();
    let mut checkpoints = vec![(
        fs::metadata(&path).unwrap().len() as usize,
        store.state().unwrap().clone(),
    )];
    for (i, changes) in [
        delta(&[(0, 1, 10), (2, u64::MAX, 3)]),
        delta(&[(1, 2, u64::MAX)]),
        delta(&[]),
    ]
    .iter()
    .enumerate()
    {
        store.apply(7, i as u64, i as u64 + 1, changes).unwrap();
        checkpoints.push((
            fs::metadata(&path).unwrap().len() as usize,
            store.state().unwrap().clone(),
        ));
    }
    drop(store);
    (fs::read(path).unwrap(), checkpoints)
}

#[test]
fn every_byte_corruption_is_rejected_without_repairing_complete_records() {
    let original = Fixture::new();
    let (bytes, _) = history(&original);
    let target = Fixture::new();
    for index in 0..bytes.len() {
        let mut damaged = bytes.clone();
        damaged[index] ^= 0x80;
        fs::write(target.path(), &damaged).unwrap();
        assert!(
            LiveAverageStore::open(target.path()).is_err(),
            "accepted byte corruption at {index}"
        );
        assert_eq!(
            fs::read(target.path()).unwrap(),
            damaged,
            "modified corrupt file at {index}"
        );
    }
    println!("complete single-byte corruption cases={}", bytes.len());
}

#[test]
fn every_truncated_history_recovers_only_complete_events_and_can_continue() {
    let original = Fixture::new();
    let (bytes, checkpoints) = history(&original);
    let target = Fixture::new();
    for cut in 0..=bytes.len() {
        fs::write(target.path(), &bytes[..cut]).unwrap();
        let Some((length, expected)) = checkpoints.iter().rev().find(|(len, _)| *len <= cut) else {
            assert!(
                LiveAverageStore::open(target.path()).is_err(),
                "accepted partial snapshot at {cut}"
            );
            assert_eq!(fs::read(target.path()).unwrap(), bytes[..cut]);
            continue;
        };
        let (mut recovered, report) = LiveAverageStore::open(target.path()).unwrap();
        assert_eq!(
            recovered.state().unwrap(),
            expected,
            "wrong prefix at {cut}"
        );
        assert_eq!(report.replayed_events, expected.version());
        assert_eq!(report.discarded_tail_bytes as usize, cut - length);
        let v = expected.version();
        recovered
            .apply(7, v, v + 1, &delta(&[(0, expected.values()[0], 91)]))
            .unwrap();
        let continued = recovered.state().unwrap().clone();
        drop(recovered);
        let reopened = LiveAverageStore::open(target.path()).unwrap().0;
        assert_eq!(reopened.state().unwrap(), &continued);
    }
    println!("truncated history cases={}", bytes.len() + 1);
}

fn replace_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn reseal(bytes: &mut [u8], frame: usize) {
    let len = u64::from_le_bytes(bytes[frame..frame + 8].try_into().unwrap()) as usize;
    let hash = bytes[frame + 24..frame + 24 + len]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        });
    replace_u64(bytes, frame + 16, hash);
}

#[test]
fn valid_checksums_do_not_bypass_semantic_and_sequence_validation() {
    let original = Fixture::new();
    let (bytes, checkpoints) = history(&original);
    let event = checkpoints[0].0;
    let payload = event + 24;
    let target = Fixture::new();
    let cases = [
        (8, 40, 1),                      // snapshot version/event mismatch
        (8, 56, 0),                      // wrong artifact
        (8, 64, 0),                      // empty source
        (8, 64, u64::MAX),               // count overflow
        (8, 72, 42),                     // incorrect aggregate
        (event, payload, 8),             // wrong source
        (event, payload + 8, 1),         // wrong base version
        (event, payload + 16, 0),        // replayed sequence
        (event, payload + 16, 2),        // sequence gap
        (event, payload + 24, 1),        // wrong replacement count
        (event, payload + 24, u64::MAX), // count overflow
        (event, payload + 32, 3),        // index out of range
        (event, payload + 40, 99),       // stale old value
        (event, payload + 56, 0),        // duplicate index
    ];
    for (frame, offset, value) in cases {
        let mut damaged = bytes.clone();
        replace_u64(&mut damaged, offset, value);
        reseal(&mut damaged, frame);
        fs::write(target.path(), &damaged).unwrap();
        assert!(
            LiveAverageStore::open(target.path()).is_err(),
            "accepted semantic corruption at {offset}"
        );
        assert_eq!(fs::read(target.path()).unwrap(), damaged);
    }
    // A valid maximum-version checkpoint may be queried, but must never wrap.
    let mut maximum = bytes[..event].to_vec();
    replace_u64(&mut maximum, 40, u64::MAX);
    replace_u64(&mut maximum, 48, u64::MAX);
    reseal(&mut maximum, 8);
    fs::write(target.path(), &maximum).unwrap();
    let mut store = LiveAverageStore::open(target.path()).unwrap().0;
    assert!(matches!(
        store.apply(7, u64::MAX, 0, &delta(&[])),
        Err(LiveError::WrongVersion)
    ));
    assert_eq!(store.state().unwrap().version(), u64::MAX);
    drop(store);
    assert_eq!(fs::read(target.path()).unwrap(), maximum);
}

#[test]
fn resource_limits_reject_before_mutation_and_compaction_renews_event_budget() {
    let fixture = Fixture::new();
    let limits = StoreLimits {
        max_source_values: 3,
        max_changes_per_event: 1,
        max_journal_events: 1,
    };
    let too_small = StoreLimits {
        max_source_values: 2,
        ..limits
    };
    assert!(matches!(
        LiveAverageStore::create_with_limits(fixture.path(), state(), too_small),
        Err(LiveError::ResourceLimit(_))
    ));
    assert!(!fixture.path().exists());
    drop(LiveAverageStore::create_with_limits(fixture.path(), state(), limits).unwrap());
    let before = fs::read(fixture.path()).unwrap();
    let mut store = LiveAverageStore::open_with_limits(fixture.path(), limits)
        .unwrap()
        .0;
    assert!(matches!(
        store.apply(7, 0, 1, &delta(&[(0, 1, 3), (1, 2, 4)])),
        Err(LiveError::ResourceLimit(_))
    ));
    assert_eq!(store.state().unwrap(), &state());
    drop(store);
    assert_eq!(fs::read(fixture.path()).unwrap(), before);
    let mut store = LiveAverageStore::open_with_limits(fixture.path(), limits)
        .unwrap()
        .0;
    store.apply(7, 0, 1, &delta(&[(0, 1, 3)])).unwrap();
    drop(store);
    let after = fs::read(fixture.path()).unwrap();
    let mut store = LiveAverageStore::open_with_limits(fixture.path(), limits)
        .unwrap()
        .0;
    assert!(matches!(
        store.apply(7, 1, 2, &delta(&[])),
        Err(LiveError::ResourceLimit(_))
    ));
    drop(store);
    assert_eq!(fs::read(fixture.path()).unwrap(), after);
    for restricted in [
        too_small,
        StoreLimits {
            max_changes_per_event: 0,
            ..limits
        },
        StoreLimits {
            max_journal_events: 0,
            ..limits
        },
    ] {
        assert!(matches!(
            LiveAverageStore::open_with_limits(fixture.path(), restricted),
            Err(LiveError::ResourceLimit(_))
        ));
        assert_eq!(fs::read(fixture.path()).unwrap(), after);
    }
    let mut store = LiveAverageStore::open_with_limits(fixture.path(), limits)
        .unwrap()
        .0;
    assert!(store.apply(7, 1, 2, &delta(&[])).is_err());
    store.compact().unwrap();
    store.apply(7, 1, 2, &delta(&[(1, 2, 0)])).unwrap();
    drop(store);
    let (store, recovery) = LiveAverageStore::open_with_limits(fixture.path(), limits).unwrap();
    assert_eq!(recovery.replayed_events, 1);
    assert_eq!(store.state().unwrap().version(), 2);
    assert_eq!(store.state().unwrap().values(), &[3, 0, u64::MAX]);
}
