use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{FoldSpec, LiveAverage, LiveError, Replace, ReplaceDelta, SemanticArtifact};

const MAGIC: &[u8; 8] = b"AXLIVE01";
const HEADER_BYTES: u64 = 24;
static CHECKPOINT_ID: AtomicU64 = AtomicU64::new(0);

/// Single-writer snapshot plus append-only event log. A sealed, synced frame commits
/// an event; a crash after sync but before acknowledgement may still commit it.
pub struct LiveAverageStore {
    file: File,
    path: PathBuf,
    _writer_lock: File,
    state: LiveAverage,
    needs_recovery: bool,
    limits: StoreLimits,
    journal_events: u64,
}

/// Resource limits apply before allocations or durable writes. Compacting resets
/// the journal-event budget without changing source version or event sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoreLimits {
    pub max_source_values: u64,
    pub max_changes_per_event: u64,
    pub max_journal_events: u64,
}

impl Default for StoreLimits {
    fn default() -> Self {
        Self {
            max_source_values: 134_217_728,
            max_changes_per_event: 1_048_576,
            max_journal_events: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecoveryReport {
    pub replayed_events: u64,
    pub discarded_tail_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactionReport {
    pub bytes_before: u64,
    pub bytes_after: u64,
    pub checkpoint_event: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckpointStage {
    Created,
    Synced,
    Published,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AppendStage {
    HeaderWritten,
    PayloadWritten,
    Synced,
    Applied,
}

impl LiveAverageStore {
    /// Creates a new file, never replacing an existing source.
    pub fn create(path: impl AsRef<Path>, state: LiveAverage) -> Result<Self, LiveError> {
        Self::create_with_limits(path, state, StoreLimits::default())
    }

    pub fn create_with_limits(
        path: impl AsRef<Path>,
        state: LiveAverage,
        limits: StoreLimits,
    ) -> Result<Self, LiveError> {
        if state.values().len() as u64 > limits.max_source_values {
            return Err(LiveError::ResourceLimit("source values"));
        }
        let path = resolved_path(path.as_ref())?;
        let writer_lock = writer_lock(&path)?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)?;
        file.try_lock().map_err(|e| LiveError::Io(e.into()))?;
        write_snapshot(&mut file, &state)?;
        Ok(Self {
            file,
            path,
            _writer_lock: writer_lock,
            state,
            needs_recovery: false,
            limits,
            journal_events: 0,
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<(Self, RecoveryReport), LiveError> {
        Self::open_with_limits(path, StoreLimits::default())
    }

    pub fn open_with_limits(
        path: impl AsRef<Path>,
        limits: StoreLimits,
    ) -> Result<(Self, RecoveryReport), LiveError> {
        let path = fs::canonicalize(path)?;
        let writer_lock = writer_lock(&path)?;
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        file.try_lock().map_err(|e| LiveError::Io(e.into()))?;
        let mut magic = [0; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(LiveError::InvalidRecord);
        }
        let file_len = file.metadata()?.len();
        let mut state = read_snapshot(&mut file, file_len, limits)?;
        let count = state.values().len() as u64;
        let max_event_bytes = 32_u64
            .checked_add(
                count
                    .min(limits.max_changes_per_event)
                    .checked_mul(24)
                    .ok_or(LiveError::InvalidRecord)?,
            )
            .ok_or(LiveError::InvalidRecord)?;
        let mut report = RecoveryReport::default();
        loop {
            let start = file.stream_position()?;
            if start == file_len {
                break;
            }
            if report.replayed_events >= limits.max_journal_events {
                return Err(LiveError::ResourceLimit("journal events"));
            }
            let Some(record) = read_frame(&mut file, file_len, max_event_bytes)? else {
                report.discarded_tail_bytes = file_len - start;
                file.set_len(start)?;
                file.sync_all()?;
                file.seek(SeekFrom::Start(start))?;
                break;
            };
            let mut reader = record.as_slice();
            let source_id = take_u64(&mut reader)?;
            let version = take_u64(&mut reader)?;
            let event = take_u64(&mut reader)?;
            let changes = take_u64(&mut reader)?;
            if changes.checked_mul(24) != Some(reader.len() as u64) {
                return Err(LiveError::InvalidRecord);
            }
            let mut replacements = Vec::new();
            replacements
                .try_reserve_exact(reader.len() / 24)
                .map_err(|_| LiveError::ResourceLimit("event allocation"))?;
            while !reader.is_empty() {
                let index = usize::try_from(take_u64(&mut reader)?)
                    .map_err(|_| LiveError::InvalidRecord)?;
                let old = take_u64(&mut reader)?;
                let new = take_u64(&mut reader)?;
                replacements.push(Replace::new(index, old, new));
            }
            let delta = ReplaceDelta::try_new(replacements).map_err(LiveError::Change)?;
            state.apply(source_id, version, event, &delta)?;
            report.replayed_events += 1;
        }
        Ok((
            Self {
                file,
                path,
                _writer_lock: writer_lock,
                state,
                needs_recovery: false,
                limits,
                journal_events: report.replayed_events,
            },
            report,
        ))
    }

    pub fn state(&self) -> Result<&LiveAverage, LiveError> {
        if self.needs_recovery {
            return Err(LiveError::NeedsRecovery);
        }
        Ok(&self.state)
    }

    /// Writes a complete synced snapshot beside the journal, then replaces it.
    /// The stable sidecar lock stays held across replacement of the data inode.
    pub fn compact(&mut self) -> Result<CompactionReport, LiveError> {
        self.compact_observed(|_| {})
    }

    fn compact_observed(
        &mut self,
        mut observe: impl FnMut(CheckpointStage),
    ) -> Result<CompactionReport, LiveError> {
        self.state()?;
        let before = self.file.metadata()?.len();
        let event = self.state.last_event();
        let mut name = self.path.as_os_str().to_os_string();
        name.push(format!(
            ".checkpoint-{}-{}",
            std::process::id(),
            CHECKPOINT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let temporary = PathBuf::from(name);
        let mut replacement = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        replacement
            .try_lock()
            .map_err(|e| LiveError::Io(e.into()))?;
        observe(CheckpointStage::Created);
        if let Err(error) = write_snapshot(&mut replacement, &self.state) {
            drop(replacement);
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        let after = replacement.metadata()?.len();
        observe(CheckpointStage::Synced);
        self.needs_recovery = true;
        if let Err(error) = fs::rename(&temporary, &self.path) {
            drop(replacement);
            let _ = fs::remove_file(&temporary);
            return Err(error.into());
        }
        observe(CheckpointStage::Published);
        self.file = replacement;
        // Directory synchronization is available through std on Unix. Windows
        // process-crash tests do not establish directory durability after power loss.
        #[cfg(unix)]
        File::open(self.path.parent().expect("resolved parent"))?.sync_all()?;
        self.needs_recovery = false;
        self.journal_events = 0;
        Ok(CompactionReport {
            bytes_before: before,
            bytes_after: after,
            checkpoint_event: event,
        })
    }

    pub fn apply(
        &mut self,
        source_id: u64,
        version: u64,
        event: u64,
        delta: &ReplaceDelta,
    ) -> Result<(), LiveError> {
        self.apply_observed(source_id, version, event, delta, |_| {})
    }

    fn apply_observed(
        &mut self,
        source_id: u64,
        version: u64,
        event: u64,
        delta: &ReplaceDelta,
        mut observe: impl FnMut(AppendStage),
    ) -> Result<(), LiveError> {
        self.state()?.validate(source_id, version, event, delta)?;
        if delta.changes().len() as u64 > self.limits.max_changes_per_event {
            return Err(LiveError::ResourceLimit("changes per event"));
        }
        if self.journal_events >= self.limits.max_journal_events {
            return Err(LiveError::ResourceLimit("journal events"));
        }
        let bytes = delta
            .changes()
            .len()
            .checked_mul(24)
            .and_then(|n| n.checked_add(32))
            .ok_or(LiveError::ResourceLimit("event size"))?;
        let mut record = Vec::new();
        record
            .try_reserve_exact(bytes)
            .map_err(|_| LiveError::ResourceLimit("event allocation"))?;
        for value in [source_id, version, event, delta.changes().len() as u64] {
            record.extend_from_slice(&value.to_le_bytes());
        }
        for change in delta.changes() {
            for value in [change.index() as u64, change.old(), change.new_value()] {
                record.extend_from_slice(&value.to_le_bytes());
            }
        }
        // An I/O error has an uncertain commit outcome. Refuse further reads/writes
        // until reopen resolves it from the sealed log, rather than append past damage.
        self.needs_recovery = true;
        write_frame(&mut self.file, &record, &mut observe)?;
        self.file.sync_all()?;
        observe(AppendStage::Synced);
        self.state.apply(source_id, version, event, delta)?;
        self.journal_events += 1;
        self.needs_recovery = false;
        observe(AppendStage::Applied);
        Ok(())
    }
}

fn resolved_path(path: &Path) -> Result<PathBuf, LiveError> {
    let absolute = std::path::absolute(path)?;
    let parent = fs::canonicalize(absolute.parent().ok_or(LiveError::InvalidRecord)?)?;
    Ok(parent.join(absolute.file_name().ok_or(LiveError::InvalidRecord)?))
}

fn writer_lock(path: &Path) -> Result<File, LiveError> {
    let mut name = path.as_os_str().to_os_string();
    name.push(".lock");
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(PathBuf::from(name))?;
    lock.try_lock().map_err(|e| LiveError::Io(e.into()))?;
    Ok(lock)
}

fn write_snapshot(file: &mut File, state: &LiveAverage) -> Result<(), LiveError> {
    let mut header = Vec::with_capacity(56);
    for value in [
        state.source_id(),
        state.version(),
        state.last_event(),
        state.semantic().hash().value(),
        state.values().len() as u64,
    ] {
        header.extend_from_slice(&value.to_le_bytes());
    }
    header.extend_from_slice(&state.query().numerator().to_le_bytes());
    let hash = state
        .values()
        .iter()
        .fold(checksum(&header), |hash, value| {
            checksum_extend(hash, &value.to_le_bytes())
        });
    let len = 56 + state.values().len() as u64 * 8;
    let mut writer = std::io::BufWriter::with_capacity(65536, &mut *file);
    writer.write_all(MAGIC)?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&(!len).to_le_bytes())?;
    writer.write_all(&hash.to_le_bytes())?;
    writer.write_all(&header)?;
    for value in state.values() {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    drop(writer);
    file.sync_all()?;
    Ok(())
}

fn write_frame(
    file: &mut File,
    payload: &[u8],
    observe: &mut impl FnMut(AppendStage),
) -> Result<(), LiveError> {
    let len = payload.len() as u64;
    file.write_all(&len.to_le_bytes())?;
    file.write_all(&(!len).to_le_bytes())?;
    file.write_all(&checksum(payload).to_le_bytes())?;
    observe(AppendStage::HeaderWritten);
    file.write_all(payload)?;
    observe(AppendStage::PayloadWritten);
    Ok(())
}

fn read_snapshot(file: &mut File, end: u64, limits: StoreLimits) -> Result<LiveAverage, LiveError> {
    let remaining = end
        .checked_sub(file.stream_position()?)
        .ok_or(LiveError::InvalidRecord)?;
    if remaining < HEADER_BYTES + 56 {
        return Err(LiveError::InvalidRecord);
    }
    let mut framing = [0_u8; 24];
    file.read_exact(&mut framing)?;
    let mut reader = framing.as_slice();
    let len = take_u64(&mut reader)?;
    let complement = take_u64(&mut reader)?;
    let expected_hash = take_u64(&mut reader)?;
    if complement != !len || len < 56 || len > remaining - HEADER_BYTES {
        return Err(LiveError::InvalidRecord);
    }
    let mut header = [0; 56];
    file.read_exact(&mut header)?;
    let mut reader = header.as_slice();
    let source_id = take_u64(&mut reader)?;
    let version = take_u64(&mut reader)?;
    let event = take_u64(&mut reader)?;
    let artifact_hash = take_u64(&mut reader)?;
    let count = take_u64(&mut reader)?;
    let total = take_u128(&mut reader)?;
    if count == 0
        || count.checked_mul(8).and_then(|n| n.checked_add(56)) != Some(len)
        || version != event
    {
        return Err(LiveError::InvalidRecord);
    }
    if count > limits.max_source_values {
        return Err(LiveError::ResourceLimit("source values"));
    }
    let semantic = SemanticArtifact::synthesize(FoldSpec::AverageExactU64)
        .map_err(|_| LiveError::InvalidArtifact)?;
    if artifact_hash != semantic.hash().value() {
        return Err(LiveError::InvalidArtifact);
    }
    let count = usize::try_from(count).map_err(|_| LiveError::ResourceLimit("address space"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| LiveError::ResourceLimit("source allocation"))?;
    let mut buffer = [0_u8; 65536];
    let mut hash = checksum(&header);
    let mut bytes_left = len - 56;
    while bytes_left != 0 {
        let bytes = bytes_left.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..bytes])?;
        hash = checksum_extend(hash, &buffer[..bytes]);
        values.extend(
            buffer[..bytes]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|v| u64::from_le_bytes(*v)),
        );
        bytes_left -= bytes as u64;
    }
    if hash != expected_hash {
        return Err(LiveError::InvalidRecord);
    }
    let mut state = LiveAverage::materialize(source_id, values, semantic)?;
    if state.query().numerator() != total {
        return Err(LiveError::InvalidRecord);
    }
    state.restore_version(version, event)?;
    Ok(state)
}

fn read_frame(file: &mut File, end: u64, max_len: u64) -> Result<Option<Vec<u8>>, LiveError> {
    let remaining = end - file.stream_position()?;
    if remaining < HEADER_BYTES {
        return Ok(None);
    }
    let mut header = [0; HEADER_BYTES as usize];
    file.read_exact(&mut header)?;
    let mut reader = header.as_slice();
    let len = take_u64(&mut reader)?;
    let complement = take_u64(&mut reader)?;
    let hash = take_u64(&mut reader)?;
    if complement != !len {
        return Err(LiveError::InvalidRecord);
    }
    if len > max_len {
        return Err(LiveError::ResourceLimit("event frame"));
    }
    if len > remaining - HEADER_BYTES {
        return Ok(None);
    }
    let len = usize::try_from(len).map_err(|_| LiveError::ResourceLimit("address space"))?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(len)
        .map_err(|_| LiveError::ResourceLimit("event allocation"))?;
    payload.resize(len, 0);
    file.read_exact(&mut payload)?;
    if checksum(&payload) != hash {
        return Err(LiveError::InvalidRecord);
    }
    Ok(Some(payload))
}

pub(crate) fn checksum(bytes: &[u8]) -> u64 {
    checksum_extend(0xcbf2_9ce4_8422_2325_u64, bytes)
}

fn checksum_extend(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

fn take_u64(input: &mut &[u8]) -> Result<u64, LiveError> {
    Ok(u64::from_le_bytes(take(input)?))
}

fn take_u128(input: &mut &[u8]) -> Result<u128, LiveError> {
    Ok(u128::from_le_bytes(take(input)?))
}

fn take<const N: usize>(input: &mut &[u8]) -> Result<[u8; N], LiveError> {
    let bytes = input.get(..N).ok_or(LiveError::InvalidRecord)?;
    let value = bytes.try_into().map_err(|_| LiveError::InvalidRecord)?;
    *input = &input[N..];
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_child() {
        let Ok(root) = std::env::var("AXON_APPEND_TEST_ROOT") else {
            return;
        };
        let target = std::env::var("AXON_APPEND_TEST_STAGE").unwrap();
        let mut store = LiveAverageStore::open(Path::new(&root).join("state.bin"))
            .unwrap()
            .0;
        let delta = ReplaceDelta::try_new(vec![Replace::new(0, 1, 99)]).unwrap();
        store
            .apply_observed(7, 0, 1, &delta, |stage| {
                if format!("{stage:?}") == target {
                    fs::write(Path::new(&root).join("ready"), b"ready").unwrap();
                    loop {
                        std::thread::park();
                    }
                }
            })
            .unwrap();
    }

    #[test]
    fn process_crashes_during_append_recover_whole_events_without_duplicate_application() {
        for stage in [
            AppendStage::HeaderWritten,
            AppendStage::PayloadWritten,
            AppendStage::Synced,
            AppendStage::Applied,
        ] {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root =
                std::env::temp_dir().join(format!("axon-append-{}-{nonce}", std::process::id()));
            fs::create_dir(&root).unwrap();
            let path = root.join("state.bin");
            let semantic = SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap();
            let initial = LiveAverage::materialize(7, vec![1, 2], semantic).unwrap();
            drop(LiveAverageStore::create(&path, initial).unwrap());
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "live_store::tests::append_child", "--nocapture"])
                .env("AXON_APPEND_TEST_ROOT", &root)
                .env("AXON_APPEND_TEST_STAGE", format!("{stage:?}"))
                .stdout(std::process::Stdio::null())
                .spawn()
                .unwrap();
            let start = std::time::Instant::now();
            while !root.join("ready").exists() && start.elapsed().as_secs() < 10 {
                if child.try_wait().unwrap().is_some() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            child.kill().unwrap();
            child.wait().unwrap();
            assert!(
                root.join("ready").exists(),
                "append stage not reached: {stage:?}"
            );
            let (mut recovered, report) = LiveAverageStore::open(&path).unwrap();
            let expected_version = u64::from(stage != AppendStage::HeaderWritten);
            assert_eq!(recovered.state().unwrap().version(), expected_version);
            assert_eq!(report.replayed_events, expected_version);
            assert_eq!(
                report.discarded_tail_bytes,
                if expected_version == 0 {
                    HEADER_BYTES
                } else {
                    0
                }
            );
            let delta = ReplaceDelta::try_new(vec![Replace::new(0, 1, 99)]).unwrap();
            if expected_version == 0 {
                recovered.apply(7, 0, 1, &delta).unwrap();
            } else {
                assert!(recovered.apply(7, 0, 1, &delta).is_err());
            }
            assert_eq!(recovered.state().unwrap().values(), &[99, 2]);
            drop(recovered);
            let reopened = LiveAverageStore::open(&path).unwrap().0;
            assert_eq!(reopened.state().unwrap().version(), 1);
            drop(reopened);
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn write_error_poisoning_blocks_queries_and_further_appends_until_reopen() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("axon-write-failure-{}-{nonce}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let path = root.join("state.bin");
        let semantic = SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap();
        let initial = LiveAverage::materialize(7, vec![1, 2], semantic).unwrap();
        let mut store = LiveAverageStore::create(&path, initial.clone()).unwrap();
        // Substitute a real read-only handle; the first OS write must fail.
        store.file = File::open(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let delta = ReplaceDelta::try_new(vec![Replace::new(0, 1, 99)]).unwrap();
        assert!(matches!(
            store.apply(7, 0, 1, &delta),
            Err(LiveError::Io(_))
        ));
        assert_eq!(store.state, initial);
        assert!(matches!(store.state(), Err(LiveError::NeedsRecovery)));
        assert!(matches!(
            store.apply(7, 0, 1, &delta),
            Err(LiveError::NeedsRecovery)
        ));
        assert!(matches!(store.compact(), Err(LiveError::NeedsRecovery)));
        assert_eq!(fs::read(&path).unwrap(), before);
        drop(store);
        let mut recovered = LiveAverageStore::open(&path).unwrap().0;
        assert_eq!(recovered.state().unwrap(), &initial);
        recovered.apply(7, 0, 1, &delta).unwrap();
        drop(recovered);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkpoint_child() {
        let Ok(root) = std::env::var("AXON_CHECKPOINT_TEST_ROOT") else {
            return;
        };
        let target = std::env::var("AXON_CHECKPOINT_TEST_STAGE").unwrap();
        let path = Path::new(&root).join("state.bin");
        let mut store = LiveAverageStore::open(&path).unwrap().0;
        store
            .compact_observed(|stage| {
                if format!("{stage:?}") == target {
                    fs::write(Path::new(&root).join("ready"), b"ready").unwrap();
                    loop {
                        std::thread::park();
                    }
                }
            })
            .unwrap();
    }

    #[test]
    fn process_crashes_at_checkpoint_boundaries_keep_the_committed_state() {
        for stage in [
            CheckpointStage::Created,
            CheckpointStage::Synced,
            CheckpointStage::Published,
        ] {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir()
                .join(format!("axon-checkpoint-{}-{nonce}", std::process::id()));
            fs::create_dir(&root).unwrap();
            let path = root.join("state.bin");
            let semantic = SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap();
            let live = LiveAverage::materialize(7, vec![1, 2], semantic).unwrap();
            let mut store = LiveAverageStore::create(&path, live).unwrap();
            for event in 1..=17 {
                let old = store.state().unwrap().values()[0];
                store
                    .apply(
                        7,
                        event - 1,
                        event,
                        &ReplaceDelta::try_new(vec![Replace::new(0, old, event + 10)]).unwrap(),
                    )
                    .unwrap();
            }
            let expected = store.state().unwrap().clone();
            drop(store);
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "live_store::tests::checkpoint_child",
                    "--nocapture",
                ])
                .env("AXON_CHECKPOINT_TEST_ROOT", &root)
                .env("AXON_CHECKPOINT_TEST_STAGE", format!("{stage:?}"))
                .stdout(std::process::Stdio::null())
                .spawn()
                .unwrap();
            let start = std::time::Instant::now();
            while !root.join("ready").exists() && start.elapsed().as_secs() < 10 {
                if child.try_wait().unwrap().is_some() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            child.kill().unwrap();
            child.wait().unwrap();
            assert!(
                root.join("ready").exists(),
                "checkpoint stage was not reached: {stage:?}"
            );
            let (mut recovered, report) = LiveAverageStore::open(&path).unwrap();
            assert_eq!(recovered.state().unwrap(), &expected);
            assert_eq!(
                report.replayed_events,
                if stage == CheckpointStage::Published {
                    0
                } else {
                    17
                }
            );
            recovered
                .apply(
                    7,
                    17,
                    18,
                    &ReplaceDelta::try_new(vec![Replace::new(1, 2, 99)]).unwrap(),
                )
                .unwrap();
            assert_eq!(recovered.state().unwrap().values(), &[27, 99]);
            recovered.compact().unwrap();
            drop(recovered);
            // The entire directory was created exclusively for this test.
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn failed_checkpoint_publication_requires_reopen_without_losing_the_old_file() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "axon-checkpoint-failure-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let path = root.join("state.bin");
        let original = root.join("original.bin");
        let semantic = SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap();
        let live = LiveAverage::materialize(7, vec![1, 2], semantic).unwrap();
        let mut store = LiveAverageStore::create(&path, live.clone()).unwrap();
        let result = store.compact_observed(|stage| {
            if stage == CheckpointStage::Synced {
                fs::rename(&path, &original).unwrap();
                fs::create_dir(&path).unwrap();
            }
        });
        assert!(result.is_err());
        assert!(matches!(store.state(), Err(LiveError::NeedsRecovery)));
        assert!(store.compact().is_err());
        drop(store);
        fs::remove_dir(&path).unwrap();
        fs::rename(&original, &path).unwrap();
        let recovered = LiveAverageStore::open(&path).unwrap().0;
        assert_eq!(recovered.state().unwrap(), &live);
        drop(recovered);
        fs::remove_dir_all(root).unwrap();
    }
}
