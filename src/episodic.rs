use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crate::live_store::checksum;

const MAGIC: &[u8; 8] = b"AXEP0001";
const VERSION: u32 = 1;
const HEADER_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EpisodeLimits {
    pub max_file_bytes: u64,
    pub max_episodes: usize,
    pub max_frame_bytes: usize,
    pub max_text_bytes: usize,
}

impl Default for EpisodeLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 64 * 1024 * 1024,
            max_episodes: 100_000,
            max_frame_bytes: 32 * 1024,
            max_text_bytes: 4_096,
        }
    }
}

#[derive(Debug)]
pub enum EpisodeError {
    Io(std::io::Error),
    Invalid(&'static str),
    Limit(&'static str),
    UnknownEpisode(u64),
    AlreadyReplaced(u64),
    SourceMismatch,
    EventCollision,
    NeedsRecovery,
}

impl fmt::Display for EpisodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "episode storage error: {error}"),
            Self::Invalid(reason) => write!(f, "invalid episode: {reason}"),
            Self::Limit(resource) => write!(f, "episode limit exceeded: {resource}"),
            Self::UnknownEpisode(id) => write!(f, "unknown episode {id}"),
            Self::AlreadyReplaced(id) => write!(f, "episode {id} was already revised or retracted"),
            Self::SourceMismatch => f.write_str("only the same source may revise its observation"),
            Self::EventCollision => {
                f.write_str("source event was already recorded with other data")
            }
            Self::NeedsRecovery => {
                f.write_str("episode store must be reopened after an I/O failure")
            }
        }
    }
}

impl std::error::Error for EpisodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for EpisodeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FactKey {
    pub subject: String,
    pub predicate: String,
    pub context: String,
}

impl FactKey {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            context: context.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EpisodeValue {
    Text(String),
    Integer(i64),
    Boolean(bool),
}

impl fmt::Display for EpisodeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(value) => value.fmt(f),
            Self::Integer(value) => value.fmt(f),
            Self::Boolean(value) => value.fmt(f),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpisodeStamp {
    pub source: String,
    pub event: String,
    pub effective_at: u64,
}

impl EpisodeStamp {
    pub fn new(source: impl Into<String>, event: impl Into<String>, effective_at: u64) -> Self {
        Self {
            source: source.into(),
            event: event.into(),
            effective_at,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EpisodeData {
    Fact {
        key: FactKey,
        value: EpisodeValue,
        replaces: Option<u64>,
    },
    Retraction {
        target: u64,
    },
    Outcome {
        context: String,
        action: String,
        success: bool,
        cost_ns: u64,
        detail: String,
    },
    Trace {
        context: String,
        channel: String,
        fields: BTreeMap<String, EpisodeValue>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Episode {
    id: u64,
    stamp: EpisodeStamp,
    data: EpisodeData,
}

impl Episode {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn stamp(&self) -> &EpisodeStamp {
        &self.stamp
    }
    pub fn data(&self) -> &EpisodeData {
        &self.data
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EpisodeReceipt {
    pub id: u64,
    pub duplicate: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EpisodeRecovery {
    pub replayed_episodes: usize,
    pub discarded_tail_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BeliefStatus {
    Consistent,
    Conflict,
    Missing,
}

impl BeliefStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Consistent => "consistent",
            Self::Conflict => "conflict",
            Self::Missing => "missing",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactEvidence {
    pub episode_id: u64,
    pub source: String,
    pub effective_at: u64,
    pub value: EpisodeValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BeliefResult {
    pub status: BeliefStatus,
    pub evidence: Vec<FactEvidence>,
    pub examined_episodes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ActionSummary {
    pub observations: u64,
    pub successes: u64,
    pub total_cost_ns: u128,
}

impl ActionSummary {
    pub fn success_fraction(self) -> (u128, u128) {
        (
            u128::from(self.successes) + 1,
            u128::from(self.observations) + 2,
        )
    }

    fn updated(self, success: bool, cost_ns: u64) -> Result<Self, EpisodeError> {
        let observations = self
            .observations
            .checked_add(1)
            .ok_or(EpisodeError::Limit("outcome count"))?;
        if observations > u64::MAX - 2 {
            return Err(EpisodeError::Limit("outcome count"));
        }
        Ok(Self {
            observations,
            successes: self
                .successes
                .checked_add(u64::from(success))
                .ok_or(EpisodeError::Limit("success count"))?,
            total_cost_ns: self
                .total_cost_ns
                .checked_add(u128::from(cost_ns))
                .ok_or(EpisodeError::Limit("cost sum"))?,
        })
    }
}

pub struct EpisodeStore {
    file: File,
    path: PathBuf,
    _writer_lock: File,
    limits: EpisodeLimits,
    episodes: Vec<Episode>,
    by_fact: BTreeMap<FactKey, Vec<usize>>,
    by_source_event: BTreeMap<(String, String), usize>,
    replaced: BTreeSet<u64>,
    outcomes: BTreeMap<(String, String), ActionSummary>,
    by_trace: BTreeMap<(String, String), Vec<usize>>,
    last_hash: u64,
    needs_recovery: bool,
}

impl EpisodeStore {
    pub fn default_path() -> PathBuf {
        PathBuf::from(".axon-memory/episodes.axep")
    }

    pub fn open(path: impl AsRef<Path>) -> Result<(Self, EpisodeRecovery), EpisodeError> {
        Self::open_with_limits(path, EpisodeLimits::default())
    }

    pub fn open_with_limits(
        path: impl AsRef<Path>,
        limits: EpisodeLimits,
    ) -> Result<(Self, EpisodeRecovery), EpisodeError> {
        if limits.max_file_bytes < HEADER_BYTES
            || limits.max_episodes == 0
            || limits.max_frame_bytes < 64
            || limits.max_frame_bytes > u32::MAX as usize
            || limits.max_text_bytes == 0
            || limits.max_text_bytes > limits.max_frame_bytes
        {
            return Err(EpisodeError::Invalid("resource limits"));
        }
        let absolute = std::path::absolute(path.as_ref())?;
        let parent = absolute
            .parent()
            .ok_or(EpisodeError::Invalid("store path"))?;
        fs::create_dir_all(parent)?;
        let path = if absolute.exists() {
            fs::canonicalize(&absolute)?
        } else {
            fs::canonicalize(parent)?.join(
                absolute
                    .file_name()
                    .ok_or(EpisodeError::Invalid("store name"))?,
            )
        };
        let mut lock_name = path.as_os_str().to_os_string();
        lock_name.push(".lock");
        let writer_lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(PathBuf::from(lock_name))?;
        writer_lock
            .try_lock()
            .map_err(|error| EpisodeError::Io(error.into()))?;
        let new_file = !path.exists();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(new_file)
            .open(&path)?;
        file.try_lock()
            .map_err(|error| EpisodeError::Io(error.into()))?;
        if new_file {
            file.write_all(MAGIC)?;
            file.write_all(&VERSION.to_le_bytes())?;
            file.write_all(&(!VERSION).to_le_bytes())?;
            file.sync_all()?;
        }
        let length = file.metadata()?.len();
        if length > limits.max_file_bytes {
            return Err(EpisodeError::Limit("file bytes"));
        }
        let mut header = [0_u8; HEADER_BYTES as usize];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header)?;
        if &header[..8] != MAGIC
            || header[8..12] != VERSION.to_le_bytes()
            || header[12..16] != (!VERSION).to_le_bytes()
        {
            return Err(EpisodeError::Invalid("header"));
        }
        let mut store = Self {
            file,
            path,
            _writer_lock: writer_lock,
            limits,
            episodes: Vec::new(),
            by_fact: BTreeMap::new(),
            by_source_event: BTreeMap::new(),
            replaced: BTreeSet::new(),
            outcomes: BTreeMap::new(),
            by_trace: BTreeMap::new(),
            last_hash: 0,
            needs_recovery: false,
        };
        let mut recovery = EpisodeRecovery::default();
        loop {
            let start = store.file.stream_position()?;
            if start == length {
                break;
            }
            let remaining = length - start;
            if remaining < 8 {
                recovery.discarded_tail_bytes = remaining;
                break;
            }
            let mut prefix = [0_u8; 16];
            store.file.read_exact(&mut prefix[..8])?;
            let payload_len = u32::from_le_bytes(prefix[..4].try_into().expect("fixed width"));
            let complement = u32::from_le_bytes(prefix[4..8].try_into().expect("fixed width"));
            if payload_len != !complement {
                return Err(EpisodeError::Invalid("frame length complement"));
            }
            if payload_len as usize > limits.max_frame_bytes {
                return Err(EpisodeError::Limit("frame bytes"));
            }
            if remaining < 16 {
                recovery.discarded_tail_bytes = remaining;
                break;
            }
            store.file.read_exact(&mut prefix[8..])?;
            if u64::from_le_bytes(prefix[8..].try_into().expect("fixed width")) != store.last_hash {
                return Err(EpisodeError::Invalid("journal chain"));
            }
            if remaining < u64::from(payload_len) + 24 {
                recovery.discarded_tail_bytes = remaining;
                break;
            }
            let mut sealed = Vec::new();
            sealed
                .try_reserve_exact(payload_len as usize + 16)
                .map_err(|_| EpisodeError::Limit("frame allocation"))?;
            sealed.extend_from_slice(&prefix);
            sealed.resize(payload_len as usize + 16, 0);
            store.file.read_exact(&mut sealed[16..])?;
            let mut hash = [0_u8; 8];
            store.file.read_exact(&mut hash)?;
            let hash = u64::from_le_bytes(hash);
            if checksum(&sealed) != hash {
                return Err(EpisodeError::Invalid("frame checksum"));
            }
            let episode = decode_episode(&sealed[16..], limits)?;
            let outcome = store.validate_next(&episode)?;
            store.apply(episode, outcome);
            store.last_hash = hash;
            recovery.replayed_episodes += 1;
        }
        if recovery.discarded_tail_bytes > 0 {
            let valid_length = length - recovery.discarded_tail_bytes;
            store.file.set_len(valid_length)?;
            store.file.sync_all()?;
            store.file.seek(SeekFrom::Start(valid_length))?;
        }
        Ok((store, recovery))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn len(&self) -> usize {
        self.episodes.len()
    }
    pub fn episodes(&self) -> Result<&[Episode], EpisodeError> {
        self.healthy()?;
        Ok(&self.episodes)
    }
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
    pub fn file_bytes(&self) -> Result<u64, EpisodeError> {
        Ok(self.file.metadata()?.len())
    }

    /// Copy through the owning handle while keeping the writer lock. Reopening the
    /// source conflicts with Windows byte-range locks; releasing it permits a torn snapshot.
    pub fn snapshot_new(&mut self, path: &Path) -> Result<(), EpisodeError> {
        self.healthy()?;
        let length = self.file.metadata()?.len();
        if length > self.limits.max_file_bytes {
            return Err(EpisodeError::Limit("snapshot bytes"));
        }
        let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
        self.file.seek(SeekFrom::Start(0))?;
        let copied = std::io::copy(&mut Read::by_ref(&mut self.file).take(length), &mut output)?;
        if copied != length {
            return Err(EpisodeError::Invalid("incomplete snapshot"));
        }
        output.sync_all()?;
        Ok(())
    }
    pub fn next_id(&self) -> u64 {
        self.episodes.len() as u64 + 1
    }

    pub fn observe(
        &mut self,
        stamp: EpisodeStamp,
        key: FactKey,
        value: EpisodeValue,
    ) -> Result<EpisodeReceipt, EpisodeError> {
        self.record(
            stamp,
            EpisodeData::Fact {
                key,
                value,
                replaces: None,
            },
        )
    }

    pub fn revise(
        &mut self,
        target: u64,
        source: impl Into<String>,
        event: impl Into<String>,
        value: EpisodeValue,
    ) -> Result<EpisodeReceipt, EpisodeError> {
        let previous = self.episode(target)?;
        let EpisodeData::Fact { key, .. } = &previous.data else {
            return Err(EpisodeError::Invalid("revision target is not a fact"));
        };
        let key = key.clone();
        let at = previous.stamp.effective_at;
        self.record(
            EpisodeStamp::new(source, event, at),
            EpisodeData::Fact {
                key,
                value,
                replaces: Some(target),
            },
        )
    }

    pub fn retract(
        &mut self,
        target: u64,
        source: impl Into<String>,
        event: impl Into<String>,
    ) -> Result<EpisodeReceipt, EpisodeError> {
        let at = self.episode(target)?.stamp.effective_at;
        self.record(
            EpisodeStamp::new(source, event, at),
            EpisodeData::Retraction { target },
        )
    }

    pub fn record(
        &mut self,
        stamp: EpisodeStamp,
        data: EpisodeData,
    ) -> Result<EpisodeReceipt, EpisodeError> {
        self.record_observed(stamp, data, |_| {})
    }

    fn record_observed(
        &mut self,
        stamp: EpisodeStamp,
        data: EpisodeData,
        mut observe: impl FnMut(AppendStage),
    ) -> Result<EpisodeReceipt, EpisodeError> {
        self.healthy()?;
        let episode = Episode {
            id: self.next_id(),
            stamp,
            data,
        };
        if let Some(&index) = self
            .by_source_event
            .get(&(episode.stamp.source.clone(), episode.stamp.event.clone()))
        {
            let previous = &self.episodes[index];
            if episode.stamp == previous.stamp && episode.data == previous.data {
                return Ok(EpisodeReceipt {
                    id: previous.id,
                    duplicate: true,
                });
            }
            return Err(EpisodeError::EventCollision);
        }
        let outcome = self.validate_next(&episode)?;
        self.check_size(&episode)?;
        let payload = encode_episode(&episode);
        let length = payload.len() as u32;
        let mut frame = Vec::with_capacity(payload.len() + 24);
        frame.extend_from_slice(&length.to_le_bytes());
        frame.extend_from_slice(&(!length).to_le_bytes());
        frame.extend_from_slice(&self.last_hash.to_le_bytes());
        frame.extend_from_slice(&payload);
        let hash = checksum(&frame);
        frame.extend_from_slice(&hash.to_le_bytes());
        if self
            .file
            .metadata()?
            .len()
            .checked_add(frame.len() as u64)
            .is_none_or(|length| length > self.limits.max_file_bytes)
        {
            return Err(EpisodeError::Limit("file bytes"));
        }
        let write_result = (|| -> std::io::Result<()> {
            self.file.seek(SeekFrom::End(0))?;
            self.file.write_all(&frame[..16])?;
            observe(AppendStage::HeaderWritten);
            self.file.write_all(&frame[16..])?;
            observe(AppendStage::PayloadWritten);
            self.file.sync_all()?;
            observe(AppendStage::Synced);
            Ok(())
        })();
        if let Err(error) = write_result {
            self.needs_recovery = true;
            return Err(EpisodeError::Io(error));
        }
        let id = episode.id;
        self.apply(episode, outcome);
        self.last_hash = hash;
        observe(AppendStage::Applied);
        Ok(EpisodeReceipt {
            id,
            duplicate: false,
        })
    }

    fn healthy(&self) -> Result<(), EpisodeError> {
        if self.needs_recovery {
            Err(EpisodeError::NeedsRecovery)
        } else {
            Ok(())
        }
    }

    pub fn check_record(
        &self,
        stamp: &EpisodeStamp,
        data: &EpisodeData,
    ) -> Result<(), EpisodeError> {
        self.healthy()?;
        let episode = Episode {
            id: self.next_id(),
            stamp: stamp.clone(),
            data: data.clone(),
        };
        self.validate_next(&episode)?;
        self.check_size(&episode)
    }

    fn check_size(&self, episode: &Episode) -> Result<(), EpisodeError> {
        let length = encoded_length(episode)?;
        if length > self.limits.max_frame_bytes {
            return Err(EpisodeError::Limit("frame bytes"));
        }
        if self
            .file
            .metadata()?
            .len()
            .checked_add(length as u64 + 24)
            .is_none_or(|length| length > self.limits.max_file_bytes)
        {
            return Err(EpisodeError::Limit("file bytes"));
        }
        Ok(())
    }

    pub fn episode(&self, id: u64) -> Result<&Episode, EpisodeError> {
        self.healthy()?;
        id.checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| self.episodes.get(index))
            .ok_or(EpisodeError::UnknownEpisode(id))
    }

    pub fn history(&self, key: &FactKey) -> Result<Vec<&Episode>, EpisodeError> {
        self.healthy()?;
        validate_key(key, self.limits)?;
        Ok(self
            .by_fact
            .get(key)
            .into_iter()
            .flatten()
            .map(|index| &self.episodes[*index])
            .collect())
    }

    pub fn belief(
        &self,
        key: &FactKey,
        effective_at: Option<u64>,
        known_at: Option<u64>,
    ) -> Result<BeliefResult, EpisodeError> {
        self.healthy()?;
        validate_key(key, self.limits)?;
        let known_at = known_at.unwrap_or(self.len() as u64);
        if known_at > self.len() as u64 {
            return Err(EpisodeError::UnknownEpisode(known_at));
        }
        let at = effective_at.unwrap_or(u64::MAX);
        let indices = self.by_fact.get(key).map_or(&[][..], Vec::as_slice);
        let visible = indices
            .iter()
            .map(|index| &self.episodes[*index])
            .filter(|episode| episode.id <= known_at && episode.stamp.effective_at <= at)
            .collect::<Vec<_>>();
        let mut inactive = BTreeSet::new();
        for episode in &visible {
            match episode.data {
                EpisodeData::Fact {
                    replaces: Some(target),
                    ..
                }
                | EpisodeData::Retraction { target } => {
                    inactive.insert(target);
                }
                _ => {}
            }
        }
        let mut latest = BTreeMap::<&str, (u64, Vec<&Episode>)>::new();
        for episode in visible {
            if inactive.contains(&episode.id) {
                continue;
            }
            let entry = latest
                .entry(&episode.stamp.source)
                .or_insert((episode.stamp.effective_at, Vec::new()));
            if episode.stamp.effective_at > entry.0 {
                entry.0 = episode.stamp.effective_at;
                entry.1.clear();
            }
            if episode.stamp.effective_at == entry.0 {
                entry.1.push(episode);
            }
        }
        let evidence = latest
            .into_values()
            .flat_map(|(_, episodes)| episodes)
            .filter_map(|episode| {
                if let EpisodeData::Fact { value, .. } = &episode.data {
                    Some(FactEvidence {
                        episode_id: episode.id,
                        source: episode.stamp.source.clone(),
                        effective_at: episode.stamp.effective_at,
                        value: value.clone(),
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let values = evidence
            .iter()
            .map(|evidence| &evidence.value)
            .collect::<BTreeSet<_>>();
        let status = match values.len() {
            0 => BeliefStatus::Missing,
            1 => BeliefStatus::Consistent,
            _ => BeliefStatus::Conflict,
        };
        Ok(BeliefResult {
            status,
            evidence,
            examined_episodes: indices.len(),
        })
    }

    pub fn experience(&self, context: &str) -> Result<Vec<(&str, ActionSummary)>, EpisodeError> {
        self.healthy()?;
        validate_text(context, self.limits.max_text_bytes)?;
        let lower = (context.to_owned(), String::new());
        Ok(self
            .outcomes
            .range(lower..)
            .take_while(|((candidate, _), _)| candidate == context)
            .map(|((_, action), summary)| (action.as_str(), *summary))
            .collect())
    }

    pub fn traces(&self, context: &str, channel: &str) -> Result<Vec<&Episode>, EpisodeError> {
        self.healthy()?;
        validate_text(context, self.limits.max_text_bytes)?;
        validate_text(channel, 256.min(self.limits.max_text_bytes))?;
        Ok(self
            .by_trace
            .get(&(context.to_owned(), channel.to_owned()))
            .into_iter()
            .flatten()
            .map(|&index| &self.episodes[index])
            .collect())
    }

    fn validate_next(&self, episode: &Episode) -> Result<Option<ActionSummary>, EpisodeError> {
        if self.episodes.len() >= self.limits.max_episodes {
            return Err(EpisodeError::Limit("episode count"));
        }
        if episode.id != self.next_id() {
            return Err(EpisodeError::Invalid("episode sequence"));
        }
        validate_text(&episode.stamp.source, 256.min(self.limits.max_text_bytes))?;
        validate_text(&episode.stamp.event, 256.min(self.limits.max_text_bytes))?;
        if self
            .by_source_event
            .contains_key(&(episode.stamp.source.clone(), episode.stamp.event.clone()))
        {
            return Err(EpisodeError::EventCollision);
        }
        match &episode.data {
            EpisodeData::Trace {
                context,
                channel,
                fields,
            } => {
                validate_text(context, self.limits.max_text_bytes)?;
                validate_text(channel, 256.min(self.limits.max_text_bytes))?;
                if fields.is_empty() || fields.len() > 16 {
                    return Err(EpisodeError::Limit("trace fields"));
                }
                for (name, value) in fields {
                    validate_text(name, 256.min(self.limits.max_text_bytes))?;
                    if let EpisodeValue::Text(value) = value {
                        validate_text(value, self.limits.max_text_bytes)?;
                    }
                }
                Ok(None)
            }
            EpisodeData::Fact {
                key,
                value,
                replaces,
            } => {
                validate_key(key, self.limits)?;
                if let EpisodeValue::Text(value) = value {
                    validate_text(value, self.limits.max_text_bytes)?;
                }
                if let Some(target) = replaces {
                    let previous = self.validate_target(*target, episode)?;
                    if !matches!(&previous.data, EpisodeData::Fact { key: previous_key, .. } if previous_key == key)
                    {
                        return Err(EpisodeError::Invalid("revision key changed"));
                    }
                }
                Ok(None)
            }
            EpisodeData::Retraction { target } => {
                self.validate_target(*target, episode)?;
                Ok(None)
            }
            EpisodeData::Outcome {
                context,
                action,
                success,
                cost_ns,
                detail,
            } => {
                validate_text(context, self.limits.max_text_bytes)?;
                validate_text(action, self.limits.max_text_bytes)?;
                validate_text(detail, self.limits.max_text_bytes)?;
                self.outcomes
                    .get(&(context.clone(), action.clone()))
                    .copied()
                    .unwrap_or_default()
                    .updated(*success, *cost_ns)
                    .map(Some)
            }
        }
    }

    fn validate_target(&self, target: u64, episode: &Episode) -> Result<&Episode, EpisodeError> {
        let previous = self.episode(target)?;
        if !matches!(previous.data, EpisodeData::Fact { .. }) {
            return Err(EpisodeError::Invalid("target is not a fact"));
        }
        if self.replaced.contains(&target) {
            return Err(EpisodeError::AlreadyReplaced(target));
        }
        if previous.stamp.source != episode.stamp.source {
            return Err(EpisodeError::SourceMismatch);
        }
        if previous.stamp.effective_at != episode.stamp.effective_at {
            return Err(EpisodeError::Invalid("revision effective time changed"));
        }
        Ok(previous)
    }

    fn apply(&mut self, episode: Episode, outcome: Option<ActionSummary>) {
        let index = self.episodes.len();
        match &episode.data {
            EpisodeData::Trace {
                context, channel, ..
            } => {
                self.by_trace
                    .entry((context.clone(), channel.clone()))
                    .or_default()
                    .push(index);
            }
            EpisodeData::Fact { key, replaces, .. } => {
                self.by_fact.entry(key.clone()).or_default().push(index);
                if let Some(target) = replaces {
                    self.replaced.insert(*target);
                }
            }
            EpisodeData::Retraction { target } => {
                let EpisodeData::Fact { key, .. } = &self.episodes[*target as usize - 1].data
                else {
                    unreachable!("validated fact target")
                };
                self.by_fact.entry(key.clone()).or_default().push(index);
                self.replaced.insert(*target);
            }
            EpisodeData::Outcome {
                context, action, ..
            } => {
                self.outcomes.insert(
                    (context.clone(), action.clone()),
                    outcome.expect("validated outcome"),
                );
            }
        }
        self.by_source_event.insert(
            (episode.stamp.source.clone(), episode.stamp.event.clone()),
            index,
        );
        self.episodes.push(episode);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AppendStage {
    HeaderWritten,
    PayloadWritten,
    Synced,
    Applied,
}

fn validate_text(value: &str, maximum: usize) -> Result<(), EpisodeError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(EpisodeError::Invalid(
            "empty, padded or control-containing text",
        ));
    }
    if value.len() > maximum {
        return Err(EpisodeError::Limit("text bytes"));
    }
    Ok(())
}

fn validate_key(key: &FactKey, limits: EpisodeLimits) -> Result<(), EpisodeError> {
    for value in [&key.subject, &key.predicate, &key.context] {
        validate_text(value, 256.min(limits.max_text_bytes))?;
    }
    Ok(())
}

fn encode_episode(episode: &Episode) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&episode.id.to_le_bytes());
    bytes.extend_from_slice(&episode.stamp.effective_at.to_le_bytes());
    put_text(&mut bytes, &episode.stamp.source);
    put_text(&mut bytes, &episode.stamp.event);
    match &episode.data {
        EpisodeData::Trace {
            context,
            channel,
            fields,
        } => {
            bytes.push(3);
            put_text(&mut bytes, context);
            put_text(&mut bytes, channel);
            bytes.push(fields.len() as u8);
            for (name, value) in fields {
                put_text(&mut bytes, name);
                put_value(&mut bytes, value);
            }
        }
        EpisodeData::Fact {
            key,
            value,
            replaces,
        } => {
            bytes.push(0);
            for value in [&key.subject, &key.predicate, &key.context] {
                put_text(&mut bytes, value);
            }
            bytes.extend_from_slice(&replaces.unwrap_or(0).to_le_bytes());
            match value {
                EpisodeValue::Text(value) => {
                    bytes.push(0);
                    put_text(&mut bytes, value);
                }
                EpisodeValue::Integer(value) => {
                    bytes.push(1);
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                EpisodeValue::Boolean(value) => {
                    bytes.push(2);
                    bytes.push(u8::from(*value));
                }
            }
        }
        EpisodeData::Retraction { target } => {
            bytes.push(1);
            bytes.extend_from_slice(&target.to_le_bytes());
        }
        EpisodeData::Outcome {
            context,
            action,
            success,
            cost_ns,
            detail,
        } => {
            bytes.push(2);
            put_text(&mut bytes, context);
            put_text(&mut bytes, action);
            bytes.push(u8::from(*success));
            bytes.extend_from_slice(&cost_ns.to_le_bytes());
            put_text(&mut bytes, detail);
        }
    }
    bytes
}

fn encoded_length(episode: &Episode) -> Result<usize, EpisodeError> {
    let mut length = 17_usize;
    let mut text = |value: &str| -> Result<(), EpisodeError> {
        length = length
            .checked_add(4)
            .and_then(|n| n.checked_add(value.len()))
            .ok_or(EpisodeError::Limit("frame bytes"))?;
        Ok(())
    };
    text(&episode.stamp.source)?;
    text(&episode.stamp.event)?;
    match &episode.data {
        EpisodeData::Trace {
            context,
            channel,
            fields,
        } => {
            text(context)?;
            text(channel)?;
            let mut extra = 1;
            for (name, value) in fields {
                text(name)?;
                extra += 1;
                match value {
                    EpisodeValue::Text(value) => text(value)?,
                    EpisodeValue::Integer(_) => extra += 8,
                    EpisodeValue::Boolean(_) => extra += 1,
                }
            }
            length = length
                .checked_add(extra)
                .ok_or(EpisodeError::Limit("frame bytes"))?;
        }
        EpisodeData::Fact { key, value, .. } => {
            for value in [&key.subject, &key.predicate, &key.context] {
                text(value)?;
            }
            if let EpisodeValue::Text(value) = value {
                text(value)?;
            }
            // A fact adds its replacement id and value tag; numeric values add eight bytes.
            length = length
                .checked_add(
                    9 + match value {
                        EpisodeValue::Text(_) => 0,
                        EpisodeValue::Integer(_) => 8,
                        EpisodeValue::Boolean(_) => 1,
                    },
                )
                .ok_or(EpisodeError::Limit("frame bytes"))?;
        }
        EpisodeData::Retraction { .. } => {
            length += 8;
        }
        EpisodeData::Outcome {
            context,
            action,
            detail,
            ..
        } => {
            for value in [context, action, detail] {
                text(value)?;
            }
            length = length
                .checked_add(9)
                .ok_or(EpisodeError::Limit("frame bytes"))?;
        }
    }
    Ok(length)
}

fn put_text(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn put_value(bytes: &mut Vec<u8>, value: &EpisodeValue) {
    match value {
        EpisodeValue::Text(value) => {
            bytes.push(0);
            put_text(bytes, value);
        }
        EpisodeValue::Integer(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        EpisodeValue::Boolean(value) => {
            bytes.push(2);
            bytes.push(u8::from(*value));
        }
    }
}

fn take_value(bytes: &mut &[u8], limits: EpisodeLimits) -> Result<EpisodeValue, EpisodeError> {
    match take(bytes, 1)?[0] {
        0 => Ok(EpisodeValue::Text(take_text(bytes, limits)?)),
        1 => Ok(EpisodeValue::Integer(i64::from_le_bytes(
            take(bytes, 8)?.try_into().expect("fixed width"),
        ))),
        2 => Ok(EpisodeValue::Boolean(take_bool(bytes)?)),
        _ => Err(EpisodeError::Invalid("value kind")),
    }
}

fn decode_episode(mut bytes: &[u8], limits: EpisodeLimits) -> Result<Episode, EpisodeError> {
    let id = u64::from_le_bytes(take(&mut bytes, 8)?.try_into().expect("fixed width"));
    let effective_at = u64::from_le_bytes(take(&mut bytes, 8)?.try_into().expect("fixed width"));
    let source = take_text(&mut bytes, limits)?;
    let event = take_text(&mut bytes, limits)?;
    let data = match take(&mut bytes, 1)?[0] {
        0 => {
            let key = FactKey::new(
                take_text(&mut bytes, limits)?,
                take_text(&mut bytes, limits)?,
                take_text(&mut bytes, limits)?,
            );
            let replaces =
                u64::from_le_bytes(take(&mut bytes, 8)?.try_into().expect("fixed width"));
            let value = match take(&mut bytes, 1)?[0] {
                0 => EpisodeValue::Text(take_text(&mut bytes, limits)?),
                1 => EpisodeValue::Integer(i64::from_le_bytes(
                    take(&mut bytes, 8)?.try_into().expect("fixed width"),
                )),
                2 => EpisodeValue::Boolean(take_bool(&mut bytes)?),
                _ => return Err(EpisodeError::Invalid("value kind")),
            };
            EpisodeData::Fact {
                key,
                value,
                replaces: (replaces != 0).then_some(replaces),
            }
        }
        1 => EpisodeData::Retraction {
            target: u64::from_le_bytes(take(&mut bytes, 8)?.try_into().expect("fixed width")),
        },
        2 => EpisodeData::Outcome {
            context: take_text(&mut bytes, limits)?,
            action: take_text(&mut bytes, limits)?,
            success: take_bool(&mut bytes)?,
            cost_ns: u64::from_le_bytes(take(&mut bytes, 8)?.try_into().expect("fixed width")),
            detail: take_text(&mut bytes, limits)?,
        },
        3 => {
            let context = take_text(&mut bytes, limits)?;
            let channel = take_text(&mut bytes, limits)?;
            let count = take(&mut bytes, 1)?[0];
            if count == 0 || count > 16 {
                return Err(EpisodeError::Limit("trace fields"));
            }
            let mut fields = BTreeMap::new();
            for _ in 0..count {
                let name = take_text(&mut bytes, limits)?;
                let value = take_value(&mut bytes, limits)?;
                if fields.insert(name, value).is_some() {
                    return Err(EpisodeError::Invalid("duplicate trace field"));
                }
            }
            EpisodeData::Trace {
                context,
                channel,
                fields,
            }
        }
        _ => return Err(EpisodeError::Invalid("episode kind")),
    };
    if !bytes.is_empty() {
        return Err(EpisodeError::Invalid("trailing payload bytes"));
    }
    Ok(Episode {
        id,
        stamp: EpisodeStamp::new(source, event, effective_at),
        data,
    })
}

fn take<'a>(bytes: &mut &'a [u8], length: usize) -> Result<&'a [u8], EpisodeError> {
    if length > bytes.len() {
        return Err(EpisodeError::Invalid("truncated payload"));
    }
    let (value, rest) = bytes.split_at(length);
    *bytes = rest;
    Ok(value)
}

fn take_text(bytes: &mut &[u8], limits: EpisodeLimits) -> Result<String, EpisodeError> {
    let length = u32::from_le_bytes(take(bytes, 4)?.try_into().expect("fixed width")) as usize;
    if length > limits.max_text_bytes {
        return Err(EpisodeError::Limit("text bytes"));
    }
    String::from_utf8(take(bytes, length)?.to_vec()).map_err(|_| EpisodeError::Invalid("UTF-8"))
}

fn take_bool(bytes: &mut &[u8]) -> Result<bool, EpisodeError> {
    match take(bytes, 1)?[0] {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(EpisodeError::Invalid("boolean value")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        process::{Command, Stdio},
        sync::atomic::{AtomicU64, Ordering},
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    struct Temp(PathBuf);
    impl Temp {
        fn candidate_path() -> PathBuf {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            std::env::temp_dir().join(format!(
                "axon-episodes-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ))
        }
        fn new() -> Self {
            let root = Self::candidate_path();
            fs::create_dir(&root).unwrap();
            Self(root)
        }
        fn path(&self) -> PathBuf {
            self.0.join("episodes.axep")
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    #[test]
    fn concurrent_fixture_names_do_not_depend_on_clock_resolution() {
        let workers = (0..8)
            .map(|_| {
                std::thread::spawn(|| (0..256).map(|_| Temp::candidate_path()).collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        let paths = workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(paths.len(), 2048);
    }
    fn key() -> FactKey {
        FactKey::new("pump", "state", "lab")
    }
    fn stamp(event: &str, at: u64) -> EpisodeStamp {
        EpisodeStamp::new("sensor", event, at)
    }
    fn value(n: i64) -> EpisodeValue {
        EpisodeValue::Integer(n)
    }

    #[test]
    fn temporal_observations_separate_effective_time_from_commit_order() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        store.observe(stamp("new", 20), key(), value(2)).unwrap();
        store.observe(stamp("late", 10), key(), value(1)).unwrap();
        assert_eq!(
            store.belief(&key(), None, None).unwrap().evidence[0].value,
            value(2)
        );
        assert_eq!(
            store.belief(&key(), Some(15), Some(1)).unwrap().status,
            BeliefStatus::Missing
        );
        assert_eq!(
            store.belief(&key(), Some(15), Some(2)).unwrap().evidence[0].value,
            value(1)
        );
        assert_eq!(
            store.belief(&key(), None, Some(0)).unwrap().status,
            BeliefStatus::Missing
        );
        assert!(store.belief(&key(), None, Some(3)).is_err());
    }

    #[test]
    fn revisions_keep_historical_knowledge_and_retractions_do_not_resurrect_old_facts() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        store.observe(stamp("old", 10), key(), value(1)).unwrap();
        store.observe(stamp("new", 20), key(), value(2)).unwrap();
        let correction = store.revise(2, "sensor", "fix", value(3)).unwrap();
        assert_eq!(
            store.belief(&key(), None, Some(2)).unwrap().evidence[0].value,
            value(2)
        );
        assert_eq!(
            store.belief(&key(), None, None).unwrap().evidence[0].value,
            value(3)
        );
        store.retract(correction.id, "sensor", "withdraw").unwrap();
        assert_eq!(
            store.belief(&key(), None, None).unwrap().status,
            BeliefStatus::Missing
        );
        assert_eq!(
            store.belief(&key(), Some(15), None).unwrap().evidence[0].value,
            value(1)
        );
        assert_eq!(store.history(&key()).unwrap().len(), 4);
        drop(store);
        let (store, recovery) = EpisodeStore::open(temp.path()).unwrap();
        assert_eq!(recovery.replayed_episodes, 4);
        assert_eq!(
            store.belief(&key(), None, Some(3)).unwrap().evidence[0].value,
            value(3)
        );
    }

    #[test]
    fn conflicting_sources_and_equal_time_claims_are_not_silently_overwritten() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        for (source, event, n) in [("a", "1", 1), ("b", "1", 1), ("c", "1", 2), ("a", "2", 2)] {
            store
                .observe(EpisodeStamp::new(source, event, 10), key(), value(n))
                .unwrap();
        }
        let result = store.belief(&key(), None, None).unwrap();
        assert_eq!(result.status, BeliefStatus::Conflict);
        assert_eq!(result.evidence.len(), 4);
        store.retract(4, "a", "3").unwrap();
        assert_eq!(
            store.belief(&key(), None, None).unwrap().status,
            BeliefStatus::Conflict
        );
        store.revise(3, "c", "2", value(1)).unwrap();
        assert_eq!(
            store.belief(&key(), None, None).unwrap().status,
            BeliefStatus::Consistent
        );
    }

    #[test]
    fn event_deduplication_and_revision_authority_are_durable() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        store.observe(stamp("1", 10), key(), value(1)).unwrap();
        let bytes = store.file_bytes().unwrap();
        assert!(
            store
                .observe(stamp("1", 10), key(), value(1))
                .unwrap()
                .duplicate
        );
        assert!(matches!(
            store.observe(stamp("1", 10), key(), value(2)),
            Err(EpisodeError::EventCollision)
        ));
        assert!(matches!(
            store.revise(1, "other", "fix", value(2)),
            Err(EpisodeError::SourceMismatch)
        ));
        assert_eq!(store.file_bytes().unwrap(), bytes);
        store.revise(1, "sensor", "fix", value(2)).unwrap();
        assert!(
            store
                .revise(1, "sensor", "fix", value(2))
                .unwrap()
                .duplicate
        );
        assert!(matches!(
            store.retract(1, "sensor", "withdraw"),
            Err(EpisodeError::AlreadyReplaced(1))
        ));
        drop(store);
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        assert!(
            store
                .revise(1, "sensor", "fix", value(2))
                .unwrap()
                .duplicate
        );
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn outcome_learning_is_context_specific_exact_and_idempotent() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        for (id, context, action, success, cost) in [
            ("1", "lab", "full", true, 10),
            ("2", "lab", "full", false, 20),
            ("3", "field", "full", true, 99),
            ("4", "lab", "delta", true, 4),
        ] {
            let data = EpisodeData::Outcome {
                context: context.into(),
                action: action.into(),
                success,
                cost_ns: cost,
                detail: "measured".into(),
            };
            store.record(stamp(id, 1), data.clone()).unwrap();
            assert!(store.record(stamp(id, 1), data).unwrap().duplicate);
        }
        let summaries = store.experience("lab").unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(
            summaries[1].1,
            ActionSummary {
                observations: 2,
                successes: 1,
                total_cost_ns: 30
            }
        );
        assert_eq!(summaries[1].1.success_fraction(), (2, 4));
        drop(store);
        let store = EpisodeStore::open(temp.path()).unwrap().0;
        assert_eq!(store.experience("field").unwrap()[0].1.total_cost_ns, 99);
        assert_eq!(
            ActionSummary {
                observations: u64::MAX,
                successes: u64::MAX,
                total_cost_ns: 0
            }
            .success_fraction(),
            (u128::from(u64::MAX) + 1, u128::from(u64::MAX) + 2)
        );
    }

    #[test]
    fn input_and_resource_limits_reject_before_mutation() {
        let temp = Temp::new();
        let limits = EpisodeLimits {
            max_episodes: 1,
            max_text_bytes: 8,
            ..EpisodeLimits::default()
        };
        let mut store = EpisodeStore::open_with_limits(temp.path(), limits)
            .unwrap()
            .0;
        for invalid in ["", " padded", "line\nbreak", "123456789"] {
            assert!(
                store
                    .observe(stamp("1", 1), key(), EpisodeValue::Text(invalid.into()))
                    .is_err()
            );
            assert_eq!(store.file_bytes().unwrap(), HEADER_BYTES);
        }
        store.observe(stamp("1", 1), key(), value(1)).unwrap();
        let bytes = store.file_bytes().unwrap();
        assert!(
            store
                .observe(stamp("1", 1), key(), value(1))
                .unwrap()
                .duplicate
        );
        assert!(matches!(
            store.observe(stamp("2", 1), key(), value(1)),
            Err(EpisodeError::Limit("episode count"))
        ));
        assert_eq!(store.file_bytes().unwrap(), bytes);
        let other = temp.0.join("small.axep");
        let mut small = EpisodeStore::open_with_limits(
            &other,
            EpisodeLimits {
                max_file_bytes: HEADER_BYTES,
                ..EpisodeLimits::default()
            },
        )
        .unwrap()
        .0;
        assert!(matches!(
            small.observe(stamp("1", 1), key(), value(1)),
            Err(EpisodeError::Limit("file bytes"))
        ));
        assert_eq!(small.file_bytes().unwrap(), HEADER_BYTES);
    }

    #[test]
    fn frame_limit_is_checked_before_encoding_large_fields() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open_with_limits(
            temp.path(),
            EpisodeLimits {
                max_frame_bytes: 64,
                max_text_bytes: 64,
                ..EpisodeLimits::default()
            },
        )
        .unwrap()
        .0;
        assert!(matches!(
            store.observe(stamp("1", 1), key(), EpisodeValue::Text("x".repeat(60))),
            Err(EpisodeError::Limit("frame bytes"))
        ));
        assert_eq!(store.file_bytes().unwrap(), HEADER_BYTES);
    }

    #[test]
    fn payload_length_matches_encoder_for_all_variants_and_utf8() {
        let data = [
            EpisodeData::Trace {
                context: "lab".into(),
                channel: "transition/v1".into(),
                fields: BTreeMap::from([
                    ("text".into(), EpisodeValue::Text("caf\u{e9}".into())),
                    ("bool".into(), EpisodeValue::Boolean(true)),
                    ("integer".into(), EpisodeValue::Integer(i64::MIN)),
                ]),
            },
            EpisodeData::Fact {
                key: key(),
                value: EpisodeValue::Text("caf\u{e9}".into()),
                replaces: None,
            },
            EpisodeData::Fact {
                key: key(),
                value: value(i64::MIN),
                replaces: Some(1),
            },
            EpisodeData::Fact {
                key: key(),
                value: EpisodeValue::Boolean(false),
                replaces: None,
            },
            EpisodeData::Retraction { target: 1 },
            EpisodeData::Outcome {
                context: "c".into(),
                action: "a".into(),
                success: true,
                cost_ns: u64::MAX,
                detail: "d".into(),
            },
        ];
        for data in data {
            let episode = Episode {
                id: 1,
                stamp: stamp("1", u64::MAX),
                data,
            };
            let bytes = encode_episode(&episode);
            assert_eq!(encoded_length(&episode).unwrap(), bytes.len());
            assert_eq!(
                decode_episode(&bytes, EpisodeLimits::default()).unwrap(),
                episode
            );
        }
    }

    #[test]
    fn every_partial_frame_recovers_only_the_committed_prefix() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        store.observe(stamp("1", 1), key(), value(1)).unwrap();
        let boundary = store.file_bytes().unwrap() as usize;
        store.observe(stamp("2", 2), key(), value(2)).unwrap();
        drop(store);
        let bytes = fs::read(temp.path()).unwrap();
        for cut in boundary..bytes.len() {
            let path = temp.0.join("cut.axep");
            fs::write(&path, &bytes[..cut]).unwrap();
            let (mut store, report) = EpisodeStore::open(&path).unwrap();
            assert_eq!(store.len(), 1, "cut {cut}");
            assert_eq!(report.discarded_tail_bytes, (cut - boundary) as u64);
            store.file.seek(SeekFrom::Start(0)).unwrap();
            let mut recovered = Vec::new();
            store.file.read_to_end(&mut recovered).unwrap();
            assert_eq!(recovered, bytes[..boundary]);
            store.observe(stamp("2", 2), key(), value(2)).unwrap();
            drop(store);
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
    }

    #[test]
    fn complete_single_byte_corruption_is_refused_without_rewriting() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        store.observe(stamp("1", 1), key(), value(1)).unwrap();
        store.observe(stamp("2", 2), key(), value(2)).unwrap();
        drop(store);
        let bytes = fs::read(temp.path()).unwrap();
        for index in 0..bytes.len() {
            let path = temp.0.join("corrupt.axep");
            let mut corrupt = bytes.clone();
            corrupt[index] ^= 1;
            fs::write(&path, &corrupt).unwrap();
            assert!(EpisodeStore::open(&path).is_err(), "byte {index}");
            assert_eq!(fs::read(&path).unwrap(), corrupt);
        }
    }

    #[test]
    fn forged_checksums_do_not_bypass_semantic_validation_or_chain_order() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        store.observe(stamp("1", 1), key(), value(1)).unwrap();
        let boundary = store.file_bytes().unwrap() as usize;
        store.observe(stamp("2", 2), key(), value(2)).unwrap();
        drop(store);
        let original = fs::read(temp.path()).unwrap();
        let mut forged = original.clone();
        forged[32..40].copy_from_slice(&2_u64.to_le_bytes());
        let hash = checksum(&forged[16..boundary - 8]);
        forged[boundary - 8..boundary].copy_from_slice(&hash.to_le_bytes());
        fs::write(temp.path(), &forged).unwrap();
        assert!(matches!(
            EpisodeStore::open(temp.path()),
            Err(EpisodeError::Invalid("episode sequence"))
        ));
        assert_eq!(fs::read(temp.path()).unwrap(), forged);
        let reordered = [
            &original[..16],
            &original[boundary..],
            &original[16..boundary],
        ]
        .concat();
        fs::write(temp.path(), &reordered).unwrap();
        assert!(matches!(
            EpisodeStore::open(temp.path()),
            Err(EpisodeError::Invalid("journal chain"))
        ));
    }

    #[test]
    fn exclusive_lock_includes_hardlink_aliases() {
        let temp = Temp::new();
        let store = EpisodeStore::open(temp.path()).unwrap().0;
        assert!(EpisodeStore::open(temp.path()).is_err());
        let alias = temp.0.join("alias.axep");
        fs::hard_link(temp.path(), &alias).unwrap();
        assert!(EpisodeStore::open(alias).is_err());
        drop(store);
        assert!(EpisodeStore::open(temp.path()).is_ok());
    }

    #[test]
    fn io_failure_poisoning_prevents_unsynced_queries_and_appends() {
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        store.file = File::open(temp.path()).unwrap();
        assert!(matches!(
            store.observe(stamp("1", 1), key(), value(1)),
            Err(EpisodeError::Io(_))
        ));
        assert!(matches!(
            store.belief(&key(), None, None),
            Err(EpisodeError::NeedsRecovery)
        ));
        assert!(matches!(
            store.experience("lab"),
            Err(EpisodeError::NeedsRecovery)
        ));
        assert!(matches!(
            store.history(&key()),
            Err(EpisodeError::NeedsRecovery)
        ));
        assert!(matches!(
            store.observe(stamp("1", 1), key(), value(1)),
            Err(EpisodeError::NeedsRecovery)
        ));
        assert_eq!(store.len(), 0);
        drop(store);
        assert_eq!(EpisodeStore::open(temp.path()).unwrap().0.len(), 0);
    }

    #[test]
    fn failed_experience_write_reports_completed_solver_effects_explicitly() {
        let temp = Temp::new();
        let mut episodes = EpisodeStore::open(temp.path()).unwrap().0;
        episodes.file = File::open(temp.path()).unwrap();
        let artifacts = crate::ArtifactStore::open(temp.0.join("artifacts"));
        let task = crate::AxonTask::parse("task completed {\n data numbers: Vec<u64> = [1, 2]\n goal derive IncrementalArtifact<AddModU64>\n}\n").unwrap();
        let result = crate::solve_task_with_experience(
            &artifacts,
            &task,
            &mut episodes,
            "fixed-host/2-items",
        );
        let Err(crate::ExperiencedSolveError::Memory {
            executed: Some(execution),
            ..
        }) = result
        else {
            panic!("must report separate execution and recording outcomes");
        };
        let receipt = execution.unwrap();
        assert_eq!(receipt.outcome(), &crate::ArtifactOutcome::ModularSum(3));
        assert!(receipt.artifact_path().exists());
        assert!(matches!(
            episodes.experience("fixed-host/2-items"),
            Err(EpisodeError::NeedsRecovery)
        ));
    }

    #[test]
    fn world_action_is_reported_and_session_stops_when_memory_write_fails() {
        use crate::{Authority, Effect, world::*};
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        store.file = File::open(temp.path()).unwrap();
        let shape = WorldShape::new(2, 1).unwrap();
        let mut session = WorldSession::restore(&mut store, shape, "fault-test").unwrap();
        let mut scope = ProbeScope::unrestricted(shape, 100);
        let authority = Authority::from([Effect::Simulate, Effect::WriteData]);
        let probe = Probe {
            state: 0,
            action: 0,
            cost: 1,
        };
        let mut calls = 0;
        assert!(matches!(
            session.intervene(probe, &mut scope, &authority, |s, _| {
                calls += 1;
                s ^ 1
            }),
            Err(WorldError::ActionOccurred { after: 1, .. })
        ));
        assert_eq!(scope.remaining, 99);
        assert_eq!(session.learner().records(), 0);
        assert!(
            session
                .intervene(probe, &mut scope, &authority, |s, _| {
                    calls += 1;
                    s
                })
                .is_err()
        );
        assert_eq!(calls, 1);
    }

    #[test]
    fn trace_decoder_refuses_excess_or_duplicate_fields() {
        let mut prefix = Vec::new();
        prefix.extend_from_slice(&1_u64.to_le_bytes());
        prefix.extend_from_slice(&1_u64.to_le_bytes());
        put_text(&mut prefix, "test");
        put_text(&mut prefix, "one");
        prefix.push(3);
        put_text(&mut prefix, "lab");
        put_text(&mut prefix, "trace");
        let mut excess = prefix.clone();
        excess.push(17);
        assert!(matches!(
            decode_episode(&excess, EpisodeLimits::default()),
            Err(EpisodeError::Limit("trace fields"))
        ));
        prefix.push(2);
        for _ in 0..2 {
            put_text(&mut prefix, "same");
            put_value(&mut prefix, &EpisodeValue::Boolean(true));
        }
        assert!(matches!(
            decode_episode(&prefix, EpisodeLimits::default()),
            Err(EpisodeError::Invalid("duplicate trace field"))
        ));
    }

    #[test]
    fn behavior_query_reports_effects_and_suspends_after_failed_commit() {
        use crate::{Authority, Effect, behavior::*};
        struct Adapter;
        impl Resettable for Adapter {
            fn reset(&mut self) -> Result<(), BehaviorError> {
                Ok(())
            }
            fn step(&mut self, _: u8) -> Result<u16, BehaviorError> {
                Ok(7)
            }
        }
        let temp = Temp::new();
        let mut store = EpisodeStore::open(temp.path()).unwrap().0;
        store.file = File::open(temp.path()).unwrap();
        let mut memory = BehaviorMemory::restore(
            &mut store,
            "commit-failure",
            1,
            BehaviorBudget::default(),
            Authority::from([Effect::Simulate, Effect::WriteData]),
            Adapter,
        )
        .unwrap();
        assert!(
            matches!(memory.query(&[0,0]), Err(BehaviorError::QueryOccurred { reply, .. }) if reply == vec![7,7])
        );
        assert!(memory.suspended());
        assert!(memory.query(&[0]).is_err());
        drop(memory);
        assert_eq!(store.len(), 0);
        assert!(store.traces("commit-failure", "behavior.query/v1").is_err());
    }

    #[test]
    fn append_child() {
        let Ok(root) = std::env::var("AXON_EPISODE_TEST_ROOT") else {
            return;
        };
        let target = std::env::var("AXON_EPISODE_TEST_STAGE").unwrap();
        let mut store = EpisodeStore::open(Path::new(&root).join("episodes.axep"))
            .unwrap()
            .0;
        store
            .record_observed(
                stamp("1", 1),
                EpisodeData::Fact {
                    key: key(),
                    value: value(1),
                    replaces: None,
                },
                |stage| {
                    if format!("{stage:?}") == target {
                        fs::write(Path::new(&root).join("ready"), b"ready").unwrap();
                        loop {
                            std::thread::park();
                        }
                    }
                },
            )
            .unwrap();
    }

    #[test]
    fn process_kill_at_four_append_stages_recovers_without_duplicate_application() {
        for stage in [
            AppendStage::HeaderWritten,
            AppendStage::PayloadWritten,
            AppendStage::Synced,
            AppendStage::Applied,
        ] {
            let temp = Temp::new();
            drop(EpisodeStore::open(temp.path()).unwrap());
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "episodic::tests::append_child", "--nocapture"])
                .env("AXON_EPISODE_TEST_ROOT", &temp.0)
                .env("AXON_EPISODE_TEST_STAGE", format!("{stage:?}"))
                .stdout(Stdio::null())
                .spawn()
                .unwrap();
            let start = Instant::now();
            while !temp.0.join("ready").exists() && start.elapsed() < Duration::from_secs(10) {
                if child.try_wait().unwrap().is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            child.kill().unwrap();
            child.wait().unwrap();
            assert!(temp.0.join("ready").exists(), "stage {stage:?} not reached");
            let (mut store, report) = EpisodeStore::open(temp.path()).unwrap();
            let committed = stage != AppendStage::HeaderWritten;
            assert_eq!(store.len(), usize::from(committed));
            assert_eq!(report.discarded_tail_bytes, if committed { 0 } else { 16 });
            let receipt = store.observe(stamp("1", 1), key(), value(1)).unwrap();
            assert_eq!(receipt.duplicate, committed);
            assert_eq!(store.len(), 1);
        }
    }
}
