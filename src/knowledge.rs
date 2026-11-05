use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAGIC: &[u8; 8] = b"AXMEM001";
const LEGACY_FORMAT_VERSION: u32 = 1;
const FORMAT_VERSION: u32 = 2;
const HEADER_BYTES: usize = 36;
const MAX_CHUNK_BYTES: usize = 1_536;
const MAX_STORED_CHUNK_BYTES: usize = 64 * 1024;
const MAX_FEEDBACK_TERM_BYTES: usize = 256;
const INDEXING_VERSION: u64 = 6;
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;
const MIN_WEIGHTED_QUERY_COVERAGE: f64 = 0.3;
const MIN_COMPOSED_QUERY_COVERAGE: f64 = 0.45;
const MIN_COMPOSED_EXACT_COVERAGE: f64 = 0.35;
const MAX_EVIDENCE_PER_BUNDLE: usize = 3;
const MIN_SUBWORD_SIMILARITY: f64 = 0.52;
const MAX_GRAPH_CANDIDATES: usize = 96;
const MAX_GRAPH_SEEDS: usize = MAX_GRAPH_CANDIDATES;
const MAX_EVIDENCE_PER_PATH: usize = 6;
const MAX_EVIDENCE_PER_SOURCE_PATH: usize = 3;
const MAX_SOURCES_PER_PATH: usize = 4;
const MAX_BRIDGE_TERMS: usize = 3;
const BRIDGE_DOCUMENT_DIVISOR: usize = 20;
const MIN_PATH_QUERY_COVERAGE: f64 = 0.72;
const MIN_PATH_EXACT_COVERAGE: f64 = 0.70;
const MIN_DIRECT_QUERY_COVERAGE: f64 = 0.75;
const MIN_DIRECT_EXACT_COVERAGE: f64 = 0.70;
const DEFAULT_MEMORY_PATH: &str = ".axon-memory/knowledge.axm";
static TEMPORARY_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnowledgeLimits {
    pub max_index_bytes: u64,
    pub max_input_file_bytes: u64,
    pub max_sources: usize,
    pub max_chunks: usize,
    pub max_total_text_bytes: u64,
    pub max_path_bytes: usize,
    pub max_feedback_records: usize,
    pub max_feedback_terms: usize,
}

impl Default for KnowledgeLimits {
    fn default() -> Self {
        Self {
            max_index_bytes: 512 * 1024 * 1024,
            max_input_file_bytes: 32 * 1024 * 1024,
            max_sources: 100_000,
            max_chunks: 1_000_000,
            max_total_text_bytes: 256 * 1024 * 1024,
            max_path_bytes: 32 * 1024,
            max_feedback_records: 10_000,
            max_feedback_terms: 32,
        }
    }
}

#[derive(Debug)]
pub enum KnowledgeError {
    Io {
        action: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidFormat(&'static str),
    UnsupportedVersion(u32),
    NonUnicodePath(PathBuf),
    InvalidUtf8(PathBuf),
    InputTooLarge {
        path: PathBuf,
        bytes: u64,
        maximum: u64,
    },
    CapacityExceeded(&'static str),
    UnknownSource(PathBuf),
    EmptyQuery,
    InvalidLimit,
}

impl fmt::Display for KnowledgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { action, path, .. } => {
                write!(formatter, "cannot {action} `{}`", path.display())
            }
            Self::InvalidFormat(reason) => write!(formatter, "invalid AXON memory: {reason}"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported AXON memory version {version}")
            }
            Self::NonUnicodePath(path) => {
                write!(
                    formatter,
                    "source path is not valid Unicode: `{}`",
                    path.display()
                )
            }
            Self::InvalidUtf8(path) => {
                write!(formatter, "source is not UTF-8 text: `{}`", path.display())
            }
            Self::InputTooLarge {
                path,
                bytes,
                maximum,
            } => write!(
                formatter,
                "source `{}` has {bytes} bytes, above the {maximum}-byte limit",
                path.display()
            ),
            Self::CapacityExceeded(resource) => {
                write!(formatter, "AXON memory exceeds its {resource} limit")
            }
            Self::UnknownSource(path) => write!(
                formatter,
                "source is not present in AXON memory: `{}`",
                path.display()
            ),
            Self::EmptyQuery => formatter.write_str("query has no searchable terms"),
            Self::InvalidLimit => formatter.write_str("result limit must be between 1 and 100"),
        }
    }
}

impl std::error::Error for KnowledgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LearnReport {
    discovered_files: usize,
    added_sources: usize,
    updated_sources: usize,
    unchanged_sources: usize,
    skipped_files: usize,
    changed_chunks: usize,
    bytes_read: u64,
    total_sources: usize,
    total_chunks: usize,
    index_bytes: u64,
}

impl LearnReport {
    pub fn discovered_files(&self) -> usize {
        self.discovered_files
    }

    pub fn added_sources(&self) -> usize {
        self.added_sources
    }

    pub fn updated_sources(&self) -> usize {
        self.updated_sources
    }

    pub fn unchanged_sources(&self) -> usize {
        self.unchanged_sources
    }

    pub fn skipped_files(&self) -> usize {
        self.skipped_files
    }

    pub fn changed_chunks(&self) -> usize {
        self.changed_chunks
    }

    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    pub fn total_sources(&self) -> usize {
        self.total_sources
    }

    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }

    pub fn index_bytes(&self) -> u64 {
        self.index_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KnowledgeStats {
    sources: usize,
    chunks: usize,
    text_bytes: u64,
    index_bytes: u64,
    feedback_records: usize,
    feedback_observations: u64,
}

impl KnowledgeStats {
    pub fn sources(self) -> usize {
        self.sources
    }

    pub fn chunks(self) -> usize {
        self.chunks
    }

    pub fn text_bytes(self) -> u64 {
        self.text_bytes
    }

    pub fn index_bytes(self) -> u64 {
        self.index_bytes
    }

    pub fn feedback_records(self) -> usize {
        self.feedback_records
    }

    pub fn feedback_observations(self) -> u64 {
        self.feedback_observations
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnowledgeFeedbackLabel {
    Relevant,
    Irrelevant,
}

impl KnowledgeFeedbackLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Relevant => "relevant",
            Self::Irrelevant => "irrelevant",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnowledgeFeedbackReport {
    source: PathBuf,
    label: KnowledgeFeedbackLabel,
    relevant_observations: u32,
    irrelevant_observations: u32,
    total_records: usize,
}

impl KnowledgeFeedbackReport {
    pub fn source(&self) -> &Path {
        &self.source
    }

    pub fn label(&self) -> KnowledgeFeedbackLabel {
        self.label
    }

    pub fn relevant_observations(&self) -> u32 {
        self.relevant_observations
    }

    pub fn irrelevant_observations(&self) -> u32 {
        self.irrelevant_observations
    }

    pub fn total_records(&self) -> usize {
        self.total_records
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecallStatus {
    Found,
    WeakEvidence,
    NoEvidence,
}

impl RecallStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Found => "found",
            Self::WeakEvidence => "weak_evidence",
            Self::NoEvidence => "no_evidence",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecallHit {
    source: PathBuf,
    start_line: u32,
    end_line: u32,
    score: f64,
    feedback_factor: f64,
    matched_terms: Vec<String>,
    text: String,
}

impl RecallHit {
    pub fn source(&self) -> &Path {
        &self.source
    }

    pub fn start_line(&self) -> u32 {
        self.start_line
    }

    pub fn end_line(&self) -> u32 {
        self.end_line
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn feedback_factor(&self) -> f64 {
        self.feedback_factor
    }

    pub fn matched_terms(&self) -> &[String] {
        &self.matched_terms
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecallResult {
    status: RecallStatus,
    query_terms: Vec<String>,
    search_work: SearchWork,
    hits: Vec<RecallHit>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchWork {
    indexed_documents: usize,
    scored_documents: usize,
    posting_visits: usize,
}

impl SearchWork {
    pub fn indexed_documents(self) -> usize {
        self.indexed_documents
    }

    pub fn scored_documents(self) -> usize {
        self.scored_documents
    }

    pub fn posting_visits(self) -> usize {
        self.posting_visits
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceBundle {
    source: PathBuf,
    score: f64,
    weighted_coverage: f64,
    exact_coverage: f64,
    covered_terms: Vec<String>,
    missing_terms: Vec<String>,
    evidence: Vec<RecallHit>,
}

impl EvidenceBundle {
    pub fn source(&self) -> &Path {
        &self.source
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn weighted_coverage(&self) -> f64 {
        self.weighted_coverage
    }

    pub fn exact_coverage(&self) -> f64 {
        self.exact_coverage
    }

    pub fn covered_terms(&self) -> &[String] {
        &self.covered_terms
    }

    pub fn missing_terms(&self) -> &[String] {
        &self.missing_terms
    }

    pub fn evidence(&self) -> &[RecallHit] {
        &self.evidence
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceResult {
    status: RecallStatus,
    query_terms: Vec<String>,
    search_work: SearchWork,
    bundles: Vec<EvidenceBundle>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceLinkKind {
    Adjacent,
    SharedTerms,
    SourceEntity,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EvidenceSourceRole {
    Documentation,
    Implementation,
    Test,
    Data,
    Other,
}

impl EvidenceSourceRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Documentation => "documentation",
            Self::Implementation => "implementation",
            Self::Test => "test",
            Self::Data => "data",
            Self::Other => "other",
        }
    }
}

impl EvidenceLinkKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Adjacent => "adjacent",
            Self::SharedTerms => "shared_terms",
            Self::SourceEntity => "source_entity",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceLink {
    from_evidence: usize,
    to_evidence: usize,
    kind: EvidenceLinkKind,
    score: f64,
    bridge_terms: Vec<String>,
}

impl EvidenceLink {
    pub fn from_evidence(&self) -> usize {
        self.from_evidence
    }

    pub fn to_evidence(&self) -> usize {
        self.to_evidence
    }

    pub fn kind(&self) -> EvidenceLinkKind {
        self.kind
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn bridge_terms(&self) -> &[String] {
        &self.bridge_terms
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidencePath {
    score: f64,
    weighted_coverage: f64,
    exact_coverage: f64,
    coherence: f64,
    source_coverage: f64,
    source_union_coverage: f64,
    source_union_exact_coverage: f64,
    corroborated_coverage: f64,
    source_name_affinity: f64,
    anchor_source_count: usize,
    anchor_definition_count: usize,
    query_example_count: usize,
    covered_terms: Vec<String>,
    missing_terms: Vec<String>,
    evidence: Vec<RecallHit>,
    evidence_roles: Vec<EvidenceSourceRole>,
    links: Vec<EvidenceLink>,
}

impl EvidencePath {
    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn weighted_coverage(&self) -> f64 {
        self.weighted_coverage
    }

    pub fn exact_coverage(&self) -> f64 {
        self.exact_coverage
    }

    pub fn coherence(&self) -> f64 {
        self.coherence
    }

    pub fn source_coverage(&self) -> f64 {
        self.source_coverage
    }

    pub fn source_union_coverage(&self) -> f64 {
        self.source_union_coverage
    }

    pub fn source_union_exact_coverage(&self) -> f64 {
        self.source_union_exact_coverage
    }

    pub fn corroborated_coverage(&self) -> f64 {
        self.corroborated_coverage
    }

    pub fn source_name_affinity(&self) -> f64 {
        self.source_name_affinity
    }

    pub fn anchor_source_count(&self) -> usize {
        self.anchor_source_count
    }

    pub fn anchor_definition_count(&self) -> usize {
        self.anchor_definition_count
    }

    pub fn query_example_count(&self) -> usize {
        self.query_example_count
    }

    pub fn covered_terms(&self) -> &[String] {
        &self.covered_terms
    }

    pub fn missing_terms(&self) -> &[String] {
        &self.missing_terms
    }

    pub fn evidence(&self) -> &[RecallHit] {
        &self.evidence
    }

    pub fn evidence_roles(&self) -> &[EvidenceSourceRole] {
        &self.evidence_roles
    }

    pub fn distinct_role_count(&self) -> usize {
        self.evidence_roles
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn links(&self) -> &[EvidenceLink] {
        &self.links
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InvestigationResult {
    status: RecallStatus,
    query_terms: Vec<String>,
    anchor_terms: Vec<String>,
    unsupported_terms: Vec<String>,
    search_work: SearchWork,
    paths: Vec<EvidencePath>,
}

impl InvestigationResult {
    pub fn search_work(&self) -> SearchWork {
        self.search_work
    }

    pub fn status(&self) -> RecallStatus {
        self.status
    }

    pub fn query_terms(&self) -> &[String] {
        &self.query_terms
    }

    pub fn anchor_terms(&self) -> &[String] {
        &self.anchor_terms
    }

    pub fn unsupported_terms(&self) -> &[String] {
        &self.unsupported_terms
    }

    pub fn paths(&self) -> &[EvidencePath] {
        &self.paths
    }
}

impl EvidenceResult {
    pub fn search_work(&self) -> SearchWork {
        self.search_work
    }

    pub fn status(&self) -> RecallStatus {
        self.status
    }

    pub fn query_terms(&self) -> &[String] {
        &self.query_terms
    }

    pub fn bundles(&self) -> &[EvidenceBundle] {
        &self.bundles
    }
}

impl RecallResult {
    pub fn search_work(&self) -> SearchWork {
        self.search_work
    }

    pub fn status(&self) -> RecallStatus {
        self.status
    }

    pub fn query_terms(&self) -> &[String] {
        &self.query_terms
    }

    pub fn hits(&self) -> &[RecallHit] {
        &self.hits
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Fingerprint {
    bytes: u64,
    first: u64,
    second: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredChunk {
    start_line: u32,
    end_line: u32,
    text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceRecord {
    fingerprint: Fingerprint,
    chunks: Vec<StoredChunk>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct FeedbackKey {
    source: String,
    query_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FeedbackRecord {
    relevant: u32,
    irrelevant: u32,
}

#[derive(Debug, Default)]
struct StoredMemory {
    sources: BTreeMap<String, SourceRecord>,
    feedback: BTreeMap<FeedbackKey, FeedbackRecord>,
}

#[derive(Debug, Default)]
struct SearchIndex {
    documents: Vec<IndexedDocument>,
    postings: BTreeMap<String, Vec<usize>>,
    document_frequency: BTreeMap<String, usize>,
    source_frequency: BTreeMap<String, usize>,
    source_tokens: BTreeMap<String, BTreeSet<String>>,
    source_vocabulary: BTreeMap<String, BTreeSet<String>>,
    source_definitions: BTreeMap<String, BTreeSet<String>>,
    subword_signatures: BTreeMap<String, Box<[u64]>>,
    average_length: f64,
}

impl SearchIndex {
    fn build(sources: &BTreeMap<String, SourceRecord>) -> Self {
        let mut documents = Vec::new();
        let mut postings = BTreeMap::<String, Vec<usize>>::new();
        let mut document_frequency = BTreeMap::<String, usize>::new();
        let mut source_frequency = BTreeMap::<String, usize>::new();
        let mut source_tokens = BTreeMap::new();
        let mut source_vocabulary = BTreeMap::new();
        let mut source_definitions = BTreeMap::new();
        let mut total_tokens = 0_usize;
        for (source, record) in sources {
            source_tokens.insert(source.clone(), tokenize(source).into_iter().collect());
            let mut vocabulary = BTreeSet::new();
            let mut definitions = BTreeSet::new();
            for (chunk_index, chunk) in record.chunks.iter().enumerate() {
                let tokens = tokenize(&chunk.text);
                let unique_tokens = tokens.iter().cloned().collect::<BTreeSet<_>>();
                vocabulary.extend(unique_tokens.iter().cloned());
                if evidence_source_role(Path::new(source)) == EvidenceSourceRole::Implementation {
                    definitions.extend(declared_identifiers(&chunk.text));
                }
                total_tokens = total_tokens.saturating_add(tokens.len());
                for token in &unique_tokens {
                    *document_frequency.entry(token.clone()).or_default() += 1;
                    postings
                        .entry(token.clone())
                        .or_default()
                        .push(documents.len());
                }
                documents.push(IndexedDocument {
                    source: source.clone(),
                    chunk_index,
                    tokens,
                    unique_tokens,
                    has_numeric_answer: contains_numeric_answer(&chunk.text),
                    has_size_answer: contains_size_answer(&chunk.text),
                });
            }
            for token in &vocabulary {
                *source_frequency.entry(token.clone()).or_default() += 1;
            }
            source_vocabulary.insert(source.clone(), vocabulary);
            source_definitions.insert(source.clone(), definitions);
        }
        let average_length = if documents.is_empty() {
            0.0
        } else {
            total_tokens as f64 / documents.len() as f64
        };
        let subword_signatures = document_frequency
            .keys()
            .map(|term| (term.clone(), subword_signature(term).into_boxed_slice()))
            .collect();
        Self {
            documents,
            postings,
            document_frequency,
            source_frequency,
            source_tokens,
            source_vocabulary,
            source_definitions,
            subword_signatures,
            average_length,
        }
    }
}

#[derive(Debug)]
struct IndexedDocument {
    source: String,
    chunk_index: usize,
    tokens: Vec<String>,
    unique_tokens: BTreeSet<String>,
    has_numeric_answer: bool,
    has_size_answer: bool,
}

pub struct KnowledgeStore {
    path: PathBuf,
    _lock: File,
    limits: KnowledgeLimits,
    sources: BTreeMap<String, SourceRecord>,
    feedback: BTreeMap<FeedbackKey, FeedbackRecord>,
    search_index: SearchIndex,
}

impl KnowledgeStore {
    pub fn default_path() -> PathBuf {
        PathBuf::from(DEFAULT_MEMORY_PATH)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, KnowledgeError> {
        Self::open_with_limits(path, KnowledgeLimits::default())
    }

    pub fn open_with_limits(
        path: impl AsRef<Path>,
        limits: KnowledgeLimits,
    ) -> Result<Self, KnowledgeError> {
        validate_limits(limits)?;
        let path = resolve_store_path(path.as_ref())?;
        let lock_path = sidecar_path(&path, ".lock");
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|source| io_error("open memory lock", &lock_path, source))?;
        lock.try_lock()
            .map_err(|source| io_error("lock memory", &lock_path, source.into()))?;

        recover_interrupted_publish(&path)?;
        let stored = if path.exists() {
            load_memory(&path, limits)?
        } else {
            StoredMemory::default()
        };
        let sources = stored.sources;
        let feedback = stored.feedback;
        let backup = sidecar_path(&path, ".previous");
        if path.exists() && backup.exists() {
            fs::remove_file(&backup)
                .map_err(|source| io_error("remove old memory backup", &backup, source))?;
        }

        let search_index = SearchIndex::build(&sources);
        Ok(Self {
            path,
            _lock: lock,
            limits,
            sources,
            feedback,
            search_index,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn learn_path(&mut self, input: impl AsRef<Path>) -> Result<LearnReport, KnowledgeError> {
        let input = input.as_ref();
        let metadata =
            fs::metadata(input).map_err(|source| io_error("inspect source", input, source))?;
        let explicit_file = metadata.is_file();
        let mut report = LearnReport::default();
        let mut files = Vec::new();
        if explicit_file {
            files.push(
                fs::canonicalize(input)
                    .map_err(|source| io_error("resolve source", input, source))?,
            );
        } else if metadata.is_dir() {
            let mut visited_directories = BTreeSet::new();
            collect_text_files(
                input,
                &mut files,
                &mut report.skipped_files,
                self.limits.max_sources,
                0,
                &mut visited_directories,
            )?;
        } else {
            return Err(KnowledgeError::InvalidFormat(
                "learning input must be a file or directory",
            ));
        }
        files.sort();
        files.dedup();
        report.discovered_files = files.len();

        let mut updates = Vec::new();
        for file in files {
            let metadata =
                fs::metadata(&file).map_err(|source| io_error("inspect source", &file, source))?;
            let bytes = metadata.len();
            if bytes > self.limits.max_input_file_bytes {
                if explicit_file {
                    return Err(KnowledgeError::InputTooLarge {
                        path: file,
                        bytes,
                        maximum: self.limits.max_input_file_bytes,
                    });
                }
                report.skipped_files += 1;
                continue;
            }
            let raw = read_bounded_file(&file, self.limits.max_input_file_bytes)?;
            report.bytes_read = report
                .bytes_read
                .checked_add(raw.len() as u64)
                .ok_or(KnowledgeError::CapacityExceeded("input byte count"))?;
            let text = match std::str::from_utf8(&raw) {
                Ok(text) => text,
                Err(_) if !explicit_file => {
                    report.skipped_files += 1;
                    continue;
                }
                Err(_) => return Err(KnowledgeError::InvalidUtf8(file)),
            };
            let source = file
                .to_str()
                .ok_or_else(|| KnowledgeError::NonUnicodePath(file.clone()))?
                .to_owned();
            if source.len() > self.limits.max_path_bytes {
                return Err(KnowledgeError::CapacityExceeded("source path byte count"));
            }
            let fingerprint = fingerprint(&raw);
            if self.sources.get(&source).map(|entry| entry.fingerprint) == Some(fingerprint) {
                report.unchanged_sources += 1;
                continue;
            }
            let chunks = chunk_text(text)?;
            report.changed_chunks = report
                .changed_chunks
                .checked_add(chunks.len())
                .ok_or(KnowledgeError::CapacityExceeded("chunk count"))?;
            if self.sources.contains_key(&source) {
                report.updated_sources += 1;
            } else {
                report.added_sources += 1;
            }
            updates.push((
                source,
                SourceRecord {
                    fingerprint,
                    chunks,
                },
            ));
        }

        let updated_sources = updates
            .iter()
            .filter(|(source, _)| self.sources.contains_key(source))
            .map(|(source, _)| source.clone())
            .collect::<BTreeSet<_>>();
        for (source, record) in updates {
            self.sources.insert(source, record);
        }
        if !updated_sources.is_empty() {
            self.feedback
                .retain(|key, _| !updated_sources.contains(&key.source));
        }
        validate_sources(&self.sources, self.limits)?;
        if report.added_sources + report.updated_sources > 0 {
            self.search_index = SearchIndex::build(&self.sources);
            self.persist()?;
        }
        let stats = self.stats()?;
        report.total_sources = stats.sources;
        report.total_chunks = stats.chunks;
        report.index_bytes = stats.index_bytes;
        Ok(report)
    }

    pub fn forget(&mut self, source: impl AsRef<Path>) -> Result<usize, KnowledgeError> {
        let source = source.as_ref();
        let resolved = if source.exists() {
            fs::canonicalize(source).map_err(|error| io_error("resolve source", source, error))?
        } else {
            std::path::absolute(source)
                .map_err(|error| io_error("resolve source", source, error))?
        };
        let prefix = resolved
            .to_str()
            .ok_or_else(|| KnowledgeError::NonUnicodePath(resolved.clone()))?;
        let separator = std::path::MAIN_SEPARATOR;
        let directory_prefix = format!("{prefix}{separator}");
        let before = self.sources.len();
        self.sources
            .retain(|stored, _| stored != prefix && !stored.starts_with(&directory_prefix));
        let removed = before - self.sources.len();
        if removed > 0 {
            self.feedback.retain(|key, _| {
                key.source != prefix && !key.source.starts_with(&directory_prefix)
            });
            self.search_index = SearchIndex::build(&self.sources);
            self.persist()?;
        }
        Ok(removed)
    }

    pub fn stats(&self) -> Result<KnowledgeStats, KnowledgeError> {
        let (chunks, text_bytes) = source_totals(&self.sources)?;
        let index_bytes = if self.path.exists() {
            fs::metadata(&self.path)
                .map_err(|source| io_error("inspect memory", &self.path, source))?
                .len()
        } else {
            0
        };
        let feedback_observations = feedback_observation_count(&self.feedback)?;
        Ok(KnowledgeStats {
            sources: self.sources.len(),
            chunks,
            text_bytes,
            index_bytes,
            feedback_records: self.feedback.len(),
            feedback_observations,
        })
    }

    pub fn record_feedback(
        &mut self,
        query: &str,
        source: impl AsRef<Path>,
        label: KnowledgeFeedbackLabel,
    ) -> Result<KnowledgeFeedbackReport, KnowledgeError> {
        let query_terms = feedback_query_terms(query, self.limits.max_feedback_terms)?;
        let source = resolve_source_path(source.as_ref())?;
        let source_key = source
            .to_str()
            .ok_or_else(|| KnowledgeError::NonUnicodePath(source.clone()))?
            .to_owned();
        if !self.sources.contains_key(&source_key) {
            return Err(KnowledgeError::UnknownSource(source));
        }
        let key = FeedbackKey {
            source: source_key,
            query_terms,
        };
        if !self.feedback.contains_key(&key)
            && self.feedback.len() >= self.limits.max_feedback_records
        {
            return Err(KnowledgeError::CapacityExceeded("feedback record count"));
        }

        let previous = self.feedback.get(&key).copied();
        let record = self.feedback.entry(key.clone()).or_insert(FeedbackRecord {
            relevant: 0,
            irrelevant: 0,
        });
        match label {
            KnowledgeFeedbackLabel::Relevant => {
                record.relevant =
                    record
                        .relevant
                        .checked_add(1)
                        .ok_or(KnowledgeError::CapacityExceeded(
                            "relevant feedback observation count",
                        ))?;
            }
            KnowledgeFeedbackLabel::Irrelevant => {
                record.irrelevant =
                    record
                        .irrelevant
                        .checked_add(1)
                        .ok_or(KnowledgeError::CapacityExceeded(
                            "irrelevant feedback observation count",
                        ))?;
            }
        }
        let updated = *record;
        if let Err(error) = self.persist() {
            match previous {
                Some(record) => {
                    self.feedback.insert(key, record);
                }
                None => {
                    self.feedback.remove(&key);
                }
            }
            return Err(error);
        }
        Ok(KnowledgeFeedbackReport {
            source,
            label,
            relevant_observations: updated.relevant,
            irrelevant_observations: updated.irrelevant,
            total_records: self.feedback.len(),
        })
    }

    pub fn recall(&self, query: &str, limit: usize) -> Result<RecallResult, KnowledgeError> {
        if !(1..=100).contains(&limit) {
            return Err(KnowledgeError::InvalidLimit);
        }
        let ScoredQuery {
            query_terms,
            query_weights,
            total_query_weight,
            mut hits,
            search_work,
            ..
        } = self.score_query(query)?;
        sort_scored_hits(&mut hits);
        hits.truncate(limit);
        let status = recall_status(&query_terms, &query_weights, total_query_weight, &hits);
        if status == RecallStatus::NoEvidence {
            hits.clear();
        }
        let hits = hits.into_iter().map(|candidate| candidate.hit).collect();
        Ok(RecallResult {
            status,
            query_terms,
            search_work,
            hits,
        })
    }

    pub fn reason(&self, query: &str, limit: usize) -> Result<EvidenceResult, KnowledgeError> {
        if !(1..=100).contains(&limit) {
            return Err(KnowledgeError::InvalidLimit);
        }
        let ScoredQuery {
            query_terms,
            query_weights,
            total_query_weight,
            hits,
            search_work,
            ..
        } = self.score_query(query)?;
        let mut bundles = compose_evidence(hits, &query_terms, &query_weights, total_query_weight);
        bundles.sort_by(|left, right| {
            right
                .bundle
                .score
                .total_cmp(&left.bundle.score)
                .then_with(|| left.bundle.source.cmp(&right.bundle.source))
        });
        bundles.truncate(limit);
        let status = composed_status(&query_terms, &bundles);
        if status == RecallStatus::NoEvidence {
            bundles.clear();
        }
        Ok(EvidenceResult {
            status,
            query_terms,
            search_work,
            bundles: bundles
                .into_iter()
                .map(|candidate| candidate.bundle)
                .collect(),
        })
    }

    pub fn investigate(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<InvestigationResult, KnowledgeError> {
        if !(1..=100).contains(&limit) {
            return Err(KnowledgeError::InvalidLimit);
        }
        let ScoredQuery {
            query_terms,
            query_weights,
            total_query_weight,
            hits,
            search_work,
            groundable_indices,
        } = self.score_query(query)?;
        let anchor_terms = query_anchor_terms(query);
        let supported_terms = hits
            .iter()
            .filter(|hit| !hit.query_example)
            .flat_map(|hit| hit.matched_indices.iter().copied())
            .filter(|index| groundable_indices.contains(index))
            .collect::<BTreeSet<_>>();
        let unsupported_terms = query_terms
            .iter()
            .enumerate()
            .filter(|(index, _)| !supported_terms.contains(index))
            .map(|(_, term)| term.clone())
            .collect::<Vec<_>>();
        if !unsupported_terms.is_empty() {
            return Ok(InvestigationResult {
                status: RecallStatus::NoEvidence,
                query_terms,
                anchor_terms: anchor_terms.into_iter().collect(),
                unsupported_terms,
                search_work,
                paths: Vec::new(),
            });
        }
        let mut paths = compose_evidence_paths(
            hits,
            &query_terms,
            &query_weights,
            total_query_weight,
            &self.search_index,
            &anchor_terms,
        );
        paths.retain(|path| path_is_grounded(&query_terms, path));
        paths.sort_by(|left, right| {
            right
                .path
                .score
                .total_cmp(&left.path.score)
                .then_with(|| evidence_path_key(&left.path).cmp(&evidence_path_key(&right.path)))
        });
        let mut seen_source_sets = BTreeSet::new();
        paths.retain(|path| seen_source_sets.insert(evidence_source_key(&path.path)));
        paths.truncate(limit);
        let status = if paths.is_empty() {
            RecallStatus::NoEvidence
        } else {
            RecallStatus::Found
        };
        Ok(InvestigationResult {
            status,
            query_terms,
            anchor_terms: anchor_terms.into_iter().collect(),
            unsupported_terms,
            search_work,
            paths: paths.into_iter().map(|candidate| candidate.path).collect(),
        })
    }

    fn score_query(&self, query: &str) -> Result<ScoredQuery, KnowledgeError> {
        let mut seen_query_terms = BTreeSet::new();
        let query_terms = tokenize(query)
            .into_iter()
            .filter(|term| seen_query_terms.insert(term.clone()))
            .collect::<Vec<_>>();
        if query_terms.is_empty() {
            return Err(KnowledgeError::EmptyQuery);
        }

        if self.search_index.documents.is_empty() {
            return Ok(ScoredQuery {
                query_terms,
                query_weights: Vec::new(),
                total_query_weight: 0.0,
                search_work: SearchWork::default(),
                groundable_indices: BTreeSet::new(),
                hits: Vec::new(),
            });
        }

        let expanded = expand_query(&query_terms, &self.search_index);
        let groundable_indices = expanded
            .iter()
            .filter(|term| {
                term.exact
                    || bounded_edit_distance(&query_terms[term.query_index], &term.token, 1)
                        .is_some()
            })
            .map(|term| term.query_index)
            .collect();
        let (candidate_documents, search_work) =
            candidate_document_indices(&expanded, &self.search_index);
        let average_length = self.search_index.average_length;
        let document_count = self.search_index.documents.len() as f64;
        let query_weights = query_terms
            .iter()
            .map(|term| {
                inverse_document_frequency(
                    document_count,
                    self.search_index
                        .document_frequency
                        .get(term)
                        .copied()
                        .unwrap_or(0) as f64,
                )
            })
            .collect::<Vec<_>>();
        let total_query_weight = query_weights.iter().sum::<f64>();
        let numeric_intent = query_numeric_intent(query);
        let feedback_factors = feedback_factors(&query_terms, &self.feedback);
        let mut hits = Vec::<ScoredHit>::new();

        for document_index in candidate_documents {
            let document = &self.search_index.documents[document_index];
            let source = &document.source;
            let chunk = &self
                .sources
                .get(source)
                .expect("indexed source exists")
                .chunks[document.chunk_index];
            let tokens = &document.tokens;
            let mut counts = HashMap::<&str, usize>::new();
            for token in tokens {
                *counts.entry(token.as_str()).or_default() += 1;
            }
            let mut score = 0.0;
            let mut matched_indices = BTreeSet::new();
            let mut exact_indices = BTreeSet::new();
            for term in &expanded {
                let Some(&frequency) = counts.get(term.token.as_str()) else {
                    continue;
                };
                let df = *self
                    .search_index
                    .document_frequency
                    .get(&term.token)
                    .expect("expanded terms come from the document vocabulary")
                    as f64;
                let inverse_document_frequency = inverse_document_frequency(document_count, df);
                let normalized_length = if average_length == 0.0 {
                    1.0
                } else {
                    tokens.len() as f64 / average_length
                };
                let frequency = frequency as f64;
                let saturation = frequency * (BM25_K1 + 1.0)
                    / (frequency + BM25_K1 * (1.0 - BM25_B + BM25_B * normalized_length));
                score += term.weight * inverse_document_frequency * saturation;
                matched_indices.insert(term.query_index);
                if term.exact {
                    exact_indices.insert(term.query_index);
                }
            }
            if score == 0.0 {
                continue;
            }
            if query_terms.len() >= 2 && contains_sequence(tokens, &query_terms) {
                score += 1.0;
            }
            let source_tokens = self
                .search_index
                .source_tokens
                .get(source)
                .expect("indexed source tokens exist");
            let path_score = query_terms
                .iter()
                .filter(|term| source_tokens.contains(term.as_str()))
                .map(|term| {
                    let df = self
                        .search_index
                        .document_frequency
                        .get(term)
                        .copied()
                        .unwrap_or(1) as f64;
                    inverse_document_frequency(document_count, df)
                })
                .sum::<f64>();
            score += path_score * 1.5;
            let coverage = matched_indices.len() as f64 / query_terms.len() as f64;
            score *= 0.75 + 0.5 * coverage;
            let matches_numeric_intent = match numeric_intent {
                Some(NumericIntent::Count) => document.has_numeric_answer,
                Some(NumericIntent::Size) => document.has_size_answer,
                None => false,
            };
            if matches_numeric_intent {
                score *= 1.6;
            }
            let feedback_factor = feedback_factors.get(source).copied().unwrap_or(1.0);
            score *= feedback_factor;
            let matched_terms = matched_indices
                .iter()
                .map(|index| query_terms[*index].clone())
                .collect();
            hits.push(ScoredHit {
                document_index,
                query_example: is_query_example(&chunk.text),
                hit: RecallHit {
                    source: PathBuf::from(source),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    score,
                    feedback_factor,
                    matched_terms,
                    text: chunk.text.clone(),
                },
                matched_indices,
                exact_indices,
            });
        }
        Ok(ScoredQuery {
            query_terms,
            query_weights,
            total_query_weight,
            search_work,
            groundable_indices,
            hits,
        })
    }

    fn persist(&self) -> Result<(), KnowledgeError> {
        validate_sources(&self.sources, self.limits)?;
        validate_feedback(&self.feedback, &self.sources, self.limits)?;
        let bytes = encode_memory(&self.sources, &self.feedback)?;
        if bytes.len() as u64 > self.limits.max_index_bytes {
            return Err(KnowledgeError::CapacityExceeded("index byte count"));
        }
        let temporary = sidecar_path(
            &self.path,
            &format!(
                ".temporary-{}-{}",
                std::process::id(),
                TEMPORARY_ID.fetch_add(1, Ordering::Relaxed)
            ),
        );
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| io_error("create temporary memory", &temporary, source))?;
        if let Err(error) = file
            .write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|source| io_error("write temporary memory", &temporary, source))
        {
            drop(file);
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        drop(file);
        publish_replacement(&temporary, &self.path)
    }
}

#[derive(Clone, Debug)]
struct ExpandedTerm {
    token: String,
    query_index: usize,
    weight: f64,
    exact: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct ScoredHit {
    document_index: usize,
    hit: RecallHit,
    matched_indices: BTreeSet<usize>,
    exact_indices: BTreeSet<usize>,
    query_example: bool,
}

struct ScoredQuery {
    query_terms: Vec<String>,
    query_weights: Vec<f64>,
    total_query_weight: f64,
    search_work: SearchWork,
    groundable_indices: BTreeSet<usize>,
    hits: Vec<ScoredHit>,
}

fn candidate_document_indices(
    expanded: &[ExpandedTerm],
    search_index: &SearchIndex,
) -> (BTreeSet<usize>, SearchWork) {
    let mut documents = BTreeSet::new();
    let mut posting_visits = 0;
    for term in expanded {
        if let Some(postings) = search_index.postings.get(&term.token) {
            posting_visits += postings.len();
            documents.extend(postings.iter().copied());
        }
    }
    let search_work = SearchWork {
        indexed_documents: search_index.documents.len(),
        scored_documents: documents.len(),
        posting_visits,
    };
    (documents, search_work)
}

struct ScoredBundle {
    bundle: EvidenceBundle,
    matched_indices: BTreeSet<usize>,
    exact_indices: BTreeSet<usize>,
}

struct ScoredPath {
    path: EvidencePath,
    matched_indices: BTreeSet<usize>,
    exact_indices: BTreeSet<usize>,
}

#[derive(Default)]
struct SourceQueryMatches {
    matched_indices: BTreeSet<usize>,
    exact_indices: BTreeSet<usize>,
}

fn sort_scored_hits(hits: &mut [ScoredHit]) {
    hits.sort_by(|left, right| {
        right
            .hit
            .score
            .total_cmp(&left.hit.score)
            .then_with(|| left.hit.source.cmp(&right.hit.source))
            .then_with(|| left.hit.start_line.cmp(&right.hit.start_line))
    });
}

fn recall_status(
    query_terms: &[String],
    query_weights: &[f64],
    total_query_weight: f64,
    hits: &[ScoredHit],
) -> RecallStatus {
    match hits.first() {
        None => RecallStatus::NoEvidence,
        Some(best)
            if query_terms.len() >= 3
                && (best.matched_indices.len() < 2 || best.exact_indices.len() < 2) =>
        {
            RecallStatus::NoEvidence
        }
        Some(best) if query_terms.len() == 2 && best.exact_indices.is_empty() => {
            RecallStatus::NoEvidence
        }
        Some(best) if query_terms.len() == 1 && best.exact_indices.len() == 1 => {
            RecallStatus::Found
        }
        Some(best)
            if best.matched_indices.len() >= 2
                && weighted_coverage(&best.matched_indices, query_weights, total_query_weight)
                    >= MIN_WEIGHTED_QUERY_COVERAGE =>
        {
            RecallStatus::Found
        }
        Some(_) if query_terms.len() >= 3 => RecallStatus::NoEvidence,
        Some(_) => RecallStatus::WeakEvidence,
    }
}

fn compose_evidence(
    hits: Vec<ScoredHit>,
    query_terms: &[String],
    query_weights: &[f64],
    total_query_weight: f64,
) -> Vec<ScoredBundle> {
    let mut by_source = BTreeMap::<PathBuf, Vec<ScoredHit>>::new();
    for hit in hits {
        by_source
            .entry(hit.hit.source.clone())
            .or_default()
            .push(hit);
    }

    let mut bundles = Vec::with_capacity(by_source.len());
    for (source, mut candidates) in by_source {
        sort_scored_hits(&mut candidates);
        let mut selected = Vec::new();
        let mut matched_indices = BTreeSet::new();
        let mut exact_indices = BTreeSet::new();
        while selected.len() < MAX_EVIDENCE_PER_BUNDLE && !candidates.is_empty() {
            let choice = candidates
                .iter()
                .enumerate()
                .map(|(index, candidate)| {
                    let new_matched = candidate
                        .matched_indices
                        .difference(&matched_indices)
                        .map(|query_index| query_weights[*query_index])
                        .sum::<f64>();
                    let new_exact = candidate
                        .exact_indices
                        .difference(&exact_indices)
                        .map(|query_index| query_weights[*query_index])
                        .sum::<f64>();
                    let utility =
                        (new_matched + 0.75 * new_exact) * 10.0 + candidate.hit.score.ln_1p();
                    (index, new_matched, utility)
                })
                .max_by(|left, right| {
                    left.2
                        .total_cmp(&right.2)
                        .then_with(|| right.0.cmp(&left.0))
                });
            let Some((index, new_matched, _)) = choice else {
                break;
            };
            if !selected.is_empty() && new_matched == 0.0 {
                break;
            }
            let candidate = candidates.remove(index);
            matched_indices.extend(candidate.matched_indices.iter().copied());
            exact_indices.extend(candidate.exact_indices.iter().copied());
            selected.push(candidate.hit);
        }
        if selected.is_empty() {
            continue;
        }

        let coverage = weighted_coverage(&matched_indices, query_weights, total_query_weight);
        let exact_coverage = weighted_coverage(&exact_indices, query_weights, total_query_weight);
        let evidence_score = selected
            .iter()
            .enumerate()
            .map(|(index, hit)| hit.score * if index == 0 { 1.0 } else { 0.3 })
            .sum::<f64>();
        let covered_terms = query_terms
            .iter()
            .enumerate()
            .filter(|(index, _)| matched_indices.contains(index))
            .map(|(_, term)| term.clone())
            .collect();
        let missing_terms = query_terms
            .iter()
            .enumerate()
            .filter(|(index, _)| !matched_indices.contains(index))
            .map(|(_, term)| term.clone())
            .collect();
        bundles.push(ScoredBundle {
            bundle: EvidenceBundle {
                source,
                score: evidence_score + coverage * 10.0 + exact_coverage * 6.0,
                weighted_coverage: coverage,
                exact_coverage,
                covered_terms,
                missing_terms,
                evidence: selected,
            },
            matched_indices,
            exact_indices,
        });
    }
    bundles
}

fn composed_status(query_terms: &[String], bundles: &[ScoredBundle]) -> RecallStatus {
    let Some(best) = bundles.first() else {
        return RecallStatus::NoEvidence;
    };
    match query_terms.len() {
        0 => RecallStatus::NoEvidence,
        1 if best.exact_indices.len() == 1 => RecallStatus::Found,
        1 => RecallStatus::WeakEvidence,
        2 if !best.exact_indices.is_empty() && best.matched_indices.len() == 2 => {
            RecallStatus::Found
        }
        2 => RecallStatus::WeakEvidence,
        _ if best.exact_indices.len() < 2
            || best.matched_indices.len() < 2
            || best.bundle.weighted_coverage < MIN_COMPOSED_QUERY_COVERAGE
            || best.bundle.exact_coverage < MIN_COMPOSED_EXACT_COVERAGE =>
        {
            RecallStatus::NoEvidence
        }
        _ => RecallStatus::Found,
    }
}

fn weighted_coverage(
    indices: &BTreeSet<usize>,
    query_weights: &[f64],
    total_query_weight: f64,
) -> f64 {
    if total_query_weight == 0.0 {
        return 0.0;
    }
    indices
        .iter()
        .map(|index| query_weights[*index])
        .sum::<f64>()
        / total_query_weight
}

#[derive(Clone)]
struct CandidateLink {
    from_evidence: usize,
    kind: EvidenceLinkKind,
    score: f64,
    bridge_terms: Vec<String>,
}

#[derive(Clone)]
struct GraphEdge {
    kind: EvidenceLinkKind,
    score: f64,
    bridge_terms: Vec<String>,
}

#[derive(Clone)]
struct PathExpansion {
    candidate_index: usize,
    link: CandidateLink,
    utility: f64,
}

fn aggregate_source_query_matches(hits: &[ScoredHit]) -> BTreeMap<PathBuf, SourceQueryMatches> {
    let mut sources = BTreeMap::<PathBuf, SourceQueryMatches>::new();
    for hit in hits {
        let source = sources.entry(hit.hit.source.clone()).or_default();
        source
            .matched_indices
            .extend(hit.matched_indices.iter().copied());
        source
            .exact_indices
            .extend(hit.exact_indices.iter().copied());
    }
    sources
}

fn compose_evidence_paths(
    mut hits: Vec<ScoredHit>,
    query_terms: &[String],
    query_weights: &[f64],
    total_query_weight: f64,
    search_index: &SearchIndex,
    anchor_terms: &BTreeSet<String>,
) -> Vec<ScoredPath> {
    hits.retain(|hit| !hit.query_example);
    let source_query_matches = aggregate_source_query_matches(&hits);
    sort_scored_hits(&mut hits);
    hits.truncate(MAX_GRAPH_CANDIDATES);
    let edge_cache = build_evidence_edges(&hits, search_index, query_terms, anchor_terms);
    let seed_count = hits.len().min(MAX_GRAPH_SEEDS);
    let mut unique_paths = BTreeMap::<Vec<usize>, ScoredPath>::new();

    for (seed_index, prioritize_roles) in
        (0..seed_count).flat_map(|seed_index| [(seed_index, false), (seed_index, true)])
    {
        let mut selected = vec![seed_index];
        let mut matched_indices = hits[seed_index].matched_indices.clone();
        let mut exact_indices = hits[seed_index].exact_indices.clone();
        let mut links = Vec::new();

        while selected.len() < MAX_EVIDENCE_PER_PATH {
            let mut best_expansion: Option<PathExpansion> = None;
            let mut best_adds_role = false;
            for candidate_index in 0..hits.len() {
                if selected.contains(&candidate_index) {
                    continue;
                }
                let candidate = &hits[candidate_index];
                if selected
                    .iter()
                    .filter(|index| hits[**index].hit.source == candidate.hit.source)
                    .count()
                    >= MAX_EVIDENCE_PER_SOURCE_PATH
                {
                    continue;
                }
                let new_matched = candidate
                    .matched_indices
                    .difference(&matched_indices)
                    .map(|query_index| query_weights[*query_index])
                    .sum::<f64>();
                let new_exact = candidate
                    .exact_indices
                    .difference(&exact_indices)
                    .map(|query_index| query_weights[*query_index])
                    .sum::<f64>();
                let Some(link) = strongest_candidate_link(&selected, candidate_index, &edge_cache)
                else {
                    continue;
                };
                let introduces_source = selected
                    .iter()
                    .all(|selected_index| hits[*selected_index].hit.source != candidate.hit.source);
                if introduces_source
                    && selected
                        .iter()
                        .map(|index| &hits[*index].hit.source)
                        .collect::<BTreeSet<_>>()
                        .len()
                        >= MAX_SOURCES_PER_PATH
                {
                    continue;
                }
                let introduces_source_role =
                    introduces_new_source_role(&selected, candidate_index, &hits);
                let linked_by_explicit_anchor = link
                    .bridge_terms
                    .iter()
                    .any(|term| anchor_terms.contains(term));
                let corroborates = new_matched == 0.0
                    && new_exact == 0.0
                    && introduces_source
                    && introduces_source_role
                    && (linked_by_explicit_anchor
                        || (link.bridge_terms.len() >= 2 && link.score >= 0.70));
                if new_matched == 0.0 && new_exact == 0.0 && !corroborates {
                    continue;
                }
                let utility = (new_matched + 0.75 * new_exact) * 10.0
                    + graph_hit_score(candidate)
                    + definition_bonus(&candidate.hit.source, search_index, anchor_terms)
                    + link.score * 4.0
                    + if introduces_source { 2.0 } else { 0.0 }
                    + if introduces_source_role { 6.0 } else { 0.0 }
                    + if corroborates { 0.5 } else { 0.0 };
                let expansion = PathExpansion {
                    candidate_index,
                    link,
                    utility,
                };
                let replace = best_expansion.as_ref().is_none_or(|current| {
                    (prioritize_roles && introduces_source_role)
                        .cmp(&(prioritize_roles && best_adds_role))
                        .then_with(|| expansion.utility.total_cmp(&current.utility))
                        .then_with(|| {
                            hits[current.candidate_index]
                                .hit
                                .source
                                .cmp(&hits[expansion.candidate_index].hit.source)
                        })
                        .then_with(|| current.candidate_index.cmp(&expansion.candidate_index))
                        .is_gt()
                });
                if replace {
                    best_adds_role = introduces_source_role;
                    best_expansion = Some(expansion);
                }
            }
            let Some(expansion) = best_expansion else {
                break;
            };
            let candidate = &hits[expansion.candidate_index];
            let to_evidence = selected.len();
            links.push(EvidenceLink {
                from_evidence: expansion.link.from_evidence,
                to_evidence,
                kind: expansion.link.kind,
                score: expansion.link.score,
                bridge_terms: expansion.link.bridge_terms,
            });
            matched_indices.extend(candidate.matched_indices.iter().copied());
            exact_indices.extend(candidate.exact_indices.iter().copied());
            selected.push(expansion.candidate_index);
        }

        let mut path = build_scored_path(
            &selected,
            links,
            &hits,
            query_terms,
            query_weights,
            total_query_weight,
        );
        enrich_path_metrics(
            &mut path.path,
            search_index,
            query_terms,
            query_weights,
            total_query_weight,
            anchor_terms,
            &source_query_matches,
        );
        if path_is_grounded(query_terms, &path) {
            insert_unique_path(&mut unique_paths, &selected, &hits, path);
        }
    }

    for left in 0..hits.len() {
        for right in left + 1..hits.len() {
            let Some(edge) = edge_cache[left][right].as_ref() else {
                continue;
            };
            let selected = [left, right];
            let links = vec![EvidenceLink {
                from_evidence: 0,
                to_evidence: 1,
                kind: edge.kind,
                score: edge.score,
                bridge_terms: edge.bridge_terms.clone(),
            }];
            let mut path = build_scored_path(
                &selected,
                links,
                &hits,
                query_terms,
                query_weights,
                total_query_weight,
            );
            enrich_path_metrics(
                &mut path.path,
                search_index,
                query_terms,
                query_weights,
                total_query_weight,
                anchor_terms,
                &source_query_matches,
            );
            if path_is_grounded(query_terms, &path) {
                insert_unique_path(&mut unique_paths, &selected, &hits, path);
            }

            let matched_indices = selected
                .iter()
                .flat_map(|index| hits[*index].matched_indices.iter().copied())
                .collect::<BTreeSet<_>>();
            let exact_indices = selected
                .iter()
                .flat_map(|index| hits[*index].exact_indices.iter().copied())
                .collect::<BTreeSet<_>>();
            let mut best_expansion: Option<PathExpansion> = None;
            let mut best_internal_expansion: Option<PathExpansion> = None;
            let mut best_role_expansions = BTreeMap::<EvidenceSourceRole, PathExpansion>::new();
            for candidate_index in 0..hits.len() {
                if selected.contains(&candidate_index) {
                    continue;
                }
                let candidate = &hits[candidate_index];
                let new_matched = candidate
                    .matched_indices
                    .difference(&matched_indices)
                    .map(|query_index| query_weights[*query_index])
                    .sum::<f64>();
                let new_exact = candidate
                    .exact_indices
                    .difference(&exact_indices)
                    .map(|query_index| query_weights[*query_index])
                    .sum::<f64>();
                let Some(link) = strongest_candidate_link(&selected, candidate_index, &edge_cache)
                else {
                    continue;
                };
                let introduces_source = selected
                    .iter()
                    .all(|selected_index| hits[*selected_index].hit.source != candidate.hit.source);
                let introduces_source_role =
                    introduces_new_source_role(&selected, candidate_index, &hits);
                let linked_by_explicit_anchor = link
                    .bridge_terms
                    .iter()
                    .any(|term| anchor_terms.contains(term));
                let corroborates = new_matched == 0.0
                    && new_exact == 0.0
                    && introduces_source
                    && (link.kind == EvidenceLinkKind::SourceEntity
                        || linked_by_explicit_anchor
                        || (link.bridge_terms.len() >= 2 && link.score >= 0.70));
                if new_matched == 0.0 && new_exact == 0.0 && !corroborates {
                    continue;
                }
                let utility = (new_matched + 0.75 * new_exact) * 10.0
                    + graph_hit_score(candidate)
                    + definition_bonus(&candidate.hit.source, search_index, anchor_terms)
                    + link.score * 4.0
                    + if introduces_source { 2.0 } else { 0.0 }
                    + if introduces_source_role { 6.0 } else { 0.0 }
                    + if corroborates { 0.5 } else { 0.0 };
                let expansion = PathExpansion {
                    candidate_index,
                    link,
                    utility,
                };
                let stays_within_sources = selected
                    .iter()
                    .any(|selected_index| hits[*selected_index].hit.source == candidate.hit.source);
                if stays_within_sources
                    && best_internal_expansion
                        .as_ref()
                        .is_none_or(|current| expansion.utility > current.utility)
                {
                    best_internal_expansion = Some(expansion.clone());
                }
                if introduces_source_role {
                    let role = evidence_source_role(&candidate.hit.source);
                    if matches!(
                        role,
                        EvidenceSourceRole::Documentation
                            | EvidenceSourceRole::Implementation
                            | EvidenceSourceRole::Test
                    ) && best_role_expansions
                        .get(&role)
                        .is_none_or(|current| expansion.utility > current.utility)
                    {
                        best_role_expansions.insert(role, expansion.clone());
                    }
                }
                if best_expansion
                    .as_ref()
                    .is_none_or(|current| expansion.utility > current.utility)
                {
                    best_expansion = Some(expansion);
                }
            }
            let mut expansions = Vec::new();
            if let Some(expansion) = best_expansion {
                expansions.push(expansion);
            }
            if let Some(expansion) = best_internal_expansion
                && expansions
                    .iter()
                    .all(|current| current.candidate_index != expansion.candidate_index)
            {
                expansions.push(expansion);
            }
            for expansion in best_role_expansions.into_values() {
                if expansions
                    .iter()
                    .all(|current| current.candidate_index != expansion.candidate_index)
                {
                    expansions.push(expansion);
                }
            }
            for expansion in expansions {
                let triple_selected = [left, right, expansion.candidate_index];
                let triple_links = vec![
                    EvidenceLink {
                        from_evidence: 0,
                        to_evidence: 1,
                        kind: edge.kind,
                        score: edge.score,
                        bridge_terms: edge.bridge_terms.clone(),
                    },
                    EvidenceLink {
                        from_evidence: expansion.link.from_evidence,
                        to_evidence: 2,
                        kind: expansion.link.kind,
                        score: expansion.link.score,
                        bridge_terms: expansion.link.bridge_terms,
                    },
                ];
                let (triple_selected, triple_links) = extend_role_path(
                    triple_selected.to_vec(),
                    triple_links,
                    &hits,
                    query_weights,
                    &edge_cache,
                );
                let mut triple = build_scored_path(
                    &triple_selected,
                    triple_links,
                    &hits,
                    query_terms,
                    query_weights,
                    total_query_weight,
                );
                enrich_path_metrics(
                    &mut triple.path,
                    search_index,
                    query_terms,
                    query_weights,
                    total_query_weight,
                    anchor_terms,
                    &source_query_matches,
                );
                if path_is_grounded(query_terms, &triple) {
                    insert_unique_path(&mut unique_paths, &triple_selected, &hits, triple);
                }
            }
        }
    }

    unique_paths.into_values().collect()
}

fn definition_bonus(
    source: &Path,
    search_index: &SearchIndex,
    anchor_terms: &BTreeSet<String>,
) -> f64 {
    search_index
        .source_definitions
        .get(source.to_string_lossy().as_ref())
        .map_or(0.0, |definitions| {
            if anchor_terms
                .iter()
                .any(|anchor| definitions.contains(anchor))
            {
                6.0
            } else {
                0.0
            }
        })
}

fn extend_role_path(
    mut selected: Vec<usize>,
    mut links: Vec<EvidenceLink>,
    hits: &[ScoredHit],
    query_weights: &[f64],
    edge_cache: &[Vec<Option<GraphEdge>>],
) -> (Vec<usize>, Vec<EvidenceLink>) {
    let roles = selected
        .iter()
        .map(|index| evidence_source_role(&hits[*index].hit.source))
        .collect::<BTreeSet<_>>();
    if roles
        != BTreeSet::from([
            EvidenceSourceRole::Documentation,
            EvidenceSourceRole::Implementation,
            EvidenceSourceRole::Test,
        ])
    {
        return (selected, links);
    }
    let sources = selected
        .iter()
        .map(|index| &hits[*index].hit.source)
        .collect::<BTreeSet<_>>();
    let mut matched = selected
        .iter()
        .flat_map(|index| hits[*index].matched_indices.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut exact = selected
        .iter()
        .flat_map(|index| hits[*index].exact_indices.iter().copied())
        .collect::<BTreeSet<_>>();
    while selected.len() < MAX_EVIDENCE_PER_PATH {
        let mut best: Option<PathExpansion> = None;
        for (candidate_index, candidate) in hits.iter().enumerate() {
            if selected.contains(&candidate_index)
                || !sources.contains(&candidate.hit.source)
                || selected
                    .iter()
                    .filter(|index| hits[**index].hit.source == candidate.hit.source)
                    .count()
                    >= MAX_EVIDENCE_PER_SOURCE_PATH
            {
                continue;
            }
            let gain = candidate
                .matched_indices
                .difference(&matched)
                .map(|index| query_weights[*index])
                .sum::<f64>()
                + candidate
                    .exact_indices
                    .difference(&exact)
                    .map(|index| query_weights[*index] * 0.75)
                    .sum::<f64>();
            if gain == 0.0 {
                continue;
            }
            let Some(link) = strongest_candidate_link(&selected, candidate_index, edge_cache)
            else {
                continue;
            };
            let utility = gain * 10.0 + graph_hit_score(candidate) + link.score * 4.0;
            if best
                .as_ref()
                .is_none_or(|current| utility > current.utility)
            {
                best = Some(PathExpansion {
                    candidate_index,
                    link,
                    utility,
                });
            }
        }
        let Some(expansion) = best else {
            break;
        };
        links.push(EvidenceLink {
            from_evidence: expansion.link.from_evidence,
            to_evidence: selected.len(),
            kind: expansion.link.kind,
            score: expansion.link.score,
            bridge_terms: expansion.link.bridge_terms,
        });
        matched.extend(
            hits[expansion.candidate_index]
                .matched_indices
                .iter()
                .copied(),
        );
        exact.extend(
            hits[expansion.candidate_index]
                .exact_indices
                .iter()
                .copied(),
        );
        selected.push(expansion.candidate_index);
    }
    (selected, links)
}

fn build_scored_path(
    selected: &[usize],
    links: Vec<EvidenceLink>,
    hits: &[ScoredHit],
    query_terms: &[String],
    query_weights: &[f64],
    total_query_weight: f64,
) -> ScoredPath {
    let matched_indices = selected
        .iter()
        .flat_map(|index| hits[*index].matched_indices.iter().copied())
        .collect::<BTreeSet<_>>();
    let exact_indices = selected
        .iter()
        .flat_map(|index| hits[*index].exact_indices.iter().copied())
        .collect::<BTreeSet<_>>();
    let coverage = weighted_coverage(&matched_indices, query_weights, total_query_weight);
    let exact_coverage = weighted_coverage(&exact_indices, query_weights, total_query_weight);
    let coherence = if links.is_empty() {
        0.0
    } else {
        links.iter().map(EvidenceLink::score).sum::<f64>() / links.len() as f64
    };
    let evidence_score = selected
        .iter()
        .enumerate()
        .map(|(position, index)| {
            graph_hit_score(&hits[*index]) * if position == 0 { 2.0 } else { 0.75 }
        })
        .sum::<f64>();
    let query_example_count = selected
        .iter()
        .filter(|index| hits[**index].query_example)
        .count();
    let evidence = selected
        .iter()
        .map(|index| hits[*index].hit.clone())
        .collect::<Vec<_>>();
    let evidence_roles = evidence
        .iter()
        .map(|hit| evidence_source_role(&hit.source))
        .collect::<Vec<_>>();
    let covered_terms = query_terms
        .iter()
        .enumerate()
        .filter(|(index, _)| matched_indices.contains(index))
        .map(|(_, term)| term.clone())
        .collect();
    let missing_terms = query_terms
        .iter()
        .enumerate()
        .filter(|(index, _)| !matched_indices.contains(index))
        .map(|(_, term)| term.clone())
        .collect();
    let distinct_sources = evidence
        .iter()
        .map(|hit| hit.source.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let source_diversity_bonus = match distinct_sources {
        0 | 1 => 0.0,
        2 => 6.0,
        _ => 8.0,
    };
    let source_roles = evidence_roles
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    let provenance_bonus = match source_roles {
        0 | 1 => 0.0,
        2 => 8.0,
        _ => 14.0,
    };
    let hierarchy_bonus = links
        .iter()
        .filter(|link| link.kind == EvidenceLinkKind::SourceEntity)
        .count() as f64
        * 1.5;
    let path_length_penalty = selected.len().saturating_sub(1) as f64 * 2.5;
    ScoredPath {
        path: EvidencePath {
            score: evidence_score
                + coverage * 10.0
                + exact_coverage * 6.0
                + coherence * 4.0
                + source_diversity_bonus
                + provenance_bonus
                + hierarchy_bonus
                - path_length_penalty
                - query_example_count as f64 * 3.0,
            weighted_coverage: coverage,
            exact_coverage,
            coherence,
            source_coverage: 0.0,
            source_union_coverage: 0.0,
            source_union_exact_coverage: 0.0,
            corroborated_coverage: 0.0,
            source_name_affinity: 0.0,
            anchor_source_count: 0,
            anchor_definition_count: 0,
            query_example_count,
            covered_terms,
            missing_terms,
            evidence,
            evidence_roles,
            links,
        },
        matched_indices,
        exact_indices,
    }
}

fn graph_hit_score(hit: &ScoredHit) -> f64 {
    hit.hit.score.ln_1p() * if hit.query_example { 0.2 } else { 1.0 }
}

fn enrich_path_metrics(
    path: &mut EvidencePath,
    search_index: &SearchIndex,
    query_terms: &[String],
    query_weights: &[f64],
    total_query_weight: f64,
    anchor_terms: &BTreeSet<String>,
    source_query_matches: &BTreeMap<PathBuf, SourceQueryMatches>,
) {
    let source_paths = path
        .evidence
        .iter()
        .map(|hit| hit.source.clone())
        .collect::<BTreeSet<_>>();
    let sources = source_paths
        .iter()
        .map(|source| source.to_string_lossy().into_owned())
        .collect::<BTreeSet<_>>();
    let source_coverages = sources
        .iter()
        .filter_map(|source| search_index.source_vocabulary.get(source))
        .map(|vocabulary| {
            let supported = query_weights
                .iter()
                .enumerate()
                .filter(|(index, _)| vocabulary.contains(&query_terms[*index]))
                .map(|(_, weight)| *weight)
                .sum::<f64>();
            if total_query_weight == 0.0 {
                0.0
            } else {
                supported / total_query_weight
            }
        })
        .collect::<Vec<_>>();
    path.source_coverage = if source_coverages.is_empty() {
        0.0
    } else {
        source_coverages.iter().sum::<f64>() / source_coverages.len() as f64
    };

    let mut source_union_matches = BTreeSet::new();
    let mut source_union_exact = BTreeSet::new();
    for source in &source_paths {
        if let Some(matches) = source_query_matches.get(source) {
            source_union_matches.extend(matches.matched_indices.iter().copied());
            source_union_exact.extend(matches.exact_indices.iter().copied());
        }
    }
    path.source_union_coverage =
        weighted_coverage(&source_union_matches, query_weights, total_query_weight);
    path.source_union_exact_coverage =
        weighted_coverage(&source_union_exact, query_weights, total_query_weight);

    if total_query_weight > 0.0 && sources.len() >= 2 {
        let corroborated = query_weights
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                sources
                    .iter()
                    .filter_map(|source| search_index.source_vocabulary.get(source))
                    .filter(|vocabulary| vocabulary.contains(&query_terms[*index]))
                    .take(2)
                    .count()
                    == 2
            })
            .map(|(_, weight)| *weight)
            .sum::<f64>();
        path.corroborated_coverage = corroborated / total_query_weight;
    }
    path.source_name_affinity = sources
        .iter()
        .map(|source| source_name_affinity(Path::new(source), anchor_terms))
        .fold(0.0, f64::max);
    path.anchor_source_count = sources
        .iter()
        .filter_map(|source| search_index.source_vocabulary.get(source))
        .filter(|vocabulary| {
            anchor_terms
                .iter()
                .any(|anchor| vocabulary.contains(anchor))
        })
        .count();
    path.anchor_definition_count = sources
        .iter()
        .filter_map(|source| search_index.source_definitions.get(source))
        .filter(|definitions| {
            anchor_terms
                .iter()
                .any(|anchor| definitions.contains(anchor))
        })
        .count();
    path.score += path.corroborated_coverage * 8.0
        + path.source_union_coverage * 4.0
        + path.source_union_exact_coverage * 2.0
        + path.anchor_source_count.min(3) as f64 * 0.5
        + path.anchor_definition_count.min(3) as f64 * 6.0;
}

fn declared_identifiers(text: &str) -> BTreeSet<String> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let declaration = line.strip_prefix("pub ").unwrap_or(line);
            ["struct ", "enum ", "trait ", "fn ", "const fn ", "impl "]
                .iter()
                .find_map(|prefix| declaration.strip_prefix(prefix))
        })
        .filter_map(|suffix| {
            suffix
                .split(|character: char| !character.is_alphanumeric())
                .next()
        })
        .filter(|name| !name.is_empty())
        .flat_map(tokenize)
        .collect()
}

fn source_name_affinity(source: &Path, anchor_terms: &BTreeSet<String>) -> f64 {
    let Some(stem) = source.file_stem().and_then(|stem| stem.to_str()) else {
        return 0.0;
    };
    let source_terms = tokenize(stem)
        .into_iter()
        .filter(|term| term.chars().count() >= 3)
        .collect::<BTreeSet<_>>();
    anchor_terms
        .iter()
        .map(|anchor| {
            let matched_characters = source_terms
                .iter()
                .filter(|term| anchor.contains(term.as_str()))
                .map(|term| term.chars().count())
                .sum::<usize>();
            matched_characters as f64 / anchor.chars().count().max(1) as f64
        })
        .fold(0.0, f64::max)
        .min(1.0)
}

fn evidence_source_role(source: &Path) -> EvidenceSourceRole {
    let has_component = |expected: &str| {
        source.components().any(|component| {
            component
                .as_os_str()
                .to_str()
                .is_some_and(|value| value.eq_ignore_ascii_case(expected))
        })
    };
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let extension = source
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if extension == "rs"
        && (has_component("tests")
            || stem.ends_with("_test")
            || stem.ends_with("_tests")
            || stem.ends_with("_contract")
            || stem.ends_with("_contracts"))
    {
        EvidenceSourceRole::Test
    } else if has_component("src") && extension == "rs" {
        EvidenceSourceRole::Implementation
    } else if matches!(extension.as_str(), "md" | "txt" | "rst" | "adoc") {
        EvidenceSourceRole::Documentation
    } else if matches!(
        extension.as_str(),
        "csv" | "tsv" | "json" | "jsonl" | "toml" | "yaml" | "yml"
    ) {
        EvidenceSourceRole::Data
    } else {
        EvidenceSourceRole::Other
    }
}

fn is_query_example(text: &str) -> bool {
    let lowercase = text.to_ascii_lowercase();
    [
        "investigate \"",
        "investigate '",
        "recall \"",
        "recall '",
        "ask \"",
        "ask '",
        "example query",
        "query example",
        "consulta de exemplo",
    ]
    .iter()
    .any(|marker| lowercase.contains(marker))
}

fn insert_unique_path(
    unique_paths: &mut BTreeMap<Vec<usize>, ScoredPath>,
    selected: &[usize],
    hits: &[ScoredHit],
    path: ScoredPath,
) {
    let mut key = selected
        .iter()
        .map(|index| hits[*index].document_index)
        .collect::<Vec<_>>();
    key.sort_unstable();
    if unique_paths
        .get(&key)
        .is_none_or(|current| path.path.score > current.path.score)
    {
        unique_paths.insert(key, path);
    }
}

fn strongest_candidate_link(
    selected: &[usize],
    candidate_index: usize,
    edge_cache: &[Vec<Option<GraphEdge>>],
) -> Option<CandidateLink> {
    selected
        .iter()
        .enumerate()
        .filter_map(|(position, selected_index)| {
            edge_cache[*selected_index][candidate_index]
                .as_ref()
                .map(|edge| CandidateLink {
                    from_evidence: position,
                    kind: edge.kind,
                    score: edge.score,
                    bridge_terms: edge.bridge_terms.clone(),
                })
        })
        .max_by(|left, right| {
            left.score
                .total_cmp(&right.score)
                .then_with(|| right.from_evidence.cmp(&left.from_evidence))
        })
}

fn introduces_new_source_role(
    selected: &[usize],
    candidate_index: usize,
    hits: &[ScoredHit],
) -> bool {
    let candidate_role = evidence_source_role(&hits[candidate_index].hit.source);
    selected.iter().all(|selected_index| {
        evidence_source_role(&hits[*selected_index].hit.source) != candidate_role
    })
}

fn build_evidence_edges(
    hits: &[ScoredHit],
    search_index: &SearchIndex,
    query_terms: &[String],
    anchor_terms: &BTreeSet<String>,
) -> Vec<Vec<Option<GraphEdge>>> {
    let mut edges = vec![vec![None; hits.len()]; hits.len()];
    for left in 0..hits.len() {
        for right in left + 1..hits.len() {
            let Some((kind, score, bridge_terms)) = evidence_link(
                &hits[left],
                &hits[right],
                search_index,
                query_terms,
                anchor_terms,
            ) else {
                continue;
            };
            let edge = GraphEdge {
                kind,
                score,
                bridge_terms,
            };
            edges[left][right] = Some(edge.clone());
            edges[right][left] = Some(edge);
        }
    }
    edges
}

fn evidence_link(
    left: &ScoredHit,
    right: &ScoredHit,
    search_index: &SearchIndex,
    query_terms: &[String],
    anchor_terms: &BTreeSet<String>,
) -> Option<(EvidenceLinkKind, f64, Vec<String>)> {
    let left_document = &search_index.documents[left.document_index];
    let right_document = &search_index.documents[right.document_index];
    let adjacent = left_document.source == right_document.source
        && left_document
            .chunk_index
            .abs_diff(right_document.chunk_index)
            == 1;
    let maximum_document_frequency =
        (search_index.documents.len() / BRIDGE_DOCUMENT_DIVISOR).max(2);
    let maximum_phrase_frequency = (search_index.documents.len() / 100).max(2);
    let document_count = search_index.documents.len() as f64;
    let maximum_idf = inverse_document_frequency(document_count, 1.0);
    let mut bridge_terms = left_document
        .unique_tokens
        .intersection(&right_document.unique_tokens)
        .filter_map(|term| {
            let frequency = search_index.document_frequency.get(term).copied()?;
            (term.chars().count() >= 4 && frequency <= maximum_document_frequency).then_some((
                inverse_document_frequency(document_count, frequency as f64),
                term.clone(),
            ))
        })
        .collect::<Vec<_>>();
    bridge_terms.sort_by(|left, right| {
        anchor_terms
            .contains(&right.1)
            .cmp(&anchor_terms.contains(&left.1))
            .then_with(|| {
                query_terms
                    .contains(&right.1)
                    .cmp(&query_terms.contains(&left.1))
            })
            .then_with(|| right.0.total_cmp(&left.0))
            .then_with(|| left.1.cmp(&right.1))
    });
    let eligible_terms = bridge_terms
        .iter()
        .map(|(_, term)| term.as_str())
        .collect::<BTreeSet<_>>();
    let phrase_terms = bridge_terms
        .iter()
        .filter(|(_, term)| {
            search_index
                .document_frequency
                .get(term)
                .is_some_and(|frequency| *frequency <= maximum_phrase_frequency)
        })
        .map(|(_, term)| term.as_str())
        .collect::<BTreeSet<_>>();
    let query_anchored = query_terms
        .iter()
        .any(|term| eligible_terms.contains(term.as_str()));
    let strong_query_anchor = anchor_terms
        .iter()
        .any(|term| eligible_terms.contains(term.as_str()));
    let phrase_bridged =
        shares_bridge_phrase(&left_document.tokens, &right_document.tokens, &phrase_terms);
    bridge_terms.truncate(MAX_BRIDGE_TERMS);
    let bridge_score = bridge_terms
        .iter()
        .enumerate()
        .map(|(index, (idf, _))| idf * if index == 0 { 1.0 } else { 0.25 })
        .sum::<f64>()
        / (maximum_idf * 1.5);
    let terms = bridge_terms
        .into_iter()
        .map(|(_, term)| term)
        .collect::<Vec<_>>();
    if adjacent {
        return Some((
            EvidenceLinkKind::Adjacent,
            (0.60 + bridge_score * 0.40).min(1.0),
            terms,
        ));
    }
    if (query_anchored || phrase_bridged)
        && (left_document.source == right_document.source
            || terms.len() >= 2
            || strong_query_anchor)
        && !terms.is_empty()
    {
        return Some((EvidenceLinkKind::SharedTerms, bridge_score.min(1.0), terms));
    }
    source_entity_link(left_document, right_document, search_index, anchor_terms)
}

fn shares_bridge_phrase(
    left: &[String],
    right: &[String],
    eligible_terms: &BTreeSet<&str>,
) -> bool {
    left.windows(2).any(|left_pair| {
        eligible_terms.contains(left_pair[0].as_str())
            && eligible_terms.contains(left_pair[1].as_str())
            && right.windows(2).any(|right_pair| right_pair == left_pair)
    })
}

fn source_entity_link(
    left: &IndexedDocument,
    right: &IndexedDocument,
    search_index: &SearchIndex,
    anchor_terms: &BTreeSet<String>,
) -> Option<(EvidenceLinkKind, f64, Vec<String>)> {
    if left.source == right.source {
        return None;
    }
    let left_vocabulary = search_index.source_vocabulary.get(&left.source)?;
    let right_vocabulary = search_index.source_vocabulary.get(&right.source)?;
    let source_count = search_index.source_vocabulary.len() as f64;
    let maximum_idf = inverse_document_frequency(source_count, 1.0);
    let maximum_source_frequency = (search_index.source_vocabulary.len() / 4).max(2);
    let mut entities = left_vocabulary
        .intersection(right_vocabulary)
        .filter_map(|term| {
            let frequency = search_index.source_frequency.get(term).copied()?;
            (anchor_terms.contains(term) && frequency <= maximum_source_frequency).then_some((
                inverse_document_frequency(source_count, frequency as f64),
                term.clone(),
            ))
        })
        .collect::<Vec<_>>();
    entities.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    entities.truncate(MAX_BRIDGE_TERMS);
    let strongest = entities.first()?.0 / maximum_idf;
    let terms = entities.into_iter().map(|(_, term)| term).collect();
    Some((
        EvidenceLinkKind::SourceEntity,
        (0.65 + strongest * 0.35).min(1.0),
        terms,
    ))
}

fn path_is_grounded(query_terms: &[String], path: &ScoredPath) -> bool {
    match query_terms.len() {
        0 => false,
        1 => path.exact_indices.len() == 1,
        2 if path.path.evidence.len() == 1 => {
            !path.exact_indices.is_empty()
                && path.matched_indices.len() == 2
                && path.path.weighted_coverage >= MIN_DIRECT_QUERY_COVERAGE
                && path.path.exact_coverage >= MIN_DIRECT_EXACT_COVERAGE
        }
        2 => {
            !path.exact_indices.is_empty()
                && path.matched_indices.len() == 2
                && path.path.links.len() + 1 == path.path.evidence.len()
        }
        _ if path.path.evidence.len() == 1 => {
            path.exact_indices.len() >= 2
                && path.path.weighted_coverage >= MIN_DIRECT_QUERY_COVERAGE
                && path.path.exact_coverage >= MIN_DIRECT_EXACT_COVERAGE
        }
        _ => {
            let connected = path.path.links.len() + 1 == path.path.evidence.len();
            let chunk_grounded = path.exact_indices.len() >= 2
                && path.matched_indices.len() >= 2
                && path.path.weighted_coverage >= MIN_PATH_QUERY_COVERAGE
                && path.path.exact_coverage >= MIN_PATH_EXACT_COVERAGE;
            connected && chunk_grounded
        }
    }
}

fn evidence_path_key(path: &EvidencePath) -> Vec<(PathBuf, u32, u32)> {
    path.evidence
        .iter()
        .map(|hit| (hit.source.clone(), hit.start_line, hit.end_line))
        .collect()
}

fn evidence_source_key(path: &EvidencePath) -> Vec<PathBuf> {
    path.evidence
        .iter()
        .map(|hit| hit.source.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn validate_limits(limits: KnowledgeLimits) -> Result<(), KnowledgeError> {
    if limits.max_index_bytes < HEADER_BYTES as u64
        || limits.max_input_file_bytes == 0
        || limits.max_sources == 0
        || limits.max_chunks == 0
        || limits.max_total_text_bytes == 0
        || limits.max_path_bytes == 0
        || limits.max_feedback_records == 0
        || limits.max_feedback_terms == 0
    {
        return Err(KnowledgeError::InvalidFormat(
            "memory limits must all be positive",
        ));
    }
    Ok(())
}

fn io_error(action: &'static str, path: &Path, source: std::io::Error) -> KnowledgeError {
    KnowledgeError::Io {
        action,
        path: path.to_path_buf(),
        source,
    }
}

fn resolve_store_path(path: &Path) -> Result<PathBuf, KnowledgeError> {
    let absolute = std::path::absolute(path)
        .map_err(|source| io_error("resolve memory path", path, source))?;
    let parent = absolute.parent().ok_or(KnowledgeError::InvalidFormat(
        "memory path has no parent directory",
    ))?;
    fs::create_dir_all(parent)
        .map_err(|source| io_error("create memory directory", parent, source))?;
    let parent = fs::canonicalize(parent)
        .map_err(|source| io_error("resolve memory directory", parent, source))?;
    let file_name = absolute.file_name().ok_or(KnowledgeError::InvalidFormat(
        "memory path has no file name",
    ))?;
    Ok(parent.join(file_name))
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

fn recover_interrupted_publish(path: &Path) -> Result<(), KnowledgeError> {
    let backup = sidecar_path(path, ".previous");
    if !path.exists() && backup.exists() {
        fs::rename(&backup, path)
            .map_err(|source| io_error("recover previous memory", &backup, source))?;
    }
    Ok(())
}

fn publish_replacement(temporary: &Path, destination: &Path) -> Result<(), KnowledgeError> {
    let backup = sidecar_path(destination, ".previous");
    if backup.exists() {
        fs::remove_file(&backup)
            .map_err(|source| io_error("remove stale memory backup", &backup, source))?;
    }
    if destination.exists() {
        fs::rename(destination, &backup)
            .map_err(|source| io_error("stage previous memory", destination, source))?;
    }
    if let Err(source) = fs::rename(temporary, destination) {
        if backup.exists() {
            let _ = fs::rename(&backup, destination);
        }
        let _ = fs::remove_file(temporary);
        return Err(io_error("publish memory", destination, source));
    }
    #[cfg(unix)]
    File::open(destination.parent().expect("resolved memory has a parent"))
        .and_then(|directory| directory.sync_all())
        .map_err(|source| io_error("sync memory directory", destination, source))?;
    if backup.exists() {
        fs::remove_file(&backup)
            .map_err(|source| io_error("remove previous memory", &backup, source))?;
    }
    Ok(())
}

fn read_bounded_file(path: &Path, maximum: u64) -> Result<Vec<u8>, KnowledgeError> {
    let metadata = fs::metadata(path).map_err(|source| io_error("inspect file", path, source))?;
    if metadata.len() > maximum {
        return Err(KnowledgeError::InputTooLarge {
            path: path.to_path_buf(),
            bytes: metadata.len(),
            maximum,
        });
    }
    let capacity = usize::try_from(metadata.len())
        .map_err(|_| KnowledgeError::CapacityExceeded("file byte count"))?;
    let mut bytes = Vec::with_capacity(capacity);
    File::open(path)
        .map(|file| file.take(maximum.saturating_add(1)))
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|source| io_error("read file", path, source))?;
    if bytes.len() as u64 > maximum {
        return Err(KnowledgeError::InputTooLarge {
            path: path.to_path_buf(),
            bytes: bytes.len() as u64,
            maximum,
        });
    }
    Ok(bytes)
}

fn load_memory(path: &Path, limits: KnowledgeLimits) -> Result<StoredMemory, KnowledgeError> {
    let bytes = read_bounded_file(path, limits.max_index_bytes)?;
    decode_memory(&bytes, limits)
}

#[cfg(test)]
fn encode_sources(sources: &BTreeMap<String, SourceRecord>) -> Result<Vec<u8>, KnowledgeError> {
    encode_memory(sources, &BTreeMap::new())
}

fn encode_memory(
    sources: &BTreeMap<String, SourceRecord>,
    feedback: &BTreeMap<FeedbackKey, FeedbackRecord>,
) -> Result<Vec<u8>, KnowledgeError> {
    let mut payload = Vec::new();
    encode_source_records(&mut payload, sources)?;
    write_u32(&mut payload, feedback.len(), "feedback record count")?;
    for (key, record) in feedback {
        write_string(&mut payload, &key.source, "feedback source path")?;
        write_u32(
            &mut payload,
            key.query_terms.len(),
            "feedback query term count",
        )?;
        for term in &key.query_terms {
            write_string(&mut payload, term, "feedback term byte count")?;
        }
        payload.extend_from_slice(&record.relevant.to_le_bytes());
        payload.extend_from_slice(&record.irrelevant.to_le_bytes());
    }
    Ok(finish_encoding(FORMAT_VERSION, payload))
}

fn encode_source_records(
    payload: &mut Vec<u8>,
    sources: &BTreeMap<String, SourceRecord>,
) -> Result<(), KnowledgeError> {
    write_u32(payload, sources.len(), "source count")?;
    for (source, record) in sources {
        write_string(payload, source, "source path")?;
        payload.extend_from_slice(&record.fingerprint.bytes.to_le_bytes());
        payload.extend_from_slice(&record.fingerprint.first.to_le_bytes());
        payload.extend_from_slice(&record.fingerprint.second.to_le_bytes());
        write_u32(payload, record.chunks.len(), "source chunk count")?;
        for chunk in &record.chunks {
            payload.extend_from_slice(&chunk.start_line.to_le_bytes());
            payload.extend_from_slice(&chunk.end_line.to_le_bytes());
            write_string(payload, &chunk.text, "chunk byte count")?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn encode_legacy_sources(
    sources: &BTreeMap<String, SourceRecord>,
) -> Result<Vec<u8>, KnowledgeError> {
    let mut payload = Vec::new();
    encode_source_records(&mut payload, sources)?;
    Ok(finish_encoding(LEGACY_FORMAT_VERSION, payload))
}

fn finish_encoding(version: u32, payload: Vec<u8>) -> Vec<u8> {
    let payload_len = payload.len() as u64;
    let mut encoded = Vec::with_capacity(HEADER_BYTES + payload.len());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&version.to_le_bytes());
    encoded.extend_from_slice(&payload_len.to_le_bytes());
    encoded.extend_from_slice(&(!payload_len).to_le_bytes());
    encoded.extend_from_slice(&checksum(&payload).to_le_bytes());
    encoded.extend_from_slice(&payload);
    encoded
}

fn write_u32(
    output: &mut Vec<u8>,
    value: usize,
    resource: &'static str,
) -> Result<(), KnowledgeError> {
    let value = u32::try_from(value).map_err(|_| KnowledgeError::CapacityExceeded(resource))?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_string(
    output: &mut Vec<u8>,
    value: &str,
    resource: &'static str,
) -> Result<(), KnowledgeError> {
    write_u32(output, value.len(), resource)?;
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

#[cfg(test)]
fn decode_sources(
    encoded: &[u8],
    limits: KnowledgeLimits,
) -> Result<BTreeMap<String, SourceRecord>, KnowledgeError> {
    decode_memory(encoded, limits).map(|memory| memory.sources)
}

fn decode_memory(encoded: &[u8], limits: KnowledgeLimits) -> Result<StoredMemory, KnowledgeError> {
    if encoded.len() < HEADER_BYTES {
        return Err(KnowledgeError::InvalidFormat("truncated header"));
    }
    if &encoded[..8] != MAGIC {
        return Err(KnowledgeError::InvalidFormat("wrong magic"));
    }
    let version = u32::from_le_bytes(encoded[8..12].try_into().expect("fixed header width"));
    if version != LEGACY_FORMAT_VERSION && version != FORMAT_VERSION {
        return Err(KnowledgeError::UnsupportedVersion(version));
    }
    let payload_len = u64::from_le_bytes(encoded[12..20].try_into().expect("fixed header width"));
    let complement = u64::from_le_bytes(encoded[20..28].try_into().expect("fixed header width"));
    if complement != !payload_len {
        return Err(KnowledgeError::InvalidFormat("invalid length complement"));
    }
    if payload_len > limits.max_index_bytes {
        return Err(KnowledgeError::CapacityExceeded("index byte count"));
    }
    let payload_len = usize::try_from(payload_len)
        .map_err(|_| KnowledgeError::CapacityExceeded("index byte count"))?;
    let expected_length = HEADER_BYTES
        .checked_add(payload_len)
        .ok_or(KnowledgeError::CapacityExceeded("index byte count"))?;
    if encoded.len() != expected_length {
        return Err(KnowledgeError::InvalidFormat("file length mismatch"));
    }
    let expected_checksum =
        u64::from_le_bytes(encoded[28..36].try_into().expect("fixed header width"));
    let payload = &encoded[HEADER_BYTES..];
    if checksum(payload) != expected_checksum {
        return Err(KnowledgeError::InvalidFormat("checksum mismatch"));
    }

    let mut decoder = Decoder::new(payload);
    let source_count = decoder.read_u32()? as usize;
    if source_count > limits.max_sources {
        return Err(KnowledgeError::CapacityExceeded("source count"));
    }
    let mut sources = BTreeMap::new();
    let mut previous_source: Option<String> = None;
    let mut total_chunks = 0_usize;
    let mut total_text_bytes = 0_u64;
    for _ in 0..source_count {
        let source = decoder.read_string(limits.max_path_bytes, "source path")?;
        if source.is_empty() {
            return Err(KnowledgeError::InvalidFormat("empty source path"));
        }
        if previous_source
            .as_ref()
            .is_some_and(|previous| previous >= &source)
        {
            return Err(KnowledgeError::InvalidFormat(
                "source records are not strictly ordered",
            ));
        }
        previous_source = Some(source.clone());
        let fingerprint = Fingerprint {
            bytes: decoder.read_u64()?,
            first: decoder.read_u64()?,
            second: decoder.read_u64()?,
        };
        let chunk_count = decoder.read_u32()? as usize;
        total_chunks = total_chunks
            .checked_add(chunk_count)
            .ok_or(KnowledgeError::CapacityExceeded("chunk count"))?;
        if total_chunks > limits.max_chunks {
            return Err(KnowledgeError::CapacityExceeded("chunk count"));
        }
        let mut chunks = Vec::with_capacity(chunk_count);
        let mut previous_range = (0, 0);
        for _ in 0..chunk_count {
            let start_line = decoder.read_u32()?;
            let end_line = decoder.read_u32()?;
            if start_line == 0
                || end_line < start_line
                || previous_range != (0, 0) && (start_line, end_line) < previous_range
            {
                return Err(KnowledgeError::InvalidFormat("invalid chunk line range"));
            }
            let text = decoder.read_string(MAX_STORED_CHUNK_BYTES, "chunk byte count")?;
            if text.trim().is_empty() {
                return Err(KnowledgeError::InvalidFormat("empty chunk"));
            }
            total_text_bytes = total_text_bytes
                .checked_add(text.len() as u64)
                .ok_or(KnowledgeError::CapacityExceeded("stored text byte count"))?;
            if total_text_bytes > limits.max_total_text_bytes {
                return Err(KnowledgeError::CapacityExceeded("stored text byte count"));
            }
            chunks.push(StoredChunk {
                start_line,
                end_line,
                text,
            });
            previous_range = (start_line, end_line);
        }
        sources.insert(
            source,
            SourceRecord {
                fingerprint,
                chunks,
            },
        );
    }
    if version == LEGACY_FORMAT_VERSION {
        if !decoder.is_finished() {
            return Err(KnowledgeError::InvalidFormat("trailing payload bytes"));
        }
        return Ok(StoredMemory {
            sources,
            feedback: BTreeMap::new(),
        });
    }

    let feedback_count = decoder.read_u32()? as usize;
    if feedback_count > limits.max_feedback_records {
        return Err(KnowledgeError::CapacityExceeded("feedback record count"));
    }
    let mut feedback = BTreeMap::new();
    let mut previous_key: Option<FeedbackKey> = None;
    for _ in 0..feedback_count {
        let source = decoder.read_string(limits.max_path_bytes, "feedback source path")?;
        if !sources.contains_key(&source) {
            return Err(KnowledgeError::InvalidFormat(
                "feedback references an unknown source",
            ));
        }
        let term_count = decoder.read_u32()? as usize;
        if term_count == 0 {
            return Err(KnowledgeError::InvalidFormat("feedback query has no terms"));
        }
        if term_count > limits.max_feedback_terms {
            return Err(KnowledgeError::CapacityExceeded(
                "feedback query term count",
            ));
        }
        let mut query_terms = Vec::with_capacity(term_count);
        for _ in 0..term_count {
            let term = decoder.read_string(MAX_FEEDBACK_TERM_BYTES, "feedback term byte count")?;
            if term.is_empty()
                || query_terms
                    .last()
                    .is_some_and(|previous: &String| previous >= &term)
            {
                return Err(KnowledgeError::InvalidFormat(
                    "feedback terms are not strictly ordered",
                ));
            }
            query_terms.push(term);
        }
        let record = FeedbackRecord {
            relevant: decoder.read_u32()?,
            irrelevant: decoder.read_u32()?,
        };
        if record.relevant == 0 && record.irrelevant == 0 {
            return Err(KnowledgeError::InvalidFormat(
                "feedback record has no observations",
            ));
        }
        let key = FeedbackKey {
            source,
            query_terms,
        };
        if previous_key
            .as_ref()
            .is_some_and(|previous| previous >= &key)
        {
            return Err(KnowledgeError::InvalidFormat(
                "feedback records are not strictly ordered",
            ));
        }
        previous_key = Some(key.clone());
        feedback.insert(key, record);
    }
    if !decoder.is_finished() {
        return Err(KnowledgeError::InvalidFormat("trailing payload bytes"));
    }
    validate_feedback(&feedback, &sources, limits)?;
    Ok(StoredMemory { sources, feedback })
}

struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read_exact(&mut self, length: usize) -> Result<&'a [u8], KnowledgeError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(KnowledgeError::InvalidFormat("record length overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(KnowledgeError::InvalidFormat("truncated record"))?;
        self.position = end;
        Ok(value)
    }

    fn read_u32(&mut self) -> Result<u32, KnowledgeError> {
        Ok(u32::from_le_bytes(
            self.read_exact(4)?
                .try_into()
                .expect("decoder returned four bytes"),
        ))
    }

    fn read_u64(&mut self) -> Result<u64, KnowledgeError> {
        Ok(u64::from_le_bytes(
            self.read_exact(8)?
                .try_into()
                .expect("decoder returned eight bytes"),
        ))
    }

    fn read_string(
        &mut self,
        maximum: usize,
        resource: &'static str,
    ) -> Result<String, KnowledgeError> {
        let length = self.read_u32()? as usize;
        if length > maximum {
            return Err(KnowledgeError::CapacityExceeded(resource));
        }
        let bytes = self.read_exact(length)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| KnowledgeError::InvalidFormat("record contains invalid UTF-8"))
    }

    fn is_finished(&self) -> bool {
        self.position == self.bytes.len()
    }
}

fn validate_sources(
    sources: &BTreeMap<String, SourceRecord>,
    limits: KnowledgeLimits,
) -> Result<(), KnowledgeError> {
    if sources.len() > limits.max_sources {
        return Err(KnowledgeError::CapacityExceeded("source count"));
    }
    for source in sources.keys() {
        if source.is_empty() || source.len() > limits.max_path_bytes {
            return Err(KnowledgeError::CapacityExceeded("source path byte count"));
        }
    }
    let (chunks, text_bytes) = source_totals(sources)?;
    if chunks > limits.max_chunks {
        return Err(KnowledgeError::CapacityExceeded("chunk count"));
    }
    if text_bytes > limits.max_total_text_bytes {
        return Err(KnowledgeError::CapacityExceeded("stored text byte count"));
    }
    Ok(())
}

fn validate_feedback(
    feedback: &BTreeMap<FeedbackKey, FeedbackRecord>,
    sources: &BTreeMap<String, SourceRecord>,
    limits: KnowledgeLimits,
) -> Result<(), KnowledgeError> {
    if feedback.len() > limits.max_feedback_records {
        return Err(KnowledgeError::CapacityExceeded("feedback record count"));
    }
    for (key, record) in feedback {
        if !sources.contains_key(&key.source) {
            return Err(KnowledgeError::InvalidFormat(
                "feedback references an unknown source",
            ));
        }
        if key.query_terms.is_empty() {
            return Err(KnowledgeError::InvalidFormat("feedback query has no terms"));
        }
        if key.query_terms.len() > limits.max_feedback_terms {
            return Err(KnowledgeError::CapacityExceeded(
                "feedback query term count",
            ));
        }
        for (index, term) in key.query_terms.iter().enumerate() {
            if term.is_empty() {
                return Err(KnowledgeError::InvalidFormat(
                    "feedback query contains an empty term",
                ));
            }
            if term.len() > MAX_FEEDBACK_TERM_BYTES {
                return Err(KnowledgeError::CapacityExceeded("feedback term byte count"));
            }
            if index > 0 && key.query_terms[index - 1] >= *term {
                return Err(KnowledgeError::InvalidFormat(
                    "feedback terms are not strictly ordered",
                ));
            }
        }
        if record.relevant == 0 && record.irrelevant == 0 {
            return Err(KnowledgeError::InvalidFormat(
                "feedback record has no observations",
            ));
        }
    }
    feedback_observation_count(feedback)?;
    Ok(())
}

fn feedback_observation_count(
    feedback: &BTreeMap<FeedbackKey, FeedbackRecord>,
) -> Result<u64, KnowledgeError> {
    feedback.values().try_fold(0_u64, |total, record| {
        total
            .checked_add(u64::from(record.relevant))
            .and_then(|value| value.checked_add(u64::from(record.irrelevant)))
            .ok_or(KnowledgeError::CapacityExceeded(
                "feedback observation count",
            ))
    })
}

fn feedback_query_terms(query: &str, maximum: usize) -> Result<Vec<String>, KnowledgeError> {
    let mut terms = tokenize(query);
    terms.sort();
    terms.dedup();
    if terms.is_empty() {
        return Err(KnowledgeError::EmptyQuery);
    }
    if terms.len() > maximum {
        return Err(KnowledgeError::CapacityExceeded(
            "feedback query term count",
        ));
    }
    if terms
        .iter()
        .any(|term| term.len() > MAX_FEEDBACK_TERM_BYTES)
    {
        return Err(KnowledgeError::CapacityExceeded("feedback term byte count"));
    }
    Ok(terms)
}

fn feedback_factors(
    query_terms: &[String],
    feedback: &BTreeMap<FeedbackKey, FeedbackRecord>,
) -> BTreeMap<String, f64> {
    let mut query = query_terms.iter().map(String::as_str).collect::<Vec<_>>();
    query.sort_unstable();
    query.dedup();
    let mut log_factors = BTreeMap::<String, f64>::new();
    for (key, record) in feedback {
        let (intersection, union) = sorted_overlap(&query, &key.query_terms);
        let exact_single_term =
            query.len() == 1 && key.query_terms.len() == 1 && query[0] == key.query_terms[0];
        if intersection < 2 && !exact_single_term {
            continue;
        }
        let similarity = intersection as f64 / union as f64;
        let learned_log_odds =
            ((f64::from(record.relevant) + 1.0) / (f64::from(record.irrelevant) + 1.0)).ln();
        *log_factors.entry(key.source.clone()).or_default() += similarity * learned_log_odds;
    }
    log_factors
        .into_iter()
        .map(|(source, value)| (source, value.clamp(-4.0, 4.0).exp()))
        .collect()
}

fn sorted_overlap(query: &[&str], stored: &[String]) -> (usize, usize) {
    let mut query_index = 0;
    let mut stored_index = 0;
    let mut intersection = 0;
    while query_index < query.len() && stored_index < stored.len() {
        match query[query_index].cmp(stored[stored_index].as_str()) {
            std::cmp::Ordering::Less => query_index += 1,
            std::cmp::Ordering::Greater => stored_index += 1,
            std::cmp::Ordering::Equal => {
                intersection += 1;
                query_index += 1;
                stored_index += 1;
            }
        }
    }
    (intersection, query.len() + stored.len() - intersection)
}

fn resolve_source_path(source: &Path) -> Result<PathBuf, KnowledgeError> {
    if source.exists() {
        fs::canonicalize(source).map_err(|error| io_error("resolve source", source, error))
    } else {
        std::path::absolute(source).map_err(|error| io_error("resolve source", source, error))
    }
}

fn source_totals(sources: &BTreeMap<String, SourceRecord>) -> Result<(usize, u64), KnowledgeError> {
    let mut chunk_count = 0_usize;
    let mut text_bytes = 0_u64;
    for record in sources.values() {
        chunk_count = chunk_count
            .checked_add(record.chunks.len())
            .ok_or(KnowledgeError::CapacityExceeded("chunk count"))?;
        for chunk in &record.chunks {
            text_bytes = text_bytes
                .checked_add(chunk.text.len() as u64)
                .ok_or(KnowledgeError::CapacityExceeded("stored text byte count"))?;
        }
    }
    Ok((chunk_count, text_bytes))
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn fingerprint(bytes: &[u8]) -> Fingerprint {
    let first = checksum(bytes);
    let second = bytes
        .iter()
        .enumerate()
        .fold(0x9e3779b97f4a7c15_u64, |hash, (index, byte)| {
            hash.rotate_left(7)
                .wrapping_add(u64::from(*byte))
                .wrapping_add(index as u64 ^ 0xa0761d6478bd642f)
                .wrapping_mul(0xe7037ed1a0b428db)
        })
        ^ INDEXING_VERSION.wrapping_mul(0x9e3779b97f4a7c15);
    Fingerprint {
        bytes: bytes.len() as u64,
        first,
        second,
    }
}

fn collect_text_files(
    directory: &Path,
    files: &mut Vec<PathBuf>,
    skipped: &mut usize,
    maximum: usize,
    depth: usize,
    visited_directories: &mut BTreeSet<PathBuf>,
) -> Result<(), KnowledgeError> {
    if depth > 64 {
        return Err(KnowledgeError::CapacityExceeded("source directory depth"));
    }
    let canonical_directory = fs::canonicalize(directory)
        .map_err(|source| io_error("resolve source directory", directory, source))?;
    if !visited_directories.insert(canonical_directory) {
        *skipped += 1;
        return Ok(());
    }
    let mut entries = fs::read_dir(directory)
        .map_err(|source| io_error("read source directory", directory, source))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| io_error("read source directory", directory, source))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|source| io_error("inspect source entry", &path, source))?;
        if metadata.file_type().is_symlink() {
            *skipped += 1;
            continue;
        }
        if metadata.is_dir() {
            if ignored_directory(&entry.file_name().to_string_lossy()) {
                *skipped += 1;
            } else {
                collect_text_files(
                    &path,
                    files,
                    skipped,
                    maximum,
                    depth + 1,
                    visited_directories,
                )?;
            }
        } else if metadata.is_file() {
            if supported_text_path(&path) {
                if files.len() >= maximum {
                    return Err(KnowledgeError::CapacityExceeded("discovered source count"));
                }
                files.push(
                    fs::canonicalize(&path)
                        .map_err(|source| io_error("resolve source", &path, source))?,
                );
            } else {
                *skipped += 1;
            }
        }
    }
    Ok(())
}

fn ignored_directory(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        ".git" | ".axon-memory" | ".axon-artifacts" | ".ecc" | "target" | "node_modules" | "vendor"
    )
}

fn supported_text_path(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    if matches!(
        extension.as_deref(),
        Some(
            "txt"
                | "md"
                | "rst"
                | "adoc"
                | "rs"
                | "toml"
                | "json"
                | "jsonl"
                | "yaml"
                | "yml"
                | "csv"
                | "tsv"
                | "xml"
                | "html"
                | "css"
                | "js"
                | "jsx"
                | "ts"
                | "tsx"
                | "py"
                | "go"
                | "java"
                | "c"
                | "h"
                | "cpp"
                | "hpp"
                | "sh"
                | "ps1"
                | "sql"
                | "ini"
                | "cfg"
                | "axon"
        )
    ) {
        return true;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(
        name.as_str(),
        "readme" | "license" | "changelog" | "makefile" | "dockerfile"
    )
}

#[derive(Clone, Debug)]
struct TextPiece {
    start_line: u32,
    end_line: u32,
    text: String,
}

fn chunk_text(text: &str) -> Result<Vec<StoredChunk>, KnowledgeError> {
    let mut paragraphs = Vec::<TextPiece>::new();
    let mut current_text = String::new();
    let mut current_start = 0_u32;
    let mut current_end = 0_u32;
    for (index, line) in text.lines().enumerate() {
        let line_number = u32::try_from(index + 1)
            .map_err(|_| KnowledgeError::CapacityExceeded("source line count"))?;
        let trimmed = line.trim_end();
        if trimmed.trim().is_empty() {
            flush_paragraph(
                &mut paragraphs,
                &mut current_text,
                current_start,
                current_end,
            );
            current_start = 0;
            current_end = 0;
            continue;
        }
        if current_text.is_empty() {
            current_start = line_number;
        } else {
            current_text.push('\n');
        }
        current_text.push_str(trimmed);
        current_end = line_number;
    }
    flush_paragraph(
        &mut paragraphs,
        &mut current_text,
        current_start,
        current_end,
    );

    let mut pieces = Vec::new();
    for paragraph in paragraphs {
        split_piece(paragraph, &mut pieces)?;
    }
    let mut chunks = Vec::new();
    let mut current: Option<StoredChunk> = None;
    for piece in pieces {
        let starts_section = is_markdown_heading(&piece.text);
        match current.as_mut() {
            Some(chunk)
                if !starts_section
                    && chunk.text.len() + 2 + piece.text.len() <= MAX_CHUNK_BYTES =>
            {
                chunk.text.push_str("\n\n");
                chunk.text.push_str(&piece.text);
                chunk.end_line = piece.end_line;
            }
            Some(_) => {
                chunks.push(current.take().expect("current chunk exists"));
                current = Some(StoredChunk {
                    start_line: piece.start_line,
                    end_line: piece.end_line,
                    text: piece.text,
                });
            }
            None => {
                current = Some(StoredChunk {
                    start_line: piece.start_line,
                    end_line: piece.end_line,
                    text: piece.text,
                });
            }
        }
    }
    if let Some(chunk) = current {
        chunks.push(chunk);
    }
    Ok(chunks)
}

fn is_markdown_heading(text: &str) -> bool {
    let trimmed = text.trim_start();
    let hashes = trimmed
        .chars()
        .take_while(|character| *character == '#')
        .count();
    (1..=6).contains(&hashes) && trimmed.as_bytes().get(hashes) == Some(&b' ')
}

fn flush_paragraph(
    paragraphs: &mut Vec<TextPiece>,
    text: &mut String,
    start_line: u32,
    end_line: u32,
) {
    let trimmed = text.trim().to_owned();
    text.clear();
    if !trimmed.is_empty() {
        paragraphs.push(TextPiece {
            start_line,
            end_line,
            text: trimmed,
        });
    }
}

fn split_piece(piece: TextPiece, output: &mut Vec<TextPiece>) -> Result<(), KnowledgeError> {
    if piece.text.len() <= MAX_CHUNK_BYTES {
        output.push(piece);
        return Ok(());
    }
    let mut line_number = piece.start_line;
    for line in piece.text.lines() {
        if line.len() <= MAX_CHUNK_BYTES {
            output.push(TextPiece {
                start_line: line_number,
                end_line: line_number,
                text: line.to_owned(),
            });
        } else {
            let mut start = 0;
            while start < line.len() {
                let mut end = (start + MAX_CHUNK_BYTES).min(line.len());
                while !line.is_char_boundary(end) {
                    end -= 1;
                }
                if end < line.len() {
                    let minimum = start + MAX_CHUNK_BYTES / 2;
                    if let Some(boundary) = line[start..end]
                        .char_indices()
                        .rev()
                        .find(|(offset, character)| {
                            start + offset >= minimum
                                && (character.is_whitespace()
                                    || matches!(character, '.' | '!' | '?' | ';' | ','))
                        })
                        .map(|(offset, character)| start + offset + character.len_utf8())
                    {
                        end = boundary;
                    }
                }
                if end == start {
                    return Err(KnowledgeError::InvalidFormat(
                        "cannot split an oversized source line",
                    ));
                }
                let part = line[start..end].trim();
                if !part.is_empty() {
                    output.push(TextPiece {
                        start_line: line_number,
                        end_line: line_number,
                        text: part.to_owned(),
                    });
                }
                start = end;
                while start < line.len()
                    && line[start..]
                        .chars()
                        .next()
                        .is_some_and(char::is_whitespace)
                {
                    start += line[start..]
                        .chars()
                        .next()
                        .expect("checked above")
                        .len_utf8();
                }
            }
        }
        line_number = line_number
            .checked_add(1)
            .ok_or(KnowledgeError::CapacityExceeded("source line count"))?;
    }
    Ok(())
}

fn expand_query(query_terms: &[String], search_index: &SearchIndex) -> Vec<ExpandedTerm> {
    let document_frequency = &search_index.document_frequency;
    let mut expanded = Vec::new();
    for (query_index, query) in query_terms.iter().enumerate() {
        if document_frequency.contains_key(query) {
            expanded.push(ExpandedTerm {
                token: query.clone(),
                query_index,
                weight: 1.0,
                exact: true,
            });
            continue;
        }
        let maximum_distance = match query.chars().count() {
            0..=3 => 0,
            4..=7 => 1,
            _ => 2,
        };
        if maximum_distance == 0 || document_frequency.len() > 100_000 {
            continue;
        }
        let mut candidates = document_frequency
            .iter()
            .filter_map(|(candidate, frequency)| {
                bounded_edit_distance(query, candidate, maximum_distance)
                    .map(|distance| (distance, *frequency, candidate))
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| right.1.cmp(&left.1))
                .then_with(|| left.2.cmp(right.2))
        });
        if let Some(best_distance) = candidates.first().map(|candidate| candidate.0) {
            let weight = if best_distance == 1 { 0.70 } else { 0.45 };
            for (_, _, candidate) in candidates
                .into_iter()
                .take_while(|candidate| candidate.0 == best_distance)
                .take(3)
            {
                expanded.push(ExpandedTerm {
                    token: candidate.clone(),
                    query_index,
                    weight,
                    exact: false,
                });
            }
            continue;
        }

        let query_signature = subword_signature(query);
        if query_signature.is_empty() {
            continue;
        }
        let query_length = query.chars().count();
        let mut subword_candidates = search_index
            .subword_signatures
            .iter()
            .filter_map(|(candidate, signature)| {
                let candidate_length = candidate.chars().count();
                if candidate_length < 4
                    || query_length.abs_diff(candidate_length) > 3.max(query_length / 2)
                {
                    return None;
                }
                let similarity = subword_similarity(&query_signature, signature);
                (similarity >= MIN_SUBWORD_SIMILARITY).then_some((
                    similarity,
                    document_frequency.get(candidate).copied().unwrap_or(0),
                    candidate,
                ))
            })
            .collect::<Vec<_>>();
        subword_candidates.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| right.1.cmp(&left.1))
                .then_with(|| left.2.cmp(right.2))
        });
        let Some(best_similarity) = subword_candidates.first().map(|candidate| candidate.0) else {
            continue;
        };
        for (similarity, _, candidate) in subword_candidates
            .into_iter()
            .take_while(|candidate| best_similarity - candidate.0 <= 0.03)
            .take(3)
        {
            expanded.push(ExpandedTerm {
                token: candidate.clone(),
                query_index,
                weight: 0.25 + 0.45 * similarity,
                exact: false,
            });
        }
    }
    expanded
}

fn inverse_document_frequency(document_count: f64, document_frequency: f64) -> f64 {
    (1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)).ln()
}

fn bounded_edit_distance(left: &str, right: &str, maximum: usize) -> Option<usize> {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    if left.len().abs_diff(right.len()) > maximum {
        return None;
    }
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut previous_previous = previous.clone();
    let mut current = vec![0; right.len() + 1];
    for (left_index, left_character) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_character) in right.iter().enumerate() {
            let substitution =
                previous[right_index] + usize::from(left_character != right_character);
            current[right_index + 1] = (current[right_index] + 1)
                .min(previous[right_index + 1] + 1)
                .min(substitution);
            if left_index > 0
                && right_index > 0
                && left[left_index] == right[right_index - 1]
                && left[left_index - 1] == right[right_index]
            {
                current[right_index + 1] =
                    current[right_index + 1].min(previous_previous[right_index - 1] + 1);
            }
        }
        std::mem::swap(&mut previous_previous, &mut previous);
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[right.len()] <= maximum).then_some(previous[right.len()])
}

fn subword_signature(term: &str) -> Vec<u64> {
    let bounded = std::iter::once('^')
        .chain(term.chars())
        .chain(std::iter::once('$'))
        .collect::<Vec<_>>();
    if bounded.len() < 3 {
        return Vec::new();
    }
    let mut signature = bounded
        .windows(3)
        .map(|window| {
            window
                .iter()
                .fold(0xcbf29ce484222325_u64, |hash, character| {
                    (hash ^ u64::from(u32::from(*character))).wrapping_mul(0x100000001b3)
                })
        })
        .collect::<Vec<_>>();
    signature.sort_unstable();
    signature.dedup();
    signature
}

fn subword_similarity(left: &[u64], right: &[u64]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let mut left_index = 0;
    let mut right_index = 0;
    let mut intersection = 0_usize;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                intersection += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    2.0 * intersection as f64 / (left.len() + right.len()) as f64
}

fn contains_sequence(document: &[String], query: &[String]) -> bool {
    query.len() <= document.len() && document.windows(query.len()).any(|window| window == query)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NumericIntent {
    Count,
    Size,
}

fn query_numeric_intent(query: &str) -> Option<NumericIntent> {
    let words = normalized_words(query);
    if words.iter().any(|word| word == "tamanho") {
        return Some(NumericIntent::Size);
    }
    words
        .iter()
        .any(|word| {
            matches!(
                word.as_str(),
                "quanto"
                    | "quantos"
                    | "quanta"
                    | "quantas"
                    | "numero"
                    | "numeros"
                    | "quantidade"
                    | "quantidades"
            )
        })
        .then_some(NumericIntent::Count)
}

fn contains_numeric_answer(text: &str) -> bool {
    normalized_words(text)
        .iter()
        .any(|word| word.chars().all(|character| character.is_ascii_digit()))
}

fn contains_size_answer(text: &str) -> bool {
    normalized_words(text).iter().any(|word| {
        matches!(
            word.as_str(),
            "byte" | "bytes" | "kb" | "kib" | "mb" | "mib" | "gb" | "gib" | "palavra" | "palavras"
        )
    })
}

fn normalized_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            for lower in character.to_lowercase() {
                current.push(fold_latin(lower));
            }
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            for lower in character.to_lowercase() {
                current.push(fold_latin(lower));
            }
        } else {
            push_token(&mut tokens, &mut current);
        }
    }
    push_token(&mut tokens, &mut current);
    tokens
}

fn query_anchor_terms(query: &str) -> BTreeSet<String> {
    query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty() && is_code_like_word(word))
        .flat_map(tokenize)
        .collect()
}

fn is_code_like_word(word: &str) -> bool {
    let has_letter = word.chars().any(char::is_alphabetic);
    let has_digit = word.chars().any(|character| character.is_ascii_digit());
    let has_lowercase = word.chars().any(char::is_lowercase);
    let uppercase_count = word
        .chars()
        .filter(|character| character.is_uppercase())
        .count();
    let camel_case = has_lowercase && word.chars().skip(1).any(char::is_uppercase);
    let acronym = !has_lowercase && uppercase_count >= 2;
    camel_case || acronym || (has_letter && has_digit)
}

fn push_token(tokens: &mut Vec<String>, current: &mut String) {
    if current.is_empty() {
        return;
    }
    let token = std::mem::take(current);
    if token.chars().count() == 1 && !token.chars().all(|character| character.is_ascii_digit()) {
        return;
    }
    if is_stop_word(&token) {
        return;
    }
    tokens.push(stem(&token));
}

fn fold_latin(character: char) -> char {
    match character {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
        'ç' => 'c',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'ñ' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'ý' | 'ÿ' => 'y',
        other => other,
    }
}

fn is_stop_word(token: &str) -> bool {
    matches!(
        token,
        "a" | "ao"
            | "aos"
            | "as"
            | "atual"
            | "atuais"
            | "com"
            | "como"
            | "depois"
            | "deve"
            | "devem"
            | "da"
            | "das"
            | "de"
            | "do"
            | "dos"
            | "e"
            | "eh"
            | "em"
            | "essa"
            | "esse"
            | "esta"
            | "este"
            | "foi"
            | "mais"
            | "mas"
            | "na"
            | "nas"
            | "no"
            | "nos"
            | "o"
            | "os"
            | "ou"
            | "para"
            | "por"
            | "porque"
            | "qual"
            | "quais"
            | "quando"
            | "quanto"
            | "quantos"
            | "quantas"
            | "que"
            | "real"
            | "reais"
            | "se"
            | "sem"
            | "ser"
            | "sob"
            | "sobre"
            | "seu"
            | "seus"
            | "sua"
            | "suas"
            | "onde"
            | "um"
            | "uma"
            | "what"
            | "when"
            | "where"
            | "which"
            | "why"
            | "how"
            | "the"
            | "and"
            | "for"
            | "from"
            | "in"
            | "is"
            | "of"
            | "or"
            | "to"
            | "with"
    )
}

fn stem(token: &str) -> String {
    const SUFFIXES: &[&str] = &[
        "izacoes", "imentos", "amentos", "izacao", "idades", "imento", "amento", "acoes", "mente",
        "adores", "adoras", "idade", "izados", "izadas", "izado", "izada", "adora", "ador", "ando",
        "endo", "indo", "acoes", "acao", "icoes", "icao", "ados", "adas", "idos", "idas", "icos",
        "icas", "ico", "ica", "coes", "cao", "oes", "ais", "eis", "es", "ar", "er", "ir", "s",
    ];
    for suffix in SUFFIXES {
        if token.ends_with(suffix) && token.len() >= suffix.len() + 4 {
            return token[..token.len() - suffix.len()].to_owned();
        }
    }
    token.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "axon-knowledge-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn feedback_fixture(label: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
        let root = temporary_root(label);
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        let initially_first = corpus.join("cobalt-engine.md");
        let preferred = corpus.join("manual.md");
        let text = "Cobalt engine calibration follows the verified protocol.";
        fs::write(&initially_first, text).unwrap();
        fs::write(&preferred, text).unwrap();
        (
            root.clone(),
            root.join("memory.axm"),
            fs::canonicalize(initially_first).unwrap(),
            fs::canonicalize(preferred).unwrap(),
        )
    }

    #[test]
    fn tokenizer_folds_accents_stems_related_forms_and_ignores_stop_words() {
        assert_eq!(
            tokenize("A reconstrução e reconstruir"),
            ["reconstru", "reconstru"]
        );
        assert_eq!(tokenize("VARIÂNCIA exata"), ["variancia", "exata"]);
    }

    #[test]
    fn query_anchors_detect_identifiers_acronyms_and_alphanumeric_types() {
        assert_eq!(
            query_anchor_terms("ZXMEM001 QuartzArtifact usa QXR e v128, palavras comuns"),
            BTreeSet::from([
                "quartzartifact".to_owned(),
                "qxr".to_owned(),
                "v128".to_owned(),
                "zxmem001".to_owned(),
            ])
        );
        let anchors = BTreeSet::from(["relaystore".to_owned()]);
        assert_eq!(
            source_name_affinity(Path::new("src/relay_store.rs"), &anchors),
            1.0
        );
    }

    #[test]
    fn evidence_roles_distinguish_docs_implementation_tests_and_data() {
        assert_eq!(
            evidence_source_role(Path::new("docs/guide.md")),
            EvidenceSourceRole::Documentation
        );
        assert_eq!(
            evidence_source_role(Path::new("src/engine.rs")),
            EvidenceSourceRole::Implementation
        );
        assert_eq!(
            evidence_source_role(Path::new("tests/engine_contracts.rs")),
            EvidenceSourceRole::Test
        );
        assert_eq!(
            evidence_source_role(Path::new("fixtures/cases.json")),
            EvidenceSourceRole::Data
        );
        assert_eq!(
            evidence_source_role(Path::new("docs/BENCH_CONTRACT.md")),
            EvidenceSourceRole::Documentation
        );
        assert_eq!(
            evidence_source_role(Path::new("tests/README.md")),
            EvidenceSourceRole::Documentation
        );
        assert!(is_query_example(
            "Example query: axon investigate quartz relay"
        ));
        assert!(!is_query_example(
            "QuartzRelay combines a checkpoint with durable recovery."
        ));
    }

    #[test]
    fn bounded_typo_matching_has_an_explicit_radius() {
        assert_eq!(bounded_edit_distance("memoria", "memroia", 1), Some(1));
        assert_eq!(bounded_edit_distance("memoria", "morfologia", 2), None);
    }

    #[test]
    fn sparse_retrieval_matches_exhaustive_scoring_with_typo_numeric_and_feedback() {
        let root = temporary_root("sparse-differential");
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        for index in 0..2_048 {
            let text = "The meadow contains grass and soil.";
            store.sources.insert(
                format!("noise-{index}.md"),
                SourceRecord {
                    fingerprint: fingerprint(text.as_bytes()),
                    chunks: vec![StoredChunk {
                        start_line: 1,
                        end_line: 1,
                        text: text.to_owned(),
                    }],
                },
            );
        }
        let text = "Nebula engine uses ceramic material and occupies 24 bytes.";
        store.sources.insert(
            "engine.md".to_owned(),
            SourceRecord {
                fingerprint: fingerprint(text.as_bytes()),
                chunks: vec![StoredChunk {
                    start_line: 1,
                    end_line: 1,
                    text: text.to_owned(),
                }],
            },
        );
        store.feedback.insert(
            FeedbackKey {
                source: "engine.md".to_owned(),
                query_terms: vec!["engine".to_owned(), "nebula".to_owned()],
            },
            FeedbackRecord {
                relevant: 3,
                irrelevant: 1,
            },
        );
        store.search_index = SearchIndex::build(&store.sources);
        let queries = [
            "nebula engine ceramic",
            "neubla engine",
            "how many bytes nebula",
            "grass soil",
            "orchid chlorine",
        ];
        for query in queries {
            let sparse = store.score_query(query).unwrap();
            let original = store.search_index.postings.clone();
            let all_documents = (0..store.search_index.documents.len()).collect::<Vec<_>>();
            for postings in store.search_index.postings.values_mut() {
                *postings = all_documents.clone();
            }
            let exhaustive = store.score_query(query).unwrap();
            store.search_index.postings = original;
            assert_eq!(sparse.hits, exhaustive.hits, "query={query}");
        }
        let result = store.recall("nebula engine ceramic", 5).unwrap();
        assert_eq!(result.status(), RecallStatus::Found);
        assert_eq!(result.search_work().indexed_documents(), 2_049);
        assert_eq!(result.search_work().scored_documents(), 1);
        assert_eq!(result.search_work().posting_visits(), 3);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn subword_signature_bridges_morphology_without_matching_unrelated_terms() {
        let desalocar = subword_signature("desaloc");
        let alocar = subword_signature("aloc");
        let fotografia = subword_signature("fotografia");

        assert!(subword_similarity(&desalocar, &alocar) >= MIN_SUBWORD_SIMILARITY);
        assert!(subword_similarity(&desalocar, &fotografia) < MIN_SUBWORD_SIMILARITY);
    }

    #[test]
    fn chunks_keep_non_overlapping_source_line_ranges() {
        let chunks = chunk_text("# Title\n\nfirst paragraph\nline two\n\nlast").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 6);
        assert!(chunks[0].text.contains("first paragraph"));
    }

    #[test]
    fn encoded_memory_rejects_every_truncation() {
        let sources = BTreeMap::from([(
            "source.md".to_owned(),
            SourceRecord {
                fingerprint: fingerprint(b"durable evidence"),
                chunks: vec![StoredChunk {
                    start_line: 1,
                    end_line: 1,
                    text: "durable evidence".to_owned(),
                }],
            },
        )]);
        let encoded = encode_sources(&sources).unwrap();
        assert_eq!(
            decode_sources(&encoded, KnowledgeLimits::default()).unwrap(),
            sources
        );
        for end in 0..encoded.len() {
            assert!(decode_sources(&encoded[..end], KnowledgeLimits::default()).is_err());
        }
    }

    #[test]
    fn checksum_detects_payload_corruption() {
        let sources = BTreeMap::from([(
            "source.md".to_owned(),
            SourceRecord {
                fingerprint: fingerprint(b"evidence"),
                chunks: vec![StoredChunk {
                    start_line: 1,
                    end_line: 1,
                    text: "evidence".to_owned(),
                }],
            },
        )]);
        let mut encoded = encode_sources(&sources).unwrap();
        let last = encoded.len() - 1;
        encoded[last] ^= 0x80;
        assert!(decode_sources(&encoded, KnowledgeLimits::default()).is_err());
    }

    #[test]
    fn every_single_byte_corruption_is_rejected() {
        let sources = BTreeMap::from([(
            "source.md".to_owned(),
            SourceRecord {
                fingerprint: fingerprint(b"evidence with provenance"),
                chunks: vec![StoredChunk {
                    start_line: 2,
                    end_line: 3,
                    text: "evidence with provenance".to_owned(),
                }],
            },
        )]);
        let encoded = encode_sources(&sources).unwrap();
        for index in 0..encoded.len() {
            let mut damaged = encoded.clone();
            damaged[index] ^= 0x01;
            assert!(
                decode_sources(&damaged, KnowledgeLimits::default()).is_err(),
                "corruption at byte {index} was accepted"
            );
        }
    }

    #[test]
    fn decoder_rejects_invalid_feedback_even_with_a_valid_checksum() {
        let sources = BTreeMap::from([(
            "source.md".to_owned(),
            SourceRecord {
                fingerprint: fingerprint(b"grounded evidence"),
                chunks: vec![StoredChunk {
                    start_line: 1,
                    end_line: 1,
                    text: "grounded evidence".to_owned(),
                }],
            },
        )]);
        let cases = [
            BTreeMap::from([(
                FeedbackKey {
                    source: "missing.md".to_owned(),
                    query_terms: vec!["evidence".to_owned()],
                },
                FeedbackRecord {
                    relevant: 1,
                    irrelevant: 0,
                },
            )]),
            BTreeMap::from([(
                FeedbackKey {
                    source: "source.md".to_owned(),
                    query_terms: vec!["zeta".to_owned(), "alpha".to_owned()],
                },
                FeedbackRecord {
                    relevant: 1,
                    irrelevant: 0,
                },
            )]),
            BTreeMap::from([(
                FeedbackKey {
                    source: "source.md".to_owned(),
                    query_terms: vec!["evidence".to_owned()],
                },
                FeedbackRecord {
                    relevant: 0,
                    irrelevant: 0,
                },
            )]),
        ];
        for feedback in cases {
            let encoded = encode_memory(&sources, &feedback).unwrap();
            assert!(decode_memory(&encoded, KnowledgeLimits::default()).is_err());
        }
    }

    #[test]
    fn long_utf8_line_splits_on_character_boundaries_and_roundtrips() {
        let text = "á".repeat(MAX_CHUNK_BYTES);
        let chunks = chunk_text(&text).unwrap();
        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.text.len() <= MAX_CHUNK_BYTES)
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.start_line == 1 && chunk.end_line == 1)
        );
        let sources = BTreeMap::from([(
            "unicode.md".to_owned(),
            SourceRecord {
                fingerprint: fingerprint(text.as_bytes()),
                chunks,
            },
        )]);
        let encoded = encode_sources(&sources).unwrap();
        assert_eq!(
            decode_sources(&encoded, KnowledgeLimits::default()).unwrap(),
            sources
        );
    }

    #[test]
    fn learning_is_incremental_persistent_and_replaces_changed_content() {
        let root = temporary_root("incremental");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("facts.md");
        let memory = root.join("memory.axm");
        fs::write(&source, "Mercury is the closest planet to the Sun.").unwrap();
        let mut store = KnowledgeStore::open(&memory).unwrap();

        let first = store.learn_path(&source).unwrap();
        let second = store.learn_path(&source).unwrap();
        assert_eq!(first.added_sources(), 1);
        assert_eq!(second.unchanged_sources(), 1);
        assert_eq!(second.updated_sources(), 0);
        fs::write(&source, "Venus has a dense carbon dioxide atmosphere.").unwrap();
        let third = store.learn_path(&source).unwrap();
        assert_eq!(third.updated_sources(), 1);
        drop(store);

        let reopened = KnowledgeStore::open(&memory).unwrap();
        assert_eq!(
            reopened
                .recall("dense atmosphere Venus", 3)
                .unwrap()
                .status(),
            RecallStatus::Found
        );
        assert_eq!(
            reopened.recall("Mercury closest Sun", 3).unwrap().status(),
            RecallStatus::NoEvidence
        );
        drop(reopened);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ranking_prefers_the_grounded_source_over_distractors_and_handles_a_typo() {
        let root = temporary_root("ranking");
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        fs::write(
            corpus.join("memory.md"),
            "A memoria do AXON guarda fontes, intervalos de linha, versoes e evidencias locais.",
        )
        .unwrap();
        fs::write(
            corpus.join("weather.md"),
            "Weather observations record temperature and atmospheric pressure.",
        )
        .unwrap();
        fs::write(
            corpus.join("build.md"),
            "The compiler builds a release binary from Rust source files.",
        )
        .unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&corpus).unwrap();

        let result = store
            .recall("onde a memroia guarda evidencias e fontes", 3)
            .unwrap();
        assert_ne!(result.status(), RecallStatus::NoEvidence);
        assert!(
            result.hits()[0]
                .source()
                .file_name()
                .unwrap()
                .eq_ignore_ascii_case("memory.md")
        );
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn composed_reasoning_covers_a_query_with_three_separate_passages() {
        let root = temporary_root("composed-evidence");
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        let source = corpus.join("durability.md");
        fs::write(
            &source,
            "# Integrity\n\nAxiom validates stored content.\n\n# Exclusion\n\nMutex prevents a second writer.\n\n# Recovery\n\nPriorstate survives an interrupted publication.\n",
        )
        .unwrap();
        fs::write(
            corpus.join("distractor.md"),
            "Axiom axiom axiom protects an unrelated packet.",
        )
        .unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&corpus).unwrap();
        let query = "axiom mutex priorstate";

        assert_eq!(
            store.recall(query, 5).unwrap().status(),
            RecallStatus::NoEvidence
        );
        let result = store.reason(query, 5).unwrap();
        assert_eq!(result.status(), RecallStatus::Found);
        assert_eq!(
            result.bundles()[0].source(),
            fs::canonicalize(source).unwrap()
        );
        assert_eq!(result.bundles()[0].evidence().len(), 3);
        assert_eq!(
            result.bundles()[0].covered_terms(),
            ["axiom", "mutex", "priorstate"]
        );
        assert!(result.bundles()[0].missing_terms().is_empty());

        let unrelated = store.reason("axiom wallet crypton", 5).unwrap();
        assert_eq!(unrelated.status(), RecallStatus::NoEvidence);
        assert!(unrelated.bundles().is_empty());
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn composed_reasoning_uses_a_subword_bridge_only_with_an_exact_anchor() {
        let root = temporary_root("composed-subword");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("allocation.md");
        fs::write(
            &source,
            "A regiao zirconio mantem alocacao persistente de vetor.",
        )
        .unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&source).unwrap();

        let grounded = store.reason("desalocar zirconio regiao vetor", 3).unwrap();
        assert_eq!(grounded.status(), RecallStatus::Found);
        assert!(
            grounded.bundles()[0]
                .covered_terms()
                .iter()
                .any(|term| term == "desaloc")
        );
        let unanchored = store.reason("desalocar fotografia noturna", 3).unwrap();
        assert_eq!(unanchored.status(), RecallStatus::NoEvidence);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn investigation_builds_a_cross_source_path_through_rare_bridge_terms() {
        let root = temporary_root("cross-source-evidence-path");
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        fs::write(
            corpus.join("origin.md"),
            "The orion engine delegates through the cobalt relay.",
        )
        .unwrap();
        fs::write(
            corpus.join("destination.md"),
            "The cobalt relay opens the cedar vault.",
        )
        .unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&corpus).unwrap();

        let result = store.investigate("orion engine cedar vault", 3).unwrap();
        assert_eq!(result.status(), RecallStatus::Found);
        assert_eq!(result.paths()[0].evidence().len(), 2);
        assert_eq!(result.paths()[0].links().len(), 1);
        assert_eq!(
            result.paths()[0].links()[0].kind(),
            EvidenceLinkKind::SharedTerms
        );
        assert!(
            result.paths()[0].links()[0]
                .bridge_terms()
                .iter()
                .any(|term| term == "cobalt")
        );
        assert_eq!(result.paths()[0].weighted_coverage(), 1.0);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn investigation_links_sources_through_a_short_explicit_acronym() {
        let root = temporary_root("short-source-entity");
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        fs::write(corpus.join("origin.md"), "QXR controls the amber gate.").unwrap();
        fs::write(corpus.join("destination.md"), "QXR opens the violet key.").unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&corpus).unwrap();

        let result = store.investigate("QXR amber gate violet key", 3).unwrap();
        assert_eq!(result.anchor_terms(), ["qxr"]);
        assert_eq!(result.status(), RecallStatus::Found);
        assert_eq!(result.paths()[0].evidence().len(), 2);
        assert_eq!(
            result.paths()[0].links()[0].kind(),
            EvidenceLinkKind::SourceEntity
        );
        assert_eq!(result.paths()[0].links()[0].bridge_terms(), ["qxr"]);
        assert!(result.paths()[0].corroborated_coverage() > 0.0);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn investigation_prefers_documentation_implementation_and_test_evidence() {
        let root = temporary_root("source-role-evidence-path");
        let corpus = root.join("corpus");
        for directory in ["docs", "src", "tests"] {
            fs::create_dir_all(corpus.join(directory)).unwrap();
        }
        fs::write(
            corpus.join("docs/guide.md"),
            "QuartzRelay combines the amber intake with durable recovery.",
        )
        .unwrap();
        fs::write(
            corpus.join("src/relay.rs"),
            "pub struct QuartzRelay; // cobalt checkpoint protocol implementation",
        )
        .unwrap();
        fs::write(
            corpus.join("tests/relay_contracts.rs"),
            "#[test] fn violet_replay() { /* checkpoint protocol validates violet replay. */ }",
        )
        .unwrap();
        fs::write(
            corpus.join("docs/report.md"),
            "Example query: investigate amber cobalt violet replay durable.",
        )
        .unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&corpus).unwrap();

        let result = store
            .investigate(
                "QuartzRelay amber cobalt checkpoint protocol violet replay durable",
                5,
            )
            .unwrap();
        assert_eq!(result.status(), RecallStatus::Found);
        assert_eq!(result.paths()[0].distinct_role_count(), 3);
        assert_eq!(result.paths()[0].anchor_definition_count(), 1);
        assert!(result.paths()[0].evidence().len() <= MAX_EVIDENCE_PER_PATH);
        assert!(result.paths()[0].evidence().iter().all(|hit| {
            result.paths()[0]
                .evidence()
                .iter()
                .filter(|other| other.source() == hit.source())
                .count()
                <= MAX_EVIDENCE_PER_SOURCE_PATH
        }));
        assert_eq!(
            result.paths()[0]
                .evidence_roles()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                EvidenceSourceRole::Documentation,
                EvidenceSourceRole::Implementation,
                EvidenceSourceRole::Test,
            ])
        );
        assert!(result.paths()[0].evidence().iter().all(|hit| {
            hit.source()
                .file_name()
                .is_none_or(|name| !name.eq_ignore_ascii_case("report.md"))
        }));
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn investigation_does_not_treat_repeated_example_queries_as_evidence() {
        let root = temporary_root("query-only-evidence");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("examples.md");
        fs::write(
            &source,
            "Example query: investigate QuartzRelay amber cobalt violet replay durable.",
        )
        .unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&source).unwrap();
        let result = store
            .investigate("QuartzRelay amber cobalt violet replay durable", 5)
            .unwrap();
        assert_eq!(result.status(), RecallStatus::NoEvidence);
        assert!(result.paths().is_empty());
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn investigation_reports_an_unrepresented_concept_instead_of_ignoring_it() {
        let root = temporary_root("missing-query-concept");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("relay.md");
        fs::write(&source, "QuartzRelay stores the amber checkpoint durably.").unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&source).unwrap();
        let result = store
            .investigate("QuartzRelay amber checkpoint quasiparticle", 5)
            .unwrap();
        assert_eq!(result.status(), RecallStatus::NoEvidence);
        assert!(
            result
                .unsupported_terms()
                .iter()
                .any(|term| term == "quasiparticle")
        );
        assert!(result.paths().is_empty());
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn investigation_accepts_one_edit_but_refuses_a_more_distant_word() {
        let root = temporary_root("conservative-query-corrections");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("relay.md");
        fs::write(
            &source,
            "QuartzRelay keeps a durable chromatic silver rotor.",
        )
        .unwrap();
        for index in 0..24 {
            fs::write(
                root.join(format!("noise-{index}.md")),
                "A meadow contains grass and soil.",
            )
            .unwrap();
        }
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&root).unwrap();
        assert_eq!(
            store
                .investigate("QuartzRelay durable chromtaic silver rotor", 3)
                .unwrap()
                .status(),
            RecallStatus::Found,
        );
        let result = store
            .investigate("QuartzRelay durable chromaxix silver rotor", 3)
            .unwrap();
        assert_eq!(result.status(), RecallStatus::NoEvidence);
        assert!(
            result
                .unsupported_terms()
                .iter()
                .any(|term| term == "chromaxix")
        );
        assert_eq!(
            store
                .recall("QuartzRelay durable chromaxix silver rotor", 3)
                .unwrap()
                .status(),
            RecallStatus::Found,
        );
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_wide_matches_and_roles_cannot_ground_missing_cited_terms() {
        let query = ["relay", "amber", "cobalt", "violet", "durable"].map(str::to_owned);
        let hits = ["docs/relay.md", "src/relay.rs", "tests/relay.rs"]
            .into_iter()
            .enumerate()
            .map(|(document_index, source)| ScoredHit {
                document_index,
                hit: RecallHit {
                    source: PathBuf::from(source),
                    start_line: 1,
                    end_line: 1,
                    score: 1.0,
                    feedback_factor: 1.0,
                    matched_terms: vec!["relay".to_owned()],
                    text: "relay".to_owned(),
                },
                matched_indices: BTreeSet::from([0]),
                exact_indices: BTreeSet::from([0]),
                query_example: false,
            })
            .collect::<Vec<_>>();
        let links = (1..=2)
            .map(|to_evidence| EvidenceLink {
                from_evidence: 0,
                to_evidence,
                kind: EvidenceLinkKind::SourceEntity,
                score: 1.0,
                bridge_terms: vec!["relay".to_owned()],
            })
            .collect();
        let mut path = build_scored_path(&[0, 1, 2], links, &hits, &query, &[1.0; 5], 5.0);
        path.path.source_union_coverage = 1.0;
        path.path.source_union_exact_coverage = 1.0;
        path.path.anchor_source_count = 3;
        assert_eq!(path.path.distinct_role_count(), 3);
        assert!(!path_is_grounded(&query, &path));
    }

    #[test]
    fn investigation_uses_adjacency_but_refuses_disconnected_word_collection() {
        let root = temporary_root("adjacent-evidence-path");
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        let first = format!("Aurora engine {}", "xenon ".repeat(150));
        let second = format!("Cedar vault {}", "quartz ".repeat(150));
        fs::write(corpus.join("adjacent.md"), format!("{first}\n\n{second}")).unwrap();
        fs::write(
            corpus.join("isolated-origin.md"),
            "A topaz motor contains a silver conduit.",
        )
        .unwrap();
        fs::write(
            corpus.join("isolated-destination.md"),
            "A juniper archive contains an ivory chamber.",
        )
        .unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&corpus).unwrap();

        let adjacent = store.investigate("aurora engine cedar vault", 3).unwrap();
        assert_eq!(adjacent.status(), RecallStatus::Found);
        assert_eq!(
            adjacent.paths()[0].links()[0].kind(),
            EvidenceLinkKind::Adjacent
        );
        let disconnected = store.investigate("topaz motor ivory chamber", 3).unwrap();
        assert_eq!(disconnected.status(), RecallStatus::NoEvidence);
        assert!(disconnected.paths().is_empty());
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn relevant_feedback_reranks_existing_evidence_and_survives_reopen() {
        let (root, memory, initially_first, preferred) = feedback_fixture("feedback-rerank");
        let mut store = KnowledgeStore::open(&memory).unwrap();
        store.learn_path(root.join("corpus")).unwrap();
        let query = "cobalt engine calibration";

        let before = store.recall(query, 2).unwrap();
        assert_eq!(before.hits()[0].source(), initially_first);
        let report = store
            .record_feedback(query, &preferred, KnowledgeFeedbackLabel::Relevant)
            .unwrap();
        assert_eq!(report.relevant_observations(), 1);
        assert_eq!(report.irrelevant_observations(), 0);
        let after = store.recall(query, 2).unwrap();
        assert_eq!(after.hits()[0].source(), preferred);
        assert!((after.hits()[0].feedback_factor() - 2.0).abs() < 1e-9);
        assert_eq!(store.stats().unwrap().feedback_records(), 1);
        drop(store);

        let reopened = KnowledgeStore::open(&memory).unwrap();
        let persisted = reopened.recall(query, 2).unwrap();
        assert_eq!(persisted.hits()[0].source(), preferred);
        assert_eq!(reopened.stats().unwrap().feedback_observations(), 1);
        drop(reopened);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn feedback_is_query_local_and_cannot_create_evidence() {
        let (root, memory, _, preferred) = feedback_fixture("feedback-local");
        let mut store = KnowledgeStore::open(&memory).unwrap();
        store.learn_path(root.join("corpus")).unwrap();
        let similar_query = "protocol calibration";
        let before = store.recall(similar_query, 2).unwrap();
        let before_sources = before
            .hits()
            .iter()
            .map(|hit| hit.source().to_owned())
            .collect::<Vec<_>>();
        store
            .record_feedback(
                "cobalt engine calibration",
                &preferred,
                KnowledgeFeedbackLabel::Relevant,
            )
            .unwrap();

        let similar = store.recall(similar_query, 2).unwrap();
        assert_eq!(
            similar
                .hits()
                .iter()
                .map(|hit| hit.source().to_owned())
                .collect::<Vec<_>>(),
            before_sources
        );
        assert!(
            similar
                .hits()
                .iter()
                .all(|hit| (hit.feedback_factor() - 1.0).abs() < 1e-9)
        );
        let unrelated = store
            .recall("orchid chlorophyll photosynthesis", 2)
            .unwrap();
        assert_eq!(unrelated.status(), RecallStatus::NoEvidence);
        assert!(unrelated.hits().is_empty());
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn irrelevant_feedback_demotes_existing_evidence() {
        let (root, memory, initially_first, preferred) = feedback_fixture("feedback-demote");
        let mut store = KnowledgeStore::open(&memory).unwrap();
        store.learn_path(root.join("corpus")).unwrap();
        let query = "cobalt engine calibration";
        assert_eq!(
            store.recall(query, 2).unwrap().hits()[0].source(),
            initially_first
        );

        store
            .record_feedback(query, &initially_first, KnowledgeFeedbackLabel::Irrelevant)
            .unwrap();
        let after = store.recall(query, 2).unwrap();
        assert_eq!(after.hits()[0].source(), preferred);
        let demoted = after
            .hits()
            .iter()
            .find(|hit| hit.source() == initially_first)
            .unwrap();
        assert!((demoted.feedback_factor() - 0.5).abs() < 1e-9);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn feedback_refuses_unknown_sources_and_expires_with_changed_content() {
        let root = temporary_root("feedback-lifecycle");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("facts.md");
        let unknown = root.join("unknown.md");
        fs::write(&source, "Aster engine uses a ceramic rotor.").unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&source).unwrap();

        assert!(matches!(
            store.record_feedback(
                "aster ceramic rotor",
                &unknown,
                KnowledgeFeedbackLabel::Relevant
            ),
            Err(KnowledgeError::UnknownSource(path)) if path == unknown
        ));
        store
            .record_feedback(
                "aster ceramic rotor",
                &source,
                KnowledgeFeedbackLabel::Relevant,
            )
            .unwrap();
        assert_eq!(store.stats().unwrap().feedback_records(), 1);

        fs::write(&source, "Aster engine now uses a carbon rotor.").unwrap();
        store.learn_path(&source).unwrap();
        assert_eq!(store.stats().unwrap().feedback_records(), 0);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn feedback_limits_fail_without_mutating_prior_experience() {
        let root = temporary_root("feedback-limits");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("facts.md");
        fs::write(&source, "Alpha beta gamma delta evidence.").unwrap();
        let limits = KnowledgeLimits {
            max_feedback_records: 1,
            max_feedback_terms: 2,
            ..KnowledgeLimits::default()
        };
        let mut store = KnowledgeStore::open_with_limits(root.join("memory.axm"), limits).unwrap();
        store.learn_path(&source).unwrap();
        store
            .record_feedback("alpha beta", &source, KnowledgeFeedbackLabel::Relevant)
            .unwrap();

        assert!(matches!(
            store.record_feedback("gamma delta", &source, KnowledgeFeedbackLabel::Relevant),
            Err(KnowledgeError::CapacityExceeded("feedback record count"))
        ));
        assert!(matches!(
            store.record_feedback(
                "alpha beta gamma",
                &source,
                KnowledgeFeedbackLabel::Relevant
            ),
            Err(KnowledgeError::CapacityExceeded(
                "feedback query term count"
            ))
        ));
        let stats = store.stats().unwrap();
        assert_eq!(stats.feedback_records(), 1);
        assert_eq!(stats.feedback_observations(), 1);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_memory_opens_and_migrates_when_feedback_is_recorded() {
        let root = temporary_root("feedback-v1-migration");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("legacy.md");
        let memory = root.join("memory.axm");
        fs::write(&source, "Legacy cobalt evidence remains readable.").unwrap();
        let source_key = fs::canonicalize(&source)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let sources = BTreeMap::from([(
            source_key,
            SourceRecord {
                fingerprint: fingerprint(b"Legacy cobalt evidence remains readable."),
                chunks: vec![StoredChunk {
                    start_line: 1,
                    end_line: 1,
                    text: "Legacy cobalt evidence remains readable.".to_owned(),
                }],
            },
        )]);
        fs::write(&memory, encode_legacy_sources(&sources).unwrap()).unwrap();

        let mut store = KnowledgeStore::open(&memory).unwrap();
        assert_eq!(store.stats().unwrap().feedback_records(), 0);
        store
            .record_feedback(
                "legacy cobalt evidence",
                &source,
                KnowledgeFeedbackLabel::Relevant,
            )
            .unwrap();
        drop(store);
        let encoded = fs::read(&memory).unwrap();
        assert_eq!(
            u32::from_le_bytes(encoded[8..12].try_into().unwrap()),
            FORMAT_VERSION
        );
        let reopened = KnowledgeStore::open(&memory).unwrap();
        assert_eq!(reopened.stats().unwrap().feedback_records(), 1);
        drop(reopened);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unrelated_query_abstains_instead_of_returning_a_random_chunk() {
        let root = temporary_root("abstain");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("facts.md");
        fs::write(&source, "Exact variance uses three arithmetic moments.").unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&source).unwrap();

        let result = store
            .recall("orchid photosynthesis chlorophyll", 3)
            .unwrap();
        assert_eq!(result.status(), RecallStatus::NoEvidence);
        assert!(result.hits().is_empty());
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rare_missing_terms_outweigh_two_generic_matches() {
        let root = temporary_root("weighted-abstention");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("facts.md");
        fs::write(
            &source,
            "The current experiment compares real execution costs and local evidence.",
        )
        .unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&source).unwrap();

        let result = store
            .recall("current Bitcoin quotation in real currency", 3)
            .unwrap();
        assert_eq!(result.status(), RecallStatus::NoEvidence);
        assert!(result.hits().is_empty());
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn forgetting_a_directory_removes_only_its_sources() {
        let root = temporary_root("forget");
        let first = root.join("first");
        let second = root.join("second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        fs::write(first.join("a.md"), "alpha evidence").unwrap();
        fs::write(second.join("b.md"), "beta evidence").unwrap();
        let memory = root.join("memory.axm");
        let mut store = KnowledgeStore::open(&memory).unwrap();
        store.learn_path(&first).unwrap();
        store.learn_path(&second).unwrap();
        store
            .record_feedback(
                "alpha evidence",
                first.join("a.md"),
                KnowledgeFeedbackLabel::Relevant,
            )
            .unwrap();
        store
            .record_feedback(
                "beta evidence",
                second.join("b.md"),
                KnowledgeFeedbackLabel::Relevant,
            )
            .unwrap();

        assert_eq!(store.forget(&first).unwrap(), 1);
        assert_eq!(store.stats().unwrap().sources(), 1);
        assert_eq!(store.stats().unwrap().feedback_records(), 1);
        drop(store);
        let reopened = KnowledgeStore::open(&memory).unwrap();
        assert_eq!(reopened.stats().unwrap().sources(), 1);
        assert_eq!(reopened.stats().unwrap().feedback_records(), 1);
        drop(reopened);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_publish_recovers_the_previous_snapshot() {
        let root = temporary_root("recovery");
        fs::create_dir_all(&root).unwrap();
        let memory = root.join("memory.axm");
        let source = root.join("source.md");
        fs::write(&source, "recoverable knowledge").unwrap();
        let mut store = KnowledgeStore::open(&memory).unwrap();
        store.learn_path(&source).unwrap();
        drop(store);
        let backup = sidecar_path(&memory, ".previous");
        fs::rename(&memory, &backup).unwrap();

        let recovered = KnowledgeStore::open(&memory).unwrap();
        assert_eq!(recovered.stats().unwrap().sources(), 1);
        assert!(memory.exists());
        assert!(!backup.exists());
        drop(recovered);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sidecar_lock_refuses_a_second_open_writer() {
        let root = temporary_root("exclusive-lock");
        fs::create_dir_all(&root).unwrap();
        let memory = root.join("memory.axm");
        let first = KnowledgeStore::open(&memory).unwrap();
        assert!(KnowledgeStore::open(&memory).is_err());
        drop(first);
        assert!(KnowledgeStore::open(&memory).is_ok());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_learning_skips_unsupported_and_non_utf8_files() {
        let root = temporary_root("skip-binary");
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        fs::write(corpus.join("valid.md"), "grounded text").unwrap();
        fs::write(corpus.join("unsupported.exe"), b"plain but unsupported").unwrap();
        fs::write(corpus.join("invalid.md"), [0xff, 0xfe, 0xfd]).unwrap();
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();

        let report = store.learn_path(&corpus).unwrap();
        assert_eq!(report.added_sources(), 1);
        assert_eq!(report.skipped_files(), 2);
        assert_eq!(store.stats().unwrap().sources(), 1);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_input_refuses_oversized_source_without_learning_it() {
        let root = temporary_root("bounded-input");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("large.md");
        fs::write(&source, "more than four bytes").unwrap();
        let limits = KnowledgeLimits {
            max_input_file_bytes: 4,
            ..KnowledgeLimits::default()
        };
        let mut store = KnowledgeStore::open_with_limits(root.join("memory.axm"), limits).unwrap();

        assert!(matches!(
            store.learn_path(&source),
            Err(KnowledgeError::InputTooLarge { .. })
        ));
        assert_eq!(store.stats().unwrap().sources(), 0);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn retrieval_finds_each_unique_source_among_sixty_four_distractors() {
        let root = temporary_root("retrieval-grid");
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        for index in 0..64 {
            fs::write(
                corpus.join(format!("record-{index:02}.md")),
                format!(
                    "Archive record beacon{index:02} contains verified channel {index:02} evidence."
                ),
            )
            .unwrap();
        }
        let mut store = KnowledgeStore::open(root.join("memory.axm")).unwrap();
        store.learn_path(&corpus).unwrap();

        for index in 0..64 {
            let result = store
                .recall(&format!("verified archive beacon{index:02}"), 3)
                .unwrap();
            assert_eq!(result.status(), RecallStatus::Found);
            assert_eq!(
                result.hits()[0]
                    .source()
                    .file_name()
                    .unwrap()
                    .to_string_lossy(),
                format!("record-{index:02}.md")
            );
        }
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }
}
