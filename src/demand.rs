//! Immutable learned modules, dormant storage, and bounded demand-driven execution.
use crate::{
    Authority, Effect, TextPrediction, TextSkill, TextValuePrediction,
    behavior::BehaviorModel,
    live_store::checksum,
    numeric::evolution::EvolvedSkill,
    reactive::{ReactiveCircuit, ReactiveSkill},
    text_skill::{publish_new, read_bounded},
    world::WorldModel,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    mem::size_of,
    path::{Path, PathBuf},
    time::Instant,
};

const MAX_MODULES: usize = 256;
const MAX_CATALOG: u64 = 1_048_576;

#[path = "goal_controller.rs"]
pub mod controller;

#[path = "genome.rs"]
pub mod genome;

#[derive(Debug)]
pub enum DemandError {
    Invalid(&'static str),
    Budget(&'static str),
    Missing(String),
    Unauthorized(Effect),
    Context,
    Identity,
    NoRoute,
    NeedsExample {
        row: usize,
        module: String,
        input: String,
    },
    Suspended,
    Drift {
        expected: u16,
        actual: u16,
    },
    Io(std::io::Error),
    Module(String),
}
impl fmt::Display for DemandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid demand execution: {s}"),
            Self::Budget(s) => write!(f, "seed budget exhausted: {s}"),
            Self::Missing(s) => write!(f, "unknown module: {s}"),
            Self::Unauthorized(e) => write!(f, "effect not authorized: {e:?}"),
            Self::Context => f.write_str("module context mismatch"),
            Self::Identity => f.write_str("module content or residency contract changed"),
            Self::NoRoute => f.write_str("no affordable declared route to the goal"),
            Self::NeedsExample { row, module, input } => {
                write!(
                    f,
                    "module {module:?}, row {row} needs a demonstration for {input:?}; batch refused"
                )
            }
            Self::Suspended => f.write_str("behavior cursor suspended"),
            Self::Drift { expected, actual } => {
                write!(f, "behavior drift: expected {expected}, observed {actual}")
            }
            Self::Io(e) => e.fmt(f),
            Self::Module(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for DemandError {}
impl From<std::io::Error> for DemandError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
fn storage(e: crate::TextSkillError) -> DemandError {
    DemandError::Module(e.to_string())
}
fn name(s: &str) -> Result<(), DemandError> {
    if s.is_empty() || s.len() > 256 || s.trim() != s || s.chars().any(char::is_control) {
        return Err(DemandError::Invalid("identifier"));
    }
    Ok(())
}
/// Validate a portable sibling filename; this never grants filesystem authority.
pub fn file_name(s: &str) -> Result<(), DemandError> {
    if s.is_empty()
        || s.len() > 128
        || s.starts_with('.')
        || s.ends_with('.')
        || !s
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
    {
        return Err(DemandError::Invalid(
            "module must be a plain sibling filename",
        ));
    }
    let stem = s.split('.').next().expect("nonempty").to_ascii_uppercase();
    if ["CON", "PRN", "AUX", "NUL"].contains(&stem.as_str())
        || ((stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.len() == 4
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
    {
        return Err(DemandError::Invalid("reserved device filename"));
    }
    Ok(())
}
fn require(authority: &Authority, effect: Effect) -> Result<(), DemandError> {
    if authority.allows(effect) {
        Ok(())
    } else {
        Err(DemandError::Unauthorized(effect))
    }
}
pub(crate) fn parent_or_current(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

/// Innate policy, not a learned model. The byte cap covers retained modules, not process RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Seed {
    pub active_bytes: usize,
    pub active_modules: usize,
    pub max_rows: usize,
    pub max_value_bytes: usize,
    pub batch_bytes: usize,
    pub max_plan_cost: u64,
}
impl Default for Seed {
    fn default() -> Self {
        Self {
            active_bytes: 65_536,
            active_modules: 1,
            max_rows: 20_000,
            max_value_bytes: 1024,
            batch_bytes: 16_777_216,
            max_plan_cost: 1_000_000,
        }
    }
}
impl Seed {
    fn validate(self) -> Result<(), DemandError> {
        if self.active_bytes == 0
            || self.active_bytes > 67_108_864
            || self.active_modules == 0
            || self.active_modules > MAX_MODULES
            || self.max_rows == 0
            || self.max_rows > 100_000
            || self.max_value_bytes == 0
            || self.max_value_bytes > 1024
            || self.batch_bytes == 0
            || self.batch_bytes > 268_435_456
            || self.max_plan_cost == 0
        {
            return Err(DemandError::Invalid("seed limits"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModuleKind {
    Text,
    Behavior,
    World,
    Evolved,
    Reactive,
    Numeric,
    Circuit,
    Study,
    Semantic,
    #[cfg(feature = "curriculum")]
    Structural,
    #[cfg(feature = "curriculum")]
    Sequence,
    #[cfg(feature = "curriculum")]
    CompactSequence,
    #[cfg(feature = "curriculum")]
    EnhancedSequence,
    #[cfg(feature = "curriculum")]
    FactorizedSequence,
    #[cfg(feature = "curriculum")]
    Dependency,
    #[cfg(feature = "curriculum")]
    RelationNetwork,
    #[cfg(feature = "curriculum")]
    JointSequence,
    #[cfg(feature = "curriculum")]
    LexicalEncoder,
    #[cfg(feature = "curriculum")]
    SegmentSequence,
    #[cfg(feature = "curriculum")]
    RelationalPointer,
    #[cfg(feature = "curriculum")]
    ContextPointer,
    #[cfg(feature = "curriculum")]
    Synaptic,
}
impl ModuleKind {
    fn tag(self) -> u8 {
        match self {
            Self::Text => 0,
            Self::Behavior => 1,
            Self::World => 2,
            Self::Evolved => 3,
            Self::Reactive => 4,
            Self::Numeric => 5,
            Self::Circuit => 18,
            Self::Study => 6,
            Self::Semantic => 7,
            #[cfg(feature = "curriculum")]
            Self::Structural => 8,
            #[cfg(feature = "curriculum")]
            Self::Sequence => 9,
            #[cfg(feature = "curriculum")]
            Self::CompactSequence => 10,
            #[cfg(feature = "curriculum")]
            Self::EnhancedSequence => 11,
            #[cfg(feature = "curriculum")]
            Self::FactorizedSequence => 12,
            #[cfg(feature = "curriculum")]
            Self::Dependency => 13,
            #[cfg(feature = "curriculum")]
            Self::RelationNetwork => 14,
            #[cfg(feature = "curriculum")]
            Self::JointSequence => 15,
            #[cfg(feature = "curriculum")]
            Self::LexicalEncoder => 16,
            #[cfg(feature = "curriculum")]
            Self::SegmentSequence => 17,
            #[cfg(feature = "curriculum")]
            Self::RelationalPointer => 19,
            #[cfg(feature = "curriculum")]
            Self::ContextPointer => 20,
            #[cfg(feature = "curriculum")]
            Self::Synaptic => 21,
        }
    }
    fn maximum(self) -> u64 {
        match self {
            Self::Text => 2_097_152,
            Self::Behavior => 16_384,
            Self::World => 512,
            Self::Evolved => 65_536,
            Self::Reactive => 1_048_576,
            Self::Numeric => 32768,
            Self::Circuit => 65_536,
            Self::Study => crate::cumulative::MAX_BYTES as u64,
            Self::Semantic => crate::semantic::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::Structural => crate::structural::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::Sequence => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::CompactSequence => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::EnhancedSequence => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::FactorizedSequence => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::Dependency => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::RelationNetwork => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::JointSequence => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::LexicalEncoder => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::SegmentSequence => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::RelationalPointer => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::ContextPointer => crate::structural::sequence::MAX_BYTES as u64,
            #[cfg(feature = "curriculum")]
            Self::Synaptic => 196608,
        }
    }
}

#[derive(Debug)]
pub enum Module {
    Text(TextSkill),
    Behavior(BehaviorModel),
    World(WorldModel),
    Evolved(Box<EvolvedSkill>),
    Reactive(Box<(ReactiveSkill, ReactiveCircuit)>),
    Numeric(Box<crate::numeric::NumericSkill>),
    Circuit(Box<crate::numeric::NumericExecution>),
    Study(Box<crate::cumulative::StudySeed>),
    Semantic(Box<crate::semantic::SemanticSeed>),
    #[cfg(feature = "curriculum")]
    Structural(Box<crate::structural::StructuralSeed>),
    #[cfg(feature = "curriculum")]
    Sequence(Box<crate::structural::sequence::Model>),
    #[cfg(feature = "curriculum")]
    CompactSequence(Box<crate::structural::sequence::compact::CompactModel>),
    #[cfg(feature = "curriculum")]
    EnhancedSequence(Box<crate::structural::sequence::compact::EnhancedModel>),
    #[cfg(feature = "curriculum")]
    FactorizedSequence(Box<crate::structural::sequence::factorized::Model>),
    #[cfg(feature = "curriculum")]
    Dependency(Box<crate::structural::sequence::dependency::Model>),
    #[cfg(feature = "curriculum")]
    RelationNetwork(Box<crate::structural::sequence::relation_network::Model>),
    #[cfg(feature = "curriculum")]
    JointSequence(Box<crate::structural::sequence::joint::Model>),
    #[cfg(feature = "curriculum")]
    LexicalEncoder(Box<crate::structural::sequence::lexical_encoder::Model>),
    #[cfg(feature = "curriculum")]
    SegmentSequence(Box<crate::structural::sequence::segment::Model>),
    #[cfg(feature = "curriculum")]
    RelationalPointer(Box<crate::structural::sequence::lexical_encoder::relational_pointer::Model>),
    #[cfg(feature = "curriculum")]
    ContextPointer(Box<crate::structural::sequence::lexical_encoder::relational_pointer::context_read::ContextModel>),
    #[cfg(feature = "curriculum")]
    Synaptic(Box<crate::structural::sequence::lexical_encoder::synaptic::Model>),
}
impl Module {
    fn decode_reserved(
        kind: ModuleKind,
        bytes: &[u8],
        expected: usize,
    ) -> Result<Self, DemandError> {
        let m = Self::decode(kind, bytes)?;
        #[cfg(feature = "curriculum")]
        let m = match m {
            Self::Dependency(mut model) => {
                model
                    .restore_legacy_reservation(expected.saturating_sub(size_of::<Self>()))
                    .map_err(|error| DemandError::Module(error.to_string()))?;
                Self::Dependency(model)
            }
            other => other,
        };
        if m.retained_bytes() != expected {
            return Err(DemandError::Identity);
        }
        Ok(m)
    }
    fn decode(kind: ModuleKind, bytes: &[u8]) -> Result<Self, DemandError> {
        match kind {
            #[cfg(feature = "curriculum")]
            ModuleKind::Synaptic => crate::structural::sequence::lexical_encoder::synaptic::Model::from_bytes(bytes)
                .map(|m| Self::Synaptic(Box::new(m)))
                .map_err(|e| DemandError::Module(e.to_string())),
            #[cfg(feature = "curriculum")]
            ModuleKind::ContextPointer => {
                crate::structural::sequence::lexical_encoder::relational_pointer::context_read::ContextModel::from_bytes(bytes)
                    .map(|m|Self::ContextPointer(Box::new(m)))
                    .map_err(|e|DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::RelationalPointer => {
                crate::structural::sequence::lexical_encoder::relational_pointer::Model::from_bytes(
                    bytes,
                )
                .map(|m| Self::RelationalPointer(Box::new(m)))
                .map_err(|e| DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::SegmentSequence => {
                crate::structural::sequence::segment::Model::from_bytes(bytes)
                    .map(|m| Self::SegmentSequence(Box::new(m)))
                    .map_err(|e| DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::LexicalEncoder => {
                crate::structural::sequence::lexical_encoder::Model::from_bytes(bytes)
                    .map(|m| Self::LexicalEncoder(Box::new(m)))
                    .map_err(|e| DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::JointSequence => {
                crate::structural::sequence::joint::Model::from_bytes(bytes)
                    .map(|m| Self::JointSequence(Box::new(m)))
                    .map_err(|e| DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::RelationNetwork => {
                crate::structural::sequence::relation_network::Model::from_bytes(bytes)
                    .map(|m| Self::RelationNetwork(Box::new(m)))
                    .map_err(|e| DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::Dependency => {
                crate::structural::sequence::dependency::Model::from_bytes(bytes)
                    .map(|m| Self::Dependency(Box::new(m)))
                    .map_err(|e| DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::FactorizedSequence => {
                crate::structural::sequence::factorized::Model::from_bytes(bytes)
                    .map(|m| Self::FactorizedSequence(Box::new(m)))
                    .map_err(|e| DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::EnhancedSequence => {
                crate::structural::sequence::compact::EnhancedModel::from_bytes(bytes)
                    .map(|m| Self::EnhancedSequence(Box::new(m)))
                    .map_err(|e| DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::CompactSequence => {
                crate::structural::sequence::compact::CompactModel::from_bytes(bytes)
                    .map(|m| Self::CompactSequence(Box::new(m)))
                    .map_err(|e| DemandError::Module(e.to_string()))
            }
            #[cfg(feature = "curriculum")]
            ModuleKind::Sequence => crate::structural::sequence::Model::from_bytes(bytes)
                .map(|m| Self::Sequence(Box::new(m)))
                .map_err(|e| DemandError::Module(e.to_string())),
            #[cfg(feature = "curriculum")]
            ModuleKind::Structural => crate::structural::StructuralSeed::from_bytes(bytes)
                .map(|s| Self::Structural(Box::new(s)))
                .map_err(|e| DemandError::Module(e.to_string())),
            ModuleKind::Semantic => crate::semantic::SemanticSeed::from_bytes(bytes)
                .map(|s| Self::Semantic(Box::new(s)))
                .map_err(|e| DemandError::Module(e.to_string())),
            ModuleKind::Numeric => crate::numeric::NumericSkill::from_bytes(bytes)
                .map(|s| Self::Numeric(Box::new(s)))
                .map_err(|e| DemandError::Module(e.to_string())),
            ModuleKind::Circuit => {
                crate::numeric::evolution::constructive::CircuitDna::from_bytes(bytes)
                    .and_then(|dna| dna.program())
                    .map(|program| Self::Circuit(Box::new(program.compile_execution())))
                    .map_err(|e| DemandError::Module(e.to_string()))
            }
            ModuleKind::Study => crate::cumulative::StudySeed::from_bytes(bytes)
                .map(|s| Self::Study(Box::new(s)))
                .map_err(|e| DemandError::Module(e.to_string())),
            ModuleKind::Text => TextSkill::from_bytes(bytes)
                .map(Self::Text)
                .map_err(storage),
            ModuleKind::Behavior => BehaviorModel::from_bytes(bytes)
                .map(Self::Behavior)
                .map_err(|e| DemandError::Module(e.to_string())),
            ModuleKind::World => WorldModel::from_bytes(bytes)
                .map(Self::World)
                .map_err(|e| DemandError::Module(e.to_string())),
            ModuleKind::Evolved => EvolvedSkill::from_bytes(bytes)
                .map(|s| Self::Evolved(Box::new(s)))
                .map_err(|e| DemandError::Module(e.to_string())),
            ModuleKind::Reactive => {
                let s = ReactiveSkill::from_bytes(bytes)
                    .map_err(|e| DemandError::Module(e.to_string()))?;
                let circuit = s
                    .compile()
                    .map_err(|e| DemandError::Module(e.to_string()))?;
                Ok(Self::Reactive(Box::new((s, circuit))))
            }
        }
    }
    pub fn context(&self) -> &str {
        match self {
            Self::Text(m) => m.context(),
            Self::Behavior(m) => m.context(),
            Self::World(m) => m.context(),
            Self::Evolved(m) => m.context(),
            Self::Reactive(m) => m.0.context(),
            Self::Numeric(m) => m.context(),
            Self::Circuit(_) => crate::numeric::evolution::constructive::CONTEXT,
            Self::Study(m) => m.context(),
            Self::Semantic(m) => m.context(),
            #[cfg(feature = "curriculum")]
            Self::Structural(m) => m.context(),
            #[cfg(feature = "curriculum")]
            Self::Sequence(_) => crate::structural::sequence::CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::CompactSequence(_) => crate::structural::sequence::compact::CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::EnhancedSequence(_) => crate::structural::sequence::compact::ENHANCED_CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::FactorizedSequence(_) => crate::structural::sequence::factorized::CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::Dependency(_) => crate::structural::sequence::dependency::CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::RelationNetwork(_) => crate::structural::sequence::relation_network::CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::JointSequence(_) => crate::structural::sequence::joint::CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::LexicalEncoder(_) => crate::structural::sequence::lexical_encoder::CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::SegmentSequence(_) => crate::structural::sequence::segment::CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::RelationalPointer(_) => {
                crate::structural::sequence::lexical_encoder::relational_pointer::CONTEXT
            }
            #[cfg(feature = "curriculum")]
            Self::ContextPointer(_) => crate::structural::sequence::lexical_encoder::relational_pointer::context_read::CONTEXT,
            #[cfg(feature = "curriculum")]
            Self::Synaptic(_) => crate::structural::sequence::lexical_encoder::synaptic::CONTEXT,
        }
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + match self {
                Self::Text(m) => m.retained_bytes() - size_of::<TextSkill>(),
                Self::Behavior(m) => m.retained_bytes() - size_of::<BehaviorModel>(),
                Self::World(m) => m.retained_bytes() - size_of::<WorldModel>(),
                Self::Evolved(m) => m.retained_bytes(),
                Self::Reactive(m) => m.0.retained_bytes() + m.1.retained_bytes(),
                Self::Numeric(m) => m.retained_bytes(),
                Self::Circuit(m) => m.retained_bytes(),
                Self::Study(m) => m.retained_bytes(),
                Self::Semantic(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::Structural(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::Sequence(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::CompactSequence(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::EnhancedSequence(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::FactorizedSequence(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::Dependency(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::RelationNetwork(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::JointSequence(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::LexicalEncoder(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::SegmentSequence(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::RelationalPointer(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::ContextPointer(m) => m.retained_bytes(),
                #[cfg(feature = "curriculum")]
                Self::Synaptic(m) => m.retained_bytes(),
            }
    }
}

/// Type names are assertions supplied by the installer, not inferred semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Descriptor {
    pub id: String,
    pub kind: ModuleKind,
    pub file: String,
    pub context: String,
    pub input: String,
    pub output: String,
    pub cost: u32,
    pub encoded_bytes: usize,
    pub retained_bytes: usize,
    pub identity: u64,
}
pub struct ModuleSpec<'a> {
    pub id: &'a str,
    pub kind: ModuleKind,
    pub file: &'a str,
    pub input: &'a str,
    pub output: &'a str,
    pub cost: u32,
}
#[derive(Debug)]
pub struct Catalog {
    root: PathBuf,
    entries: Vec<Descriptor>,
}
impl Catalog {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let root = if root.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            root
        };
        Self {
            root,
            entries: Vec::new(),
        }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn entries(&self) -> &[Descriptor] {
        &self.entries
    }
    pub fn get(&self, id: &str) -> Option<&Descriptor> {
        self.entries.iter().find(|e| e.id == id)
    }
    pub fn register(
        &mut self,
        spec: ModuleSpec<'_>,
        authority: &Authority,
    ) -> Result<(), DemandError> {
        self.check_registration(&spec)?;
        let ModuleSpec {
            id,
            kind,
            file,
            input,
            output,
            cost,
        } = spec;
        require(authority, Effect::ReadData)?;
        let bytes = read_bounded(&self.root.join(file), kind.maximum()).map_err(storage)?;
        let module = Module::decode(kind, &bytes)?;
        self.entries.push(Descriptor {
            id: id.into(),
            kind,
            file: file.into(),
            context: module.context().into(),
            input: input.into(),
            output: output.into(),
            cost,
            encoded_bytes: bytes.len(),
            retained_bytes: module.retained_bytes(),
            identity: checksum(&bytes),
        });
        Ok(())
    }
    pub fn check_registration(&self, spec: &ModuleSpec<'_>) -> Result<(), DemandError> {
        name(spec.id)?;
        name(spec.input)?;
        name(spec.output)?;
        file_name(spec.file)?;
        if spec.cost == 0 || self.get(spec.id).is_some() {
            return Err(DemandError::Invalid("duplicate module or zero cost"));
        }
        if self.entries.len() >= MAX_MODULES {
            return Err(DemandError::Budget("catalog modules"));
        }
        Ok(())
    }
    /// A new immutable text version must preserve context and cite the old model as its parent.
    /// This is a bounded training/install operation, not a demand-runtime working-set operation.
    pub fn revise_text(
        &mut self,
        id: &str,
        file: &str,
        authority: &Authority,
    ) -> Result<(), DemandError> {
        require(authority, Effect::ReadData)?;
        file_name(file)?;
        let index = self
            .entries
            .iter()
            .position(|e| e.id == id)
            .ok_or_else(|| DemandError::Missing(id.into()))?;
        let old = &self.entries[index];
        if old.kind != ModuleKind::Text {
            return Err(DemandError::Invalid("only text revisions supported"));
        }
        let old_bytes =
            read_bounded(&self.root.join(&old.file), old.encoded_bytes as u64).map_err(storage)?;
        if checksum(&old_bytes) != old.identity {
            return Err(DemandError::Identity);
        }
        let parent = TextSkill::from_bytes(&old_bytes).map_err(storage)?;
        let bytes =
            read_bounded(&self.root.join(file), ModuleKind::Text.maximum()).map_err(storage)?;
        let skill = TextSkill::from_bytes(&bytes).map_err(storage)?;
        if skill.context() != old.context
            || skill.parent() != Some(parent.fingerprint())
            || skill.revision()
                != parent
                    .revision()
                    .checked_add(1)
                    .ok_or(DemandError::Budget("revision"))?
        {
            return Err(DemandError::Identity);
        }
        let module = Module::Text(skill);
        let e = &mut self.entries[index];
        e.file = file.into();
        e.encoded_bytes = bytes.len();
        e.retained_bytes = module.retained_bytes();
        e.identity = checksum(&bytes);
        Ok(())
    }
    pub fn save_new(
        &self,
        path: impl AsRef<Path>,
        authority: &Authority,
    ) -> Result<(), DemandError> {
        require(authority, Effect::WriteData)?;
        // Sibling references remain portable when the whole bundle is moved.
        let path = path.as_ref();
        if std::fs::canonicalize(parent_or_current(path))? != std::fs::canonicalize(&self.root)? {
            return Err(DemandError::Invalid(
                "catalog must be saved alongside modules",
            ));
        }
        publish_new(path, &self.to_bytes()).map_err(storage)
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = b"AXSD0001".to_vec();
        b.extend_from_slice(&(self.entries.len() as u16).to_le_bytes());
        for e in &self.entries {
            b.push(e.kind.tag());
            for s in [&e.id, &e.file, &e.context, &e.input, &e.output] {
                b.extend_from_slice(&(s.len() as u16).to_le_bytes());
                b.extend_from_slice(s.as_bytes());
            }
            b.extend_from_slice(&e.cost.to_le_bytes());
            b.extend_from_slice(&(e.encoded_bytes as u64).to_le_bytes());
            b.extend_from_slice(&(e.retained_bytes as u64).to_le_bytes());
            b.extend_from_slice(&e.identity.to_le_bytes());
        }
        b.extend_from_slice(&checksum(&b).to_le_bytes());
        b
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.to_bytes())
    }
    /// Reads descriptors only; no learned module is opened or decoded.
    pub fn load(path: impl AsRef<Path>, authority: &Authority) -> Result<Self, DemandError> {
        require(authority, Effect::ReadData)?;
        let path = path.as_ref();
        let b = read_bounded(path, MAX_CATALOG).map_err(storage)?;
        Self::from_bytes(&b, parent_or_current(path))
    }
    /// Decode the same portable catalog contract without opening any payload files.
    pub fn from_bytes(b: &[u8], root: impl Into<PathBuf>) -> Result<Self, DemandError> {
        if b.len() < 18
            || b.len() as u64 > MAX_CATALOG
            || &b[..8] != b"AXSD0001"
            || checksum(&b[..b.len() - 8])
                != u64::from_le_bytes(b[b.len() - 8..].try_into().expect("width"))
        {
            return Err(DemandError::Invalid("catalog header/checksum"));
        }
        let mut r = Reader(&b[8..b.len() - 8]);
        let n = r.u16()? as usize;
        if n > MAX_MODULES {
            return Err(DemandError::Budget("catalog modules"));
        }
        let mut result = Self::new(root);
        for _ in 0..n {
            let kind = match r.take(1)?[0] {
                0 => ModuleKind::Text,
                1 => ModuleKind::Behavior,
                2 => ModuleKind::World,
                3 => ModuleKind::Evolved,
                4 => ModuleKind::Reactive,
                5 => ModuleKind::Numeric,
                6 => ModuleKind::Study,
                7 => ModuleKind::Semantic,
                18 => ModuleKind::Circuit,
                #[cfg(feature = "curriculum")]
                8 => ModuleKind::Structural,
                #[cfg(feature = "curriculum")]
                9 => ModuleKind::Sequence,
                #[cfg(feature = "curriculum")]
                10 => ModuleKind::CompactSequence,
                #[cfg(feature = "curriculum")]
                11 => ModuleKind::EnhancedSequence,
                #[cfg(feature = "curriculum")]
                12 => ModuleKind::FactorizedSequence,
                #[cfg(feature = "curriculum")]
                13 => ModuleKind::Dependency,
                #[cfg(feature = "curriculum")]
                14 => ModuleKind::RelationNetwork,
                #[cfg(feature = "curriculum")]
                15 => ModuleKind::JointSequence,
                #[cfg(feature = "curriculum")]
                16 => ModuleKind::LexicalEncoder,
                #[cfg(feature = "curriculum")]
                17 => ModuleKind::SegmentSequence,
                #[cfg(feature = "curriculum")]
                19 => ModuleKind::RelationalPointer,
                #[cfg(feature = "curriculum")]
                20 => ModuleKind::ContextPointer,
                #[cfg(feature = "curriculum")]
                21 => ModuleKind::Synaptic,
                _ => return Err(DemandError::Invalid("module kind")),
            };
            let id = r.text()?;
            let file = r.text()?;
            let context = r.text()?;
            let input = r.text()?;
            let output = r.text()?;
            let cost = u32::from_le_bytes(r.take(4)?.try_into().expect("width"));
            let encoded_bytes =
                usize::try_from(r.u64()?).map_err(|_| DemandError::Invalid("encoded size"))?;
            let retained_bytes =
                usize::try_from(r.u64()?).map_err(|_| DemandError::Invalid("resident size"))?;
            let identity = r.u64()?;
            file_name(&file)?;
            for s in [&id, &context, &input, &output] {
                name(s)?;
            }
            if cost == 0
                || encoded_bytes == 0
                || encoded_bytes as u64 > kind.maximum()
                || retained_bytes < size_of::<Module>()
                || retained_bytes > 67_108_864
                || result.get(&id).is_some()
            {
                return Err(DemandError::Invalid("module descriptor"));
            }
            result.entries.push(Descriptor {
                id,
                kind,
                file,
                context,
                input,
                output,
                cost,
                encoded_bytes,
                retained_bytes,
                identity,
            });
        }
        if !r.0.is_empty() {
            return Err(DemandError::Invalid("trailing catalog data"));
        }
        Ok(result)
    }
    pub fn dormant_payload_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.encoded_bytes).sum()
    }
    pub fn eager_retained_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.retained_bytes).sum()
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.root.capacity()
            + self.entries.capacity() * size_of::<Descriptor>()
            + self
                .entries
                .iter()
                .map(|e| {
                    e.id.capacity()
                        + e.file.capacity()
                        + e.context.capacity()
                        + e.input.capacity()
                        + e.output.capacity()
                })
                .sum::<usize>()
    }
}
struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], DemandError> {
        if n > self.0.len() {
            return Err(DemandError::Invalid("truncated catalog"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u16(&mut self) -> Result<u16, DemandError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("width")))
    }
    fn u64(&mut self) -> Result<u64, DemandError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("width")))
    }
    fn text(&mut self) -> Result<String, DemandError> {
        let n = self.u16()? as usize;
        if n > 256 {
            return Err(DemandError::Invalid("descriptor text length"));
        }
        let s = std::str::from_utf8(self.take(n)?)
            .map_err(|_| DemandError::Invalid("descriptor UTF-8"))?
            .to_owned();
        name(&s)?;
        Ok(s)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPlan {
    pub input: String,
    pub goal: String,
    pub modules: Vec<String>,
    pub cost: u64,
    identities: Vec<u64>,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DemandStats {
    pub loads: u64,
    pub hits: u64,
    pub evictions: u64,
    pub file_reads: u64,
    pub bytes_read: u64,
    pub load_ns: u128,
    pub active_bytes: usize,
    pub peak_active_bytes: usize,
    pub active_modules: usize,
    pub peak_active_modules: usize,
}
struct Slot {
    module: Option<Box<Module>>,
    used: u64,
}
pub struct DemandRuntime {
    catalog: Catalog,
    seed: Seed,
    slots: Vec<Slot>,
    tick: u64,
    stats: DemandStats,
}
impl DemandRuntime {
    pub fn new(catalog: Catalog, seed: Seed) -> Result<Self, DemandError> {
        seed.validate()?;
        let slots = (0..catalog.entries.len())
            .map(|_| Slot {
                module: None,
                used: 0,
            })
            .collect();
        Ok(Self {
            catalog,
            seed,
            slots,
            tick: 0,
            stats: DemandStats::default(),
        })
    }
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }
    pub fn seed(&self) -> Seed {
        self.seed
    }
    pub fn stats(&self) -> DemandStats {
        self.stats
    }
    pub fn metadata_bytes(&self) -> usize {
        size_of::<Self>() - size_of::<Catalog>()
            + self.catalog.retained_bytes()
            + self.slots.capacity() * size_of::<Slot>()
    }
    pub fn active_ids(&self) -> Vec<&str> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.module.is_some())
            .map(|(i, _)| self.catalog.entries[i].id.as_str())
            .collect()
    }
    fn index(&self, id: &str, context: &str) -> Result<usize, DemandError> {
        let i = self
            .catalog
            .entries
            .iter()
            .position(|e| e.id == id)
            .ok_or_else(|| DemandError::Missing(id.into()))?;
        if self.catalog.entries[i].context != context {
            return Err(DemandError::Context);
        }
        Ok(i)
    }
    fn evict(&mut self, i: usize) {
        if let Some(m) = self.slots[i].module.take() {
            self.stats.active_bytes -= m.retained_bytes();
            self.stats.active_modules -= 1;
            self.stats.evictions += 1;
        }
    }
    pub fn unload_all(&mut self) {
        for i in 0..self.slots.len() {
            self.evict(i);
        }
    }
    pub fn predict_numeric(
        &mut self,
        id: &str,
        context: &str,
        input: &[f64],
        authority: &Authority,
    ) -> Result<f64, DemandError> {
        if !(1..=4).contains(&input.len()) || input.iter().any(|n| !n.is_finite()) {
            return Err(DemandError::Invalid("finite numeric inputs"));
        }
        self.with_module(id, context, authority, |module| {
            let result = match module {
                Module::Circuit(program) => program.evaluate(input),
                Module::Numeric(skill) => skill.predict(context, input),
                Module::Evolved(skill) => skill.predict(context, input),
                _ => return Err(DemandError::Invalid("numeric module required")),
            };
            result.map_err(|e| DemandError::Module(e.to_string()))
        })
    }
    /// The closure borrow pins the module; safe Rust cannot evict or leak a reference from it.
    pub fn with_module<R>(
        &mut self,
        id: &str,
        context: &str,
        authority: &Authority,
        operation: impl FnOnce(&Module) -> Result<R, DemandError>,
    ) -> Result<R, DemandError> {
        require(authority, Effect::ReadData)?;
        let i = self.index(id, context)?;
        self.tick = self
            .tick
            .checked_add(1)
            .ok_or(DemandError::Budget("access counter"))?;
        if self.slots[i].module.is_none() {
            let e = &self.catalog.entries[i];
            if e.retained_bytes > self.seed.active_bytes {
                return Err(DemandError::Budget("single module residency"));
            }
            let reserve = e.retained_bytes;
            while self.stats.active_bytes + reserve > self.seed.active_bytes
                || self.stats.active_modules >= self.seed.active_modules
            {
                let oldest = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.module.is_some())
                    .min_by_key(|(_, s)| s.used)
                    .map(|(j, _)| j)
                    .expect("resident entries exist");
                self.evict(oldest);
            }
            let e = &self.catalog.entries[i];
            let start = Instant::now();
            self.stats.file_reads += 1;
            let b = read_bounded(&self.catalog.root.join(&e.file), e.encoded_bytes as u64)
                .map_err(storage)?;
            self.stats.bytes_read += b.len() as u64;
            if b.len() != e.encoded_bytes || checksum(&b) != e.identity {
                return Err(DemandError::Identity);
            }
            let m = Module::decode_reserved(e.kind, &b, reserve)?;
            if m.context() != e.context || m.retained_bytes() != reserve {
                return Err(DemandError::Identity);
            }
            self.slots[i].module = Some(Box::new(m));
            self.stats.load_ns += start.elapsed().as_nanos();
            self.stats.loads += 1;
            self.stats.active_bytes += reserve;
            self.stats.active_modules += 1;
            self.stats.peak_active_bytes =
                self.stats.peak_active_bytes.max(self.stats.active_bytes);
            self.stats.peak_active_modules = self
                .stats
                .peak_active_modules
                .max(self.stats.active_modules);
        } else {
            self.stats.hits += 1;
        }
        self.slots[i].used = self.tick;
        operation(self.slots[i].module.as_deref().expect("admitted"))
    }
    /// Uniform-cost search over declared contracts, without loading model payloads.
    pub fn plan_text(&self, input: &str, goal: &str) -> Result<TextPlan, DemandError> {
        name(input)?;
        name(goal)?;
        let mut frontier = BTreeMap::from([(input.to_owned(), (0_u64, Vec::<usize>::new()))]);
        let mut done = BTreeSet::new();
        while let Some((current, (cost, path))) = frontier
            .iter()
            .filter(|(k, _)| !done.contains(k.as_str()))
            .min_by_key(|(k, (c, _))| (*c, k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            if current == goal {
                return Ok(TextPlan {
                    input: input.into(),
                    goal: goal.into(),
                    cost,
                    modules: path
                        .iter()
                        .map(|&i| self.catalog.entries[i].id.clone())
                        .collect(),
                    identities: path
                        .iter()
                        .map(|&i| self.catalog.entries[i].identity)
                        .collect(),
                });
            }
            done.insert(current.clone());
            for (i, e) in self.catalog.entries.iter().enumerate() {
                if e.kind != ModuleKind::Text
                    || e.input != current
                    || e.retained_bytes > self.seed.active_bytes
                {
                    continue;
                }
                let Some(next_cost) = cost.checked_add(u64::from(e.cost)) else {
                    continue;
                };
                if next_cost > self.seed.max_plan_cost {
                    continue;
                }
                if frontier.get(&e.output).is_none_or(|(c, _)| next_cost < *c) {
                    let mut next_path = path.clone();
                    next_path.push(i);
                    frontier.insert(e.output.clone(), (next_cost, next_path));
                }
            }
        }
        Err(DemandError::NoRoute)
    }
    fn validate_plan(&self, plan: &TextPlan) -> Result<(), DemandError> {
        name(&plan.input)?;
        name(&plan.goal)?;
        if plan.modules.len() != plan.identities.len() || plan.modules.len() > MAX_MODULES {
            return Err(DemandError::Invalid("plan dimensions"));
        }
        let mut current = plan.input.as_str();
        let mut cost = 0_u64;
        for (id, identity) in plan.modules.iter().zip(&plan.identities) {
            let e = self
                .catalog
                .get(id)
                .ok_or_else(|| DemandError::Missing(id.clone()))?;
            if e.identity != *identity || e.kind != ModuleKind::Text || e.input != current {
                return Err(DemandError::Identity);
            }
            current = &e.output;
            cost = cost
                .checked_add(u64::from(e.cost))
                .ok_or(DemandError::Budget("plan cost"))?;
        }
        if current != plan.goal || cost != plan.cost || cost > self.seed.max_plan_cost {
            return Err(DemandError::Invalid("goal/cost contract"));
        }
        Ok(())
    }
    fn batch_size(&self, values: &[String]) -> Result<usize, DemandError> {
        if values.len() > self.seed.max_rows {
            return Err(DemandError::Budget("batch rows"));
        }
        if values.iter().any(|s| s.len() > self.seed.max_value_bytes) {
            return Err(DemandError::Budget("value bytes"));
        }
        let n = values
            .iter()
            .try_fold(std::mem::size_of_val(values), |n, s| {
                n.checked_add(s.capacity())
            })
            .ok_or(DemandError::Budget("batch overflow"))?;
        if n > self.seed.batch_bytes {
            return Err(DemandError::Budget("batch retained bytes"));
        }
        Ok(n)
    }
    /// Stage-major evaluation: a batch does not cause a cold load for each row.
    pub fn execute_text(
        &mut self,
        plan: &TextPlan,
        input: &[String],
        authority: &Authority,
    ) -> Result<Vec<String>, DemandError> {
        self.execute_text_impl(plan, input, authority, false)
    }
    /// Diagnostic baseline: retain the original explanatory interpreter's decisions.
    pub fn execute_text_with_witnesses(
        &mut self,
        plan: &TextPlan,
        input: &[String],
        authority: &Authority,
    ) -> Result<Vec<String>, DemandError> {
        self.execute_text_impl(plan, input, authority, true)
    }
    fn execute_text_impl(
        &mut self,
        plan: &TextPlan,
        input: &[String],
        authority: &Authority,
        explain: bool,
    ) -> Result<Vec<String>, DemandError> {
        self.validate_plan(plan)?;
        self.batch_size(input)?;
        require(authority, Effect::ReadData)?;
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let mut values = input.to_vec();
        for id in &plan.modules {
            let context = self.catalog.get(id).expect("validated").context.clone();
            let limit = self.seed;
            let old_bytes = self.batch_size(&values)?;
            values = self.with_module(id, &context, authority, |module| {
                let Module::Text(skill) = module else {
                    return Err(DemandError::Invalid("text module expected"));
                };
                let mut next = Vec::with_capacity(values.len());
                let mut bytes = next.capacity() * size_of::<String>();
                if old_bytes + bytes > limit.batch_bytes {
                    return Err(DemandError::Budget("overlapping batches"));
                }
                for (row, v) in values.iter().enumerate() {
                    let output = if explain {
                        let TextPrediction::Agreed(w) =
                            skill.predict(&context, v).map_err(storage)?
                        else {
                            return Err(DemandError::NeedsExample {
                                row: row + 1,
                                module: id.clone(),
                                input: v.clone(),
                            });
                        };
                        w.output
                    } else {
                        let TextValuePrediction::Agreed(s) =
                            skill.predict_value(&context, v).map_err(storage)?
                        else {
                            return Err(DemandError::NeedsExample {
                                row: row + 1,
                                module: id.clone(),
                                input: v.clone(),
                            });
                        };
                        s
                    };
                    if output.len() > limit.max_value_bytes {
                        return Err(DemandError::Budget("output bytes"));
                    }
                    bytes = bytes
                        .checked_add(output.capacity())
                        .ok_or(DemandError::Budget("batch overflow"))?;
                    if old_bytes + bytes > limit.batch_bytes {
                        return Err(DemandError::Budget("overlapping batches"));
                    }
                    next.push(output);
                }
                Ok(next)
            })?;
        }
        Ok(values)
    }
    pub fn behavior_cursor(&self, id: &str, context: &str) -> Result<DormantCursor, DemandError> {
        let e = &self.catalog.entries[self.index(id, context)?];
        if e.kind != ModuleKind::Behavior {
            return Err(DemandError::Invalid("behavior module expected"));
        }
        Ok(DormantCursor {
            identity: e.identity,
            state: 0,
            observations: 0,
            suspended: false,
        })
    }
    pub fn observe_behavior(
        &mut self,
        id: &str,
        context: &str,
        cursor: &mut DormantCursor,
        action: u8,
        output: u16,
        authority: &Authority,
    ) -> Result<(), DemandError> {
        if cursor.suspended {
            return Err(DemandError::Suspended);
        }
        let e = &self.catalog.entries[self.index(id, context)?];
        if e.identity != cursor.identity {
            return Err(DemandError::Identity);
        }
        self.with_module(id, context, authority, |m| {
            let Module::Behavior(model) = m else {
                return Err(DemandError::Invalid("behavior module expected"));
            };
            let edge = model
                .edge(cursor.state, action)
                .map_err(|e| DemandError::Module(e.to_string()))?;
            cursor.observations = cursor
                .observations
                .checked_add(1)
                .ok_or(DemandError::Budget("cursor observations"))?;
            if edge.output != output {
                cursor.suspended = true;
                return Err(DemandError::Drift {
                    expected: edge.output,
                    actual: output,
                });
            }
            cursor.state = edge.next;
            Ok(())
        })
    }
    pub fn plan_behavior(
        &mut self,
        id: &str,
        context: &str,
        cursor: &DormantCursor,
        goal: u16,
        forbidden: &BTreeSet<u16>,
        authority: &Authority,
    ) -> Result<Option<Vec<u8>>, DemandError> {
        if cursor.suspended {
            return Err(DemandError::Suspended);
        }
        let e = &self.catalog.entries[self.index(id, context)?];
        if e.identity != cursor.identity {
            return Err(DemandError::Identity);
        }
        self.with_module(id, context, authority, |m| {
            let Module::Behavior(model) = m else {
                return Err(DemandError::Invalid("behavior module expected"));
            };
            model
                .plan_from(cursor.state, goal, forbidden)
                .map_err(|e| DemandError::Module(e.to_string()))
        })
    }
}

/// Continuity belongs to a small cursor, not to the evictable model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DormantCursor {
    identity: u64,
    state: u16,
    observations: u64,
    suspended: bool,
}
impl DormantCursor {
    pub fn state(&self) -> u16 {
        self.state
    }
    pub fn observations(&self) -> u64 {
        self.observations
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    pub fn save_new(
        &self,
        path: impl AsRef<Path>,
        authority: &Authority,
    ) -> Result<(), DemandError> {
        require(authority, Effect::WriteData)?;
        let mut b = b"AXCR0001".to_vec();
        b.extend_from_slice(&self.identity.to_le_bytes());
        b.extend_from_slice(&self.state.to_le_bytes());
        b.extend_from_slice(&self.observations.to_le_bytes());
        b.push(u8::from(self.suspended));
        b.extend_from_slice(&checksum(&b).to_le_bytes());
        publish_new(path.as_ref(), &b).map_err(storage)
    }
    pub fn load(path: impl AsRef<Path>, authority: &Authority) -> Result<Self, DemandError> {
        require(authority, Effect::ReadData)?;
        let b = read_bounded(path.as_ref(), 35).map_err(storage)?;
        if b.len() != 35
            || &b[..8] != b"AXCR0001"
            || b[26] > 1
            || checksum(&b[..27]) != u64::from_le_bytes(b[27..].try_into().expect("width"))
        {
            return Err(DemandError::Invalid("cursor checksum/format"));
        }
        let state = u16::from_le_bytes(b[16..18].try_into().expect("width"));
        if state >= 128 {
            return Err(DemandError::Invalid("cursor state"));
        }
        Ok(Self {
            identity: u64::from_le_bytes(b[8..16].try_into().expect("width")),
            state,
            observations: u64::from_le_bytes(b[18..26].try_into().expect("width")),
            suspended: b[26] == 1,
        })
    }
}

/// An innate exact reducer; only unsigned decimal integers belong to this contract.
pub fn exact_decimal_sum(values: &[String]) -> Result<u128, DemandError> {
    values.iter().try_fold(0_u128, |sum, s| {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(DemandError::Invalid("unsigned decimal required"));
        }
        let n = s
            .parse::<u128>()
            .map_err(|_| DemandError::Budget("decimal overflow"))?;
        sum.checked_add(n)
            .ok_or(DemandError::Budget("sum overflow"))
    })
}
