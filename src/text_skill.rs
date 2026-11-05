use crate::{
    TextBudget, TextExample, TextForgeError, TextPrediction, TextProgram, TextValuePrediction,
    live_store::checksum,
    text_forge::{
        Atom, Case, Edge, Label, Node, Selector, TokenClass, validate_budget, validate_example,
    },
};
use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

const MAGIC: &[u8; 8] = b"AXTX0001";
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum TextSkillError {
    Io(std::io::Error),
    Forge(TextForgeError),
    Invalid(&'static str),
    ContextMismatch,
}
impl fmt::Display for TextSkillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "text skill storage error: {e}"),
            Self::Forge(e) => e.fmt(f),
            Self::Invalid(s) => write!(f, "invalid text skill: {s}"),
            Self::ContextMismatch => {
                f.write_str("text skill context does not match the declared task")
            }
        }
    }
}
impl std::error::Error for TextSkillError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Forge(e) => Some(e),
            _ => None,
        }
    }
}
impl From<std::io::Error> for TextSkillError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<TextForgeError> for TextSkillError {
    fn from(e: TextForgeError) -> Self {
        Self::Forge(e)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextSkill {
    name: String,
    context: String,
    revision: u32,
    parent: Option<u64>,
    examples: Vec<TextExample>,
    program: TextProgram,
}
impl TextSkill {
    pub fn learn(
        name: impl Into<String>,
        context: impl Into<String>,
        examples: Vec<TextExample>,
        budget: TextBudget,
    ) -> Result<Self, TextSkillError> {
        let name = name.into();
        let context = context.into();
        identifier(&name)?;
        identifier(&context)?;
        let program = TextProgram::learn(&examples, budget)?;
        let skill = Self {
            name,
            context,
            revision: 1,
            parent: None,
            examples,
            program,
        };
        skill.validate()?;
        Ok(skill)
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn revision(&self) -> u32 {
        self.revision
    }
    pub fn parent(&self) -> Option<u64> {
        self.parent
    }
    pub fn examples(&self) -> &[TextExample] {
        &self.examples
    }
    pub fn program(&self) -> &TextProgram {
        &self.program
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.payload())
    }
    pub fn predict(&self, context: &str, input: &str) -> Result<TextPrediction, TextSkillError> {
        if context != self.context {
            return Err(TextSkillError::ContextMismatch);
        }
        Ok(self.program.predict(input)?)
    }
    pub fn predict_value(
        &self,
        context: &str,
        input: &str,
    ) -> Result<TextValuePrediction, TextSkillError> {
        if context != self.context {
            return Err(TextSkillError::ContextMismatch);
        }
        Ok(self.program.predict_value(input)?)
    }
    pub fn refine(
        &self,
        context: &str,
        additions: &[TextExample],
        replace: bool,
    ) -> Result<Self, TextSkillError> {
        if context != self.context {
            return Err(TextSkillError::ContextMismatch);
        }
        if additions.is_empty() || additions.len() > self.program.budget.max_examples {
            return Err(TextSkillError::Invalid("refinement size"));
        }
        let mut examples = self.examples.clone();
        let mut seen = std::collections::BTreeMap::new();
        for example in additions {
            validate_example(example, self.program.budget)?;
            if seen
                .insert(&example.input, &example.output)
                .is_some_and(|output| output != &example.output)
            {
                return Err(TextForgeError::ContradictoryExamples.into());
            }
            if let Some(old) = examples.iter_mut().find(|old| old.input == example.input) {
                if old.output != example.output {
                    if !replace {
                        return Err(TextForgeError::ContradictoryExamples.into());
                    }
                    old.output = example.output.clone();
                }
            } else {
                examples.push(example.clone());
            }
        }
        let mut skill = Self::learn(&self.name, &self.context, examples, self.program.budget)?;
        skill.revision = self
            .revision
            .checked_add(1)
            .ok_or(TextSkillError::Invalid("revision overflow"))?;
        skill.parent = Some(self.fingerprint());
        Ok(skill)
    }
    pub fn save_new(&self, path: impl AsRef<Path>) -> Result<u64, TextSkillError> {
        let bytes = self.to_bytes()?;
        publish_new(path.as_ref(), &bytes)?;
        Ok(bytes.len() as u64)
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, TextSkillError> {
        self.validate()?;
        let payload = self.payload();
        let mut bytes = Vec::with_capacity(payload.len() + 24);
        let length = payload.len() as u32;
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(&(!length).to_le_bytes());
        bytes.extend_from_slice(&payload);
        let hash = checksum(&bytes);
        bytes.extend_from_slice(&hash.to_le_bytes());
        if bytes.len() as u64 > MAX_FILE_BYTES {
            return Err(TextSkillError::Invalid("file limit"));
        }
        Ok(bytes)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TextSkillError> {
        let bytes = read_bounded(path.as_ref(), MAX_FILE_BYTES)?;
        Self::from_bytes(&bytes)
    }
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, TextSkillError> {
        if bytes.len() as u64 > MAX_FILE_BYTES {
            return Err(TextSkillError::Invalid("file limit"));
        }
        if bytes.len() < 24 || &bytes[..8] != MAGIC {
            return Err(TextSkillError::Invalid("header"));
        }
        let length = u32::from_le_bytes(bytes[8..12].try_into().expect("fixed width"));
        let complement = u32::from_le_bytes(bytes[12..16].try_into().expect("fixed width"));
        if length != !complement || length as usize != bytes.len() - 24 {
            return Err(TextSkillError::Invalid("payload length"));
        }
        let hash = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().expect("fixed width"));
        if checksum(&bytes[..bytes.len() - 8]) != hash {
            return Err(TextSkillError::Invalid("checksum"));
        }
        let skill = Self::decode(&bytes[16..bytes.len() - 8])?;
        skill.validate()?;
        Ok(skill)
    }
    /// Retained Rust payload, including spare capacity; excludes allocator metadata and scratch.
    pub fn retained_bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<Self>()
            + self.name.capacity()
            + self.context.capacity()
            + self.examples.capacity() * size_of::<TextExample>()
            + self
                .examples
                .iter()
                .map(|e| e.input.capacity() + e.output.capacity())
                .sum::<usize>()
            + self.program.atoms.capacity() * size_of::<Atom>()
            + self
                .program
                .atoms
                .iter()
                .map(|a| match &a.selector {
                    Selector::Field { delimiter, .. } => delimiter.capacity(),
                    _ => 0,
                })
                .sum::<usize>()
            + self.program.nodes.capacity() * size_of::<Node>()
            + self
                .program
                .nodes
                .iter()
                .map(|n| {
                    n.edges.capacity() * size_of::<Edge>()
                        + n.edges
                            .iter()
                            .map(|e| match &e.label {
                                Label::Literal(s) => s.capacity(),
                                Label::Atoms(ids) => ids.capacity() * size_of::<usize>(),
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
    fn validate(&self) -> Result<(), TextSkillError> {
        identifier(&self.name)?;
        identifier(&self.context)?;
        if self.revision == 0 || (self.revision == 1) != self.parent.is_none() {
            return Err(TextSkillError::Invalid("revision chain"));
        }
        if self.examples.is_empty() || self.examples.len() > self.program.budget.max_examples {
            return Err(TextSkillError::Invalid("examples count"));
        }
        self.program.validate()?;
        if self.encoded_length() as u64 + 24 > MAX_FILE_BYTES {
            return Err(TextSkillError::Invalid("file limit"));
        }
        for example in &self.examples {
            validate_example(example, self.program.budget)?;
            if !matches!(self.program.predict(&example.input)?, TextPrediction::Agreed(w) if w.output == example.output)
            {
                return Err(TextSkillError::Invalid("demonstration replay"));
            }
        }
        Ok(())
    }
    fn payload(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_text(&mut bytes, &self.name);
        put_text(&mut bytes, &self.context);
        put_u32(&mut bytes, self.revision);
        bytes.push(u8::from(self.parent.is_some()));
        bytes.extend_from_slice(&self.parent.unwrap_or(0).to_le_bytes());
        let b = self.program.budget;
        for n in [
            b.max_examples,
            b.max_input_bytes,
            b.max_output_bytes,
            b.max_pieces,
            b.max_nodes,
            b.max_edges,
            b.max_atoms,
            b.max_delimiters,
        ] {
            put_u32(&mut bytes, n as u32);
        }
        bytes.extend_from_slice(&b.max_work.to_le_bytes());
        put_u32(&mut bytes, self.program.pieces as u32);
        bytes.extend_from_slice(&self.program.work.to_le_bytes());
        put_u32(&mut bytes, self.examples.len() as u32);
        for example in &self.examples {
            put_text(&mut bytes, &example.input);
            put_text(&mut bytes, &example.output);
        }
        put_u32(&mut bytes, self.program.atoms.len() as u32);
        for atom in &self.program.atoms {
            match &atom.selector {
                Selector::Whole => bytes.push(0),
                Selector::Field {
                    delimiter,
                    index,
                    reverse,
                } => {
                    bytes.push(1);
                    put_text(&mut bytes, delimiter);
                    bytes.push(*index);
                    bytes.push(u8::from(*reverse));
                }
                Selector::Token {
                    class,
                    index,
                    reverse,
                } => {
                    bytes.push(2);
                    bytes.push(match class {
                        TokenClass::Word => 0,
                        TokenClass::Letters => 1,
                        TokenClass::Digits => 2,
                    });
                    bytes.push(*index);
                    bytes.push(u8::from(*reverse));
                }
            }
            bytes.push(u8::from(atom.trim));
            bytes.push(u8::from(atom.collapse));
            bytes.push(match atom.case {
                Case::Preserve => 0,
                Case::Lower => 1,
                Case::Upper => 2,
            });
        }
        put_u32(&mut bytes, self.program.nodes.len() as u32);
        for node in &self.program.nodes {
            bytes.push(u8::from(node.terminal));
            put_u32(&mut bytes, node.edges.len() as u32);
            for edge in &node.edges {
                put_u32(&mut bytes, edge.next as u32);
                match &edge.label {
                    Label::Literal(value) => {
                        bytes.push(0);
                        put_text(&mut bytes, value);
                    }
                    Label::Atoms(ids) => {
                        bytes.push(1);
                        put_u32(&mut bytes, ids.len() as u32);
                        for id in ids {
                            put_u32(&mut bytes, *id as u32);
                        }
                    }
                }
            }
        }
        bytes
    }
    fn encoded_length(&self) -> usize {
        85 + self.name.len()
            + self.context.len()
            + self
                .examples
                .iter()
                .map(|e| 8 + e.input.len() + e.output.len())
                .sum::<usize>()
            + self
                .program
                .atoms
                .iter()
                .map(|a| match &a.selector {
                    Selector::Whole => 4,
                    Selector::Field { delimiter, .. } => 10 + delimiter.len(),
                    Selector::Token { .. } => 7,
                })
                .sum::<usize>()
            + self
                .program
                .nodes
                .iter()
                .map(|n| {
                    5 + n
                        .edges
                        .iter()
                        .map(|e| {
                            9 + match &e.label {
                                Label::Literal(s) => s.len(),
                                Label::Atoms(ids) => 4 * ids.len(),
                            }
                        })
                        .sum::<usize>()
                })
                .sum::<usize>()
    }
    fn decode(bytes: &[u8]) -> Result<Self, TextSkillError> {
        let mut r = Reader(bytes);
        let name = r.text(256)?;
        let context = r.text(256)?;
        let revision = r.u32()?;
        let has_parent = r.boolean()?;
        let parent = r.u64()?;
        if !has_parent && parent != 0 {
            return Err(TextSkillError::Invalid("parent field"));
        }
        let mut numbers = [0_usize; 8];
        for n in &mut numbers {
            *n = r.u32()? as usize;
        }
        let budget = TextBudget {
            max_examples: numbers[0],
            max_input_bytes: numbers[1],
            max_output_bytes: numbers[2],
            max_pieces: numbers[3],
            max_nodes: numbers[4],
            max_edges: numbers[5],
            max_atoms: numbers[6],
            max_delimiters: numbers[7],
            max_work: r.u64()?,
        };
        validate_budget(budget)?;
        let pieces = r.count(budget.max_pieces)?;
        let work = r.u64()?;
        let count = r.count(budget.max_examples)?;
        let mut examples = Vec::new();
        for _ in 0..count {
            examples.push(TextExample::new(
                r.text(budget.max_input_bytes)?,
                r.text(budget.max_output_bytes)?,
            ));
        }
        let count = r.count(budget.max_atoms)?;
        let mut atoms = Vec::new();
        for _ in 0..count {
            let selector = match r.byte()? {
                0 => Selector::Whole,
                1 => Selector::Field {
                    delimiter: r.text(12)?,
                    index: r.byte()?,
                    reverse: r.boolean()?,
                },
                2 => Selector::Token {
                    class: match r.byte()? {
                        0 => TokenClass::Word,
                        1 => TokenClass::Letters,
                        2 => TokenClass::Digits,
                        _ => return Err(TextSkillError::Invalid("token class")),
                    },
                    index: r.byte()?,
                    reverse: r.boolean()?,
                },
                _ => return Err(TextSkillError::Invalid("selector kind")),
            };
            atoms.push(Atom {
                selector,
                trim: r.boolean()?,
                collapse: r.boolean()?,
                case: match r.byte()? {
                    0 => Case::Preserve,
                    1 => Case::Lower,
                    2 => Case::Upper,
                    _ => return Err(TextSkillError::Invalid("case kind")),
                },
            });
        }
        let count = r.count(budget.max_nodes)?;
        let mut nodes = Vec::new();
        let mut total_edges = 0;
        for _ in 0..count {
            let terminal = r.boolean()?;
            let count = r.count(budget.max_edges)?;
            total_edges += count;
            if total_edges > budget.max_edges {
                return Err(TextSkillError::Invalid("edge limit"));
            }
            let mut edges = Vec::new();
            for _ in 0..count {
                let next = r.u32()? as usize;
                let label = match r.byte()? {
                    0 => Label::Literal(r.text(budget.max_output_bytes)?),
                    1 => {
                        let count = r.count(atoms.len())?;
                        let mut ids = Vec::new();
                        for _ in 0..count {
                            ids.push(r.u32()? as usize);
                        }
                        Label::Atoms(ids)
                    }
                    _ => return Err(TextSkillError::Invalid("label kind")),
                };
                edges.push(Edge { next, label });
            }
            nodes.push(Node { terminal, edges });
        }
        if !r.0.is_empty() {
            return Err(TextSkillError::Invalid("trailing payload"));
        }
        Ok(Self {
            name,
            context,
            revision,
            parent: has_parent.then_some(parent),
            examples,
            program: TextProgram {
                atoms,
                nodes,
                budget,
                pieces,
                work,
            },
        })
    }
}
fn identifier(value: &str) -> Result<(), TextSkillError> {
    if value.is_empty()
        || value.len() > 256
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(TextSkillError::Invalid("name or context"));
    }
    Ok(())
}
fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_text(bytes: &mut Vec<u8>, value: &str) {
    put_u32(bytes, value.len() as u32);
    bytes.extend_from_slice(value.as_bytes());
}
struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take(&mut self, length: usize) -> Result<&[u8], TextSkillError> {
        if length > self.0.len() {
            return Err(TextSkillError::Invalid("truncated payload"));
        }
        let (value, rest) = self.0.split_at(length);
        self.0 = rest;
        Ok(value)
    }
    fn byte(&mut self) -> Result<u8, TextSkillError> {
        Ok(self.take(1)?[0])
    }
    fn boolean(&mut self) -> Result<bool, TextSkillError> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(TextSkillError::Invalid("boolean")),
        }
    }
    fn u32(&mut self) -> Result<u32, TextSkillError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed width"),
        ))
    }
    fn u64(&mut self) -> Result<u64, TextSkillError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed width"),
        ))
    }
    fn count(&mut self, maximum: usize) -> Result<usize, TextSkillError> {
        let n = self.u32()? as usize;
        if n > maximum {
            Err(TextSkillError::Invalid("count limit"))
        } else {
            Ok(n)
        }
    }
    fn text(&mut self, maximum: usize) -> Result<String, TextSkillError> {
        let n = self.count(maximum)?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| TextSkillError::Invalid("UTF-8"))
    }
}
pub fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, TextSkillError> {
    let file = File::open(path)?;
    if file.metadata()?.len() > maximum {
        return Err(TextSkillError::Invalid("input file limit"));
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err(TextSkillError::Invalid("input grew beyond limit"));
    }
    Ok(bytes)
}
// Atomic no-clobber publication; directory durability after power loss is not established.
pub(crate) fn publish_new(path: &Path, bytes: &[u8]) -> Result<(), TextSkillError> {
    let path = std::path::absolute(path)?;
    let parent = path
        .parent()
        .ok_or(TextSkillError::Invalid("output path"))?;
    fs::create_dir_all(parent)?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let staged = parent.join(format!(".axon-text-stage-{}-{nonce}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged)?;
    let result = (|| -> Result<(), TextSkillError> {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::hard_link(&staged, &path)?;
        Ok(())
    })();
    let _ = fs::remove_file(&staged);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn skill() -> TextSkill {
        TextSkill::learn(
            "case",
            "strings",
            vec![
                TextExample::new("Alpha", "ALPHA"),
                TextExample::new("Beta", "BETA"),
            ],
            TextBudget::default(),
        )
        .unwrap()
    }
    #[test]
    fn encoded_size_is_exact_including_revision_metadata() {
        let old = skill();
        assert_eq!(old.encoded_length(), old.payload().len());
        let new = old
            .refine("strings", &[TextExample::new("Gamma", "GAMMA")], false)
            .unwrap();
        assert_eq!(new.encoded_length(), new.payload().len());
    }
    #[test]
    fn valid_binary_payloads_do_not_bypass_graph_and_training_validation() {
        let mut bad = skill();
        bad.program.nodes[0].edges[0].next = 0;
        assert!(
            TextSkill::decode(&bad.payload())
                .unwrap()
                .validate()
                .is_err()
        );
        let mut bad = skill();
        bad.examples[0].output = "different".into();
        assert!(
            TextSkill::decode(&bad.payload())
                .unwrap()
                .validate()
                .is_err()
        );
        let mut bad = skill();
        bad.revision = 0;
        assert!(
            TextSkill::decode(&bad.payload())
                .unwrap()
                .validate()
                .is_err()
        );
    }
    #[test]
    fn bounded_decoder_rejects_forged_counts_and_random_payloads_without_panicking() {
        let mut seed = 20261012_u64;
        for length in 0..1024 {
            let mut bytes = Vec::new();
            for _ in 0..length {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                bytes.push((seed >> 32) as u8);
            }
            assert!(TextSkill::decode(&bytes).is_err());
        }
    }
}
