//! Bounded induction of pure numerical programs. Example agreement is not a proof.
use crate::{
    live_store::checksum,
    text_skill::{TextSkillError, publish_new, read_bounded},
};
use std::{collections::HashMap, fmt, mem::size_of, path::Path, time::Instant};

const MAGIC: &[u8; 8] = b"AXNP0001";
const MAX_OPS: usize = 256;
const MAX_STACK: usize = 64;
const MAX_FILE: u64 = 32_768;

#[path = "numeric_continuous.rs"]
pub mod continuous;
#[path = "numeric_evolution.rs"]
pub mod evolution;
#[path = "numeric_execution.rs"]
mod execution;
#[path = "numeric_library.rs"]
pub mod library;
#[path = "numeric_seed.rs"]
pub mod seed;
pub use execution::NumericExecution;

#[derive(Clone, Debug, PartialEq)]
pub struct NumericExample {
    pub input: Vec<f64>,
    pub output: f64,
}
impl NumericExample {
    pub fn new(input: impl Into<Vec<f64>>, output: f64) -> Self {
        Self {
            input: input.into(),
            output,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericTolerance {
    pub absolute: f64,
    pub relative: f64,
}
impl Default for NumericTolerance {
    fn default() -> Self {
        Self {
            absolute: 1e-9,
            relative: 1e-9,
        }
    }
}
impl NumericTolerance {
    pub fn agrees(self, actual: f64, expected: f64) -> bool {
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs() <= self.absolute + self.relative * expected.abs()
    }
    fn validate(self) -> Result<(), NumericError> {
        if [self.absolute, self.relative]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0 || *v > 0.001)
        {
            return Err(NumericError::Invalid("tolerance"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct NumericBudget {
    pub max_cost: usize,
    pub max_terms: usize,
    pub max_work: u64,
}
impl Default for NumericBudget {
    fn default() -> Self {
        Self {
            max_cost: 8,
            max_terms: 100_000,
            max_work: 20_000_000,
        }
    }
}
impl NumericBudget {
    pub fn validate(self) -> Result<(), NumericError> {
        if !(1..=12).contains(&self.max_cost)
            || !(1..=200_000).contains(&self.max_terms)
            || !(1..=100_000_000).contains(&self.max_work)
        {
            return Err(NumericError::Invalid("search budget"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NumericSearch {
    pub candidates: u64,
    pub scalar_work: u64,
    pub retained_terms: usize,
    pub invalid_terms: u64,
    pub search_cost: usize,
    pub library_expansions: usize,
    pub elapsed_ns: u64,
}
#[derive(Debug)]
pub enum NumericError {
    Invalid(&'static str),
    Domain(&'static str),
    Context,
    Contradiction,
    Budget(NumericSearch),
    NoProgram(NumericSearch),
    Evolution {
        reason: &'static str,
        evaluations: u64,
        work: u64,
        loss: f64,
    },
    LibraryBudget {
        patterns: usize,
        visits: u64,
    },
    Development {
        attempts: usize,
        scalar_work: u64,
        reason: &'static str,
    },
    Suspended,
    Drift {
        expected: f64,
        actual: f64,
    },
    Evaluation {
        correct: usize,
        total: usize,
        refused: usize,
        maximum_error: f64,
    },
    Storage(TextSkillError),
}
impl fmt::Display for NumericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid numerical foundry: {s}"),
            Self::Domain(s) => write!(f, "numerical domain guard: {s}"),
            Self::Context => f.write_str("numerical capability context mismatch"),
            Self::Contradiction => f.write_str("same input has incompatible demonstrations"),
            Self::Budget(s) => write!(
                f,
                "numerical search budget exhausted after {} candidates, {} retained terms",
                s.candidates, s.retained_terms
            ),
            Self::NoProgram(s) => write!(
                f,
                "no compatible program in the explored grammar after {} candidates",
                s.candidates
            ),
            Self::Evolution {
                reason,
                evaluations,
                work,
                loss,
            } => write!(
                f,
                "evolution not admitted: {reason}; {evaluations} genomes, {work} work, best loss {loss}"
            ),
            Self::LibraryBudget { patterns, visits } => write!(
                f,
                "abstraction mining budget exhausted: {patterns} patterns, {visits} tree visits"
            ),
            Self::Development {
                attempts,
                scalar_work,
                reason,
            } => write!(
                f,
                "bounded numerical development stopped: {reason}; {attempts} searches, {scalar_work} scalar work"
            ),
            Self::Suspended => {
                f.write_str("numerical monitor suspended; a new revision is required")
            }
            Self::Drift { expected, actual } => {
                write!(f, "numerical drift: expected {expected}, observed {actual}")
            }
            Self::Evaluation {
                correct,
                total,
                refused,
                maximum_error,
            } => write!(
                f,
                "EMPIRICAL EVALUATION REFUTED: {correct}/{total} correct, {refused} refused, maximum absolute error {maximum_error}"
            ),
            Self::Storage(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for NumericError {}
impl From<TextSkillError> for NumericError {
    fn from(e: TextSkillError) -> Self {
        Self::Storage(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Input(u8),
    Constant(f64),
    Add,
    Subtract,
    Multiply,
    Divide,
    Sqrt,
}
impl Op {
    fn integer_binary(self, a: i64, b: i64) -> Result<i64, NumericError> {
        match self {
            Self::Add => a.checked_add(b),
            Self::Subtract => a.checked_sub(b),
            Self::Multiply => a.checked_mul(b),
            Self::Divide if a.checked_rem(b) == Some(0) => a.checked_div(b),
            _ => None,
        }
        .ok_or(NumericError::Domain("integer overflow or inexact division"))
    }
    fn binary(self, a: f64, b: f64) -> Result<f64, NumericError> {
        let value = match self {
            Self::Add => a + b,
            Self::Subtract => a - b,
            Self::Multiply => a * b,
            Self::Divide if b != 0.0 => a / b,
            Self::Divide => return Err(NumericError::Domain("division by zero")),
            _ => return Err(NumericError::Invalid("binary instruction")),
        };
        if value == 0.0 && a != 0.0 && b != 0.0 && matches!(self, Self::Multiply | Self::Divide) {
            return Err(NumericError::Domain("arithmetic underflow"));
        }
        finite(value)
    }
}
fn finite(value: f64) -> Result<f64, NumericError> {
    if value.is_finite() {
        Ok(if value == 0.0 { 0.0 } else { value })
    } else {
        Err(NumericError::Domain("non-finite arithmetic"))
    }
}
fn root(value: f64) -> Result<f64, NumericError> {
    if value < 0.0 {
        Err(NumericError::Domain("negative square root"))
    } else {
        finite(value.sqrt())
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct NumericProgram {
    arity: usize,
    ops: Vec<Op>,
}
impl NumericProgram {
    #[cfg(feature = "curriculum")]
    pub(crate) fn encode_program(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        library::write_program(&mut bytes, self);
        bytes
    }
    #[cfg(feature = "curriculum")]
    pub(crate) fn decode_program(bytes: &[u8]) -> Result<Self, NumericError> {
        if bytes.len() > 4096 {
            return Err(NumericError::Invalid("pure program size"));
        }
        let mut reader = Reader(bytes);
        let program = library::read_program(&mut reader)?;
        if !reader.0.is_empty() {
            return Err(NumericError::Invalid("pure program trailing bytes"));
        }
        Ok(program)
    }
    /// Exact partial integer semantics for a program proposed by the floating-point search.
    /// Fractional intermediates and overflow are refusals, never rounded answers.
    pub(crate) fn evaluate_integer(&self, input: &[i64]) -> Result<i64, NumericError> {
        if input.len() != self.arity {
            return Err(NumericError::Domain("integer arity"));
        }
        let mut stack = [0i64; MAX_STACK];
        let mut top = 0;
        for op in &self.ops {
            match *op {
                Op::Input(i) => {
                    stack[top] = input[i as usize];
                    top += 1;
                }
                Op::Constant(n) if n.fract() == 0.0 && n.abs() <= (1u64 << 53) as f64 => {
                    stack[top] = n as i64;
                    top += 1;
                }
                Op::Constant(_) => return Err(NumericError::Domain("noninteger constant")),
                Op::Sqrt => {
                    let n = u64::try_from(stack[top - 1])
                        .map_err(|_| NumericError::Domain("negative integer root"))?;
                    let r = n.isqrt();
                    if r * r != n {
                        return Err(NumericError::Domain("irrational integer root"));
                    }
                    stack[top - 1] = r as i64;
                }
                _ => {
                    let b = stack[top - 1];
                    top -= 1;
                    let a = stack[top - 1];
                    stack[top - 1] = op.integer_binary(a, b)?;
                }
            }
        }
        Ok(stack[0])
    }
    pub fn compile_execution(&self) -> NumericExecution {
        NumericExecution::compile(self)
    }
    pub fn arity(&self) -> usize {
        self.arity
    }
    pub fn instructions(&self) -> usize {
        self.ops.len()
    }
    pub fn scratch_bytes(&self) -> usize {
        size_of::<[f64; MAX_STACK]>()
    }
    pub fn evaluate(&self, input: &[f64]) -> Result<f64, NumericError> {
        if input.len() != self.arity || input.iter().any(|n| !n.is_finite()) {
            return Err(NumericError::Domain("input arity or finite values"));
        }
        let mut stack = [0.0; MAX_STACK];
        let mut top = 0;
        for op in &self.ops {
            match *op {
                Op::Input(i) => {
                    stack[top] = input[i as usize];
                    top += 1;
                }
                Op::Constant(n) => {
                    stack[top] = n;
                    top += 1;
                }
                Op::Sqrt => stack[top - 1] = root(stack[top - 1])?,
                _ => {
                    let b = stack[top - 1];
                    top -= 1;
                    stack[top - 1] = op.binary(stack[top - 1], b)?;
                }
            }
        }
        finite(stack[0])
    }
    pub fn expression(&self) -> String {
        let mut stack: Vec<String> = Vec::new();
        for op in &self.ops {
            match *op {
                Op::Input(i) => stack.push(format!("x{i}")),
                Op::Constant(n) => stack.push(n.to_string()),
                Op::Sqrt => {
                    let a = stack.pop().expect("validated");
                    stack.push(format!("sqrt({a})"));
                }
                _ => {
                    let b = stack.pop().expect("validated");
                    let a = stack.pop().expect("validated");
                    let symbol = match op {
                        Op::Add => "+",
                        Op::Subtract => "-",
                        Op::Multiply => "*",
                        _ => "/",
                    };
                    stack.push(format!("({a} {symbol} {b})"));
                }
            }
        }
        stack.pop().expect("validated")
    }
    fn validate(&self) -> Result<(), NumericError> {
        if !(1..=4).contains(&self.arity) || self.ops.is_empty() || self.ops.len() > MAX_OPS {
            return Err(NumericError::Invalid("program size"));
        }
        let mut depth = 0;
        for op in &self.ops {
            match op {
                Op::Input(i) if (*i as usize) < self.arity => depth += 1,
                Op::Constant(n) if n.is_finite() => depth += 1,
                Op::Sqrt if depth >= 1 => {}
                Op::Add | Op::Subtract | Op::Multiply | Op::Divide if depth >= 2 => depth -= 1,
                _ => return Err(NumericError::Invalid("program stack or operand")),
            }
            if depth > MAX_STACK {
                return Err(NumericError::Invalid("program stack budget"));
            }
        }
        if depth != 1 {
            return Err(NumericError::Invalid("program result stack"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct NumericSkill {
    name: String,
    context: String,
    revision: u32,
    parent: Option<u64>,
    examples: Vec<NumericExample>,
    tolerance: NumericTolerance,
    program: NumericProgram,
    dependencies: Vec<u64>,
    search: NumericSearch,
}
impl NumericSkill {
    /// Search and execution share partial integer semantics; no rounded intermediates.
    #[cfg(feature = "curriculum")]
    pub(crate) fn learn_integer_from_programs(
        examples: Vec<NumericExample>,
        programs: &[(&NumericProgram, u64)],
        budget: NumericBudget,
    ) -> Result<Self, NumericError> {
        budget.validate()?;
        if programs.len() > 16
            || examples.iter().any(|e| {
                e.input
                    .iter()
                    .chain([&e.output])
                    .any(|n| n.fract() != 0.0 || n.abs() > (1u64 << 53) as f64)
            })
        {
            return Err(NumericError::Invalid("integer learning interface"));
        }
        for (p, _) in programs {
            p.validate()?;
        }
        let tolerance = NumericTolerance {
            absolute: 0.0,
            relative: 0.0,
        };
        let arity = validate_examples(&examples, tolerance)?;
        let tools: Vec<_> = programs
            .iter()
            .map(|(program, identity)| NumericTool {
                program,
                identity: *identity,
            })
            .collect();
        let mut search = Search::new(&examples, &tools, budget, tolerance);
        search.exact_integer = true;
        let (program, dependencies, search) = search.run(arity)?;
        Ok(Self {
            name: "integer-derived".into(),
            context: "structural/integer-v1".into(),
            revision: 1,
            parent: None,
            examples,
            tolerance,
            program,
            dependencies,
            search,
        })
    }
    /// Reuse validated acquired programs without re-running their ancestral searches.
    pub fn learn_from_programs(
        name: impl Into<String>,
        context: impl Into<String>,
        examples: Vec<NumericExample>,
        programs: &[(&NumericProgram, u64)],
        budget: NumericBudget,
        tolerance: NumericTolerance,
    ) -> Result<Self, NumericError> {
        if programs.len() > 16 {
            return Err(NumericError::Invalid("inherited program count"));
        }
        for (program, _) in programs {
            program.validate()?;
        }
        let tools: Vec<_> = programs
            .iter()
            .map(|(program, identity)| NumericTool {
                program,
                identity: *identity,
            })
            .collect();
        Self::learn_tools(
            name.into(),
            context.into(),
            examples,
            &tools,
            budget,
            tolerance,
        )
    }
    pub fn learn(
        name: impl Into<String>,
        context: impl Into<String>,
        examples: Vec<NumericExample>,
        library: &[&NumericSkill],
        budget: NumericBudget,
        tolerance: NumericTolerance,
    ) -> Result<Self, NumericError> {
        let tools: Vec<_> = library
            .iter()
            .map(|skill| NumericTool {
                program: &skill.program,
                identity: skill.fingerprint(),
            })
            .collect();
        Self::learn_tools(
            name.into(),
            context.into(),
            examples,
            &tools,
            budget,
            tolerance,
        )
    }
    pub fn learn_from_library(
        name: impl Into<String>,
        context: impl Into<String>,
        examples: Vec<NumericExample>,
        library: &library::NumericLibrary,
        budget: NumericBudget,
        tolerance: NumericTolerance,
    ) -> Result<Self, NumericError> {
        let tools = library.tools();
        Self::learn_tools(
            name.into(),
            context.into(),
            examples,
            &tools,
            budget,
            tolerance,
        )
    }
    fn learn_tools(
        name: String,
        context: String,
        examples: Vec<NumericExample>,
        library: &[NumericTool<'_>],
        budget: NumericBudget,
        tolerance: NumericTolerance,
    ) -> Result<Self, NumericError> {
        identifier(&name)?;
        identifier(&context)?;
        budget.validate()?;
        tolerance.validate()?;
        let arity = validate_examples(&examples, tolerance)?;
        if library.len() > 16 {
            return Err(NumericError::Invalid(
                "library supports at most 16 capabilities",
            ));
        }
        let (program, dependencies, search) =
            Search::new(&examples, library, budget, tolerance).run(arity)?;
        Ok(Self {
            name,
            context,
            revision: 1,
            parent: None,
            examples,
            tolerance,
            program,
            dependencies,
            search,
        })
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
    pub fn program(&self) -> &NumericProgram {
        &self.program
    }
    pub fn search(&self) -> NumericSearch {
        self.search
    }
    pub fn examples(&self) -> &[NumericExample] {
        &self.examples
    }
    pub fn dependencies(&self) -> &[u64] {
        &self.dependencies
    }
    pub fn tolerance(&self) -> NumericTolerance {
        self.tolerance
    }
    pub fn predict(&self, context: &str, input: &[f64]) -> Result<f64, NumericError> {
        if context != self.context {
            return Err(NumericError::Context);
        }
        self.program.evaluate(input)
    }
    pub fn revise(
        &self,
        context: &str,
        examples: Vec<NumericExample>,
        library: &[&NumericSkill],
        budget: NumericBudget,
    ) -> Result<Self, NumericError> {
        if context != self.context {
            return Err(NumericError::Context);
        }
        let mut next = Self::learn(
            self.name.clone(),
            self.context.clone(),
            examples,
            library,
            budget,
            self.tolerance,
        )?;
        next.revision = self
            .revision
            .checked_add(1)
            .ok_or(NumericError::Invalid("revision overflow"))?;
        next.parent = Some(self.fingerprint());
        Ok(next)
    }
    pub fn revise_from_library(
        &self,
        context: &str,
        examples: Vec<NumericExample>,
        library: &library::NumericLibrary,
        budget: NumericBudget,
    ) -> Result<Self, NumericError> {
        if context != self.context {
            return Err(NumericError::Context);
        }
        let mut next = Self::learn_from_library(
            self.name.clone(),
            self.context.clone(),
            examples,
            library,
            budget,
            self.tolerance,
        )?;
        next.revision = self
            .revision
            .checked_add(1)
            .ok_or(NumericError::Invalid("revision overflow"))?;
        next.parent = Some(self.fingerprint());
        Ok(next)
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.name.capacity()
            + self.context.capacity()
            + self.program.ops.capacity() * size_of::<Op>()
            + self.dependencies.capacity() * 8
            + self.examples.capacity() * size_of::<NumericExample>()
            + self
                .examples
                .iter()
                .map(|e| e.input.capacity() * 8)
                .sum::<usize>()
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.payload(false))
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(self.payload(true));
        bytes.extend(checksum(&bytes).to_le_bytes());
        bytes
    }
    pub fn save_new(&self, path: &Path) -> Result<(), NumericError> {
        publish_new(path, &self.to_bytes()).map_err(Into::into)
    }
    pub fn open(path: &Path, context: &str) -> Result<Self, NumericError> {
        let skill = Self::from_bytes(&read_bounded(path, MAX_FILE)?)?;
        if context != skill.context {
            return Err(NumericError::Context);
        }
        Ok(skill)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NumericError> {
        if bytes.len() < 16 || bytes.len() as u64 > MAX_FILE || &bytes[..8] != MAGIC {
            return Err(NumericError::Invalid("capsule header or size"));
        }
        let body = bytes.len() - 8;
        if checksum(&bytes[..body])
            != u64::from_le_bytes(bytes[body..].try_into().expect("eight bytes"))
        {
            return Err(NumericError::Invalid("capsule checksum"));
        }
        let mut r = Reader(&bytes[8..body]);
        let name = r.text()?;
        let context = r.text()?;
        identifier(&name)?;
        identifier(&context)?;
        let revision = r.u32()?;
        let has_parent = r.byte()?;
        let raw_parent = r.u64()?;
        if revision == 0
            || has_parent > 1
            || (has_parent == 0 && raw_parent != 0)
            || (revision == 1) != (has_parent == 0)
        {
            return Err(NumericError::Invalid("revision lineage"));
        }
        let tolerance = NumericTolerance {
            absolute: r.number()?,
            relative: r.number()?,
        };
        tolerance.validate()?;
        let arity = r.byte()? as usize;
        if !(1..=4).contains(&arity) {
            return Err(NumericError::Invalid("arity"));
        }
        let count = r.count(32)?;
        let mut examples = Vec::with_capacity(count);
        for _ in 0..count {
            let mut input = Vec::with_capacity(arity);
            for _ in 0..arity {
                input.push(r.number()?);
            }
            examples.push(NumericExample::new(input, r.number()?));
        }
        validate_examples(&examples, tolerance)?;
        let count = r.count(16)?;
        let mut dependencies = Vec::with_capacity(count);
        for _ in 0..count {
            dependencies.push(r.u64()?);
        }
        let count = r.count(MAX_OPS)?;
        let mut ops = Vec::with_capacity(count);
        for _ in 0..count {
            ops.push(match r.byte()? {
                0 => Op::Input(r.byte()?),
                1 => Op::Constant(r.number()?),
                2 => Op::Add,
                3 => Op::Subtract,
                4 => Op::Multiply,
                5 => Op::Divide,
                6 => Op::Sqrt,
                _ => return Err(NumericError::Invalid("opcode")),
            });
        }
        let program = NumericProgram { arity, ops };
        program.validate()?;
        let search = NumericSearch {
            candidates: r.u64()?,
            scalar_work: r.u64()?,
            retained_terms: r.u32()? as usize,
            invalid_terms: r.u64()?,
            search_cost: r.byte()? as usize,
            library_expansions: r.u32()? as usize,
            elapsed_ns: r.u64()?,
        };
        if !r.0.is_empty()
            || !(1..=12).contains(&search.search_cost)
            || search.retained_terms > 200_000
        {
            return Err(NumericError::Invalid("search record or trailing data"));
        }
        for e in &examples {
            if !tolerance.agrees(program.evaluate(&e.input)?, e.output) {
                return Err(NumericError::Invalid(
                    "program does not reproduce demonstrations",
                ));
            }
        }
        Ok(Self {
            name,
            context,
            revision,
            parent: (has_parent == 1).then_some(raw_parent),
            examples,
            tolerance,
            program,
            dependencies,
            search,
        })
    }
    fn payload(&self, statistics: bool) -> Vec<u8> {
        let mut b = Vec::new();
        for s in [&self.name, &self.context] {
            b.extend((s.len() as u32).to_le_bytes());
            b.extend(s.as_bytes());
        }
        b.extend(self.revision.to_le_bytes());
        b.push(u8::from(self.parent.is_some()));
        b.extend(self.parent.unwrap_or(0).to_le_bytes());
        b.extend(self.tolerance.absolute.to_le_bytes());
        b.extend(self.tolerance.relative.to_le_bytes());
        b.push(self.program.arity as u8);
        b.extend((self.examples.len() as u32).to_le_bytes());
        for e in &self.examples {
            for n in &e.input {
                b.extend(n.to_le_bytes());
            }
            b.extend(e.output.to_le_bytes());
        }
        b.extend((self.dependencies.len() as u32).to_le_bytes());
        for n in &self.dependencies {
            b.extend(n.to_le_bytes());
        }
        b.extend((self.program.ops.len() as u32).to_le_bytes());
        for op in &self.program.ops {
            match op {
                Op::Input(i) => {
                    b.push(0);
                    b.push(*i);
                }
                Op::Constant(n) => {
                    b.push(1);
                    b.extend(n.to_le_bytes());
                }
                Op::Add => b.push(2),
                Op::Subtract => b.push(3),
                Op::Multiply => b.push(4),
                Op::Divide => b.push(5),
                Op::Sqrt => b.push(6),
            }
        }
        if statistics {
            let s = self.search;
            b.extend(s.candidates.to_le_bytes());
            b.extend(s.scalar_work.to_le_bytes());
            b.extend((s.retained_terms as u32).to_le_bytes());
            b.extend(s.invalid_terms.to_le_bytes());
            b.push(s.search_cost as u8);
            b.extend((s.library_expansions as u32).to_le_bytes());
            b.extend(s.elapsed_ns.to_le_bytes());
        }
        b
    }
}

pub struct NumericMonitor<'a> {
    skill: &'a NumericSkill,
    observations: u64,
    suspended: bool,
}
impl<'a> NumericMonitor<'a> {
    pub fn new(skill: &'a NumericSkill) -> Self {
        Self {
            skill,
            observations: 0,
            suspended: false,
        }
    }
    pub fn observations(&self) -> u64 {
        self.observations
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    pub fn predict(&self, input: &[f64]) -> Result<f64, NumericError> {
        if self.suspended {
            return Err(NumericError::Suspended);
        }
        self.skill.program.evaluate(input)
    }
    pub fn observe(&mut self, input: &[f64], actual: f64) -> Result<(), NumericError> {
        if self.suspended {
            return Err(NumericError::Suspended);
        }
        let expected = match self.skill.program.evaluate(input) {
            Ok(n) => n,
            Err(e) => {
                self.suspended = true;
                return Err(e);
            }
        };
        self.observations = self
            .observations
            .checked_add(1)
            .ok_or(NumericError::Invalid("observation counter"))?;
        if !self.skill.tolerance.agrees(expected, actual) {
            self.suspended = true;
            return Err(NumericError::Drift { expected, actual });
        }
        Ok(())
    }
}

fn identifier(s: &str) -> Result<(), NumericError> {
    if s.is_empty() || s.len() > 256 || s.trim() != s || s.chars().any(char::is_control) {
        return Err(NumericError::Invalid("identifier"));
    }
    Ok(())
}
fn validate_examples(
    examples: &[NumericExample],
    tolerance: NumericTolerance,
) -> Result<usize, NumericError> {
    if !(2..=32).contains(&examples.len()) {
        return Err(NumericError::Invalid("requires 2..32 demonstrations"));
    }
    let arity = examples[0].input.len();
    if !(1..=4).contains(&arity)
        || examples.iter().any(|e| {
            e.input.len() != arity
                || !e.output.is_finite()
                || e.input.iter().any(|n| !n.is_finite())
        })
    {
        return Err(NumericError::Invalid(
            "demonstration dimensions or finite values",
        ));
    }
    for (i, a) in examples.iter().enumerate() {
        for b in &examples[..i] {
            if a.input == b.input && !tolerance.agrees(a.output, b.output) {
                return Err(NumericError::Contradiction);
            }
        }
    }
    Ok(arity)
}

#[derive(Clone, Copy)]
enum Node {
    Leaf(Op),
    Unary(usize),
    Binary(Op, usize, usize),
    Tool(usize, [usize; 4]),
}
struct Term {
    node: Node,
    values: Vec<f64>,
}
struct NumericTool<'a> {
    program: &'a NumericProgram,
    identity: u64,
}
struct Search<'a> {
    examples: &'a [NumericExample],
    library: &'a [NumericTool<'a>],
    budget: NumericBudget,
    tolerance: NumericTolerance,
    terms: Vec<Term>,
    by_cost: Vec<Vec<usize>>,
    signatures: HashMap<Vec<u64>, usize>,
    stats: NumericSearch,
    started: Instant,
    exact_integer: bool,
}
impl<'a> Search<'a> {
    fn new(
        examples: &'a [NumericExample],
        library: &'a [NumericTool<'a>],
        budget: NumericBudget,
        tolerance: NumericTolerance,
    ) -> Self {
        Self {
            examples,
            library,
            budget,
            tolerance,
            terms: Vec::new(),
            by_cost: vec![Vec::new(); budget.max_cost + 1],
            signatures: HashMap::new(),
            stats: NumericSearch::default(),
            started: Instant::now(),
            exact_integer: false,
        }
    }
    fn stats(&self) -> NumericSearch {
        NumericSearch {
            retained_terms: self.terms.len(),
            elapsed_ns: self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            ..self.stats
        }
    }
    fn insert(&mut self, node: Node, cost: usize) -> Result<Option<usize>, NumericError> {
        self.stats.candidates += 1;
        let unit = match node {
            Node::Tool(i, ..) => self.library[i].program.instructions() as u64,
            _ => 1,
        };
        let work = self.examples.len() as u64 * unit;
        if self.stats.scalar_work + work > self.budget.max_work {
            return Err(NumericError::Budget(self.stats()));
        }
        self.stats.scalar_work += work;
        let mut values = Vec::with_capacity(self.examples.len());
        for (i, example) in self.examples.iter().enumerate() {
            let value = if self.exact_integer {
                let integer = match node {
                    Node::Leaf(Op::Input(n)) => Ok(example.input[n as usize] as i64),
                    Node::Leaf(Op::Constant(n))
                        if n.fract() == 0.0 && n.abs() <= (1u64 << 53) as f64 =>
                    {
                        Ok(n as i64)
                    }
                    Node::Unary(a) => {
                        let n = self.terms[a].values[i] as i64;
                        u64::try_from(n)
                            .ok()
                            .and_then(|n| {
                                let r = n.isqrt();
                                (r * r == n).then_some(r as i64)
                            })
                            .ok_or(NumericError::Domain("integer root"))
                    }
                    Node::Binary(op, a, b) => op.integer_binary(
                        self.terms[a].values[i] as i64,
                        self.terms[b].values[i] as i64,
                    ),
                    Node::Tool(tool, args) => {
                        let mut inputs = [0i64; 4];
                        for j in 0..self.library[tool].program.arity {
                            inputs[j] = self.terms[args[j]].values[i] as i64;
                        }
                        self.library[tool]
                            .program
                            .evaluate_integer(&inputs[..self.library[tool].program.arity])
                    }
                    _ => Err(NumericError::Domain("integer leaf")),
                };
                integer.and_then(|n| {
                    if n.unsigned_abs() <= 1u64 << 53 {
                        Ok(n as f64)
                    } else {
                        Err(NumericError::Domain("integer search range"))
                    }
                })
            } else {
                match node {
                    Node::Leaf(Op::Input(n)) => Ok(example.input[n as usize]),
                    Node::Leaf(Op::Constant(n)) => Ok(n),
                    Node::Unary(a) => root(self.terms[a].values[i]),
                    Node::Binary(op, a, b) => {
                        op.binary(self.terms[a].values[i], self.terms[b].values[i])
                    }
                    Node::Tool(tool, args) => {
                        let mut inputs = [0.0; 4];
                        for j in 0..self.library[tool].program.arity {
                            inputs[j] = self.terms[args[j]].values[i];
                        }
                        self.library[tool]
                            .program
                            .evaluate(&inputs[..self.library[tool].program.arity])
                    }
                    _ => return Err(NumericError::Invalid("search leaf")),
                }
            };
            match value {
                Ok(n) => values.push(if n == 0.0 { 0.0 } else { n }),
                Err(_) => {
                    self.stats.invalid_terms += 1;
                    return Ok(None);
                }
            }
        }
        // Exact observed signatures prune search only. They do not prove equivalence outside these examples.
        let signature: Vec<u64> = values.iter().map(|v| v.to_bits()).collect();
        if self.signatures.contains_key(&signature) {
            return Ok(None);
        }
        if self.terms.len() >= self.budget.max_terms {
            return Err(NumericError::Budget(self.stats()));
        }
        let matches = values
            .iter()
            .zip(self.examples)
            .all(|(v, e)| self.tolerance.agrees(*v, e.output));
        let id = self.terms.len();
        self.terms.push(Term { node, values });
        self.signatures.insert(signature, id);
        self.by_cost[cost].push(id);
        if matches {
            self.stats.search_cost = cost;
            return Ok(Some(id));
        }
        Ok(None)
    }
    fn compile(
        &self,
        id: usize,
        output: &mut Vec<Op>,
        dependencies: &mut Vec<u64>,
        calls: &mut usize,
    ) -> Result<(), NumericError> {
        match self.terms[id].node {
            Node::Leaf(op) => output.push(op),
            Node::Unary(a) => {
                self.compile(a, output, dependencies, calls)?;
                output.push(Op::Sqrt);
            }
            Node::Binary(op, a, b) => {
                self.compile(a, output, dependencies, calls)?;
                self.compile(b, output, dependencies, calls)?;
                output.push(op);
            }
            Node::Tool(i, args) => {
                let skill = &self.library[i];
                let identity = skill.identity;
                if !dependencies.contains(&identity) {
                    dependencies.push(identity);
                }
                *calls += 1;
                for op in &skill.program.ops {
                    if let Op::Input(input) = op {
                        self.compile(args[*input as usize], output, dependencies, calls)?;
                    } else {
                        output.push(*op);
                    }
                    if output.len() > MAX_OPS {
                        return Err(NumericError::Budget(self.stats()));
                    }
                }
            }
        }
        if output.len() > MAX_OPS {
            return Err(NumericError::Budget(self.stats()));
        }
        Ok(())
    }
    fn finish(
        &self,
        id: usize,
        arity: usize,
    ) -> Result<(NumericProgram, Vec<u64>, NumericSearch), NumericError> {
        let mut ops = Vec::new();
        let mut dependencies = Vec::new();
        let mut calls = 0;
        self.compile(id, &mut ops, &mut dependencies, &mut calls)?;
        let program = NumericProgram { arity, ops };
        program.validate()?;
        for e in self.examples {
            if !self.tolerance.agrees(program.evaluate(&e.input)?, e.output) {
                return Err(NumericError::Invalid(
                    "compiled program changed observed semantics",
                ));
            }
        }
        Ok((
            program,
            dependencies,
            NumericSearch {
                library_expansions: calls,
                ..self.stats()
            },
        ))
    }
    fn tool_args(
        &mut self,
        tool: usize,
        cost: usize,
        slot: usize,
        remaining: usize,
        mut args: [usize; 4],
    ) -> Result<Option<usize>, NumericError> {
        let arity = self.library[tool].program.arity;
        if slot == arity {
            return if remaining == 0 {
                self.insert(Node::Tool(tool, args), cost)
            } else {
                Ok(None)
            };
        }
        let slots = arity - slot;
        if remaining < slots {
            return Ok(None);
        }
        for part in 1..=remaining - (slots - 1) {
            for id in self.by_cost[part].clone() {
                args[slot] = id;
                if let Some(found) = self.tool_args(tool, cost, slot + 1, remaining - part, args)? {
                    return Ok(Some(found));
                }
            }
        }
        Ok(None)
    }
    fn run(
        mut self,
        arity: usize,
    ) -> Result<(NumericProgram, Vec<u64>, NumericSearch), NumericError> {
        let mut literals = vec![0.0, 1.0, 2.0];
        // Constants inferred from observed affine projections are proposals, not guaranteed laws.
        for column in 0..arity {
            let a = &self.examples[0];
            if let Some(b) = self
                .examples
                .iter()
                .find(|b| b.input[column] != a.input[column])
            {
                let work = 4 + 3 * self.examples.len() as u64;
                if self.stats.scalar_work + work > self.budget.max_work {
                    return Err(NumericError::Budget(self.stats()));
                }
                self.stats.scalar_work += work;
                let slope = (b.output - a.output) / (b.input[column] - a.input[column]);
                let offset = a.output - slope * a.input[column];
                if slope.is_finite()
                    && offset.is_finite()
                    && self.examples.iter().all(|e| {
                        self.tolerance
                            .agrees(slope * e.input[column] + offset, e.output)
                    })
                {
                    for n in [slope, offset] {
                        let n = if n == 0.0 { 0.0 } else { n };
                        if !literals.contains(&n) {
                            literals.push(n);
                        }
                    }
                }
            }
        }
        if self
            .examples
            .iter()
            .all(|e| self.tolerance.agrees(e.output, self.examples[0].output))
            && !literals.contains(&self.examples[0].output)
        {
            literals.push(self.examples[0].output);
        }
        for input in 0..arity {
            if let Some(id) = self.insert(Node::Leaf(Op::Input(input as u8)), 1)? {
                return self.finish(id, arity);
            }
        }
        for n in literals {
            if let Some(id) = self.insert(Node::Leaf(Op::Constant(n)), 1)? {
                return self.finish(id, arity);
            }
        }
        for cost in 2..=self.budget.max_cost {
            for a in self.by_cost[cost - 1].clone() {
                if let Some(id) = self.insert(Node::Unary(a), cost)? {
                    return self.finish(id, arity);
                }
                for tool in 0..self.library.len() {
                    if self.library[tool].program.arity == 1
                        && let Some(id) = self.insert(Node::Tool(tool, [a, 0, 0, 0]), cost)?
                    {
                        return self.finish(id, arity);
                    }
                }
            }
            for left_cost in 1..cost - 1 {
                let right_cost = cost - 1 - left_cost;
                let left = self.by_cost[left_cost].clone();
                let right = self.by_cost[right_cost].clone();
                for &a in &left {
                    for &b in &right {
                        for op in [Op::Add, Op::Multiply, Op::Subtract, Op::Divide] {
                            if matches!(op, Op::Add | Op::Multiply) && a > b {
                                continue;
                            }
                            if let Some(id) = self.insert(Node::Binary(op, a, b), cost)? {
                                return self.finish(id, arity);
                            }
                        }
                        for tool in 0..self.library.len() {
                            if self.library[tool].program.arity == 2
                                && let Some(id) =
                                    self.insert(Node::Tool(tool, [a, b, 0, 0]), cost)?
                            {
                                return self.finish(id, arity);
                            }
                        }
                    }
                }
            }
            // Cheap outer compositions must get a turn before high-arity Cartesian products.
            for tool in 0..self.library.len() {
                if self.library[tool].program.arity >= 3
                    && cost > self.library[tool].program.arity
                    && let Some(id) = self.tool_args(tool, cost, 0, cost - 1, [0; 4])?
                {
                    return self.finish(id, arity);
                }
            }
        }
        Err(NumericError::NoProgram(self.stats()))
    }
}

struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], NumericError> {
        if n > self.0.len() {
            return Err(NumericError::Invalid("truncated capsule"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn byte(&mut self) -> Result<u8, NumericError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, NumericError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, NumericError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn number(&mut self) -> Result<f64, NumericError> {
        let value = f64::from_bits(self.u64()?);
        if !value.is_finite() {
            return Err(NumericError::Invalid("non-finite stored number"));
        }
        Ok(value)
    }
    fn count(&mut self, max: usize) -> Result<usize, NumericError> {
        let n = self.u32()? as usize;
        if n > max {
            return Err(NumericError::Invalid("stored count limit"));
        }
        Ok(n)
    }
    fn text(&mut self) -> Result<String, NumericError> {
        let n = self.count(256)?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| NumericError::Invalid("UTF-8"))
    }
}
