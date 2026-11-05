//! Empirical induction of bounded integer transducers, not a biological memory model.
use crate::{
    live_store::checksum,
    text_skill::{TextSkillError, publish_new, read_bounded},
};
use std::{fmt, mem::size_of, path::Path, sync::Arc};

#[path = "reactive_agent.rs"]
pub mod agent;
#[path = "reactive_forge.rs"]
mod forge;
#[path = "reactive_storage.rs"]
mod storage;

const MAX_CELLS: usize = 32;
const MAX_DEPTH: usize = 4;
const MAX_EXPR: usize = 96;
const MAX_FILE: u64 = 1_048_576;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReactiveEvent {
    pub input: Vec<i64>,
    pub output: i64,
}
impl ReactiveEvent {
    pub fn new(input: impl Into<Vec<i64>>, output: i64) -> Self {
        Self {
            input: input.into(),
            output,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReactiveTrace {
    pub events: Vec<ReactiveEvent>,
}
impl ReactiveTrace {
    pub fn new(events: Vec<ReactiveEvent>) -> Self {
        Self { events }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ReactiveBudget {
    pub max_work: u64,
    pub max_candidates: u64,
    pub max_samples: usize,
    pub max_ticks: usize,
    pub max_tools: usize,
    pub max_guard_constants: usize,
}
impl Default for ReactiveBudget {
    fn default() -> Self {
        Self {
            max_work: 50_000_000,
            max_candidates: 50_000,
            max_samples: 256,
            max_ticks: 64,
            max_tools: 4,
            max_guard_constants: 12,
        }
    }
}
impl ReactiveBudget {
    fn validate(self) -> Result<(), ReactiveError> {
        if !(1..=100_000_000).contains(&self.max_work)
            || !(1..=200_000).contains(&self.max_candidates)
            || !(4..=256).contains(&self.max_samples)
            || !(4..=64).contains(&self.max_ticks)
            || self.max_tools > 4
            || !(1..=16).contains(&self.max_guard_constants)
        {
            return Err(ReactiveError::Invalid("search policy"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReactiveSearch {
    pub candidates: u64,
    pub work: u64,
    pub guarded_candidates: u64,
    pub refused_tools: u64,
    pub elapsed_ns: u64,
}
#[derive(Debug)]
pub enum ReactiveError {
    Invalid(&'static str),
    Domain(&'static str),
    Context,
    Identity,
    Suspended,
    Budget(ReactiveSearch),
    NoProgram(ReactiveSearch),
    Drift { expected: i64, actual: i64 },
    Storage(TextSkillError),
}
impl fmt::Display for ReactiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reactive: {self:?}")
    }
}
impl std::error::Error for ReactiveError {}
impl From<TextSkillError> for ReactiveError {
    fn from(e: TextSkillError) -> Self {
        Self::Storage(e)
    }
}
impl From<std::io::Error> for ReactiveError {
    fn from(e: std::io::Error) -> Self {
        Self::Storage(TextSkillError::Io(e))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Expr {
    Input(u8),
    State,
    Tool(u8),
    Constant(i64),
    Add(Arc<Expr>, Arc<Expr>),
    Subtract(Arc<Expr>, Arc<Expr>),
    Multiply(Arc<Expr>, Arc<Expr>),
    IfLess(Arc<Expr>, Arc<Expr>, Arc<Expr>, Arc<Expr>),
}
#[derive(Clone, Copy)]
struct Frame<'a> {
    input: &'a [i64],
    state: i64,
    tools: &'a [i64],
}
impl Expr {
    fn eval(&self, frame: Frame<'_>, work: &mut u64, limit: u64) -> Result<i64, ReactiveError> {
        if *work >= limit {
            return Err(ReactiveError::Domain("execution fuel"));
        }
        *work += 1;
        match self {
            Self::Input(i) => frame
                .input
                .get(*i as usize)
                .copied()
                .ok_or(ReactiveError::Invalid("input reference")),
            Self::State => Ok(frame.state),
            Self::Tool(i) => frame
                .tools
                .get(*i as usize)
                .copied()
                .ok_or(ReactiveError::Invalid("tool reference")),
            Self::Constant(n) => Ok(*n),
            Self::IfLess(a, b, t, f) => {
                let yes = a.eval(frame, work, limit)? < b.eval(frame, work, limit)?;
                if yes {
                    t.eval(frame, work, limit)
                } else {
                    f.eval(frame, work, limit)
                }
            }
            Self::Add(a, b) | Self::Subtract(a, b) | Self::Multiply(a, b) => {
                let a = a.eval(frame, work, limit)?;
                let b = b.eval(frame, work, limit)?;
                match self {
                    Self::Add(..) => a.checked_add(b),
                    Self::Subtract(..) => a.checked_sub(b),
                    _ => a.checked_mul(b),
                }
                .ok_or(ReactiveError::Domain("integer overflow"))
            }
        }
    }
    fn nodes(&self) -> usize {
        match self {
            Self::Add(a, b) | Self::Subtract(a, b) | Self::Multiply(a, b) => {
                1 + a.nodes() + b.nodes()
            }
            Self::IfLess(a, b, t, f) => 1 + a.nodes() + b.nodes() + t.nodes() + f.nodes(),
            _ => 1,
        }
    }
    fn has_state(&self) -> bool {
        match self {
            Self::State => true,
            Self::Add(a, b) | Self::Subtract(a, b) | Self::Multiply(a, b) => {
                a.has_state() || b.has_state()
            }
            Self::IfLess(a, b, t, f) => {
                a.has_state() || b.has_state() || t.has_state() || f.has_state()
            }
            _ => false,
        }
    }
    fn visit_tools(&self, used: &mut [bool; 4]) {
        match self {
            Self::Tool(i) => {
                if (*i as usize) < 4 {
                    used[*i as usize] = true;
                }
            }
            Self::Add(a, b) | Self::Subtract(a, b) | Self::Multiply(a, b) => {
                a.visit_tools(used);
                b.visit_tools(used);
            }
            Self::IfLess(a, b, t, f) => {
                for e in [a, b, t, f] {
                    e.visit_tools(used);
                }
            }
            _ => {}
        }
    }
    fn remap(&self, map: &[u8; 4]) -> Arc<Self> {
        Arc::new(match self {
            Self::Tool(i) => Self::Tool(map[*i as usize]),
            Self::Add(a, b) => Self::Add(a.remap(map), b.remap(map)),
            Self::Subtract(a, b) => Self::Subtract(a.remap(map), b.remap(map)),
            Self::Multiply(a, b) => Self::Multiply(a.remap(map), b.remap(map)),
            Self::IfLess(a, b, t, f) => {
                Self::IfLess(a.remap(map), b.remap(map), t.remap(map), f.remap(map))
            }
            e => e.clone(),
        })
    }
    fn validate(
        &self,
        arity: usize,
        tools: usize,
        state: bool,
        depth: usize,
    ) -> Result<(), ReactiveError> {
        if depth > 16 || self.nodes() > MAX_EXPR {
            return Err(ReactiveError::Invalid("expression quota"));
        }
        match self {
            Self::Input(i) if (*i as usize) >= arity => {
                Err(ReactiveError::Invalid("input reference"))
            }
            Self::Tool(i) if (*i as usize) >= tools => {
                Err(ReactiveError::Invalid("tool reference"))
            }
            Self::State if !state => Err(ReactiveError::Invalid("undeclared state")),
            Self::Add(a, b) | Self::Subtract(a, b) | Self::Multiply(a, b) => {
                a.validate(arity, tools, state, depth + 1)?;
                b.validate(arity, tools, state, depth + 1)
            }
            Self::IfLess(a, b, t, f) => {
                for e in [a, b, t, f] {
                    e.validate(arity, tools, state, depth + 1)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    fn describe(&self) -> String {
        match self {
            Self::Input(i) => format!("x{i}"),
            Self::State => "s".into(),
            Self::Tool(i) => format!("tool{i}"),
            Self::Constant(n) => n.to_string(),
            Self::IfLess(a, b, t, f) => format!(
                "if {} < {} then {} else {}",
                a.describe(),
                b.describe(),
                t.describe(),
                f.describe()
            ),
            Self::Add(a, b) | Self::Subtract(a, b) | Self::Multiply(a, b) => format!(
                "({} {} {})",
                a.describe(),
                match self {
                    Self::Add(..) => "+",
                    Self::Subtract(..) => "-",
                    _ => "*",
                },
                b.describe()
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReactiveSkill {
    name: String,
    context: String,
    arity: usize,
    update: Option<Arc<Expr>>,
    initial: i64,
    after_update: bool,
    output: Arc<Expr>,
    tools: Vec<ReactiveSkill>,
    traces: Vec<ReactiveTrace>,
    search: ReactiveSearch,
}
impl ReactiveSkill {
    pub fn learn(
        name: &str,
        context: &str,
        traces: Vec<ReactiveTrace>,
        library: &[&Self],
        budget: ReactiveBudget,
    ) -> Result<Self, ReactiveError> {
        forge::learn(name, context, traces, library, budget)
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn arity(&self) -> usize {
        self.arity
    }
    pub fn search(&self) -> ReactiveSearch {
        self.search
    }
    pub fn traces(&self) -> &[ReactiveTrace] {
        &self.traces
    }
    pub fn dependencies(&self) -> Vec<u64> {
        self.tools.iter().map(Self::fingerprint).collect()
    }
    pub fn memory_registers(&self) -> usize {
        usize::from(self.update.is_some())
            + self.tools.iter().map(Self::memory_registers).sum::<usize>()
    }
    pub fn cells(&self) -> usize {
        1 + self.tools.iter().map(Self::cells).sum::<usize>()
    }
    pub fn instructions(&self) -> usize {
        self.output.nodes()
            + self.update.as_ref().map_or(0, |e| e.nodes())
            + self.tools.iter().map(Self::instructions).sum::<usize>()
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.name.capacity()
            + self.context.capacity()
            + (self.output.nodes() + self.update.as_ref().map_or(0, |e| e.nodes()))
                * size_of::<Expr>()
            + self.tools.capacity() * size_of::<Self>()
            + self
                .tools
                .iter()
                .map(|s| s.retained_bytes() - size_of::<Self>())
                .sum::<usize>()
            + self.traces.capacity() * size_of::<ReactiveTrace>()
            + self
                .traces
                .iter()
                .map(|t| {
                    t.events.capacity() * size_of::<ReactiveEvent>()
                        + t.events
                            .iter()
                            .map(|e| e.input.capacity() * size_of::<i64>())
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
    pub fn description(&self) -> String {
        format!(
            "init={}; update={}; output({})={}; tools={}",
            self.initial,
            self.update
                .as_ref()
                .map_or_else(|| "none".into(), |e| e.describe()),
            if self.after_update { "after" } else { "before" },
            self.output.describe(),
            self.tools.len()
        )
    }
    pub fn compile(&self) -> Result<ReactiveCircuit, ReactiveError> {
        self.validate_shape(0)?;
        let mut circuit = ReactiveCircuit {
            context: self.context.clone(),
            arity: self.arity,
            identity: self.fingerprint(),
            cells: Vec::with_capacity(self.cells()),
        };
        circuit.append(self, &mut Vec::new());
        Ok(circuit)
    }
    fn validate_shape(&self, depth: usize) -> Result<(), ReactiveError> {
        identifier(&self.name)?;
        identifier(&self.context)?;
        if depth > MAX_DEPTH
            || !(1..=2).contains(&self.arity)
            || self.tools.len() > 4
            || self.cells() > MAX_CELLS
            || self.instructions() > 1024
            || (!self.after_update && self.update.is_none())
            || (self.update.is_none() && self.initial != 0)
        {
            return Err(ReactiveError::Invalid("circuit shape"));
        }
        if let Some(update) = &self.update {
            update.validate(self.arity, self.tools.len(), true, 0)?;
        }
        self.output
            .validate(self.arity, self.tools.len(), self.update.is_some(), 0)?;
        let mut used = [false; 4];
        self.output.visit_tools(&mut used);
        if let Some(e) = &self.update {
            e.visit_tools(&mut used);
        }
        if used[..self.tools.len()].iter().any(|u| !u) {
            return Err(ReactiveError::Invalid("unused tool payload"));
        }
        for tool in &self.tools {
            if tool.arity != self.arity {
                return Err(ReactiveError::Invalid("tool schema"));
            }
            tool.validate_shape(depth + 1)?;
        }
        Ok(())
    }
    fn validate(&self) -> Result<(), ReactiveError> {
        self.validate_shape(0)?;
        validate_traces(&self.traces, self.arity, ReactiveBudget::default())?;
        let circuit = self.compile()?;
        for trace in &self.traces {
            let mut cursor = circuit.cursor();
            for e in &trace.events {
                if circuit.step(&mut cursor, self.context(), &e.input)? != e.output {
                    return Err(ReactiveError::Invalid("training replay"));
                }
            }
        }
        for tool in &self.tools {
            tool.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Cell {
    update: Option<Arc<Expr>>,
    output: Arc<Expr>,
    initial: i64,
    after_update: bool,
    tools: [u8; 4],
    tool_count: usize,
}
#[derive(Debug)]
pub struct ReactiveCircuit {
    context: String,
    arity: usize,
    identity: u64,
    cells: Vec<Cell>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReactiveCursor {
    identity: u64,
    ticks: u64,
    states: [i64; MAX_CELLS],
    suspended: bool,
}
impl ReactiveCursor {
    pub fn ticks(&self) -> u64 {
        self.ticks
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
    }
}
impl ReactiveCircuit {
    fn append(&mut self, skill: &ReactiveSkill, seen: &mut Vec<(Vec<u8>, u8)>) -> u8 {
        let bytes = skill.to_bytes();
        if let Some((_, id)) = seen.iter().find(|(b, _)| *b == bytes) {
            return *id;
        }
        let mut tools = [0; 4];
        for (i, tool) in skill.tools.iter().enumerate() {
            tools[i] = self.append(tool, seen);
        }
        let id = self.cells.len() as u8;
        self.cells.push(Cell {
            update: skill.update.clone(),
            output: skill.output.clone(),
            initial: skill.initial,
            after_update: skill.after_update,
            tools,
            tool_count: skill.tools.len(),
        });
        seen.push((bytes, id));
        id
    }
    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn cells(&self) -> usize {
        self.cells.len()
    }
    pub fn memory_registers(&self) -> usize {
        self.cells.iter().filter(|c| c.update.is_some()).count()
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>() + self.context.capacity() + self.cells.capacity() * size_of::<Cell>()
    }
    pub fn cursor(&self) -> ReactiveCursor {
        let mut states = [0; MAX_CELLS];
        for (i, cell) in self.cells.iter().enumerate() {
            states[i] = cell.initial;
        }
        ReactiveCursor {
            identity: self.identity,
            ticks: 0,
            states,
            suspended: false,
        }
    }
    pub fn step(
        &self,
        cursor: &mut ReactiveCursor,
        context: &str,
        input: &[i64],
    ) -> Result<i64, ReactiveError> {
        let mut work = 0;
        self.step_counted(cursor, context, input, &mut work, 1024)
    }
    fn step_counted(
        &self,
        cursor: &mut ReactiveCursor,
        context: &str,
        input: &[i64],
        work: &mut u64,
        limit: u64,
    ) -> Result<i64, ReactiveError> {
        if context != self.context {
            return Err(ReactiveError::Context);
        }
        if cursor.identity != self.identity {
            return Err(ReactiveError::Identity);
        }
        if cursor.suspended {
            return Err(ReactiveError::Suspended);
        }
        if input.len() != self.arity {
            return Err(ReactiveError::Domain("input schema"));
        }
        let ticks = cursor
            .ticks
            .checked_add(1)
            .ok_or(ReactiveError::Domain("tick overflow"))?;
        let mut states = cursor.states;
        let mut outputs = [0; MAX_CELLS];
        // Children are topologically ordered and run once; state commits only after all guards pass.
        for (i, cell) in self.cells.iter().enumerate() {
            let mut values = [0; 4];
            for (j, value) in values.iter_mut().enumerate().take(cell.tool_count) {
                *value = outputs[cell.tools[j] as usize];
            }
            let before = states[i];
            if let Some(update) = &cell.update {
                states[i] = update.eval(
                    Frame {
                        input,
                        state: before,
                        tools: &values[..cell.tool_count],
                    },
                    work,
                    limit,
                )?;
            }
            outputs[i] = cell.output.eval(
                Frame {
                    input,
                    state: if cell.after_update { states[i] } else { before },
                    tools: &values[..cell.tool_count],
                },
                work,
                limit,
            )?;
        }
        cursor.states = states;
        cursor.ticks = ticks;
        Ok(outputs[self.cells.len() - 1])
    }
    pub fn feedback(
        &self,
        cursor: &mut ReactiveCursor,
        predicted: i64,
        actual: i64,
    ) -> Result<(), ReactiveError> {
        if cursor.identity != self.identity {
            return Err(ReactiveError::Identity);
        }
        if cursor.suspended {
            return Err(ReactiveError::Suspended);
        }
        if predicted != actual {
            cursor.suspended = true;
            return Err(ReactiveError::Drift {
                expected: predicted,
                actual,
            });
        }
        Ok(())
    }
}
fn identifier(s: &str) -> Result<(), ReactiveError> {
    if s.is_empty() || s.len() > 256 || s.chars().any(char::is_control) {
        return Err(ReactiveError::Invalid("identifier"));
    }
    Ok(())
}
fn validate_traces(
    traces: &[ReactiveTrace],
    arity: usize,
    budget: ReactiveBudget,
) -> Result<(), ReactiveError> {
    budget.validate()?;
    let samples = traces
        .iter()
        .try_fold(0usize, |sum, t| sum.checked_add(t.events.len()))
        .ok_or(ReactiveError::Invalid("sample overflow"))?;
    if !(1..=16).contains(&traces.len())
        || !(4..=budget.max_samples).contains(&samples)
        || traces.iter().any(|t| {
            t.events.is_empty()
                || t.events.len() > budget.max_ticks
                || t.events.iter().any(|e| e.input.len() != arity)
        })
        || !(1..=2).contains(&arity)
    {
        return Err(ReactiveError::Invalid("demonstration shape or quota"));
    }
    Ok(())
}
