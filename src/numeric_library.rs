//! Bounded structural abstraction mining. Compression is a selection signal, not truth.
use super::{
    NumericBudget, NumericError, NumericExample, NumericProgram, NumericSearch, NumericSkill,
    NumericTolerance, NumericTool, Op, Reader, identifier,
};
use crate::{
    live_store::checksum,
    text_skill::{publish_new, read_bounded},
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    mem::size_of,
    path::Path,
    time::Instant,
};

const MAGIC: &[u8; 8] = b"AXNL0001";
const MAX_FILE: u64 = 262_144;
const MAX_CORPUS_OPS: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub struct MiningBudget {
    pub max_patterns: usize,
    pub max_visits: u64,
    pub max_modules: usize,
    pub max_body: usize,
}
impl Default for MiningBudget {
    fn default() -> Self {
        Self {
            max_patterns: 4096,
            max_visits: 10_000_000,
            max_modules: 8,
            max_body: 32,
        }
    }
}
impl MiningBudget {
    pub(super) fn validate(self) -> Result<(), NumericError> {
        if !(1..=16_384).contains(&self.max_patterns)
            || !(1..=100_000_000).contains(&self.max_visits)
            || !(1..=16).contains(&self.max_modules)
            || !(3..=64).contains(&self.max_body)
        {
            return Err(NumericError::Invalid("abstraction mining budget"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DevelopmentBudget {
    pub max_attempts: usize,
    pub max_work: u64,
    pub mining: MiningBudget,
}
impl Default for DevelopmentBudget {
    fn default() -> Self {
        Self {
            max_attempts: 8,
            max_work: 100_000_000,
            mining: MiningBudget::default(),
        }
    }
}
#[derive(Clone, Debug)]
pub struct DevelopmentAttempt {
    pub proposed_tool: Option<u64>,
    pub compatible: bool,
    pub search: NumericSearch,
}
#[derive(Clone, Debug, Default)]
pub struct DevelopmentTrace {
    pub attempts: Vec<DevelopmentAttempt>,
    pub scalar_work: u64,
    pub proposal_patterns: usize,
    pub proposal_visits: u64,
    pub elapsed_ns: u64,
    pub admitted: Option<NumericAbstraction>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MiningStatistics {
    pub supplied_sources: usize,
    pub unique_sources: usize,
    pub patterns: usize,
    pub tree_visits: u64,
    pub original_nodes: usize,
    pub rewritten_nodes: usize,
    pub definition_nodes: usize,
    pub net_saved_nodes: usize,
    pub elapsed_ns: u64,
}

#[derive(Clone, Debug)]
pub struct NumericAbstraction {
    program: NumericProgram,
    sources: Vec<u64>,
    net_saved_nodes: usize,
}
impl NumericAbstraction {
    pub fn program(&self) -> &NumericProgram {
        &self.program
    }
    pub fn sources(&self) -> &[u64] {
        &self.sources
    }
    pub fn net_saved_nodes(&self) -> usize {
        self.net_saved_nodes
    }
    pub fn fingerprint(&self) -> u64 {
        let mut b = Vec::new();
        write_program(&mut b, &self.program);
        checksum(&b)
    }
}

#[derive(Clone, Debug)]
struct Source {
    identity: u64,
    program: NumericProgram,
}

/// The archive includes source programs for reproducible structural audits.
/// It does not authenticate source authors or retain a general proof of their validity.
#[derive(Clone, Debug)]
pub struct NumericLibrary {
    context: String,
    corpus: Vec<Source>,
    modules: Vec<NumericAbstraction>,
    statistics: MiningStatistics,
}
impl NumericLibrary {
    /// Reopen recurring mechanisms on impasse. Only labeled task demonstrations guide admission.
    pub fn develop(
        &self,
        name: impl Into<String>,
        context: impl Into<String>,
        examples: Vec<NumericExample>,
        search_budget: NumericBudget,
        development: DevelopmentBudget,
        tolerance: NumericTolerance,
    ) -> Result<(NumericSkill, DevelopmentTrace), NumericError> {
        let started = Instant::now();
        search_budget.validate()?;
        development.mining.validate()?;
        if !(1..=16).contains(&development.max_attempts)
            || !(1..=100_000_000).contains(&development.max_work)
        {
            return Err(NumericError::Invalid("development budget"));
        }
        let name = name.into();
        let context = context.into();
        let mut trace = DevelopmentTrace::default();
        let tools = self.tools();
        let first = try_task(
            &name,
            &context,
            &examples,
            &tools,
            None,
            search_budget,
            development,
            &mut trace,
            tolerance,
        )?;
        if let Some(skill) = first {
            trace.elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            return Ok((skill, trace));
        }
        if tools.len() >= 16 {
            return Err(development_error(&trace, "active tool limit"));
        }
        let trees: Vec<_> = self
            .corpus
            .iter()
            .map(|s| Tree::from_program(&s.program))
            .collect();
        let mut counter = Counter {
            visits: 0,
            maximum: development.mining.max_visits,
            patterns: 0,
        };
        let candidates = proposals(&trees, development.mining, &mut counter)?;
        let original: Vec<_> = trees.iter().map(Tree::size).collect();
        let mut ranked = Vec::new();
        for candidate in candidates {
            let program = candidate.program();
            let mut encoded = Vec::new();
            write_program(&mut encoded, &program);
            let identity = checksum(&encoded);
            if tools.iter().any(|tool| tool.identity == identity) {
                continue;
            }
            let costs = corpus_costs(&trees, std::slice::from_ref(&candidate), &mut counter)?;
            let sources: Vec<_> = original
                .iter()
                .zip(&costs)
                .enumerate()
                .filter(|(_, (old, new))| new < old)
                .map(|(i, _)| self.corpus[i].identity)
                .collect();
            let gross = original.iter().sum::<usize>() - costs.iter().sum::<usize>();
            if sources.len() >= 2 && gross > candidate.definition_cost() {
                ranked.push((
                    gross - candidate.definition_cost(),
                    candidate,
                    program,
                    sources,
                ));
            }
        }
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        trace.proposal_patterns = counter.patterns;
        trace.proposal_visits = counter.visits;
        for (net_saved_nodes, _, program, sources) in ranked {
            let module = NumericAbstraction {
                program,
                sources,
                net_saved_nodes,
            };
            let identity = module.fingerprint();
            let mut trial = self.tools();
            trial.push(NumericTool {
                program: &module.program,
                identity,
            });
            let result = try_task(
                &name,
                &context,
                &examples,
                &trial,
                Some(identity),
                search_budget,
                development,
                &mut trace,
                tolerance,
            );
            drop(trial);
            if let Some(skill) = result? {
                trace.admitted = Some(module);
                trace.elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                return Ok((skill, trace));
            }
        }
        Err(development_error(
            &trace,
            "no compatible recurring proposal in the explored pool",
        ))
    }
    pub fn mine(
        context: impl Into<String>,
        skills: &[&NumericSkill],
        budget: MiningBudget,
    ) -> Result<Self, NumericError> {
        let started = Instant::now();
        let context = context.into();
        identifier(&context)?;
        budget.validate()?;
        if skills.is_empty()
            || skills.len() > 64
            || skills
                .iter()
                .map(|s| s.program().instructions())
                .sum::<usize>()
                > MAX_CORPUS_OPS
        {
            return Err(NumericError::Invalid(
                "requires 1..64 source skills, at most 4096 instructions",
            ));
        }
        // Renaming or copying an artifact cannot manufacture independent structural support.
        let mut unique: BTreeMap<Tree, Source> = BTreeMap::new();
        let mut counter = Counter {
            visits: 0,
            maximum: budget.max_visits,
            patterns: 0,
        };
        for skill in skills {
            let tree = Tree::from_program(skill.program());
            let key = generalize(&tree, &[], &mut Vec::new(), &mut counter)?.expect("source arity");
            let source = Source {
                identity: skill.fingerprint(),
                program: skill.program().clone(),
            };
            match unique.get_mut(&key) {
                Some(previous) if source.identity < previous.identity => *previous = source,
                Some(_) => {}
                None => {
                    unique.insert(key, source);
                }
            }
        }
        let corpus: Vec<_> = unique.into_values().collect();
        let trees: Vec<_> = corpus
            .iter()
            .map(|s| Tree::from_program(&s.program))
            .collect();
        let candidates = proposals(&trees, budget, &mut counter)?;
        let mut selected: Vec<Tree> = Vec::new();
        let mut modules = Vec::new();
        let original_nodes = trees.iter().map(Tree::size).sum::<usize>();
        let mut costs: Vec<_> = trees.iter().map(Tree::size).collect();
        while selected.len() < budget.max_modules {
            let mut best: Option<(Tree, Vec<usize>, Vec<u64>, usize)> = None;
            for candidate in &candidates {
                if selected.contains(candidate) {
                    continue;
                }
                let mut trial = selected.clone();
                trial.push(candidate.clone());
                let next = corpus_costs(&trees, &trial, &mut counter)?;
                let sources: Vec<_> = costs
                    .iter()
                    .zip(&next)
                    .enumerate()
                    .filter(|(_, (old, new))| new < old)
                    .map(|(i, _)| corpus[i].identity)
                    .collect();
                let gross = costs.iter().sum::<usize>() - next.iter().sum::<usize>();
                let charge = candidate.definition_cost();
                if sources.len() < 2 || gross <= charge {
                    continue;
                }
                let net = gross - charge;
                // Sorted candidates and strict comparison make ties deterministic.
                if best.as_ref().is_none_or(|(_, _, _, score)| net > *score) {
                    best = Some((candidate.clone(), next, sources, net));
                }
            }
            let Some((pattern, next, sources, net_saved_nodes)) = best else {
                break;
            };
            modules.push(NumericAbstraction {
                program: pattern.program(),
                sources,
                net_saved_nodes,
            });
            selected.push(pattern);
            costs = next;
        }
        let rewritten_nodes = costs.iter().sum();
        let definition_nodes = selected.iter().map(Tree::definition_cost).sum::<usize>();
        let statistics = MiningStatistics {
            supplied_sources: skills.len(),
            unique_sources: corpus.len(),
            patterns: candidates.len(),
            tree_visits: counter.visits,
            original_nodes,
            rewritten_nodes,
            definition_nodes,
            net_saved_nodes: original_nodes - rewritten_nodes - definition_nodes,
            elapsed_ns: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        };
        Ok(Self {
            context,
            corpus,
            modules,
            statistics,
        })
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn modules(&self) -> &[NumericAbstraction] {
        &self.modules
    }
    pub fn statistics(&self) -> MiningStatistics {
        self.statistics
    }
    pub(super) fn tools(&self) -> Vec<NumericTool<'_>> {
        self.modules
            .iter()
            .map(|m| NumericTool {
                program: &m.program,
                identity: m.fingerprint(),
            })
            .collect()
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.context.capacity()
            + self.corpus.capacity() * size_of::<Source>()
            + self.modules.capacity() * size_of::<NumericAbstraction>()
            + self
                .corpus
                .iter()
                .map(|s| s.program.ops.capacity() * size_of::<Op>())
                .sum::<usize>()
            + self
                .modules
                .iter()
                .map(|m| m.program.ops.capacity() * size_of::<Op>() + m.sources.capacity() * 8)
                .sum::<usize>()
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.payload(false))
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        b.extend(self.payload(true));
        b.extend(checksum(&b).to_le_bytes());
        b
    }
    fn payload(&self, statistics: bool) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend((self.context.len() as u32).to_le_bytes());
        b.extend(self.context.as_bytes());
        b.extend((self.corpus.len() as u32).to_le_bytes());
        for source in &self.corpus {
            b.extend(source.identity.to_le_bytes());
            write_program(&mut b, &source.program);
        }
        b.extend((self.modules.len() as u32).to_le_bytes());
        for module in &self.modules {
            write_program(&mut b, &module.program);
            b.extend((module.sources.len() as u32).to_le_bytes());
            for id in &module.sources {
                b.extend(id.to_le_bytes());
            }
            b.extend((module.net_saved_nodes as u32).to_le_bytes());
        }
        if statistics {
            let s = self.statistics;
            for n in [
                s.supplied_sources,
                s.unique_sources,
                s.patterns,
                s.original_nodes,
                s.rewritten_nodes,
                s.definition_nodes,
                s.net_saved_nodes,
            ] {
                b.extend((n as u32).to_le_bytes());
            }
            b.extend(s.tree_visits.to_le_bytes());
            b.extend(s.elapsed_ns.to_le_bytes());
        }
        b
    }
    pub fn save_new(&self, path: &Path) -> Result<(), NumericError> {
        publish_new(path, &self.to_bytes()).map_err(Into::into)
    }
    pub fn open(path: &Path, context: &str) -> Result<Self, NumericError> {
        let library = Self::from_bytes(&read_bounded(path, MAX_FILE)?)?;
        if library.context != context {
            return Err(NumericError::Context);
        }
        Ok(library)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NumericError> {
        if bytes.len() < 16 || bytes.len() as u64 > MAX_FILE || &bytes[..8] != MAGIC {
            return Err(NumericError::Invalid("library header or size"));
        }
        let body = bytes.len() - 8;
        if checksum(&bytes[..body])
            != u64::from_le_bytes(bytes[body..].try_into().expect("eight bytes"))
        {
            return Err(NumericError::Invalid("library checksum"));
        }
        let mut r = Reader(&bytes[8..body]);
        let context = r.text()?;
        identifier(&context)?;
        let mut corpus = Vec::new();
        for _ in 0..r.count(64)? {
            corpus.push(Source {
                identity: r.u64()?,
                program: read_program(&mut r)?,
            });
        }
        let mut modules = Vec::new();
        for _ in 0..r.count(16)? {
            let program = read_program(&mut r)?;
            let mut sources = Vec::new();
            for _ in 0..r.count(64)? {
                sources.push(r.u64()?);
            }
            modules.push(NumericAbstraction {
                program,
                sources,
                net_saved_nodes: r.u32()? as usize,
            });
        }
        let statistics = MiningStatistics {
            supplied_sources: r.u32()? as usize,
            unique_sources: r.u32()? as usize,
            patterns: r.u32()? as usize,
            original_nodes: r.u32()? as usize,
            rewritten_nodes: r.u32()? as usize,
            definition_nodes: r.u32()? as usize,
            net_saved_nodes: r.u32()? as usize,
            tree_visits: r.u64()?,
            elapsed_ns: r.u64()?,
        };
        if !r.0.is_empty() {
            return Err(NumericError::Invalid("library trailing data"));
        }
        let library = Self {
            context,
            corpus,
            modules,
            statistics,
        };
        library.audit()?;
        Ok(library)
    }
    fn audit(&self) -> Result<(), NumericError> {
        let s = self.statistics;
        if self.corpus.is_empty()
            || s.unique_sources != self.corpus.len()
            || s.supplied_sources < self.corpus.len()
            || s.supplied_sources > 64
            || s.patterns > 16_384
            || s.tree_visits > 100_000_000
            || self
                .corpus
                .iter()
                .map(|s| s.program.instructions())
                .sum::<usize>()
                > MAX_CORPUS_OPS
        {
            return Err(NumericError::Invalid("library corpus or statistics"));
        }
        let trees: Vec<_> = self
            .corpus
            .iter()
            .map(|s| Tree::from_program(&s.program))
            .collect();
        let mut counter = Counter {
            visits: 0,
            maximum: 100_000_000,
            patterns: s.patterns,
        };
        let mut unique = BTreeSet::new();
        let mut identities = BTreeSet::new();
        for (tree, source) in trees.iter().zip(&self.corpus) {
            let key = generalize(tree, &[], &mut Vec::new(), &mut counter)?.expect("arity");
            if !unique.insert(key) || !identities.insert(source.identity) {
                return Err(NumericError::Invalid("duplicate library source"));
            }
        }
        let original_nodes = trees.iter().map(Tree::size).sum::<usize>();
        let mut costs: Vec<_> = trees.iter().map(Tree::size).collect();
        let mut selected = Vec::new();
        for module in &self.modules {
            let pattern = Tree::from_program(&module.program);
            let canonical =
                generalize(&pattern, &[], &mut Vec::new(), &mut counter)?.expect("arity");
            if pattern != canonical
                || pattern.arity() != module.program.arity()
                || pattern.size() > 64
                || pattern.size() <= pattern.arity() + 1
                || selected.contains(&pattern)
            {
                return Err(NumericError::Invalid("noncanonical or trivial abstraction"));
            }
            selected.push(pattern.clone());
            let next = corpus_costs(&trees, &selected, &mut counter)?;
            let sources: Vec<_> = costs
                .iter()
                .zip(&next)
                .enumerate()
                .filter(|(_, (old, new))| new < old)
                .map(|(i, _)| self.corpus[i].identity)
                .collect();
            let gross = costs.iter().sum::<usize>() - next.iter().sum::<usize>();
            if sources.len() < 2
                || sources != module.sources
                || gross <= pattern.definition_cost()
                || gross - pattern.definition_cost() != module.net_saved_nodes
            {
                return Err(NumericError::Invalid(
                    "abstraction support or compression audit",
                ));
            }
            costs = next;
        }
        let rewritten_nodes = costs.iter().sum::<usize>();
        let definition_nodes = selected.iter().map(Tree::definition_cost).sum::<usize>();
        if original_nodes != s.original_nodes
            || rewritten_nodes != s.rewritten_nodes
            || definition_nodes != s.definition_nodes
            || original_nodes - rewritten_nodes - definition_nodes != s.net_saved_nodes
        {
            return Err(NumericError::Invalid("library compression totals"));
        }
        Ok(())
    }
}

fn development_error(trace: &DevelopmentTrace, reason: &'static str) -> NumericError {
    NumericError::Development {
        attempts: trace.attempts.len(),
        scalar_work: trace.scalar_work,
        reason,
    }
}
#[allow(clippy::too_many_arguments)]
fn try_task(
    name: &str,
    context: &str,
    examples: &[NumericExample],
    tools: &[NumericTool<'_>],
    proposed_tool: Option<u64>,
    mut budget: NumericBudget,
    development: DevelopmentBudget,
    trace: &mut DevelopmentTrace,
    tolerance: NumericTolerance,
) -> Result<Option<NumericSkill>, NumericError> {
    if trace.attempts.len() >= development.max_attempts || trace.scalar_work >= development.max_work
    {
        return Err(development_error(
            trace,
            "aggregate search budget exhausted",
        ));
    }
    budget.max_work = budget
        .max_work
        .min(development.max_work - trace.scalar_work);
    let result = NumericSkill::learn_tools(
        name.to_owned(),
        context.to_owned(),
        examples.to_vec(),
        tools,
        budget,
        tolerance,
    );
    let (skill, search) = match result {
        Ok(skill) => {
            let search = skill.search();
            (Some(skill), search)
        }
        Err(NumericError::NoProgram(s) | NumericError::Budget(s)) => (None, s),
        Err(error) => return Err(error),
    };
    trace.scalar_work += search.scalar_work;
    trace.attempts.push(DevelopmentAttempt {
        proposed_tool,
        compatible: skill.is_some(),
        search,
    });
    Ok(skill)
}

fn proposals(
    trees: &[Tree],
    budget: MiningBudget,
    counter: &mut Counter,
) -> Result<BTreeSet<Tree>, NumericError> {
    let mut candidates = BTreeSet::new();
    for tree in trees {
        let mut nodes = Vec::new();
        tree.locations(&mut nodes);
        for node in nodes {
            let mut descendants = Vec::new();
            node.locations(&mut descendants);
            propose(node, &[], &mut candidates, counter, budget)?;
            for i in 1..descendants.len() {
                propose(node, &[descendants[i]], &mut candidates, counter, budget)?;
                for j in i + 1..descendants.len() {
                    propose(
                        node,
                        &[descendants[i], descendants[j]],
                        &mut candidates,
                        counter,
                        budget,
                    )?;
                }
            }
        }
    }
    Ok(candidates)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Tree {
    Input(u8),
    Constant(u64),
    Root(Box<Tree>),
    Binary(u8, Box<Tree>, Box<Tree>),
}
impl Tree {
    fn from_program(program: &NumericProgram) -> Self {
        let mut stack = Vec::new();
        for op in &program.ops {
            let node = match *op {
                Op::Input(i) => Self::Input(i),
                Op::Constant(n) => Self::Constant(n.to_bits()),
                Op::Sqrt => Self::Root(Box::new(stack.pop().expect("validated"))),
                _ => {
                    let b = stack.pop().expect("validated");
                    let a = stack.pop().expect("validated");
                    let tag = match op {
                        Op::Add => 2,
                        Op::Subtract => 3,
                        Op::Multiply => 4,
                        _ => 5,
                    };
                    Self::Binary(tag, Box::new(a), Box::new(b))
                }
            };
            stack.push(node);
        }
        stack.pop().expect("validated")
    }
    fn size(&self) -> usize {
        match self {
            Self::Root(a) => 1 + a.size(),
            Self::Binary(_, a, b) => 1 + a.size() + b.size(),
            _ => 1,
        }
    }
    fn arity(&self) -> usize {
        match self {
            Self::Input(i) => *i as usize + 1,
            Self::Root(a) => a.arity(),
            Self::Binary(_, a, b) => a.arity().max(b.arity()),
            _ => 0,
        }
    }
    fn definition_cost(&self) -> usize {
        self.size() + self.arity() + 1
    }
    fn locations<'a>(&'a self, nodes: &mut Vec<&'a Tree>) {
        nodes.push(self);
        match self {
            Self::Root(a) => a.locations(nodes),
            Self::Binary(_, a, b) => {
                a.locations(nodes);
                b.locations(nodes);
            }
            _ => {}
        }
    }
    fn program(&self) -> NumericProgram {
        let mut ops = Vec::new();
        self.emit(&mut ops);
        NumericProgram {
            arity: self.arity(),
            ops,
        }
    }
    fn emit(&self, ops: &mut Vec<Op>) {
        match self {
            Self::Input(i) => ops.push(Op::Input(*i)),
            Self::Constant(bits) => ops.push(Op::Constant(f64::from_bits(*bits))),
            Self::Root(a) => {
                a.emit(ops);
                ops.push(Op::Sqrt);
            }
            Self::Binary(tag, a, b) => {
                a.emit(ops);
                b.emit(ops);
                ops.push(binary(*tag));
            }
        }
    }
}
fn binary(tag: u8) -> Op {
    match tag {
        2 => Op::Add,
        3 => Op::Subtract,
        4 => Op::Multiply,
        _ => Op::Divide,
    }
}
struct Counter {
    visits: u64,
    maximum: u64,
    patterns: usize,
}
impl Counter {
    fn tick(&mut self) -> Result<(), NumericError> {
        if self.visits >= self.maximum {
            return Err(NumericError::LibraryBudget {
                patterns: self.patterns,
                visits: self.visits,
            });
        }
        self.visits += 1;
        Ok(())
    }
}
fn generalize<'a>(
    tree: &'a Tree,
    cuts: &[&Tree],
    bindings: &mut Vec<&'a Tree>,
    counter: &mut Counter,
) -> Result<Option<Tree>, NumericError> {
    counter.tick()?;
    if matches!(tree, Tree::Input(_)) || cuts.contains(&tree) {
        let slot = match bindings.iter().position(|x| *x == tree) {
            Some(i) => i,
            None if bindings.len() < 4 => {
                bindings.push(tree);
                bindings.len() - 1
            }
            None => return Ok(None),
        };
        return Ok(Some(Tree::Input(slot as u8)));
    }
    Ok(Some(match tree {
        Tree::Constant(bits) => Tree::Constant(*bits),
        Tree::Root(a) => {
            let Some(a) = generalize(a, cuts, bindings, counter)? else {
                return Ok(None);
            };
            Tree::Root(Box::new(a))
        }
        Tree::Binary(tag, a, b) => {
            let Some(a) = generalize(a, cuts, bindings, counter)? else {
                return Ok(None);
            };
            let Some(b) = generalize(b, cuts, bindings, counter)? else {
                return Ok(None);
            };
            Tree::Binary(*tag, Box::new(a), Box::new(b))
        }
        Tree::Input(_) => unreachable!(),
    }))
}
fn propose(
    tree: &Tree,
    cuts: &[&Tree],
    candidates: &mut BTreeSet<Tree>,
    counter: &mut Counter,
    budget: MiningBudget,
) -> Result<(), NumericError> {
    if let Some(pattern) = generalize(tree, cuts, &mut Vec::new(), counter)?
        && pattern.arity() > 0
        && pattern.size() > pattern.arity() + 1
        && pattern.size() <= budget.max_body
        && !candidates.contains(&pattern)
    {
        if candidates.len() >= budget.max_patterns {
            return Err(NumericError::LibraryBudget {
                patterns: candidates.len(),
                visits: counter.visits,
            });
        }
        candidates.insert(pattern);
        counter.patterns = candidates.len();
    }
    Ok(())
}
fn bind<'a>(
    pattern: &Tree,
    tree: &'a Tree,
    args: &mut [Option<&'a Tree>; 4],
    counter: &mut Counter,
) -> Result<bool, NumericError> {
    counter.tick()?;
    Ok(match (pattern, tree) {
        (Tree::Input(i), _) => match args[*i as usize] {
            Some(previous) => previous == tree,
            None => {
                args[*i as usize] = Some(tree);
                true
            }
        },
        (Tree::Constant(a), Tree::Constant(b)) => a == b,
        (Tree::Root(a), Tree::Root(b)) => bind(a, b, args, counter)?,
        (Tree::Binary(tag, a, b), Tree::Binary(other, c, d)) if tag == other => {
            bind(a, c, args, counter)? && bind(b, d, args, counter)?
        }
        _ => false,
    })
}
fn compressed_cost<'a>(
    tree: &'a Tree,
    patterns: &[Tree],
    counter: &mut Counter,
    memo: &mut HashMap<&'a Tree, usize>,
) -> Result<usize, NumericError> {
    counter.tick()?;
    if let Some(cost) = memo.get(tree) {
        return Ok(*cost);
    }
    let mut cost = match tree {
        Tree::Root(a) => 1 + compressed_cost(a, patterns, counter, memo)?,
        Tree::Binary(_, a, b) => {
            1 + compressed_cost(a, patterns, counter, memo)?
                + compressed_cost(b, patterns, counter, memo)?
        }
        _ => 1,
    };
    for pattern in patterns {
        let mut args = [None; 4];
        if bind(pattern, tree, &mut args, counter)? {
            let mut call = 1;
            for arg in args[..pattern.arity()].iter().flatten() {
                call += compressed_cost(arg, patterns, counter, memo)?;
            }
            cost = cost.min(call);
        }
    }
    memo.insert(tree, cost);
    Ok(cost)
}
fn corpus_costs(
    trees: &[Tree],
    patterns: &[Tree],
    counter: &mut Counter,
) -> Result<Vec<usize>, NumericError> {
    trees
        .iter()
        .map(|tree| compressed_cost(tree, patterns, counter, &mut HashMap::new()))
        .collect()
}
pub(super) fn write_program(b: &mut Vec<u8>, program: &NumericProgram) {
    b.push(program.arity() as u8);
    b.extend((program.instructions() as u32).to_le_bytes());
    for op in &program.ops {
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
}
pub(super) fn read_program(r: &mut Reader<'_>) -> Result<NumericProgram, NumericError> {
    let arity = r.byte()? as usize;
    let mut ops = Vec::new();
    for _ in 0..r.count(super::MAX_OPS)? {
        ops.push(match r.byte()? {
            0 => Op::Input(r.byte()?),
            1 => Op::Constant(r.number()?),
            2 => Op::Add,
            3 => Op::Subtract,
            4 => Op::Multiply,
            5 => Op::Divide,
            6 => Op::Sqrt,
            _ => return Err(NumericError::Invalid("library opcode")),
        });
    }
    let program = NumericProgram { arity, ops };
    program.validate()?;
    Ok(program)
}
