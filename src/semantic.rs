//! Bounded relational induction. Supervised roles and a numeric instruction set are priors,
//! not innate Portuguese semantics. Personal state is deliberately outside the inherited seed.
use crate::{
    live_store::checksum,
    numeric::{NumericBudget, NumericExample, NumericSkill, NumericTolerance},
    text_skill::{publish_new, read_bounded},
};
use std::{collections::BTreeMap, fmt, mem::size_of, path::Path};
#[path = "semantic_binding.rs"]
mod binding;
use binding::Relational;

pub const MAX_BYTES: usize = 131_072;
const MAGIC: &[u8; 8] = b"AXSM0001";
const CONTEXT: &str = "axon/relational-v1";

#[derive(Debug)]
pub enum SemanticError {
    Invalid(&'static str),
    Unknown,
    Ambiguous,
    Missing,
    Domain,
    Storage(String),
}
impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Invalid(s) => s,
            Self::Unknown => "unlearned sentence",
            Self::Ambiguous => "ambiguous rule or reference",
            Self::Missing => "missing prior fact",
            Self::Domain => "numeric domain refused",
            Self::Storage(s) => s,
        })
    }
}
impl std::error::Error for SemanticError {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Focus {
    pub owners: Vec<String>,
    pub items: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct Change {
    pub owner: String,
    pub item: String,
    pub before: Option<f64>,
    pub after: f64,
}
#[derive(Clone, Debug)]
pub enum Meaning {
    Change(Vec<Change>),
    Question {
        cells: Vec<(String, String, f64)>,
        answer: f64,
    },
    Calculate {
        arguments: Vec<f64>,
        answer: f64,
    },
    Ignore,
}
#[derive(Clone, Debug)]
pub struct Demonstration {
    pub sentence: String,
    pub focus: Focus,
    pub meaning: Meaning,
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Source {
    Token(u8),
    Owner,
    Item,
}
#[derive(Clone, Debug)]
struct Cell {
    owner: Source,
    item: Source,
}
#[derive(Clone, Debug)]
struct Update {
    cell: Cell,
    quantity: Source,
    initialize: bool,
    program: NumericSkill,
}
#[derive(Clone, Debug)]
enum Action {
    Change(Vec<Update>),
    Question(Vec<Cell>, NumericSkill),
    Calculate(Vec<Source>, NumericSkill),
    Relational(Box<Relational>),
    Ignore,
}
#[derive(Clone, Debug)]
pub struct Rule {
    pattern: Vec<Option<String>>,
    action: Action,
}
#[derive(Clone, Debug, Default)]
pub struct SemanticSeed {
    rules: Vec<Rule>,
}

fn words(s: &str) -> Result<Vec<String>, SemanticError> {
    if s.is_empty() || s.len() > 1024 || s.chars().any(char::is_control) {
        return Err(SemanticError::Invalid("sentence bounds"));
    }
    let v: Vec<_> = s
        .trim()
        .trim_end_matches(['.', '?', '!'])
        .split_whitespace()
        .map(str::to_lowercase)
        .collect();
    if v.is_empty() || v.len() > 32 {
        return Err(SemanticError::Invalid("token bounds"));
    }
    Ok(v)
}
fn number(s: &str) -> Result<f64, SemanticError> {
    s.replace(',', ".")
        .parse::<f64>()
        .ok()
        .filter(|n| n.is_finite() && n.abs() <= 1e12)
        .ok_or(SemanticError::Domain)
}
fn value(source: &Source, w: &[String], focus: &Focus) -> Result<String, SemanticError> {
    match source {
        Source::Token(i) => w.get(*i as usize).cloned().ok_or(SemanticError::Unknown),
        Source::Owner | Source::Item => {
            let v = if *source == Source::Owner {
                &focus.owners
            } else {
                &focus.items
            };
            if v.len() != 1 {
                return Err(SemanticError::Ambiguous);
            }
            Ok(v[0].to_lowercase())
        }
    }
}
fn infer_source(
    w: &[Vec<String>],
    ds: &[Demonstration],
    desired: &[String],
) -> Result<Source, SemanticError> {
    let candidates = (0..w[0].len())
        .map(|i| Source::Token(i as u8))
        .chain([Source::Owner, Source::Item]);
    let valid: Vec<_> = candidates
        .filter(|s| {
            ds.iter().enumerate().all(|(i, d)| {
                value(s, &w[i], &d.focus).is_ok_and(|v| v == desired[i].to_lowercase())
            })
        })
        .collect();
    if valid.len() != 1 {
        return Err(SemanticError::Ambiguous);
    }
    Ok(valid[0].clone())
}
fn fit(
    examples: Vec<NumericExample>,
    budget: NumericBudget,
) -> Result<NumericSkill, SemanticError> {
    NumericSkill::learn(
        "induced",
        CONTEXT,
        examples,
        &[],
        budget,
        NumericTolerance::default(),
    )
    .map_err(|e| SemanticError::Storage(e.to_string()))
}
impl Rule {
    /// Learn local lexical role bindings across different sentence orders.
    pub fn learn_relational(
        ds: &[Demonstration],
        budget: NumericBudget,
        window: u8,
        parents: &[&Rule],
    ) -> Result<Self, SemanticError> {
        Relational::learn(ds, budget, window, parents)
    }
    pub fn learning_window(&self) -> Option<u8> {
        match &self.action {
            Action::Relational(r) => Some(r.window),
            _ => None,
        }
    }
    pub fn search_work(&self) -> u64 {
        match &self.action {
            Action::Relational(r) => r.program.search().scalar_work,
            Action::Calculate(_, p) | Action::Question(_, p) => p.search().scalar_work,
            Action::Change(v) => v.iter().map(|u| u.program.search().scalar_work).sum(),
            Action::Ignore => 0,
        }
    }
    fn matches(&self, w: &[String]) -> bool {
        match &self.action {
            Action::Relational(r) => {
                self.pattern.iter().flatten().all(|p| w.contains(p)) && r.bind(w).is_ok()
            }
            _ => {
                self.pattern.len() == w.len()
                    && self
                        .pattern
                        .iter()
                        .zip(w)
                        .all(|(p, v)| p.as_ref().is_none_or(|p| p == v))
            }
        }
    }
    /// No verb dictionary: literal anchors, role projections and numeric programs come from examples.
    pub fn learn(ds: &[Demonstration], budget: NumericBudget) -> Result<Self, SemanticError> {
        if !(4..=32).contains(&ds.len()) {
            return Err(SemanticError::Invalid("need 4..32 demonstrations"));
        }
        let w: Vec<_> = ds
            .iter()
            .map(|d| words(&d.sentence))
            .collect::<Result<_, _>>()?;
        if w.iter().any(|v| v.len() != w[0].len()) {
            return Err(SemanticError::Invalid("unaligned demonstrations"));
        }
        let pattern: Vec<_> = (0..w[0].len())
            .map(|i| w.iter().all(|v| v[i] == w[0][i]).then(|| w[0][i].clone()))
            .collect();
        if pattern.iter().all(Option::is_none) {
            return Err(SemanticError::Invalid("no discriminating anchor"));
        }
        let cell = |owners: Vec<String>, items: Vec<String>| -> Result<Cell, SemanticError> {
            Ok(Cell {
                owner: infer_source(&w, ds, &owners)?,
                item: infer_source(&w, ds, &items)?,
            })
        };
        let action = match &ds[0].meaning {
            Meaning::Change(first) if !first.is_empty() && first.len() <= 4 => {
                let all: Vec<_> = ds
                    .iter()
                    .map(|d| match &d.meaning {
                        Meaning::Change(v) if v.len() == first.len() => Ok(v),
                        _ => Err(SemanticError::Invalid("mixed demonstration signatures")),
                    })
                    .collect::<Result<_, _>>()?;
                let mut effects = Vec::new();
                for j in 0..first.len() {
                    let initialize = first[j].before.is_none();
                    if all.iter().any(|v| v[j].before.is_none() != initialize) {
                        return Err(SemanticError::Invalid("mixed initialization"));
                    }
                    let target = cell(
                        all.iter().map(|v| v[j].owner.clone()).collect(),
                        all.iter().map(|v| v[j].item.clone()).collect(),
                    )?;
                    let mut hypotheses = Vec::new();
                    for i in 0..w[0].len() {
                        if pattern[i].is_some() {
                            continue;
                        }
                        let qs: Result<Vec<_>, _> = w.iter().map(|v| number(&v[i])).collect();
                        if let Ok(qs) = qs {
                            let examples = all
                                .iter()
                                .zip(qs)
                                .map(|(v, q)| {
                                    NumericExample::new(
                                        vec![v[j].before.unwrap_or(0.0), q],
                                        v[j].after,
                                    )
                                })
                                .collect();
                            if let Ok(program) = fit(examples, budget) {
                                hypotheses.push((Source::Token(i as u8), program));
                            }
                        }
                    }
                    if hypotheses.len() != 1 {
                        return Err(SemanticError::Ambiguous);
                    }
                    let (quantity, program) = hypotheses.pop().expect("one");
                    effects.push(Update {
                        cell: target,
                        quantity,
                        initialize,
                        program,
                    });
                }
                Action::Change(effects)
            }
            Meaning::Question { cells: first, .. } if !first.is_empty() && first.len() <= 4 => {
                let all: Vec<_> = ds
                    .iter()
                    .map(|d| match &d.meaning {
                        Meaning::Question { cells, answer } if cells.len() == first.len() => {
                            Ok((cells, *answer))
                        }
                        _ => Err(SemanticError::Invalid("mixed questions")),
                    })
                    .collect::<Result<_, _>>()?;
                let mut cells = Vec::new();
                for j in 0..first.len() {
                    cells.push(cell(
                        all.iter().map(|(v, _)| v[j].0.clone()).collect(),
                        all.iter().map(|(v, _)| v[j].1.clone()).collect(),
                    )?);
                }
                let program = fit(
                    all.iter()
                        .map(|(v, a)| {
                            NumericExample::new(v.iter().map(|c| c.2).collect::<Vec<_>>(), *a)
                        })
                        .collect(),
                    budget,
                )?;
                Action::Question(cells, program)
            }
            Meaning::Calculate {
                arguments: first, ..
            } if !first.is_empty() && first.len() <= 4 => {
                let all: Vec<_> = ds
                    .iter()
                    .map(|d| match &d.meaning {
                        Meaning::Calculate { arguments, answer }
                            if arguments.len() == first.len() =>
                        {
                            Ok((arguments, *answer))
                        }
                        _ => Err(SemanticError::Invalid("mixed calculations")),
                    })
                    .collect::<Result<_, _>>()?;
                let mut sources = Vec::new();
                for j in 0..first.len() {
                    // Numeric equivalence permits decimal commas, unlike symbolic role matching.
                    let valid: Vec<_> = (0..w[0].len())
                        .filter(|i| {
                            all.iter()
                                .enumerate()
                                .all(|(k, (v, _))| number(&w[k][*i]).is_ok_and(|n| n == v[j]))
                        })
                        .collect();
                    if valid.len() != 1 {
                        return Err(SemanticError::Ambiguous);
                    }
                    sources.push(Source::Token(valid[0] as u8));
                }
                let program = fit(
                    all.iter()
                        .map(|(v, a)| NumericExample::new((*v).clone(), *a))
                        .collect(),
                    budget,
                )?;
                Action::Calculate(sources, program)
            }
            Meaning::Ignore if ds.iter().all(|d| matches!(d.meaning, Meaning::Ignore)) => {
                Action::Ignore
            }
            _ => return Err(SemanticError::Invalid("invalid teaching signature")),
        };
        let rule = Self { pattern, action };
        rule.validate()?;
        Ok(rule)
    }
    fn validate(&self) -> Result<(), SemanticError> {
        if self.pattern.is_empty()
            || self.pattern.len() > 32
            || self.pattern.iter().all(Option::is_none)
            || self
                .pattern
                .iter()
                .flatten()
                .any(|s| s.is_empty() || s.len() > 128 || s.chars().any(char::is_whitespace))
        {
            return Err(SemanticError::Invalid("rule pattern"));
        }
        let source = |s: &Source| match s {
            Source::Token(i) if *i as usize >= self.pattern.len() => {
                Err(SemanticError::Invalid("projection bounds"))
            }
            _ => Ok(()),
        };
        let cell = |c: &Cell| {
            source(&c.owner)?;
            source(&c.item)
        };
        match &self.action {
            Action::Relational(r) => r.validate(&self.pattern)?,
            Action::Change(v) if !v.is_empty() && v.len() <= 4 => {
                for u in v {
                    cell(&u.cell)?;
                    source(&u.quantity)?;
                    if u.program.program().arity() != 2 || u.program.context() != CONTEXT {
                        return Err(SemanticError::Invalid("update program"));
                    }
                }
            }
            Action::Question(v, p)
                if !v.is_empty()
                    && v.len() <= 4
                    && p.program().arity() == v.len()
                    && p.context() == CONTEXT =>
            {
                for c in v {
                    cell(c)?;
                }
            }
            Action::Calculate(v, p)
                if !v.is_empty()
                    && v.len() <= 4
                    && p.program().arity() == v.len()
                    && p.context() == CONTEXT =>
            {
                for s in v {
                    source(s)?;
                }
            }
            Action::Ignore => {}
            _ => return Err(SemanticError::Invalid("rule action")),
        }
        Ok(())
    }
    pub fn expression(&self) -> String {
        match &self.action {
            Action::Change(v) => v
                .iter()
                .map(|u| u.program.program().expression())
                .collect::<Vec<_>>()
                .join("; "),
            Action::Question(_, p) | Action::Calculate(_, p) => p.program().expression(),
            Action::Relational(r) => r.program.program().expression(),
            Action::Ignore => "no change".into(),
        }
    }
}
impl SemanticSeed {
    pub fn learning_window(&self) -> Option<u8> {
        self.rules.iter().filter_map(Rule::learning_window).max()
    }
    pub fn bind_calculation(&self, sentence: &str) -> Result<Vec<f64>, SemanticError> {
        let words = words(sentence)?;
        let matches: Vec<_> = self.rules.iter().filter(|r| r.matches(&words)).collect();
        if matches.len() != 1 {
            return Err(if matches.is_empty() {
                SemanticError::Unknown
            } else {
                SemanticError::Ambiguous
            });
        }
        match &matches[0].action {
            Action::Relational(r) => r.bind(&words),
            _ => Err(SemanticError::Invalid("relational calculation required")),
        }
    }
    /// Continue acquisition using inherited bindings, programs and the selected local policy.
    pub fn teach_calculation(
        &self,
        ds: &[Demonstration],
        budget: NumericBudget,
    ) -> Result<Self, SemanticError> {
        let parents: Vec<_> = self.rules.iter().collect();
        let rule =
            Rule::learn_relational(ds, budget, self.learning_window().unwrap_or(1), &parents)?;
        self.merge(&Self::new(vec![rule])?)
    }
    /// Combine acquired structures, never personal episodes or authority.
    /// Conflicting learned meanings for identical anchors are rejected, not overwritten.
    pub fn merge(&self, other: &Self) -> Result<Self, SemanticError> {
        let mut rules = self.rules.clone();
        for incoming in &other.rules {
            if let Some(old) = rules.iter().find(|r| r.pattern == incoming.pattern) {
                let a = Self::new(vec![old.clone()])?.to_bytes()?;
                let b = Self::new(vec![incoming.clone()])?.to_bytes()?;
                if a != b {
                    return Err(SemanticError::Ambiguous);
                }
            } else {
                rules.push(incoming.clone());
            }
        }
        rules.sort_by(|a, b| a.pattern.cmp(&b.pattern));
        Self::new(rules)
    }
    pub fn new(rules: Vec<Rule>) -> Result<Self, SemanticError> {
        let seed = Self { rules };
        seed.to_bytes()?;
        Ok(seed)
    }
    pub fn context(&self) -> &str {
        CONTEXT
    }
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }
    pub fn subset(&self, bits: u64) -> Result<Self, SemanticError> {
        Self::new(
            self.rules
                .iter()
                .enumerate()
                .filter(|(i, _)| bits & (1 << i) != 0)
                .map(|(_, r)| r.clone())
                .collect(),
        )
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.rules.capacity() * size_of::<Rule>()
            + self
                .rules
                .iter()
                .map(|r| {
                    r.pattern.capacity() * size_of::<Option<String>>()
                        + r.pattern
                            .iter()
                            .flatten()
                            .map(String::capacity)
                            .sum::<usize>()
                        + match &r.action {
                            Action::Relational(r) => r.retained_bytes(),
                            Action::Change(v) => {
                                v.capacity() * size_of::<Update>()
                                    + v.iter().map(|u| u.program.retained_bytes()).sum::<usize>()
                            }
                            Action::Question(v, p) => {
                                v.capacity() * size_of::<Cell>() + p.retained_bytes()
                            }
                            Action::Calculate(v, p) => {
                                v.capacity() * size_of::<Source>() + p.retained_bytes()
                            }
                            Action::Ignore => 0,
                        }
                })
                .sum::<usize>()
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.to_bytes().expect("validated seed"))
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, SemanticError> {
        self.encode(false)
    }
    /// Stable decoder identity, excluding examples, ancestry and search timing metadata.
    pub fn structural_bytes(&self) -> Result<Vec<u8>, SemanticError> {
        self.encode(true)
    }
    pub fn mask_for(&self, child: &Self) -> Result<u64, SemanticError> {
        let mut mask = 0;
        for r in &child.rules {
            let key = Self::new(vec![r.clone()])?.structural_bytes()?;
            let matches: Vec<_> = self
                .rules
                .iter()
                .enumerate()
                .filter_map(|(i, source)| {
                    (Self::new(vec![source.clone()])
                        .and_then(|s| s.structural_bytes())
                        .ok()
                        .as_ref()
                        == Some(&key))
                    .then_some(i)
                })
                .collect();
            if matches.len() != 1 || mask & (1u64 << matches[0]) != 0 {
                return Err(SemanticError::Ambiguous);
            }
            mask |= 1 << matches[0];
        }
        Ok(mask)
    }
    fn encode(&self, structural: bool) -> Result<Vec<u8>, SemanticError> {
        if self.rules.len() > 64 {
            return Err(SemanticError::Invalid("rule budget"));
        }
        let relational = self
            .rules
            .iter()
            .any(|r| matches!(r.action, Action::Relational(_)));
        let mut b = if structural {
            if relational {
                b"AXSI0002".to_vec()
            } else {
                b"AXSI0001".to_vec()
            }
        } else {
            if relational {
                b"AXSM0002".to_vec()
            } else {
                MAGIC.to_vec()
            }
        };
        let program = |b: &mut Vec<u8>, p: &NumericSkill| {
            if structural {
                b.push(p.program().arity() as u8);
                text(b, &p.program().expression());
            } else {
                blob(b, &p.to_bytes());
            }
        };
        b.push(self.rules.len() as u8);
        for r in &self.rules {
            r.validate()?;
            b.push(r.pattern.len() as u8);
            for p in &r.pattern {
                text(&mut b, p.as_deref().unwrap_or(""));
            }
            match &r.action {
                Action::Relational(r) => {
                    b.push(4);
                    r.write(&mut b);
                    program(&mut b, &r.program);
                }
                Action::Change(v) => {
                    b.push(0);
                    b.push(v.len() as u8);
                    for u in v {
                        put_cell(&mut b, &u.cell);
                        put_source(&mut b, &u.quantity);
                        b.push(u8::from(u.initialize));
                        program(&mut b, &u.program);
                    }
                }
                Action::Question(v, p) => {
                    b.push(1);
                    b.push(v.len() as u8);
                    for c in v {
                        put_cell(&mut b, c);
                    }
                    program(&mut b, p);
                }
                Action::Calculate(v, p) => {
                    b.push(2);
                    b.push(v.len() as u8);
                    for s in v {
                        put_source(&mut b, s);
                    }
                    program(&mut b, p);
                }
                Action::Ignore => b.push(3),
            }
        }
        b.extend(checksum(&b).to_le_bytes());
        if b.len() > MAX_BYTES {
            return Err(SemanticError::Invalid("seed byte budget"));
        }
        Ok(b)
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self, SemanticError> {
        if b.len() < 17 || b.len() > MAX_BYTES || (&b[..8] != MAGIC && &b[..8] != b"AXSM0002") {
            return Err(SemanticError::Invalid("seed header"));
        }
        let end = b.len() - 8;
        if checksum(&b[..end]) != u64::from_le_bytes(b[end..].try_into().expect("width")) {
            return Err(SemanticError::Invalid("seed checksum"));
        }
        let mut r = Reader(&b[8..end]);
        let count = r.count(64)?;
        let mut rules = Vec::new();
        for _ in 0..count {
            let n = r.count(32)?;
            let mut pattern = Vec::new();
            for _ in 0..n {
                let s = r.text()?;
                pattern.push((!s.is_empty()).then_some(s));
            }
            let action = match r.byte()? {
                0 => {
                    let n = r.count(4)?;
                    let mut v = Vec::new();
                    for _ in 0..n {
                        let cell = r.cell()?;
                        let quantity = r.source()?;
                        let initialize = match r.byte()? {
                            0 => false,
                            1 => true,
                            _ => return Err(SemanticError::Invalid("initialization flag")),
                        };
                        let program = r.program()?;
                        v.push(Update {
                            cell,
                            quantity,
                            initialize,
                            program,
                        });
                    }
                    Action::Change(v)
                }
                1 => {
                    let n = r.count(4)?;
                    let mut v = Vec::new();
                    for _ in 0..n {
                        v.push(r.cell()?);
                    }
                    Action::Question(v, r.program()?)
                }
                2 => {
                    let n = r.count(4)?;
                    let mut v = Vec::new();
                    for _ in 0..n {
                        v.push(r.source()?);
                    }
                    Action::Calculate(v, r.program()?)
                }
                3 => Action::Ignore,
                4 if &b[..8] == b"AXSM0002" => {
                    Action::Relational(Box::new(Relational::read(&mut r)?))
                }
                _ => return Err(SemanticError::Invalid("action tag")),
            };
            rules.push(Rule { pattern, action });
        }
        if !r.0.is_empty() {
            return Err(SemanticError::Invalid("trailing bytes"));
        }
        if &b[..8] == b"AXSM0002"
            && !rules
                .iter()
                .any(|r| matches!(r.action, Action::Relational(_)))
        {
            return Err(SemanticError::Invalid("noncanonical relational version"));
        }
        Self::new(rules)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), SemanticError> {
        publish_new(path, &self.to_bytes()?).map_err(|e| SemanticError::Storage(e.to_string()))
    }
    pub fn open(path: &Path) -> Result<Self, SemanticError> {
        Self::from_bytes(
            &read_bounded(path, MAX_BYTES as u64)
                .map_err(|e| SemanticError::Storage(e.to_string()))?,
        )
    }
}
fn text(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u16).to_le_bytes());
    b.extend(s.as_bytes());
}
fn blob(b: &mut Vec<u8>, v: &[u8]) {
    b.extend((v.len() as u32).to_le_bytes());
    b.extend(v);
}
fn put_source(b: &mut Vec<u8>, s: &Source) {
    match s {
        Source::Token(i) => {
            b.push(0);
            b.push(*i);
        }
        Source::Owner => b.push(1),
        Source::Item => b.push(2),
    }
}
fn put_cell(b: &mut Vec<u8>, c: &Cell) {
    put_source(b, &c.owner);
    put_source(b, &c.item);
}
struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], SemanticError> {
        if n > self.0.len() {
            return Err(SemanticError::Invalid("truncated seed"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn byte(&mut self) -> Result<u8, SemanticError> {
        Ok(self.take(1)?[0])
    }
    fn count(&mut self, max: u8) -> Result<usize, SemanticError> {
        let n = self.byte()?;
        if n > max {
            return Err(SemanticError::Invalid("count budget"));
        }
        Ok(n as usize)
    }
    fn text(&mut self) -> Result<String, SemanticError> {
        let n = u16::from_le_bytes(self.take(2)?.try_into().expect("width")) as usize;
        if n > 128 {
            return Err(SemanticError::Invalid("anchor budget"));
        }
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| SemanticError::Invalid("utf8"))
    }
    fn source(&mut self) -> Result<Source, SemanticError> {
        match self.byte()? {
            0 => Ok(Source::Token(self.byte()?)),
            1 => Ok(Source::Owner),
            2 => Ok(Source::Item),
            _ => Err(SemanticError::Invalid("source tag")),
        }
    }
    fn cell(&mut self) -> Result<Cell, SemanticError> {
        Ok(Cell {
            owner: self.source()?,
            item: self.source()?,
        })
    }
    fn program(&mut self) -> Result<NumericSkill, SemanticError> {
        let n = u32::from_le_bytes(self.take(4)?.try_into().expect("width")) as usize;
        if n > 32768 {
            return Err(SemanticError::Invalid("program bytes"));
        }
        NumericSkill::from_bytes(self.take(n)?).map_err(|e| SemanticError::Storage(e.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evidence {
    pub event: usize,
    pub rule: usize,
    pub owner: String,
    pub item: String,
    pub before: Option<f64>,
    pub after: f64,
    pub expression: String,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub value: f64,
    pub rule: usize,
    pub expression: String,
    pub evidence: Vec<Evidence>,
}
#[derive(Clone, Debug, Default)]
pub struct Session {
    facts: BTreeMap<(String, String), f64>,
    focus: Focus,
    events: usize,
    evidence: Vec<Evidence>,
}
impl Session {
    pub fn fact(&self, owner: &str, item: &str) -> Option<f64> {
        self.facts
            .get(&(owner.to_lowercase(), item.to_lowercase()))
            .copied()
    }
    pub fn evidence(&self) -> &[Evidence] {
        &self.evidence
    }
    pub fn apply(
        &mut self,
        seed: &SemanticSeed,
        sentence: &str,
    ) -> Result<Option<Answer>, SemanticError> {
        if self.events >= 128 {
            return Err(SemanticError::Invalid("session event budget"));
        }
        let w = words(sentence)?;
        let matches: Vec<_> = seed
            .rules
            .iter()
            .enumerate()
            .filter(|(_, r)| r.matches(&w))
            .collect();
        if matches.len() > 1
            && matches
                .iter()
                .any(|(_, r)| matches!(r.action, Action::Relational(_)))
        {
            return Err(SemanticError::Ambiguous);
        }
        // More observed anchors outrank a wildcard interpretation of a learned pronoun.
        let specificity = matches
            .iter()
            .map(|(_, r)| r.pattern.iter().flatten().count())
            .max();
        let matched: Vec<_> = matches
            .into_iter()
            .filter(|(_, r)| Some(r.pattern.iter().flatten().count()) == specificity)
            .collect();
        if matched.is_empty() {
            return Err(SemanticError::Unknown);
        }
        if matched.len() != 1 {
            return Err(SemanticError::Ambiguous);
        }
        let (id, rule) = matched[0];
        let key = |c: &Cell| -> Result<(String, String), SemanticError> {
            Ok((
                value(&c.owner, &w, &self.focus)?,
                value(&c.item, &w, &self.focus)?,
            ))
        };
        let mut changes = Vec::new();
        let result = match &rule.action {
            Action::Relational(r) => Some(Answer {
                value: r
                    .program
                    .predict(CONTEXT, &r.bind(&w)?)
                    .map_err(|_| SemanticError::Domain)?,
                rule: id,
                expression: r.program.program().expression(),
                evidence: self.evidence.clone(),
            }),
            Action::Change(v) => {
                for u in v {
                    let k = key(&u.cell)?;
                    let before = self.facts.get(&k).copied();
                    if u.initialize && before.is_some() {
                        return Err(SemanticError::Ambiguous);
                    }
                    let old = if u.initialize {
                        0.0
                    } else {
                        before.ok_or(SemanticError::Missing)?
                    };
                    let q = number(&value(&u.quantity, &w, &self.focus)?)?;
                    let after = u
                        .program
                        .predict(CONTEXT, &[old, q])
                        .map_err(|_| SemanticError::Domain)?;
                    if after.abs() > 1e12 {
                        return Err(SemanticError::Domain);
                    }
                    if changes.iter().any(|(prior, _)| prior == &k) {
                        return Err(SemanticError::Ambiguous);
                    }
                    changes.push((
                        k,
                        Evidence {
                            event: self.events,
                            rule: id,
                            owner: String::new(),
                            item: String::new(),
                            before,
                            after,
                            expression: u.program.program().expression(),
                        },
                    ));
                }
                if self.facts.len()
                    + changes
                        .iter()
                        .filter(|(k, _)| !self.facts.contains_key(k))
                        .count()
                    > 16
                {
                    return Err(SemanticError::Invalid("fact budget"));
                }
                None
            }
            Action::Question(v, p) => {
                let input: Vec<_> = v
                    .iter()
                    .map(|c| {
                        self.facts
                            .get(&key(c)?)
                            .copied()
                            .ok_or(SemanticError::Missing)
                    })
                    .collect::<Result<_, _>>()?;
                let n = p
                    .predict(CONTEXT, &input)
                    .map_err(|_| SemanticError::Domain)?;
                Some(Answer {
                    value: n,
                    rule: id,
                    expression: p.program().expression(),
                    evidence: self.evidence.clone(),
                })
            }
            Action::Calculate(v, p) => {
                let input: Vec<_> = v
                    .iter()
                    .map(|s| number(&value(s, &w, &self.focus)?))
                    .collect::<Result<_, _>>()?;
                let n = p
                    .predict(CONTEXT, &input)
                    .map_err(|_| SemanticError::Domain)?;
                Some(Answer {
                    value: n,
                    rule: id,
                    expression: p.program().expression(),
                    evidence: self.evidence.clone(),
                })
            }
            Action::Ignore => None,
        };
        if !changes.is_empty() {
            let mut focus = Focus::default();
            for (k, mut e) in changes {
                if !focus.owners.contains(&k.0) {
                    focus.owners.push(k.0.clone());
                }
                if !focus.items.contains(&k.1) {
                    focus.items.push(k.1.clone());
                }
                e.owner = k.0.clone();
                e.item = k.1.clone();
                self.facts.insert(k, e.after);
                self.evidence.push(e);
            }
            self.focus = focus;
        }
        self.events += 1;
        Ok(result)
    }
    /// Atomic narrative ingestion; an unrecognized clause cannot silently corrupt ongoing state.
    pub fn narrative(&mut self, seed: &SemanticSeed, text: &str) -> Result<Answer, SemanticError> {
        if text.len() > 8192 {
            return Err(SemanticError::Invalid("narrative byte budget"));
        }
        let mut copy = self.clone();
        let mut answer = None;
        let mut start = 0;
        for (i, c) in text.char_indices() {
            let decimal = c == '.'
                && i > 0
                && text.as_bytes()[i - 1].is_ascii_digit()
                && text.as_bytes().get(i + 1).is_some_and(u8::is_ascii_digit);
            if ['.', '?', '!'].contains(&c) && !decimal {
                let s = text[start..i].trim();
                if !s.is_empty() {
                    answer = copy.apply(seed, s)?;
                }
                start = i + c.len_utf8();
            }
        }
        if !text[start..].trim().is_empty() {
            answer = copy.apply(seed, text[start..].trim())?;
        }
        let answer = answer.ok_or(SemanticError::Missing)?;
        *self = copy;
        Ok(answer)
    }
}
