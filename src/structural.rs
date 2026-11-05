//! Example-induced tree transductions. Symbols have no built-in linguistic or calculus meaning.
use crate::text_skill::{publish_new, read_bounded};
use serde::{Deserialize, Serialize};
use std::{fmt, path::Path};
#[path = "structural_agent.rs"]
pub mod agent;
#[path = "structural_algorithm.rs"]
pub mod algorithm;
#[path = "structural_alignment.rs"]
pub mod alignment;
#[path = "structural_dependencies.rs"]
mod dependencies;
#[path = "structural_dialogue.rs"]
pub mod dialogue;
#[path = "structural_induction.rs"]
pub mod induction;
#[path = "structural_language.rs"]
pub mod language;
#[path = "structural_lexical.rs"]
mod lexical;
#[path = "structural_numeric.rs"]
mod numeric;
#[path = "structural_phrase.rs"]
mod phrase;
#[path = "structural_sequence.rs"]
pub mod sequence;
#[path = "structural_session.rs"]
pub mod session;

pub const MAX_BYTES: usize = 262_144;
pub const CONTEXT: &str = "axon/structural-v1";
const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 48;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Term {
    Integer(i64),
    Symbol(String),
    Tree(Vec<Term>),
}
impl Term {
    pub fn validate(&self) -> Result<(), Error> {
        fn walk(t: &Term, depth: usize, count: &mut usize) -> Result<(), Error> {
            *count += 1;
            if depth > MAX_DEPTH || *count > MAX_NODES {
                return Err(Error::Budget);
            }
            match t {
                Term::Symbol(s)
                    if s.is_empty() || s.len() > 256 || s.chars().any(char::is_control) =>
                {
                    Err(Error::Invalid("symbol bounds"))
                }
                Term::Tree(v) => {
                    if v.len() < 2 || v.len() > 32 || !matches!(&v[0], Term::Symbol(_)) {
                        return Err(Error::Invalid(
                            "tree requires a symbol head and 1..31 children",
                        ));
                    }
                    for child in v {
                        walk(child, depth + 1, count)?;
                    }
                    Ok(())
                }
                _ => Ok(()),
            }
        }
        walk(self, 0, &mut 0)
    }
    pub fn at(&self, path: &[usize]) -> Option<&Self> {
        let mut t = self;
        for i in path {
            let Self::Tree(v) = t else { return None };
            t = v.get(*i)?;
        }
        Some(t)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Invalid(&'static str),
    Unsupported,
    Ambiguous,
    Budget,
    Storage(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => f.write_str(s),
            Self::Unsupported => f.write_str("unlearned structural transformation"),
            Self::Ambiguous => f.write_str("conflicting structural interpretations"),
            Self::Budget => f.write_str("structural work or size quota"),
            Self::Storage(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub input: Term,
    pub output: Term,
}

/// Explicit domain restrictions, not inferred mathematical facts.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Guard {
    Vocabulary {
        path: Vec<usize>,
        values: Vec<String>,
    },
    Symbols {
        path: Vec<usize>,
        values: Vec<String>,
    },
    IntegerRange {
        path: Vec<usize>,
        min: i64,
        max: i64,
    },
    Different {
        left: Vec<usize>,
        right: Vec<usize>,
    },
}
impl Guard {
    fn accepts(&self, t: &Term) -> bool {
        match self {
            Self::Vocabulary { path, values } => {
                matches!(t.at(path), Some(Term::Tree(words)) if words.iter().all(|word| matches!(word,Term::Symbol(s) if values.binary_search(s).is_ok())))
            }
            Self::Symbols { path, values } => {
                matches!(t.at(path), Some(Term::Symbol(s)) if values.contains(s))
            }
            Self::IntegerRange { path, min, max } => {
                matches!(t.at(path), Some(Term::Integer(n)) if n >= min && n <= max)
            }
            Self::Different { left, right } => {
                matches!((t.at(left),t.at(right)),(Some(a),Some(b)) if a != b)
            }
        }
    }
    fn validate(&self) -> Result<(), Error> {
        let valid = |p: &[usize]| p.len() <= MAX_DEPTH && p.iter().all(|i| *i < 32);
        let ok = match self {
            Self::Vocabulary { path, values } => {
                valid(path)
                    && (2..=512).contains(&values.len())
                    && values.windows(2).all(|p| p[0] < p[1])
                    && values.iter().all(|s| {
                        !s.chars().any(char::is_whitespace)
                            && Term::Symbol(s.clone()).validate().is_ok()
                    })
            }
            Self::Symbols { path, values } => {
                valid(path)
                    && (2..=32).contains(&values.len())
                    && values.windows(2).all(|p| p[0] < p[1])
                    && values
                        .iter()
                        .all(|s| Term::Symbol(s.clone()).validate().is_ok())
            }
            Self::IntegerRange { path, min, max } => valid(path) && min <= max,
            Self::Different { left, right } => valid(left) && valid(right) && left != right,
        };
        if ok {
            Ok(())
        } else {
            Err(Error::Invalid("invalid structural guard"))
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lesson {
    pub id: String,
    pub examples: Vec<Example>,
    #[serde(default)]
    pub guards: Vec<Guard>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum Pattern {
    Literal(Term),
    Tree(Vec<Pattern>),
    Choice(Vec<Pattern>),
    Hole(usize, u8),
    Calculate(numeric::Calculation),
    Phrase(Vec<Pattern>),
    Span(usize),
    Algorithm(algorithm::Binding),
    Lookup {
        slot: usize,
        entries: Vec<(String, Term)>,
    },
}
fn kind(t: &Term) -> u8 {
    match t {
        Term::Integer(_) => 0,
        Term::Symbol(_) => 1,
        Term::Tree(_) => 2,
    }
}
fn infer_input(rows: &[Term], holes: &mut Vec<Vec<Term>>) -> Result<Pattern, Error> {
    if rows.iter().all(|t| *t == rows[0]) {
        return Ok(Pattern::Literal(rows[0].clone()));
    }
    if let Term::Tree(first) = &rows[0]
        && rows
            .iter()
            .all(|t| matches!(t,Term::Tree(v) if v[0]==first[0]))
        && rows
            .iter()
            .any(|t| matches!(t,Term::Tree(v) if v.len()!=first.len()))
        && rows
            .iter()
            .all(|t| matches!(t,Term::Tree(v) if v.iter().all(|c| matches!(c, Term::Symbol(_)))))
    {
        return phrase::learn(rows, holes);
    }
    if let Term::Tree(first) = &rows[0]
        && rows
            .iter()
            .all(|t| matches!(t,Term::Tree(v) if v.len()==first.len() && v[0]==first[0]))
    {
        let mut children = Vec::new();
        for i in 0..first.len() {
            let column = rows
                .iter()
                .map(|t| {
                    let Term::Tree(v) = t else { unreachable!() };
                    v[i].clone()
                })
                .collect::<Vec<_>>();
            children.push(infer_input(&column, holes)?);
        }
        return Ok(Pattern::Tree(children));
    }
    let tag = if rows.iter().any(|t| kind(t) != kind(&rows[0])) {
        3
    } else {
        kind(&rows[0])
    };
    if let Some(i) = holes.iter().position(|h| h == rows) {
        return Ok(Pattern::Hole(i, tag));
    }
    if holes.len() >= 64 {
        return Err(Error::Budget);
    }
    let i = holes.len();
    holes.push(rows.to_vec());
    Ok(Pattern::Hole(i, tag))
}
fn infer_output(rows: &[Term], holes: &[Vec<Term>], work: &mut u64) -> Result<Pattern, Error> {
    if let Some(i) = holes.iter().position(|h| h == rows) {
        return Ok(Pattern::Hole(
            i,
            if rows.iter().any(|t| kind(t) != kind(&rows[0])) {
                3
            } else {
                kind(&rows[0])
            },
        ));
    }
    if rows.iter().all(|t| *t == rows[0]) {
        return Ok(Pattern::Literal(rows[0].clone()));
    }
    if let Term::Tree(first) = &rows[0]
        && rows
            .iter()
            .all(|t| matches!(t,Term::Tree(v) if v.len()==first.len() && v[0]==first[0]))
    {
        return Ok(Pattern::Tree(
            (0..first.len())
                .map(|i| {
                    let column = rows
                        .iter()
                        .map(|t| {
                            let Term::Tree(v) = t else { unreachable!() };
                            v[i].clone()
                        })
                        .collect::<Vec<_>>();
                    infer_output(&column, holes, work)
                })
                .collect::<Result<_, _>>()?,
        ));
    }
    if let Ok(calculation) = numeric::Calculation::learn(rows, holes, work) {
        return Ok(Pattern::Calculate(calculation));
    }
    lexical::infer(rows, holes).ok_or(Error::Unsupported)
}
impl Pattern {
    fn bind(&self, t: &Term, values: &mut Vec<Option<Term>>) -> Result<bool, Error> {
        Ok(match self {
            Self::Choice(branches) => {
                let mut agreed = None;
                for branch in branches {
                    let mut trial = values.clone();
                    if branch.bind(t, &mut trial)? {
                        if agreed.as_ref().is_some_and(|old| old != &trial) {
                            return Err(Error::Ambiguous);
                        }
                        agreed = Some(trial);
                    }
                }
                if let Some(bound) = agreed {
                    *values = bound;
                    true
                } else {
                    false
                }
            }
            Self::Phrase(pieces) => return phrase::bind(pieces, t, values),
            Self::Calculate(_) | Self::Span(_) | Self::Algorithm(_) | Self::Lookup { .. } => false,
            Self::Literal(v) => v == t,
            Self::Tree(p) => match t {
                Term::Tree(v) if p.len() == v.len() => {
                    let mut yes = true;
                    for (p, t) in p.iter().zip(v) {
                        if !p.bind(t, values)? {
                            yes = false;
                            break;
                        }
                    }
                    yes
                }
                _ => false,
            },
            Self::Hole(i, tag) => {
                if *tag != 3 && *tag != kind(t) {
                    return Ok(false);
                }
                if let Some(old) = &values[*i] {
                    old == t
                } else {
                    values[*i] = Some(t.clone());
                    true
                }
            }
        })
    }
    fn instantiate(&self, values: &[Option<Term>], count: &mut usize) -> Result<Term, Error> {
        *count += 1;
        if *count > MAX_NODES {
            return Err(Error::Budget);
        }
        match self {
            Self::Lookup { slot, entries } => {
                let Some(Some(Term::Symbol(key))) = values.get(*slot) else {
                    return Err(Error::Unsupported);
                };
                entries
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, value)| value.clone())
                    .ok_or(Error::Unsupported)
            }
            Self::Choice(_) | Self::Phrase(_) | Self::Span(_) => {
                Err(Error::Invalid("phrase cannot construct output"))
            }
            Self::Calculate(c) => c.evaluate(values),
            Self::Algorithm(p) => {
                let t = p.evaluate(values)?;
                *count += term_nodes(&t) - 1;
                if *count > MAX_NODES {
                    return Err(Error::Budget);
                }
                Ok(t)
            }
            Self::Literal(t) => {
                let n = term_nodes(t);
                *count += n - 1;
                if *count > MAX_NODES {
                    return Err(Error::Budget);
                }
                Ok(t.clone())
            }
            Self::Tree(v) => Ok(Term::Tree(
                v.iter()
                    .map(|p| p.instantiate(values, count))
                    .collect::<Result<_, _>>()?,
            )),
            Self::Hole(i, _) => {
                let t = values[*i].as_ref().expect("input bound all output holes");
                *count += term_nodes(t) - 1;
                if *count > MAX_NODES {
                    return Err(Error::Budget);
                }
                Ok(t.clone())
            }
        }
    }
    fn specificity(&self) -> usize {
        match self {
            Self::Literal(Term::Tree(v)) => 1 + v.len(),
            Self::Literal(_) => 1,
            Self::Tree(v) | Self::Phrase(v) => v.iter().map(Self::specificity).sum(),
            Self::Choice(v) => v.first().map_or(0, Self::specificity),
            Self::Hole(_, _)
            | Self::Calculate(_)
            | Self::Span(_)
            | Self::Algorithm(_)
            | Self::Lookup { .. } => 0,
        }
    }
}
fn term_nodes(t: &Term) -> usize {
    match t {
        Term::Tree(v) => 1 + v.iter().map(term_nodes).sum::<usize>(),
        _ => 1,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RuleWire")]
pub struct Rule {
    id: String,
    guards: Vec<Guard>,
    teaching_checksum: u64,
    input: Pattern,
    output: Pattern,
    holes: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    dependencies: Vec<dependencies::Dependency>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleWire {
    id: String,
    guards: Vec<Guard>,
    teaching_checksum: u64,
    input: Pattern,
    output: Pattern,
    holes: usize,
    #[serde(default)]
    dependencies: Vec<dependencies::Dependency>,
}
impl TryFrom<RuleWire> for Rule {
    type Error = Error;
    fn try_from(w: RuleWire) -> Result<Self, Error> {
        let r = Self {
            id: w.id,
            guards: w.guards,
            teaching_checksum: w.teaching_checksum,
            input: w.input,
            output: w.output,
            holes: w.holes,
            dependencies: w.dependencies,
        };
        r.validate()?;
        Ok(r)
    }
}
impl Rule {
    pub fn learn(lesson: Lesson) -> Result<Self, Error> {
        Self::learn_with(lesson, false)
    }
    fn learn_with(lesson: Lesson, spans: bool) -> Result<Self, Error> {
        if lesson.id.is_empty()
            || lesson.id.len() > 64
            || !lesson
                .id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
            || !(3..=32).contains(&lesson.examples.len())
            || lesson.guards.len() > 8
        {
            return Err(Error::Invalid("lesson bounds"));
        }
        for g in &lesson.guards {
            g.validate()?;
        }
        for e in &lesson.examples {
            e.input.validate()?;
            e.output.validate()?;
            if !matches!(e.input, Term::Tree(_))
                || !lesson.guards.iter().all(|g| g.accepts(&e.input))
            {
                return Err(Error::Invalid("example violates signature or guard"));
            }
        }
        let xs = lesson
            .examples
            .iter()
            .map(|e| e.input.clone())
            .collect::<Vec<_>>();
        let ys = lesson
            .examples
            .iter()
            .map(|e| e.output.clone())
            .collect::<Vec<_>>();
        let mut holes = Vec::new();
        let input = if spans {
            phrase::learn_supervised(&xs, &ys, &mut holes)?
        } else {
            infer_input(&xs, &mut holes)?
        };
        if matches!(input, Pattern::Hole(_, _)) || holes.is_empty() {
            return Err(Error::Invalid(
                "need varying examples and a shared structure",
            ));
        }
        let output = infer_output(&ys, &holes, &mut 2_000_000)?;
        let rule = Self {
            id: lesson.id.clone(),
            guards: lesson.guards.clone(),
            teaching_checksum: crate::live_store::checksum(
                &serde_json::to_vec(&lesson).map_err(|e| Error::Storage(e.to_string()))?,
            ),
            input,
            output,
            holes: holes.len(),
            dependencies: vec![],
        };
        if !lesson
            .examples
            .iter()
            .all(|e| rule.apply(&e.input).as_ref() == Some(&e.output))
        {
            return Err(Error::Invalid("training replay failed"));
        }
        Ok(rule)
    }
    pub fn apply(&self, t: &Term) -> Option<Term> {
        self.checked_apply(t).ok().flatten()
    }
    fn checked_apply(&self, t: &Term) -> Result<Option<Term>, Error> {
        if !self.guards.iter().all(|g| g.accepts(t)) {
            return Ok(None);
        }
        let mut values = vec![None; self.holes];
        if self.input.bind(t, &mut values)? {
            match self.output.instantiate(&values, &mut 0) {
                Ok(t) => Ok(Some(t)),
                Err(Error::Unsupported) => Ok(None),
                Err(e) => Err(e),
            }
        } else {
            Ok(None)
        }
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    fn validate(&self) -> Result<(), Error> {
        if self.id.is_empty()
            || self.id.len() > 64
            || !self
                .id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
            || !(1..=64).contains(&self.holes)
            || self.guards.len() > 8
            || !matches!(
                self.input,
                Pattern::Tree(_)
                    | Pattern::Phrase(_)
                    | Pattern::Choice(_)
                    | Pattern::Literal(Term::Tree(_))
            )
        {
            return Err(Error::Invalid("compiled rule bounds"));
        }
        for g in &self.guards {
            g.validate()?;
        }
        fn walk(
            p: &Pattern,
            depth: usize,
            count: &mut usize,
            slots: &mut [Option<u8>],
            input: bool,
        ) -> Result<(), Error> {
            *count += 1;
            if depth > MAX_DEPTH || *count > MAX_NODES {
                return Err(Error::Budget);
            }
            match p {
                Pattern::Choice(v) => {
                    if !input || depth != 0 || !(2..=8).contains(&v.len()) {
                        return Err(Error::Invalid("input choice bounds"));
                    }
                    let mut agreed = None;
                    let score = v[0].specificity();
                    for (i, branch) in v.iter().enumerate() {
                        if !matches!(
                            branch,
                            Pattern::Tree(_) | Pattern::Phrase(_) | Pattern::Literal(Term::Tree(_))
                        ) || branch.specificity() != score
                            || v[..i]
                                .iter()
                                .any(|old| !lexical::disjoint_alternatives(old, branch))
                        {
                            return Err(Error::Invalid("input choice signature"));
                        }
                        let mut local = slots.to_vec();
                        walk(branch, depth + 1, count, &mut local, true)?;
                        if local.iter().any(Option::is_none)
                            || agreed.as_ref().is_some_and(|old| old != &local)
                        {
                            return Err(Error::Invalid("input choice bindings"));
                        }
                        agreed = Some(local);
                    }
                    slots.copy_from_slice(&agreed.unwrap());
                    Ok(())
                }
                Pattern::Phrase(v) => {
                    if !input
                        || v.len() < 2
                        || v.len() > 32
                        || !matches!(v.first(), Some(Pattern::Literal(Term::Symbol(_))))
                        || v.windows(2)
                            .any(|pair| matches!(pair, [Pattern::Span(_), Pattern::Span(_)]))
                        || v.iter().any(|p| {
                            !matches!(p, Pattern::Literal(Term::Symbol(_)) | Pattern::Span(_))
                        })
                    {
                        return Err(Error::Invalid("phrase signature"));
                    }
                    for p in v {
                        walk(p, depth + 1, count, slots, input)?;
                    }
                    Ok(())
                }
                Pattern::Span(i) => {
                    if !input {
                        return Err(Error::Invalid("output span"));
                    }
                    let slot = slots.get_mut(*i).ok_or(Error::Invalid("span slot"))?;
                    if slot.is_some_and(|tag| tag != 1) {
                        return Err(Error::Invalid("span type"));
                    }
                    *slot = Some(1);
                    Ok(())
                }
                Pattern::Calculate(c) => {
                    if input || c.slots.iter().any(|i| slots.get(*i) != Some(&Some(0))) {
                        return Err(Error::Invalid("calculation requires bound integer slots"));
                    }
                    Ok(())
                }
                Pattern::Lookup { slot, entries } => {
                    if input
                        || slots.get(*slot) != Some(&Some(1))
                        || !(2..=32).contains(&entries.len())
                        || !entries.windows(2).all(|p| p[0].0 < p[1].0)
                        || entries.iter().any(|(key, value)| {
                            Term::Symbol(key.clone()).validate().is_err()
                                || !matches!(value, Term::Symbol(_) | Term::Integer(_))
                                || value.validate().is_err()
                        })
                    {
                        return Err(Error::Invalid("lexical lookup domain"));
                    }
                    Ok(())
                }
                Pattern::Algorithm(p) => {
                    if input || !slots.get(p.slot).is_some_and(Option::is_some) {
                        return Err(Error::Invalid("algorithm requires a bound value slot"));
                    }
                    Ok(())
                }
                Pattern::Literal(t) => t.validate(),
                Pattern::Hole(i, tag) => {
                    let Some(slot) = slots.get_mut(*i) else {
                        return Err(Error::Invalid("hole index"));
                    };
                    if *tag > 3 || slot.is_some_and(|old| old != *tag) || !input && slot.is_none() {
                        return Err(Error::Invalid("hole type or unbound output"));
                    }
                    *slot = Some(*tag);
                    Ok(())
                }
                Pattern::Tree(v) => {
                    if v.len() < 2
                        || v.len() > 32
                        || !matches!(v.first(), Some(Pattern::Literal(Term::Symbol(_))))
                    {
                        return Err(Error::Invalid("compiled tree signature"));
                    }
                    for child in v {
                        if matches!(child, Pattern::Span(_)) {
                            return Err(Error::Invalid("span requires phrase parent"));
                        }
                        walk(child, depth + 1, count, slots, input)?;
                    }
                    Ok(())
                }
            }
        }
        let mut slots = vec![None; self.holes];
        walk(&self.input, 0, &mut 0, &mut slots, true)?;
        if slots.iter().any(Option::is_none) {
            return Err(Error::Invalid("unbound input hole"));
        }
        walk(&self.output, 0, &mut 0, &mut slots, false)
    }
    pub fn derived_programs(&self) -> Vec<String> {
        fn walk(p: &Pattern, out: &mut Vec<String>) {
            match p {
                Pattern::Calculate(c) => out.push(c.expression()),
                Pattern::Tree(v) => {
                    for p in v {
                        walk(p, out);
                    }
                }
                _ => (),
            }
        }
        let mut out = Vec::new();
        walk(&self.output, &mut out);
        out
    }
    pub fn algorithm_program(&self) -> Option<&algorithm::Program> {
        match &self.output {
            Pattern::Algorithm(p) => Some(&p.program),
            _ => None,
        }
    }
    fn requires_v7(&self) -> bool {
        fn has(p: &Pattern) -> bool {
            match p {
                Pattern::Lookup { .. } => true,
                Pattern::Tree(v) | Pattern::Phrase(v) => v.iter().any(has),
                _ => false,
            }
        }
        self.guards
            .iter()
            .any(|g| matches!(g, Guard::Symbols { .. }))
            || has(&self.output)
    }
    fn requires_v8(&self) -> bool {
        self.guards
            .iter()
            .any(|g| matches!(g, Guard::Vocabulary { .. }))
    }
    fn requires_v9(&self) -> bool {
        matches!(self.input, Pattern::Choice(_))
    }
    fn specificity(&self) -> usize {
        self.input.specificity()
            + self
                .guards
                .iter()
                .filter(|g| matches!(g, Guard::Symbols { .. }))
                .count()
    }
    pub(super) fn algorithm_programs(&self) -> Vec<&algorithm::Program> {
        fn walk<'a>(p: &'a Pattern, out: &mut Vec<&'a algorithm::Program>) {
            match p {
                Pattern::Algorithm(binding) => out.push(&binding.program),
                Pattern::Tree(v) | Pattern::Phrase(v) => {
                    for p in v {
                        walk(p, out);
                    }
                }
                _ => (),
            }
        }
        let mut out = Vec::new();
        walk(&self.output, &mut out);
        out
    }
    fn requires_v4(&self) -> bool {
        fn has(p: &Pattern) -> bool {
            match p {
                Pattern::Algorithm(_) | Pattern::Hole(_, 3) => true,
                Pattern::Tree(v) | Pattern::Phrase(v) => v.iter().any(has),
                _ => false,
            }
        }
        has(&self.input) || has(&self.output)
    }
    fn requires_v6(&self) -> bool {
        fn tag(p: &Pattern, slot: usize) -> Option<u8> {
            match p {
                Pattern::Hole(i, kind) if *i == slot => Some(*kind),
                Pattern::Span(i) if *i == slot => Some(1),
                Pattern::Tree(v) | Pattern::Phrase(v) => v.iter().find_map(|p| tag(p, slot)),
                _ => None,
            }
        }
        fn walk(p: &Pattern, input: &Pattern) -> bool {
            match p {
                Pattern::Algorithm(binding) => tag(input, binding.slot) != Some(2),
                Pattern::Tree(v) | Pattern::Phrase(v) => v.iter().any(|p| walk(p, input)),
                _ => false,
            }
        }
        walk(&self.output, &self.input)
    }
    fn extended(&self) -> bool {
        fn phrase(p: &Pattern) -> bool {
            match p {
                Pattern::Phrase(_) => true,
                Pattern::Tree(v) => v.iter().any(phrase),
                _ => false,
            }
        }
        phrase(&self.input) || !self.derived_programs().is_empty() || self.requires_v4()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Encoded {
    version: u32,
    rules: Vec<Rule>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    development: Option<algorithm::Policy>,
}
#[derive(Clone, Debug, Default)]
pub struct StructuralSeed {
    rules: Vec<Rule>,
    development: Option<algorithm::Policy>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prediction {
    pub output: Term,
    pub rules: Vec<String>,
    pub visits: usize,
}
impl StructuralSeed {
    pub fn algorithm_policy(&self) -> algorithm::Policy {
        self.development.unwrap_or_default()
    }
    pub fn set_algorithm_policy(&mut self, policy: algorithm::Policy) -> Result<(), Error> {
        policy.validate()?;
        self.development = Some(policy);
        Ok(())
    }
    pub fn context(&self) -> &str {
        CONTEXT
    }
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }
    pub fn interpretations(&self, t: &Term) -> Result<Vec<Term>, Error> {
        t.validate()?;
        let mut best = Vec::new();
        let mut specificity = 0;
        for r in &self.rules {
            if let Some(out) = r.checked_apply(t)? {
                let score = r.specificity();
                if best.is_empty() || score > specificity {
                    best.clear();
                    specificity = score;
                }
                if score == specificity && !best.contains(&out) {
                    best.push(out);
                }
            }
        }
        Ok(best)
    }
    pub fn teach(&mut self, lesson: Lesson, admission: &[Example]) -> Result<(), Error> {
        self.teach_with(lesson, admission, false)
    }
    fn teach_with(
        &mut self,
        lesson: Lesson,
        admission: &[Example],
        spans: bool,
    ) -> Result<(), Error> {
        if self.rules.len() >= 128
            || self.rules.iter().any(|r| r.id() == lesson.id)
            || !(1..=32).contains(&admission.len())
        {
            return Err(Error::Invalid("admission or rule count"));
        }
        let training = lesson
            .examples
            .iter()
            .map(|e| e.input.clone())
            .collect::<Vec<_>>();
        let r = Rule::learn_with(lesson, spans)?;
        for e in admission {
            e.input.validate()?;
            e.output.validate()?;
            if training.contains(&e.input) {
                return Err(Error::Invalid("admission overlaps training"));
            }
            if r.apply(&e.input).as_ref() != Some(&e.output) {
                return Err(Error::Invalid("independent admission refuted rule"));
            }
        }
        self.rules.push(r);
        if self.to_bytes().is_err() {
            self.rules.pop();
            return Err(Error::Budget);
        }
        Ok(())
    }
    pub fn transform(&self, t: &Term, max_steps: usize) -> Result<Prediction, Error> {
        t.validate()?;
        if max_steps == 0 || max_steps > 4096 {
            return Err(Error::Invalid("rewrite quota"));
        }
        let mut prediction = Prediction {
            output: t.clone(),
            rules: Vec::new(),
            visits: 0,
        };
        fn walk(
            seed: &StructuralSeed,
            t: &Term,
            depth: usize,
            limit: usize,
            p: &mut Prediction,
        ) -> Result<Term, Error> {
            p.visits += 1;
            if p.visits > 131072 || depth > MAX_DEPTH {
                return Err(Error::Budget);
            }
            let mut best = None;
            let mut specificity = 0;
            let mut conflict = false;
            for r in &seed.rules {
                if let Some(out) = r.checked_apply(t)? {
                    let score = r.specificity();
                    if best.is_none() || score > specificity {
                        best = Some((out, r.id().to_owned()));
                        specificity = score;
                        conflict = false;
                    } else if score == specificity
                        && best.as_ref().is_some_and(|(old, _)| *old != out)
                    {
                        conflict = true;
                    }
                }
            }
            if conflict {
                return Err(Error::Ambiguous);
            }
            if let Some((out, id)) = best {
                if p.rules.len() >= limit || out == *t {
                    return Err(Error::Budget);
                }
                out.validate()?;
                p.rules.push(id);
                return walk(seed, &out, depth + 1, limit, p);
            }
            if let Term::Tree(v) = t {
                let mut out = vec![v[0].clone()];
                for child in &v[1..] {
                    out.push(walk(seed, child, depth + 1, limit, p)?);
                }
                let out = Term::Tree(out);
                out.validate()?;
                if out != *t {
                    return walk(seed, &out, depth + 1, limit, p);
                }
                Ok(out)
            } else {
                Ok(t.clone())
            }
        }
        prediction.output = walk(self, t, 0, max_steps, &mut prediction)?;
        Ok(prediction)
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let tracked = self.rules.iter().any(|r| !r.dependencies.is_empty());
        let values = self.rules.iter().any(Rule::requires_v6);
        self.validate_dependencies(tracked || values)?;
        let b = serde_json::to_vec(&Encoded {
            version: if self.rules.iter().any(Rule::requires_v9) {
                9
            } else if self.rules.iter().any(Rule::requires_v8) {
                8
            } else if self.rules.iter().any(Rule::requires_v7) {
                7
            } else if values {
                6
            } else if tracked {
                5
            } else if self.development.is_some() || self.rules.iter().any(Rule::requires_v4) {
                4
            } else if self.rules.iter().any(Rule::extended) {
                3
            } else {
                2
            },
            rules: self.rules.clone(),
            development: self.development,
        })
        .map_err(|e| Error::Storage(e.to_string()))?;
        if b.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(b)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        let encoded: Encoded =
            serde_json::from_slice(bytes).map_err(|e| Error::Storage(e.to_string()))?;
        if ![2, 3, 4, 5, 6, 7, 8, 9].contains(&encoded.version) || encoded.rules.len() > 128 {
            return Err(Error::Invalid("structural seed version or count"));
        }
        let mut rules = Vec::<Rule>::new();
        if let Some(p) = encoded.development {
            p.validate()?;
        }
        if encoded.version < 4
            && (encoded.development.is_some() || encoded.rules.iter().any(Rule::requires_v4))
        {
            return Err(Error::Invalid("algorithm seed requires version 4"));
        }
        if encoded.version < 5 && encoded.rules.iter().any(|r| !r.dependencies.is_empty()) {
            return Err(Error::Invalid("dependency witnesses require version 5"));
        }
        if encoded.version < 6 && encoded.rules.iter().any(Rule::requires_v6) {
            return Err(Error::Invalid(
                "scalar algorithm bindings require version 6",
            ));
        }
        if encoded.version < 7 && encoded.rules.iter().any(Rule::requires_v7) {
            return Err(Error::Invalid("lexical tables require version 7"));
        }
        if encoded.version < 8 && encoded.rules.iter().any(Rule::requires_v8) {
            return Err(Error::Invalid("vocabulary guards require version 8"));
        }
        if encoded.version < 9 && encoded.rules.iter().any(Rule::requires_v9) {
            return Err(Error::Invalid("input choices require version 9"));
        }
        for r in encoded.rules {
            if encoded.version == 2 && r.extended() {
                return Err(Error::Invalid("v2 seed cannot contain calculations"));
            }
            if rules.iter().any(|old| old.id() == r.id()) {
                return Err(Error::Invalid("duplicate structural lesson"));
            }
            r.validate()?;
            rules.push(r);
        }
        let seed = Self {
            rules,
            development: encoded.development,
        };
        seed.validate_dependencies(encoded.version >= 5)?;
        Ok(seed)
    }
    pub fn retained_bytes(&self) -> usize {
        use std::mem::size_of;
        fn term(t: &Term) -> usize {
            match t {
                Term::Symbol(s) => s.len(),
                Term::Tree(v) => v.len() * size_of::<Term>() + v.iter().map(term).sum::<usize>(),
                _ => 0,
            }
        }
        fn pattern(p: &Pattern) -> usize {
            match p {
                Pattern::Calculate(c) => c.retained_bytes(),
                Pattern::Algorithm(p) => p.program.retained_bytes(),
                Pattern::Literal(t) => term(t),
                Pattern::Lookup { entries, .. } => {
                    entries.len() * size_of::<(String, Term)>()
                        + entries
                            .iter()
                            .map(|(key, value)| key.len() + term(value))
                            .sum::<usize>()
                }
                Pattern::Tree(v) | Pattern::Phrase(v) | Pattern::Choice(v) => {
                    v.len() * size_of::<Pattern>() + v.iter().map(pattern).sum::<usize>()
                }
                _ => 0,
            }
        }
        size_of::<Self>()
            + self.rules.len() * size_of::<Rule>()
            + self
                .rules
                .iter()
                .map(|r| {
                    r.id.len()
                        + r.dependencies
                            .iter()
                            .map(|d| d.rule.len() + d.expected.retained_bytes())
                            .sum::<usize>()
                        + r.guards.len() * size_of::<Guard>()
                        + r.guards
                            .iter()
                            .map(|g| match g {
                                Guard::Symbols { path, values }
                                | Guard::Vocabulary { path, values } => {
                                    path.len() * size_of::<usize>()
                                        + values.len() * size_of::<String>()
                                        + values.iter().map(String::len).sum::<usize>()
                                }
                                Guard::IntegerRange { path, .. } => path.len() * size_of::<usize>(),
                                Guard::Different { left, right } => {
                                    (left.len() + right.len()) * size_of::<usize>()
                                }
                            })
                            .sum::<usize>()
                        + pattern(&r.input)
                        + pattern(&r.output)
                })
                .sum::<usize>()
    }
    pub fn save_new(&self, path: &Path) -> Result<(), Error> {
        publish_new(path, &self.to_bytes()?).map_err(|e| Error::Storage(e.to_string()))
    }
    pub fn open(path: &Path) -> Result<Self, Error> {
        Self::from_bytes(
            &read_bounded(path, MAX_BYTES as u64).map_err(|e| Error::Storage(e.to_string()))?,
        )
    }
}
