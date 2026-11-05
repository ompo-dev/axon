//! Goal-directed acquisition. The environment exposes observations, never target programs.
use super::{Error, Pattern, Rule, StructuralSeed, Term, numeric::Calculation};
use crate::numeric::{NumericBudget, NumericError, NumericExample, NumericProgram, NumericSkill};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[path = "algorithm_agent.rs"]
mod collections;
pub use collections::{AlgorithmAcquisition, AlgorithmEvent, AlgorithmLesson};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Interface {
    pub arity: usize,
    pub epoch: u64,
    pub probes: Vec<Vec<i64>>,
}
pub trait Environment {
    fn describe(&self, name: &str) -> Option<Interface>;
    fn query(&mut self, name: &str, epoch: u64, input: &[i64]) -> Result<i64, Error>;
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub queries: usize,
    pub confirmations: usize,
    pub search_work: u64,
    pub active: bool,
    pub inherit: bool,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            queries: 24,
            confirmations: 4,
            search_work: 4_000_000,
            active: true,
            inherit: true,
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub phase: String,
    pub operation: String,
    pub epoch: u64,
    pub input: Vec<i64>,
    pub actual: Option<i64>,
    pub proposed: Option<i64>,
    pub expression: Option<String>,
    pub work: u64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Statistics {
    pub queries: usize,
    pub refutations: usize,
    pub acquisitions: usize,
    pub reuses: usize,
    pub search_work: u64,
    pub probe_work: u64,
    pub language_work: u64,
}
pub struct Agent {
    seed: StructuralSeed,
    budget: Budget,
    pub statistics: Statistics,
}
pub(crate) fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 24
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}
fn prefix(name: &str) -> String {
    format!("acquired-{name}-")
}
fn identity(name: &str, epoch: u64) -> String {
    format!("{}{epoch}", prefix(name))
}
fn owns(rule: &Rule, name: &str) -> bool {
    matches!(&rule.input, Pattern::Tree(v) if rule.id().starts_with("acquired-")
        && matches!(v.get(1), Some(Pattern::Literal(Term::Symbol(s))) if s == name))
}
fn call(name: &str, input: &[i64]) -> Term {
    Term::Tree(
        [Term::Symbol("invoke".into()), Term::Symbol(name.into())]
            .into_iter()
            .chain(input.iter().copied().map(Term::Integer))
            .collect(),
    )
}

/// Parentheses are grouping syntax, not a dictionary of commands. Word meaning is learned.
pub fn utterance(text: &str) -> Result<Term, Error> {
    if text.is_empty() || text.len() > 4096 || text.chars().any(char::is_control) {
        return Err(Error::Invalid("goal text bounds"));
    }
    let normalized = text.to_lowercase().replace('(', " ( ").replace(')', " ) ");
    let tokens: Vec<_> = normalized
        .trim_end_matches(['.', '?', '!'])
        .split_whitespace()
        .collect();
    fn group(tokens: &[&str], pos: &mut usize, depth: usize, nested: bool) -> Result<Term, Error> {
        if depth > 16 {
            return Err(Error::Budget);
        }
        let mut terms = vec![Term::Symbol("goal".into())];
        let mut closed = !nested;
        while let Some(&token) = tokens.get(*pos) {
            *pos += 1;
            match token {
                "(" => terms.push(group(tokens, pos, depth + 1, true)?),
                ")" if nested => {
                    closed = true;
                    break;
                }
                ")" => return Err(Error::Invalid("unbalanced goal grouping")),
                s => terms.push(
                    s.parse::<i64>()
                        .map(Term::Integer)
                        .unwrap_or_else(|_| Term::Symbol(s.into())),
                ),
            }
        }
        if !closed {
            return Err(Error::Invalid("unbalanced goal grouping"));
        }
        let term = Term::Tree(terms);
        term.validate()?;
        Ok(term)
    }
    group(&tokens, &mut 0, 0, false)
}
impl Agent {
    pub fn new(seed: StructuralSeed, budget: Budget) -> Result<Self, Error> {
        if !(7..=64).contains(&budget.queries)
            || !(2..=16).contains(&budget.confirmations)
            || budget.confirmations + 3 > budget.queries
            || !(1..=100_000_000).contains(&budget.search_work)
        {
            return Err(Error::Invalid("agent budget"));
        }
        Ok(Self {
            seed,
            budget,
            statistics: Statistics::default(),
        })
    }
    pub fn seed(&self) -> &StructuralSeed {
        &self.seed
    }
    pub fn into_seed(self) -> StructuralSeed {
        self.seed
    }
    /// An explicit conflicting observation suspends the procedure, not the whole agent.
    pub fn feedback(
        &mut self,
        name: &str,
        epoch: u64,
        input: &[i64],
        actual: i64,
    ) -> Result<bool, Error> {
        if !valid_name(name) || !(1..=4).contains(&input.len()) {
            return Err(Error::Invalid("feedback signature"));
        }
        let id = identity(name, epoch);
        let Some(rule) = self.seed.rules.iter().find(|r| r.id() == id) else {
            return Ok(false);
        };
        let predicted = rule.checked_apply(&call(name, input))?;
        if predicted == Some(Term::Integer(actual)) {
            return Ok(false);
        }
        self.seed.rules.retain(|r| r.id() != id);
        self.statistics.refutations += 1;
        Ok(true)
    }
    pub fn answer(
        &mut self,
        text: &str,
        environment: &mut impl Environment,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<i64, Error> {
        // Do not execute acquired procedures while interpreting a request: epochs must be checked first.
        let mut grammar = self.seed.clone();
        grammar.rules.retain(|r| !r.id().starts_with("acquired-"));
        let direct = grammar.transform(&utterance(text)?, 512)?;
        let mut language_work = 0;
        let plan = if direct.rules.is_empty() {
            let mut meanings = grammar.composed_interpretations(
                text,
                super::language::DEFAULT_WORK.min(self.budget.search_work as usize),
            )?;
            if meanings.len() != 1 {
                return Err(Error::Ambiguous);
            }
            let meaning = meanings.remove(0);
            language_work = meaning.visits as u64;
            self.statistics.language_work += language_work;
            emit(&Event {
                phase: "interpretation".into(),
                operation: "language".into(),
                epoch: 0,
                input: vec![],
                actual: None,
                proposed: None,
                expression: Some(
                    serde_json::to_string(&meaning.output)
                        .map_err(|e| Error::Storage(e.to_string()))?,
                ),
                work: language_work,
            })?;
            meaning.output
        } else {
            direct.output
        };
        self.execute_plan(&plan, language_work, environment, emit)
    }
    /// Execute a host-grounded plan through the same authority/epoch preflight.
    pub fn answer_plan(
        &mut self,
        plan: &Term,
        environment: &mut impl Environment,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<i64, Error> {
        self.execute_plan(plan, 0, environment, emit)
    }
    fn execute_plan(
        &mut self,
        plan: &Term,
        language_work: u64,
        environment: &mut impl Environment,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<i64, Error> {
        plan.validate()?;
        let mut interfaces = BTreeMap::new();
        fn preflight(
            t: &Term,
            env: &impl Environment,
            interfaces: &mut BTreeMap<String, Interface>,
            count: &mut usize,
        ) -> Result<(), Error> {
            *count += 1;
            if *count > 32 {
                return Err(Error::Budget);
            }
            if matches!(t, Term::Integer(_)) {
                return Ok(());
            }
            let Term::Tree(v) = t else {
                return Err(Error::Unsupported);
            };
            let Some(Term::Symbol(name)) = v.get(1) else {
                return Err(Error::Unsupported);
            };
            if v[0] != Term::Symbol("invoke".into()) || !valid_name(name) {
                return Err(Error::Unsupported);
            }
            let interface = env.describe(name).ok_or(Error::Unsupported)?;
            if !(1..=4).contains(&interface.arity)
                || v.len() != interface.arity + 2
                || !(7..=128).contains(&interface.probes.len())
                || interface.probes.iter().enumerate().any(|(i, p)| {
                    p.len() != interface.arity
                        || p.iter().any(|n| n.unsigned_abs() > 1u64 << 53)
                        || interface.probes[..i].contains(p)
                })
            {
                return Err(Error::Invalid("experiment interface bounds"));
            }
            interfaces.insert(name.clone(), interface);
            for child in &v[2..] {
                preflight(child, env, interfaces, count)?;
            }
            Ok(())
        }
        preflight(plan, environment, &mut interfaces, &mut 0)?;
        let mut remaining = self.budget.queries;
        let mut work = self.budget.search_work.saturating_sub(language_work);
        self.execute(
            plan,
            environment,
            &interfaces,
            &mut remaining,
            &mut work,
            emit,
        )
    }
    fn execute(
        &mut self,
        t: &Term,
        env: &mut impl Environment,
        interfaces: &BTreeMap<String, Interface>,
        queries: &mut usize,
        work: &mut u64,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<i64, Error> {
        if let Term::Integer(n) = t {
            return Ok(*n);
        }
        let Term::Tree(v) = t else {
            return Err(Error::Unsupported);
        };
        let Term::Symbol(name) = &v[1] else {
            return Err(Error::Unsupported);
        };
        let mut input = Vec::new();
        for child in &v[2..] {
            input.push(self.execute(child, env, interfaces, queries, work, emit)?);
        }
        let interface = &interfaces[name];
        let id = identity(name, interface.epoch);
        if let Some(rule) = self.seed.rules.iter().find(|r| r.id() == id) {
            let Some(Term::Integer(value)) = rule.checked_apply(&call(name, &input))? else {
                return Err(Error::Unsupported);
            };
            self.statistics.reuses += 1;
            emit(&Event {
                phase: "reuse".into(),
                operation: name.clone(),
                epoch: interface.epoch,
                input,
                actual: Some(value),
                proposed: None,
                expression: None,
                work: 0,
            })?;
            return Ok(value);
        }
        // A host-announced revision invalidates the old epoch before any new observations.
        self.seed.rules.retain(|r| !owns(r, name));
        let program = self.investigate(name, interface, env, queries, work, emit)?;
        let value = program
            .evaluate_integer(&input)
            .map_err(|_| Error::Unsupported)?;
        let mut pattern = vec![
            Pattern::Literal(Term::Symbol("invoke".into())),
            Pattern::Literal(Term::Symbol(name.clone())),
        ];
        pattern.extend((0..interface.arity).map(|i| Pattern::Hole(i, 0)));
        let rule = Rule {
            id,
            input: Pattern::Tree(pattern),
            output: Pattern::Calculate(Calculation {
                slots: (0..interface.arity).collect(),
                program,
            }),
            holes: interface.arity,
            guards: vec![],
            teaching_checksum: 0,
            dependencies: vec![],
        };
        rule.validate()?;
        if self.seed.rules.len() >= 128 {
            return Err(Error::Budget);
        }
        let mut next = self.seed.clone();
        next.rules.push(rule);
        next.to_bytes()?;
        self.seed = next;
        self.statistics.acquisitions += 1;
        Ok(value)
    }
    fn investigate(
        &mut self,
        name: &str,
        interface: &Interface,
        env: &mut impl Environment,
        queries: &mut usize,
        work: &mut u64,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<NumericProgram, Error> {
        let mut observed: Vec<(Vec<i64>, i64)> = Vec::new();
        let mut candidate: Option<NumericProgram> = None;
        let mut confirmations = 0;
        let mut alternatives: Vec<NumericProgram> = if self.budget.inherit {
            self.seed
                .rules
                .iter()
                .filter_map(|r| match &r.output {
                    Pattern::Calculate(c) if c.program.arity() == interface.arity => {
                        Some(c.program.clone())
                    }
                    _ => None,
                })
                .take(16)
                .collect()
        } else {
            vec![]
        };
        emit(&Event {
            phase: "need".into(),
            operation: name.into(),
            epoch: interface.epoch,
            input: vec![],
            actual: None,
            proposed: None,
            expression: None,
            work: 0,
        })?;
        loop {
            if *queries == 0 || observed.len() >= 32 {
                return Err(Error::Budget);
            }
            let mut best = None;
            let mut score_work = 0;
            for point in &interface.probes {
                if observed.iter().any(|(x, _)| x == point) {
                    continue;
                }
                let mut predictions = Vec::new();
                if self.budget.active {
                    for p in alternatives.iter().chain(candidate.iter()) {
                        score_work += p.instructions() as u64;
                        if score_work > 1_000_000 {
                            return Err(Error::Budget);
                        }
                        let value = p.evaluate_integer(point).ok();
                        if !predictions.contains(&value) {
                            predictions.push(value);
                        }
                    }
                }
                let score = predictions.len();
                if best.as_ref().is_none_or(|(_, old)| score > *old) {
                    best = Some((point.clone(), score));
                }
            }
            self.statistics.probe_work += score_work;
            let (point, _) = best.ok_or(Error::Budget)?;
            let predicted = candidate
                .as_ref()
                .and_then(|p| p.evaluate_integer(&point).ok());
            *queries -= 1;
            self.statistics.queries += 1;
            let actual = env.query(name, interface.epoch, &point)?;
            if env.describe(name).is_none_or(|current| {
                current.epoch != interface.epoch || current.arity != interface.arity
            }) {
                return Err(Error::Invalid("environment changed during investigation"));
            }
            if actual.unsigned_abs() > 1u64 << 53 {
                return Err(Error::Invalid("observation outside exact search range"));
            }
            let refuted = candidate.is_some() && predicted != Some(actual);
            if refuted {
                self.statistics.refutations += 1;
            }
            emit(&Event {
                phase: if refuted { "refutation" } else { "observation" }.into(),
                operation: name.into(),
                epoch: interface.epoch,
                input: point.clone(),
                actual: Some(actual),
                proposed: predicted,
                expression: candidate.as_ref().map(NumericProgram::expression),
                work: score_work,
            })?;
            observed.push((point, actual));
            if candidate.is_some() && !refuted {
                confirmations += 1;
                if confirmations >= self.budget.confirmations {
                    emit(&Event {
                        phase: "consolidation".into(),
                        operation: name.into(),
                        epoch: interface.epoch,
                        input: vec![],
                        actual: None,
                        proposed: None,
                        expression: candidate.as_ref().map(NumericProgram::expression),
                        work: 0,
                    })?;
                    return candidate.ok_or(Error::Unsupported);
                }
                continue;
            }
            if observed.len() < 3 {
                continue;
            }
            if let Some(p) = candidate.take() {
                alternatives.push(p);
                if alternatives.len() > 16 {
                    alternatives.remove(0);
                }
            }
            confirmations = 0;
            if *work == 0 {
                return Err(Error::Budget);
            }
            let examples = observed
                .iter()
                .map(|(x, y)| {
                    NumericExample::new(x.iter().map(|n| *n as f64).collect::<Vec<_>>(), *y as f64)
                })
                .collect();
            let library: Vec<_> = if self.budget.inherit {
                alternatives
                    .iter()
                    .map(|p| (p, crate::live_store::checksum(&p.encode_program())))
                    .collect()
            } else {
                vec![]
            };
            let result = NumericSkill::learn_integer_from_programs(
                examples,
                &library,
                NumericBudget {
                    max_cost: 8,
                    max_terms: 30_000,
                    max_work: *work,
                },
            );
            let spent = match &result {
                Ok(s) => s.search().scalar_work,
                Err(NumericError::Budget(s) | NumericError::NoProgram(s)) => s.scalar_work,
                Err(_) => return Err(Error::Unsupported),
            };
            *work = work.saturating_sub(spent);
            self.statistics.search_work += spent;
            match result {
                Ok(s) => {
                    let p = s.program().clone();
                    if !observed
                        .iter()
                        .all(|(x, y)| p.evaluate_integer(x).ok() == Some(*y))
                    {
                        return Err(Error::Unsupported);
                    }
                    emit(&Event {
                        phase: "hypothesis".into(),
                        operation: name.into(),
                        epoch: interface.epoch,
                        input: vec![],
                        actual: None,
                        proposed: None,
                        expression: Some(p.expression()),
                        work: spent,
                    })?;
                    candidate = Some(p);
                }
                Err(_) => return Err(Error::Budget),
            }
        }
    }
}
