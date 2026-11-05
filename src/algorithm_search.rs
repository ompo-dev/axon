//! Bounded, task-name-independent construction of typed iteration programs.
use super::{Error, Expr as E, Meter, Program, Value as V};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fmt};

#[path = "algorithm_enumeration.rs"]
mod enumeration;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub input: V,
    pub output: V,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub order: u16,
}
impl Default for Policy {
    fn default() -> Self {
        Self { order: 96 }
    }
}
impl Policy {
    pub fn validate(self) -> Result<(), Error> {
        if self.order >= 720 {
            Err(Error::Invalid("algorithm policy genotype"))
        } else {
            Ok(())
        }
    }
    fn lanes(self) -> Vec<usize> {
        let mut n = self.order as usize;
        let mut remaining = vec![0, 1, 2, 3, 4, 5];
        let mut out = vec![];
        for divisor in [120, 24, 6, 2, 1, 1] {
            let i = n / divisor;
            n %= divisor;
            out.push(remaining.remove(i));
        }
        out
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SearchBudget {
    pub candidates: usize,
    pub work: u64,
    pub policy: Policy,
}
impl Default for SearchBudget {
    fn default() -> Self {
        Self {
            candidates: 50000,
            work: 30_000_000,
            policy: Policy::default(),
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct SearchReceipt {
    pub program: Program,
    pub candidates: u64,
    pub work: u64,
    pub inherited: bool,
    pub abstracted: bool,
    pub templates: usize,
    pub mining_work: u64,
}
#[derive(Debug)]
pub struct SearchFailure {
    pub error: Error,
    pub candidates: u64,
    pub work: u64,
    pub templates: usize,
    pub mining_work: u64,
}
#[derive(Debug)]
pub struct HypothesisSample {
    pub hypotheses: Vec<(Program, Vec<V>)>,
    pub candidates: u64,
    pub work: u64,
    pub templates: usize,
    pub mining_work: u64,
    pub budget_exhausted: bool,
}
type SearchOutcome = (Result<SearchReceipt, SearchFailure>, Vec<(Program, Vec<V>)>);
pub(crate) struct LibrarySearch {
    pub result: Result<SearchReceipt, SearchFailure>,
    pub windows: usize,
}

pub(crate) fn learn_libraries(
    examples: &[Example],
    libraries: &[Vec<Program>],
    budget: SearchBudget,
) -> LibrarySearch {
    let mut work = 0;
    let mut candidates = 0;
    let mut templates = 0;
    let mut mining_work = 0;
    let mut error = Error::Unsupported;
    let mut windows = 0;
    if libraries.is_empty()
        || libraries.len() > 8
        || budget.candidates == 0
        || budget.candidates > 100000
        || budget.work == 0
        || budget.work > 100_000_000
    {
        error = Error::Invalid("algorithm catalogue search bounds");
    } else {
        for (index, library) in libraries.iter().enumerate() {
            let left = libraries.len() - index;
            let share = SearchBudget {
                candidates: (budget.candidates - candidates as usize) / left,
                work: (budget.work - work) / left as u64,
                ..budget
            };
            if share.candidates == 0 || share.work == 0 {
                error = Error::Budget;
                break;
            }
            windows += 1;
            match learn(examples, library, share) {
                Ok(mut receipt) => {
                    receipt.work += work;
                    receipt.candidates += candidates;
                    receipt.templates += templates;
                    receipt.mining_work += mining_work;
                    return LibrarySearch {
                        result: Ok(receipt),
                        windows,
                    };
                }
                Err(failure) => {
                    work += failure.work;
                    candidates += failure.candidates;
                    templates += failure.templates;
                    mining_work += failure.mining_work;
                    error = failure.error;
                    if !matches!(error, Error::Budget | Error::Unsupported) {
                        break;
                    }
                }
            }
        }
    }
    LibrarySearch {
        result: Err(SearchFailure {
            error,
            candidates,
            work,
            templates,
            mining_work,
        }),
        windows,
    }
}
impl fmt::Display for SearchFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({} candidates, {} work)",
            self.error, self.candidates, self.work
        )
    }
}
impl std::error::Error for SearchFailure {}
struct Search<'a> {
    examples: &'a [Example],
    budget: SearchBudget,
    meter: Meter,
    candidates: u64,
    seen: BTreeSet<Vec<u8>>,
    templates: usize,
    mining_work: u64,
    queries: &'a [V],
    hypotheses: Vec<(Program, Vec<V>)>,
    hypothesis_limit: usize,
    prediction_nodes: usize,
}
impl Search<'_> {
    fn propose(
        &mut self,
        e: E,
        inherited: bool,
        abstracted: bool,
    ) -> Result<Option<SearchReceipt>, Error> {
        self.observe(e, inherited, abstracted, false)
            .map(|(receipt, _)| receipt)
    }
    fn observe(
        &mut self,
        e: E,
        inherited: bool,
        abstracted: bool,
        collect: bool,
    ) -> Result<(Option<SearchReceipt>, Option<Vec<V>>), Error> {
        if e.nodes() > 256 {
            return Ok((None, None));
        }
        let bytes = serde_json::to_vec(&e).map_err(|e| Error::Storage(e.to_string()))?;
        let novel = !self.seen.contains(&bytes);
        if !novel && !collect {
            return Ok((None, None));
        }
        if novel {
            if self.candidates >= self.budget.candidates as u64 {
                return Err(Error::Budget);
            }
            self.seen.insert(bytes);
            self.candidates += 1;
        }
        self.meter.charge(e.nodes() as u64)?;
        let Ok(program) = Program::new(e) else {
            return Ok((None, None));
        };
        let mut outputs = Vec::new();
        let mut matched = true;
        for ex in self.examples {
            match program.run(&ex.input, &mut self.meter) {
                Ok(v) => {
                    if v != ex.output {
                        matched = false;
                        if !collect {
                            return Ok((None, None));
                        }
                    }
                    if collect {
                        outputs.push(v);
                    }
                }
                Err(Error::Budget) if self.meter.work >= self.budget.work => {
                    return Err(Error::Budget);
                }
                _ => return Ok((None, None)),
            }
        }
        if !matched {
            return Ok((None, Some(outputs)));
        }
        if self.hypothesis_limit > 0 {
            let mut predictions = Vec::new();
            let mut nodes = 0;
            for input in self.queries {
                match program.run(input, &mut self.meter) {
                    Ok(value) => {
                        nodes += value.nodes();
                        if nodes > 16384 {
                            return Ok((None, collect.then_some(outputs)));
                        }
                        predictions.push(value);
                    }
                    Err(Error::Budget) if self.meter.work >= self.budget.work => {
                        return Err(Error::Budget);
                    }
                    _ => return Ok((None, collect.then_some(outputs))),
                }
            }
            self.meter
                .charge((nodes * (self.hypotheses.len() + 1)) as u64)?;
            // Diversity is observed only on the authorized input pool, never its labels.
            if self.hypotheses.iter().any(|(_, old)| old == &predictions)
                || self.prediction_nodes + nodes > 32768
            {
                return Ok((None, collect.then_some(outputs)));
            }
            self.prediction_nodes += nodes;
            self.hypotheses.push((program.clone(), predictions));
            if self.hypotheses.len() < self.hypothesis_limit {
                return Ok((None, collect.then_some(outputs)));
            }
        }
        Ok((
            Some(SearchReceipt {
                program,
                candidates: self.candidates,
                work: self.meter.work,
                inherited,
                abstracted,
                templates: self.templates,
                mining_work: self.mining_work,
            }),
            collect.then_some(outputs),
        ))
    }
}
fn boxed(e: E) -> Box<E> {
    Box::new(e)
}
fn int(n: i64) -> E {
    E::Literal(V::Integer(n))
}
fn fold(seq: &E, init: E, body: E) -> E {
    E::Fold(boxed(seq.clone()), boxed(init), boxed(body))
}
fn call(p: &Program, arg: E) -> E {
    E::Call(boxed(p.expression().clone()), boxed(arg))
}
fn push(value: E) -> E {
    E::Push(boxed(E::Acc(0)), boxed(value))
}
fn choose(p: E, a: E, b: E) -> E {
    E::If(boxed(p), boxed(a), boxed(b))
}
fn same_kind(a: &V, b: &V) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn state_initializers(
    search: &mut Search<'_>,
    library: &[Program],
    parameter: &E,
) -> Result<Vec<E>, Error> {
    let mut proposals = vec![int(0), int(1), E::Empty, parameter.clone()];
    for p in library {
        for argument in [int(0), int(1), E::Empty, parameter.clone()] {
            proposals.push(call(p, argument));
        }
    }
    let mut selected = Vec::new();
    let mut signatures = Vec::new();
    let mut retained_nodes = 0;
    for expression in proposals {
        if expression.nodes() > 256 {
            continue;
        }
        let Ok(program) = Program::new(expression.clone()) else {
            continue;
        };
        let mut values = Vec::new();
        for example in search.examples {
            match program.run(&example.input, &mut search.meter) {
                Ok(value) if same_kind(&value, &example.output) => values.push(value),
                Err(Error::Budget) if search.meter.work >= search.budget.work => {
                    return Err(Error::Budget);
                }
                _ => break,
            }
        }
        let nodes: usize = values.iter().map(V::nodes).sum();
        if values.len() != search.examples.len() || retained_nodes + nodes > 16384 {
            continue;
        }
        search
            .meter
            .charge((nodes * (signatures.len() + 1)) as u64)?;
        if signatures.contains(&values) {
            continue;
        }
        retained_nodes += nodes;
        signatures.push(values);
        selected.push(expression);
        if selected.len() == 24 {
            break;
        }
    }
    Ok(selected)
}

pub fn learn(
    examples: &[Example],
    library: &[Program],
    budget: SearchBudget,
) -> Result<SearchReceipt, SearchFailure> {
    search(examples, library, budget, &[], 0).0
}

pub fn hypotheses(
    examples: &[Example],
    library: &[Program],
    budget: SearchBudget,
    inputs: &[V],
    maximum: usize,
) -> Result<HypothesisSample, SearchFailure> {
    let validation = (|| {
        if inputs.is_empty()
            || inputs.len() > 64
            || !(2..=16).contains(&maximum)
            || inputs.iter().map(V::nodes).sum::<usize>() > 8192
        {
            return Err(Error::Invalid("algorithm hypothesis bounds"));
        }
        for input in inputs {
            input.validate()?;
        }
        Ok(())
    })();
    if let Err(error) = validation {
        return Err(SearchFailure {
            error,
            candidates: 0,
            work: 0,
            templates: 0,
            mining_work: 0,
        });
    }
    let (result, hypotheses) = search(examples, library, budget, inputs, maximum);
    let (candidates, work, templates, mining_work, budget_exhausted) = match result {
        Ok(r) => (r.candidates, r.work, r.templates, r.mining_work, false),
        Err(f) if matches!(f.error, Error::Budget | Error::Unsupported) => (
            f.candidates,
            f.work,
            f.templates,
            f.mining_work,
            matches!(f.error, Error::Budget),
        ),
        Err(f) => return Err(f),
    };
    Ok(HypothesisSample {
        hypotheses,
        candidates,
        work,
        templates,
        mining_work,
        budget_exhausted,
    })
}

fn search(
    examples: &[Example],
    library: &[Program],
    budget: SearchBudget,
    queries: &[V],
    hypothesis_limit: usize,
) -> SearchOutcome {
    let mut search = Search {
        examples,
        budget,
        meter: Meter::new(budget.work),
        candidates: 0,
        seen: BTreeSet::new(),
        templates: 0,
        mining_work: 0,
        queries,
        hypotheses: Vec::new(),
        hypothesis_limit,
        prediction_nodes: 0,
    };
    let result = (|| {
        budget.policy.validate()?;
        if !(3..=32).contains(&examples.len())
            || library.len() > 32
            || budget.candidates == 0
            || budget.candidates > 100000
            || budget.work == 0
            || budget.work > 100_000_000
        {
            return Err(Error::Invalid("algorithm search bounds"));
        }
        for (i, e) in examples.iter().enumerate() {
            e.input.validate()?;
            e.output.validate()?;
            if examples[..i].iter().any(|old| old.input == e.input) {
                return Err(Error::Invalid("duplicate algorithm example"));
            }
        }
        let output = &examples[0].output;
        if !examples.iter().all(|e| same_kind(&e.output, output)) {
            return Err(Error::Invalid("mixed algorithm output kinds"));
        }
        let sources=vec![E::Input,E::Get(boxed(E::Input),0),E::Get(boxed(E::Input),1)].into_iter().filter(|source|examples.iter().all(|e|{
            let v=match source{E::Input=>Some(&e.input),E::Get(_,i)=>match &e.input{V::List(v)=>v.get(*i),_=>None},_=>None};
            matches!(v,Some(V::List(items)) if items.first().is_none_or(|first|items.iter().all(|v|same_kind(v,first))))
        })).collect::<Vec<_>>();
        let grid=examples.iter().all(|e|matches!((&e.input,&e.output),(V::List(a),V::List(b)) if !a.is_empty()&&!b.is_empty()&&a.iter().chain(b).all(|v|matches!(v,V::List(_)))));
        let item = E::Item(0);
        let acc = E::Acc(0);
        let parameter = E::Get(boxed(E::Input), 1);
        let mut atoms = vec![item.clone(), parameter.clone(), int(0), int(1)];
        for i in 0..3 {
            atoms.push(E::Get(boxed(item.clone()), i));
        }
        let mut literals = BTreeSet::new();
        fn collect(v: &V, out: &mut BTreeSet<V>) {
            match v {
                V::List(v) => {
                    for v in v {
                        collect(v, out)
                    }
                }
                V::Integer(n) if (-10..=10).contains(n) => {
                    out.insert(v.clone());
                }
                V::Text(_) | V::Boolean(_) => {
                    out.insert(v.clone());
                }
                _ => (),
            }
        }
        for e in examples {
            collect(&e.input, &mut literals);
            collect(&e.output, &mut literals);
        }
        for v in literals.into_iter().take(12) {
            let e = E::Literal(v);
            if !atoms.contains(&e) {
                atoms.push(e);
            }
        }
        let mut values = atoms.clone();
        for a in atoms.iter().take(7) {
            for b in atoms.iter().take(7) {
                values.push(E::Multiply(boxed(a.clone()), boxed(b.clone())));
                values.push(E::Add(boxed(a.clone()), boxed(b.clone())));
                values.push(E::Remainder(boxed(a.clone()), boxed(b.clone())));
            }
        }
        let mut predicates = vec![
            E::Less(boxed(acc.clone()), boxed(item.clone())),
            E::Less(boxed(item.clone()), boxed(acc.clone())),
            E::Equal(boxed(acc.clone()), boxed(item.clone())),
        ];
        let mut operands = atoms.clone();
        operands.push(acc.clone());
        for a in &operands {
            for b in &operands {
                if a != b {
                    predicates.push(E::Equal(boxed(a.clone()), boxed(b.clone())));
                    predicates.push(E::Less(boxed(a.clone()), boxed(b.clone())));
                }
            }
        }
        for a in values.iter().skip(atoms.len()).take(147) {
            predicates.push(E::Equal(boxed(a.clone()), boxed(int(0))));
        }
        let mut aggregates = vec![item.clone(), acc.clone()];
        for a in &values {
            aggregates.push(E::Add(boxed(acc.clone()), boxed(a.clone())));
            aggregates.push(E::Multiply(boxed(acc.clone()), boxed(a.clone())));
            aggregates.push(E::Subtract(boxed(acc.clone()), boxed(a.clone())));
        }
        macro_rules! attempt {
            ($expr:expr,$inherited:expr) => {
                if let Some(p) = search.propose($expr, $inherited, false)? {
                    return Ok(Some(p));
                }
            };
        }
        // Reserve a fair share for every construction family. Unused work rolls forward,
        // but an expensive early family cannot exhaust the whole learner's budget.
        let mut exhausted = false;
        for (index, lane) in budget.policy.lanes().into_iter().enumerate() {
            let remaining = (6 - index) as u64;
            search.budget.candidates = search.candidates as usize
                + (budget.candidates - search.candidates as usize) / remaining as usize;
            search.budget.work = search.meter.work + (budget.work - search.meter.work) / remaining;
            search.meter.limit = search.budget.work;
            let result = (|| {
                match lane {
                    0 => {
                        for p in library {
                            attempt!(p.expression().clone(), true);
                        }
                        let before = search.meter.work;
                        let mined = super::abstraction::mine(library, &mut search.meter);
                        search.mining_work += search.meter.work - before;
                        let templates = mined?;
                        search.templates = templates.len();
                        let mut expansions = Vec::new();
                        for template in templates {
                            let before = search.meter.work;
                            let choices = template.choices(&values, &predicates, &mut search.meter);
                            search.mining_work += search.meter.work - before;
                            let choices = choices?;
                            expansions.push((template, choices));
                        }
                        let rounds = expansions
                            .iter()
                            .map(|(_, c)| c.iter().map(Vec::len).sum())
                            .max()
                            .unwrap_or(0);
                        // Dovetail both templates and holes; no early pattern owns the entire quota.
                        for round in 0..rounds {
                            for (template, choices) in &expansions {
                                if choices.len() == 1 {
                                    if let Some(a) = choices[0].get(round)
                                        && let Some(p) = search.propose(
                                            template.instantiate(&[a]),
                                            true,
                                            true,
                                        )?
                                    {
                                        return Ok(Some(p));
                                    }
                                } else if choices.len() == 2 {
                                    for (i, a) in choices[0].iter().enumerate().take(round + 1) {
                                        if let Some(b) = choices[1].get(round - i)
                                            && let Some(p) = search.propose(
                                                template.instantiate(&[a, b]),
                                                true,
                                                true,
                                            )?
                                        {
                                            return Ok(Some(p));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    1 => {
                        if matches!(output, V::Integer(_) | V::Boolean(_)) {
                            for seq in &sources {
                                for init in [
                                    int(0),
                                    int(1),
                                    E::Get(boxed(seq.clone()), 0),
                                    E::Literal(V::Boolean(false)),
                                ] {
                                    for body in &aggregates {
                                        attempt!(fold(seq, init.clone(), body.clone()), false);
                                    }
                                }
                            }
                        }
                    }
                    2 => {
                        if !matches!(output, V::List(_)) {
                            for seq in &sources {
                                for p in &predicates {
                                    // Small accumulator updates get a turn before larger scalar expressions.
                                    for body in aggregates.iter().take(2 + atoms.len() * 3) {
                                        for init in [int(0), E::Get(boxed(seq.clone()), 0)] {
                                            attempt!(
                                                fold(
                                                    seq,
                                                    init,
                                                    choose(p.clone(), body.clone(), acc.clone())
                                                ),
                                                false
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    3 => {
                        if matches!(output, V::List(_)) && !grid {
                            for seq in &sources {
                                for v in &values {
                                    attempt!(fold(seq, E::Empty, push(v.clone())), false);
                                }
                                for p in &predicates {
                                    for v in atoms.iter().take(7) {
                                        for inverted in [false, true] {
                                            let (yes, no) = if inverted {
                                                (acc.clone(), push(v.clone()))
                                            } else {
                                                (push(v.clone()), acc.clone())
                                            };
                                            attempt!(
                                                fold(seq, E::Empty, choose(p.clone(), yes, no)),
                                                false
                                            );
                                        }
                                    }
                                }
                                for p in &predicates {
                                    for a in atoms.iter().take(7) {
                                        for b in atoms.iter().take(4) {
                                            attempt!(
                                                fold(
                                                    seq,
                                                    E::Empty,
                                                    push(choose(p.clone(), a.clone(), b.clone()))
                                                ),
                                                false
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    4 => {
                        if matches!(output, V::List(_)) {
                            // A fold may retain an entire record, not only a scalar or appended list.
                            for seq in &sources {
                                for field in 0..3 {
                                    let a = E::Get(boxed(acc.clone()), field);
                                    let b = E::Get(boxed(item.clone()), field);
                                    for predicate in [
                                        E::Less(boxed(a.clone()), boxed(b.clone())),
                                        E::Less(boxed(b.clone()), boxed(a.clone())),
                                    ] {
                                        for (yes, no) in [
                                            (item.clone(), acc.clone()),
                                            (acc.clone(), item.clone()),
                                        ] {
                                            attempt!(
                                                fold(
                                                    seq,
                                                    E::Get(boxed(seq.clone()), 0),
                                                    choose(predicate.clone(), yes, no)
                                                ),
                                                false
                                            );
                                        }
                                    }
                                }
                            }
                            for v in &atoms {
                                let inner = fold(&item, E::Empty, push(v.clone()));
                                attempt!(fold(&E::Input, E::Empty, push(inner)), false);
                            }
                            for from in &atoms {
                                for to in &atoms {
                                    let pixel = choose(
                                        E::Equal(boxed(item.clone()), boxed(from.clone())),
                                        to.clone(),
                                        item.clone(),
                                    );
                                    attempt!(
                                        fold(
                                            &E::Input,
                                            E::Empty,
                                            push(fold(&item, E::Empty, push(pixel)))
                                        ),
                                        false
                                    );
                                }
                            }
                        }
                        if let Some(receipt) = enumeration::learn(&mut search, library)? {
                            return Ok(Some(receipt));
                        }
                    }
                    5 => {
                        for p in library {
                            for seq in &sources {
                                for init in [int(0), int(1), E::Empty] {
                                    attempt!(
                                        fold(
                                            seq,
                                            init,
                                            call(p, E::List(vec![acc.clone(), item.clone()]))
                                        ),
                                        true
                                    );
                                }
                                if matches!(output, V::List(_)) {
                                    attempt!(
                                        fold(seq, E::Empty, push(call(p, item.clone()))),
                                        true
                                    );
                                }
                            }
                            for q in library {
                                attempt!(call(p, call(q, E::Input)), true);
                                attempt!(
                                    call(p, E::List(vec![call(q, E::Input), parameter.clone()])),
                                    true
                                );
                            }
                            if matches!(output, V::Integer(_)) {
                                for seq in &sources {
                                    let value =
                                        call(p, E::List(vec![item.clone(), parameter.clone()]));
                                    attempt!(
                                        fold(seq, int(0), E::Add(boxed(acc.clone()), boxed(value))),
                                        true
                                    );
                                }
                            }
                        }
                        // Learn an initial-state constructor too; no target-specific state is embedded here.
                        for init in state_initializers(&mut search, library, &parameter)? {
                            for seq in &sources {
                                if matches!(output, V::List(_)) {
                                    attempt!(
                                        fold(seq, init.clone(), push(item.clone())),
                                        matches!(&init, E::Call(..))
                                    );
                                }
                                for p in library {
                                    attempt!(
                                        fold(
                                            seq,
                                            init.clone(),
                                            call(p, E::List(vec![acc.clone(), item.clone()]))
                                        ),
                                        true
                                    );
                                }
                            }
                        }
                    }
                    _ => (),
                }
                Ok(None)
            })();
            match result {
                Ok(Some(receipt)) => return Ok(Some(receipt)),
                Ok(None) => (),
                Err(Error::Budget) => exhausted = true,
                Err(error) => return Err(error),
            }
        }
        if exhausted {
            Err(Error::Budget)
        } else {
            Ok(None)
        }
    })();
    let result = match result {
        Ok(Some(r)) => Ok(r),
        Ok(None) => Err(SearchFailure {
            error: Error::Unsupported,
            candidates: search.candidates,
            work: search.meter.work,
            templates: search.templates,
            mining_work: search.mining_work,
        }),
        Err(error) => Err(SearchFailure {
            error,
            candidates: search.candidates,
            work: search.meter.work,
            templates: search.templates,
            mining_work: search.mining_work,
        }),
    };
    (result, search.hypotheses)
}
