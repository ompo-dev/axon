//! Collection procedures share the existing agent, grammar, compiled seed and feedback gate.
use super::{Agent, Error, Pattern, Rule, StructuralSeed, Term, utterance, valid_name};
use crate::structural::algorithm::{
    Binding, DEFAULT_FUEL, Example, InquiryEvent, InquiryPlan, InquiryReceipt, Meter, Program,
    Representation, SearchBudget, SearchReceipt, Value, inquire, learn_represented,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

impl StructuralSeed {
    pub(crate) fn algorithm_library_windows(&self) -> Vec<Vec<Program>> {
        let selected = self.algorithm_library();
        let all: Vec<_> = self
            .rules
            .iter()
            .filter_map(Rule::algorithm_program)
            .collect();
        let mut recent = Vec::new();
        for program in all.iter().rev() {
            if !recent.contains(program) {
                recent.push(*program);
            }
            if recent.len() == 16 {
                break;
            }
        }
        let mut omitted = Vec::new();
        for program in all {
            if !selected.contains(program) && !omitted.contains(&program) {
                omitted.push(program);
            }
        }
        let mut windows = vec![selected];
        for chunk in omitted.chunks(16) {
            windows.push(recent.iter().chain(chunk).map(|p| (*p).clone()).collect());
        }
        windows
    }
    pub(crate) fn algorithm_library(&self) -> Vec<Program> {
        let all: Vec<_> = self
            .rules
            .iter()
            .filter_map(Rule::algorithm_program)
            .collect();
        if all.len() <= 32 {
            return all.into_iter().cloned().collect();
        }
        // Keep recent complex skills and small reusable foundations, not just the oldest entries.
        let mut selected: Vec<&Program> = Vec::new();
        for p in all.iter().rev() {
            if !selected.contains(p) {
                selected.push(p);
            }
            if selected.len() == 16 {
                break;
            }
        }
        let mut compact = all.clone();
        compact.sort_by_key(|p| p.expression().nodes());
        for p in compact {
            if !selected.contains(&p) {
                selected.push(p);
            }
            if selected.len() == 32 {
                break;
            }
        }
        selected.into_iter().cloned().collect()
    }
}

#[derive(Debug, Serialize)]
pub struct AlgorithmAcquisition {
    pub program: Program,
    pub attempts: usize,
    pub candidates: u64,
    pub work: u64,
    pub inherited: bool,
    pub abstracted: bool,
    pub templates: usize,
    pub mining_work: u64,
    pub confirmations: usize,
    pub invalidated: Vec<String>,
    pub library_programs: usize,
    pub library_windows: usize,
    pub inquiry: Option<InquiryReceipt>,
}
#[derive(Debug, Serialize)]
pub struct AlgorithmEvent {
    pub phase: String,
    pub operation: String,
    pub candidates: u64,
    pub work: u64,
    pub templates: usize,
    pub mining_work: u64,
    pub program: Option<Program>,
    pub library_programs: usize,
    pub library_windows: usize,
    pub inquiry: Option<InquiryEvent>,
}
pub struct AlgorithmLesson<'a> {
    pub name: &'a str,
    pub examples: &'a [Example],
    pub admission: &'a [Example],
}
impl Agent {
    pub fn learn_algorithm_inquiring(
        &mut self,
        lesson: AlgorithmLesson<'_>,
        plan: &InquiryPlan,
        mut budget: SearchBudget,
        inherit: bool,
        oracle: &mut impl FnMut(&Value) -> Result<Value, Error>,
        emit: &mut impl FnMut(&AlgorithmEvent) -> Result<(), Error>,
    ) -> Result<AlgorithmAcquisition, Error> {
        let AlgorithmLesson {
            name,
            examples,
            admission,
        } = lesson;
        if !valid_name(name) || !(2..=8).contains(&admission.len()) {
            return Err(Error::Invalid("algorithm teaching bounds"));
        }
        plan.validate(examples)?;
        for (i, witness) in admission.iter().enumerate() {
            witness.input.validate()?;
            witness.output.validate()?;
            if examples
                .iter()
                .chain(&admission[..i])
                .any(|e| e.input == witness.input)
                || plan.inputs.contains(&witness.input)
            {
                return Err(Error::Invalid("algorithm admission overlaps inquiry"));
            }
        }
        let id = format!("algorithm-{name}");
        if self
            .seed
            .rules
            .iter()
            .any(|r| r.id() == id && r.algorithm_program().is_none())
        {
            return Err(Error::Invalid("algorithm rule namespace collision"));
        }
        let mut available = self.seed.clone();
        available.prepare_algorithm_dependencies()?;
        available.remove_algorithm_family(&id);
        let library = if inherit {
            available.algorithm_library()
        } else {
            Vec::new()
        };
        budget.policy = self.seed.algorithm_policy();
        let inquiry = inquire(examples, &library, budget, plan, oracle, &mut |e| {
            emit(&AlgorithmEvent {
                phase: e.phase.clone(),
                operation: name.into(),
                candidates: e.candidates,
                work: e.work,
                templates: 0,
                mining_work: 0,
                program: None,
                library_programs: library.len(),
                library_windows: 0,
                inquiry: Some(e.clone()),
            })
        })?;
        budget.candidates = budget
            .candidates
            .saturating_sub(inquiry.candidates as usize);
        budget.work = budget.work.saturating_sub(inquiry.work);
        let mut acquired = self.learn_algorithm(
            name,
            &inquiry.examples,
            admission,
            budget,
            inherit,
            &mut |e| {
                emit(&AlgorithmEvent {
                    phase: e.phase.clone(),
                    operation: e.operation.clone(),
                    candidates: e.candidates + inquiry.candidates,
                    work: e.work + inquiry.work,
                    templates: e.templates + inquiry.templates,
                    mining_work: e.mining_work + inquiry.mining_work,
                    program: e.program.clone(),
                    library_programs: e.library_programs,
                    library_windows: e.library_windows,
                    inquiry: None,
                })
            },
        )?;
        acquired.candidates += inquiry.candidates;
        acquired.work += inquiry.work;
        acquired.templates += inquiry.templates;
        acquired.mining_work += inquiry.mining_work;
        acquired.inquiry = Some(inquiry);
        Ok(acquired)
    }
    pub fn learn_algorithm(
        &mut self,
        name: &str,
        examples: &[Example],
        admission: &[Example],
        budget: SearchBudget,
        inherit: bool,
        emit: &mut impl FnMut(&AlgorithmEvent) -> Result<(), Error>,
    ) -> Result<AlgorithmAcquisition, Error> {
        self.learn_algorithm_represented(
            AlgorithmLesson {
                name,
                examples,
                admission,
            },
            Representation::Enumerate,
            budget,
            inherit,
            emit,
        )
    }

    /// Opt-in proposal representation; shares the existing refutation/atomic commit gate.
    pub fn learn_algorithm_represented(
        &mut self,
        lesson: AlgorithmLesson<'_>,
        representation: Representation,
        mut budget: SearchBudget,
        inherit: bool,
        emit: &mut impl FnMut(&AlgorithmEvent) -> Result<(), Error>,
    ) -> Result<AlgorithmAcquisition, Error> {
        let AlgorithmLesson {
            name,
            examples,
            admission,
        } = lesson;
        if !valid_name(name)
            || !(3..=24).contains(&examples.len())
            || !(2..=8).contains(&admission.len())
        {
            return Err(Error::Invalid("algorithm teaching bounds"));
        }
        let id = format!("algorithm-{name}");
        if self
            .seed
            .rules
            .iter()
            .any(|r| r.id() == id && r.algorithm_program().is_none())
        {
            return Err(Error::Invalid("algorithm rule namespace collision"));
        }
        for (i, e) in admission.iter().enumerate() {
            e.input.validate()?;
            e.output.validate()?;
            if examples
                .iter()
                .chain(&admission[..i])
                .any(|old| old.input == e.input)
            {
                return Err(Error::Invalid("algorithm admission overlaps teaching"));
            }
        }
        let mut base = self.seed.clone();
        base.prepare_algorithm_dependencies()?;
        let mut available = base.clone();
        available.remove_algorithm_family(&id);
        let libraries = if inherit {
            available.algorithm_library_windows()
        } else {
            vec![vec![]]
        };
        let library_programs = libraries.iter().map(Vec::len).max().unwrap_or(0);
        budget.policy = self.seed.algorithm_policy();
        let mut training = examples.to_vec();
        let mut witnesses = admission.to_vec();
        let mut candidates = 0;
        let mut work = 0;
        let mut templates = 0;
        let mut mining_work = 0;
        let mut library_windows = 0;
        emit(&AlgorithmEvent {
            phase: "need".into(),
            operation: name.into(),
            candidates,
            work,
            templates,
            mining_work,
            program: None,
            library_programs,
            library_windows,
            inquiry: None,
        })?;
        for attempt in 1..=7 {
            let search = learn_represented(&training, &libraries, budget, representation);
            library_windows += search.windows;
            let receipt = match search.result {
                Ok(r) => r,
                Err(f) => {
                    emit(&AlgorithmEvent {
                        phase: "search_failure".into(),
                        operation: name.into(),
                        candidates: candidates + f.candidates,
                        work: work + f.work,
                        templates: templates + f.templates,
                        mining_work: mining_work + f.mining_work,
                        program: None,
                        library_programs,
                        library_windows,
                        inquiry: None,
                    })?;
                    return Err(f.error);
                }
            };
            let SearchReceipt {
                program,
                inherited,
                abstracted,
                ..
            } = receipt;
            candidates += receipt.candidates;
            work += receipt.work;
            templates += receipt.templates;
            mining_work += receipt.mining_work;
            budget.candidates = budget
                .candidates
                .saturating_sub(receipt.candidates as usize);
            budget.work = budget.work.saturating_sub(receipt.work);
            emit(&AlgorithmEvent {
                phase: "hypothesis".into(),
                operation: name.into(),
                candidates,
                work,
                templates,
                mining_work,
                program: Some(program.clone()),
                library_programs,
                library_windows,
                inquiry: None,
            })?;
            let mut refuted = None;
            for (i, witness) in witnesses.iter().enumerate() {
                let limit = DEFAULT_FUEL.min(budget.work);
                if limit == 0 {
                    return Err(Error::Budget);
                }
                let mut meter = Meter::new(limit);
                let result = program.run(&witness.input, &mut meter);
                work += meter.work;
                budget.work = budget.work.saturating_sub(meter.work);
                emit(&AlgorithmEvent {
                    phase: "verification".into(),
                    operation: name.into(),
                    candidates,
                    work,
                    templates,
                    mining_work,
                    program: None,
                    library_programs,
                    library_windows,
                    inquiry: None,
                })?;
                if matches!(&result, Err(Error::Budget)) && meter.work >= limit {
                    return Err(Error::Budget);
                }
                if result.ok() != Some(witness.output.clone()) {
                    refuted = Some(i);
                    break;
                }
            }
            if let Some(i) = refuted {
                self.statistics.refutations += 1;
                emit(&AlgorithmEvent {
                    phase: "refutation".into(),
                    operation: name.into(),
                    candidates,
                    work,
                    templates,
                    mining_work,
                    program: Some(program),
                    library_programs,
                    library_windows,
                    inquiry: None,
                })?;
                training.push(witnesses.remove(i));
                if witnesses.len() < 2
                    || training.len() > 32
                    || budget.candidates == 0
                    || budget.work == 0
                {
                    return Err(Error::Budget);
                }
                continue;
            }
            let rule = Rule {
                id: id.clone(),
                guards: vec![],
                teaching_checksum: crate::live_store::checksum(
                    &serde_json::to_vec(&training).map_err(|e| Error::Storage(e.to_string()))?,
                ),
                holes: 1,
                input: Pattern::Tree(vec![
                    Pattern::Literal(Term::Symbol("invoke".into())),
                    Pattern::Literal(Term::Symbol(name.into())),
                    Pattern::Hole(0, {
                        let tag = |v: &Value| match v {
                            Value::Integer(_) => 0,
                            Value::Text(_) => 1,
                            _ => 2,
                        };
                        let first = tag(&training[0].input);
                        if training
                            .iter()
                            .chain(&witnesses)
                            .all(|e| tag(&e.input) == first)
                        {
                            first
                        } else {
                            3
                        }
                    }),
                ]),
                output: Pattern::Algorithm(Binding {
                    slot: 0,
                    program: program.clone(),
                }),
                dependencies: available.dependencies_for(&program)?,
            };
            rule.validate()?;
            let unchanged = base
                .rules
                .iter()
                .find(|r| r.id() == id)
                .and_then(Rule::algorithm_program)
                .is_some_and(|old| old == &program);
            let mut next = base.clone();
            let invalidated = if unchanged {
                next.rules.retain(|r| r.id() != id);
                vec![]
            } else {
                next.remove_algorithm_family(&id)
                    .into_iter()
                    .filter(|name| name != &id)
                    .collect()
            };
            if next.rules.len() >= 128 {
                return Err(Error::Budget);
            }
            next.rules.push(rule);
            next.to_bytes()?;
            emit(&AlgorithmEvent {
                phase: "consolidation".into(),
                operation: name.into(),
                candidates,
                work,
                templates,
                mining_work,
                program: Some(program.clone()),
                library_programs,
                library_windows,
                inquiry: None,
            })?;
            self.seed = next;
            self.statistics.acquisitions += 1;
            return Ok(AlgorithmAcquisition {
                program,
                attempts: attempt,
                candidates,
                work,
                inherited,
                abstracted,
                templates,
                mining_work,
                confirmations: witnesses.len(),
                invalidated,
                library_programs,
                library_windows,
                inquiry: None,
            });
        }
        Err(Error::Budget)
    }
    pub fn algorithm_feedback(
        &mut self,
        name: &str,
        input: &Value,
        actual: &Value,
    ) -> Result<bool, Error> {
        Ok(!self
            .algorithm_feedback_affected(name, input, actual)?
            .is_empty())
    }
    pub fn algorithm_feedback_affected(
        &mut self,
        name: &str,
        input: &Value,
        actual: &Value,
    ) -> Result<Vec<String>, Error> {
        if !valid_name(name) {
            return Err(Error::Invalid("algorithm feedback name"));
        }
        input.validate()?;
        actual.validate()?;
        let id = format!("algorithm-{name}");
        let Some(program) = self
            .seed
            .rules
            .iter()
            .find(|r| r.id() == id)
            .and_then(Rule::algorithm_program)
        else {
            return Ok(vec![]);
        };
        if program
            .evaluate(input, DEFAULT_FUEL)
            .ok()
            .is_some_and(|e| e.value == *actual)
        {
            return Ok(vec![]);
        }
        let mut next = self.seed.clone();
        next.prepare_algorithm_dependencies()?;
        let affected = next.remove_algorithm_family(&id);
        next.to_bytes()?;
        self.seed = next;
        self.statistics.refutations += 1;
        Ok(affected)
    }
    pub fn value_plan(&self, text: &str) -> Result<crate::structural::Prediction, Error> {
        let mut grammar = self.seed.clone();
        grammar
            .rules
            .retain(|r| !r.id().starts_with("acquired-") && r.algorithm_program().is_none());
        let direct = grammar.transform(&utterance(text)?, 512)?;
        if !direct.rules.is_empty() {
            return Ok(direct);
        }
        let mut meanings =
            grammar.composed_interpretations(text, crate::structural::language::DEFAULT_WORK)?;
        if meanings.len() != 1 {
            return Err(Error::Ambiguous);
        }
        Ok(meanings.remove(0))
    }
    pub fn answer_value(
        &self,
        text: &str,
        bindings: &BTreeMap<String, Value>,
    ) -> Result<Value, Error> {
        self.answer_value_plan(&self.value_plan(text)?.output, bindings)
    }
    /// Evaluate an already interpreted pure collection plan; never queries an environment.
    pub fn answer_value_plan(
        &self,
        plan: &Term,
        bindings: &BTreeMap<String, Value>,
    ) -> Result<Value, Error> {
        Ok(self.answer_value_plan_with_inputs(plan, bindings)?.0)
    }
    /// Names record actual binding reads, not names merely claimed by a plan.
    pub fn answer_value_plan_with_inputs(
        &self,
        plan: &Term,
        bindings: &BTreeMap<String, Value>,
    ) -> Result<(Value, Vec<String>), Error> {
        if bindings.len() > 16 {
            return Err(Error::Budget);
        }
        for (name, value) in bindings {
            if !valid_name(name) {
                return Err(Error::Invalid("binding name"));
            }
            value.validate()?;
        }
        plan.validate()?;
        fn resolve(
            agent: &Agent,
            t: &Term,
            bindings: &BTreeMap<String, Value>,
            calls: &mut usize,
            inputs: &mut BTreeSet<String>,
        ) -> Result<Value, Error> {
            *calls += 1;
            if *calls > 32 {
                return Err(Error::Budget);
            }
            match t {
                Term::Integer(n) => Ok(Value::Integer(*n)),
                Term::Symbol(name) => {
                    let value = bindings.get(name).ok_or(Error::Unsupported)?;
                    inputs.insert(name.clone());
                    Ok(value.clone())
                }
                Term::Tree(v) if v.len() >= 3 && v[0] == Term::Symbol("invoke".into()) => {
                    let mut request = vec![v[0].clone(), v[1].clone()];
                    for arg in &v[2..] {
                        request.push(
                            resolve(agent, arg, bindings, calls, inputs)?.to_transport_term()?,
                        );
                    }
                    let p = agent.seed.transform(&Term::Tree(request), 512)?;
                    if p.rules.is_empty() {
                        return Err(Error::Unsupported);
                    }
                    Value::from_term(&p.output)
                }
                _ => Err(Error::Unsupported),
            }
        }
        let mut inputs = BTreeSet::new();
        let value = resolve(self, plan, bindings, &mut 0, &mut inputs)?;
        Ok((value, inputs.into_iter().collect()))
    }
}
