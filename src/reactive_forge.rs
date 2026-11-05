use super::*;
use std::{
    collections::{BTreeSet, HashSet},
    time::Instant,
};

#[derive(Clone, Copy)]
struct Row {
    input: [i64; 2],
    tools: [i64; 4],
    target: i64,
    reset: bool,
    before: i64,
    after: i64,
}
struct Search {
    budget: ReactiveBudget,
    stats: ReactiveSearch,
    start: Instant,
}
impl Search {
    fn stats(&self) -> ReactiveSearch {
        ReactiveSearch {
            elapsed_ns: self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            ..self.stats
        }
    }
    fn charge(&mut self, n: u64) -> Result<(), ReactiveError> {
        if n > self.budget.max_work - self.stats.work {
            return Err(ReactiveError::Budget(self.stats()));
        }
        self.stats.work += n;
        Ok(())
    }
    fn candidate(&mut self) -> Result<(), ReactiveError> {
        if self.stats.candidates >= self.budget.max_candidates {
            return Err(ReactiveError::Budget(self.stats()));
        }
        self.stats.candidates += 1;
        self.charge(1)
    }
    fn eval(&mut self, e: &Expr, frame: Frame<'_>) -> Result<i64, ReactiveError> {
        let result = e.eval(frame, &mut self.stats.work, self.budget.max_work);
        if matches!(result, Err(ReactiveError::Domain("execution fuel"))) {
            return Err(ReactiveError::Budget(self.stats()));
        }
        result
    }
}
fn atoms(arity: usize, tools: usize, state: bool) -> Vec<Arc<Expr>> {
    let mut v = Vec::new();
    for i in 0..arity {
        v.push(Arc::new(Expr::Input(i as u8)));
    }
    for i in 0..tools {
        v.push(Arc::new(Expr::Tool(i as u8)));
    }
    if state {
        v.push(Arc::new(Expr::State));
    }
    for n in [0, 1, -1] {
        v.push(Arc::new(Expr::Constant(n)));
    }
    v
}
fn arithmetic(atoms: &[Arc<Expr>]) -> Vec<Arc<Expr>> {
    let mut pool = atoms.to_vec();
    let mut seen: HashSet<_> = pool.iter().cloned().collect();
    for (i, a) in atoms.iter().enumerate() {
        for (j, b) in atoms.iter().enumerate() {
            for op in 0..3 {
                if op != 1 && i > j {
                    continue;
                }
                let e = Arc::new(match op {
                    0 => Expr::Add(a.clone(), b.clone()),
                    1 => Expr::Subtract(a.clone(), b.clone()),
                    _ => Expr::Multiply(a.clone(), b.clone()),
                });
                if seen.insert(e.clone()) {
                    pool.push(e);
                }
            }
        }
    }
    pool
}
fn guards(base: &[Arc<Expr>], rows: &[Row], arity: usize, max: usize) -> Vec<Arc<Expr>> {
    let values: BTreeSet<_> = rows
        .iter()
        .flat_map(|r| r.input[..arity].iter().copied())
        .collect();
    let values: Vec<_> = values.into_iter().collect();
    let mut result = base.to_vec();
    let n = values.len().min(max);
    for i in 0..n {
        let value = values[if n == 1 {
            0
        } else {
            i * (values.len() - 1) / (n - 1)
        }];
        let e = Arc::new(Expr::Constant(value));
        if !result.contains(&e) {
            result.push(e);
        }
    }
    result
}
fn frame<'a>(row: &'a Row, arity: usize, tools: usize, after: bool) -> Frame<'a> {
    Frame {
        input: &row.input[..arity],
        state: if after { row.after } else { row.before },
        tools: &row.tools[..tools],
    }
}
type Mask = [u64; 4];
fn contains(a: &Mask, b: &Mask) -> bool {
    a.iter().zip(b).all(|(a, b)| a & b == *b)
}
fn fit_output(
    search: &mut Search,
    rows: &[Row],
    schema: (usize, usize),
    after: bool,
    pool: &[Arc<Expr>],
    guard: &[Arc<Expr>],
    conditional: bool,
) -> Result<Option<Arc<Expr>>, ReactiveError> {
    let (arity, tools) = schema;
    let mut matches = Vec::<Mask>::new();
    for e in pool {
        let mut mask = [0; 4];
        let mut all = true;
        for (i, row) in rows.iter().enumerate() {
            let result = search.eval(e, frame(row, arity, tools, after));
            if matches!(result, Err(ReactiveError::Budget(_))) {
                return result.map(|_| None);
            }
            if result.is_ok_and(|n| n == row.target) {
                mask[i / 64] |= 1 << (i % 64);
            } else {
                all = false;
                if !conditional {
                    break;
                }
            }
        }
        if all {
            return Ok(Some(e.clone()));
        }
        if conditional {
            matches.push(mask);
        }
    }
    if !conditional {
        return Ok(None);
    }
    let mut full = [0; 4];
    for i in 0..rows.len() {
        full[i / 64] |= 1 << (i % 64);
    }
    for (i, a) in guard.iter().enumerate() {
        for (j, b) in guard.iter().enumerate() {
            if i == j {
                continue;
            }
            let mut yes = [0; 4];
            for (k, row) in rows.iter().enumerate() {
                if search.eval(a, frame(row, arity, tools, after))?
                    < search.eval(b, frame(row, arity, tools, after))?
                {
                    yes[k / 64] |= 1 << (k % 64);
                }
            }
            if yes.iter().all(|n| *n == 0) || yes == full {
                continue;
            }
            let mut no = [0; 4];
            for k in 0..4 {
                no[k] = full[k] & !yes[k];
            }
            let mut t = None;
            let mut f = None;
            for (k, m) in matches.iter().enumerate() {
                search.charge(8)?;
                if t.is_none() && contains(m, &yes) {
                    t = Some(k);
                }
                if f.is_none() && contains(m, &no) {
                    f = Some(k);
                }
                if t.is_some() && f.is_some() {
                    break;
                }
            }
            if let (Some(t), Some(f)) = (t, f) {
                return Ok(Some(Arc::new(Expr::IfLess(
                    a.clone(),
                    b.clone(),
                    pool[t].clone(),
                    pool[f].clone(),
                ))));
            }
        }
    }
    Ok(None)
}
fn rollout(
    search: &mut Search,
    rows: &[Row],
    arity: usize,
    tools: usize,
    update: &Expr,
    initial: i64,
) -> Result<Option<Vec<Row>>, ReactiveError> {
    let mut result = Vec::with_capacity(rows.len());
    let mut state = initial;
    for row in rows {
        if row.reset {
            state = initial;
        }
        let mut row = *row;
        row.before = state;
        match search.eval(
            update,
            Frame {
                input: &row.input[..arity],
                state,
                tools: &row.tools[..tools],
            },
        ) {
            Ok(value) => state = value,
            Err(ReactiveError::Budget(s)) => return Err(ReactiveError::Budget(s)),
            Err(_) => {
                search.stats.guarded_candidates += 1;
                return Ok(None);
            }
        }
        row.after = state;
        result.push(row);
    }
    Ok(Some(result))
}
struct Candidate {
    update: Option<Arc<Expr>>,
    initial: i64,
    after: bool,
    output: Arc<Expr>,
}
fn try_update(
    search: &mut Search,
    rows: &[Row],
    arity: usize,
    tools: usize,
    update: Arc<Expr>,
    pool: &[Arc<Expr>],
) -> Result<Option<Candidate>, ReactiveError> {
    for initial in [0, 1, -1] {
        search.candidate()?;
        let Some(states) = rollout(search, rows, arity, tools, &update, initial)? else {
            continue;
        };
        for after in [true, false] {
            if after && !update.has_state() {
                continue;
            }
            if let Some(output) =
                fit_output(search, &states, (arity, tools), after, pool, &[], false)?
                && output.has_state()
            {
                return Ok(Some(Candidate {
                    update: Some(update.clone()),
                    initial,
                    after,
                    output,
                }));
            }
        }
    }
    Ok(None)
}

pub(super) fn learn(
    name: &str,
    context: &str,
    traces: Vec<ReactiveTrace>,
    library: &[&ReactiveSkill],
    budget: ReactiveBudget,
) -> Result<ReactiveSkill, ReactiveError> {
    identifier(name)?;
    identifier(context)?;
    budget.validate()?;
    let arity = traces
        .first()
        .and_then(|t| t.events.first())
        .ok_or(ReactiveError::Invalid("empty experience"))?
        .input
        .len();
    validate_traces(&traces, arity, budget)?;
    if library.len() > budget.max_tools {
        return Err(ReactiveError::Invalid("library quota"));
    }
    let mut search = Search {
        budget,
        stats: ReactiveSearch::default(),
        start: Instant::now(),
    };
    let mut rows = Vec::new();
    for trace in &traces {
        for (i, event) in trace.events.iter().enumerate() {
            let mut input = [0; 2];
            input[..arity].copy_from_slice(&event.input);
            rows.push(Row {
                input,
                tools: [0; 4],
                target: event.output,
                reset: i == 0,
                before: 0,
                after: 0,
            });
        }
    }
    let mut tools = Vec::new();
    for source in library {
        if source.arity != arity {
            return Err(ReactiveError::Invalid("library schema"));
        }
        let circuit = source.compile()?;
        let mut cursor = circuit.cursor();
        let mut values = Vec::new();
        let mut valid = true;
        for row in &rows {
            if row.reset {
                cursor = circuit.cursor();
            }
            match circuit.step_counted(
                &mut cursor,
                source.context(),
                &row.input[..arity],
                &mut search.stats.work,
                budget.max_work,
            ) {
                Ok(v) => values.push(v),
                Err(ReactiveError::Domain("execution fuel")) => {
                    return Err(ReactiveError::Budget(search.stats()));
                }
                Err(_) => {
                    valid = false;
                    search.stats.refused_tools += 1;
                    break;
                }
            }
        }
        if valid {
            for (row, value) in rows.iter_mut().zip(values) {
                row.tools[tools.len()] = value;
            }
            tools.push((*source).clone());
        }
    }
    let base = atoms(arity, tools.len(), false);
    let pure = arithmetic(&base);
    let guard = guards(&base, &rows, arity, budget.max_guard_constants);
    search.candidate()?;
    let found = fit_output(
        &mut search,
        &rows,
        (arity, tools.len()),
        true,
        &pure,
        &guard,
        true,
    )?;
    let mut candidate = found.map(|output| Candidate {
        update: None,
        initial: 0,
        after: true,
        output,
    });
    if candidate.is_none() {
        let base = atoms(arity, tools.len(), true);
        let pure = arithmetic(&base);
        let guard = guards(&base, &rows, arity, budget.max_guard_constants);
        for e in &pure {
            candidate = try_update(&mut search, &rows, arity, tools.len(), e.clone(), &pure)?;
            if candidate.is_some() {
                break;
            }
        }
        // Enumerate generic comparison/branch compositions; no task-specific recurrence is proposed.
        'conditions: for cost in [5, 7, 9] {
            for (i, a) in guard.iter().enumerate() {
                for (j, b) in guard.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    for t in &pure {
                        for f in &pure {
                            if t == f || 3 + t.nodes() + f.nodes() != cost {
                                continue;
                            }
                            let e =
                                Arc::new(Expr::IfLess(a.clone(), b.clone(), t.clone(), f.clone()));
                            if candidate.is_none() {
                                candidate =
                                    try_update(&mut search, &rows, arity, tools.len(), e, &pure)?;
                            }
                            if candidate.is_some() {
                                break 'conditions;
                            }
                        }
                    }
                }
            }
        }
    }
    let Some(candidate) = candidate else {
        return Err(ReactiveError::NoProgram(search.stats()));
    };
    let mut used = [false; 4];
    candidate.output.visit_tools(&mut used);
    if let Some(e) = &candidate.update {
        e.visit_tools(&mut used);
    }
    let mut selected = Vec::new();
    let mut map = [0; 4];
    for (i, tool) in tools.into_iter().enumerate() {
        if used[i] {
            map[i] = selected.len() as u8;
            selected.push(tool);
        }
    }
    let skill = ReactiveSkill {
        name: name.into(),
        context: context.into(),
        arity,
        update: candidate.update.map(|e| e.remap(&map)),
        initial: candidate.initial,
        after_update: candidate.after,
        output: candidate.output.remap(&map),
        tools: selected,
        traces,
        search: search.stats(),
    };
    skill.validate()?;
    Ok(skill)
}
