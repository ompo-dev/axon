//! Bounded evolution of numerical program graphs; not evolution of a general cognitive seed.
use super::*;
use crate::behavior_lab::Rng;

const MAX_GENES: usize = 64;
const MAX_SIGNALS: usize = MAX_GENES + 8;
const MAGIC: &[u8; 8] = b"AXEV0002";
const FILE_LIMIT: u64 = 65_536;

#[path = "numeric_constructive.rs"]
pub mod constructive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvolutionBudget {
    pub population: usize,
    pub generations: usize,
    pub genes: usize,
    pub mutations: usize,
    pub max_work: u64,
    pub local_refinement: bool,
    pub refinement_slice_work: u64,
    pub refinement_share_percent: u8,
    pub refine_inherited_calls: bool,
}
impl Default for EvolutionBudget {
    fn default() -> Self {
        Self {
            population: 64,
            generations: 2000,
            genes: 32,
            mutations: 3,
            max_work: 100_000_000,
            local_refinement: true,
            refinement_slice_work: 0,
            refinement_share_percent: 100,
            refine_inherited_calls: false,
        }
    }
}
impl EvolutionBudget {
    pub fn balanced() -> Self {
        Self {
            refinement_slice_work: 4_000_000,
            refinement_share_percent: 50,
            refine_inherited_calls: true,
            ..Self::default()
        }
    }
    pub(super) fn validate(self) -> Result<(), NumericError> {
        if !(4..=128).contains(&self.population)
            || !(1..=10_000).contains(&self.generations)
            || !(2..=MAX_GENES).contains(&self.genes)
            || !(1..=8).contains(&self.mutations)
            || !(1..=200_000_000).contains(&self.max_work)
            || self.refinement_share_percent > 100
            || self.refinement_slice_work > 200_000_000
        {
            return Err(NumericError::Invalid("evolution budget"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EvolutionStatistics {
    pub evaluations: u64,
    pub work: u64,
    pub generation: usize,
    pub mutations: u64,
    pub crossovers: u64,
    pub fresh: u64,
    pub invalid_nodes: u64,
    pub unexportable: u64,
    pub admission_work: u64,
    pub local_proposals: u64,
    pub best_loss: f64,
    pub elapsed_ns: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvolutionStop {
    Admitted,
    Refuted,
    WorkLimit,
    GenerationLimit,
}
#[derive(Clone, Debug)]
pub struct GenerationRecord {
    pub generation: usize,
    pub evaluations: u64,
    pub work: u64,
    pub loss: f64,
    pub correct: usize,
    pub active_genes: usize,
    pub distinct_genomes: usize,
    pub genome: u64,
    pub parents: [u64; 2],
}
pub struct EvolutionRun {
    pub skill: Option<EvolvedSkill>,
    pub stop: EvolutionStop,
    pub statistics: EvolutionStatistics,
    pub generations: Vec<GenerationRecord>,
    pub refutation: Option<EvolutionRefutation>,
    pub allocation: SearchAllocation,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchAllocation {
    pub refinement_work: u64,
    pub refinements: u64,
    pub truncated_refinements: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvolutionPhase {
    Search,
    Generation,
    Finished,
}
#[derive(Clone, Copy, Debug)]
pub struct EvolutionProgress {
    pub phase: EvolutionPhase,
    pub statistics: EvolutionStatistics,
    pub allocation: SearchAllocation,
    pub correct: usize,
    pub active_genes: usize,
    pub distinct_genomes: usize,
    pub stop: Option<EvolutionStop>,
}
#[derive(Clone, Debug)]
pub struct EvolutionRefutation {
    pub input: Vec<f64>,
    pub expected: f64,
    pub actual: Option<f64>,
    pub expression: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Gene {
    function: u8,
    args: [u8; 4],
}
#[derive(Clone, Debug)]
struct Genome {
    arity: usize,
    genes: Vec<Gene>,
    parents: [u64; 2],
}
#[derive(Clone, Debug)]
struct Inherited {
    identity: u64,
    program: NumericProgram,
}
#[derive(Clone, Debug)]
pub struct EvolvedSkill {
    name: String,
    context: String,
    genome: Genome,
    output: usize,
    tools: Vec<Inherited>,
    program: NumericProgram,
    dependencies: Vec<u64>,
    training: Vec<NumericExample>,
    admission: Vec<NumericExample>,
    tolerance: NumericTolerance,
    statistics: EvolutionStatistics,
}
#[derive(Clone)]
struct Fitness {
    errors: [f64; 32],
    loss: f64,
    correct: usize,
    output: usize,
    active: usize,
}
impl Fitness {
    fn better(&self, other: &Self, samples: usize) -> bool {
        let exact = self.correct == samples;
        let old_exact = other.correct == samples;
        if exact != old_exact {
            return exact;
        }
        if exact {
            return (self.active, self.loss) < (other.active, other.loss);
        }
        if self.loss == other.loss {
            self.active < other.active
        } else {
            self.loss < other.loss
        }
    }
}
struct Individual {
    genome: Genome,
    fitness: Fitness,
}
fn base(arity: usize) -> usize {
    arity + 4
}
fn arguments(gene: Gene, tools: &[Inherited]) -> usize {
    match gene.function {
        4 => 1,
        0..=3 => 2,
        i => tools[i as usize - 5].program.arity(),
    }
}
fn active(genome: &Genome, output: usize, tools: &[Inherited]) -> [bool; MAX_GENES] {
    let mut needed = [false; MAX_GENES];
    if output >= base(genome.arity) {
        needed[output - base(genome.arity)] = true;
    }
    for i in (0..genome.genes.len()).rev() {
        if needed[i] {
            let gene = genome.genes[i];
            for a in &gene.args[..arguments(gene, tools)] {
                if *a as usize >= base(genome.arity) {
                    needed[*a as usize - base(genome.arity)] = true;
                }
            }
        }
    }
    needed
}
fn charge(stats: &mut EvolutionStatistics, limit: u64, work: u64) -> bool {
    if work > limit.saturating_sub(stats.work) {
        return false;
    }
    stats.work += work;
    true
}
fn signal_values(
    genome: &Genome,
    tools: &[Inherited],
    rows: &[NumericExample],
    stats: &mut EvolutionStatistics,
    limit: u64,
) -> Option<[[f64; 32]; MAX_SIGNALS]> {
    let mut values = [[0.0; 32]; MAX_SIGNALS];
    for (r, row) in rows.iter().enumerate() {
        if !charge(stats, limit, base(genome.arity) as u64) {
            return None;
        }
        for (i, value) in row.input.iter().enumerate() {
            values[i][r] = *value;
        }
        for (i, value) in [-1.0, 0.0, 1.0, 2.0].into_iter().enumerate() {
            values[genome.arity + i][r] = value;
        }
    }
    for (i, gene) in genome.genes.iter().enumerate() {
        for (r, _) in rows.iter().enumerate() {
            if !charge(stats, limit, 1) {
                return None;
            }
            let a = values[gene.args[0] as usize][r];
            let b = values[gene.args[1] as usize][r];
            let result = match gene.function {
                0..=3 => [Op::Add, Op::Subtract, Op::Multiply, Op::Divide][gene.function as usize]
                    .binary(a, b),
                4 => root(a),
                tool => {
                    let source = &tools[tool as usize - 5].program;
                    let mut args = [0.0; 4];
                    for (j, slot) in args.iter_mut().enumerate().take(source.arity()) {
                        *slot = values[gene.args[j] as usize][r];
                    }
                    if !charge(stats, limit, source.instructions() as u64) {
                        return None;
                    }
                    source.evaluate(&args[..source.arity()])
                }
            };
            values[base(genome.arity) + i][r] = result.unwrap_or(f64::NAN);
        }
        if values[base(genome.arity) + i][..rows.len()]
            .iter()
            .any(|n| !n.is_finite())
        {
            stats.invalid_nodes += 1;
        }
    }
    Some(values)
}
fn score(
    values: &[f64; 32],
    rows: &[NumericExample],
    tolerance: NumericTolerance,
    stats: &mut EvolutionStatistics,
    limit: u64,
) -> Option<Fitness> {
    if !charge(stats, limit, rows.len() as u64) {
        return None;
    }
    let mut fitness = Fitness {
        errors: [0.0; 32],
        loss: 0.0,
        correct: 0,
        output: 0,
        active: 0,
    };
    for (i, row) in rows.iter().enumerate() {
        fitness.correct += usize::from(tolerance.agrees(values[i], row.output));
        let error = ((values[i] - row.output).abs() / (1.0 + row.output.abs())).min(1e6);
        fitness.errors[i] = if values[i].is_finite() && error.is_finite() {
            error
        } else {
            1e6
        };
        fitness.loss += fitness.errors[i];
    }
    fitness.loss /= rows.len() as f64;
    Some(fitness)
}
fn evaluate(
    genome: &Genome,
    tools: &[Inherited],
    rows: &[NumericExample],
    tolerance: NumericTolerance,
    stats: &mut EvolutionStatistics,
    limit: u64,
) -> Option<Fitness> {
    let values = signal_values(genome, tools, rows, stats, limit)?;
    let signals = base(genome.arity) + genome.genes.len();
    let mut best: Option<Fitness> = None;
    // Readout selection is supplied; each genome exposes all of its computed signals.
    for (output, values) in values.iter().enumerate().take(signals) {
        let mut fitness = score(values, rows, tolerance, stats, limit)?;
        fitness.output = output;
        if best
            .as_ref()
            .is_none_or(|b| fitness.correct == rows.len() || fitness.loss <= b.loss)
        {
            fitness.active = active(genome, output, tools)
                .into_iter()
                .filter(|n| *n)
                .count();
            if best.as_ref().is_none_or(|b| fitness.better(b, rows.len())) {
                best = Some(fitness);
            }
        }
    }
    stats.evaluations += 1;
    best
}

#[allow(clippy::too_many_arguments)]
fn refine(
    genome: &Genome,
    fitness: &Fitness,
    tools: &[Inherited],
    rows: &[NumericExample],
    tolerance: NumericTolerance,
    stats: &mut EvolutionStatistics,
    limit: u64,
    keep_partial: bool,
    refine_calls: bool,
) -> Option<(Genome, Fitness, bool)> {
    let values = signal_values(genome, tools, rows, stats, limit)?;
    let tail = genome.genes.len()
        - if refine_calls && genome.genes.len() >= 3 {
            3
        } else {
            2
        };
    let signal = base(genome.arity) + tail;
    let mut features: Vec<_> = (0..signal)
        .filter(|i| values[*i][..rows.len()].iter().all(|v| v.is_finite()))
        .collect();
    features.sort_by_key(|i| active(genome, *i, tools).into_iter().filter(|v| *v).count());
    let mut seen = std::collections::HashSet::new();
    features.retain(|i| {
        seen.insert(
            values[*i][..rows.len()]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        )
    });
    features.truncate(16);
    let mut best = fitness.clone();
    let mut chosen = genome.clone();
    let mut consider = |replacement: &[Gene],
                        output_values: &[f64; 32],
                        stats: &mut EvolutionStatistics|
     -> Option<bool> {
        stats.local_proposals += 1;
        let mut proposed = score(output_values, rows, tolerance, stats, limit)?;
        if proposed.correct == rows.len() || proposed.loss < best.loss {
            let mut child = genome.clone();
            child.genes[tail..tail + replacement.len()].copy_from_slice(replacement);
            proposed.output = signal + replacement.len() - 1;
            proposed.active = active(&child, proposed.output, tools)
                .into_iter()
                .filter(|v| *v)
                .count();
            if proposed.better(&best, rows.len()) {
                best = proposed;
                chosen = child;
            }
        }
        Some(best.correct == rows.len())
    };
    // Search bounded one/two-operation mutations over observed signals, not named formula templates.
    let completed = (|| -> Option<()> {
        if refine_calls && !features.is_empty() {
            let mut order: Vec<_> = (0..tools.len()).collect();
            order.sort_by_key(|i| (tools[*i].program.arity(), tools[*i].program.instructions()));
            for i in order {
                let tool = &tools[i].program;
                for binding in 0..features.len().pow(tool.arity() as u32) {
                    let mut rest = binding;
                    let mut args = [0; 4];
                    for arg in args.iter_mut().take(tool.arity()) {
                        *arg = features[rest % features.len()] as u8;
                        rest /= features.len();
                    }
                    if !charge(stats, limit, (tool.instructions() * rows.len()) as u64) {
                        return None;
                    }
                    let mut result = [f64::NAN; 32];
                    for (r, value) in result.iter_mut().enumerate().take(rows.len()) {
                        let mut input = [0.0; 4];
                        for (a, slot) in args.iter().zip(input.iter_mut()).take(tool.arity()) {
                            *slot = values[*a as usize][r];
                        }
                        *value = tool.evaluate(&input[..tool.arity()]).unwrap_or(f64::NAN);
                    }
                    if consider(
                        &[Gene {
                            function: (5 + i) as u8,
                            args,
                        }],
                        &result,
                        stats,
                    )? {
                        return Some(());
                    }
                }
                // Bind acquired binary operations to two new channel transformations.
                if tool.arity() == 2 && genome.genes.len() >= 3 {
                    let mut transforms = Vec::new();
                    for a in 0..genome.arity {
                        for b in 0..genome.arity {
                            for (op, operator) in [Op::Add, Op::Subtract, Op::Multiply, Op::Divide]
                                .into_iter()
                                .enumerate()
                            {
                                if matches!(operator, Op::Add | Op::Multiply) && a > b {
                                    continue;
                                }
                                if !charge(stats, limit, rows.len() as u64) {
                                    return None;
                                }
                                let mut result = [f64::NAN; 32];
                                for r in 0..rows.len() {
                                    result[r] = operator
                                        .binary(values[a][r], values[b][r])
                                        .unwrap_or(f64::NAN);
                                }
                                if result[..rows.len()].iter().all(|v| v.is_finite()) {
                                    transforms.push((
                                        Gene {
                                            function: op as u8,
                                            args: [a as u8, b as u8, 0, 0],
                                        },
                                        result,
                                    ));
                                }
                            }
                        }
                    }
                    for (left, x) in &transforms {
                        for (right, y) in &transforms {
                            if !charge(stats, limit, (tool.instructions() * rows.len()) as u64) {
                                return None;
                            }
                            let mut result = [f64::NAN; 32];
                            for r in 0..rows.len() {
                                result[r] = tool.evaluate(&[x[r], y[r]]).unwrap_or(f64::NAN);
                            }
                            if consider(
                                &[
                                    *left,
                                    *right,
                                    Gene {
                                        function: (5 + i) as u8,
                                        args: [signal as u8, (signal + 1) as u8, 0, 0],
                                    },
                                ],
                                &result,
                                stats,
                            )? {
                                return Some(());
                            }
                        }
                    }
                }
            }
        }
        'search: for &a in &features {
            for &b in &features {
                for (op, operator) in [Op::Add, Op::Subtract, Op::Multiply, Op::Divide]
                    .into_iter()
                    .enumerate()
                {
                    if matches!(operator, Op::Add | Op::Multiply) && a > b {
                        continue;
                    }
                    let first = Gene {
                        function: op as u8,
                        args: [a as u8, b as u8, 0, 0],
                    };
                    let mut inner = [0.0; 32];
                    if !charge(stats, limit, rows.len() as u64) {
                        return None;
                    }
                    for r in 0..rows.len() {
                        inner[r] = operator
                            .binary(values[a][r], values[b][r])
                            .unwrap_or(f64::NAN);
                    }
                    if consider(&[first], &inner, stats)? {
                        break 'search;
                    }
                    let mut transformed = [0.0; 32];
                    if !charge(stats, limit, rows.len() as u64) {
                        return None;
                    }
                    for r in 0..rows.len() {
                        transformed[r] = root(inner[r]).unwrap_or(f64::NAN);
                    }
                    if consider(
                        &[
                            first,
                            Gene {
                                function: 4,
                                args: [signal as u8, 0, 0, 0],
                            },
                        ],
                        &transformed,
                        stats,
                    )? {
                        break 'search;
                    }
                    for &c in &features {
                        for (outer, operator) in [Op::Add, Op::Subtract, Op::Multiply, Op::Divide]
                            .into_iter()
                            .enumerate()
                        {
                            for reverse in [false, true] {
                                if !charge(stats, limit, rows.len() as u64) {
                                    return None;
                                }
                                for r in 0..rows.len() {
                                    let (x, y) = if reverse {
                                        (values[c][r], inner[r])
                                    } else {
                                        (inner[r], values[c][r])
                                    };
                                    transformed[r] = operator.binary(x, y).unwrap_or(f64::NAN);
                                }
                                let (a, b) = if reverse { (c, signal) } else { (signal, c) };
                                if consider(
                                    &[
                                        first,
                                        Gene {
                                            function: outer as u8,
                                            args: [a as u8, b as u8, 0, 0],
                                        },
                                    ],
                                    &transformed,
                                    stats,
                                )? {
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
        }
        Some(())
    })();
    if completed.is_none() && !keep_partial {
        return None;
    }
    if best.better(fitness, rows.len()) {
        chosen.parents = [identity(genome, fitness.output, tools), 0];
        stats.mutations += chosen
            .genes
            .iter()
            .zip(&genome.genes)
            .filter(|(a, b)| a != b)
            .count() as u64;
    }
    Some((chosen, best, completed.is_none()))
}
fn reference(rng: &mut Rng, limit: usize, inputs: usize) -> u8 {
    let available = if rng.pick(4) == 0 { inputs } else { limit };
    rng.pick(available) as u8
}
fn gene(rng: &mut Rng, limit: usize, inputs: usize, tools: usize) -> Gene {
    let mut args = [0; 4];
    for a in &mut args {
        *a = reference(rng, limit, inputs);
    }
    if rng.pick(5) == 0 {
        args[1] = args[0];
    }
    Gene {
        function: rng.pick(5 + tools) as u8,
        args,
    }
}
fn fresh(rng: &mut Rng, arity: usize, genes: usize, tools: &[Inherited]) -> Genome {
    let mut genome = Genome {
        arity,
        genes: (0..genes)
            .map(|i| gene(rng, base(arity) + i, base(arity), tools.len()))
            .collect(),
        parents: [0; 2],
    };
    // Expose acquired operations on raw channels; no target law or semantic name is consulted.
    for (i, tool) in tools.iter().enumerate().take(genes) {
        let mut args = [0; 4];
        let direct = tool.program.arity() == arity && rng.pick(2) == 0;
        for (j, a) in args.iter_mut().enumerate() {
            *a = if direct && j < arity {
                j as u8
            } else {
                rng.pick(arity) as u8
            };
        }
        genome.genes[i] = Gene {
            function: (5 + i) as u8,
            args,
        };
    }
    genome
}
fn identity(genome: &Genome, output: usize, tools: &[Inherited]) -> u64 {
    let mut b = vec![genome.arity as u8, output as u8];
    for gene in &genome.genes {
        b.push(gene.function);
        b.extend(gene.args);
    }
    for tool in tools {
        b.extend(tool.identity.to_le_bytes());
        library::write_program(&mut b, &tool.program);
    }
    checksum(&b)
}
fn select<T: std::borrow::Borrow<Individual>>(pop: &[T], rows: usize, rng: &mut Rng) -> usize {
    if rng.pick(2) == 0 {
        let mut best = rng.pick(pop.len());
        for _ in 0..3 {
            let i = rng.pick(pop.len());
            if pop[i]
                .borrow()
                .fitness
                .better(&pop[best].borrow().fitness, rows)
            {
                best = i;
            }
        }
        return best;
    }
    let mut cases: Vec<_> = (0..rows).collect();
    for i in (1..cases.len()).rev() {
        let j = rng.pick(i + 1);
        cases.swap(i, j);
    }
    let mut survivors: Vec<_> = (0..pop.len()).collect();
    for case in cases {
        let minimum = survivors
            .iter()
            .map(|i| pop[*i].borrow().fitness.errors[case])
            .fold(f64::INFINITY, f64::min);
        let mut errors: Vec<_> = survivors
            .iter()
            .map(|i| pop[*i].borrow().fitness.errors[case])
            .collect();
        errors.sort_by(f64::total_cmp);
        let median = errors[errors.len() / 2];
        for e in &mut errors {
            *e = (*e - median).abs();
        }
        errors.sort_by(f64::total_cmp);
        let epsilon = errors[errors.len() / 2].max(1e-12);
        survivors.retain(|i| pop[*i].borrow().fitness.errors[case] <= minimum + epsilon);
        if survivors.len() <= 1 {
            break;
        }
    }
    survivors[rng.pick(survivors.len())]
}
fn compile(
    genome: &Genome,
    output: usize,
    tools: &[Inherited],
) -> Result<(NumericProgram, Vec<u64>), NumericError> {
    fn emit(
        g: &Genome,
        signal: usize,
        tools: &[Inherited],
        ops: &mut Vec<Op>,
        dependencies: &mut Vec<u64>,
    ) -> Result<(), NumericError> {
        if ops.len() >= MAX_OPS {
            return Err(NumericError::Invalid("evolved expansion limit"));
        }
        if signal < g.arity {
            ops.push(Op::Input(signal as u8));
        } else if signal < base(g.arity) {
            ops.push(Op::Constant([-1.0, 0.0, 1.0, 2.0][signal - g.arity]));
        } else {
            let gene = g.genes[signal - base(g.arity)];
            match gene.function {
                0..=4 => {
                    for a in &gene.args[..arguments(gene, tools)] {
                        emit(g, *a as usize, tools, ops, dependencies)?;
                    }
                    ops.push(
                        [Op::Add, Op::Subtract, Op::Multiply, Op::Divide, Op::Sqrt]
                            [gene.function as usize],
                    );
                }
                tool => {
                    let tool = &tools[tool as usize - 5];
                    if !dependencies.contains(&tool.identity) {
                        dependencies.push(tool.identity);
                    }
                    for op in &tool.program.ops {
                        if let Op::Input(i) = op {
                            emit(g, gene.args[*i as usize] as usize, tools, ops, dependencies)?;
                        } else {
                            ops.push(*op);
                        }
                        if ops.len() > MAX_OPS {
                            return Err(NumericError::Invalid("evolved expansion limit"));
                        }
                    }
                }
            }
        }
        if ops.len() > MAX_OPS {
            return Err(NumericError::Invalid("evolved expansion limit"));
        }
        Ok(())
    }
    let mut ops = Vec::new();
    let mut dependencies = Vec::new();
    emit(genome, output, tools, &mut ops, &mut dependencies)?;
    let program = NumericProgram {
        arity: genome.arity,
        ops,
    };
    program.validate()?;
    Ok((program, dependencies))
}

impl EvolvedSkill {
    #[allow(clippy::too_many_arguments)]
    pub fn evolve(
        name: &str,
        context: &str,
        training: Vec<NumericExample>,
        admission: Vec<NumericExample>,
        inherited: &[&EvolvedSkill],
        seed: u64,
        budget: EvolutionBudget,
        tolerance: NumericTolerance,
    ) -> Result<EvolutionRun, NumericError> {
        Self::evolve_observed(
            name,
            context,
            training,
            admission,
            inherited,
            seed,
            budget,
            tolerance,
            |_| {},
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn evolve_observed(
        name: &str,
        context: &str,
        training: Vec<NumericExample>,
        admission: Vec<NumericExample>,
        inherited: &[&EvolvedSkill],
        seed: u64,
        budget: EvolutionBudget,
        tolerance: NumericTolerance,
        mut observe: impl FnMut(EvolutionProgress),
    ) -> Result<EvolutionRun, NumericError> {
        identifier(name)?;
        identifier(context)?;
        budget.validate()?;
        tolerance.validate()?;
        let arity = validate_examples(&training, tolerance)?;
        if validate_examples(&admission, tolerance)? != arity || inherited.len() > 8 {
            return Err(NumericError::Invalid("evolution schema/library"));
        }
        if admission
            .iter()
            .any(|a| training.iter().any(|t| a.input == t.input))
        {
            return Err(NumericError::Invalid("admission overlaps training"));
        }
        let tools: Vec<_> = inherited
            .iter()
            .map(|s| Inherited {
                identity: s.fingerprint(),
                program: s.program.clone(),
            })
            .collect();
        let mut rng = Rng::new(seed);
        let started = Instant::now();
        let mut stats = EvolutionStatistics {
            best_loss: 1e6,
            ..EvolutionStatistics::default()
        };
        let mut trace = Vec::new();
        let mut population = Vec::<Individual>::new();
        let mut stop = EvolutionStop::GenerationLimit;
        let mut winner = None;
        let mut refutation = None;
        let mut allocation = SearchAllocation::default();
        let refinement_quota = budget.max_work * u64::from(budget.refinement_share_percent) / 100;
        let keep_partial =
            budget.refinement_slice_work > 0 || budget.refinement_share_percent < 100;
        'generations: for generation in 0..=budget.generations {
            stats.generation = generation;
            let mut next = Vec::with_capacity(budget.population);
            if !population.is_empty() {
                let best = (0..population.len())
                    .min_by(|a, b| {
                        if population[*a]
                            .fitness
                            .better(&population[*b].fitness, training.len())
                        {
                            std::cmp::Ordering::Less
                        } else if population[*b]
                            .fitness
                            .better(&population[*a].fitness, training.len())
                        {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .expect("nonempty");
                next.push(Individual {
                    genome: population[best].genome.clone(),
                    fitness: population[best].fitness.clone(),
                });
            }
            while next.len() < budget.population {
                let genome = if population.is_empty() || rng.pick(10) == 0 {
                    stats.fresh += 1;
                    fresh(&mut rng, arity, budget.genes, &tools)
                } else {
                    let a = select(&population, training.len(), &mut rng);
                    let parent = &population[a];
                    let mut g = parent.genome.clone();
                    g.parents = [identity(&g, parent.fitness.output, &tools), 0];
                    if rng.pick(4) == 0 {
                        let b = select(&population, training.len(), &mut rng);
                        g.parents[1] =
                            identity(&population[b].genome, population[b].fitness.output, &tools);
                        let cut = rng.pick(g.genes.len());
                        g.genes[cut..].copy_from_slice(&population[b].genome.genes[cut..]);
                        stats.crossovers += 1;
                    }
                    let mask = active(&g, parent.fitness.output, &tools);
                    let active: Vec<_> = mask[..g.genes.len()]
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| **v)
                        .map(|(i, _)| i)
                        .collect();
                    for _ in 0..1 + rng.pick(budget.mutations) {
                        let i = if !active.is_empty() && rng.pick(2) == 0 {
                            active[rng.pick(active.len())]
                        } else {
                            rng.pick(g.genes.len())
                        };
                        if rng.pick(3) == 0 {
                            g.genes[i] = gene(&mut rng, base(arity) + i, base(arity), tools.len());
                        } else {
                            let slot = rng.pick(4);
                            g.genes[i].args[slot] =
                                reference(&mut rng, base(arity) + i, base(arity));
                        }
                        stats.mutations += 1;
                    }
                    g
                };
                let Some(mut fitness) = evaluate(
                    &genome,
                    &tools,
                    &training,
                    tolerance,
                    &mut stats,
                    budget.max_work,
                ) else {
                    stop = EvolutionStop::WorkLimit;
                    break 'generations;
                };
                let mut genome = genome;
                if budget.local_refinement
                    && fitness.correct != training.len()
                    && next.len() % 16 == 0
                    && allocation.refinement_work < refinement_quota
                {
                    let before = stats.work;
                    let available = (refinement_quota - allocation.refinement_work)
                        .min(budget.max_work - stats.work);
                    let slice = if budget.refinement_slice_work == 0 {
                        available
                    } else {
                        available.min(budget.refinement_slice_work)
                    };
                    let local_limit = stats.work.saturating_add(slice);
                    let refined = refine(
                        &genome,
                        &fitness,
                        &tools,
                        &training,
                        tolerance,
                        &mut stats,
                        local_limit,
                        keep_partial,
                        budget.refine_inherited_calls,
                    );
                    allocation.refinements += 1;
                    allocation.refinement_work += stats.work - before;
                    if let Some((g, f, truncated)) = refined {
                        allocation.truncated_refinements += u64::from(truncated);
                        genome = g;
                        fitness = f;
                    } else if keep_partial {
                        allocation.truncated_refinements += 1;
                    } else {
                        stop = EvolutionStop::WorkLimit;
                        break 'generations;
                    }
                }
                stats.best_loss = stats.best_loss.min(fitness.loss);
                if stats.evaluations.is_multiple_of(16) || stats.evaluations == 1 {
                    observe(EvolutionProgress {
                        phase: EvolutionPhase::Search,
                        statistics: stats,
                        allocation,
                        correct: fitness.correct,
                        active_genes: fitness.active,
                        distinct_genomes: 0,
                        stop: None,
                    });
                }
                if fitness.correct == training.len() {
                    if let Ok((program, dependencies)) = compile(&genome, fitness.output, &tools) {
                        let required =
                            (program.instructions() * (training.len() + admission.len())) as u64;
                        if !charge(&mut stats, budget.max_work, required) {
                            stop = EvolutionStop::WorkLimit;
                            break 'generations;
                        }
                        stats.admission_work += required;
                        // Check the deployed interpreter, not just graph fitness. Admission labels never guide mutation.
                        let train_matches = training.iter().all(|e| {
                            program
                                .evaluate(&e.input)
                                .is_ok_and(|v| tolerance.agrees(v, e.output))
                        });
                        if !train_matches {
                            stats.unexportable += 1;
                        } else {
                            let admitted = admission.iter().all(|e| {
                                program
                                    .evaluate(&e.input)
                                    .is_ok_and(|v| tolerance.agrees(v, e.output))
                            });
                            stop = if admitted {
                                EvolutionStop::Admitted
                            } else {
                                EvolutionStop::Refuted
                            };
                            if !admitted {
                                refutation = admission.iter().find_map(|e| {
                                    let actual = program.evaluate(&e.input).ok();
                                    if actual.is_some_and(|v| tolerance.agrees(v, e.output)) {
                                        None
                                    } else {
                                        Some(EvolutionRefutation {
                                            input: e.input.clone(),
                                            expected: e.output,
                                            actual,
                                            expression: program.expression(),
                                        })
                                    }
                                });
                            }
                            stats.elapsed_ns =
                                started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                            let record = record(generation, &genome, &fitness, &tools, &stats, 1);
                            trace.push(record);
                            if admitted {
                                winner = Some(Self {
                                    name: name.into(),
                                    context: context.into(),
                                    genome,
                                    output: fitness.output,
                                    tools,
                                    program,
                                    dependencies,
                                    training,
                                    admission,
                                    tolerance,
                                    statistics: stats,
                                });
                            }
                            break 'generations;
                        }
                    } else {
                        stats.unexportable += 1;
                    }
                }
                next.push(Individual { genome, fitness });
            }
            let best = next
                .iter()
                .reduce(|a, b| {
                    if b.fitness.better(&a.fitness, training.len()) {
                        b
                    } else {
                        a
                    }
                })
                .expect("population");
            let distinct = next
                .iter()
                .map(|i| identity(&i.genome, i.fitness.output, &tools))
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            trace.push(record(
                generation,
                &best.genome,
                &best.fitness,
                &tools,
                &stats,
                distinct,
            ));
            observe(EvolutionProgress {
                phase: EvolutionPhase::Generation,
                statistics: stats,
                allocation,
                correct: best.fitness.correct,
                active_genes: best.fitness.active,
                distinct_genomes: distinct,
                stop: None,
            });
            population = next;
        }
        stats.elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        if let Some(skill) = &mut winner {
            skill.statistics = stats;
        }
        observe(EvolutionProgress {
            phase: EvolutionPhase::Finished,
            statistics: stats,
            allocation,
            correct: trace.last().map_or(0, |g| g.correct),
            active_genes: trace.last().map_or(0, |g| g.active_genes),
            distinct_genomes: trace.last().map_or(0, |g| g.distinct_genomes),
            stop: Some(stop),
        });
        Ok(EvolutionRun {
            skill: winner,
            stop,
            statistics: stats,
            generations: trace,
            refutation,
            allocation,
        })
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn program(&self) -> &NumericProgram {
        &self.program
    }
    /// Owned payload accounting, excluding allocator overhead and process RSS.
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.name.capacity()
            + self.context.capacity()
            + self.genome.genes.capacity() * std::mem::size_of::<Gene>()
            + self.tools.capacity() * std::mem::size_of::<Inherited>()
            + self
                .tools
                .iter()
                .map(|t| t.program.ops.capacity() * std::mem::size_of::<Op>())
                .sum::<usize>()
            + self.program.ops.capacity() * std::mem::size_of::<Op>()
            + self.dependencies.capacity() * std::mem::size_of::<u64>()
            + [&self.training, &self.admission]
                .iter()
                .map(|rows| {
                    rows.capacity() * std::mem::size_of::<NumericExample>()
                        + rows
                            .iter()
                            .map(|e| e.input.capacity() * std::mem::size_of::<f64>())
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
    pub fn dependencies(&self) -> &[u64] {
        &self.dependencies
    }
    /// Partial adapters only supply evaluations, work, admission work and best loss.
    /// Check `has_complete_search_statistics` before interpreting other counters.
    pub fn statistics(&self) -> EvolutionStatistics {
        self.statistics
    }
    /// Generational runs always record at least one fresh proposal. Adapters
    /// with zero fresh proposals retain partial legacy statistics, not measured zeros.
    pub fn has_complete_search_statistics(&self) -> bool {
        self.statistics.fresh > 0
    }
    pub fn genome_identity(&self) -> u64 {
        identity(&self.genome, self.output, &self.tools)
    }
    pub fn active_genes(&self) -> usize {
        active(&self.genome, self.output, &self.tools)
            .into_iter()
            .filter(|a| *a)
            .count()
    }
    /// Seven values per gene: function, arity, four argument signal IDs, active flag.
    pub fn wiring(&self) -> (usize, usize, Vec<u64>) {
        let needed = active(&self.genome, self.output, &self.tools);
        let mut flat = Vec::with_capacity(self.genome.genes.len() * 7);
        for (i, g) in self.genome.genes.iter().enumerate() {
            flat.extend([
                u64::from(g.function),
                arguments(*g, &self.tools) as u64,
                u64::from(g.args[0]),
                u64::from(g.args[1]),
                u64::from(g.args[2]),
                u64::from(g.args[3]),
                u64::from(needed[i]),
            ]);
        }
        (self.genome.arity, self.output, flat)
    }
    pub fn predict(&self, context: &str, input: &[f64]) -> Result<f64, NumericError> {
        if context != self.context {
            return Err(NumericError::Context);
        }
        self.program.evaluate(input)
    }
    pub fn fingerprint(&self) -> u64 {
        let mut bytes = Vec::new();
        for s in [&self.name, &self.context] {
            bytes.extend((s.len() as u32).to_le_bytes());
            bytes.extend(s.as_bytes());
        }
        bytes.extend(self.genome_identity().to_le_bytes());
        bytes.extend(self.tolerance.absolute.to_le_bytes());
        bytes.extend(self.tolerance.relative.to_le_bytes());
        for rows in [&self.training, &self.admission] {
            bytes.extend((rows.len() as u32).to_le_bytes());
            for row in rows {
                for v in &row.input {
                    bytes.extend(v.to_le_bytes());
                }
                bytes.extend(row.output.to_le_bytes());
            }
        }
        checksum(&bytes)
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        for s in [&self.name, &self.context] {
            b.extend((s.len() as u32).to_le_bytes());
            b.extend(s.as_bytes());
        }
        b.extend(self.tolerance.absolute.to_le_bytes());
        b.extend(self.tolerance.relative.to_le_bytes());
        b.extend([
            self.genome.arity as u8,
            self.genome.genes.len() as u8,
            self.output as u8,
            self.tools.len() as u8,
        ]);
        for g in &self.genome.genes {
            b.push(g.function);
            b.extend(g.args);
        }
        for p in self.genome.parents {
            b.extend(p.to_le_bytes());
        }
        for t in &self.tools {
            b.extend(t.identity.to_le_bytes());
            library::write_program(&mut b, &t.program);
        }
        for rows in [&self.training, &self.admission] {
            b.extend((rows.len() as u32).to_le_bytes());
            for row in rows {
                for n in &row.input {
                    b.extend(n.to_le_bytes());
                }
                b.extend(row.output.to_le_bytes());
            }
        }
        let s = self.statistics;
        for n in [
            s.evaluations,
            s.work,
            s.generation as u64,
            s.mutations,
            s.crossovers,
            s.fresh,
            s.invalid_nodes,
            s.unexportable,
            s.admission_work,
            s.elapsed_ns,
        ] {
            b.extend(n.to_le_bytes());
        }
        b.extend(s.best_loss.to_le_bytes());
        b.extend(s.local_proposals.to_le_bytes());
        b.extend(checksum(&b).to_le_bytes());
        b
    }
    pub fn save_new(&self, path: &Path) -> Result<(), NumericError> {
        publish_new(path, &self.to_bytes()).map_err(Into::into)
    }
    pub fn open(path: &Path, context: &str) -> Result<Self, NumericError> {
        let s = Self::from_bytes(&read_bounded(path, FILE_LIMIT)?)?;
        if s.context != context {
            return Err(NumericError::Context);
        }
        Ok(s)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NumericError> {
        if bytes.len() < 16
            || bytes.len() as u64 > FILE_LIMIT
            || (&bytes[..8] != MAGIC && &bytes[..8] != b"AXEV0001")
        {
            return Err(NumericError::Invalid("evolved capsule header"));
        }
        let end = bytes.len() - 8;
        if checksum(&bytes[..end]) != u64::from_le_bytes(bytes[end..].try_into().expect("eight")) {
            return Err(NumericError::Invalid("evolved checksum"));
        }
        let mut r = Reader(&bytes[8..end]);
        let name = r.text()?;
        let context = r.text()?;
        identifier(&name)?;
        identifier(&context)?;
        let tolerance = NumericTolerance {
            absolute: r.number()?,
            relative: r.number()?,
        };
        tolerance.validate()?;
        let arity = r.byte()? as usize;
        let count = r.byte()? as usize;
        let output = r.byte()? as usize;
        let tool_count = r.byte()? as usize;
        if !(1..=4).contains(&arity)
            || !(2..=MAX_GENES).contains(&count)
            || output >= base(arity) + count
            || tool_count > 8
        {
            return Err(NumericError::Invalid("evolved genome schema"));
        }
        let mut genes = Vec::new();
        for i in 0..count {
            let function = r.byte()?;
            let mut args = [0; 4];
            for a in &mut args {
                *a = r.byte()?;
            }
            if function as usize >= 5 + tool_count
                || args.iter().any(|a| *a as usize >= base(arity) + i)
            {
                return Err(NumericError::Invalid("evolved forward reference/function"));
            }
            genes.push(Gene { function, args });
        }
        let parents = [r.u64()?, r.u64()?];
        let mut tools = Vec::new();
        for _ in 0..tool_count {
            tools.push(Inherited {
                identity: r.u64()?,
                program: library::read_program(&mut r)?,
            });
        }
        let genome = Genome {
            arity,
            genes,
            parents,
        };
        let mut evidence = Vec::new();
        for _ in 0..2 {
            let mut rows = Vec::new();
            for _ in 0..r.count(32)? {
                let mut input = Vec::new();
                for _ in 0..arity {
                    input.push(r.number()?);
                }
                rows.push(NumericExample::new(input, r.number()?));
            }
            if validate_examples(&rows, tolerance)? != arity {
                return Err(NumericError::Invalid("evolved evidence"));
            }
            evidence.push(rows);
        }
        let mut statistics = EvolutionStatistics {
            evaluations: r.u64()?,
            work: r.u64()?,
            generation: usize::try_from(r.u64()?)
                .map_err(|_| NumericError::Invalid("evolved generation width"))?,
            mutations: r.u64()?,
            crossovers: r.u64()?,
            fresh: r.u64()?,
            invalid_nodes: r.u64()?,
            unexportable: r.u64()?,
            admission_work: r.u64()?,
            elapsed_ns: r.u64()?,
            best_loss: r.number()?,
            local_proposals: 0,
        };
        if &bytes[..8] == MAGIC {
            statistics.local_proposals = r.u64()?;
        }
        if !r.0.is_empty()
            || statistics.evaluations == 0
            || statistics.work == 0
            || statistics.work > 200_000_000
            || statistics.generation > 10_000
            || statistics.admission_work > statistics.work
            || statistics.best_loss < 0.0
            || statistics.best_loss > 1e6
        {
            return Err(NumericError::Invalid("evolved statistics/trailing data"));
        }
        if evidence[1]
            .iter()
            .any(|a| evidence[0].iter().any(|t| a.input == t.input))
        {
            return Err(NumericError::Invalid("evolved overlapping admission"));
        }
        let (program, dependencies) = compile(&genome, output, &tools)?;
        for e in evidence.iter().flatten() {
            if !tolerance.agrees(program.evaluate(&e.input)?, e.output) {
                return Err(NumericError::Invalid("evolved evidence replay"));
            }
        }
        let admission = evidence.pop().expect("two");
        let training = evidence.pop().expect("two");
        Ok(Self {
            name,
            context,
            genome,
            output,
            tools,
            program,
            dependencies,
            training,
            admission,
            tolerance,
            statistics,
        })
    }
}
fn record(
    generation: usize,
    genome: &Genome,
    fitness: &Fitness,
    tools: &[Inherited],
    stats: &EvolutionStatistics,
    distinct_genomes: usize,
) -> GenerationRecord {
    GenerationRecord {
        generation,
        evaluations: stats.evaluations,
        work: stats.work,
        loss: fitness.loss,
        correct: fitness.correct,
        active_genes: fitness.active,
        distinct_genomes,
        genome: identity(genome, fitness.output, tools),
        parents: genome.parents,
    }
}
