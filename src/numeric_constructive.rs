//! Persistent graph evolution with meaningful growth, using the existing numerical VM.
//! Exact genotype uniqueness is not equivalence of the functions computed by two graphs.
use super::*;
use crate::{EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue};
use std::collections::{BTreeMap, BTreeSet};

pub const CONTEXT: &str = "axon/constructive-v1";
const DNA_MAGIC: &[u8; 8] = b"AXCG0001";

/// A self-contained executable circuit: no examples, fitness, RNG or personal episodes.
#[derive(Clone, Debug)]
pub struct CircuitDna {
    genome: Genome,
    output: usize,
    tools: Vec<Inherited>,
}
fn graph_bytes(g: &Genome) -> Vec<u8> {
    let mut bytes = vec![g.arity as u8, g.genes.len() as u8];
    for gene in &g.genes {
        bytes.push(gene.function);
        bytes.extend(gene.args);
    }
    bytes
}
fn normalize(g: &mut Genome, tools: &[Inherited]) {
    for gene in &mut g.genes {
        let n = arguments(*gene, tools);
        gene.args[n..].fill(0);
        if matches!(gene.function, 0 | 2) && gene.args[0] > gene.args[1] {
            gene.args.swap(0, 1);
        }
    }
}
fn read_graph(r: &mut Reader<'_>, tools: &[Inherited]) -> Result<Genome, NumericError> {
    let arity = r.take(1)?[0] as usize;
    let count = r.take(1)?[0] as usize;
    if !(1..=4).contains(&arity) || !(1..=MAX_GENES).contains(&count) {
        return Err(NumericError::Invalid("circuit dimensions"));
    }
    let mut genes = Vec::with_capacity(count);
    for i in 0..count {
        let function = r.take(1)?[0];
        let args: [u8; 4] = r.take(4)?.try_into().expect("four");
        if function as usize >= 5 + tools.len() {
            return Err(NumericError::Invalid("circuit operation"));
        }
        let gene = Gene { function, args };
        let n = arguments(gene, tools);
        if args[..n].iter().any(|a| *a as usize >= base(arity) + i) {
            return Err(NumericError::Invalid("circuit forward/cyclic reference"));
        }
        genes.push(gene);
    }
    let mut graph = Genome {
        arity,
        genes,
        parents: [0; 2],
    };
    let original = graph_bytes(&graph);
    normalize(&mut graph, tools);
    if graph_bytes(&graph) != original {
        return Err(NumericError::Invalid("noncanonical circuit"));
    }
    Ok(graph)
}
fn decimal(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut result = String::from("1");
    for byte in bytes {
        write!(result, "{byte:03}").expect("string");
    }
    result
}
fn undecimal(s: &str) -> Result<Vec<u8>, NumericError> {
    if s.len() > 3 * FILE_LIMIT as usize + 1
        || s.len() < 4
        || s.len() % 3 != 1
        || !s.starts_with('1')
        || !s.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(NumericError::Invalid("circuit decimal encoding"));
    }
    s.as_bytes()[1..]
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| {
            let n =
                u16::from(c[0] - b'0') * 100 + u16::from(c[1] - b'0') * 10 + u16::from(c[2] - b'0');
            u8::try_from(n).map_err(|_| NumericError::Invalid("circuit decimal byte"))
        })
        .collect()
}
impl CircuitDna {
    /// Export an acquired program, not the foundry's examples or private metadata.
    /// This does not certify privacy of constants encoded by the learned program.
    pub fn from_program(program: &NumericProgram) -> Result<Self, NumericError> {
        program.validate()?;
        let arity = program.arity();
        let mut args = [0; 4];
        for (i, arg) in args.iter_mut().enumerate().take(arity) {
            *arg = i as u8;
        }
        let dna = Self {
            genome: Genome {
                arity,
                genes: vec![Gene { function: 5, args }],
                parents: [0; 2],
            },
            output: base(arity),
            tools: vec![Inherited {
                identity: 0,
                program: program.clone(),
            }],
        };
        dna.program()?;
        Ok(dna)
    }
    pub fn from_skill(skill: &EvolvedSkill) -> Self {
        let mut genome = skill.genome.clone();
        normalize(&mut genome, &skill.tools);
        Self {
            genome,
            output: skill.output,
            tools: skill.tools.clone(),
        }
    }
    pub fn program(&self) -> Result<NumericProgram, NumericError> {
        compile(&self.genome, self.output, &self.tools).map(|p| p.0)
    }
    pub fn genes(&self) -> usize {
        self.genome.genes.len()
    }
    pub fn number(&self) -> String {
        decimal(&self.to_bytes())
    }
    pub fn from_number(number: &str) -> Result<Self, NumericError> {
        Self::from_bytes(&undecimal(number)?)
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = DNA_MAGIC.to_vec();
        bytes.push(self.tools.len() as u8);
        for tool in &self.tools {
            library::write_program(&mut bytes, &tool.program);
        }
        bytes.extend(graph_bytes(&self.genome));
        bytes.push(self.output as u8);
        bytes.extend(checksum(&bytes).to_le_bytes());
        bytes
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NumericError> {
        if bytes.len() < 20 || bytes.len() > FILE_LIMIT as usize || &bytes[..8] != DNA_MAGIC {
            return Err(NumericError::Invalid("circuit header"));
        }
        let end = bytes.len() - 8;
        if checksum(&bytes[..end]) != u64::from_le_bytes(bytes[end..].try_into().expect("eight")) {
            return Err(NumericError::Invalid("circuit checksum"));
        }
        let mut r = Reader(&bytes[8..end]);
        let count = r.take(1)?[0] as usize;
        if count > 8 {
            return Err(NumericError::Invalid("circuit inherited tools"));
        }
        let mut tools = Vec::new();
        for _ in 0..count {
            let program = library::read_program(&mut r)?;
            tools.push(Inherited {
                identity: 0,
                program,
            });
        }
        let genome = read_graph(&mut r, &tools)?;
        let output = r.take(1)?[0] as usize;
        if !r.0.is_empty() || output >= base(genome.arity) + genome.genes.len() {
            return Err(NumericError::Invalid("circuit output/trailing bytes"));
        }
        let dna = Self {
            genome,
            output,
            tools,
        };
        dna.program()?;
        Ok(dna)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstructiveBudget {
    pub population: usize,
    pub initial_genes: usize,
    pub max_genes: usize,
    pub patience: usize,
    pub max_candidates: usize,
    pub max_work: u64,
}
impl Default for ConstructiveBudget {
    fn default() -> Self {
        Self {
            population: 32,
            initial_genes: 2,
            max_genes: 16,
            patience: 3,
            max_candidates: 20000,
            max_work: 100_000_000,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConstructivePolicy {
    #[default]
    Elitist,
    Diverse,
    Reinforced,
}
impl ConstructivePolicy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Elitist => "elitist",
            Self::Diverse => "diverse",
            Self::Reinforced => "reinforced",
        }
    }
}
/// Search-level operator feedback, not an individual's general learning policy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConstructiveFeedback {
    pub trials: [u64; 4],
    pub rewards: [f64; 4],
}
impl ConstructiveFeedback {
    fn choose(&self) -> usize {
        if let Some(i) = self.trials.iter().position(|n| *n == 0) {
            return i;
        }
        let total = self.trials.iter().sum::<u64>() as f64;
        (0..4)
            .max_by(|a, b| {
                let value = |i: usize| {
                    self.rewards[i] / self.trials[i] as f64
                        + (2.0 * total.ln() / self.trials[i] as f64).sqrt()
                };
                value(*a).total_cmp(&value(*b))
            })
            .expect("four operators")
    }
    fn observe(&mut self, action: usize, reward: f64) {
        self.trials[action] += 1;
        self.rewards[action] += reward;
    }
}
fn composition(
    rng: &mut Rng,
    arity: usize,
    genes: usize,
    tools: &[Inherited],
    mut cursor: usize,
) -> Genome {
    let functions: Vec<_> = (0..5 + tools.len())
        .filter(|f| {
            arguments(
                Gene {
                    function: *f as u8,
                    args: [0; 4],
                },
                tools,
            ) <= 2
        })
        .collect();
    let mut choices = [0; 3];
    for f in &mut choices {
        *f = functions[cursor % functions.len()] as u8;
        cursor /= functions.len();
    }
    let wrap = cursor % 2 == 1;
    cursor /= 2;
    let mut input = [0; 4];
    if arity == 4 {
        let mut available: Vec<_> = (0..4).collect();
        for slot in &mut input {
            let n = available.len();
            *slot = available.remove(cursor % n);
            cursor /= n;
        }
    } else {
        for slot in &mut input {
            *slot = cursor % arity;
            cursor /= arity;
        }
    }
    let b = base(arity) as u8;
    let mut graph = Genome {
        arity,
        genes: (0..genes)
            .map(|i| gene(rng, base(arity) + i, base(arity), tools.len()))
            .collect(),
        parents: [0; 2],
    };
    graph.genes[..4].copy_from_slice(&[
        Gene {
            function: choices[0],
            args: [input[0] as u8, input[1] as u8, 0, 0],
        },
        Gene {
            function: choices[1],
            args: [input[2] as u8, input[3] as u8, 0, 0],
        },
        Gene {
            function: choices[2],
            args: [b, b + 1, 0, 0],
        },
        Gene {
            function: if wrap { 4 } else { 0 },
            args: [b + 2, (arity + 1) as u8, 0, 0],
        },
    ]);
    graph
}
#[derive(Clone, Copy, Debug)]
pub struct ConstructiveProgress {
    pub completed: usize,
    pub seen: usize,
    pub genes: usize,
    pub growths: usize,
    pub duplicate_proposals: usize,
    pub correct: usize,
    pub loss: f64,
    pub work: u64,
}
#[derive(Debug)]
pub enum ConstructiveError {
    Numeric(NumericError),
    Storage(crate::EpisodeError),
    Invalid(&'static str),
}
impl std::fmt::Display for ConstructiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Numeric(e) => e.fmt(f),
            Self::Storage(e) => e.fmt(f),
            Self::Invalid(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for ConstructiveError {}
impl From<NumericError> for ConstructiveError {
    fn from(e: NumericError) -> Self {
        Self::Numeric(e)
    }
}
impl From<crate::EpisodeError> for ConstructiveError {
    fn from(e: crate::EpisodeError) -> Self {
        Self::Storage(e)
    }
}
type Result<T, E = ConstructiveError> = std::result::Result<T, E>;
fn text(v: impl ToString) -> EpisodeValue {
    EpisodeValue::Text(v.to_string())
}
fn field<'a>(fields: &'a BTreeMap<String, EpisodeValue>, key: &str) -> Result<&'a str> {
    match fields.get(key) {
        Some(EpisodeValue::Text(v)) => Ok(v),
        _ => Err(ConstructiveError::Invalid("constructive journal field")),
    }
}

/// Reservations precede fitness evaluation. Interrupted reservations remain quarantined.
pub struct ConstructiveSearch {
    store: EpisodeStore,
    budget: ConstructiveBudget,
    training: Vec<NumericExample>,
    tolerance: NumericTolerance,
    tools: Vec<Inherited>,
    seen: BTreeSet<Vec<u8>>,
    records: Vec<Individual>,
    completed_keys: BTreeSet<Vec<u8>>,
    front: Vec<usize>,
    best_index: Option<usize>,
    genes: usize,
    stale: usize,
    growths: usize,
    work: u64,
    rng: Rng,
    duplicates: usize,
    policy: ConstructivePolicy,
    feedback: ConstructiveFeedback,
    composition_cursor: usize,
}
impl ConstructiveSearch {
    pub fn open(
        path: &Path,
        training: Vec<NumericExample>,
        inherited: &[&EvolvedSkill],
        budget: ConstructiveBudget,
        tolerance: NumericTolerance,
        rng: u64,
    ) -> Result<Self> {
        Self::open_with_policy(
            path,
            training,
            inherited,
            budget,
            tolerance,
            rng,
            ConstructivePolicy::Elitist,
        )
    }
    pub fn open_with_policy(
        path: &Path,
        training: Vec<NumericExample>,
        inherited: &[&EvolvedSkill],
        budget: ConstructiveBudget,
        tolerance: NumericTolerance,
        rng: u64,
        policy: ConstructivePolicy,
    ) -> Result<Self> {
        tolerance.validate()?;
        let arity = validate_examples(&training, tolerance)?;
        if !(4..=128).contains(&budget.population)
            || budget.initial_genes < 2
            || budget.initial_genes > budget.max_genes
            || budget.max_genes > MAX_GENES
            || !(1..=100).contains(&budget.patience)
            || !(1..=40000).contains(&budget.max_candidates)
            || !(1..=200_000_000).contains(&budget.max_work)
            || inherited.len() > 8
        {
            return Err(ConstructiveError::Invalid("constructive budget"));
        }
        let mut tools: Vec<_> = inherited
            .iter()
            .map(|s| Inherited {
                identity: s.fingerprint(),
                program: s.program.clone(),
            })
            .collect();
        if policy != ConstructivePolicy::Elitist {
            tools.retain(|t| {
                !matches!(
                    t.program.ops.as_slice(),
                    [
                        Op::Input(0),
                        Op::Input(1),
                        Op::Add | Op::Subtract | Op::Multiply | Op::Divide
                    ]
                )
            });
        }
        // Exact scope bytes, not a hash-only acceptance test. RNG is not genotype identity.
        let mut scope =
            format!("exportable-judge-v3;{budget:?};{tolerance:?};arity={arity};").into_bytes();
        if policy != ConstructivePolicy::Elitist {
            scope.extend(format!("case-portfolio-v3;policy={};", policy.name()).bytes());
        }
        for row in &training {
            for n in row.input.iter().chain([&row.output]) {
                scope.extend(n.to_le_bytes());
            }
        }
        for tool in &tools {
            library::write_program(&mut scope, &tool.program);
        }
        let scope = decimal(&scope);
        if scope.len() > 28000 {
            return Err(ConstructiveError::Invalid("constructive scope size"));
        }
        let (mut store, _) = EpisodeStore::open(path)?;
        let fields = scope
            .as_bytes()
            .chunks(4096)
            .enumerate()
            .map(|(i, c)| {
                (
                    format!("scope{i:02}"),
                    text(std::str::from_utf8(c).expect("ASCII")),
                )
            })
            .collect();
        let metadata = EpisodeData::Trace {
            context: CONTEXT.into(),
            channel: "metadata".into(),
            fields,
        };
        if store.is_empty() {
            store.record(
                EpisodeStamp::new("constructive-meta", "v1", 0),
                metadata.clone(),
            )?;
        }
        if store.episode(1)?.data() != &metadata {
            return Err(ConstructiveError::Invalid(
                "constructive decoder/examples/budget mismatch",
            ));
        }
        let mut search = Self {
            store,
            budget,
            training,
            tolerance,
            tools,
            seen: BTreeSet::new(),
            records: Vec::new(),
            completed_keys: BTreeSet::new(),
            front: Vec::new(),
            best_index: None,
            genes: budget.initial_genes,
            stale: 0,
            growths: 0,
            work: 0,
            rng: Rng::new(rng),
            duplicates: 0,
            policy,
            feedback: ConstructiveFeedback::default(),
            composition_cursor: 0,
        };
        for id in 2..=search.store.len() as u64 {
            let episode = search.store.episode(id)?;
            let EpisodeData::Trace {
                context,
                channel,
                fields,
            } = episode.data()
            else {
                return Err(ConstructiveError::Invalid("foreign constructive record"));
            };
            if context != CONTEXT {
                return Err(ConstructiveError::Invalid("constructive context"));
            }
            match channel.as_str() {
                "growth" => {
                    if field(fields, "genes")? != (search.genes + 1).to_string()
                        || search.genes >= budget.max_genes
                    {
                        return Err(ConstructiveError::Invalid("constructive growth"));
                    }
                    search.genes += 1;
                    search.growths += 1;
                    search.stale = 0;
                }
                "reservation" => {
                    let bytes = undecimal(field(fields, "dna")?)?;
                    let mut r = Reader(&bytes);
                    let graph = read_graph(&mut r, &search.tools)?;
                    if !r.0.is_empty()
                        || graph.arity != arity
                        || graph.genes.len() != search.genes
                        || !search.seen.insert(bytes)
                        || search.seen.len() > budget.max_candidates
                    {
                        return Err(ConstructiveError::Invalid(
                            "duplicate/invalid constructive reservation",
                        ));
                    }
                }
                "result" => {
                    let bytes = undecimal(field(fields, "dna")?)?;
                    let data = undecimal(field(fields, "fitness")?)?;
                    let mut r = Reader(&bytes);
                    let mut graph = read_graph(&mut r, &search.tools)?;
                    let mut r = Reader(&data);
                    let mut errors = [0.0; 32];
                    for n in &mut errors {
                        *n = f64::from_le_bytes(r.take(8)?.try_into().expect("eight"));
                    }
                    let loss = f64::from_le_bytes(r.take(8)?.try_into().expect("eight"));
                    let correct = r.take(1)?[0] as usize;
                    let output = r.take(1)?[0] as usize;
                    let active = r.take(1)?[0] as usize;
                    for p in &mut graph.parents {
                        *p = u64::from_le_bytes(r.take(8)?.try_into().expect("eight"));
                    }
                    let work = u64::from_le_bytes(r.take(8)?.try_into().expect("eight"));
                    if !r.0.is_empty()
                        || !search.seen.contains(&bytes)
                        || !search.completed_keys.insert(bytes)
                        || !loss.is_finite()
                        || errors.iter().any(|e| !e.is_finite() || *e < 0.0)
                        || correct > search.training.len()
                        || output >= base(arity) + graph.genes.len()
                        || active > graph.genes.len()
                        || work > budget.max_work.saturating_sub(search.work)
                    {
                        return Err(ConstructiveError::Invalid("constructive result schema"));
                    }
                    search.work += work;
                    let record = Individual {
                        genome: graph,
                        fitness: Fitness {
                            errors,
                            loss,
                            correct,
                            output,
                            active,
                        },
                    };
                    if policy != ConstructivePolicy::Elitist {
                        let action: usize = field(fields, "action")?
                            .parse()
                            .map_err(|_| ConstructiveError::Invalid("constructive action"))?;
                        let reward: f64 = field(fields, "reward")?
                            .parse()
                            .map_err(|_| ConstructiveError::Invalid("constructive reward"))?;
                        if action >= 4
                            || !reward.is_finite()
                            || !(0.0..=1.0).contains(&reward)
                            || reward != search.reward(&record.fitness)
                        {
                            return Err(ConstructiveError::Invalid(
                                "constructive feedback mismatch",
                            ));
                        }
                        search.feedback.observe(action, reward);
                        let cursor: usize = field(fields, "cursor")?.parse().map_err(|_| {
                            ConstructiveError::Invalid("constructive composition cursor")
                        })?;
                        if cursor < search.composition_cursor
                            || cursor > 4_000_000_000
                            || (action == 3
                                && search.genes >= 4
                                && cursor == search.composition_cursor)
                        {
                            return Err(ConstructiveError::Invalid(
                                "constructive composition cursor",
                            ));
                        }
                        search.composition_cursor = cursor;
                    }
                    search.install(record);
                }
                _ => return Err(ConstructiveError::Invalid("constructive channel")),
            }
        }
        Ok(search)
    }
    fn append(&mut self, channel: &str, fields: BTreeMap<String, EpisodeValue>) -> Result<()> {
        self.store.record(
            EpisodeStamp::new(
                "constructive",
                format!("{channel}-{}", self.store.len()),
                self.store.len() as u64,
            ),
            EpisodeData::Trace {
                context: CONTEXT.into(),
                channel: channel.into(),
                fields,
            },
        )?;
        Ok(())
    }
    fn best(&self) -> Option<&Individual> {
        self.best_index.map(|i| &self.records[i])
    }
    pub fn feedback(&self) -> Option<ConstructiveFeedback> {
        (self.policy != ConstructivePolicy::Elitist).then_some(self.feedback)
    }
    fn reward(&self, fitness: &Fitness) -> f64 {
        if self.front.is_empty() {
            return 0.0;
        }
        (0..self.training.len())
            .map(|case| {
                let previous = self
                    .front
                    .iter()
                    .map(|i| self.records[*i].fitness.errors[case])
                    .fold(1e6, f64::min);
                ((previous - fitness.errors[case]) / (1.0 + previous)).max(0.0)
            })
            .sum::<f64>()
            / self.training.len() as f64
    }
    fn install(&mut self, record: Individual) {
        let better = self
            .best()
            .is_none_or(|best| record.fitness.better(&best.fitness, self.training.len()));
        // A fitting DAG may expand beyond VM limits. Its measured fitness stays
        // recorded, but only an executable child can stop the search or be born.
        let capability = self.best().is_none_or(|best| {
            record.fitness.correct > best.fitness.correct
                || record.fitness.loss < best.fitness.loss * (1.0 - 1e-9)
        });
        let eligible = (better || capability)
            && compile(&record.genome, record.fitness.output, &self.tools).is_ok();
        let improved = eligible && capability;
        self.stale = if improved { 0 } else { self.stale + 1 };
        let index = self.records.len();
        if eligible && better {
            self.best_index = Some(index);
        }
        self.records.push(record);
        if self.policy != ConstructivePolicy::Elitist {
            // Keep every per-case champion before filling with recent proposals.
            let mut pool = self.front.clone();
            pool.push(index);
            let mut front = Vec::new();
            if let Some(best) = self.best_index {
                front.push(best);
            }
            for case in 0..self.training.len() {
                let champion = *pool
                    .iter()
                    .min_by(|a, b| {
                        self.records[**a].fitness.errors[case]
                            .total_cmp(&self.records[**b].fitness.errors[case])
                            .then(
                                self.records[**a]
                                    .fitness
                                    .loss
                                    .total_cmp(&self.records[**b].fitness.loss),
                            )
                    })
                    .expect("new record");
                if !front.contains(&champion) && front.len() < self.budget.population {
                    front.push(champion);
                }
            }
            for i in pool.into_iter().rev() {
                if front.len() == self.budget.population {
                    break;
                }
                if !front.contains(&i) {
                    front.push(i);
                }
            }
            self.front = front;
            return;
        }
        self.front.push(index);
        self.front.sort_by(|a, b| {
            self.records[*a]
                .fitness
                .loss
                .total_cmp(&self.records[*b].fitness.loss)
                .then(
                    self.records[*a]
                        .fitness
                        .active
                        .cmp(&self.records[*b].fitness.active),
                )
        });
        self.front.truncate(self.budget.population);
    }
    pub fn progress(&self) -> ConstructiveProgress {
        let best = self.best();
        ConstructiveProgress {
            completed: self.records.len(),
            seen: self.seen.len(),
            genes: self.genes,
            growths: self.growths,
            duplicate_proposals: self.duplicates,
            correct: best.map_or(0, |i| i.fitness.correct),
            loss: best.map_or(1e6, |i| i.fitness.loss),
            work: self.work,
        }
    }
    pub fn snapshot_new(&mut self, path: &Path) -> Result<()> {
        self.store.snapshot_new(path)?;
        Ok(())
    }
    pub fn winner(&self) -> Option<CircuitDna> {
        let best = self.best()?;
        (best.fitness.correct == self.training.len()).then(|| CircuitDna {
            genome: best.genome.clone(),
            output: best.fitness.output,
            tools: self.tools.clone(),
        })
    }
    pub fn best_dna(&self) -> Option<CircuitDna> {
        self.best().map(|best| CircuitDna {
            genome: best.genome.clone(),
            output: best.fitness.output,
            tools: self.tools.clone(),
        })
    }
    pub fn certify(
        &self,
        name: &str,
        context: &str,
        admission: Vec<NumericExample>,
    ) -> Result<EvolvedSkill> {
        identifier(name)?;
        identifier(context)?;
        let dna = self.winner().ok_or(ConstructiveError::Invalid(
            "no constructive training winner",
        ))?;
        if validate_examples(&admission, self.tolerance)? != dna.genome.arity
            || admission
                .iter()
                .any(|a| self.training.iter().any(|t| t.input == a.input))
        {
            return Err(ConstructiveError::Invalid(
                "constructive admission schema/overlap",
            ));
        }
        let (program, dependencies) = compile(&dna.genome, dna.output, &dna.tools)?;
        let admission_work =
            (program.instructions() * (self.training.len() + admission.len())) as u64;
        if admission_work > self.budget.max_work.saturating_sub(self.work) {
            return Err(ConstructiveError::Invalid(
                "constructive admission work limit",
            ));
        }
        if !self.training.iter().chain(&admission).all(|e| {
            program
                .evaluate(&e.input)
                .is_ok_and(|v| self.tolerance.agrees(v, e.output))
        }) {
            return Err(ConstructiveError::Invalid(
                "constructive candidate refuted by admission",
            ));
        }
        Ok(EvolvedSkill {
            name: name.into(),
            context: context.into(),
            genome: dna.genome,
            output: dna.output,
            tools: dna.tools,
            program,
            dependencies,
            training: self.training.clone(),
            admission,
            tolerance: self.tolerance,
            statistics: EvolutionStatistics {
                evaluations: self.records.len() as u64,
                work: self.work + admission_work,
                admission_work,
                best_loss: 0.0,
                ..EvolutionStatistics::default()
            },
        })
    }
    /// Returns new evaluations only; cached elites are parents, never evaluated again.
    pub fn run(
        &mut self,
        limit: usize,
        mut observe: impl FnMut(ConstructiveProgress),
    ) -> Result<usize> {
        if limit == 0 || limit > 40000 {
            return Err(ConstructiveError::Invalid("constructive batch limit"));
        }
        let start = self.records.len();
        let mut attempts = 0usize;
        observe(self.progress());
        while self.records.len() - start < limit
            && self.seen.len() < self.budget.max_candidates
            && self.work < self.budget.max_work
            && self.winner().is_none()
        {
            if self.stale >= self.budget.population * self.budget.patience
                && self.genes < self.budget.max_genes
            {
                self.append(
                    "growth",
                    BTreeMap::from([("genes".into(), text(self.genes + 1))]),
                )?;
                self.genes += 1;
                self.growths += 1;
                self.stale = 0;
                observe(self.progress());
            }
            let arity = self.training[0].input.len();
            let population: Vec<_> = self.front.iter().map(|i| &self.records[*i]).collect();
            let action = if self.policy != ConstructivePolicy::Elitist
                && self.genes >= 4
                && self.records.len().is_multiple_of(2)
            {
                3
            } else if population.is_empty() {
                0
            } else if self.policy == ConstructivePolicy::Reinforced {
                (self.feedback.choose() + attempts) % 4
            } else if self.policy == ConstructivePolicy::Diverse {
                self.rng.pick(4)
            } else {
                1
            };
            let novel = self.policy != ConstructivePolicy::Elitist;
            let mut graph = if novel && action == 3 && self.genes >= 4 {
                let graph = composition(
                    &mut self.rng,
                    arity,
                    self.genes,
                    &self.tools,
                    self.composition_cursor,
                );
                self.composition_cursor += 1;
                graph
            } else if population.is_empty()
                || if novel {
                    action == 0
                } else {
                    self.rng.pick(5) == 0
                }
            {
                if novel {
                    Genome {
                        arity,
                        genes: (0..self.genes)
                            .map(|i| {
                                gene(
                                    &mut self.rng,
                                    base(arity) + i,
                                    base(arity),
                                    self.tools.len(),
                                )
                            })
                            .collect(),
                        parents: [0; 2],
                    }
                } else {
                    fresh(&mut self.rng, arity, self.genes, &self.tools)
                }
            } else {
                let parent = population[if novel {
                    select(&population, self.training.len(), &mut self.rng)
                } else {
                    self.rng.pick(population.len())
                }];
                let mut g = parent.genome.clone();
                g.parents = [identity(&g, parent.fitness.output, &self.tools), 0];
                while g.genes.len() < self.genes {
                    g.genes.push(gene(
                        &mut self.rng,
                        base(arity) + g.genes.len(),
                        base(arity),
                        self.tools.len(),
                    ));
                }
                if if novel {
                    action == 2
                } else {
                    self.rng.pick(4) == 0
                } {
                    let other = population[self.rng.pick(population.len())];
                    g.parents[1] = identity(&other.genome, other.fitness.output, &self.tools);
                    let cut = self.rng.pick(other.genome.genes.len());
                    g.genes[cut..other.genome.genes.len()]
                        .copy_from_slice(&other.genome.genes[cut..]);
                }
                for _ in 0..1 + self.rng.pick(3) {
                    let i = self.rng.pick(g.genes.len());
                    if self.rng.pick(3) == 0 {
                        g.genes[i] = gene(
                            &mut self.rng,
                            base(arity) + i,
                            base(arity),
                            self.tools.len(),
                        );
                    } else {
                        let slot = self.rng.pick(arguments(g.genes[i], &self.tools));
                        g.genes[i].args[slot] =
                            reference(&mut self.rng, base(arity) + i, base(arity));
                    }
                }
                if novel && action == 3 {
                    // Generic composition: reuse an existing output in a later operation.
                    let start = parent.fitness.output.saturating_sub(base(arity))
                        + usize::from(parent.fitness.output >= base(arity));
                    if start < g.genes.len() {
                        let i = start + self.rng.pick(g.genes.len() - start);
                        let slot = self.rng.pick(arguments(g.genes[i], &self.tools));
                        g.genes[i].args[slot] = parent.fitness.output as u8;
                    }
                }
                g
            };
            normalize(&mut graph, &self.tools);
            let key = graph_bytes(&graph);
            if self.seen.contains(&key) {
                self.duplicates += 1;
                attempts += 1;
                if attempts >= 100000 {
                    break;
                }
                continue;
            }
            attempts = 0;
            // Upper bound for this evaluation, including expanded inherited calls. Do not reserve work that cannot finish.
            let cost = self.training.len() as u64
                * (2 * (base(arity) + self.genes)
                    + self.genes
                        * self
                            .tools
                            .iter()
                            .map(|t| t.program.instructions())
                            .max()
                            .unwrap_or(0)) as u64;
            if cost > self.budget.max_work.saturating_sub(self.work) {
                break;
            }
            self.append(
                "reservation",
                BTreeMap::from([("dna".into(), text(decimal(&key)))]),
            )?;
            self.seen.insert(key.clone());
            let mut stats = EvolutionStatistics::default();
            let fitness = evaluate(
                &graph,
                &self.tools,
                &self.training,
                self.tolerance,
                &mut stats,
                cost,
            )
            .ok_or(ConstructiveError::Invalid("constructive work estimate"))?;
            let mut data = Vec::new();
            for n in fitness.errors.iter().chain([&fitness.loss]) {
                data.extend(n.to_le_bytes());
            }
            data.extend([
                fitness.correct as u8,
                fitness.output as u8,
                fitness.active as u8,
            ]);
            for p in graph.parents {
                data.extend(p.to_le_bytes());
            }
            data.extend(stats.work.to_le_bytes());
            let reward = if novel { self.reward(&fitness) } else { 0.0 };
            let mut fields = BTreeMap::from([
                ("dna".into(), text(decimal(&key))),
                ("fitness".into(), text(decimal(&data))),
            ]);
            if novel {
                fields.insert("action".into(), text(action));
                fields.insert("reward".into(), text(reward));
                fields.insert("cursor".into(), text(self.composition_cursor));
            }
            self.append("result", fields)?;
            self.work += stats.work;
            self.completed_keys.insert(key);
            if novel {
                self.feedback.observe(action, reward);
            }
            self.install(Individual {
                genome: graph,
                fitness,
            });
            if (self.records.len() - start).is_multiple_of(32) || self.winner().is_some() {
                observe(self.progress());
            }
        }
        observe(self.progress());
        Ok(self.records.len() - start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exported_circuit_preserves_four_input_float_bits_and_domain_refusals() {
        let program = NumericProgram {
            arity: 4,
            ops: vec![
                Op::Input(0),
                Op::Input(1),
                Op::Divide,
                Op::Input(2),
                Op::Sqrt,
                Op::Add,
                Op::Input(3),
                Op::Multiply,
            ],
        };
        let dna = CircuitDna::from_program(&program).unwrap();
        let raw = dna.to_bytes();
        let restored = CircuitDna::from_bytes(&raw).unwrap();
        assert_eq!(restored.to_bytes(), raw);
        assert_eq!(restored.program().unwrap(), program);
        let execution = restored.program().unwrap().compile_execution();
        for a in [-0.0, 0.0, 1.0, -1.0, 1e-300, 1e300, f64::MAX] {
            for b in [-0.0, 0.0, 2.0, -3.0, 1e-300, 1e300, f64::MIN_POSITIVE] {
                for c in [0.0, 4.0, -1.0, 1e300] {
                    for d in [0.0, 1.0, -1.0] {
                        let input = [a, b, c, d];
                        match (program.evaluate(&input), execution.evaluate(&input)) {
                            (Ok(x), Ok(y)) => assert_eq!(x.to_bits(), y.to_bits()),
                            (Err(x), Err(y)) => assert_eq!(x.to_string(), y.to_string()),
                            (a, b) => panic!("circuit changed arithmetic: {a:?} != {b:?}"),
                        }
                    }
                }
            }
        }
        for input in [
            vec![1.0],
            vec![1.0, 2.0, f64::NAN, 4.0],
            vec![1.0, 2.0, 3.0, f64::INFINITY],
        ] {
            assert!(execution.evaluate(&input).is_err());
        }
    }
    #[test]
    fn circuit_export_is_independent_of_witnesses_names_and_contexts() {
        let learn = |name: &str, offset: f64| {
            NumericSkill::learn(
                name,
                format!("private/{name}"),
                [-7.0, 0.0, 0.25, 3.0, 16.0]
                    .into_iter()
                    .map(|x| {
                        let x = x + offset;
                        NumericExample::new(vec![x], 2.0 * x + 1.0)
                    })
                    .collect(),
                &[],
                NumericBudget::default(),
                NumericTolerance::default(),
            )
            .unwrap()
        };
        let first = learn("private-person-alpha", 0.0);
        let second = learn("private-person-beta", 31.0);
        assert_ne!(first.to_bytes(), second.to_bytes());
        assert_eq!(first.program(), second.program());
        let a = CircuitDna::from_program(first.program())
            .unwrap()
            .to_bytes();
        let b = CircuitDna::from_program(second.program())
            .unwrap()
            .to_bytes();
        assert_eq!(a, b);
        assert!(a.len() < first.to_bytes().len());
        assert!(
            !a.windows("private-person".len())
                .any(|w| w == b"private-person")
        );
        let mut dna = CircuitDna::from_bytes(&a).unwrap();
        dna.genome.genes[0].args[0] = base(1) as u8;
        assert!(CircuitDna::from_bytes(&dna.to_bytes()).is_err());
        let mut dna = CircuitDna::from_bytes(&a).unwrap();
        dna.genome.genes[0].args[3] = 1;
        assert!(CircuitDna::from_bytes(&dna.to_bytes()).is_err());
        let mut raw = a;
        raw[0] ^= 1;
        assert!(CircuitDna::from_bytes(&raw).is_err());
        for program in [
            NumericProgram {
                arity: 0,
                ops: vec![Op::Constant(1.0)],
            },
            NumericProgram {
                arity: 1,
                ops: vec![Op::Input(2)],
            },
            NumericProgram {
                arity: 1,
                ops: vec![Op::Constant(f64::NAN)],
            },
        ] {
            assert!(CircuitDna::from_program(&program).is_err());
        }
    }
    #[test]
    fn operator_reinforcement_explores_and_credits_observed_improvement() {
        let mut feedback = ConstructiveFeedback::default();
        for i in 0..4 {
            assert_eq!(feedback.choose(), i);
            feedback.observe(i, f64::from(i == 2));
        }
        for _ in 0..100 {
            let i = feedback.choose();
            feedback.observe(i, f64::from(i == 2));
        }
        assert_eq!(feedback.trials.iter().sum::<u64>(), 104);
        assert!(feedback.trials.iter().all(|n| *n > 1));
        assert!(feedback.trials[2] > feedback.trials[0] * 2);
    }
    #[test]
    fn case_specialists_survive_inferior_average_loss_and_new_arrivals() {
        let path =
            std::env::temp_dir().join(format!("axon-case-portfolio-{}.axep", std::process::id()));
        let mut search = ConstructiveSearch::open_with_policy(
            &path,
            vec![
                NumericExample::new(vec![0.0], 0.0),
                NumericExample::new(vec![1.0], 1.0),
            ],
            &[],
            ConstructiveBudget {
                population: 4,
                ..ConstructiveBudget::default()
            },
            NumericTolerance::default(),
            1,
            ConstructivePolicy::Diverse,
        )
        .unwrap();
        let record = |errors: [f64; 2]| Individual {
            genome: Genome {
                arity: 1,
                genes: vec![
                    Gene {
                        function: 2,
                        args: [0; 4]
                    };
                    2
                ],
                parents: [0; 2],
            },
            fitness: Fitness {
                errors: std::array::from_fn(|i| if i < 2 { errors[i] } else { 0.0 }),
                loss: (errors[0] + errors[1]) / 2.0,
                correct: 1,
                output: 5,
                active: 1,
            },
        };
        search.install(record([0.0, 0.1]));
        search.install(record([0.9, 0.0]));
        for _ in 0..20 {
            search.install(record([0.2, 0.2]));
        }
        assert!(search.front.contains(&0));
        assert!(search.front.contains(&1));
        assert_eq!(search.front.len(), 4);
        drop(search);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn plateau_progress_is_not_restricted_to_the_promoted_elite() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "axon-constructive-policy-{}-{}.axep",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let training = (0..4)
            .map(|x| NumericExample::new(vec![x as f64], (x * x) as f64))
            .collect();
        let mut search = ConstructiveSearch::open(
            &path,
            training,
            &[],
            ConstructiveBudget::default(),
            NumericTolerance::default(),
            1,
        )
        .unwrap();
        let genome = Genome {
            arity: 1,
            genes: vec![
                Gene {
                    function: 2,
                    args: [0; 4]
                };
                2
            ],
            parents: [0; 2],
        };
        // Policy fixtures: more correct cases can accompany a worse aggregate loss.
        let fitness = Fitness {
            errors: [0.0; 32],
            loss: 0.1,
            correct: 1,
            output: 5,
            active: 1,
        };
        search.install(Individual {
            genome: genome.clone(),
            fitness: fitness.clone(),
        });
        search.stale = 7;
        search.install(Individual {
            genome,
            fitness: Fitness {
                correct: 2,
                loss: 0.2,
                ..fitness.clone()
            },
        });
        assert_eq!(search.stale, 0);
        assert_eq!(search.best().unwrap().fitness.correct, 1);
        // A perfectly fitting graph is still not a winner if expansion cannot be deployed.
        let mut ops = vec![Op::Input(0)];
        for _ in 0..127 {
            ops.extend([Op::Input(0), Op::Add]);
        }
        search.tools.push(Inherited {
            identity: 1,
            program: NumericProgram { arity: 1, ops },
        });
        let graph = Genome {
            arity: 1,
            genes: vec![
                Gene {
                    function: 5,
                    args: [0; 4],
                },
                Gene {
                    function: 2,
                    args: [5, 5, 0, 0],
                },
            ],
            parents: [0; 2],
        };
        search.install(Individual {
            genome: graph,
            fitness: Fitness {
                correct: 4,
                loss: 0.0,
                output: 6,
                active: 2,
                ..fitness
            },
        });
        assert!(search.winner().is_none());
        assert_eq!(search.stale, 1);
        drop(search);
        std::fs::remove_file(path).unwrap();
    }
}
