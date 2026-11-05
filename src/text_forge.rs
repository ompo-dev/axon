use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextExample {
    pub input: String,
    pub output: String,
}
impl TextExample {
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextBudget {
    pub max_examples: usize,
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_pieces: usize,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub max_work: u64,
    pub max_atoms: usize,
    pub max_delimiters: usize,
}
impl Default for TextBudget {
    fn default() -> Self {
        Self {
            max_examples: 32,
            max_input_bytes: 1024,
            max_output_bytes: 256,
            max_pieces: 6,
            max_nodes: 4096,
            max_edges: 32768,
            max_work: 2_000_000,
            max_atoms: 4096,
            max_delimiters: 32,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextForgeError {
    Invalid(&'static str),
    Limit(&'static str),
    ContradictoryExamples,
    Unsupported,
}
impl std::fmt::Display for TextForgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid text synthesis: {s}"),
            Self::Limit(s) => write!(f, "text synthesis budget exhausted: {s}"),
            Self::ContradictoryExamples => {
                f.write_str("the same input has incompatible demonstrations")
            }
            Self::Unsupported => f.write_str("no program found in the declared text grammar"),
        }
    }
}
impl std::error::Error for TextForgeError {}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum TokenClass {
    Word,
    Letters,
    Digits,
}
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Selector {
    Whole,
    Field {
        delimiter: String,
        index: u8,
        reverse: bool,
    },
    Token {
        class: TokenClass,
        index: u8,
        reverse: bool,
    },
}
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Case {
    Preserve,
    Lower,
    Upper,
}
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Atom {
    pub selector: Selector,
    pub trim: bool,
    pub collapse: bool,
    pub case: Case,
}
impl Atom {
    pub(crate) fn evaluate(&self, input: &str) -> Option<String> {
        let selected = match &self.selector {
            Selector::Whole => input,
            Selector::Field {
                delimiter,
                index,
                reverse,
            } => {
                if !input.contains(delimiter) {
                    return None;
                }
                let fields = input.split(delimiter).collect::<Vec<_>>();
                let index = if *reverse {
                    fields.len().checked_sub(usize::from(*index) + 1)?
                } else {
                    usize::from(*index)
                };
                fields.get(index).copied()?
            }
            Selector::Token {
                class,
                index,
                reverse,
            } => {
                let tokens = tokens(input, *class);
                let index = if *reverse {
                    tokens.len().checked_sub(usize::from(*index) + 1)?
                } else {
                    usize::from(*index)
                };
                tokens.get(index).copied()?
            }
        };
        let selected = if self.trim { selected.trim() } else { selected };
        let selected = if self.collapse {
            selected.split_whitespace().collect::<Vec<_>>().join(" ")
        } else {
            selected.to_owned()
        };
        Some(match self.case {
            Case::Preserve => selected,
            Case::Lower => selected.to_lowercase(),
            Case::Upper => selected.to_uppercase(),
        })
    }
}

fn tokens(input: &str, class: TokenClass) -> Vec<&str> {
    input
        .split(|c: char| !match class {
            TokenClass::Word => c.is_alphanumeric(),
            TokenClass::Letters => c.is_alphabetic(),
            TokenClass::Digits => c.is_ascii_digit(),
        })
        .filter(|s| !s.is_empty())
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Label {
    Atoms(Vec<usize>),
    Literal(String),
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Edge {
    pub next: usize,
    pub label: Label,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Node {
    pub terminal: bool,
    pub edges: Vec<Edge>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextProgram {
    pub(crate) atoms: Vec<Atom>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) budget: TextBudget,
    pub(crate) pieces: usize,
    pub(crate) work: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextWitness {
    pub output: String,
    pub steps: Vec<String>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextPrediction {
    Agreed(TextWitness),
    Ambiguous {
        first: TextWitness,
        second: TextWitness,
    },
    OutOfDomain,
}
/// Same version-space decision without allocating human-readable proof traces.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextValuePrediction {
    Agreed(String),
    Ambiguous { first: String, second: String },
    OutOfDomain,
}
impl TextPrediction {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Agreed(_) => "agreed",
            Self::Ambiguous { .. } => "needs_example",
            Self::OutOfDomain => "out_of_domain",
        }
    }
}

impl TextProgram {
    pub fn learn(examples: &[TextExample], budget: TextBudget) -> Result<Self, TextForgeError> {
        validate_budget(budget)?;
        if examples.is_empty() {
            return Err(TextForgeError::Invalid("no demonstrations"));
        }
        if examples.len() > budget.max_examples {
            return Err(TextForgeError::Limit("examples"));
        }
        let mut unique = BTreeMap::new();
        for example in examples {
            validate_example(example, budget)?;
            if unique
                .insert(&example.input, &example.output)
                .is_some_and(|output| output != &example.output)
            {
                return Err(TextForgeError::ContradictoryExamples);
            }
        }
        let mut examples = examples.to_vec();
        examples.sort_by(|a, b| {
            b.output
                .len()
                .cmp(&a.output.len())
                .then(a.input.cmp(&b.input))
        });
        examples.dedup();
        if examples.iter().all(|example| example.output.is_empty()) {
            return Ok(Self {
                atoms: Vec::new(),
                nodes: vec![Node {
                    terminal: true,
                    edges: Vec::new(),
                }],
                budget,
                pieces: 0,
                work: 0,
            });
        }
        let mut work = 0;
        let atoms = generate_atoms(&examples, budget, &mut work)?;
        let mut groups = BTreeMap::<Vec<String>, Vec<usize>>::new();
        for (id, atom) in atoms.iter().enumerate() {
            charge(&mut work, examples.len() as u64, budget)?;
            if let Some(values) = examples
                .iter()
                .map(|e| atom.evaluate(&e.input))
                .collect::<Option<Vec<_>>>()
                && !values[0].is_empty()
                && values
                    .iter()
                    .zip(&examples)
                    .all(|(value, example)| value.len() <= example.output.len())
            {
                groups.entry(values).or_default().push(id);
            }
        }
        let finish = examples.iter().map(|e| e.output.len()).collect::<Vec<_>>();
        let mut positions = vec![vec![0; examples.len()]];
        let mut nodes = vec![Node {
            terminal: false,
            edges: Vec::new(),
        }];
        let mut layer = vec![0_usize];
        let mut edge_count = 0;
        let mut alternative_refs = 0_usize;
        let mut terminal = None;
        let mut pieces = 0;
        for depth in 1..=budget.max_pieces {
            let mut next_positions = BTreeMap::<Vec<usize>, usize>::new();
            for &id in &layer {
                let position = positions[id].clone();
                let mut proposals = Vec::new();
                for (values, choices) in &groups {
                    charge(&mut work, examples.len() as u64, budget)?;
                    if examples
                        .iter()
                        .zip(&position)
                        .zip(values)
                        .all(|((e, p), value)| e.output[*p..].starts_with(value))
                    {
                        let next = position
                            .iter()
                            .zip(values)
                            .map(|(p, value)| p + value.len())
                            .collect::<Vec<_>>();
                        proposals.push((next, Label::Atoms(choices.clone())));
                    }
                }
                let suffixes = examples
                    .iter()
                    .zip(&position)
                    .map(|(e, p)| &e.output[*p..])
                    .collect::<Vec<_>>();
                let first = suffixes[0];
                for end in first
                    .char_indices()
                    .skip(1)
                    .map(|(i, _)| i)
                    .chain(std::iter::once(first.len()))
                    .filter(|end| *end > 0)
                {
                    charge(&mut work, examples.len() as u64, budget)?;
                    let literal = &first[..end];
                    if suffixes.iter().all(|s| s.starts_with(literal)) {
                        proposals.push((
                            position.iter().map(|p| p + end).collect(),
                            Label::Literal(literal.into()),
                        ));
                    } else {
                        break;
                    }
                }
                for (next, label) in proposals {
                    if let Label::Atoms(ids) = &label {
                        alternative_refs += ids.len();
                    }
                    if alternative_refs > 131_072 {
                        return Err(TextForgeError::Limit("alternative references"));
                    }
                    if edge_count >= budget.max_edges {
                        return Err(TextForgeError::Limit("edges"));
                    }
                    let next_id = if let Some(id) = next_positions.get(&next) {
                        *id
                    } else {
                        if nodes.len() >= budget.max_nodes {
                            return Err(TextForgeError::Limit("nodes"));
                        }
                        let new_id = nodes.len();
                        let done = next == finish;
                        positions.push(next.clone());
                        nodes.push(Node {
                            terminal: done,
                            edges: Vec::new(),
                        });
                        next_positions.insert(next, new_id);
                        if done {
                            terminal = Some(new_id);
                        }
                        new_id
                    };
                    nodes[id].edges.push(Edge {
                        next: next_id,
                        label,
                    });
                    edge_count += 1;
                }
            }
            if terminal.is_some() {
                pieces = depth;
                break;
            }
            layer = next_positions.into_values().collect();
            if layer.is_empty() {
                break;
            }
        }
        terminal.ok_or(TextForgeError::Unsupported)?;
        // A compact version space retains every shortest path, not just one fitted recipe.
        let mut viable = vec![false; nodes.len()];
        for id in (0..nodes.len()).rev() {
            viable[id] = nodes[id].terminal || nodes[id].edges.iter().any(|edge| viable[edge.next]);
        }
        let mut remap = BTreeMap::new();
        for (id, value) in viable.iter().enumerate() {
            if *value {
                remap.insert(id, remap.len());
            }
        }
        let mut compact = Vec::new();
        for (id, node) in nodes.into_iter().enumerate() {
            if !viable[id] {
                continue;
            }
            compact.push(Node {
                terminal: node.terminal,
                edges: node
                    .edges
                    .into_iter()
                    .filter(|e| viable[e.next])
                    .map(|mut e| {
                        e.next = remap[&e.next];
                        e
                    })
                    .collect(),
            });
        }
        let used = compact
            .iter()
            .flat_map(|n| &n.edges)
            .filter_map(|e| {
                if let Label::Atoms(ids) = &e.label {
                    Some(ids)
                } else {
                    None
                }
            })
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let atom_map = used
            .iter()
            .enumerate()
            .map(|(new, old)| (*old, new))
            .collect::<BTreeMap<_, _>>();
        for node in &mut compact {
            for edge in &mut node.edges {
                if let Label::Atoms(ids) = &mut edge.label {
                    for id in ids {
                        *id = atom_map[id];
                    }
                }
            }
        }
        let program = Self {
            atoms: used.into_iter().map(|id| atoms[id].clone()).collect(),
            nodes: compact,
            budget,
            pieces,
            work,
        };
        program.validate()?;
        for example in &examples {
            if !matches!(program.predict(&example.input)?, TextPrediction::Agreed(w) if w.output == example.output)
            {
                return Err(TextForgeError::Invalid("training replay"));
            }
        }
        Ok(program)
    }

    pub fn pieces(&self) -> usize {
        self.pieces
    }
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn atoms(&self) -> usize {
        self.atoms.len()
    }
    pub fn search_work(&self) -> u64 {
        self.work
    }
    pub fn representative_steps(&self) -> Vec<String> {
        let mut steps = Vec::new();
        let mut id = 0;
        while !self.nodes[id].terminal {
            let edge = &self.nodes[id].edges[0];
            steps.push(match &edge.label {
                Label::Literal(s) => format!("literal({s:?})"),
                Label::Atoms(ids) => format!("{:?}", self.atoms[ids[0]]),
            });
            id = edge.next;
        }
        steps
    }

    pub fn predict(&self, input: &str) -> Result<TextPrediction, TextForgeError> {
        if input.len() > self.budget.max_input_bytes {
            return Err(TextForgeError::Limit("input bytes"));
        }
        let values = self
            .atoms
            .iter()
            .map(|atom| atom.evaluate(input))
            .collect::<Vec<_>>();
        let mut results = vec![Vec::<TextWitness>::new(); self.nodes.len()];
        let mut work = 0;
        for id in (0..self.nodes.len()).rev() {
            let node = &self.nodes[id];
            if node.terminal {
                results[id].push(TextWitness {
                    output: String::new(),
                    steps: Vec::new(),
                });
                continue;
            }
            'edges: for edge in &node.edges {
                let choices = match &edge.label {
                    Label::Literal(value) => vec![(value.as_str(), format!("literal({value:?})"))],
                    Label::Atoms(ids) => ids
                        .iter()
                        .filter_map(|id| {
                            values[*id]
                                .as_deref()
                                .map(|value| (value, format!("{:?}", self.atoms[*id])))
                        })
                        .collect(),
                };
                for (prefix, step) in choices {
                    for suffix in results[edge.next].clone() {
                        charge(&mut work, 1, self.budget)?;
                        if prefix.len() + suffix.output.len() > self.budget.max_output_bytes {
                            return Err(TextForgeError::Limit("prediction output bytes"));
                        }
                        let output = format!("{prefix}{}", suffix.output);
                        if results[id].iter().any(|w| w.output == output) {
                            continue;
                        }
                        let mut steps = vec![step.clone()];
                        steps.extend(suffix.steps);
                        results[id].push(TextWitness { output, steps });
                        if results[id].len() == 2 {
                            break 'edges;
                        }
                    }
                }
            }
        }
        Ok(match results[0].len() {
            0 => TextPrediction::OutOfDomain,
            1 => TextPrediction::Agreed(results[0].remove(0)),
            _ => {
                let first = results[0].remove(0);
                let second = results[0].remove(0);
                TextPrediction::Ambiguous { first, second }
            }
        })
    }

    pub fn predict_value(&self, input: &str) -> Result<TextValuePrediction, TextForgeError> {
        if input.len() > self.budget.max_input_bytes {
            return Err(TextForgeError::Limit("input bytes"));
        }
        let values = self
            .atoms
            .iter()
            .map(|a| a.evaluate(input))
            .collect::<Vec<_>>();
        let mut results = vec![Vec::<String>::new(); self.nodes.len()];
        let mut work = 0;
        for id in (0..self.nodes.len()).rev() {
            let node = &self.nodes[id];
            if node.terminal {
                results[id].push(String::new());
                continue;
            }
            let mut outputs = Vec::new();
            'edges: for edge in &node.edges {
                let mut append = |prefix: &str| -> Result<bool, TextForgeError> {
                    for suffix in &results[edge.next] {
                        charge(&mut work, 1, self.budget)?;
                        let length = prefix.len() + suffix.len();
                        if length > self.budget.max_output_bytes {
                            return Err(TextForgeError::Limit("prediction output bytes"));
                        }
                        if outputs.iter().any(|s: &String| {
                            s.len() == length
                                && s.starts_with(prefix)
                                && &s[prefix.len()..] == suffix
                        }) {
                            continue;
                        }
                        let mut output = String::with_capacity(length);
                        output.push_str(prefix);
                        output.push_str(suffix);
                        outputs.push(output);
                        if outputs.len() == 2 {
                            break;
                        }
                    }
                    Ok(outputs.len() == 2)
                };
                match &edge.label {
                    Label::Literal(s) => {
                        if append(s)? {
                            break;
                        }
                    }
                    Label::Atoms(ids) => {
                        for atom in ids {
                            if let Some(value) = &values[*atom]
                                && append(value)?
                            {
                                break 'edges;
                            }
                        }
                    }
                }
            }
            results[id] = outputs;
        }
        Ok(match results[0].len() {
            0 => TextValuePrediction::OutOfDomain,
            1 => TextValuePrediction::Agreed(results[0].remove(0)),
            _ => TextValuePrediction::Ambiguous {
                first: results[0].remove(0),
                second: results[0].remove(0),
            },
        })
    }

    pub(crate) fn validate(&self) -> Result<(), TextForgeError> {
        validate_budget(self.budget)?;
        if self.nodes.is_empty()
            || self.nodes.len() > self.budget.max_nodes
            || self.atoms.len() > self.budget.max_atoms
            || self.pieces > self.budget.max_pieces
            || self.work > self.budget.max_work
        {
            return Err(TextForgeError::Invalid("program dimensions"));
        }
        let mut edges = 0;
        let mut alternative_refs = 0_usize;
        let mut reachable = vec![false; self.nodes.len()];
        let mut depths = vec![0; self.nodes.len()];
        reachable[0] = true;
        for (id, node) in self.nodes.iter().enumerate() {
            if !reachable[id] || node.terminal == !node.edges.is_empty() {
                return Err(TextForgeError::Invalid("terminal or unreachable node"));
            }
            if node.terminal && depths[id] != self.pieces {
                return Err(TextForgeError::Invalid("program depth"));
            }
            for edge in &node.edges {
                if let Label::Atoms(ids) = &edge.label {
                    alternative_refs += ids.len();
                }
                if alternative_refs > 131_072 {
                    return Err(TextForgeError::Limit("alternative references"));
                }
                edges += 1;
                if edge.next <= id || edge.next >= self.nodes.len() {
                    return Err(TextForgeError::Invalid("program DAG"));
                }
                let depth = depths[id] + 1;
                if reachable[edge.next] && depths[edge.next] != depth {
                    return Err(TextForgeError::Invalid("inconsistent program depth"));
                }
                depths[edge.next] = depth;
                reachable[edge.next] = true;
                match &edge.label {
                    Label::Literal(value)
                        if !value.is_empty() && value.len() <= self.budget.max_output_bytes => {}
                    Label::Atoms(ids)
                        if !ids.is_empty()
                            && ids.len() <= self.atoms.len()
                            && ids.iter().all(|id| *id < self.atoms.len())
                            && ids.windows(2).all(|pair| pair[0] < pair[1]) => {}
                    _ => return Err(TextForgeError::Invalid("edge label")),
                }
            }
        }
        if edges > self.budget.max_edges {
            return Err(TextForgeError::Limit("edges"));
        }
        for atom in &self.atoms {
            match &atom.selector {
                Selector::Field {
                    delimiter, index, ..
                } if delimiter.is_empty() || delimiter.chars().count() > 3 || *index >= 8 => {
                    return Err(TextForgeError::Invalid("field selector"));
                }
                Selector::Token { index, .. } if *index >= 8 => {
                    return Err(TextForgeError::Invalid("token selector"));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

pub(crate) fn validate_budget(budget: TextBudget) -> Result<(), TextForgeError> {
    if budget.max_examples == 0
        || budget.max_examples > 32
        || budget.max_input_bytes == 0
        || budget.max_input_bytes > 4096
        || budget.max_output_bytes == 0
        || budget.max_output_bytes > 1024
        || budget.max_pieces == 0
        || budget.max_pieces > 8
        || budget.max_nodes == 0
        || budget.max_nodes > 8192
        || budget.max_edges == 0
        || budget.max_edges > 65536
        || budget.max_work == 0
        || budget.max_work > 20_000_000
        || budget.max_atoms == 0
        || budget.max_atoms > 8192
        || budget.max_delimiters == 0
        || budget.max_delimiters > 64
    {
        return Err(TextForgeError::Invalid("resource limits"));
    }
    Ok(())
}
pub(crate) fn validate_example(
    example: &TextExample,
    budget: TextBudget,
) -> Result<(), TextForgeError> {
    if example.input.len() > budget.max_input_bytes
        || example.output.len() > budget.max_output_bytes
    {
        return Err(TextForgeError::Limit("example bytes"));
    }
    Ok(())
}
fn charge(work: &mut u64, amount: u64, budget: TextBudget) -> Result<(), TextForgeError> {
    *work = work
        .checked_add(amount)
        .ok_or(TextForgeError::Limit("work"))?;
    if *work > budget.max_work {
        return Err(TextForgeError::Limit("work"));
    }
    Ok(())
}

fn generate_atoms(
    examples: &[TextExample],
    budget: TextBudget,
    work: &mut u64,
) -> Result<Vec<Atom>, TextForgeError> {
    let mut delimiters = BTreeSet::new();
    for example in examples {
        let chars = example.input.chars().collect::<Vec<_>>();
        for start in 0..chars.len() {
            for length in 1..=3 {
                if start + length > chars.len() {
                    break;
                }
                let part = &chars[start..start + length];
                if part.iter().all(|c| !c.is_alphanumeric()) {
                    let delimiter = part.iter().collect::<String>();
                    charge(work, examples.len() as u64, budget)?;
                    if examples.iter().all(|e| e.input.contains(&delimiter)) {
                        delimiters.insert(delimiter);
                    }
                    if delimiters.len() > budget.max_delimiters {
                        return Err(TextForgeError::Limit("delimiters"));
                    }
                }
            }
        }
    }
    let mut selectors = vec![Selector::Whole];
    for delimiter in delimiters {
        for index in 0..8 {
            for reverse in [false, true] {
                selectors.push(Selector::Field {
                    delimiter: delimiter.clone(),
                    index,
                    reverse,
                });
            }
        }
    }
    for class in [TokenClass::Word, TokenClass::Letters, TokenClass::Digits] {
        for index in 0..8 {
            for reverse in [false, true] {
                selectors.push(Selector::Token {
                    class,
                    index,
                    reverse,
                });
            }
        }
    }
    let mut atoms = Vec::new();
    for selector in selectors {
        for trim in [false, true] {
            for collapse in [false, true] {
                for case in [Case::Preserve, Case::Lower, Case::Upper] {
                    let atom = Atom {
                        selector: selector.clone(),
                        trim,
                        collapse,
                        case,
                    };
                    charge(work, examples.len() as u64, budget)?;
                    if examples.iter().all(|e| atom.evaluate(&e.input).is_some()) {
                        if atoms.len() >= budget.max_atoms {
                            return Err(TextForgeError::Limit("atoms"));
                        }
                        atoms.push(atom);
                    }
                }
            }
        }
    }
    Ok(atoms)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn learn(pairs: &[(&str, &str)]) -> TextProgram {
        TextProgram::learn(
            &pairs
                .iter()
                .map(|(a, b)| TextExample::new(*a, *b))
                .collect::<Vec<_>>(),
            TextBudget::default(),
        )
        .unwrap()
    }
    fn agreed(program: &TextProgram, input: &str, output: &str) {
        assert!(
            matches!(program.predict(input).unwrap(), TextPrediction::Agreed(w) if w.output == output),
            "prediction={:?}",
            program.predict(input)
        );
    }
    #[test]
    fn learns_normalization_without_a_ready_transformation() {
        let program = learn(&[
            ("  Alpha  Beta  ", "alpha beta"),
            (" Gamma \t Delta ", "gamma delta"),
        ]);
        agreed(&program, "  NOVA   Equipe ", "nova equipe");
        assert_eq!(program.pieces(), 1);
    }
    #[test]
    fn discovers_and_reuses_a_three_piece_program() {
        let program = learn(&[
            ("Ana Maria|Silva Costa", "Silva Costa, Ana Maria"),
            ("Bruno|Lima Rocha", "Lima Rocha, Bruno"),
        ]);
        agreed(
            &program,
            "Carla Sofia|Pereira Alves",
            "Pereira Alves, Carla Sofia",
        );
        assert_eq!(program.pieces(), 3);
        assert!(
            program
                .representative_steps()
                .iter()
                .any(|s| s.contains("literal"))
        );
    }
    #[test]
    fn ambiguity_has_two_executable_witnesses_and_an_extra_example_resolves_it() {
        let initial = learn(&[("Ana Silva", "Silva"), ("Bruno Costa", "Costa")]);
        let TextPrediction::Ambiguous { first, second } =
            initial.predict("Carla de Souza").unwrap()
        else {
            panic!("must request a demonstration");
        };
        assert_ne!(first.output, second.output);
        let refined = learn(&[
            ("Ana Silva", "Silva"),
            ("Bruno Costa", "Costa"),
            ("Carla de Souza", "Souza"),
        ]);
        agreed(&refined, "Daniel da Rocha", "Rocha");
    }
    #[test]
    fn unrelated_inputs_are_not_completed_with_a_guessed_value() {
        let program = learn(&[
            ("Name|North Zone", "North Zone"),
            ("Other|South District", "South District"),
        ]);
        assert_eq!(
            program.predict("Broken").unwrap(),
            TextPrediction::OutOfDomain
        );
    }
    #[test]
    fn utf8_boundaries_and_expanding_case_conversion_are_supported() {
        let program = learn(&[("  stra\u{df}e ", "STRASSE"), (" caf\u{e9} ", "CAF\u{c9}")]);
        agreed(&program, " \u{e1}gua ", "\u{c1}GUA");
    }
    #[test]
    fn preserves_leading_zeroes_in_learned_extraction() {
        let program = learn(&[
            ("ticket AB-003", "ID:003"),
            ("receipt XYZ-810", "ID:810"),
            ("case K-042", "ID:042"),
        ]);
        agreed(&program, "order LM-0007", "ID:0007");
    }
    #[test]
    fn duplicates_are_idempotent_and_contradictions_are_refused() {
        assert_eq!(
            learn(&[("aa", "AA"), ("bb", "BB")]),
            learn(&[("aa", "AA"), ("bb", "BB"), ("aa", "AA")])
        );
        assert_eq!(
            TextProgram::learn(
                &[TextExample::new("x", "a"), TextExample::new("x", "b")],
                TextBudget::default()
            ),
            Err(TextForgeError::ContradictoryExamples)
        );
    }
    #[test]
    fn unavailable_operations_are_not_claimed_as_learned() {
        assert_eq!(
            TextProgram::learn(
                &[
                    TextExample::new("abcd", "dcba"),
                    TextExample::new("wxyz", "zyxw")
                ],
                TextBudget::default()
            ),
            Err(TextForgeError::Unsupported)
        );
    }
    #[test]
    fn bounded_search_refuses_incomplete_version_spaces() {
        let examples = [
            TextExample::new("  aa ", "AA"),
            TextExample::new(" bb ", "BB"),
        ];
        for budget in [
            TextBudget {
                max_work: 1,
                ..TextBudget::default()
            },
            TextBudget {
                max_atoms: 1,
                ..TextBudget::default()
            },
            TextBudget {
                max_nodes: 1,
                ..TextBudget::default()
            },
            TextBudget {
                max_edges: 1,
                ..TextBudget::default()
            },
        ] {
            assert!(matches!(
                TextProgram::learn(&examples, budget),
                Err(TextForgeError::Limit(_))
            ));
        }
    }
    #[test]
    fn empty_output_is_an_explicit_constant_program() {
        let program = learn(&[("one", ""), ("two", "")]);
        agreed(&program, "three", "");
        assert_eq!(program.pieces(), 0);
    }
    #[test]
    fn two_witness_predictions_match_a_full_output_set_enumerator() {
        let programs = [
            learn(&[("Ana Silva", "Silva"), ("Bruno Costa", "Costa")]),
            learn(&[("a|b", "b/a"), ("c|d", "d/c")]),
        ];
        for program in programs {
            for input in [
                "Carla de Souza",
                "a|b",
                "alpha|beta",
                "x",
                "one two three",
                "left side|right side",
            ] {
                let mut sets = vec![BTreeSet::<String>::new(); program.nodes.len()];
                for id in (0..program.nodes.len()).rev() {
                    if program.nodes[id].terminal {
                        sets[id].insert(String::new());
                        continue;
                    }
                    for edge in &program.nodes[id].edges {
                        let values = match &edge.label {
                            Label::Literal(s) => vec![s.clone()],
                            Label::Atoms(ids) => ids
                                .iter()
                                .filter_map(|id| program.atoms[*id].evaluate(input))
                                .collect(),
                        };
                        let suffixes = sets[edge.next].clone();
                        for prefix in values {
                            for suffix in &suffixes {
                                sets[id].insert(format!("{prefix}{suffix}"));
                            }
                        }
                    }
                }
                match program.predict(input).unwrap() {
                    TextPrediction::OutOfDomain => assert!(sets[0].is_empty()),
                    TextPrediction::Agreed(w) => assert_eq!(sets[0], BTreeSet::from([w.output])),
                    TextPrediction::Ambiguous { first, second } => {
                        assert!(sets[0].len() > 1);
                        assert!(
                            sets[0].contains(&first.output) && sets[0].contains(&second.output)
                        );
                        assert_ne!(first.output, second.output);
                    }
                }
            }
        }
    }
    #[test]
    fn prediction_limits_do_not_silently_remove_alternative_programs() {
        let program = learn(&[("aa", "aa"), ("bb", "bb")]);
        assert!(matches!(
            program.predict(&"x".repeat(300)),
            Err(TextForgeError::Limit("prediction output bytes"))
        ));
        assert!(matches!(
            program.predict(&"x".repeat(1025)),
            Err(TextForgeError::Limit("input bytes"))
        ));
    }
}
