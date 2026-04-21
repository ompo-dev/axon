use std::collections::{HashMap, VecDeque};

use crate::cli::RunMode;
use crate::memory::{MemoryHypothesis, MemoryState, NodeKind};
use crate::storage::{MutationKind, MutationRecord};

#[derive(Clone, Debug)]
pub struct Assembly {
    pub id: u32,
    pub symbol: Option<char>,
    pub activation: f32,
    pub stability: f32,
    pub support_count: u32,
    pub novelty: f32,
    pub last_tick: u64,
}

#[derive(Clone, Debug)]
pub struct DeltaEdge {
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    pub utility: f32,
    pub weak_ticks: u32,
}

#[derive(Clone, Debug)]
pub struct CsrGraph {
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub weights: Vec<f32>,
}

impl CsrGraph {
    pub fn empty() -> Self {
        Self {
            row_ptr: vec![0],
            col_idx: Vec::new(),
            weights: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BrainStats {
    pub spawn_count: u64,
    pub merge_count: u64,
    pub prune_count: u64,
    pub emitted_chars: u64,
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub from_user: bool,
    pub text: String,
    pub tick: u64,
}

#[derive(Clone, Debug)]
pub struct BrainState {
    pub tick: u64,
    pub mode: RunMode,
    pub assemblies: Vec<Assembly>,
    pub activations: Vec<f32>,
    pub csr: CsrGraph,
    pub delta_edges: Vec<DeltaEdge>,
    pub char_nodes: HashMap<u32, u32>,
    pub char_frequency: HashMap<u32, u64>,
    pub transitions: HashMap<(u32, u32), u64>,
    pub last_input_char: Option<char>,
    pub pending_input: VecDeque<char>,
    pub pending_output: VecDeque<char>,
    pub recent_user_buffer: String,
    pub transcript: Vec<ChatMessage>,
    pub stats: BrainStats,
    pub rng: XorShift64,
}

impl BrainState {
    pub fn new(mode: RunMode, seed: u64) -> Self {
        Self {
            tick: 0,
            mode,
            assemblies: Vec::new(),
            activations: Vec::new(),
            csr: CsrGraph::empty(),
            delta_edges: Vec::new(),
            char_nodes: HashMap::new(),
            char_frequency: HashMap::new(),
            transitions: HashMap::new(),
            last_input_char: None,
            pending_input: VecDeque::new(),
            pending_output: VecDeque::new(),
            recent_user_buffer: String::new(),
            transcript: Vec::new(),
            stats: BrainStats {
                spawn_count: 0,
                merge_count: 0,
                prune_count: 0,
                emitted_chars: 0,
            },
            rng: XorShift64::new(seed),
        }
    }

    pub fn queue_user_text(&mut self, text: &str) {
        for ch in text.chars() {
            self.pending_input.push_back(ch);
        }
        self.pending_input.push_back('\n');
        self.transcript.push(ChatMessage {
            from_user: true,
            text: text.to_string(),
            tick: self.tick,
        });
    }

    pub fn step(
        &mut self,
        memory: &mut MemoryState,
        outgoing_mutations: &mut Vec<MutationRecord>,
        use_gpu: bool,
    ) -> Option<char> {
        self.tick = self.tick.saturating_add(1);
        let mut stimulus_idx: Option<usize> = None;
        if let Some(ch) = self.pending_input.pop_front() {
            let canonical = canonical_char(ch);
            let idx = self.ensure_symbol_node(canonical) as usize;
            stimulus_idx = Some(idx);
            if canonical != '\n' {
                self.recent_user_buffer.push(canonical);
                if self.recent_user_buffer.len() > 512 {
                    self.recent_user_buffer.remove(0);
                }
            }
            outgoing_mutations.push(MutationRecord {
                kind: MutationKind::InputChar,
                flags: 0,
                tick: self.tick,
                a: canonical as u32,
                b: idx as u32,
                c: 0,
                value: 1.0,
                extra: 0.0,
            });
            *self.char_frequency.entry(canonical as u32).or_insert(0) += 1;
            if let Some(prev) = self.last_input_char {
                *self
                    .transitions
                    .entry((prev as u32, canonical as u32))
                    .or_insert(0) += 1;
            }
            if canonical == '\n' {
                let message = self.recent_user_buffer.trim().to_string();
                if !message.is_empty() {
                    memory.observe_text(self.tick, &message, outgoing_mutations);
                    self.plan_response(memory);
                } else {
                    self.emit_response_text("pode enviar uma frase?\n");
                }
                self.recent_user_buffer.clear();
                self.last_input_char = None;
            } else {
                self.last_input_char = Some(canonical);
            }
        }

        self.propagate_field(stimulus_idx, use_gpu);
        self.apply_plasticity(outgoing_mutations);
        self.apply_structural_rules(outgoing_mutations);
        memory.decay_to_tick(self.tick, outgoing_mutations);

        let emitted = if let Some(ch) = self.pending_output.pop_front() {
            self.stats.emitted_chars = self.stats.emitted_chars.saturating_add(1);
            outgoing_mutations.push(MutationRecord {
                kind: MutationKind::OutputChar,
                flags: 0,
                tick: self.tick,
                a: ch as u32,
                b: 0,
                c: 0,
                value: 1.0,
                extra: 0.0,
            });
            Some(ch)
        } else {
            None
        };

        if let Some(ch) = emitted {
            if let Some(last) = self.transcript.last_mut() {
                if !last.from_user {
                    last.text.push(ch);
                } else {
                    self.transcript.push(ChatMessage {
                        from_user: false,
                        text: ch.to_string(),
                        tick: self.tick,
                    });
                }
            } else {
                self.transcript.push(ChatMessage {
                    from_user: false,
                    text: ch.to_string(),
                    tick: self.tick,
                });
            }
            if ch == '\n' {
                self.transcript.push(ChatMessage {
                    from_user: false,
                    text: String::new(),
                    tick: self.tick,
                });
            }
        }

        emitted
    }

    pub fn apply_correction(
        &mut self,
        memory: &mut MemoryState,
        wrong: &str,
        correct: &str,
        outgoing_mutations: &mut Vec<MutationRecord>,
    ) {
        memory.apply_correction(self.tick, wrong, correct, outgoing_mutations);
        let feedback = format!("corrigido: '{}' -> '{}'\n", wrong.trim(), correct.trim());
        self.emit_response_text(&feedback);
    }

    fn propagate_field(&mut self, stimulus_idx: Option<usize>, _use_gpu: bool) {
        if self.assemblies.is_empty() {
            return;
        }
        let n = self.assemblies.len();
        let mut drive = vec![0.0_f32; n];
        for row in 0..n {
            let start = self.csr.row_ptr[row] as usize;
            let end = self.csr.row_ptr[row + 1] as usize;
            let source_activation = self.activations[row];
            if source_activation.abs() < 0.0001 {
                continue;
            }
            for edge_idx in start..end {
                let to = self.csr.col_idx[edge_idx] as usize;
                drive[to] += self.csr.weights[edge_idx] * source_activation;
            }
        }
        for edge in &self.delta_edges {
            let from = edge.from as usize;
            let to = edge.to as usize;
            if from < self.activations.len() && to < drive.len() {
                drive[to] += self.activations[from] * edge.weight;
            }
        }
        if let Some(idx) = stimulus_idx {
            if idx < drive.len() {
                drive[idx] += 1.0;
            }
        }

        let lambda = 0.15_f32;
        for i in 0..n {
            let inhibition = if self.activations[i].abs() > 0.8 {
                0.12
            } else {
                0.03
            };
            let raw = drive[i] - inhibition;
            let next = (1.0 - lambda) * self.activations[i] + lambda * raw.tanh();
            self.activations[i] = next.clamp(-1.0, 1.0);
            self.assemblies[i].activation = self.activations[i];
            self.assemblies[i].stability = (self.assemblies[i].stability * 0.995
                + self.activations[i].abs() * 0.005)
                .clamp(0.0, 1.0);
            self.assemblies[i].last_tick = self.tick;
            if self.activations[i].abs() > 0.2 {
                self.assemblies[i].support_count =
                    self.assemblies[i].support_count.saturating_add(1);
            }
        }
    }

    fn apply_plasticity(&mut self, outgoing_mutations: &mut Vec<MutationRecord>) {
        let eta = 0.02_f32;
        let eta_r = 0.002_f32;
        let active: Vec<usize> = self
            .activations
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| if value.abs() > 0.25 { Some(idx) } else { None })
            .collect();
        for i in 0..active.len() {
            for j in 0..active.len() {
                if i == j {
                    continue;
                }
                let from = active[i] as u32;
                let to = active[j] as u32;
                let ai = self.activations[from as usize];
                let aj = self.activations[to as usize];
                let edge_idx = self.find_or_create_delta_edge(from, to);
                let edge = &mut self.delta_edges[edge_idx];
                let delta = eta * (ai * aj - aj * aj * edge.weight);
                edge.weight = (edge.weight + delta + eta_r * edge.utility).clamp(-1.5, 1.5);
                edge.utility = (edge.utility * 0.99 + (ai.abs() + aj.abs()) * 0.01).clamp(0.0, 1.0);
                outgoing_mutations.push(MutationRecord {
                    kind: MutationKind::EdgeUpdate,
                    flags: 0,
                    tick: self.tick,
                    a: from,
                    b: to,
                    c: 0,
                    value: edge.weight,
                    extra: delta,
                });
            }
        }
    }

    fn apply_structural_rules(&mut self, outgoing_mutations: &mut Vec<MutationRecord>) {
        if self.assemblies.is_empty() {
            return;
        }
        let novelty = self
            .assemblies
            .iter()
            .map(|a| (1.0 - a.stability) * a.activation.abs())
            .fold(0.0_f32, f32::max);
        if novelty > 0.72 && self.assemblies.len() < 16_384 {
            let id = self.add_assembly(None);
            self.stats.spawn_count = self.stats.spawn_count.saturating_add(1);
            outgoing_mutations.push(MutationRecord {
                kind: MutationKind::Spawn,
                flags: 0,
                tick: self.tick,
                a: id,
                b: 0,
                c: 0,
                value: novelty,
                extra: 0.0,
            });
        }

        if self.assemblies.len() >= 2 {
            let mut top_a = 0usize;
            let mut top_b = 1usize;
            for i in 0..self.assemblies.len() {
                if self.activations[i].abs() > self.activations[top_a].abs() {
                    top_b = top_a;
                    top_a = i;
                } else if i != top_a && self.activations[i].abs() > self.activations[top_b].abs() {
                    top_b = i;
                }
            }
            let sim = 1.0 - (self.activations[top_a] - self.activations[top_b]).abs();
            if sim > 0.92
                && self.assemblies[top_a].support_count > 32
                && self.assemblies[top_b].support_count > 32
            {
                self.assemblies[top_a].stability =
                    (self.assemblies[top_a].stability + self.assemblies[top_b].stability) * 0.5;
                self.assemblies[top_b].stability *= 0.8;
                self.stats.merge_count = self.stats.merge_count.saturating_add(1);
                outgoing_mutations.push(MutationRecord {
                    kind: MutationKind::Merge,
                    flags: 0,
                    tick: self.tick,
                    a: top_a as u32,
                    b: top_b as u32,
                    c: 0,
                    value: sim,
                    extra: 0.0,
                });
            }
        }

        let before = self.delta_edges.len();
        for edge in &mut self.delta_edges {
            if edge.utility < 0.02 {
                edge.weak_ticks = edge.weak_ticks.saturating_add(1);
            } else {
                edge.weak_ticks = 0;
            }
        }
        self.delta_edges.retain(|edge| edge.weak_ticks < 500);
        let pruned = before.saturating_sub(self.delta_edges.len());
        if pruned > 0 {
            self.stats.prune_count = self.stats.prune_count.saturating_add(pruned as u64);
            outgoing_mutations.push(MutationRecord {
                kind: MutationKind::Prune,
                flags: 0,
                tick: self.tick,
                a: pruned as u32,
                b: 0,
                c: 0,
                value: 0.0,
                extra: 0.0,
            });
        }
    }

    fn ensure_symbol_node(&mut self, ch: char) -> u32 {
        if let Some(existing) = self.char_nodes.get(&(ch as u32)).copied() {
            return existing;
        }
        let id = self.add_assembly(Some(ch));
        self.char_nodes.insert(ch as u32, id);
        id
    }

    fn add_assembly(&mut self, symbol: Option<char>) -> u32 {
        let id = self.assemblies.len() as u32;
        self.assemblies.push(Assembly {
            id,
            symbol,
            activation: 0.0,
            stability: 0.05,
            support_count: 0,
            novelty: 1.0,
            last_tick: self.tick,
        });
        self.activations.push(0.0);
        self.csr
            .row_ptr
            .push(self.csr.row_ptr.last().copied().unwrap_or(0));
        id
    }

    fn find_or_create_delta_edge(&mut self, from: u32, to: u32) -> usize {
        for (idx, edge) in self.delta_edges.iter().enumerate() {
            if edge.from == from && edge.to == to {
                return idx;
            }
        }
        self.delta_edges.push(DeltaEdge {
            from,
            to,
            weight: 0.0,
            utility: 0.5,
            weak_ticks: 0,
        });
        self.delta_edges.len() - 1
    }

    fn plan_response(&mut self, memory: &MemoryState) {
        if !self.pending_output.is_empty() {
            return;
        }
        let hypotheses = memory.rank_hypotheses(&self.recent_user_buffer, 6);
        if hypotheses.is_empty() {
            self.emit_response_text("pode detalhar melhor?\n");
            return;
        }

        let choice_idx = self.pick_hypothesis_index(&hypotheses);
        let picked = hypotheses
            .get(choice_idx)
            .cloned()
            .unwrap_or_else(|| hypotheses[0].clone());

        if picked.score < 0.10 {
            self.emit_response_text("tenho mais de uma leitura, pode especificar um pouco?\n");
            return;
        }

        let response = self.build_associative_response(&picked, &hypotheses);
        self.emit_response_text(&response);
    }

    fn pick_hypothesis_index(&mut self, hypotheses: &[MemoryHypothesis]) -> usize {
        match self.mode {
            RunMode::Deterministic => 0,
            RunMode::Stochastic => {
                let mut weights = Vec::with_capacity(hypotheses.len());
                for item in hypotheses {
                    weights.push(item.score.max(0.0) + 0.01);
                }
                let total: f32 = weights.iter().sum();
                if total <= 0.0 {
                    return 0;
                }
                let mut r = (self.rng.next_u64() as f64 / u64::MAX as f64) as f32 * total;
                for (idx, w) in weights.iter().enumerate() {
                    if r <= *w {
                        return idx;
                    }
                    r -= *w;
                }
                0
            }
        }
    }

    fn build_associative_response(
        &self,
        picked: &MemoryHypothesis,
        hypotheses: &[MemoryHypothesis],
    ) -> String {
        let mut out = String::new();
        match picked.kind {
            NodeKind::Episode => {
                out.push_str("linha dominante: ");
                out.push_str(&picked.label);
            }
            _ => {
                out.push_str("associacao dominante: ");
                out.push_str(&picked.label);
            }
        }
        if hypotheses.len() > 1 {
            let mut alts = Vec::new();
            for alt in hypotheses.iter().skip(1).take(2) {
                if alt.label != picked.label {
                    alts.push(alt.label.clone());
                }
            }
            if !alts.is_empty() {
                out.push_str(" | alternativas: ");
                out.push_str(&alts.join(" | "));
            }
        }
        out.push('\n');
        out
    }

    pub fn top_active_assemblies(&self, n: usize) -> Vec<&Assembly> {
        let mut refs: Vec<&Assembly> = self.assemblies.iter().collect();
        refs.sort_by(|a, b| b.activation.abs().total_cmp(&a.activation.abs()));
        refs.into_iter().take(n).collect()
    }

    pub fn ensure_assembly_capacity(&mut self, target_len: usize) {
        while self.assemblies.len() < target_len {
            self.add_assembly(None);
        }
    }

    pub fn set_delta_edge_weight(&mut self, from: u32, to: u32, weight: f32) {
        let idx = self.find_or_create_delta_edge(from, to);
        self.delta_edges[idx].weight = weight;
    }

    fn emit_response_text(&mut self, text: &str) {
        for ch in text.chars() {
            self.pending_output.push_back(ch);
        }
    }
}

#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

fn canonical_char(ch: char) -> char {
    let lowered = ch.to_lowercase().next().unwrap_or(ch);
    if lowered.is_control() && lowered != '\n' {
        ' '
    } else {
        lowered
    }
}
