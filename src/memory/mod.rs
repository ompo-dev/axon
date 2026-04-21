use std::collections::{HashMap, VecDeque};

use crate::storage::{MutationKind, MutationRecord};

const TEMP_ALPHA: f32 = 0.50;
const TEMP_BETA: f32 = 0.30;
const TEMP_GAMMA: f32 = 0.20;
const TEMP_TAU_TICKS: f32 = 10_000.0;
const TICKS_PER_DAY: u64 = 8_640_000; // 100 Hz * 60 * 60 * 24
const MAX_PROP_DEPTH: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum NodeKind {
    Concept = 1,
    Episode = 2,
    Temporal = 3,
    Cue = 4,
}

impl NodeKind {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Concept),
            2 => Some(Self::Episode),
            3 => Some(Self::Temporal),
            4 => Some(Self::Cue),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EdgeKind {
    CoActivation = 1,
    TemporalBinding = 2,
    ContextBinding = 3,
    Contrast = 4,
    Correction = 5,
}

impl EdgeKind {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::CoActivation),
            2 => Some(Self::TemporalBinding),
            3 => Some(Self::ContextBinding),
            4 => Some(Self::Contrast),
            5 => Some(Self::Correction),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryNode {
    pub id: u64,
    pub kind: NodeKind,
    pub label: String,
    pub last_tick: u64,
    pub recurrence: u32,
    pub frequency: f32,
    pub salience: f32,
    pub temperature: f32,
    pub amplitude: f32,
    pub phase: f32,
    pub omega: f32,
}

#[derive(Clone, Debug)]
pub struct MemoryEdge {
    pub from: u64,
    pub to: u64,
    pub kind: EdgeKind,
    pub strength: f32,
    pub last_tick: u64,
    pub recurrence: u32,
    pub frequency: f32,
    pub salience: f32,
    pub temperature: f32,
    pub delay: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct TemporalAnchor {
    pub cue: String,
    pub node_id: u64,
    pub last_rebind_tick: u64,
}

#[derive(Clone, Debug)]
pub struct MemoryHypothesis {
    pub node_id: u64,
    pub label: String,
    pub kind: NodeKind,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct MemoryState {
    pub nodes: Vec<MemoryNode>,
    pub edges: Vec<MemoryEdge>,
    pub temporal_anchors: Vec<TemporalAnchor>,
    pub next_node_id: u64,
    node_index: HashMap<String, u64>,
    adjacency: HashMap<u64, Vec<usize>>,
    recent_cues: VecDeque<String>,
}

pub type UnifiedGraph = MemoryState;

impl MemoryState {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            temporal_anchors: Vec::new(),
            next_node_id: 1,
            node_index: HashMap::new(),
            adjacency: HashMap::new(),
            recent_cues: VecDeque::with_capacity(32),
        }
    }

    pub fn observe_text(&mut self, tick: u64, text: &str, mutations: &mut Vec<MutationRecord>) {
        let words = canonical_words(text);
        if words.is_empty() {
            return;
        }
        self.push_recent_cues(&words);

        let episode_label = canonical_sentence(text);
        let episode_id =
            self.get_or_create_node(NodeKind::Episode, &episode_label, tick, 1.0, mutations);

        let mut concept_ids = Vec::with_capacity(words.len());
        for word in &words {
            let concept_id =
                self.get_or_create_node(NodeKind::Concept, word, tick, 0.85, mutations);
            concept_ids.push(concept_id);
            self.link_bidirectional(
                concept_id,
                episode_id,
                EdgeKind::ContextBinding,
                0.08,
                tick,
                0.8,
                mutations,
            );
        }

        for pair in concept_ids.windows(2) {
            let from = pair[0];
            let to = pair[1];
            self.link_bidirectional(
                from,
                to,
                EdgeKind::CoActivation,
                0.07,
                tick,
                0.75,
                mutations,
            );
        }

        for cue in words.iter().take(2) {
            let cue_id = self.get_or_create_node(NodeKind::Cue, cue, tick, 0.70, mutations);
            if let Some(target) = concept_ids.first().copied() {
                self.link_directed(
                    cue_id,
                    target,
                    EdgeKind::ContextBinding,
                    0.05,
                    tick,
                    0.65,
                    mutations,
                );
            }
        }

        let day_bucket = tick / TICKS_PER_DAY;
        for cue in words.iter().filter(|word| is_temporal_cue(word)) {
            let temporal_label = format!("{cue}@{day_bucket}");
            let temporal_id =
                self.get_or_create_node(NodeKind::Temporal, &temporal_label, tick, 1.0, mutations);
            if let Some(anchor_idx) = self.temporal_anchors.iter().position(|a| a.cue == *cue) {
                let previous = self.temporal_anchors[anchor_idx].node_id;
                if previous != temporal_id {
                    self.temporal_anchors[anchor_idx].node_id = temporal_id;
                    self.temporal_anchors[anchor_idx].last_rebind_tick = tick;
                    mutations.push(MutationRecord {
                        kind: MutationKind::TemporalRebind,
                        flags: 0,
                        tick,
                        a: previous as u32,
                        b: temporal_id as u32,
                        c: cue.len() as u32,
                        value: 1.0,
                        extra: day_bucket as f32,
                    });
                }
            } else {
                self.temporal_anchors.push(TemporalAnchor {
                    cue: cue.clone(),
                    node_id: temporal_id,
                    last_rebind_tick: tick,
                });
                mutations.push(MutationRecord {
                    kind: MutationKind::TemporalRebind,
                    flags: 0,
                    tick,
                    a: 0,
                    b: temporal_id as u32,
                    c: cue.len() as u32,
                    value: 1.0,
                    extra: day_bucket as f32,
                });
            }

            for concept_id in &concept_ids {
                self.link_bidirectional(
                    temporal_id,
                    *concept_id,
                    EdgeKind::TemporalBinding,
                    0.08,
                    tick,
                    0.9,
                    mutations,
                );
            }
        }

        self.decay_to_tick(tick, mutations);
    }

    pub fn ingest_dictionary_entry(
        &mut self,
        tick: u64,
        lemma: &str,
        definition: &str,
        mutations: &mut Vec<MutationRecord>,
    ) -> bool {
        let canon_lemma = canonical_sentence(lemma);
        if canon_lemma.is_empty() {
            return false;
        }

        let concept_id =
            self.get_or_create_node(NodeKind::Concept, &canon_lemma, tick, 0.95, mutations);
        let def_episode_label = format!("def:{canon_lemma}");
        let def_episode_id =
            self.get_or_create_node(NodeKind::Episode, &def_episode_label, tick, 0.75, mutations);

        self.link_bidirectional(
            concept_id,
            def_episode_id,
            EdgeKind::ContextBinding,
            0.10,
            tick,
            0.85,
            mutations,
        );

        let mut words = canonical_words(definition);
        words.retain(|w| !w.is_empty() && w != &canon_lemma);
        let mut inserted_any = false;
        for word in words.into_iter().take(48) {
            let related_id =
                self.get_or_create_node(NodeKind::Concept, &word, tick, 0.7, mutations);
            self.link_bidirectional(
                concept_id,
                related_id,
                EdgeKind::CoActivation,
                0.05,
                tick,
                0.7,
                mutations,
            );
            self.link_bidirectional(
                related_id,
                def_episode_id,
                EdgeKind::ContextBinding,
                0.04,
                tick,
                0.65,
                mutations,
            );
            inserted_any = true;
        }

        inserted_any
    }

    pub fn decay_to_tick(&mut self, tick: u64, mutations: &mut Vec<MutationRecord>) {
        for node in &mut self.nodes {
            let prev_tick = node.last_tick;
            let prev_temp = node.temperature;
            let dt = tick.saturating_sub(prev_tick) as f32;
            node.temperature = temperature_formula(prev_tick, tick, node.frequency, node.salience);
            let amp_decay = (-dt / (TEMP_TAU_TICKS * 0.6)).exp();
            node.amplitude = (node.amplitude * amp_decay + node.temperature * 0.12).clamp(0.0, 1.0);
            node.phase = wrap_phase(node.phase + node.omega * dt * 0.001);
            let delta = (node.temperature - prev_temp).abs();
            if delta >= 0.02 {
                mutations.push(MutationRecord {
                    kind: MutationKind::TempUpdate,
                    flags: 0,
                    tick,
                    a: node.id as u32,
                    b: node.kind as u32,
                    c: 0,
                    value: node.temperature,
                    extra: delta,
                });
            }
        }

        let mut pruned = 0usize;
        for edge in &mut self.edges {
            let prev_tick = edge.last_tick;
            edge.temperature = temperature_formula(prev_tick, tick, edge.frequency, edge.salience);
            edge.confidence = (edge.confidence * 0.999).clamp(0.05, 1.0);
            if edge.temperature < 0.02 && edge.recurrence < 2 {
                edge.strength *= 0.90;
                if edge.strength.abs() < 0.005 {
                    edge.strength = 0.0;
                    pruned = pruned.saturating_add(1);
                }
            }
        }

        if pruned > 0 {
            mutations.push(MutationRecord {
                kind: MutationKind::LinkWeaken,
                flags: 0,
                tick,
                a: pruned as u32,
                b: 0,
                c: 0,
                value: 0.0,
                extra: 0.0,
            });
        }
    }

    pub fn apply_correction(
        &mut self,
        tick: u64,
        wrong: &str,
        correct: &str,
        mutations: &mut Vec<MutationRecord>,
    ) {
        let wrong_clean = canonical_sentence(wrong);
        let correct_clean = canonical_sentence(correct);
        if wrong_clean.is_empty() || correct_clean.is_empty() {
            return;
        }
        let wrong_id =
            self.get_or_create_node(NodeKind::Concept, &wrong_clean, tick, 0.95, mutations);
        let correct_id =
            self.get_or_create_node(NodeKind::Concept, &correct_clean, tick, 1.0, mutations);

        self.link_directed(
            wrong_id,
            correct_id,
            EdgeKind::Correction,
            0.20,
            tick,
            1.0,
            mutations,
        );

        let edge_indexes = self
            .adjacency
            .get(&wrong_id)
            .cloned()
            .unwrap_or_else(Vec::new);
        for edge_idx in edge_indexes {
            if let Some(edge) = self.edges.get_mut(edge_idx) {
                if edge.kind == EdgeKind::CoActivation && edge.to != correct_id {
                    let previous = edge.strength;
                    edge.strength *= 0.82;
                    edge.salience *= 0.92;
                    edge.frequency = (edge.frequency + 0.2).min(16.0);
                    edge.confidence = (edge.confidence * 0.95).max(0.05);
                    edge.last_tick = tick;
                    mutations.push(MutationRecord {
                        kind: MutationKind::LinkWeaken,
                        flags: 0,
                        tick,
                        a: edge.from as u32,
                        b: edge.to as u32,
                        c: edge.kind as u32,
                        value: edge.strength,
                        extra: previous,
                    });
                }
            }
        }

        self.link_bidirectional(
            correct_id,
            wrong_id,
            EdgeKind::Contrast,
            0.03,
            tick,
            0.7,
            mutations,
        );

        mutations.push(MutationRecord {
            kind: MutationKind::CorrectionApplied,
            flags: 0,
            tick,
            a: wrong_id as u32,
            b: correct_id as u32,
            c: 0,
            value: 1.0,
            extra: 0.0,
        });
    }

    pub fn recall_for_text(&self, text: &str, limit: usize) -> Vec<String> {
        self.rank_hypotheses(text, limit)
            .into_iter()
            .map(|entry| entry.label)
            .collect()
    }

    pub fn recall_from_cues(&self, cues: &[String], limit: usize) -> Vec<String> {
        self.rank_hypotheses(&cues.join(" "), limit)
            .into_iter()
            .map(|entry| entry.label)
            .collect()
    }

    pub fn rank_hypotheses(&self, text: &str, limit: usize) -> Vec<MemoryHypothesis> {
        let cues = canonical_words(text);
        let mut seed_ids = Vec::new();
        for cue in &cues {
            let key = node_key(NodeKind::Concept, cue);
            if let Some(id) = self.node_index.get(&key).copied() {
                seed_ids.push(id);
            }
            let cue_key = node_key(NodeKind::Cue, cue);
            if let Some(id) = self.node_index.get(&cue_key).copied() {
                seed_ids.push(id);
            }
            if let Some(anchor_id) = self
                .temporal_anchors
                .iter()
                .find(|anchor| anchor.cue == *cue)
                .map(|anchor| anchor.node_id)
            {
                seed_ids.push(anchor_id);
            }
        }
        if seed_ids.is_empty() {
            for recent in self.recent_cues.iter().rev().take(4) {
                if let Some(id) = self
                    .node_index
                    .get(&node_key(NodeKind::Concept, recent))
                    .copied()
                {
                    seed_ids.push(id);
                }
            }
        }
        if seed_ids.is_empty() {
            return Vec::new();
        }

        let mut excitation: HashMap<u64, f32> = HashMap::new();
        let mut inhibition: HashMap<u64, f32> = HashMap::new();
        let mut frontier: VecDeque<(u64, f32, u8)> =
            seed_ids.iter().map(|id| (*id, 1.0, 0)).collect();

        while let Some((node_id, energy, depth)) = frontier.pop_front() {
            let Some(source) = self.node_by_id(node_id) else {
                continue;
            };

            if energy >= 0.0 {
                *excitation.entry(node_id).or_insert(0.0) += energy;
            } else {
                *inhibition.entry(node_id).or_insert(0.0) += -energy;
            }

            if depth >= MAX_PROP_DEPTH {
                continue;
            }
            let source_drive =
                energy.abs() * source.temperature.max(0.05) * source.amplitude.max(0.05);
            if source_drive < 0.01 {
                continue;
            }

            let Some(edges) = self.adjacency.get(&node_id) else {
                continue;
            };

            for edge_idx in edges {
                let Some(edge) = self.edges.get(*edge_idx) else {
                    continue;
                };
                if edge.temperature < 0.02 || edge.confidence < 0.05 {
                    continue;
                }
                let Some(target) = self.node_by_id(edge.to) else {
                    continue;
                };
                let phase_delta = source.phase - target.phase - edge.delay;
                let interference = phase_delta.cos();
                let wave = source_drive
                    * signed_weight(edge.kind, edge.strength)
                    * edge.temperature
                    * edge.confidence
                    * kind_gain(edge.kind)
                    * interference;
                if wave.abs() < 0.008 {
                    continue;
                }
                frontier.push_back((edge.to, wave * 0.78, depth + 1));
            }
        }

        let mut scored = Vec::new();
        for node in &self.nodes {
            if node.kind != NodeKind::Concept && node.kind != NodeKind::Episode {
                continue;
            }
            let e = excitation.get(&node.id).copied().unwrap_or(0.0);
            let i = inhibition.get(&node.id).copied().unwrap_or(0.0);
            let score = e - (i * 0.62)
                + node.temperature * 0.08
                + node.salience * 0.05
                + node.amplitude * 0.05;
            if score > 0.02 {
                scored.push(MemoryHypothesis {
                    node_id: node.id,
                    label: node.label.clone(),
                    kind: node.kind,
                    score,
                });
            }
        }
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut out = Vec::new();
        for item in scored {
            if !out
                .iter()
                .any(|existing: &MemoryHypothesis| existing.label == item.label)
            {
                out.push(item);
            }
            if out.len() >= limit {
                break;
            }
        }
        out
    }

    pub fn top_hot_nodes(&self, n: usize) -> Vec<&MemoryNode> {
        let mut refs: Vec<&MemoryNode> = self.nodes.iter().collect();
        refs.sort_by(|a, b| {
            let sa = a.temperature * 0.7 + a.amplitude * 0.3;
            let sb = b.temperature * 0.7 + b.amplitude * 0.3;
            sb.total_cmp(&sa)
        });
        refs.into_iter().take(n).collect()
    }

    pub fn rebuild_indexes(&mut self) {
        self.node_index.clear();
        self.adjacency.clear();
        for node in &self.nodes {
            self.node_index
                .insert(node_key(node.kind, &node.label), node.id);
            self.next_node_id = self.next_node_id.max(node.id.saturating_add(1));
        }
        for (idx, edge) in self.edges.iter().enumerate() {
            self.adjacency.entry(edge.from).or_default().push(idx);
        }
    }

    pub fn ingest_legacy_episode(
        &mut self,
        tick: u64,
        trace: &str,
        salience: f32,
        recall_score: f32,
    ) {
        let mut sink = Vec::new();
        self.observe_text(tick, trace, &mut sink);
        let score = (salience + recall_score).clamp(0.0, 1.0);
        for node in self.nodes.iter_mut().rev().take(8) {
            node.salience = node.salience.max(score);
            node.temperature = node.temperature.max(score);
            node.amplitude = node.amplitude.max(score * 0.9);
        }
    }

    fn get_or_create_node(
        &mut self,
        kind: NodeKind,
        label: &str,
        tick: u64,
        salience: f32,
        mutations: &mut Vec<MutationRecord>,
    ) -> u64 {
        let key = node_key(kind, label);
        if let Some(id) = self.node_index.get(&key).copied() {
            if let Some(node) = self.node_by_id_mut(id) {
                node.recurrence = node.recurrence.saturating_add(1);
                node.frequency = (node.frequency + 1.0).min(64.0);
                node.salience = node.salience.max(salience);
                let prev_tick = node.last_tick;
                node.last_tick = tick;
                node.temperature =
                    temperature_formula(prev_tick, tick, node.frequency, node.salience);
                node.amplitude =
                    ((1.0 - 0.18) * node.amplitude + 0.18 * node.temperature).clamp(0.0, 1.0);
                node.phase = wrap_phase(node.phase + node.omega * 0.35 + salience * 0.05);
                mutations.push(MutationRecord {
                    kind: MutationKind::TempUpdate,
                    flags: 0,
                    tick,
                    a: id as u32,
                    b: kind as u32,
                    c: 0,
                    value: node.temperature,
                    extra: node.frequency,
                });
            }
            return id;
        }

        let id = self.next_node_id;
        self.next_node_id = self.next_node_id.saturating_add(1);
        let node = MemoryNode {
            id,
            kind,
            label: label.to_string(),
            last_tick: tick,
            recurrence: 1,
            frequency: 1.0,
            salience,
            temperature: temperature_formula(tick, tick, 1.0, salience),
            amplitude: salience.clamp(0.1, 1.0),
            phase: 0.0,
            omega: base_omega(label),
        };
        self.nodes.push(node);
        self.node_index.insert(key, id);
        id
    }

    fn link_bidirectional(
        &mut self,
        from: u64,
        to: u64,
        kind: EdgeKind,
        strength_delta: f32,
        tick: u64,
        salience: f32,
        mutations: &mut Vec<MutationRecord>,
    ) {
        self.link_directed(from, to, kind, strength_delta, tick, salience, mutations);
        self.link_directed(to, from, kind, strength_delta, tick, salience, mutations);
    }

    fn link_directed(
        &mut self,
        from: u64,
        to: u64,
        kind: EdgeKind,
        strength_delta: f32,
        tick: u64,
        salience: f32,
        mutations: &mut Vec<MutationRecord>,
    ) {
        if let Some(idx) = self.find_edge(from, to, kind) {
            if let Some(edge) = self.edges.get_mut(idx) {
                let previous = edge.strength;
                edge.recurrence = edge.recurrence.saturating_add(1);
                edge.frequency = (edge.frequency + 1.0).min(64.0);
                edge.salience = edge.salience.max(salience);
                let prev_tick = edge.last_tick;
                edge.last_tick = tick;
                let signed_delta = if kind == EdgeKind::Contrast {
                    -strength_delta.abs()
                } else {
                    strength_delta.abs()
                };
                edge.strength = (edge.strength + signed_delta).clamp(-1.5, 1.5);
                edge.confidence =
                    (edge.confidence * 0.98 + salience * 0.02 + edge.frequency * 0.001)
                        .clamp(0.05, 1.0);
                edge.temperature =
                    temperature_formula(prev_tick, tick, edge.frequency, edge.salience);
                mutations.push(MutationRecord {
                    kind: MutationKind::LinkStrengthen,
                    flags: 0,
                    tick,
                    a: from as u32,
                    b: to as u32,
                    c: kind as u32,
                    value: edge.strength,
                    extra: previous,
                });
            }
            return;
        }

        let raw_strength = if kind == EdgeKind::Contrast {
            -strength_delta.abs()
        } else {
            strength_delta.abs()
        };
        let edge = MemoryEdge {
            from,
            to,
            kind,
            strength: raw_strength.clamp(-1.5, 1.5),
            last_tick: tick,
            recurrence: 1,
            frequency: 1.0,
            salience,
            temperature: temperature_formula(tick, tick, 1.0, salience),
            delay: edge_default_delay(kind),
            confidence: (0.45 + salience * 0.5).clamp(0.05, 1.0),
        };
        let idx = self.edges.len();
        self.edges.push(edge);
        self.adjacency.entry(from).or_default().push(idx);
        mutations.push(MutationRecord {
            kind: MutationKind::LinkCreate,
            flags: 0,
            tick,
            a: from as u32,
            b: to as u32,
            c: kind as u32,
            value: raw_strength,
            extra: salience,
        });
    }

    fn find_edge(&self, from: u64, to: u64, kind: EdgeKind) -> Option<usize> {
        let indexes = self.adjacency.get(&from)?;
        for idx in indexes {
            let edge = self.edges.get(*idx)?;
            if edge.to == to && edge.kind == kind {
                return Some(*idx);
            }
        }
        None
    }

    fn node_by_id(&self, id: u64) -> Option<&MemoryNode> {
        if id == 0 {
            return None;
        }
        self.nodes.get((id - 1) as usize)
    }

    fn node_by_id_mut(&mut self, id: u64) -> Option<&mut MemoryNode> {
        if id == 0 {
            return None;
        }
        self.nodes.get_mut((id - 1) as usize)
    }

    fn push_recent_cues(&mut self, words: &[String]) {
        for word in words.iter().take(4) {
            if self.recent_cues.len() >= 32 {
                self.recent_cues.pop_front();
            }
            self.recent_cues.push_back(word.clone());
        }
    }
}

fn node_key(kind: NodeKind, label: &str) -> String {
    format!("{}:{}", kind as u8, label)
}

fn canonical_words(text: &str) -> Vec<String> {
    canonical_sentence(text)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn canonical_sentence(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        let lower = ch.to_lowercase().next().unwrap_or(ch);
        if lower.is_alphanumeric() || lower == '-' || lower.is_whitespace() {
            out.push(lower);
        } else {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_temporal_cue(word: &str) -> bool {
    matches!(
        word,
        "hoje" | "ontem" | "amanha" | "agora" | "depois" | "antes"
    )
}

fn kind_gain(kind: EdgeKind) -> f32 {
    match kind {
        EdgeKind::CoActivation => 1.00,
        EdgeKind::TemporalBinding => 1.10,
        EdgeKind::ContextBinding => 0.95,
        EdgeKind::Contrast => 0.90,
        EdgeKind::Correction => 1.25,
    }
}

fn signed_weight(kind: EdgeKind, weight: f32) -> f32 {
    match kind {
        EdgeKind::Contrast => -weight.abs(),
        _ => weight,
    }
}

fn edge_default_delay(kind: EdgeKind) -> f32 {
    match kind {
        EdgeKind::CoActivation => 0.02,
        EdgeKind::TemporalBinding => 0.15,
        EdgeKind::ContextBinding => 0.06,
        EdgeKind::Contrast => 0.18,
        EdgeKind::Correction => 0.10,
    }
}

fn base_omega(label: &str) -> f32 {
    let mut h = 0x811C9DC5u32;
    for b in label.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    let norm = (h % 1000) as f32 / 1000.0;
    0.012 + norm * 0.036
}

fn wrap_phase(value: f32) -> f32 {
    let two_pi = std::f32::consts::PI * 2.0;
    let mut v = value % two_pi;
    if v > std::f32::consts::PI {
        v -= two_pi;
    } else if v < -std::f32::consts::PI {
        v += two_pi;
    }
    v
}

fn temperature_formula(last_tick: u64, now_tick: u64, frequency: f32, salience: f32) -> f32 {
    let dt = now_tick.saturating_sub(last_tick) as f32;
    let recency = (-dt / TEMP_TAU_TICKS).exp();
    let freq_term = 1.0 - (-frequency * 0.22).exp();
    let salience_term = salience.clamp(0.0, 1.0);
    (TEMP_ALPHA * recency + TEMP_BETA * freq_term + TEMP_GAMMA * salience_term).clamp(0.0, 1.0)
}
