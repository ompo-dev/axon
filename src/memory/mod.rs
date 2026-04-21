use std::collections::{HashMap, VecDeque};

use crate::storage::{MutationKind, MutationRecord};

const TEMP_ALPHA: f32 = 0.50;
const TEMP_BETA: f32 = 0.30;
const TEMP_GAMMA: f32 = 0.20;
const TEMP_TAU_TICKS: f32 = 10_000.0;
const TICKS_PER_DAY: u64 = 8_640_000; // 100 Hz * 60 * 60 * 24

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
}

#[derive(Clone, Debug)]
pub struct TemporalAnchor {
    pub cue: String,
    pub node_id: u64,
    pub last_rebind_tick: u64,
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

    pub fn observe_text(
        &mut self,
        tick: u64,
        text: &str,
        mutations: &mut Vec<MutationRecord>,
    ) {
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
            let concept_id = self.get_or_create_node(NodeKind::Concept, word, tick, 0.8, mutations);
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
                0.06,
                tick,
                0.7,
                mutations,
            );
        }

        for cue in words.iter().take(2) {
            let cue_id = self.get_or_create_node(NodeKind::Cue, cue, tick, 0.65, mutations);
            if let Some(target) = concept_ids.first().copied() {
                self.link_directed(
                    cue_id,
                    target,
                    EdgeKind::ContextBinding,
                    0.05,
                    tick,
                    0.6,
                    mutations,
                );
            }
        }

        let day_bucket = tick / TICKS_PER_DAY;
        for cue in words.iter().filter(|word| is_temporal_cue(word)) {
            let temporal_label = format!("{cue}@{day_bucket}");
            let temporal_id = self.get_or_create_node(
                NodeKind::Temporal,
                &temporal_label,
                tick,
                1.0,
                mutations,
            );
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
                    0.07,
                    tick,
                    0.9,
                    mutations,
                );
            }
        }

        self.decay_to_tick(tick, mutations);
    }

    pub fn decay_to_tick(&mut self, tick: u64, mutations: &mut Vec<MutationRecord>) {
        for node in &mut self.nodes {
            let next = temperature_formula(node.last_tick, tick, node.frequency, node.salience);
            let delta = (next - node.temperature).abs();
            node.temperature = next;
            if delta >= 0.02 {
                mutations.push(MutationRecord {
                    kind: MutationKind::TempUpdate,
                    flags: 0,
                    tick,
                    a: node.id as u32,
                    b: node.kind as u32,
                    c: 0,
                    value: next,
                    extra: delta,
                });
            }
        }

        let mut pruned = 0usize;
        for edge in &mut self.edges {
            let next = temperature_formula(edge.last_tick, tick, edge.frequency, edge.salience);
            edge.temperature = next;
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
        let cues = canonical_words(text);
        self.recall_from_cues(&cues, limit)
    }

    pub fn recall_from_cues(&self, cues: &[String], limit: usize) -> Vec<String> {
        let mut seed_ids = Vec::new();
        for cue in cues {
            let key = node_key(NodeKind::Concept, cue);
            if let Some(id) = self.node_index.get(&key).copied() {
                seed_ids.push(id);
            }
            let cue_key = node_key(NodeKind::Cue, cue);
            if let Some(id) = self.node_index.get(&cue_key).copied() {
                seed_ids.push(id);
            }
        }
        if seed_ids.is_empty() {
            for recent in self.recent_cues.iter().rev().take(3) {
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

        let mut scores: HashMap<u64, f32> = HashMap::new();
        let mut frontier: Vec<(u64, f32, u8)> = seed_ids.iter().map(|id| (*id, 1.0, 0)).collect();
        while let Some((node_id, energy, depth)) = frontier.pop() {
            *scores.entry(node_id).or_insert(0.0) += energy;
            if depth >= 3 || energy < 0.02 {
                continue;
            }
            let Some(edges) = self.adjacency.get(&node_id) else {
                continue;
            };
            for edge_idx in edges {
                let Some(edge) = self.edges.get(*edge_idx) else {
                    continue;
                };
                if edge.strength == 0.0 || edge.temperature < 0.03 {
                    continue;
                }
                let gain = kind_gain(edge.kind);
                let propagated = energy * edge.strength.max(0.0) * edge.temperature * gain;
                if propagated < 0.015 {
                    continue;
                }
                frontier.push((edge.to, propagated * 0.72, depth + 1));
            }
        }

        let mut scored_labels: Vec<(String, f32)> = scores
            .into_iter()
            .filter_map(|(node_id, score)| {
                let node = self.node_by_id(node_id)?;
                if node.kind == NodeKind::Concept || node.kind == NodeKind::Episode {
                    Some((node.label.clone(), score * node.temperature.max(0.05)))
                } else {
                    None
                }
            })
            .collect();
        scored_labels.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut out = Vec::new();
        for (label, _) in scored_labels {
            if !out.iter().any(|existing| existing == &label) {
                out.push(label);
            }
            if out.len() >= limit {
                break;
            }
        }
        out
    }

    pub fn top_hot_nodes(&self, n: usize) -> Vec<&MemoryNode> {
        let mut refs: Vec<&MemoryNode> = self.nodes.iter().collect();
        refs.sort_by(|a, b| b.temperature.total_cmp(&a.temperature));
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
                node.last_tick = tick;
                node.temperature =
                    temperature_formula(node.last_tick, tick, node.frequency, node.salience);
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
                edge.last_tick = tick;
                edge.strength = (edge.strength + strength_delta).clamp(0.0, 1.5);
                edge.temperature =
                    temperature_formula(edge.last_tick, tick, edge.frequency, edge.salience);
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

        let edge = MemoryEdge {
            from,
            to,
            kind,
            strength: strength_delta.clamp(0.0, 1.5),
            last_tick: tick,
            recurrence: 1,
            frequency: 1.0,
            salience,
            temperature: temperature_formula(tick, tick, 1.0, salience),
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
            value: strength_delta,
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
        EdgeKind::TemporalBinding => 1.08,
        EdgeKind::ContextBinding => 0.94,
        EdgeKind::Contrast => 0.55,
        EdgeKind::Correction => 1.20,
    }
}

fn temperature_formula(last_tick: u64, now_tick: u64, frequency: f32, salience: f32) -> f32 {
    let dt = now_tick.saturating_sub(last_tick) as f32;
    let recency = (-dt / TEMP_TAU_TICKS).exp();
    let freq_term = 1.0 - (-frequency * 0.22).exp();
    let salience_term = salience.clamp(0.0, 1.0);
    (TEMP_ALPHA * recency + TEMP_BETA * freq_term + TEMP_GAMMA * salience_term).clamp(0.0, 1.0)
}
