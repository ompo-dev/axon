use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::AxonError;

#[derive(Clone, Debug)]
pub struct Concept {
    pub id: u32,
    pub lemma: String,
    pub canonical: String,
    pub definition: String,
    pub recurrence: u32,
    pub connectivity: f32,
    pub stability: f32,
}

#[derive(Clone, Debug)]
pub struct SemanticState {
    pub concepts: Vec<Concept>,
    pub index: HashMap<String, u32>,
    pub char_to_concepts: HashMap<u32, Vec<u32>>,
    pub promotions: u64,
}

impl SemanticState {
    pub fn new() -> Self {
        Self {
            concepts: Vec::new(),
            index: HashMap::new(),
            char_to_concepts: HashMap::new(),
            promotions: 0,
        }
    }

    pub fn ingest_txt_file(&mut self, path: &Path) -> Result<usize, AxonError> {
        let raw = fs::read_to_string(path)?;
        self.ingest_txt(&raw)
    }

    pub fn ingest_txt(&mut self, raw: &str) -> Result<usize, AxonError> {
        let entries = parse_structured_entries(raw);
        let mut inserted = 0usize;
        for (lemma, definition) in entries {
            if self.add_or_update_concept(&lemma, &definition) {
                inserted = inserted.saturating_add(1);
            }
        }
        Ok(inserted)
    }

    pub fn add_or_update_concept(&mut self, lemma: &str, definition: &str) -> bool {
        let canonical = canonicalize(lemma);
        if canonical.is_empty() {
            return false;
        }
        if let Some(existing) = self.index.get(&canonical).copied() {
            if let Some(concept) = self.concepts.get_mut(existing as usize) {
                concept.definition = merge_definition(&concept.definition, definition);
                concept.recurrence = concept.recurrence.saturating_add(1);
                concept.stability = (concept.stability + 0.01).min(1.0);
                return false;
            }
        }
        let id = self.concepts.len() as u32;
        let concept = Concept {
            id,
            lemma: lemma.to_string(),
            canonical: canonical.clone(),
            definition: definition.to_string(),
            recurrence: 1,
            connectivity: 0.0,
            stability: 0.1,
        };
        for code in canonical.chars().map(|ch| ch as u32) {
            self.char_to_concepts.entry(code).or_default().push(id);
        }
        self.index.insert(canonical, id);
        self.concepts.push(concept);
        true
    }

    pub fn concept_boost_for_char(&self, ch: char) -> f32 {
        let key = ch as u32;
        self.char_to_concepts
            .get(&key)
            .map(|ids| ids.len() as f32 * 0.01)
            .unwrap_or(0.0)
    }

    pub fn reinforce_from_context(&mut self, context: &str) {
        let canon = canonicalize(context);
        if canon.is_empty() {
            return;
        }
        for concept in &mut self.concepts {
            if canon.contains(&concept.canonical) {
                concept.recurrence = concept.recurrence.saturating_add(1);
                concept.connectivity = (concept.connectivity + 0.02).min(1.0);
                concept.stability = (concept.stability + 0.015).min(1.0);
            } else {
                concept.connectivity *= 0.999;
            }
        }
        self.promotions = self.promotions.saturating_add(1);
    }

    pub fn top_concepts(&self, n: usize) -> Vec<&Concept> {
        let mut refs: Vec<&Concept> = self.concepts.iter().collect();
        refs.sort_by(|a, b| {
            let sa = a.stability + a.connectivity + (a.recurrence as f32 * 0.001);
            let sb = b.stability + b.connectivity + (b.recurrence as f32 * 0.001);
            sb.total_cmp(&sa)
        });
        refs.into_iter().take(n).collect()
    }
}

pub fn canonicalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        let lower = ch.to_lowercase().next().unwrap_or(ch);
        if lower.is_alphanumeric() || lower.is_whitespace() || lower == '-' {
            out.push(lower);
        } else {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn merge_definition(existing: &str, incoming: &str) -> String {
    if incoming.trim().is_empty() {
        return existing.to_string();
    }
    if existing.contains(incoming.trim()) {
        return existing.to_string();
    }
    let mut merged = String::with_capacity(existing.len() + incoming.len() + 2);
    merged.push_str(existing);
    if !existing.ends_with('\n') {
        merged.push('\n');
    }
    merged.push_str(incoming.trim());
    merged
}

fn parse_structured_entries(raw: &str) -> Vec<(String, String)> {
    let mut entries = Vec::new();
    let mut current = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                if let Some(parsed) = parse_entry_block(&current.join("\n")) {
                    entries.push(parsed);
                }
                current.clear();
            }
            continue;
        }
        current.push(line.to_string());
    }
    if !current.is_empty() {
        if let Some(parsed) = parse_entry_block(&current.join("\n")) {
            entries.push(parsed);
        }
    }
    entries
}

fn parse_entry_block(block: &str) -> Option<(String, String)> {
    let mut lines = block.lines();
    let first = lines.next()?.trim();
    if first.is_empty() {
        return None;
    }
    if let Some((lemma, definition)) = first.split_once(':') {
        let rest = lines.collect::<Vec<_>>().join("\n");
        let mut full_def = definition.trim().to_string();
        if !rest.trim().is_empty() {
            full_def.push('\n');
            full_def.push_str(rest.trim());
        }
        return Some((lemma.trim().to_string(), full_def));
    }
    let definition = lines.collect::<Vec<_>>().join("\n");
    Some((first.to_string(), definition))
}
