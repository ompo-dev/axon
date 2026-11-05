//! Experimental learned segmentation, POS and projective arc-standard parsing.
//! Reuses the sequence learner/quantizer. No Portuguese vocabulary is hardcoded.
//! Arc-standard reference: Chen & Manning (2014), https://aclanthology.org/D14-1082/.
//! Syntax, margins and predicted relations are not semantics or action authority.
use super::factorized::{Learner, Quantized, random, take};
use super::{Error, MAX_BYTES, hash, unique};
use crate::live_store::checksum;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;

pub const CONTEXT: &str = "axon/dependency-v1";
const MAGIC: &[u8; 8] = b"AXDP0002";
const LEGACY_MAGIC: &[u8; 8] = b"AXDP0001";
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub seed: u64,
    pub buckets: usize,
    pub dimensions: usize,
    pub epochs: usize,
    pub learning_rate: f32,
}
impl Budget {
    fn validate(self) -> Result<(), Error> {
        if self.seed == 0
            || !self.buckets.is_power_of_two()
            || !(256..=8192).contains(&self.buckets)
            || !(8..=32).contains(&self.dimensions)
            || !(1..=32).contains(&self.epochs)
            || !self.learning_rate.is_finite()
            || !(0.001..=1.0).contains(&self.learning_rate)
        {
            return Err(Error::Invalid("dependency budget"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Word {
    pub form: String,
    pub upos: String,
    pub head: usize,
    pub deprel: String,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Surface {
    pub form: String,
    pub words: Vec<String>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub text: String,
    pub surfaces: Vec<Surface>,
    pub words: Vec<Word>,
}
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum Action {
    Shift,
    Left(String),
    Right(String),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Variant {
    words: Vec<String>,
    count: usize,
}
// Keep the published object wire format, but retain sorted contiguous entries.
mod expansion_map {
    use super::Variant;
    use serde::{
        Deserializer, Serializer,
        de::{Error as _, MapAccess, Visitor},
        ser::SerializeMap,
    };
    pub fn serialize<S: Serializer>(
        entries: &[(String, Vec<Variant>)],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(entries.len()))?;
        for (key, value) in entries {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<(String, Vec<Variant>)>, D::Error> {
        struct Entries;
        impl<'de> Visitor<'de> for Entries {
            type Value = Vec<(String, Vec<Variant>)>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("bounded sorted expansion map")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut result: Self::Value = vec![];
                while let Some((key, value)) = map.next_entry::<String, Vec<Variant>>()? {
                    if result.len() >= 1024
                        || result.last().is_some_and(|(previous, _)| previous >= &key)
                    {
                        return Err(A::Error::custom("duplicate/unsorted expansion or quota"));
                    }
                    result.push((key, value));
                }
                Ok(result)
            }
        }
        deserializer.deserialize_map(Entries)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    revision: u8,
    seed: u64,
    buckets: usize,
    dimensions: usize,
    pos: Vec<String>,
    actions: Vec<Action>,
    #[serde(with = "expansion_map")]
    expansions: Vec<(String, Vec<Variant>)>,
}
#[derive(Clone, Debug)]
pub struct Model {
    header: Header,
    input: Quantized,
    output: Quantized,
    bias: Vec<f32>,
}
#[derive(Debug, Serialize)]
pub struct Event {
    pub epoch: usize,
    pub examples: usize,
    pub nonprojective: usize,
    pub updates: usize,
    pub errors: usize,
}
#[derive(Debug, Serialize)]
pub struct Prediction {
    pub words: Vec<Word>,
    pub ambiguous_expansions: usize,
    pub minimum_margin: f32,
    pub work: usize,
}
fn form(s: &str) -> bool {
    !s.is_empty() && s.len() <= 1024 && !s.chars().any(char::is_control)
}
pub(super) fn pos(s: &str) -> bool {
    matches!(
        s,
        "ADJ"
            | "ADP"
            | "ADV"
            | "AUX"
            | "CCONJ"
            | "DET"
            | "INTJ"
            | "NOUN"
            | "NUM"
            | "PART"
            | "PRON"
            | "PROPN"
            | "PUNCT"
            | "SCONJ"
            | "SYM"
            | "VERB"
            | "X"
    )
}
fn rel(s: &str) -> bool {
    let parts = s.split(':').collect::<Vec<_>>();
    (1..=2).contains(&parts.len())
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_lowercase()))
}
pub(super) fn validate_words(words: &[Word]) -> Result<(), Error> {
    if words.is_empty() || words.len() > 256 || words.iter().filter(|w| w.head == 0).count() != 1 {
        return Err(Error::Invalid("dependency words or root"));
    }
    for (i, w) in words.iter().enumerate() {
        if !form(&w.form)
            || !pos(&w.upos)
            || !rel(&w.deprel)
            || w.head > words.len()
            || w.head == i + 1
            || (w.head == 0) != (w.deprel == "root")
        {
            return Err(Error::Invalid("dependency supervision"));
        }
    }
    for (i, w) in words.iter().enumerate() {
        let mut seen = BTreeSet::from([i + 1]);
        let mut node = w.head;
        while node != 0 {
            if !seen.insert(node) {
                return Err(Error::Invalid("dependency cycle"));
            }
            node = words[node - 1].head;
        }
    }
    Ok(())
}
fn shape(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_uppercase() {
                'A'
            } else if c.is_alphabetic() {
                'a'
            } else if c.is_numeric() {
                '0'
            } else if c.is_whitespace() {
                '_'
            } else {
                c
            }
        })
        .take(16)
        .collect()
}
impl Header {
    fn compact(mut self) -> Self {
        self.pos = self
            .pos
            .into_iter()
            .map(|s| s.into_boxed_str().into_string())
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .into_vec();
        self.actions = self
            .actions
            .into_iter()
            .map(|a| match a {
                Action::Shift => a,
                Action::Left(s) => Action::Left(s.into_boxed_str().into_string()),
                Action::Right(s) => Action::Right(s.into_boxed_str().into_string()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .into_vec();
        self.expansions = self
            .expansions
            .into_iter()
            .map(|(k, v)| {
                (
                    k.into_boxed_str().into_string(),
                    v.into_iter()
                        .map(|v| Variant {
                            count: v.count,
                            words: v
                                .words
                                .into_iter()
                                .map(|s| s.into_boxed_str().into_string())
                                .collect::<Vec<_>>()
                                .into_boxed_slice()
                                .into_vec(),
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                        .into_vec(),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .into_vec();
        self
    }
    fn classes(&self) -> usize {
        2 + self.pos.len() + self.actions.len()
    }
    fn f(&self, kind: u8, parts: &[&str]) -> usize {
        hash(self.seed, kind, parts, self.buckets)
    }
    fn character(&self, chars: &[char], index: usize) -> Vec<usize> {
        let context = (-2..=2)
            .map(|d| {
                index
                    .checked_add_signed(d)
                    .and_then(|i| chars.get(i))
                    .map_or("<end>".into(), char::to_string)
            })
            .collect::<Vec<_>>();
        let mut f = vec![self.f(0, &[])];
        for (i, c) in context.iter().enumerate() {
            f.push(self.f(1 + i as u8, &[c]));
            f.push(self.f(6 + i as u8, &[&shape(c)]));
        }
        f.push(self.f(11, &[&context[1], &context[2]]));
        f.push(self.f(12, &[&context[2], &context[3]]));
        unique(&mut f);
        f
    }
    fn word(&self, words: &[String], i: usize, previous: &[String]) -> Vec<usize> {
        let mut f = vec![self.f(20, &[])];
        for d in -2..=2 {
            let w = i
                .checked_add_signed(d)
                .and_then(|j| words.get(j))
                .map_or("<end>", String::as_str);
            f.push(self.f(21 + (d + 2) as u8, &[&w.to_lowercase()]));
        }
        let chars = words[i].to_lowercase().chars().collect::<Vec<_>>();
        for n in 1..=4.min(chars.len()) {
            f.push(self.f(26 + n as u8, &[&chars[..n].iter().collect::<String>()]));
            f.push(self.f(
                31 + n as u8,
                &[&chars[chars.len() - n..].iter().collect::<String>()],
            ));
        }
        f.push(self.f(36, &[&shape(&words[i])]));
        f.push(self.f(37, &[previous.last().map_or("<start>", String::as_str)]));
        f.push(
            self.f(
                38,
                &[
                    previous
                        .iter()
                        .rev()
                        .nth(1)
                        .map_or("<start2>", String::as_str),
                    previous.last().map_or("<start>", String::as_str),
                ],
            ),
        );
        unique(&mut f);
        f
    }
    fn parse_features(&self, forms: &[String], tags: &[String], state: &State) -> Vec<usize> {
        let mut nodes = state
            .stack
            .iter()
            .rev()
            .take(3)
            .copied()
            .map(Some)
            .collect::<Vec<_>>();
        nodes.resize(3, None);
        nodes.extend((state.next..state.next + 3).map(|i| (i < forms.len()).then_some(i)));
        let mut f = vec![self.f(40, &[])];
        let word = |i: Option<usize>| i.map_or("<none>", |i| forms[i].as_str());
        let tag = |i: Option<usize>| i.map_or("<none>", |i| tags[i].as_str());
        for (slot, node) in nodes.iter().enumerate() {
            f.push(self.f(41 + slot as u8, &[&word(*node).to_lowercase()]));
            f.push(self.f(47 + slot as u8, &[tag(*node)]));
        }
        for (j, k) in [(0, 1), (0, 3), (1, 3), (3, 4)] {
            let kind = 53 + j * 4 + k;
            f.push(self.f(kind as u8, &[tag(nodes[j]), tag(nodes[k])]));
            f.push(self.f((kind + 20) as u8, &[word(nodes[j]), tag(nodes[k])]));
            f.push(self.f((kind + 40) as u8, &[tag(nodes[j]), word(nodes[k])]));
        }
        f.push(self.f(110, &[tag(nodes[0]), tag(nodes[1]), tag(nodes[3])]));
        let distance = nodes[0]
            .zip(nodes[1])
            .map_or(0, |(a, b)| a.abs_diff(b).min(8));
        f.push(self.f(111, &[&distance.to_string(), tag(nodes[0]), tag(nodes[1])]));
        for (slot, node) in nodes[..2].iter().enumerate() {
            if let Some(node) = node {
                let children = (1..forms.len())
                    .filter(|i| state.heads[*i] == *node)
                    .collect::<Vec<_>>();
                for (side, child) in [children.first(), children.last()].iter().enumerate() {
                    let child = child.copied();
                    f.push(self.f((112 + slot * 4 + side) as u8, &[tag(child)]));
                    f.push(self.f(
                        (122 + slot * 4 + side) as u8,
                        &[child.map_or("<none>", |i| state.labels[i].as_str())],
                    ));
                }
            }
        }
        unique(&mut f);
        f
    }
    fn validate(&self) -> Result<(), Error> {
        Budget {
            seed: self.seed,
            buckets: self.buckets,
            dimensions: self.dimensions,
            epochs: 1,
            learning_rate: 0.1,
        }
        .validate()?;
        if !matches!(self.revision, 1 | 2)
            || self.pos.is_empty()
            || self.pos.len() > 17
            || !self.pos.iter().all(|p| pos(p))
            || self.pos.windows(2).any(|p| p[0] >= p[1])
            || !(2..=128).contains(&self.actions.len())
            || self.actions.windows(2).any(|p| p[0] >= p[1])
            || self.actions[0] != Action::Shift
            || !self.actions.contains(&Action::Right("root".into()))
            || self.actions.iter().any(|a| match a {
                Action::Shift => false,
                Action::Left(r) => !rel(r) || r == "root",
                Action::Right(r) => !rel(r),
            })
            || self.expansions.len() > 1024
            || self.expansions.windows(2).any(|p| p[0].0 >= p[1].0)
        {
            return Err(Error::Invalid("dependency header"));
        }
        for (key, variants) in &self.expansions {
            if !form(key)
                || variants.is_empty()
                || variants.len() > 16
                || variants.iter().any(|v| {
                    v.count == 0
                        || v.count > 1_000_000
                        || v.words.is_empty()
                        || v.words.len() > 16
                        || !v.words.iter().all(|w| form(w))
                })
                || variants.windows(2).any(|v| {
                    v[0].count < v[1].count
                        || (v[0].count == v[1].count && v[0].words >= v[1].words)
                })
            {
                return Err(Error::Invalid("dependency expansions"));
            }
        }
        Ok(())
    }
}
struct State {
    stack: Vec<usize>,
    next: usize,
    heads: Vec<usize>,
    labels: Vec<String>,
}
impl State {
    fn new(n: usize) -> Self {
        Self {
            stack: vec![0],
            next: 1,
            heads: vec![usize::MAX; n + 1],
            labels: vec![String::new(); n + 1],
        }
    }
    fn done(&self, n: usize) -> bool {
        self.next > n && self.stack.len() == 1
    }
    fn legal(&self, action: &Action, n: usize) -> bool {
        if matches!(action, Action::Shift) {
            return self.next <= n;
        }
        if self.stack.len() < 2 {
            return false;
        }
        let a = self.stack[self.stack.len() - 2];
        let b = *self.stack.last().unwrap();
        match action {
            Action::Left(r) => a != 0 && r != "root",
            Action::Right(r) => {
                b != 0
                    && ((a == 0 && r == "root" && self.next > n && self.stack.len() == 2)
                        || (a != 0 && r != "root"))
            }
            _ => false,
        }
    }
    fn apply(&mut self, action: &Action, n: usize) -> Result<(), Error> {
        if !self.legal(action, n) {
            return Err(Error::Invalid("illegal dependency action"));
        }
        match action {
            Action::Shift => {
                self.stack.push(self.next);
                self.next += 1;
            }
            Action::Left(label) => {
                let dep = self.stack.remove(self.stack.len() - 2);
                self.heads[dep] = *self.stack.last().unwrap();
                self.labels[dep] = label.clone();
            }
            Action::Right(label) => {
                let dep = self.stack.pop().unwrap();
                self.heads[dep] = *self.stack.last().unwrap();
                self.labels[dep] = label.clone();
            }
        }
        Ok(())
    }
    fn oracle(&self, words: &[Word]) -> Option<Action> {
        if self.stack.len() >= 2 {
            let a = self.stack[self.stack.len() - 2];
            let b = *self.stack.last().unwrap();
            let complete = |node| {
                words
                    .iter()
                    .enumerate()
                    .all(|(i, w)| w.head != node || self.heads[i + 1] != usize::MAX)
            };
            if a != 0 && words[a - 1].head == b && complete(a) {
                return Some(Action::Left(words[a - 1].deprel.clone()));
            }
            if b != 0
                && words[b - 1].head == a
                && complete(b)
                && self.legal(&Action::Right(words[b - 1].deprel.clone()), words.len())
            {
                return Some(Action::Right(words[b - 1].deprel.clone()));
            }
        }
        (self.next <= words.len()).then_some(Action::Shift)
    }
}
type Prepared = Vec<(Vec<usize>, usize, std::ops::Range<usize>)>;
impl Model {
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        budget.validate()?;
        if !(4..=10000).contains(&examples.len()) {
            return Err(Error::Invalid("dependency training count"));
        }
        let mut positions = BTreeSet::new();
        let mut actions = BTreeSet::from([Action::Shift]);
        let mut lexical = BTreeMap::<String, BTreeMap<Vec<String>, usize>>::new();
        for e in examples {
            validate_words(&e.words)?;
            if e.text.is_empty()
                || e.text.len() > 16384
                || e.surfaces.is_empty()
                || e.surfaces.len() > 256
            {
                return Err(Error::Invalid("dependency text quota"));
            }
            let flattened = e
                .surfaces
                .iter()
                .flat_map(|s| s.words.iter())
                .collect::<Vec<_>>();
            if flattened != e.words.iter().map(|w| &w.form).collect::<Vec<_>>() {
                return Err(Error::Invalid("dependency surface alignment"));
            }
            for s in &e.surfaces {
                if !form(&s.form) || s.words.is_empty() || s.words.len() > 16 {
                    return Err(Error::Invalid("dependency surface"));
                }
                *lexical
                    .entry(s.form.clone())
                    .or_default()
                    .entry(s.words.clone())
                    .or_default() += 1;
            }
            for w in &e.words {
                positions.insert(w.upos.clone());
            }
        }
        // The action vocabulary is learned only from teacher trees.
        for e in examples {
            for (i, w) in e.words.iter().enumerate() {
                actions.insert(if w.head > i + 1 {
                    Action::Left(w.deprel.clone())
                } else {
                    Action::Right(w.deprel.clone())
                });
            }
        }
        let expansions = lexical
            .into_iter()
            .filter(|(k, v)| v.keys().any(|w| w.len() != 1 || w[0] != *k))
            .map(|(k, v)| {
                let mut variants = v
                    .into_iter()
                    .map(|(words, count)| Variant { words, count })
                    .collect::<Vec<_>>();
                variants.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.words.cmp(&b.words)));
                (k, variants)
            })
            .collect();
        let header = Header {
            revision: 2,
            seed: budget.seed,
            buckets: budget.buckets,
            dimensions: budget.dimensions,
            pos: positions.into_iter().collect(),
            actions: actions.into_iter().collect(),
            expansions,
        }
        .compact();
        header.validate()?;
        let header_bytes = serde_json::to_vec(&header)
            .map_err(|e| Error::Storage(e.to_string()))?
            .len();
        let planned_bytes = 20
            + header_bytes
            + (header.buckets + header.classes()) * (header.dimensions + 4)
            + header.classes() * 4;
        if planned_bytes > MAX_BYTES {
            return Err(Error::Budget);
        }
        let mut prepared = Vec::<Prepared>::new();
        let mut nonprojective = 0;
        let mut total = 0usize;
        for e in examples {
            let mut p = vec![];
            let mut cursor = 0;
            let mut ends = BTreeSet::new();
            for surface in &e.surfaces {
                let start = cursor
                    + e.text[cursor..]
                        .find(&surface.form)
                        .ok_or(Error::Invalid("dependency source text"))?;
                if !e.text[cursor..start].chars().all(char::is_whitespace) {
                    return Err(Error::Invalid("dependency text gap"));
                }
                cursor = start + surface.form.len();
                ends.insert(cursor);
            }
            if !e.text[cursor..].chars().all(char::is_whitespace) {
                return Err(Error::Invalid("dependency text tail"));
            }
            let chars = e.text.chars().collect::<Vec<_>>();
            for (i, (offset, ch)) in e.text.char_indices().enumerate() {
                if !ch.is_whitespace() {
                    p.push((
                        header.character(&chars, i),
                        usize::from(ends.contains(&(offset + ch.len_utf8()))),
                        0..2,
                    ));
                }
            }
            let forms = e.words.iter().map(|w| w.form.clone()).collect::<Vec<_>>();
            let mut previous = vec![];
            for (i, w) in e.words.iter().enumerate() {
                p.push((
                    header.word(&forms, i, &previous),
                    2 + header.pos.binary_search(&w.upos).unwrap(),
                    2..2 + header.pos.len(),
                ));
                previous.push(w.upos.clone());
            }
            let mut state = State::new(e.words.len());
            let mut transitions = vec![];
            let forms = std::iter::once("<root>".into())
                .chain(forms)
                .collect::<Vec<_>>();
            let tags = std::iter::once("<root>".into())
                .chain(previous)
                .collect::<Vec<_>>();
            while !state.done(e.words.len()) {
                let Some(action) = state.oracle(&e.words) else {
                    nonprojective += 1;
                    transitions.clear();
                    break;
                };
                let index = header.actions.binary_search(&action).unwrap();
                transitions.push((
                    header.parse_features(&forms, &tags, &state),
                    2 + header.pos.len() + index,
                    2 + header.pos.len()..header.classes(),
                ));
                state.apply(&action, e.words.len())?;
            }
            p.extend(transitions);
            total = total.checked_add(p.len()).ok_or(Error::Budget)?;
            if total > 3_000_000 {
                return Err(Error::Budget);
            }
            prepared.push(p);
        }
        let dimensions = budget.dimensions;
        let mut rng = budget.seed;
        let mut learner = Learner {
            input: (0..budget.buckets * dimensions)
                .map(|_| {
                    ((random(&mut rng) >> 40) as f32 / (1u32 << 24) as f32 - 0.5)
                        / (dimensions as f32).sqrt()
                })
                .collect(),
            output: vec![0.0; header.classes() * dimensions],
            bias: vec![0.0; header.classes()],
            dimensions,
            tanh: true,
        };
        let mut order = (0..examples.len()).collect::<Vec<_>>();
        for epoch in 0..budget.epochs {
            for i in (1..order.len()).rev() {
                order.swap(i, random(&mut rng) as usize % (i + 1));
            }
            let mut event = Event {
                epoch: epoch + 1,
                examples: examples.len(),
                nonprojective,
                updates: 0,
                errors: 0,
            };
            for (step, index) in order.iter().enumerate() {
                let progress = (epoch * examples.len() + step) as f32
                    / (budget.epochs * examples.len()) as f32;
                for (f, gold, head) in &prepared[*index] {
                    event.errors += usize::from(learner.update(
                        f,
                        *gold,
                        head.clone(),
                        budget.learning_rate * (1.0 - progress).max(0.1),
                    ));
                    event.updates += 1;
                }
            }
            emit(&event)?;
        }
        let model = Self {
            header,
            input: Quantized::encode(&learner.input, dimensions),
            output: Quantized::encode(&learner.output, dimensions),
            bias: learner.bias,
        };
        model.to_bytes()?;
        Ok(model)
    }
    fn scores(&self, features: &[usize], work: &mut usize) -> Vec<f32> {
        let d = self.header.dimensions;
        let mut h = vec![0.0; d];
        let norm = (features.len() as f32).sqrt();
        *work += features.len() * d + self.bias.len() * d;
        for f in features {
            for (i, h) in h.iter_mut().enumerate() {
                *h += f32::from(self.input.values[f * d + i]) * self.input.scales[*f] / norm;
            }
        }
        for x in &mut h {
            *x = x.tanh();
        }
        (0..self.bias.len())
            .map(|c| {
                self.bias[c]
                    + h.iter()
                        .zip(&self.output.values[c * d..(c + 1) * d])
                        .map(|(h, w)| h * f32::from(*w) * self.output.scales[c])
                        .sum::<f32>()
            })
            .collect()
    }
    fn choose(
        &self,
        f: &[usize],
        allowed: impl Iterator<Item = usize>,
        margin: &mut f32,
        work: &mut usize,
    ) -> Result<usize, Error> {
        let scores = self.scores(f, work);
        let mut ranked = allowed.collect::<Vec<_>>();
        ranked.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then_with(|| a.cmp(b)));
        let best = *ranked
            .first()
            .ok_or(Error::Invalid("dependency dead end"))?;
        if ranked.len() > 1 {
            *margin = margin.min(scores[best] - scores[ranked[1]]);
        }
        Ok(best)
    }
    pub fn predict(&self, text: &str) -> Result<Prediction, Error> {
        if text.is_empty()
            || text.len() > 16384
            || text.chars().any(|c| c.is_control() && !c.is_whitespace())
        {
            return Err(Error::Invalid("dependency input"));
        }
        let chars = text.chars().collect::<Vec<_>>();
        let mut surfaces = vec![];
        let mut start = 0;
        let mut margin = f32::MAX;
        let mut work = 0;
        for (i, (offset, ch)) in text.char_indices().enumerate() {
            if ch.is_whitespace() {
                continue;
            }
            let boundary = self.choose(
                &self.header.character(&chars, i),
                0..2,
                &mut margin,
                &mut work,
            )?;
            if boundary == 1 {
                let end = offset + ch.len_utf8();
                let s = text[start..end].trim();
                if !form(s) {
                    return Err(Error::Invalid("predicted dependency token"));
                }
                surfaces.push(s.to_string());
                start = end;
            }
        }
        let tail = text[start..].trim();
        if !tail.is_empty() {
            if !form(tail) {
                return Err(Error::Invalid("predicted dependency tail"));
            }
            surfaces.push(tail.to_string());
        }
        let mut forms = vec![];
        let mut ambiguous = 0;
        for s in surfaces {
            if let Ok(index) = self
                .header
                .expansions
                .binary_search_by(|(key, _)| key.cmp(&s))
            {
                let v = &self.header.expansions[index].1;
                ambiguous += usize::from(v.len() > 1);
                forms.extend(v[0].words.clone());
            } else {
                forms.push(s);
            }
        }
        if forms.is_empty() || forms.len() > 256 {
            return Err(Error::Invalid("dependency predicted word quota"));
        }
        let mut tags = vec![];
        for i in 0..forms.len() {
            let guess = self.choose(
                &self.header.word(&forms, i, &tags),
                2..2 + self.header.pos.len(),
                &mut margin,
                &mut work,
            )?;
            tags.push(self.header.pos[guess - 2].clone());
        }
        let n = forms.len();
        let forms = std::iter::once("<root>".into())
            .chain(forms)
            .collect::<Vec<_>>();
        let tags = std::iter::once("<root>".into())
            .chain(tags)
            .collect::<Vec<_>>();
        let mut state = State::new(n);
        for _ in 0..2 * n {
            let base = 2 + self.header.pos.len();
            let allowed = self
                .header
                .actions
                .iter()
                .enumerate()
                .filter_map(|(i, a)| state.legal(a, n).then_some(base + i));
            let guess = self.choose(
                &self.header.parse_features(&forms, &tags, &state),
                allowed,
                &mut margin,
                &mut work,
            )?;
            state.apply(&self.header.actions[guess - base], n)?;
        }
        if !state.done(n) {
            return Err(Error::Invalid("dependency incomplete"));
        }
        let words = (1..=n)
            .map(|i| Word {
                form: forms[i].clone(),
                upos: tags[i].clone(),
                head: state.heads[i],
                deprel: state.labels[i].clone(),
            })
            .collect::<Vec<_>>();
        validate_words(&words)?;
        Ok(Prediction {
            words,
            ambiguous_expansions: ambiguous,
            minimum_margin: margin,
            work,
        })
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.input.retained_bytes()
            + self.output.retained_bytes()
            + self.bias.capacity() * 4
            + self.header.pos.capacity() * size_of::<String>()
            + self.header.pos.iter().map(String::capacity).sum::<usize>()
            + self.header.actions.capacity() * size_of::<Action>()
            + self
                .header
                .actions
                .iter()
                .map(|a| match a {
                    Action::Shift => 0,
                    Action::Left(r) | Action::Right(r) => r.capacity(),
                })
                .sum::<usize>()
            + self.header.expansions.capacity() * size_of::<(String, Vec<Variant>)>()
            + self
                .header
                .expansions
                .iter()
                .map(|(k, v)| {
                    k.capacity()
                        + v.capacity() * size_of::<Variant>()
                        + v.iter()
                            .map(|v| {
                                v.words.capacity() * size_of::<String>()
                                    + v.words.iter().map(String::capacity).sum::<usize>()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
    pub(crate) fn restore_legacy_reservation(&mut self, expected: usize) -> Result<(), Error> {
        let current = self.retained_bytes();
        if current == expected {
            return Ok(());
        }
        // Only the two published v1 layouts are admissible. Allocate the old
        // reservation physically; never weaken the catalog's exact identity check.
        let extra = 128 * self.header.expansions.len();
        if self.header.revision != 1 || current.checked_add(extra) != Some(expected) {
            return Err(Error::Invalid("dependency legacy residency witness"));
        }
        let n = self.bias.len();
        let mut reserved = vec![0.0; n + extra / size_of::<f32>()];
        reserved[..n].copy_from_slice(&self.bias);
        reserved.truncate(n);
        self.bias = reserved;
        if self.retained_bytes() != expected {
            return Err(Error::Invalid("dependency legacy allocation"));
        }
        Ok(())
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let header = serde_json::to_vec(&self.header).map_err(|e| Error::Storage(e.to_string()))?;
        let mut bytes = if self.header.revision == 1 {
            LEGACY_MAGIC
        } else {
            MAGIC
        }
        .to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend(header);
        self.input.append(&mut bytes);
        self.output.append(&mut bytes);
        for b in &self.bias {
            bytes.extend_from_slice(&b.to_le_bytes());
        }
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if !(20..=MAX_BYTES).contains(&bytes.len()) {
            return Err(Error::Invalid("dependency length"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if checksum(body).to_le_bytes() != digest {
            return Err(Error::Invalid("dependency checksum"));
        }
        let mut cursor = Cursor::new(body);
        let magic = take::<8>(&mut cursor)?;
        if &magic != MAGIC && &magic != LEGACY_MAGIC {
            return Err(Error::Invalid("dependency magic"));
        }
        let size = u32::from_le_bytes(take(&mut cursor)?) as usize;
        let header: Header = serde_json::from_slice(
            body.get(12..12usize.checked_add(size).ok_or(Error::Budget)?)
                .ok_or(Error::Invalid("dependency header length"))?,
        )
        .map_err(|e| Error::Storage(e.to_string()))?;
        header.validate()?;
        if (header.revision == 1) != (&magic == LEGACY_MAGIC) {
            return Err(Error::Invalid("dependency codec revision"));
        }
        let header = header.compact();
        if body.len()
            != 12
                + size
                + (header.buckets + header.classes()) * (header.dimensions + 4)
                + header.classes() * 4
        {
            return Err(Error::Invalid("dependency payload"));
        }
        cursor.set_position((12 + size) as u64);
        let input = Quantized::read(&mut cursor, header.buckets, header.dimensions)?;
        let output = Quantized::read(&mut cursor, header.classes(), header.dimensions)?;
        let mut bias = Vec::with_capacity(header.classes());
        for _ in 0..header.classes() {
            let x = f32::from_le_bytes(take(&mut cursor)?);
            if !x.is_finite() || x.abs() > 16.0 {
                return Err(Error::Invalid("dependency bias"));
            }
            bias.push(x);
        }
        Ok(Self {
            header,
            input,
            output,
            bias,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub fn data() -> Vec<Example> {
        (0..24)
            .map(|i| {
                let a = if i % 2 == 0 { "alice" } else { "bruno" };
                let b = if i % 3 == 0 { "caixa" } else { "bola" };
                Example {
                    text: format!("{a} move {b}"),
                    surfaces: [a, "move", b]
                        .map(|s| Surface {
                            form: s.into(),
                            words: vec![s.into()],
                        })
                        .into(),
                    words: vec![
                        Word {
                            form: a.into(),
                            upos: "PROPN".into(),
                            head: 2,
                            deprel: "nsubj".into(),
                        },
                        Word {
                            form: "move".into(),
                            upos: "VERB".into(),
                            head: 0,
                            deprel: "root".into(),
                        },
                        Word {
                            form: b.into(),
                            upos: "NOUN".into(),
                            head: 2,
                            deprel: "obj".into(),
                        },
                    ],
                }
            })
            .collect()
    }
    pub fn budget() -> Budget {
        Budget {
            seed: 47,
            buckets: 1024,
            dimensions: 16,
            epochs: 8,
            learning_rate: 0.3,
        }
    }
    #[test]
    fn teacher_validation_checks_all_heads_before_traversal() {
        let mut words = data()[0].words.clone();
        words[2].head = 999;
        assert!(validate_words(&words).is_err());
        words[2].head = 1;
        words[0].head = 3;
        assert!(validate_words(&words).is_err());
        assert!(
            Model::learn(
                &data(),
                Budget {
                    seed: 0,
                    ..budget()
                },
                &mut |_| Ok(())
            )
            .is_err()
        );
        let mut examples = data();
        examples[0].text = "unrelated text".into();
        assert!(Model::learn(&examples, budget(), &mut |_| Ok(())).is_err());
        let singles = (0..4)
            .map(|_| Example {
                text: "word".into(),
                surfaces: vec![Surface {
                    form: "word".into(),
                    words: vec!["word".into()],
                }],
                words: vec![Word {
                    form: "word".into(),
                    upos: "NOUN".into(),
                    head: 0,
                    deprel: "root".into(),
                }],
            })
            .collect::<Vec<_>>();
        assert!(Model::learn(&singles, budget(), &mut |_| Ok(())).is_ok());
    }
    #[test]
    fn expansion_storage_preserves_object_wire_format_and_rejects_duplicate_keys() {
        let mut value = serde_json::to_value(Header {
            revision: 1,
            seed: 47,
            buckets: 256,
            dimensions: 8,
            pos: vec!["NOUN".into()],
            actions: vec![
                Action::Shift,
                Action::Left("dep".into()),
                Action::Right("root".into()),
            ],
            expansions: vec![(
                "ab".into(),
                vec![Variant {
                    words: vec!["a".into(), "b".into()],
                    count: 3,
                }],
            )],
        })
        .unwrap();
        assert!(value["expansions"].is_object());
        let header: Header = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(header.expansions[0].0, "ab");
        value["expansions"] = serde_json::json!({"ab":[{"words":["a","b"],"count":3}]});
        let text = serde_json::to_string(&value)
            .unwrap()
            .replace("\"ab\":[", "\"ab\":[],\"ab\":[");
        assert!(serde_json::from_str::<Header>(&text).is_err());
        assert!(
            serde_json::from_str::<Header>(
                &serde_json::to_string(&value)
                    .unwrap()
                    .replace("\"ab\":[", "\"zz\":[],\"ab\":[")
            )
            .is_err()
        );
    }
    #[test]
    fn legacy_codec_keeps_its_residency_witness_without_weakening_catalog_checks() {
        let mut model = Model::learn(&data(), budget(), &mut |_| Ok(())).unwrap();
        model.header.expansions.push((
            "xy".into(),
            vec![Variant {
                words: vec!["x".into(), "y".into()],
                count: 1,
            }],
        ));
        model.header = model.header.compact();
        let modern = model.to_bytes().unwrap();
        let charged = model.retained_bytes();
        model.header.revision = 1;
        let legacy = model.to_bytes().unwrap();
        assert_eq!(&modern[..8], MAGIC);
        assert_eq!(&legacy[..8], LEGACY_MAGIC);
        let mut restored = Model::from_bytes(&legacy).unwrap();
        assert_eq!(restored.retained_bytes(), charged);
        assert!(restored.restore_legacy_reservation(charged + 1).is_err());
        restored.restore_legacy_reservation(charged + 128).unwrap();
        assert_eq!(restored.retained_bytes(), charged + 128);
        assert_eq!(restored.to_bytes().unwrap(), legacy);
        assert_eq!(
            serde_json::to_value(restored.predict("alice move bola").unwrap()).unwrap(),
            serde_json::to_value(
                Model::from_bytes(&modern)
                    .unwrap()
                    .predict("alice move bola")
                    .unwrap()
            )
            .unwrap()
        );
    }
    #[test]
    fn oracle_recovers_all_small_projective_trees_and_detects_nonprojective_trees() {
        for n in 1usize..=4 {
            for code in 0..(n + 1).pow(n as u32) {
                let mut code = code;
                let words = (1..=n)
                    .map(|i| {
                        let head = code % (n + 1);
                        code /= n + 1;
                        Word {
                            form: format!("w{i}"),
                            upos: "NOUN".into(),
                            head,
                            deprel: if head == 0 { "root" } else { "dep" }.into(),
                        }
                    })
                    .collect::<Vec<_>>();
                if validate_words(&words).is_err() {
                    continue;
                }
                let projective = words.iter().enumerate().all(|(i, w)| {
                    ((i + 1).min(w.head) + 1..(i + 1).max(w.head)).all(|j| {
                        let mut node = j;
                        while node != 0 && node != w.head {
                            node = words[node - 1].head;
                        }
                        node == w.head
                    })
                });
                let mut state = State::new(n);
                let mut complete = true;
                for _ in 0..2 * n {
                    if let Some(a) = state.oracle(&words) {
                        state.apply(&a, n).unwrap();
                    } else {
                        complete = false;
                        break;
                    }
                }
                assert_eq!(complete, projective);
                if complete {
                    assert!(state.done(n));
                    assert_eq!(
                        state.heads[1..],
                        words.iter().map(|w| w.head).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[test]
    fn learned_raw_text_parser_roundtrips_and_rejects_repaired_malformed_payloads() {
        let mut events = vec![];
        let model = Model::learn(&data(), budget(), &mut |e| {
            events.push(e.updates);
            Ok(())
        })
        .unwrap();
        assert_eq!(events.len(), 8);
        assert!(events.iter().all(|n| *n > 0));
        let bytes = model.to_bytes().unwrap();
        let restored = Model::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, restored.to_bytes().unwrap());
        assert_eq!(model.retained_bytes(), restored.retained_bytes());
        let result = model.predict("alice move bola").unwrap();
        validate_words(&result.words).unwrap();
        assert_eq!(
            serde_json::to_value(&result).unwrap(),
            serde_json::to_value(restored.predict("alice move bola").unwrap()).unwrap()
        );
        assert!(result.work > 0 && result.minimum_margin.is_finite());
        assert!(model.retained_bytes() < 262144);
        assert!(model.predict("").is_err());
        assert!(model.predict("  ").is_err());
        assert!(model.predict("a\u{0}b").is_err());
        let mut corrupted = bytes.clone();
        corrupted[20] ^= 1;
        assert!(Model::from_bytes(&corrupted).is_err());
        let size = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut corrupted = bytes.clone();
        corrupted[12 + size..16 + size].copy_from_slice(&f32::NAN.to_le_bytes());
        let n = corrupted.len() - 8;
        let digest = checksum(&corrupted[..n]);
        corrupted[n..].copy_from_slice(&digest.to_le_bytes());
        assert!(Model::from_bytes(&corrupted).is_err());
        let mut corrupted = bytes[..bytes.len() - 8].to_vec();
        corrupted.push(0);
        corrupted.extend_from_slice(&checksum(&corrupted).to_le_bytes());
        assert!(Model::from_bytes(&corrupted).is_err());
    }
}
