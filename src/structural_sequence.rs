//! Bounded hashed averaged perceptrons for sentence labels and BIO sequences.
//! Greedy decoding is empirical, not a symbolic certificate or calibrated probability.
//! Algorithm references: Collins (2002), https://aclanthology.org/W02-1001/;
//! NLTK's greedy averaged-perceptron documentation. This is an independent implementation.
use super::{Error, Term};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[path = "structural_sequence_compact.rs"]
pub mod compact;
#[path = "structural_dependency.rs"]
pub mod dependency;
#[path = "structural_sequence_factorized.rs"]
pub mod factorized;
#[path = "structural_sequence_joint.rs"]
pub mod joint;
#[path = "structural_lexical_encoder.rs"]
pub mod lexical_encoder;
#[path = "structural_relation_network.rs"]
pub mod relation_network;
#[path = "structural_sequence_search.rs"]
pub mod search;
#[path = "structural_sequence_segment.rs"]
pub mod segment;

const MAX_WEIGHTS: usize = 24000;
pub const MAX_BYTES: usize = 196608;
pub const CONTEXT: &str = "axon/sequence-v1";
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub seed: u64,
    pub buckets: usize,
    pub epochs: usize,
    pub weights: usize,
    #[serde(default)]
    pub intent_weights: usize,
    #[serde(default)]
    pub subwords: bool,
    #[serde(default)]
    pub shared_roles: bool,
    pub minimum_margin: u16,
}
impl Budget {
    fn validate(self) -> Result<(), Error> {
        self.validate_limit(MAX_WEIGHTS)
    }
    fn validate_limit(self, maximum: usize) -> Result<(), Error> {
        if self.seed == 0
            || !self.buckets.is_power_of_two()
            || !(256..=16384).contains(&self.buckets)
            || !(1..=16).contains(&self.epochs)
            || !(256..=maximum).contains(&self.weights)
            || self.intent_weights > self.weights.saturating_sub(128)
            || !(1..=255).contains(&self.minimum_margin)
        {
            return Err(Error::Invalid("sequence budget"));
        }
        Ok(())
    }
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub text: String,
    pub intent: String,
    pub tags: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct Event {
    pub epoch: usize,
    pub sentences: usize,
    pub intent_errors: usize,
    pub tag_errors: usize,
}
#[derive(Debug, Serialize)]
pub struct Prediction {
    pub output: Term,
    pub intent_margin: i32,
    pub tag_margin: i32,
    pub work: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "Wire", into = "Wire")]
pub struct Model {
    seed: u64,
    buckets: usize,
    intents: Vec<String>,
    tags: Vec<String>,
    minimum_margin: u16,
    weights: Vec<(u32, i8)>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    revision: u8,
    seed: u64,
    buckets: usize,
    intents: Vec<String>,
    tags: Vec<String>,
    minimum_margin: u16,
    packed: String,
}
fn label(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 96
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
}
fn role(s: &str) -> Option<&str> {
    s.get(2..)
        .filter(|_| s.starts_with("B:") || s.starts_with("I:"))
}
fn legal(tag: &str, previous: &str) -> bool {
    !tag.starts_with("I:") || role(tag) == role(previous)
}
fn class_labels(
    examples: &[Example],
    initial: Option<&Model>,
) -> Result<(Vec<String>, Vec<String>), Error> {
    if !(4..=10000).contains(&examples.len()) {
        return Err(Error::Invalid("sequence training count"));
    }
    let mut intents = BTreeSet::new();
    let mut tags = BTreeSet::new();
    if let Some(m) = initial {
        intents.extend(m.intents.iter().cloned());
        tags.extend(m.tags.iter().skip(1).cloned());
    }
    for e in examples {
        if !label(&e.intent) || e.tags.len() != tokens(&e.text)?.len() {
            return Err(Error::Invalid("sequence supervision"));
        }
        intents.insert(e.intent.clone());
        let mut prev = "O";
        for t in &e.tags {
            if t != "O" {
                let r = role(t)
                    .filter(|r| label(r))
                    .ok_or(Error::Invalid("sequence BIO tag"))?;
                tags.insert(format!("B:{r}"));
                tags.insert(format!("I:{r}"));
            }
            if !legal(t, prev) {
                return Err(Error::Invalid("sequence BIO transition"));
            }
            prev = t;
        }
    }
    let intents = intents.into_iter().collect::<Vec<_>>();
    let tags = std::iter::once("O".into()).chain(tags).collect::<Vec<_>>();
    if !(2..=96).contains(&intents.len()) || tags.len() > 192 || intents.len() + tags.len() > 256 {
        return Err(Error::Invalid("sequence classes"));
    }
    Ok((intents, tags))
}
impl From<Model> for Wire {
    fn from(m: Model) -> Self {
        let mut packed = String::with_capacity(m.weights.len() * 8);
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for (key, weight) in m.weights {
            for byte in [
                (key & 255) as u8,
                ((key >> 8) & 255) as u8,
                ((key >> 16) & 255) as u8,
                weight as u8,
            ] {
                packed.push(HEX[(byte >> 4) as usize] as char);
                packed.push(HEX[(byte & 15) as usize] as char);
            }
        }
        Self {
            revision: 1,
            seed: m.seed,
            buckets: m.buckets,
            intents: m.intents,
            tags: m.tags,
            minimum_margin: m.minimum_margin,
            packed,
        }
    }
}
impl TryFrom<Wire> for Model {
    type Error = Error;
    fn try_from(w: Wire) -> Result<Self, Error> {
        validate_header(&w)?;
        if w.packed.len() > MAX_WEIGHTS * 8 || !w.packed.len().is_multiple_of(8) {
            return Err(Error::Invalid("sequence encoding"));
        }
        let mut weights = Vec::with_capacity(w.packed.len() / 8);
        for record in w.packed.as_bytes().as_chunks::<8>().0 {
            let mut bytes = [0u8; 4];
            for (i, pair) in record.as_chunks::<2>().0.iter().enumerate() {
                let digit = |b: u8| match b {
                    b'0'..=b'9' => Ok(b - b'0'),
                    b'a'..=b'f' => Ok(b - b'a' + 10),
                    _ => Err(Error::Invalid("sequence hex")),
                };
                bytes[i] = (digit(pair[0])? << 4) | digit(pair[1])?;
            }
            let key =
                u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
            let value = bytes[3] as i8;
            if key as usize >= w.buckets * (w.intents.len() + w.tags.len())
                || value == 0
                || weights.last().is_some_and(|(old, _)| *old >= key)
            {
                return Err(Error::Invalid("sequence weights"));
            }
            weights.push((key, value));
        }
        if weights.is_empty() {
            return Err(Error::Invalid("empty sequence model"));
        }
        let mut model = metadata(w);
        model.weights = weights;
        Ok(model)
    }
}
fn validate_header(w: &Wire) -> Result<(), Error> {
    Budget {
        seed: w.seed,
        buckets: w.buckets,
        epochs: 1,
        weights: 256,
        intent_weights: 0,
        subwords: false,
        shared_roles: false,
        minimum_margin: w.minimum_margin,
    }
    .validate()?;
    if w.revision != 1
        || !(2..=96).contains(&w.intents.len())
        || !(1..=192).contains(&w.tags.len())
        || w.intents.len() + w.tags.len() > 256
        || !w.intents.iter().all(|s| label(s))
        || w.intents.windows(2).any(|p| p[0] >= p[1])
        || w.tags[0] != "O"
        || w.tags[1..].windows(2).any(|p| p[0] >= p[1])
        || w.tags[1..]
            .iter()
            .any(|s| role(s).is_none_or(|r| !label(r)))
        || w.tags[1..].iter().any(|s| {
            !w.tags.contains(&format!(
                "{}:{}",
                if s.starts_with("B:") { "I" } else { "B" },
                role(s).unwrap()
            ))
        })
    {
        return Err(Error::Invalid("sequence encoding"));
    }
    Ok(())
}
fn metadata(w: Wire) -> Model {
    let compact_labels = |labels: Vec<String>| {
        labels
            .into_iter()
            .map(|s| s.into_boxed_str().into_string())
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .into_vec()
    };
    Model {
        seed: w.seed,
        buckets: w.buckets,
        intents: compact_labels(w.intents),
        tags: compact_labels(w.tags),
        minimum_margin: w.minimum_margin,
        weights: Vec::new(),
    }
}
pub fn tokens(text: &str) -> Result<Vec<String>, Error> {
    tokens_mode(text, true)
}
fn tokens_mode(text: &str, strip_terminal_punctuation: bool) -> Result<Vec<String>, Error> {
    if text.is_empty() || text.len() > 4096 || text.chars().any(char::is_control) {
        return Err(Error::Invalid("sequence text"));
    }
    let lowered = text.to_lowercase();
    let trimmed = lowered.trim();
    let stream = if strip_terminal_punctuation {
        trimmed.trim_end_matches(['.', '?', '!'])
    } else {
        trimmed
    };
    let words = stream
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if words.is_empty() || words.len() > 64 || words.iter().any(|s| s.len() > 256) {
        return Err(Error::Invalid("sequence tokens"));
    }
    Ok(words)
}
fn hash(seed: u64, kind: u8, parts: &[&str], buckets: usize) -> usize {
    let mut h = seed ^ 0xcbf29ce484222325;
    for b in std::iter::once(kind).chain(
        parts
            .iter()
            .flat_map(|p| p.bytes().chain(std::iter::once(255))),
    ) {
        h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
    (h as usize) & (buckets - 1)
}
fn unique(v: &mut Vec<usize>) {
    v.sort_unstable();
    v.dedup();
}
fn intent_features(words: &[String], b: Budget) -> Vec<usize> {
    let mut f = vec![hash(b.seed, 0, &[], b.buckets)];
    for word in words {
        f.push(hash(b.seed, 1, &[word], b.buckets));
        if b.subwords {
            subwords(word, b, 50, &mut f);
        }
    }
    for pair in words.windows(2) {
        f.push(hash(b.seed, 2, &[&pair[0], &pair[1]], b.buckets));
    }
    f.push(hash(b.seed, 3, &[&words[0]], b.buckets));
    f.push(hash(b.seed, 4, &[words.last().unwrap()], b.buckets));
    unique(&mut f);
    f
}
fn word_features(words: &[String], index: usize, b: Budget) -> Vec<usize> {
    let mut f = vec![hash(b.seed, 10, &[], b.buckets)];
    for offset in -2isize..=2 {
        let word = index
            .checked_add_signed(offset)
            .and_then(|i| words.get(i))
            .map(String::as_str)
            .unwrap_or(if offset < 0 { "<start>" } else { "<end>" });
        f.push(hash(b.seed, (13 + offset) as u8, &[word], b.buckets));
    }
    let word = &words[index];
    if b.subwords {
        subwords(word, b, 40, &mut f);
    }
    let chars = word.chars().collect::<Vec<_>>();
    for len in 1..=3.min(chars.len()) {
        let prefix = chars[..len].iter().collect::<String>();
        let suffix = chars[chars.len() - len..].iter().collect::<String>();
        f.push(hash(b.seed, 20 + len as u8, &[&prefix], b.buckets));
        f.push(hash(b.seed, 24 + len as u8, &[&suffix], b.buckets));
    }
    let shape = if word.chars().all(|c| c.is_ascii_digit()) {
        "digits"
    } else if word.contains('-') {
        "dash"
    } else {
        "text"
    };
    f.push(hash(b.seed, 28, &[shape], b.buckets));
    unique(&mut f);
    f
}
fn subwords(word: &str, b: Budget, kind: u8, features: &mut Vec<usize>) {
    let characters = word.chars().collect::<Vec<_>>();
    for width in 2..=4.min(characters.len()) {
        for part in characters.windows(width) {
            let value = part.iter().collect::<String>();
            features.push(hash(b.seed, kind + width as u8, &[&value], b.buckets));
        }
    }
}
fn contextual(
    base: &[usize],
    previous: &str,
    previous2: &str,
    intent: &str,
    b: Budget,
) -> Vec<usize> {
    let mut f = base.to_vec();
    for (kind, parts) in [
        (30, vec![previous]),
        (31, vec![previous2]),
        (32, vec![previous, previous2]),
        (33, vec![intent]),
        (34, vec![previous, intent]),
    ] {
        f.push(hash(b.seed, kind, &parts, b.buckets));
    }
    unique(&mut f);
    f
}
fn best(scores: &[i32], start: usize, end: usize, tags: Option<(&[String], &str)>) -> (usize, i32) {
    best_allowed(scores, start, end, tags, |_| true)
}
fn best_allowed(
    scores: &[i32],
    start: usize,
    end: usize,
    tags: Option<(&[String], &str)>,
    allowed: impl Fn(usize) -> bool,
) -> (usize, i32) {
    let mut ranked = (start..end)
        .filter(|i| allowed(*i))
        .filter(|i| tags.is_none_or(|(ts, prev)| legal(&ts[*i - start], prev)))
        .collect::<Vec<_>>();
    ranked.sort_by_key(|i| (std::cmp::Reverse(scores[*i]), *i));
    let first = ranked[0];
    let margin = ranked
        .get(1)
        .map_or(i32::MAX, |i| scores[first] - scores[*i]);
    (first, margin)
}
struct Learner {
    weights: Vec<i32>,
    totals: Vec<i64>,
    last: Vec<u32>,
    clock: u32,
    classes: usize,
}
fn quantized(value: f64, maximum: f64) -> i8 {
    (value / (maximum / 127.0).max(1.0)).round() as i8
}
impl Learner {
    fn finish(
        self,
        intents: Vec<String>,
        tags: Vec<String>,
        budget: Budget,
        shared_scale: bool,
    ) -> Model {
        let n = intents.len();
        let classes = self.classes;
        let mut averaged = Vec::with_capacity(self.weights.len());
        let mut maximum = [0.0f64; 2];
        for key in 0..self.weights.len() {
            let total = self.totals[key]
                + i64::from(self.clock - self.last[key]) * i64::from(self.weights[key]);
            let value = total as f64 / f64::from(self.clock);
            let head = usize::from(key % classes >= n);
            maximum[head] = maximum[head].max(value.abs());
            averaged.push(value);
        }
        if shared_scale {
            let shared = maximum[0].max(maximum[1]);
            maximum.fill(shared);
        }
        let mut weights = Vec::new();
        for (key, average) in averaged.into_iter().enumerate() {
            let head = usize::from(key % classes >= n);
            let value = quantized(average, maximum[head]);
            if value != 0 {
                weights.push((key as u32, value));
            }
        }
        let mut model = metadata(Wire {
            revision: 1,
            seed: budget.seed,
            buckets: budget.buckets,
            intents,
            tags,
            minimum_margin: budget.minimum_margin,
            packed: String::new(),
        });
        model.weights = weights;
        model.pruned(budget)
    }
    fn scores(&self, features: &[usize]) -> Vec<i32> {
        let mut scores = vec![0; self.classes];
        for f in features {
            for (s, w) in scores
                .iter_mut()
                .zip(&self.weights[f * self.classes..(f + 1) * self.classes])
            {
                *s += w;
            }
        }
        scores
    }
    fn update(&mut self, features: &[usize], gold: usize, guess: usize) {
        self.update_shared(features, gold, guess, None)
    }
    fn add(&mut self, feature: usize, class: usize, delta: i32) {
        let key = feature * self.classes + class;
        self.totals[key] += i64::from(self.clock - self.last[key]) * i64::from(self.weights[key]);
        self.last[key] = self.clock;
        self.weights[key] = (self.weights[key] + delta).clamp(-32767, 32767);
    }
    fn update_shared(
        &mut self,
        features: &[usize],
        gold: usize,
        guess: usize,
        sharing: Option<(&[usize], &[Option<usize>])>,
    ) {
        self.clock += 1;
        if gold == guess {
            return;
        }
        for f in features {
            for (class, delta) in [(gold, 8), (guess, -8)] {
                self.add(*f, class, delta);
                if let Some((base, siblings)) = sharing
                    && base.binary_search(f).is_ok()
                    && let Some(sibling) = siblings[class]
                {
                    self.add(*f, sibling, delta);
                }
            }
        }
    }
    fn update_path(
        &mut self,
        features: &[usize],
        words: &[Vec<usize>],
        labels: (&[String], &[String]),
        gold: (&usize, &[usize]),
        guess: &search::Path,
        budget: Budget,
    ) {
        self.clock += 1;
        let (intents, tags) = labels;
        let n = intents.len();
        if *gold.0 == guess.intent && gold.1.iter().copied().eq(guess.tags.iter().map(|i| n + i)) {
            return;
        }
        for f in features {
            self.add(*f, *gold.0, 8);
            self.add(*f, guess.intent, -8);
        }
        for (intent, sequence, delta) in [
            (*gold.0, gold.1.iter().map(|i| i - n).collect::<Vec<_>>(), 8),
            (guess.intent, guess.tags.clone(), -8),
        ] {
            let (mut previous, mut previous2) = ("<start>", "<start2>");
            for (base, tag) in words.iter().zip(sequence) {
                for f in contextual(base, previous, previous2, &intents[intent], budget) {
                    self.add(f, n + tag, delta);
                }
                previous2 = previous;
                previous = &tags[tag];
            }
        }
    }
}
impl Model {
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let bytes = serde_json::to_vec(self).map_err(|e| Error::Storage(e.to_string()))?;
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        serde_json::from_slice(bytes).map_err(|e| Error::Storage(e.to_string()))
    }
    fn budget(&self) -> Budget {
        Budget {
            seed: self.seed,
            buckets: self.buckets,
            epochs: 1,
            weights: MAX_WEIGHTS,
            intent_weights: 0,
            subwords: false,
            shared_roles: false,
            minimum_margin: self.minimum_margin,
        }
    }
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        initial: Option<&Self>,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        if budget.subwords {
            return Err(Error::Invalid("subword model requires enhanced codec"));
        }
        Self::try_from(Wire::from(Self::fit(
            examples,
            budget,
            initial,
            MAX_WEIGHTS,
            emit,
        )?))
    }
    fn fit(
        examples: &[Example],
        budget: Budget,
        initial: Option<&Self>,
        maximum: usize,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        Self::fit_mode(examples, budget, initial, maximum, None, emit)
    }
    fn fit_mode(
        examples: &[Example],
        budget: Budget,
        initial: Option<&Self>,
        maximum: usize,
        structured: Option<search::Budget>,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        budget.validate_limit(maximum)?;
        if let Some(search) = structured {
            search.validate()?;
            if budget.shared_roles {
                return Err(Error::Invalid("joint shared-role schema"));
            }
        }
        if !(4..=10000).contains(&examples.len()) {
            return Err(Error::Invalid("sequence training count"));
        }
        if initial.is_some_and(|m| m.seed != budget.seed || m.buckets != budget.buckets) {
            return Err(Error::Invalid("sequence refinement schema"));
        }
        let (intents, tags) = class_labels(examples, initial)?;
        let n = intents.len();
        let classes = n + tags.len();
        let mut siblings = vec![None; classes];
        for (i, tag) in tags.iter().enumerate().skip(1) {
            let paired = format!(
                "{}:{}",
                if tag.starts_with("B:") { "I" } else { "B" },
                role(tag).unwrap()
            );
            siblings[n + i] = Some(n + tags.iter().position(|t| t == &paired).unwrap());
        }
        let size = budget.buckets * classes;
        let mut learner = Learner {
            weights: vec![0; size],
            totals: vec![0; size],
            last: vec![0; size],
            clock: 0,
            classes,
        };
        if let Some(m) = initial {
            let old = m.intents.len() + m.tags.len();
            for (key, value) in &m.weights {
                let feature = *key as usize / old;
                let class = *key as usize % old;
                let new = if class < m.intents.len() {
                    intents.binary_search(&m.intents[class]).unwrap()
                } else {
                    n + tags
                        .iter()
                        .position(|t| t == &m.tags[class - m.intents.len()])
                        .unwrap()
                };
                learner.weights[feature * classes + new] = i32::from(*value);
            }
        }
        let mut prepared = Vec::new();
        for e in examples {
            let words = tokens(&e.text)?;
            prepared.push((
                intent_features(&words, budget),
                (0..words.len())
                    .map(|i| word_features(&words, i, budget))
                    .collect::<Vec<_>>(),
                intents.binary_search(&e.intent).unwrap(),
                e.tags
                    .iter()
                    .map(|t| n + tags.iter().position(|s| s == t).unwrap())
                    .collect::<Vec<_>>(),
            ));
        }
        let mut order = (0..examples.len()).collect::<Vec<_>>();
        let mut rng = budget.seed;
        for epoch in 0..budget.epochs {
            for i in (1..order.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                order.swap(i, rng as usize % (i + 1));
            }
            let mut event = Event {
                epoch: epoch + 1,
                sentences: examples.len(),
                intent_errors: 0,
                tag_errors: 0,
            };
            for index in &order {
                let (features, word_features, intent, gold_tags) = &prepared[*index];
                if let Some(search) = structured {
                    let decoded = search::decode(
                        (&intents, &tags),
                        features,
                        word_features,
                        budget,
                        search,
                        |f, work| {
                            *work += f.len() * classes;
                            learner.scores(f)
                        },
                    )?;
                    let guess = &decoded.paths[0];
                    event.intent_errors += usize::from(guess.intent != *intent);
                    event.tag_errors += gold_tags
                        .iter()
                        .zip(&guess.tags)
                        .filter(|(a, b)| **a != n + **b)
                        .count();
                    learner.update_path(
                        features,
                        word_features,
                        (&intents, &tags),
                        (intent, gold_tags),
                        guess,
                        budget,
                    );
                    continue;
                }
                let (guess, _) = best(&learner.scores(features), 0, n, None);
                event.intent_errors += usize::from(guess != *intent);
                learner.update(features, *intent, guess);
                let (mut previous, mut previous2) = ("<start>", "<start2>");
                for (base, gold) in word_features.iter().zip(gold_tags) {
                    let f = contextual(base, previous, previous2, &intents[*intent], budget);
                    let (guess, _) = best(&learner.scores(&f), n, classes, Some((&tags, previous)));
                    event.tag_errors += usize::from(guess != *gold);
                    learner.update_shared(
                        &f,
                        *gold,
                        guess,
                        budget
                            .shared_roles
                            .then_some((base.as_slice(), siblings.as_slice())),
                    );
                    previous2 = previous;
                    // Training histories must admit the supervised BIO continuation.
                    previous = &tags[*gold - n];
                }
            }
            emit(&event)?;
        }
        Ok(learner.finish(intents, tags, budget, structured.is_some()))
    }
    fn pruned(mut self, budget: Budget) -> Self {
        let mut parts = [Vec::new(), Vec::new()];
        let n = self.intents.len();
        let classes = n + self.tags.len();
        for (key, value) in std::mem::take(&mut self.weights) {
            parts[usize::from(key as usize % classes >= n)].push((key, value));
        }
        let intent_limit = if budget.intent_weights == 0 {
            budget.weights / 3
        } else {
            budget.intent_weights
        };
        let limits = [intent_limit, budget.weights - intent_limit];
        let mut weights = Vec::new();
        for (mut part, limit) in parts.into_iter().zip(limits) {
            part.sort_by_key(|(key, value)| (std::cmp::Reverse(i16::from(*value).abs()), *key));
            part.truncate(limit);
            weights.extend(part);
        }
        weights.sort_by_key(|(key, _)| *key);
        self.weights = weights.into_boxed_slice().into_vec();
        self
    }
    // Laboratory-only comparison. The wide model is never encoded or admitted.
    pub(crate) fn compression_models(
        examples: &[Example],
        budget: Budget,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<(Self, Self), Error> {
        budget.validate_limit(compact::MAX_WEIGHTS)?;
        let mut wide = budget;
        wide.weights = budget.buckets * 512;
        wide.intent_weights = budget.buckets * 256;
        let unpruned = Self::fit(examples, wide, None, wide.weights, emit)?;
        let pruned = unpruned.clone().pruned(budget);
        Ok((unpruned, pruned))
    }
    fn scores(&self, features: &[usize], work: &mut usize) -> Vec<i32> {
        let classes = self.intents.len() + self.tags.len();
        let mut scores = vec![0; classes];
        for f in features {
            let start = (f * classes) as u32;
            let end = start + classes as u32;
            let first = self.weights.partition_point(|(key, _)| *key < start);
            for (key, value) in self.weights[first..]
                .iter()
                .take_while(|(key, _)| *key < end)
            {
                scores[(*key - start) as usize] += i32::from(*value);
                *work += 1;
            }
            *work += 1;
        }
        scores
    }
    pub fn predict(&self, text: &str) -> Result<Prediction, Error> {
        self.predict_using(text, |f, work| self.scores(f, work))
    }
    pub(crate) fn probe_predict(&self, text: &str, subwords: bool) -> Result<Prediction, Error> {
        self.predict_configured(text, subwords, |f, work| self.scores(f, work))
    }
    fn predict_using(
        &self,
        text: &str,
        scores: impl FnMut(&[usize], &mut usize) -> Vec<i32>,
    ) -> Result<Prediction, Error> {
        self.predict_configured(text, false, scores)
    }
    fn predict_configured(
        &self,
        text: &str,
        subwords: bool,
        mut scores: impl FnMut(&[usize], &mut usize) -> Vec<i32>,
    ) -> Result<Prediction, Error> {
        let words = tokens(text)?;
        let mut budget = self.budget();
        budget.subwords = subwords;
        let n = self.intents.len();
        let mut work = 0;
        let (intent, intent_margin) = best(
            &scores(&intent_features(&words, budget), &mut work),
            0,
            n,
            None,
        );
        if intent_margin < i32::from(self.minimum_margin) {
            return Err(Error::Ambiguous);
        }
        let (mut previous, mut previous2) = ("<start>", "<start2>");
        let mut tag_margin = i32::MAX;
        let mut selected = Vec::new();
        for index in 0..words.len() {
            let f = contextual(
                &word_features(&words, index, budget),
                previous,
                previous2,
                &self.intents[intent],
                budget,
            );
            let (tag, margin) = best(
                &scores(&f, &mut work),
                n,
                n + self.tags.len(),
                Some((&self.tags, previous)),
            );
            tag_margin = tag_margin.min(margin);
            if margin < i32::from(self.minimum_margin) {
                return Err(Error::Ambiguous);
            }
            let tag = &self.tags[tag - n];
            selected.push(tag.as_str());
            previous2 = previous;
            previous = tag;
        }
        let output = search::output(&words, &self.intents[intent], &selected)?;
        Ok(Prediction {
            output,
            intent_margin,
            tag_margin,
            work,
        })
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.weights.capacity() * std::mem::size_of::<(u32, i8)>()
            + (self.intents.capacity() + self.tags.capacity()) * std::mem::size_of::<String>()
            + self
                .intents
                .iter()
                .chain(&self.tags)
                .map(String::capacity)
                .sum::<usize>()
    }
    pub fn weights(&self) -> usize {
        self.weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compression_probe_reproduces_the_same_pruned_coefficients_and_honors_cancellation() {
        let examples = (0..32)
            .flat_map(|i| {
                ["send", "cancel"].map(move |intent| Example {
                    text: format!("{intent} letter to person-{i}"),
                    intent: intent.into(),
                    tags: ["O", "O", "O", "B:person"].map(str::to_owned).into(),
                })
            })
            .collect::<Vec<_>>();
        let b = Budget {
            seed: 47,
            buckets: 4096,
            epochs: 10,
            weights: 256,
            intent_weights: 128,
            subwords: false,
            shared_roles: false,
            minimum_margin: 1,
        };
        let (wide, pruned) = Model::compression_models(&examples, b, &mut |_| Ok(())).unwrap();
        let direct = Model::learn(&examples, b, None, &mut |_| Ok(())).unwrap();
        assert!(wide.weights() >= pruned.weights());
        assert_eq!(pruned.to_bytes().unwrap(), direct.to_bytes().unwrap());
        assert!(matches!(
            Model::compression_models(&examples, b, &mut |_| Err(Error::Budget)),
            Err(Error::Budget)
        ));
        let mut invalid = b;
        invalid.weights = 0;
        assert!(Model::compression_models(&examples, invalid, &mut |_| Ok(())).is_err());
    }
    #[test]
    fn shared_role_updates_tie_lexical_features_without_tying_boundary_context() {
        let mut learner = Learner {
            weights: vec![0; 6],
            totals: vec![0; 6],
            last: vec![0; 6],
            clock: 0,
            classes: 3,
        };
        let siblings = [None, Some(2), Some(1)];
        learner.update_shared(&[0, 1], 1, 0, Some((&[0], &siblings)));
        assert_eq!(learner.weights, vec![-8, 8, 8, -8, 8, 0]);
        learner.update_shared(&[0, 1], 2, 1, Some((&[0], &siblings)));
        assert_eq!(learner.weights, vec![-8, 8, 8, -8, 0, 8]);
        assert_eq!(learner.clock, 2);
    }
    #[test]
    fn quantization_preserves_head_scale_instead_of_clipping_each_large_weight() {
        assert_eq!(quantized(1016.0, 1016.0), 127);
        assert_eq!(quantized(-512.0, 1016.0), -64);
        assert_eq!(quantized(256.0, 1016.0), 32);
        assert_eq!(quantized(8.0, 16.0), 8);
        assert_eq!(quantized(0.0, 0.0), 0);
    }
}
