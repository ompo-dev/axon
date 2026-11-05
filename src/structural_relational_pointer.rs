//! Joint learned labels and source positions; provided candidates are not semantic answers.
//! Diagonal biaffine scoring, not the full BiLSTM parser of Dozat/Manning (2017).
use super::{
    Error, Lexicon,
    factorized::{Quantized, random, take},
    optimizer, ordered_context,
};
use crate::live_store::checksum;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
    mem::size_of,
};

pub const CONTEXT: &str = "axon/relational-pointer-v1";
pub const MODULE: &str = "semantic-relational-pointer-v1";
pub const MAX_WORDS: usize = 544;
pub const MAX_TEXT: usize = 8192;
const MAGIC: &[u8; 8] = b"AXRP0001";
const MAX_BYTES: usize = 196608;
#[path = "structural_pointer_read.rs"]
pub mod context_read;
use context_read::Read;

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub seed: u64,
    pub vocabulary: usize,
    pub dimensions: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub minibatch: usize,
    #[serde(default)]
    pub candidate_vocabulary: CandidateVocabulary,
    #[serde(default)]
    pub independent_shuffle: bool,
}
#[derive(Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CandidateVocabulary {
    #[default]
    Retain,
    Omit,
    Reallocate,
}
impl Budget {
    fn validate(self) -> Result<(), Error> {
        if self.seed == 0
            || !(16..=4096).contains(&self.vocabulary)
            || !(8..=16).contains(&self.dimensions)
            || self.epochs > 64
            || !(1..=64).contains(&self.minibatch)
            || !self.learning_rate.is_finite()
            || !(0.0001..=0.1).contains(&self.learning_rate)
        {
            return Err(Error::Invalid("pointer budget"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub label: String,
    pub left: usize,
    pub right: Option<usize>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub text: String,
    pub candidates: Vec<usize>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub input: Input,
    pub targets: Vec<Selection>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Label {
    name: String,
    arity: usize,
}
#[derive(Clone, Debug)]
pub struct Model {
    lexicon: Lexicon,
    labels: Box<[Label]>,
    dim: usize,
    input: Quantized,
    recurrent: Quantized,
    head: Quantized,
    bias: Vec<f32>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    revision: u8,
    words: usize,
    word_bytes: usize,
    dimensions: usize,
    labels: Vec<Label>,
}
#[derive(Serialize)]
pub struct Event {
    pub epoch: usize,
    pub examples: usize,
    pub errors: usize,
    pub updates: usize,
    pub loss: f64,
    pub order_fingerprint: String,
}
#[derive(Serialize)]
pub struct Prediction {
    pub selection: Selection,
    pub work: usize,
    pub workspace_bytes: usize,
    pub margin: Option<f32>,
    pub candidates: usize,
    pub blocks: usize,
}
pub struct Measurement {
    pub result: Result<Prediction, Error>,
    pub work: usize,
}

pub fn tokens(text: &str) -> Result<Vec<String>, Error> {
    if text.is_empty() || text.len() > MAX_TEXT || text.chars().any(char::is_control) {
        return Err(Error::Invalid("pointer text"));
    }
    let normalized = text.to_lowercase();
    let words = normalized
        .trim()
        .trim_end_matches(['.', '?', '!'])
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if words.is_empty() || words.len() > MAX_WORDS || words.iter().any(|w| w.len() > 256) {
        return Err(Error::Invalid("pointer tokens"));
    }
    Ok(words)
}
fn validate_input(input: &Input) -> Result<Vec<String>, Error> {
    let words = tokens(&input.text)?;
    if input.candidates.is_empty()
        || input.candidates.len() > 32
        || input.candidates.iter().any(|i| *i >= words.len())
        || input.candidates.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(Error::Invalid("pointer candidate positions"));
    }
    Ok(words)
}
fn valid_label(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 96
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
}
fn choices(
    labels: &[Label],
    count: usize,
) -> impl Iterator<Item = (usize, usize, Option<usize>)> + '_ {
    labels.iter().enumerate().flat_map(move |(c, label)| {
        (0..count).flat_map(move |a| {
            (0..if label.arity == 2 { count } else { 1 })
                .map(move |b| (c, a, (label.arity == 2).then_some(b)))
        })
    })
}
fn encode_words(
    words: &[String],
    lexicon: &Lexicon,
    dim: usize,
    mut row: impl FnMut(usize, usize) -> f32,
) -> (Vec<usize>, Vec<f32>) {
    let ids = words
        .iter()
        .map(|w| lexicon.index(w).map_or(2, |i| i + 3))
        .collect::<Vec<_>>();
    let embedded = ids
        .iter()
        .flat_map(|i| (0..dim).map(|d| (*i, d)))
        .map(|(i, d)| row(i, d))
        .collect();
    (ids, embedded)
}
struct Features {
    global: Vec<f32>,
    candidates: Vec<f32>,
    count: usize,
    dim: usize,
}
impl Features {
    fn new(
        cache: &ordered_context::Cache,
        embedded: &[f32],
        positions: &[usize],
        dim: usize,
    ) -> Self {
        let count = embedded.len() / dim;
        let mut global = Vec::with_capacity(3 * dim);
        global.extend_from_slice(cache.state(0, count - 1, dim));
        global.extend_from_slice(cache.state(1, 0, dim));
        global.extend_from_slice(&embedded[..dim]);
        let mut candidates = Vec::with_capacity(positions.len() * 4 * dim);
        for &i in positions {
            candidates.extend_from_slice(cache.state(0, i, dim));
            candidates.extend_from_slice(cache.state(1, i, dim));
            for neighbor in [i.checked_sub(1), (i + 1 < count).then_some(i + 1)] {
                if let Some(j) = neighbor {
                    candidates.extend_from_slice(&embedded[j * dim..(j + 1) * dim]);
                } else {
                    candidates.extend(std::iter::repeat_n(0.0, dim));
                }
            }
        }
        Self {
            global,
            candidates,
            count: positions.len(),
            dim,
        }
    }
    fn candidate(&self, index: usize) -> &[f32] {
        &self.candidates[index * 4 * self.dim..(index + 1) * 4 * self.dim]
    }
    fn workspace_bytes(&self) -> usize {
        (self.global.capacity() + self.candidates.capacity()) * 4
    }
    fn scatter(
        &self,
        global: &[f32],
        candidates: &[f32],
        positions: &[usize],
        words: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        self.scatter_core(global, candidates, positions, words, true)
    }
    fn scatter_core(
        &self,
        global: &[f32],
        candidates: &[f32],
        positions: &[usize],
        words: usize,
        neighbors: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let d = self.dim;
        let mut states = vec![0.0; 2 * words * d];
        let mut direct = vec![0.0; words * d];
        for k in 0..d {
            states[(words - 1) * d + k] += global[k];
            states[words * d + k] += global[d + k];
            direct[k] += global[2 * d + k];
        }
        for (j, &i) in positions.iter().enumerate() {
            let g = &candidates[j * 4 * d..(j + 1) * 4 * d];
            for k in 0..d {
                states[i * d + k] += g[k];
                states[(words + i) * d + k] += g[d + k];
                if neighbors && i > 0 {
                    direct[(i - 1) * d + k] += g[2 * d + k];
                }
                if neighbors && i + 1 < words {
                    direct[(i + 1) * d + k] += g[3 * d + k];
                }
            }
        }
        (states, direct)
    }
}
struct Scorer<'a> {
    head: &'a [f32],
    features: &'a Features,
    base: Vec<f32>,
    left: Vec<f32>,
    right: Vec<f32>,
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
impl<'a> Scorer<'a> {
    fn new(head: &'a [f32], bias: &[f32], features: &'a Features, work: &mut usize) -> Self {
        let d = features.dim;
        let n = bias.len();
        let k = features.count;
        let mut result = Self {
            head,
            features,
            base: Vec::with_capacity(n),
            left: Vec::with_capacity(n * k),
            right: Vec::with_capacity(n * k),
        };
        for (c, &b) in bias.iter().enumerate() {
            let row = &head[c * 15 * d..(c + 1) * 15 * d];
            result.base.push(b + dot(&row[..3 * d], &features.global));
            for i in 0..k {
                let h = features.candidate(i);
                result.left.push(dot(&row[3 * d..7 * d], h));
                result.right.push(dot(&row[7 * d..11 * d], h));
            }
        }
        *work += 3 * n * (3 * d + 2 * k * 4 * d);
        result
    }
    fn score(&self, c: usize, a: usize, b: Option<usize>, work: &mut usize) -> f32 {
        let d = self.features.dim;
        let k = self.features.count;
        let mut result = self.base[c] + self.left[c * k + a];
        *work += 1;
        if let Some(b) = b {
            result += self.right[c * k + b];
            let weights = &self.head[c * 15 * d + 11 * d..(c + 1) * 15 * d];
            result += weights
                .iter()
                .zip(self.features.candidate(a))
                .zip(self.features.candidate(b))
                .map(|((w, a), b)| w * a * b)
                .sum::<f32>();
            *work += 3 * 4 * d + 2;
        }
        result
    }
    fn workspace_bytes(&self) -> usize {
        (self.base.capacity() + self.left.capacity() + self.right.capacity()) * 4
    }
}
struct HeadGradient {
    head: Vec<f32>,
    bias: Vec<f32>,
    global: Vec<f32>,
    candidates: Vec<f32>,
    loss: f64,
    error: bool,
}
fn head_gradient(
    head: &[f32],
    bias: &[f32],
    features: &Features,
    labels: &[Label],
    gold: &BTreeSet<(usize, usize, Option<usize>)>,
) -> Result<HeadGradient, Error> {
    let mut work = 0;
    let scorer = Scorer::new(head, bias, features, &mut work);
    let options = choices(labels, features.count).collect::<Vec<_>>();
    let scores = options
        .iter()
        .map(|&(c, a, b)| scorer.score(c, a, b, &mut work))
        .collect::<Vec<_>>();
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let gold_max = options
        .iter()
        .zip(&scores)
        .filter(|(p, _)| gold.contains(p))
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() || !gold_max.is_finite() {
        return Err(Error::Invalid("pointer target domain"));
    }
    let total = scores
        .iter()
        .map(|s| (f64::from(*s) - f64::from(maximum)).exp())
        .sum::<f64>();
    let gold_total = options
        .iter()
        .zip(&scores)
        .filter(|(p, _)| gold.contains(p))
        .map(|(_, s)| (f64::from(*s) - f64::from(gold_max)).exp())
        .sum::<f64>();
    let mut dh = vec![0.0; head.len()];
    let mut db = vec![0.0; bias.len()];
    let mut dg = vec![0.0; features.global.len()];
    let mut dc = vec![0.0; features.candidates.len()];
    let d = features.dim;
    let mut best = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    for (&(c, a, b), &s) in options.iter().zip(&scores) {
        let error = ((f64::from(s) - f64::from(maximum)).exp() / total
            - if gold.contains(&(c, a, b)) {
                (f64::from(s) - f64::from(gold_max)).exp() / gold_total
            } else {
                0.0
            }) as f32;
        db[c] += error;
        let start = c * 15 * d;
        for k in 0..3 * d {
            dh[start + k] += error * features.global[k];
            dg[k] += error * head[start + k];
        }
        for k in 0..4 * d {
            dh[start + 3 * d + k] += error * features.candidate(a)[k];
            dc[a * 4 * d + k] += error * head[start + 3 * d + k];
            if let Some(b) = b {
                let (x, y) = (features.candidate(a)[k], features.candidate(b)[k]);
                dh[start + 7 * d + k] += error * y;
                dh[start + 11 * d + k] += error * x * y;
                dc[a * 4 * d + k] += error * head[start + 11 * d + k] * y;
                dc[b * 4 * d + k] +=
                    error * (head[start + 7 * d + k] + head[start + 11 * d + k] * x);
            }
        }
    }
    Ok(HeadGradient {
        head: dh,
        bias: db,
        global: dg,
        candidates: dc,
        loss: f64::from(maximum) - f64::from(gold_max) + total.ln() - gold_total.ln(),
        error: !gold.contains(&options[best]),
    })
}
fn accumulate(
    example: &Example,
    lexicon: &Lexicon,
    labels: &[Label],
    d: usize,
    parameters: [&[f32]; 3],
    context: &ordered_context::Train,
    gradients: &mut [Vec<f32>; 3],
) -> Result<(f64, bool), Error> {
    accumulate_read(
        example,
        lexicon,
        labels,
        d,
        (Read::Neighbors, parameters),
        context,
        gradients,
    )
}
fn accumulate_read(
    example: &Example,
    lexicon: &Lexicon,
    labels: &[Label],
    d: usize,
    parameters: (Read, [&[f32]; 3]),
    context: &ordered_context::Train,
    gradients: &mut [Vec<f32>; 3],
) -> Result<(f64, bool), Error> {
    let (read, [input, output, bias]) = parameters;
    let rn = 2 * d * d;
    let words = validate_input(&example.input)?;
    let (ids, embedded) = encode_words(&words, lexicon, d, |i, k| input[i * d + k]);
    let cache = context.encode_streamed(&embedded, 32)?;
    let (features, reader) =
        context_read::features(read, &cache, &embedded, &example.input.candidates, d, true);
    let gold = example
        .targets
        .iter()
        .map(|t| {
            (
                labels.iter().position(|l| l.name == t.label).unwrap(),
                example.input.candidates.binary_search(&t.left).unwrap(),
                t.right
                    .map(|r| example.input.candidates.binary_search(&r).unwrap()),
            )
        })
        .collect::<BTreeSet<_>>();
    let head = head_gradient(&output[rn..], &bias[2 * d..], &features, labels, &gold)?;
    let (ds, mut direct) = reader.scatter(
        &features,
        &cache,
        &embedded,
        &head.global,
        &head.candidates,
        &example.input.candidates,
    );
    let recurrent = context.gradient_states(&cache, &ds)?;
    let (dw, drb) = recurrent.parameters();
    for (dst, src) in gradients[1][..rn].iter_mut().zip(dw) {
        *dst += src;
    }
    for (dst, src) in gradients[1][rn..].iter_mut().zip(head.head) {
        *dst += src;
    }
    for (dst, src) in gradients[2][..2 * d].iter_mut().zip(drb) {
        *dst += src;
    }
    for (dst, src) in gradients[2][2 * d..].iter_mut().zip(head.bias) {
        *dst += src;
    }
    for (x, g) in direct.iter_mut().zip(&recurrent.inputs) {
        *x += g;
    }
    for (j, id) in ids.iter().enumerate() {
        for k in 0..d {
            gradients[0][id * d + k] += direct[j * d + k];
        }
    }
    Ok((head.loss, head.error))
}
impl Model {
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        Self::learn_read(examples, budget, Read::Neighbors, emit)
    }
    fn learn_read(
        examples: &[Example],
        budget: Budget,
        read: Read,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        budget.validate()?;
        if !(4..=10000).contains(&examples.len()) {
            return Err(Error::Invalid("pointer training count"));
        }
        let mut counts = BTreeMap::<String, usize>::new();
        let mut candidate_words = BTreeSet::new();
        let mut classes = BTreeMap::new();
        for e in examples {
            let words = validate_input(&e.input)?;
            if e.targets.is_empty() || e.targets.len() > 8 {
                return Err(Error::Invalid("pointer targets"));
            }
            for &i in &e.input.candidates {
                candidate_words.insert(words[i].clone());
            }
            for w in words {
                *counts.entry(w).or_default() += 1;
            }
            for t in &e.targets {
                if !valid_label(&t.label)
                    || !e.input.candidates.contains(&t.left)
                    || t.right.is_some_and(|r| !e.input.candidates.contains(&r))
                {
                    return Err(Error::Invalid("pointer teaching source"));
                }
                let arity = 1 + usize::from(t.right.is_some());
                if classes
                    .insert(t.label.clone(), arity)
                    .is_some_and(|old| old != arity)
                {
                    return Err(Error::Invalid("pointer label arity"));
                }
            }
        }
        if !(2..=16).contains(&classes.len()) {
            return Err(Error::Invalid("pointer label count"));
        }
        let labels = classes
            .into_iter()
            .map(|(name, arity)| Label { name, arity })
            .collect::<Vec<_>>();
        let mut vocabulary = counts.into_iter().collect::<Vec<_>>();
        if budget.candidate_vocabulary == CandidateVocabulary::Reallocate {
            vocabulary.retain(|(word, _)| !candidate_words.contains(word));
        }
        vocabulary.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        vocabulary.truncate(budget.vocabulary);
        if budget.candidate_vocabulary == CandidateVocabulary::Omit {
            vocabulary.retain(|(word, _)| !candidate_words.contains(word));
        }
        vocabulary.sort_by(|a, b| a.0.cmp(&b.0));
        let raw = vocabulary
            .iter()
            .flat_map(|(w, _)| w.bytes().chain(std::iter::once(0)))
            .collect::<Vec<_>>();
        let lexicon = Lexicon::from_bytes(&raw, vocabulary.len())?;
        let d = budget.dimensions;
        let rn = 2 * d * d;
        let hn = labels.len() * 15 * d;
        let mut state = budget.seed;
        let mut input = (0..(lexicon.len() + 3) * d)
            .map(|_| {
                ((random(&mut state) >> 40) as f32 / (1u32 << 24) as f32 - 0.5) / (d as f32).sqrt()
            })
            .collect::<Vec<_>>();
        let mut context = ordered_context::Train::new(d, budget.seed)?;
        let mut output = context.weights.clone();
        output.resize(rn + hn, 0.0);
        let mut bias = vec![0.0; 2 * d + labels.len()];
        let mut adam = optimizer::Adam::new([input.len(), output.len(), bias.len()])?;
        let mut order = (0..examples.len()).collect::<Vec<_>>();
        if budget.independent_shuffle {
            state = budget.seed.wrapping_add(0xd1b54a32d192ed03).max(1);
        }
        for epoch in 0..budget.epochs {
            for j in (1..order.len()).rev() {
                let k = random(&mut state) as usize % (j + 1);
                order.swap(j, k);
            }
            let mut errors = 0;
            let mut loss = 0.0;
            let mut batch = 0;
            for &i in &order {
                let (l, error) = if read == Read::Neighbors {
                    accumulate(
                        &examples[i],
                        &lexicon,
                        &labels,
                        d,
                        [&input, &output, &bias],
                        &context,
                        &mut adam.gradients,
                    )?
                } else {
                    accumulate_read(
                        &examples[i],
                        &lexicon,
                        &labels,
                        d,
                        (read, [&input, &output, &bias]),
                        &context,
                        &mut adam.gradients,
                    )?
                };
                loss += l;
                errors += usize::from(error);
                batch += 1;
                if batch == budget.minibatch || i == *order.last().unwrap() {
                    adam.step(
                        [&mut input, &mut output, &mut bias],
                        batch,
                        budget.learning_rate,
                    )?;
                    context.weights.copy_from_slice(&output[..rn]);
                    context.bias.copy_from_slice(&bias[..2 * d]);
                    batch = 0;
                }
            }
            emit(&Event {
                epoch: epoch + 1,
                examples: examples.len(),
                errors,
                updates: adam.steps(),
                loss: loss / examples.len() as f64,
                order_fingerprint: format!(
                    "{:016x}",
                    checksum(
                        &order
                            .iter()
                            .flat_map(|&i| (i as u64).to_le_bytes())
                            .collect::<Vec<_>>()
                    )
                ),
            })?;
        }
        let model = Self {
            lexicon,
            labels: labels.into_boxed_slice(),
            dim: d,
            input: Quantized::encode(&input, d),
            recurrent: Quantized::encode(&output[..rn], d),
            head: Quantized::encode(&output[rn..], 15 * d),
            bias,
        };
        model.to_bytes()?;
        Ok(model)
    }
    pub fn words(&self) -> usize {
        self.lexicon.len()
    }
    pub fn candidate_lexemes_retained(&self, examples: &[Example]) -> Result<usize, Error> {
        let mut candidate_words = BTreeSet::new();
        for e in examples {
            let words = validate_input(&e.input)?;
            for &i in &e.input.candidates {
                candidate_words.insert(words[i].clone());
            }
        }
        Ok(candidate_words
            .iter()
            .filter(|w| self.lexicon.index(w).is_some())
            .count())
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>() + self.lexicon.retained_bytes() - size_of::<Lexicon>()
            + self.input.retained_bytes()
            + self.recurrent.retained_bytes()
            + self.head.retained_bytes()
            + self.bias.capacity() * 4
            + self.labels.len() * size_of::<Label>()
            + self.labels.iter().map(|l| l.name.capacity()).sum::<usize>()
    }
    pub fn measure(&self, input: &Input, max_work: usize, max_workspace: usize) -> Measurement {
        self.measure_read(input, max_work, max_workspace, Read::Neighbors)
    }
    fn measure_read(
        &self,
        input: &Input,
        max_work: usize,
        max_workspace: usize,
        read: Read,
    ) -> Measurement {
        let mut work = 0;
        let result = (|| {
            let words = validate_input(input)?;
            let d = self.dim;
            let count = words.len();
            let k = input.candidates.len();
            let classes = self.labels.len();
            let word_bytes = words.capacity() * size_of::<String>()
                + words.iter().map(String::capacity).sum::<usize>();
            let reservation = word_bytes
                + count * (size_of::<usize>() + 3 * d * 4)
                + (3 * d + 4 * d * k + classes * (1 + 2 * k) + self.head.values.len()) * 4
                + read.scratch_bytes(count);
            if max_work == 0 || max_work > 2_000_000 || reservation > max_workspace {
                return Err(Error::Budget);
            }
            let (ids, embedded) = encode_words(&words, &self.lexicon, d, |i, k| {
                f32::from(self.input.values[i * d + k]) * self.input.scales[i]
            });
            work += count * d * 2;
            let cache = ordered_context::encode_states(
                &embedded,
                d,
                |direction, i, k| {
                    let row = direction * d + i;
                    f32::from(self.recurrent.values[row * d + k]) * self.recurrent.scales[row]
                },
                &self.bias[..2 * d],
                32,
            )?;
            work += cache.work;
            if work > max_work {
                return Err(Error::Budget);
            }
            let read_work = read.work(count, k, d);
            if work.saturating_add(read_work) > max_work {
                return Err(Error::Budget);
            }
            let (features, reader) =
                context_read::features(read, &cache, &embedded, &input.candidates, d, false);
            work += read_work;
            let head = self
                .head
                .values
                .iter()
                .enumerate()
                .map(|(i, w)| f32::from(*w) * self.head.scales[i / (15 * d)])
                .collect::<Vec<_>>();
            let scorer = Scorer::new(&head, &self.bias[2 * d..], &features, &mut work);
            let workspace = word_bytes
                + ids.capacity() * size_of::<usize>()
                + embedded.capacity() * 4
                + cache.workspace_bytes()
                + features.workspace_bytes()
                + head.capacity() * 4
                + scorer.workspace_bytes()
                + reader.scratch_bytes;
            if workspace > max_workspace {
                return Err(Error::Budget);
            }
            let mut best = None;
            let mut second = f32::NEG_INFINITY;
            for (c, a, b) in choices(&self.labels, k) {
                let score = scorer.score(c, a, b, &mut work);
                if work > max_work {
                    return Err(Error::Budget);
                }
                if !score.is_finite() {
                    return Err(Error::Invalid("pointer finite score"));
                }
                if best.is_none_or(|(_, _, _, s)| score > s) {
                    if let Some((_, _, _, old)) = best {
                        second = old;
                    }
                    best = Some((c, a, b, score));
                } else if score > second {
                    second = score;
                }
            }
            let (c, a, b, score) = best.ok_or(Error::Unsupported)?;
            Ok(Prediction {
                selection: Selection {
                    label: self.labels[c].name.clone(),
                    left: input.candidates[a],
                    right: b.map(|j| input.candidates[j]),
                },
                work,
                workspace_bytes: workspace,
                margin: second.is_finite().then_some(score - second),
                candidates: k,
                blocks: count.div_ceil(32),
            })
        })();
        Measurement { result, work }
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let header = serde_json::to_vec(&Header {
            revision: 1,
            words: self.lexicon.len(),
            word_bytes: self.lexicon.bytes.len(),
            dimensions: self.dim,
            labels: self.labels.to_vec(),
        })
        .map_err(|e| Error::Storage(e.to_string()))?;
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend(header);
        bytes.extend_from_slice(&self.lexicon.bytes);
        self.input.append(&mut bytes);
        self.recurrent.append(&mut bytes);
        self.head.append(&mut bytes);
        for n in &self.bias {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if !(20..=MAX_BYTES).contains(&bytes.len()) || bytes.get(..8) != Some(MAGIC) {
            return Err(Error::Invalid("pointer header"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if checksum(body).to_le_bytes() != digest {
            return Err(Error::Invalid("pointer checksum"));
        }
        let mut cursor = Cursor::new(body);
        cursor.set_position(8);
        let n = u32::from_le_bytes(take(&mut cursor)?) as usize;
        let end = 12usize
            .checked_add(n)
            .filter(|n| *n <= body.len())
            .ok_or(Error::Invalid("pointer header size"))?;
        let h: Header =
            serde_json::from_slice(&body[12..end]).map_err(|e| Error::Storage(e.to_string()))?;
        let d = h.dimensions;
        if h.revision != 1
            || !(8..=16).contains(&d)
            || !(2..=16).contains(&h.labels.len())
            || h.labels
                .iter()
                .any(|l| !valid_label(&l.name) || !(1..=2).contains(&l.arity))
            || h.labels.windows(2).any(|p| p[0].name >= p[1].name)
        {
            return Err(Error::Invalid("pointer schema"));
        }
        let stop = end
            .checked_add(h.word_bytes)
            .filter(|n| *n <= body.len())
            .ok_or(Error::Invalid("pointer lexicon size"))?;
        let lexicon = Lexicon::from_bytes(&body[end..stop], h.words)?;
        cursor.set_position(stop as u64);
        let input = Quantized::read(&mut cursor, h.words + 3, d)?;
        let recurrent = Quantized::read(&mut cursor, 2 * d, d)?;
        let head = Quantized::read(&mut cursor, h.labels.len(), 15 * d)?;
        let mut bias = Vec::with_capacity(2 * d + h.labels.len());
        for _ in 0..2 * d + h.labels.len() {
            let n = f32::from_le_bytes(take(&mut cursor)?);
            if !n.is_finite() || n.abs() > 16.0 {
                return Err(Error::Invalid("pointer bias"));
            }
            bias.push(n);
        }
        if cursor.position() != body.len() as u64 {
            return Err(Error::Invalid("pointer trailing bytes"));
        }
        Ok(Self {
            lexicon,
            labels: h.labels.into_boxed_slice(),
            dim: d,
            input,
            recurrent,
            head,
            bias,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn entire_joint_gradient_matches_finite_differences() {
        let d = 8;
        let lexicon = Lexicon::from_bytes(b"a\0b\0c\0", 3).unwrap();
        let labels = vec![
            Label {
                name: "one".into(),
                arity: 1,
            },
            Label {
                name: "two".into(),
                arity: 2,
            },
        ];
        let example = Example {
            input: Input {
                text: "a b c a".into(),
                candidates: vec![0, 2, 3],
            },
            targets: vec![
                Selection {
                    label: "two".into(),
                    left: 0,
                    right: Some(3),
                },
                Selection {
                    label: "two".into(),
                    left: 3,
                    right: Some(0),
                },
            ],
        };
        for read in [Read::Neighbors, Read::Uniform, Read::Attention] {
            let mut parameters = [
                (0..48)
                    .map(|i| (i as f32 - 23.0) * 0.008)
                    .collect::<Vec<_>>(),
                (0..2 * d * d + 2 * 15 * d)
                    .map(|i| ((i % 23) as f32 - 11.0) * 0.009)
                    .collect(),
                (0..2 * d + 2).map(|i| (i as f32 - 8.0) * 0.007).collect(),
            ];
            if read == Read::Attention {
                for x in &mut parameters[0] {
                    *x *= 2.5;
                }
                for x in &mut parameters[1] {
                    *x *= 2.0;
                }
            }
            let evaluate = |p: &[Vec<f32>; 3]| {
                let mut context = ordered_context::Train::new(d, 47).unwrap();
                context.weights.copy_from_slice(&p[1][..2 * d * d]);
                context.bias.copy_from_slice(&p[2][..2 * d]);
                let mut g = p.clone().map(|v| vec![0.0; v.len()]);
                let loss = accumulate_read(
                    &example,
                    &lexicon,
                    &labels,
                    d,
                    (read, [&p[0], &p[1], &p[2]]),
                    &context,
                    &mut g,
                )
                .unwrap()
                .0;
                (loss, g)
            };
            let (_, gradient) = evaluate(&parameters);
            for (group, indices) in [
                (0, vec![24, 29, 40, 47]),
                (1, vec![0, 63, 127, 128, 159, 241, 367]),
                (2, vec![0, 9, 16, 17]),
            ] {
                for i in indices {
                    let old = parameters[group][i];
                    parameters[group][i] = old + 0.001;
                    let high = evaluate(&parameters).0;
                    parameters[group][i] = old - 0.001;
                    let low = evaluate(&parameters).0;
                    parameters[group][i] = old;
                    let numerical = (high - low) / 0.002;
                    assert!(
                        (numerical - f64::from(gradient[group][i])).abs() < 0.0003,
                        "{read:?} group{group}:{i} {numerical} vs {}",
                        gradient[group][i]
                    );
                }
            }
        }
    }
    #[test]
    fn candidate_vocabulary_policy_preserves_defaults_and_variable_invariance() {
        let legacy = r#"{"seed":47,"vocabulary":16,"dimensions":8,"epochs":2,"learning_rate":0.01,"minibatch":4}"#;
        let default: Budget = serde_json::from_str(legacy).unwrap();
        assert!(default.candidate_vocabulary == CandidateVocabulary::Retain);
        assert!(!default.independent_shuffle);
        let examples = (0..32)
            .map(|i| Example {
                input: Input {
                    text: format!(
                        "{} red {} blue {} filler rare-{i}",
                        if i % 2 == 0 { "pick-red" } else { "pick-blue" },
                        i + 2,
                        i + 100
                    ),
                    candidates: vec![2, 4],
                },
                targets: vec![Selection {
                    label: if i % 2 == 0 { "red" } else { "blue" }.into(),
                    left: if i % 2 == 0 { 2 } else { 4 },
                    right: None,
                }],
            })
            .collect::<Vec<_>>();
        let original = Model::learn(&examples, default, &mut |_| Ok(())).unwrap();
        let explicit = Model::learn(
            &examples,
            Budget {
                candidate_vocabulary: CandidateVocabulary::Retain,
                independent_shuffle: false,
                ..default
            },
            &mut |_| Ok(()),
        )
        .unwrap();
        assert_eq!(original.to_bytes().unwrap(), explicit.to_bytes().unwrap());
        let mut orders = Vec::new();
        let mut lengths = Vec::new();
        for policy in [
            CandidateVocabulary::Retain,
            CandidateVocabulary::Omit,
            CandidateVocabulary::Reallocate,
        ] {
            let mut seen = Vec::new();
            let model = Model::learn(
                &examples,
                Budget {
                    candidate_vocabulary: policy,
                    independent_shuffle: true,
                    ..default
                },
                &mut |event| {
                    seen.push(event.order_fingerprint.clone());
                    Ok(())
                },
            )
            .unwrap();
            orders.push(seen);
            lengths.push(model.words());
            if policy != CandidateVocabulary::Retain {
                assert_eq!(model.candidate_lexemes_retained(&examples).unwrap(), 0);
                let first = Input {
                    text: "pick-red red 2 blue 100 filler rare-0".into(),
                    candidates: vec![2, 4],
                };
                let changed = Input {
                    text: "pick-red red 719 blue -431 filler rare-0".into(),
                    candidates: vec![2, 4],
                };
                let a = model.measure(&first, 2_000_000, 262144).result.unwrap();
                let b = model.measure(&changed, 2_000_000, 262144).result.unwrap();
                assert_eq!(a.selection, b.selection);
                assert_eq!(a.margin, b.margin);
                assert_eq!(a.work, b.work);
                let restored = Model::from_bytes(&model.to_bytes().unwrap()).unwrap();
                assert_eq!(restored.retained_bytes(), model.retained_bytes());
                assert_eq!(
                    restored
                        .measure(&changed, 2_000_000, 262144)
                        .result
                        .unwrap()
                        .selection,
                    a.selection
                );
            } else {
                assert!(model.candidate_lexemes_retained(&examples).unwrap() > 0);
            }
        }
        assert_eq!(orders[0], orders[1]);
        assert_eq!(orders[1], orders[2]);
        assert!(lengths[1] < lengths[0]);
        assert_eq!(lengths[2], lengths[0]);
        assert!(
            serde_json::from_str::<Budget>(&legacy.replace(
                "\"seed\":47",
                "\"candidate_vocabulary\":\"magic\",\"seed\":47"
            ))
            .is_err()
        );
    }
    #[test]
    fn pointers_learn_positions_with_long_context_and_survive_strict_codec() {
        let mut examples = Vec::new();
        for n in 0..32 {
            for red in [true, false] {
                let padding = if n % 2 == 0 { 0 } else { 80 };
                let text = format!(
                    "{} red {} {} blue {}",
                    if red { "pick-red" } else { "pick-blue" },
                    n + 2,
                    vec!["filler"; padding].join(" "),
                    3 * n + 99
                )
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
                let b = 4 + padding;
                examples.push(Example {
                    input: Input {
                        text,
                        candidates: vec![2, b],
                    },
                    targets: vec![Selection {
                        label: if red { "red" } else { "blue" }.into(),
                        left: if red { 2 } else { b },
                        right: None,
                    }],
                });
            }
        }
        let model = Model::learn(
            &examples,
            Budget {
                seed: 47,
                vocabulary: 128,
                dimensions: 8,
                epochs: 32,
                learning_rate: 0.01,
                minibatch: 4,
                candidate_vocabulary: CandidateVocabulary::Retain,
                independent_shuffle: false,
            },
            &mut |_| Ok(()),
        )
        .unwrap();
        let bytes = model.to_bytes().unwrap();
        let restored = Model::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, restored.to_bytes().unwrap());
        assert_eq!(model.retained_bytes(), restored.retained_bytes());
        for red in [true, false] {
            let input = Input {
                text: format!(
                    "{} red 719 {} blue 813",
                    if red { "pick-red" } else { "pick-blue" },
                    vec!["filler"; 80].join(" ")
                ),
                candidates: vec![2, 84],
            };
            let prediction = restored.measure(&input, 2_000_000, 262144).result.unwrap();
            assert_eq!(prediction.selection.left, if red { 2 } else { 84 });
            assert_eq!(prediction.selection.label, if red { "red" } else { "blue" });
            assert!(prediction.blocks > 1);
            assert!(prediction.workspace_bytes < 262144);
            assert!(restored.measure(&input, 1, 262144).result.is_err());
            assert!(restored.measure(&input, 2_000_000, 1).result.is_err());
        }
        for i in [0, 7, 12, bytes.len() - 1] {
            let mut corrupt = bytes.clone();
            corrupt[i] ^= 1;
            assert!(Model::from_bytes(&corrupt).is_err());
        }
        for candidates in [vec![], vec![2, 2], vec![9], (0..33).collect()] {
            assert!(
                restored
                    .measure(
                        &Input {
                            text: "a b c".into(),
                            candidates
                        },
                        2_000_000,
                        262144
                    )
                    .result
                    .is_err()
            );
        }
        assert!(tokens(&vec!["x"; 545].join(" ")).is_err());
        assert!(tokens("a\nb").is_err());
        assert!(
            serde_json::from_str::<Input>(r#"{"text":"a","candidates":[0],"expected":1}"#).is_err()
        );
    }
}
