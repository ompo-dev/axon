//! Train-only lexical indices and shared word/subword vectors with explicit context positions.
//! Window representations are established methods, not an AXON invention:
//! https://www.jmlr.org/papers/v12/collobert11a.html
use super::*;
use crate::live_store::checksum;
use factorized::{Quantized, random, take};
use std::{collections::BTreeMap, io::Cursor};
const MAGIC: &[u8; 8] = b"AXSQ0006";
const DEEP_MAGIC: &[u8; 8] = b"AXSQ0007";
const REGIONS_MAGIC: &[u8; 8] = b"AXSQ0009";
const BYTE_MAGIC: &[u8; 8] = b"AXSQ0010";
const ORDERED_MAGIC: &[u8; 8] = b"AXSQ0011";
const ROLES_MAGIC: &[u8; 8] = b"AXSQ0012";
const SPLIT_MAGIC: &[u8; 8] = b"AXSQ0013";
const GRAMS_MAGIC: &[u8; 8] = b"AXSQ0014";
const SPLIT_ORDERED_MAGIC: &[u8; 8] = b"AXSQ0015";
const BOUNDARY_MAGIC: &[u8; 8] = b"AXSQ0016";
const ROLES_SCALE: f32 = 1.0 / 16.0;
const POSITIONS: usize = 7;
const BOUNDARY_ROWS: usize = 4;
type PathConstraint<'a> = dyn Fn(&str, &[&str], usize, &str) -> bool + 'a;
#[path = "structural_byte_convolution.rs"]
mod byte_convolution;
#[path = "structural_lexical_graph.rs"]
mod lexical_graph;
#[path = "structural_optimizer.rs"]
mod optimizer;
pub use lexical_graph::Graph as LexicalGraph;
#[path = "structural_ordered_context.rs"]
mod ordered_context;
#[path = "structural_relational_pointer.rs"]
pub mod relational_pointer;
#[path = "structural_synaptic.rs"]
pub mod synaptic;
pub const CONTEXT: &str = "axon/lexical-encoder-v1";

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub seed: u64,
    pub vocabulary: usize,
    pub characters: usize,
    pub dimensions: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub balanced: bool,
    pub minimum_margin: u16,
    #[serde(default)]
    pub collision_control: bool,
    #[serde(default)]
    pub hidden: usize,
    #[serde(default)]
    pub input_dropout: f32,
    #[serde(default)]
    pub relative_regions: bool,
    #[serde(default)]
    pub byte_convolution: bool,
    #[serde(default)]
    pub freeze_byte_convolution: bool,
    #[serde(default)]
    pub ordered_context: bool,
    #[serde(default)]
    pub freeze_ordered_context: bool,
    #[serde(default)]
    pub mean_token_loss: bool,
    #[serde(default, skip_serializing_if = "zero")]
    pub minibatch: usize,
    #[serde(default, skip_serializing_if = "disabled")]
    pub typed_roles: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    pub constrained_loss: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    pub split_heads: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    pub boundary_features: bool,
}
impl Budget {
    fn validate(self) -> Result<(), Error> {
        if self.seed == 0
            || (self.split_heads
                && (self.hidden == 0
                    || !self.relative_regions
                    || self.byte_convolution
                    || self.typed_roles))
            || (self.boundary_features && !self.relative_regions)
            || (self.constrained_loss && !self.typed_roles)
            || (self.typed_roles
                && (!self.relative_regions || self.byte_convolution || self.ordered_context))
            || (self.minibatch != 0
                && (!(8..=64).contains(&self.minibatch)
                    || self.byte_convolution
                    || self.ordered_context))
            || (self.byte_convolution && self.characters != 0)
            || (self.freeze_byte_convolution && !self.byte_convolution)
            || (self.ordered_context && (self.byte_convolution || self.characters != 0))
            || (self.freeze_ordered_context && !self.ordered_context)
            || !self.input_dropout.is_finite()
            || !(0.0..=0.75).contains(&self.input_dropout)
            || (self.hidden != 0 && !(8..=64).contains(&self.hidden))
            || !(16..=4096).contains(&self.vocabulary)
            || (self.characters != 0
                && (!self.characters.is_power_of_two() || !(256..=1024).contains(&self.characters)))
            || !(8..=16).contains(&self.dimensions)
            || !(1..=32).contains(&self.epochs)
            || !self.learning_rate.is_finite()
            || !(0.001..=1.0).contains(&self.learning_rate)
            || !(1..=255).contains(&self.minimum_margin)
        {
            return Err(Error::Invalid("lexical encoder budget"));
        }
        Ok(())
    }
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuxiliaryExample {
    pub text: String,
    pub labels: Vec<String>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Auxiliary {
    pub weight: f32,
    pub interval: usize,
    pub examples: Vec<AuxiliaryExample>,
}
impl Auxiliary {
    fn labels(&self) -> Result<Vec<String>, Error> {
        if !self.weight.is_finite()
            || !(0.0..=1.0).contains(&self.weight)
            || !(1..=64).contains(&self.interval)
            || !(4..=10000).contains(&self.examples.len())
        {
            return Err(Error::Invalid("auxiliary grammar budget"));
        }
        let mut labels = BTreeSet::new();
        for example in &self.examples {
            if tokens_mode(&example.text, false)?.len() != example.labels.len()
                || example.labels.iter().any(|p| !dependency::pos(p))
            {
                return Err(Error::Invalid("auxiliary grammar supervision"));
            }
            labels.extend(example.labels.iter().cloned());
        }
        if labels.len() < 2 {
            return Err(Error::Invalid("auxiliary grammar classes"));
        }
        Ok(labels.into_iter().collect())
    }
}
pub struct TrainingEvent {
    pub primary: Event,
    pub auxiliary_sentences: usize,
    pub auxiliary_words: usize,
    pub auxiliary_errors: usize,
    pub optimizer_steps: usize,
    pub lexical_edge_updates: usize,
    pub primary_tag_normalization_terms: usize,
}
#[derive(Clone, Debug)]
struct Lexicon {
    bytes: Box<[u8]>,
    offsets: Box<[u32]>,
}
impl Lexicon {
    fn from_bytes(bytes: &[u8], count: usize) -> Result<Self, Error> {
        if count == 0 || count > 4096 || bytes.len() > 65536 || bytes.last() != Some(&0) {
            return Err(Error::Invalid("encoder lexicon bounds"));
        }
        let mut offsets = vec![0u32];
        let mut previous: Option<&str> = None;
        for (i, b) in bytes.iter().enumerate() {
            if *b != 0 {
                continue;
            }
            let word = std::str::from_utf8(&bytes[*offsets.last().unwrap() as usize..i])
                .map_err(|_| Error::Invalid("encoder lexicon UTF8"))?;
            if word.is_empty()
                || word.len() > 256
                || word.chars().any(|c| c.is_whitespace() || c.is_control())
                || word.to_lowercase() != word
                || previous.is_some_and(|p| p >= word)
            {
                return Err(Error::Invalid("encoder lexicon order/token"));
            }
            previous = Some(word);
            offsets.push((i + 1) as u32);
        }
        if offsets.len() != count + 1 {
            return Err(Error::Invalid("encoder lexicon count"));
        }
        Ok(Self {
            bytes: bytes.into(),
            offsets: offsets.into_boxed_slice(),
        })
    }
    fn word(&self, i: usize) -> &str {
        std::str::from_utf8(&self.bytes[self.offsets[i] as usize..self.offsets[i + 1] as usize - 1])
            .expect("validated lexical UTF8")
    }
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
    fn index(&self, word: &str) -> Option<usize> {
        let mut low = 0;
        let mut high = self.len();
        while low < high {
            let mid = (low + high) / 2;
            match self.word(mid).cmp(word) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.bytes.len() + self.offsets.len() * 4
    }
}
type Groups = [Vec<usize>; POSITIONS];
#[derive(Clone, Debug)]
pub struct Model {
    metadata: super::Model,
    lexicon: Lexicon,
    dimensions: usize,
    characters: usize,
    collision_control: bool,
    relative_regions: bool,
    byte_convolution: bool,
    ordered_context: bool,
    typed_roles: bool,
    split_heads: bool,
    word_grams: bool,
    boundary_features: bool,
    input: Quantized,
    output: Quantized,
    bias: Vec<f32>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    metadata: Wire,
    dimensions: usize,
    characters: usize,
    words: usize,
    word_bytes: usize,
    #[serde(default)]
    collision_control: bool,
    #[serde(default, skip_serializing_if = "zero")]
    hidden: usize,
    #[serde(default, skip_serializing_if = "disabled")]
    relative_regions: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    byte_convolution: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    ordered_context: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    typed_roles: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    split_heads: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    word_grams: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    boundary_features: bool,
}
fn zero(n: &usize) -> bool {
    *n == 0
}
fn disabled(flag: &bool) -> bool {
    !*flag
}
// shortcut: these feature flags are part of the encoded format identity;
// bundling them would obscure the one-to-one mapping used by the decoder.
#[allow(clippy::too_many_arguments)]
fn format_magic(
    word_grams: bool,
    split_heads: bool,
    typed_roles: bool,
    ordered_context: bool,
    byte_convolution: bool,
    relative_regions: bool,
    hidden: usize,
    boundary_features: bool,
) -> &'static [u8; 8] {
    if boundary_features {
        BOUNDARY_MAGIC
    } else if split_heads && ordered_context {
        SPLIT_ORDERED_MAGIC
    } else if word_grams {
        GRAMS_MAGIC
    } else if split_heads {
        SPLIT_MAGIC
    } else if typed_roles {
        ROLES_MAGIC
    } else if ordered_context {
        ORDERED_MAGIC
    } else if byte_convolution {
        BYTE_MAGIC
    } else if relative_regions {
        REGIONS_MAGIC
    } else if hidden == 0 {
        MAGIC
    } else {
        DEEP_MAGIC
    }
}
struct Learner {
    input: Vec<f32>,
    output: Vec<f32>,
    bias: Vec<f32>,
    dimensions: usize,
    hidden: usize,
    optimizer: Option<optimizer::Adam>,
    task_boundary: Option<usize>,
}
fn descend(value: &mut f32, delta: f32, bound: f32, accumulated: Option<&mut f32>) {
    if let Some(sum) = accumulated {
        *sum += delta;
    } else {
        *value = (*value - delta).clamp(-bound, bound);
    }
}
fn hidden(
    groups: &Groups,
    dimensions: usize,
    mut row: impl FnMut(usize, usize) -> f32,
) -> Vec<f32> {
    let mut h = vec![0.0; dimensions * POSITIONS];
    for (slot, group) in groups.iter().enumerate() {
        let norm = (group.len().max(1) as f32).sqrt();
        for id in group {
            for d in 0..dimensions {
                h[slot * dimensions + d] += row(*id, d) / norm;
            }
        }
    }
    for v in &mut h {
        *v = v.tanh();
    }
    h
}
fn logits(h: &[f32], bias: &[f32], mut weight: impl FnMut(usize, usize) -> f32) -> Vec<f32> {
    (0..bias.len())
        .map(|c| {
            bias[c]
                + h.iter()
                    .enumerate()
                    .map(|(d, v)| v * weight(c, d))
                    .sum::<f32>()
        })
        .collect()
}
fn log_scores(
    values: Vec<i32>,
    range: std::ops::Range<usize>,
    divisor: usize,
    work: &mut usize,
) -> Result<Vec<i64>, Error> {
    if range.is_empty() || range.end > values.len() || divisor == 0 {
        return Err(Error::Invalid("normalized score range"));
    }
    let maximum = range.clone().map(|i| values[i]).max().unwrap();
    let partition = range
        .clone()
        .map(|i| ((f64::from(values[i]) - f64::from(maximum)) / 1000.0).exp())
        .sum::<f64>()
        .ln();
    *work = work.checked_add(range.len() * 3).ok_or(Error::Budget)?;
    let mut scores = vec![0; values.len()];
    for i in range {
        scores[i] = ((((f64::from(values[i]) - f64::from(maximum)) / 1000.0 - partition)
            / divisor as f64)
            * 1_000_000.0)
            .round() as i64;
    }
    Ok(scores)
}
fn dropout_mask(width: usize, probability: f32, state: &mut u64) -> Vec<f32> {
    if probability == 0.0 {
        return Vec::new();
    }
    (0..width)
        .map(|_| {
            if (random(state) >> 40) as f32 / ((1u32 << 24) as f32) < probability {
                0.0
            } else {
                1.0 / (1.0 - probability)
            }
        })
        .collect()
}
impl Learner {
    fn update(
        &mut self,
        groups: &Groups,
        gold: usize,
        head: impl Iterator<Item = usize> + Clone,
        rate: f32,
        mask: &[f32],
        mut deferred: Option<(usize, &mut [f32])>,
    ) -> bool {
        let dim = self.dimensions;
        let width = dim * POSITIONS;
        let mixers = self.hidden * (1 + usize::from(self.task_boundary.is_some()));
        let branch = if self.task_boundary.is_some_and(|n| gold >= n) {
            self.hidden
        } else {
            0
        };
        let unmasked = hidden(groups, dim, |id, d| self.input[id * dim + d]);
        let h = unmasked
            .iter()
            .enumerate()
            .map(|(d, v)| v * mask.get(d).copied().unwrap_or(1.0))
            .collect::<Vec<_>>();
        let latent = if self.hidden == 0 {
            h.clone()
        } else {
            logits(&h, &self.bias[branch..branch + self.hidden], |c, d| {
                self.output[(branch + c) * width + d]
            })
            .into_iter()
            .map(f32::tanh)
            .collect()
        };
        let output_width = latent.len();
        let offset = mixers * width;
        let values = logits(&latent, &self.bias[mixers..], |c, d| {
            self.output[offset + c * output_width + d]
        });
        let guess = head
            .clone()
            .max_by(|a, b| values[*a].total_cmp(&values[*b]).then_with(|| b.cmp(a)))
            .unwrap();
        let maximum = head
            .clone()
            .map(|c| values[c])
            .fold(f32::NEG_INFINITY, f32::max);
        let total = head
            .clone()
            .map(|c| (values[c] - maximum).exp())
            .sum::<f32>();
        let mut gradient = vec![0.0; output_width];
        for c in head {
            let error = (values[c] - maximum).exp() / total - f32::from(c == gold);
            descend(
                &mut self.bias[mixers + c],
                rate * error,
                16.0,
                self.optimizer
                    .as_mut()
                    .map(|o| &mut o.gradients[2][mixers + c]),
            );
            for d in 0..output_width {
                let w = &mut self.output[offset + c * output_width + d];
                gradient[d] += error * *w;
                descend(
                    w,
                    rate * error * latent[d],
                    8.0,
                    self.optimizer
                        .as_mut()
                        .map(|o| &mut o.gradients[1][offset + c * output_width + d]),
                );
            }
        }
        if self.hidden != 0 {
            let mut back = vec![0.0; width];
            for j in 0..self.hidden {
                let g = gradient[j] * (1.0 - latent[j] * latent[j]);
                descend(
                    &mut self.bias[branch + j],
                    rate * g,
                    16.0,
                    self.optimizer
                        .as_mut()
                        .map(|o| &mut o.gradients[2][branch + j]),
                );
                for d in 0..width {
                    let w = &mut self.output[(branch + j) * width + d];
                    back[d] += g * *w;
                    descend(
                        w,
                        rate * g * h[d],
                        8.0,
                        self.optimizer
                            .as_mut()
                            .map(|o| &mut o.gradients[1][(branch + j) * width + d]),
                    );
                }
            }
            gradient = back;
        }
        for (d, (g, v)) in gradient.iter_mut().zip(unmasked).enumerate() {
            *g *= (1.0 - v * v) * mask.get(d).copied().unwrap_or(1.0);
        }
        for (slot, group) in groups.iter().enumerate() {
            let norm = (group.len().max(1) as f32).sqrt();
            for id in group {
                for d in 0..dim {
                    if let Some((base, deltas)) = deferred.as_mut()
                        && *id >= *base
                    {
                        deltas[(*id - *base) * dim + d] += rate * gradient[slot * dim + d] / norm;
                    } else {
                        let w = &mut self.input[id * dim + d];
                        if let Some(optimizer) = self.optimizer.as_mut() {
                            optimizer.gradients[0][id * dim + d] +=
                                rate * gradient[slot * dim + d] / norm;
                        } else {
                            *w = (*w - rate * gradient[slot * dim + d] / norm).clamp(-8.0, 8.0);
                        }
                    }
                }
            }
        }
        guess != gold
    }
}
enum ExtraCache {
    None,
    Bytes(Vec<byte_convolution::Cache>),
    Ordered(ordered_context::Cache, Vec<usize>),
}
impl ExtraCache {
    fn output(&self, i: usize, d: usize, dim: usize) -> f32 {
        match self {
            Self::None => unreachable!("No virtual rows without an encoder"),
            Self::Bytes(caches) => caches[i].output[d],
            Self::Ordered(cache, _) => cache.output[i * dim + d],
        }
    }
}
fn prepare_extra_training(
    learner: &mut Learner,
    convolution: Option<&byte_convolution::Train>,
    ordered: Option<&ordered_context::Train>,
    words: &[String],
    features: &[Vec<usize>],
    base: usize,
) -> Result<(ExtraCache, Vec<f32>), Error> {
    if let Some(ordered) = ordered {
        let rows = features.iter().map(|f| f[0]).collect::<Vec<_>>();
        let inputs = rows
            .iter()
            .flat_map(|r| {
                learner.input[r * learner.dimensions..(r + 1) * learner.dimensions]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        let cache = ordered.encode(&inputs)?;
        learner.input.truncate(base * learner.dimensions);
        learner.input.extend_from_slice(&cache.output);
        let deltas = vec![0.0; learner.input.len()];
        return Ok((ExtraCache::Ordered(cache, rows), deltas));
    }
    let Some(convolution) = convolution else {
        return Ok((ExtraCache::None, Vec::new()));
    };
    let caches = words
        .iter()
        .map(|w| convolution.encode(w))
        .collect::<Result<Vec<_>, _>>()?;
    learner.input.truncate(base * learner.dimensions);
    for c in &caches {
        learner.input.extend_from_slice(&c.output);
    }
    let deltas = vec![0.0; words.len() * learner.dimensions];
    Ok((ExtraCache::Bytes(caches), deltas))
}
fn apply_ordered_training(
    ordered: &mut ordered_context::Train,
    cache: &ExtraCache,
    deltas: &[f32],
    learner: &mut Learner,
    scale: f32,
    frozen: bool,
) -> Result<(), Error> {
    let ExtraCache::Ordered(cache, rows) = cache else {
        return Err(Error::Invalid("ordered training cache"));
    };
    let real_length = deltas.len() - cache.output.len();
    let gradient = ordered.gradient(cache, &deltas[real_length..])?;
    let mut input_gradient = deltas[..real_length].to_vec();
    for (i, row) in rows.iter().enumerate() {
        for d in 0..learner.dimensions {
            input_gradient[row * learner.dimensions + d] +=
                gradient.inputs[i * learner.dimensions + d];
        }
    }
    let applied_scale = ordered.apply(&gradient, scale, frozen, &input_gradient)?;
    for (value, gradient) in learner.input[..real_length].iter_mut().zip(input_gradient) {
        *value = (*value - applied_scale * gradient).clamp(-8.0, 8.0);
    }
    Ok(())
}
fn join_quantized(a: Quantized, b: Quantized) -> Quantized {
    Quantized {
        values: a
            .values
            .into_iter()
            .chain(b.values)
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .into_vec(),
        scales: a
            .scales
            .into_iter()
            .chain(b.scales)
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .into_vec(),
    }
}
impl Model {
    fn character_base(&self) -> usize {
        3 + self.lexicon.len()
    }
    fn tag_base(&self) -> usize {
        self.character_base() + self.characters
    }
    fn intent_base(&self) -> usize {
        self.tag_base() + self.metadata.tags.len()
    }
    fn input_rows(&self) -> usize {
        self.boundary_base()
            + if self.boundary_features {
                BOUNDARY_ROWS
            } else {
                0
            }
    }
    fn boundary_base(&self) -> usize {
        self.intent_base() + self.metadata.intents.len()
    }
    fn hidden_units(&self) -> usize {
        (self.bias.len()
            - self.metadata.intents.len()
            - self.metadata.tags.len()
            - if self.byte_convolution {
                self.dimensions
            } else if self.ordered_context {
                2 * self.dimensions
            } else {
                0
            })
            / (1 + usize::from(self.split_heads))
    }
    fn magic(&self) -> &'static [u8; 8] {
        format_magic(
            self.word_grams,
            self.split_heads,
            self.typed_roles,
            self.ordered_context,
            self.byte_convolution,
            self.relative_regions,
            self.hidden_units(),
            self.boundary_features,
        )
    }
    fn role_stride(&self) -> usize {
        self.metadata.tags.len().div_ceil(7)
    }
    fn role_cells(&self) -> usize {
        if self.typed_roles {
            self.metadata.intents.len() * self.role_stride()
        } else {
            0
        }
    }
    fn role_allowed(&self, intent: usize, tag: usize) -> bool {
        !self.typed_roles
            || self.output.values[self.output.values.len() - self.role_cells()
                + intent * self.role_stride()
                + tag / 7] as u8
                & (1 << (tag % 7))
                != 0
    }
    fn learn_role_schema(&self, examples: &[Example]) -> Vec<i8> {
        if !self.typed_roles {
            return Vec::new();
        }
        let mut cells = vec![0i8; self.role_cells()];
        let stride = self.role_stride();
        for intent in 0..self.metadata.intents.len() {
            cells[intent * stride] = 1;
        }
        for example in examples {
            let intent = self
                .metadata
                .intents
                .binary_search(&example.intent)
                .unwrap();
            for tag in &example.tags {
                if let Some(name) = role(tag) {
                    for prefix in ["B:", "I:"] {
                        let tag = self
                            .metadata
                            .tags
                            .iter()
                            .position(|t| t == &format!("{prefix}{name}"))
                            .unwrap();
                        cells[intent * stride + tag / 7] |= 1 << (tag % 7);
                    }
                }
            }
        }
        cells
    }
    fn word_features(&self, word: &str) -> Result<Vec<usize>, Error> {
        let mut f = vec![self.lexicon.index(word).map_or(2, |i| {
            if self.collision_control {
                3 + hash(
                    self.metadata.seed,
                    59,
                    &[word],
                    self.lexicon.len().next_power_of_two(),
                ) % self.lexicon.len()
            } else {
                i + 3
            }
        })];
        if self.characters != 0 {
            let chars = word.chars().collect::<Vec<_>>();
            for width in 2..=4.min(chars.len()) {
                for part in chars.windows(width) {
                    if f.len() >= 768 {
                        return Err(Error::Budget);
                    }
                    f.push(
                        self.character_base()
                            + hash(
                                self.metadata.seed,
                                60 + width as u8,
                                &[&part.iter().collect::<String>()],
                                self.characters,
                            ),
                    );
                }
            }
        }
        unique(&mut f);
        Ok(f)
    }
    fn prepare(&self, text: &str) -> Result<Vec<Vec<usize>>, Error> {
        self.prepare_using(text, true)
    }
    fn prepare_using(
        &self,
        text: &str,
        strip_terminal_punctuation: bool,
    ) -> Result<Vec<Vec<usize>>, Error> {
        let raw = tokens_mode(text, strip_terminal_punctuation)?;
        raw.iter()
            .enumerate()
            .map(|(i, w)| {
                let mut features = self.word_features(w)?;
                if self.word_grams {
                    let previous = i
                        .checked_sub(1)
                        .and_then(|j| raw.get(j))
                        .map(String::as_str)
                        .unwrap_or("<start>");
                    let next = raw.get(i + 1).map(String::as_str).unwrap_or("<end>");
                    let bucket = self.character_base();
                    features.push(
                        bucket
                            + hash(
                                self.metadata.seed,
                                80,
                                &[previous, w.as_str()],
                                self.characters,
                            ),
                    );
                    features.push(
                        bucket + hash(self.metadata.seed, 81, &[w.as_str(), next], self.characters),
                    );
                    features.push(
                        bucket
                            + hash(
                                self.metadata.seed,
                                82,
                                &[previous, w.as_str(), next],
                                self.characters,
                            ),
                    );
                    unique(&mut features);
                }
                if self.boundary_features {
                    let boundary = if raw.len() == 1 {
                        0
                    } else if i == 0 {
                        1
                    } else if i + 1 == raw.len() {
                        2
                    } else {
                        3
                    };
                    features.push(self.boundary_base() + boundary);
                }
                if self.byte_convolution || self.ordered_context {
                    features.push(self.input_rows() + i);
                }
                Ok(features)
            })
            .collect()
    }
    fn sentence(&self, words: &[Vec<usize>]) -> Groups {
        let mut groups: Groups = std::array::from_fn(|_| Vec::new());
        if self.ordered_context {
            // Read a contextual endpoint per region; averaging all recurrent states can cancel order.
            let mut endpoints = [None; POSITIONS];
            for (index, word) in words.iter().enumerate() {
                let region = index * POSITIONS / words.len();
                groups[region].extend(word.iter().copied().filter(|id| *id < self.input_rows()));
                endpoints[region] = Some(self.input_rows() + index);
            }
            for (group, endpoint) in groups.iter_mut().zip(endpoints) {
                unique(group);
                if let Some(endpoint) = endpoint {
                    group.push(endpoint);
                }
            }
            return groups;
        }
        if self.relative_regions {
            for (index, word) in words.iter().enumerate() {
                groups[index * POSITIONS / words.len()].extend(word);
            }
            for group in &mut groups {
                unique(group);
            }
            return groups;
        }
        groups[0] = words[0].clone();
        groups[4] = words.last().unwrap().clone();
        groups[2] = words.iter().flatten().copied().collect();
        unique(&mut groups[2]);
        groups
    }
    fn context(
        &self,
        words: &[Vec<usize>],
        index: usize,
        previous: Option<usize>,
        intent: usize,
    ) -> Groups {
        let mut groups: Groups = std::array::from_fn(|_| Vec::new());
        for (slot, offset) in (-2isize..=2).enumerate() {
            groups[slot] = index
                .checked_add_signed(offset)
                .and_then(|i| words.get(i))
                .cloned()
                .unwrap_or_else(|| vec![usize::from(offset >= 0)]);
        }
        if self.relative_regions {
            groups[0] = words[..index.saturating_sub(1)]
                .iter()
                .flatten()
                .copied()
                .collect();
            groups[4] = words[(index + 2).min(words.len())..]
                .iter()
                .flatten()
                .copied()
                .collect();
            for (slot, boundary) in [(0, 0), (4, 1)] {
                unique(&mut groups[slot]);
                if groups[slot].is_empty() {
                    groups[slot].push(boundary);
                }
            }
        }
        groups[5] = vec![previous.map_or(0, |p| self.tag_base() + p)];
        groups[6] = vec![self.intent_base() + intent];
        groups
    }
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        Self::learn_with_auxiliary(examples, budget, None, &mut |event| emit(&event.primary))
    }
    pub fn learn_with_auxiliary(
        examples: &[Example],
        budget: Budget,
        auxiliary: Option<&Auxiliary>,
        emit: &mut impl FnMut(&TrainingEvent) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        Self::learn_with_sources(examples, budget, auxiliary, None, emit)
    }
    pub fn learn_with_sources(
        examples: &[Example],
        budget: Budget,
        auxiliary: Option<&Auxiliary>,
        lexical: Option<&LexicalGraph>,
        emit: &mut impl FnMut(&TrainingEvent) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        Self::fit(examples, budget, auxiliary, lexical, None, emit)
    }
    pub fn refine_with_sources(
        examples: &[Example],
        budget: Budget,
        initial: &Self,
        auxiliary: Option<&Auxiliary>,
        lexical: Option<&LexicalGraph>,
        emit: &mut impl FnMut(&TrainingEvent) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        Self::fit(examples, budget, auxiliary, lexical, Some(initial), emit)
    }
    fn fit(
        examples: &[Example],
        budget: Budget,
        auxiliary: Option<&Auxiliary>,
        lexical: Option<&LexicalGraph>,
        initial: Option<&Self>,
        emit: &mut impl FnMut(&TrainingEvent) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        budget.validate()?;
        if initial.is_some_and(|m| {
            m.metadata.seed != budget.seed
                || m.dimensions != budget.dimensions
                || m.characters != budget.characters
                || m.hidden_units() != budget.hidden
                || m.relative_regions != budget.relative_regions
                || m.split_heads != budget.split_heads
                || m.typed_roles != budget.typed_roles
                || m.word_grams != (budget.split_heads && budget.characters != 0)
                || m.boundary_features != budget.boundary_features
                || m.collision_control
                || budget.collision_control
                || m.byte_convolution
                || budget.byte_convolution
                || m.ordered_context
                || budget.ordered_context
                || m.lexicon.len() > budget.vocabulary
        }) {
            return Err(Error::Invalid("encoder refinement schema"));
        }
        let auxiliary_labels = auxiliary
            .map(Auxiliary::labels)
            .transpose()?
            .unwrap_or_default();
        let auxiliary = auxiliary.filter(|a| a.weight > 0.0);
        if budget.split_heads && auxiliary.is_some() {
            return Err(Error::Invalid("split mixer auxiliary not supported"));
        }
        if lexical.is_some_and(|g| g.weight > 0.0)
            && (budget.minibatch != 0
                || budget.byte_convolution
                || budget.ordered_context
                || budget.collision_control
                || budget.characters != 0
                || auxiliary.is_some())
        {
            return Err(Error::Invalid("lexical graph training combination"));
        }
        if budget.minibatch != 0 && auxiliary.is_some() {
            return Err(Error::Invalid("minibatch auxiliary not supported"));
        }
        let (intents, tags) = class_labels(examples, initial.map(|m| &m.metadata))?;
        let n = intents.len();
        let classes = n + tags.len();
        let auxiliary_classes = if auxiliary.is_some() {
            auxiliary_labels.len()
        } else {
            0
        };
        let mut counts = BTreeMap::<String, usize>::new();
        for e in examples {
            for word in tokens(&e.text)? {
                *counts.entry(word).or_default() += 1;
            }
        }
        let mut vocabulary = counts.into_iter().collect::<Vec<_>>();
        vocabulary.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        if let Some(m) = initial {
            vocabulary.retain(|(word, _)| m.lexicon.index(word).is_none());
            vocabulary.truncate(budget.vocabulary - m.lexicon.len());
            vocabulary.extend((0..m.lexicon.len()).map(|i| (m.lexicon.word(i).to_owned(), 0)));
        } else {
            vocabulary.truncate(budget.vocabulary);
        }
        vocabulary.sort_by(|a, b| a.0.cmp(&b.0));
        let raw = vocabulary
            .iter()
            .flat_map(|(w, _)| w.bytes().chain(std::iter::once(0)))
            .collect::<Vec<_>>();
        let lexicon = Lexicon::from_bytes(&raw, vocabulary.len())?;
        let mut model = Self {
            metadata: metadata(Wire {
                revision: 1,
                seed: budget.seed,
                buckets: budget.characters.max(256),
                intents,
                tags,
                minimum_margin: budget.minimum_margin,
                packed: String::new(),
            }),
            lexicon,
            dimensions: budget.dimensions,
            characters: budget.characters,
            collision_control: budget.collision_control,
            relative_regions: budget.relative_regions,
            byte_convolution: budget.byte_convolution,
            ordered_context: budget.ordered_context,
            typed_roles: budget.typed_roles,
            split_heads: budget.split_heads,
            word_grams: budget.split_heads && budget.characters != 0,
            boundary_features: budget.boundary_features,
            input: Quantized {
                values: vec![],
                scales: vec![],
            },
            output: Quantized {
                values: vec![],
                scales: vec![],
            },
            bias: vec![],
        };
        let dim = budget.dimensions;
        let mut learned_role_cells = model.learn_role_schema(examples);
        if let Some(m) = initial.filter(|m| m.typed_roles) {
            for (old_intent, name) in m.metadata.intents.iter().enumerate() {
                let intent = model.metadata.intents.binary_search(name).unwrap();
                for (old_tag, name) in m.metadata.tags.iter().enumerate() {
                    if m.role_allowed(old_intent, old_tag) {
                        let tag = model.metadata.tags.iter().position(|t| t == name).unwrap();
                        learned_role_cells[intent * model.role_stride() + tag / 7] |=
                            1 << (tag % 7);
                    }
                }
            }
        }
        let lexical_edges = lexical
            .map(|g| g.prepare(&model))
            .transpose()?
            .unwrap_or_default();
        let lexical = lexical.filter(|g| g.weight > 0.0);
        let mut lexical_order = (0..lexical_edges.len()).collect::<Vec<_>>();
        let mut lexical_cursor = 0;
        let mut lexical_state = budget.seed.wrapping_add(0xd1b54a32d192ed03).max(1);
        let width = dim * POSITIONS;
        let mut state = budget.seed;
        let mut learner = Learner {
            input: (0..model.input_rows() * dim)
                .map(|_| {
                    ((random(&mut state) >> 40) as f32 / (1u32 << 24) as f32 - 0.5)
                        / (dim as f32).sqrt()
                })
                .collect(),
            output: (0..budget.hidden * width)
                .map(|_| {
                    ((random(&mut state) >> 40) as f32 / (1u32 << 24) as f32 - 0.5)
                        / (width as f32).sqrt()
                })
                .chain(std::iter::repeat_n(
                    0.0,
                    (classes + auxiliary_classes)
                        * if budget.hidden == 0 {
                            width
                        } else {
                            budget.hidden
                        },
                ))
                .collect(),
            bias: vec![0.0; classes + budget.hidden + auxiliary_classes],
            dimensions: dim,
            hidden: budget.hidden,
            optimizer: None,
            task_boundary: budget.split_heads.then_some(n),
        };
        let mixing_rows = budget.hidden * (1 + usize::from(budget.split_heads));
        if budget.split_heads {
            // Start both mixers identically without consuming the primary RNG again.
            let block = learner.output[..budget.hidden * width].to_vec();
            learner
                .output
                .splice(budget.hidden * width..budget.hidden * width, block);
            learner.bias.splice(
                budget.hidden..budget.hidden,
                std::iter::repeat_n(0.0, budget.hidden),
            );
        }
        if let Some(m) = initial {
            model.inherit_parameters(m, &mut learner);
        }
        if budget.minibatch != 0 {
            learner.optimizer = Some(optimizer::Adam::new([
                learner.input.len(),
                learner.output.len(),
                learner.bias.len(),
            ])?);
        }
        let mut convolution = if budget.byte_convolution {
            Some(byte_convolution::Train::new(dim, budget.seed)?)
        } else {
            None
        };
        let mut ordered = if budget.ordered_context {
            Some(ordered_context::Train::new(dim, budget.seed)?)
        } else {
            None
        };
        let byte_base = model.input_rows();
        let mut prepared = Vec::new();
        let mut nodes = 0usize;
        let mut frequencies = vec![0usize; n];
        for e in examples {
            let words = model.prepare(&e.text)?;
            nodes += words.iter().map(Vec::len).sum::<usize>();
            if nodes > 8_000_000 {
                return Err(Error::Budget);
            }
            let intent = model.metadata.intents.binary_search(&e.intent).unwrap();
            frequencies[intent] += 1;
            let tags = e
                .tags
                .iter()
                .map(|t| model.metadata.tags.iter().position(|x| x == t).unwrap())
                .collect::<Vec<_>>();
            let raw = if budget.byte_convolution {
                tokens(&e.text)?
            } else {
                Vec::new()
            };
            prepared.push((words, intent, tags, raw));
        }
        let mut prepared_auxiliary = Vec::new();
        if let Some(auxiliary) = auxiliary {
            for example in &auxiliary.examples {
                let words = model.prepare_using(&example.text, false)?;
                nodes += words.iter().map(Vec::len).sum::<usize>();
                if nodes > 8_000_000 {
                    return Err(Error::Budget);
                }
                let labels = example
                    .labels
                    .iter()
                    .map(|p| auxiliary_labels.binary_search(p).unwrap())
                    .collect::<Vec<_>>();
                let raw = if budget.byte_convolution {
                    tokens_mode(&example.text, false)?
                } else {
                    Vec::new()
                };
                prepared_auxiliary.push((words, labels, raw));
            }
        }
        let mut auxiliary_state = budget.seed.wrapping_add(0x9e3779b97f4a7c15).max(1);
        let mut auxiliary_order = (0..prepared_auxiliary.len()).collect::<Vec<_>>();
        let mut auxiliary_cursor = 0;
        let mut order = (0..examples.len()).collect::<Vec<_>>();
        for epoch in 0..budget.epochs {
            let steps_before = learner.optimizer.as_ref().map_or(0, optimizer::Adam::steps);
            for i in (1..order.len()).rev() {
                order.swap(i, random(&mut state) as usize % (i + 1));
            }
            let mut event = Event {
                epoch: epoch + 1,
                sentences: examples.len(),
                intent_errors: 0,
                tag_errors: 0,
            };
            let mut auxiliary_sentences = 0;
            let mut auxiliary_words = 0;
            let mut auxiliary_errors = 0;
            let mut lexical_edge_updates = 0;
            let mut primary_tag_normalization_terms = 0;
            for (step, index) in order.iter().enumerate() {
                let (words, intent, tags, raw) = &prepared[*index];
                let (extra_cache, mut byte_deltas) = prepare_extra_training(
                    &mut learner,
                    convolution.as_ref(),
                    ordered.as_ref(),
                    raw,
                    words,
                    byte_base,
                )?;
                let progress = (epoch * examples.len() + step) as f32
                    / (budget.epochs * examples.len()) as f32;
                let balance = if budget.balanced {
                    (examples.len() as f32 / (n * frequencies[*intent]) as f32)
                        .sqrt()
                        .clamp(0.25, 4.0)
                } else {
                    1.0
                };
                let rate = if budget.minibatch == 0 {
                    budget.learning_rate * (1.0 - progress).max(0.1) * balance
                } else {
                    balance
                };
                let mask = dropout_mask(width, budget.input_dropout, &mut state);
                event.intent_errors += usize::from(learner.update(
                    &model.sentence(words),
                    *intent,
                    0..n,
                    rate,
                    &mask,
                    (budget.byte_convolution || budget.ordered_context).then_some((
                        if budget.ordered_context { 0 } else { byte_base },
                        byte_deltas.as_mut_slice(),
                    )),
                ));
                let mut previous: Option<usize> = None;
                let token_rate = if budget.mean_token_loss {
                    rate / tags.len() as f32
                } else {
                    rate
                };
                for (i, tag) in tags.iter().enumerate() {
                    let mask = dropout_mask(width, budget.input_dropout, &mut state);
                    let previous_name =
                        previous.map_or("<start>", |p| model.metadata.tags[p].as_str());
                    let head = (n..classes).filter(|class| {
                        let tag = *class - n;
                        !budget.constrained_loss
                            || (learned_role_cells[intent * model.role_stride() + tag / 7] as u8
                                & (1 << (tag % 7))
                                != 0
                                && legal(&model.metadata.tags[tag], previous_name))
                    });
                    primary_tag_normalization_terms += head.clone().count();
                    event.tag_errors += usize::from(learner.update(
                        &model.context(words, i, previous, *intent),
                        n + *tag,
                        head,
                        token_rate,
                        &mask,
                        (budget.byte_convolution || budget.ordered_context).then_some((
                            if budget.ordered_context { 0 } else { byte_base },
                            byte_deltas.as_mut_slice(),
                        )),
                    ));
                    previous = Some(*tag);
                }
                if let Some(optimizer) = learner.optimizer.as_mut()
                    && ((step + 1).is_multiple_of(budget.minibatch) || step + 1 == order.len())
                {
                    optimizer.step(
                        [&mut learner.input, &mut learner.output, &mut learner.bias],
                        step % budget.minibatch + 1,
                        budget.learning_rate * (1.0 - progress).max(0.1),
                    )?;
                }
                if let Some(graph) = lexical
                    && (epoch * examples.len() + step + 1).is_multiple_of(graph.interval)
                {
                    if lexical_cursor == 0 {
                        for i in (1..lexical_order.len()).rev() {
                            lexical_order.swap(i, random(&mut lexical_state) as usize % (i + 1));
                        }
                    }
                    lexical_graph::apply(
                        &mut learner.input,
                        dim,
                        lexical_edges[lexical_order[lexical_cursor]],
                        budget.learning_rate * (1.0 - progress).max(0.1) * graph.weight,
                    )?;
                    lexical_cursor = (lexical_cursor + 1) % lexical_order.len();
                    lexical_edge_updates += 1;
                }
                // CNN parameters are held fixed inside each sentence; average its head gradients.
                if let Some(convolution) = convolution.as_mut()
                    && !budget.freeze_byte_convolution
                {
                    let ExtraCache::Bytes(byte_caches) = &extra_cache else {
                        return Err(Error::Invalid("byte training cache"));
                    };
                    let gradient = convolution.gradient(byte_caches, &byte_deltas)?;
                    convolution.apply(
                        gradient,
                        if budget.mean_token_loss {
                            0.5
                        } else {
                            1.0 / (tags.len() + 1) as f32
                        },
                    )?;
                }
                if let Some(ordered) = ordered.as_mut() {
                    apply_ordered_training(
                        ordered,
                        &extra_cache,
                        &byte_deltas,
                        &mut learner,
                        if budget.mean_token_loss {
                            0.5
                        } else {
                            1.0 / (tags.len() + 1) as f32
                        },
                        budget.freeze_ordered_context,
                    )?;
                }
                if let Some(auxiliary) = auxiliary
                    && (epoch * examples.len() + step + 1).is_multiple_of(auxiliary.interval)
                {
                    if auxiliary_cursor == 0 {
                        for i in (1..auxiliary_order.len()).rev() {
                            auxiliary_order
                                .swap(i, random(&mut auxiliary_state) as usize % (i + 1));
                        }
                    }
                    let (words, labels, raw) =
                        &prepared_auxiliary[auxiliary_order[auxiliary_cursor]];
                    let (extra_cache, mut byte_deltas) = prepare_extra_training(
                        &mut learner,
                        convolution.as_ref(),
                        ordered.as_ref(),
                        raw,
                        words,
                        byte_base,
                    )?;
                    auxiliary_cursor = (auxiliary_cursor + 1) % auxiliary_order.len();
                    auxiliary_sentences += 1;
                    auxiliary_words += words.len();
                    // No main-task intent or BIO annotation is fabricated for source grammar.
                    for (i, label) in labels.iter().enumerate() {
                        let mut groups = model.context(words, i, None, 0);
                        groups[5].clear();
                        groups[6].clear();
                        let mask = dropout_mask(width, budget.input_dropout, &mut auxiliary_state);
                        auxiliary_errors += usize::from(learner.update(
                            &groups,
                            classes + *label,
                            classes..classes + auxiliary_classes,
                            budget.learning_rate * (1.0 - progress).max(0.1) * auxiliary.weight
                                / if budget.mean_token_loss {
                                    labels.len() as f32
                                } else {
                                    1.0
                                },
                            &mask,
                            (budget.byte_convolution || budget.ordered_context).then_some((
                                if budget.ordered_context { 0 } else { byte_base },
                                byte_deltas.as_mut_slice(),
                            )),
                        ));
                    }
                    if let Some(convolution) = convolution.as_mut()
                        && !budget.freeze_byte_convolution
                    {
                        let ExtraCache::Bytes(byte_caches) = &extra_cache else {
                            return Err(Error::Invalid("byte training cache"));
                        };
                        let gradient = convolution.gradient(byte_caches, &byte_deltas)?;
                        convolution.apply(
                            gradient,
                            if budget.mean_token_loss {
                                1.0
                            } else {
                                1.0 / labels.len() as f32
                            },
                        )?;
                    }
                    if let Some(ordered) = ordered.as_mut() {
                        apply_ordered_training(
                            ordered,
                            &extra_cache,
                            &byte_deltas,
                            &mut learner,
                            if budget.mean_token_loss {
                                1.0
                            } else {
                                1.0 / labels.len() as f32
                            },
                            budget.freeze_ordered_context,
                        )?;
                    }
                }
            }
            emit(&TrainingEvent {
                primary: event,
                auxiliary_sentences,
                auxiliary_words,
                auxiliary_errors,
                optimizer_steps: learner.optimizer.as_ref().map_or(0, optimizer::Adam::steps)
                    - steps_before,
                lexical_edge_updates,
                primary_tag_normalization_terms,
            })?;
        }
        // Auxiliary outputs teach shared vectors/mixing only; no POS head is deployed.
        learner.output.truncate(
            mixing_rows * width
                + classes
                    * if budget.hidden == 0 {
                        width
                    } else {
                        budget.hidden
                    },
        );
        learner.bias.truncate(mixing_rows + classes);
        learner.input.truncate(byte_base * dim);
        model.input = Quantized::encode(&learner.input, dim);
        model.output = if budget.hidden == 0 {
            Quantized::encode(&learner.output, width)
        } else {
            let a = Quantized::encode(&learner.output[..mixing_rows * width], width);
            let b = Quantized::encode(&learner.output[mixing_rows * width..], budget.hidden);
            join_quantized(a, b)
        };
        if let Some(convolution) = convolution {
            model.input = join_quantized(
                model.input,
                Quantized::encode(&convolution.embedding, byte_convolution::EMBED),
            );
            model.output = join_quantized(
                model.output,
                Quantized::encode(&convolution.filters, byte_convolution::WIDTH),
            );
            learner.bias.extend(convolution.bias);
        }
        if let Some(ordered) = ordered {
            model.output = join_quantized(model.output, Quantized::encode(&ordered.weights, dim));
            learner.bias.extend(ordered.bias);
        }
        if model.typed_roles {
            // Categorical adjacency, seven bits/byte with a canonical bounded scale marker.
            // These bytes are not neural coefficients and are never used by a dot product.
            model.output = join_quantized(
                model.output,
                Quantized {
                    values: learned_role_cells,
                    scales: vec![ROLES_SCALE],
                },
            );
        }
        model.bias = learner.bias.into_boxed_slice().into_vec();
        model.to_bytes()?;
        Ok(model)
    }
    fn byte_cache(
        &self,
        words: &[String],
        work: &mut usize,
    ) -> Result<Vec<byte_convolution::Cache>, Error> {
        if !self.byte_convolution {
            return Ok(Vec::new());
        }
        let base = self.input_rows();
        let classes = self.metadata.intents.len() + self.metadata.tags.len();
        let hidden = self.hidden_units();
        let width = self.dimensions * POSITIONS;
        let offset = hidden * width + classes * if hidden == 0 { width } else { hidden };
        words
            .iter()
            .map(|word| {
                let cache = byte_convolution::encode(
                    word,
                    self.dimensions,
                    |r, d| {
                        f32::from(
                            self.input.values
                                [base * self.dimensions + r * byte_convolution::EMBED + d],
                        ) * self.input.scales[base + r]
                    },
                    |r, d| {
                        f32::from(self.output.values[offset + r * byte_convolution::WIDTH + d])
                            * self.output.scales[hidden + classes + r]
                    },
                    &self.bias[hidden + classes..],
                )?;
                *work = work.checked_add(cache.work).ok_or(Error::Budget)?;
                Ok(cache)
            })
            .collect()
    }
    fn scores(&self, groups: &Groups, cache: &ExtraCache, work: &mut usize) -> Vec<i32> {
        let dim = self.dimensions;
        let width = dim * POSITIONS;
        let h = hidden(groups, dim, |id, d| {
            if (self.byte_convolution || self.ordered_context) && id >= self.input_rows() {
                cache.output(id - self.input_rows(), d, dim)
            } else {
                f32::from(self.input.values[id * dim + d]) * self.input.scales[id]
            }
        });
        let classes = self.metadata.intents.len() + self.metadata.tags.len();
        let hidden = self.hidden_units();
        let mixing_rows = hidden * (1 + usize::from(self.split_heads));
        let tag_context = self.split_heads
            && groups[6]
                .iter()
                .any(|id| *id >= self.intent_base() && *id < self.input_rows());
        let branch = if tag_context { hidden } else { 0 };
        let head = if self.split_heads {
            if tag_context {
                self.metadata.intents.len()..classes
            } else {
                0..self.metadata.intents.len()
            }
        } else {
            0..classes
        };
        let latent = if hidden == 0 {
            h
        } else {
            logits(&h, &self.bias[branch..branch + hidden], |c, d| {
                f32::from(self.output.values[(branch + c) * width + d])
                    * self.output.scales[branch + c]
            })
            .into_iter()
            .map(f32::tanh)
            .collect::<Vec<_>>()
        };
        let output_width = latent.len();
        let offset = mixing_rows * width;
        *work += groups.iter().map(Vec::len).sum::<usize>() * dim
            + hidden * width
            + head.len() * output_width;
        let values = logits(
            &latent,
            &self.bias[mixing_rows + head.start..mixing_rows + head.end],
            |c, d| {
                f32::from(self.output.values[offset + (head.start + c) * output_width + d])
                    * self.output.scales[mixing_rows + head.start + c]
            },
        )
        .into_iter()
        .map(|v| (v * 1000.0).round() as i32)
        .collect::<Vec<_>>();
        if !self.split_heads {
            return values;
        }
        let mut result = vec![0; classes];
        result[head].copy_from_slice(&values);
        result
    }
    pub fn predict(&self, text: &str) -> Result<Prediction, Error> {
        self.measure(text).result
    }
    pub fn measure(&self, text: &str) -> search::Measurement<Prediction> {
        let mut work = 0;
        let result = self.predict_using(text, &mut work);
        search::Measurement { result, work }
    }
    pub fn measure_search(
        &self,
        text: &str,
        budget: search::Budget,
        mean_tags: bool,
    ) -> search::Measurement {
        self.measure_search_using(text, budget, mean_tags, None)
    }
    /// A host-supplied pure shape constraint, not a label or a learned semantic certificate.
    /// Logical work counts constraint consultations, not their physical CPU instructions.
    pub fn measure_search_constrained(
        &self,
        text: &str,
        budget: search::Budget,
        mean_tags: bool,
        allowed: &PathConstraint<'_>,
    ) -> search::Measurement {
        self.measure_search_using(text, budget, mean_tags, Some(allowed))
    }
    fn measure_search_using(
        &self,
        text: &str,
        budget: search::Budget,
        mean_tags: bool,
        allowed: Option<&PathConstraint<'_>>,
    ) -> search::Measurement {
        let mut work = 0;
        let result = (|| {
            budget.validate()?;
            if budget.beam == 1 && budget.intents == 1 && allowed.is_none() {
                let measured = self.measure(text);
                work = measured.work;
                if work > budget.max_work {
                    return Err(Error::Budget);
                }
                return measured.result.map(|p| search::Prediction {
                    output: p.output,
                    work,
                    expansions: 0,
                    surviving_paths: 1,
                    pruned: true,
                    retained_path_margin: None,
                });
            }
            let raw = tokens(text)?;
            let words = self.prepare(text)?;
            let cache = self.extra_cache(&raw, &words, &mut work)?;
            let n = self.metadata.intents.len();
            let classes = n + self.metadata.tags.len();
            let initial = log_scores(
                self.scores(&self.sentence(&words), &cache, &mut work),
                0..n,
                1,
                &mut work,
            )?;
            let mut cached_position = None;
            let mut cached_scores = BTreeMap::<(usize, Option<usize>), Vec<i64>>::new();
            let role_filter = |path: &search::Path, index, tag| {
                if !self.role_allowed(path.intent, tag) {
                    return false;
                }
                allowed.is_none_or(|filter| {
                    filter(
                        &self.metadata.intents[path.intent],
                        &path
                            .tags
                            .iter()
                            .map(|i| self.metadata.tags[*i].as_str())
                            .collect::<Vec<_>>(),
                        index,
                        &self.metadata.tags[tag],
                    )
                })
            };
            let outcome = search::decode_conditioned_paths(
                (&self.metadata.intents, &self.metadata.tags),
                words.len(),
                initial,
                budget,
                &mut |index, path, work| {
                    if cached_position != Some(index) {
                        cached_scores.clear();
                        cached_position = Some(index);
                    }
                    *work = work.checked_add(1).ok_or(Error::Budget)?;
                    let key = (path.intent, path.tags.last().copied());
                    if let Some(scores) = cached_scores.get(&key) {
                        return Ok(scores.clone());
                    }
                    let scores = self.scores(
                        &self.context(&words, index, path.tags.last().copied(), path.intent),
                        &cache,
                        work,
                    );
                    let scores = log_scores(
                        scores,
                        n..classes,
                        if mean_tags { words.len() } else { 1 },
                        work,
                    )?;
                    cached_scores.insert(key, scores.clone());
                    Ok(scores)
                },
                &mut work,
                (self.typed_roles || allowed.is_some())
                    .then_some(&role_filter as &dyn Fn(&search::Path, usize, usize) -> bool),
            )?;
            let best = &outcome.paths[0];
            let margin = outcome
                .paths
                .get(1)
                .map(|p| best.score.checked_sub(p.score).ok_or(Error::Budget))
                .transpose()?;
            if margin == Some(0) {
                return Err(Error::Ambiguous);
            }
            let output = search::output(
                &raw,
                &self.metadata.intents[best.intent],
                &best
                    .tags
                    .iter()
                    .map(|i| self.metadata.tags[*i].as_str())
                    .collect::<Vec<_>>(),
            )?;
            Ok(search::Prediction {
                output,
                work,
                expansions: outcome.expansions,
                surviving_paths: outcome.paths.len(),
                pruned: outcome.pruned,
                retained_path_margin: margin,
            })
        })();
        search::Measurement { result, work }
    }
    fn predict_using(&self, text: &str, work: &mut usize) -> Result<Prediction, Error> {
        let raw = tokens(text)?;
        let words = self.prepare(text)?;
        let cache = self.extra_cache(&raw, &words, work)?;
        let n = self.metadata.intents.len();
        let (intent, intent_margin) = best(
            &self.scores(&self.sentence(&words), &cache, work),
            0,
            n,
            None,
        );
        if intent_margin < i32::from(self.metadata.minimum_margin) {
            return Err(Error::Ambiguous);
        }
        let mut selected = Vec::new();
        let mut previous = None;
        let mut tag_margin = i32::MAX;
        for i in 0..words.len() {
            if self.typed_roles {
                *work += self.metadata.tags.len();
            }
            let (tag, margin) = best_allowed(
                &self.scores(&self.context(&words, i, previous, intent), &cache, work),
                n,
                n + self.metadata.tags.len(),
                Some((
                    &self.metadata.tags,
                    previous.map_or("<start>", |p| &self.metadata.tags[p]),
                )),
                |tag| self.role_allowed(intent, tag - n),
            );
            if margin < i32::from(self.metadata.minimum_margin) {
                return Err(Error::Ambiguous);
            }
            tag_margin = tag_margin.min(margin);
            selected.push(tag - n);
            previous = Some(tag - n);
        }
        let output = search::output(
            &raw,
            &self.metadata.intents[intent],
            &selected
                .iter()
                .map(|i| self.metadata.tags[*i].as_str())
                .collect::<Vec<_>>(),
        )?;
        Ok(Prediction {
            output,
            intent_margin,
            tag_margin,
            work: *work,
        })
    }
    fn inherit_parameters(&self, initial: &Self, learner: &mut Learner) {
        let dim = self.dimensions;
        let mut rows = (0..3).map(|i| (i, i)).collect::<Vec<_>>();
        rows.extend((0..initial.lexicon.len()).map(|i| {
            (
                3 + i,
                3 + self.lexicon.index(initial.lexicon.word(i)).unwrap(),
            )
        }));
        rows.extend(
            (0..self.characters).map(|i| (initial.character_base() + i, self.character_base() + i)),
        );
        let classes = initial
            .metadata
            .intents
            .iter()
            .map(|name| self.metadata.intents.binary_search(name).unwrap())
            .chain(initial.metadata.tags.iter().map(|name| {
                self.metadata.intents.len()
                    + self.metadata.tags.iter().position(|t| t == name).unwrap()
            }))
            .collect::<Vec<_>>();
        for (old, new) in classes.iter().copied().enumerate() {
            let old_row = if old < initial.metadata.intents.len() {
                initial.intent_base() + old
            } else {
                initial.tag_base() + old - initial.metadata.intents.len()
            };
            let new_row = if new < self.metadata.intents.len() {
                self.intent_base() + new
            } else {
                self.tag_base() + new - self.metadata.intents.len()
            };
            rows.push((old_row, new_row));
        }
        for (old, new) in rows {
            for d in 0..dim {
                learner.input[new * dim + d] =
                    f32::from(initial.input.values[old * dim + d]) * initial.input.scales[old];
            }
        }
        let width = dim * POSITIONS;
        let mixers = learner.hidden * (1 + usize::from(self.split_heads));
        for row in 0..mixers {
            for d in 0..width {
                learner.output[row * width + d] =
                    f32::from(initial.output.values[row * width + d]) * initial.output.scales[row];
            }
            learner.bias[row] = initial.bias[row];
        }
        let output_width = if learner.hidden == 0 {
            width
        } else {
            learner.hidden
        };
        let offset = mixers * width;
        for (old, new) in classes.into_iter().enumerate() {
            for d in 0..output_width {
                learner.output[offset + new * output_width + d] =
                    f32::from(initial.output.values[offset + old * output_width + d])
                        * initial.output.scales[mixers + old];
            }
            learner.bias[mixers + new] = initial.bias[mixers + old];
        }
    }
    fn extra_cache(
        &self,
        raw: &[String],
        words: &[Vec<usize>],
        work: &mut usize,
    ) -> Result<ExtraCache, Error> {
        let cache = if self.ordered_context {
            let dim = self.dimensions;
            let inputs = words
                .iter()
                .flat_map(|f| {
                    (0..dim).map(move |d| {
                        f32::from(self.input.values[f[0] * dim + d]) * self.input.scales[f[0]]
                    })
                })
                .collect::<Vec<_>>();
            let units = self.hidden_units();
            let classes = self.metadata.intents.len() + self.metadata.tags.len();
            let mixing_rows = units * (1 + usize::from(self.split_heads));
            let offset = mixing_rows * dim * POSITIONS
                + classes * if units == 0 { dim * POSITIONS } else { units };
            let encoded = ordered_context::encode(
                &inputs,
                dim,
                |direction, d, k| {
                    let row = direction * dim + d;
                    f32::from(self.output.values[offset + row * dim + k])
                        * self.output.scales[mixing_rows + classes + row]
                },
                &self.bias[mixing_rows + classes..],
            )?;
            *work = work.checked_add(encoded.work).ok_or(Error::Budget)?;
            ExtraCache::Ordered(encoded, Vec::new())
        } else if self.byte_convolution {
            ExtraCache::Bytes(self.byte_cache(raw, work)?)
        } else {
            ExtraCache::None
        };
        Ok(cache)
    }
    pub fn words(&self) -> usize {
        self.lexicon.len()
    }
    pub fn codec(&self) -> &'static str {
        if self.boundary_features {
            "AXSQ0016"
        } else if self.split_heads && self.ordered_context {
            "AXSQ0015"
        } else if self.word_grams {
            "AXSQ0014"
        } else if self.split_heads {
            "AXSQ0013"
        } else if self.typed_roles {
            "AXSQ0012"
        } else if self.ordered_context {
            "AXSQ0011"
        } else if self.byte_convolution {
            "AXSQ0010"
        } else if self.relative_regions {
            "AXSQ0009"
        } else if self.bias.len() == self.metadata.intents.len() + self.metadata.tags.len() {
            "AXSQ0006"
        } else {
            "AXSQ0007"
        }
    }
    pub fn weights(&self) -> usize {
        self.input.values.len() + self.output.values.len() - self.role_cells()
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            - std::mem::size_of::<super::Model>()
            - std::mem::size_of::<Lexicon>()
            + self.metadata.retained_bytes()
            + self.lexicon.retained_bytes()
            + self.input.retained_bytes()
            + self.output.retained_bytes()
            + self.bias.capacity() * 4
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let header = serde_json::to_vec(&Header {
            metadata: Wire::from(self.metadata.clone()),
            dimensions: self.dimensions,
            characters: self.characters,
            words: self.lexicon.len(),
            word_bytes: self.lexicon.bytes.len(),
            collision_control: self.collision_control,
            hidden: self.hidden_units(),
            relative_regions: self.relative_regions,
            byte_convolution: self.byte_convolution,
            ordered_context: self.ordered_context,
            typed_roles: self.typed_roles,
            split_heads: self.split_heads,
            word_grams: self.word_grams,
            boundary_features: self.boundary_features,
        })
        .map_err(|e| Error::Storage(e.to_string()))?;
        let mut bytes = self.magic().to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&self.lexicon.bytes);
        self.input.append(&mut bytes);
        self.output.append(&mut bytes);
        for v in &self.bias {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if !(20..=MAX_BYTES).contains(&bytes.len()) {
            return Err(Error::Invalid("encoder length"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if checksum(body).to_le_bytes() != digest
            || ![
                MAGIC.as_slice(),
                DEEP_MAGIC.as_slice(),
                REGIONS_MAGIC.as_slice(),
                BYTE_MAGIC.as_slice(),
                ORDERED_MAGIC.as_slice(),
                ROLES_MAGIC.as_slice(),
                SPLIT_MAGIC.as_slice(),
                GRAMS_MAGIC.as_slice(),
                SPLIT_ORDERED_MAGIC.as_slice(),
                BOUNDARY_MAGIC.as_slice(),
            ]
            .contains(&&body[..8])
        {
            return Err(Error::Invalid("encoder checksum/magic"));
        }
        let mut cursor = Cursor::new(body);
        cursor.set_position(8);
        let size = u32::from_le_bytes(take(&mut cursor)?) as usize;
        let end = 12usize
            .checked_add(size)
            .filter(|end| *end <= body.len())
            .ok_or(Error::Invalid("encoder header length"))?;
        let header: Header =
            serde_json::from_slice(&body[12..end]).map_err(|e| Error::Storage(e.to_string()))?;
        validate_header(&header.metadata)?;
        Budget {
            seed: header.metadata.seed,
            vocabulary: header.words.max(16),
            characters: header.characters,
            dimensions: header.dimensions,
            epochs: 1,
            learning_rate: 0.1,
            balanced: false,
            minimum_margin: header.metadata.minimum_margin,
            collision_control: header.collision_control,
            hidden: header.hidden,
            input_dropout: 0.0,
            relative_regions: header.relative_regions,
            byte_convolution: header.byte_convolution,
            freeze_byte_convolution: false,
            ordered_context: header.ordered_context,
            freeze_ordered_context: false,
            mean_token_loss: false,
            minibatch: 0,
            typed_roles: header.typed_roles,
            constrained_loss: false,
            split_heads: header.split_heads,
            boundary_features: header.boundary_features,
        }
        .validate()?;
        if body.get(..8)
            != Some(
                format_magic(
                    header.word_grams,
                    header.split_heads,
                    header.typed_roles,
                    header.ordered_context,
                    header.byte_convolution,
                    header.relative_regions,
                    header.hidden,
                    header.boundary_features,
                )
                .as_slice(),
            )
            || !header.metadata.packed.is_empty()
            || header.metadata.buckets != header.characters.max(256)
            || header.word_grams != (header.split_heads && header.characters != 0)
        {
            return Err(Error::Invalid("encoder metadata"));
        }
        let lexical_end = end
            .checked_add(header.word_bytes)
            .filter(|end| *end <= body.len())
            .ok_or(Error::Invalid("encoder words length"))?;
        let lexicon = Lexicon::from_bytes(&body[end..lexical_end], header.words)?;
        let metadata = metadata(header.metadata);
        let rows = 3
            + lexicon.len()
            + header.characters
            + metadata.tags.len()
            + metadata.intents.len()
            + if header.boundary_features {
                BOUNDARY_ROWS
            } else {
                0
            };
        let classes = metadata.intents.len() + metadata.tags.len();
        let width = header.dimensions * POSITIONS;
        let mixing_rows = header.hidden * (1 + usize::from(header.split_heads));
        let output_weights = if header.hidden == 0 {
            classes * width
        } else {
            mixing_rows * width + classes * header.hidden
        };
        let byte_rows = if header.byte_convolution {
            byte_convolution::ROWS
        } else {
            0
        };
        let byte_units = if header.byte_convolution {
            header.dimensions
        } else {
            0
        };
        let ordered_units = if header.ordered_context {
            2 * header.dimensions
        } else {
            0
        };
        let role_rows = usize::from(header.typed_roles);
        let role_cells = if header.typed_roles {
            metadata.intents.len() * metadata.tags.len().div_ceil(7)
        } else {
            0
        };
        if body.len()
            != lexical_end
                + rows * (header.dimensions + 4)
                + byte_rows * (byte_convolution::EMBED + 4)
                + output_weights
                + byte_units * byte_convolution::WIDTH
                + ordered_units * header.dimensions
                + role_cells
                + role_rows * 4
                + (classes + mixing_rows + byte_units + ordered_units) * 8
        {
            return Err(Error::Invalid("encoder payload length"));
        }
        cursor.set_position(lexical_end as u64);
        let input = Quantized::read_packed(
            &mut cursor,
            rows + byte_rows,
            rows * header.dimensions + byte_rows * byte_convolution::EMBED,
        )?;
        let output = Quantized::read_packed(
            &mut cursor,
            classes + mixing_rows + byte_units + ordered_units + role_rows,
            output_weights
                + byte_units * byte_convolution::WIDTH
                + ordered_units * header.dimensions
                + role_cells,
        )?;
        let mut bias = Vec::with_capacity(classes + mixing_rows + byte_units + ordered_units);
        for _ in 0..classes + mixing_rows + byte_units + ordered_units {
            let v = f32::from_le_bytes(take(&mut cursor)?);
            if !v.is_finite() || v.abs() > 16.0 {
                return Err(Error::Invalid("encoder bias"));
            }
            bias.push(v);
        }
        let model = Self {
            metadata,
            lexicon,
            dimensions: header.dimensions,
            characters: header.characters,
            collision_control: header.collision_control,
            relative_regions: header.relative_regions,
            byte_convolution: header.byte_convolution,
            ordered_context: header.ordered_context,
            typed_roles: header.typed_roles,
            split_heads: header.split_heads,
            word_grams: header.word_grams,
            boundary_features: header.boundary_features,
            input,
            output,
            bias,
        };
        if model.typed_roles {
            if model.output.scales[classes + header.hidden..]
                .iter()
                .any(|s| *s != ROLES_SCALE)
            {
                return Err(Error::Invalid("role schema canonical scale"));
            }
            let remainder = model.metadata.tags.len() % 7;
            for intent in 0..model.metadata.intents.len() {
                if !model.role_allowed(intent, 0) {
                    return Err(Error::Invalid("role schema outside tag"));
                }
                if remainder != 0
                    && (model.output.values[model.output.values.len() - role_cells
                        + (intent + 1) * model.role_stride()
                        - 1] as u8)
                        >> remainder
                        != 0
                {
                    return Err(Error::Invalid("role schema unused bits"));
                }
                for (tag, name) in model.metadata.tags.iter().enumerate().skip(1) {
                    let prefix = if name.starts_with("B:") { "I:" } else { "B:" };
                    let other = model
                        .metadata
                        .tags
                        .iter()
                        .position(|t| t == &format!("{prefix}{}", role(name).unwrap()))
                        .ok_or(Error::Invalid("role schema pair"))?;
                    if model.role_allowed(intent, tag) != model.role_allowed(intent, other) {
                        return Err(Error::Invalid("role schema pair support"));
                    }
                }
            }
            for tag in 0..model.metadata.tags.len() {
                if !(0..model.metadata.intents.len()).any(|intent| model.role_allowed(intent, tag))
                {
                    return Err(Error::Invalid("role schema unseen class"));
                }
            }
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_log_scores_are_shift_invariant_bounded_and_length_explicit() {
        let a = log_scores(vec![-2000, 1000, 4000], 0..3, 1, &mut 0).unwrap();
        let b = log_scores(vec![498000, 501000, 504000], 0..3, 1, &mut 0).unwrap();
        assert_eq!(a, b);
        assert!(
            (a.iter()
                .map(|v| (*v as f64 / 1_000_000.0).exp())
                .sum::<f64>()
                - 1.0)
                .abs()
                < 0.000002
        );
        let mean = log_scores(vec![-2000, 1000, 4000], 0..3, 3, &mut 0).unwrap();
        for i in 0..3 {
            assert!((a[i] - 3 * mean[i]).abs() <= 2);
        }
        let extreme = log_scores(vec![i32::MIN, i32::MAX], 0..2, 1, &mut 0).unwrap();
        assert_eq!(extreme[1], 0);
        assert!(extreme[0] < -1_000_000_000);
        assert!(log_scores(vec![0], 0..0, 1, &mut 0).is_err());
        assert!(log_scores(vec![0], 0..2, 1, &mut 0).is_err());
        assert!(log_scores(vec![0], 0..1, 0, &mut 0).is_err());
        let mut work = usize::MAX;
        assert!(log_scores(vec![0], 0..1, 1, &mut work).is_err());
    }
    #[test]
    fn learned_role_domains_preserve_neural_weights_and_numeric_birth() {
        let examples = data()
            .into_iter()
            .map(|mut e| {
                if e.intent == "cancel" {
                    e.tags[3] = "B:object".into();
                }
                e
            })
            .collect::<Vec<_>>();
        let config = Budget {
            characters: 0,
            relative_regions: true,
            epochs: 32,
            hidden: 16,
            ..budget()
        };
        let mut old = Model::learn(&examples, config, &mut |_| Ok(())).unwrap();
        let typed = Model::learn(
            &examples,
            Budget {
                typed_roles: true,
                ..config
            },
            &mut |_| Ok(()),
        )
        .unwrap();
        assert_eq!(old.input.values, typed.input.values);
        assert_eq!(old.input.scales, typed.input.scales);
        assert_eq!(old.bias, typed.bias);
        assert_eq!(
            old.output.values,
            typed.output.values[..old.output.values.len()]
        );
        assert_eq!(
            old.output.scales,
            typed.output.scales[..old.output.scales.len()]
        );
        assert_eq!(old.weights(), typed.weights());
        assert_eq!(
            typed.retained_bytes() - old.retained_bytes(),
            typed.role_cells() + 4
        );
        let raw = typed.to_bytes().unwrap();
        assert_eq!(&raw[..8], ROLES_MAGIC);
        let mut child = Model::from_bytes(&raw).unwrap();
        assert_eq!(raw, child.to_bytes().unwrap());
        for intent in ["send", "cancel"] {
            let role = if intent == "send" { "person" } else { "object" };
            let text = format!("{intent} letter to unknown-value");
            assert_eq!(
                child.predict(&text).unwrap().output,
                search::output(
                    &tokens(&text).unwrap(),
                    intent,
                    &["O", "O", "O", &format!("B:{role}")]
                )
                .unwrap()
            );
        }
        let object = child
            .metadata
            .tags
            .iter()
            .position(|t| t == "B:object")
            .unwrap();
        let bias = child.hidden_units() + child.metadata.intents.len() + object;
        old.bias[bias] = 16.0;
        child.bias[bias] = 16.0;
        assert_ne!(
            old.predict("send letter to unknown-value").unwrap().output,
            child
                .predict("send letter to unknown-value")
                .unwrap()
                .output
        );
        let send = child
            .metadata
            .intents
            .iter()
            .position(|s| s == "send")
            .unwrap();
        assert!(!child.role_allowed(send, object));
        assert!(
            child
                .measure_search(
                    "send letter to unknown-value",
                    search::Budget {
                        beam: 4,
                        intents: 2,
                        max_work: 5_000_000
                    },
                    true
                )
                .result
                .is_ok()
        );
        for mode in 0..4 {
            let mut damaged = Model::from_bytes(&raw).unwrap();
            let base = damaged.output.values.len() - damaged.role_cells();
            match mode {
                0 => damaged.output.values[base] &= !1,
                1 => damaged.output.values[base] ^= 2,
                2 => damaged.output.values[base] |= 64,
                _ => *damaged.output.scales.last_mut().unwrap() = 0.5,
            }
            assert!(
                Model::from_bytes(&damaged.to_bytes().unwrap()).is_err(),
                "repaired categorical payload {mode}"
            );
        }
        for invalid in [
            Budget {
                relative_regions: false,
                typed_roles: true,
                ..config
            },
            Budget {
                ordered_context: true,
                typed_roles: true,
                ..config
            },
            Budget {
                byte_convolution: true,
                typed_roles: true,
                ..config
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }
    #[test]
    fn search_greedy_control_preserves_predictions_errors_and_work_in_all_encoders() {
        for (byte_convolution, ordered_context) in [(false, false), (true, false), (false, true)] {
            let model = Model::learn(
                &data(),
                Budget {
                    characters: 0,
                    hidden: 16,
                    epochs: 32,
                    relative_regions: true,
                    byte_convolution,
                    ordered_context,
                    ..budget()
                },
                &mut |_| Ok(()),
            )
            .unwrap();
            for text in [
                "send letter to person-new",
                "cancel letter to person-other",
                "",
            ] {
                let reference = model.measure(text);
                for mean in [false, true] {
                    let tested = model.measure_search(
                        text,
                        search::Budget {
                            beam: 1,
                            intents: 1,
                            max_work: 5_000_000,
                        },
                        mean,
                    );
                    assert_eq!(tested.work, reference.work);
                    assert_eq!(
                        tested.result.as_ref().ok().map(|p| &p.output),
                        reference.result.as_ref().ok().map(|p| &p.output)
                    );
                    assert_eq!(
                        tested.result.as_ref().err().map(ToString::to_string),
                        reference.result.as_ref().err().map(ToString::to_string)
                    );
                }
            }
            let refused = model.measure_search(
                "send letter to person-new",
                search::Budget {
                    beam: 4,
                    intents: 2,
                    max_work: 1,
                },
                false,
            );
            assert!(matches!(refused.result, Err(Error::Budget)));
            assert!(refused.work > 0);
            for mean in [false, true] {
                assert!(
                    model
                        .measure_search(
                            "send letter to person-new",
                            search::Budget {
                                beam: 4,
                                intents: 2,
                                max_work: 5_000_000
                            },
                            mean
                        )
                        .result
                        .is_ok()
                );
            }
        }
    }
    fn auxiliary() -> Auxiliary {
        Auxiliary {
            weight: 0.1,
            interval: 4,
            examples: (0..4)
                .map(|i| AuxiliaryExample {
                    text: format!("aux-only-{i} arrives"),
                    labels: vec!["NOUN".into(), "VERB".into()],
                })
                .collect(),
        }
    }
    fn data() -> Vec<Example> {
        (0..16)
            .flat_map(|i| {
                ["send", "cancel"].map(move |intent| Example {
                    text: format!("{intent} letter to person-{i}"),
                    intent: intent.into(),
                    tags: vec!["O".into(), "O".into(), "O".into(), "B:person".into()],
                })
            })
            .collect()
    }
    fn budget() -> Budget {
        Budget {
            seed: 47,
            vocabulary: 64,
            characters: 256,
            dimensions: 8,
            epochs: 16,
            learning_rate: 0.1,
            balanced: false,
            minimum_margin: 1,
            collision_control: false,
            hidden: 0,
            input_dropout: 0.0,
            relative_regions: false,
            byte_convolution: false,
            freeze_byte_convolution: false,
            ordered_context: false,
            freeze_ordered_context: false,
            mean_token_loss: false,
            minibatch: 0,
            typed_roles: false,
            constrained_loss: false,
            split_heads: false,
            boundary_features: false,
        }
    }
    #[test]
    fn minibatch_accumulates_exact_loss_gradients_without_changing_parameters() {
        let dim = 8;
        let width = dim * POSITIONS;
        let units = 8;
        let groups: Groups = [
            vec![0, 1],
            vec![1],
            vec![],
            vec![2],
            vec![0],
            vec![],
            vec![2],
        ];
        let input = (0..3 * dim)
            .map(|i| (i as f32 - 10.0) * 0.02)
            .collect::<Vec<_>>();
        let output = (0..units * width + 2 * units)
            .map(|i| (i as f32 % 13.0 - 6.0) * 0.01)
            .collect::<Vec<_>>();
        let bias = vec![0.01; units + 2];
        let loss = |parameters: &[Vec<f32>; 3]| {
            let h = hidden(&groups, dim, |r, d| parameters[0][r * dim + d]);
            let latent = logits(&h, &parameters[2][..units], |r, d| {
                parameters[1][r * width + d]
            })
            .into_iter()
            .map(f32::tanh)
            .collect::<Vec<_>>();
            let s = logits(&latent, &parameters[2][units..], |r, d| {
                parameters[1][units * width + r * units + d]
            });
            (s[0].exp() + s[1].exp()).ln() - s[1]
        };
        let original = [input.clone(), output.clone(), bias.clone()];
        let mut learner = Learner {
            input,
            output,
            bias,
            dimensions: dim,
            hidden: units,
            optimizer: Some(optimizer::Adam::new(original.each_ref().map(Vec::len)).unwrap()),
            task_boundary: None,
        };
        for _ in 0..2 {
            learner.update(&groups, 1, 0..2, 0.5, &[], None);
        }
        assert_eq!(
            [&learner.input, &learner.output, &learner.bias],
            original.each_ref()
        );
        for group in 0..3 {
            for i in 0..original[group].len() {
                let mut hi = original.clone();
                let mut lo = original.clone();
                hi[group][i] += 0.001;
                lo[group][i] -= 0.001;
                let expected = (loss(&hi) - loss(&lo)) / 0.002;
                let actual = learner.optimizer.as_ref().unwrap().gradients[group][i];
                assert!(
                    (actual - expected).abs() < 0.0002,
                    "group{group} coordinate{i}: {actual} != {expected}"
                );
            }
        }
    }
    #[test]
    fn refinement_remaps_lexical_class_and_mixer_parameters_without_reinitializing_them() {
        for (units, split, typed, chars) in [
            (0, false, false, 256),
            (16, false, false, 0),
            (16, true, false, 0),
            (16, false, true, 0),
        ] {
            let config = Budget {
                hidden: units,
                split_heads: split,
                typed_roles: typed,
                characters: chars,
                relative_regions: units != 0,
                ..budget()
            };
            let parent = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
            let original = parent.to_bytes().unwrap();
            let examples = (0..16)
                .map(|i| Example {
                    text: format!("archive file-{i}"),
                    intent: "aarchive".into(),
                    tags: vec!["O".into(), "B:document".into()],
                })
                .collect::<Vec<_>>();
            let (intents, tags) = class_labels(&examples, Some(&parent.metadata)).unwrap();
            let mut next = parent.clone();
            next.metadata.intents = intents;
            next.metadata.tags = tags;
            let mut words = (0..parent.words())
                .map(|i| parent.lexicon.word(i).to_owned())
                .collect::<Vec<_>>();
            words.push("aardvark".into());
            words.sort();
            let bytes = words
                .iter()
                .flat_map(|w| w.bytes().chain(std::iter::once(0)))
                .collect::<Vec<_>>();
            next.lexicon = Lexicon::from_bytes(&bytes, words.len()).unwrap();
            let width = config.dimensions * POSITIONS;
            let mixers = units * (1 + usize::from(split));
            let classes = next.metadata.intents.len() + next.metadata.tags.len();
            let output_width = if units == 0 { width } else { units };
            let mut learner = Learner {
                input: vec![0.125; next.input_rows() * config.dimensions],
                output: vec![0.0; mixers * width + classes * output_width],
                bias: vec![0.0; mixers + classes],
                dimensions: config.dimensions,
                hidden: units,
                task_boundary: split.then_some(next.metadata.intents.len()),
                optimizer: None,
            };
            next.inherit_parameters(&parent, &mut learner);
            let check_input = |old: usize, new: usize| {
                for d in 0..config.dimensions {
                    assert_eq!(
                        learner.input[new * config.dimensions + d],
                        f32::from(parent.input.values[old * config.dimensions + d])
                            * parent.input.scales[old]
                    );
                }
            };
            for i in 0..3 {
                check_input(i, i);
            }
            for i in 0..parent.words() {
                check_input(
                    3 + i,
                    3 + next.lexicon.index(parent.lexicon.word(i)).unwrap(),
                );
            }
            for i in 0..chars {
                check_input(parent.character_base() + i, next.character_base() + i);
            }
            for row in 0..mixers {
                assert_eq!(learner.bias[row], parent.bias[row]);
                for d in 0..width {
                    assert_eq!(
                        learner.output[row * width + d],
                        f32::from(parent.output.values[row * width + d])
                            * parent.output.scales[row]
                    );
                }
            }
            for (old, name) in parent
                .metadata
                .intents
                .iter()
                .chain(&parent.metadata.tags)
                .enumerate()
            {
                let intent = old < parent.metadata.intents.len();
                let new = if intent {
                    next.metadata.intents.binary_search(name).unwrap()
                } else {
                    next.metadata.intents.len()
                        + next.metadata.tags.iter().position(|t| t == name).unwrap()
                };
                check_input(
                    if intent {
                        parent.intent_base() + old
                    } else {
                        parent.tag_base() + old - parent.metadata.intents.len()
                    },
                    if intent {
                        next.intent_base() + new
                    } else {
                        next.tag_base() + new - next.metadata.intents.len()
                    },
                );
                assert_eq!(learner.bias[mixers + new], parent.bias[mixers + old]);
                for d in 0..output_width {
                    assert_eq!(
                        learner.output[mixers * width + new * output_width + d],
                        f32::from(parent.output.values[mixers * width + old * output_width + d])
                            * parent.output.scales[mixers + old]
                    );
                }
            }
            let new = next
                .metadata
                .intents
                .binary_search(&"aarchive".to_owned())
                .unwrap();
            assert!(
                learner.output[mixers * width + new * output_width
                    ..mixers * width + (new + 1) * output_width]
                    .iter()
                    .all(|v| *v == 0.0)
            );
            let new_word = 3 + next.lexicon.index("aardvark").unwrap();
            assert!(
                learner.input[new_word * config.dimensions..(new_word + 1) * config.dimensions]
                    .iter()
                    .all(|v| *v == 0.125)
            );
            let child = Model::refine_with_sources(
                &examples,
                Budget {
                    epochs: 32,
                    ..config
                },
                &parent,
                None,
                None,
                &mut |_| Ok(()),
            )
            .unwrap();
            for i in 0..parent.words() {
                assert!(child.lexicon.index(parent.lexicon.word(i)).is_some());
            }
            if typed {
                for (i, name) in parent.metadata.intents.iter().enumerate() {
                    let j = child.metadata.intents.binary_search(name).unwrap();
                    for (t, tag) in parent.metadata.tags.iter().enumerate() {
                        let u = child.metadata.tags.iter().position(|s| s == tag).unwrap();
                        assert_eq!(parent.role_allowed(i, t), child.role_allowed(j, u));
                    }
                }
            }
            let raw = child.to_bytes().unwrap();
            let born = Model::from_bytes(&raw).unwrap();
            assert_eq!(born.to_bytes().unwrap(), raw);
            assert_eq!(
                born.predict("archive file-new").unwrap().output,
                Term::Tree(vec![
                    Term::Symbol("intent".into()),
                    Term::Symbol("aarchive".into()),
                    Term::Tree(vec![
                        Term::Symbol("slots".into()),
                        Term::Tree(vec![
                            Term::Symbol("document".into()),
                            Term::Symbol("file-new".into())
                        ])
                    ])
                ])
            );
            assert_eq!(parent.to_bytes().unwrap(), original);
        }
    }
    #[test]
    fn refinement_refuses_schema_changes_and_callback_failure_leaves_the_parent_untouched() {
        let config = budget();
        let parent = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
        let raw = parent.to_bytes().unwrap();
        for invalid in [
            Budget {
                seed: config.seed + 1,
                ..config
            },
            Budget {
                dimensions: 16,
                ..config
            },
            Budget {
                hidden: 8,
                ..config
            },
            Budget {
                characters: 0,
                ..config
            },
            Budget {
                relative_regions: true,
                ..config
            },
            Budget {
                vocabulary: 16,
                ..config
            },
            Budget {
                collision_control: true,
                ..config
            },
        ] {
            assert!(
                Model::refine_with_sources(&data(), invalid, &parent, None, None, &mut |_| Ok(()))
                    .is_err()
            );
        }
        assert!(matches!(
            Model::refine_with_sources(&data(), config, &parent, None, None, &mut |_| Err(
                Error::Budget
            )),
            Err(Error::Budget)
        ));
        assert_eq!(parent.to_bytes().unwrap(), raw);
    }
    #[test]
    fn split_mixers_route_both_loss_gradients_without_touching_the_other_task() {
        let dim = 8;
        let width = dim * POSITIONS;
        let units = 8;
        let groups: Groups = [
            vec![0, 1],
            vec![1],
            vec![],
            vec![2],
            vec![0],
            vec![],
            vec![2],
        ];
        let original = [
            (0..3 * dim)
                .map(|i| (i as f32 - 10.0) * 0.02)
                .collect::<Vec<_>>(),
            (0..2 * units * width + 4 * units)
                .map(|i| (i as f32 % 17.0 - 8.0) * 0.03)
                .collect(),
            (0..2 * units + 4)
                .map(|i| (i as f32 % 5.0 - 2.0) * 0.01)
                .collect(),
        ];
        for (start, branch) in [(0, 0), (2, units)] {
            let loss = |parameters: &[Vec<f32>; 3]| {
                let h = hidden(&groups, dim, |r, d| parameters[0][r * dim + d]);
                let latent = logits(&h, &parameters[2][branch..branch + units], |r, d| {
                    parameters[1][(branch + r) * width + d]
                })
                .into_iter()
                .map(f32::tanh)
                .collect::<Vec<_>>();
                let s = logits(
                    &latent,
                    &parameters[2][2 * units + start..2 * units + start + 2],
                    |r, d| parameters[1][2 * units * width + (start + r) * units + d],
                );
                (s[0].exp() + s[1].exp()).ln() - s[1]
            };
            let mut learner = Learner {
                input: original[0].clone(),
                output: original[1].clone(),
                bias: original[2].clone(),
                dimensions: dim,
                hidden: units,
                task_boundary: Some(2),
                optimizer: Some(optimizer::Adam::new(original.each_ref().map(Vec::len)).unwrap()),
            };
            learner.update(&groups, start + 1, start..start + 2, 1.0, &[], None);
            assert_eq!(
                [&learner.input, &learner.output, &learner.bias],
                original.each_ref()
            );
            let gradients = &learner.optimizer.as_ref().unwrap().gradients;
            for group in 0..3 {
                for i in 0..original[group].len() {
                    let mut hi = original.clone();
                    let mut lo = original.clone();
                    hi[group][i] += 0.001;
                    lo[group][i] -= 0.001;
                    let expected = (loss(&hi) - loss(&lo)) / 0.002;
                    assert!(
                        (gradients[group][i] - expected).abs() < 0.0002,
                        "task{start} group{group} coordinate{i}: {} != {expected}",
                        gradients[group][i]
                    );
                }
            }
            let inactive = units - branch;
            assert!(
                gradients[1][inactive * width..(inactive + units) * width]
                    .iter()
                    .all(|v| *v == 0.0)
            );
            assert!(
                gradients[2][inactive..inactive + units]
                    .iter()
                    .all(|v| *v == 0.0)
            );
            assert!(gradients[0].iter().any(|v| v.abs() > 1e-6));
        }
    }
    #[test]
    fn split_mixers_have_strict_shapes_and_teacher_free_numeric_birth() {
        let config = Budget {
            characters: 0,
            hidden: 16,
            relative_regions: true,
            split_heads: true,
            epochs: 32,
            ..budget()
        };
        let model = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
        let raw = model.to_bytes().unwrap();
        assert_eq!(&raw[..8], SPLIT_MAGIC);
        let child = Model::from_bytes(&raw).unwrap();
        assert_eq!(child.codec(), "AXSQ0013");
        assert_eq!(child.hidden_units(), 16);
        assert_eq!(child.to_bytes().unwrap(), raw);
        let ordered_config = Budget {
            ordered_context: true,
            ..config
        };
        let ordered = Model::learn(&data(), ordered_config, &mut |_| Ok(())).unwrap();
        let ordered_raw = ordered.to_bytes().unwrap();
        assert_eq!(&ordered_raw[..8], SPLIT_ORDERED_MAGIC);
        let ordered_child = Model::from_bytes(&ordered_raw).unwrap();
        assert_eq!(ordered_child.codec(), "AXSQ0015");
        assert!(ordered_child.predict("send letter to person-new").is_ok());
        let boundary_config = Budget {
            boundary_features: true,
            ..config
        };
        let boundary = Model::learn(&data(), boundary_config, &mut |_| Ok(())).unwrap();
        let boundary_raw = boundary.to_bytes().unwrap();
        assert_eq!(&boundary_raw[..8], BOUNDARY_MAGIC);
        let boundary_child = Model::from_bytes(&boundary_raw).unwrap();
        assert_eq!(boundary_child.codec(), "AXSQ0016");
        assert!(boundary_child.predict("send letter to person-new").is_ok());
        assert_eq!(
            Model::learn(&data(), config, &mut |_| Ok(()))
                .unwrap()
                .to_bytes()
                .unwrap(),
            raw
        );
        let text = "send letter to person-new";
        assert_eq!(
            child.predict(text).unwrap().output,
            Term::Tree(vec![
                Term::Symbol("intent".into()),
                Term::Symbol("send".into()),
                Term::Tree(vec![
                    Term::Symbol("slots".into()),
                    Term::Tree(vec![
                        Term::Symbol("person".into()),
                        Term::Symbol("person-new".into())
                    ])
                ])
            ])
        );
        let words = child.prepare(text).unwrap();
        let sentence = child.sentence(&words);
        assert!(sentence[6].iter().all(|id| *id < child.intent_base()));
        let mut work = 0;
        let values = child.scores(&sentence, &ExtraCache::None, &mut work);
        assert!(
            values[child.metadata.intents.len()..]
                .iter()
                .all(|v| *v == 0)
        );
        let context = child.context(&words, 3, Some(0), 0);
        assert!(context[6].iter().any(|id| *id >= child.intent_base()));
        let values = child.scores(&context, &ExtraCache::None, &mut work);
        assert!(
            values[..child.metadata.intents.len()]
                .iter()
                .all(|v| *v == 0)
        );
        for invalid in [
            Budget {
                hidden: 0,
                ..config
            },
            Budget {
                relative_regions: false,
                ..config
            },
            Budget {
                typed_roles: true,
                ..config
            },
            Budget {
                byte_convolution: true,
                ..config
            },
            Budget {
                freeze_ordered_context: true,
                ..config
            },
        ] {
            assert!(invalid.validate().is_err());
        }
        let mut corrupted = model.clone();
        corrupted.split_heads = false;
        assert!(Model::from_bytes(&corrupted.to_bytes().unwrap()).is_err());
        let mut corrupted = model.clone();
        corrupted.output.values.pop();
        assert!(Model::from_bytes(&corrupted.to_bytes().unwrap()).is_err());
        let mut corrupted = model;
        corrupted.output.scales[0] = f32::NAN;
        assert!(Model::from_bytes(&corrupted.to_bytes().unwrap()).is_err());
    }
    #[test]
    fn restricted_softmax_gradient_excludes_disallowed_classes_and_matches_loss() {
        let dim = 8;
        let width = dim * POSITIONS;
        let groups: Groups = [vec![0], vec![1], vec![], vec![], vec![], vec![], vec![]];
        let input = (0..16).map(|i| i as f32 * 0.01).collect::<Vec<_>>();
        let output = (0..3 * width)
            .map(|i| (i / width) as f32 * 0.02)
            .collect::<Vec<_>>();
        let bias = vec![0.1, 0.2, 9.0];
        let mut learner = Learner {
            optimizer: Some(optimizer::Adam::new([input.len(), output.len(), bias.len()]).unwrap()),
            task_boundary: None,
            input: input.clone(),
            output: output.clone(),
            bias: bias.clone(),
            dimensions: dim,
            hidden: 0,
        };
        assert!(!learner.update(&groups, 1, (0..3).filter(|i| *i != 2), 1.0, &[], None));
        assert_eq!(learner.input, input);
        assert_eq!(learner.output, output);
        assert_eq!(learner.bias, bias);
        let gradients = &learner.optimizer.as_ref().unwrap().gradients;
        assert_eq!(gradients[2][2], 0.0);
        assert!(gradients[1][2 * width..].iter().all(|v| *v == 0.0));
        let loss = |x: &[f32]| {
            let h = hidden(&groups, dim, |r, d| x[r * dim + d]);
            let s = logits(&h, &bias, |r, d| output[r * width + d]);
            (s[0].exp() + s[1].exp()).ln() - s[1]
        };
        for i in 0..input.len() {
            let mut hi = input.clone();
            let mut lo = input.clone();
            hi[i] += 0.001;
            lo[i] -= 0.001;
            assert!((gradients[0][i] - (loss(&hi) - loss(&lo)) / 0.002).abs() < 0.0002);
        }
    }
    #[test]
    fn constrained_training_uses_learned_support_without_deploying_policy_or_gold() {
        let config = Budget {
            characters: 0,
            relative_regions: true,
            typed_roles: true,
            constrained_loss: true,
            epochs: 32,
            hidden: 16,
            ..budget()
        };
        assert!(
            Budget {
                typed_roles: false,
                ..config
            }
            .validate()
            .is_err()
        );
        for mean_token_loss in [false, true] {
            let config = Budget {
                mean_token_loss,
                ..config
            };
            let model = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
            let raw = model.to_bytes().unwrap();
            assert_eq!(&raw[..8], ROLES_MAGIC);
            let child = Model::from_bytes(&raw).unwrap();
            assert_eq!(raw, child.to_bytes().unwrap());
            assert_eq!(
                raw,
                Model::learn(&data(), config, &mut |_| Ok(()))
                    .unwrap()
                    .to_bytes()
                    .unwrap()
            );
            assert!(!String::from_utf8_lossy(&raw).contains("constrained_loss"));
            for intent in ["send", "cancel"] {
                let text = format!("{intent} letter to novel-person");
                assert_eq!(
                    child.predict(&text).unwrap().output,
                    search::output(
                        &tokens(&text).unwrap(),
                        intent,
                        &["O", "O", "O", "B:person"]
                    )
                    .unwrap()
                );
            }
        }
    }
    #[test]
    fn minibatch_training_is_reproducible_and_optimizer_is_not_deployed() {
        let config = Budget {
            minibatch: 8,
            epochs: 32,
            learning_rate: 0.03,
            mean_token_loss: true,
            ..budget()
        };
        let mut steps = 0;
        let model = Model::learn_with_auxiliary(&data(), config, None, &mut |e| {
            steps += e.optimizer_steps;
            Ok(())
        })
        .unwrap();
        assert_eq!(steps, 128);
        let bytes = model.to_bytes().unwrap();
        assert_eq!(
            bytes,
            Model::learn(&data(), config, &mut |_| Ok(()))
                .unwrap()
                .to_bytes()
                .unwrap()
        );
        let born = Model::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, born.to_bytes().unwrap());
        for intent in ["send", "cancel"] {
            assert_eq!(
                born.predict(&format!("{intent} letter to stranger"))
                    .unwrap()
                    .output,
                search::output(
                    &tokens(&format!("{intent} letter to stranger")).unwrap(),
                    intent,
                    &["O", "O", "O", "B:person"]
                )
                .unwrap()
            );
        }
        for invalid in [
            Budget {
                minibatch: 1,
                ..config
            },
            Budget {
                minibatch: 65,
                ..config
            },
            Budget {
                ordered_context: true,
                characters: 0,
                ..config
            },
            Budget {
                byte_convolution: true,
                characters: 0,
                ..config
            },
        ] {
            assert!(invalid.validate().is_err());
        }
        assert!(!String::from_utf8_lossy(&bytes).contains("minibatch"));
    }
    #[test]
    fn lexical_graph_is_train_only_zero_preserves_rng_and_updates_are_counted() {
        let config = Budget {
            characters: 0,
            relative_regions: true,
            epochs: 8,
            ..budget()
        };
        let graph = LexicalGraph {
            weight: 0.0,
            interval: 3,
            edges: vec![["cancel".into(), "send".into()]],
        };
        let old = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
        let mut updates = 0;
        let zero = Model::learn_with_sources(&data(), config, None, Some(&graph), &mut |e| {
            updates += e.lexical_edge_updates;
            Ok(())
        })
        .unwrap();
        assert_eq!(old.to_bytes().unwrap(), zero.to_bytes().unwrap());
        assert_eq!(updates, 0);
        let active = LexicalGraph {
            weight: 0.2,
            ..graph.clone()
        };
        let learned = Model::learn_with_sources(&data(), config, None, Some(&active), &mut |e| {
            updates += e.lexical_edge_updates;
            Ok(())
        })
        .unwrap();
        assert_eq!(updates, data().len() * config.epochs / active.interval);
        assert_eq!(old.retained_bytes(), learned.retained_bytes());
        assert_eq!(old.weights(), learned.weights());
        assert_eq!(old.lexicon.bytes, learned.lexicon.bytes);
        let bytes = learned.to_bytes().unwrap();
        assert_ne!(old.to_bytes().unwrap(), bytes);
        assert_eq!(
            bytes,
            Model::from_bytes(&bytes).unwrap().to_bytes().unwrap()
        );
        assert_eq!(
            bytes,
            Model::learn_with_sources(&data(), config, None, Some(&active), &mut |_| Ok(()))
                .unwrap()
                .to_bytes()
                .unwrap()
        );
        assert!(!String::from_utf8_lossy(&bytes).contains("edges"));
        for invalid in [
            LexicalGraph {
                weight: f32::NAN,
                ..active.clone()
            },
            LexicalGraph {
                interval: 0,
                ..active.clone()
            },
            LexicalGraph {
                edges: vec![["cancel".into(), "unseen".into()]],
                ..active.clone()
            },
            LexicalGraph {
                edges: vec![["send".into(), "cancel".into()]],
                ..active.clone()
            },
            LexicalGraph {
                edges: vec![["cancel".into(), "send".into()]; 2],
                ..active.clone()
            },
        ] {
            assert!(
                Model::learn_with_sources(&data(), config, None, Some(&invalid), &mut |_| Ok(()))
                    .is_err()
            );
        }
        assert!(
            Model::learn_with_sources(
                &data(),
                Budget {
                    minibatch: 8,
                    ..config
                },
                None,
                Some(&active),
                &mut |_| Ok(())
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<LexicalGraph>(
                r#"{"weight":0.1,"interval":1,"edges":[["cancel","send"]],"teacher":"send"}"#
            )
            .is_err()
        );
    }
    #[test]
    fn positions_share_words_without_hash_collisions_and_roundtrip_without_sentences() {
        let model = Model::learn(&data(), budget(), &mut |_| Ok(())).unwrap();
        let raw = model.to_bytes().unwrap();
        let child = Model::from_bytes(&raw).unwrap();
        assert_eq!(
            child.predict("send letter to new-person").unwrap().output,
            search::output(
                &tokens("send letter to new-person").unwrap(),
                "send",
                &["O", "O", "O", "B:person"]
            )
            .unwrap()
        );
        assert_eq!(raw, child.to_bytes().unwrap());
        assert!(model.retained_bytes() < MAX_BYTES);
        assert!(!String::from_utf8_lossy(&raw).contains("send letter to person-0"));
        assert_ne!(model.lexicon.index("send"), model.lexicon.index("cancel"));
        for i in 0..raw.len() {
            let mut broken = raw.clone();
            broken[i] ^= 1;
            assert!(Model::from_bytes(&broken).is_err());
        }
        assert!(Lexicon::from_bytes(b"x\0x\0", 2).is_err());
        assert!(Lexicon::from_bytes(b"X\0", 1).is_err());
        assert!(Lexicon::from_bytes(b"a b\0", 1).is_err());
    }
    #[test]
    fn shared_embedding_gradient_matches_finite_difference_across_positions() {
        let dim = 8;
        let width = dim * POSITIONS;
        let groups: [Vec<usize>; POSITIONS] = [
            vec![0, 1],
            vec![1],
            vec![],
            vec![2],
            vec![0],
            vec![],
            vec![2],
        ];
        let input = (0..3 * dim)
            .map(|i| (i as f32 - 10.0) * 0.02)
            .collect::<Vec<_>>();
        let output = (0..2 * width)
            .map(|i| (i as f32 % 13.0 - 6.0) * 0.01)
            .collect::<Vec<_>>();
        let loss = |input: &[f32]| {
            let h = hidden(&groups, dim, |id, d| input[id * dim + d]);
            let s = logits(&h, &[0.0, 0.0], |c, d| output[c * width + d]);
            (s[0].exp() + s[1].exp()).ln() - s[1]
        };
        let mut learner = Learner {
            input: input.clone(),
            output: output.clone(),
            bias: vec![0.0; 2],
            dimensions: dim,
            hidden: 0,
            optimizer: None,
            task_boundary: None,
        };
        learner.update(&groups, 1, 0..2, 0.001, &[], None);
        for i in [0, 9, 20] {
            let mut high = input.clone();
            let mut low = input.clone();
            high[i] += 0.001;
            low[i] -= 0.001;
            let numerical = (loss(&high) - loss(&low)) / 0.002;
            let actual = (input[i] - learner.input[i]) / 0.001;
            assert!(
                (numerical - actual).abs() < 0.0002,
                "{i}: {numerical} vs {actual}"
            );
        }
    }
    #[test]
    fn mixing_layer_gradient_and_deep_codec_preserve_the_learned_computation() {
        let dim = 8;
        let width = dim * POSITIONS;
        let units = 8;
        let groups: Groups = [
            vec![0, 1],
            vec![1],
            vec![],
            vec![2],
            vec![0],
            vec![],
            vec![2],
        ];
        let input = (0..3 * dim)
            .map(|i| (i as f32 - 10.0) * 0.03)
            .collect::<Vec<_>>();
        let output = (0..units * width + 2 * units)
            .map(|i| (i as f32 % 17.0 - 8.0) * 0.03)
            .collect::<Vec<_>>();
        let mask = (0..width)
            .map(|i| if i % 3 == 0 { 0.0 } else { 1.5 })
            .collect::<Vec<_>>();
        let loss = |input: &[f32], output: &[f32]| {
            let mut h = hidden(&groups, dim, |id, d| input[id * dim + d]);
            for (v, m) in h.iter_mut().zip(&mask) {
                *v *= m;
            }
            let latent = logits(&h, &vec![0.0; units], |c, d| output[c * width + d])
                .into_iter()
                .map(f32::tanh)
                .collect::<Vec<_>>();
            let s = logits(&latent, &[0.0, 0.0], |c, d| {
                output[units * width + c * units + d]
            });
            (s[0].exp() + s[1].exp()).ln() - s[1]
        };
        let mut learner = Learner {
            input: input.clone(),
            output: output.clone(),
            bias: vec![0.0; units + 2],
            dimensions: dim,
            hidden: units,
            optimizer: None,
            task_boundary: None,
        };
        learner.update(&groups, 1, 0..2, 0.01, &mask, None);
        for i in [0, 9, 20] {
            let mut high = input.clone();
            let mut low = input.clone();
            high[i] += 0.002;
            low[i] -= 0.002;
            let expected = (loss(&high, &output) - loss(&low, &output)) / 0.004;
            let actual = (input[i] - learner.input[i]) / 0.01;
            assert!((expected - actual).abs() < 0.0002);
        }
        for i in [0, 17, 300] {
            let mut high = output.clone();
            let mut low = output.clone();
            high[i] += 0.002;
            low[i] -= 0.002;
            let expected = (loss(&input, &high) - loss(&input, &low)) / 0.004;
            let actual = (output[i] - learner.output[i]) / 0.01;
            assert!((expected - actual).abs() < 0.0002);
        }
        assert!(
            input
                .iter()
                .zip(&learner.input)
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
        let model = Model::learn(
            &data(),
            Budget {
                hidden: 16,
                epochs: 24,
                ..budget()
            },
            &mut |_| Ok(()),
        )
        .unwrap();
        let raw = model.to_bytes().unwrap();
        assert_eq!(&raw[..8], DEEP_MAGIC);
        let child = Model::from_bytes(&raw).unwrap();
        assert_eq!(raw, child.to_bytes().unwrap());
        assert_eq!(
            model.predict("send letter to person-3").unwrap().output,
            child.predict("send letter to person-3").unwrap().output
        );
        assert_eq!(child.codec(), "AXSQ0007");
    }
    #[test]
    fn repaired_checksums_do_not_bypass_shape_scale_or_lexical_validation() {
        let model = Model::learn(&data(), budget(), &mut |_| Ok(())).unwrap();
        let raw = model.to_bytes().unwrap();
        let size = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let end = 12 + size;
        let header: serde_json::Value = serde_json::from_slice(&raw[12..end]).unwrap();
        for field in ["dimensions", "characters", "words", "word_bytes", "hidden"] {
            let mut forged = header.clone();
            forged[field] = serde_json::json!(usize::MAX);
            let encoded = serde_json::to_vec(&forged).unwrap();
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            bytes.extend(encoded);
            bytes.extend_from_slice(&raw[end..raw.len() - 8]);
            bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
            assert!(Model::from_bytes(&bytes).is_err(), "{field}");
        }
        let lexical_end = end + model.lexicon.bytes.len();
        for offset in [lexical_end, raw.len() - 12] {
            let mut bytes = raw[..raw.len() - 8].to_vec();
            bytes[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
            assert!(Model::from_bytes(&bytes).is_err());
        }
        assert!(
            Model::learn(
                &data(),
                Budget {
                    vocabulary: usize::MAX,
                    ..budget()
                },
                &mut |_| Ok(())
            )
            .is_err()
        );
    }
    #[test]
    fn dropout_is_training_only_reproducible_and_default_preserves_rng() {
        let mut state = 47;
        assert!(dropout_mask(64, 0.0, &mut state).is_empty());
        assert_eq!(state, 47);
        let mask = dropout_mask(64, 0.25, &mut state);
        assert!(mask.contains(&0.0));
        assert!(mask.contains(&(1.0 / 0.75)));
        let mut replay = 47;
        assert_eq!(mask, dropout_mask(64, 0.25, &mut replay));
        let config = Budget {
            input_dropout: 0.25,
            hidden: 16,
            ..budget()
        };
        let a = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
        let b = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
        assert_eq!(a.to_bytes().unwrap(), b.to_bytes().unwrap());
        let raw = a.to_bytes().unwrap();
        let size = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&raw[12..12 + size]).unwrap();
        assert!(header.get("input_dropout").is_none());
        let child = Model::from_bytes(&raw).unwrap();
        assert_eq!(
            a.predict("send letter to new-person").unwrap().output,
            child.predict("send letter to new-person").unwrap().output
        );
        for input_dropout in [f32::NAN, -0.1, 0.76, f32::INFINITY] {
            assert!(
                Budget {
                    input_dropout,
                    ..budget()
                }
                .validate()
                .is_err()
            );
        }
    }
    #[test]
    fn ordered_regions_distinguish_equal_bags_and_include_distant_context() {
        let mut model = Model::learn(&data(), budget(), &mut |_| Ok(())).unwrap();
        let a = vec![vec![3], vec![4], vec![5], vec![6], vec![7]];
        let b = vec![vec![3], vec![6], vec![5], vec![4], vec![7]];
        assert_eq!(model.sentence(&a), model.sentence(&b));
        model.relative_regions = true;
        assert_ne!(model.sentence(&a), model.sentence(&b));
        for length in 1..=64 {
            let words = (0..length).map(|i| vec![i + 3]).collect::<Vec<_>>();
            let all = model
                .sentence(&words)
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(all.len(), length);
            assert_eq!(all, (3..length + 3).collect::<Vec<_>>());
        }
        // Regions retain coarse order only: permutations inside a region still collapse.
        let long = (0..20).map(|i| vec![i + 3]).collect::<Vec<_>>();
        let mut within_region = long.clone();
        within_region.swap(4, 5);
        assert_eq!(model.sentence(&long), model.sentence(&within_region));
        let words = (0..9).map(|i| vec![i + 3]).collect::<Vec<_>>();
        let mut changed = words.clone();
        changed[0] = vec![23];
        assert_ne!(
            model.context(&words, 4, Some(0), 0),
            model.context(&changed, 4, Some(0), 0)
        );
        model.relative_regions = false;
        assert_eq!(
            model.context(&words, 4, Some(0), 0),
            model.context(&changed, 4, Some(0), 0)
        );
    }
    #[test]
    fn region_order_is_learned_and_survives_quantization_with_an_unseen_value() {
        let examples = (0..24)
            .flat_map(|i| {
                [("left", "alpha", "beta"), ("right", "beta", "alpha")].map(
                    move |(intent, a, b)| Example {
                        text: format!("start {a} filler-{i} {b} end"),
                        intent: intent.into(),
                        tags: vec!["O".into(); 5],
                    },
                )
            })
            .collect::<Vec<_>>();
        let model = Model::learn(
            &examples,
            Budget {
                relative_regions: true,
                characters: 0,
                hidden: 16,
                epochs: 32,
                ..budget()
            },
            &mut |_| Ok(()),
        )
        .unwrap();
        let raw = model.to_bytes().unwrap();
        assert_eq!(&raw[..8], REGIONS_MAGIC);
        let child = Model::from_bytes(&raw).unwrap();
        assert_eq!(child.codec(), "AXSQ0009");
        assert_eq!(raw, child.to_bytes().unwrap());
        assert_eq!(model.retained_bytes(), child.retained_bytes());
        for (intent, a, b) in [("left", "alpha", "beta"), ("right", "beta", "alpha")] {
            let text = format!("start {a} unseen-value {b} end");
            assert_eq!(
                child.predict(&text).unwrap().output,
                search::output(&tokens(&text).unwrap(), intent, &["O"; 5]).unwrap()
            );
        }
        assert!(!String::from_utf8_lossy(&raw).contains("start alpha filler-0 beta end"));
        assert!(child.retained_bytes() < MAX_BYTES);
    }
    #[test]
    fn repaired_checksums_cannot_reinterpret_region_or_legacy_layouts() {
        for relative_regions in [false, true] {
            for hidden in [0, 16] {
                let model = Model::learn(
                    &data(),
                    Budget {
                        relative_regions,
                        hidden,
                        ..budget()
                    },
                    &mut |_| Ok(()),
                )
                .unwrap();
                let raw = model.to_bytes().unwrap();
                let size = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
                let end = 12 + size;
                let header: serde_json::Value = serde_json::from_slice(&raw[12..end]).unwrap();
                assert_eq!(
                    header.get("relative_regions"),
                    relative_regions.then_some(&serde_json::Value::Bool(true))
                );
                for magic in [MAGIC, DEEP_MAGIC, REGIONS_MAGIC] {
                    if magic == &raw[..8] {
                        continue;
                    }
                    let mut forged = raw[..raw.len() - 8].to_vec();
                    forged[..8].copy_from_slice(magic);
                    forged.extend_from_slice(&checksum(&forged).to_le_bytes());
                    assert!(Model::from_bytes(&forged).is_err());
                }
                for value in [serde_json::json!(!relative_regions), serde_json::json!(1)] {
                    let mut changed = header.clone();
                    changed["relative_regions"] = value;
                    let encoded = serde_json::to_vec(&changed).unwrap();
                    let mut forged = raw[..8].to_vec();
                    forged.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
                    forged.extend(encoded);
                    forged.extend_from_slice(&raw[end..raw.len() - 8]);
                    forged.extend_from_slice(&checksum(&forged).to_le_bytes());
                    assert!(Model::from_bytes(&forged).is_err());
                }
            }
        }
    }
    #[test]
    fn auxiliary_zero_preserves_primary_rng_bytes_and_no_auxiliary_updates_occur() {
        let zero = Auxiliary {
            weight: 0.0,
            ..auxiliary()
        };
        for relative_regions in [false, true] {
            for hidden in [0, 16] {
                let budget = Budget {
                    relative_regions,
                    hidden,
                    ..budget()
                };
                let original = Model::learn(&data(), budget, &mut |_| Ok(())).unwrap();
                let mut updates = 0;
                let control = Model::learn_with_auxiliary(&data(), budget, Some(&zero), &mut |e| {
                    updates += e.auxiliary_words + e.auxiliary_sentences + e.auxiliary_errors;
                    Ok(())
                })
                .unwrap();
                assert_eq!(updates, 0);
                assert_eq!(original.to_bytes().unwrap(), control.to_bytes().unwrap());
                assert_eq!(original.retained_bytes(), control.retained_bytes());
            }
        }
    }
    #[test]
    fn auxiliary_teaches_shared_weights_but_neither_vocabulary_nor_head_is_deployed() {
        let budget = Budget {
            hidden: 16,
            relative_regions: true,
            ..budget()
        };
        let aux = auxiliary();
        let plain = Model::learn(&data(), budget, &mut |_| Ok(())).unwrap();
        let mut sentences = 0;
        let mut words = 0;
        let learned = Model::learn_with_auxiliary(&data(), budget, Some(&aux), &mut |e| {
            assert!(e.auxiliary_errors <= e.auxiliary_words);
            sentences += e.auxiliary_sentences;
            words += e.auxiliary_words;
            Ok(())
        })
        .unwrap();
        let replay =
            Model::learn_with_auxiliary(&data(), budget, Some(&aux), &mut |_| Ok(())).unwrap();
        assert_eq!(sentences, data().len() * budget.epochs / aux.interval);
        assert_eq!(words, sentences * 2);
        assert_eq!(learned.lexicon.bytes, plain.lexicon.bytes);
        assert_eq!(learned.lexicon.index("aux-only-0"), None);
        assert_eq!(learned.weights(), plain.weights());
        assert_eq!(learned.retained_bytes(), plain.retained_bytes());
        assert_eq!(learned.metadata.intents, plain.metadata.intents);
        assert_eq!(learned.metadata.tags, plain.metadata.tags);
        assert_ne!(learned.input.values, plain.input.values);
        let raw = learned.to_bytes().unwrap();
        assert_eq!(raw, replay.to_bytes().unwrap());
        let child = Model::from_bytes(&raw).unwrap();
        assert_eq!(raw, child.to_bytes().unwrap());
        assert_eq!(learned.retained_bytes(), child.retained_bytes());
        let size = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&raw[12..12 + size]).unwrap();
        assert!(header.get("auxiliary").is_none());
        assert_eq!(
            learned.bias.len(),
            budget.hidden + learned.metadata.intents.len() + learned.metadata.tags.len()
        );
    }
    #[test]
    fn auxiliary_invalid_supervision_and_budgets_fail_before_any_training_event() {
        let mut bad = Vec::new();
        for weight in [-0.1, 1.01, f32::NAN, f32::INFINITY] {
            bad.push(Auxiliary {
                weight,
                ..auxiliary()
            });
        }
        for interval in [0, 65] {
            bad.push(Auxiliary {
                interval,
                ..auxiliary()
            });
        }
        let mut count = auxiliary();
        count.examples.truncate(3);
        bad.push(count);
        let mut shape = auxiliary();
        shape.examples[0].labels.pop();
        bad.push(shape);
        let mut label = auxiliary();
        label.examples[0].labels[0] = "root".into();
        label.weight = 0.0;
        bad.push(label);
        let mut overlong = auxiliary();
        overlong.examples[0].text = vec!["word"; 65].join(" ");
        overlong.examples[0].labels = vec!["NOUN".into(); 65];
        bad.push(overlong);
        for auxiliary in bad {
            let mut events = 0;
            assert!(
                Model::learn_with_auxiliary(&data(), budget(), Some(&auxiliary), &mut |_| {
                    events += 1;
                    Ok(())
                })
                .is_err()
            );
            assert_eq!(events, 0);
        }
        assert!(
            serde_json::from_value::<AuxiliaryExample>(
                serde_json::json!({"text":"x","labels":["NOUN"],"expected":"secret"})
            )
            .is_err()
        );
    }
    #[test]
    fn auxiliary_keeps_annotated_terminal_punctuation_without_changing_primary_tokens() {
        let mut aux = auxiliary();
        for row in &mut aux.examples {
            row.text.push_str(" .");
            row.labels.push("PUNCT".into());
        }
        assert_eq!(tokens(&aux.examples[0].text).unwrap().len(), 2);
        assert_eq!(tokens_mode(&aux.examples[0].text, false).unwrap().len(), 3);
        assert_eq!(tokens("send letter.").unwrap(), vec!["send", "letter"]);
        let mut words = 0;
        Model::learn_with_auxiliary(&data(), budget(), Some(&aux), &mut |e| {
            words += e.auxiliary_words;
            Ok(())
        })
        .unwrap();
        assert_eq!(words, data().len() * budget().epochs / aux.interval * 3);
    }
    #[test]
    fn deferred_virtual_embedding_gradient_matches_loss_without_updating_cache_rows() {
        let dim = 8;
        let width = dim * POSITIONS;
        let groups: Groups = [
            vec![0, 1],
            vec![1],
            vec![],
            vec![2],
            vec![0],
            vec![],
            vec![2],
        ];
        let input = (0..3 * dim)
            .map(|i| (i as f32 - 10.0) * 0.02)
            .collect::<Vec<_>>();
        let output = (0..2 * width)
            .map(|i| (i as f32 % 13.0 - 6.0) * 0.01)
            .collect::<Vec<_>>();
        let loss = |x: &[f32]| {
            let h = hidden(&groups, dim, |r, d| x[r * dim + d]);
            let s = logits(&h, &[0.0, 0.0], |r, d| output[r * width + d]);
            (s[0].exp() + s[1].exp()).ln() - s[1]
        };
        let mut learner = Learner {
            input: input.clone(),
            output: output.clone(),
            bias: vec![0.0; 2],
            dimensions: dim,
            hidden: 0,
            optimizer: None,
            task_boundary: None,
        };
        let mut deltas = vec![0.0; dim];
        learner.update(&groups, 1, 0..2, 0.01, &[], Some((2, &mut deltas)));
        assert_eq!(&learner.input[2 * dim..], &input[2 * dim..]);
        for d in 0..dim {
            let mut hi = input.clone();
            let mut lo = input.clone();
            hi[2 * dim + d] += 0.001;
            lo[2 * dim + d] -= 0.001;
            let expected = (loss(&hi) - loss(&lo)) / 0.002;
            assert!((deltas[d] / 0.01 - expected).abs() < 0.0002);
        }
    }
    #[test]
    fn byte_model_roundtrip_keeps_shapes_new_values_work_and_strict_flags() {
        let config = Budget {
            characters: 0,
            hidden: 16,
            epochs: 32,
            relative_regions: true,
            byte_convolution: true,
            ..budget()
        };
        let model = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
        let bytes = model.to_bytes().unwrap();
        assert_eq!(&bytes[..8], BYTE_MAGIC);
        let child = Model::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, child.to_bytes().unwrap());
        assert_eq!(child.codec(), "AXSQ0010");
        assert_eq!(model.retained_bytes(), child.retained_bytes());
        assert!(bytes.len() < MAX_BYTES);
        for intent in ["send", "cancel"] {
            let text = format!("{intent} letter to person-never");
            let p = child.predict(&text).unwrap();
            assert_eq!(
                p.output,
                search::output(
                    &tokens(&text).unwrap(),
                    intent,
                    &["O", "O", "O", "B:person"]
                )
                .unwrap()
            );
            assert!(
                p.work
                    > child
                        .byte_cache(&tokens(&text).unwrap(), &mut 0)
                        .unwrap()
                        .iter()
                        .map(|c| c.work)
                        .sum::<usize>()
            );
        }
        let raw = child
            .byte_cache(&["new-alpha".into(), "new-beta".into()], &mut 0)
            .unwrap();
        assert_ne!(raw[0].output, raw[1].output);
        let size = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let end = 12 + size;
        let header: serde_json::Value = serde_json::from_slice(&bytes[12..end]).unwrap();
        assert!(header.get("freeze_byte_convolution").is_none());
        for (field, value) in [
            ("byte_convolution", serde_json::json!(false)),
            ("characters", serde_json::json!(512)),
            ("hidden", serde_json::json!(usize::MAX)),
        ] {
            let mut h = header.clone();
            h[field] = value;
            let encoded = serde_json::to_vec(&h).unwrap();
            let mut forged = BYTE_MAGIC.to_vec();
            forged.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            forged.extend(encoded);
            forged.extend_from_slice(&bytes[end..bytes.len() - 8]);
            forged.extend_from_slice(&checksum(&forged).to_le_bytes());
            assert!(Model::from_bytes(&forged).is_err(), "{field}");
        }
        for magic in [MAGIC, DEEP_MAGIC, REGIONS_MAGIC] {
            let mut forged = bytes[..bytes.len() - 8].to_vec();
            forged[..8].copy_from_slice(magic);
            forged.extend_from_slice(&checksum(&forged).to_le_bytes());
            assert!(Model::from_bytes(&forged).is_err());
        }
        for bad in [
            Budget {
                freeze_byte_convolution: true,
                ..budget()
            },
            Budget {
                byte_convolution: true,
                ..budget()
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }
    #[test]
    fn frozen_byte_features_reproduce_initial_parameters_and_learning_changes_them() {
        let base = Budget {
            characters: 0,
            hidden: 16,
            epochs: 32,
            byte_convolution: true,
            ..budget()
        };
        let frozen = Model::learn(
            &data(),
            Budget {
                freeze_byte_convolution: true,
                ..base
            },
            &mut |_| Ok(()),
        )
        .unwrap();
        let initial = byte_convolution::Train::new(base.dimensions, base.seed).unwrap();
        let quantized = Quantized::encode(&initial.embedding, byte_convolution::EMBED);
        let offset = frozen.input_rows() * base.dimensions;
        assert_eq!(&frozen.input.values[offset..], quantized.values.as_slice());
        let learned = Model::learn(&data(), base, &mut |_| Ok(())).unwrap();
        assert_ne!(
            &learned.input.values[offset..],
            &frozen.input.values[offset..]
        );
        assert_eq!(learned.words(), frozen.words());
        assert_eq!(learned.weights(), frozen.weights());
        assert_eq!(learned.retained_bytes(), frozen.retained_bytes());
        let replay = Model::learn(&data(), base, &mut |_| Ok(())).unwrap();
        assert_eq!(learned.to_bytes().unwrap(), replay.to_bytes().unwrap());
    }
    #[test]
    fn ordered_context_roundtrip_new_values_and_repaired_flags_are_strict() {
        let config = Budget {
            characters: 0,
            hidden: 16,
            epochs: 32,
            ordered_context: true,
            relative_regions: true,
            ..budget()
        };
        let model = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
        let bytes = model.to_bytes().unwrap();
        assert_eq!(&bytes[..8], ORDERED_MAGIC);
        assert_eq!(model.codec(), "AXSQ0011");
        let child = Model::from_bytes(&bytes).unwrap();
        assert_eq!(child.to_bytes().unwrap(), bytes);
        assert_eq!(child.retained_bytes(), model.retained_bytes());
        assert_eq!(child.words(), model.words());
        for intent in ["send", "cancel"] {
            let text = format!("{intent} letter to person-never");
            assert_eq!(
                child.predict(&text).unwrap().output,
                search::output(
                    &tokens(&text).unwrap(),
                    intent,
                    &["O", "O", "O", "B:person"]
                )
                .unwrap()
            );
            assert_eq!(model.measure(&text).work, child.measure(&text).work);
        }
        let end = 12 + u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[12..end]).unwrap();
        assert!(header.get("freeze_ordered_context").is_none());
        for (field, value) in [
            ("ordered_context", serde_json::json!(false)),
            ("byte_convolution", serde_json::json!(true)),
            ("characters", serde_json::json!(256)),
            ("dimensions", serde_json::json!(usize::MAX)),
        ] {
            let mut h = header.clone();
            h[field] = value;
            let encoded = serde_json::to_vec(&h).unwrap();
            let mut forged = ORDERED_MAGIC.to_vec();
            forged.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
            forged.extend(encoded);
            forged.extend_from_slice(&bytes[end..bytes.len() - 8]);
            forged.extend_from_slice(&checksum(&forged).to_le_bytes());
            assert!(Model::from_bytes(&forged).is_err(), "{field}");
        }
        for magic in [MAGIC, DEEP_MAGIC, REGIONS_MAGIC, BYTE_MAGIC] {
            let mut forged = bytes[..bytes.len() - 8].to_vec();
            forged[..8].copy_from_slice(magic);
            forged.extend_from_slice(&checksum(&forged).to_le_bytes());
            assert!(Model::from_bytes(&forged).is_err());
        }
        for bad in [
            Budget {
                freeze_ordered_context: true,
                ..budget()
            },
            Budget {
                ordered_context: true,
                ..budget()
            },
            Budget {
                byte_convolution: true,
                ..config
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }
    #[test]
    fn frozen_ordered_context_matches_initial_parameters_and_is_reproducible() {
        let config = Budget {
            characters: 0,
            hidden: 16,
            epochs: 32,
            ordered_context: true,
            relative_regions: true,
            ..budget()
        };
        let frozen = Model::learn(
            &data(),
            Budget {
                freeze_ordered_context: true,
                ..config
            },
            &mut |_| Ok(()),
        )
        .unwrap();
        let initial = ordered_context::Train::new(config.dimensions, config.seed).unwrap();
        let quantized = Quantized::encode(&initial.weights, config.dimensions);
        let offset = frozen.output.values.len() - initial.weights.len();
        assert_eq!(&frozen.output.values[offset..], quantized.values.as_slice());
        let learned = Model::learn(&data(), config, &mut |_| Ok(())).unwrap();
        assert_ne!(
            &learned.output.values[offset..],
            &frozen.output.values[offset..]
        );
        assert_eq!(learned.weights(), frozen.weights());
        assert_eq!(learned.retained_bytes(), frozen.retained_bytes());
        assert_eq!(
            learned.to_bytes().unwrap(),
            Model::learn(&data(), config, &mut |_| Ok(()))
                .unwrap()
                .to_bytes()
                .unwrap()
        );
    }
    #[test]
    fn order_inside_one_region_can_be_taught_and_new_value_copied() {
        let sentence = |reverse: bool, value: &str| {
            let mut words = vec![
                "start".to_owned(),
                if reverse { "bob" } else { "ana" }.to_owned(),
                if reverse { "ana" } else { "bob" }.to_owned(),
            ];
            words.extend((0..23).map(|i| format!("filler-{i}")));
            words.push(value.to_owned());
            words.push("end".to_owned());
            words.join(" ")
        };
        let examples = (0..32)
            .flat_map(|i| {
                [false, true].map({
                    let sentence = &sentence;
                    move |reverse| {
                        let mut tags = vec!["O".to_owned(); 28];
                        tags[26] = "B:value".to_owned();
                        Example {
                            text: sentence(reverse, &format!("value-{i}")),
                            intent: if reverse { "reverse" } else { "forward" }.to_owned(),
                            tags,
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        let config = Budget {
            characters: 0,
            relative_regions: true,
            ordered_context: true,
            mean_token_loss: true,
            hidden: 0,
            epochs: 32,
            learning_rate: 0.2,
            vocabulary: 32,
            ..budget()
        };
        let learned = Model::learn(&examples, config, &mut |_| Ok(())).unwrap();
        let plain = Model::learn(
            &examples,
            Budget {
                ordered_context: false,
                ..config
            },
            &mut |_| Ok(()),
        )
        .unwrap();
        let first = sentence(false, "new-value");
        let second = sentence(true, "new-value");
        assert_eq!(
            plain.sentence(&plain.prepare(&first).unwrap()),
            plain.sentence(&plain.prepare(&second).unwrap())
        );
        let child = Model::from_bytes(&learned.to_bytes().unwrap()).unwrap();
        for (text, intent) in [(first, "forward"), (second, "reverse")] {
            let mut tags = vec!["O"; 28];
            tags[26] = "B:value";
            assert_eq!(
                child.predict(&text).unwrap().output,
                search::output(&tokens(&text).unwrap(), intent, &tags).unwrap()
            );
        }
    }
    #[test]
    fn recurrent_and_direct_word_gradients_share_one_frozen_sentence_input() {
        let dim = 8;
        let initial = (0..3 * dim)
            .map(|i| (i as f32 - 11.0) * 0.02)
            .collect::<Vec<_>>();
        let weights = (0..2 * dim * POSITIONS)
            .map(|i| (i as f32 % 13.0 - 6.0) * 0.01)
            .collect::<Vec<_>>();
        let mut ordered = ordered_context::Train::new(dim, 47).unwrap();
        let recurrent_before = ordered.weights.clone();
        let features = vec![vec![0], vec![1], vec![0]];
        let groups = [
            vec![3, 4],
            vec![],
            vec![0, 5],
            vec![],
            vec![2],
            vec![],
            vec![],
        ];
        let loss = |input: &[f32]| {
            let source = features
                .iter()
                .flat_map(|f| input[f[0] * dim..(f[0] + 1) * dim].iter().copied())
                .collect::<Vec<_>>();
            let cache = ordered.encode(&source).unwrap();
            let mut all = input.to_vec();
            all.extend(cache.output);
            let h = hidden(&groups, dim, |r, d| all[r * dim + d]);
            let scores = logits(&h, &[0.0, 0.0], |r, d| weights[r * dim * POSITIONS + d]);
            (scores[0].exp() + scores[1].exp()).ln() - scores[1]
        };
        let expected = (0..initial.len())
            .map(|i| {
                let mut hi = initial.clone();
                let mut lo = initial.clone();
                hi[i] += 0.001;
                lo[i] -= 0.001;
                (loss(&hi) - loss(&lo)) / 0.002
            })
            .collect::<Vec<_>>();
        let mut learner = Learner {
            input: initial.clone(),
            output: weights,
            bias: vec![0.0; 2],
            dimensions: dim,
            hidden: 0,
            optimizer: None,
            task_boundary: None,
        };
        let (cache, mut deltas) =
            prepare_extra_training(&mut learner, None, Some(&ordered), &[], &features, 3).unwrap();
        learner.update(&groups, 1, 0..2, 0.01, &[], Some((0, &mut deltas)));
        assert_eq!(&learner.input[..initial.len()], initial);
        apply_ordered_training(&mut ordered, &cache, &deltas, &mut learner, 1.0, true).unwrap();
        assert_eq!(ordered.weights, recurrent_before);
        for (i, expected) in expected.iter().enumerate() {
            assert!(
                ((initial[i] - learner.input[i]) / 0.01 - expected).abs() < 0.0002,
                "word coordinate {i}"
            );
        }
    }
}
