//! Experimental low-rank feature sharing, trained locally; not a pretrained LLM.
use super::{Model as Metadata, *};
use crate::live_store::checksum;
use std::io::{Cursor, Read};
pub const CONTEXT: &str = "axon/factorized-sequence-v1";
const MAGIC: &[u8; 8] = b"AXSQ0004";
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub seed: u64,
    pub buckets: usize,
    pub dimensions: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub subwords: bool,
    #[serde(default)]
    pub tanh: bool,
    #[serde(default)]
    pub balanced: bool,
    pub minimum_margin: u16,
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
            || !(1..=255).contains(&self.minimum_margin)
            || self
                .buckets
                .saturating_mul(self.dimensions.saturating_add(4))
                >= MAX_BYTES
        {
            return Err(Error::Invalid("factorized budget"));
        }
        Ok(())
    }
    fn features(self) -> super::Budget {
        super::Budget {
            seed: self.seed,
            buckets: self.buckets,
            epochs: 1,
            weights: 256,
            intent_weights: 128,
            subwords: self.subwords,
            shared_roles: false,
            minimum_margin: self.minimum_margin,
        }
    }
}
#[derive(Clone, Debug)]
pub(super) struct Quantized {
    pub(super) values: Vec<i8>,
    pub(super) scales: Vec<f32>,
}
impl Quantized {
    pub(super) fn encode(matrix: &[f32], dimensions: usize) -> Self {
        let mut values = Vec::with_capacity(matrix.len());
        let mut scales = Vec::with_capacity(matrix.len() / dimensions);
        for row in matrix.chunks_exact(dimensions) {
            let maximum = row.iter().map(|x| x.abs()).fold(0.0, f32::max);
            let scale = if maximum == 0.0 {
                1.0 / 127.0
            } else {
                maximum / 127.0
            };
            scales.push(scale);
            values.extend(
                row.iter()
                    .map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8),
            );
        }
        Self { values, scales }
    }
    pub(super) fn append(&self, bytes: &mut Vec<u8>) {
        for scale in &self.scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        bytes.extend(self.values.iter().map(|v| *v as u8));
    }
    pub(super) fn read(
        input: &mut Cursor<&[u8]>,
        rows: usize,
        dimensions: usize,
    ) -> Result<Self, Error> {
        Self::read_packed(input, rows, rows * dimensions)
    }
    pub(super) fn read_packed(
        input: &mut Cursor<&[u8]>,
        rows: usize,
        values: usize,
    ) -> Result<Self, Error> {
        let mut scales = Vec::with_capacity(rows);
        for _ in 0..rows {
            let x = f32::from_le_bytes(take(input)?);
            if !x.is_finite() || !(1e-10..=0.064).contains(&x) {
                return Err(Error::Invalid("factorized scale"));
            }
            scales.push(x);
        }
        let mut raw = vec![0u8; values];
        input
            .read_exact(&mut raw)
            .map_err(|_| Error::Invalid("factorized truncation"))?;
        if raw.contains(&128) {
            return Err(Error::Invalid("factorized coefficient"));
        }
        Ok(Self {
            values: raw.into_iter().map(|x| x as i8).collect(),
            scales,
        })
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.values.capacity() + self.scales.capacity() * 4
    }
}
#[derive(Clone, Debug)]
pub struct Model {
    metadata: Metadata,
    dimensions: usize,
    subwords: bool,
    tanh: bool,
    input: Quantized,
    output: Quantized,
    bias: Vec<f32>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    metadata: Wire,
    dimensions: usize,
    subwords: bool,
    #[serde(default, skip_serializing_if = "disabled")]
    tanh: bool,
}
fn disabled(flag: &bool) -> bool {
    !*flag
}
pub(super) fn take<const N: usize>(input: &mut Cursor<&[u8]>) -> Result<[u8; N], Error> {
    let mut bytes = [0; N];
    input
        .read_exact(&mut bytes)
        .map_err(|_| Error::Invalid("factorized truncation"))?;
    Ok(bytes)
}
pub(super) fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
pub(super) struct Learner {
    pub(super) input: Vec<f32>,
    pub(super) output: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) dimensions: usize,
    pub(super) tanh: bool,
}
impl Learner {
    fn hidden(&self, features: &[usize]) -> Vec<f32> {
        let mut h = vec![0.0; self.dimensions];
        let norm = (features.len() as f32).sqrt();
        for f in features {
            for (h, w) in h
                .iter_mut()
                .zip(&self.input[f * self.dimensions..(f + 1) * self.dimensions])
            {
                *h += w / norm;
            }
        }
        if self.tanh {
            for x in &mut h {
                *x = x.tanh();
            }
        }
        h
    }
    pub(super) fn update(
        &mut self,
        features: &[usize],
        gold: usize,
        head: std::ops::Range<usize>,
        rate: f32,
    ) -> bool {
        let h = self.hidden(features);
        let mut scores = head
            .clone()
            .map(|c| {
                self.bias[c]
                    + h.iter()
                        .zip(&self.output[c * self.dimensions..(c + 1) * self.dimensions])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let guess = (0..scores.len())
            .max_by(|a, b| scores[*a].total_cmp(&scores[*b]).then_with(|| b.cmp(a)))
            .unwrap()
            + head.start;
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for x in &mut scores {
            *x = (*x - maximum).exp();
        }
        let total = scores.iter().sum::<f32>();
        let mut gradient = vec![0.0; self.dimensions];
        for (c, probability) in head.zip(scores) {
            let error = probability / total - if c == gold { 1.0 } else { 0.0 };
            self.bias[c] = (self.bias[c] - rate * error).clamp(-16.0, 16.0);
            for ((w, h), g) in self.output[c * self.dimensions..(c + 1) * self.dimensions]
                .iter_mut()
                .zip(&h)
                .zip(&mut gradient)
            {
                *g += error * *w;
                *w = (*w - rate * error * h).clamp(-8.0, 8.0);
            }
        }
        let norm = (features.len() as f32).sqrt();
        if self.tanh {
            for (g, h) in gradient.iter_mut().zip(&h) {
                *g *= 1.0 - h * h;
            }
        }
        for f in features {
            for (w, g) in self.input[f * self.dimensions..(f + 1) * self.dimensions]
                .iter_mut()
                .zip(&gradient)
            {
                *w = (*w - rate * g / norm).clamp(-8.0, 8.0);
            }
        }
        guess != gold
    }
}
impl Model {
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        Self::fit(examples, budget, None, emit)
    }
    pub fn refine(
        examples: &[Example],
        budget: Budget,
        initial: &Self,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        Self::fit(examples, budget, Some(initial), emit)
    }
    fn fit(
        examples: &[Example],
        budget: Budget,
        initial: Option<&Self>,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        budget.validate()?;
        if initial.is_some_and(|m| {
            m.metadata.seed != budget.seed
                || m.metadata.buckets != budget.buckets
                || m.dimensions != budget.dimensions
                || m.subwords != budget.subwords
                || m.tanh != budget.tanh
        }) {
            return Err(Error::Invalid("factorized refinement schema"));
        }
        let (intents, tags) = class_labels(examples, initial.map(|m| &m.metadata))?;
        let n = intents.len();
        let classes = n + tags.len();
        let dimensions = budget.dimensions;
        let mut state = budget.seed;
        let mut learner = Learner {
            input: (0..budget.buckets * dimensions)
                .map(|_| {
                    ((random(&mut state) >> 40) as f32 / (1u32 << 24) as f32 - 0.5)
                        / (dimensions as f32).sqrt()
                })
                .collect(),
            output: vec![0.0; classes * dimensions],
            bias: vec![0.0; classes],
            dimensions,
            tanh: budget.tanh,
        };
        if let Some(m) = initial {
            for row in 0..budget.buckets {
                for column in 0..dimensions {
                    learner.input[row * dimensions + column] =
                        f32::from(m.input.values[row * dimensions + column]) * m.input.scales[row];
                }
            }
            for old in 0..m.bias.len() {
                let new = if old < m.metadata.intents.len() {
                    intents.binary_search(&m.metadata.intents[old]).unwrap()
                } else {
                    n + tags
                        .iter()
                        .position(|t| t == &m.metadata.tags[old - m.metadata.intents.len()])
                        .unwrap()
                };
                for column in 0..dimensions {
                    learner.output[new * dimensions + column] =
                        f32::from(m.output.values[old * dimensions + column])
                            * m.output.scales[old];
                }
                learner.bias[new] = m.bias[old];
            }
        }
        let features = budget.features();
        let mut prepared = Vec::with_capacity(examples.len());
        for e in examples {
            let words = tokens(&e.text)?;
            prepared.push((
                intent_features(&words, features),
                (0..words.len())
                    .map(|i| word_features(&words, i, features))
                    .collect::<Vec<_>>(),
                intents.binary_search(&e.intent).unwrap(),
                e.tags
                    .iter()
                    .map(|t| n + tags.iter().position(|s| s == t).unwrap())
                    .collect::<Vec<_>>(),
            ));
        }
        let mut order = (0..examples.len()).collect::<Vec<_>>();
        let mut counts = vec![0usize; n];
        for (_, _, intent, _) in &prepared {
            counts[*intent] += 1;
        }
        let class_weights = counts
            .iter()
            .map(|count| {
                (examples.len() as f32 / (n * (*count).max(1)) as f32)
                    .sqrt()
                    .clamp(0.25, 4.0)
            })
            .collect::<Vec<_>>();
        for epoch in 0..budget.epochs {
            for i in (1..order.len()).rev() {
                order.swap(i, random(&mut state) as usize % (i + 1));
            }
            let mut event = Event {
                epoch: epoch + 1,
                sentences: examples.len(),
                intent_errors: 0,
                tag_errors: 0,
            };
            for (step, index) in order.iter().enumerate() {
                let progress = (epoch * examples.len() + step) as f32
                    / (budget.epochs * examples.len()) as f32;
                let rate = budget.learning_rate * (1.0 - progress).max(0.1);
                let (f, words, intent, gold_tags) = &prepared[*index];
                let rate = if budget.balanced {
                    rate * class_weights[*intent]
                } else {
                    rate
                };
                event.intent_errors += usize::from(learner.update(f, *intent, 0..n, rate));
                let (mut previous, mut previous2) = ("<start>", "<start2>");
                for (base, gold) in words.iter().zip(gold_tags) {
                    let f = contextual(base, previous, previous2, &intents[*intent], features);
                    event.tag_errors += usize::from(learner.update(&f, *gold, n..classes, rate));
                    previous2 = previous;
                    previous = &tags[*gold - n];
                }
            }
            emit(&event)?;
        }
        let model = Self {
            metadata: metadata(Wire {
                revision: 1,
                seed: budget.seed,
                buckets: budget.buckets,
                intents,
                tags,
                minimum_margin: budget.minimum_margin,
                packed: String::new(),
            }),
            dimensions,
            subwords: budget.subwords,
            tanh: budget.tanh,
            input: Quantized::encode(&learner.input, dimensions),
            output: Quantized::encode(&learner.output, dimensions),
            bias: learner.bias.into_boxed_slice().into_vec(),
        };
        model.to_bytes()?;
        Ok(model)
    }
    fn scores(&self, features: &[usize], work: &mut usize) -> Vec<i32> {
        let mut h = vec![0.0; self.dimensions];
        let norm = (features.len() as f32).sqrt();
        for f in features {
            for (h, w) in h
                .iter_mut()
                .zip(&self.input.values[f * self.dimensions..(f + 1) * self.dimensions])
            {
                *h += f32::from(*w) * self.input.scales[*f] / norm;
            }
        }
        if self.tanh {
            for x in &mut h {
                *x = x.tanh();
            }
        }
        *work += (features.len() + self.bias.len()) * self.dimensions;
        (0..self.bias.len())
            .map(|c| {
                let score = self.bias[c]
                    + h.iter()
                        .zip(&self.output.values[c * self.dimensions..(c + 1) * self.dimensions])
                        .map(|(h, w)| h * f32::from(*w) * self.output.scales[c])
                        .sum::<f32>();
                (score * 1000.0).round() as i32
            })
            .collect()
    }
    pub fn predict(&self, text: &str) -> Result<Prediction, Error> {
        self.metadata
            .predict_configured(text, self.subwords, |f, w| self.scores(f, w))
    }
    pub fn probe_search(
        &self,
        text: &str,
        budget: search::Budget,
    ) -> Result<search::Prediction, Error> {
        search::predict(&self.metadata, text, self.subwords, budget, |f, work| {
            self.scores(f, work)
        })
    }
    pub fn measure_search(&self, text: &str, budget: search::Budget) -> search::Measurement {
        search::measure(&self.metadata, text, self.subwords, budget, |f, work| {
            self.scores(f, work)
        })
    }
    pub fn weights(&self) -> usize {
        self.input.values.len() + self.output.values.len()
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() - std::mem::size_of::<Metadata>()
            + self.metadata.retained_bytes()
            + self.input.retained_bytes()
            + self.output.retained_bytes()
            + self.bias.capacity() * 4
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let header = serde_json::to_vec(&Header {
            metadata: Wire::from(self.metadata.clone()),
            dimensions: self.dimensions,
            subwords: self.subwords,
            tanh: self.tanh,
        })
        .map_err(|e| Error::Storage(e.to_string()))?;
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&header);
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
            return Err(Error::Invalid("factorized length"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if checksum(body).to_le_bytes() != digest {
            return Err(Error::Invalid("factorized checksum"));
        }
        let mut cursor = Cursor::new(body);
        if &take::<8>(&mut cursor)? != MAGIC {
            return Err(Error::Invalid("factorized magic"));
        }
        let size = u32::from_le_bytes(take(&mut cursor)?) as usize;
        let header = body
            .get(12..12usize.checked_add(size).ok_or(Error::Budget)?)
            .ok_or(Error::Invalid("factorized header length"))?;
        let header: Header =
            serde_json::from_slice(header).map_err(|e| Error::Storage(e.to_string()))?;
        validate_header(&header.metadata)?;
        if header.metadata.buckets > 8192
            || !(8..=32).contains(&header.dimensions)
            || !header.metadata.packed.is_empty()
        {
            return Err(Error::Invalid("factorized header"));
        }
        cursor.set_position((12 + size) as u64);
        let metadata = metadata(header.metadata);
        let classes = metadata.intents.len() + metadata.tags.len();
        if body.len()
            != 12 + size + (metadata.buckets + classes) * (header.dimensions + 4) + classes * 4
        {
            return Err(Error::Invalid("factorized payload length"));
        }
        let input = Quantized::read(&mut cursor, metadata.buckets, header.dimensions)?;
        let output = Quantized::read(&mut cursor, classes, header.dimensions)?;
        let mut bias = Vec::with_capacity(classes);
        for _ in 0..classes {
            let value = f32::from_le_bytes(take(&mut cursor)?);
            if !value.is_finite() || value.abs() > 16.0 {
                return Err(Error::Invalid("factorized bias"));
            }
            bias.push(value);
        }
        if cursor.position() != body.len() as u64 {
            return Err(Error::Invalid("factorized trailing bytes"));
        }
        Ok(Self {
            metadata,
            dimensions: header.dimensions,
            subwords: header.subwords,
            tanh: header.tanh,
            input,
            output,
            bias,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn data() -> Vec<Example> {
        (0..16)
            .flat_map(|i| {
                ["send", "cancel"].map(move |intent| Example {
                    text: format!("{intent} letter to person-{i}"),
                    intent: intent.into(),
                    tags: ["O", "O", "O", "B:person"].map(str::to_owned).into(),
                })
            })
            .collect()
    }
    fn budget() -> Budget {
        Budget {
            seed: 47,
            buckets: 256,
            dimensions: 16,
            epochs: 32,
            learning_rate: 0.3,
            subwords: false,
            tanh: false,
            balanced: false,
            minimum_margin: 1,
        }
    }
    #[test]
    fn balancing_is_neutral_for_equal_class_frequencies_but_changes_skewed_training() {
        let b = Budget {
            balanced: true,
            ..budget()
        };
        let equal = Model::learn(&data(), budget(), &mut |_| Ok(()))
            .unwrap()
            .to_bytes()
            .unwrap();
        assert_eq!(
            equal,
            Model::learn(&data(), b, &mut |_| Ok(()))
                .unwrap()
                .to_bytes()
                .unwrap()
        );
        let skew = data()
            .into_iter()
            .enumerate()
            .filter(|(i, e)| e.intent == "send" || i % 8 == 1)
            .map(|(_, e)| e)
            .collect::<Vec<_>>();
        let ordinary = Model::learn(&skew, budget(), &mut |_| Ok(()))
            .unwrap()
            .to_bytes()
            .unwrap();
        assert_ne!(
            ordinary,
            Model::learn(&skew, b, &mut |_| Ok(()))
                .unwrap()
                .to_bytes()
                .unwrap()
        );
    }
    #[test]
    fn nonlinear_gradient_matches_finite_difference_and_flag_survives_roundtrip() {
        let mut learner = Learner {
            input: vec![0.2; 16],
            output: [vec![0.3; 8], vec![-0.2; 8]].concat(),
            bias: vec![0.0; 2],
            dimensions: 8,
            tanh: true,
        };
        let loss = |l: &Learner| {
            let h = l.hidden(&[0, 1]);
            let scores = (0..2)
                .map(|c| {
                    l.bias[c]
                        + h.iter()
                            .zip(&l.output[c * 8..(c + 1) * 8])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let max = scores[0].max(scores[1]);
            max + scores.iter().map(|s| (*s - max).exp()).sum::<f32>().ln() - scores[1]
        };
        let delta = 0.001;
        let initial = learner.input[0];
        learner.input[0] = initial + delta;
        let high = loss(&learner);
        learner.input[0] = initial - delta;
        let low = loss(&learner);
        learner.input[0] = initial;
        let expected = (high - low) / (2.0 * delta);
        learner.update(&[0, 1], 1, 0..2, 0.001);
        let actual = (initial - learner.input[0]) / 0.001;
        assert!((actual - expected).abs() < 0.001, "{actual} != {expected}");
        let b = Budget {
            tanh: true,
            ..budget()
        };
        let model = Model::learn(&data(), b, &mut |_| Ok(())).unwrap();
        let bytes = model.to_bytes().unwrap();
        let child = Model::from_bytes(&bytes).unwrap();
        assert!(child.tanh);
        assert_eq!(bytes, child.to_bytes().unwrap());
        assert!(Model::refine(&data(), budget(), &child, &mut |_| Ok(())).is_err());
        let linear = Model::learn(&data(), budget(), &mut |_| Ok(()))
            .unwrap()
            .to_bytes()
            .unwrap();
        let size = u32::from_le_bytes(linear[8..12].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&linear[12..12 + size]).unwrap();
        assert!(header.get("tanh").is_none());
    }
    #[test]
    fn learned_shared_features_roundtrip_and_do_not_change_old_codecs() {
        let model = Model::learn(&data(), budget(), &mut |_| Ok(())).unwrap();
        let bytes = model.to_bytes().unwrap();
        let child = Model::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, child.to_bytes().unwrap());
        assert_eq!(model.retained_bytes(), child.retained_bytes());
        let mut correct = 0;
        for e in data() {
            let a = model
                .predict(&e.text)
                .map(|p| p.output)
                .map_err(|e| e.to_string());
            let b = child
                .predict(&e.text)
                .map(|p| p.output)
                .map_err(|e| e.to_string());
            assert_eq!(a, b);
            if let Ok(p) = a {
                let p = serde_json::to_value(p).unwrap();
                correct += usize::from(p[1] == e.intent && p[2][1][0] == "person");
            }
        }
        assert_eq!(correct, 32);
        assert!(super::super::Model::from_bytes(&bytes).is_err());
        assert!(compact::CompactModel::from_bytes(&bytes).is_err());
        assert_eq!(
            bytes,
            Model::learn(&data(), budget(), &mut |_| Ok(()))
                .unwrap()
                .to_bytes()
                .unwrap()
        );
    }
    #[test]
    fn refinement_inherits_features_remaps_new_roles_and_refuses_incompatible_schema() {
        let parent = Model::learn(&data(), budget(), &mut |_| Ok(())).unwrap();
        let bytes = parent.to_bytes().unwrap();
        let mut examples = data();
        examples.extend((0..16).map(|i| Example {
            text: format!("archive letter priority level-{i}"),
            intent: "archive".into(),
            tags: ["O", "O", "O", "B:importance"].map(str::to_owned).into(),
        }));
        let child = Model::refine(&examples, budget(), &parent, &mut |_| Ok(())).unwrap();
        assert_eq!(child.metadata.intents, ["archive", "cancel", "send"]);
        let child = Model::from_bytes(&child.to_bytes().unwrap()).unwrap();
        for e in &examples {
            let p = serde_json::to_value(child.predict(&e.text).unwrap().output).unwrap();
            assert_eq!(p[1], e.intent);
            assert_eq!(
                p[2][1][0],
                if e.intent == "archive" {
                    "importance"
                } else {
                    "person"
                }
            );
        }
        assert_eq!(bytes, parent.to_bytes().unwrap());
        for b in [
            Budget {
                seed: 48,
                ..budget()
            },
            Budget {
                dimensions: 8,
                ..budget()
            },
            Budget {
                subwords: true,
                ..budget()
            },
        ] {
            assert!(Model::refine(&examples, b, &parent, &mut |_| Ok(())).is_err());
        }
    }
    #[test]
    fn bad_scales_bias_dimensions_truncation_and_cancellation_are_rejected() {
        let model = Model::learn(&data(), budget(), &mut |_| Ok(())).unwrap();
        let bytes = model.to_bytes().unwrap();
        for length in [0, 19, bytes.len() - 1] {
            assert!(Model::from_bytes(&bytes[..length]).is_err());
        }
        let start = 12 + u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        for (offset, value) in [
            (start, f32::NAN),
            (start, -1.0),
            (bytes.len() - 12, f32::INFINITY),
        ] {
            let mut forged = bytes[..bytes.len() - 8].to_vec();
            forged[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            forged.extend_from_slice(&checksum(&forged).to_le_bytes());
            assert!(Model::from_bytes(&forged).is_err());
        }
        let mut b = budget();
        b.dimensions = 33;
        assert!(Model::learn(&data(), b, &mut |_| Ok(())).is_err());
        let mut header: serde_json::Value = serde_json::from_slice(&bytes[12..start]).unwrap();
        header["dimensions"] = serde_json::json!(33);
        let header = serde_json::to_vec(&header).unwrap();
        let mut forged = MAGIC.to_vec();
        forged.extend_from_slice(&(header.len() as u32).to_le_bytes());
        forged.extend_from_slice(&header);
        forged.extend_from_slice(&bytes[start..bytes.len() - 8]);
        forged.extend_from_slice(&checksum(&forged).to_le_bytes());
        assert!(Model::from_bytes(&forged).is_err());
        assert!(matches!(
            Model::learn(&data(), budget(), &mut |_| Err(Error::Budget)),
            Err(Error::Budget)
        ));
    }
}
