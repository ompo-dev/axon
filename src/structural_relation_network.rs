//! Supervised paired-text relations. Features are observations, never truth rules.
//! Hypothesis-only control: https://aclanthology.org/N18-2017/.
use super::dependency::Prediction as Graph;
use super::factorized::{Learner, Quantized, random, take};
use super::{Error, MAX_BYTES, hash, unique};
use crate::live_store::checksum;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
};
pub const CONTEXT: &str = "axon/relation-network-v1";
const MAGIC: &[u8; 8] = b"AXRL0001";
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Hypothesis,
    Pair,
    Graph,
    BoundGraph,
}
impl Mode {
    pub fn uses_syntax(self) -> bool {
        matches!(self, Self::Graph | Self::BoundGraph)
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub seed: u64,
    pub buckets: usize,
    pub dimensions: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub mode: Mode,
    pub minimum_margin: f32,
}
impl Budget {
    fn validate(self) -> Result<(), Error> {
        if self.seed == 0
            || !self.buckets.is_power_of_two()
            || !(256..=4096).contains(&self.buckets)
            || !(8..=32).contains(&self.dimensions)
            || !(1..=32).contains(&self.epochs)
            || !self.learning_rate.is_finite()
            || !(0.001..=1.0).contains(&self.learning_rate)
            || !self.minimum_margin.is_finite()
            || !(0.0..=10.0).contains(&self.minimum_margin)
        {
            return Err(Error::Invalid("relation network budget"));
        }
        Ok(())
    }
    fn f(self, kind: u8, parts: &[&str]) -> usize {
        hash(self.seed, kind, parts, self.buckets)
    }
}
fn words(text: &str) -> Result<Vec<String>, Error> {
    if text.is_empty()
        || text.len() > 16384
        || text.chars().any(|c| c.is_control() && !c.is_whitespace())
    {
        return Err(Error::Invalid("relation text"));
    }
    let words = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>();
    if words.is_empty() || words.len() > 256 || words.iter().any(|w| w.len() > 1024) {
        return Err(Error::Invalid("relation words"));
    }
    Ok(words)
}
fn lexical(b: Budget, kind: u8, words: &[String], f: &mut Vec<usize>) {
    for word in words {
        f.push(b.f(kind, &[word]));
        let chars = word.chars().collect::<Vec<_>>();
        if chars.len() > 3 {
            f.push(b.f(kind + 1, &[&chars[..3].iter().collect::<String>()]));
            f.push(b.f(
                kind + 2,
                &[&chars[chars.len() - 3..].iter().collect::<String>()],
            ));
        }
    }
    for pair in words.windows(2) {
        f.push(b.f(kind + 3, &[&pair[0], &pair[1]]));
    }
}
fn edges(graph: &Graph) -> BTreeSet<(String, String, String)> {
    graph
        .words
        .iter()
        .map(|w| {
            (
                w.deprel.clone(),
                if w.head == 0 {
                    "<root>".into()
                } else {
                    graph.words[w.head - 1].form.to_lowercase()
                },
                w.form.to_lowercase(),
            )
        })
        .collect()
}
type Frame = (String, String, String, String, String);
fn binding_frames(a: &Graph, c: &Graph) -> Result<(BTreeSet<Frame>, BTreeSet<Frame>), Error> {
    let mut names = BTreeMap::new();
    for w in a.words.iter().chain(&c.words) {
        if matches!(w.upos.as_str(), "NOUN" | "PROPN") {
            let n = names.len();
            names.entry(w.form.to_lowercase()).or_insert(n);
        }
    }
    let build = |g: &Graph| -> Result<BTreeSet<Frame>, Error> {
        let mut frames = BTreeSet::new();
        let mut work = 0;
        for (index, head) in g.words.iter().enumerate() {
            let mut children = g
                .words
                .iter()
                .filter(|w| w.head == index + 1)
                .map(|w| {
                    let form = w.form.to_lowercase();
                    let slot = if matches!(w.upos.as_str(), "NOUN" | "PROPN") {
                        format!("bound:{}", names[&form])
                    } else {
                        format!("literal:{form}")
                    };
                    (w.deprel.clone(), slot)
                })
                .collect::<Vec<_>>();
            children.sort();
            for i in 0..children.len() {
                for j in i + 1..children.len() {
                    work += 1;
                    if work > 4096 {
                        return Err(Error::Budget);
                    }
                    frames.insert((
                        head.form.to_lowercase(),
                        children[i].0.clone(),
                        children[i].1.clone(),
                        children[j].0.clone(),
                        children[j].1.clone(),
                    ));
                }
            }
        }
        Ok(frames)
    };
    Ok((build(a)?, build(c)?))
}
/// Only raw text and predicted graphs enter. No target, reviewer or split metadata.
pub fn features(
    b: Budget,
    premise: &str,
    hypothesis: &str,
    graphs: Option<(&Graph, &Graph)>,
) -> Result<Vec<usize>, Error> {
    b.validate()?;
    let h = words(hypothesis)?;
    let mut f = vec![b.f(0, &[])];
    lexical(b, 1, &h, &mut f);
    if b.mode == Mode::Hypothesis {
        unique(&mut f);
        return Ok(f.into_boxed_slice().into_vec());
    }
    let p = words(premise)?;
    lexical(b, 10, &p, &mut f);
    let ps = p.iter().collect::<BTreeSet<_>>();
    let hs = h.iter().collect::<BTreeSet<_>>();
    let removed = ps.difference(&hs).copied().collect::<Vec<_>>();
    let added = hs.difference(&ps).copied().collect::<Vec<_>>();
    for w in ps.intersection(&hs) {
        f.push(b.f(20, &[w]));
    }
    for w in &removed {
        f.push(b.f(21, &[w]));
    }
    for w in &added {
        f.push(b.f(22, &[w]));
    }
    if removed.len() * added.len() <= 128 {
        for a in &removed {
            for c in &added {
                f.push(b.f(23, &[a, c]));
            }
        }
    }
    for (kind, n) in [
        (24, removed.len()),
        (25, added.len()),
        (26, ps.intersection(&hs).count()),
        (27, p.len().abs_diff(h.len())),
    ] {
        f.push(b.f(kind, &[&n.min(16).to_string()]));
    }
    f.push(b.f(
        28,
        &[if p.len() < h.len() {
            "shorter"
        } else if p.len() == h.len() {
            "equal"
        } else {
            "longer"
        }],
    ));
    for (kind, v) in [(30, &p), (31, &h)] {
        for pair in v.windows(2) {
            let other = if kind == 30 { &h } else { &p };
            if !other.windows(2).any(|x| x == pair) {
                f.push(b.f(kind, &[&pair[0], &pair[1]]));
            }
        }
    }
    if b.mode.uses_syntax() {
        if let Some((a, c)) = graphs {
            super::dependency::validate_words(&a.words)?;
            super::dependency::validate_words(&c.words)?;
            let ae = edges(a);
            let ce = edges(c);
            for (kind, values) in [
                (40, ae.intersection(&ce).collect::<Vec<_>>()),
                (41, ae.difference(&ce).collect()),
                (42, ce.difference(&ae).collect()),
            ] {
                for (r, head, word) in values {
                    f.push(b.f(kind, &[r, head, word]));
                    f.push(b.f(kind + 3, &[r, word]));
                }
            }
            let mut matches = 0;
            for aw in &a.words {
                for cw in &c.words {
                    if aw.form.to_lowercase() == cw.form.to_lowercase() {
                        matches += 1;
                        if matches > 4096 {
                            return Err(Error::Budget);
                        }
                        f.push(b.f(50, &[&aw.deprel, &cw.deprel]));
                        f.push(b.f(51, &[&aw.form.to_lowercase(), &aw.deprel, &cw.deprel]));
                        f.push(b.f(52, &[&aw.upos, &cw.upos]));
                    }
                }
            }
            f.push(b.f(
                53,
                &[
                    &ae.difference(&ce).count().min(16).to_string(),
                    &ce.difference(&ae).count().min(16).to_string(),
                ],
            ));
            if b.mode == Mode::BoundGraph {
                // Equal surface forms share lexical slots, not proven entities.
                // Pairing edges under one predicate preserves role combinations.
                let (af, cf) = binding_frames(a, c)?;
                for (kind, values) in [
                    (60, af.intersection(&cf).collect::<Vec<_>>()),
                    (61, af.difference(&cf).collect()),
                    (62, cf.difference(&af).collect()),
                ] {
                    for (head, r1, s1, r2, s2) in values {
                        f.push(b.f(kind, &[head, r1, s1, r2, s2]));
                        f.push(b.f(kind + 3, &[r1, s1, r2, s2]));
                    }
                }
                f.push(b.f(
                    66,
                    &[
                        &af.difference(&cf).count().min(16).to_string(),
                        &cf.difference(&af).count().min(16).to_string(),
                    ],
                ));
            }
            f.push(b.f(
                54,
                &[
                    &a.ambiguous_expansions.min(4).to_string(),
                    &c.ambiguous_expansions.min(4).to_string(),
                ],
            ));
        } else {
            f.push(b.f(55, &["syntax-unavailable"]));
        }
    }
    unique(&mut f);
    if f.len() > 4096 {
        return Err(Error::Budget);
    }
    Ok(f.into_boxed_slice().into_vec())
}

#[cfg(test)]
mod tests {
    use super::super::dependency::Word;
    use super::*;
    fn budget(mode: Mode) -> Budget {
        Budget {
            seed: 47,
            buckets: 256,
            dimensions: 8,
            epochs: 32,
            learning_rate: 0.3,
            mode,
            minimum_margin: 0.0,
        }
    }
    fn graph(reverse: bool) -> Graph {
        Graph {
            words: vec![
                Word {
                    form: "a".into(),
                    upos: "NOUN".into(),
                    head: 2,
                    deprel: if reverse { "obj" } else { "nsubj" }.into(),
                },
                Word {
                    form: "x".into(),
                    upos: "VERB".into(),
                    head: 0,
                    deprel: "root".into(),
                },
                Word {
                    form: "b".into(),
                    upos: "NOUN".into(),
                    head: 2,
                    deprel: if reverse { "nsubj" } else { "obj" }.into(),
                },
            ],
            ambiguous_expansions: 0,
            minimum_margin: 1.0,
            work: 0,
        }
    }
    #[test]
    fn directional_graph_changes_are_observations_not_embedded_truth_rules() {
        let a = graph(false);
        let b = graph(true);
        let spec = budget(Mode::Graph);
        assert_ne!(
            features(spec, "a x b", "a x b", Some((&a, &a))).unwrap(),
            features(spec, "a x b", "a x b", Some((&a, &b))).unwrap()
        );
        let hypothesis = budget(Mode::Hypothesis);
        assert_eq!(
            features(hypothesis, "different premise", "h only", None).unwrap(),
            features(hypothesis, "", "h only", None).unwrap()
        );
        assert_ne!(
            features(budget(Mode::Pair), "a x b", "b x a", None).unwrap(),
            features(budget(Mode::Pair), "b x a", "a x b", None).unwrap()
        );
        let mut bad = graph(false);
        bad.words[0].head = 999;
        assert!(features(spec, "a x b", "a x b", Some((&bad, &a))).is_err());
    }
    #[test]
    fn predicate_binding_quotients_names_without_assuming_coreference_or_truth() {
        let a = graph(false);
        let c = graph(true);
        let mut renamed = graph(false);
        renamed.words[0].form = "newname".into();
        renamed.words[2].form = "othername".into();
        let original = binding_frames(&a, &a).unwrap();
        let changed = binding_frames(&renamed, &renamed).unwrap();
        assert_eq!(original, changed);
        let reversal = binding_frames(&a, &c).unwrap();
        assert_ne!(reversal.0, reversal.1);
        let b = budget(Mode::BoundGraph);
        assert_ne!(
            features(b, "a x b", "a x b", Some((&a, &a))).unwrap(),
            features(b, "a x b", "a x b", Some((&a, &c))).unwrap()
        );
    }
    #[test]
    fn local_learning_quantization_and_codec_preserve_predictions_and_allocation() {
        let data = (0..192)
            .map(|i| (vec![0, 1 + i % 3], (i % 3) as u8))
            .collect::<Vec<_>>();
        let model = Model::learn(&data, budget(Mode::Pair), &mut |_| Ok(())).unwrap();
        let bytes = model.to_bytes().unwrap();
        let restored = Model::from_bytes(&bytes).unwrap();
        assert_eq!(model.retained_bytes(), restored.retained_bytes());
        assert_eq!(bytes, restored.to_bytes().unwrap());
        for (f, label) in &data[..3] {
            let p = model.predict(f).unwrap();
            assert_eq!(p.label, Some(*label));
            assert_eq!(
                serde_json::to_value(p).unwrap(),
                serde_json::to_value(restored.predict(f).unwrap()).unwrap()
            );
        }
        assert!(model.predict(&[999]).is_err());
        assert!(model.predict(&[1, 1]).is_err());
        assert!(model.predict(&[]).is_err());
        assert!(
            Model::learn(
                &data,
                Budget {
                    seed: 0,
                    ..budget(Mode::Pair)
                },
                &mut |_| Ok(())
            )
            .is_err()
        );
        let mut corrupt = bytes.clone();
        corrupt[20] ^= 1;
        assert!(Model::from_bytes(&corrupt).is_err());
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut corrupt = bytes.clone();
        corrupt[12 + n..16 + n].copy_from_slice(&f32::NAN.to_le_bytes());
        let end = corrupt.len() - 8;
        let sum = checksum(&corrupt[..end]);
        corrupt[end..].copy_from_slice(&sum.to_le_bytes());
        assert!(Model::from_bytes(&corrupt).is_err());
    }
}
#[derive(Clone, Debug)]
pub struct Model {
    budget: Budget,
    input: Quantized,
    output: Quantized,
    bias: Vec<f32>,
}
#[derive(Debug, Serialize)]
pub struct Event {
    pub epoch: usize,
    pub examples: usize,
    pub errors: usize,
}
#[derive(Debug, Serialize)]
pub struct Prediction {
    pub label: Option<u8>,
    pub margin: f32,
    pub scores: [f32; 3],
    pub features: usize,
    pub work: usize,
}
impl Model {
    pub fn budget(&self) -> Budget {
        self.budget
    }
    pub fn learn(
        data: &[(Vec<usize>, u8)],
        b: Budget,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        b.validate()?;
        if !(4..=10000).contains(&data.len())
            || data.iter().any(|(f, y)| {
                *y > 2
                    || f.is_empty()
                    || f.len() > 4096
                    || f.iter().any(|i| *i >= b.buckets)
                    || f.windows(2).any(|p| p[0] >= p[1])
            })
            || !(0..3).all(|y| data.iter().any(|(_, label)| *label == y))
        {
            return Err(Error::Invalid("relation supervision"));
        }
        let d = b.dimensions;
        let mut state = b.seed;
        let mut l = Learner {
            input: (0..b.buckets * d)
                .map(|_| {
                    ((random(&mut state) >> 40) as f32 / (1u32 << 24) as f32 - 0.5)
                        / (d as f32).sqrt()
                })
                .collect(),
            output: vec![0.0; 3 * d],
            bias: vec![0.0; 3],
            dimensions: d,
            tanh: true,
        };
        let mut order = (0..data.len()).collect::<Vec<_>>();
        for epoch in 0..b.epochs {
            for i in (1..order.len()).rev() {
                order.swap(i, random(&mut state) as usize % (i + 1));
            }
            let mut event = Event {
                epoch: epoch + 1,
                examples: data.len(),
                errors: 0,
            };
            for (step, index) in order.iter().enumerate() {
                let rate = b.learning_rate
                    * (1.0 - (epoch * data.len() + step) as f32 / (b.epochs * data.len()) as f32)
                        .max(0.1);
                event.errors +=
                    usize::from(l.update(&data[*index].0, data[*index].1 as usize, 0..3, rate));
            }
            emit(&event)?;
        }
        let model = Self {
            budget: b,
            input: Quantized::encode(&l.input, d),
            output: Quantized::encode(&l.output, d),
            bias: l.bias,
        };
        model.to_bytes()?;
        Ok(model)
    }
    pub fn predict(&self, f: &[usize]) -> Result<Prediction, Error> {
        if f.is_empty()
            || f.len() > 4096
            || f.iter().any(|i| *i >= self.budget.buckets)
            || f.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(Error::Invalid("relation features"));
        }
        let d = self.budget.dimensions;
        let mut h = vec![0.0; d];
        let norm = (f.len() as f32).sqrt();
        for i in f {
            for (j, h) in h.iter_mut().enumerate() {
                *h += f32::from(self.input.values[i * d + j]) * self.input.scales[*i] / norm;
            }
        }
        for h in &mut h {
            *h = h.tanh();
        }
        let scores = std::array::from_fn(|c| {
            self.bias[c]
                + h.iter()
                    .zip(&self.output.values[c * d..(c + 1) * d])
                    .map(|(h, w)| h * f32::from(*w) * self.output.scales[c])
                    .sum::<f32>()
        });
        let mut order = [0, 1, 2];
        order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then_with(|| a.cmp(b)));
        let margin = scores[order[0]] - scores[order[1]];
        Ok(Prediction {
            label: (margin >= self.budget.minimum_margin).then_some(order[0] as u8),
            margin,
            scores,
            features: f.len(),
            work: (f.len() + 3) * d,
        })
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.input.retained_bytes()
            + self.output.retained_bytes()
            + self.bias.capacity() * 4
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let header = serde_json::to_vec(&self.budget).map_err(|e| Error::Storage(e.to_string()))?;
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend(header);
        self.input.append(&mut bytes);
        self.output.append(&mut bytes);
        for x in &self.bias {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if !(20..=MAX_BYTES).contains(&bytes.len()) {
            return Err(Error::Invalid("relation length"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if checksum(body).to_le_bytes() != digest {
            return Err(Error::Invalid("relation checksum"));
        }
        let mut r = Cursor::new(body);
        if &take::<8>(&mut r)? != MAGIC {
            return Err(Error::Invalid("relation magic"));
        }
        let n = u32::from_le_bytes(take(&mut r)?) as usize;
        let b: Budget = serde_json::from_slice(
            body.get(12..12usize.checked_add(n).ok_or(Error::Budget)?)
                .ok_or(Error::Invalid("relation header"))?,
        )
        .map_err(|e| Error::Storage(e.to_string()))?;
        b.validate()?;
        if body.len() != 12 + n + (b.buckets + 3) * (b.dimensions + 4) + 12 {
            return Err(Error::Invalid("relation payload"));
        }
        r.set_position((12 + n) as u64);
        let input = Quantized::read(&mut r, b.buckets, b.dimensions)?;
        let output = Quantized::read(&mut r, 3, b.dimensions)?;
        let mut bias = Vec::with_capacity(3);
        for _ in 0..3 {
            let x = f32::from_le_bytes(take(&mut r)?);
            if !x.is_finite() || x.abs() > 16.0 {
                return Err(Error::Invalid("relation bias"));
            }
            bias.push(x);
        }
        Ok(Self {
            budget: b,
            input,
            output,
            bias,
        })
    }
}
