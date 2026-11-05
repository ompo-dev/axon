//! Learned segment features with exact top-two bounded segmentation for a selected intent.
//! Averaged perceptron, not a probabilistic CRF or a general semantic certificate.
//! Segment representation reference: Sarawagi/Cohen (NeurIPS 2004).
use super::*;
use crate::live_store::checksum;
const MAGIC: &[u8; 8] = b"AXSQ0008";
pub const CONTEXT: &str = "axon/segment-sequence-v1";

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub learner: super::Budget,
    pub max_span: usize,
    pub typed_roles: bool,
    pub max_work: usize,
    #[serde(default)]
    pub span_replacements: f32,
}
impl Budget {
    fn validate(self) -> Result<(), Error> {
        self.learner.validate_limit(compact::MAX_WEIGHTS)?;
        if !(1..=16).contains(&self.max_span)
            || !(1..=50_000_000).contains(&self.max_work)
            || self.learner.shared_roles
            || !self.span_replacements.is_finite()
            || !(0.0..=1.0).contains(&self.span_replacements)
        {
            return Err(Error::Invalid("segment budget"));
        }
        Ok(())
    }
}
pub struct TrainingEvent {
    pub base: super::Event,
    pub augmented_sentences: usize,
    pub replaced_spans: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Span {
    start: usize,
    end: usize,
    tag: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Path {
    spans: Vec<Span>,
    score: i64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    subwords: bool,
    max_span: usize,
    typed_roles: bool,
    max_work: usize,
    roles: Vec<Vec<usize>>,
}
#[derive(Clone, Debug)]
pub struct Model {
    base: compact::CompactModel,
    subwords: bool,
    max_span: usize,
    typed_roles: bool,
    max_work: usize,
    roles: Box<[Box<[usize]>]>,
}
fn gold_spans(tags: &[String], labels: &[String]) -> Vec<Span> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < tags.len() {
        let start = i;
        let tag = labels.iter().position(|t| t == &tags[i]).unwrap();
        i += 1;
        if tags[start] != "O" {
            let continuation = format!("I:{}", role(&tags[start]).unwrap());
            while tags.get(i) == Some(&continuation) {
                i += 1;
            }
        }
        out.push(Span { start, end: i, tag });
    }
    out
}
fn recombine(
    words: &[String],
    spans: &[Span],
    pool: &[Vec<Vec<String>>],
    probability: f32,
    state: &mut u64,
) -> (Vec<String>, Vec<Span>, usize) {
    if probability == 0.0 {
        return (words.to_vec(), spans.to_vec(), 0);
    }
    let mut out = Vec::new();
    let mut labels = Vec::new();
    let mut replaced = 0;
    for s in spans {
        let mut value = &words[s.start..s.end];
        if s.tag != 0
            && !pool[s.tag].is_empty()
            && (factorized::random(state) >> 40) as f32 / ((1u32 << 24) as f32) < probability
        {
            let candidate = &pool[s.tag][factorized::random(state) as usize % pool[s.tag].len()];
            if candidate.as_slice() != value
                && out.len() + candidate.len() + words.len() - s.end <= 64
            {
                value = candidate;
                replaced += 1;
            }
        }
        let start = out.len();
        out.extend_from_slice(value);
        labels.push(Span {
            start,
            end: out.len(),
            tag: s.tag,
        });
    }
    (out, labels, replaced)
}
fn features(
    words: &[String],
    start: usize,
    end: usize,
    intent: &str,
    b: super::Budget,
) -> Vec<usize> {
    let before = start.checked_sub(1).map_or("<start>", |i| &words[i]);
    let after = words.get(end).map_or("<end>", String::as_str);
    let first = &words[start];
    let last = &words[end - 1];
    let length = (end - start).to_string();
    let mut f = Vec::new();
    for (kind, parts) in [
        (70, vec![]),
        (71, vec![length.as_str()]),
        (72, vec![before]),
        (73, vec![after]),
        (74, vec![first.as_str()]),
        (75, vec![last.as_str()]),
        (76, vec![before, first]),
        (77, vec![last, after]),
        (78, vec![before, after]),
        (79, vec![intent]),
        (80, vec![intent, before]),
        (81, vec![intent, after]),
        (82, vec![intent, length.as_str()]),
    ] {
        f.push(hash(b.seed, kind, &parts, b.buckets));
    }
    for word in &words[start..end] {
        f.push(hash(b.seed, 83, &[word], b.buckets));
    }
    for (kind, word) in [(84, first), (88, last)] {
        let chars = word.chars().collect::<Vec<_>>();
        for len in 1..=3.min(chars.len()) {
            f.push(hash(
                b.seed,
                kind + len as u8,
                &[&chars[chars.len() - len..].iter().collect::<String>()],
                b.buckets,
            ));
        }
        if b.subwords {
            subwords(word, b, 92, &mut f);
        }
    }
    unique(&mut f);
    f
}
fn keep_two(paths: &mut Vec<Path>, candidate: Path) {
    if paths.iter().any(|p| p.spans == candidate.spans) {
        return;
    }
    paths.push(candidate);
    paths.sort_by(|a, b| b.score.cmp(&a.score).then(a.spans.cmp(&b.spans)));
    paths.truncate(2);
}
fn decode(
    words: &[String],
    intent: &str,
    roles: &[usize],
    budget: Budget,
    n: usize,
    mut scores: impl FnMut(&[usize], &mut usize) -> Vec<i32>,
    work: &mut usize,
) -> Result<Vec<Path>, Error> {
    let mut chart = vec![Vec::<Path>::new(); words.len() + 1];
    chart[0].push(Path {
        spans: vec![],
        score: 0,
    });
    for end in 1..=words.len() {
        for len in 1..=budget.max_span.min(end) {
            let start = end - len;
            let f = features(words, start, end, intent, budget.learner);
            let values = scores(&f, work);
            if *work > budget.max_work {
                return Err(Error::Budget);
            }
            let previous = chart[start].clone();
            for tag in roles.iter().filter(|tag| **tag != 0 || len == 1) {
                for old in &previous {
                    *work += 1;
                    if *work > budget.max_work {
                        return Err(Error::Budget);
                    }
                    let score = old.score + i64::from(values[n + tag]);
                    if chart[end].len() == 2 && score < chart[end][1].score {
                        continue;
                    }
                    let mut spans = old.spans.clone();
                    spans.push(Span {
                        start,
                        end,
                        tag: *tag,
                    });
                    keep_two(&mut chart[end], Path { spans, score });
                }
            }
        }
    }
    Ok(chart.pop().unwrap())
}
impl Model {
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        emit: &mut impl FnMut(&TrainingEvent) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        budget.validate()?;
        if !(4..=10000).contains(&examples.len()) {
            return Err(Error::Invalid("segment training count"));
        }
        let (intents, tags) = class_labels(examples, None)?;
        let n = intents.len();
        let classes = n + tags.len();
        let mut roles = vec![vec![0]; n];
        let mut prepared = Vec::new();
        for e in examples {
            let words = tokens(&e.text)?;
            let intent = intents.binary_search(&e.intent).unwrap();
            let spans = gold_spans(&e.tags, &tags);
            if spans.iter().any(|s| s.end - s.start > budget.max_span) {
                return Err(Error::Invalid("training span exceeds declared bound"));
            }
            roles[intent].extend(spans.iter().map(|s| s.tag));
            prepared.push((words, intent, spans));
        }
        if !budget.typed_roles {
            let all = tags
                .iter()
                .enumerate()
                .filter_map(|(i, t)| (t == "O" || t.starts_with("B:")).then_some(i))
                .collect::<Vec<_>>();
            roles.fill(all);
        }
        for r in &mut roles {
            unique(r);
        }
        let mut pool = vec![Vec::new(); tags.len()];
        if budget.span_replacements != 0.0 {
            for (words, _, spans) in &prepared {
                for s in spans {
                    if s.tag != 0 {
                        pool[s.tag].push(words[s.start..s.end].to_vec());
                    }
                }
            }
            for values in &mut pool {
                values.sort();
                values.dedup();
            }
        }
        let size = budget.learner.buckets * classes;
        let mut learner = Learner {
            weights: vec![0; size],
            totals: vec![0; size],
            last: vec![0; size],
            clock: 0,
            classes,
        };
        let mut order = (0..examples.len()).collect::<Vec<_>>();
        let mut rng = budget.learner.seed;
        for epoch in 0..budget.learner.epochs {
            for i in (1..order.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                order.swap(i, rng as usize % (i + 1));
            }
            let mut event = TrainingEvent {
                base: Event {
                    epoch: epoch + 1,
                    sentences: examples.len(),
                    intent_errors: 0,
                    tag_errors: 0,
                },
                augmented_sentences: 0,
                replaced_spans: 0,
            };
            for index in &order {
                let (words, intent, gold) = &prepared[*index];
                let mut train_one = |words: &[String], gold: &[Span]| -> Result<(), Error> {
                    let f = intent_features(words, budget.learner);
                    let (guess, _) = best(&learner.scores(&f), 0, n, None);
                    event.base.intent_errors += usize::from(guess != *intent);
                    learner.update(&f, *intent, guess);
                    let paths = decode(
                        words,
                        &intents[*intent],
                        &roles[*intent],
                        budget,
                        n,
                        |f, work| {
                            let mut values = vec![0; classes];
                            for feature in f {
                                for tag in &roles[*intent] {
                                    values[n + tag] += learner.weights[feature * classes + n + tag];
                                    *work += 1;
                                }
                            }
                            values
                        },
                        &mut 0,
                    )?;
                    let predicted = &paths[0].spans;
                    event.base.tag_errors += usize::from(predicted != gold);
                    learner.clock += 1;
                    if predicted != gold {
                        for (path, delta) in [(gold, 8), (predicted.as_slice(), -8)] {
                            for s in path {
                                for f in features(
                                    words,
                                    s.start,
                                    s.end,
                                    &intents[*intent],
                                    budget.learner,
                                ) {
                                    learner.add(f, n + s.tag, delta);
                                }
                            }
                        }
                    }
                    Ok(())
                };
                train_one(words, gold)?;
                if budget.span_replacements != 0.0 {
                    let (augmented, labels, replaced) =
                        recombine(words, gold, &pool, budget.span_replacements, &mut rng);
                    if replaced != 0 {
                        train_one(&augmented, &labels)?;
                        event.augmented_sentences += 1;
                        event.replaced_spans += replaced;
                    }
                }
            }
            emit(&event)?;
        }
        let base = compact::CompactModel::from_legacy(learner.finish(
            intents,
            tags,
            budget.learner,
            false,
        ))?;
        let model = Self {
            base,
            subwords: budget.learner.subwords,
            max_span: budget.max_span,
            typed_roles: budget.typed_roles,
            max_work: budget.max_work,
            roles: roles
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        model.to_bytes()?;
        Ok(model)
    }
    fn budget(&self) -> Budget {
        let mut learner = self.base.metadata().budget();
        learner.subwords = self.subwords;
        Budget {
            learner,
            max_span: self.max_span,
            typed_roles: self.typed_roles,
            max_work: self.max_work,
            span_replacements: 0.0,
        }
    }
    pub fn measure(&self, text: &str) -> search::Measurement<Prediction> {
        let mut work = 0;
        let result = (|| {
            let words = tokens(text)?;
            let meta = self.base.metadata();
            let b = self.budget();
            let n = meta.intents.len();
            let scores = self
                .base
                .scores(&intent_features(&words, b.learner), &mut work);
            if work > self.max_work {
                return Err(Error::Budget);
            }
            let (intent, intent_margin) = best(&scores, 0, n, None);
            if intent_margin < i32::from(meta.minimum_margin) {
                return Err(Error::Ambiguous);
            }
            let paths = decode(
                &words,
                &meta.intents[intent],
                &self.roles[intent],
                b,
                n,
                |f, w| self.base.scores(f, w),
                &mut work,
            )?;
            let margin = paths.get(1).map(|p| paths[0].score - p.score);
            if margin.is_some_and(|m| m < i64::from(meta.minimum_margin)) {
                return Err(Error::Ambiguous);
            }
            let mut sequence = vec![0; words.len()];
            for s in &paths[0].spans {
                sequence[s.start] = s.tag;
                if s.tag != 0 {
                    let inside = format!("I:{}", role(&meta.tags[s.tag]).unwrap());
                    let tag = meta.tags.iter().position(|t| t == &inside).unwrap();
                    sequence[s.start + 1..s.end].fill(tag);
                }
            }
            let output = search::output(
                &words,
                &meta.intents[intent],
                &sequence
                    .iter()
                    .map(|i| meta.tags[*i].as_str())
                    .collect::<Vec<_>>(),
            )?;
            Ok(Prediction {
                output,
                intent_margin,
                tag_margin: margin.map_or(i32::MAX, |m| m.min(i64::from(i32::MAX)) as i32),
                work,
            })
        })();
        search::Measurement { result, work }
    }
    pub fn weights(&self) -> usize {
        self.base.weights()
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() - std::mem::size_of::<compact::CompactModel>()
            + self.base.retained_bytes()
            + self.roles.len() * std::mem::size_of::<Box<[usize]>>()
            + self
                .roles
                .iter()
                .map(|r| r.len() * std::mem::size_of::<usize>())
                .sum::<usize>()
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let header = serde_json::to_vec(&Header {
            subwords: self.subwords,
            max_span: self.max_span,
            typed_roles: self.typed_roles,
            max_work: self.max_work,
            roles: self.roles.iter().map(|r| r.to_vec()).collect(),
        })
        .map_err(|e| Error::Storage(e.to_string()))?;
        if header.len() > 32768 {
            return Err(Error::Budget);
        }
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend(header);
        bytes.extend(self.base.to_bytes()?);
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if !(28..=MAX_BYTES).contains(&bytes.len()) {
            return Err(Error::Invalid("segment length"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if body.get(..8) != Some(MAGIC.as_slice()) || checksum(body).to_le_bytes() != digest {
            return Err(Error::Invalid("segment checksum/magic"));
        }
        let size = u32::from_le_bytes(body[8..12].try_into().unwrap()) as usize;
        if size > 32768 {
            return Err(Error::Budget);
        }
        let (header, base) = body[12..]
            .split_at_checked(size)
            .ok_or(Error::Invalid("segment header"))?;
        let h: Header =
            serde_json::from_slice(header).map_err(|e| Error::Storage(e.to_string()))?;
        if !(1..=16).contains(&h.max_span) || !(1..=50_000_000).contains(&h.max_work) {
            return Err(Error::Invalid("segment policy"));
        }
        let base = compact::CompactModel::from_bytes(base)?;
        let meta = base.metadata();
        let all = meta
            .tags
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (t == "O" || t.starts_with("B:")).then_some(i))
            .collect::<Vec<_>>();
        if h.roles.len() != meta.intents.len()
            || h.roles.iter().any(|r| {
                r.first() != Some(&0)
                    || r.windows(2).any(|p| p[0] >= p[1])
                    || r.iter().any(|i| !all.contains(i))
                    || !h.typed_roles && r != &all
            })
        {
            return Err(Error::Invalid("segment role schema"));
        }
        Ok(Self {
            base,
            subwords: h.subwords,
            max_span: h.max_span,
            typed_roles: h.typed_roles,
            max_work: h.max_work,
            roles: h
                .roles
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn budget() -> Budget {
        Budget {
            learner: super::super::Budget {
                seed: 47,
                buckets: 2048,
                epochs: 16,
                weights: 8000,
                intent_weights: 2000,
                subwords: false,
                shared_roles: false,
                minimum_margin: 1,
            },
            max_span: 4,
            typed_roles: true,
            max_work: 1_000_000,
            span_replacements: 0.0,
        }
    }
    fn data() -> Vec<Example> {
        (0..16)
            .flat_map(|i| {
                ["send", "cancel"].map(move |verb| Example {
                    text: format!("{verb} to person-{i} surname-{i} now"),
                    intent: verb.into(),
                    tags: ["O", "O", "B:person", "I:person", "O"]
                        .map(str::to_owned)
                        .to_vec(),
                })
            })
            .collect()
    }
    #[test]
    fn top_two_segmentation_matches_exhaustive_search_and_preserves_refusal_work() {
        let b = budget();
        let words = ["a", "b", "c", "d"].map(str::to_owned);
        let scores = |f: &[usize]| {
            let s = f.iter().sum::<usize>() as i32;
            vec![0, 0, s % 17 - 8, s % 13 - 6]
        };
        fn enumerate(
            words: &[String],
            b: Budget,
            start: usize,
            path: Path,
            out: &mut Vec<Path>,
            scores: &impl Fn(&[usize]) -> Vec<i32>,
        ) {
            if start == words.len() {
                out.push(path);
                return;
            }
            for end in start + 1..=(start + b.max_span).min(words.len()) {
                let values = scores(&features(words, start, end, "intent", b.learner));
                for tag in 0..2 {
                    if tag == 0 && end - start != 1 {
                        continue;
                    }
                    let mut p = path.clone();
                    p.spans.push(Span { start, end, tag });
                    p.score += i64::from(values[2 + tag]);
                    enumerate(words, b, end, p, out, scores);
                }
            }
        }
        let mut all = Vec::new();
        enumerate(
            &words,
            b,
            0,
            Path {
                spans: vec![],
                score: 0,
            },
            &mut all,
            &scores,
        );
        all.sort_by(|a, b| b.score.cmp(&a.score).then(a.spans.cmp(&b.spans)));
        all.truncate(2);
        let mut work = 0;
        let exact = decode(
            &words,
            "intent",
            &[0, 1],
            b,
            2,
            |f, w| {
                *w += f.len();
                scores(f)
            },
            &mut work,
        )
        .unwrap();
        assert_eq!(exact, all);
        assert!(work > 0);
        let mut used = 0;
        assert_eq!(
            decode(
                &words,
                "intent",
                &[0, 1],
                Budget { max_work: 1, ..b },
                2,
                |f, w| {
                    *w += f.len();
                    scores(f)
                },
                &mut used
            )
            .unwrap_err(),
            Error::Budget
        );
        assert!(used > 1);
    }
    #[test]
    fn learned_segments_copy_new_values_roundtrip_and_keep_exact_residency() {
        let model = Model::learn(&data(), budget(), &mut |_| Ok(())).unwrap();
        let raw = model.to_bytes().unwrap();
        let child = Model::from_bytes(&raw).unwrap();
        assert_eq!(raw, child.to_bytes().unwrap());
        assert_eq!(model.retained_bytes(), child.retained_bytes());
        let input = "send to new name now";
        let p = child.measure(input).result.unwrap();
        assert_eq!(
            p.output,
            search::output(
                &tokens(input).unwrap(),
                "send",
                &["O", "O", "B:person", "I:person", "O"]
            )
            .unwrap()
        );
        assert!(!String::from_utf8_lossy(&raw).contains("person-0 surname-0"));
        let capped = Model {
            max_work: 1,
            ..child
        };
        let m = capped.measure(input);
        assert_eq!(m.result.unwrap_err(), Error::Budget);
        assert!(m.work > 0);
        assert!(model.measure("x\0y").result.is_err());
        assert!(
            Model::learn(
                &data(),
                Budget {
                    max_span: 1,
                    ..budget()
                },
                &mut |_| Ok(())
            )
            .is_err()
        );
    }
    #[test]
    fn repaired_checksums_cannot_change_segment_bounds_roles_or_header_shape() {
        let model = Model::learn(&data(), budget(), &mut |_| Ok(())).unwrap();
        let raw = model.to_bytes().unwrap();
        let size = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let end = 12 + size;
        let h: serde_json::Value = serde_json::from_slice(&raw[12..end]).unwrap();
        let inside = model
            .base
            .metadata()
            .tags
            .iter()
            .position(|t| t.starts_with("I:"))
            .unwrap();
        for (field, value) in [
            ("max_span", serde_json::json!(usize::MAX)),
            ("max_work", serde_json::json!(0)),
            ("roles", serde_json::json!([[0, 0], [0]])),
            ("roles", serde_json::json!([[0, inside], [0]])),
            ("roles", serde_json::json!([[0]])),
        ] {
            let mut changed = h.clone();
            changed[field] = value;
            let header = serde_json::to_vec(&changed).unwrap();
            let mut b = MAGIC.to_vec();
            b.extend_from_slice(&(header.len() as u32).to_le_bytes());
            b.extend(header);
            b.extend_from_slice(&raw[end..raw.len() - 8]);
            b.extend_from_slice(&checksum(&b).to_le_bytes());
            assert!(Model::from_bytes(&b).is_err(), "{field}");
        }
        for i in [0, 7, 8, 12, raw.len() - 9, raw.len() - 1] {
            let mut b = raw.clone();
            b[i] ^= 1;
            assert!(Model::from_bytes(&b).is_err());
        }
        assert!(
            Budget {
                max_work: usize::MAX,
                ..budget()
            }
            .validate()
            .is_err()
        );
        assert!(Model::learn(&data()[..2], budget(), &mut |_| Ok(())).is_err());
    }
    #[test]
    fn recombination_is_train_only_bounded_and_zero_probability_preserves_rng() {
        let words = ["send", "to", "old", "name", "now"].map(str::to_owned);
        let spans = vec![
            Span {
                start: 0,
                end: 1,
                tag: 0,
            },
            Span {
                start: 1,
                end: 2,
                tag: 0,
            },
            Span {
                start: 2,
                end: 4,
                tag: 1,
            },
            Span {
                start: 4,
                end: 5,
                tag: 0,
            },
        ];
        let pool = vec![
            vec![],
            vec![
                vec!["new".into()],
                vec!["another".into(), "new".into(), "name".into()],
            ],
        ];
        let mut state = 47;
        let original = state;
        assert_eq!(
            recombine(&words, &spans, &pool, 0.0, &mut state),
            (words.to_vec(), spans.clone(), 0)
        );
        assert_eq!(state, original);
        let actual = recombine(&words, &spans, &pool, 1.0, &mut state);
        let mut repeat = 47;
        assert_eq!(actual, recombine(&words, &spans, &pool, 1.0, &mut repeat));
        assert_eq!(&actual.0[..2], &words[..2]);
        assert_eq!(actual.0.last(), words.last());
        assert_eq!(actual.2, 1);
        assert!(
            actual
                .1
                .iter()
                .all(|s| s.start < s.end && s.end <= actual.0.len())
        );
        let long = vec!["fixed".to_owned(); 64];
        let labels = (0..64)
            .map(|i| Span {
                start: i,
                end: i + 1,
                tag: usize::from(i == 32),
            })
            .collect::<Vec<_>>();
        let oversized = vec![vec![], vec![vec!["a".into(), "b".into(), "c".into()]]];
        let mut state = 47;
        assert_eq!(
            recombine(&long, &labels, &oversized, 1.0, &mut state),
            (long, labels, 0)
        );
        let mut changed = 0;
        let config = Budget {
            span_replacements: 0.7,
            ..budget()
        };
        let model = Model::learn(&data(), config, &mut |e| {
            changed += e.replaced_spans;
            assert!(e.augmented_sentences <= e.base.sentences);
            Ok(())
        })
        .unwrap();
        assert!(changed > 0);
        let raw = model.to_bytes().unwrap();
        assert_eq!(raw, Model::from_bytes(&raw).unwrap().to_bytes().unwrap());
        assert!(!String::from_utf8_lossy(&raw).contains("span_replacements"));
        for span_replacements in [f32::NAN, -0.1, 1.1] {
            assert!(
                Budget {
                    span_replacements,
                    ..budget()
                }
                .validate()
                .is_err()
            );
        }
    }
}
