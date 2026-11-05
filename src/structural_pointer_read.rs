//! Candidate-conditioned reads of learned context; not a Transformer or new arithmetic.
use super::*;

const MAGIC: &[u8; 8] = b"AXCR0001";
pub const CONTEXT: &str = "axon/context-relational-pointer-v1";
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Read {
    Neighbors,
    Uniform,
    Attention,
}
impl Read {
    pub(super) fn scratch_bytes(self, words: usize) -> usize {
        if self == Self::Neighbors {
            0
        } else {
            words * 4
        }
    }
    pub(super) fn work(self, words: usize, candidates: usize, dim: usize) -> usize {
        if self == Self::Neighbors {
            0
        } else {
            2 * candidates
                * words
                * (2 * dim
                    + if self == Self::Attention {
                        3 * dim + 8
                    } else {
                        1
                    })
        }
    }
}
pub(super) struct Cache {
    read: Read,
    weights: Vec<f32>,
    pub(super) scratch_bytes: usize,
}
pub(super) fn features(
    read: Read,
    cache: &ordered_context::Cache,
    embedded: &[f32],
    positions: &[usize],
    d: usize,
    training: bool,
) -> (Features, Cache) {
    let mut result = Features::new(cache, embedded, positions, d);
    if read == Read::Neighbors {
        return (
            result,
            Cache {
                read,
                weights: Vec::new(),
                scratch_bytes: 0,
            },
        );
    }
    let words = embedded.len() / d;
    let mut alpha = vec![0.0; words];
    let mut weights = if training {
        Vec::with_capacity(2 * positions.len() * words)
    } else {
        Vec::new()
    };
    for (a, &position) in positions.iter().enumerate() {
        for direction in 0..2 {
            if read == Read::Attention {
                let query = cache.state(direction, position, d);
                for (j, s) in alpha.iter_mut().enumerate() {
                    *s = dot(query, cache.state(direction, j, d)) / (d as f32).sqrt();
                }
                let max = alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut alpha {
                    *s = (*s - max).exp();
                }
                let sum: f32 = alpha.iter().sum();
                for s in &mut alpha {
                    *s /= sum;
                }
            } else {
                alpha.fill(1.0 / words as f32);
            }
            let output =
                &mut result.candidates[(a * 4 + 2 + direction) * d..(a * 4 + 3 + direction) * d];
            output.fill(0.0);
            for (j, &weight) in alpha.iter().enumerate() {
                for k in 0..d {
                    output[k] += weight * embedded[j * d + k];
                }
            }
            if training {
                weights.extend_from_slice(&alpha);
            }
        }
    }
    (
        result,
        Cache {
            read,
            weights,
            scratch_bytes: alpha.capacity() * 4,
        },
    )
}
impl Cache {
    pub(super) fn scatter(
        &self,
        features: &Features,
        cache: &ordered_context::Cache,
        embedded: &[f32],
        global: &[f32],
        candidates: &[f32],
        positions: &[usize],
    ) -> (Vec<f32>, Vec<f32>) {
        let words = embedded.len() / features.dim;
        if self.read == Read::Neighbors {
            return features.scatter(global, candidates, positions, words);
        }
        let d = features.dim;
        assert_eq!(self.weights.len(), positions.len() * 2 * words);
        let (mut states, mut direct) =
            features.scatter_core(global, candidates, positions, words, false);
        for (a, &position) in positions.iter().enumerate() {
            for direction in 0..2 {
                let weights =
                    &self.weights[(a * 2 + direction) * words..(a * 2 + direction + 1) * words];
                let gradient =
                    &candidates[(a * 4 + 2 + direction) * d..(a * 4 + 3 + direction) * d];
                let output =
                    &features.candidates[(a * 4 + 2 + direction) * d..(a * 4 + 3 + direction) * d];
                let center = dot(gradient, output);
                for (j, &weight) in weights.iter().enumerate() {
                    for k in 0..d {
                        direct[j * d + k] += weight * gradient[k];
                    }
                    if self.read == Read::Attention {
                        let score = weight
                            * (dot(gradient, &embedded[j * d..(j + 1) * d]) - center)
                            / (d as f32).sqrt();
                        for k in 0..d {
                            states[(direction * words + position) * d + k] +=
                                score * cache.state(direction, j, d)[k];
                            states[(direction * words + j) * d + k] +=
                                score * cache.state(direction, position, d)[k];
                        }
                    }
                }
            }
        }
        (states, direct)
    }
}
#[derive(Debug)]
pub struct ContextModel {
    model: Model,
    read: Read,
}

impl ContextModel {
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        read: Read,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        if read == Read::Neighbors {
            return Err(Error::Invalid("context pointer read mode"));
        }
        let result = Self {
            model: Model::learn_read(examples, budget, read, emit)?,
            read,
        };
        result.to_bytes()?;
        Ok(result)
    }
    pub fn base(&self) -> &Model {
        &self.model
    }
    pub fn read(&self) -> Read {
        self.read
    }
    pub fn retained_bytes(&self) -> usize {
        self.model.retained_bytes() + size_of::<Self>() - size_of::<Model>()
    }
    pub fn measure(&self, input: &Input, max_work: usize, max_workspace: usize) -> Measurement {
        self.model
            .measure_read(input, max_work, max_workspace, self.read)
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut bytes = MAGIC.to_vec();
        bytes.push(match self.read {
            Read::Uniform => 1,
            Read::Attention => 2,
            Read::Neighbors => return Err(Error::Invalid("context pointer read mode")),
        });
        bytes.extend(self.model.to_bytes()?);
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if !(37..=MAX_BYTES).contains(&bytes.len()) || bytes.get(..8) != Some(MAGIC) {
            return Err(Error::Invalid("context pointer header"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if checksum(body).to_le_bytes() != digest {
            return Err(Error::Invalid("context pointer checksum"));
        }
        let read = match body[8] {
            1 => Read::Uniform,
            2 => Read::Attention,
            _ => return Err(Error::Invalid("context pointer read mode")),
        };
        Ok(Self {
            model: Model::from_bytes(&body[9..])?,
            read,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learned_reads_are_portable_bounded_and_strictly_versioned() {
        let examples = (0..32)
            .map(|i| Example {
                input: Input {
                    text: format!(
                        "{} {} x {}",
                        if i % 2 == 0 { "red" } else { "blue" },
                        i + 1,
                        i + 40
                    ),
                    candidates: vec![1, 3],
                },
                targets: vec![Selection {
                    label: if i % 2 == 0 { "one" } else { "two" }.into(),
                    left: if i % 2 == 0 { 1 } else { 3 },
                    right: None,
                }],
            })
            .collect::<Vec<_>>();
        let budget = Budget {
            seed: 47,
            vocabulary: 64,
            dimensions: 8,
            epochs: 64,
            learning_rate: 0.03,
            minibatch: 8,
            candidate_vocabulary: CandidateVocabulary::Omit,
            independent_shuffle: true,
        };
        for read in [Read::Uniform, Read::Attention] {
            let model = ContextModel::learn(&examples, budget, read, &mut |_| Ok(())).unwrap();
            let bytes = model.to_bytes().unwrap();
            assert_eq!(&bytes[..8], MAGIC);
            assert!(Model::from_bytes(&bytes).is_err());
            assert!(ContextModel::from_bytes(&model.base().to_bytes().unwrap()).is_err());
            let restored = ContextModel::from_bytes(&bytes).unwrap();
            assert_eq!(restored.read(), read);
            assert_eq!(restored.to_bytes().unwrap(), bytes);
            assert_eq!(model.retained_bytes(), restored.retained_bytes());
            for e in &examples {
                let first = model.measure(&e.input, 2_000_000, 262144).result.unwrap();
                let second = restored
                    .measure(&e.input, 2_000_000, 262144)
                    .result
                    .unwrap();
                assert_eq!(first.selection, e.targets[0]);
                assert_eq!(second.selection, first.selection);
                assert_eq!(second.margin, first.margin);
                assert_eq!(second.work, first.work);
                assert_eq!(second.workspace_bytes, first.workspace_bytes);
                assert!(model.measure(&e.input, 1, 262144).result.is_err());
                assert!(model.measure(&e.input, 2_000_000, 0).result.is_err());
            }
            let mut bad = bytes.clone();
            bad[8] = 3;
            let n = bad.len() - 8;
            let hash = checksum(&bad[..n]).to_le_bytes();
            bad[n..].copy_from_slice(&hash);
            assert!(ContextModel::from_bytes(&bad).is_err());
            assert!(ContextModel::from_bytes(&bytes[..bytes.len() - 1]).is_err());
            let input = Input {
                text: std::iter::repeat_n("x", 544).collect::<Vec<_>>().join(" "),
                candidates: vec![0],
            };
            let p = restored.measure(&input, 2_000_000, 262144).result.unwrap();
            assert_eq!(p.blocks, 17);
            assert!(p.work < 2_000_000 && p.workspace_bytes < 262144);
        }
        assert!(ContextModel::learn(&examples, budget, Read::Neighbors, &mut |_| Ok(())).is_err());
    }
}
