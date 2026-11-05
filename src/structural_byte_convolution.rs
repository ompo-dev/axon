//! Bounded UTF-8 byte embeddings, width-three convolution, tanh and max pooling.
//! Character CNN precedent: Ma/Hovy 2016, https://aclanthology.org/P16-1101/.
//! This module is not their BiLSTM/CRF, nor a universal byte or semantic interpreter.
use super::{Error, random};
pub(super) const ROWS: usize = 258;
pub(super) const EMBED: usize = 8;
pub(super) const WINDOW: usize = 3;
pub(super) const WIDTH: usize = EMBED * WINDOW;

pub(super) struct Cache {
    bytes: Vec<u16>,
    winners: Vec<u16>,
    pub output: Vec<f32>,
    pub work: usize,
}
pub(super) fn encode(
    word: &str,
    dimensions: usize,
    mut embedding: impl FnMut(usize, usize) -> f32,
    mut filter: impl FnMut(usize, usize) -> f32,
    bias: &[f32],
) -> Result<Cache, Error> {
    if word.is_empty()
        || word.len() > 256
        || word.chars().any(|c| c.is_control() || c.is_whitespace())
        || !(8..=16).contains(&dimensions)
        || bias.len() != dimensions
    {
        return Err(Error::Invalid("byte convolution input"));
    }
    let bytes = std::iter::once(256)
        .chain(word.bytes().map(u16::from))
        .chain(std::iter::once(257))
        .collect::<Vec<_>>();
    let windows = bytes.len() - WINDOW + 1;
    let mut output = vec![0.0; dimensions];
    let mut winners = vec![0u16; dimensions];
    for d in 0..dimensions {
        let mut best = f32::NEG_INFINITY;
        for start in 0..windows {
            let mut value = bias[d];
            for k in 0..WIDTH {
                value += embedding(bytes[start + k / EMBED] as usize, k % EMBED) * filter(d, k);
            }
            if value > best {
                best = value;
                winners[d] = start as u16;
            }
        }
        output[d] = best.tanh();
    }
    Ok(Cache {
        bytes,
        winners,
        output,
        work: windows * dimensions * (WIDTH * 3 + 2),
    })
}
#[derive(Clone)]
pub(super) struct Train {
    pub embedding: Vec<f32>,
    pub filters: Vec<f32>,
    pub bias: Vec<f32>,
    dimensions: usize,
}
pub(super) struct Gradient {
    embedding: Vec<f32>,
    filters: Vec<f32>,
    bias: Vec<f32>,
}
impl Train {
    pub fn new(dimensions: usize, seed: u64) -> Result<Self, Error> {
        if !(8..=16).contains(&dimensions) || seed == 0 {
            return Err(Error::Invalid("byte convolution budget"));
        }
        let mut state = seed.wrapping_add(0xd1b54a32d192ed03).max(1);
        let mut draw = |width: usize| {
            ((random(&mut state) >> 40) as f32 / (1u32 << 24) as f32 - 0.5) / (width as f32).sqrt()
        };
        Ok(Self {
            embedding: (0..ROWS * EMBED).map(|_| draw(EMBED)).collect(),
            filters: (0..dimensions * WIDTH).map(|_| draw(WIDTH)).collect(),
            bias: vec![0.0; dimensions],
            dimensions,
        })
    }
    pub fn encode(&self, word: &str) -> Result<Cache, Error> {
        encode(
            word,
            self.dimensions,
            |r, d| self.embedding[r * EMBED + d],
            |r, d| self.filters[r * WIDTH + d],
            &self.bias,
        )
    }
    pub fn gradient(&self, caches: &[Cache], deltas: &[f32]) -> Result<Gradient, Error> {
        if deltas.len() != caches.len() * self.dimensions || deltas.iter().any(|g| !g.is_finite()) {
            return Err(Error::Invalid("byte convolution gradient"));
        }
        let mut out = Gradient {
            embedding: vec![0.0; self.embedding.len()],
            filters: vec![0.0; self.filters.len()],
            bias: vec![0.0; self.dimensions],
        };
        for (i, cache) in caches.iter().enumerate() {
            for d in 0..self.dimensions {
                let g = deltas[i * self.dimensions + d] * (1.0 - cache.output[d] * cache.output[d]);
                out.bias[d] += g;
                let start = cache.winners[d] as usize;
                for k in 0..WIDTH {
                    let row = cache.bytes[start + k / EMBED] as usize * EMBED + k % EMBED;
                    out.filters[d * WIDTH + k] += g * self.embedding[row];
                    out.embedding[row] += g * self.filters[d * WIDTH + k];
                }
            }
        }
        Ok(out)
    }
    pub fn apply(&mut self, gradient: Gradient, scale: f32) -> Result<(), Error> {
        if !scale.is_finite() || !(0.0..=1.0).contains(&scale) {
            return Err(Error::Invalid("byte convolution step"));
        }
        for (values, gradients, bound) in [
            (&mut self.embedding, gradient.embedding, 8.0),
            (&mut self.filters, gradient.filters, 8.0),
            (&mut self.bias, gradient.bias, 16.0),
        ] {
            for (w, g) in values.iter_mut().zip(gradients) {
                *w = (*w - scale * g).clamp(-bound, bound);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_order_unicode_boundaries_and_work_are_explicit_and_bounded() {
        let model = Train::new(8, 47).unwrap();
        assert_ne!(
            model.encode("ab").unwrap().output,
            model.encode("ba").unwrap().output
        );
        for word in ["a", "a\u{e7}\u{e3}o", "\u{732b}", &"x".repeat(256)] {
            let cache = model.encode(word).unwrap();
            assert_eq!(cache.bytes.len(), word.len() + 2);
            assert_eq!(cache.bytes[0], 256);
            assert_eq!(cache.bytes.last(), Some(&257));
            assert!(cache.work > 0);
            assert_eq!(cache.output.len(), 8);
            assert!(
                cache
                    .output
                    .iter()
                    .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
            );
        }
        for word in ["", "two words", "a\nb", &"x".repeat(257)] {
            assert!(model.encode(word).is_err());
        }
        assert!(Train::new(7, 47).is_err());
        assert!(Train::new(16, 0).is_err());
    }
    #[test]
    fn convolution_embedding_filter_and_bias_gradients_match_finite_differences() {
        let model = Train::new(8, 47).unwrap();
        let words = ["aba", "abc"];
        let deltas = (0..16).map(|i| (i as f32 - 7.0) * 0.07).collect::<Vec<_>>();
        let loss = |m: &Train| {
            words
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    m.encode(w)
                        .unwrap()
                        .output
                        .iter()
                        .enumerate()
                        .map(|(d, v)| v * deltas[i * 8 + d])
                        .sum::<f32>()
                })
                .sum::<f32>()
        };
        let caches = words
            .iter()
            .map(|w| model.encode(w).unwrap())
            .collect::<Vec<_>>();
        let gradient = model.gradient(&caches, &deltas).unwrap();
        for (kind, indices) in [
            (
                0,
                vec![
                    97 * EMBED,
                    97 * EMBED + 3,
                    98 * EMBED + 5,
                    256 * EMBED,
                    257 * EMBED + 7,
                ],
            ),
            (1, vec![0, 5, 24, 83, 191]),
            (2, (0..8).collect()),
        ] {
            for i in indices {
                let mut hi = model.clone();
                let mut lo = model.clone();
                let expected = match kind {
                    0 => {
                        hi.embedding[i] += 0.001;
                        lo.embedding[i] -= 0.001;
                        gradient.embedding[i]
                    }
                    1 => {
                        hi.filters[i] += 0.001;
                        lo.filters[i] -= 0.001;
                        gradient.filters[i]
                    }
                    _ => {
                        hi.bias[i] += 0.001;
                        lo.bias[i] -= 0.001;
                        gradient.bias[i]
                    }
                };
                let numeric = (loss(&hi) - loss(&lo)) / 0.002;
                assert!(
                    (numeric - expected).abs() < 0.00015,
                    "{kind}/{i}: {numeric} vs {expected}"
                );
            }
        }
        assert!(model.gradient(&caches, &[0.0]).is_err());
    }
    #[test]
    fn shared_byte_filters_acquire_a_taught_distinction_without_a_word_table() {
        let mut model = Train::new(8, 47).unwrap();
        let words = ["aa0", "aa1", "zz0", "zz1"];
        let target = [0.5, 0.5, -0.5, -0.5];
        let loss = |m: &Train| {
            words
                .iter()
                .enumerate()
                .map(|(i, w)| (m.encode(w).unwrap().output[0] - target[i]).powi(2))
                .sum::<f32>()
        };
        let before = loss(&model);
        for _ in 0..400 {
            let caches = words
                .iter()
                .map(|w| model.encode(w).unwrap())
                .collect::<Vec<_>>();
            let mut deltas = vec![0.0; words.len() * 8];
            for (i, c) in caches.iter().enumerate() {
                deltas[i * 8] = 2.0 * (c.output[0] - target[i]);
            }
            let g = model.gradient(&caches, &deltas).unwrap();
            model.apply(g, 0.1).unwrap();
        }
        assert!(loss(&model) < before * 0.1);
        assert!(
            model.encode("aaX").unwrap().output[0] > model.encode("zzX").unwrap().output[0] + 0.2
        );
        assert_eq!(model.embedding.len(), ROWS * EMBED);
    }
}
