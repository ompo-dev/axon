//! Bounded bidirectional Elman context, not an LSTM or general semantic parser.
//! Established direction: https://aclanthology.org/N18-2050/ (different architecture).
use super::{Error, random};
pub(super) struct Cache {
    inputs: Vec<f32>,
    states: Vec<f32>,
    pub output: Vec<f32>,
    pub work: usize,
}
pub(super) fn encode(
    inputs: &[f32],
    dim: usize,
    mut weight: impl FnMut(usize, usize, usize) -> f32,
    bias: &[f32],
) -> Result<Cache, Error> {
    encode_bounded(inputs, dim, &mut weight, bias, 64, 64, true)
}
pub(super) fn encode_streamed(
    inputs: &[f32],
    dim: usize,
    mut weight: impl FnMut(usize, usize, usize) -> f32,
    bias: &[f32],
    block: usize,
) -> Result<Cache, Error> {
    encode_bounded(inputs, dim, &mut weight, bias, 544, block, true)
}
pub(super) fn encode_states(
    inputs: &[f32],
    dim: usize,
    mut weight: impl FnMut(usize, usize, usize) -> f32,
    bias: &[f32],
    block: usize,
) -> Result<Cache, Error> {
    encode_bounded(inputs, dim, &mut weight, bias, 544, block, false)
}
fn encode_bounded(
    inputs: &[f32],
    dim: usize,
    weight: &mut impl FnMut(usize, usize, usize) -> f32,
    bias: &[f32],
    maximum: usize,
    block: usize,
    training_cache: bool,
) -> Result<Cache, Error> {
    if !(8..=16).contains(&dim)
        || inputs.is_empty()
        || !inputs.len().is_multiple_of(dim)
        || inputs.len() / dim > maximum
        || !(1..=64).contains(&block)
        || inputs.iter().any(|v| !v.is_finite() || v.abs() > 8.0)
        || bias.len() != 2 * dim
        || bias.iter().any(|v| !v.is_finite() || v.abs() > 16.0)
    {
        return Err(Error::Invalid("ordered context input"));
    }
    let count = inputs.len() / dim;
    let mut states = vec![0.0; inputs.len() * 2];
    let mut output = if training_cache {
        vec![0.0; inputs.len()]
    } else {
        Vec::new()
    };
    for direction in 0..2 {
        for step in (0..count)
            .step_by(block)
            .flat_map(|start| start..(start + block).min(count))
        {
            let i = if direction == 0 {
                step
            } else {
                count - step - 1
            };
            let previous = if step == 0 {
                None
            } else if direction == 0 {
                Some(i - 1)
            } else {
                Some(i + 1)
            };
            for d in 0..dim {
                let mut value = inputs[i * dim + d] + bias[direction * dim + d];
                if let Some(p) = previous {
                    for k in 0..dim {
                        value +=
                            weight(direction, d, k) * states[(direction * count + p) * dim + k];
                    }
                }
                let state = value.tanh();
                states[(direction * count + i) * dim + d] = state;
                if training_cache {
                    output[i * dim + d] += state * std::f32::consts::FRAC_1_SQRT_2;
                }
            }
        }
    }
    Ok(Cache {
        inputs: if training_cache {
            inputs.to_vec()
        } else {
            Vec::new()
        },
        states,
        output,
        work: 2 * count * dim * (3 * dim + 4),
    })
}
pub(super) struct Train {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    dim: usize,
}
pub(super) struct Gradient {
    weights: Vec<f32>,
    bias: Vec<f32>,
    pub inputs: Vec<f32>,
}
impl Cache {
    pub fn state(&self, direction: usize, index: usize, dim: usize) -> &[f32] {
        let start = (direction * (self.states.len() / (2 * dim)) + index) * dim;
        &self.states[start..start + dim]
    }
    pub fn workspace_bytes(&self) -> usize {
        (self.inputs.capacity() + self.states.capacity() + self.output.capacity()) * 4
    }
}
impl Gradient {
    pub fn parameters(&self) -> (&[f32], &[f32]) {
        (&self.weights, &self.bias)
    }
}
impl Train {
    pub fn new(dim: usize, seed: u64) -> Result<Self, Error> {
        if !(8..=16).contains(&dim) || seed == 0 {
            return Err(Error::Invalid("ordered context budget"));
        }
        let mut state = seed.wrapping_add(0x94d049bb133111eb).max(1);
        let weights = (0..2 * dim * dim)
            .map(|i| {
                let noise = ((random(&mut state) >> 40) as f32 / (1u32 << 24) as f32 - 0.5)
                    / (dim as f32).sqrt();
                noise + if i % dim == (i / dim) % dim { 0.5 } else { 0.0 }
            })
            .collect();
        Ok(Self {
            weights,
            bias: vec![0.0; 2 * dim],
            dim,
        })
    }
    pub fn encode(&self, inputs: &[f32]) -> Result<Cache, Error> {
        encode(
            inputs,
            self.dim,
            |direction, d, k| self.weights[(direction * self.dim + d) * self.dim + k],
            &self.bias,
        )
    }
    pub fn encode_streamed(&self, inputs: &[f32], block: usize) -> Result<Cache, Error> {
        encode_streamed(
            inputs,
            self.dim,
            |direction, d, k| self.weights[(direction * self.dim + d) * self.dim + k],
            &self.bias,
            block,
        )
    }
    pub fn gradient(&self, cache: &Cache, deltas: &[f32]) -> Result<Gradient, Error> {
        self.gradient_using(cache, deltas, false)
    }
    pub fn gradient_states(&self, cache: &Cache, deltas: &[f32]) -> Result<Gradient, Error> {
        self.gradient_using(cache, deltas, true)
    }
    fn gradient_using(
        &self,
        cache: &Cache,
        deltas: &[f32],
        separate: bool,
    ) -> Result<Gradient, Error> {
        if cache.inputs.is_empty()
            || deltas.len() != cache.inputs.len() * if separate { 2 } else { 1 }
            || deltas.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("ordered context gradient"));
        }
        let dim = self.dim;
        let count = cache.inputs.len() / dim;
        let mut result = Gradient {
            weights: vec![0.0; self.weights.len()],
            bias: vec![0.0; self.bias.len()],
            inputs: vec![0.0; cache.inputs.len()],
        };
        for direction in 0..2 {
            let mut carry = vec![0.0; dim];
            for step in 0..count {
                let i = if direction == 0 {
                    count - step - 1
                } else {
                    step
                };
                let previous = if direction == 0 {
                    i.checked_sub(1)
                } else {
                    (i + 1 < count).then_some(i + 1)
                };
                let mut next = vec![0.0; dim];
                for d in 0..dim {
                    let h = cache.states[(direction * count + i) * dim + d];
                    let direct = if separate {
                        deltas[(direction * count + i) * dim + d]
                    } else {
                        deltas[i * dim + d] * std::f32::consts::FRAC_1_SQRT_2
                    };
                    let g = (direct + carry[d]) * (1.0 - h * h);
                    result.inputs[i * dim + d] += g;
                    result.bias[direction * dim + d] += g;
                    if let Some(p) = previous {
                        for (k, n) in next.iter_mut().enumerate() {
                            let index = (direction * dim + d) * dim + k;
                            result.weights[index] +=
                                g * cache.states[(direction * count + p) * dim + k];
                            *n += g * self.weights[index];
                        }
                    }
                }
                carry = next;
            }
        }
        if result
            .inputs
            .iter()
            .chain(&result.weights)
            .chain(&result.bias)
            .any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("ordered context finite gradient"));
        }
        Ok(result)
    }
    pub fn apply(
        &mut self,
        gradient: &Gradient,
        scale: f32,
        frozen: bool,
        input_gradient: &[f32],
    ) -> Result<f32, Error> {
        if !scale.is_finite()
            || !(0.0..=1.0).contains(&scale)
            || input_gradient.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("ordered context step"));
        }
        // Clip the shared parameter/input step together, not the analytical derivative.
        let norm = gradient
            .weights
            .iter()
            .chain(&gradient.bias)
            .chain(input_gradient)
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = scale * (1.0 / norm.max(1.0)) as f32;
        if !frozen {
            for (values, gradients, bound) in [
                (&mut self.weights, &gradient.weights, 8.0),
                (&mut self.bias, &gradient.bias, 16.0),
            ] {
                for (w, g) in values.iter_mut().zip(gradients) {
                    *w = (*w - scale * g).clamp(-bound, bound);
                }
            }
        }
        Ok(scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inference_states_are_identical_without_training_copies() {
        for dim in [8, 16] {
            let model = Train::new(dim, 47).unwrap();
            for count in [1, 64, 129, 544] {
                let inputs = (0..count * dim)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
                    .collect::<Vec<_>>();
                let full = model.encode_streamed(&inputs, 32).unwrap();
                let compact = encode_states(
                    &inputs,
                    dim,
                    |direction, d, k| model.weights[(direction * dim + d) * dim + k],
                    &model.bias,
                    32,
                )
                .unwrap();
                assert_eq!(compact.states, full.states);
                assert_eq!(compact.work, full.work);
                assert!(compact.inputs.is_empty() && compact.output.is_empty());
                assert_eq!(
                    full.workspace_bytes() - compact.workspace_bytes(),
                    2 * inputs.len() * 4
                );
                for direction in 0..2 {
                    for i in 0..count {
                        assert_eq!(
                            compact.state(direction, i, dim),
                            full.state(direction, i, dim)
                        );
                    }
                }
                assert!(model.gradient_states(&compact, &[]).is_err());
            }
        }
    }
    #[test]
    fn streamed_context_preserves_boundaries_and_legacy_limit() {
        let model = Train::new(8, 47).unwrap();
        for count in [7, 64, 129, 544] {
            let input = (0..count * 8)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
                .collect::<Vec<_>>();
            let expected = model.encode_streamed(&input, 1).unwrap();
            for block in [7, 32, 64] {
                let actual = model.encode_streamed(&input, block).unwrap();
                assert_eq!(actual.states, expected.states);
                assert_eq!(actual.output, expected.output);
                assert_eq!(actual.work, expected.work);
            }
            if count <= 64 {
                assert_eq!(model.encode(&input).unwrap().states, expected.states);
            } else {
                assert!(model.encode(&input).is_err());
            }
        }
        assert!(model.encode_streamed(&vec![0.0; 545 * 8], 32).is_err());
        assert!(model.encode_streamed(&[0.0; 8], 0).is_err());
    }
    #[test]
    fn context_has_order_two_directions_and_input_limits() {
        let model = Train::new(8, 47).unwrap();
        let inputs = (0..24)
            .map(|i| (i as f32 - 11.0) * 0.02)
            .collect::<Vec<_>>();
        let mut reversed = inputs
            .as_chunks::<8>()
            .0
            .iter()
            .rev()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert_ne!(
            model.encode(&inputs).unwrap().output,
            model.encode(&reversed).unwrap().output
        );
        let first = model.encode(&inputs).unwrap();
        reversed.copy_from_slice(&inputs);
        reversed[23] += 0.2;
        let changed = model.encode(&reversed).unwrap();
        assert_ne!(
            &first.output[..8],
            &changed.output[..8],
            "Future context must reach earlier positions"
        );
        assert_eq!(first.work, 2 * 3 * 8 * (3 * 8 + 4));
        for invalid in [
            vec![],
            vec![0.0; 7],
            vec![0.0; 65 * 8],
            vec![f32::NAN; 8],
            vec![9.0; 8],
        ] {
            assert!(model.encode(&invalid).is_err());
        }
        assert!(Train::new(7, 47).is_err());
        assert!(Train::new(8, 0).is_err());
    }
    #[test]
    fn bidirectional_bptt_matches_input_matrix_and_bias_finite_differences() {
        let mut model = Train::new(8, 47).unwrap();
        let inputs = (0..32)
            .map(|i| (i as f32 - 15.0) * 0.01)
            .collect::<Vec<_>>();
        let deltas = (0..32)
            .map(|i| (i as f32 - 13.0) * 0.03)
            .collect::<Vec<_>>();
        let loss = |m: &Train, x: &[f32]| {
            m.encode(x)
                .unwrap()
                .output
                .iter()
                .zip(&deltas)
                .map(|(v, g)| v * g)
                .sum::<f32>()
        };
        let gradient = model
            .gradient(&model.encode(&inputs).unwrap(), &deltas)
            .unwrap();
        for kind in 0..3 {
            let count = match kind {
                0 => inputs.len(),
                1 => model.weights.len(),
                _ => model.bias.len(),
            };
            for i in 0..count {
                let mut x = inputs.clone();
                let expected = match kind {
                    0 => {
                        x[i] += 0.001;
                        gradient.inputs[i]
                    }
                    1 => {
                        model.weights[i] += 0.001;
                        gradient.weights[i]
                    }
                    _ => {
                        model.bias[i] += 0.001;
                        gradient.bias[i]
                    }
                };
                let high = loss(&model, &x);
                match kind {
                    0 => x[i] -= 0.002,
                    1 => model.weights[i] -= 0.002,
                    _ => model.bias[i] -= 0.002,
                }
                let low = loss(&model, &x);
                match kind {
                    0 => {}
                    1 => model.weights[i] += 0.001,
                    _ => model.bias[i] += 0.001,
                }
                let numerical = (high - low) / 0.002;
                assert!(
                    (numerical - expected).abs() < 0.0015,
                    "kind {kind} index {i}: {numerical} != {expected}"
                );
            }
        }
        let initial = loss(&model, &inputs);
        model
            .apply(&gradient, 0.2, false, &gradient.inputs)
            .unwrap();
        assert!(loss(&model, &inputs) < initial);
        assert!(
            model
                .gradient(&model.encode(&inputs).unwrap(), &[0.0])
                .is_err()
        );
        let before = model.weights.clone();
        let scale = model.apply(&gradient, 1.0, true, &[1000.0]).unwrap();
        assert!(scale <= 0.001);
        assert_eq!(model.weights, before);
    }
}
