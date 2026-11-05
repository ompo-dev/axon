//! Train-only dense Adam; no optimizer state is deployed with the numeric genome.
//! Algorithm: Kingma and Ba, https://arxiv.org/abs/1412.6980 (not an AXON invention).
use super::Error;

pub(super) struct Adam {
    pub gradients: [Vec<f32>; 3],
    first: [Vec<f32>; 3],
    second: [Vec<f32>; 3],
    steps: i32,
}
impl Adam {
    pub fn new(lengths: [usize; 3]) -> Result<Self, Error> {
        if lengths.contains(&0)
            || lengths
                .iter()
                .try_fold(0usize, |n, v| n.checked_add(*v))
                .is_none_or(|n| n > 500_000)
        {
            return Err(Error::Budget);
        }
        let zeros = || lengths.map(|n| vec![0.0; n]);
        Ok(Self {
            gradients: zeros(),
            first: zeros(),
            second: zeros(),
            steps: 0,
        })
    }
    pub fn steps(&self) -> usize {
        self.steps as usize
    }
    pub fn step(
        &mut self,
        parameters: [&mut [f32]; 3],
        sentences: usize,
        rate: f32,
    ) -> Result<(), Error> {
        if !(1..=64).contains(&sentences)
            || !rate.is_finite()
            || !(0.0..=1.0).contains(&rate)
            || self.steps >= 100_000
            || parameters
                .iter()
                .zip(&self.gradients)
                .any(|(p, g)| p.len() != g.len() || p.iter().chain(g).any(|v| !v.is_finite()))
        {
            return Err(Error::Invalid("Adam batch/parameters"));
        }
        let divisor = sentences as f64;
        let norm = self
            .gradients
            .iter()
            .flatten()
            .map(|g| (f64::from(*g) / divisor).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = (1.0 / norm.max(1.0) / divisor) as f32;
        self.steps += 1;
        let correction1 = 1.0 - 0.9f64.powi(self.steps);
        let correction2 = 1.0 - 0.999f64.powi(self.steps);
        for (group, values) in parameters.into_iter().enumerate() {
            let bound = if group == 2 { 16.0 } else { 8.0 };
            for (i, value) in values.iter_mut().enumerate() {
                let gradient = self.gradients[group][i] * scale;
                let m = &mut self.first[group][i];
                let v = &mut self.second[group][i];
                *m = 0.9 * *m + 0.1 * gradient;
                *v = 0.999 * *v + 0.001 * gradient * gradient;
                let update =
                    (f64::from(*m) / correction1) / ((f64::from(*v) / correction2).sqrt() + 1e-8);
                *value = (*value - rate * update as f32).clamp(-bound, bound);
                self.gradients[group][i] = 0.0;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adam_matches_two_scalar_reference_steps_and_clears_batch() {
        let mut state = Adam::new([1; 3]).unwrap();
        let (mut x, mut y, mut z) = ([1.0], [2.0], [3.0]);
        let (mut first, mut second, mut reference) = (0.0f64, 0.0f64, 1.0f64);
        for (i, g) in [0.3f32, -0.2].into_iter().enumerate() {
            state.gradients[0][0] = 2.0 * g;
            first = 0.9 * first + 0.1 * f64::from(g);
            second = 0.999 * second + 0.001 * f64::from(g).powi(2);
            reference -= 0.01 * (first / (1.0 - 0.9f64.powi(i as i32 + 1)))
                / ((second / (1.0 - 0.999f64.powi(i as i32 + 1))).sqrt() + 1e-8);
            state.step([&mut x, &mut y, &mut z], 2, 0.01).unwrap();
            assert!((f64::from(x[0]) - reference).abs() < 1e-6);
            assert!(state.gradients.iter().flatten().all(|v| *v == 0.0));
        }
        assert_eq!(y, [2.0]);
        assert_eq!(z, [3.0]);
    }
    #[test]
    fn adam_global_clip_zero_batch_and_invalid_step_are_bounded() {
        let mut state = Adam::new([1; 3]).unwrap();
        let (mut x, mut y, mut z) = ([0.0], [0.0], [0.0]);
        state.gradients[0][0] = 6.0;
        state.gradients[1][0] = 8.0;
        state.step([&mut x, &mut y, &mut z], 1, 0.1).unwrap();
        assert!((state.first[0][0] - 0.06).abs() < 1e-6);
        assert!((state.first[1][0] - 0.08).abs() < 1e-6);
        state.gradients[2][0] = f32::NAN;
        let before = (x, y, z, state.steps);
        assert!(state.step([&mut x, &mut y, &mut z], 1, 0.1).is_err());
        assert_eq!((x, y, z, state.steps), before);
        assert!(state.step([&mut x, &mut y, &mut z], 0, 0.1).is_err());
        assert!(Adam::new([500_000, 1, 1]).is_err());
    }
}
