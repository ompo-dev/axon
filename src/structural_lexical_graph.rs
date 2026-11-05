//! Train-only soft lexical prior, not sense disambiguation or safe word substitution.
use super::{Error, Model};
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub weight: f32,
    pub interval: usize,
    pub edges: Vec<[String; 2]>,
}
impl Graph {
    pub(super) fn prepare(&self, model: &Model) -> Result<Vec<[usize; 2]>, Error> {
        if !self.weight.is_finite()
            || !(0.0..=1.0).contains(&self.weight)
            || !(1..=64).contains(&self.interval)
            || !(1..=50000).contains(&self.edges.len())
        {
            return Err(Error::Invalid("lexical graph budget"));
        }
        let mut seen = BTreeSet::new();
        let mut edges = Vec::with_capacity(self.edges.len());
        for pair in &self.edges {
            if pair[0] >= pair[1] || !seen.insert(pair) {
                return Err(Error::Invalid("lexical graph pair"));
            }
            let rows = pair
                .each_ref()
                .map(|word| model.lexicon.index(word).map(|i| i + 3));
            edges.push([
                rows[0].ok_or(Error::Invalid("lexical graph vocabulary"))?,
                rows[1].ok_or(Error::Invalid("lexical graph vocabulary"))?,
            ]);
        }
        Ok(edges)
    }
}
pub(super) fn apply(
    input: &mut [f32],
    dimensions: usize,
    edge: [usize; 2],
    rate: f32,
) -> Result<(), Error> {
    if !(8..=16).contains(&dimensions)
        || !input.len().is_multiple_of(dimensions)
        || edge[0] == edge[1]
        || edge.iter().any(|r| *r >= input.len() / dimensions)
        || !rate.is_finite()
        || !(0.0..=1.0).contains(&rate)
    {
        return Err(Error::Invalid("lexical graph update"));
    }
    for row in edge {
        if input[row * dimensions..(row + 1) * dimensions]
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 8.0)
        {
            return Err(Error::Invalid("lexical graph vector"));
        }
    }
    for d in 0..dimensions {
        let (a, b) = (edge[0] * dimensions + d, edge[1] * dimensions + d);
        let difference = input[a] - input[b];
        input[a] = (input[a] - rate * difference).clamp(-8.0, 8.0);
        input[b] = (input[b] + rate * difference).clamp(-8.0, 8.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn symmetric_pair_step_matches_gradient_and_never_mutates_other_rows() {
        let initial = (0..24).map(|i| (i as f32 - 12.0) * 0.1).collect::<Vec<_>>();
        let mut input = initial.clone();
        let loss = |x: &[f32]| {
            (0..8)
                .map(|d| 0.5 * (x[d] - x[16 + d]).powi(2))
                .sum::<f32>()
        };
        apply(&mut input, 8, [0, 2], 0.01).unwrap();
        assert_eq!(&input[8..16], &initial[8..16]);
        assert!(loss(&input) < loss(&initial));
        for i in (0..8).chain(16..24) {
            let mut hi = initial.clone();
            let mut lo = initial.clone();
            hi[i] += 0.001;
            lo[i] -= 0.001;
            assert!(
                ((initial[i] - input[i]) / 0.01 - (loss(&hi) - loss(&lo)) / 0.002).abs() < 0.001
            );
        }
        let before = input.clone();
        assert!(apply(&mut input, 8, [0, 3], 0.01).is_err());
        assert_eq!(input, before);
        assert!(apply(&mut input, 8, [0, 0], 0.01).is_err());
        assert_eq!(input, before);
    }
}
