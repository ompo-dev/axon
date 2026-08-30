/// Evidência de robustez (ou perda) da álgebra de ligação hiperdimensional.
#[derive(Clone, Debug, PartialEq)]
pub struct HypercellReport {
    pub trials: u32,
    pub dimension: usize,
    pub dense_exact_recovery: f32,
    pub dense_noisy_retrieval: f32,
    pub sparse_exact_recovery: f32,
    pub sparse_signal_retention: f32,
}

pub(super) fn run() -> HypercellReport {
    const TRIALS: u32 = 96;
    const DIMENSION: usize = 257;
    const CODEBOOK: usize = 32;
    const NOISY_FLIPS: usize = 8;

    let mut rng = Lcg::new(0xA81E_B001);
    let mut dense_exact = 0_u32;
    let mut dense_noisy = 0_u32;
    let mut sparse_exact = 0_u32;
    let mut sparse_retention_sum = 0.0_f32;

    for _ in 0..TRIALS {
        let codebook = (0..CODEBOOK)
            .map(|_| dense_vector(&mut rng, DIMENSION))
            .collect::<Vec<_>>();
        let target_index = rng.below(CODEBOOK as u32) as usize;
        let target = &codebook[target_index];
        let dense_key = dense_vector(&mut rng, DIMENSION);
        let dense_bound = bind(target, &dense_key);
        let dense_recovered = bind(&dense_bound, &dense_key);
        dense_exact += u32::from(dense_recovered == *target);

        let mut noisy_bound = dense_bound.clone();
        for _ in 0..NOISY_FLIPS {
            let position = rng.below(DIMENSION as u32) as usize;
            noisy_bound[position] = -noisy_bound[position];
        }
        let noisy_recovered = bind(&noisy_bound, &dense_key);
        dense_noisy += u32::from(nearest(&noisy_recovered, &codebook) == target_index);

        let sparse_value = sparse_vector(&mut rng, DIMENSION);
        let sparse_key = sparse_vector(&mut rng, DIMENSION);
        let sparse_recovered = bind(&bind(&sparse_value, &sparse_key), &sparse_key);
        sparse_exact += u32::from(sparse_recovered == sparse_value);
        sparse_retention_sum += active_agreement(&sparse_value, &sparse_recovered);
    }

    HypercellReport {
        trials: TRIALS,
        dimension: DIMENSION,
        dense_exact_recovery: dense_exact as f32 / TRIALS as f32,
        dense_noisy_retrieval: dense_noisy as f32 / TRIALS as f32,
        sparse_exact_recovery: sparse_exact as f32 / TRIALS as f32,
        sparse_signal_retention: sparse_retention_sum / TRIALS as f32,
    }
}

fn dense_vector(rng: &mut Lcg, dimension: usize) -> Vec<i8> {
    (0..dimension)
        .map(|_| if rng.below(2) == 0 { -1 } else { 1 })
        .collect()
}

fn sparse_vector(rng: &mut Lcg, dimension: usize) -> Vec<i8> {
    (0..dimension)
        .map(|_| match rng.below(4) {
            0 => -1,
            1 => 1,
            _ => 0,
        })
        .collect()
}

fn bind(left: &[i8], right: &[i8]) -> Vec<i8> {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            if *left == 0 || *right == 0 {
                0
            } else {
                left * right
            }
        })
        .collect()
}

fn nearest(query: &[i8], codebook: &[Vec<i8>]) -> usize {
    codebook
        .iter()
        .enumerate()
        .max_by_key(|(_, candidate)| dot(query, candidate))
        .map(|(index, _)| index)
        .expect("experiment codebook must be nonempty")
}

fn dot(left: &[i8], right: &[i8]) -> i32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| i32::from(*left) * i32::from(*right))
        .sum()
}

fn active_agreement(original: &[i8], recovered: &[i8]) -> f32 {
    let active = original
        .iter()
        .zip(recovered)
        .filter(|(original, _)| **original != 0)
        .collect::<Vec<_>>();
    if active.is_empty() {
        return 1.0;
    }
    active
        .iter()
        .filter(|(original, recovered)| **original == **recovered)
        .count() as f32
        / active.len() as f32
}

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn below(&mut self, upper: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 32) as u32) % upper
    }
}
