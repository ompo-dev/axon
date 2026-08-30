/// Generalização fora das combinações observadas de dois fatores independentes.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizationReport {
    pub training_cases: u32,
    pub holdout_cases: u32,
    pub lookup_holdout_accuracy: f32,
    pub factorized_holdout_accuracy: f32,
}

pub(super) fn run() -> GeneralizationReport {
    let all = (0..4)
        .flat_map(|left| (0..4).map(move |right| Example::new(left, right)))
        .collect::<Vec<_>>();
    let holdout_positions = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 3),
        (2, 0),
        (2, 3),
        (3, 2),
        (3, 3),
    ];
    let (holdout, training): (Vec<_>, Vec<_>) = all
        .into_iter()
        .partition(|example| holdout_positions.contains(&(example.left, example.right)));
    let lookup = AssociativeLookup::learn(&training);
    let factorized = LocalFactorRule::learn(&training, 80);

    GeneralizationReport {
        training_cases: training.len() as u32,
        holdout_cases: holdout.len() as u32,
        lookup_holdout_accuracy: accuracy(&holdout, |example| lookup.predict(example)),
        factorized_holdout_accuracy: accuracy(&holdout, |example| factorized.predict(example)),
    }
}

#[derive(Clone, Copy)]
struct Example {
    left: usize,
    right: usize,
    label: i8,
}

impl Example {
    fn new(left: usize, right: usize) -> Self {
        const WEIGHTS: [i8; 4] = [-3, -1, 1, 3];
        Self {
            left,
            right,
            label: if WEIGHTS[left] + WEIGHTS[right] > 0 {
                1
            } else {
                -1
            },
        }
    }
}

struct AssociativeLookup {
    seen: [[Option<i8>; 4]; 4],
    fallback: i8,
}

impl AssociativeLookup {
    fn learn(training: &[Example]) -> Self {
        let positives = training.iter().filter(|example| example.label > 0).count();
        let fallback = if positives * 2 >= training.len() {
            1
        } else {
            -1
        };
        let mut seen = [[None; 4]; 4];
        for example in training {
            seen[example.left][example.right] = Some(example.label);
        }
        Self { seen, fallback }
    }

    fn predict(&self, example: &Example) -> i8 {
        self.seen[example.left][example.right].unwrap_or(self.fallback)
    }
}

/// Delta rule local: apenas os dois fatores ativos recebem o erro do exemplo.
struct LocalFactorRule {
    left: [f32; 4],
    right: [f32; 4],
}

impl LocalFactorRule {
    fn learn(training: &[Example], epochs: u32) -> Self {
        let mut model = Self {
            left: [0.0; 4],
            right: [0.0; 4],
        };
        for _ in 0..epochs {
            for example in training {
                let prediction = model.predict(example);
                if prediction != example.label {
                    let delta = 0.25 * f32::from(example.label);
                    model.left[example.left] += delta;
                    model.right[example.right] += delta;
                }
            }
        }
        model
    }

    fn predict(&self, example: &Example) -> i8 {
        if self.left[example.left] + self.right[example.right] > 0.0 {
            1
        } else {
            -1
        }
    }
}

fn accuracy(examples: &[Example], predict: impl Fn(&Example) -> i8) -> f32 {
    examples
        .iter()
        .filter(|example| predict(example) == example.label)
        .count() as f32
        / examples.len() as f32
}
