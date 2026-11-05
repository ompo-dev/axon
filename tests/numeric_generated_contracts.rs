use axon_uic::{behavior_lab::Rng, numeric::*};
#[derive(Debug)]
enum Truth {
    Input(usize),
    Constant(f64),
    Binary(u8, Box<Truth>, Box<Truth>),
}
impl Truth {
    fn generate(rng: &mut Rng, arity: usize, cost: usize) -> Self {
        if cost == 1 {
            return if rng.pick(4) == 0 {
                Self::Constant(rng.pick(3) as f64)
            } else {
                Self::Input(rng.pick(arity))
            };
        }
        let left_cost = 1 + 2 * rng.pick((cost - 1) / 2);
        Self::Binary(
            rng.pick(3) as u8,
            Box::new(Self::generate(rng, arity, left_cost)),
            Box::new(Self::generate(rng, arity, cost - 1 - left_cost)),
        )
    }
    fn evaluate(&self, input: &[f64]) -> f64 {
        match self {
            Self::Input(i) => input[*i],
            Self::Constant(n) => *n,
            Self::Binary(op, a, b) => {
                let (a, b) = (a.evaluate(input), b.evaluate(input));
                match op {
                    0 => a + b,
                    1 => a - b,
                    _ => a * b,
                }
            }
        }
    }
}
#[test]
fn programs_generated_without_family_labels_are_learned_then_checked_on_fractional_extrapolation() {
    let mut goals = Rng::new(37489153);
    let mut training_rng = Rng::new(17381921);
    let mut testing_rng = Rng::new(99371659);
    for case in 0..40 {
        let arity = 2 + goals.pick(2);
        let cost = 3 + 2 * goals.pick(3);
        let truth = Truth::generate(&mut goals, arity, cost);
        let rows = (0..10)
            .map(|_| {
                let input = (0..arity)
                    .map(|_| training_rng.pick(21) as f64 - 10.0)
                    .collect::<Vec<_>>();
                let output = truth.evaluate(&input);
                NumericExample::new(input, output)
            })
            .collect();
        let skill = NumericSkill::learn(
            format!("opaque-{case}"),
            "generated/arithmetic",
            rows,
            &[],
            NumericBudget::default(),
            NumericTolerance::default(),
        )
        .unwrap_or_else(|e| panic!("goal {truth:?}: {e}"));
        let execution = skill.program().compile_execution();
        for _ in 0..512 {
            let input = (0..arity)
                .map(|_| (testing_rng.pick(1999) as f64 - 999.0) / 7.0)
                .collect::<Vec<_>>();
            let expected = truth.evaluate(&input);
            let value = skill.predict("generated/arithmetic", &input).unwrap();
            assert_eq!(
                value.to_bits(),
                execution.evaluate(&input).unwrap().to_bits()
            );
            assert!(
                skill.tolerance().agrees(value, expected),
                "truth {truth:?}; learned {}; input {input:?}; {value} != {expected}",
                skill.program().expression()
            );
        }
    }
}
