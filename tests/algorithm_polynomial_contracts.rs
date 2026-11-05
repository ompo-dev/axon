#![cfg(feature = "curriculum")]
use axon_uic::structural::{
    Error, StructuralSeed,
    agent::{Agent, AlgorithmLesson, Budget},
    algorithm::{Example, Program, Representation, SearchBudget, Value, learn_polynomial},
};

fn rows(xs: &[i64], f: impl Fn(i64) -> i64) -> Vec<Example> {
    xs.iter()
        .map(|&x| Example {
            input: Value::Integer(x),
            output: Value::Integer(f(x)),
        })
        .collect()
}
fn evaluate(p: &Program, x: i64) -> i64 {
    let Value::Integer(n) = p.evaluate(&Value::Integer(x), 10000).unwrap().value else {
        panic!()
    };
    n
}

#[test]
fn exact_cubics_transfer_with_unseen_coefficients_and_roundtrip() {
    for k in -12..=12 {
        let f = |x| (k * x + 7) * x * x - 13 * x + 19;
        let receipt =
            learn_polynomial(&rows(&[-3, -1, 0, 2, 4, 5], f), SearchBudget::default()).unwrap();
        let restored: Program =
            serde_json::from_slice(&serde_json::to_vec(&receipt.program).unwrap()).unwrap();
        for x in [-4097, -17, -7, 1, 7, 11, 29, 200] {
            assert_eq!(evaluate(&restored, x), f(x));
        }
        assert!(receipt.work < 30000000 && receipt.candidates <= 4);
        assert!(!receipt.inherited);
    }
}

#[test]
fn exact_bivariate_cubic_interactions_transfer_off_training_triangle() {
    let f = |x: i64, y: i64| 7 * x * x * y - 11 * x * y * y + 13 * y * y * y - 17 * x + 23 * y - 29;
    let examples: Vec<_> = (0..=3)
        .flat_map(|d| (0..=d).map(move |y| (d - y, y)))
        .map(|(x, y)| Example {
            input: Value::List(vec![Value::Integer(x), Value::Integer(y)]),
            output: Value::Integer(f(x, y)),
        })
        .collect();
    let receipt = learn_polynomial(&examples, SearchBudget::default()).unwrap();
    for (x, y) in [(-7, 3), (9, 11), (0, -17), (21, -4)] {
        assert_eq!(
            receipt
                .program
                .evaluate(
                    &Value::List(vec![Value::Integer(x), Value::Integer(y)]),
                    10000
                )
                .unwrap()
                .value,
            Value::Integer(f(x, y))
        );
    }
}

#[test]
fn refutation_repairs_ambiguous_samples_and_seed_retains_only_program() {
    let teaching = rows(&[-1, 0, 1], |x| x * x * x);
    let admission = rows(&[2, 3, 4, 5], |x| x * x * x);
    let wrong = learn_polynomial(&teaching, SearchBudget::default()).unwrap();
    assert_eq!(evaluate(&wrong.program, 2), 2); // x and x^3 initially collide.
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    let mut phases = Vec::new();
    let result = agent
        .learn_algorithm_represented(
            AlgorithmLesson {
                name: "cubo",
                examples: &teaching,
                admission: &admission,
            },
            Representation::Polynomial,
            SearchBudget::default(),
            false,
            &mut |event| {
                phases.push(event.phase.clone());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(result.attempts, 2);
    assert_eq!(phases.iter().filter(|s| *s == "refutation").count(), 1);
    let bytes = agent.seed().to_bytes().unwrap();
    let restored = StructuralSeed::from_bytes(&bytes).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), bytes);
    assert_eq!(
        evaluate(
            restored
                .rules()
                .iter()
                .find_map(|r| r.algorithm_program())
                .unwrap(),
            19
        ),
        6859
    );
}

#[test]
fn out_of_class_and_conflicting_evidence_fail_without_commit() {
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    let before = agent.seed().to_bytes().unwrap();
    let teaching = rows(&[-2, -1, 0, 1, 2, 3], |x| x.pow(4));
    let admission = rows(&[4, 5, 6], |x| x.pow(4));
    assert!(
        agent
            .learn_algorithm_represented(
                AlgorithmLesson {
                    name: "quarto",
                    examples: &teaching,
                    admission: &admission
                },
                Representation::Polynomial,
                SearchBudget::default(),
                true,
                &mut |_| Ok(()),
            )
            .is_err()
    );
    assert_eq!(agent.seed().to_bytes().unwrap(), before);
    let mut conflicting = rows(&[0, 1, 2], |x| x + 1);
    conflicting.push(Example {
        input: Value::Integer(1),
        output: Value::Integer(99),
    });
    assert!(learn_polynomial(&conflicting, SearchBudget::default()).is_err());
    assert!(
        learn_polynomial(
            &rows(&[-4, -2, 0, 2, 4, 6], |x| x / 2),
            SearchBudget::default()
        )
        .is_err()
    );
}

#[test]
fn a_fit_cannot_certify_an_unknown_domain_or_detect_hidden_quartic() {
    let examples = rows(&[-1, 0, 1], |x| x * x * x + x * x * (x - 1) * (x + 1));
    let p = learn_polynomial(&examples, SearchBudget::default())
        .unwrap()
        .program;
    assert_eq!(evaluate(&p, 2), 2);
    assert_ne!(evaluate(&p, 2), 20); // publish this false generalization, do not call it understanding.
}

#[test]
fn work_candidates_coordinate_shapes_and_execution_overflow_are_bounded() {
    let teaching = rows(&[-3, -1, 0, 2, 4, 5], |x| 7 * x * x * x + 19);
    for (candidates, work) in [(1, 30000000), (50000, 1)] {
        let e = learn_polynomial(
            &teaching,
            SearchBudget {
                candidates,
                work,
                ..SearchBudget::default()
            },
        )
        .unwrap_err();
        assert!(matches!(e.error, Error::Budget));
        assert!(e.work <= work && e.candidates <= candidates as u64);
    }
    for xs in [&[0, 1][..], &[0, 1, 4097][..], &[0, 1, i64::MIN][..]] {
        assert!(learn_polynomial(&rows(xs, |_| 1), SearchBudget::default()).is_err());
    }
    let mut bad = teaching.clone();
    bad[0].input = Value::List(vec![Value::Integer(0); 3]);
    assert!(learn_polynomial(&bad, SearchBudget::default()).is_err());
    let p = learn_polynomial(&teaching, SearchBudget::default())
        .unwrap()
        .program;
    assert!(p.evaluate(&Value::Integer(i64::MAX), 10000).is_err());
}

#[test]
fn event_failure_keeps_existing_seed_atomic() {
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    let before = agent.seed().to_bytes().unwrap();
    assert!(
        agent
            .learn_algorithm_represented(
                AlgorithmLesson {
                    name: "nova",
                    examples: &rows(&[0, 1, 2], |x| x + 17),
                    admission: &rows(&[3, 4], |x| x + 17)
                },
                Representation::Polynomial,
                SearchBudget::default(),
                true,
                &mut |e| {
                    if e.phase == "consolidation" {
                        Err(Error::Budget)
                    } else {
                        Ok(())
                    }
                },
            )
            .is_err()
    );
    assert_eq!(agent.seed().to_bytes().unwrap(), before);
}
