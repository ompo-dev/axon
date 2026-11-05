#![cfg(feature = "curriculum")]
use axon_uic::structural::{
    StructuralSeed,
    agent::{Agent, Budget},
    algorithm::{Example, Program, SearchBudget, Value, learn},
};
use serde_json::{Value as Json, json};

fn value(v: Json) -> Value {
    serde_json::from_value(v).unwrap()
}

fn samples(reduce: bool, column: usize, equal: bool, weighted: bool) -> Vec<Example> {
    let inputs = [
        json!([[[0, 12, 13], [2, 14, 12], [0, 13, 17]], 1]),
        json!([[[1, -12, 14], [0, 15, 12], [1, 17, -11]], 1]),
        json!([[[1, 16, 12], [2, 11, 19], [0, -13, -12]], 2]),
        json!([[[0, 14, -11], [1, 12, 15], [2, 17, 12]], 1]),
        json!([[[1, -14, 13], [1, 12, -12], [2, 11, 11]], 1]),
    ];
    inputs
        .into_iter()
        .map(|input| {
            let limit = input[1].as_i64().unwrap();
            let results: Vec<i64> = input[0]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| {
                    let k = r[0].as_i64().unwrap();
                    if equal { k == limit } else { k < limit }
                })
                .map(|r| {
                    if weighted {
                        r[1].as_i64().unwrap() * r[2].as_i64().unwrap()
                    } else {
                        r[column].as_i64().unwrap()
                    }
                })
                .collect();
            Example {
                input: value(input),
                output: value(if reduce {
                    json!(results.iter().sum::<i64>())
                } else {
                    json!(results)
                }),
            }
        })
        .collect()
}

fn acquire(examples: &[Example], library: &[Program]) -> Program {
    let found = learn(&examples[..3], library, SearchBudget::default())
        .unwrap_or_else(|e| panic!("acquisition with {} source programs: {e}", library.len()));
    for e in &examples[3..] {
        assert_eq!(
            found.program.evaluate(&e.input, 100_000).unwrap().value,
            e.output
        );
    }
    found.program
}

#[test]
fn learned_structure_transfers_to_conditional_record_transformations() {
    check_transfer(false);
}

#[test]
fn learned_structure_transfers_to_conditional_weighted_reductions() {
    check_transfer(true);
}

fn check_transfer(reduce: bool) {
    let a = acquire(&samples(reduce, 1, false, false), &[]);
    let b = acquire(&samples(reduce, 2, true, false), &[]);
    let target = samples(reduce, 0, false, true);
    assert!(learn(&target[..3], &[], SearchBudget::default()).is_err());
    let duplicates = learn(
        &target[..3],
        &[a.clone(), a.clone()],
        SearchBudget::default(),
    )
    .unwrap_err();
    assert_eq!(duplicates.templates, 0);
    let unrelated: Vec<_> = [false, true]
        .into_iter()
        .map(|product| {
            let examples: Vec<_> = [vec![12_i64, 13], vec![16, -12], vec![11, 17, 14]]
                .into_iter()
                .map(|xs| Example {
                    output: value(if product {
                        json!(xs.iter().product::<i64>())
                    } else {
                        json!(xs.iter().sum::<i64>())
                    }),
                    input: value(json!(xs)),
                })
                .collect();
            learn(&examples, &[], SearchBudget::default())
                .unwrap()
                .program
        })
        .collect();
    assert!(learn(&target[..3], &unrelated, SearchBudget::default()).is_err());
    let library = [a, b];
    let receipt = learn(&target[..3], &library, SearchBudget::default()).unwrap();
    assert!(receipt.abstracted);
    assert!(receipt.templates > 0 && receipt.templates <= 64);
    assert!(receipt.mining_work > 0 && receipt.mining_work < receipt.work);
    let again = learn(&target[..3], &library, SearchBudget::default()).unwrap();
    assert_eq!(receipt.program, again.program);
    assert_eq!(receipt.work, again.work);
    let program = acquire(&target, &library);
    for length in 0..40 {
        for limit in -3..5 {
            let rows: Vec<[i64; 3]> = (0..length)
                .map(|i| [(i * 17 % 13) - 6, (i * 3 % 11) - 5, (i * 7 % 17) - 8])
                .collect();
            let results: Vec<i64> = rows
                .iter()
                .filter(|r| r[0] < limit)
                .map(|r| r[1] * r[2])
                .collect();
            let expected = if reduce {
                json!(results.iter().sum::<i64>())
            } else {
                json!(results)
            };
            assert_eq!(
                program
                    .evaluate(&value(json!([rows, limit])), 100_000)
                    .unwrap()
                    .value,
                value(expected)
            );
        }
    }
}

#[test]
fn descendant_reconstructs_abstractions_from_compiled_sources_without_parent_examples() {
    let mut parent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    let previous: Json =
        serde_json::from_str(include_str!("../evals/curricula/dependencies-v1-data.json")).unwrap();
    for lesson in previous["lessons"].as_array().unwrap() {
        let examples: Vec<Example> = serde_json::from_value(lesson["examples"].clone()).unwrap();
        let admission: Vec<Example> = serde_json::from_value(lesson["admission"].clone()).unwrap();
        parent
            .learn_algorithm(
                lesson["id"].as_str().unwrap(),
                &examples,
                &admission,
                SearchBudget::default(),
                true,
                &mut |_| Ok(()),
            )
            .unwrap();
    }
    for (name, reduce, column, equal) in [
        ("project-before", false, 1, false),
        ("project-equal", false, 2, true),
        ("total-before", true, 1, false),
        ("total-equal", true, 2, true),
    ] {
        let rows = samples(reduce, column, equal, false);
        parent
            .learn_algorithm(
                name,
                &rows[..3],
                &rows[3..],
                SearchBudget::default(),
                true,
                &mut |_| Ok(()),
            )
            .unwrap();
    }
    let bytes = parent.seed().to_bytes().unwrap();
    drop(parent);
    let mut child = Agent::new(
        StructuralSeed::from_bytes(&bytes).unwrap(),
        Budget::default(),
    )
    .unwrap();
    for (name, reduce) in [("weighted-list", false), ("weighted-total", true)] {
        let rows = samples(reduce, 0, false, true);
        let acquired = child
            .learn_algorithm(
                name,
                &rows[..3],
                &rows[3..],
                SearchBudget::default(),
                true,
                &mut |_| Ok(()),
            )
            .unwrap();
        if reduce {
            assert!(acquired.inherited);
        } else {
            assert!(acquired.abstracted);
        }
        assert!(acquired.mining_work > 0);
        assert_eq!(acquired.confirmations, 2);
    }
    let reopened = StructuralSeed::from_bytes(&child.seed().to_bytes().unwrap()).unwrap();
    assert_eq!(reopened.rules().len(), 25);
}

#[test]
fn mining_and_failed_acquisition_cannot_exceed_the_existing_work_quota() {
    let a = acquire(&samples(false, 1, false, false), &[]);
    let b = acquire(&samples(false, 2, true, false), &[]);
    let target = samples(false, 0, false, true);
    let failure = learn(
        &target[..3],
        &[a, b],
        SearchBudget {
            work: 1000,
            ..SearchBudget::default()
        },
    )
    .unwrap_err();
    assert!(failure.work <= 1000);
    assert!(failure.mining_work > 0 && failure.mining_work <= failure.work);
    assert!(failure.candidates <= SearchBudget::default().candidates as u64);
}
