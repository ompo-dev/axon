#![cfg(feature = "curriculum")]
use axon_uic::structural::algorithm::{Example, Program, SearchBudget, learn};
use serde_json::{Value as Json, json};

fn examples(rows: Vec<(Json, Json)>) -> Vec<Example> {
    rows.into_iter()
        .map(|(input, output)| Example {
            input: serde_json::from_value(input).unwrap(),
            output: serde_json::from_value(output).unwrap(),
        })
        .collect()
}
fn acquire(rows: &[Example], library: &[Program]) -> Program {
    let receipt = learn(&rows[..3], library, SearchBudget::default()).unwrap();
    for row in &rows[3..] {
        assert_eq!(
            receipt.program.evaluate(&row.input, 100000).unwrap().value,
            row.output
        );
    }
    receipt.program
}
#[test]
fn learned_constructors_can_initialize_a_different_list_transformation() {
    let wrap = acquire(
        &examples(vec![
            (json!(-4), json!([-4])),
            (json!(0), json!([0])),
            (json!(7), json!([7])),
            (json!(12), json!([12])),
            (json!(-9), json!([-9])),
        ]),
        &[],
    );
    let rows = examples(vec![
        (json!([[], 3]), json!([3])),
        (json!([[1, 2], 7]), json!([7, 1, 2])),
        (json!([[-4, 5, 1], 0]), json!([0, -4, 5, 1])),
        (json!([[2], 9]), json!([9, 2])),
        (json!([[-1, 8], -3]), json!([-3, -1, 8])),
    ]);
    acquire(&rows, &[wrap]);
}
#[test]
fn structured_initial_values_and_local_updates_are_induced_within_existing_types() {
    acquire(
        &examples(vec![
            (json!(-4), json!([-4, []])),
            (json!(0), json!([0, []])),
            (json!(7), json!([7, []])),
            (json!(12), json!([12, []])),
            (json!(-9), json!([-9, []])),
        ]),
        &[],
    );
    acquire(
        &examples(vec![
            (json!([[0, []], 3]), json!([3, [3]])),
            (json!([[5, [2, 5]], -2]), json!([3, [2, 5, 3]])),
            (json!([[-3, [9]], 7]), json!([4, [9, 4]])),
            (json!([[4, [1, -8]], 0]), json!([4, [1, -8, 4]])),
            (json!([[7, []], -5]), json!([2, [2]])),
        ]),
        &[],
    );
}
