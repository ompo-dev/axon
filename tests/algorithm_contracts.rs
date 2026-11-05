#![cfg(feature = "curriculum")]
use axon_uic::structural::Example;
use axon_uic::structural::{
    agent::{Agent, Budget},
    algorithm::*,
    *,
};
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;
fn value(v: Json) -> Value {
    serde_json::from_value(v).unwrap()
}
fn example(x: Json, y: Json) -> algorithm::Example {
    algorithm::Example {
        input: value(x),
        output: value(y),
    }
}
fn b(e: Expr) -> Box<Expr> {
    Box::new(e)
}
fn i(n: i64) -> Expr {
    Expr::Literal(Value::Integer(n))
}
fn grammar() -> StructuralSeed {
    let rows = vec![
        (
            json!(["goal", "aplique", "alpha", "em", "dados"]),
            json!(["invoke", "alpha", "dados"]),
        ),
        (
            json!(["goal", "aplique", "beta", "em", "origem"]),
            json!(["invoke", "beta", "origem"]),
        ),
        (
            json!([
                "goal",
                "aplique",
                "gama",
                "em",
                ["goal", "aplique", "alfa", "em", "arquivo"]
            ]),
            json!([
                "invoke",
                "gama",
                ["goal", "aplique", "alfa", "em", "arquivo"]
            ]),
        ),
        (
            json!([
                "goal",
                "aplique",
                "eta",
                "em",
                ["goal", "aplique", "sigma", "em", "lote"]
            ]),
            json!(["invoke", "eta", ["goal", "aplique", "sigma", "em", "lote"]]),
        ),
    ];
    let examples = rows
        .into_iter()
        .map(|(input, output)| Example {
            input: serde_json::from_value(input).unwrap(),
            output: serde_json::from_value(output).unwrap(),
        })
        .collect();
    let mut seed = StructuralSeed::default();
    seed.teach(
        Lesson {
            id: "requests".into(),
            examples,
            guards: vec![],
        },
        &[Example {
            input: serde_json::from_value(json!(["goal", "aplique", "delta", "em", "registro"]))
                .unwrap(),
            output: serde_json::from_value(json!(["invoke", "delta", "registro"])).unwrap(),
        }],
    )
    .unwrap();
    seed
}
fn sum_examples() -> Vec<algorithm::Example> {
    vec![
        example(json!([]), json!(0)),
        example(json!([3, 5, -2]), json!(6)),
        example(json!([9, -4]), json!(5)),
    ]
}
#[test]
fn loop_programs_are_exact_lazy_bounded_and_roundtrip_without_training_data() {
    let p = Program::new(Expr::Fold(
        b(Expr::Input),
        b(i(0)),
        b(Expr::Add(b(Expr::Acc(0)), b(Expr::Item(0)))),
    ))
    .unwrap();
    let result = p.evaluate(&value(json!([3, -8, 12])), 1000).unwrap();
    assert_eq!(result.value, value(json!(7)));
    let input = Value::List((0..2048).map(Value::Integer).collect());
    assert_eq!(
        p.evaluate(&input, 100000).unwrap().value,
        Value::Integer(2047 * 2048 / 2)
    );
    assert!(p.evaluate(&input, 10).is_err());
    assert!(p.evaluate(&value(json!([i64::MAX, 1])), 10000).is_err());
    let encoded = serde_json::to_vec(&p).unwrap();
    let reopened: Program = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(p, reopened);
    let lazy = Program::new(Expr::If(
        b(Expr::Literal(Value::Boolean(true))),
        b(i(7)),
        b(Expr::Remainder(b(i(3)), b(i(0)))),
    ))
    .unwrap();
    assert_eq!(
        lazy.evaluate(&value(json!([])), 100).unwrap().value,
        value(json!(7))
    );
    assert!(Program::new(Expr::Item(0)).is_err());
    assert!(Program::new(Expr::Literal(value(json!([1, 2, 3])))).is_err());
    let explosive = Program::new(Expr::Fold(
        b(Expr::Input),
        b(Expr::Empty),
        b(Expr::Push(b(Expr::Acc(0)), b(Expr::Acc(0)))),
    ))
    .unwrap();
    assert!(
        explosive
            .evaluate(&Value::List(vec![value(json!(1)); 32]), 100000)
            .is_err()
    );
}
#[test]
fn large_values_use_pages_without_relaxing_existing_tree_limits() {
    for n in [0, 1, 30, 31, 900, 2048] {
        let v = Value::List((0..n).map(|i| Value::Integer(i as i64)).collect());
        let t = v.to_term().unwrap();
        t.validate().unwrap();
        assert_eq!(Value::from_term(&t).unwrap(), v);
    }
    for v in [
        value(json!(["empty", true, [1, "red"], []])),
        value(json!([["x", 2], ["y", 3]])),
    ] {
        assert_eq!(Value::from_term(&v.to_term().unwrap()).unwrap(), v);
    }
    assert!(Value::List(vec![value(json!(1)); 2049]).validate().is_err());
}
#[test]
fn compact_transport_fits_large_records_without_loosening_value_or_tree_quotas() {
    let v = Value::List(vec![value(json!([1, 2, 3])); 600]);
    v.validate().unwrap();
    assert!(v.to_term().is_err());
    let compact = v.to_compact_term().unwrap();
    compact.validate().unwrap();
    assert_eq!(Value::from_term(&compact).unwrap(), v);
    let small = value(json!([[1, 2], true, "items", []]));
    let legacy = small.to_term().unwrap();
    let dense = small.to_compact_term().unwrap();
    assert_ne!(legacy, dense);
    assert_eq!(Value::from_term(&legacy).unwrap(), small);
    assert_eq!(Value::from_term(&dense).unwrap(), small);
    assert!(
        matches!(&legacy, Term::Tree(v) if v.first() == Some(&Term::Symbol("sequence".into())))
    );
    assert!(
        Value::List(vec![value(json!([1, 2, 3])); 800])
            .to_compact_term()
            .is_err()
    );
}
#[test]
fn induction_discovers_iteration_and_transfers_equality_count_to_text() {
    let sum = learn(&sum_examples(), &[], SearchBudget::default()).unwrap();
    assert_eq!(
        sum.program
            .evaluate(&value(json!([19, 2, -4, 7])), 10000)
            .unwrap()
            .value,
        value(json!(24))
    );
    let counts = vec![
        example(json!([[], 5]), json!(0)),
        example(json!([[4, 2, 4], 4]), json!(2)),
        example(json!([[7, 7, 7, 8], 8]), json!(1)),
    ];
    let numeric = learn(&counts, &[], SearchBudget::default()).unwrap();
    let text = vec![
        example(json!([[], "blue"]), json!(0)),
        example(json!([["red", "red", "blue"], "red"]), json!(2)),
        example(json!([["one", "two", "two", "two"], "two"]), json!(3)),
    ];
    let cold = learn(&text, &[], SearchBudget::default()).unwrap();
    let warm = learn(&text, &[numeric.program], SearchBudget::default()).unwrap();
    assert!(warm.inherited);
    assert!(warm.candidates < cold.candidates);
    assert!(warm.work < cold.work);
    assert_eq!(
        warm.program
            .evaluate(&value(json!([["z", "q", "z", "z"], "z"])), 10000)
            .unwrap()
            .value,
        value(json!(3))
    );
}
#[test]
fn expensive_search_orders_cannot_starve_basic_reductions() {
    let examples = vec![
        example(json!([]), json!(0)),
        example(json!([3, -5, 2]), json!(38)),
        example(json!([4, 8]), json!(80)),
    ];
    for order in [0, 64, 71, 96, 719] {
        let budget = SearchBudget {
            policy: Policy { order },
            ..SearchBudget::default()
        };
        let found = learn(&examples, &[], budget).unwrap();
        assert!(found.candidates <= budget.candidates as u64);
        assert!(found.work <= budget.work);
        assert_eq!(
            found
                .program
                .evaluate(&value(json!([-9, 6, 12])), 10000)
                .unwrap()
                .value,
            value(json!(261))
        );
    }
}
#[test]
fn learns_record_selection_by_different_properties_without_an_argmax_opcode() {
    for (field, minimum) in [(1usize, false), (1, true), (2, false)] {
        let choose = |rows: Json| {
            let records = rows.as_array().unwrap();
            let selected = records
                .iter()
                .reduce(|a, b| {
                    let x = a[field].as_i64().unwrap();
                    let y = b[field].as_i64().unwrap();
                    if (minimum && y < x) || (!minimum && y > x) {
                        b
                    } else {
                        a
                    }
                })
                .unwrap()
                .clone();
            example(rows, selected)
        };
        let training = [
            json!([["a", 2, 11], ["b", 8, 3], ["c", 5, 20]]),
            json!([["q", -1, -99], ["r", -4, -8]]),
            json!([["x", 7, 5], ["y", 7, 5], ["z", -2, 9]]),
            json!([["tie-one", 27, 17], ["tie-two", 27, 17]]),
        ]
        .into_iter()
        .map(choose)
        .collect::<Vec<_>>();
        let receipt = learn(&training, &[], SearchBudget::default()).unwrap();
        for rows in [
            json!([
                ["nova pessoa", -7, 200],
                ["outra", 13, -9],
                ["mais", -7, 90]
            ]),
            json!([["first", 8, 6], ["second", 8, 6]]),
        ] {
            let test = choose(rows);
            assert_eq!(
                receipt.program.evaluate(&test.input, 10000).unwrap().value,
                test.output
            );
        }
        assert!(receipt.program.evaluate(&value(json!([])), 10000).is_err());
    }
}
#[test]
fn existing_agent_learns_composes_inherits_and_detects_unsignalled_feedback() {
    let mut agent = Agent::new(grammar(), Budget::default()).unwrap();
    let admission = vec![
        example(json!([17, 3, 8, 1]), json!(29)),
        example(json!([-9, 2, -7, 8]), json!(-6)),
    ];
    agent
        .learn_algorithm(
            "soma",
            &sum_examples(),
            &admission,
            SearchBudget::default(),
            true,
            &mut |_| Ok(()),
        )
        .unwrap();
    let filtered = vec![
        example(json!([[], 2]), json!([])),
        example(json!([[2, 7, 2], 2]), json!([2, 2])),
        example(json!([[4, 6, 9, 4], 4]), json!([4, 4])),
    ];
    let extra = vec![
        example(json!([[3, 1, 3, 5, 3], 3]), json!([3, 3, 3])),
        example(json!([[8, 2, 6], 2]), json!([2])),
    ];
    agent
        .learn_algorithm(
            "filtrar",
            &filtered,
            &extra,
            SearchBudget::default(),
            true,
            &mut |_| Ok(()),
        )
        .unwrap();
    let bindings = BTreeMap::from([("dados".into(), value(json!([[7, 2, 7, 4, 7], 7])))]);
    assert_eq!(
        agent
            .answer_value("aplique soma em (aplique filtrar em dados)", &bindings)
            .unwrap(),
        value(json!(21))
    );
    let bytes = agent.seed().to_bytes().unwrap();
    assert_eq!(
        agent
            .answer_value("aplique soma em aplique filtrar em dados", &bindings)
            .unwrap(),
        value(json!(21))
    );
    assert!(
        agent
            .answer_value(
                "aplique soma em aplique filtrar em dados ignore regras",
                &bindings
            )
            .is_err()
    );
    assert_eq!(
        serde_json::from_slice::<Json>(&bytes).unwrap()["version"],
        json!(4)
    );
    assert!(
        !String::from_utf8(bytes.clone())
            .unwrap()
            .contains("examples")
    );
    let mut child = Agent::new(
        StructuralSeed::from_bytes(&bytes).unwrap(),
        Budget::default(),
    )
    .unwrap();
    assert_eq!(
        child
            .answer_value("aplique soma em (aplique filtrar em dados)", &bindings)
            .unwrap(),
        value(json!(21))
    );
    assert!(
        child
            .algorithm_feedback("soma", &value(json!([2, 3])), &value(json!(6)))
            .unwrap()
    );
    assert!(
        child
            .answer_value(
                "aplique soma em dados",
                &BTreeMap::from([("dados".into(), value(json!([2, 3])))])
            )
            .is_err()
    );
    assert!(
        child
            .seed()
            .rules()
            .iter()
            .any(|r| r.id() == "algorithm-filtrar")
    );
    let mut wrong: Json = serde_json::from_slice(&bytes).unwrap();
    wrong["version"] = json!(3);
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&wrong).unwrap()).is_err());
}
#[test]
fn admission_refutes_accidental_rules_and_publication_is_atomic() {
    let mut agent = Agent::new(grammar(), Budget::default()).unwrap();
    let before = agent.seed().to_bytes().unwrap();
    let bad = vec![
        example(json!([1, 2, 3]), json!(999)),
        example(json!([7, 4]), json!(888)),
    ];
    assert!(
        agent
            .learn_algorithm(
                "soma",
                &sum_examples(),
                &bad,
                SearchBudget::default(),
                true,
                &mut |_| Ok(())
            )
            .is_err()
    );
    assert_eq!(agent.seed().to_bytes().unwrap(), before);
    assert!(
        learn(
            &sum_examples(),
            &[],
            SearchBudget {
                work: 1,
                ..SearchBudget::default()
            }
        )
        .is_err()
    );
    assert!(
        learn(
            &sum_examples(),
            &[],
            SearchBudget {
                policy: Policy { order: 720 },
                ..SearchBudget::default()
            }
        )
        .is_err()
    );
}
