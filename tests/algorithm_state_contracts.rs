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
fn examples(rows: Vec<(Json, Json)>) -> Vec<Example> {
    rows.into_iter()
        .map(|(input, output)| Example {
            input: value(input),
            output: value(output),
        })
        .collect()
}
fn acquired(rows: &[Example], library: &[Program]) -> Program {
    let receipt = learn(&rows[..3], library, SearchBudget::default())
        .unwrap_or_else(|e| panic!("acquisition with {} sources: {e}", library.len()));
    for e in &rows[3..] {
        assert_eq!(
            receipt.program.evaluate(&e.input, 100_000).unwrap().value,
            e.output
        );
    }
    receipt.program
}
fn queries() -> Vec<Example> {
    examples(vec![
        (json!([[], "a"]), json!(0)),
        (json!([[["a", 12], ["b", -4]], "a"]), json!(12)),
        (json!([[["c", 6], ["d", 7]], "d"]), json!(7)),
        (json!([[["x", -9], ["y", 4]], "x"]), json!(-9)),
        (json!([[["a", 3], ["b", 8]], "z"]), json!(0)),
    ])
}
fn removals() -> Vec<Example> {
    examples(vec![
        (json!([[], "a"]), json!([])),
        (
            json!([[["a", 2], ["b", 3], ["c", 4]], "b"]),
            json!([["a", 2], ["c", 4]]),
        ),
        (
            json!([[["z", 4], ["b", -2], ["a", 7]], "z"]),
            json!([["b", -2], ["a", 7]]),
        ),
        (
            json!([[["x", 5], ["y", -3], ["z", 7]], "z"]),
            json!([["x", 5], ["y", -3]]),
        ),
        (json!([[["k", 4]], "other"]), json!([["k", 4]])),
    ])
}
fn updates() -> Vec<Example> {
    examples(vec![
        (json!([[], ["azul", 2]]), json!([["azul", 2]])),
        (
            json!([[["azul", 7], ["verde", -2]], ["azul", 3]]),
            json!([["verde", -2], ["azul", 10]]),
        ),
        (
            json!([[["amarelo", 5], ["verde", 9]], ["novo", -4]]),
            json!([["amarelo", 5], ["verde", 9], ["novo", -4]]),
        ),
        (
            json!([[["u", -7], ["v", 12]], ["u", 2]]),
            json!([["v", 12], ["u", -5]]),
        ),
        (json!([[["u", 9]], ["u", -9]]), json!([["u", 0]])),
    ])
}
fn grouped(rows: &[(String, i64)]) -> Json {
    let mut state: Vec<(String, i64)> = Vec::new();
    for (key, amount) in rows {
        let previous = state.iter().find(|r| r.0 == *key).map_or(0, |r| r.1);
        state.retain(|r| r.0 != *key);
        state.push((key.clone(), previous + amount));
    }
    json!(state)
}

#[test]
fn expression_search_acquires_record_construction_and_scalar_identity_wrapping() {
    let rows = examples(vec![
        (json!(["a", 2, 3]), json!(["a", 6])),
        (json!(["b", -4, 5]), json!(["b", -20])),
        (json!(["c", 7, -2]), json!(["c", -14])),
        (json!(["new", 9, 8]), json!(["new", 72])),
        (json!(["other", 0, -3]), json!(["other", 0])),
    ]);
    acquired(&rows, &[]);
    let rows = examples(vec![
        (json!("a"), json!(["a", 1])),
        (json!("b"), json!(["b", 1])),
        (json!("c"), json!(["c", 1])),
        (json!("new"), json!(["new", 1])),
        (json!("other"), json!(["other", 1])),
    ]);
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    agent
        .learn_algorithm(
            "unit",
            &rows[..3],
            &rows[3..],
            SearchBudget::default(),
            true,
            &mut |_| Ok(()),
        )
        .unwrap();
    let encoded: Json = serde_json::from_slice(&agent.seed().to_bytes().unwrap()).unwrap();
    let rule = &encoded["rules"][0];
    assert_eq!(rule["input"]["Tree"][2]["Hole"][1], json!(1));
    assert_eq!(encoded["version"], json!(6));
    let mut downgraded = encoded.clone();
    downgraded["version"] = json!(5);
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&downgraded).unwrap()).is_err());
    StructuralSeed::from_bytes(&agent.seed().to_bytes().unwrap()).unwrap();
    let invocation = serde_json::from_value(json!(["invoke", "unit", "nunca-visto"])).unwrap();
    let result = agent.seed().transform(&invocation, 512).unwrap();
    assert_eq!(
        Value::from_term(&result.output).unwrap(),
        value(json!(["nunca-visto", 1]))
    );
}

#[test]
fn learned_key_operations_compose_into_a_state_transition_and_lift_to_batch_grouping() {
    let query = acquired(&queries(), &[]);
    let remove = acquired(&removals(), &[]);
    let transition = acquired(&updates(), &[query.clone(), remove.clone()]);
    let training = examples(vec![
        (json!([]), json!([])),
        (
            json!([["a", 2], ["b", 3], ["a", 5]]),
            json!([["b", 3], ["a", 7]]),
        ),
        (
            json!([["c", -4], ["c", 3], ["b", 0], ["c", 2]]),
            json!([["b", 0], ["c", 1]]),
        ),
        (
            json!([["q", -3], ["q", 7], ["r", 12]]),
            json!([["q", 4], ["r", 12]]),
        ),
        (
            json!([["u", 0], ["v", 9], ["u", -2], ["v", -9]]),
            json!([["u", -2], ["v", 0]]),
        ),
    ]);
    assert!(learn(&training[..3], &[], SearchBudget::default()).is_err());
    let batch = acquired(&training, &[query, remove, transition]);
    for length in 0..50 {
        for shift in 0..5 {
            let rows: Vec<_> = (0..length)
                .map(|i| {
                    (
                        format!("nova-chave-{}", (i * 7 + shift) % 13),
                        i * 3 % 19 - 9,
                    )
                })
                .collect();
            assert_eq!(
                batch.evaluate(&value(json!(rows)), 500_000).unwrap().value,
                value(grouped(&rows))
            );
        }
    }
}

#[test]
fn observational_equivalence_is_refuted_by_independent_admission_not_treated_as_a_proof() {
    let rows = examples(vec![
        (json!(-1), json!(-1)),
        (json!(0), json!(0)),
        (json!(1), json!(1)),
        (json!(2), json!(8)),
        (json!(3), json!(27)),
        (json!(4), json!(64)),
    ]);
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    let receipt = agent
        .learn_algorithm(
            "cube",
            &rows[..3],
            &rows[3..],
            SearchBudget::default(),
            true,
            &mut |_| Ok(()),
        )
        .unwrap();
    assert!(receipt.attempts >= 2);
    assert_eq!(receipt.confirmations, 2);
    let request = serde_json::from_value(json!(["invoke", "cube", 7])).unwrap();
    assert_eq!(
        agent.seed().transform(&request, 512).unwrap().output,
        axon_uic::structural::Term::Integer(343)
    );
}

#[test]
fn bounded_expression_search_never_spends_unreserved_candidates_or_work() {
    let rows = examples(vec![
        (json!(["a", 2, 3]), json!(["a", 6])),
        (json!(["b", -4, 5]), json!(["b", -20])),
        (json!(["c", 7, -2]), json!(["c", -14])),
    ]);
    for candidates in [1, 2, 7, 31] {
        let budget = SearchBudget {
            candidates,
            work: 1000,
            ..SearchBudget::default()
        };
        let failure = learn(&rows, &[], budget).unwrap_err();
        assert!(failure.candidates <= candidates as u64);
        assert!(failure.work <= 1000);
    }
}

#[test]
fn larger_catalogues_keep_recent_transition_sources_and_refute_their_descendants_selectively() {
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    for (name, rows) in [
        ("query", queries()),
        ("without", removals()),
        ("update", updates()),
    ] {
        agent
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
    let mut wire: Json = serde_json::from_slice(&agent.seed().to_bytes().unwrap()).unwrap();
    let template = wire["rules"][0].clone();
    let mut rules = Vec::new();
    // Synthetic routing fixtures, not a claim that these unrelated functions were learned.
    for n in 0..36 {
        let mut rule = template.clone();
        rule["id"] = json!(format!("algorithm-unrelated-{n}"));
        rule["input"]["Tree"][1]["Literal"] = json!(format!("unrelated-{n}"));
        rule["input"]["Tree"][2]["Hole"][1] = json!(0);
        rule["output"]["Algorithm"]["program"] = json!({"Add":["Input",{"Literal":n}]});
        rule["teaching_checksum"] = json!(0);
        rules.push(rule);
    }
    rules.extend(wire["rules"].as_array().unwrap().iter().cloned());
    wire["rules"] = json!(rules);
    wire["version"] = json!(6);
    let mut child = Agent::new(
        StructuralSeed::from_bytes(&serde_json::to_vec(&wire).unwrap()).unwrap(),
        Budget::default(),
    )
    .unwrap();
    let rows = examples(vec![
        (json!([]), json!([])),
        (
            json!([["a", 2], ["b", 3], ["a", 5]]),
            json!([["b", 3], ["a", 7]]),
        ),
        (
            json!([["c", -4], ["c", 3], ["b", 0], ["c", 2]]),
            json!([["b", 0], ["c", 1]]),
        ),
        (
            json!([["q", -3], ["q", 7], ["r", 12]]),
            json!([["q", 4], ["r", 12]]),
        ),
        (
            json!([["u", 0], ["v", 9], ["u", -2], ["v", -9]]),
            json!([["u", -2], ["v", 0]]),
        ),
    ]);
    let result = child
        .learn_algorithm(
            "group",
            &rows[..3],
            &rows[3..],
            SearchBudget::default(),
            true,
            &mut |_| Ok(()),
        )
        .unwrap();
    assert!(result.library_programs <= 32);
    assert_eq!(child.seed().rules().len(), 40);
    let bytes = child.seed().to_bytes().unwrap();
    let mut restarted = Agent::new(
        StructuralSeed::from_bytes(&bytes).unwrap(),
        Budget::default(),
    )
    .unwrap();
    let affected = restarted
        .algorithm_feedback_affected(
            "query",
            &value(json!([[["a", 12]], "a"])),
            &value(json!(13)),
        )
        .unwrap();
    assert!(affected.contains(&"algorithm-update".into()));
    assert!(affected.contains(&"algorithm-group".into()));
    assert!(
        restarted
            .seed()
            .rules()
            .iter()
            .any(|r| r.id() == "algorithm-without")
    );
    assert!(
        restarted
            .seed()
            .rules()
            .iter()
            .any(|r| r.id() == "algorithm-unrelated-35")
    );
}

#[test]
fn old_sources_outside_the_primary_window_share_one_total_search_budget() {
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    for (name, rows) in [
        ("query", queries()),
        ("without", removals()),
        ("update", updates()),
    ] {
        agent
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
    let mut wire: Json = serde_json::from_slice(&agent.seed().to_bytes().unwrap()).unwrap();
    let template = wire["rules"][0].clone();
    // Unrelated synthetic routing fixtures deliberately displace the old, larger sources.
    for n in 0..36 {
        let mut rule = template.clone();
        rule["id"] = json!(format!("algorithm-later-{n}"));
        rule["input"]["Tree"][1]["Literal"] = json!(format!("later-{n}"));
        rule["input"]["Tree"][2]["Hole"][1] = json!(0);
        rule["output"]["Algorithm"]["program"] = json!({"Add":["Input",{"Literal":100+n}]});
        rule["teaching_checksum"] = json!(0);
        wire["rules"].as_array_mut().unwrap().push(rule);
    }
    wire["version"] = json!(6);
    let mut child = Agent::new(
        StructuralSeed::from_bytes(&serde_json::to_vec(&wire).unwrap()).unwrap(),
        Budget::default(),
    )
    .unwrap();
    let rows = examples(vec![
        (json!([]), json!([])),
        (
            json!([["a", 2], ["b", 3], ["a", 5]]),
            json!([["b", 3], ["a", 7]]),
        ),
        (
            json!([["c", -4], ["c", 3], ["b", 0], ["c", 2]]),
            json!([["b", 0], ["c", 1]]),
        ),
        (
            json!([["q", -3], ["q", 7], ["r", 12]]),
            json!([["q", 4], ["r", 12]]),
        ),
        (
            json!([["u", 0], ["v", 9], ["u", -2], ["v", -9]]),
            json!([["u", -2], ["v", 0]]),
        ),
    ]);
    let result = child
        .learn_algorithm(
            "group",
            &rows[..3],
            &rows[3..],
            SearchBudget::default(),
            true,
            &mut |_| Ok(()),
        )
        .unwrap();
    assert!(result.library_windows >= 2);
    assert!(result.library_programs <= 32);
    assert!(result.candidates <= 50000);
    assert!(result.work <= 30_000_000);
    assert_eq!(child.seed().rules().len(), 40);
}
