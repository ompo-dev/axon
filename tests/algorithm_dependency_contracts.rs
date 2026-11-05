#![cfg(feature = "curriculum")]
use axon_uic::structural::{
    Error, StructuralSeed,
    agent::{Agent, AlgorithmAcquisition, Budget},
    algorithm::{Example, Program, SearchBudget, Value},
};
use serde_json::{Value as Json, json};

fn value(v: Json) -> Value {
    serde_json::from_value(v).unwrap()
}
fn teach(agent: &mut Agent, name: &str, rows: Vec<(Json, Json)>) -> AlgorithmAcquisition {
    let examples: Vec<_> = rows
        .into_iter()
        .map(|(input, output)| Example {
            input: value(input),
            output: value(output),
        })
        .collect();
    let split = examples.len() - 2;
    agent
        .learn_algorithm(
            name,
            &examples[..split],
            &examples[split..],
            SearchBudget::default(),
            true,
            &mut |_| Ok(()),
        )
        .unwrap()
}
fn program<'a>(agent: &'a Agent, name: &str) -> Option<&'a Program> {
    agent
        .seed()
        .rules()
        .iter()
        .find(|r| r.id() == format!("algorithm-{name}"))
        .and_then(|r| r.algorithm_program())
}
fn learner() -> Agent {
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    teach(
        &mut agent,
        "sum",
        vec![
            (json!([]), json!(0)),
            (json!([2, 3]), json!(5)),
            (json!([-2, 9]), json!(7)),
            (json!([7, 5, -3]), json!(9)),
            (json!([-1, 2, 8]), json!(9)),
        ],
    );
    teach(
        &mut agent,
        "squares",
        vec![
            (json!([]), json!([])),
            (json!([2, -3]), json!([4, 9])),
            (json!([1, 4, 5]), json!([1, 16, 25])),
            (json!([7, 2]), json!([49, 4])),
            (json!([-2, 6, -5]), json!([4, 36, 25])),
        ],
    );
    teach(
        &mut agent,
        "select",
        vec![
            (json!([[], 2]), json!([])),
            (json!([[2, 3, 2], 2]), json!([2, 2])),
            (json!([[4, 1, 4], 4]), json!([4, 4])),
            (json!([[7, 7, 2], 7]), json!([7, 7])),
            (json!([[-3, 5, -3, 8], -3]), json!([-3, -3])),
        ],
    );
    teach(
        &mut agent,
        "chosen",
        vec![
            (json!([[], 2]), json!([])),
            (json!([[2, 3, 2], 2]), json!([4, 4])),
            (json!([[4, 1, 4], 4]), json!([16, 16])),
            (json!([[7, 2, 7], 7]), json!([49, 49])),
            (json!([[-3, 5, -3, 8], -3]), json!([9, 9])),
        ],
    );
    teach(
        &mut agent,
        "report",
        vec![
            (json!([[], 2]), json!(0)),
            (json!([[2, 3, 2], 2]), json!(8)),
            (json!([[4, 1, 4], 4]), json!(32)),
            (json!([[7, 2, 7], 7]), json!(98)),
            (json!([[-3, 5, -3, 8], -3]), json!(18)),
        ],
    );
    agent
}

#[test]
fn divergent_feedback_invalidates_captured_calls_without_erasing_unrelated_skills() {
    let mut agent = learner();
    assert_eq!(
        program(&agent, "report")
            .unwrap()
            .evaluate(&value(json!([[7, 2, 7], 7])), 10000)
            .unwrap()
            .value,
        value(json!(98))
    );
    assert!(
        agent
            .algorithm_feedback("sum", &value(json!([2, 3])), &value(json!(6)))
            .unwrap()
    );
    assert!(program(&agent, "sum").is_none());
    assert!(
        program(&agent, "report").is_none(),
        "a caller must not retain a disproved source version"
    );
    assert!(program(&agent, "chosen").is_some());
    assert_eq!(
        program(&agent, "squares")
            .unwrap()
            .evaluate(&value(json!([2, 3])), 10000)
            .unwrap()
            .value,
        value(json!([4, 9]))
    );
}

#[test]
fn roundtrip_preserves_transitive_invalidation_and_independent_copies() {
    let mut parent = learner();
    teach(
        &mut parent,
        "other-sum",
        vec![
            (json!([]), json!(0)),
            (json!([2, 3]), json!(5)),
            (json!([-2, 9]), json!(7)),
            (json!([7, 5, -3]), json!(9)),
            (json!([-1, 2, 8]), json!(9)),
        ],
    );
    let bytes = parent.seed().to_bytes().unwrap();
    let wire: Json = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["version"], json!(5));
    let report = wire["rules"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == "algorithm-report")
        .unwrap();
    assert!(
        report["dependencies"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["rule"] == "algorithm-chosen")
    );
    let mut child = Agent::new(
        StructuralSeed::from_bytes(&bytes).unwrap(),
        Budget::default(),
    )
    .unwrap();
    let affected = child
        .algorithm_feedback_affected("select", &value(json!([[2, 3, 2], 2])), &value(json!([])))
        .unwrap();
    let ids: std::collections::BTreeSet<_> = affected.into_iter().collect();
    assert_eq!(
        ids,
        std::collections::BTreeSet::from([
            "algorithm-select".into(),
            "algorithm-chosen".into(),
            "algorithm-report".into()
        ])
    );
    assert!(program(&child, "squares").is_some());
    assert!(program(&child, "sum").is_some());
    assert!(program(&child, "other-sum").is_some());
    StructuralSeed::from_bytes(&child.seed().to_bytes().unwrap()).unwrap();
}

#[test]
fn source_replacement_invalidates_callers_only_when_the_program_changes() {
    let mut agent = learner();
    let unchanged = teach(
        &mut agent,
        "sum",
        vec![
            (json!([]), json!(0)),
            (json!([2, 3]), json!(5)),
            (json!([-2, 9]), json!(7)),
            (json!([7, 5, -3]), json!(9)),
            (json!([-1, 2, 8]), json!(9)),
        ],
    );
    assert!(unchanged.invalidated.is_empty());
    assert!(program(&agent, "report").is_some());
    let changed = teach(
        &mut agent,
        "sum",
        vec![
            (json!([]), json!(0)),
            (json!([2, 3]), json!(7)),
            (json!([-2, 9]), json!(9)),
            (json!([7, 5, -3]), json!(12)),
            (json!([-1, 2, 8]), json!(12)),
        ],
    );
    assert_eq!(changed.invalidated, vec!["algorithm-report"]);
    assert!(program(&agent, "report").is_none());
    assert!(program(&agent, "chosen").is_some());
    assert_eq!(
        program(&agent, "sum")
            .unwrap()
            .evaluate(&value(json!([7, 3, 2])), 10000)
            .unwrap()
            .value,
        value(json!(15))
    );
}

#[test]
fn failed_admission_or_publication_keeps_the_entire_previous_graph() {
    let mut agent = learner();
    let before = agent.seed().to_bytes().unwrap();
    let training = vec![
        Example {
            input: value(json!([])),
            output: value(json!(0)),
        },
        Example {
            input: value(json!([2, 3])),
            output: value(json!(7)),
        },
        Example {
            input: value(json!([-2, 9])),
            output: value(json!(9)),
        },
    ];
    let admission = vec![
        Example {
            input: value(json!([7, 5, -3])),
            output: value(json!(12)),
        },
        Example {
            input: value(json!([-1, 2, 8])),
            output: value(json!(12)),
        },
    ];
    assert!(
        agent
            .learn_algorithm(
                "sum",
                &training,
                &admission,
                SearchBudget::default(),
                true,
                &mut |e| {
                    if e.phase == "consolidation" {
                        Err(Error::Invalid("test publication failure"))
                    } else {
                        Ok(())
                    }
                }
            )
            .is_err()
    );
    assert_eq!(before, agent.seed().to_bytes().unwrap());
    let mut wrong = admission.clone();
    wrong[0].output = value(json!(999));
    assert!(
        agent
            .learn_algorithm(
                "sum",
                &training,
                &wrong,
                SearchBudget::default(),
                true,
                &mut |_| Ok(())
            )
            .is_err()
    );
    assert_eq!(before, agent.seed().to_bytes().unwrap());
    assert_eq!(
        program(&agent, "report")
            .unwrap()
            .evaluate(&value(json!([[7, 2, 7], 7])), 10000)
            .unwrap()
            .value,
        value(json!(98))
    );
}

#[test]
fn imported_graph_rejects_dangling_stale_fabricated_or_downgraded_witnesses() {
    let agent = learner();
    let original: Json = serde_json::from_slice(&agent.seed().to_bytes().unwrap()).unwrap();
    let index = original["rules"]
        .as_array()
        .unwrap()
        .iter()
        .position(|r| r["id"] == "algorithm-report")
        .unwrap();
    let rejects = |wire: Json| {
        assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&wire).unwrap()).is_err())
    };
    let mut bad = original.clone();
    bad["version"] = json!(4);
    rejects(bad);
    let mut bad = original.clone();
    bad["rules"][index]["dependencies"] = json!([]);
    rejects(bad);
    let mut bad = original.clone();
    bad["rules"][index]["dependencies"][0]["rule"] = json!("missing");
    rejects(bad);
    let mut bad = original.clone();
    bad["rules"][index]["dependencies"][0]["rule"] = json!("algorithm-report");
    rejects(bad);
    let mut bad = original.clone();
    bad["rules"][index]["dependencies"][0]["expected"] = json!("Empty");
    rejects(bad);
    let mut bad = original.clone();
    let duplicate = bad["rules"][index]["dependencies"][0].clone();
    bad["rules"][index]["dependencies"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    rejects(bad);
    let mut bad = original.clone();
    bad["rules"]
        .as_array_mut()
        .unwrap()
        .retain(|r| r["id"] != "algorithm-sum");
    rejects(bad);
}

#[test]
fn legacy_v4_inference_is_preserved_and_known_dependencies_migrate_on_feedback() {
    let agent = learner();
    let mut legacy: Json = serde_json::from_slice(&agent.seed().to_bytes().unwrap()).unwrap();
    legacy["version"] = json!(4);
    for rule in legacy["rules"].as_array_mut().unwrap() {
        rule.as_object_mut().unwrap().remove("dependencies");
    }
    let seed = StructuralSeed::from_bytes(&serde_json::to_vec(&legacy).unwrap()).unwrap();
    let mut child = Agent::new(seed, Budget::default()).unwrap();
    assert_eq!(
        program(&child, "report")
            .unwrap()
            .evaluate(&value(json!([[7, 2, 7], 7])), 10000)
            .unwrap()
            .value,
        value(json!(98))
    );
    let affected = child
        .algorithm_feedback_affected("sum", &value(json!([2, 3])), &value(json!(6)))
        .unwrap();
    assert!(affected.contains(&"algorithm-report".into()));
    assert!(program(&child, "chosen").is_some());
    let current: Json = serde_json::from_slice(&child.seed().to_bytes().unwrap()).unwrap();
    assert_eq!(current["version"], json!(5));
}

#[test]
fn nested_algorithm_bindings_cannot_bypass_dependency_validation() {
    let agent = learner();
    let mut wire: Json = serde_json::from_slice(&agent.seed().to_bytes().unwrap()).unwrap();
    let index = wire["rules"]
        .as_array()
        .unwrap()
        .iter()
        .position(|r| r["id"] == "algorithm-report")
        .unwrap();
    let output = wire["rules"][index]["output"].take();
    wire["rules"][index]["output"] = json!({"Tree":[{"Literal":"response"},output]});
    let seed = StructuralSeed::from_bytes(&serde_json::to_vec(&wire).unwrap()).unwrap();
    let mut child = Agent::new(seed, Budget::default()).unwrap();
    let affected = child
        .algorithm_feedback_affected("sum", &value(json!([2, 3])), &value(json!(6)))
        .unwrap();
    assert!(affected.contains(&"algorithm-report".into()));
    wire["rules"][index]["dependencies"] = json!([]);
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&wire).unwrap()).is_err());
}

#[test]
fn unknown_legacy_inline_calls_remain_readable_but_refuse_unwitnessed_adaptation_atomically() {
    let original = learner();
    let mut wire: Json = serde_json::from_slice(&original.seed().to_bytes().unwrap()).unwrap();
    wire["version"] = json!(4);
    for rule in wire["rules"].as_array_mut().unwrap() {
        rule.as_object_mut().unwrap().remove("dependencies");
        if rule["id"] == "algorithm-report" {
            rule["output"]["Algorithm"]["program"] = json!({"Call":[{"Literal":17}, "Input"]});
        }
    }
    let legacy = serde_json::to_vec(&wire).unwrap();
    let mut child = Agent::new(
        StructuralSeed::from_bytes(&legacy).unwrap(),
        Budget::default(),
    )
    .unwrap();
    assert_eq!(
        program(&child, "report")
            .unwrap()
            .evaluate(&value(json!([2, 3])), 1000)
            .unwrap()
            .value,
        value(json!(17))
    );
    let before = child.seed().to_bytes().unwrap();
    assert!(
        child
            .algorithm_feedback_affected("sum", &value(json!([2, 3])), &value(json!(6)))
            .is_err()
    );
    assert_eq!(before, child.seed().to_bytes().unwrap());
}
