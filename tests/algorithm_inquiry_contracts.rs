#![cfg(feature = "curriculum")]
use axon_uic::structural::{
    StructuralSeed,
    agent::{Agent, AlgorithmLesson, Budget},
    algorithm::{Example, InquiryPlan, InquiryStrategy, SearchBudget, Value, inquire, learn},
};
use serde_json::json;

fn v(value: serde_json::Value) -> Value {
    serde_json::from_value(value).unwrap()
}
fn sum(input: &Value) -> Value {
    let Value::List(items) = input else {
        panic!("list")
    };
    Value::Integer(
        items
            .iter()
            .map(|x| match x {
                Value::Integer(n) => *n,
                _ => panic!("integer"),
            })
            .sum(),
    )
}
fn teaching() -> Vec<Example> {
    [-3, 0, 7]
        .into_iter()
        .map(|n| Example {
            input: v(json!([n])),
            output: v(json!(n)),
        })
        .collect()
}
fn plan(strategy: InquiryStrategy) -> InquiryPlan {
    InquiryPlan {
        inputs: vec![
            v(json!([11])),
            v(json!([-14])),
            v(json!([22])),
            v(json!([3, 5, -2])),
            v(json!([-9, 4, 12, 2])),
        ],
        max_queries: 1,
        max_hypotheses: 8,
        strategy,
    }
}

#[test]
fn question_precedes_its_answer_and_distinguishes_unlabeled_hypotheses() {
    let mut selected = None;
    let mut events = Vec::new();
    let receipt = inquire(
        &teaching(),
        &[],
        SearchBudget::default(),
        &plan(InquiryStrategy::Disagreement),
        &mut |input| {
            selected = Some(input.clone());
            Ok(sum(input))
        },
        &mut |event| {
            events.push(event.clone());
            Ok(())
        },
    )
    .unwrap();
    assert!(receipt.hypotheses >= 2, "{receipt:?}");
    assert_eq!(receipt.queries, 1);
    assert!(matches!(selected, Some(Value::List(ref items)) if items.len() > 1));
    let before = events
        .iter()
        .position(|e| e.phase == "query_selected")
        .unwrap();
    let after = events
        .iter()
        .position(|e| e.phase == "query_observed")
        .unwrap();
    assert!(before < after);
    assert!(events[before].actual.is_none());
    assert!(events[before].predictions.len() >= 2);
    assert!(events[before].disagreement_pairs > 0);
    assert_eq!(
        events[after].actual,
        Some(sum(events[before].input.as_ref().unwrap()))
    );
    assert!(receipt.work > 0 && receipt.candidates > 0);
    assert!(receipt.work < SearchBudget::default().work);
}

#[test]
fn same_single_oracle_call_can_disambiguate_where_a_passive_prefix_cannot() {
    let rows = teaching();
    for strategy in [InquiryStrategy::Disagreement, InquiryStrategy::Passive] {
        let receipt = inquire(
            &rows,
            &[],
            SearchBudget::default(),
            &plan(strategy),
            &mut |input| Ok(sum(input)),
            &mut |_| Ok(()),
        )
        .unwrap();
        let acquired = learn(
            &receipt.examples,
            &[],
            SearchBudget {
                candidates: SearchBudget::default().candidates - receipt.candidates as usize,
                work: SearchBudget::default().work - receipt.work,
                ..SearchBudget::default()
            },
        )
        .unwrap();
        let mut correct = 0;
        for n in 1..101 {
            let input = v(json!([n, -2 * n, 3 * n + 1]));
            correct += usize::from(
                acquired.program.evaluate(&input, 10_000).unwrap().value == sum(&input),
            );
        }
        assert_eq!(
            correct,
            if strategy == InquiryStrategy::Disagreement {
                100
            } else {
                0
            }
        );
        assert_eq!(receipt.queries, 1);
    }
}

#[test]
fn active_acquisition_keeps_admission_independent_and_child_needs_no_teacher() {
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    let admission: Vec<_> = [vec![2, 4, 8], vec![-4, 11, -9]]
        .into_iter()
        .map(|row| {
            let input = v(json!(row));
            Example {
                output: sum(&input),
                input,
            }
        })
        .collect();
    let mut events = Vec::new();
    let receipt = agent
        .learn_algorithm_inquiring(
            AlgorithmLesson {
                name: "unknown-name",
                examples: &teaching(),
                admission: &admission,
            },
            &plan(InquiryStrategy::Disagreement),
            SearchBudget::default(),
            true,
            &mut |input| Ok(sum(input)),
            &mut |event| {
                events.push(serde_json::to_value(event).unwrap());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(receipt.confirmations, 2);
    assert_eq!(receipt.inquiry.as_ref().unwrap().queries, 1);
    assert!(
        events
            .windows(2)
            .all(|p| p[0]["work"].as_u64() <= p[1]["work"].as_u64())
    );
    assert_eq!(events.last().unwrap()["work"], json!(receipt.work));
    let child = StructuralSeed::from_bytes(&agent.seed().to_bytes().unwrap()).unwrap();
    let invocation = axon_uic::structural::Term::Tree(vec![
        axon_uic::structural::Term::Symbol("invoke".into()),
        axon_uic::structural::Term::Symbol("unknown-name".into()),
        v(json!([91, -12, 8, -1])).to_term().unwrap(),
    ]);
    assert_eq!(
        child.transform(&invocation, 512).unwrap().output,
        axon_uic::structural::Term::Integer(86)
    );
}

#[test]
fn overlap_invalid_oracle_and_event_cancellation_cannot_publish() {
    let mut agent = Agent::new(StructuralSeed::default(), Budget::default()).unwrap();
    let before = agent.seed().to_bytes().unwrap();
    let mut calls = 0;
    let admission = vec![
        Example {
            input: plan(InquiryStrategy::Disagreement).inputs[0].clone(),
            output: v(json!(11)),
        },
        Example {
            input: v(json!([90, 2])),
            output: v(json!(92)),
        },
    ];
    assert!(
        agent
            .learn_algorithm_inquiring(
                AlgorithmLesson {
                    name: "test",
                    examples: &teaching(),
                    admission: &admission
                },
                &plan(InquiryStrategy::Disagreement),
                SearchBudget::default(),
                true,
                &mut |_| {
                    calls += 1;
                    Ok(v(json!(0)))
                },
                &mut |_| Ok(())
            )
            .is_err()
    );
    assert_eq!(calls, 0);
    assert_eq!(before, agent.seed().to_bytes().unwrap());
    let result = inquire(
        &teaching(),
        &[],
        SearchBudget::default(),
        &plan(InquiryStrategy::Disagreement),
        &mut |_| {
            calls += 1;
            Ok(v(json!(0)))
        },
        &mut |_| Err(axon_uic::structural::Error::Invalid("cancelled")),
    );
    assert!(result.is_err());
    assert_eq!(calls, 0);
    let result = inquire(
        &teaching(),
        &[],
        SearchBudget::default(),
        &plan(InquiryStrategy::Disagreement),
        &mut |_| Ok(Value::Text(String::new())),
        &mut |_| Ok(()),
    );
    assert!(result.is_err());
    assert_eq!(before, agent.seed().to_bytes().unwrap());
}

#[test]
fn inquiry_caps_duplicates_and_consensus_are_not_silent_certificates() {
    let rows = teaching();
    let mut p = plan(InquiryStrategy::Disagreement);
    p.inputs.push(p.inputs[0].clone());
    assert!(p.validate(&rows).is_err());
    p = plan(InquiryStrategy::Disagreement);
    p.max_queries = 5;
    assert!(p.validate(&rows).is_err());
    p = plan(InquiryStrategy::Disagreement);
    p.inputs = vec![v(json!([81])), v(json!([-97]))];
    let r = inquire(
        &rows,
        &[],
        SearchBudget::default(),
        &p,
        &mut |x| Ok(sum(x)),
        &mut |_| Ok(()),
    )
    .unwrap();
    assert!(r.queries <= 1);
    assert!(!r.stop.is_empty());
    for work in [2, 31, 100] {
        let mut recorded = 0;
        let _ = inquire(
            &rows,
            &[],
            SearchBudget {
                candidates: 4,
                work,
                ..SearchBudget::default()
            },
            &plan(InquiryStrategy::Disagreement),
            &mut |x| Ok(sum(x)),
            &mut |event| {
                recorded = event.work;
                Ok(())
            },
        );
        assert!(recorded <= work);
    }
}

#[test]
fn new_answers_rebuild_the_sample_instead_of_certifying_a_surviving_shortcut() {
    let minimum = |input: &Value| match input {
        Value::List(items) => items.iter().min().cloned().unwrap(),
        _ => panic!("list"),
    };
    let mut p = plan(InquiryStrategy::Disagreement);
    p.inputs = vec![
        v(json!([11])),
        v(json!([-24])),
        v(json!([38])),
        v(json!([2, 5, -3])),
        v(json!([-9, -4, -2])),
        v(json!([3, 6, 11])),
        v(json!([0, -4, 7, 2])),
    ];
    p.max_queries = 2;
    p.max_hypotheses = 16;
    let receipt = inquire(
        &teaching(),
        &[],
        SearchBudget::default(),
        &p,
        &mut |input| Ok(minimum(input)),
        &mut |_| Ok(()),
    )
    .unwrap();
    let learned = learn(
        &receipt.examples,
        &[],
        SearchBudget {
            candidates: SearchBudget::default().candidates - receipt.candidates as usize,
            work: SearchBudget::default().work - receipt.work,
            ..SearchBudget::default()
        },
    )
    .unwrap();
    for input in [
        json!([8, 2, 5]),
        json!([-4, -11, -3]),
        json!([9, -6, 2]),
        json!([12, 4, 0]),
    ] {
        let input = v(input);
        assert_eq!(
            learned.program.evaluate(&input, 10_000).unwrap().value,
            minimum(&input)
        );
    }
}
