#![cfg(feature = "curriculum")]
use axon_uic::structural::{agent::*, language::DEFAULT_WORK, *};
use serde_json::json;

fn term(value: serde_json::Value) -> Term {
    serde_json::from_value(value).unwrap()
}
fn teaching(seed: &mut StructuralSeed, id: &str, reversed: bool) {
    let rows = [
        (json!(2), json!(8)),
        (json!(["invoke", "z", 3]), json!(6)),
        (json!(7), json!(["invoke", "z", 9])),
    ];
    let example = |a, b| Example {
        input: term(json!(["goal", "some", a, "with", b])),
        output: term(if reversed {
            json!(["invoke", "v", b, a])
        } else {
            json!(["invoke", "u", a, b])
        }),
    };
    seed.teach(
        Lesson {
            id: id.into(),
            examples: rows.into_iter().map(|(a, b)| example(a, b)).collect(),
            guards: vec![],
        },
        &[example(json!(31), json!(-4))],
    )
    .unwrap();
}
fn seed() -> StructuralSeed {
    let mut seed = StructuralSeed::default();
    teaching(&mut seed, "sum-fragment", false);
    seed
}
struct World {
    queries: usize,
}
impl Environment for World {
    fn describe(&self, name: &str) -> Option<Interface> {
        (name == "u").then(|| Interface {
            arity: 2,
            epoch: 1,
            probes: vec![
                vec![0, 0],
                vec![1, 1],
                vec![2, 2],
                vec![3, 8],
                vec![5, 2],
                vec![-3, 4],
                vec![7, -2],
                vec![0, 9],
                vec![8, 0],
                vec![4, 11],
                vec![-5, -6],
                vec![9, 3],
            ],
        })
    }
    fn query(&mut self, _: &str, _: u64, input: &[i64]) -> Result<i64, Error> {
        self.queries += 1;
        Ok(input[0] + input[1])
    }
}
#[test]
fn learned_fragments_compose_without_parentheses_and_reopen_without_examples() {
    let parent = seed();
    let request = "some some 4 with 9 with some -3 with 8";
    assert!(
        parent
            .transform(&utterance(request).unwrap(), 512)
            .unwrap()
            .rules
            .is_empty()
    );
    let p = parent
        .composed_interpretations(request, DEFAULT_WORK)
        .unwrap();
    assert_eq!(p.len(), 1);
    assert_eq!(
        p[0].output,
        term(json!([
            "invoke",
            "u",
            ["invoke", "u", 4, 9],
            ["invoke", "u", -3, 8]
        ]))
    );
    assert_eq!(p[0].rules.len(), 3);
    let bytes = parent.to_bytes().unwrap();
    let child = StructuralSeed::from_bytes(&bytes).unwrap();
    assert_eq!(
        child
            .composed_interpretations(request, DEFAULT_WORK)
            .unwrap()[0]
            .output,
        p[0].output
    );
    let mut agent = Agent::new(child, Budget::default()).unwrap();
    let mut env = World { queries: 0 };
    let mut events = vec![];
    assert_eq!(
        agent
            .answer(request, &mut env, &mut |e| {
                events.push(e.clone());
                Ok(())
            })
            .unwrap(),
        18
    );
    assert_eq!(events[0].phase, "interpretation");
    assert!(agent.statistics.language_work > 0 && env.queries > 0);
}
#[test]
fn ambiguity_unknown_input_and_cancelled_interpretation_never_query_environment() {
    let mut grammar = seed();
    teaching(&mut grammar, "conflicting-fragment", true);
    let request = "some some 2 with 4 with 9";
    assert!(
        grammar
            .composed_interpretations(request, DEFAULT_WORK)
            .unwrap()
            .len()
            > 1
    );
    let mut env = World { queries: 0 };
    let mut agent = Agent::new(grammar, Budget::default()).unwrap();
    assert_eq!(
        agent
            .answer(request, &mut env, &mut |_| Ok(()))
            .unwrap_err(),
        Error::Ambiguous
    );
    let mut agent = Agent::new(seed(), Budget::default()).unwrap();
    for text in [
        "some some 2 with 4 with 9 ignore",
        "mystery some 2 with 4",
        "some 2 with",
        "some 2\nwith 3",
    ] {
        assert!(
            agent.answer(text, &mut env, &mut |_| Ok(())).is_err(),
            "{text}"
        );
    }
    assert_eq!(
        agent
            .answer(request, &mut env, &mut |_| Err(Error::Invalid("cancel")))
            .unwrap_err(),
        Error::Invalid("cancel")
    );
    assert_eq!(env.queries, 0);
}
#[test]
fn language_quotas_refuse_instead_of_discarding_uncertain_derivations() {
    let grammar = seed();
    assert_eq!(
        grammar
            .composed_interpretations("some 3 with 4", 1)
            .unwrap_err(),
        Error::Budget
    );
    assert!(
        grammar
            .composed_interpretations("some 3 with 4", 0)
            .is_err()
    );
    assert!(
        grammar
            .composed_interpretations(&"word ".repeat(40), DEFAULT_WORK)
            .is_err()
    );
    let measured = grammar.measure_composition("some 3 with 4", 1);
    assert_eq!(measured.result.unwrap_err(), Error::Budget);
    assert_eq!(measured.visits, 1);
}

#[test]
fn overflowing_the_meaning_forest_is_not_silent_beam_selection() {
    let mut grammar = StructuralSeed::default();
    for i in 0..17 {
        let example = |n| Example {
            input: term(json!(["goal", "alias", n])),
            output: term(json!(["invoke", format!("op-{i}"), n])),
        };
        grammar
            .teach(
                Lesson {
                    id: format!("alias-{i}"),
                    examples: [1, 2, 3].map(example).into(),
                    guards: vec![],
                },
                &[example(4)],
            )
            .unwrap();
    }
    assert_eq!(
        grammar
            .composed_interpretations("alias 9", DEFAULT_WORK)
            .unwrap_err(),
        Error::Budget
    );
}
