#![cfg(feature = "curriculum")]
use axon_uic::structural::{induction::*, *};
use serde_json::json;

fn example(a: &str, b: &str, opposite: bool) -> Example {
    Example {
        input: serde_json::from_value(json!([
            "pair",
            ["observation", a, b],
            [if opposite { "changed" } else { "observation" }, a, b]
        ]))
        .unwrap(),
        output: Term::Integer(i64::from(opposite)),
    }
}
fn examples() -> Vec<Example> {
    (0..12)
        .flat_map(|i| {
            [false, true]
                .map(|opposite| example(&format!("entity-{i}"), &format!("property-{i}"), opposite))
        })
        .collect()
}
fn budget() -> Budget {
    Budget {
        seed: 47,
        proposals: 64,
        rules: 16,
        checks: 100000,
        minimum_holes: 2,
        structural_groups: false,
    }
}
#[test]
fn induction_is_generic_deterministic_portable_and_transfers_unseen_bindings() {
    let examples = examples();
    let (seed, stats) = learn(&examples, budget(), &mut |_| Ok(())).unwrap();
    assert_eq!(stats.correct, examples.len());
    assert_eq!(stats.wrong, 0);
    assert!(stats.selected <= 2);
    let raw = seed.to_bytes().unwrap();
    let other = learn(&examples, budget(), &mut |_| Ok(())).unwrap().0;
    assert_eq!(raw, other.to_bytes().unwrap());
    let child = StructuralSeed::from_bytes(&raw).unwrap();
    for opposite in [false, true] {
        let e = example("unseen-42", "unseen-13", opposite);
        assert_eq!(child.transform(&e.input, 32).unwrap().output, e.output);
    }
    let mut outside = example("unseen-42", "unseen-13", false);
    if let Term::Tree(v) = &mut outside.input {
        v[0] = Term::Symbol("unknown".into());
    }
    assert!(
        child
            .transform(&outside.input, 32)
            .unwrap()
            .rules
            .is_empty()
    );
}
#[test]
fn contradictory_labels_refute_rules_instead_of_becoming_correct_abstentions() {
    let mut data = examples();
    let mut conflict = data[0].clone();
    conflict.output = Term::Integer(2);
    data.push(conflict);
    let (seed, stats) = learn(&data, budget(), &mut |_| Ok(())).unwrap();
    assert!(stats.refuted > 0);
    assert_eq!(stats.wrong, 0);
    assert_eq!(stats.correct + stats.abstained, data.len());
    assert!(seed.transform(&data[0].input, 32).unwrap().rules.is_empty());
}
#[test]
fn induction_respects_work_bounds_and_cancellation() {
    let mut tiny = budget();
    tiny.checks = 1;
    let (seed, stats) = learn(&examples(), tiny, &mut |_| Ok(())).unwrap();
    assert_eq!(stats.checked, 1);
    assert!(stats.quota_reached);
    assert!(seed.rules().is_empty());
    assert!(matches!(
        learn(&examples(), budget(), &mut |_| Err(Error::Budget)),
        Err(Error::Budget)
    ));
    tiny.proposals = 0;
    assert!(learn(&examples(), tiny, &mut |_| Ok(())).is_err());
    let mut data = examples();
    data[0].output = data[0].input.clone();
    assert!(learn(&data, budget(), &mut |_| Ok(())).is_err());
}

#[test]
fn structured_families_bind_unseen_multiword_values_and_preserve_roles() {
    fn row(name: &str, place: &str) -> Example {
        Example {
            input: axon_uic::structural::agent::utterance(&format!("send to {name} from {place}"))
                .unwrap(),
            output: serde_json::from_value(json!([
                "delivery",
                ["recipient", name],
                ["origin", place]
            ]))
            .unwrap(),
        }
    }
    let rows = (0..12)
        .map(|i| {
            row(
                &format!("person{i} {}", "name ".repeat(i % 3 + 1).trim()),
                &format!("place{i}"),
            )
        })
        .collect::<Vec<_>>();
    let mut b = budget();
    b.minimum_holes = 1;
    let (seed, stats) = learn_grouped(std::slice::from_ref(&rows), b, &mut |_| Ok(())).unwrap();
    assert_eq!(stats.wrong, 0);
    assert!(
        stats.correct >= 6,
        "{}",
        serde_json::to_string(&stats).unwrap()
    );
    let child = StructuralSeed::from_bytes(&seed.to_bytes().unwrap()).unwrap();
    assert!(
        !String::from_utf8(seed.to_bytes().unwrap())
            .unwrap()
            .contains("person0")
    );
    let unseen = row("another whole name", "new town");
    assert_eq!(
        child.transform(&unseen.input, 32).unwrap().output,
        unseen.output
    );
    for i in 0..64 {
        let e = row(
            &format!("new{i} {}", "unseen ".repeat(i % 4 + 1).trim()),
            &format!("different town-{i}"),
        );
        assert_eq!(child.transform(&e.input, 32).unwrap().output, e.output);
    }
    let ambiguous = axon_uic::structural::agent::utterance("send to a from b from c").unwrap();
    assert!(matches!(
        child.transform(&ambiguous, 32),
        Err(Error::Ambiguous)
    ));
    let mut conflict = rows[0].clone();
    conflict.output = serde_json::from_value(json!(["different", "meaning"])).unwrap();
    let (refused, stats) =
        learn_grouped(&[rows, vec![conflict.clone()]], b, &mut |_| Ok(())).unwrap();
    assert_eq!(stats.wrong, 0);
    assert!(
        refused
            .transform(&conflict.input, 32)
            .unwrap()
            .rules
            .is_empty()
    );
    assert!(learn_grouped(&[vec![]], b, &mut |_| Ok(())).is_err());
    let rows = (0..12)
        .map(|i| row(&format!("name{i}"), &format!("town{i}")))
        .collect::<Vec<_>>();
    assert!(matches!(
        learn_grouped(&[rows], b, &mut |_| Err(Error::Budget)),
        Err(Error::Budget)
    ));
}

#[test]
fn learned_lexical_domains_refuse_suffix_capture_without_losing_compositional_values() {
    let colors = ["red", "green", "blue"];
    let family = colors
        .iter()
        .flat_map(|a| {
            colors.iter().map(move |b| Example {
                input: serde_json::from_value(json!(["utterance", "say", a, b])).unwrap(),
                output: serde_json::from_value(json!(["tag", ["color", format!("{a} {b}")]]))
                    .unwrap(),
            })
        })
        .collect::<Vec<_>>();
    let suffix = Example {
        input: serde_json::from_value(json!(["utterance", "say", "red", "green", "please"]))
            .unwrap(),
        output: serde_json::from_value(json!(["tag", ["color", "red green"]])).unwrap(),
    };
    let families = vec![family, vec![suffix.clone()]];
    let mut b = budget();
    b.minimum_holes = 1;
    let (plain, _) = learn_grouped(&families, b, &mut |_| Ok(())).unwrap();
    let (typed, stats) = learn_grouped_domains(&families, b, &mut |_| Ok(())).unwrap();
    assert_eq!(stats.wrong, 0);
    assert!(typed.rules().len() > plain.rules().len());
    let raw = typed.to_bytes().unwrap();
    let child = StructuralSeed::from_bytes(&raw).unwrap();
    let unseen =
        serde_json::from_value(json!(["utterance", "say", "green", "blue", "red"])).unwrap();
    assert_eq!(
        child.transform(&unseen, 32).unwrap().output,
        serde_json::from_value::<Term>(json!(["tag", ["color", "green blue red"]])).unwrap()
    );
    assert!(child.transform(&suffix.input, 32).unwrap().rules.is_empty());
    let unknown = serde_json::from_value(json!(["utterance", "say", "ultraviolet"])).unwrap();
    assert!(child.transform(&unknown, 32).unwrap().rules.is_empty());
    let mut wire: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(wire["version"], 8);
    wire["version"] = json!(7);
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&wire).unwrap()).is_err());
    let bad = Lesson {
        id: "bad-domain".into(),
        examples: families[0][..3].to_vec(),
        guards: vec![Guard::Vocabulary {
            path: vec![],
            values: vec!["x".into(), "x".into()],
        }],
    };
    assert!(Rule::learn(bad).is_err());
}

#[test]
fn vocabulary_guard_bounds_are_validated_before_learning() {
    let examples = ["a", "b", "c"]
        .iter()
        .map(|word| Example {
            input: serde_json::from_value(json!(["observation", word])).unwrap(),
            output: Term::Integer(1),
        })
        .collect::<Vec<_>>();
    for values in [
        vec![],
        vec!["a".into()],
        vec!["b".into(), "a".into()],
        vec!["a".into(), "a".into()],
        vec!["a b".into(), "c".into()],
        (0..513).map(|i| format!("word-{i:04}")).collect(),
    ] {
        assert!(
            Rule::learn(Lesson {
                id: "invalid".into(),
                examples: examples.clone(),
                guards: vec![Guard::Vocabulary {
                    path: vec![],
                    values
                }]
            })
            .is_err()
        );
    }
    assert!(
        Rule::learn(Lesson {
            id: "invalid".into(),
            examples,
            guards: vec![Guard::Vocabulary {
                path: vec![32],
                values: vec!["a".into(), "b".into()]
            }]
        })
        .is_err()
    );
}
#[test]
fn closed_associations_require_independent_observations_and_refuse_unknown_or_conflicting_keys() {
    let rows = (0..6)
        .flat_map(|key| {
            (0..3).map(move |sample| Example {
                input: serde_json::from_value(json!([
                    "observation",
                    format!("class-{key}"),
                    format!("private-source-{key}-{sample}"),
                    format!("private-target-{key}-{sample}")
                ]))
                .unwrap(),
                output: Term::Integer(i64::from(key >= 3)),
            })
        })
        .collect::<Vec<_>>();
    let (seed, stats) = learn_closed(&rows, budget(), &[1], &mut |_| Ok(())).unwrap();
    assert_eq!(stats.correct, rows.len());
    assert_eq!(stats.wrong, 0);
    let bytes = seed.to_bytes().unwrap();
    assert!(
        !String::from_utf8(bytes.clone())
            .unwrap()
            .contains("private-")
    );
    let child = StructuralSeed::from_bytes(&bytes).unwrap();
    for key in 0..6 {
        let input = serde_json::from_value(json!([
            "observation",
            format!("class-{key}"),
            "unseen source",
            "unseen target"
        ]))
        .unwrap();
        assert_eq!(
            child.transform(&input, 64).unwrap().output,
            Term::Integer(i64::from(key >= 3))
        );
    }
    let unknown = serde_json::from_value(json!([
        "observation",
        "unknown",
        "unseen source",
        "unseen target"
    ]))
    .unwrap();
    assert!(child.transform(&unknown, 64).unwrap().rules.is_empty());
    let mut conflict = rows.clone();
    let mut bad = rows[0].clone();
    bad.output = Term::Integer(2);
    conflict.push(bad);
    let seed = learn_closed(&conflict, budget(), &[1], &mut |_| Ok(()))
        .unwrap()
        .0;
    assert!(seed.transform(&rows[0].input, 64).unwrap().rules.is_empty());
    let mut tiny = budget();
    tiny.checks = 1;
    assert!(matches!(
        learn_closed(&rows, tiny, &[1], &mut |_| Ok(())),
        Err(Error::Budget)
    ));
    assert!(learn_closed(&rows, budget(), &[99], &mut |_| Ok(())).is_err());
}
#[test]
fn one_recurring_key_can_be_learned_without_widening_the_symbol_guard_contract() {
    let rows = (0..6)
        .map(|i| Example {
            input: serde_json::from_value(json!([
                "observed",
                "known",
                format!("source-{i}"),
                format!("target-{i}")
            ]))
            .unwrap(),
            output: Term::Integer(2),
        })
        .collect::<Vec<_>>();
    let (seed, stats) = learn_closed(&rows, budget(), &[1], &mut |_| Ok(())).unwrap();
    assert_eq!(stats.correct, 6);
    assert_eq!(stats.selected, 1);
    let unseen =
        serde_json::from_value(json!(["observed", "known", "novel source", "novel target"]))
            .unwrap();
    assert_eq!(
        seed.transform(&unseen, 64).unwrap().output,
        Term::Integer(2)
    );
    let unknown = serde_json::from_value(json!([
        "observed",
        "unknown",
        "novel source",
        "novel target"
    ]))
    .unwrap();
    assert!(seed.transform(&unknown, 64).unwrap().rules.is_empty());
}
