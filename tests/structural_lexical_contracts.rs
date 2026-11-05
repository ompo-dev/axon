#![cfg(feature = "curriculum")]
use axon_uic::structural::{agent::utterance, language::DEFAULT_WORK, *};
use serde_json::json;

fn term(value: serde_json::Value) -> Term {
    serde_json::from_value(value).unwrap()
}
fn unit(n: i64, word: &str, unit: &str) -> Example {
    Example {
        input: term(json!(["goal", n, word])),
        output: term(json!(["quantity", n, unit])),
    }
}
fn taught() -> StructuralSeed {
    let mut seed = StructuralSeed::default();
    seed.teach(
        Lesson {
            id: "units".into(),
            guards: vec![],
            examples: vec![
                unit(2, "reais", "BRL"),
                unit(3, "quilos", "KG"),
                unit(7, "reais", "BRL"),
                unit(9, "quilos", "KG"),
            ],
        },
        &[unit(31, "reais", "BRL"), unit(-12, "quilos", "KG")],
    )
    .unwrap();
    seed
}
#[test]
fn lexical_correspondence_is_learned_closed_and_portable() {
    let seed = taught();
    let raw = seed.to_bytes().unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(wire["version"], 7);
    let child = StructuralSeed::from_bytes(&raw).unwrap();
    for (word, tag) in [("reais", "BRL"), ("quilos", "KG")] {
        for n in [-2047, -1, 0, 43, 9001] {
            assert_eq!(
                child
                    .composed_interpretations(&format!("{n} {word}"), DEFAULT_WORK)
                    .unwrap()[0]
                    .output,
                term(json!(["quantity", n, tag]))
            );
        }
    }
    for word in ["euros", "real", "gramas", "ignore"] {
        assert_eq!(
            child
                .composed_interpretations(&format!("4 {word}"), DEFAULT_WORK)
                .unwrap_err(),
            Error::Unsupported
        );
        assert!(
            child
                .transform(&utterance(&format!("4 {word}")).unwrap(), 64)
                .unwrap()
                .rules
                .is_empty()
        );
    }
    assert_eq!(child.to_bytes().unwrap(), raw);
}
#[test]
fn new_lexeme_requires_evidence_and_conflicting_table_does_not_publish() {
    let mut seed = taught();
    let before = seed.to_bytes().unwrap();
    assert!(
        seed.teach(
            Lesson {
                id: "unseen".into(),
                guards: vec![],
                examples: vec![
                    unit(2, "reais", "BRL"),
                    unit(3, "quilos", "KG"),
                    unit(7, "reais", "BRL"),
                ]
            },
            &[unit(31, "euros", "EUR")]
        )
        .is_err()
    );
    assert_eq!(seed.to_bytes().unwrap(), before);
    assert!(
        seed.teach(
            Lesson {
                id: "conflict".into(),
                guards: vec![],
                examples: vec![
                    unit(2, "reais", "BRL"),
                    unit(3, "reais", "KG"),
                    unit(7, "quilos", "KG"),
                ]
            },
            &[unit(31, "reais", "BRL")]
        )
        .is_err()
    );
    assert_eq!(seed.to_bytes().unwrap(), before);
}
#[test]
fn incidental_names_do_not_replace_an_inducible_numeric_rule_with_a_table() {
    let make = |name: &str, n: i64| Example {
        input: term(json!(["square", name, n])),
        output: term(json!(["result", n * n])),
    };
    let mut seed = StructuralSeed::default();
    seed.teach(
        Lesson {
            id: "square".into(),
            guards: vec![],
            examples: vec![make("a", 2), make("b", 5), make("c", -3)],
        },
        &[make("new", 13)],
    )
    .unwrap();
    assert_eq!(
        seed.transform(&make("unseen", 17).input, 64)
            .unwrap()
            .output,
        make("unseen", 17).output
    );
}
#[test]
fn lexical_codec_rejects_wrong_version_duplicate_keys_and_executable_values() {
    let raw = taught().to_bytes().unwrap();
    let value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let mut old = value.clone();
    old["version"] = json!(6);
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&old).unwrap()).is_err());
    let path = &["rules", "0", "output", "Tree", "2", "Lookup"];
    assert!(
        value["rules"][0]["output"]["Tree"][2]
            .get("Lookup")
            .is_some(),
        "{path:?}"
    );
    let mut duplicate = value.clone();
    duplicate["rules"][0]["output"]["Tree"][2]["Lookup"]["entries"][1][0] = json!("quilos");
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&duplicate).unwrap()).is_err());
    let mut executable = value;
    executable["rules"][0]["output"]["Tree"][2]["Lookup"]["entries"][0][1] =
        json!(["invoke", "hidden"]);
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&executable).unwrap()).is_err());
}
fn separate() -> StructuralSeed {
    let mut seed = StructuralSeed::default();
    for (word, tag) in [("reais", "BRL"), ("quilos", "KG")] {
        seed.teach(
            Lesson {
                id: word.into(),
                guards: vec![],
                examples: [2, 4, 7].into_iter().map(|n| unit(n, word, tag)).collect(),
            },
            &[unit(31, word, tag)],
        )
        .unwrap();
    }
    seed
}
#[test]
fn consolidation_preserves_exact_domains_precedence_and_outputs() {
    let old = separate();
    let mut compact = old.clone();
    let receipts = compact.consolidate_language().unwrap();
    assert_eq!(receipts.len(), 1);
    assert_eq!(compact.rules().len(), 1);
    let compact = StructuralSeed::from_bytes(&compact.to_bytes().unwrap()).unwrap();
    for word in ["reais", "quilos", "euros", "unknown"] {
        for n in -100..=100 {
            let input = utterance(&format!("{n} {word}")).unwrap();
            let a = old.transform(&input, 64).unwrap();
            let b = compact.transform(&input, 64).unwrap();
            assert_eq!(a.output, b.output);
            assert_eq!(a.rules.is_empty(), b.rules.is_empty());
        }
    }
    assert!(compact.clone().consolidate_language().unwrap().is_empty());
}
#[test]
fn consolidation_does_not_create_cross_products_or_resolve_conflicts() {
    let mut seed = separate();
    seed.teach(
        Lesson {
            id: "conflicting-reais".into(),
            guards: vec![],
            examples: [2, 4, 7]
                .into_iter()
                .map(|n| unit(n, "reais", "OTHER"))
                .collect(),
        },
        &[unit(31, "reais", "OTHER")],
    )
    .unwrap();
    let input = utterance("16 reais").unwrap();
    assert_eq!(seed.transform(&input, 64).unwrap_err(), Error::Ambiguous);
    seed.consolidate_language().unwrap();
    assert_eq!(seed.transform(&input, 64).unwrap_err(), Error::Ambiguous);
    assert!(
        seed.transform(&utterance("16 euros").unwrap(), 64)
            .unwrap()
            .rules
            .is_empty()
    );
}

fn alternative_seed() -> StructuralSeed {
    let mut seed = StructuralSeed::default();
    for (id, verb, particle) in [
        ("a", "aplique", "em"),
        ("b", "execute", "sobre"),
        ("c", "quero", "para"),
    ] {
        let example = |n, k| Example {
            input: term(json!(["goal", verb, n, particle, k])),
            output: term(json!(["result", n, k])),
        };
        seed.teach(
            Lesson {
                id: id.into(),
                guards: vec![],
                examples: vec![example(2, 13), example(4, -1), example(7, 3)],
            },
            &[example(31, -8)],
        )
        .unwrap();
    }
    let example = |n| Example {
        input: term(json!(["goal", "aplique", 7, "em", n])),
        output: term(json!(["override", n])),
    };
    seed.teach(
        Lesson {
            id: "specific".into(),
            guards: vec![],
            examples: [2, 4, 7].map(example).to_vec(),
        },
        &[example(31)],
    )
    .unwrap();
    seed
}

#[test]
fn exact_alternatives_preserve_cross_products_bindings_conflicts_and_priority() {
    let old = alternative_seed();
    let before = old.to_bytes().unwrap();
    let mut compact = old.clone();
    assert_eq!(compact.consolidate_alternatives().unwrap().len(), 2);
    assert_eq!(compact.rules().len(), 2);
    let raw = compact.to_bytes().unwrap();
    let child = StructuralSeed::from_bytes(&raw).unwrap();
    assert_eq!(raw, child.to_bytes().unwrap());
    assert_eq!(before, old.to_bytes().unwrap());
    for verb in ["aplique", "execute", "quero", "unknown"] {
        for particle in ["em", "sobre", "para", "unknown"] {
            for n in -10..=10 {
                let input = term(json!(["goal", verb, n, particle, 19]));
                let a = old.transform(&input, 64).unwrap();
                let b = child.transform(&input, 64).unwrap();
                assert_eq!(a.output, b.output, "{verb} {n} {particle}");
                assert_eq!(a.rules.is_empty(), b.rules.is_empty());
                if n == 0 || n == 7 {
                    let text = format!("{verb} {n} {particle} 19");
                    let meanings = |s: &StructuralSeed| {
                        s.composed_interpretations(&text, DEFAULT_WORK)
                            .map(|predictions| {
                                predictions
                                    .into_iter()
                                    .map(|p| p.output)
                                    .collect::<Vec<_>>()
                            })
                    };
                    assert_eq!(meanings(&old), meanings(&child), "composition: {text}");
                    if [("aplique", "em"), ("execute", "sobre"), ("quero", "para")]
                        .contains(&(verb, particle))
                    {
                        assert!(meanings(&child).is_ok(), "known composition: {text}");
                    }
                }
            }
        }
    }
    let mut conflicting = compact.clone();
    let example = |n, k| Example {
        input: term(json!(["goal", "execute", n, "sobre", k])),
        output: term(json!(["other", n, k])),
    };
    conflicting
        .teach(
            Lesson {
                id: "conflict".into(),
                guards: vec![],
                examples: vec![example(2, 13), example(4, -1), example(7, 3)],
            },
            &[example(31, -8)],
        )
        .unwrap();
    assert_eq!(
        conflicting
            .transform(&term(json!(["goal", "execute", 5, "sobre", 2])), 64)
            .unwrap_err(),
        Error::Ambiguous
    );
    assert!(conflicting.consolidate_alternatives().unwrap().is_empty());
    assert_eq!(
        conflicting
            .transform(&term(json!(["goal", "execute", 5, "sobre", 2])), 64)
            .unwrap_err(),
        Error::Ambiguous
    );
}

#[test]
fn alternative_codec_rejects_old_versions_overlaps_unbound_slots_and_output_choices() {
    let mut seed = alternative_seed();
    seed.consolidate_alternatives().unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&seed.to_bytes().unwrap()).unwrap();
    assert_eq!(wire["version"], 9);
    let branches = wire["rules"][0]["input"]["Choice"].as_array().unwrap();
    for version in [2, 7, 8] {
        let mut forged = wire.clone();
        forged["version"] = json!(version);
        assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&forged).unwrap()).is_err());
    }
    for choices in [
        vec![],
        vec![branches[0].clone()],
        vec![branches[0].clone(), branches[0].clone()],
        vec![branches[0].clone(); 9],
    ] {
        let mut forged = wire.clone();
        forged["rules"][0]["input"]["Choice"] = json!(choices);
        assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&forged).unwrap()).is_err());
    }
    for (field, value) in [
        ("output", wire["rules"][0]["input"].clone()),
        (
            "input",
            json!({"Choice":[branches[0], {"Tree":[{"Literal":"goal"},{"Hole":[0,0]}]}]}),
        ),
    ] {
        let mut forged = wire.clone();
        forged["rules"][0][field] = value;
        assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&forged).unwrap()).is_err());
    }
}
