#![cfg(feature = "curriculum")]
use axon_uic::structural::*;
use serde_json::json;
fn term(v: serde_json::Value) -> Term {
    serde_json::from_value(v).unwrap()
}
fn example(a: serde_json::Value, b: serde_json::Value) -> Example {
    Example {
        input: term(a),
        output: term(b),
    }
}
fn lesson(id: &str, word: &str) -> Lesson {
    Lesson {
        id: id.into(),
        guards: vec![],
        examples: [("ana", "recife"), ("bruno", "natal"), ("caique", "manaus")]
            .into_iter()
            .map(|(a, b)| {
                example(
                    json!(["utterance", a, word, b]),
                    json!(["tell", a, "city", b]),
                )
            })
            .collect(),
    }
}
#[test]
fn learns_typed_shared_roles_and_transfers_unseen_symbols() {
    let r = Rule::learn(lesson("fact", "mora")).unwrap();
    assert_eq!(
        r.apply(&term(json!(["utterance", "dora", "mora", "belem"]))),
        Some(term(json!(["tell", "dora", "city", "belem"])))
    );
    assert!(
        r.apply(&term(json!(["utterance", 12, "mora", "belem"])))
            .is_none()
    );
    assert!(
        r.apply(&term(json!(["utterance", "dora", "visita", "belem"])))
            .is_none()
    );
}
#[test]
fn admission_is_independent_atomic_and_replay_validates_storage() {
    let mut s = StructuralSeed::default();
    let l = lesson("fact", "mora");
    assert!(s.teach(l.clone(), &[l.examples[0].clone()]).is_err());
    assert!(s.rules().is_empty());
    s.teach(
        l,
        &[example(
            json!(["utterance", "dora", "mora", "belem"]),
            json!(["tell", "dora", "city", "belem"]),
        )],
    )
    .unwrap();
    let bytes = s.to_bytes().unwrap();
    assert_eq!(
        StructuralSeed::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    for n in [0, 1, bytes.len() / 2, bytes.len() - 1] {
        assert!(StructuralSeed::from_bytes(&bytes[..n]).is_err());
    }
    let mut bad: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    bad["version"] = json!(99);
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&bad).unwrap()).is_err());
}
#[test]
fn repeated_variables_are_constraints_not_independent_wildcards() {
    let r = Rule::learn(Lesson {
        id: "same".into(),
        guards: vec![],
        examples: ["a", "b", "c"]
            .map(|s| example(json!(["same", s, s]), json!(["yes", s])))
            .to_vec(),
    })
    .unwrap();
    assert!(r.apply(&term(json!(["same", "d", "e"]))).is_none());
    assert_eq!(
        r.apply(&term(json!(["same", "z", "z"]))),
        Some(term(json!(["yes", "z"])))
    );
}
#[test]
fn conflicting_teaching_refuses_and_loops_exhaust_quota() {
    let mut s = StructuralSeed::default();
    let make = |id: &str, out: &str| Lesson {
        id: id.into(),
        guards: vec![],
        examples: [1, 2, 3]
            .map(|n| example(json!(["a", n]), json!([out, n])))
            .to_vec(),
    };
    s.teach(
        make("one", "b"),
        &[example(json!(["a", 4]), json!(["b", 4]))],
    )
    .unwrap();
    s.teach(
        make("two", "c"),
        &[example(json!(["a", 5]), json!(["c", 5]))],
    )
    .unwrap();
    assert_eq!(
        s.transform(&term(json!(["a", 8])), 8).unwrap_err(),
        Error::Ambiguous
    );
    let mut loop_seed = StructuralSeed::default();
    loop_seed
        .teach(
            make("one", "a"),
            &[example(json!(["a", 7]), json!(["a", 7]))],
        )
        .unwrap();
    assert_eq!(
        loop_seed.transform(&term(json!(["a", 8])), 8).unwrap_err(),
        Error::Budget
    );
}
#[test]
fn taught_structural_derivative_composes_without_a_builtin_derivative() {
    let mut s = StructuralSeed::default();
    for (id, make) in [
        (
            "constant",
            (0..4)
                .map(|i| {
                    example(
                        json!(["d", ["num", i], ["var", format!("v{i}")]]),
                        json!(["num", 0]),
                    )
                })
                .collect::<Vec<_>>(),
        ),
        (
            "variable",
            (0..4)
                .map(|i| {
                    example(
                        json!(["d", ["var", format!("v{i}")], ["var", format!("v{i}")]]),
                        json!(["num", 1]),
                    )
                })
                .collect::<Vec<_>>(),
        ),
    ] {
        let mut es = make;
        let last = es.pop().unwrap();
        s.teach(
            Lesson {
                id: id.into(),
                guards: vec![],
                examples: es,
            },
            &[last],
        )
        .unwrap();
    }
    let make = |i: i64| {
        example(
            json!([
                "d",
                ["add", ["var", format!("x{i}")], ["num", i]],
                ["var", format!("x{i}")]
            ]),
            json!([
                "add",
                ["d", ["var", format!("x{i}")], ["var", format!("x{i}")]],
                ["d", ["num", i], ["var", format!("x{i}")]]
            ]),
        )
    };
    s.teach(
        Lesson {
            id: "sum".into(),
            guards: vec![],
            examples: (1..4).map(make).collect(),
        },
        &[make(4)],
    )
    .unwrap();
    let out = s
        .transform(
            &term(json!([
                "d",
                ["add", ["var", "new"], ["num", 91]],
                ["var", "new"]
            ])),
            32,
        )
        .unwrap();
    assert_eq!(out.output, term(json!(["add", ["num", 1], ["num", 0]])));
    assert_eq!(out.rules.len(), 3);
}

fn language_seed() -> StructuralSeed {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../evals/curricula/structural-v3.json")).unwrap();
    let mut seed = StructuralSeed::default();
    for value in fixture["lessons"].as_array().unwrap() {
        let id = value["id"].as_str().unwrap();
        if id.starts_with("d-") || id.starts_with("lim-") || id.starts_with("solve-") {
            continue;
        }
        let mut teaching = value.clone();
        let admission = teaching
            .as_object_mut()
            .unwrap()
            .remove("admission")
            .unwrap();
        seed.teach(
            serde_json::from_value(teaching).unwrap(),
            &serde_json::from_value::<Vec<Example>>(admission).unwrap(),
        )
        .unwrap();
    }
    seed
}
struct Temporary(std::path::PathBuf);
impl Temporary {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let name = format!(
            "axon-structural-{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let p = self.0.canonicalize().unwrap();
        assert!(p.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        std::fs::remove_dir_all(p).unwrap();
    }
}
#[test]
fn dialogue_uses_context_preserves_real_ambiguity_and_resumes_personal_memory() {
    use axon_uic::structural::dialogue::Dialogue;
    let seed = language_seed();
    let temp = Temporary::new();
    let path = temp.0.join("session.axep");
    let mut d = Dialogue::open(&path).unwrap();
    assert_eq!(
        d.turn(&seed, "onde ele mora", "user").unwrap().status,
        "clarify"
    );
    d.turn(&seed, "ana mora em recife", "user").unwrap();
    assert_eq!(
        d.turn(&seed, "onde ele mora", "user")
            .unwrap()
            .text
            .as_deref(),
        Some("ana mora em recife")
    );
    drop(d);
    let mut d = Dialogue::open(&path).unwrap();
    assert_eq!(
        d.turn(&seed, "onde ele mora", "user")
            .unwrap()
            .text
            .as_deref(),
        Some("ana mora em recife")
    );
    d.turn(&seed, "ele mora em belem", "user").unwrap();
    d.turn(&seed, "ana mora em recife", "user").unwrap();
    assert_eq!(
        d.turn(&seed, "onde ele mora", "user").unwrap().status,
        "clarify"
    );
    let mut fresh = Dialogue::open(&temp.0.join("fresh.axep")).unwrap();
    assert_eq!(
        fresh.turn(&seed, "onde ana mora", "user").unwrap().status,
        "unknown"
    );
}
#[test]
fn online_supervision_adds_a_new_form_without_erasing_old_grammar_or_facts() {
    use axon_uic::structural::dialogue::Dialogue;
    let mut seed = language_seed();
    let temp = Temporary::new();
    let mut d = Dialogue::open(&temp.0.join("live.axep")).unwrap();
    assert_eq!(
        d.turn(&seed, "dora habita em natal", "user")
            .unwrap()
            .status,
        "unknown"
    );
    let mut l = lesson("habita", "habita");
    for e in &mut l.examples {
        if let Term::Tree(v) = &mut e.input {
            v.insert(3, Term::Symbol("em".into()));
        }
    }
    // This new teacher maps the verb to the same existing functional predicate.
    for e in &mut l.examples {
        if let Term::Tree(v) = &mut e.output {
            v[2] = Term::Symbol("mora".into());
        }
    }
    seed.teach(
        l,
        &[example(
            json!(["utterance", "erica", "habita", "em", "palmas"]),
            json!(["tell", "erica", "mora", "palmas"]),
        )],
    )
    .unwrap();
    assert_eq!(
        d.turn(&seed, "dora habita em natal", "user")
            .unwrap()
            .status,
        "recorded"
    );
    assert_eq!(
        d.turn(&seed, "em que cidade dora mora", "user")
            .unwrap()
            .text
            .as_deref(),
        Some("dora mora em natal")
    );
    let roundtrip = StructuralSeed::from_bytes(&seed.to_bytes().unwrap()).unwrap();
    assert_eq!(
        d.turn(&roundtrip, "onde dora mora", "user")
            .unwrap()
            .text
            .as_deref(),
        Some("dora mora em natal")
    );
}
#[test]
fn output_growth_is_refused_before_instantiating_an_oversized_tree() {
    let make = |i: i64| {
        let a = if i % 2 == 0 {
            json!(["a", i])
        } else {
            json!(["b", i])
        };
        example(json!(["copy", a]), json!(["pair", a, a]))
    };
    let mut seed = StructuralSeed::default();
    seed.teach(
        Lesson {
            id: "copy".into(),
            guards: vec![],
            examples: (0..4).map(make).collect(),
        },
        &[make(8)],
    )
    .unwrap();
    fn tree(n: usize) -> Term {
        if n == 0 {
            term(json!(["leaf", 1]))
        } else {
            Term::Tree(vec![
                Term::Symbol("branch".into()),
                tree(n - 1),
                tree(n - 1),
            ])
        }
    }
    let input = Term::Tree(vec![Term::Symbol("copy".into()), tree(9)]);
    input.validate().unwrap();
    assert_eq!(seed.transform(&input, 32).unwrap_err(), Error::Budget);
}
#[test]
fn serialized_seed_contains_compiled_rules_not_examples_and_rejects_forged_holes() {
    let mut seed = StructuralSeed::default();
    seed.teach(
        lesson("one", "mora"),
        &[example(
            json!(["utterance", "dora", "mora", "belem"]),
            json!(["tell", "dora", "city", "belem"]),
        )],
    )
    .unwrap();
    let bytes = seed.to_bytes().unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(!text.contains("\"examples\""));
    assert!(!text.contains("\"ana\""));
    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["version"], 2);
    wire["rules"][0]["output"] = json!({"Hole":[99,1]});
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&wire).unwrap()).is_err());
    assert!(serde_json::from_value::<Rule>(wire["rules"][0].clone()).is_err());
}

#[test]
fn discovers_integer_output_programs_instead_of_copying_values() {
    let mut seed = StructuralSeed::default();
    let ex = |a: i64, b: i64| example(json!(["unknown", a, b]), json!(["num", a * a + b]));
    seed.teach(
        Lesson {
            id: "discovered".into(),
            guards: vec![],
            examples: [(1, 7), (3, -2), (-4, 5), (6, 1), (2, 19)]
                .map(|(a, b)| ex(a, b))
                .to_vec(),
        },
        &[ex(11, -5), ex(-8, 21)],
    )
    .unwrap();
    assert_eq!(
        seed.transform(&term(json!(["unknown", -17, 31])), 32)
            .unwrap()
            .output,
        term(json!(["num", 320]))
    );
    assert_eq!(seed.rules()[0].derived_programs().len(), 1);
    let bytes = seed.to_bytes().unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["version"],
        3
    );
    let restored = StructuralSeed::from_bytes(&bytes).unwrap();
    assert_eq!(
        restored
            .transform(&term(json!(["unknown", 19, -7])), 32)
            .unwrap()
            .output,
        term(json!(["num", 354]))
    );
    let mut bad: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    bad["rules"][0]["output"]["Tree"][1]["Calculate"]["slots"] = json!([99, 0]);
    assert!(StructuralSeed::from_bytes(&serde_json::to_vec(&bad).unwrap()).is_err());
}

#[test]
fn exact_integer_execution_refuses_overflow_and_inexact_roots() {
    let mut seed = StructuralSeed::default();
    let ex = |a: i64, b: i64| example(json!(["combine", a, b]), json!(["num", a + b]));
    seed.teach(
        Lesson {
            id: "addition".into(),
            guards: vec![],
            examples: [(1, 7), (3, 2), (-4, 5), (6, 1)]
                .map(|(a, b)| ex(a, b))
                .to_vec(),
        },
        &[ex(17, -3)],
    )
    .unwrap();
    let input = term(json!(["combine", i64::MAX, 1]));
    assert!(seed.transform(&input, 16).unwrap().rules.is_empty());
    assert_eq!(
        seed.transform(&term(json!(["combine", 9_007_199_254_740_993i64, 2])), 16)
            .unwrap()
            .output,
        term(json!(["num", 9_007_199_254_740_995i64]))
    );
    let root = |n: i64| example(json!(["root", n * n]), json!(["num", n]));
    seed.teach(
        Lesson {
            id: "root".into(),
            guards: vec![],
            examples: [2, 5, 9, 13].map(root).to_vec(),
        },
        &[root(17)],
    )
    .unwrap();
    assert!(
        seed.transform(&term(json!(["root", 2])), 16)
            .unwrap()
            .rules
            .is_empty()
    );
    assert!(
        seed.transform(&term(json!(["root", -4])), 16)
            .unwrap()
            .rules
            .is_empty()
    );
}

#[test]
fn variable_length_phrases_learn_roles_without_memorizing_entity_names() {
    let utterance = |s: &str| {
        Term::Tree(
            std::iter::once(Term::Symbol("utterance".into()))
                .chain(s.split_whitespace().map(|s| Term::Symbol(s.into())))
                .collect(),
        )
    };
    let ex = |a: &str, b: &str| Example {
        input: utterance(&format!("{a} mora em {b}")),
        output: term(json!(["tell", a, "mora", b])),
    };
    let mut seed = language_seed();
    seed.teach(
        Lesson {
            id: "span-fact".into(),
            guards: vec![],
            examples: [
                ("ana maria", "rio de janeiro"),
                ("bruno", "natal"),
                ("caio silva neto", "belo horizonte"),
                ("dora lima", "recife velho"),
            ]
            .map(|(a, b)| ex(a, b))
            .to_vec(),
        },
        &[ex("eva santos", "sao paulo"), ex("fabio", "porto alegre")],
    )
    .unwrap();
    let input = utterance("heitor de almeida costa mora em vila nova do norte");
    let expected = term(json!([
        "tell",
        "heitor de almeida costa",
        "mora",
        "vila nova do norte"
    ]));
    assert_eq!(seed.transform(&input, 32).unwrap().output, expected);
    let restored = StructuralSeed::from_bytes(&seed.to_bytes().unwrap()).unwrap();
    assert_eq!(restored.transform(&input, 32).unwrap().output, expected);
    let ambiguous = utterance("ana mora em bruno mora em recife");
    assert_eq!(
        restored.transform(&ambiguous, 32).unwrap_err(),
        Error::Ambiguous
    );
    let dir = Temporary::new();
    let mut dialogue =
        axon_uic::structural::dialogue::Dialogue::open(&dir.0.join("dialogue.axep")).unwrap();
    assert_eq!(
        dialogue
            .turn(&restored, "ana mora em bruno mora em recife", "observer")
            .unwrap()
            .status,
        "clarify"
    );
    assert_eq!(dialogue.episodes(), 0);
    let question = |a: &str| Example {
        input: utterance(&format!("qual a capital do pais de {a}")),
        output: term(json!(["path", a, "mora", "capital"])),
    };
    seed.teach(
        Lesson {
            id: "span-path".into(),
            guards: vec![],
            examples: [
                "ana maria",
                "bruno",
                "fabio de almeida",
                "caio de souza neto",
            ]
            .map(question)
            .to_vec(),
        },
        &[question("dora de lima"), question("eva")],
    )
    .unwrap();
    assert_eq!(
        seed.transform(
            &utterance("qual a capital do pais de heitor de oliveira"),
            32
        )
        .unwrap()
        .output,
        term(json!(["path", "heitor de oliveira", "mora", "capital"]))
    );
}
