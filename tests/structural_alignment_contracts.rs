#![cfg(feature = "curriculum")]
use axon_uic::structural::{alignment::*, induction::*, *};
use serde_json::json;
fn term(value: serde_json::Value) -> Term {
    serde_json::from_value(value).unwrap()
}
#[test]
fn contrast_preserves_changed_words_without_assigning_their_meaning() {
    let a = unique("one sees two", "one never sees two").unwrap();
    assert_eq!(
        a,
        term(json!([
            "contrast-v1",
            3,
            4,
            [
                "steps",
                ["copy", 1],
                ["replace", "empty", ["tokens", "never"]],
                ["copy", 2]
            ]
        ]))
    );
    assert_eq!(
        a,
        unique("alpha hears beta", "alpha never hears beta").unwrap()
    );
    assert_ne!(a, unique("one sees two", "one rarely sees two").unwrap());
    assert_ne!(unique("one sees two", "one sees two").unwrap(), a);
    assert_eq!(
        unique("one sees two", "two sees one"),
        Err(Error::Ambiguous)
    );
    assert_eq!(
        unique("1 2 3", "1 2 3").unwrap(),
        unique("x y z", "x y z").unwrap()
    );
}
#[test]
fn tied_alignments_are_exposed_and_bounds_are_refusals() {
    assert_eq!(alternatives("a b", "b a").unwrap().len(), 2);
    assert_eq!(unique("a b", "b a"), Err(Error::Ambiguous));
    assert!(unique("", "hello").is_err());
    assert!(unique("hello\nworld", "hello").is_err());
    assert!(unique(&"word ".repeat(100), "hello").is_err());
    assert!(unique("a (b c)", "a b c").is_err());
    assert!(alternatives(&"a ".repeat(31), &"a ".repeat(15)).is_err());
}
#[test]
fn topology_ignores_only_copy_lengths_and_never_erases_changed_words_or_direction() {
    assert_eq!(
        topology("one sees two", "one never sees two").unwrap(),
        topology(
            "several new people see things",
            "several new people never see things"
        )
        .unwrap()
    );
    assert_ne!(
        topology("one sees two", "one never sees two").unwrap(),
        topology("one never sees two", "one sees two").unwrap()
    );
    assert_ne!(
        topology("one sees two", "one never sees two").unwrap(),
        topology("one sees two", "one rarely sees two").unwrap()
    );
    assert_eq!(
        topology("one", "one").unwrap(),
        topology(
            "many completely different words",
            "many completely different words"
        )
        .unwrap()
    );
    assert!(topology("a b", "b a").is_err());
}
fn copied(t: &Term) -> usize {
    match t {
        Term::Tree(v) if v[0] == Term::Symbol("copy".into()) => match v[1] {
            Term::Integer(n) => n as usize,
            _ => panic!("copy type"),
        },
        Term::Tree(v) => v.iter().map(copied).sum(),
        _ => 0,
    }
}
#[test]
fn alignment_matches_exhaustive_subsequence_oracle() {
    let mut rows = Vec::new();
    for n in 1..=4 {
        for bits in 0..1 << n {
            rows.push(
                (0..n)
                    .map(|i| if bits & (1 << i) == 0 { "x" } else { "y" })
                    .collect::<Vec<_>>(),
            );
        }
    }
    for a in &rows {
        for b in &rows {
            let mut maximum = 0;
            for mask in 1..1 << a.len() {
                let sub = a
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, v)| v)
                    .collect::<Vec<_>>();
                let mut at = 0;
                for word in b {
                    if sub.get(at).is_some_and(|v| *v == word) {
                        at += 1;
                    }
                }
                if at == sub.len() {
                    maximum = maximum.max(at);
                }
            }
            match alternatives(&a.join(" "), &b.join(" ")) {
                Ok(outputs) => {
                    assert!(!outputs.is_empty());
                    assert!(outputs.iter().all(|out| copied(out) == maximum));
                }
                Err(Error::Ambiguous | Error::Budget) => (),
                Err(error) => panic!("unexpected {error}"),
            }
        }
    }
}
#[test]
fn identity_meaning_is_learned_from_supervision_not_hardcoded() {
    for label in [1, 2] {
        let data = (2..=15)
            .map(|n| {
                let text = (0..n)
                    .map(|i| format!("item-{i}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                Example {
                    input: unique(&text, &text).unwrap(),
                    output: Term::Integer(label),
                }
            })
            .collect::<Vec<_>>();
        let budget = Budget {
            seed: 47,
            proposals: 64,
            rules: 16,
            checks: 100000,
            minimum_holes: 1,
            structural_groups: true,
        };
        let (seed, stats) = learn(&data, budget, &mut |_| Ok(())).unwrap();
        assert_eq!(stats.correct, data.len());
        assert_eq!(stats.wrong, 0);
        let child = StructuralSeed::from_bytes(&seed.to_bytes().unwrap()).unwrap();
        let query = unique(
            "new arbitrary words outside all training",
            "new arbitrary words outside all training",
        )
        .unwrap();
        assert_eq!(
            child.transform(&query, 64).unwrap().output,
            Term::Integer(label)
        );
        let changed = unique("new arbitrary words", "new arbitrary other words").unwrap();
        assert!(child.transform(&changed, 64).unwrap().rules.is_empty());
    }
}
