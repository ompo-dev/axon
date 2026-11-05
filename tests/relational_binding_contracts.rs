use axon_uic::{numeric::NumericBudget, semantic::*};
#[cfg(feature = "curriculum")]
#[test]
fn decimal_labels_and_sentence_numbers_have_identical_roundtrip() {
    for s in [
        "2.4615384615384617",
        "4.6923076923076925",
        "7.230769230769231",
        "12.23076923076923",
    ] {
        assert_eq!(
            serde_json::from_str::<f64>(s).unwrap().to_bits(),
            s.parse::<f64>().unwrap().to_bits(),
            "{s}"
        );
    }
    for i in 1..=12 {
        for j in 0..3 {
            let x = ((i * (31 + j * 18) + j * 11) % 997 + 1) as f64 / 13.0;
            let text = x.to_string();
            let decoded: serde_json::Value =
                serde_json::from_str(&format!("{{\"arguments\":[{text}]}}")).unwrap();
            assert_eq!(
                decoded["arguments"][0].as_f64().unwrap().to_bits(),
                text.parse::<f64>().unwrap().to_bits(),
                "{text}"
            );
        }
    }
}
fn teaching(arity: usize, label: &str, f: impl Fn(&[f64]) -> f64) -> Vec<Demonstration> {
    (0..12)
        .map(|i| {
            let x: Vec<_> = (0..arity)
                .map(|j| ((i * 13 + j * 7 + 3) % 31) as f64 / 3.0 + 0.2)
                .collect();
            let mut pairs: Vec<_> = x
                .iter()
                .enumerate()
                .map(|(j, n)| format!("campo{j} {n}"))
                .collect();
            if i % 2 == 1 {
                pairs.reverse();
            }
            let sentence = if i % 3 == 0 {
                format!("agora calcule {label} com {}", pairs.join(" e "))
            } else if i % 3 == 1 {
                format!("{label} {} agora", pairs.join(" com "))
            } else {
                format!("calcule {label} {}", pairs.join(" e "))
            };
            Demonstration {
                sentence,
                focus: Focus::default(),
                meaning: Meaning::Calculate {
                    answer: f(&x),
                    arguments: x,
                },
            }
        })
        .collect()
}
#[test]
fn roles_transfer_order_not_unknown_words_and_new_codec_is_checked() {
    let rule = Rule::learn_relational(
        &teaching(2, "compra", |x| x[0] - x[1]),
        NumericBudget::default(),
        1,
        &[],
    )
    .unwrap();
    let seed = SemanticSeed::new(vec![rule]).unwrap();
    assert_eq!(
        Session::default()
            .narrative(&seed, "com campo1 3 agora compra e campo0 13 calcule")
            .unwrap()
            .value,
        10.0
    );
    for input in [
        "nao compra campo0 13 campo1 3",
        "compra campo0 13 campo1 3 campo1 8",
        "compra campo0 13",
        "compra campo0 13 campo1 NaN",
    ] {
        assert!(Session::default().narrative(&seed, input).is_err());
    }
    let encoded = seed.to_bytes().unwrap();
    assert_eq!(&encoded[..8], b"AXSM0002");
    let reopened = SemanticSeed::from_bytes(&encoded).unwrap();
    assert_eq!(
        reopened.structural_bytes().unwrap(),
        seed.structural_bytes().unwrap()
    );
    for i in 0..encoded.len() {
        assert!(SemanticSeed::from_bytes(&encoded[..i]).is_err());
        let mut broken = encoded.clone();
        broken[i] ^= 1;
        assert!(SemanticSeed::from_bytes(&broken).is_err());
    }
}
#[test]
fn inherited_algorithm_learns_a_new_composition_under_same_search_ceiling() {
    let parent = Rule::learn_relational(
        &teaching(2, "norma", |x| x[0].hypot(x[1])),
        NumericBudget::default(),
        1,
        &[],
    )
    .unwrap();
    let ds = teaching(3, "espaco", |x| x[0].hypot(x[1]).hypot(x[2]));
    let budget = NumericBudget {
        max_cost: 5,
        ..NumericBudget::default()
    };
    assert!(Rule::learn_relational(&ds, budget, 1, &[]).is_err());
    let child =
        Rule::learn_relational(&ds, budget, parent.learning_window().unwrap(), &[&parent]).unwrap();
    let original = SemanticSeed::new(vec![parent]).unwrap();
    let next = original
        .merge(&SemanticSeed::new(vec![child]).unwrap())
        .unwrap();
    assert_eq!(original.rules().len(), 1);
    assert_eq!(next.rules().len(), 2);
    assert!(
        axon_uic::numeric::NumericTolerance::default().agrees(
            Session::default()
                .narrative(&next, "espaco campo2 12 campo0 3 campo1 4")
                .unwrap()
                .value,
            13.0
        )
    );
    assert_eq!(
        Session::default()
            .narrative(&next, "norma campo1 4 campo0 3")
            .unwrap()
            .value,
        5.0
    );
}
