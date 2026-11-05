use axon_uic::{TextBudget, TextExample, TextPrediction, TextProgram, TextValuePrediction};
fn stripped(p: TextPrediction) -> TextValuePrediction {
    match p {
        TextPrediction::Agreed(w) => TextValuePrediction::Agreed(w.output),
        TextPrediction::Ambiguous { first, second } => TextValuePrediction::Ambiguous {
            first: first.output,
            second: second.output,
        },
        TextPrediction::OutOfDomain => TextValuePrediction::OutOfDomain,
    }
}
#[test]
fn value_only_interpreter_matches_explanatory_reference_on_unseen_strings() {
    let families = [
        vec![
            TextExample::new("Ana Maria|Silva", "Silva, Ana Maria"),
            TextExample::new("Bruno|Costa", "Costa, Bruno"),
        ],
        vec![
            TextExample::new("Ana Silva", "Silva"),
            TextExample::new("Bruno Costa", "Costa"),
        ],
        vec![
            TextExample::new(" Ana Maria | 1250 ", "ANA MARIA|1250"),
            TextExample::new("bruno | 902", "BRUNO|902"),
            TextExample::new("Carla Sofia Rocha | 34567", "CARLA SOFIA ROCHA|34567"),
        ],
        vec![
            TextExample::new("  ana   maria  ", "ANA MARIA"),
            TextExample::new("bruno  costa", "BRUNO COSTA"),
        ],
    ];
    let parts = [
        "",
        "ana",
        "Carla Sofia",
        "123",
        "  Two   Words  ",
        "Stra\u{df}e",
        "\u{00e7}\u{00e3}\u{00f5}",
        "\u{1f642}",
        "one two three",
        "a|b",
    ];
    let mut checked = 0;
    for examples in families {
        let p = TextProgram::learn(&examples, TextBudget::default()).unwrap();
        for a in parts {
            for b in parts {
                for delimiter in ["|", " ", "", ":", "\t"] {
                    let input = format!("{a}{delimiter}{b}");
                    assert_eq!(
                        p.predict_value(&input),
                        p.predict(&input).map(stripped),
                        "input {input:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(
            p.predict_value(&"x".repeat(1025)),
            p.predict(&"x".repeat(1025)).map(stripped)
        );
    }
    assert_eq!(checked, 2000);
}
#[test]
fn value_only_interpreter_preserves_expanding_unicode_output_refusal() {
    let budget = TextBudget {
        max_output_bytes: 8,
        ..TextBudget::default()
    };
    let p = TextProgram::learn(
        &[
            TextExample::new("abc", "ABC"),
            TextExample::new("def", "DEF"),
        ],
        budget,
    )
    .unwrap();
    for input in [
        "\u{df}\u{df}\u{df}\u{df}\u{df}",
        "hello world",
        "abc",
        "",
        "\u{fb03}\u{fb03}\u{fb03}",
    ] {
        assert_eq!(p.predict_value(input), p.predict(input).map(stripped));
    }
}
