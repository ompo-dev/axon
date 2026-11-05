#![cfg(feature = "curriculum")]
use axon_uic::structural::{sequence::*, *};
use serde_json::json;
fn budget() -> Budget {
    Budget {
        seed: 47,
        buckets: 4096,
        epochs: 10,
        weights: 6000,
        intent_weights: 0,
        subwords: false,
        shared_roles: false,
        minimum_margin: 1,
    }
}
fn examples() -> Vec<sequence::Example> {
    (0..32)
        .flat_map(|i| {
            [
                sequence::Example {
                    text: format!("send letter to person-{i}"),
                    intent: "send".into(),
                    tags: vec!["O".into(), "O".into(), "O".into(), "B:person".into()],
                },
                sequence::Example {
                    text: format!("cancel letter to person-{i}"),
                    intent: "cancel".into(),
                    tags: vec!["O".into(), "O".into(), "O".into(), "B:person".into()],
                },
            ]
        })
        .collect()
}
#[test]
fn sequence_learning_is_deterministic_compact_portable_and_does_not_replace_symbolic_transform() {
    let data = examples();
    let model = Model::learn(&data, budget(), None, &mut |_| Ok(())).unwrap();
    let predicted = model.predict("send letter to unseen-name").unwrap();
    assert_eq!(
        predicted.output,
        serde_json::from_value::<Term>(json!([
            "intent",
            "send",
            ["slots", ["person", "unseen-name"]]
        ]))
        .unwrap()
    );
    let seed = StructuralSeed::default();
    let old = seed.to_bytes().unwrap();
    let bytes = model.to_bytes().unwrap();
    assert!(bytes.len() < 60000);
    let model2 = Model::learn(&data, budget(), None, &mut |_| Ok(())).unwrap();
    assert_eq!(bytes, model2.to_bytes().unwrap());
    let child = Model::from_bytes(&bytes).unwrap();
    assert_eq!(model.retained_bytes(), child.retained_bytes());
    assert!(model.retained_bytes() >= model.weights() * 8);
    assert_eq!(
        child.predict("send letter to unseen-name").unwrap().output,
        predicted.output
    );
    assert!(
        seed.transform(
            &serde_json::from_value(json!(["goal", "send", "letter"])).unwrap(),
            32
        )
        .unwrap()
        .rules
        .is_empty()
    );
    assert_eq!(
        StructuralSeed::from_bytes(&old)
            .unwrap()
            .to_bytes()
            .unwrap(),
        old
    );
    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["revision"], 1);
    wire["revision"] = json!(0);
    assert!(Model::from_bytes(&serde_json::to_vec(&wire).unwrap()).is_err());
    assert!(
        !String::from_utf8(bytes.clone())
            .unwrap()
            .contains("person-31")
    );
    wire = serde_json::from_slice(&bytes).unwrap();
    wire["packed"] = json!("00000000");
    assert!(Model::from_bytes(&serde_json::to_vec(&wire).unwrap()).is_err());
}
#[test]
fn sequence_bounds_cancellation_and_supervision_are_enforced() {
    let mut b = budget();
    b.buckets = 4000;
    assert!(Model::learn(&examples(), b, None, &mut |_| Ok(())).is_err());
    b = budget();
    b.intent_weights = b.weights;
    assert!(Model::learn(&examples(), b, None, &mut |_| Ok(())).is_err());
    assert!(matches!(
        Model::learn(&examples(), budget(), None, &mut |_| Err(Error::Budget)),
        Err(Error::Budget)
    ));
    for text in ["", "x\ny"] {
        assert!(tokens(text).is_err());
    }
    assert!(tokens(&"x ".repeat(65)).is_err());
    assert_eq!(tokens("Say 003.").unwrap(), vec!["say", "003"]);
    let mut data = examples();
    data[0].tags[0] = "I:person".into();
    assert!(Model::learn(&data, budget(), None, &mut |_| Ok(())).is_err());
    let model = Model::learn(&examples(), budget(), None, &mut |_| Ok(())).unwrap();
    let mut mismatch = budget();
    mismatch.seed += 1;
    assert!(Model::learn(&examples(), mismatch, Some(&model), &mut |_| Ok(())).is_err());
}

#[test]
fn refinement_preserves_schema_and_inherited_bindings_with_explicit_replay() {
    let data = examples();
    let model = Model::learn(&data, budget(), None, &mut |_| Ok(())).unwrap();
    let refined = Model::learn(&data, budget(), Some(&model), &mut |_| Ok(())).unwrap();
    for (text, meaning) in [
        ("send letter to person-1", "send"),
        ("cancel letter to person-1", "cancel"),
    ] {
        let before = model.predict(text).unwrap();
        let prediction = refined.predict(text).unwrap();
        assert_eq!(prediction.output, before.output);
        let Term::Tree(v) = prediction.output else {
            panic!("missing structure")
        };
        assert_eq!(v[1], Term::Symbol(meaning.into()));
    }
    let original = refined.to_bytes().unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
    for packed in ["00000000", "ffffff01", "0000000100000001", "x0000001"] {
        value["packed"] = json!(packed);
        assert!(Model::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
