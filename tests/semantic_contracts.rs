#![cfg(feature = "text-workflows")]
use axon_uic::{demand::genome::Genome, numeric::NumericBudget, semantic::*, semantic_cli::*};
use std::{
    path::{Path, PathBuf},
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
fn seed() -> &'static SemanticSeed {
    static SEED: OnceLock<SemanticSeed> = OnceLock::new();
    SEED.get_or_init(|| {
        let lessons = load_lessons(Path::new("evals/fixtures/relational-v1-lessons.tsv")).unwrap();
        SemanticSeed::new(
            lessons
                .values()
                .map(|ds| Rule::learn(ds, NumericBudget::default()).unwrap())
                .collect(),
        )
        .unwrap()
    })
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "axon-semantic-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let p = self.0.canonicalize().unwrap();
        assert!(p.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        std::fs::remove_dir_all(p).unwrap();
    }
}
#[test]
fn composed_transfers_preserve_roles_and_emit_real_evidence() {
    let mut s = Session::default();
    let a=s.narrative(seed(),"Sofia tinha 18 livros. Bruno tinha 0 livros. Sofia emprestou 5 livros para Bruno. Sofia recebeu 2 livros. Bruno devolveu 3 livros para Sofia. Quantos livros ficaram com Sofia?").unwrap();
    assert_eq!(a.value, 18.0);
    assert_eq!(s.fact("Bruno", "livros"), Some(2.0));
    assert_eq!(a.evidence.len(), 7);
    assert!(a.evidence.iter().any(|e| e.owner == "sofia"
        && e.before == Some(18.0)
        && e.after == 13.0
        && e.expression.contains('-')));
    assert_eq!(Session::default().narrative(seed(),"Sofia tinha 18 livros. Bruno tinha 0 livros. Sofia emprestou 5 livros para Bruno. Quantos livros ficaram com Sofia?").unwrap().value,13.0);
}
#[test]
fn ongoing_world_changes_without_retraining_and_questions_do_not_reset_it() {
    let mut s = Session::default();
    s.apply(seed(), "Vera tinha 28 caixas").unwrap();
    assert_eq!(
        s.apply(seed(), "Quantos caixas ficaram com Vera")
            .unwrap()
            .unwrap()
            .value,
        28.0
    );
    s.apply(seed(), "Ela recebeu 6 caixas").unwrap();
    assert_eq!(
        s.apply(seed(), "Quantos caixas ela tem")
            .unwrap()
            .unwrap()
            .value,
        34.0
    );
    s.apply(seed(), "Vera gastou 3 caixas").unwrap();
    assert_eq!(s.fact("Vera", "caixas"), Some(31.0));
}
#[test]
fn ambiguity_missing_facts_unknown_words_and_units_are_not_invented() {
    let mut s = Session::default();
    assert!(matches!(
        s.apply(seed(), "Quantos livros ficaram com Ana"),
        Err(SemanticError::Missing)
    ));
    s.apply(seed(), "Ana tinha 18 livros").unwrap();
    s.apply(seed(), "Bruno tinha 0 livros").unwrap();
    s.apply(seed(), "Ana emprestou 5 livros para Bruno")
        .unwrap();
    assert!(matches!(
        s.apply(seed(), "Ela recebeu 2 livros"),
        Err(SemanticError::Ambiguous)
    ));
    assert!(matches!(
        s.apply(seed(), "Ana emprestou 1 livros para Ana"),
        Err(SemanticError::Ambiguous)
    ));
    assert!(matches!(
        s.apply(seed(), "Ana queimou 4 livros"),
        Err(SemanticError::Unknown)
    ));
    assert!(matches!(
        s.apply(seed(), "Quantos moedas ficaram com Ana"),
        Err(SemanticError::Missing)
    ));
    assert_eq!(s.fact("Ana", "livros"), Some(13.0));
}
#[test]
fn effects_and_whole_narratives_are_atomic_on_failure() {
    let mut s = Session::default();
    s.apply(seed(), "Ana tinha 18 livros").unwrap();
    assert!(
        s.apply(seed(), "Ana emprestou 5 livros para Bruno")
            .is_err()
    );
    assert_eq!(s.fact("Ana", "livros"), Some(18.0));
    let before = s.evidence().to_vec();
    assert!(
        s.narrative(
            seed(),
            "Ana recebeu 4 livros. Ana queimou 2 livros. Quantos livros ficaram com Ana?"
        )
        .is_err()
    );
    assert_eq!(s.fact("Ana", "livros"), Some(18.0));
    assert_eq!(s.evidence(), before);
}
#[test]
fn numeric_programs_generalize_decimals_negative_inputs_and_domain_refusal() {
    let mut s = Session::default();
    for (text, n) in [
        ("Quanto custa comprar 2,5 unidades a 8 por unidade?", 20.0),
        ("Quanto custa comprar 2.5 unidades a 8 por unidade?", 20.0),
        ("Qual x satisfaz -3 vezes x mais 5 igual a 14?", -3.0),
        ("Qual a norma de -16 e 63?", 65.0),
    ] {
        assert!((s.narrative(seed(), text).unwrap().value - n).abs() < 1e-8);
    }
    assert!(matches!(
        s.narrative(seed(), "Qual x satisfaz 0 vezes x mais 5 igual a 14?"),
        Err(SemanticError::Domain)
    ));
}
#[test]
fn negation_is_a_learned_rule_not_substring_arithmetic() {
    let mut s = Session::default();
    assert_eq!(
        s.narrative(
            seed(),
            "Ana tinha 18 livros. Ana nao recebeu 7 livros. Quantos livros ficaram com Ana?"
        )
        .unwrap()
        .value,
        18.0
    );
    assert_eq!(s.evidence().len(), 1);
    assert!(s.apply(seed(), "Ana nunca recebeu 7 livros").is_err());
}
#[test]
fn portable_number_is_sufficient_for_birth_but_contains_no_personal_state() {
    let t = Temp::new();
    let genome = pack(seed(), &t.0.join("source")).unwrap();
    let g = Genome::from_number(&genome.number()).unwrap();
    assert_eq!(g.to_bytes(), genome.to_bytes());
    std::fs::rename(t.0.join("source"), t.0.join("unavailable")).unwrap();
    for name in ["birth1", "birth2"] {
        let copy = born(&g, &t.0.join(name)).unwrap();
        assert_eq!(copy.fingerprint(), seed().fingerprint());
        let mut s = Session::default();
        assert!(matches!(
            s.apply(&copy, "Quantos livros ficaram com Ana"),
            Err(SemanticError::Missing)
        ));
        assert_eq!(
            s.narrative(
                &copy,
                "Ana tinha 7 livros. Ana recebeu 4 livros. Quantos livros ficaram com Ana?"
            )
            .unwrap()
            .value,
            11.0
        );
    }
    assert!(born(&g, &t.0.join("birth1")).is_err());
}
#[test]
fn codec_detects_corruption_and_duplicate_rules_refuse_instead_of_choosing() {
    let bytes = seed().to_bytes().unwrap();
    assert_eq!(
        SemanticSeed::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    for i in [0, 8, 24, bytes.len() - 1] {
        let mut copy = bytes.clone();
        copy[i] ^= 1;
        assert!(SemanticSeed::from_bytes(&copy).is_err());
    }
    for n in [0, 8, 16, bytes.len() - 1] {
        assert!(SemanticSeed::from_bytes(&bytes[..n]).is_err());
    }
    let duplicate =
        SemanticSeed::new(vec![seed().rules()[0].clone(), seed().rules()[0].clone()]).unwrap();
    assert!(matches!(
        Session::default().apply(&duplicate, "Ana tinha 18 livros"),
        Err(SemanticError::Ambiguous)
    ));
}
#[test]
fn underdetermined_role_supervision_is_rejected() {
    let mut lessons = load_lessons(Path::new("evals/fixtures/relational-v1-lessons.tsv")).unwrap();
    let ds = lessons.get_mut("08-reference").unwrap();
    for d in &mut *ds {
        if let Meaning::Change(v) = &d.meaning {
            d.focus.items = vec![v[0].item.clone()];
        }
    }
    assert!(matches!(
        Rule::learn(ds, NumericBudget::default()),
        Err(SemanticError::Ambiguous)
    ));
}
#[test]
fn empty_seed_does_not_come_with_portuguese_or_formulas() {
    let empty = SemanticSeed::default();
    let copy = SemanticSeed::from_bytes(&empty.to_bytes().unwrap()).unwrap();
    assert!(matches!(
        Session::default().narrative(
            &copy,
            "Ana tinha 18 livros. Quantos livros ficaram com Ana?"
        ),
        Err(SemanticError::Unknown)
    ));
}

#[test]
fn complementary_structures_merge_deterministically_without_reteaching() {
    let a = seed().subset(0b101010101010101).unwrap();
    let b = seed().subset(0b010101010101010).unwrap();
    let child = a.merge(&b).unwrap();
    assert_eq!(child.rules().len(), 15);
    assert_eq!(
        child.to_bytes().unwrap(),
        b.merge(&a).unwrap().to_bytes().unwrap()
    );
    assert_eq!(
        child.to_bytes().unwrap(),
        child.merge(&child).unwrap().to_bytes().unwrap()
    );
    let mut world = Session::default();
    assert_eq!(world.narrative(&child,"Ana tinha 18 livros. Bruno tinha 0 livros. Ana emprestou 5 livros para Bruno. Ana recebeu 2 livros. Quantos livros ficaram com Ana?").unwrap().value,15.0);
    assert!(matches!(
        Session::default().apply(&child, "Quantos livros ficaram com Ana"),
        Err(SemanticError::Missing)
    ));
}

#[test]
fn conflicting_inherited_meanings_are_rejected_without_changing_parents() {
    let mut lessons = load_lessons(Path::new("evals/fixtures/relational-v1-lessons.tsv")).unwrap();
    let ds = lessons.get_mut("01-declare").unwrap();
    for d in &mut *ds {
        d.meaning = Meaning::Ignore;
    }
    let wrong =
        SemanticSeed::new(vec![Rule::learn(ds, NumericBudget::default()).unwrap()]).unwrap();
    let identity = seed().fingerprint();
    assert!(matches!(
        seed().merge(&wrong),
        Err(SemanticError::Ambiguous)
    ));
    assert_eq!(seed().fingerprint(), identity);
}

#[test]
fn structural_identity_and_masks_ignore_training_time_not_semantic_roles() {
    let lessons = load_lessons(Path::new("evals/fixtures/relational-v1-lessons.tsv")).unwrap();
    let independent = SemanticSeed::new(
        lessons
            .values()
            .map(|ds| Rule::learn(ds, NumericBudget::default()).unwrap())
            .collect(),
    )
    .unwrap();
    assert_eq!(
        independent.structural_bytes().unwrap(),
        seed().structural_bytes().unwrap()
    );
    let partial = independent.subset(17321).unwrap();
    assert_eq!(seed().mask_for(&partial).unwrap(), 17321);
    assert_eq!(
        seed()
            .mask_for(
                &seed()
                    .subset(1)
                    .unwrap()
                    .merge(&seed().subset(2).unwrap())
                    .unwrap()
            )
            .unwrap(),
        3
    );
}
