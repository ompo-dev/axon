use axon_uic::{
    cli,
    numeric::{evolution::*, *},
};
use std::{
    fs,
    path::PathBuf,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
fn square() -> &'static EvolvedSkill {
    static MODEL: OnceLock<EvolvedSkill> = OnceLock::new();
    MODEL.get_or_init(|| {
        EvolvedSkill::evolve(
            "source",
            "evo/source",
            [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
                .into_iter()
                .map(|x| NumericExample::new(vec![x], x * x))
                .collect(),
            [-4.0, 4.0, 2.5, -2.5]
                .into_iter()
                .map(|x| NumericExample::new(vec![x], x * x))
                .collect(),
            &[],
            91,
            EvolutionBudget::default(),
            NumericTolerance::default(),
        )
        .unwrap()
        .skill
        .expect("source learned")
    })
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "axon-evo-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        if let (Ok(p), Ok(root)) = (self.0.canonicalize(), std::env::temp_dir().canonicalize()) {
            assert!(p.starts_with(root));
            let _ = fs::remove_dir_all(p);
        }
    }
}
#[test]
fn names_and_contexts_do_not_choose_a_handwritten_solution() {
    let s = square();
    let a = EvolvedSkill::evolve(
        "unrelated",
        "different/context",
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
            .into_iter()
            .map(|x| NumericExample::new(vec![x], x * x))
            .collect(),
        [-4.0, 4.0, 2.5, -2.5]
            .into_iter()
            .map(|x| NumericExample::new(vec![x], x * x))
            .collect(),
        &[],
        91,
        EvolutionBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap()
    .skill
    .unwrap();
    assert_eq!(s.genome_identity(), a.genome_identity());
    assert_eq!(s.program(), a.program());
    for i in -100..=100 {
        let x = i as f64 / 7.0;
        assert_eq!(s.predict(s.context(), &[x]).unwrap(), x * x);
    }
}
#[test]
fn evolution_is_reproducible_except_wall_clock_measurements() {
    let a = EvolvedSkill::evolve(
        "test",
        "repro",
        square_inputs(),
        admission(),
        &[],
        51,
        EvolutionBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap();
    let b = EvolvedSkill::evolve(
        "test",
        "repro",
        square_inputs(),
        admission(),
        &[],
        51,
        EvolutionBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap();
    assert_eq!(a.stop, b.stop);
    let mut sa = a.statistics;
    let mut sb = b.statistics;
    sa.elapsed_ns = 0;
    sb.elapsed_ns = 0;
    assert_eq!(sa, sb);
    assert_eq!(
        a.skill.unwrap().fingerprint(),
        b.skill.unwrap().fingerprint()
    );
}
fn square_inputs() -> Vec<NumericExample> {
    [-3.0, -2.0, 0.0, 1.0, 2.0, 3.0]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], x * x))
        .collect()
}
#[test]
fn observation_does_not_change_evolution_and_wiring_matches_the_export() {
    let plain = EvolvedSkill::evolve(
        "observed",
        "observed",
        square_inputs(),
        admission(),
        &[],
        51,
        EvolutionBudget::balanced(),
        NumericTolerance::default(),
    )
    .unwrap();
    let mut progress = Vec::new();
    let observed = EvolvedSkill::evolve_observed(
        "observed",
        "observed",
        square_inputs(),
        admission(),
        &[],
        51,
        EvolutionBudget::balanced(),
        NumericTolerance::default(),
        |p| progress.push(p),
    )
    .unwrap();
    assert_eq!(plain.stop, observed.stop);
    assert_eq!(plain.allocation, observed.allocation);
    let a = plain.skill.unwrap();
    let b = observed.skill.unwrap();
    assert_eq!(a.fingerprint(), b.fingerprint());
    assert_eq!(a.program(), b.program());
    assert_eq!(progress.last().unwrap().stop, Some(EvolutionStop::Admitted));
    assert!(
        progress
            .windows(2)
            .all(|p| p[0].statistics.work <= p[1].statistics.work)
    );
    let (arity, output, wiring) = b.wiring();
    assert_eq!(arity, 1);
    assert_eq!(wiring.len() % 7, 0);
    assert!(output < arity + 4 + wiring.len() / 7);
    assert_eq!(
        wiring
            .as_chunks::<7>()
            .0
            .iter()
            .filter(|g| g[6] == 1)
            .count(),
        b.active_genes()
    );
    for (i, g) in wiring.as_chunks::<7>().0.iter().enumerate() {
        assert!((1..=4).contains(&g[1]));
        assert!(
            g[2..2 + g[1] as usize]
                .iter()
                .all(|v| *v < (arity + 4 + i) as u64)
        );
    }
}

#[test]
fn sliced_refinement_preserves_budget_for_mutation_and_crossover() {
    let rows = |offset: f64| {
        (0..8)
            .map(|i| {
                let x = (i as f64 + offset) / 3.0;
                NumericExample::new(vec![x], x.sin())
            })
            .collect()
    };
    let budget = EvolutionBudget {
        population: 32,
        generations: 8,
        genes: 8,
        max_work: 200_000,
        refinement_slice_work: 1000,
        refinement_share_percent: 20,
        ..EvolutionBudget::balanced()
    };
    let r = EvolvedSkill::evolve(
        "sliced",
        "sliced",
        rows(1.0),
        rows(11.0),
        &[],
        7189,
        budget,
        NumericTolerance::default(),
    )
    .unwrap();
    assert!(r.skill.is_none());
    assert!(r.statistics.work <= budget.max_work);
    assert!(r.allocation.refinement_work <= budget.max_work / 5);
    assert!(r.allocation.truncated_refinements > 0);
    assert!(r.statistics.evaluations > 32);
    assert!(r.statistics.mutations > 0 && r.statistics.crossovers > 0);
}
fn admission() -> Vec<NumericExample> {
    [-4.0, 4.0, 5.0]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], x * x))
        .collect()
}
#[test]
fn unseen_admission_refutes_an_ambiguous_candidate_and_stops_search() {
    let r = EvolvedSkill::evolve(
        "alias",
        "alias",
        vec![
            NumericExample::new(vec![0.0], 0.0),
            NumericExample::new(vec![1.0], 1.0),
        ],
        vec![
            NumericExample::new(vec![2.0], 4.0),
            NumericExample::new(vec![3.0], 9.0),
        ],
        &[],
        19,
        EvolutionBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap();
    assert_eq!(r.stop, EvolutionStop::Refuted);
    assert!(r.skill.is_none());
    assert_eq!(r.statistics.evaluations, 1);
    assert!(r.statistics.admission_work > 0);
    let witness = r.refutation.unwrap();
    assert_eq!(witness.input, vec![2.0]);
    assert_eq!(witness.expected, 4.0);
    assert_eq!(witness.actual, Some(2.0));
}
#[test]
fn quotas_invalid_inputs_and_overlap_never_publish_a_model() {
    let r = EvolvedSkill::evolve(
        "tiny",
        "tiny",
        square_inputs(),
        admission(),
        &[],
        1,
        EvolutionBudget {
            max_work: 1,
            ..EvolutionBudget::default()
        },
        NumericTolerance::default(),
    )
    .unwrap();
    assert_eq!(r.stop, EvolutionStop::WorkLimit);
    assert!(r.skill.is_none());
    assert!(r.statistics.work <= 1);
    for b in [
        EvolutionBudget {
            population: 0,
            ..EvolutionBudget::default()
        },
        EvolutionBudget {
            genes: 65,
            ..EvolutionBudget::default()
        },
        EvolutionBudget {
            max_work: 200_000_001,
            ..EvolutionBudget::default()
        },
    ] {
        assert!(
            EvolvedSkill::evolve(
                "bad",
                "bad",
                square_inputs(),
                admission(),
                &[],
                1,
                b,
                NumericTolerance::default()
            )
            .is_err()
        );
    }
    assert!(
        EvolvedSkill::evolve(
            "bad",
            "bad",
            square_inputs(),
            square_inputs(),
            &[],
            1,
            EvolutionBudget::default(),
            NumericTolerance::default()
        )
        .is_err()
    );
    let mut invalid = square_inputs();
    invalid[0].input[0] = f64::NAN;
    assert!(
        EvolvedSkill::evolve(
            "bad",
            "bad",
            invalid,
            admission(),
            &[],
            1,
            EvolutionBudget::default(),
            NumericTolerance::default()
        )
        .is_err()
    );
}
#[test]
fn mutations_and_crossover_do_not_make_an_unsupported_target_admitted() {
    let rows = |offset: f64| {
        (0..8)
            .map(|i| {
                let x = (i as f64 + offset) / 3.0;
                NumericExample::new(vec![x], x.sin())
            })
            .collect()
    };
    let r = EvolvedSkill::evolve(
        "opaque",
        "unsupported",
        rows(1.0),
        rows(11.0),
        &[],
        7189,
        EvolutionBudget {
            population: 8,
            generations: 5,
            genes: 8,
            max_work: 2_000_000,
            local_refinement: false,
            ..EvolutionBudget::default()
        },
        NumericTolerance::default(),
    )
    .unwrap();
    assert_eq!(r.stop, EvolutionStop::GenerationLimit);
    assert!(r.skill.is_none());
    assert!(r.statistics.mutations > 0);
    assert!(r.statistics.crossovers > 0);
    assert_eq!(r.statistics.local_proposals, 0);
    assert!(r.generations.iter().any(|g| g.distinct_genomes > 1));
}
#[test]
fn acquired_structure_is_reused_on_new_channels_and_survives_restart() {
    let rows = |coords: &[(f64, f64)]| {
        coords
            .iter()
            .map(|(x, y)| NumericExample::new(vec![*x, *y], x.hypot(*y)))
            .collect()
    };
    let r = EvolvedSkill::evolve(
        "new",
        "norm",
        rows(&[
            (3.0, 4.0),
            (5.0, 12.0),
            (-3.0, 1.0),
            (1.0, 0.0),
            (0.0, 2.0),
            (2.0, 2.0),
        ]),
        rows(&[(4.0, 7.0), (11.0, 1.0), (-4.0, -5.0), (0.0, 3.0)]),
        &[square()],
        823,
        EvolutionBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap();
    let a = r.skill.unwrap();
    assert_eq!(a.dependencies(), &[square().fingerprint()]);
    let b = EvolvedSkill::from_bytes(&a.to_bytes()).unwrap();
    assert_eq!(a.program(), b.program());
    assert_eq!(a.fingerprint(), b.fingerprint());
    for i in -50..=50 {
        let x = i as f64 / 7.0;
        let y = (i + 17) as f64 / 11.0;
        assert!(
            NumericTolerance::default().agrees(b.predict("norm", &[x, y]).unwrap(), x.hypot(y))
        );
    }
}
#[test]
fn publication_context_schema_and_nonfinite_guards_are_preserved() {
    let t = Temp::new();
    let file = t.0.join("model.axevo");
    let a = square();
    a.save_new(&file).unwrap();
    let bytes = fs::read(&file).unwrap();
    assert!(a.save_new(&file).is_err());
    assert_eq!(bytes, fs::read(&file).unwrap());
    assert!(EvolvedSkill::open(&file, "wrong").is_err());
    assert!(a.predict("wrong", &[1.0]).is_err());
    assert!(a.predict(a.context(), &[1.0, 2.0]).is_err());
    assert!(a.predict(a.context(), &[f64::MAX]).is_err());
    assert!(a.predict(a.context(), &[f64::INFINITY]).is_err());
}
#[test]
fn byte_corruption_and_every_truncation_are_refused() {
    let b = square().to_bytes();
    for i in 0..b.len() {
        assert!(EvolvedSkill::from_bytes(&b[..i]).is_err());
        let mut bad = b.clone();
        bad[i] ^= 1;
        assert!(EvolvedSkill::from_bytes(&bad).is_err());
    }
}
fn resign(b: &mut [u8]) {
    let end = b.len() - 8;
    let hash = b[..end].iter().fold(0xcbf29ce484222325u64, |h, v| {
        (h ^ u64::from(*v)).wrapping_mul(0x100000001b3)
    });
    b[end..].copy_from_slice(&hash.to_le_bytes());
}
#[test]
fn semantic_validation_rejects_forward_references_even_with_recomputed_checksum() {
    let s = square();
    let mut b = s.to_bytes();
    let start = 8 + 4 + s.name().len() + 4 + s.context().len() + 16 + 4;
    b[start + 1] = 255;
    resign(&mut b);
    assert!(EvolvedSkill::from_bytes(&b).is_err());
}
#[test]
fn experimental_v1_capsules_reopen_with_stable_semantic_identity() {
    let s = square();
    let mut b = s.to_bytes();
    let end = b.len() - 8;
    b.drain(end - 8..end);
    b[..8].copy_from_slice(b"AXEV0001");
    resign(&mut b);
    let old = EvolvedSkill::from_bytes(&b).unwrap();
    assert_eq!(old.program(), s.program());
    assert_eq!(old.fingerprint(), s.fingerprint());
    assert_eq!(old.statistics().local_proposals, 0);
}
fn cmd(args: &[&str]) -> Result<String, cli::CliError> {
    cli::command(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}
#[test]
fn the_actual_axon_cli_fits_reopens_predicts_and_refuses_overwrite() {
    let t = Temp::new();
    let train = t.0.join("train.tsv");
    let valid = t.0.join("valid.tsv");
    let model = t.0.join("new.axevo");
    fs::write(&train, "x0\ty\n-3\t9\n-2\t4\n0\t0\n1\t1\n2\t4\n3\t9\n").unwrap();
    fs::write(&valid, "x0\ty\n-4\t16\n4\t16\n5\t25\n").unwrap();
    let fit = [
        "numeric-evolve",
        train.to_str().unwrap(),
        valid.to_str().unwrap(),
        "--name",
        "test",
        "--context",
        "cli/evo",
        "--skill",
        model.to_str().unwrap(),
    ];
    assert!(cmd(&fit).unwrap().contains("ADMITTED EVOLVED"));
    let bytes = fs::read(&model).unwrap();
    assert!(cmd(&fit).is_err());
    assert_eq!(bytes, fs::read(&model).unwrap());
    for backend in ["stack", "shared"] {
        let p = cmd(&[
            "evolved-predict",
            model.to_str().unwrap(),
            "--context",
            "cli/evo",
            "--values",
            "7",
            "--backend",
            backend,
        ])
        .unwrap();
        assert!(p.contains("value=49.000000000000000"));
    }
    assert!(
        cmd(&[
            "evolved-inspect",
            model.to_str().unwrap(),
            "--context",
            "other"
        ])
        .is_err()
    );
}
