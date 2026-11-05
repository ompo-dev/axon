use axon_uic::{
    cli,
    numeric::{evolution::*, seed::*, *},
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
        let rows = |inputs: &[f64]| {
            inputs
                .iter()
                .map(|x| NumericExample::new(vec![*x], x * x))
                .collect()
        };
        EvolvedSkill::evolve(
            "module",
            "seed/module",
            rows(&[-3., -2., 0., 1., 2., 3.]),
            rows(&[-4., 4., 5.]),
            &[],
            91,
            EvolutionBudget::balanced(),
            NumericTolerance::default(),
        )
        .unwrap()
        .skill
        .unwrap()
    })
}
fn founder() -> DevelopmentSeed {
    DevelopmentSeed::new("scope", EvolutionBudget::balanced(), 65536).unwrap()
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "axon-dna-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let p = self.0.canonicalize().unwrap();
        assert!(p.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        fs::remove_dir_all(p).unwrap();
    }
}
fn command(args: &[&str]) -> Result<String, cli::CliError> {
    cli::command(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}

#[test]
fn offspring_are_reproducible_keep_caps_and_do_not_forget_or_relax_the_verifier() {
    let parent = founder().extend("scope", square().clone()).unwrap();
    let original = parent.to_bytes();
    for random in 1..64 {
        let a = parent.mutate("scope", random).unwrap();
        let b = parent.mutate("scope", random).unwrap();
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.revision_id(), b.revision_id());
        assert_eq!(a.to_bytes(), b.to_bytes());
        assert_ne!(a.fingerprint(), parent.fingerprint());
        assert_ne!(a.revision_id(), parent.revision_id());
        assert_eq!(a.parents(), [parent.revision_id(), 0]);
        assert_eq!(a.generation(), parent.generation() + 1);
        assert_eq!(a.policy().max_work, parent.policy().max_work);
        assert_eq!(a.byte_cap(), parent.byte_cap());
        assert_eq!(a.skills()[0].to_bytes(), square().to_bytes());
        assert_eq!(a.skills()[0].predict("seed/module", &[7.]).unwrap(), 49.);
        assert!(a.skills()[0].predict("wrong", &[7.]).is_err());
    }
    assert_eq!(original, parent.to_bytes());
    assert!(parent.mutate("other", 1).is_err());
    let crossed = parent
        .crossover("scope", &parent.mutate("scope", 11).unwrap(), 51)
        .unwrap();
    assert_eq!(crossed.skills().len(), 1);
    assert_eq!(crossed.parents()[0], parent.revision_id());
    assert_ne!(crossed.parents()[1], 0);
    assert!(parent.extend("scope", square().clone()).is_err());
    assert!(
        parent
            .crossover(
                "scope",
                &DevelopmentSeed::new("scope", EvolutionBudget::balanced(), 512).unwrap(),
                1
            )
            .is_err()
    );
    assert_eq!(parent.budget(99).unwrap().max_work, 99);
    assert!(parent.budget(0).is_err());
    assert!(parent.budget(200000001).is_err());
}
#[test]
fn growth_is_refused_transactionally_and_context_and_corruption_are_checked() {
    let seed = DevelopmentSeed::new("scope", EvolutionBudget::balanced(), 512).unwrap();
    let before = seed.to_bytes();
    assert!(seed.extend("scope", square().clone()).is_err());
    assert_eq!(before, seed.to_bytes());
    let seed = founder().extend("scope", square().clone()).unwrap();
    let bytes = seed.to_bytes();
    assert_eq!(
        seed.revision_id(),
        DevelopmentSeed::from_bytes(&bytes).unwrap().revision_id()
    );
    for i in 0..bytes.len() {
        assert!(DevelopmentSeed::from_bytes(&bytes[..i]).is_err());
        let mut damaged = bytes.clone();
        damaged[i] ^= 1;
        assert!(DevelopmentSeed::from_bytes(&damaged).is_err());
    }
    let temp = Temp::new();
    let path = temp.0.join("seed.axdna");
    seed.save_new(&path).unwrap();
    assert!(seed.save_new(&path).is_err());
    assert_eq!(bytes, fs::read(&path).unwrap());
    assert!(DevelopmentSeed::open(&path, "other").is_err());
    let mut bad = bytes.clone();
    let local_flag = 8 + 4 + seed.context().len() + 7 * 8;
    bad[local_flag] = 2;
    let end = bad.len() - 8;
    let digest = bad[..end].iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    bad[end..].copy_from_slice(&digest.to_le_bytes());
    assert!(DevelopmentSeed::from_bytes(&bad).is_err());
}
#[test]
fn pareto_selection_preserves_specialists_and_rejects_invalid_scores() {
    let base = SeedScore {
        capabilities: 1,
        tasks: 3,
        retention_ok: true,
        violations: 0,
        work: 10,
        bytes: 100,
        instructions: 5,
    };
    let scores = [
        base,
        SeedScore {
            capabilities: 2,
            ..base
        },
        SeedScore { work: 20, ..base },
        SeedScore {
            capabilities: 3,
            work: 30,
            ..base
        },
        SeedScore {
            capabilities: 7,
            retention_ok: false,
            ..base
        },
        SeedScore {
            capabilities: 7,
            violations: 1,
            ..base
        },
    ];
    assert_eq!(pareto_front(&scores).unwrap(), vec![0, 1, 3]);
    assert_eq!(
        pareto_front(&[
            base,
            SeedScore {
                capabilities: 3,
                ..base
            }
        ])
        .unwrap(),
        vec![1]
    );
    assert!(
        pareto_front(&[SeedScore {
            capabilities: 8,
            ..base
        }])
        .is_err()
    );
    assert!(pareto_front(&[base, SeedScore { tasks: 2, ..base }]).is_err());
    assert!(pareto_front(&[]).is_err());
}
#[test]
fn empty_evidence_cannot_create_a_skill_or_modify_inherited_knowledge() {
    let seed = founder().extend("scope", square().clone()).unwrap();
    let bytes = seed.to_bytes();
    let result = EvolvedSkill::evolve(
        "unknown",
        "unknown",
        Vec::new(),
        vec![
            NumericExample::new(vec![1.], 1.),
            NumericExample::new(vec![2.], 2.),
        ],
        &seed.skills().iter().collect::<Vec<_>>(),
        51,
        seed.budget(1000).unwrap(),
        NumericTolerance::default(),
    );
    assert!(result.is_err());
    assert_eq!(seed.to_bytes(), bytes);
}
#[test]
fn real_cli_creates_mutates_extends_reopens_and_uses_a_genome_for_new_learning() {
    let t = Temp::new();
    let module = t.0.join("source.axevo");
    square().save_new(&module).unwrap();
    let list = t.0.join("capsules.txt");
    fs::write(&list, "source.axevo\n").unwrap();
    let parent = t.0.join("parent.axdna");
    let child = t.0.join("child.axdna");
    let create = [
        "dna-create",
        list.to_str().unwrap(),
        "--context",
        "scope",
        "--dna",
        parent.to_str().unwrap(),
    ];
    assert!(command(&create).unwrap().contains("modules=1"));
    assert!(command(&create).is_err());
    assert!(
        command(&[
            "dna-mutate",
            parent.to_str().unwrap(),
            "--context",
            "scope",
            "--dna",
            child.to_str().unwrap(),
            "--seed",
            "11"
        ])
        .unwrap()
        .contains("generation=2")
    );
    let identity = format!("{:016x}", square().fingerprint());
    assert!(
        command(&[
            "dna-predict",
            child.to_str().unwrap(),
            "--context",
            "scope",
            "--module",
            &identity,
            "--values",
            "7"
        ])
        .unwrap()
        .contains("value=49.000000000000000")
    );
    let train = t.0.join("train.tsv");
    let admission = t.0.join("admission.tsv");
    fs::write(
        &train,
        "x0\tx1\ty\n1\t2\t5\n2\t3\t13\n3\t4\t25\n-1\t-3\t10\n0\t2\t4\n4\t0\t16\n",
    )
    .unwrap();
    fs::write(&admission, "x0\tx1\ty\n5\t1\t26\n7\t3\t58\n-3\t4\t25\n").unwrap();
    let learned = t.0.join("sum-squares.axevo");
    let fit = [
        "numeric-evolve",
        train.to_str().unwrap(),
        admission.to_str().unwrap(),
        "--context",
        "new/task",
        "--name",
        "opaque",
        "--skill",
        learned.to_str().unwrap(),
        "--genome",
        parent.to_str().unwrap(),
        "--genome-context",
        "scope",
        "--seed",
        "823",
    ];
    assert!(command(&fit).unwrap().contains("ADMITTED EVOLVED"));
    let revision = t.0.join("expanded.axdna");
    assert!(
        command(&[
            "dna-extend",
            parent.to_str().unwrap(),
            learned.to_str().unwrap(),
            "--context",
            "scope",
            "--dna",
            revision.to_str().unwrap()
        ])
        .unwrap()
        .contains("modules=2")
    );
    let mut overridden = fit.to_vec();
    overridden.extend(["--refinement", "mutation-only"]);
    assert!(command(&overridden).is_err());
    assert!(
        command(&[
            "dna-inspect",
            parent.to_str().unwrap(),
            "--context",
            "wrong"
        ])
        .is_err()
    );
}
