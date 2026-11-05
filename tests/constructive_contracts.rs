use axon_uic::numeric::{
    evolution::{constructive::*, *},
    *,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "axon-constructive-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let path = self.0.canonicalize().unwrap();
        assert!(path.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        fs::remove_dir_all(path).unwrap();
    }
}
fn examples(irrational: bool) -> Vec<NumericExample> {
    [-3.0_f64, -1.5, 0.0, 0.7, 1.0, 2.0, 4.0]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], if irrational { x.sin() } else { x * x }))
        .collect()
}
#[test]
fn new_circuit_is_not_a_subset_and_birth_needs_no_examples_or_teacher() {
    let temp = Temp::new();
    let mut search = ConstructiveSearch::open(
        &temp.0.join("search.axep"),
        examples(false),
        &[],
        ConstructiveBudget::default(),
        NumericTolerance::default(),
        91,
    )
    .unwrap();
    search.run(2000, |_| {}).unwrap();
    let dna = search.winner().expect("square acquired from primitives");
    let number = dna.number();
    assert!(number.len() > 20);
    let file = temp.0.join("circuit-number.txt");
    fs::write(&file, number.as_bytes()).unwrap();
    let predict = |path: &std::path::Path, input: &str| {
        std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
            .arg("circuit-predict")
            .arg(path)
            .arg(input)
            .output()
            .unwrap()
    };
    let output = predict(&file, "3.5");
    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("value=12.25")
    );
    assert!(!predict(&temp.0.join("missing.txt"), "1").status.success());
    assert!(!predict(&file, "NaN").status.success());
    let born = CircuitDna::from_number(&number).unwrap().program().unwrap();
    for i in -333..=333 {
        let x = f64::from(i) / 17.0;
        assert!(NumericTolerance::default().agrees(born.evaluate(&[x]).unwrap(), x * x));
    }
    let admission = [7.0, -6.0, 2.25, -0.1]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], x * x))
        .collect();
    let skill = search
        .certify("arbitrary", "new/context", admission)
        .unwrap();
    assert_eq!(born, *skill.program());
    assert!(!skill.has_complete_search_statistics());
    let proof = temp.0.join("proof.axevo");
    skill.save_new(&proof).unwrap();
    let inspected = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .arg("evolved-inspect")
        .arg(&proof)
        .args(["--context", "new/context"])
        .output()
        .unwrap();
    assert!(inspected.status.success());
    let report = String::from_utf8(inspected.stdout).unwrap();
    assert!(report.contains("search_statistics=partial"));
    assert!(report.contains("mutations=unavailable"));
    assert_eq!(
        skill.statistics().admission_work,
        (born.instructions() * 11) as u64
    );
    assert_eq!(
        skill.statistics().work,
        search.progress().work + skill.statistics().admission_work
    );
    assert!(EvolvedSkill::from_bytes(&skill.to_bytes()).is_ok());
    assert!(born.evaluate(&[f64::NAN]).is_err());
    assert!(born.evaluate(&[1.0, 2.0]).is_err());
    let wrong = [7.0, -6.0, 2.25, -0.1]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], x * x + 1.0))
        .collect();
    assert!(search.certify("wrong", "wrong", wrong).is_err());
    assert!(
        search
            .certify("overlap", "overlap", examples(false))
            .is_err()
    );
}
#[test]
fn growth_resume_and_exact_genotype_dedup_are_durable() {
    let temp = Temp::new();
    let path = temp.0.join("search.axep");
    let budget = ConstructiveBudget {
        population: 4,
        patience: 1,
        max_genes: 6,
        ..ConstructiveBudget::default()
    };
    let before;
    {
        let mut search = ConstructiveSearch::open(
            &path,
            examples(true),
            &[],
            budget,
            NumericTolerance::default(),
            52,
        )
        .unwrap();
        assert_eq!(search.run(120, |_| {}).unwrap(), 120);
        before = search.progress();
        assert!(before.growths > 0);
        search.snapshot_new(&temp.0.join("snapshot.axep")).unwrap();
        assert!(search.snapshot_new(&temp.0.join("snapshot.axep")).is_err());
    }
    let mut resumed = ConstructiveSearch::open(
        &path,
        examples(true),
        &[],
        budget,
        NumericTolerance::default(),
        52,
    )
    .unwrap();
    assert_eq!(resumed.progress().completed, before.completed);
    assert_eq!(resumed.progress().work, before.work);
    assert_eq!(resumed.run(120, |_| {}).unwrap(), 120);
    let after = resumed.progress();
    assert_eq!(after.seen, 240);
    assert_eq!(after.completed, 240);
    assert!(
        after.duplicate_proposals > 0,
        "same RNG proposes cached graphs; they must not run twice"
    );
    assert!(resumed.certify("x", "x", examples(false)).is_err());
}
#[test]
fn labels_budgets_and_tolerance_cannot_reuse_stale_fitness() {
    let temp = Temp::new();
    let path = temp.0.join("search.axep");
    drop(
        ConstructiveSearch::open(
            &path,
            examples(true),
            &[],
            ConstructiveBudget::default(),
            NumericTolerance::default(),
            1,
        )
        .unwrap(),
    );
    assert!(
        ConstructiveSearch::open(
            &path,
            examples(false),
            &[],
            ConstructiveBudget::default(),
            NumericTolerance::default(),
            1
        )
        .is_err()
    );
    assert!(
        ConstructiveSearch::open(
            &path,
            examples(true),
            &[],
            ConstructiveBudget {
                max_genes: 32,
                ..ConstructiveBudget::default()
            },
            NumericTolerance::default(),
            1
        )
        .is_err()
    );
    assert!(
        ConstructiveSearch::open(
            &path,
            examples(true),
            &[],
            ConstructiveBudget::default(),
            NumericTolerance {
                absolute: 0.1,
                relative: 0.1
            },
            1
        )
        .is_err()
    );
}
#[test]
fn malformed_decimal_and_corrupted_genes_fail_closed() {
    for s in ["", "1", "1000 ", "10000", "1999", "-1000", "1é0"] {
        assert!(CircuitDna::from_number(s).is_err());
    }
    let temp = Temp::new();
    let mut search = ConstructiveSearch::open(
        &temp.0.join("s.axep"),
        examples(false),
        &[],
        ConstructiveBudget::default(),
        NumericTolerance::default(),
        91,
    )
    .unwrap();
    search.run(2000, |_| {}).unwrap();
    let bytes = search.winner().unwrap().to_bytes();
    for i in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[i] ^= 1;
        assert!(CircuitDna::from_bytes(&bad).is_err());
        assert!(CircuitDna::from_bytes(&bytes[..i]).is_err());
    }
}

#[test]
fn interrupted_reservation_is_not_reexecuted_or_promoted() {
    use axon_uic::{EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue};
    use std::collections::BTreeMap;
    let temp = Temp::new();
    let path = temp.0.join("search.axep");
    drop(
        ConstructiveSearch::open(
            &path,
            examples(true),
            &[],
            ConstructiveBudget::default(),
            NumericTolerance::default(),
            52,
        )
        .unwrap(),
    );
    let graph = [1u8, 2, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0];
    let dna = format!(
        "1{}",
        graph.iter().map(|b| format!("{b:03}")).collect::<String>()
    );
    {
        let (mut store, _) = EpisodeStore::open(&path).unwrap();
        store
            .record(
                EpisodeStamp::new("test", "crashed-before-result", 1),
                EpisodeData::Trace {
                    context: "axon/constructive-v1".into(),
                    channel: "reservation".into(),
                    fields: BTreeMap::from([("dna".into(), EpisodeValue::Text(dna))]),
                },
            )
            .unwrap();
    }
    let mut search = ConstructiveSearch::open(
        &path,
        examples(true),
        &[],
        ConstructiveBudget::default(),
        NumericTolerance::default(),
        52,
    )
    .unwrap();
    assert_eq!(search.progress().seen, 1);
    assert_eq!(search.progress().completed, 0);
    assert!(search.winner().is_none());
    assert_eq!(search.run(100, |_| {}).unwrap(), 100);
    assert_eq!(search.progress().seen, 101);
    assert_eq!(search.progress().completed, 100);
}

#[test]
fn valid_checksum_does_not_authorize_cycles_opcodes_or_neutral_aliases() {
    fn seal(bytes: &mut Vec<u8>) {
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
        bytes.extend(hash.to_le_bytes());
    }
    let mut original = b"AXCG0001".to_vec();
    original.extend([0, 1, 2, 4, 0, 0, 0, 0, 0, 1, 5, 0, 0, 6]);
    let mut valid = original.clone();
    seal(&mut valid);
    let program = CircuitDna::from_bytes(&valid).unwrap().program().unwrap();
    assert_eq!(program.evaluate(&[9.0]).unwrap(), 2.0);
    assert!(program.evaluate(&[-1.0]).is_err());
    for (position, value) in [(11, 255), (12, 5), (13, 1), (21, 255)] {
        let mut bytes = original.clone();
        bytes[position] = value;
        seal(&mut bytes);
        assert!(CircuitDna::from_bytes(&bytes).is_err());
    }
}

#[test]
fn tiny_work_limits_do_not_reserve_unfinishable_candidates() {
    let temp = Temp::new();
    let budget = ConstructiveBudget {
        max_work: 1,
        ..ConstructiveBudget::default()
    };
    let mut search = ConstructiveSearch::open(
        &temp.0.join("s.axep"),
        examples(true),
        &[],
        budget,
        NumericTolerance::default(),
        1,
    )
    .unwrap();
    assert_eq!(search.run(100, |_| {}).unwrap(), 0);
    assert_eq!(search.progress().seen, 0);
    assert_eq!(search.progress().work, 0);
    assert!(search.run(40001, |_| {}).is_err());
}

#[test]
fn inherited_calls_must_expand_into_a_bounded_deployable_program() {
    fn seal(bytes: &mut Vec<u8>) {
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
        bytes.extend(hash.to_le_bytes());
    }
    let mut bytes = b"AXCG0001".to_vec();
    bytes.extend([1, 1]); // one inherited unary program
    bytes.extend(255u32.to_le_bytes());
    bytes.extend([0, 0]); // x0
    for _ in 0..127 {
        bytes.extend([0, 0, 2]);
    } // add x0: 255 instructions
    bytes.extend([1, 2, 5, 0, 0, 0, 0, 2, 5, 5, 0, 0]);
    let mut one_call = bytes.clone();
    one_call.push(5);
    seal(&mut one_call);
    let program = CircuitDna::from_bytes(&one_call)
        .unwrap()
        .program()
        .unwrap();
    assert_eq!(program.instructions(), 255);
    assert_eq!(program.evaluate(&[2.0]).unwrap(), 256.0);
    bytes.push(6);
    seal(&mut bytes); // squaring the call expands past the VM limit
    assert!(CircuitDna::from_bytes(&bytes).is_err());
}

#[test]
fn reinforced_feedback_and_specialists_resume_without_replaying_candidates() {
    for policy in [ConstructivePolicy::Diverse, ConstructivePolicy::Reinforced] {
        let temp = Temp::new();
        let path = temp.0.join("feedback.axep");
        let budget = ConstructiveBudget {
            population: 16,
            max_genes: 8,
            ..ConstructiveBudget::default()
        };
        let before;
        {
            let mut search = ConstructiveSearch::open_with_policy(
                &path,
                examples(true),
                &[],
                budget,
                NumericTolerance::default(),
                137,
                policy,
            )
            .unwrap();
            assert_eq!(search.run(160, |_| {}).unwrap(), 160);
            before = search.feedback().unwrap();
            assert_eq!(before.trials.iter().sum::<u64>(), 160);
            assert!(before.trials.iter().all(|n| *n > 0));
            assert!(before.rewards.iter().all(|r| r.is_finite() && *r >= 0.0));
        }
        assert!(
            ConstructiveSearch::open(
                &path,
                examples(true),
                &[],
                budget,
                NumericTolerance::default(),
                137
            )
            .is_err()
        );
        let mut resumed = ConstructiveSearch::open_with_policy(
            &path,
            examples(true),
            &[],
            budget,
            NumericTolerance::default(),
            137,
            policy,
        )
        .unwrap();
        assert_eq!(resumed.feedback().unwrap(), before);
        assert_eq!(resumed.progress().completed, 160);
        assert_eq!(resumed.run(160, |_| {}).unwrap(), 160);
        assert_eq!(resumed.progress().seen, 320);
        assert_eq!(resumed.feedback().unwrap().trials.iter().sum::<u64>(), 320);
    }
}

#[test]
fn reinforced_circuit_transfers_to_numbers_never_used_for_selection() {
    let temp = Temp::new();
    let mut search = ConstructiveSearch::open_with_policy(
        &temp.0.join("square.axep"),
        examples(false),
        &[],
        ConstructiveBudget::default(),
        NumericTolerance::default(),
        91,
        ConstructivePolicy::Reinforced,
    )
    .unwrap();
    search.run(2000, |_| {}).unwrap();
    let born = CircuitDna::from_number(&search.winner().unwrap().number())
        .unwrap()
        .program()
        .unwrap();
    for i in -100..=100 {
        let x = f64::from(i) / 13.0;
        assert!(NumericTolerance::default().agrees(born.evaluate(&[x]).unwrap(), x * x));
    }
}

#[test]
fn inherited_operations_cannot_monopolize_fresh_proposals_or_starve_mutation() {
    let temp = Temp::new();
    let teach = |name: &str, subtract: bool| {
        let rows = [-3.0, -1.0, 0.0, 0.7, 1.0, 2.0, 4.0]
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let y = i as f64 / 3.0 + 0.2;
                NumericExample::new(vec![x, y], if subtract { x - y } else { x + y })
            })
            .collect();
        let mut search = ConstructiveSearch::open_with_policy(
            &temp.0.join(format!("{name}.axep")),
            rows,
            &[],
            ConstructiveBudget::default(),
            NumericTolerance::default(),
            91,
            ConstructivePolicy::Reinforced,
        )
        .unwrap();
        search.run(4000, |_| {}).unwrap();
        let admission = [7.0, -6.0, 2.25, -0.1]
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let y = i as f64 + 1.3;
                NumericExample::new(vec![x, y], if subtract { x - y } else { x + y })
            })
            .collect();
        search.certify(name, name, admission).unwrap()
    };
    let add = teach("add", false);
    let sub = teach("sub", true);
    let train = (0..16)
        .map(|i| {
            let x = f64::from(i) - 7.1;
            let y = f64::from(i * i) / 17.0 - 2.3;
            NumericExample::new(vec![x, y], x * y)
        })
        .collect();
    let mut search = ConstructiveSearch::open_with_policy(
        &temp.0.join("product.axep"),
        train,
        &[&add, &sub],
        ConstructiveBudget::default(),
        NumericTolerance::default(),
        75218341,
        ConstructivePolicy::Reinforced,
    )
    .unwrap();
    search.run(5000, |_| {}).unwrap();
    assert!(
        search.winner().is_some(),
        "new multiplication must remain reachable after inheriting addition and subtraction"
    );
}

#[test]
fn valid_hash_chain_cannot_invent_reward_or_rewind_composition_cursor() {
    use axon_uic::{EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue};
    let temp = Temp::new();
    let source = temp.0.join("source.axep");
    let budget = ConstructiveBudget {
        initial_genes: 4,
        ..ConstructiveBudget::default()
    };
    {
        let mut search = ConstructiveSearch::open_with_policy(
            &source,
            examples(true),
            &[],
            budget,
            NumericTolerance::default(),
            91,
            ConstructivePolicy::Reinforced,
        )
        .unwrap();
        search.run(1, |_| {}).unwrap();
    }
    let (store, _) = EpisodeStore::open(&source).unwrap();
    let data: Vec<_> = (1..=3)
        .map(|id| store.episode(id).unwrap().data().clone())
        .collect();
    for (name, value) in [("reward", "0.5"), ("cursor", "0"), ("action", "4")] {
        let target = temp.0.join(format!("invalid-{name}.axep"));
        {
            let (mut forged, _) = EpisodeStore::open(&target).unwrap();
            for (i, item) in data.iter().enumerate() {
                let mut item = item.clone();
                if i == 2 {
                    let EpisodeData::Trace { fields, .. } = &mut item else {
                        panic!("trace")
                    };
                    fields.insert(name.into(), EpisodeValue::Text(value.into()));
                }
                forged
                    .record(EpisodeStamp::new("test", format!("{i}"), i as u64), item)
                    .unwrap();
            }
        }
        assert!(
            ConstructiveSearch::open_with_policy(
                &target,
                examples(true),
                &[],
                budget,
                NumericTolerance::default(),
                91,
                ConstructivePolicy::Reinforced
            )
            .is_err()
        );
    }
    drop(store);
}
