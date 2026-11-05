use axon_uic::numeric::evolution::*;
use std::{fs, process::Command};
#[test]
fn complete_population_lab_reopens_discovered_real_quadratic_branches_including_singular_counterexamples()
 {
    let root = std::env::temp_dir().join(format!("axon-evo-lab-{}", std::process::id()));
    assert!(!root.exists());
    let output = Command::new(env!("CARGO_BIN_EXE_axon-uic-evolution-lab"))
        .arg(&root)
        .args(["82641973", "1000"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = fs::read_to_string(root.join("summary.txt")).unwrap();
    assert!(
        summary.contains("passed=7"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(summary.contains("correct=7000"));
    let plus = EvolvedSkill::open(&root.join("case-4.axevo"), "evolution/case-4").unwrap();
    let minus = EvolvedSkill::open(&root.join("case-5.axevo"), "evolution/case-5").unwrap();
    for (input, expected) in [
        ([1.0, -5.0, 6.0], [3.0, 2.0]),
        ([1.0, 1.0, 0.0], [0.0, -1.0]),
        ([1.0, -4.0, 4.0], [2.0, 2.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0]),
        ([-1.0, 5.0, -6.0], [2.0, 3.0]),
    ] {
        for (s, y) in [(&plus, expected[0]), (&minus, expected[1])] {
            let x = s.predict(s.context(), &input).unwrap();
            assert!((x - y).abs() < 1e-9, "{}", s.program().expression());
            assert!((input[0] * x * x + input[1] * x + input[2]).abs() < 1e-9);
        }
    }
    for input in [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]] {
        assert!(plus.predict(plus.context(), &input).is_err());
        assert!(minus.predict(minus.context(), &input).is_err());
    }
    assert!(!plus.dependencies().is_empty());
    assert!(!minus.dependencies().is_empty());
    let absolute = root.canonicalize().unwrap();
    assert!(absolute.starts_with(std::env::temp_dir().canonicalize().unwrap()));
    fs::remove_dir_all(absolute).unwrap();
}
#[test]
fn balanced_population_composes_acquired_tools_on_new_channels_and_exports_lineage() {
    let root = std::env::temp_dir().join(format!("axon-evo-transfer-lab-{}", std::process::id()));
    assert!(!root.exists());
    let output = Command::new(env!("CARGO_BIN_EXE_axon-uic-evolution-lab"))
        .arg(&root)
        .args(["82641973", "1000", "balanced", "transfer"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = fs::read_to_string(root.join("summary.txt")).unwrap();
    assert!(
        summary.contains("passed=8") && summary.contains("correct=8000"),
        "{summary}"
    );
    let skill = EvolvedSkill::open(&root.join("case-7.axevo"), "evolution/case-7").unwrap();
    assert!(!skill.dependencies().is_empty());
    for (input, expected) in [
        ([0., 0., 3., 4.], 5.),
        ([7., -2., 7., -2.], 0.),
        ([-1., -1., 2., 3.], 5.),
    ] {
        assert!((skill.predict(skill.context(), &input).unwrap() - expected).abs() < 1e-9);
    }
    let trace = fs::read_to_string(root.join("case-7.generations.csv")).unwrap();
    assert!(trace.contains("parent_a,parent_b") && trace.lines().count() > 1);
    let absolute = root.canonicalize().unwrap();
    assert!(absolute.starts_with(std::env::temp_dir().canonicalize().unwrap()));
    fs::remove_dir_all(absolute).unwrap();
}
