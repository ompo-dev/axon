#![cfg(feature = "text-workflows")]
use axon_uic::{
    Authority, Effect, EpisodeStamp, casework::*, cumulative::*, demand::*,
    grounded_language::LanguageFrame, numeric::*,
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
        let p = std::env::temp_dir().join(format!(
            "axon-casework-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn auth() -> Authority {
    Authority::from([Effect::ReadData, Effect::WriteData])
}
fn runtime(t: &Temp) -> DemandRuntime {
    let frame = LanguageFrame::learn(&[
        (
            "pedido alpha 20 kits cada kit usa 3 pecas limite 30000 centavos".into(),
            "alpha".into(),
            "20,3,30000".into(),
        ),
        (
            "pedido beta 7 kits cada kit usa 5 pecas limite 15000 centavos".into(),
            "beta".into(),
            "7,5,15000".into(),
        ),
        (
            "pedido gamma 11 kits cada kit usa 2 pecas limite 40000 centavos".into(),
            "gamma".into(),
            "11,2,40000".into(),
        ),
    ])
    .unwrap();
    let inputs = [
        vec![20., 3., 17.],
        vec![7., 5., 10.],
        vec![11., 2., 9.],
        vec![0., 7., 3.],
        vec![2., 1., 12.],
        vec![5., 4., 0.],
        vec![1., 9., 3.],
    ];
    let skill = NumericSkill::learn(
        "opaque",
        "case/learned",
        inputs
            .into_iter()
            .map(|x| {
                let y = x[0] * x[1] - x[2];
                NumericExample::new(x, y)
            })
            .collect(),
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap();
    skill.save_new(&t.0.join("need.axnum")).unwrap();
    let a = auth();
    let seed = StudySeed::new(
        "study/case",
        vec![skill],
        vec![frame],
        StudyPolicy::default(),
    )
    .unwrap();
    seed.save_new(&t.0.join("study.axstudy"), &a).unwrap();
    let mut c = Catalog::new(&t.0);
    for (id, kind, file, input, output) in [
        (
            "learn",
            ModuleKind::Study,
            "study.axstudy",
            "study",
            "study",
        ),
        (
            "need",
            ModuleKind::Numeric,
            "need.axnum",
            "casework/count,count,count",
            "casework/shortfall",
        ),
    ] {
        c.register(
            ModuleSpec {
                id,
                kind,
                file,
                input,
                output,
                cost: 1,
            },
            &a,
        )
        .unwrap();
    }
    DemandRuntime::new(
        c,
        Seed {
            active_bytes: 262144,
            active_modules: 2,
            ..Seed::default()
        },
    )
    .unwrap()
}
fn files(t: &Temp, stock: u64) {
    fs::write(
        t.0.join("stock.csv"),
        format!("item,quantity,unit\nalpha,{stock},pecas\n"),
    )
    .unwrap();
    fs::write(t.0.join("offers.csv"),"supplier,item,units,price_cents,available,unit\nA,alpha,8,2100,8,pecas\nB,alpha,12,2900,8,pecas\n").unwrap();
}
fn request() -> &'static str {
    "pedido alpha 20 kits cada kit usa 3 pecas limite 30000 centavos"
}
#[test]
fn learned_roles_and_rule_produce_an_exact_checked_optimal_plan_and_adapt_to_file_changes() {
    let t = Temp::new();
    let mut r = runtime(&t);
    files(&t, 17);
    let stamp = EpisodeStamp::new("user", "first", 1);
    let p = plan_files(&mut r, &t.0, request(), stamp.clone(), &auth()).unwrap();
    assert_eq!(p.needed, 43);
    assert!(p.units >= 43);
    assert_eq!(p.cost_cents, 10800);
    let again = plan_files(&mut r, &t.0, request(), stamp, &auth()).unwrap();
    assert!(again.duplicate);
    assert_eq!(again.episode, p.episode);
    files(&t, 52);
    let new = plan_files(
        &mut r,
        &t.0,
        request(),
        EpisodeStamp::new("user", "second", 2),
        &auth(),
    )
    .unwrap();
    assert_eq!(new.needed, 8);
    assert_eq!(new.cost_cents, 2100);
    assert_ne!(new.source_stock, p.source_stock);
    files(&t, 70);
    let enough = plan_files(
        &mut r,
        &t.0,
        request(),
        EpisodeStamp::new("user", "third", 3),
        &auth(),
    )
    .unwrap();
    assert_eq!(enough.cost_cents, 0);
    assert!(enough.choices.is_empty());
}
#[test]
fn missing_context_conflicting_units_and_infeasible_budget_are_distinct_refusals() {
    let t = Temp::new();
    let mut r = runtime(&t);
    files(&t, 17);
    assert!(matches!(
        plan_files(
            &mut r,
            &t.0,
            "pedido alpha 20 kits",
            EpisodeStamp::new("u", "a", 0),
            &auth()
        ),
        Err(CaseError::Clarify(_))
    ));
    assert!(matches!(
        plan_files(
            &mut r,
            &t.0,
            "pedido alpha 20 kits cada kit usa 3 pecas limite 100 centavos",
            EpisodeStamp::new("u", "b", 0),
            &auth()
        ),
        Err(CaseError::Infeasible)
    ));
    fs::write(t.0.join("stock.csv"), "item,quantity,unit\nalpha,17,kg\n").unwrap();
    assert!(matches!(
        plan_files(
            &mut r,
            &t.0,
            request(),
            EpisodeStamp::new("u", "c", 0),
            &auth()
        ),
        Err(CaseError::Clarify(_))
    ));
    assert!(!t.0.join("individual.axep").exists());
}
#[test]
fn changed_snapshot_and_missing_authority_never_commit_a_plan() {
    let t = Temp::new();
    let mut r = runtime(&t);
    files(&t, 17);
    assert!(matches!(
        plan_files(
            &mut r,
            &t.0,
            request(),
            EpisodeStamp::new("u", "denied", 0),
            &Authority::from([Effect::ReadData])
        ),
        Err(CaseError::Demand(DemandError::Unauthorized(
            Effect::WriteData
        )))
    ));
    let result = plan_files_observed(
        &mut r,
        &t.0,
        request(),
        EpisodeStamp::new("u", "stale", 0),
        &auth(),
        || {
            files(&t, 52);
        },
    );
    assert!(matches!(result, Err(CaseError::Stale)));
    assert!(!t.0.join("individual.axep").exists());
}
#[test]
fn generic_integer_optimizer_matches_independent_exhaustive_oracles_on_varied_offers() {
    let t = Temp::new();
    let mut r = runtime(&t);
    let mut rng = axon_uic::behavior_lab::Rng::new(19284371);
    for event in 0..100 {
        let stock = rng.pick(65) as u64;
        fs::write(
            t.0.join("stock.csv"),
            format!("item,quantity,unit\nalpha,{stock},pecas\n"),
        )
        .unwrap();
        let mut text = "supplier,item,units,price_cents,available,unit\n".to_string();
        let mut offers = Vec::new();
        for i in 0..3 {
            let units = 1 + rng.pick(17) as u64;
            let cost = 1 + rng.pick(4000) as u64;
            let available = rng.pick(6) as u64;
            offers.push((units, cost, available));
            text.push_str(&format!("S{i},alpha,{units},{cost},{available},pecas\n"));
        }
        fs::write(t.0.join("offers.csv"), text).unwrap();
        let needed = 60i64 - stock as i64;
        let mut best = None;
        for a in 0..=offers[0].2 {
            for b in 0..=offers[1].2 {
                for c in 0..=offers[2].2 {
                    let q = [a, b, c];
                    let units = q.iter().zip(&offers).map(|(n, o)| n * o.0).sum::<u64>();
                    let cost = q.iter().zip(&offers).map(|(n, o)| n * o.1).sum::<u64>();
                    if units as i64 >= needed && cost <= 30000 {
                        best = Some(best.map_or(cost, |old: u64| old.min(cost)));
                    }
                }
            }
        }
        match (
            plan_files(
                &mut r,
                &t.0,
                request(),
                EpisodeStamp::new("oracle", format!("case-{event}"), event),
                &auth(),
            ),
            best,
        ) {
            (Ok(plan), Some(cost)) => assert_eq!(plan.cost_cents, cost),
            (Err(CaseError::Infeasible), None) => {}
            (result, expected) => panic!("result={result:?}; expected={expected:?}"),
        }
    }
}

#[test]
fn cli_uses_only_the_whole_genome_and_current_files_and_refuses_incomplete_requests() {
    use axon_uic::demand::{
        controller::CompositionBudget,
        genome::{Genome, MAX_BYTES},
    };
    let t = Temp::new();
    let mut r = runtime(&t);
    files(&t, 17);
    let g = Genome::capture(
        "case/whole",
        &mut r,
        CompositionBudget::default(),
        &[],
        MAX_BYTES,
        &auth(),
    )
    .unwrap();
    let number = t.0.join("whole.axnum");
    g.save_number_new(&number, &auth()).unwrap();
    fs::rename(t.0.join("need.axnum"), t.0.join("source-removed")).unwrap();
    fs::rename(t.0.join("study.axstudy"), t.0.join("study-removed")).unwrap();
    let args = vec![
        "case-plan".into(),
        number.to_string_lossy().into_owned(),
        t.0.join("cli-birth").to_string_lossy().into_owned(),
        t.0.to_string_lossy().into_owned(),
        request().into(),
        "real-order-1".into(),
    ];
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .args(&args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("cost_cents=10800"));
    assert!(text.contains("status=verified-dry-run"));
    assert!(t.0.join("individual.axep").exists());
    let mut invalid = args;
    invalid[2] = t.0.join("incomplete-birth").to_string_lossy().into_owned();
    invalid[4] = "pedido alpha 20 kits".into();
    invalid[5] = "incomplete".into();
    let denied = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .args(invalid)
        .output()
        .unwrap();
    assert!(!denied.status.success());
    assert!(String::from_utf8_lossy(&denied.stderr).contains("clarification"));
}
