#![cfg(feature = "curriculum")]
use axon_uic::structural::{agent::*, *};
use serde_json::json;

fn seed() -> StructuralSeed {
    let protocol: serde_json::Value =
        serde_json::from_str(include_str!("../evals/curricula/agent-v1.json")).unwrap();
    let mut seed = StructuralSeed::default();
    for t in protocol["lessons"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["id"].as_str().unwrap().starts_with("goal-frame"))
    {
        let l = Lesson {
            id: t["id"].as_str().unwrap().into(),
            examples: serde_json::from_value(t["examples"].clone()).unwrap(),
            guards: vec![],
        };
        seed.teach(
            l,
            &serde_json::from_value::<Vec<Example>>(t["admission"].clone()).unwrap(),
        )
        .unwrap();
    }
    seed
}
struct World {
    epoch: u64,
    queries: usize,
    forbidden: bool,
    duplicate: bool,
}
impl Environment for World {
    fn describe(&self, name: &str) -> Option<Interface> {
        if !["caixa", "caixa-extra"].contains(&name) {
            return None;
        }
        let mut probes = vec![
            vec![0, 0],
            vec![1, 1],
            vec![2, 2],
            vec![3, 8],
            vec![5, 2],
            vec![-3, 4],
            vec![7, -2],
            vec![0, 9],
            vec![8, 0],
            vec![4, 11],
            vec![-5, -6],
            vec![9, 3],
        ];
        if self.duplicate {
            probes.push(probes[0].clone());
        }
        Some(Interface {
            arity: 2,
            epoch: self.epoch,
            probes,
        })
    }
    fn query(&mut self, _: &str, epoch: u64, x: &[i64]) -> Result<i64, Error> {
        assert!(!self.forbidden, "unexpected teacher call");
        assert_eq!(epoch, self.epoch);
        self.queries += 1;
        Ok(if epoch == 1 { x[0] + x[1] } else { x[0] * x[1] })
    }
}
fn world() -> World {
    World {
        epoch: 1,
        queries: 0,
        forbidden: false,
        duplicate: false,
    }
}
#[test]
fn request_drives_acquisition_counterexamples_composition_and_portable_inheritance() {
    let mut agent = Agent::new(
        seed(),
        Budget {
            active: false,
            ..Budget::default()
        },
    )
    .unwrap();
    let mut env = world();
    let mut events = Vec::new();
    assert_eq!(
        agent
            .answer("avalie caixa com 41 e 17", &mut env, &mut |e| {
                events.push(e.clone());
                Ok(())
            })
            .unwrap(),
        58
    );
    assert!(events.iter().any(|e| e.phase == "refutation"));
    assert!(events.iter().any(|e| e.phase == "consolidation"));
    assert!(env.queries >= 8);
    let bytes = agent.seed().to_bytes().unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        wire["rules"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r.get("examples").is_none())
    );
    let mut child = Agent::new(
        StructuralSeed::from_bytes(&bytes).unwrap(),
        Budget::default(),
    )
    .unwrap();
    env.forbidden = true;
    assert_eq!(
        child
            .answer(
                "passe (use caixa em 93 e 11) e (resultado de caixa para 19 e 23) por caixa",
                &mut env,
                &mut |_| Ok(())
            )
            .unwrap(),
        146
    );
    assert_eq!(child.statistics.queries, 0);
    assert_eq!(child.statistics.reuses, 3);
    assert_eq!(child.seed().to_bytes().unwrap(), bytes);
}
#[test]
fn revised_epoch_relearns_and_explicit_unannounced_counterexample_suspends() {
    let mut agent = Agent::new(seed(), Budget::default()).unwrap();
    let mut env = world();
    assert_eq!(
        agent
            .answer("use caixa em 13 e 7", &mut env, &mut |_| Ok(()))
            .unwrap(),
        20
    );
    env.epoch = 2;
    assert_eq!(
        agent
            .answer("use caixa em 13 e 7", &mut env, &mut |_| Ok(()))
            .unwrap(),
        91
    );
    assert!(agent.feedback("caixa", 2, &[13, 7], 92).unwrap());
    assert!(
        !agent
            .seed()
            .rules()
            .iter()
            .any(|r| r.id() == "acquired-caixa-2")
    );
    assert!(!agent.feedback("caixa", 2, &[13, 7], 91).unwrap());
}
#[test]
fn refuses_unlearned_or_unauthorized_goals_before_effects_and_validates_boundaries() {
    let mut agent = Agent::new(seed(), Budget::default()).unwrap();
    let mut env = world();
    for text in [
        "resolva qualquer coisa",
        "avalie segredo com 4 e 9",
        "avalie caixa com (avalie segredo com 8 e 2) e 5",
        "avalie caixa com (use caixa em 4 e 2 e 5",
        "avalie caixa com 3 e 5)",
    ] {
        assert!(
            agent.answer(text, &mut env, &mut |_| Ok(())).is_err(),
            "{text}"
        );
    }
    assert_eq!(env.queries, 0);
    env.duplicate = true;
    assert!(
        agent
            .answer("avalie caixa com 4 e 9", &mut env, &mut |_| Ok(()))
            .is_err()
    );
    assert_eq!(env.queries, 0);
    assert!(
        Agent::new(
            seed(),
            Budget {
                queries: 3,
                ..Budget::default()
            }
        )
        .is_err()
    );
    assert!(utterance(&"x".repeat(4097)).is_err());
    assert!(utterance("x\nq").is_err());
    assert!(utterance("((((((((((((((((((x))))))))))))))))))").is_err());
}
#[test]
fn budgets_and_event_failures_never_promote_unconfirmed_procedure() {
    let initial = seed();
    let mut agent = Agent::new(
        initial.clone(),
        Budget {
            search_work: 1,
            ..Budget::default()
        },
    )
    .unwrap();
    let mut env = world();
    assert!(
        agent
            .answer("use caixa em 19 e 5", &mut env, &mut |_| Ok(()))
            .is_err()
    );
    assert_eq!(
        agent.seed().to_bytes().unwrap(),
        initial.to_bytes().unwrap()
    );
    let mut agent = Agent::new(initial.clone(), Budget::default()).unwrap();
    assert!(
        agent
            .answer("use caixa em 19 e 5", &mut env, &mut |_| Err(
                Error::Storage("journal unavailable".into())
            ))
            .is_err()
    );
    assert_eq!(
        agent.seed().to_bytes().unwrap(),
        initial.to_bytes().unwrap()
    );
}
#[test]
fn names_with_shared_prefix_do_not_erase_each_other() {
    let mut agent = Agent::new(seed(), Budget::default()).unwrap();
    let mut env = world();
    agent
        .answer("use caixa-extra em 5 e 3", &mut env, &mut |_| Ok(()))
        .unwrap();
    agent
        .answer("use caixa em 7 e 11", &mut env, &mut |_| Ok(()))
        .unwrap();
    assert!(
        agent
            .seed()
            .rules()
            .iter()
            .any(|r| r.id() == "acquired-caixa-extra-1")
    );
    assert!(
        agent
            .seed()
            .rules()
            .iter()
            .any(|r| r.id() == "acquired-caixa-1")
    );
    let plan = seed()
        .transform(&utterance("use caixa em 7 e 11").unwrap(), 512)
        .unwrap();
    assert_eq!(
        serde_json::to_value(plan.output).unwrap(),
        json!(["invoke", "caixa", 7, 11])
    );
}
#[test]
fn integer_induction_remains_executable_when_inherited_library_contains_roots() {
    let protocol: serde_json::Value =
        serde_json::from_str(include_str!("../evals/curricula/agent-v1.json")).unwrap();
    let mut full = StructuralSeed::default();
    for t in protocol["lessons"].as_array().unwrap() {
        let l = Lesson {
            id: t["id"].as_str().unwrap().into(),
            examples: serde_json::from_value(t["examples"].clone()).unwrap(),
            guards: serde_json::from_value(t["guards"].clone()).unwrap_or_default(),
        };
        full.teach(
            l,
            &serde_json::from_value::<Vec<Example>>(t["admission"].clone()).unwrap(),
        )
        .unwrap();
    }
    struct Quadratic(World);
    impl Environment for Quadratic {
        fn describe(&self, name: &str) -> Option<Interface> {
            self.0.describe(name)
        }
        fn query(&mut self, _: &str, _: u64, x: &[i64]) -> Result<i64, Error> {
            Ok(x[0] * x[0] + x[1])
        }
    }
    for active in [false, true] {
        let mut agent = Agent::new(
            full.clone(),
            Budget {
                active,
                ..Budget::default()
            },
        )
        .unwrap();
        assert_eq!(
            agent
                .answer(
                    "avalie caixa com 33 e 18",
                    &mut Quadratic(world()),
                    &mut |_| Ok(())
                )
                .unwrap(),
            1107
        );
    }
}
#[test]
fn changed_environment_stops_acquisition_and_still_charges_attempted_query() {
    struct Changing(World);
    impl Environment for Changing {
        fn describe(&self, name: &str) -> Option<Interface> {
            self.0.describe(name)
        }
        fn query(&mut self, _: &str, _: u64, _: &[i64]) -> Result<i64, Error> {
            self.0.epoch += 1;
            Ok(0)
        }
    }
    let initial = seed();
    let mut agent = Agent::new(initial.clone(), Budget::default()).unwrap();
    assert!(
        agent
            .answer("use caixa em 17 e 23", &mut Changing(world()), &mut |_| Ok(
                ()
            ))
            .is_err()
    );
    assert_eq!(agent.statistics.queries, 1);
    assert_eq!(
        agent.seed().to_bytes().unwrap(),
        initial.to_bytes().unwrap()
    );
}
