use axon_uic::{
    Authority, Effect, TextBudget, TextExample, TextSkill,
    behavior::{BehaviorBudget, BehaviorError, Resettable},
    behavior_lab::{CachedOracle, Rng, discover},
    demand::{controller::*, *},
    numeric::{NumericExample, NumericTolerance, evolution::*, seed::DevelopmentSeed},
    reactive::{ReactiveBudget, ReactiveEvent, ReactiveSkill, ReactiveTrace},
};
use std::{error::Error, fs, path::Path};

pub const GOALS: [&str; 2] = ["controller/goal-0", "controller/goal-1"];
pub const DEVICE: &str = "controller/device";
pub const DEV_SEED: u64 = 72918463;
pub const RESERVES: [u64; 3] = [61827943, 93074117, 42709631];

#[derive(Clone)]
pub struct Device {
    permutation: u8,
    changed: bool,
    state: u16,
    effects: u64,
}
impl Device {
    pub fn new(seed: u64) -> Self {
        Self {
            permutation: (seed % 3) as u8,
            changed: false,
            state: 0,
            effects: 0,
        }
    }
    pub fn change(&mut self) {
        self.changed = !self.changed;
    }
    pub fn effects(&self) -> u64 {
        self.effects
    }
    pub fn state(&self) -> u16 {
        self.state
    }
}
impl Resettable for Device {
    fn reset(&mut self) -> Result<(), BehaviorError> {
        self.state = 0;
        Ok(())
    }
    fn step(&mut self, action: u8) -> Result<u16, BehaviorError> {
        if action >= 3 {
            return Err(BehaviorError::Invalid("device action"));
        }
        self.effects += 1;
        match (action + self.permutation + u8::from(self.changed)) % 3 {
            0 => self.state = 0,
            1 => self.state = 1,
            _ => self.state ^= 1,
        }
        Ok(self.state)
    }
}
pub struct Bootstrap {
    pub numeric_work: u64,
    pub reactive_work: u64,
    pub device_queries: usize,
    pub device_steps: usize,
}
pub fn learn_device(
    root: &Path,
    version: usize,
    device: &Device,
    authority: &Authority,
) -> Result<(usize, usize), Box<dyn Error>> {
    if !authority.allows(Effect::Simulate) {
        return Err(DemandError::Unauthorized(Effect::Simulate).into());
    }
    if !authority.allows(Effect::WriteData) {
        return Err(DemandError::Unauthorized(Effect::WriteData).into());
    }
    let budget = BehaviorBudget {
        max_word: 8,
        ..BehaviorBudget::default()
    };
    let mut oracle = CachedOracle::new(device.clone(), budget);
    let (model, _, _) = discover(3, DEVICE, budget, &mut oracle)?;
    model.save_new(root.join(format!("device-{version}.axbehavior")))?;
    Ok((oracle.raw.queries, oracle.raw.steps))
}
pub fn catalog(root: &Path, version: usize, authority: &Authority) -> Result<Catalog, DemandError> {
    let mut c = Catalog::new(root);
    for (id, kind, file) in [
        ("component-a", ModuleKind::Text, "a.axtext".into()),
        ("component-b", ModuleKind::Text, "b.axtext".into()),
        ("component-c", ModuleKind::Evolved, "c.axevo".into()),
        ("component-d", ModuleKind::Evolved, "d.axevo".into()),
        ("component-e", ModuleKind::Reactive, "e.axreact".into()),
        ("component-f", ModuleKind::Reactive, "f.axreact".into()),
        (
            "device",
            ModuleKind::Behavior,
            format!("device-{version}.axbehavior"),
        ),
    ] {
        c.register(
            ModuleSpec {
                id,
                kind,
                file: &file,
                input: "declared-input",
                output: "declared-output",
                cost: 1,
            },
            authority,
        )?;
    }
    Ok(c)
}
pub fn runtime(c: Catalog, modules: usize) -> Result<DemandRuntime, DemandError> {
    DemandRuntime::new(
        c,
        Seed {
            active_modules: modules,
            active_bytes: 262144,
            ..Seed::default()
        },
    )
}
pub fn bootstrap(
    root: &Path,
    seed: u64,
    authority: &Authority,
) -> Result<Bootstrap, Box<dyn Error>> {
    if !authority.allows(Effect::WriteData) {
        return Err(DemandError::Unauthorized(Effect::WriteData).into());
    }
    if !authority.allows(Effect::Simulate) {
        return Err(DemandError::Unauthorized(Effect::Simulate).into());
    }
    fs::create_dir(root)?;
    for (index, path) in ["a.axtext", "b.axtext"].iter().enumerate() {
        let inputs = [
            "alpha|2,3",
            "source77|13,8",
            "x|19,11",
            "reading|7,21",
            "node21|0,5",
            "stream|4,9",
        ];
        let examples = inputs
            .iter()
            .map(|input| TextExample::new(*input, input.split('|').nth(index).expect("two fields")))
            .collect();
        TextSkill::learn(
            "opaque",
            format!("controller/text-{index}"),
            examples,
            TextBudget::default(),
        )?
        .save_new(root.join(path))?;
    }
    let mut numeric_work = 0;
    let mut dna = DevelopmentSeed::new(
        "controller/numerical-v1",
        EvolutionBudget::balanced(),
        65536,
    )?;
    for (case, file) in ["c.axevo", "d.axevo"].iter().enumerate() {
        let samples = |offset: u64, count: usize| {
            let mut rng = Rng::new(seed.wrapping_add(offset));
            (0..count)
                .map(|i| {
                    let x = if i == 0 {
                        0.0
                    } else {
                        rng.pick(23) as f64 - 11.0
                    };
                    let y = if i == 1 {
                        0.0
                    } else {
                        rng.pick(23) as f64 - 11.0
                    };
                    NumericExample::new(vec![x, y], if case == 0 { x + y } else { x * y })
                })
                .collect::<Vec<_>>()
        };
        let training = samples(case as u64 * 101, 16);
        let admission = samples(837191 + case as u64 * 971, 64)
            .into_iter()
            .filter(|a| !training.iter().any(|t| t.input == a.input))
            .take(8)
            .collect();
        let outcome = EvolvedSkill::evolve(
            "opaque",
            &format!("controller/metric-{case}"),
            training,
            admission,
            &[],
            seed.wrapping_add(case as u64 * 103),
            EvolutionBudget::balanced(),
            NumericTolerance::default(),
        )?;
        numeric_work += outcome.statistics.work;
        let s = outcome
            .skill
            .ok_or("numerical bootstrap failed; keep output directory")?;
        s.save_new(&root.join(file))?;
        dna = dna.extend("controller/numerical-v1", s)?;
    }
    dna.save_new(&root.join("numerical.axdna"))?;
    let inputs = [-4, 0, 1, 5, 19, 20, 21, 9, 100, 3, 18, 22, 7, -1, 2, 31];
    let classifier = ReactiveSkill::learn(
        "opaque",
        "controller/reactive-0",
        vec![ReactiveTrace::new(
            inputs
                .into_iter()
                .map(|x| ReactiveEvent::new(vec![x], i64::from(x < 20)))
                .collect(),
        )],
        &[],
        ReactiveBudget::default(),
    )?;
    let mut previous = 0;
    let delay = ReactiveSkill::learn(
        "opaque",
        "controller/reactive-1",
        vec![ReactiveTrace::new(
            [2, -1, 0, 3, 1, 9, -7, 0, 2, 5, -4, 1, 3, 0, 8, -2]
                .into_iter()
                .map(|x| {
                    let out = previous;
                    previous = x;
                    ReactiveEvent::new(vec![x], out)
                })
                .collect(),
        )],
        &[],
        ReactiveBudget::default(),
    )?;
    let reactive_work = classifier.search().work + delay.search().work;
    classifier.save_new(&root.join("e.axreact"))?;
    delay.save_new(&root.join("f.axreact"))?;
    let (device_queries, device_steps) = learn_device(root, 0, &Device::new(seed), authority)?;
    catalog(root, 0, authority)?.save_new(root.join("modules.axseed"), authority)?;
    Ok(Bootstrap {
        numeric_work,
        reactive_work,
        device_queries,
        device_steps,
    })
}
pub fn wire(prefix: u64, values: [i64; 2]) -> String {
    format!("source-{prefix}|{},{}", values[0], values[1])
}
pub fn expectation(goal: usize, values: [i64; 2], previous: &mut i64) -> i64 {
    if goal == 0 {
        i64::from(values[0] * values[1] < 20)
    } else {
        let out = *previous;
        *previous = i64::from(values[0] + values[1] < 20);
        out
    }
}
pub fn traces(goal: usize, seed: u64, count: usize) -> Vec<GoalTrace> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|trace| {
            let mut previous = 0;
            (0..16)
                .map(|i| {
                    let values = match i {
                        0 => [0, 0],
                        1 => [1, 1],
                        2 => [4, 5],
                        3 => [9, 0],
                        4 => [20, 1],
                        _ => [rng.pick(31) as i64 - 5, rng.pick(31) as i64 - 5],
                    };
                    GoalExample {
                        input: wire(seed.wrapping_add((trace * 191 + i) as u64), values),
                        output: expectation(goal, values, &mut previous),
                    }
                })
                .collect()
        })
        .collect()
}
