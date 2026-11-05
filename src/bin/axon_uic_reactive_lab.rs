use axon_uic::{
    behavior_lab::Rng,
    reactive::{agent::*, *},
    telemetry::{LabTelemetry, Value as V},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    time::Instant,
};

struct Oracle {
    case: usize,
    state: i64,
    peak: i64,
}
impl Oracle {
    fn new(case: usize) -> Self {
        Self {
            case,
            state: 0,
            peak: 0,
        }
    }
    fn step(&mut self, x: i64) -> i64 {
        match self.case {
            0 => {
                if x < 2 {
                    x * x
                } else {
                    x + x
                }
            }
            1 => {
                let before = self.state;
                self.state = x;
                before
            }
            2 => {
                self.state += x;
                self.state
            }
            _ => {
                self.state = if x > 0 { self.state + 1 } else { 0 };
                self.peak = self.peak.max(self.state);
                match self.case {
                    3 => self.state,
                    4 => self.peak,
                    _ => self.peak - self.state,
                }
            }
        }
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=3).contains(&args.len()) {
        return Err("axon-uic-reactive-lab <new-output-directory> [seed] [heldout-events-per-task 100..100000]".into());
    }
    let root = PathBuf::from(&args[0]);
    let seed = args
        .get(1)
        .map(|s| s.parse::<u64>())
        .transpose()?
        .unwrap_or(73629185);
    let count = args
        .get(2)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(50000);
    if !(100..=100000).contains(&count) {
        return Err("evaluation quota".into());
    }
    if let Some(parent) = root.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&root)?;
    let mut results = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join("results.csv"))?;
    writeln!(
        results,
        "case,acquisition_events,searches,rejections,search_work,training_ns,root_dependencies,expanded_cells,active_cells,active_registers,capsule_bytes,skill_retained_bytes,cursor_bytes,stateless_conflict_groups,heldout_events,correct,errors,primitive_fit,primitive_correct,primitive_work,inference_ns,continuation_correct,continuation_errors"
    )?;
    let start = Instant::now();
    let mut telemetry = LabTelemetry::from_environment("reactive");
    telemetry.emit(
        "lab_start",
        &[
            ("tasks", V::Count(6)),
            ("seed", V::Count(seed)),
            (
                "telemetry",
                V::Text("acquisition-and-validation-checkpoints"),
            ),
        ],
    )?;
    let mut acquired = Vec::<ReactiveSkill>::new();
    let mut rng = Rng::new(seed);
    let mut descriptions = String::new();
    let mut total_errors = 0;
    for case in 0..6 {
        telemetry.emit("acquisition_start", &[("case", V::Count(case as u64))])?;
        let context = format!("reactive-lab/{case}");
        let mut tools: Vec<_> = if case >= 4 {
            acquired.iter().rev().take(4).rev().cloned().collect()
        } else {
            Vec::new()
        };
        if !tools.is_empty() {
            let rotation = rng.pick(tools.len());
            tools.rotate_left(rotation);
        }
        let mut agent = ReactiveAgent::new(&context, 1, tools, ReactiveBudget::default())?;
        let mut oracle = Oracle::new(case);
        let training = Instant::now();
        for id in 1..=64 {
            let x = if id <= 8 {
                [0, 1, 2, 3, -1, 0, 1, 4][id - 1]
            } else {
                agent
                    .suggest_probe_with_bounds(&[(-32, 32)])?
                    .map(|v| v[0])
                    .unwrap_or((rng.pick(9) as i64) - 4)
            };
            let prediction = agent.observe(id as u64, &[x])?;
            let actual = oracle.step(x);
            let feedback = agent.feedback(id as u64, actual)?;
            if telemetry.enabled() {
                let predicted = prediction.value.map(|v| v.to_string());
                let status = agent.statistics();
                telemetry.emit(
                    "reactive_observation",
                    &[
                        ("case", V::Count(case as u64)),
                        ("observation", V::Count(id as u64)),
                        ("input", V::Text(&x.to_string())),
                        ("predicted", predicted.as_deref().map_or(V::Null, V::Text)),
                        ("actual", V::Text(&actual.to_string())),
                        ("phase", V::Text(agent.phase().as_str())),
                        ("outcome", V::Text(feedback.as_str())),
                        ("work", V::Count(status.search_work)),
                        ("correct", V::Count(status.correct)),
                        ("scored", V::Count(status.scored)),
                    ],
                )?;
            }
            if agent.phase() == ReactivePhase::Active {
                break;
            }
        }
        let training_ns = training.elapsed().as_nanos();
        let stats = agent.statistics();
        let Some(skill) = agent.skill().cloned() else {
            writeln!(
                results,
                "{case},{},{},{},{},{training_ns},,,,,,,,,0,0,0,false,0,0,0",
                stats.observations, stats.searches, stats.rejected_candidates, stats.search_work
            )?;
            results.sync_all()?;
            return Err(format!("acquisition failed for case {case}: {:?}", stats).into());
        };
        let checkpoint = root.join(format!("case-{case}.agent.axra"));
        agent.checkpoint_new(&checkpoint)?;
        let mut restored = ReactiveAgent::resume(&context, &checkpoint)?;
        if restored.statistics() != stats {
            return Err("restart counters changed".into());
        }
        let path = root.join(format!("case-{case}.axreact"));
        skill.save_new(&path)?;
        let skill = ReactiveSkill::open(&path, &context)?;
        let circuit = skill.compile()?;
        telemetry.emit(
            "skill",
            &[
                ("case", V::Count(case as u64)),
                ("expression", V::Text(&skill.description())),
                ("dependencies", V::Count(skill.dependencies().len() as u64)),
                ("capsule_bytes", V::Count(skill.to_bytes().len() as u64)),
                ("registers", V::Count(circuit.memory_registers() as u64)),
            ],
        )?;
        let mut cursor = circuit.cursor();
        let mut truth = Oracle::new(case);
        let mut by_input = BTreeMap::<Vec<i64>, BTreeSet<i64>>::new();
        for t in skill.traces() {
            for e in &t.events {
                by_input
                    .entry(e.input.clone())
                    .or_default()
                    .insert(e.output);
            }
        }
        let conflicts = by_input.values().filter(|v| v.len() > 1).count();
        // The primitive-only comparator sees the same prefix, not the heldout oracle.
        let baseline = ReactiveSkill::learn(
            "reference",
            "reactive-lab/reference",
            skill.traces().to_vec(),
            &[],
            ReactiveBudget::default(),
        );
        let baseline_work = match &baseline {
            Ok(s) => s.search().work,
            Err(ReactiveError::Budget(s) | ReactiveError::NoProgram(s)) => s.work,
            _ => 0,
        };
        let baseline = baseline.ok();
        let reference = baseline.as_ref().map(ReactiveSkill::compile).transpose()?;
        let mut ref_cursor = reference.as_ref().map(ReactiveCircuit::cursor);
        let mut correct = 0;
        let mut primitive_correct = 0;
        let mut continuation_correct = 0;
        let mut inference_ns = 0;
        for i in 0..count {
            let x = match i % 997 {
                0..=332 => {
                    if i % 113 < 90 {
                        rng.pick(30) as i64 + 1
                    } else {
                        -(rng.pick(11) as i64)
                    }
                }
                333..=665 => rng.pick(201) as i64 - 100,
                _ => {
                    if i % 997 == 996 {
                        0
                    } else {
                        rng.pick(100) as i64 + 1
                    }
                }
            };
            let expected = truth.step(x);
            let timer = Instant::now();
            let actual = circuit.step(&mut cursor, &context, &[x]);
            inference_ns += timer.elapsed().as_nanos();
            if actual.is_ok_and(|v| v == expected) {
                correct += 1;
            }
            if let (Some(c), Some(s)) = (&reference, &mut ref_cursor)
                && c.step(s, "reactive-lab/reference", &[x])
                    .is_ok_and(|v| v == expected)
            {
                primitive_correct += 1;
            }
            // Continue the acknowledged live history, not just a reset batch cursor.
            let expected_live = oracle.step(x);
            let id = stats.observations + i as u64 + 1;
            if restored.phase() == ReactivePhase::Active {
                let p = restored.observe(id, &[x])?;
                if p.value == Some(expected_live) {
                    continuation_correct += 1;
                }
                restored.feedback(id, expected_live)?;
            }
            if i == count / 2 {
                let bytes = cursor.to_bytes();
                cursor = ReactiveCursor::from_bytes(&bytes, &circuit, &context)?;
                cursor.save_new(&root.join(format!("case-{case}.midpoint.axrc")))?;
                if restored.phase() == ReactivePhase::Active {
                    let path = root.join(format!("case-{case}.midpoint.axra"));
                    restored.checkpoint_new(&path)?;
                    restored = ReactiveAgent::resume(&context, &path)?;
                }
            }
            if i % 256 == 0 || i + 1 == count {
                telemetry.emit(
                    "verification",
                    &[
                        ("case", V::Count(case as u64)),
                        ("observations", V::Count(i as u64 + 1)),
                        ("correct", V::Count(correct as u64)),
                        ("heldout", V::Count(count as u64)),
                        (
                            "continuation_correct",
                            V::Count(continuation_correct as u64),
                        ),
                    ],
                )?;
            }
        }
        total_errors += (count - correct) + (count - continuation_correct);
        cursor.save_new(&root.join(format!("case-{case}.final.axrc")))?;
        let bytes = skill.to_bytes().len();
        writeln!(
            results,
            "{case},{},{},{},{},{training_ns},{},{},{},{},{bytes},{},{},{conflicts},{count},{correct},{},{},{primitive_correct},{baseline_work},{inference_ns},{continuation_correct},{}",
            stats.observations,
            stats.searches,
            stats.rejected_candidates,
            stats.search_work,
            skill.dependencies().len(),
            skill.cells(),
            circuit.cells(),
            circuit.memory_registers(),
            skill.retained_bytes(),
            cursor.retained_bytes(),
            count - correct,
            baseline.is_some(),
            count - continuation_correct
        )?;
        results.flush()?;
        descriptions.push_str(&format!("case_{case}: {}\nregisters={}, dependencies={:?}, acquisition_events={}, rejected={}, heldout={correct}/{count}, continuation={continuation_correct}/{count}\n",skill.description(),circuit.memory_registers(),skill.dependencies(),stats.observations,stats.rejected_candidates));
        acquired.push(skill);
    }
    results.sync_all()?;
    let summary = format!(
        "AXON REACTIVE DISCOVERY LAB\nseed={seed}\ntasks=6\nheldout_events={}\ncontinuation_events={}\nerrors={total_errors}\nwhole_lab_ns={}\n{descriptions}\nPredictions frozen before heldout. Reset/schema/integer grammar and novelty probe policy within host bounds [-32,32] are supplied. Holdout mixes long runs, frequent switches and magnitudes outside training bounds. All four previous tools are offered in rotated order for composition; unused dependencies are pruned. Source task names/formulas/hidden state are not supplied. Internal synthetic benchmark, not AGI, neurons, physical control or general self-modification.\n",
        count * 6,
        count * 6,
        start.elapsed().as_nanos()
    );
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join("summary.txt"))?;
    file.write_all(summary.as_bytes())?;
    file.sync_all()?;
    print!("{summary}");
    telemetry.emit(
        "lab_result",
        &[
            ("tasks", V::Count(6)),
            ("heldout", V::Count(count as u64 * 12)),
            ("correct", V::Count(count as u64 * 12 - total_errors as u64)),
            ("errors", V::Count(total_errors as u64)),
        ],
    )?;
    if total_errors != 0 {
        return Err("heldout refuted an acquired program; outputs preserved".into());
    }
    Ok(())
}
