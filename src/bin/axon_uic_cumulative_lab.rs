use axon_uic::{
    Authority, Effect, EpisodeStamp,
    behavior_lab::Rng,
    casework::*,
    cumulative::*,
    demand::{controller::CompositionBudget, genome::*, *},
    grounded_language::LanguageFrame,
    numeric::*,
    telemetry::{LabTelemetry, Value as V},
};
use std::{
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const POPULATION: usize = 8;
const RESERVE: u64 = 58291347;
const KNOWN: [usize; 4] = [0, 1, 2, 4];
const NOVEL: [usize; 4] = [8, 9, 10, 11];
fn auth() -> Authority {
    Authority::from([Effect::ReadData, Effect::WriteData])
}
fn file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
fn truth(task: usize, x: &[f64]) -> Option<f64> {
    let value = match task {
        0 => x[0] + x[1],
        1 => x[0] * x[1],
        2 => x[0] - x[1],
        3 => x[0] / x[1],
        4 => x[0].hypot(x[1]),
        5 => x[0] * x[1] - x[2],
        6 => x[0].hypot(x[1]).hypot(x[2]),
        7 => x[0].hypot(x[1]).hypot(x[2]).hypot(x[3]),
        8 => x[0].hypot(x[1]).hypot(x[2]) / x[3],
        9 => x[0].hypot(x[1]).hypot(x[2]) * x[3],
        10 => x[0].hypot(x[1]).hypot(x[2]) - x[3],
        11 => (x[0] - x[2]).hypot(x[1] - x[3]),
        _ => unreachable!(),
    };
    value.is_finite().then_some(value)
}
fn arity(task: usize) -> usize {
    if task < 5 {
        2
    } else if task < 7 {
        3
    } else {
        4
    }
}
fn point(rng: &mut Rng, n: usize, wide: bool) -> Vec<f64> {
    (0..n)
        .map(|_| {
            if wide {
                (rng.pick(200001) as f64 - 100000.) / 37.
            } else {
                (rng.pick(401) as f64 - 200.) / 17.
            }
        })
        .collect()
}
fn frame() -> Result<LanguageFrame> {
    Ok(LanguageFrame::learn(&[
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
    ])?)
}
fn runtime(root: &Path, study: &StudySeed, shortfall: &NumericSkill) -> Result<DemandRuntime> {
    fs::create_dir(root)?;
    study.save_new(&root.join("study.axstudy"), &auth())?;
    shortfall.save_new(&root.join("shortfall.axnum"))?;
    let mut catalog = Catalog::new(root);
    for (id, kind, name, input, output) in [
        (
            "study",
            ModuleKind::Study,
            "study.axstudy",
            "study",
            "study",
        ),
        (
            "shortfall",
            ModuleKind::Numeric,
            "shortfall.axnum",
            "casework/count,count,count",
            "casework/shortfall",
        ),
    ] {
        catalog.register(
            ModuleSpec {
                id,
                kind,
                file: name,
                input,
                output,
                cost: 1,
            },
            &auth(),
        )?;
    }
    Ok(DemandRuntime::new(
        catalog,
        Seed {
            active_bytes: 262144,
            active_modules: 2,
            ..Seed::default()
        },
    )?)
}
fn embodied(genome: &Genome, root: &Path) -> Result<(StudySeed, DemandRuntime)> {
    let mut birth = genome.birth(root, BirthLimits::default(), &auth())?;
    let entry = birth
        .controller
        .runtime()
        .catalog()
        .entries()
        .iter()
        .find(|e| e.kind == ModuleKind::Study)
        .ok_or("study binding")?
        .clone();
    let seed = birth.controller.runtime_mut().with_module(
        &entry.id,
        &entry.context,
        &auth(),
        |module| {
            let Module::Study(s) = module else {
                return Err(DemandError::Invalid("study"));
            };
            Ok((**s).clone())
        },
    )?;
    let runtime = DemandRuntime::new(
        Catalog::from_bytes(&birth.controller.runtime().catalog().to_bytes(), root)?,
        genome.residency(),
    )?;
    Ok((seed, runtime))
}
fn descendant(
    parent: &Genome,
    root: &Path,
    seed: &StudySeed,
    shortfall: &NumericSkill,
) -> Result<Genome> {
    let mut r = runtime(root, seed, shortfall)?;
    Ok(parent.inherit_runtime(&mut r, &[], &auth())?)
}
struct Lesson {
    skill: Option<NumericSkill>,
    questions: u64,
    search: Option<u64>,
    probe_work: u64,
    borrowed: bool,
    admitted: bool,
}
fn error_work(error: &StudyError) -> Option<u64> {
    match error {
        StudyError::Evidence { search_work } => Some(*search_work),
        StudyError::Ambiguous | StudyError::NeedsExamples | StudyError::Conflict => Some(0),
        StudyError::Numeric(NumericError::Budget(s) | NumericError::NoProgram(s)) => {
            Some(s.scalar_work)
        }
        StudyError::Numeric(NumericError::Development { scalar_work, .. }) => Some(*scalar_work),
        _ => None,
    }
}
fn lesson(
    seed: &StudySeed,
    root: &Path,
    task: usize,
    inherit: bool,
    rng_seed: u64,
    telemetry: &mut LabTelemetry<std::io::Stdout>,
) -> Result<Lesson> {
    fs::create_dir(root)?;
    let context = format!("task-{task}");
    let n = arity(task);
    let journal = root.join("observations.axep");
    let mut investigator = Investigator::open(seed, &journal, &context, n, inherit, &auth())?;
    let mut query_log =
        String::from("query\tinput\tactual\tsurvivors\tdisagreement\tprobe_work\tepisode\n");
    let mut rng = Rng::new(rng_seed.wrapping_add(task as u64 * 193));
    let points = if task < 5 {
        (-2..=2)
            .flat_map(|a| (-2..=2).map(move |b| vec![a as f64, b as f64]))
            .collect::<Vec<_>>()
    } else {
        (0..64).map(|_| point(&mut rng, n, false)).collect()
    };
    let minimum = if task < 5 { 1 } else { 8 };
    let mut probe_work = 0;
    let mut acquisition = None;
    // The teacher is outside the learner API. Only committed query answers reach acquisition.
    for q in 0..seed.policy().questions {
        if q >= minimum && investigator.survivors() == 1 {
            acquisition = Some(investigator.acquire());
            break;
        }
        if q >= 8 && task >= 5 {
            break;
        }
        let probe = if q == 0 && task < 5 {
            Probe {
                input: vec![0.; n],
                disagreement: 0,
                work: 0,
            }
        } else {
            let Some(p) = investigator.next_probe(&points)? else {
                break;
            };
            p
        };
        probe_work += probe.work;
        let actual = truth(task, &probe.input);
        let receipt = investigator.observe(
            EpisodeStamp::new("teacher", format!("q-{q}"), q as u64),
            probe.input.clone(),
            actual,
            &auth(),
        )?;
        query_log.push_str(&format!(
            "{q}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            probe
                .input
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join(","),
            actual.map_or_else(|| "domain".into(), |v| v.to_string()),
            investigator.survivors(),
            probe.disagreement,
            probe.work,
            receipt.id
        ));
        telemetry.emit(
            "study_query",
            &[
                ("method", V::Text("Investigator::next_probe")),
                ("case", V::Count(task as u64)),
                ("questions", V::Count(investigator.observations() as u64)),
                ("survivors", V::Count(investigator.survivors() as u64)),
                ("disagreement", V::Count(probe.disagreement as u64)),
                ("probe_work", V::Count(probe.work)),
                ("episode", V::Count(receipt.id)),
            ],
        )?;
        if q == 1 {
            let observations = investigator.observations();
            let survivors = investigator.survivors();
            drop(investigator);
            investigator = Investigator::open(seed, &journal, &context, n, inherit, &auth())?;
            if investigator.observations() != observations || investigator.survivors() != survivors
            {
                return Err("durable replay mismatch".into());
            }
        }
    }
    file(&root.join("queries.tsv"), query_log.as_bytes())?;
    let questions = investigator.observations() as u64;
    let acquired = acquisition.unwrap_or_else(|| investigator.acquire());
    let mut admission = String::from("input\tprediction\texpected\taccepted\n");
    let result = match acquired {
        Ok(a) => {
            let mut check_rng = Rng::new(
                rng_seed
                    .wrapping_add(293719)
                    .wrapping_add(task as u64 * 557),
            );
            let mut passed = true;
            for _ in 0..32 {
                let input = point(&mut check_rng, n, true);
                let predicted = a.skill.predict(a.skill.context(), &input).ok();
                let expected = truth(task, &input);
                let accepted = agrees(predicted, expected);
                passed &= accepted;
                admission.push_str(&format!(
                    "{}\t{}\t{}\t{accepted}\n",
                    input
                        .iter()
                        .map(f64::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                    scalar(predicted),
                    scalar(expected)
                ));
            }
            if passed {
                a.skill.save_new(&root.join("acquired.axnum"))?;
            }
            Lesson {
                skill: passed.then_some(a.skill),
                questions,
                search: Some(a.search_work),
                probe_work,
                borrowed: a.borrowed,
                admitted: passed,
            }
        }
        Err(e) => {
            file(&root.join("refusal.txt"), e.to_string().as_bytes())?;
            Lesson {
                skill: None,
                questions,
                search: error_work(&e),
                probe_work,
                borrowed: false,
                admitted: false,
            }
        }
    };
    file(&root.join("admission.tsv"), admission.as_bytes())?;
    let expression = result.skill.as_ref().map(|s| s.program().expression());
    let identity = result
        .skill
        .as_ref()
        .map(|s| format!("{:016X}", s.fingerprint()));
    telemetry.emit(
        "skill",
        &[
            ("method", V::Text("Investigator::acquire")),
            ("case", V::Count(task as u64)),
            ("questions", V::Count(questions)),
            ("search_work", result.search.map_or(V::Null, V::Count)),
            ("probe_work", V::Count(probe_work)),
            ("borrowed", V::Flag(result.borrowed)),
            ("admitted", V::Flag(result.admitted)),
            ("expression", expression.as_deref().map_or(V::Null, V::Text)),
            ("identity", identity.as_deref().map_or(V::Null, V::Text)),
            (
                "instructions",
                result
                    .skill
                    .as_ref()
                    .map_or(V::Null, |s| V::Count(s.program().instructions() as u64)),
            ),
            (
                "outcome",
                V::Text(if result.admitted {
                    "admitted"
                } else {
                    "refused"
                }),
            ),
        ],
    )?;
    Ok(result)
}
fn scalar(v: Option<f64>) -> String {
    v.map_or_else(|| "domain".into(), |v| v.to_string())
}
fn agrees(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => NumericTolerance::default().agrees(a, b),
        (None, None) => true,
        _ => false,
    }
}
struct Candidate {
    genome: Genome,
    seed: StudySeed,
    admitted: u64,
    questions: u64,
    work: u64,
    id: usize,
}
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=4).contains(&args.len()) {
        return Err("usage: axon-uic-cumulative-lab <new-directory> [rng] [rows 100..10000] [development|reserved]".into());
    }
    let root = PathBuf::from(&args[0]);
    let rng_seed = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(39172841u64);
    let count = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1000usize);
    let protocol = args.get(3).map(String::as_str).unwrap_or("development");
    if !(100..=10000).contains(&count) || !["development", "reserved"].contains(&protocol) {
        return Err("cumulative protocol limits".into());
    }
    fs::create_dir(&root)?;
    let started = Instant::now();
    let mut telemetry = LabTelemetry::from_environment("cumulative-v1");
    telemetry.emit(
        "lab_start",
        &[
            ("method", V::Text("StudySeed::new")),
            ("protocol", V::Text(protocol)),
            ("rng", V::Count(rng_seed)),
            ("rows", V::Count(count as u64)),
            ("population", V::Count(POPULATION as u64)),
        ],
    )?;
    let mut skills = Vec::new();
    let mut teaching = String::from("task\tinput\tactual\n");
    let mut bootstrap_work = 0;
    for task in 0..6 {
        let mut rng = Rng::new(rng_seed.wrapping_add(task as u64 * 919));
        let examples = (0..12)
            .map(|_| {
                let mut x = point(&mut rng, arity(task), false);
                if task == 3 && x[1] == 0. {
                    x[1] = 0.75;
                }
                let y = truth(task, &x).expect("bootstrap domain");
                teaching.push_str(&format!(
                    "{task}\t{}\t{y}\n",
                    x.iter().map(f64::to_string).collect::<Vec<_>>().join(",")
                ));
                NumericExample::new(x, y)
            })
            .collect();
        let skill = NumericSkill::learn(
            "opaque",
            format!("source-{task}"),
            examples,
            &[],
            NumericBudget::default(),
            NumericTolerance::default(),
        )?;
        bootstrap_work += skill.search().scalar_work;
        skills.push(skill);
    }
    file(&root.join("bootstrap.tsv"), teaching.as_bytes())?;
    let shortfall = skills[5].clone();
    let founder = StudySeed::new(
        "axon/cumulative-v1",
        skills,
        vec![frame()?],
        StudyPolicy::default(),
    )?;
    let mut source = runtime(&root.join("source"), &founder, &shortfall)?;
    let initial = Genome::capture(
        "axon/cumulative-v1",
        &mut source,
        CompositionBudget::default(),
        &[],
        axon_uic::demand::genome::MAX_BYTES,
        &auth(),
    )?;
    initial.save_number_new(&root.join("founder.axnum"), &auth())?;
    let mut parent = initial.clone();
    let mut parent_seed = founder.clone();
    let mut candidates_csv = String::from(
        "generation,candidate,admitted,questions,search_work,depth,skills,encoded_bytes,parent,identity,selected\n",
    );
    let mut total_attempts = 0;
    let mut selection_work = 0;
    for generation in 0..2 {
        let mut candidates = Vec::new();
        for index in 0..POPULATION {
            let id = generation * POPULATION + index;
            let directory = root.join(format!("candidate-{id}"));
            fs::create_dir(&directory)?;
            let mutated = parent_seed.offspring(rng_seed.wrapping_add(id as u64 * 137))?;
            let genotype = descendant(&parent, &directory.join("modules"), &mutated, &shortfall)?;
            genotype.save_number_new(&directory.join("before.axnum"), &auth())?;
            let (born, _) = embodied(
                &Genome::from_number(&genotype.number())?,
                &directory.join("birth"),
            )?;
            let targets = [0, 1, 2, 4, 6 + generation];
            let mut learned = Vec::new();
            let mut admitted = 0;
            let mut questions = 0;
            let mut work = 0;
            for task in targets {
                let run = lesson(
                    &born,
                    &directory.join(format!("task-{task}")),
                    task,
                    true,
                    rng_seed.wrapping_add(generation as u64 * 781),
                    &mut telemetry,
                )?;
                questions += run.questions;
                work += run
                    .search
                    .ok_or("missing search accounting; candidate not selectable")?;
                total_attempts += 1;
                if let Some(s) = run.skill {
                    learned.push(s);
                    admitted += 1;
                }
            }
            let child = born.consolidate(&learned)?;
            child.check_descendant(&born)?;
            let genome = descendant(&genotype, &directory.join("learned"), &child, &shortfall)?;
            genome.save_number_new(&directory.join("after.axnum"), &auth())?;
            selection_work += work;
            telemetry.emit(
                "seed_candidate",
                &[
                    ("method", V::Text("StudySeed::consolidate")),
                    ("generation", V::Count(generation as u64)),
                    ("candidate", V::Count(id as u64)),
                    ("admitted", V::Count(admitted)),
                    ("questions", V::Count(questions)),
                    ("work", V::Count(work)),
                    ("encoded_bytes", V::Count(genome.to_bytes().len() as u64)),
                    ("skills", V::Count(child.skills().len() as u64)),
                    (
                        "identity",
                        V::Text(&format!("{:016X}", genome.fingerprint())),
                    ),
                    ("status", V::Text("evaluated")),
                ],
            )?;
            candidates.push(Candidate {
                genome,
                seed: child,
                admitted,
                questions,
                work,
                id,
            });
        }
        candidates.sort_by_key(|c| {
            (
                std::cmp::Reverse(c.admitted),
                c.questions,
                c.work,
                c.seed.policy().search.max_cost,
                c.genome.to_bytes().len(),
                c.id,
            )
        });
        let best = candidates.first().ok_or("empty generation")?;
        for c in &candidates {
            candidates_csv.push_str(&format!(
                "{generation},{},{},{},{},{},{},{},{:016X},{:016X},{}\n",
                c.id,
                c.admitted,
                c.questions,
                c.work,
                c.seed.policy().search.max_cost,
                c.seed.skills().len(),
                c.genome.to_bytes().len(),
                c.genome.parent(),
                c.genome.fingerprint(),
                c.id == best.id
            ));
        }
        parent = best.genome.clone();
        parent_seed = best.seed.clone();
        telemetry.emit(
            "seed_selection",
            &[
                ("method", V::Text("StudySeed::offspring")),
                ("generation", V::Count(generation as u64)),
                ("winner", V::Count(best.id as u64)),
                ("selected", V::Counts(&[best.id as u64])),
                ("admitted", V::Count(best.admitted)),
                ("skills", V::Count(best.seed.skills().len() as u64)),
                ("questions", V::Count(best.questions)),
                ("search_work", V::Count(best.work)),
            ],
        )?;
    }
    file(&root.join("candidates.csv"), candidates_csv.as_bytes())?;
    parent.save_number_new(&root.join("winner.axnum"), &auth())?;
    let freeze = format!(
        "protocol=cumulative-v1\nvalidation={protocol}\nwinner={:016X}\nfounder={:016X}\nselected_on=development-only\nnovel_reserved_tasks=8,9,10,11\nknown_controls=0,1,2,4\nteaching_queries=8 for novel; <=16 for known\nadmission=32 independent points, no feedback\nsearch_depth={}\ntotal_work_per_task={}\nscoring_rng={}\nrows_per_task={count}\n",
        parent.fingerprint(),
        initial.fingerprint(),
        parent_seed.policy().search.max_cost,
        parent_seed.policy().total_work,
        if protocol == "reserved" {
            RESERVE
        } else {
            rng_seed.wrapping_add(811)
        }
    );
    file(&root.join("freeze.txt"), freeze.as_bytes())?;
    telemetry.emit(
        "stage",
        &[
            ("method", V::Text("Cumulative::freeze")),
            (
                "detail",
                V::Text(
                    "Winner frozen before matched controls; admission does not feed the learner",
                ),
            ),
            ("bootstrap_work", V::Count(bootstrap_work)),
            ("selection_work", V::Count(selection_work)),
            ("seed_digits", V::Count(parent.number().len() as u64)),
            ("skills", V::Count(parent_seed.skills().len() as u64)),
        ],
    )?;
    let scoring_seed = if protocol == "reserved" {
        RESERVE
    } else {
        rng_seed.wrapping_add(811)
    };
    let mut summary = String::from(
        "mode,task,novel,questions,probe_work,search_work,admitted,borrowed,correct,domain_refusals,unavailable,wrong,heldout\n",
    );
    let mut predictions = csv::WriterBuilder::new().delimiter(b'\t').from_writer(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(root.join("predictions.tsv"))?,
    );
    predictions.write_record([
        "mode",
        "task",
        "row",
        "input",
        "prediction",
        "expected",
        "status",
    ])?;
    let mut checked = 0;
    let mut correct = 0;
    let mut wrong = 0;
    let mut unavailable = 0;
    let mut domain_refusals = 0;
    for (arm_index, (mode, knowledge, inherit, question_policy)) in [
        ("cold", &parent_seed, false, QuestionPolicy::Random),
        ("founder-random", &founder, true, QuestionPolicy::Random),
        ("learned-random", &parent_seed, true, QuestionPolicy::Random),
        ("learned-active", &parent_seed, true, QuestionPolicy::Split),
    ]
    .into_iter()
    .enumerate()
    {
        let directory = root.join(mode);
        fs::create_dir(&directory)?;
        let mut policy = parent_seed.policy();
        policy.question_policy = question_policy;
        let seed = knowledge.with_policy(policy)?;
        let mut r = runtime(&directory.join("source"), &seed, &shortfall)?;
        let g = Genome::capture(
            "axon/cumulative-v1",
            &mut r,
            CompositionBudget::default(),
            &[],
            axon_uic::demand::genome::MAX_BYTES,
            &auth(),
        )?;
        g.save_number_new(&directory.join("start.axnum"), &auth())?;
        let (seed, _) = embodied(&Genome::from_number(&g.number())?, &directory.join("birth"))?;
        let mut promoted = Vec::new();
        for task in KNOWN.into_iter().chain(NOVEL) {
            let mut shard = csv::WriterBuilder::new().delimiter(b'\t').from_writer(
                fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(root.join(format!("predictions-{arm_index}-{task}.tsv")))?,
            );
            shard.write_record([
                "mode",
                "task",
                "row",
                "input",
                "prediction",
                "expected",
                "status",
            ])?;
            let result = lesson(
                &seed,
                &directory.join(format!("task-{task}")),
                task,
                inherit,
                rng_seed.wrapping_add(81721),
                &mut telemetry,
            )?;
            total_attempts += 1;
            let mut rng = Rng::new(scoring_seed.wrapping_add(task as u64 * 101));
            let mut good = 0;
            let mut bad = 0;
            let mut absent = 0;
            let mut refused = 0;
            for row in 0..count {
                let mut x = point(&mut rng, arity(task), true);
                if task == 8 && row % 17 == 0 {
                    x[3] = 0.;
                }
                // Complete prediction first. The independent oracle is not an argument to prediction.
                let prediction = result
                    .skill
                    .as_ref()
                    .and_then(|s| s.predict(s.context(), &x).ok());
                let expected = truth(task, &x);
                let status = if result.skill.is_none() {
                    absent += 1;
                    "unavailable"
                } else if agrees(prediction, expected) {
                    if expected.is_none() {
                        refused += 1;
                        "domain"
                    } else {
                        good += 1;
                        "correct"
                    }
                } else {
                    bad += 1;
                    "wrong"
                };
                predictions.write_record([
                    mode,
                    &task.to_string(),
                    &row.to_string(),
                    &x.iter().map(f64::to_string).collect::<Vec<_>>().join(","),
                    &scalar(prediction),
                    &scalar(expected),
                    status,
                ])?;
                shard.write_record([
                    mode,
                    &task.to_string(),
                    &row.to_string(),
                    &x.iter().map(f64::to_string).collect::<Vec<_>>().join(","),
                    &scalar(prediction),
                    &scalar(expected),
                    status,
                ])?;
            }
            shard.flush()?;
            shard.get_ref().sync_all()?;
            let search = result
                .search
                .map_or_else(|| "unavailable".into(), |v| v.to_string());
            summary.push_str(&format!(
                "{mode},{task},{},{},{},{search},{},{},{good},{refused},{absent},{bad},{count}\n",
                task >= 8,
                result.questions,
                result.probe_work,
                result.admitted,
                result.borrowed
            ));
            checked += count as u64;
            correct += good;
            wrong += bad;
            unavailable += absent;
            domain_refusals += refused;
            telemetry.emit(
                "validation",
                &[
                    ("method", V::Text("NumericSkill::predict")),
                    ("arm", V::Text(mode)),
                    ("case", V::Count(64 + arm_index as u64 * 16 + task as u64)),
                    ("task", V::Count(task as u64)),
                    ("novel", V::Flag(task >= 8)),
                    ("questions", V::Count(result.questions)),
                    ("probe_work", V::Count(result.probe_work)),
                    ("search_work", result.search.map_or(V::Null, V::Count)),
                    ("admitted", V::Flag(result.admitted)),
                    ("borrowed", V::Flag(result.borrowed)),
                    ("correct", V::Count(good)),
                    ("domain_refusals", V::Count(refused)),
                    ("unavailable", V::Count(absent)),
                    ("wrong", V::Count(bad)),
                    ("heldout", V::Count(count as u64)),
                ],
            )?;
            if let Some(s) = result.skill {
                promoted.push(s);
            }
        }
        let child = seed.consolidate(&promoted)?;
        let final_genome = descendant(&g, &directory.join("learned"), &child, &shortfall)?;
        final_genome.save_number_new(&directory.join("after.axnum"), &auth())?;
    }
    predictions.flush()?;
    predictions.get_ref().sync_all()?;
    file(&root.join("results.csv"), summary.as_bytes())?;
    let plans = file_demo(&parent, &root.join("file-demo"), &mut telemetry)?;
    file(
        &root.join("plans.csv"),
        &fs::read(root.join("file-demo/plans.csv"))?,
    )?;
    telemetry.emit(
        "lab_result",
        &[
            ("method", V::Text("Cumulative::validation")),
            ("checked", V::Count(checked)),
            ("correct", V::Count(correct)),
            ("domain_refusals", V::Count(domain_refusals)),
            ("unavailable", V::Count(unavailable)),
            ("wrong", V::Count(wrong)),
            ("attempts", V::Count(total_attempts)),
            ("candidates", V::Count((POPULATION * 2) as u64)),
            ("skills", V::Count(parent_seed.skills().len() as u64)),
            ("file_plans", V::Count(plans)),
            ("seed_digits", V::Count(parent.number().len() as u64)),
            ("elapsed_ns", V::Count(started.elapsed().as_nanos() as u64)),
        ],
    )?;
    println!(
        "cumulative-v1 checked={checked} correct={correct} domain_refusals={domain_refusals} unavailable={unavailable} wrong={wrong} attempts={total_attempts} skills={} file_plans={plans}",
        parent_seed.skills().len()
    );
    if wrong != 0 {
        return Err("reserved predictions contained errors; see results.csv".into());
    }
    Ok(())
}
fn file_demo(
    genome: &Genome,
    root: &Path,
    telemetry: &mut LabTelemetry<std::io::Stdout>,
) -> Result<u64> {
    fs::create_dir(root)?;
    let (_, mut runtime) = embodied(&Genome::from_number(&genome.number())?, &root.join("birth"))?;
    file(&root.join("offers.csv"),b"supplier,item,units,price_cents,available,unit\nA,alpha,8,2100,8,pecas\nB,alpha,12,2900,8,pecas\n")?;
    let request = "pedido alpha 20 kits cada kit usa 3 pecas limite 30000 centavos";
    file(&root.join("request.txt"), request.as_bytes())?;
    let mut records = String::from(
        "event,stock,needed,units,cost_cents,choices,stock_hash,offers_hash,model,episode\n",
    );
    let mut count = 0;
    for (stock, cost) in [(17, 10800), (52, 2100), (70, 0)] {
        let input = format!("item,quantity,unit\nalpha,{stock},pecas\n");
        fs::write(root.join("stock.csv"), &input)?;
        file(&root.join(format!("stock-{stock}.csv")), input.as_bytes())?;
        let stamp = EpisodeStamp::new("user", format!("stock-{stock}"), stock);
        let p = plan_files(&mut runtime, root, request, stamp.clone(), &auth())?;
        if p.cost_cents != cost {
            return Err("file plan disagrees with independently enumerated fixture".into());
        }
        let duplicate = plan_files(&mut runtime, root, request, stamp, &auth())?;
        if !duplicate.duplicate || duplicate.episode != p.episode {
            return Err("plan idempotency".into());
        }
        let choices = p
            .choices
            .iter()
            .map(|c| format!("{}:{}", c.supplier, c.packs))
            .collect::<Vec<_>>()
            .join(";");
        records.push_str(&format!(
            "{count},{stock},{},{},{},{choices},{:016X},{:016X},{:016X},{}\n",
            p.needed, p.units, p.cost_cents, p.source_stock, p.source_offers, p.model, p.episode
        ));
        count += 1;
        telemetry.emit(
            "plan",
            &[
                ("method", V::Text("Casework::plan_files")),
                ("item", V::Text(&p.item)),
                ("needed", V::Number(p.needed as f64)),
                ("units", V::Count(p.units)),
                ("cost_cents", V::Count(p.cost_cents)),
                ("budget_cents", V::Count(p.budget_cents)),
                ("choices", V::Text(&choices)),
                ("stock_hash", V::Text(&format!("{:016X}", p.source_stock))),
                ("offers_hash", V::Text(&format!("{:016X}", p.source_offers))),
                ("model", V::Text(&format!("{:016X}", p.model))),
                ("episode", V::Count(p.episode)),
                ("optimizer_nodes", V::Count(p.nodes)),
                ("optimizer_iterations", V::Count(p.iterations)),
                ("outcome", V::Text("verified-dry-run")),
            ],
        )?;
    }
    let incomplete = plan_files(
        &mut runtime,
        root,
        "pedido alpha 20 kits",
        EpisodeStamp::new("user", "incomplete", 80),
        &auth(),
    );
    if !matches!(incomplete, Err(CaseError::Clarify(_))) {
        return Err("missing context was not refused".into());
    }
    let stale = plan_files_observed(
        &mut runtime,
        root,
        request,
        EpisodeStamp::new("user", "stale", 81),
        &auth(),
        || {
            fs::write(
                root.join("stock.csv"),
                b"item,quantity,unit\nalpha,17,pecas\n",
            )
            .expect("fixture update");
        },
    );
    if !matches!(stale, Err(CaseError::Stale)) {
        return Err("live file update was not refused".into());
    }
    file(&root.join("plans.csv"), records.as_bytes())?;
    file(&root.join("refusals.txt"),b"incomplete sentence: clarification required\nfile changed before commit: stale, no plan recorded\n")?;
    Ok(count)
}
