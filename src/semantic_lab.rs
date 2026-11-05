//! P0: evolve assemblies of acquired relational rules; do not call this open-ended evolution.
use crate::{
    behavior_lab::Rng,
    demand::genome::{Genome, bounded_path},
    live_store::checksum,
    numeric::{NumericBudget, NumericTolerance},
    semantic::{Rule, SemanticError, SemanticSeed, Session},
    semantic_cli::{born, load_lessons, pack},
    telemetry::{LabTelemetry, Value as V},
};
use std::{
    collections::BTreeSet,
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
const POPULATION: usize = 64;
const GENERATIONS: usize = 4;
const CAMPAIGNS: usize = 3;
#[derive(Clone)]
pub(crate) struct Case {
    id: String,
    family: String,
    text: String,
    expected: Option<f64>,
    refusal: String,
}
#[derive(Default, Clone, Copy)]
pub(crate) struct Score {
    pub(crate) checked: u64,
    pub(crate) correct: u64,
    pub(crate) wrong: u64,
    pub(crate) refused: u64,
}
fn refusal(e: &SemanticError) -> &'static str {
    match e {
        SemanticError::Unknown => "unknown",
        SemanticError::Ambiguous => "ambiguous",
        SemanticError::Missing => "missing",
        SemanticError::Domain => "domain",
        _ => "error",
    }
}
fn evaluate(seed: &SemanticSeed, c: &Case) -> (bool, Option<f64>, String) {
    match Session::default().narrative(seed, &c.text) {
        Ok(a) => (
            c.expected
                .is_some_and(|n| NumericTolerance::default().agrees(a.value, n)),
            Some(a.value),
            "answered".into(),
        ),
        Err(e) => {
            let status = refusal(&e);
            (
                c.expected.is_none() && c.refusal == status,
                None,
                status.into(),
            )
        }
    }
}
pub(crate) fn score(seed: &SemanticSeed, cases: &[Case]) -> Score {
    let mut s = Score::default();
    for c in cases {
        let (ok, p, _) = evaluate(seed, c);
        s.checked += 1;
        s.correct += u64::from(ok);
        s.wrong += u64::from(!ok && p.is_some());
        s.refused += u64::from(p.is_none());
    }
    s
}
pub(crate) fn generated(rng: u64, count: usize) -> Vec<Case> {
    let mut r = Rng::new(rng);
    let mut cases = Vec::new();
    for i in 0..count {
        let a = format!("pessoa{i}a");
        let b = format!("pessoa{i}b");
        let u = format!("objeto{i}");
        let n = (100 + r.pick(900)) as f64;
        let m = (10 + r.pick(90)) as f64;
        let q = (1 + r.pick(40)) as f64;
        let k = (1 + r.pick(17)) as f64;
        let (text, expected) = match i % 6 {
            0 => (
                format!("{a} tinha {n} {u}. {a} nao recebeu {q} {u}. Quantos {u} ficaram com {a}?"),
                n,
            ),
            1 => (
                format!("{a} possui {n} {u}. Ela recebeu {q} {u}. Quantos {u} ela tem?"),
                n + q,
            ),
            2 => (
                format!(
                    "{a} tinha {n} {u}. {a} gastou {q} {u}. {a} recebeu {k} {u}. Quantos {u} ficaram com {a}?"
                ),
                n - q + k,
            ),
            3 => (
                format!(
                    "{a} tinha {n} {u}. {b} tinha {m} {u}. {a} emprestou {q} {u} para {b}. {b} devolveu {k} {u} para {a}. Quantos {u} {a} tem a mais que {b}?"
                ),
                n - q + k - (m + q - k),
            ),
            4 => (
                format!(
                    "{a} possui {n} {u}. {b} tinha {m} {u}. {a} emprestou {q} {u} para {b}. {b} gastou {k} {u}. Quantos {u} {a} e {b} tem juntos?"
                ),
                n + m - k,
            ),
            _ => (
                format!(
                    "{a} tinha {n} {u}. {b} possui {m} {u}. {b} devolveu {k} {u} para {a}. Quantos {u} ficaram com {a}?"
                ),
                n + k,
            ),
        };
        cases.push(Case {
            id: format!("generated-{i}"),
            family: if i % 6 < 2 { "text" } else { "mixed" }.into(),
            text,
            expected: Some(expected),
            refusal: String::new(),
        });
        let (text, expected) = match i % 3 {
            0 => (
                format!("Quanto custa comprar {q} unidades a {m} por unidade?"),
                q * m,
            ),
            1 => (
                format!("Qual x satisfaz {q} vezes x mais {m} igual a {n}?"),
                (n - m) / q,
            ),
            _ => (format!("Qual a norma de {n} e {m}?"), n.hypot(m)),
        };
        cases.push(Case {
            id: format!("math-{i}"),
            family: "math".into(),
            text,
            expected: Some(expected),
            refusal: String::new(),
        });
    }
    cases
}
pub(crate) fn independent(path: &Path) -> Result<Vec<Case>, Box<dyn Error>> {
    let mut r = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    if r.headers()?
        .iter()
        .ne(["id", "family", "text", "expected", "refusal"])
    {
        return Err("independent schema".into());
    }
    let mut out = Vec::new();
    for row in r.records() {
        let row = row?;
        out.push(Case {
            id: row[0].into(),
            family: row[1].into(),
            text: row[2].into(),
            expected: if row[3].is_empty() {
                None
            } else {
                Some(row[3].parse()?)
            },
            refusal: row[4].into(),
        });
    }
    Ok(out)
}
fn file(path: &Path) -> Result<fs::File, std::io::Error> {
    OpenOptions::new().create_new(true).write(true).open(path)
}
fn write_new(path: &Path, s: &str) -> Result<(), std::io::Error> {
    let mut f = file(path)?;
    f.write_all(s.as_bytes())?;
    f.sync_all()
}
pub(crate) fn case_table(path: &Path, cases: &[Case]) -> Result<(), Box<dyn Error>> {
    let mut w = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(file(path)?);
    w.write_record(["id", "family", "text", "expected", "refusal"])?;
    for c in cases {
        w.write_record([
            &c.id,
            &c.family,
            &c.text,
            &c.expected.map(|n| n.to_string()).unwrap_or_default(),
            &c.refusal,
        ])?;
    }
    w.flush()?;
    Ok(())
}
pub fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=4).contains(&args.len()) {
        return Err("usage: axon-uic-semantic-lab <new-directory> [rng] [rows 100..10000] [development|reserved]".into());
    }
    let root = PathBuf::from(&args[0]);
    let rng = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(61423917u64);
    let count = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1000usize);
    let protocol = args.get(3).map(String::as_str).unwrap_or("development");
    if !(100..=10000).contains(&count) || !["development", "reserved"].contains(&protocol) {
        return Err("semantic protocol bounds".into());
    }
    bounded_path(&root)?;
    fs::create_dir(&root)?;
    fs::create_dir(root.join("population"))?;
    let start = Instant::now();
    let mut t = LabTelemetry::from_environment("relational-v1");
    t.emit(
        "lab_start",
        &[
            ("method", V::Text("Relational::P0")),
            ("protocol", V::Text(protocol)),
            ("rng", V::Count(rng)),
            ("rows", V::Count(count as u64)),
            ("population", V::Count(POPULATION as u64)),
        ],
    )?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("evals/fixtures");
    fs::copy(
        fixtures.join("relational-v1-lessons.tsv"),
        root.join("lessons.tsv"),
    )?;
    fs::copy(
        fixtures.join("relational-v2-independent.tsv"),
        root.join("independent.tsv"),
    )?;
    // Freeze the full evaluation manifest before fitting or selecting anything.
    let dev = generated(rng ^ 0x01943819, 48);
    case_table(&root.join("selection.tsv"), &dev)?;
    let mut validation = generated(
        if protocol == "reserved" {
            0x8acdf02983
        } else {
            rng ^ 0x82938123
        },
        count,
    );
    validation.extend(independent(&root.join("independent.tsv"))?);
    case_table(&root.join("validation.tsv"), &validation)?;
    write_new(
        &root.join("protocol.txt"),
        &format!(
            "protocol=relational-v1/P0-consolidation-v2\nvalidation={protocol}\npopulation=64\ngenerations=4\ncampaigns=3\nselection_rows=96\nevolution=rule-inclusion-structure; not arbitrary new bytecode\nteaching=annotated roles, before/after states, 120 demonstrations\npriors=tokenization, aligned templates, specificity, numeric instruction set, facts, bounded reference resolution\nmutation=1..3 rule inclusion flips\ncrossover=per-rule parental mixing\nelitism=8\nselection=correct-descending, wrong-ascending, encoded-bytes-ascending\nconsolidation=union of development winners; parental positive predictions must survive\nvalidation_feedback=none\nnegative=exact refusal reason, not mere abstention\nlarge_population_gate=independent-natural-text success, diversity, retention\nprocess_rss=unmeasured\ninput_rng={rng}\nvalidation_hash={:016X}\nlessons_hash={:016X}\n",
            checksum(&fs::read(root.join("validation.tsv"))?),
            checksum(&fs::read(root.join("lessons.tsv"))?)
        ),
    )?;
    let mut rules = Vec::new();
    let lessons = load_lessons(&root.join("lessons.tsv"))?;
    for (i, (name, ds)) in lessons.iter().enumerate() {
        let clock = Instant::now();
        let rule = Rule::learn(ds, NumericBudget::default()).map_err(|e| format!("{name}: {e}"))?;
        t.emit(
            "skill",
            &[
                ("method", V::Text("Rule::learn")),
                ("case", V::Count(i as u64)),
                ("name", V::Text(name)),
                ("expression", V::Text(&rule.expression())),
                ("elapsed_ns", V::Count(clock.elapsed().as_nanos() as u64)),
                ("demonstrations", V::Count(ds.len() as u64)),
            ],
        )?;
        rules.push(rule);
    }
    let all = SemanticSeed::new(rules)?;
    all.save_new(&root.join("acquired-library.axsem"))?;
    let max = (1u64 << all.rules().len()) - 1;
    let mut ledger = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(file(&root.join("population.tsv"))?);
    ledger.write_record([
        "campaign",
        "generation",
        "candidate",
        "genotype",
        "parent_a",
        "parent_b",
        "bits",
        "rules",
        "correct",
        "checked",
        "wrong",
        "encoded_bytes",
        "elapsed_ns",
        "selected",
    ])?;
    let mut control = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(file(&root.join("random-control.tsv"))?);
    control.write_record([
        "campaign",
        "generation",
        "candidate",
        "bits",
        "correct",
        "checked",
        "wrong",
        "encoded_bytes",
    ])?;
    let mut winners = Vec::new();
    let mut unique = BTreeSet::new();
    let mut evaluated = 0u64;
    for campaign in 0..CAMPAIGNS {
        let mut r = Rng::new(rng.wrapping_add(campaign as u64 * 7193161));
        let mut control_rng = Rng::new(rng.wrapping_add(campaign as u64 * 7193161) ^ 0x681723ab);
        let mut random_best = 0;
        let mut population: Vec<_> = (0..POPULATION)
            .map(|_| (r.next_u64() & max, String::new(), String::new()))
            .collect();
        for generation in 0..GENERATIONS {
            let mut scored = Vec::new();
            for (i, (bits, pa, pb)) in population.iter().enumerate() {
                let seed = all.subset(*bits)?;
                let id = format!("{:016X}", seed.fingerprint());
                unique.insert(id.clone());
                let clock = Instant::now();
                let s = score(&seed, &dev);
                let elapsed = clock.elapsed().as_nanos() as u64;
                let bytes = seed.to_bytes()?.len();
                let candidate = campaign * GENERATIONS * POPULATION + generation * POPULATION + i;
                let artifact = format!("population/candidate-{candidate}.axsem");
                seed.save_new(&root.join(&artifact))?;
                evaluated += 1;
                t.emit(
                    "seed_evaluation",
                    &[
                        ("method", V::Text("Relational::selection")),
                        ("candidate", V::Count(candidate as u64)),
                        (
                            "generation",
                            V::Count((campaign * GENERATIONS + generation) as u64),
                        ),
                        ("genotype", V::Text(&id)),
                        ("parent_a", V::Text(pa)),
                        ("parent_b", V::Text(pb)),
                        ("solved", V::Count(s.correct)),
                        ("checked", V::Count(s.checked)),
                        ("wrong", V::Count(s.wrong)),
                        ("genes", V::Count(seed.rules().len() as u64)),
                        ("encoded_bytes", V::Count(bytes as u64)),
                        ("elapsed_ns", V::Count(elapsed)),
                        ("artifact", V::Text(&artifact)),
                    ],
                )?;
                scored.push((i, *bits, id, s, bytes, elapsed, pa.clone(), pb.clone()));
            }
            scored.sort_by_key(|v| (std::cmp::Reverse(v.3.correct), v.3.wrong, v.4, v.0));
            let selected: Vec<_> = scored
                .iter()
                .take(8)
                .map(|v| {
                    (campaign * GENERATIONS * POPULATION + generation * POPULATION + v.0) as u64
                })
                .collect();
            for (rank, v) in scored.iter().enumerate() {
                ledger.write_record([
                    campaign.to_string(),
                    generation.to_string(),
                    (campaign * GENERATIONS * POPULATION + generation * POPULATION + v.0)
                        .to_string(),
                    v.2.clone(),
                    v.6.clone(),
                    v.7.clone(),
                    v.1.to_string(),
                    v.1.count_ones().to_string(),
                    v.3.correct.to_string(),
                    v.3.checked.to_string(),
                    v.3.wrong.to_string(),
                    v.4.to_string(),
                    v.5.to_string(),
                    (rank < 8).to_string(),
                ])?;
            }
            ledger.flush()?;
            let diverse = scored
                .iter()
                .map(|v| v.2.as_str())
                .collect::<BTreeSet<_>>()
                .len();
            for i in 0..POPULATION {
                let bits = control_rng.next_u64() & max;
                let brain = all.subset(bits)?;
                let s = score(&brain, &dev);
                random_best = random_best.max(s.correct);
                control.write_record([
                    campaign.to_string(),
                    generation.to_string(),
                    i.to_string(),
                    bits.to_string(),
                    s.correct.to_string(),
                    s.checked.to_string(),
                    s.wrong.to_string(),
                    brain.to_bytes()?.len().to_string(),
                ])?;
            }
            control.flush()?;
            t.emit(
                "seed_selection",
                &[
                    ("method", V::Text("Relational::breeding")),
                    (
                        "generation",
                        V::Count((campaign * GENERATIONS + generation) as u64),
                    ),
                    ("selected", V::Counts(&selected)),
                    ("correct", V::Count(scored[0].3.correct)),
                    ("checked", V::Count(scored[0].3.checked)),
                    ("unique", V::Count(diverse as u64)),
                    ("random_best", V::Count(random_best)),
                    ("population", V::Count(POPULATION as u64)),
                ],
            )?;
            if generation + 1 == GENERATIONS {
                winners.push(all.subset(scored[0].1)?);
                break;
            }
            population = scored
                .iter()
                .take(8)
                .map(|v| (v.1, v.2.clone(), String::new()))
                .collect();
            while population.len() < POPULATION {
                let a = &scored[r.pick(8) as usize];
                let b = &scored[r.pick(8) as usize];
                let mix = r.next_u64() & max;
                let mut bits = (a.1 & mix) | (b.1 & !mix & max);
                for _ in 0..1 + r.pick(3) {
                    bits ^= 1 << r.pick(all.rules().len());
                }
                population.push((bits, a.2.clone(), b.2.clone()));
            }
        }
    }
    let mut predictions = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(file(&root.join("predictions.tsv"))?);
    predictions.write_record([
        "arm",
        "id",
        "family",
        "prediction",
        "expected",
        "status",
        "expected_refusal",
        "correct",
        "latency_ns",
        "text",
    ])?;
    let mut total = Score::default();
    let mut natural_correct = 0u64;
    let mut natural_checked = 0u64;
    let mut negative_correct = 0u64;
    let mut negative_checked = 0u64;
    let mut consolidated = SemanticSeed::default();
    for winner in &winners {
        consolidated = consolidated.merge(winner)?;
    }
    for winner in &winners {
        for c in &dev {
            if evaluate(winner, c).0 && !evaluate(&consolidated, c).0 {
                return Err("consolidation regressed a parental development prediction".into());
            }
        }
    }
    let parent_ids: Vec<_> = winners
        .iter()
        .map(|w| format!("{:016X}", w.fingerprint()))
        .collect();
    write_new(
        &root.join("consolidation.txt"),
        &format!(
            "parents={}\nchild={:016X}\nselection=development-only\nrules={}\ndevelopment_correct={}\ndevelopment_checked={}\npersonal_episodes=0\n",
            parent_ids.join(","),
            consolidated.fingerprint(),
            consolidated.rules().len(),
            score(&consolidated, &dev).correct,
            dev.len()
        ),
    )?;
    winners.push(consolidated);
    for (i, winner) in winners.iter().enumerate() {
        let clock = Instant::now();
        let genome = pack(winner, &root.join(format!("winner-{i}")))?;
        let copy = Genome::from_number(&genome.number())?;
        let inherited = born(&copy, &root.join(format!("birth-{i}")))?;
        t.emit(
            "genome",
            &[
                ("method", V::Text("Genome::birth")),
                ("name", V::Text(&format!("winner-{i}"))),
                ("encoded_bytes", V::Count(genome.encoded_bytes() as u64)),
                ("seed_digits", V::Count(genome.number().len() as u64)),
                ("elapsed_ns", V::Count(clock.elapsed().as_nanos() as u64)),
                ("artifact", V::Text(&format!("winner-{i}/seed.axnum"))),
            ],
        )?;
        // Two births from the number verify retention without source/catalog dependencies.
        let restarted = born(&copy, &root.join(format!("restart-{i}")))?;
        if inherited.fingerprint() != restarted.fingerprint() {
            return Err("birth changed learned structures".into());
        }
        for (arm, brain) in [
            (format!("winner-{i}"), &inherited),
            (format!("cold-{i}"), &SemanticSeed::default()),
        ] {
            let mut s = Score::default();
            for (j, c) in validation.iter().enumerate() {
                let clock = Instant::now();
                let (ok, p, status) = evaluate(brain, c);
                let latency = clock.elapsed().as_nanos() as u64;
                predictions.write_record([
                    &arm,
                    &c.id,
                    &c.family,
                    &p.map(|n| n.to_string()).unwrap_or_default(),
                    &c.expected.map(|n| n.to_string()).unwrap_or_default(),
                    &status,
                    &c.refusal,
                    &ok.to_string(),
                    &latency.to_string(),
                    &c.text,
                ])?;
                s.checked += 1;
                s.correct += u64::from(ok);
                s.wrong += u64::from(p.is_some() && !ok);
                s.refused += u64::from(p.is_none());
                if arm.starts_with("winner") {
                    if c.family == "natural-probe" {
                        natural_checked += 1;
                        natural_correct += u64::from(ok);
                    }
                    if c.family == "negative" {
                        negative_checked += 1;
                        negative_correct += u64::from(ok);
                    }
                    if j < 4 || c.id.starts_with('X') || c.family == "natural-probe" {
                        t.emit(
                            "prediction",
                            &[
                                ("method", V::Text("Session::narrative")),
                                ("name", V::Text(&c.id)),
                                ("arm", V::Text(&arm)),
                                ("family", V::Text(&c.family)),
                                ("sentence", V::Text(&c.text)),
                                ("predicted", p.map(V::Number).unwrap_or(V::Null)),
                                ("actual", c.expected.map(V::Number).unwrap_or(V::Null)),
                                ("passed", V::Flag(ok)),
                                ("status", V::Text(&status)),
                                ("latencyNs", V::Count(latency)),
                            ],
                        )?;
                    }
                }
            }
            if arm.starts_with("winner") {
                total.checked += s.checked;
                total.correct += s.correct;
                total.wrong += s.wrong;
                total.refused += s.refused;
            }
            t.emit(
                "validation",
                &[
                    ("method", V::Text("Relational::heldout")),
                    ("arm", V::Text(&arm)),
                    ("checked", V::Count(s.checked)),
                    ("correct", V::Count(s.correct)),
                    ("wrong", V::Count(s.wrong)),
                    ("refused", V::Count(s.refused)),
                ],
            )?;
        }
    }
    predictions.flush()?;
    // Scale is blocked by the independent natural-language gate, even if templated tests score well.
    let gate = natural_correct * 10 >= natural_checked * 9
        && total.wrong * 100 <= total.checked
        && negative_correct * 100 >= negative_checked * 95;
    let outcome = if gate {
        "bounded-gates-passed"
    } else {
        "partial; natural-language gate blocks P1/P2"
    };
    write_new(
        &root.join("result.txt"),
        &format!(
            "outcome={outcome}\nevaluated_candidates={evaluated}\nunique_structures={}\nselection_predictions={}\nrandom_control_predictions={}\nchecked={}\ncorrect={}\nwrong={}\nrefused={}\nnatural_correct={natural_correct}\nnatural_checked={natural_checked}\nnegative_correct={negative_correct}\nnegative_checked={negative_checked}\nelapsed_ns={}\n",
            unique.len(),
            evaluated * dev.len() as u64,
            evaluated * dev.len() as u64,
            total.checked,
            total.correct,
            total.wrong,
            total.refused,
            start.elapsed().as_nanos()
        ),
    )?;
    t.emit(
        "lab_result",
        &[
            ("method", V::Text("Relational::result")),
            ("status", V::Text(outcome)),
            ("candidates", V::Count(evaluated)),
            ("unique", V::Count(unique.len() as u64)),
            ("checked", V::Count(total.checked)),
            ("correct", V::Count(total.correct)),
            ("wrong", V::Count(total.wrong)),
            ("refused", V::Count(total.refused)),
            ("natural_correct", V::Count(natural_correct)),
            ("natural_checked", V::Count(natural_checked)),
            ("negative_correct", V::Count(negative_correct)),
            ("negative_checked", V::Count(negative_checked)),
            ("elapsed_ns", V::Count(start.elapsed().as_nanos() as u64)),
            ("artifact", V::Text("result.txt")),
        ],
    )?;
    println!(
        "relational-v1 checked={} correct={} wrong={} candidates={evaluated} unique={} natural={natural_correct}/{natural_checked}; {outcome}",
        total.checked,
        total.correct,
        total.wrong,
        unique.len()
    );
    Ok(())
}
