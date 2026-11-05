use crate::{
    Authority, Effect,
    cli::CliError,
    demand::{
        Catalog, DemandError, DemandRuntime, Module, ModuleKind, ModuleSpec, Seed,
        controller::CompositionBudget,
        genome::{BirthLimits, Genome, MAX_BYTES, bounded_path},
    },
    numeric::NumericBudget,
    semantic::{Change, Demonstration, Focus, Meaning, Rule, SemanticError, SemanticSeed, Session},
};
use std::{collections::BTreeMap, path::Path};

pub const USAGE: &str = "\n  axon semantic-fit <lessons.tsv> <new-directory>\n  axon semantic-merge <new-directory> <parent.axnum> <parent.axnum> [...]\n  axon semantic-answer <seed.axnum> <new-birth-directory> <narrative>";
fn error(e: impl std::fmt::Display) -> DemandError {
    DemandError::Module(e.to_string())
}
fn list(s: &str) -> Vec<String> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split('|').map(str::to_owned).collect()
    }
}
fn numbers(s: &str) -> Result<Vec<f64>, DemandError> {
    list(s)
        .iter()
        .map(|s| {
            s.parse::<f64>()
                .ok()
                .filter(|n| n.is_finite())
                .ok_or(DemandError::Invalid("lesson number"))
        })
        .collect()
}
/// Role labels are teacher supervision. The loader never dispatches on a Portuguese word.
pub fn load_lessons(path: &Path) -> Result<BTreeMap<String, Vec<Demonstration>>, DemandError> {
    bounded_path(path)?;
    let bytes = crate::text_skill::read_bounded(path, 262144).map_err(error)?;
    let mut r = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(bytes.as_slice());
    let required = [
        "rule",
        "sentence",
        "kind",
        "owner",
        "item",
        "before",
        "after",
        "focus_owner",
        "focus_item",
        "arguments",
        "answer",
    ];
    if r.headers().map_err(error)?.iter().ne(required) {
        return Err(DemandError::Invalid("lesson schema"));
    }
    let mut groups = BTreeMap::<String, Vec<Demonstration>>::new();
    for (i, row) in r.records().enumerate() {
        if i >= 512 {
            return Err(DemandError::Budget("lesson rows"));
        }
        let row = row.map_err(error)?;
        let owners = list(&row[3]);
        let items = list(&row[4]);
        let meaning = match &row[2] {
            "change" => {
                let before: Result<Vec<_>, _> = list(&row[5])
                    .iter()
                    .map(|s| {
                        if s == "_" {
                            Ok(None)
                        } else {
                            s.parse::<f64>()
                                .ok()
                                .filter(|n| n.is_finite())
                                .map(Some)
                                .ok_or(DemandError::Invalid("lesson prior"))
                        }
                    })
                    .collect();
                let before = before?;
                let after = numbers(&row[6])?;
                if [items.len(), before.len(), after.len()]
                    .iter()
                    .any(|n| *n != owners.len())
                {
                    return Err(DemandError::Invalid("lesson update arity"));
                }
                Meaning::Change(
                    (0..owners.len())
                        .map(|j| Change {
                            owner: owners[j].clone(),
                            item: items[j].clone(),
                            before: before[j],
                            after: after[j],
                        })
                        .collect(),
                )
            }
            "question" => {
                let prior = numbers(&row[5])?;
                let answer = numbers(&row[10])?;
                if items.len() != owners.len() || prior.len() != owners.len() || answer.len() != 1 {
                    return Err(DemandError::Invalid("lesson question arity"));
                }
                Meaning::Question {
                    cells: (0..owners.len())
                        .map(|j| (owners[j].clone(), items[j].clone(), prior[j]))
                        .collect(),
                    answer: answer[0],
                }
            }
            "calculate" => {
                let answer = numbers(&row[10])?;
                if answer.len() != 1 {
                    return Err(DemandError::Invalid("lesson answer"));
                }
                Meaning::Calculate {
                    arguments: numbers(&row[9])?,
                    answer: answer[0],
                }
            }
            "ignore" => Meaning::Ignore,
            _ => return Err(DemandError::Invalid("lesson kind")),
        };
        groups
            .entry(row[0].into())
            .or_default()
            .push(Demonstration {
                sentence: row[1].into(),
                focus: Focus {
                    owners: list(&row[7]),
                    items: list(&row[8]),
                },
                meaning,
            });
        if groups.len() > 64 {
            return Err(DemandError::Budget("lesson rule count"));
        }
    }
    if groups.is_empty() {
        return Err(DemandError::Invalid("empty lessons"));
    }
    Ok(groups)
}
pub fn pack(seed: &SemanticSeed, path: &Path) -> Result<Genome, DemandError> {
    bounded_path(path)?;
    std::fs::create_dir(path)?;
    seed.save_new(&path.join("relational.axsem"))
        .map_err(error)?;
    let mut catalog = Catalog::new(path);
    catalog.register(
        ModuleSpec {
            id: "relational",
            kind: ModuleKind::Semantic,
            file: "relational.axsem",
            input: "narrative",
            output: "evidenced-number",
            cost: 1,
        },
        &auth(),
    )?;
    let mut runtime = DemandRuntime::new(
        catalog,
        Seed {
            active_bytes: 262144,
            ..Seed::default()
        },
    )?;
    let genome = Genome::capture(
        seed.context(),
        &mut runtime,
        CompositionBudget::default(),
        &[],
        MAX_BYTES,
        &auth(),
    )?;
    genome.save_number_new(&path.join("seed.axnum"), &auth())?;
    Ok(genome)
}
pub fn born(genome: &Genome, path: &Path) -> Result<SemanticSeed, DemandError> {
    let mut birth = genome.birth(path, BirthLimits::default(), &auth())?;
    birth
        .controller
        .runtime_mut()
        .with_module("relational", "axon/relational-v1", &auth(), |m| match m {
            Module::Semantic(s) => Ok((**s).clone()),
            _ => Err(DemandError::Invalid("semantic birth")),
        })
}
fn auth() -> Authority {
    Authority::from([Effect::ReadData, Effect::WriteData])
}
pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    match args.first().map(String::as_str) {
        Some("semantic-merge") if (4..=10).contains(&args.len()) => {
            let root = Path::new(&args[1]);
            bounded_path(root)?;
            std::fs::create_dir(root).map_err(DemandError::Io)?;
            let mut child = SemanticSeed::default();
            for (i, path) in args[2..].iter().enumerate() {
                let parent = Genome::open_number(Path::new(path), &auth())?;
                let brain = born(&parent, &root.join(format!("parent-{i}")))?;
                child = child.merge(&brain).map_err(error)?;
            }
            let genome = pack(&child, &root.join("child"))?;
            Ok(format!(
                "parents={}\nrules={}\nencoded_bytes={}\nseed_digits={}\npersonal_episodes=0",
                args.len() - 2,
                child.rules().len(),
                genome.encoded_bytes(),
                genome.number().len()
            ))
        }
        Some("semantic-fit") if args.len() == 3 => {
            let lessons = load_lessons(Path::new(&args[1]))?;
            let mut rules = Vec::new();
            for (name, ds) in &lessons {
                rules.push(
                    Rule::learn(ds, NumericBudget::default())
                        .map_err(|e| error(format!("{name}: {e}")))?,
                );
            }
            let seed = SemanticSeed::new(rules).map_err(error)?;
            let genome = pack(&seed, Path::new(&args[2]))?;
            Ok(format!(
                "rules={}\nencoded_bytes={}\nseed_digits={}\npersonal_episodes=0",
                seed.rules().len(),
                genome.encoded_bytes(),
                genome.number().len()
            ))
        }
        Some("semantic-answer") if args.len() == 4 => {
            let genome = Genome::open_number(Path::new(&args[1]), &auth())?;
            let seed = born(&genome, Path::new(&args[2]))?;
            match Session::default().narrative(&seed,&args[3]){
                Ok(a)=>Ok(format!("status=answered\nanswer={}\nquery_program={}\nevidence={}\n{}",a.value,a.expression,a.evidence.len(),a.evidence.iter().map(|e|format!("event={} {}.{} {:?} -> {} via {}",e.event,e.owner,e.item,e.before,e.after,e.expression)).collect::<Vec<_>>().join("\n"))),
                Err(SemanticError::Unknown|SemanticError::Missing|SemanticError::Ambiguous|SemanticError::Domain)=>Ok("status=refused\nreason=unknown, missing, ambiguous or outside the numeric domain".into()),
                Err(e)=>Err(error(e).into()),
            }
        }
        _ => Err(CliError::Usage),
    }
}
