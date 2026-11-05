use crate::{
    Authority, Effect, TextBudget, TextSkill,
    cli::CliError,
    demand::{
        Catalog, DemandError, DemandRuntime, ModuleKind, ModuleSpec, Seed, TextPlan,
        exact_decimal_sum,
    },
    live_store::checksum,
    text_cli::{self, Options, TextCommandError},
    text_skill::{publish_new, read_bounded},
};
use std::path::Path;

pub(crate) const USAGE: &str = "\n  axon seed-build <modules.tsv> <new.axseed>\n  axon seed-plan <catalog.axseed> --from <contract> --goal <contract> [--active-bytes <bytes>] [--active-modules <count>] [--max-cost <units>]\n  axon seed-preview <catalog.axseed> <input.csv> --from <contract> --goal <contract> --column <name> [--result-column <name>] [--reduce sum] [--active-bytes <bytes>] [--limit 1..100] [--delimiter comma|semicolon|tab]\n  axon seed-apply <catalog.axseed> <input.csv> --from <contract> --goal <contract> --column <name> --result-column <name> --output <new.csv> --confirm <review-token> [--reduce sum] [--active-bytes <bytes>] [--delimiter comma|semicolon|tab]";
pub(crate) const TEACH_USAGE: &str = "\n  axon seed-teach <old.axseed> <examples.csv> --id <module-id> --from <contract> --goal <contract> --context <context> --module <new.axtext> --output <new.axseed> [--delimiter comma|semicolon|tab]";

fn seed(options: &Options<'_>) -> Result<Seed, CliError> {
    let mut seed = Seed::default();
    for (flag, target) in [
        ("--active-bytes", &mut seed.active_bytes),
        ("--active-modules", &mut seed.active_modules),
    ] {
        if let Some(v) = options.values.get(flag) {
            *target = v.parse().map_err(|_| CliError::InvalidNumber {
                option: flag,
                value: (*v).into(),
            })?;
        }
    }
    if let Some(v) = options.values.get("--max-cost") {
        seed.max_plan_cost = v.parse().map_err(|_| CliError::InvalidNumber {
            option: "--max-cost",
            value: (*v).into(),
        })?;
    }
    Ok(seed)
}
fn plan_report(plan: &TextPlan) -> String {
    format!(
        "input_contract={:?}\ngoal_contract={:?}\nroute={:?}\ndeclared_cost={}",
        plan.input, plan.goal, plan.modules, plan.cost
    )
}
fn token(catalog: u64, input: u64, delimiter: u8, values: &[&str]) -> String {
    let mut b = Vec::new();
    b.extend_from_slice(&catalog.to_le_bytes());
    b.extend_from_slice(&input.to_le_bytes());
    b.push(delimiter);
    for v in values {
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    }
    format!("{:016X}", checksum(&b))
}
pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    let verb = args.first().map(String::as_str).ok_or(CliError::Usage)?;
    let authority = Authority::from([Effect::ReadData, Effect::WriteData]);
    if verb == "seed-teach" {
        return teach(args, &authority);
    }
    if verb == "seed-build" {
        if args.len() != 3 {
            return Err(CliError::Usage);
        }
        let manifest = Path::new(&args[1]);
        let b = read_bounded(manifest, 1_048_576)
            .map_err(|e| CliError::Demand(DemandError::Module(e.to_string())))?;
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(b.as_slice());
        if reader
            .headers()
            .map_err(TextCommandError::from)?
            .iter()
            .collect::<Vec<_>>()
            != ["id", "kind", "file", "input", "output", "cost"]
        {
            return Err(CliError::Demand(DemandError::Invalid(
                "manifest headers: id kind file input output cost",
            )));
        }
        let mut catalog = Catalog::new(manifest.parent().unwrap_or(Path::new(".")));
        for record in reader.records() {
            let r = record.map_err(TextCommandError::from)?;
            let kind = match &r[1] {
                "text" => ModuleKind::Text,
                "behavior" => ModuleKind::Behavior,
                "world" => ModuleKind::World,
                "evolved" => ModuleKind::Evolved,
                "reactive" => ModuleKind::Reactive,
                "numeric" => ModuleKind::Numeric,
                "circuit" => ModuleKind::Circuit,
                "study" => ModuleKind::Study,
                _ => {
                    return Err(CliError::Demand(DemandError::Invalid(
                        "manifest module kind",
                    )));
                }
            };
            let cost = r[5]
                .parse()
                .map_err(|_| CliError::Demand(DemandError::Invalid("manifest cost")))?;
            catalog.register(
                ModuleSpec {
                    id: &r[0],
                    kind,
                    file: &r[2],
                    input: &r[3],
                    output: &r[4],
                    cost,
                },
                &authority,
            )?;
        }
        catalog.save_new(&args[2], &authority)?;
        return Ok(format!(
            "status=seed_catalog_created\nmodules={}\ncatalog_id={:016X}\ndormant_payload_bytes={}\neager_retained_bytes={}",
            catalog.entries().len(),
            catalog.fingerprint(),
            catalog.dormant_payload_bytes(),
            catalog.eager_retained_bytes()
        ));
    }
    let positional = if verb == "seed-plan" { 1 } else { 2 };
    let allowed = match verb {
        "seed-plan" => &[
            "--from",
            "--goal",
            "--active-bytes",
            "--active-modules",
            "--max-cost",
        ][..],
        "seed-preview" => &[
            "--from",
            "--goal",
            "--column",
            "--result-column",
            "--reduce",
            "--active-bytes",
            "--limit",
            "--delimiter",
        ][..],
        "seed-apply" => &[
            "--from",
            "--goal",
            "--column",
            "--result-column",
            "--reduce",
            "--active-bytes",
            "--output",
            "--confirm",
            "--delimiter",
        ][..],
        _ => return Err(CliError::Usage),
    };
    let options = Options::parse(args, positional, allowed, false)?;
    let catalog = Catalog::load(&args[1], &authority)?;
    let catalog_id = catalog.fingerprint();
    let mut runtime = DemandRuntime::new(catalog, seed(&options)?)?;
    let from = options.required("--from")?;
    let goal = options.required("--goal")?;
    let plan = runtime.plan_text(from, goal)?;
    if verb == "seed-plan" {
        return Ok(format!(
            "{}\nmodule_file_reads=0\nactive_modules=0",
            plan_report(&plan)
        ));
    }
    let delimiter = options.delimiter()?;
    let column_name = options.required("--column")?;
    let result_name = if verb == "seed-apply" {
        options.required("--result-column")?
    } else {
        options
            .values
            .get("--result-column")
            .copied()
            .unwrap_or("axon_result")
    };
    let reduce = options.values.get("--reduce").copied().unwrap_or("none");
    if !["none", "sum"].contains(&reduce) {
        return Err(CliError::Usage);
    }
    let table = text_cli::table(Path::new(&args[2]), delimiter)?;
    let index = text_cli::column(&table.headers, column_name)?;
    if result_name.is_empty()
        || result_name.len() > 256
        || result_name.trim() != result_name
        || result_name.chars().any(char::is_control)
        || table.headers.iter().any(|s| s == result_name)
    {
        return Err(TextCommandError::Schema("result column must be new and valid").into());
    }
    let review = token(
        catalog_id,
        table.fingerprint,
        delimiter,
        &[from, goal, column_name, result_name, reduce],
    );
    if verb == "seed-apply" && options.required("--confirm")? != review {
        return Err(TextCommandError::Schema(
            "confirmation does not match catalog/input/contracts/columns/reducer; preview again",
        )
        .into());
    }
    let values = table
        .rows
        .iter()
        .map(|r| r[index].to_owned())
        .collect::<Vec<_>>();
    let outputs = runtime.execute_text(&plan, &values, &authority)?;
    let total = if reduce == "sum" {
        format!("\nexact_unsigned_sum={}", exact_decimal_sum(&outputs)?)
    } else {
        String::new()
    };
    let stats = runtime.stats();
    let mut report = format!(
        "{}\nrows={}\nmodule_loads={}\nmodule_file_reads={}\nmodule_evictions={}\npeak_active_modules={}\npeak_module_retained_bytes={}\nruntime_metadata_bytes={}\ncold_load_ns={}{}",
        plan_report(&plan),
        values.len(),
        stats.loads,
        stats.file_reads,
        stats.evictions,
        stats.peak_active_modules,
        stats.peak_active_bytes,
        runtime.metadata_bytes(),
        stats.load_ns,
        total
    );
    if verb == "seed-preview" {
        for (i, (input, output)) in values
            .iter()
            .zip(&outputs)
            .take(options.limit()?)
            .enumerate()
        {
            report.push_str(&format!(
                "\nrow={} input={input:?} output={output:?}",
                i + 1
            ));
        }
        report.push_str(&format!("\nreview_token={review}"));
    } else {
        if text_cli::formula_like(result_name)
            || table.headers.iter().any(text_cli::formula_like)
            || table
                .rows
                .iter()
                .flat_map(|r| r.iter())
                .any(text_cli::formula_like)
            || outputs.iter().any(|s| text_cli::formula_like(s))
        {
            return Err(TextCommandError::Schema(
                "formula-like CSV cells refused; originals unchanged",
            )
            .into());
        }
        let path = options.required("--output")?;
        let mut writer = csv::WriterBuilder::new()
            .delimiter(delimiter)
            .from_writer(Vec::new());
        let mut headers = table.headers.clone();
        headers.push_field(result_name);
        writer
            .write_record(&headers)
            .map_err(TextCommandError::from)?;
        for (mut row, output) in table.rows.into_iter().zip(outputs) {
            row.push_field(&output);
            writer.write_record(&row).map_err(TextCommandError::from)?;
            if writer.get_ref().len() > 67_108_864 {
                return Err(CliError::Demand(DemandError::Budget("CSV output bytes")));
            }
        }
        writer.flush().map_err(DemandError::from)?;
        let b = writer
            .into_inner()
            .map_err(|e| DemandError::Io(e.into_error()))?;
        if b.len() > 67_108_864 {
            return Err(CliError::Demand(DemandError::Budget("CSV output bytes")));
        }
        publish_new(Path::new(path), &b).map_err(|e| DemandError::Module(e.to_string()))?;
        report.push_str(&format!(
            "\nstatus=published\noutput={path:?}\noutput_bytes={}\noutput_fingerprint={:016X}",
            b.len(),
            checksum(&b)
        ));
    }
    Ok(report)
}

fn teach(args: &[String], authority: &Authority) -> Result<String, CliError> {
    let options = Options::parse(
        args,
        2,
        &[
            "--id",
            "--from",
            "--goal",
            "--context",
            "--module",
            "--output",
            "--delimiter",
        ],
        false,
    )?;
    let mut catalog = Catalog::load(&args[1], authority)?;
    let id = options.required("--id")?;
    let from = options.required("--from")?;
    let goal = options.required("--goal")?;
    let context = options.required("--context")?;
    let module_path = Path::new(options.required("--module")?);
    let output_path = Path::new(options.required("--output")?);
    for path in [module_path, output_path] {
        if path.exists() {
            return Err(CliError::Demand(DemandError::Invalid(
                "teaching outputs must be new",
            )));
        }
        if std::fs::canonicalize(crate::demand::parent_or_current(path))
            .map_err(DemandError::from)?
            != std::fs::canonicalize(catalog.root()).map_err(DemandError::from)?
        {
            return Err(CliError::Demand(DemandError::Invalid(
                "teaching outputs must be siblings of the catalog",
            )));
        }
    }
    let file = module_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or(CliError::Usage)?;
    crate::demand::file_name(file)?;
    let old = catalog.get(id).cloned();
    if let Some(old) = &old {
        if old.kind != ModuleKind::Text
            || old.context != context
            || old.input != from
            || old.output != goal
        {
            return Err(CliError::Demand(DemandError::Context));
        }
    } else {
        catalog.check_registration(&ModuleSpec {
            id,
            kind: ModuleKind::Text,
            file,
            input: from,
            output: goal,
            cost: 1,
        })?;
    }
    let demonstrations = text_cli::examples(Path::new(&args[2]), options.delimiter()?)?;
    let skill = if let Some(old) = &old {
        let mut runtime = DemandRuntime::new(
            catalog,
            Seed {
                active_bytes: 67_108_864,
                ..Seed::default()
            },
        )?;
        let skill = runtime.with_module(id, context, authority, |m| {
            let crate::demand::Module::Text(parent) = m else {
                return Err(DemandError::Invalid("text parent required"));
            };
            parent
                .refine(context, &demonstrations, false)
                .map_err(|e| DemandError::Module(e.to_string()))
        })?;
        // Reload descriptors only; keep the old artifact untouched.
        catalog = Catalog::load(&args[1], authority)?;
        if old.identity != catalog.get(id).ok_or(CliError::Usage)?.identity {
            return Err(CliError::Demand(DemandError::Identity));
        }
        skill
    } else {
        TextSkill::learn(id, context, demonstrations, TextBudget::default())
            .map_err(|e| DemandError::Module(e.to_string()))?
    };
    skill
        .save_new(module_path)
        .map_err(|e| DemandError::Module(e.to_string()))?;
    if old.is_some() {
        catalog.revise_text(id, file, authority)?;
    } else {
        catalog.register(
            ModuleSpec {
                id,
                kind: ModuleKind::Text,
                file,
                input: from,
                output: goal,
                cost: 1,
            },
            authority,
        )?;
    }
    // Two no-clobber publications, not a multi-file atomic transaction. A failed catalog
    // publication can leave an orphaned immutable module; it never replaces the old seed.
    catalog.save_new(output_path, authority)?;
    Ok(format!(
        "status={}\nmodule_id={id:?}\nrevision={}\nexamples={}\nmodule_fingerprint={:016X}\nnew_catalog={:?}\nold_catalog_unchanged=true",
        if old.is_some() { "refined" } else { "learned" },
        skill.revision(),
        skill.examples().len(),
        skill.fingerprint(),
        output_path
    ))
}
