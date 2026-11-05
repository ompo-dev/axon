use crate::{
    TextBudget, TextExample, TextPrediction, TextSkill, TextSkillError,
    cli::CliError,
    live_store::checksum,
    text_skill::{publish_new, read_bounded},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::{Path, PathBuf},
};

pub(crate) const USAGE: &str = "\n  axon text-fit <examples.csv> --name <name> --context <context> --skill <new.axtext> [--delimiter comma|semicolon|tab]\n  axon text-refine <old.axtext> <examples.csv> --context <context> --skill <new.axtext> [--replace] [--delimiter comma|semicolon|tab]\n  axon text-inspect <skill.axtext>\n  axon text-preview <skill.axtext> <input.csv> --column <column> --context <context> [--result-column <new-column>] [--limit 1..100] [--delimiter comma|semicolon|tab]\n  axon text-apply <skill.axtext> <input.csv> --column <column> --result-column <new-column> --context <context> --output <new.csv> --confirm <review-token> [--delimiter comma|semicolon|tab]";
const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_ROWS: usize = 100_000;

#[derive(Debug)]
pub enum TextCommandError {
    Skill(TextSkillError),
    Csv(csv::Error),
    Schema(&'static str),
    Row { row: usize, status: &'static str },
}
impl fmt::Display for TextCommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Skill(e) => e.fmt(f),
            Self::Csv(e) => write!(f, "CSV error: {e}"),
            Self::Schema(s) => write!(f, "invalid text workflow: {s}"),
            Self::Row { row, status } => write!(
                f,
                "row {row} is {status}; no output published; use text-preview or provide a new demonstration"
            ),
        }
    }
}
impl std::error::Error for TextCommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Skill(e) => Some(e),
            Self::Csv(e) => Some(e),
            _ => None,
        }
    }
}
impl From<TextSkillError> for TextCommandError {
    fn from(e: TextSkillError) -> Self {
        Self::Skill(e)
    }
}
impl From<csv::Error> for TextCommandError {
    fn from(e: csv::Error) -> Self {
        Self::Csv(e)
    }
}

pub(crate) struct Options<'a> {
    pub(crate) values: BTreeMap<&'a str, &'a str>,
    replace: bool,
}
impl<'a> Options<'a> {
    pub(crate) fn parse(
        args: &'a [String],
        positional: usize,
        allowed: &[&str],
        replace_allowed: bool,
    ) -> Result<Self, CliError> {
        if args.len() < positional + 1 || args[1..=positional].iter().any(|s| s.starts_with("--")) {
            return Err(CliError::Usage);
        }
        let mut values = BTreeMap::new();
        let mut replace = false;
        let mut index = positional + 1;
        while index < args.len() {
            let flag = args[index].as_str();
            if flag == "--replace" && replace_allowed && !replace {
                replace = true;
                index += 1;
                continue;
            }
            let value = args.get(index + 1).ok_or(CliError::Usage)?;
            if !allowed.contains(&flag) || values.insert(flag, value.as_str()).is_some() {
                return Err(CliError::Usage);
            }
            index += 2;
        }
        Ok(Self { values, replace })
    }
    pub(crate) fn required(&self, flag: &str) -> Result<&'a str, CliError> {
        self.values.get(flag).copied().ok_or(CliError::Usage)
    }
    pub(crate) fn delimiter(&self) -> Result<u8, CliError> {
        match self.values.get("--delimiter").copied().unwrap_or("comma") {
            "comma" => Ok(b','),
            "semicolon" => Ok(b';'),
            "tab" => Ok(b'\t'),
            _ => Err(CliError::Usage),
        }
    }
    pub(crate) fn limit(&self) -> Result<usize, CliError> {
        let value = self.values.get("--limit").copied().unwrap_or("20");
        match value.parse::<usize>() {
            Ok(n) if (1..=100).contains(&n) => Ok(n),
            _ => Err(CliError::InvalidNumber {
                option: "--limit",
                value: value.into(),
            }),
        }
    }
}

pub(crate) struct Table {
    pub(crate) headers: csv::StringRecord,
    pub(crate) rows: Vec<csv::StringRecord>,
    pub(crate) fingerprint: u64,
}
pub(crate) fn table(path: &Path, delimiter: u8) -> Result<Table, TextCommandError> {
    let bytes = read_bounded(path, MAX_INPUT_BYTES)?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(false)
        .from_reader(bytes.as_slice());
    let headers = reader.headers()?.clone();
    if headers.is_empty()
        || headers.len() > 128
        || headers.iter().any(|s| {
            s.is_empty() || s.len() > 256 || s.trim() != s || s.chars().any(char::is_control)
        })
        || headers.iter().collect::<BTreeSet<_>>().len() != headers.len()
    {
        return Err(TextCommandError::Schema(
            "empty, duplicate or invalid headers",
        ));
    }
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record?;
        if rows.len() >= MAX_ROWS || record.iter().any(|s| s.len() > 8192) {
            return Err(TextCommandError::Schema("row or field limit"));
        }
        rows.push(record);
    }
    Ok(Table {
        headers,
        rows,
        fingerprint: checksum(&bytes),
    })
}
pub(crate) fn examples(path: &Path, delimiter: u8) -> Result<Vec<TextExample>, TextCommandError> {
    let table = table(path, delimiter)?;
    if table.headers.len() != 2 {
        return Err(TextCommandError::Schema(
            "demonstrations require exactly input,output columns",
        ));
    }
    let input = column(&table.headers, "input")?;
    let output = column(&table.headers, "output")?;
    if table.rows.is_empty() || table.rows.len() > TextBudget::default().max_examples {
        return Err(TextCommandError::Schema(
            "demonstrations require 1..32 rows",
        ));
    }
    Ok(table
        .rows
        .iter()
        .map(|row| TextExample::new(&row[input], &row[output]))
        .collect())
}
pub(crate) fn column(headers: &csv::StringRecord, name: &str) -> Result<usize, TextCommandError> {
    headers
        .iter()
        .position(|candidate| candidate == name)
        .ok_or(TextCommandError::Schema("requested column is absent"))
}

pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    let verb = args.first().map(String::as_str).ok_or(CliError::Usage)?;
    let (positional, allowed): (usize, &[&str]) = match verb {
        "text-fit" => (1, &["--name", "--context", "--skill", "--delimiter"]),
        "text-refine" => (2, &["--context", "--skill", "--delimiter"]),
        "text-inspect" => (1, &[]),
        "text-preview" => (
            2,
            &[
                "--column",
                "--context",
                "--limit",
                "--delimiter",
                "--result-column",
            ],
        ),
        "text-apply" => (
            2,
            &[
                "--column",
                "--result-column",
                "--context",
                "--output",
                "--delimiter",
                "--confirm",
            ],
        ),
        _ => return Err(CliError::Usage),
    };
    let options = Options::parse(args, positional, allowed, verb == "text-refine")?;
    let delimiter = options.delimiter()?;
    match verb {
        "text-fit" => {
            let name = options.required("--name")?;
            let context = options.required("--context")?;
            let path = options.required("--skill")?;
            let skill = TextSkill::learn(
                name,
                context,
                examples(Path::new(&args[1]), delimiter)?,
                TextBudget::default(),
            )?;
            let bytes = skill.save_new(path)?;
            Ok(format!(
                "status=learned\nskill={path:?}\nartifact_bytes={bytes}\n{}",
                inspect(&skill)
            ))
        }
        "text-refine" => {
            let context = options.required("--context")?;
            let path = options.required("--skill")?;
            let old = TextSkill::load(&args[1])?;
            let skill = old.refine(
                context,
                &examples(Path::new(&args[2]), delimiter)?,
                options.replace,
            )?;
            let bytes = skill.save_new(path)?;
            Ok(format!(
                "status=refined\nskill={path:?}\nartifact_bytes={bytes}\n{}",
                inspect(&skill)
            ))
        }
        "text-inspect" => Ok(inspect(&TextSkill::load(&args[1])?)),
        _ => {
            let name = options.required("--column")?;
            let context = options.required("--context")?;
            let output = if verb == "text-apply" {
                Some((
                    PathBuf::from(options.required("--output")?),
                    options.required("--result-column")?,
                ))
            } else {
                None
            };
            let limit = options.limit()?;
            let skill = TextSkill::load(&args[1])?;
            if context != skill.context() {
                return Err(TextSkillError::ContextMismatch.into());
            }
            let table = table(Path::new(&args[2]), delimiter)?;
            let index = column(&table.headers, name)?;
            let result_name = options
                .values
                .get("--result-column")
                .copied()
                .unwrap_or("axon_result");
            let token = review_token(&skill, table.fingerprint, name, result_name, delimiter);
            if output.is_some() && options.required("--confirm")? != format!("{token:016X}") {
                return Err(TextCommandError::Schema(
                    "confirmation does not match this skill/input/column; preview again",
                )
                .into());
            }
            if result_name.is_empty()
                || result_name.len() > 256
                || result_name.trim() != result_name
                || result_name.chars().any(char::is_control)
                || table.headers.iter().any(|name| name == result_name)
            {
                return Err(TextCommandError::Schema("result column must be new and valid").into());
            }
            let mut preview = format!(
                "skill_id={:016X}\nrevision={}\ninput_fingerprint={:016X}\nrows={}",
                skill.fingerprint(),
                skill.revision(),
                table.fingerprint,
                table.rows.len()
            );
            let mut predictions = Vec::new();
            let mut counts = [0_usize; 3];
            for (row_id, row) in table.rows.iter().enumerate() {
                let prediction =
                    skill
                        .predict(context, &row[index])
                        .map_err(|_| TextCommandError::Row {
                            row: row_id + 1,
                            status: "resource_limit",
                        })?;
                match &prediction {
                    TextPrediction::Agreed(_) => counts[0] += 1,
                    TextPrediction::Ambiguous { .. } => counts[1] += 1,
                    TextPrediction::OutOfDomain => counts[2] += 1,
                }
                if output.is_some() && !matches!(prediction, TextPrediction::Agreed(_)) {
                    return Err(TextCommandError::Row {
                        row: row_id + 1,
                        status: prediction.as_str(),
                    }
                    .into());
                }
                if row_id < limit && output.is_none() {
                    preview.push_str(&format!(
                        "\nrow={} input={:?} status={}",
                        row_id + 1,
                        &row[index],
                        prediction.as_str()
                    ));
                    match &prediction {
                        TextPrediction::Agreed(w) => preview.push_str(&format!(" output={:?}", w.output)),
                        TextPrediction::Ambiguous { first, second } => preview.push_str(&format!(" alternatives={:?},{:?} request=provide_expected_output_for_this_input", first.output, second.output)),
                        TextPrediction::OutOfDomain => {},
                    }
                }
                predictions.push(prediction);
            }
            preview.push_str(&format!(
                "\nagreed={}\nneeds_example={}\nout_of_domain={}",
                counts[0], counts[1], counts[2]
            ));
            if counts[1] == 0 && counts[2] == 0 {
                preview.push_str(&format!(
                    "\nreview_token={token:016X}\nresult_column={result_name:?}"
                ));
            }
            if let Some((path, result_name)) = output {
                if formula_like(result_name)
                    || table.headers.iter().any(formula_like)
                    || table
                        .rows
                        .iter()
                        .flat_map(|row| row.iter())
                        .any(formula_like)
                    || predictions
                        .iter()
                        .any(|p| matches!(p, TextPrediction::Agreed(w) if formula_like(&w.output)))
                {
                    return Err(TextCommandError::Schema(
                        "formula-like CSV cells are refused; originals are unchanged",
                    )
                    .into());
                }
                let mut writer = csv::WriterBuilder::new()
                    .delimiter(delimiter)
                    .from_writer(Vec::new());
                let mut headers = table.headers.clone();
                headers.push_field(result_name);
                writer
                    .write_record(&headers)
                    .map_err(TextCommandError::from)?;
                for (mut row, prediction) in table.rows.into_iter().zip(predictions) {
                    let TextPrediction::Agreed(w) = prediction else {
                        unreachable!("prevalidated batch");
                    };
                    row.push_field(&w.output);
                    writer.write_record(&row).map_err(TextCommandError::from)?;
                    if writer.get_ref().len() > MAX_OUTPUT_BYTES {
                        return Err(TextCommandError::Schema("output byte limit").into());
                    }
                }
                writer.flush().map_err(TextSkillError::from)?;
                let bytes = writer
                    .into_inner()
                    .map_err(|e| TextSkillError::from(e.into_error()))?;
                if bytes.len() > MAX_OUTPUT_BYTES {
                    return Err(TextCommandError::Schema("output byte limit").into());
                }
                publish_new(&path, &bytes)?;
                preview.push_str(&format!(
                    "\nstatus=published\noutput={:?}\noutput_bytes={}\noutput_fingerprint={:016X}",
                    path,
                    bytes.len(),
                    checksum(&bytes)
                ));
            }
            Ok(preview)
        }
    }
}

fn inspect(skill: &TextSkill) -> String {
    let parent = skill
        .parent()
        .map(|n| format!("{n:016X}"))
        .unwrap_or_else(|| "none".into());
    format!(
        "name={:?}\ncontext={:?}\nrevision={}\nparent={parent}\nskill_id={:016X}\nexamples={}\nprogram_pieces={}\nprogram_nodes={}\nprimitive_alternatives={}\nsearch_work={}\nrepresentative_program={:?}",
        skill.name(),
        skill.context(),
        skill.revision(),
        skill.fingerprint(),
        skill.examples().len(),
        skill.program().pieces(),
        skill.program().nodes(),
        skill.program().atoms(),
        skill.program().search_work(),
        skill.program().representative_steps()
    )
}

fn review_token(skill: &TextSkill, input: u64, column: &str, result: &str, delimiter: u8) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&skill.fingerprint().to_le_bytes());
    bytes.extend_from_slice(&input.to_le_bytes());
    bytes.push(delimiter);
    for value in [column, result] {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    checksum(&bytes)
}

pub(crate) fn formula_like(value: &str) -> bool {
    let first = value.chars().next();
    matches!(first, Some('\t' | '\r' | '\n'))
        || matches!(
            value.trim_start().chars().next(),
            Some('=' | '+' | '-' | '@' | '\u{ff1d}' | '\u{ff0b}' | '\u{ff0d}' | '\u{ff20}')
        )
}
