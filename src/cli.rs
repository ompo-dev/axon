use std::{collections::BTreeSet, fmt, fs, path::PathBuf};

use crate::{
    ArtifactOutcome, ArtifactStore, AxonTask, EpisodeError, EpisodeStore, EvidenceResult,
    ExperiencedSolveError, InvestigationResult, KnowledgeError, KnowledgeFeedbackLabel,
    KnowledgeStore, RecallHit, RecallResult, RecallStatus, SolveError, TaskError, solve_task,
    solve_task_with_experience,
};

const USAGE: &str = "usage:\n  axon solve <task.axon> [--artifact-dir <directory>] [--episodes <file.axep> --experience-context <context>]\n  axon learn <file-or-directory> [--memory <memory.axm>]\n  axon recall <query> [--memory <memory.axm>] [--limit <1..100>]\n  axon ask <query> [--memory <memory.axm>] [--limit <1..100>]\n  axon investigate <query> [--memory <memory.axm>] [--limit <1..100>]\n  axon teach <query> (--relevant <source> | --irrelevant <source>) [--memory <memory.axm>]\n  axon memory-status [--memory <memory.axm>]\n  axon forget <file-or-directory> [--memory <memory.axm>]\n  axon memory-eval <queries.tsv> [--memory <memory.axm>] [--limit <1..100>]\n  axon memory-reason-eval <queries.tsv> [--memory <memory.axm>] [--limit <1..100>]\n  axon memory-coherence-eval <queries.tsv> [--memory <memory.axm>] [--limit <1..100>]";

#[derive(Debug)]
pub enum CliError {
    Usage,
    ReadTask {
        path: PathBuf,
        source: std::io::Error,
    },
    ReadEvaluation {
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidEvaluation {
        line: usize,
        reason: &'static str,
    },
    InvalidNumber {
        option: &'static str,
        value: String,
    },
    Task(TaskError),
    Solve(SolveError),
    Knowledge(KnowledgeError),
    Episode(EpisodeError),
    Experience(ExperiencedSolveError),
    World(crate::world::WorldError),
    Behavior(crate::behavior::BehaviorError),
    Demand(crate::demand::DemandError),
    Transfer(crate::transfer::TransferError),
    Factor(crate::factor::FactorError),
    Numeric(crate::numeric::NumericError),
    Study(crate::cumulative::StudyError),
    #[cfg(feature = "text-workflows")]
    Case(crate::casework::CaseError),
    #[cfg(feature = "text-workflows")]
    Text(crate::text_cli::TextCommandError),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage => write!(
                formatter,
                "{USAGE}{}{}{}{}{}{}{}{}{}{}{}{}{}",
                crate::episodic_cli::USAGE,
                text_usage(),
                crate::world_cli::USAGE,
                crate::behavior_cli::USAGE,
                demand_usage(),
                teach_usage(),
                crate::transfer_cli::USAGE,
                crate::factor_cli::USAGE,
                crate::numeric_cli::USAGE,
                crate::genome_cli::USAGE,
                language_usage(),
                cumulative_usage(),
                semantic_usage()
            ),
            Self::ReadTask { path, .. } => {
                write!(formatter, "cannot read task `{}`", path.display())
            }
            Self::ReadEvaluation { path, .. } => {
                write!(formatter, "cannot read evaluation `{}`", path.display())
            }
            Self::InvalidEvaluation { line, reason } => {
                write!(formatter, "invalid evaluation at line {line}: {reason}")
            }
            Self::InvalidNumber { option, value } => {
                write!(formatter, "invalid value `{value}` for {option}")
            }
            Self::Task(error) => error.fmt(formatter),
            Self::Solve(error) => error.fmt(formatter),
            Self::Knowledge(error) => error.fmt(formatter),
            Self::Episode(error) => error.fmt(formatter),
            Self::Experience(error) => error.fmt(formatter),
            Self::World(error) => error.fmt(formatter),
            Self::Behavior(error) => error.fmt(formatter),
            Self::Demand(error) => error.fmt(formatter),
            Self::Transfer(error) => error.fmt(formatter),
            Self::Factor(error) => error.fmt(formatter),
            Self::Numeric(error) => error.fmt(formatter),
            Self::Study(error) => error.fmt(formatter),
            #[cfg(feature = "text-workflows")]
            Self::Case(error) => error.fmt(formatter),
            #[cfg(feature = "text-workflows")]
            Self::Text(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReadTask { source, .. } | Self::ReadEvaluation { source, .. } => Some(source),
            Self::Task(error) => Some(error),
            Self::Solve(error) => Some(error),
            Self::Knowledge(error) => Some(error),
            Self::Episode(error) => Some(error),
            Self::Experience(error) => Some(error),
            Self::World(error) => Some(error),
            Self::Behavior(error) => Some(error),
            Self::Demand(error) => Some(error),
            Self::Transfer(error) => Some(error),
            Self::Factor(error) => Some(error),
            Self::Numeric(error) => Some(error),
            Self::Study(error) => Some(error),
            #[cfg(feature = "text-workflows")]
            Self::Case(error) => Some(error),
            #[cfg(feature = "text-workflows")]
            Self::Text(error) => Some(error),
            Self::Usage | Self::InvalidEvaluation { .. } | Self::InvalidNumber { .. } => None,
        }
    }
}

impl From<KnowledgeError> for CliError {
    fn from(error: KnowledgeError) -> Self {
        Self::Knowledge(error)
    }
}

impl From<EpisodeError> for CliError {
    fn from(error: EpisodeError) -> Self {
        Self::Episode(error)
    }
}
impl From<crate::demand::DemandError> for CliError {
    fn from(error: crate::demand::DemandError) -> Self {
        Self::Demand(error)
    }
}
impl From<crate::numeric::NumericError> for CliError {
    fn from(error: crate::numeric::NumericError) -> Self {
        Self::Numeric(error)
    }
}
impl From<crate::cumulative::StudyError> for CliError {
    fn from(error: crate::cumulative::StudyError) -> Self {
        Self::Study(error)
    }
}
#[cfg(feature = "text-workflows")]
impl From<crate::casework::CaseError> for CliError {
    fn from(error: crate::casework::CaseError) -> Self {
        Self::Case(error)
    }
}
#[cfg(feature = "text-workflows")]
impl From<crate::TextSkillError> for CliError {
    fn from(error: crate::TextSkillError) -> Self {
        Self::Text(crate::text_cli::TextCommandError::Skill(error))
    }
}
#[cfg(feature = "text-workflows")]
impl From<crate::text_cli::TextCommandError> for CliError {
    fn from(error: crate::text_cli::TextCommandError) -> Self {
        Self::Text(error)
    }
}

pub fn command(arguments: &[String]) -> Result<String, CliError> {
    match arguments.first().map(String::as_str) {
        #[cfg(feature = "text-workflows")]
        Some("semantic-fit" | "semantic-answer" | "semantic-merge") => {
            crate::semantic_cli::command(arguments)
        }
        #[cfg(feature = "text-workflows")]
        Some("study-inspect" | "case-plan") => crate::cumulative_cli::command(arguments),
        #[cfg(feature = "text-workflows")]
        Some("dictionary-import" | "dictionary-query" | "language-answer") => {
            crate::language_cli::command(arguments)
        }
        Some(
            "genome-pack" | "genome-inspect" | "genome-birth" | "genome-mutate" | "genome-apply"
            | "genome-number",
        ) => crate::genome_cli::command(arguments),
        Some("solve") => solve_command(arguments),
        Some(
            "numeric-fit"
            | "numeric-revise"
            | "numeric-inspect"
            | "numeric-predict"
            | "numeric-eval"
            | "numeric-mine"
            | "numeric-library-inspect"
            | "numeric-develop"
            | "numeric-evolve"
            | "evolved-inspect"
            | "evolved-predict"
            | "circuit-predict"
            | "circuit-export",
        ) => crate::numeric_cli::command(arguments),
        Some("dna-create" | "dna-inspect" | "dna-mutate" | "dna-extend" | "dna-predict") => {
            crate::numeric_cli::command(arguments)
        }
        Some("transfer-demo" | "transfer-replay") => crate::transfer_cli::command(arguments),
        Some("factor-demo" | "factor-publish") => crate::factor_cli::command(arguments),
        #[cfg(feature = "text-workflows")]
        Some("seed-build" | "seed-plan" | "seed-preview" | "seed-apply" | "seed-teach") => {
            crate::demand_cli::command(arguments)
        }
        Some("behavior-demo" | "behavior-inspect" | "behavior-plan") => {
            crate::behavior_cli::command(arguments)
        }
        Some("world-demo" | "world-inspect" | "world-plan" | "world-rebuild") => {
            crate::world_cli::command(arguments)
        }
        #[cfg(feature = "text-workflows")]
        Some("text-fit" | "text-refine" | "text-inspect" | "text-preview" | "text-apply") => {
            crate::text_cli::command(arguments)
        }
        Some(
            "observe" | "revise" | "retract" | "belief" | "episode-history" | "episodes-status"
            | "remember-outcome" | "experience",
        ) => crate::episodic_cli::command(arguments),
        Some("learn") => learn_command(arguments),
        Some("recall") => recall_command(arguments),
        Some("ask" | "reason") => ask_command(arguments),
        Some("investigate" | "connect") => investigate_command(arguments),
        Some("teach") => teach_command(arguments),
        Some("memory-status" | "status") => memory_status_command(arguments),
        Some("forget") => forget_command(arguments),
        Some("memory-eval" | "eval") => memory_eval_command(arguments),
        Some("memory-reason-eval" | "reason-eval") => memory_reason_eval_command(arguments),
        Some("memory-coherence-eval" | "coherence-eval") => {
            memory_coherence_eval_command(arguments)
        }
        Some("help" | "--help" | "-h") if arguments.len() == 1 => Ok(format!(
            "{USAGE}{}{}{}{}{}{}{}{}{}{}{}{}",
            crate::episodic_cli::USAGE,
            text_usage(),
            crate::world_cli::USAGE,
            crate::behavior_cli::USAGE,
            demand_usage(),
            teach_usage(),
            crate::transfer_cli::USAGE,
            crate::factor_cli::USAGE,
            crate::numeric_cli::USAGE,
            crate::genome_cli::USAGE,
            language_usage(),
            cumulative_usage()
        )),
        _ => Err(CliError::Usage),
    }
}

fn language_usage() -> &'static str {
    #[cfg(feature = "text-workflows")]
    {
        crate::language_cli::USAGE
    }
    #[cfg(not(feature = "text-workflows"))]
    {
        ""
    }
}
fn cumulative_usage() -> &'static str {
    #[cfg(feature = "text-workflows")]
    {
        crate::cumulative_cli::USAGE
    }
    #[cfg(not(feature = "text-workflows"))]
    {
        ""
    }
}
fn semantic_usage() -> &'static str {
    #[cfg(feature = "text-workflows")]
    {
        crate::semantic_cli::USAGE
    }
    #[cfg(not(feature = "text-workflows"))]
    {
        ""
    }
}

fn text_usage() -> &'static str {
    #[cfg(feature = "text-workflows")]
    {
        crate::text_cli::USAGE
    }
    #[cfg(not(feature = "text-workflows"))]
    {
        ""
    }
}

fn demand_usage() -> &'static str {
    #[cfg(feature = "text-workflows")]
    {
        crate::demand_cli::USAGE
    }
    #[cfg(not(feature = "text-workflows"))]
    {
        ""
    }
}
fn teach_usage() -> &'static str {
    #[cfg(feature = "text-workflows")]
    {
        crate::demand_cli::TEACH_USAGE
    }
    #[cfg(not(feature = "text-workflows"))]
    {
        ""
    }
}

pub fn solve_command(arguments: &[String]) -> Result<String, CliError> {
    let options = parse_solve_arguments(arguments)?;
    let task_path = options.task_path;
    let source = fs::read_to_string(&task_path).map_err(|source| CliError::ReadTask {
        path: task_path.clone(),
        source,
    })?;
    let task = AxonTask::parse(&source).map_err(CliError::Task)?;
    let artifacts = ArtifactStore::open(options.artifact_root);
    let (receipt, experience) = if let Some((path, context)) = options.experience {
        let mut episodes = EpisodeStore::open(path)?.0;
        let (receipt, episode) =
            solve_task_with_experience(&artifacts, &task, &mut episodes, &context)
                .map_err(CliError::Experience)?;
        (
            receipt,
            format!(
                "\nepisode_id={}\nepisodes={}",
                episode.id,
                episodes.path().display()
            ),
        )
    } else {
        (
            solve_task(&artifacts, &task).map_err(CliError::Solve)?,
            String::new(),
        )
    };
    let outcome = match receipt.outcome() {
        ArtifactOutcome::ModularSum(total) => format!("modular_sum={total}"),
        ArtifactOutcome::ExactAverage(average) => {
            format!(
                "exact_average={}/{}",
                average.numerator(),
                average.denominator()
            )
        }
        ArtifactOutcome::ExactMinimum(minimum) => format!("exact_minimum={minimum}"),
        ArtifactOutcome::ExactPopulationVariance(variance) => format!(
            "exact_population_variance={}/{}",
            variance.numerator(),
            variance.denominator()
        ),
    };
    Ok(format!(
        "task={}\nartifact={}\nsemantic_hash={}\ncertificate={}\nphysical_backend={}\nphysical_evidence={}\nartifact_path={}\nresult={outcome}{experience}",
        receipt.task_name(),
        receipt.artifact_status().as_str(),
        receipt.semantic_hash(),
        receipt.certificate_status().as_str(),
        receipt.physical().backend().as_str(),
        receipt.physical().evidence().as_str(),
        receipt.artifact_path().display()
    ))
}

fn learn_command(arguments: &[String]) -> Result<String, CliError> {
    if arguments.len() < 2 {
        return Err(CliError::Usage);
    }
    let input = PathBuf::from(&arguments[1]);
    let memory = parse_memory_only(arguments, 2)?;
    let mut store = KnowledgeStore::open(memory)?;
    let report = store.learn_path(&input)?;
    Ok(format!(
        "action=learn\nmemory={}\ninput={}\ndiscovered_files={}\nadded_sources={}\nupdated_sources={}\nunchanged_sources={}\nskipped_files={}\nchanged_chunks={}\nbytes_read={}\ntotal_sources={}\ntotal_chunks={}\nindex_bytes={}",
        store.path().display(),
        input.display(),
        report.discovered_files(),
        report.added_sources(),
        report.updated_sources(),
        report.unchanged_sources(),
        report.skipped_files(),
        report.changed_chunks(),
        report.bytes_read(),
        report.total_sources(),
        report.total_chunks(),
        report.index_bytes()
    ))
}

fn recall_command(arguments: &[String]) -> Result<String, CliError> {
    let (query, memory, limit) = parse_query_options(arguments, 1)?;
    let store = KnowledgeStore::open(memory)?;
    let result = store.recall(&query, limit)?;
    Ok(format_recall(&query, &result))
}

fn ask_command(arguments: &[String]) -> Result<String, CliError> {
    let (query, memory, limit) = parse_query_options(arguments, 1)?;
    let store = KnowledgeStore::open(memory)?;
    let result = store.reason(&query, limit)?;
    Ok(format_reason(&query, &result))
}

fn investigate_command(arguments: &[String]) -> Result<String, CliError> {
    let (query, memory, limit) = parse_query_options(arguments, 1)?;
    let store = KnowledgeStore::open(memory)?;
    let result = store.investigate(&query, limit)?;
    Ok(format_investigation(&query, &result))
}

fn teach_command(arguments: &[String]) -> Result<String, CliError> {
    let (query, source, label, memory) = parse_teach_options(arguments, 1)?;
    let mut store = KnowledgeStore::open(memory)?;
    let report = store.record_feedback(&query, &source, label)?;
    Ok(format!(
        "action=teach\nmemory={}\nquery={}\nlabel={}\nsource={}\nrelevant_observations={}\nirrelevant_observations={}\ntotal_feedback_records={}",
        store.path().display(),
        single_line(&query, 240),
        report.label().as_str(),
        report.source().display(),
        report.relevant_observations(),
        report.irrelevant_observations(),
        report.total_records()
    ))
}

fn memory_status_command(arguments: &[String]) -> Result<String, CliError> {
    let memory = parse_memory_only(arguments, 1)?;
    let store = KnowledgeStore::open(memory)?;
    let stats = store.stats()?;
    Ok(format!(
        "memory={}\nsources={}\nchunks={}\ntext_bytes={}\nfeedback_records={}\nfeedback_observations={}\nindex_bytes={}",
        store.path().display(),
        stats.sources(),
        stats.chunks(),
        stats.text_bytes(),
        stats.feedback_records(),
        stats.feedback_observations(),
        stats.index_bytes()
    ))
}

fn forget_command(arguments: &[String]) -> Result<String, CliError> {
    if arguments.len() < 2 {
        return Err(CliError::Usage);
    }
    let source = PathBuf::from(&arguments[1]);
    let memory = parse_memory_only(arguments, 2)?;
    let mut store = KnowledgeStore::open(memory)?;
    let removed = store.forget(&source)?;
    let stats = store.stats()?;
    Ok(format!(
        "action=forget\nmemory={}\nsource={}\nremoved_sources={}\ntotal_sources={}\ntotal_chunks={}\nindex_bytes={}",
        store.path().display(),
        source.display(),
        removed,
        stats.sources(),
        stats.chunks(),
        stats.index_bytes()
    ))
}

fn memory_eval_command(arguments: &[String]) -> Result<String, CliError> {
    memory_evaluation_command(arguments, false)
}

fn memory_reason_eval_command(arguments: &[String]) -> Result<String, CliError> {
    memory_evaluation_command(arguments, true)
}

fn memory_coherence_eval_command(arguments: &[String]) -> Result<String, CliError> {
    if arguments.len() < 2 {
        return Err(CliError::Usage);
    }
    let evaluation_path = PathBuf::from(&arguments[1]);
    let (memory, limit) = parse_memory_and_limit(arguments, 2)?;
    let source =
        fs::read_to_string(&evaluation_path).map_err(|source| CliError::ReadEvaluation {
            path: evaluation_path.clone(),
            source,
        })?;
    let cases = parse_evaluation(&source)?;
    let store = KnowledgeStore::open(memory)?;
    let mut top_one = 0_usize;
    let mut recalled = 0_usize;
    let mut positive_cases = 0_usize;
    let mut negative_cases = 0_usize;
    let mut correct_abstentions = 0_usize;
    let mut reciprocal_rank = 0.0_f64;
    let mut top_path_coverage = 0.0_f64;
    let mut top_path_coherence = 0.0_f64;
    let mut cases_with_paths = 0_usize;
    let mut lines = Vec::new();

    for (index, (query, expected)) in cases.iter().enumerate() {
        let result = store.investigate(query, limit)?;
        if let Some(path) = result.paths().first() {
            top_path_coverage += path.weighted_coverage();
            top_path_coherence += path.coherence();
            cases_with_paths += 1;
        }
        if expected == "-" {
            negative_cases += 1;
            let correct = result.paths().is_empty() && result.status() == RecallStatus::NoEvidence;
            correct_abstentions += usize::from(correct);
            lines.push(format!(
                "case={}\tresult={}\tstatus={}\texpected=abstain\tpaths={}\tquery={}",
                index + 1,
                if correct { "pass" } else { "fail" },
                result.status().as_str(),
                result.paths().len(),
                single_line(query, 240)
            ));
            continue;
        }
        if expected.split('&').any(|clause| clause.trim().is_empty()) {
            return Err(CliError::InvalidEvaluation {
                line: index + 1,
                reason: "required source clauses must not be empty",
            });
        }
        positive_cases += 1;
        let rank = result
            .paths()
            .iter()
            .position(|path| evidence_set_matches(path.evidence(), expected))
            .map(|position| position + 1);
        if rank == Some(1) {
            top_one += 1;
        }
        if let Some(rank) = rank {
            recalled += 1;
            reciprocal_rank += 1.0 / rank as f64;
        }
        let (evidence_count, link_count) = result
            .paths()
            .first()
            .map_or((0, 0), |path| (path.evidence().len(), path.links().len()));
        lines.push(format!(
            "case={}\trank={}\tstatus={}\texpected={}\tevidence={}\tlinks={}\tquery={}",
            index + 1,
            rank.map_or_else(|| "miss".to_owned(), |rank| rank.to_string()),
            result.status().as_str(),
            expected,
            evidence_count,
            link_count,
            single_line(query, 240)
        ));
    }

    let total = cases.len();
    let mean_reciprocal_rank = if positive_cases == 0 {
        0.0
    } else {
        reciprocal_rank / positive_cases as f64
    };
    let mean_top_path_coverage = if cases_with_paths == 0 {
        0.0
    } else {
        top_path_coverage / cases_with_paths as f64
    };
    let mean_top_path_coherence = if cases_with_paths == 0 {
        0.0
    } else {
        top_path_coherence / cases_with_paths as f64
    };
    Ok(format!(
        "memory={}\nevaluation={}\nmode=evidence_graph\ncases={}\npositive_cases={}\nnegative_cases={}\ntop1={}/{}\nrecall_at_{}={}/{}\ncorrect_abstentions={}/{}\noverall_correct={}/{}\nmrr={:.6}\nmean_top_path_coverage={:.6}\nmean_top_path_coherence={:.6}\n{}",
        store.path().display(),
        evaluation_path.display(),
        total,
        positive_cases,
        negative_cases,
        top_one,
        positive_cases,
        limit,
        recalled,
        positive_cases,
        correct_abstentions,
        negative_cases,
        top_one + correct_abstentions,
        total,
        mean_reciprocal_rank,
        mean_top_path_coverage,
        mean_top_path_coherence,
        lines.join("\n")
    ))
}

fn memory_evaluation_command(arguments: &[String], composed: bool) -> Result<String, CliError> {
    if arguments.len() < 2 {
        return Err(CliError::Usage);
    }
    let evaluation_path = PathBuf::from(&arguments[1]);
    let (memory, limit) = parse_memory_and_limit(arguments, 2)?;
    let source =
        fs::read_to_string(&evaluation_path).map_err(|source| CliError::ReadEvaluation {
            path: evaluation_path.clone(),
            source,
        })?;
    let cases = parse_evaluation(&source)?;
    let store = KnowledgeStore::open(memory)?;
    let mut top_one = 0_usize;
    let mut recalled = 0_usize;
    let mut positive_cases = 0_usize;
    let mut negative_cases = 0_usize;
    let mut correct_abstentions = 0_usize;
    let mut reciprocal_rank = 0.0_f64;
    let mut lines = Vec::new();
    for (index, (query, expected)) in cases.iter().enumerate() {
        let (status, sources) = if composed {
            let result = store.reason(query, limit)?;
            (
                result.status(),
                result
                    .bundles()
                    .iter()
                    .map(|bundle| bundle.source().to_owned())
                    .collect::<Vec<_>>(),
            )
        } else {
            let result = store.recall(query, limit)?;
            (
                result.status(),
                result
                    .hits()
                    .iter()
                    .map(|hit| hit.source().to_owned())
                    .collect::<Vec<_>>(),
            )
        };
        if expected == "-" {
            negative_cases += 1;
            let correct = sources.is_empty() && status == RecallStatus::NoEvidence;
            correct_abstentions += usize::from(correct);
            lines.push(format!(
                "case={}\tresult={}\tstatus={}\texpected=abstain\tquery={}",
                index + 1,
                if correct { "pass" } else { "fail" },
                status.as_str(),
                single_line(query, 240)
            ));
            continue;
        }
        positive_cases += 1;
        let rank = sources
            .iter()
            .position(|source| source_matches(source, expected))
            .map(|position| position + 1);
        if rank == Some(1) {
            top_one += 1;
        }
        if let Some(rank) = rank {
            recalled += 1;
            reciprocal_rank += 1.0 / rank as f64;
        }
        lines.push(format!(
            "case={}\trank={}\tstatus={}\texpected={}\tquery={}",
            index + 1,
            rank.map_or_else(|| "miss".to_owned(), |rank| rank.to_string()),
            status.as_str(),
            expected,
            single_line(query, 240)
        ));
    }
    let total = cases.len();
    let mean_reciprocal_rank = if positive_cases == 0 {
        0.0
    } else {
        reciprocal_rank / positive_cases as f64
    };
    Ok(format!(
        "memory={}\nevaluation={}\nmode={}\ncases={}\npositive_cases={}\nnegative_cases={}\ntop1={}/{}\nrecall_at_{}={}/{}\ncorrect_abstentions={}/{}\noverall_correct={}/{}\nmrr={:.6}\n{}",
        store.path().display(),
        evaluation_path.display(),
        if composed { "composed" } else { "recall" },
        total,
        positive_cases,
        negative_cases,
        top_one,
        positive_cases,
        limit,
        recalled,
        positive_cases,
        correct_abstentions,
        negative_cases,
        top_one + correct_abstentions,
        total,
        mean_reciprocal_rank,
        lines.join("\n")
    ))
}

fn format_recall(query: &str, result: &RecallResult) -> String {
    let mut output = format!(
        "query={}\nstatus={}\nevidence_count={}",
        single_line(query, 240),
        result.status().as_str(),
        result.hits().len()
    );
    append_search_work(&mut output, result.search_work());
    for (index, hit) in result.hits().iter().enumerate() {
        output.push_str(&format!(
            "\n---\nrank={}\nscore={:.6}\nfeedback_factor={:.6}\nsource={}\nlines={}-{}\nmatched_terms={}\ntext={}",
            index + 1,
            hit.score(),
            hit.feedback_factor(),
            hit.source().display(),
            hit.start_line(),
            hit.end_line(),
            hit.matched_terms().join(","),
            single_line(hit.text(), 600)
        ));
    }
    output
}

fn format_reason(query: &str, result: &EvidenceResult) -> String {
    let mut output = format!(
        "query={}\nstatus={}\nmode=extractive_composition\nbundle_count={}",
        single_line(query, 240),
        result.status().as_str(),
        result.bundles().len()
    );
    append_search_work(&mut output, result.search_work());
    for (rank, bundle) in result.bundles().iter().enumerate() {
        let answer = bundle
            .evidence()
            .iter()
            .map(|hit| single_line(hit.text(), 260))
            .collect::<Vec<_>>()
            .join(" | ");
        output.push_str(&format!(
            "\n---\nrank={}\nbundle_score={:.6}\nweighted_coverage={:.6}\nexact_coverage={:.6}\nsource={}\ncovered_terms={}\nmissing_terms={}\nanswer={}\nevidence_count={}",
            rank + 1,
            bundle.score(),
            bundle.weighted_coverage(),
            bundle.exact_coverage(),
            bundle.source().display(),
            bundle.covered_terms().join(","),
            bundle.missing_terms().join(","),
            answer,
            bundle.evidence().len()
        ));
        for (index, hit) in bundle.evidence().iter().enumerate() {
            output.push_str(&format!(
                "\nevidence_{}_score={:.6}\nevidence_{}_feedback_factor={:.6}\nevidence_{}_lines={}-{}\nevidence_{}_text={}",
                index + 1,
                hit.score(),
                index + 1,
                hit.feedback_factor(),
                index + 1,
                hit.start_line(),
                hit.end_line(),
                index + 1,
                single_line(hit.text(), 600)
            ));
        }
    }
    output
}

fn format_investigation(query: &str, result: &InvestigationResult) -> String {
    let mut output = format!(
        "query={}\nstatus={}\nmode=evidence_graph\nanchor_terms={}\nunsupported_terms={}\npath_count={}",
        single_line(query, 240),
        result.status().as_str(),
        result.anchor_terms().join(","),
        result.unsupported_terms().join(","),
        result.paths().len()
    );
    append_search_work(&mut output, result.search_work());
    for (rank, path) in result.paths().iter().enumerate() {
        let answer = path
            .evidence()
            .iter()
            .map(|hit| single_line(hit.text(), 260))
            .collect::<Vec<_>>()
            .join(" | ");
        let sources = path
            .evidence()
            .iter()
            .map(|hit| hit.source().display().to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join("|");
        let source_roles = path
            .evidence_roles()
            .iter()
            .map(|role| role.as_str())
            .collect::<Vec<_>>()
            .join(",");
        output.push_str(&format!(
            "\n---\nrank={}\npath_score={:.6}\nweighted_coverage={:.6}\nexact_coverage={:.6}\ncoherence={:.6}\nsource_coverage={:.6}\nsource_union_coverage={:.6}\nsource_union_exact_coverage={:.6}\ncorroborated_coverage={:.6}\nsource_name_affinity={:.6}\nanchor_source_count={}\nanchor_definition_count={}\ndistinct_source_roles={}\nquery_example_count={}\nsource_roles={}\nsources={}\ncovered_terms={}\nmissing_terms={}\nanswer={}\nevidence_count={}\nlink_count={}",
            rank + 1,
            path.score(),
            path.weighted_coverage(),
            path.exact_coverage(),
            path.coherence(),
            path.source_coverage(),
            path.source_union_coverage(),
            path.source_union_exact_coverage(),
            path.corroborated_coverage(),
            path.source_name_affinity(),
            path.anchor_source_count(),
            path.anchor_definition_count(),
            path.distinct_role_count(),
            path.query_example_count(),
            source_roles,
            sources,
            path.covered_terms().join(","),
            path.missing_terms().join(","),
            answer,
            path.evidence().len(),
            path.links().len()
        ));
        for (index, hit) in path.evidence().iter().enumerate() {
            output.push_str(&format!(
                "\nevidence_{}_score={:.6}\nevidence_{}_role={}\nevidence_{}_source={}\nevidence_{}_lines={}-{}\nevidence_{}_matched_terms={}\nevidence_{}_text={}",
                index + 1,
                hit.score(),
                index + 1,
                path.evidence_roles()[index].as_str(),
                index + 1,
                hit.source().display(),
                index + 1,
                hit.start_line(),
                hit.end_line(),
                index + 1,
                hit.matched_terms().join(","),
                index + 1,
                single_line(hit.text(), 600)
            ));
        }
        for (index, link) in path.links().iter().enumerate() {
            output.push_str(&format!(
                "\nlink_{}_from={}\nlink_{}_to={}\nlink_{}_kind={}\nlink_{}_score={:.6}\nlink_{}_bridge_terms={}",
                index + 1,
                link.from_evidence() + 1,
                index + 1,
                link.to_evidence() + 1,
                index + 1,
                link.kind().as_str(),
                index + 1,
                link.score(),
                index + 1,
                link.bridge_terms().join(",")
            ));
        }
    }
    output
}

fn append_search_work(output: &mut String, work: crate::SearchWork) {
    output.push_str(&format!(
        "\nindexed_documents={}\nscored_documents={}\nposting_visits={}",
        work.indexed_documents(),
        work.scored_documents(),
        work.posting_visits(),
    ));
}

fn single_line(value: &str, maximum_characters: usize) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= maximum_characters {
        return normalized;
    }
    let mut shortened = normalized
        .chars()
        .take(maximum_characters.saturating_sub(3))
        .collect::<String>();
    shortened.push_str("...");
    shortened
}

fn source_matches(source: &std::path::Path, expected: &str) -> bool {
    let source = source
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    expected.split('|').any(|candidate| {
        let candidate = candidate.trim().replace('\\', "/").to_ascii_lowercase();
        !candidate.is_empty() && source.ends_with(&candidate)
    })
}

fn evidence_set_matches(evidence: &[RecallHit], expected: &str) -> bool {
    let sources = evidence
        .iter()
        .map(|hit| hit.source().to_owned())
        .collect::<Vec<_>>();
    source_set_matches(&sources, expected)
}

fn source_set_matches(sources: &[PathBuf], expected: &str) -> bool {
    expected.split('&').all(|clause| {
        let clause = clause.trim();
        !clause.is_empty() && sources.iter().any(|source| source_matches(source, clause))
    })
}

fn parse_evaluation(source: &str) -> Result<Vec<(String, String)>, CliError> {
    let mut cases = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let line_number = index + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((query, expected)) = line.split_once('\t') else {
            return Err(CliError::InvalidEvaluation {
                line: line_number,
                reason: "expected query<TAB>source-suffix",
            });
        };
        if query.trim().is_empty() || expected.trim().is_empty() {
            return Err(CliError::InvalidEvaluation {
                line: line_number,
                reason: "query and source suffix must not be empty",
            });
        }
        cases.push((query.trim().to_owned(), expected.trim().to_owned()));
    }
    if cases.is_empty() {
        return Err(CliError::InvalidEvaluation {
            line: 0,
            reason: "evaluation has no cases",
        });
    }
    Ok(cases)
}

struct SolveOptions {
    task_path: PathBuf,
    artifact_root: PathBuf,
    experience: Option<(PathBuf, String)>,
}

fn parse_solve_arguments(arguments: &[String]) -> Result<SolveOptions, CliError> {
    if arguments.len() < 2 || arguments.first().map(String::as_str) != Some("solve") {
        return Err(CliError::Usage);
    }
    let task_path = PathBuf::from(&arguments[1]);
    let default_root = task_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(".axon-artifacts");
    let mut root = None;
    let mut episodes = None;
    let mut context = None;
    let (pairs, remainder) = arguments[2..].as_chunks::<2>();
    for pair in pairs {
        match pair[0].as_str() {
            "--artifact-dir" if root.is_none() => root = Some(PathBuf::from(&pair[1])),
            "--episodes" if episodes.is_none() => episodes = Some(PathBuf::from(&pair[1])),
            "--experience-context" if context.is_none() => context = Some(pair[1].clone()),
            _ => return Err(CliError::Usage),
        }
    }
    if !remainder.is_empty() || episodes.is_some() != context.is_some() {
        return Err(CliError::Usage);
    }
    Ok(SolveOptions {
        task_path,
        artifact_root: root.unwrap_or(default_root),
        experience: episodes.zip(context),
    })
}

fn parse_memory_only(arguments: &[String], start: usize) -> Result<PathBuf, CliError> {
    if arguments.len() == start {
        return Ok(KnowledgeStore::default_path());
    }
    if arguments.len() == start + 2 && arguments[start] == "--memory" {
        return Ok(PathBuf::from(&arguments[start + 1]));
    }
    Err(CliError::Usage)
}

fn parse_query_options(
    arguments: &[String],
    start: usize,
) -> Result<(String, PathBuf, usize), CliError> {
    let mut query = Vec::new();
    let mut memory = KnowledgeStore::default_path();
    let mut limit = 5_usize;
    let mut index = start;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--memory" if index + 1 < arguments.len() => {
                memory = PathBuf::from(&arguments[index + 1]);
                index += 2;
            }
            "--limit" if index + 1 < arguments.len() => {
                limit = parse_limit(&arguments[index + 1])?;
                index += 2;
            }
            option if option.starts_with("--") => return Err(CliError::Usage),
            value => {
                query.push(value.to_owned());
                index += 1;
            }
        }
    }
    if query.is_empty() {
        return Err(CliError::Usage);
    }
    Ok((query.join(" "), memory, limit))
}

fn parse_teach_options(
    arguments: &[String],
    start: usize,
) -> Result<(String, PathBuf, KnowledgeFeedbackLabel, PathBuf), CliError> {
    let mut query = Vec::new();
    let mut target = None;
    let mut memory = KnowledgeStore::default_path();
    let mut index = start;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--memory" if index + 1 < arguments.len() => {
                memory = PathBuf::from(&arguments[index + 1]);
                index += 2;
            }
            "--relevant" if index + 1 < arguments.len() && target.is_none() => {
                target = Some((
                    PathBuf::from(&arguments[index + 1]),
                    KnowledgeFeedbackLabel::Relevant,
                ));
                index += 2;
            }
            "--irrelevant" if index + 1 < arguments.len() && target.is_none() => {
                target = Some((
                    PathBuf::from(&arguments[index + 1]),
                    KnowledgeFeedbackLabel::Irrelevant,
                ));
                index += 2;
            }
            option if option.starts_with("--") => return Err(CliError::Usage),
            value => {
                query.push(value.to_owned());
                index += 1;
            }
        }
    }
    let Some((source, label)) = target else {
        return Err(CliError::Usage);
    };
    if query.is_empty() {
        return Err(CliError::Usage);
    }
    Ok((query.join(" "), source, label, memory))
}

fn parse_memory_and_limit(
    arguments: &[String],
    start: usize,
) -> Result<(PathBuf, usize), CliError> {
    let mut memory = KnowledgeStore::default_path();
    let mut limit = 5_usize;
    let mut index = start;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--memory" if index + 1 < arguments.len() => {
                memory = PathBuf::from(&arguments[index + 1]);
                index += 2;
            }
            "--limit" if index + 1 < arguments.len() => {
                limit = parse_limit(&arguments[index + 1])?;
                index += 2;
            }
            _ => return Err(CliError::Usage),
        }
    }
    Ok((memory, limit))
}

fn parse_limit(value: &str) -> Result<usize, CliError> {
    let limit = value
        .parse::<usize>()
        .map_err(|_| CliError::InvalidNumber {
            option: "--limit",
            value: value.to_owned(),
        })?;
    if !(1..=100).contains(&limit) {
        return Err(CliError::InvalidNumber {
            option: "--limit",
            value: value.to_owned(),
        });
    }
    Ok(limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluation_parser_requires_tab_separated_ground_truth() {
        assert_eq!(
            parse_evaluation("question\tdocs/answer.md\n# ignored\n")
                .unwrap()
                .len(),
            1
        );
        assert!(parse_evaluation("question without expected source").is_err());
    }

    #[test]
    fn expected_source_matching_is_suffix_based_and_portable() {
        assert!(source_matches(
            std::path::Path::new("C:\\work\\docs\\VISION.md"),
            "docs/VISION.md"
        ));
        assert!(!source_matches(
            std::path::Path::new("C:\\work\\docs\\OTHER.md"),
            "docs/VISION.md"
        ));
    }

    #[test]
    fn coherence_matching_requires_every_source_clause() {
        let evidence = vec![
            PathBuf::from("C:\\work\\docs\\ONE.md"),
            PathBuf::from("C:\\work\\src\\two.rs"),
        ];
        assert!(source_set_matches(
            &evidence,
            "docs/ONE.md&src/two.rs|src/alternative.rs"
        ));
        assert!(!source_set_matches(&evidence, "docs/ONE.md&src/missing.rs"));
    }
}
