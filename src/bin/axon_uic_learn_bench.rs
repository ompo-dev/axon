use axon_uic::{StructurePrior, default_learning_curriculum, train_structure_prior};

const DEFAULT_EPOCHS: usize = 8;
const MAX_EPOCHS: usize = 1_000;

fn main() {
    match parse_epochs().and_then(run) {
        Ok(()) => {}
        Err(message) => {
            eprintln!("learn bench failed: {message}");
            eprintln!("usage: axon-uic-learn-bench [--epochs 1..1000]");
            std::process::exit(2);
        }
    }
}

fn parse_epochs() -> Result<usize, String> {
    let mut epochs = DEFAULT_EPOCHS;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            return Err("help requested".to_owned());
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--epochs" => epochs = parse_bounded(&value, 1, MAX_EPOCHS, "--epochs")?,
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    Ok(epochs)
}

fn parse_bounded(value: &str, minimum: usize, maximum: usize, flag: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|_| format!("invalid integer for {flag}: {value}"))?;
    (minimum..=maximum)
        .contains(&value)
        .then_some(value)
        .ok_or_else(|| format!("{flag} must be between {minimum} and {maximum}"))
}

fn run(epochs: usize) -> Result<(), String> {
    let curriculum = default_learning_curriculum();
    let mut train = Vec::with_capacity(curriculum.len() * epochs);
    for _ in 0..epochs {
        train.extend(curriculum.iter().copied());
    }
    let mut prior = StructurePrior::new();
    let summary = train_structure_prior(&mut prior, &train, &curriculum);

    println!("# AXON-UIC LearnBench");
    println!("prior: tiny online linear ranker; no neural network, no hidden dataset");
    println!("epochs: {epochs}; evaluation tasks: {}", summary.tasks());
    println!("\n| Metric | Result |");
    println!("|---|---:|");
    println!(
        "| Candidate attempts before experience | {} |",
        summary.attempts_before()
    );
    println!(
        "| Candidate attempts after experience | {} |",
        summary.attempts_after()
    );
    println!(
        "| Search reduction | {:.2}% |",
        summary.search_reduction_basis_points() as f64 / 100.0
    );
    println!("\n| Problem | Top candidate after training | Accepted | Attempts |");
    println!("|---|---|---|---:|");
    for problem in &curriculum {
        let ranking = prior.rank(problem.features());
        let attempts = prior.attempts_for(*problem);
        println!(
            "| {} | {} | {} | {} |",
            problem.name(),
            ranking[0].as_str(),
            problem.accepted().as_str(),
            attempts
        );
        if ranking[0] != problem.accepted() {
            return Err(format!("learned prior missed {}", problem.name()));
        }
    }
    println!(
        "\nLimit: this is the first measurable learning loop. It learns candidate ordering, not language, perception, autonomy, or general discovery."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_accepts_bounded_epochs() {
        assert_eq!(parse_bounded("1", 1, 2, "--epochs"), Ok(1));
        assert!(parse_bounded("0", 1, 2, "--epochs").is_err());
        assert!(parse_bounded("3", 1, 2, "--epochs").is_err());
    }
}
