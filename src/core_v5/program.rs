//! ProgramCells: programas pequenos, interpretáveis e seguros.

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProgramStatus {
    Candidate,
    Verified,
    Compiled,
    Retired,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProgramInstruction {
    RepeatPair { first: String, second: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramCell {
    pub id: String,
    pub instruction: ProgramInstruction,
    pub input_schema: String,
    pub output_schema: String,
    pub provenance: Vec<String>,
    pub status: ProgramStatus,
}

impl ProgramCell {
    pub fn from_repeating_pair(
        id: impl Into<String>,
        first: impl Into<String>,
        second: impl Into<String>,
        _observed_repetitions: usize,
    ) -> Self {
        Self {
            id: id.into(),
            instruction: ProgramInstruction::RepeatPair {
                first: first.into(),
                second: second.into(),
            },
            input_schema: "repetitions: positive-integer".to_string(),
            output_schema: "sequence".to_string(),
            provenance: Vec::new(),
            status: ProgramStatus::Candidate,
        }
    }

    pub fn execute(&self, repetitions: usize) -> Vec<String> {
        match &self.instruction {
            ProgramInstruction::RepeatPair { first, second } => (0..repetitions)
                .flat_map(|_| [first.clone(), second.clone()])
                .collect(),
        }
    }

    pub fn verify(mut self, provenance: impl Into<String>) -> Self {
        self.provenance.push(provenance.into());
        self.status = ProgramStatus::Verified;
        self
    }

    pub fn compile(mut self) -> Option<Self> {
        if self.status != ProgramStatus::Verified {
            return None;
        }
        self.status = ProgramStatus::Compiled;
        Some(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InductionResult {
    pub cell: ProgramCell,
    pub raw_description_length: usize,
    pub program_description_length: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AbstractionCompiler {
    min_repetitions: usize,
}

impl AbstractionCompiler {
    pub const fn new(min_repetitions: usize) -> Self {
        Self { min_repetitions }
    }

    pub fn induce_repeating_pair(
        &self,
        id: impl Into<String>,
        train: &[String],
        holdout: &[String],
    ) -> Option<InductionResult> {
        if self.min_repetitions == 0
            || train.len() < self.min_repetitions.saturating_mul(2)
            || !train.len().is_multiple_of(2)
            || !holdout.len().is_multiple_of(2)
        {
            return None;
        }
        let first = train.first()?.clone();
        let second = train.get(1)?.clone();
        if train
            .chunks_exact(2)
            .chain(holdout.chunks_exact(2))
            .any(|pair| pair != [first.as_str(), second.as_str()])
        {
            return None;
        }
        let raw_description_length = train.len();
        let program_description_length = 3;
        if program_description_length >= raw_description_length {
            return None;
        }
        let repetitions = train.len() / 2;
        let cell = ProgramCell::from_repeating_pair(id, first, second, repetitions)
            .verify("compression-and-holdout");
        Some(InductionResult {
            cell,
            raw_description_length,
            program_description_length,
        })
    }
}

impl Default for AbstractionCompiler {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn compiler_promotes_compressive_pattern_only_after_holdout_verifies_it() {
        let result = AbstractionCompiler::default()
            .induce_repeating_pair(
                "alternation",
                &tokens(&["A", "B", "A", "B", "A", "B"]),
                &tokens(&["A", "B", "A", "B"]),
            )
            .expect("the repeat has compression and holdout support");

        assert!(result.program_description_length < result.raw_description_length);
        assert_eq!(result.cell.execute(2), tokens(&["A", "B", "A", "B"]));
        assert_eq!(result.cell.status, ProgramStatus::Verified);
        assert_eq!(
            result.cell.clone().compile().unwrap().status,
            ProgramStatus::Compiled
        );
    }

    #[test]
    fn compiler_rejects_spurious_repetition() {
        assert!(
            AbstractionCompiler::default()
                .induce_repeating_pair(
                    "spurious",
                    &tokens(&["A", "B", "A", "C", "A", "B"]),
                    &tokens(&["A", "B"])
                )
                .is_none()
        );
    }
}
