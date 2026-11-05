//! Demonstration-derived text frames joined to sense relations and learned numeric programs.
use crate::{
    TextBudget, TextExample, TextSkill, TextValuePrediction,
    lexicon::{Entailment, LexiconView, MAX_VISITS, Resolution},
    numeric::NumericSkill,
};
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct LanguageFrame {
    word: TextSkill,
    values: TextSkill,
    literals: Vec<Option<String>>,
}
impl LanguageFrame {
    /// Each demonstration supplies (sentence, referred word, ordered numeric arguments).
    /// Literal scaffolding is inferred, not a grammar for unrestricted natural language.
    pub fn learn(examples: &[(String, String, String)]) -> Result<Self, String> {
        Self::learn_with_budget(examples, TextBudget::default())
    }
    pub fn learn_with_budget(
        examples: &[(String, String, String)],
        budget: TextBudget,
    ) -> Result<Self, String> {
        if !(3..=32).contains(&examples.len()) {
            return Err("frame needs 3..32 demonstrations".into());
        }
        let literals = anchors(examples)?;
        let learn = |name, output: usize| {
            TextSkill::learn(
                name,
                "language/frame-v1",
                examples
                    .iter()
                    .map(|e| TextExample::new(&e.0, if output == 1 { &e.1 } else { &e.2 }))
                    .collect(),
                budget,
            )
            .map_err(|e| e.to_string())
        };
        Ok(Self {
            word: learn("word", 1)?,
            values: learn("values", 2)?,
            literals,
        })
    }
    pub fn accepts(&self, input: &str) -> bool {
        let row = tokens(input);
        input.len() <= 1024
            && !input.chars().any(char::is_control)
            && row.len() == self.literals.len()
            && self
                .literals
                .iter()
                .zip(row)
                .all(|(expected, actual)| expected.as_deref().is_none_or(|e| e == actual))
    }
    pub fn read_values(&self, input: &str) -> Result<Vec<f64>, Refusal> {
        if !self.accepts(input) {
            return Err(Refusal::Frame);
        }
        let TextValuePrediction::Agreed(output) = self
            .values
            .predict_value(self.values.context(), input)
            .map_err(|_| Refusal::Parser)?
        else {
            return Err(Refusal::Parser);
        };
        if output.len() > 256 {
            return Err(Refusal::Arguments);
        }
        let values = output
            .split(',')
            .map(|v| v.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| Refusal::Arguments)?;
        if values.is_empty() || values.len() > 4 || values.iter().any(|v| !v.is_finite()) {
            return Err(Refusal::Arguments);
        }
        Ok(values)
    }
    pub fn read_word(&self, input: &str) -> Result<String, Refusal> {
        if !self.accepts(input) {
            return Err(Refusal::Frame);
        }
        match self
            .word
            .predict_value(self.word.context(), input)
            .map_err(|_| Refusal::Parser)?
        {
            TextValuePrediction::Agreed(word) => Ok(word),
            _ => Err(Refusal::Parser),
        }
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut out = b"AXLF0001".to_vec();
        for skill in [&self.word, &self.values] {
            let bytes = skill.to_bytes().map_err(|e| e.to_string())?;
            out.extend((bytes.len() as u32).to_le_bytes());
            out.extend(bytes);
        }
        out.extend(crate::live_store::checksum(&out).to_le_bytes());
        Ok(out)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 24 || bytes.len() > 262144 || &bytes[..8] != b"AXLF0001" {
            return Err("frame header or size".into());
        }
        let body = bytes.len() - 8;
        if crate::live_store::checksum(&bytes[..body])
            != u64::from_le_bytes(bytes[body..].try_into().expect("width"))
        {
            return Err("frame checksum".into());
        }
        let mut cursor = 8;
        let mut read = || -> Result<TextSkill, String> {
            let length = bytes.get(cursor..cursor + 4).ok_or("frame truncation")?;
            cursor += 4;
            let length = u32::from_le_bytes(length.try_into().expect("width")) as usize;
            if length > body.saturating_sub(cursor) {
                return Err("frame payload".into());
            }
            let skill = TextSkill::from_bytes(&bytes[cursor..cursor + length])
                .map_err(|e| e.to_string())?;
            cursor += length;
            Ok(skill)
        };
        let word = read()?;
        let values = read()?;
        if cursor != body
            || word.context() != "language/frame-v1"
            || values.context() != word.context()
            || word.examples().len() != values.examples().len()
            || !(3..=32).contains(&word.examples().len())
        {
            return Err("frame contract".into());
        }
        let examples = word
            .examples()
            .iter()
            .zip(values.examples())
            .map(|(w, v)| {
                if w.input != v.input {
                    return Err("frame demonstration mismatch".to_string());
                }
                Ok((w.input.clone(), w.output.clone(), v.output.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let literals = anchors(&examples)?;
        Ok(Self {
            word,
            values,
            literals,
        })
    }
    pub fn identities(&self) -> (u64, u64) {
        (self.word.fingerprint(), self.values.fingerprint())
    }
    pub fn search_work(&self) -> u64 {
        self.word.program().search_work() + self.values.program().search_work()
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.word.retained_bytes() - std::mem::size_of::<TextSkill>()
            + self.values.retained_bytes()
            - std::mem::size_of::<TextSkill>()
            + self.literals.capacity() * std::mem::size_of::<Option<String>>()
            + self
                .literals
                .iter()
                .flatten()
                .map(String::capacity)
                .sum::<usize>()
    }
    pub fn save_parsers(
        &self,
        root: &std::path::Path,
        name: &str,
    ) -> Result<(), crate::TextSkillError> {
        self.word
            .save_new(root.join(format!("{name}-word.axtext")))?;
        self.values
            .save_new(root.join(format!("{name}-values.axtext")))?;
        Ok(())
    }
}
fn tokens(input: &str) -> Vec<&str> {
    input.split(' ').collect()
}
fn anchors(examples: &[(String, String, String)]) -> Result<Vec<Option<String>>, String> {
    let mut literals = tokens(&examples[0].0)
        .iter()
        .map(|s| Some((*s).to_owned()))
        .collect::<Vec<_>>();
    if literals.is_empty() || literals.len() > 64 {
        return Err("frame token limit".into());
    }
    for (input, _, _) in examples {
        let row = tokens(input);
        if input.len() > 1024 || input.chars().any(char::is_control) || row.len() != literals.len()
        {
            return Err("frame shape".into());
        }
        for (slot, value) in literals.iter_mut().zip(row) {
            if slot.as_deref().is_some_and(|s| s != value) {
                *slot = None;
            }
        }
    }
    if literals.iter().all(Option::is_none) {
        return Err("frame anchors".into());
    }
    Ok(literals)
}

pub struct GroundedTool {
    pub sense: String,
    pub skill: NumericSkill,
}
#[derive(Clone, Debug, PartialEq)]
pub struct GroundedAnswer {
    pub value: f64,
    pub word: String,
    pub sense: String,
    pub evidence: Vec<u64>,
    pub tool: u64,
    pub parsers: (u64, u64),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Refusal {
    Frame,
    Parser,
    WordUnknown,
    WordAmbiguous,
    Conflict,
    Ungrounded,
    ToolAmbiguous,
    Budget,
    Arguments,
    Domain,
}

/// No tool is selected by its name or a keyword heuristic. Bindings are taught explicitly.
pub fn answer(
    view: &LexiconView,
    frames: &[LanguageFrame],
    tools: &[GroundedTool],
    input: &str,
) -> Result<GroundedAnswer, Refusal> {
    if frames.len() > 4 || tools.len() > 16 {
        return Err(Refusal::Budget);
    }
    let candidates = frames
        .iter()
        .filter(|f| f.accepts(input))
        .collect::<Vec<_>>();
    if candidates.len() != 1 {
        return Err(Refusal::Frame);
    }
    let frame = candidates[0];
    let word = match frame
        .word
        .predict_value(frame.word.context(), input)
        .map_err(|_| Refusal::Parser)?
    {
        TextValuePrediction::Agreed(output) => output,
        _ => return Err(Refusal::Parser),
    };
    let sense = match view.resolve(&word) {
        Resolution::Unique(s) => s,
        Resolution::Unknown => return Err(Refusal::WordUnknown),
        Resolution::Ambiguous(_) => return Err(Refusal::WordAmbiguous),
        Resolution::Conflict => return Err(Refusal::Conflict),
    };
    let mut matches = Vec::new();
    let mut blocked = false;
    for tool in tools {
        match view.classify(&sense, &tool.sense, MAX_VISITS) {
            Entailment::Supported(proof) => matches.push((tool, proof)),
            Entailment::Conflict => blocked = true,
            Entailment::Budget => return Err(Refusal::Budget),
            _ => {}
        }
    }
    if blocked {
        return Err(Refusal::Conflict);
    }
    if matches.len() > 1 {
        return Err(Refusal::ToolAmbiguous);
    }
    let Some((tool, proof)) = matches.pop() else {
        return Err(Refusal::Ungrounded);
    };
    let values = match frame
        .values
        .predict_value(frame.values.context(), input)
        .map_err(|_| Refusal::Parser)?
    {
        TextValuePrediction::Agreed(output) => output,
        _ => return Err(Refusal::Parser),
    };
    if values.len() > 256 {
        return Err(Refusal::Arguments);
    }
    let arguments = values
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Refusal::Arguments)?;
    if arguments.len() != tool.skill.program().arity() || arguments.iter().any(|v| !v.is_finite()) {
        return Err(Refusal::Arguments);
    }
    let value = tool
        .skill
        .program()
        .compile_execution()
        .evaluate(&arguments)
        .map_err(|_| Refusal::Domain)?;
    let evidence = view
        .word_evidence(&word, &sense)
        .into_iter()
        .chain(proof)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(GroundedAnswer {
        value,
        word,
        sense,
        evidence,
        tool: tool.skill.fingerprint(),
        parsers: frame.identities(),
    })
}
