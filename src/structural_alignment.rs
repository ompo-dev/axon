//! Vocabulary-free contrast representation. Alignment is evidence, never a truth label.
use super::{Error, Term, agent::utterance};
use std::collections::BTreeMap;

const MAX_ALTERNATIVES: usize = 8;
const MAX_WORK: usize = 32768;
fn words(text: &str) -> Result<Vec<Term>, Error> {
    let Term::Tree(mut words) = utterance(text)? else {
        return Err(Error::Unsupported);
    };
    words.remove(0);
    if words.iter().any(|word| matches!(word, Term::Tree(_))) {
        return Err(Error::Unsupported);
    }
    Ok(words)
}
fn tokens(words: &[Term]) -> Term {
    if words.is_empty() {
        return Term::Symbol("empty".into());
    }
    let mut row = vec![Term::Symbol("tokens".into())];
    row.extend_from_slice(words);
    Term::Tree(row)
}
fn representation(a: &[Term], b: &[Term], links: &[(usize, usize)]) -> Result<Term, Error> {
    let mut steps = vec![Term::Symbol("steps".into())];
    let (mut left, mut right, mut copied) = (0, 0, 0);
    for &(i, j) in links.iter().chain(std::iter::once(&(a.len(), b.len()))) {
        if left < i || right < j {
            if copied > 0 {
                steps.push(Term::Tree(vec![
                    Term::Symbol("copy".into()),
                    Term::Integer(copied),
                ]));
                copied = 0;
            }
            steps.push(Term::Tree(vec![
                Term::Symbol("replace".into()),
                tokens(&a[left..i]),
                tokens(&b[right..j]),
            ]));
        }
        if i < a.len() {
            copied += 1;
        }
        left = i + 1;
        right = j + 1;
    }
    if copied > 0 {
        steps.push(Term::Tree(vec![
            Term::Symbol("copy".into()),
            Term::Integer(copied),
        ]));
    }
    let result = Term::Tree(vec![
        Term::Symbol("contrast-v1".into()),
        Term::Integer(a.len() as i64),
        Term::Integer(b.len() as i64),
        Term::Tree(steps),
    ]);
    result.validate()?;
    Ok(result)
}
struct Alignments<'a> {
    a: &'a [Term],
    b: &'a [Term],
    dp: Vec<Vec<usize>>,
    work: usize,
    outputs: BTreeMap<Vec<u8>, Term>,
}
impl Alignments<'_> {
    fn walk(
        &mut self,
        left: usize,
        right: usize,
        remaining: usize,
        links: &mut Vec<(usize, usize)>,
    ) -> Result<(), Error> {
        if remaining == 0 {
            let output = representation(self.a, self.b, links)?;
            let key = serde_json::to_vec(&output).map_err(|e| Error::Storage(e.to_string()))?;
            self.outputs.insert(key, output);
            if self.outputs.len() > MAX_ALTERNATIVES {
                return Err(Error::Ambiguous);
            }
            return Ok(());
        }
        for i in left..self.a.len() {
            for j in right..self.b.len() {
                self.work += 1;
                if self.work > MAX_WORK {
                    return Err(Error::Budget);
                }
                if self.a[i] == self.b[j]
                    && self.dp[i][j] == remaining
                    && self.dp[i + 1][j + 1] + 1 == remaining
                {
                    links.push((i, j));
                    self.walk(i + 1, j + 1, remaining - 1, links)?;
                    links.pop();
                }
            }
        }
        Ok(())
    }
}
pub fn alternatives(premise: &str, hypothesis: &str) -> Result<Vec<Term>, Error> {
    let (a, b) = (words(premise)?, words(hypothesis)?);
    let mut dp = vec![vec![0; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            dp[i][j] = if a[i] == b[j] {
                1 + dp[i + 1][j + 1]
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let remaining = dp[0][0];
    let mut search = Alignments {
        a: &a,
        b: &b,
        dp,
        work: 0,
        outputs: BTreeMap::new(),
    };
    search.walk(0, 0, remaining, &mut Vec::new())?;
    Ok(search.outputs.into_values().collect())
}
pub fn unique(premise: &str, hypothesis: &str) -> Result<Term, Error> {
    let mut outputs = alternatives(premise, hypothesis)?;
    if outputs.len() != 1 {
        return Err(Error::Ambiguous);
    }
    Ok(outputs.remove(0))
}
/// Quotient out the lengths of unchanged runs, but preserve edits and their order.
pub fn topology(premise: &str, hypothesis: &str) -> Result<Term, Error> {
    let mut normalized = BTreeMap::new();
    for contrast in alternatives(premise, hypothesis)? {
        let Term::Tree(mut fields) = contrast else {
            return Err(Error::Unsupported);
        };
        let Term::Tree(mut steps) = fields.remove(3) else {
            return Err(Error::Unsupported);
        };
        for step in &mut steps[1..] {
            if matches!(step,Term::Tree(v) if v[0]==Term::Symbol("copy".into())) {
                *step = Term::Symbol("copy".into());
            }
        }
        let output = Term::Tree(steps);
        output.validate()?;
        let key = serde_json::to_vec(&output).map_err(|e| Error::Storage(e.to_string()))?;
        normalized.insert(key, output);
    }
    if normalized.len() != 1 {
        return Err(Error::Ambiguous);
    }
    normalized.into_values().next().ok_or(Error::Unsupported)
}
