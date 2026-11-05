//! Bounded composition of learned semantic fragments, not a built-in dictionary.
use super::{Error, Pattern, Prediction, Rule, StructuralSeed, Term, term_nodes};

pub const DEFAULT_WORK: usize = 250_000;
const MAX_MEANINGS: usize = 16;
const MAX_RETAINED_NODES: usize = 32_768;

#[derive(Clone)]
struct Meaning {
    output: Term,
    rules: Vec<String>,
}
type Chart = Vec<Vec<Vec<Meaning>>>;

#[derive(Debug)]
pub struct Composition {
    pub result: Result<Vec<Prediction>, Error>,
    pub visits: usize,
}

struct Work {
    used: usize,
    limit: usize,
}
impl Work {
    fn charge(&mut self, n: usize) -> Result<(), Error> {
        let next = self.used.checked_add(n).ok_or(Error::Budget)?;
        if next > self.limit {
            return Err(Error::Budget);
        }
        self.used = next;
        Ok(())
    }
}
fn purely_structural(p: &Pattern) -> bool {
    match p {
        Pattern::Calculate(_) | Pattern::Algorithm(_) => false,
        Pattern::Tree(v) | Pattern::Phrase(v) => v.iter().all(purely_structural),
        _ => true,
    }
}

struct Expansion<'a> {
    rule: &'a Rule,
    chart: &'a Chart,
    existing: &'a [Meaning],
    proposed: &'a mut Vec<Meaning>,
    work: &'a mut Work,
    words: &'a [Term],
    end: usize,
}
impl Expansion<'_> {
    fn walk(
        &mut self,
        pieces: &[Pattern],
        start: usize,
        values: Vec<Option<Term>>,
        selected: &mut Vec<Meaning>,
    ) -> Result<(), Error> {
        self.work.charge(1)?;
        let Some((first, rest)) = pieces.split_first() else {
            if start != self.end {
                return Ok(());
            }
            let mut input = vec![Term::Symbol("goal".into())];
            input.extend(selected.iter().map(|n| n.output.clone()));
            let input = Term::Tree(input);
            input.validate()?;
            self.work.charge(term_nodes(&input))?;
            let Some(output) = self.rule.checked_apply(&input)? else {
                return Ok(());
            };
            output.validate()?;
            if self
                .existing
                .iter()
                .chain(self.proposed.iter())
                .any(|m| m.output == output)
            {
                return Ok(());
            }
            if self.existing.len() + self.proposed.len() >= MAX_MEANINGS {
                return Err(Error::Budget);
            }
            let mut rules = selected
                .iter()
                .flat_map(|n| n.rules.iter().cloned())
                .collect::<Vec<_>>();
            rules.push(self.rule.id().to_owned());
            if rules.len() > 128 {
                return Err(Error::Budget);
            }
            self.proposed.push(Meaning { output, rules });
            return Ok(());
        };
        if self.end.saturating_sub(start) < pieces.len() {
            return Ok(());
        }
        for next in start + 1..=self.end - rest.len() {
            if let Pattern::Span(slot) = first {
                self.work.charge(next - start)?;
                let text = self.words[start..next]
                    .iter()
                    .map(|t| match t {
                        Term::Symbol(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();
                let Some(text) = text else { continue };
                let text = text.join(" ");
                if text.len() > 256 {
                    continue;
                }
                let output = Term::Symbol(text);
                let mut bindings = values.clone();
                if bindings[*slot].as_ref().is_some_and(|old| *old != output) {
                    continue;
                }
                bindings[*slot] = Some(output.clone());
                selected.push(Meaning {
                    output,
                    rules: vec![],
                });
                self.walk(rest, next, bindings, selected)?;
                selected.pop();
                continue;
            }
            for index in 0..self.chart[start][next].len() {
                let candidate = self.chart[start][next][index].clone();
                self.work.charge(term_nodes(&candidate.output))?;
                let mut bindings = values.clone();
                if !first.bind(&candidate.output, &mut bindings)? {
                    continue;
                }
                selected.push(candidate.clone());
                self.walk(rest, next, bindings, selected)?;
                selected.pop();
            }
        }
        Ok(())
    }
}

impl StructuralSeed {
    /// Enumerate complete meanings. Saturation or any quota overflow refuses;
    /// never silently prune alternatives and then advertise a unique meaning.
    pub fn composed_interpretations(
        &self,
        text: &str,
        max_work: usize,
    ) -> Result<Vec<Prediction>, Error> {
        self.measure_composition(text, max_work).result
    }
    /// Work is retained even when interpretation is unknown, ambiguous or capped.
    pub fn measure_composition(&self, text: &str, max_work: usize) -> Composition {
        let mut work = Work {
            used: 0,
            limit: max_work,
        };
        let result = self.compose(text, &mut work);
        Composition {
            result,
            visits: work.used,
        }
    }
    fn compose(&self, text: &str, work: &mut Work) -> Result<Vec<Prediction>, Error> {
        let max_work = work.limit;
        if !(1..=1_000_000).contains(&max_work) {
            return Err(Error::Invalid("language work quota"));
        }
        let Term::Tree(input) = super::agent::utterance(text)? else {
            unreachable!()
        };
        let words = &input[1..];
        let n = words.len();
        if n == 0 || n > 31 {
            return Err(Error::Invalid("language word quota"));
        }
        let rules = self.rules.iter().flat_map(|r| {
            let branches = match &r.input { Pattern::Choice(v) => v.as_slice(), p => std::slice::from_ref(p) };
            branches.iter().filter_map(move |input| {
                let (Pattern::Tree(parts) | Pattern::Phrase(parts)) = input else { return None };
                (matches!(parts.first(), Some(Pattern::Literal(Term::Symbol(head))) if head == "goal")
                    && purely_structural(&r.output)).then_some((r, &parts[1..]))
            })
        }).collect::<Vec<_>>();
        let mut chart: Chart = vec![vec![Vec::new(); n + 1]; n + 1];
        let mut retained = 0;
        for (i, word) in words.iter().enumerate() {
            retained += term_nodes(word);
            chart[i][i + 1].push(Meaning {
                output: word.clone(),
                rules: vec![],
            });
        }
        for length in 1..=n {
            for start in 0..=n - length {
                let end = start + length;
                loop {
                    let mut proposed = Vec::new();
                    for (rule, pieces) in &rules {
                        work.charge(1)?;
                        if pieces.len() > length {
                            continue;
                        }
                        Expansion {
                            rule,
                            chart: &chart,
                            existing: &chart[start][end],
                            proposed: &mut proposed,
                            work,
                            words,
                            end,
                        }
                        .walk(
                            pieces,
                            start,
                            vec![None; rule.holes],
                            &mut vec![],
                        )?;
                    }
                    if proposed.is_empty() {
                        break;
                    }
                    for candidate in &proposed {
                        retained += term_nodes(&candidate.output);
                        if retained > MAX_RETAINED_NODES {
                            return Err(Error::Budget);
                        }
                    }
                    chart[start][end].extend(proposed);
                }
            }
        }
        let meanings = chart[0][n]
            .iter()
            .filter(|m| !m.rules.is_empty())
            .map(|m| Prediction {
                output: m.output.clone(),
                rules: m.rules.clone(),
                visits: work.used,
            })
            .collect::<Vec<_>>();
        if meanings.is_empty() {
            return Err(Error::Unsupported);
        }
        Ok(meanings)
    }
}
