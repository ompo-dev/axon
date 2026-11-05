//! Finite correspondences learned from demonstrations; unknown words never become defaults.
use super::{Error, Guard, Pattern, Rule, StructuralSeed, Term};
use serde::Serialize;
use std::collections::BTreeMap;

pub(super) fn infer(rows: &[Term], holes: &[Vec<Term>]) -> Option<Pattern> {
    if !rows
        .iter()
        .all(|t| matches!(t, Term::Symbol(_) | Term::Integer(_)))
    {
        return None;
    }
    for (slot, column) in holes.iter().enumerate() {
        let mut entries = BTreeMap::new();
        let mut consistent = true;
        for (input, output) in column.iter().zip(rows) {
            let Term::Symbol(key) = input else {
                consistent = false;
                break;
            };
            if entries
                .insert(key.clone(), output.clone())
                .is_some_and(|old| old != *output)
            {
                consistent = false;
                break;
            }
        }
        if consistent && (2..=32).contains(&entries.len()) {
            return Some(Pattern::Lookup {
                slot,
                entries: entries.into_iter().collect(),
            });
        }
    }
    None
}

fn same(a: &impl Serialize, b: &impl Serialize) -> bool {
    serde_json::to_vec(a)
        .ok()
        .zip(serde_json::to_vec(b).ok())
        .is_some_and(|(a, b)| a == b)
}
fn differences(a: &Pattern, b: &Pattern, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) -> bool {
    if same(a, b) {
        return true;
    }
    match (a, b) {
        (Pattern::Literal(Term::Symbol(_)), Pattern::Literal(Term::Symbol(_))) => {
            out.push(path.clone());
            true
        }
        (Pattern::Tree(a), Pattern::Tree(b)) if a.len() == b.len() => {
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                path.push(i);
                if !differences(a, b, path, out) {
                    return false;
                }
                path.pop();
            }
            true
        }
        _ => false,
    }
}
pub(super) fn disjoint_alternatives(a: &Pattern, b: &Pattern) -> bool {
    let mut paths = Vec::new();
    differences(a, b, &mut Vec::new(), &mut paths) && !paths.is_empty()
}
fn at<'a>(p: &'a Pattern, path: &[usize]) -> Option<&'a Pattern> {
    let mut p = p;
    for i in path {
        let Pattern::Tree(children) = p else {
            return None;
        };
        p = children.get(*i)?;
    }
    Some(p)
}
fn replace(p: &mut Pattern, path: &[usize], value: Pattern) -> Result<(), Error> {
    let mut p = p;
    for i in path {
        let Pattern::Tree(children) = p else {
            return Err(Error::Unsupported);
        };
        p = children.get_mut(*i).ok_or(Error::Unsupported)?;
    }
    *p = value;
    Ok(())
}
fn symbol(p: &Pattern) -> Result<String, Error> {
    match p {
        Pattern::Literal(Term::Symbol(s)) => Ok(s.clone()),
        _ => Err(Error::Unsupported),
    }
}
impl Rule {
    // One differing input literal and at most one differing output literal.
    // No cross product, wildcard vocabulary, procedural rewrite or changed precedence.
    fn factor(&self, other: &Self) -> Result<Option<Self>, Error> {
        if self.holes != other.holes
            || self.holes >= 64
            || self.guards.len() >= 8
            || !same(&self.guards, &other.guards)
            || !self.algorithm_programs().is_empty()
            || !other.algorithm_programs().is_empty()
            || !self.derived_programs().is_empty()
            || !other.derived_programs().is_empty()
            || !self.dependencies.is_empty()
            || !other.dependencies.is_empty()
        {
            return Ok(None);
        }
        let (mut input, mut output) = (Vec::new(), Vec::new());
        if !differences(&self.input, &other.input, &mut vec![], &mut input)
            || !differences(&self.output, &other.output, &mut vec![], &mut output)
            || input.len() != 1
            || output.len() > 1
            || input[0].last() == Some(&0)
            || output.iter().any(|p| p.last() == Some(&0))
        {
            return Ok(None);
        }
        let a = symbol(at(&self.input, &input[0]).ok_or(Error::Unsupported)?)?;
        let b = symbol(at(&other.input, &input[0]).ok_or(Error::Unsupported)?)?;
        let mut candidate = self.clone();
        let slot = candidate.holes;
        candidate.holes += 1;
        replace(&mut candidate.input, &input[0], Pattern::Hole(slot, 1))?;
        let mut values = vec![a.clone(), b.clone()];
        values.sort();
        candidate.guards.push(Guard::Symbols {
            path: input[0].clone(),
            values,
        });
        if let Some(path) = output.first() {
            let mut entries = vec![
                (
                    a,
                    Term::Symbol(symbol(at(&self.output, path).ok_or(Error::Unsupported)?)?),
                ),
                (
                    b,
                    Term::Symbol(symbol(at(&other.output, path).ok_or(Error::Unsupported)?)?),
                ),
            ];
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            replace(
                &mut candidate.output,
                path,
                Pattern::Lookup { slot, entries },
            )?;
        }
        candidate.teaching_checksum = crate::live_store::checksum(
            &serde_json::to_vec(&(self, other)).map_err(|e| Error::Storage(e.to_string()))?,
        );
        candidate.validate()?;
        Ok(Some(candidate))
    }
}

#[derive(Debug, Serialize)]
pub struct Consolidation {
    pub kept: String,
    pub removed: String,
    pub source_checksum: u64,
    pub before_bytes: usize,
    pub after_bytes: usize,
}
impl StructuralSeed {
    /// Exact input union with a shared output; no Cartesian product or new vocabulary.
    pub fn consolidate_alternatives(&mut self) -> Result<Vec<Consolidation>, Error> {
        let mut candidate = self.clone();
        let mut receipts = Vec::new();
        loop {
            let mut proposal = None;
            'pairs: for i in 0..candidate.rules.len() {
                let a = &candidate.rules[i];
                for j in i + 1..candidate.rules.len() {
                    let b = &candidate.rules[j];
                    if a.holes != b.holes
                        || a.specificity() != b.specificity()
                        || !same(&a.output, &b.output)
                        || !same(&a.guards, &b.guards)
                        || !a.dependencies.is_empty()
                        || !b.dependencies.is_empty()
                        || !a.algorithm_programs().is_empty()
                        || !a.derived_programs().is_empty()
                        || candidate.rules.iter().any(|r| {
                            r.dependencies
                                .iter()
                                .any(|d| d.rule == a.id || d.rule == b.id)
                        })
                    {
                        continue;
                    }
                    let mut branches = match &a.input {
                        Pattern::Choice(v) => v.clone(),
                        p => vec![p.clone()],
                    };
                    let other = match &b.input {
                        Pattern::Choice(v) => v.as_slice(),
                        p => std::slice::from_ref(p),
                    };
                    for p in other {
                        if !branches.iter().any(|old| same(old, p)) {
                            branches.push(p.clone());
                        }
                    }
                    if !(2..=8).contains(&branches.len()) {
                        continue;
                    }
                    let mut rule = a.clone();
                    rule.input = Pattern::Choice(branches);
                    rule.teaching_checksum = crate::live_store::checksum(
                        &serde_json::to_vec(&(a, b)).map_err(|e| Error::Storage(e.to_string()))?,
                    );
                    if rule.validate().is_err() {
                        continue;
                    }
                    proposal = Some((i, j, rule));
                    break 'pairs;
                }
            }
            let Some((i, j, rule)) = proposal else {
                break;
            };
            let before_bytes = candidate.to_bytes()?.len();
            let removed = candidate.rules.remove(j);
            candidate.rules[i] = rule;
            receipts.push(Consolidation {
                kept: candidate.rules[i].id.clone(),
                removed: removed.id,
                source_checksum: candidate.rules[i].teaching_checksum,
                before_bytes,
                after_bytes: candidate.to_bytes()?.len(),
            });
        }
        candidate.to_bytes()?;
        *self = candidate;
        Ok(receipts)
    }
    /// Conservative union of two already learned semantic rules. Opt-in and atomic.
    pub fn consolidate_language(&mut self) -> Result<Vec<Consolidation>, Error> {
        let mut candidate = self.clone();
        let mut receipts = Vec::new();
        loop {
            let mut proposal = None;
            'pairs: for i in 0..candidate.rules.len() {
                for j in i + 1..candidate.rules.len() {
                    if let Some(rule) = candidate.rules[i].factor(&candidate.rules[j])? {
                        proposal = Some((i, j, rule));
                        break 'pairs;
                    }
                }
            }
            let Some((i, j, rule)) = proposal else {
                break;
            };
            let before_bytes = candidate.to_bytes()?.len();
            let removed = candidate.rules.remove(j);
            candidate.rules[i] = rule;
            let after_bytes = candidate.to_bytes()?.len();
            receipts.push(Consolidation {
                kept: candidate.rules[i].id.clone(),
                removed: removed.id,
                source_checksum: candidate.rules[i].teaching_checksum,
                before_bytes,
                after_bytes,
            });
        }
        candidate.to_bytes()?;
        *self = candidate;
        Ok(receipts)
    }
}
