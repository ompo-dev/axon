//! Learned local anchors. Vocabulary and positional window are explicit bounded priors.
use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Anchor {
    word: String,
    offset: i8,
}
#[derive(Clone, Debug)]
pub(super) struct Relational {
    roles: Vec<Vec<Anchor>>,
    vocabulary: Vec<String>,
    pub(super) window: u8,
    pub(super) program: NumericSkill,
}
impl Anchor {
    fn positions(&self, words: &[String]) -> Vec<usize> {
        words
            .iter()
            .enumerate()
            .filter(|(_, w)| **w == self.word)
            .filter_map(|(i, _)| {
                let j = i.checked_add_signed(self.offset as isize)?;
                words.get(j).and_then(|w| number(w).ok()).map(|_| j)
            })
            .collect()
    }
}
impl Relational {
    pub(super) fn learn(
        ds: &[Demonstration],
        budget: NumericBudget,
        window: u8,
        parents: &[&Rule],
    ) -> Result<Rule, SemanticError> {
        if !(4..=32).contains(&ds.len()) || !(1..=3).contains(&window) || parents.len() > 16 {
            return Err(SemanticError::Invalid("relational learning bounds"));
        }
        let rows: Vec<_> = ds
            .iter()
            .map(|d| words(&d.sentence))
            .collect::<Result<_, _>>()?;
        let labels: Vec<_> = ds
            .iter()
            .map(|d| match &d.meaning {
                Meaning::Calculate { arguments, answer }
                    if (1..=4).contains(&arguments.len())
                        && answer.is_finite()
                        && arguments.iter().all(|n| n.is_finite()) =>
                {
                    Ok((arguments, *answer))
                }
                _ => Err(SemanticError::Invalid("calculation supervision required")),
            })
            .collect::<Result<_, _>>()?;
        let arity = labels[0].0.len();
        if labels.iter().any(|(x, _)| x.len() != arity)
            || rows
                .iter()
                .any(|w| w.iter().filter(|v| number(v).is_ok()).count() != arity)
        {
            return Err(SemanticError::Invalid("numeric role arity"));
        }
        let vocabulary: Vec<_> = rows
            .iter()
            .flatten()
            .filter(|w| number(w).is_err())
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let pattern: Vec<_> = vocabulary
            .iter()
            .filter(|w| rows.iter().all(|r| r.contains(w)))
            .cloned()
            .map(Some)
            .collect();
        if pattern.is_empty() || vocabulary.len() > 128 {
            return Err(SemanticError::Invalid(
                "no shared anchor or vocabulary ceiling",
            ));
        }
        let mut candidates = BTreeSet::new();
        for word in &vocabulary {
            for offset in -(window as i8)..=window as i8 {
                if offset != 0 {
                    candidates.insert(Anchor {
                        word: word.clone(),
                        offset,
                    });
                }
            }
        }
        let mut roles = Vec::new();
        for role in 0..arity {
            let valid: Vec<_> = candidates
                .iter()
                .filter(|a| {
                    let mut uses = 0;
                    let agrees = rows.iter().zip(&labels).all(|(w, (args, _))| {
                        a.positions(w).iter().all(|i| {
                            uses += 1;
                            number(&w[*i]).is_ok_and(|n| n == args[role])
                        })
                    });
                    agrees && uses >= 2
                })
                .cloned()
                .collect();
            let mut uncovered: BTreeSet<_> = (0..rows.len()).collect();
            let mut chosen = Vec::new();
            while !uncovered.is_empty() && chosen.len() < 32 {
                let best = valid
                    .iter()
                    .max_by_key(|a| {
                        uncovered
                            .iter()
                            .filter(|i| !a.positions(&rows[**i]).is_empty())
                            .count()
                    })
                    .ok_or(SemanticError::Ambiguous)?;
                let covered: Vec<_> = uncovered
                    .iter()
                    .copied()
                    .filter(|i| !best.positions(&rows[*i]).is_empty())
                    .collect();
                if covered.is_empty() {
                    return Err(SemanticError::Ambiguous);
                }
                for i in covered {
                    uncovered.remove(&i);
                }
                chosen.push(best.clone());
            }
            if !uncovered.is_empty() {
                return Err(SemanticError::Invalid("role anchor ceiling"));
            }
            chosen.sort();
            roles.push(chosen);
        }
        let inherited: Vec<_> = parents
            .iter()
            .filter_map(|r| match &r.action {
                Action::Relational(r) => Some(&r.program),
                Action::Calculate(_, p) => Some(p),
                _ => None,
            })
            .collect();
        let examples = labels
            .iter()
            .map(|(x, y)| NumericExample::new((*x).clone(), *y))
            .collect();
        let program = NumericSkill::learn(
            "relational",
            CONTEXT,
            examples,
            &inherited,
            budget,
            NumericTolerance::default(),
        )
        .map_err(|e| SemanticError::Storage(e.to_string()))?;
        let binding = Self {
            roles,
            vocabulary,
            window,
            program,
        };
        binding.validate(&pattern)?;
        for (w, (x, _)) in rows.iter().zip(&labels) {
            if binding.bind(w)? != **x {
                return Err(SemanticError::Ambiguous);
            }
        }
        Ok(Rule {
            pattern,
            action: Action::Relational(Box::new(binding)),
        })
    }
    pub(super) fn bind(&self, w: &[String]) -> Result<Vec<f64>, SemanticError> {
        if w.iter()
            .any(|s| number(s).is_err() && self.vocabulary.binary_search(s).is_err())
            || w.iter().filter(|s| number(s).is_ok()).count() != self.roles.len()
        {
            return Err(SemanticError::Unknown);
        }
        let mut used = BTreeSet::new();
        let mut values = Vec::new();
        for role in &self.roles {
            let indices: BTreeSet<_> = role.iter().flat_map(|a| a.positions(w)).collect();
            if indices.len() != 1 {
                return Err(SemanticError::Ambiguous);
            }
            let i = *indices.first().expect("one");
            if !used.insert(i) {
                return Err(SemanticError::Ambiguous);
            }
            values.push(number(&w[i])?);
        }
        Ok(values)
    }
    pub(super) fn validate(&self, pattern: &[Option<String>]) -> Result<(), SemanticError> {
        let valid_word = |s: &String| {
            !s.is_empty()
                && s.len() <= 128
                && !s.chars().any(char::is_whitespace)
                && number(s).is_err()
        };
        if !(1..=3).contains(&self.window)
            || !(1..=4).contains(&self.roles.len())
            || self.program.context() != CONTEXT
            || self.program.program().arity() != self.roles.len()
            || self.vocabulary.is_empty()
            || self.vocabulary.len() > 128
            || !self.vocabulary.windows(2).all(|w| w[0] < w[1])
            || !self.vocabulary.iter().all(valid_word)
            || pattern.iter().any(|p| {
                p.as_ref()
                    .is_none_or(|w| self.vocabulary.binary_search(w).is_err())
            })
            || self.roles.iter().any(|v| {
                v.is_empty()
                    || v.len() > 32
                    || !v.windows(2).all(|w| w[0] < w[1])
                    || v.iter().any(|a| {
                        a.offset == 0
                            || a.offset.unsigned_abs() > self.window
                            || self.vocabulary.binary_search(&a.word).is_err()
                    })
            })
        {
            return Err(SemanticError::Invalid("relational binding contract"));
        }
        Ok(())
    }
    pub(super) fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.program.retained_bytes()
            + self.vocabulary.iter().map(String::capacity).sum::<usize>()
            + self.vocabulary.capacity() * size_of::<String>()
            + self
                .roles
                .iter()
                .map(|r| {
                    r.capacity() * size_of::<Anchor>()
                        + r.iter().map(|a| a.word.capacity()).sum::<usize>()
                })
                .sum::<usize>()
    }
    pub(super) fn write(&self, b: &mut Vec<u8>) {
        b.push(self.window);
        b.push(self.vocabulary.len() as u8);
        for w in &self.vocabulary {
            text(b, w);
        }
        b.push(self.roles.len() as u8);
        for role in &self.roles {
            b.push(role.len() as u8);
            for a in role {
                text(b, &a.word);
                b.push((a.offset + 3) as u8);
            }
        }
    }
    pub(super) fn read(r: &mut Reader<'_>) -> Result<Self, SemanticError> {
        let window = r.byte()?;
        let n = r.count(128)?;
        let mut vocabulary = Vec::new();
        for _ in 0..n {
            vocabulary.push(r.text()?);
        }
        let n = r.count(4)?;
        let mut roles = Vec::new();
        for _ in 0..n {
            let n = r.count(32)?;
            let mut role = Vec::new();
            for _ in 0..n {
                let word = r.text()?;
                let raw = r.byte()?;
                if raw > 6 {
                    return Err(SemanticError::Invalid("anchor offset"));
                }
                role.push(Anchor {
                    word,
                    offset: raw as i8 - 3,
                });
            }
            roles.push(role);
        }
        Ok(Self {
            roles,
            vocabulary,
            window,
            program: r.program()?,
        })
    }
}
