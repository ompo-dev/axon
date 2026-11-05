//! Discriminative anti-unification over supervised trees, without a task-specific vocabulary.
use super::{Error, Example, Guard, Lesson, Pattern, Rule, StructuralSeed, Term};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub seed: u64,
    pub proposals: usize,
    pub rules: usize,
    pub checks: usize,
    #[serde(default = "default_minimum_holes")]
    pub minimum_holes: usize,
    #[serde(default)]
    pub structural_groups: bool,
}
fn default_minimum_holes() -> usize {
    2
}
#[derive(Default, Serialize)]
pub struct Statistics {
    pub proposals: usize,
    pub learned: usize,
    pub refuted: usize,
    pub checked: usize,
    pub selected: usize,
    pub correct: usize,
    pub wrong: usize,
    pub abstained: usize,
    pub quota_reached: bool,
    pub insufficient_evidence: usize,
    pub conflicting_keys: usize,
}
#[derive(Serialize)]
pub struct Event {
    pub phase: &'static str,
    pub candidate: usize,
    pub coverage: usize,
    pub checked: usize,
}
struct Candidate {
    lesson: Lesson,
    admission: Example,
    matches: Vec<usize>,
    bytes: usize,
    spans: bool,
}
fn shape(t: &Term, out: &mut Vec<String>, depth: usize, heads: bool) {
    if depth > if heads { 48 } else { 1 } {
        return;
    }
    if let Term::Tree(v) = t {
        out.push(v.len().to_string());
        if heads {
            let Term::Symbol(head) = &v[0] else {
                return;
            };
            out.push(head.clone());
        }
        for t in v {
            shape(t, out, depth + 1, heads);
        }
    }
}
fn similarity(a: &Term, b: &Term) -> usize {
    if a == b {
        return super::term_nodes(a);
    }
    match (a, b) {
        (Term::Tree(a), Term::Tree(b)) => a.iter().zip(b).map(|(a, b)| similarity(a, b)).sum(),
        _ => 0,
    }
}
fn diverse(rule: &Rule, minimum_holes: usize) -> bool {
    rule.holes >= minimum_holes
        && matches!(&rule.input,Pattern::Tree(v) if v.iter().skip(1).all(|p|!matches!(p,Pattern::Literal(Term::Tree(_)))))
}
pub fn learn(
    examples: &[Example],
    budget: Budget,
    emit: &mut impl FnMut(&Event) -> Result<(), Error>,
) -> Result<(StructuralSeed, Statistics), Error> {
    learn_impl(examples, None, None, budget, emit)
}

/// Supervised families may contain structured targets with varying slot values.
/// Families guide proposals only; every candidate is checked against all examples.
pub fn learn_grouped(
    families: &[Vec<Example>],
    budget: Budget,
    emit: &mut impl FnMut(&Event) -> Result<(), Error>,
) -> Result<(StructuralSeed, Statistics), Error> {
    grouped_impl(families, budget, false, emit)
}
pub fn learn_grouped_domains(
    families: &[Vec<Example>],
    budget: Budget,
    emit: &mut impl FnMut(&Event) -> Result<(), Error>,
) -> Result<(StructuralSeed, Statistics), Error> {
    grouped_impl(families, budget, true, emit)
}
fn grouped_impl(
    families: &[Vec<Example>],
    budget: Budget,
    domains: bool,
    emit: &mut impl FnMut(&Event) -> Result<(), Error>,
) -> Result<(StructuralSeed, Statistics), Error> {
    if families.is_empty()
        || families.len() > 10000
        || families.iter().any(Vec::is_empty)
        || families.iter().map(Vec::len).sum::<usize>() > 10000
    {
        return Err(Error::Invalid("induction families"));
    }
    let examples = families.iter().flatten().cloned().collect::<Vec<_>>();
    for e in &examples {
        e.input.validate()?;
        e.output.validate()?;
    }
    let keys = families
        .iter()
        .enumerate()
        .flat_map(|(index, rows)| std::iter::repeat_n(index.to_string(), rows.len()))
        .collect::<Vec<_>>();
    let guards = if domains {
        Some(domain_guards(families))
    } else {
        None
    };
    learn_impl(&examples, Some(&keys), guards.as_ref(), budget, emit)
}

fn typed_values(t: &Term, out: &mut Vec<(String, String)>) {
    if let Term::Tree(v) = t {
        if let [Term::Symbol(role), Term::Symbol(value)] = v.as_slice() {
            out.push((role.clone(), value.clone()));
        }
        for child in &v[1..] {
            typed_values(child, out);
        }
    }
}
fn domain_guards(families: &[Vec<Example>]) -> BTreeMap<String, Guard> {
    let mut domains: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for e in families.iter().flatten() {
        let mut values = Vec::new();
        typed_values(&e.output, &mut values);
        for (role, value) in values {
            for word in value.split_whitespace() {
                *domains
                    .entry(role.clone())
                    .or_default()
                    .entry(word.into())
                    .or_default() += 1;
            }
        }
    }
    let mut guards = BTreeMap::new();
    for (index, family) in families.iter().enumerate() {
        let mut roles = BTreeSet::new();
        let mut words = BTreeSet::new();
        for e in family {
            let Term::Tree(tokens) = &e.input else {
                continue;
            };
            words.extend(tokens.iter().filter_map(|t| {
                if let Term::Symbol(s) = t {
                    Some(s.clone())
                } else {
                    None
                }
            }));
            let mut values = Vec::new();
            typed_values(&e.output, &mut values);
            roles.extend(values.into_iter().map(|(role, _)| role));
        }
        if roles.is_empty() {
            continue;
        }
        for role in roles {
            if let Some(domain) = domains.get(&role) {
                let mut ranked = domain.iter().collect::<Vec<_>>();
                ranked.sort_by_key(|(word, count)| (std::cmp::Reverse(**count), *word));
                words.extend(ranked.into_iter().take(256).map(|(word, _)| word.clone()));
            }
        }
        if (2..=512).contains(&words.len()) {
            guards.insert(
                index.to_string(),
                Guard::Vocabulary {
                    path: vec![],
                    values: words.into_iter().collect(),
                },
            );
        }
    }
    guards
}

fn learn_impl(
    examples: &[Example],
    family_keys: Option<&[String]>,
    domain_guards: Option<&BTreeMap<String, Guard>>,
    budget: Budget,
    emit: &mut impl FnMut(&Event) -> Result<(), Error>,
) -> Result<(StructuralSeed, Statistics), Error> {
    if !(4..=10000).contains(&examples.len())
        || !(1..=1024).contains(&budget.proposals)
        || !(1..=128).contains(&budget.rules)
        || !(1..=16000000).contains(&budget.checks)
        || !(1..=64).contains(&budget.minimum_holes)
    {
        return Err(Error::Invalid("induction quota"));
    }
    let mut groups: BTreeMap<(String, Vec<String>), Vec<usize>> = BTreeMap::new();
    let mut keys = Vec::with_capacity(examples.len());
    for (i, e) in examples.iter().enumerate() {
        e.input.validate()?;
        e.output.validate()?;
        if family_keys.is_none() && !matches!(e.output, Term::Symbol(_) | Term::Integer(_)) {
            return Err(Error::Invalid("induction target"));
        }
        let mut signature = Vec::new();
        let label = if let Some(keys) = family_keys {
            keys[i].clone()
        } else {
            shape(&e.input, &mut signature, 0, budget.structural_groups);
            serde_json::to_string(&e.output).map_err(|e| Error::Storage(e.to_string()))?
        };
        let key = (label, signature);
        groups.entry(key.clone()).or_default().push(i);
        keys.push(key);
    }
    let mut rng = budget.seed.max(1);
    let mut anchors = (0..examples.len()).collect::<Vec<_>>();
    for i in (1..anchors.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        anchors.swap(i, (rng as usize) % (i + 1));
    }
    if budget.structural_groups || family_keys.is_some() {
        let rank = anchors
            .iter()
            .enumerate()
            .map(|(rank, index)| (*index, rank))
            .collect::<BTreeMap<_, _>>();
        let mut queues = groups
            .values()
            .map(|indices| {
                let mut queue = indices.clone();
                queue.sort_by_key(|index| rank[index]);
                queue
            })
            .collect::<Vec<_>>();
        queues.sort_by_key(|queue| (queue.len(), rank[&queue[0]]));
        anchors.clear();
        for offset in 0..examples.len() {
            let mut added = false;
            for queue in &queues {
                if let Some(index) = queue.get(offset) {
                    anchors.push(*index);
                    added = true;
                }
            }
            if !added {
                break;
            }
        }
    }
    let mut stats = Statistics::default();
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();
    for anchor in anchors {
        if stats.proposals >= budget.proposals || stats.checked >= budget.checks {
            stats.quota_reached = true;
            break;
        }
        let Some(group) = groups.get(&keys[anchor]) else {
            continue;
        };
        let mut nearest = group
            .iter()
            .copied()
            .filter(|i| *i != anchor)
            .map(|i| (similarity(&examples[anchor].input, &examples[i].input), i))
            .collect::<Vec<_>>();
        nearest.sort_by_key(|(score, i)| (std::cmp::Reverse(*score), *i));
        let mut chosen = vec![anchor];
        for (_, i) in nearest {
            if chosen
                .iter()
                .all(|j| examples[*j].input != examples[i].input)
            {
                chosen.push(i);
            }
            if chosen.len() == 4 {
                break;
            }
        }
        if chosen.len() != 4 {
            continue;
        }
        stats.proposals += 1;
        let lesson = Lesson {
            id: format!("induced-{}", stats.proposals),
            guards: domain_guards
                .and_then(|g| g.get(&keys[anchor].0))
                .cloned()
                .into_iter()
                .collect(),
            examples: chosen[..3].iter().map(|i| examples[*i].clone()).collect(),
        };
        let spans = family_keys.is_some() && lesson.examples.iter().all(|e| matches!(&e.input, Term::Tree(v) if v.iter().all(|t|matches!(t, Term::Symbol(_)))));
        let Ok(rule) = Rule::learn_with(lesson.clone(), spans) else {
            continue;
        };
        if !(diverse(&rule, budget.minimum_holes)
            || (family_keys.is_some()
                && rule.holes >= budget.minimum_holes
                && matches!(rule.input, Pattern::Phrase(_))))
        {
            continue;
        }
        let identity = serde_json::to_vec(&(&rule.input, &rule.output))
            .map_err(|e| Error::Storage(e.to_string()))?;
        if !seen.insert(identity) {
            continue;
        }
        let admission = examples[chosen[3]].clone();
        if rule.apply(&admission.input).as_ref() != Some(&admission.output) {
            stats.refuted += 1;
            continue;
        }
        emit(&Event {
            phase: "hypothesis",
            candidate: stats.proposals,
            coverage: 0,
            checked: stats.checked,
        })?;
        let mut matches = Vec::new();
        let mut wrong = false;
        for (i, e) in examples.iter().enumerate() {
            if stats.checked >= budget.checks {
                stats.quota_reached = true;
                wrong = true;
                break;
            }
            stats.checked += 1;
            if let Some(out) = rule.apply(&e.input) {
                if out != e.output {
                    wrong = true;
                    break;
                }
                matches.push(i);
            }
        }
        if wrong {
            stats.refuted += 1;
            continue;
        }
        if matches.len() < 6 {
            continue;
        }
        stats.learned += 1;
        emit(&Event {
            phase: "verified",
            candidate: stats.proposals,
            coverage: matches.len(),
            checked: stats.checked,
        })?;
        let bytes = serde_json::to_vec(&rule)
            .map_err(|e| Error::Storage(e.to_string()))?
            .len();
        candidates.push(Candidate {
            lesson,
            admission,
            matches,
            bytes,
            spans,
        });
    }
    let mut seed = StructuralSeed::default();
    let mut covered = BTreeSet::new();
    while seed.rules.len() < budget.rules && !candidates.is_empty() {
        let selected = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    c.matches.iter().filter(|j| !covered.contains(*j)).count(),
                    std::cmp::Reverse(c.bytes),
                    std::cmp::Reverse(i),
                )
            })
            .max();
        let Some((gain, _, std::cmp::Reverse(index))) = selected else {
            break;
        };
        if gain == 0 {
            break;
        }
        let candidate = candidates.remove(index);
        if seed
            .teach_with(candidate.lesson, &[candidate.admission], candidate.spans)
            .is_ok()
        {
            covered.extend(candidate.matches);
        }
    }
    stats.selected = seed.rules.len();
    for e in examples {
        let result = seed.transform(&e.input, 512);
        match result {
            Ok(p) if !p.rules.is_empty() && p.output == e.output => stats.correct += 1,
            Ok(p) if !p.rules.is_empty() => stats.wrong += 1,
            _ => stats.abstained += 1,
        }
    }
    Ok((seed, stats))
}

/// Closed associations between recurring abstract observations and labels. Conflicts abstain.
pub fn learn_closed(
    examples: &[Example],
    budget: Budget,
    key_path: &[usize],
    emit: &mut impl FnMut(&Event) -> Result<(), Error>,
) -> Result<(StructuralSeed, Statistics), Error> {
    if !(4..=10000).contains(&examples.len())
        || !(1..=128).contains(&budget.rules)
        || !(1..=1024).contains(&budget.proposals)
        || !(1..=16000000).contains(&budget.checks)
    {
        return Err(Error::Invalid("closed induction quota"));
    }
    let mut keys: BTreeMap<String, Vec<Example>> = BTreeMap::new();
    for example in examples {
        example.input.validate()?;
        example.output.validate()?;
        if !matches!(example.output, Term::Integer(_) | Term::Symbol(_)) {
            return Err(Error::Invalid("closed target"));
        }
        let Some(Term::Symbol(key)) = example.input.at(key_path) else {
            return Err(Error::Invalid("closed key"));
        };
        let rows = keys.entry(key.clone()).or_default();
        if !rows
            .iter()
            .any(|row| row.input == example.input && row.output == example.output)
        {
            rows.push(example.clone());
        }
    }
    let mut groups: BTreeMap<String, Vec<(String, Vec<Example>)>> = BTreeMap::new();
    let mut stats = Statistics::default();
    for (key, rows) in keys {
        if rows.iter().any(|row| row.output != rows[0].output) {
            stats.conflicting_keys += 1;
            continue;
        }
        if rows.len() < 2 {
            stats.insufficient_evidence += 1;
            continue;
        }
        let label =
            serde_json::to_string(&rows[0].output).map_err(|e| Error::Storage(e.to_string()))?;
        groups.entry(label).or_default().push((key, rows));
    }
    let mut seed = StructuralSeed::default();
    for group in groups.values_mut() {
        group.sort_by_key(|(key, rows)| (std::cmp::Reverse(rows.len()), key.clone()));
        for chunk in group.chunks(16) {
            if seed.rules.len() >= budget.rules || stats.proposals >= budget.proposals {
                stats.quota_reached = true;
                break;
            }
            let mut training = chunk
                .iter()
                .map(|(_, rows)| rows[0].clone())
                .collect::<Vec<_>>();
            let admission = chunk
                .iter()
                .map(|(_, rows)| rows.last().expect("nonempty observations").clone())
                .collect::<Vec<_>>();
            for (_, rows) in chunk {
                for example in &rows[1..rows.len() - 1] {
                    if training.len() < 3 {
                        training.push(example.clone());
                    }
                }
            }
            if training.len() < 3 {
                stats.insufficient_evidence += chunk.len();
                continue;
            }
            stats.proposals += 1;
            let mut values = chunk.iter().map(|(key, _)| key.clone()).collect::<Vec<_>>();
            values.sort();
            let lesson = Lesson {
                id: format!("association-{}", stats.proposals),
                guards: if values.len() > 1 {
                    vec![Guard::Symbols {
                        path: key_path.to_vec(),
                        values,
                    }]
                } else {
                    vec![]
                },
                examples: training,
            };
            emit(&Event {
                phase: "association",
                candidate: stats.proposals,
                coverage: chunk.iter().map(|(_, rows)| rows.len()).sum(),
                checked: stats.checked,
            })?;
            if seed.teach(lesson, &admission).is_ok() {
                stats.learned += 1;
            } else {
                stats.refuted += 1;
            }
        }
    }
    stats.selected = seed.rules.len();
    for example in examples {
        if stats.checked >= budget.checks {
            return Err(Error::Budget);
        }
        stats.checked += 1;
        match seed.transform(&example.input, 512) {
            Ok(p) if !p.rules.is_empty() && p.output == example.output => stats.correct += 1,
            Ok(p) if !p.rules.is_empty() => stats.wrong += 1,
            _ => stats.abstained += 1,
        }
    }
    Ok((seed, stats))
}
