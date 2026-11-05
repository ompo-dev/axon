//! Bounded joint beam search, not exact Viterbi, calibrated confidence or authority.
use super::*;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub beam: usize,
    pub intents: usize,
    pub max_work: usize,
}
impl Budget {
    pub(crate) fn validate(self) -> Result<(), Error> {
        if !(1..=16).contains(&self.beam)
            || !(1..=8).contains(&self.intents)
            || !(1..=50_000_000).contains(&self.max_work)
        {
            return Err(Error::Invalid("sequence search budget"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Path {
    pub intent: usize,
    pub tags: Vec<usize>,
    pub score: i64,
}
type PathGuard<'a> = dyn Fn(&Path, usize, usize) -> bool + 'a;
#[derive(Debug)]
pub(super) struct Outcome {
    pub paths: Vec<Path>,
    pub work: usize,
    pub expansions: usize,
    pub pruned: bool,
}
#[derive(Debug, Serialize)]
pub struct Prediction {
    pub output: Term,
    pub work: usize,
    pub expansions: usize,
    pub surviving_paths: usize,
    pub pruned: bool,
    pub retained_path_margin: Option<i64>,
}
pub struct Measurement<T = Prediction> {
    pub result: Result<T, Error>,
    pub work: usize,
}
pub(super) fn predict(
    metadata: &Model,
    text: &str,
    subwords: bool,
    budget: Budget,
    scores: impl FnMut(&[usize], &mut usize) -> Vec<i32>,
) -> Result<Prediction, Error> {
    measure(metadata, text, subwords, budget, scores).result
}
pub(super) fn measure(
    metadata: &Model,
    text: &str,
    subwords: bool,
    budget: Budget,
    scores: impl FnMut(&[usize], &mut usize) -> Vec<i32>,
) -> Measurement {
    let mut work = 0;
    let result = (|| {
        let words = tokens(text)?;
        let mut features = metadata.budget();
        features.subwords = subwords;
        let outcome = decode_with_work(
            (&metadata.intents, &metadata.tags),
            &intent_features(&words, features),
            &(0..words.len())
                .map(|i| word_features(&words, i, features))
                .collect::<Vec<_>>(),
            features,
            budget,
            scores,
            &mut work,
        )?;
        let first = &outcome.paths[0];
        let margin = outcome
            .paths
            .get(1)
            .map(|second| first.score - second.score);
        if margin == Some(0) {
            return Err(Error::Ambiguous);
        }
        let output = output(
            &words,
            &metadata.intents[first.intent],
            &first
                .tags
                .iter()
                .map(|i| metadata.tags[*i].as_str())
                .collect::<Vec<_>>(),
        )?;
        Ok(Prediction {
            output,
            work: outcome.work,
            expansions: outcome.expansions,
            surviving_paths: outcome.paths.len(),
            pruned: outcome.pruned,
            retained_path_margin: margin,
        })
    })();
    Measurement { result, work }
}
fn order(a: &Path, b: &Path) -> std::cmp::Ordering {
    b.score
        .cmp(&a.score)
        .then(a.intent.cmp(&b.intent))
        .then(a.tags.cmp(&b.tags))
}
pub(super) fn decode(
    labels: (&[String], &[String]),
    sentence: &[usize],
    words: &[Vec<usize>],
    features: super::Budget,
    budget: Budget,
    scores: impl FnMut(&[usize], &mut usize) -> Vec<i32>,
) -> Result<Outcome, Error> {
    decode_with_work(labels, sentence, words, features, budget, scores, &mut 0)
}
fn decode_with_work(
    labels: (&[String], &[String]),
    sentence: &[usize],
    words: &[Vec<usize>],
    features: super::Budget,
    budget: Budget,
    mut scores: impl FnMut(&[usize], &mut usize) -> Vec<i32>,
    work: &mut usize,
) -> Result<Outcome, Error> {
    budget.validate()?;
    let (intents, tags) = labels;
    let n = intents.len();
    let classes = n + tags.len();
    if words.is_empty() || words.len() > 64 || n < 2 || tags.is_empty() || classes > 256 {
        return Err(Error::Invalid("sequence search shape"));
    }
    let initial = scores(sentence, work);
    decode_conditioned(
        labels,
        words.len(),
        initial.into_iter().map(i64::from).collect(),
        budget,
        &mut |index, path, work| {
            let previous = path.tags.last().map_or("<start>", |i| &tags[*i]);
            let previous2 = path
                .tags
                .len()
                .checked_sub(2)
                .map_or("<start2>", |i| &tags[path.tags[i]]);
            let f = contextual(
                &words[index],
                previous,
                previous2,
                &intents[path.intent],
                features,
            );
            Ok(scores(&f, work).into_iter().map(i64::from).collect())
        },
        work,
    )
}
pub(super) fn decode_conditioned(
    labels: (&[String], &[String]),
    words: usize,
    initial: Vec<i64>,
    budget: Budget,
    scores: &mut impl FnMut(usize, &Path, &mut usize) -> Result<Vec<i64>, Error>,
    work: &mut usize,
) -> Result<Outcome, Error> {
    decode_conditioned_roles(labels, words, initial, budget, scores, work, None)
}
pub(super) fn decode_conditioned_roles(
    labels: (&[String], &[String]),
    words: usize,
    initial: Vec<i64>,
    budget: Budget,
    scores: &mut impl FnMut(usize, &Path, &mut usize) -> Result<Vec<i64>, Error>,
    work: &mut usize,
    allowed: Option<&dyn Fn(usize, usize) -> bool>,
) -> Result<Outcome, Error> {
    let filter = |path: &Path, _index, tag| allowed.is_none_or(|f| f(path.intent, tag));
    decode_conditioned_paths(
        labels,
        words,
        initial,
        budget,
        scores,
        work,
        allowed.map(|_| &filter as &dyn Fn(&Path, usize, usize) -> bool),
    )
}
pub(super) fn decode_conditioned_paths(
    labels: (&[String], &[String]),
    words: usize,
    initial: Vec<i64>,
    budget: Budget,
    scores: &mut impl FnMut(usize, &Path, &mut usize) -> Result<Vec<i64>, Error>,
    work: &mut usize,
    allowed: Option<&PathGuard<'_>>,
) -> Result<Outcome, Error> {
    budget.validate()?;
    let (intents, tags) = labels;
    let n = intents.len();
    let classes = n + tags.len();
    if words == 0 || words > 64 || n < 2 || tags.is_empty() || classes > 256 {
        return Err(Error::Invalid("sequence search shape"));
    }
    if initial.len() != classes {
        return Err(Error::Invalid("sequence search scores"));
    }
    if *work > budget.max_work {
        return Err(Error::Budget);
    }
    let mut ranked = (0..n).collect::<Vec<_>>();
    ranked.sort_by_key(|i| (std::cmp::Reverse(initial[*i]), *i));
    let mut paths = ranked
        .into_iter()
        .take(budget.intents)
        .map(|intent| Path {
            intent,
            tags: Vec::new(),
            score: initial[intent],
        })
        .collect::<Vec<_>>();
    let mut pruned = budget.intents < n;
    let mut expansions = 0usize;
    for index in 0..words {
        let mut next: Vec<Path> = Vec::with_capacity(budget.beam + 1);
        for path in &paths {
            let previous = path.tags.last().map_or("<start>", |i| &tags[*i]);
            let values = scores(index, path, work)?;
            if values.len() != classes {
                return Err(Error::Invalid("sequence search scores"));
            }
            if *work > budget.max_work {
                return Err(Error::Budget);
            }
            for (tag, _) in tags.iter().enumerate().filter(|(_, t)| legal(t, previous)) {
                if let Some(allowed) = allowed {
                    *work += 1;
                    if *work > budget.max_work {
                        return Err(Error::Budget);
                    }
                    if !allowed(path, index, tag) {
                        continue;
                    }
                }
                *work += 1;
                if *work > budget.max_work {
                    return Err(Error::Budget);
                }
                expansions += 1;
                let mut candidate = Path {
                    intent: path.intent,
                    tags: path.tags.clone(),
                    score: path
                        .score
                        .checked_add(values[n + tag])
                        .ok_or(Error::Budget)?,
                };
                candidate.tags.push(tag);
                let index = next.partition_point(|p| order(p, &candidate).is_lt());
                if index < budget.beam {
                    next.insert(index, candidate);
                    if next.len() > budget.beam {
                        next.pop();
                        pruned = true;
                    }
                } else {
                    pruned = true;
                }
            }
        }
        if next.is_empty() {
            return Err(Error::Unsupported);
        }
        paths = next;
    }
    Ok(Outcome {
        paths,
        work: *work,
        expansions,
        pruned,
    })
}

pub(super) fn output(words: &[String], intent: &str, tags: &[&str]) -> Result<Term, Error> {
    if words.len() != tags.len() {
        return Err(Error::Invalid("sequence output length"));
    }
    let mut spans: Vec<(String, String)> = Vec::new();
    let mut previous = "<start>";
    for (word, tag) in words.iter().zip(tags) {
        if !legal(tag, previous) {
            return Err(Error::Invalid("orphan sequence span"));
        }
        if tag.starts_with("B:") {
            spans.push((
                role(tag)
                    .ok_or(Error::Invalid("sequence output tag"))?
                    .into(),
                word.clone(),
            ));
        } else if tag.starts_with("I:") {
            let value = &mut spans
                .last_mut()
                .ok_or(Error::Invalid("orphan sequence span"))?
                .1;
            value.push(' ');
            value.push_str(word);
        } else if *tag != "O" {
            return Err(Error::Invalid("sequence output tag"));
        }
        previous = tag;
    }
    let slots = if spans.is_empty() {
        Term::Symbol("none".into())
    } else {
        Term::Tree(
            std::iter::once(Term::Symbol("slots".into()))
                .chain(spans.into_iter().map(|(role, value)| {
                    Term::Tree(vec![
                        Term::Symbol(role),
                        Term::Symbol(value.trim_end_matches(['.', '?', '!']).into()),
                    ])
                }))
                .collect(),
        )
    };
    let output = Term::Tree(vec![
        Term::Symbol("intent".into()),
        Term::Symbol(intent.into()),
        slots,
    ]);
    output.validate()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn role_domains_filter_paths_charge_lookups_and_preserve_budget_refusal() {
        let intents = vec!["a".into(), "b".into()];
        let tags = vec!["O".into(), "B:value".into(), "I:value".into()];
        let budget = Budget {
            beam: 16,
            intents: 2,
            max_work: 1000,
        };
        let allowed = |intent, tag| intent != 0 || tag == 0;
        let scores = |_: usize, _: &Path, _: &mut usize| Ok(vec![0, 0, 1, 4, 3]);
        let mut work = 0;
        let result = decode_conditioned_roles(
            (&intents, &tags),
            2,
            vec![0; 5],
            budget,
            &mut { scores },
            &mut work,
            Some(&allowed),
        )
        .unwrap();
        assert!(
            result
                .paths
                .iter()
                .all(|p| p.intent != 0 || p.tags == [0, 0])
        );
        assert_eq!(result.paths.len(), 6);
        assert!(result.work > result.expansions);
        let mut limited = 0;
        assert!(matches!(
            decode_conditioned_roles(
                (&intents, &tags),
                2,
                vec![0; 5],
                Budget {
                    max_work: 1,
                    ..budget
                },
                &mut { scores },
                &mut limited,
                Some(&allowed)
            ),
            Err(Error::Budget)
        ));
        assert_eq!(limited, 2);
    }
    fn features() -> super::super::Budget {
        super::super::Budget {
            seed: 47,
            buckets: 16384,
            epochs: 1,
            weights: 256,
            intent_weights: 128,
            subwords: false,
            shared_roles: false,
            minimum_margin: 1,
        }
    }
    #[test]
    fn conditioned_decoder_matches_exhaustive_small_paths_and_propagates_errors() {
        let intents = ["a".into(), "b".into()];
        let tags = ["O".into(), "B:x".into()];
        let initial = vec![5, -3, 0, 0];
        let score = |index: usize, p: &Path| -> Vec<i64> {
            let previous = p.tags.last().copied().unwrap_or(0) as i64;
            vec![
                0,
                0,
                (index as i64 + previous + p.intent as i64) * 3,
                7 - index as i64 - previous * 4,
            ]
        };
        let mut expected = Vec::new();
        for (intent, initial_score) in initial.iter().copied().enumerate().take(2) {
            for bits in 0..8 {
                let mut path = Path {
                    intent,
                    tags: Vec::new(),
                    score: initial_score,
                };
                for index in 0..3 {
                    let tag = (bits >> index) & 1;
                    path.score += score(index, &path)[2 + tag];
                    path.tags.push(tag);
                }
                expected.push(path);
            }
        }
        expected.sort_by(order);
        let budget = Budget {
            beam: 16,
            intents: 2,
            max_work: 10000,
        };
        let result = decode_conditioned(
            (&intents, &tags),
            3,
            initial.clone(),
            budget,
            &mut |i, p, w| {
                *w += 4;
                Ok(score(i, p))
            },
            &mut 0,
        )
        .unwrap();
        assert_eq!(result.paths, expected);
        assert!(!result.pruned);
        assert!(
            decode_conditioned(
                (&intents, &tags),
                3,
                initial.clone(),
                budget,
                &mut |_, _, _| Err(Error::Budget),
                &mut 0
            )
            .is_err()
        );
        assert!(
            decode_conditioned(
                (&intents, &tags),
                3,
                initial.clone(),
                budget,
                &mut |_, _, _| Ok(vec![]),
                &mut 0
            )
            .is_err()
        );
        assert!(
            decode_conditioned(
                (&intents, &tags),
                3,
                initial,
                budget,
                &mut |_, _, w| {
                    *w = 10001;
                    Ok(vec![0; 4])
                },
                &mut 0
            )
            .is_err()
        );
    }
    #[test]
    fn a_delayed_reward_can_change_the_whole_sequence_without_illegal_bio() {
        let intents = ["a".into(), "b".into()];
        let tags = ["O".into(), "B:x".into(), "I:x".into()];
        let b = features();
        let marker = hash(b.seed, 30, &["B:x"], b.buckets);
        let score = |f: &[usize], work: &mut usize| {
            *work += 5;
            if f == [2] {
                vec![0, -100, 0, 0, 0]
            } else if f.contains(&0) {
                vec![0, 0, 5, 4, 100]
            } else if f.contains(&marker) {
                vec![0, 0, 0, 0, 10]
            } else {
                vec![0, 0, 0, -100, 100]
            }
        };
        let greedy = decode(
            (&intents, &tags),
            &[2],
            &[vec![0], vec![1]],
            b,
            Budget {
                beam: 1,
                intents: 1,
                max_work: 1000,
            },
            score,
        )
        .unwrap();
        let joint = decode(
            (&intents, &tags),
            &[2],
            &[vec![0], vec![1]],
            b,
            Budget {
                beam: 2,
                intents: 1,
                max_work: 1000,
            },
            score,
        )
        .unwrap();
        assert_eq!(greedy.paths[0].tags, [0, 0]);
        assert_eq!(joint.paths[0].tags, [1, 2]);
        assert_eq!(joint.paths[0].score, 14);
        assert!(joint.pruned);
        assert!(
            decode(
                (&intents, &tags),
                &[2],
                &[vec![0]],
                b,
                Budget {
                    beam: 2,
                    intents: 1,
                    max_work: 1
                },
                score
            )
            .is_err()
        );
        assert!(
            decode(
                (&intents, &tags),
                &[2],
                &[vec![0]],
                b,
                Budget {
                    beam: 0,
                    intents: 1,
                    max_work: 1000
                },
                score
            )
            .is_err()
        );
        assert!(output(&["a".into()], "a", &["I:x"]).is_err());
    }
}
