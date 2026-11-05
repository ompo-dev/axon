//! Variable-length symbol spans between anchors inferred from demonstrations.
use super::{Error, Pattern, Term};

fn symbols(t: &Term) -> Result<Vec<String>, Error> {
    let Term::Tree(v) = t else {
        return Err(Error::Unsupported);
    };
    v.iter()
        .map(|t| match t {
            Term::Symbol(s) => Ok(s.clone()),
            _ => Err(Error::Unsupported),
        })
        .collect()
}
fn common(a: &[String], b: &[String]) -> Vec<String> {
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            dp[i][j] = if a[i] == b[j] {
                1 + dp[i + 1][j + 1]
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            out.push(a[i].clone());
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}
pub(super) fn learn(rows: &[Term], holes: &mut Vec<Vec<Term>>) -> Result<Pattern, Error> {
    let rows = rows.iter().map(symbols).collect::<Result<Vec<_>, _>>()?;
    learn_inner(&rows, &vec![Vec::new(); rows.len()], holes)
}

/// Varying target spans cannot become incidental literal anchors in the input.
pub(super) fn learn_supervised(
    rows: &[Term],
    targets: &[Term],
    holes: &mut Vec<Vec<Term>>,
) -> Result<Pattern, Error> {
    fn columns(rows: &[Term], out: &mut Vec<Vec<String>>) {
        if rows.iter().all(|t| *t == rows[0]) {
            return;
        }
        if rows.iter().all(|t| matches!(t, Term::Symbol(_))) {
            out.push(
                rows.iter()
                    .map(|t| {
                        let Term::Symbol(s) = t else { unreachable!() };
                        s.clone()
                    })
                    .collect(),
            );
        } else if let Term::Tree(first) = &rows[0]
            && rows
                .iter()
                .all(|t| matches!(t,Term::Tree(v) if v.len()==first.len() && v[0]==first[0]))
        {
            for index in 1..first.len() {
                columns(
                    &rows
                        .iter()
                        .map(|t| {
                            let Term::Tree(v) = t else { unreachable!() };
                            v[index].clone()
                        })
                        .collect::<Vec<_>>(),
                    out,
                );
            }
        }
    }
    let rows = rows.iter().map(symbols).collect::<Result<Vec<_>, _>>()?;
    let mut blocked = rows
        .iter()
        .map(|row| vec![false; row.len()])
        .collect::<Vec<_>>();
    let mut values = Vec::new();
    columns(targets, &mut values);
    for column in values {
        for (i, value) in column.iter().enumerate() {
            let tokens = value.split_whitespace().collect::<Vec<_>>();
            if tokens.is_empty() || tokens.len() > rows[i].len() {
                return Err(Error::Unsupported);
            }
            let positions = rows[i]
                .windows(tokens.len())
                .enumerate()
                .filter_map(|(index, words)| {
                    (index > 0 && words.iter().map(String::as_str).eq(tokens.iter().copied()))
                        .then_some(index)
                })
                .collect::<Vec<_>>();
            if positions.len() != 1 {
                return Err(Error::Unsupported);
            }
            for cell in &mut blocked[i][positions[0]..positions[0] + tokens.len()] {
                *cell = true;
            }
        }
    }
    learn_inner(&rows, &blocked, holes)
}

fn learn_inner(
    rows: &[Vec<String>],
    blocked: &[Vec<bool>],
    holes: &mut Vec<Vec<Term>>,
) -> Result<Pattern, Error> {
    let eligible = rows
        .iter()
        .zip(blocked)
        .map(|(row, mask)| {
            row.iter()
                .enumerate()
                .filter(|(i, _)| !mask.get(*i).copied().unwrap_or(false))
                .map(|(_, s)| s.clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut anchors = eligible[0].clone();
    for row in &eligible[1..] {
        anchors = common(&anchors, row);
    }
    if anchors.len() < 2 || anchors[0] != rows[0][0] {
        return Err(Error::Unsupported);
    }
    // Alignment is a proposal. Exact replay and admission reject ambiguous span bindings.
    let mut cursor = vec![0; rows.len()];
    let mut pieces = Vec::new();
    for anchor in anchors.iter().map(Some).chain(std::iter::once(None)) {
        let ends = rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                anchor
                    .map_or(row.len(), |a| {
                        cursor[i]
                            + row[cursor[i]..]
                                .iter()
                                .enumerate()
                                .position(|(offset, s)| {
                                    s == a
                                        && !blocked[i]
                                            .get(cursor[i] + offset)
                                            .copied()
                                            .unwrap_or(false)
                                })
                                .expect("monotone common anchor")
                    })
                    .max(cursor[i])
            })
            .collect::<Vec<_>>();
        let gaps = rows
            .iter()
            .enumerate()
            .map(|(i, row)| row[cursor[i]..ends[i]].join(" "))
            .collect::<Vec<_>>();
        if gaps.iter().any(|g| !g.is_empty()) {
            if gaps.iter().any(|g| g.is_empty() || g.len() > 256) {
                return Err(Error::Unsupported);
            }
            let values = gaps.into_iter().map(Term::Symbol).collect::<Vec<_>>();
            let index = if let Some(i) = holes.iter().position(|h| *h == values) {
                i
            } else {
                let i = holes.len();
                if i >= 64 {
                    return Err(Error::Budget);
                }
                holes.push(values);
                i
            };
            pieces.push(Pattern::Span(index));
        }
        if let Some(a) = anchor {
            pieces.push(Pattern::Literal(Term::Symbol(a.clone())));
        }
        for (i, end) in ends.iter().enumerate() {
            cursor[i] = end + usize::from(anchor.is_some());
        }
    }
    Ok(Pattern::Phrase(pieces))
}

pub(super) fn bind(
    pieces: &[Pattern],
    t: &Term,
    values: &mut Vec<Option<Term>>,
) -> Result<bool, Error> {
    let Term::Tree(words) = t else {
        return Ok(false);
    };
    if words.iter().any(|t| !matches!(t, Term::Symbol(_))) {
        return Ok(false);
    }
    struct Search {
        visits: usize,
        answer: Option<Vec<Option<Term>>>,
        ambiguous: bool,
    }
    fn walk(
        pieces: &[Pattern],
        words: &[Term],
        values: Vec<Option<Term>>,
        s: &mut Search,
    ) -> Result<(), Error> {
        s.visits += 1;
        if s.visits > 4096 {
            return Err(Error::Budget);
        }
        let Some((first, rest)) = pieces.split_first() else {
            if words.is_empty() {
                if s.answer.as_ref().is_some_and(|a| *a != values) {
                    s.ambiguous = true;
                } else {
                    s.answer = Some(values);
                }
            }
            return Ok(());
        };
        match first {
            Pattern::Literal(literal) if words.first() == Some(literal) => {
                walk(rest, &words[1..], values, s)?
            }
            Pattern::Span(slot) => {
                for count in 1..=words.len().saturating_sub(rest.len()) {
                    let text = words[..count]
                        .iter()
                        .map(|t| {
                            let Term::Symbol(s) = t else { unreachable!() };
                            s.as_str()
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    if text.len() > 256 {
                        break;
                    }
                    let value = Term::Symbol(text);
                    if values[*slot].as_ref().is_some_and(|old| *old != value) {
                        continue;
                    }
                    let mut next = values.clone();
                    next[*slot] = Some(value);
                    walk(rest, &words[count..], next, s)?;
                    if s.ambiguous {
                        break;
                    }
                }
            }
            _ => (),
        }
        Ok(())
    }
    let mut search = Search {
        visits: 0,
        answer: None,
        ambiguous: false,
    };
    walk(pieces, words, values.clone(), &mut search)?;
    if search.ambiguous {
        return Err(Error::Ambiguous);
    }
    if let Some(answer) = search.answer {
        *values = answer;
        Ok(true)
    } else {
        Ok(false)
    }
}
