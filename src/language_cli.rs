use crate::{
    EpisodeStamp, EpisodeStore,
    cli::CliError,
    grounded_language::{GroundedTool, LanguageFrame, answer},
    lexicon::{self, LexiconView, Relation, Resolution},
    numeric::NumericSkill,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

pub const USAGE: &str = "\n  axon dictionary-import <journal.axep> <context> <facts.tsv>\n  axon dictionary-query <journal.axep> <context> <word> <parent-sense>\n  axon language-answer <journal.axep> <context> <frames.tsv> <tools.tsv> <sentence>";
fn invalid(line: usize, reason: &'static str) -> CliError {
    CliError::InvalidEvaluation { line, reason }
}
fn table(path: &str, header: &[&str], maximum: usize) -> Result<Vec<Vec<String>>, CliError> {
    let bytes = crate::text_skill::read_bounded(Path::new(path), 67_108_864)?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(bytes.as_slice());
    if reader
        .headers()
        .map_err(|_| invalid(1, "TSV header"))?
        .iter()
        .ne(header.iter().copied())
    {
        return Err(invalid(1, "TSV columns"));
    }
    let mut rows = Vec::new();
    for (i, row) in reader.records().enumerate() {
        if rows.len() == maximum {
            return Err(invalid(i + 2, "TSV row limit"));
        }
        let row = row.map_err(|_| invalid(i + 2, "TSV row"))?;
        if row
            .iter()
            .any(|s| s.len() > 4096 || s.chars().any(char::is_control))
        {
            return Err(invalid(i + 2, "TSV field"));
        }
        rows.push(row.iter().map(str::to_owned).collect());
    }
    if rows.is_empty() {
        return Err(invalid(2, "empty TSV"));
    }
    Ok(rows)
}
fn existing(path: &str) -> Result<EpisodeStore, CliError> {
    if !Path::new(path).is_file() {
        return Err(invalid(0, "dictionary journal must exist"));
    }
    Ok(EpisodeStore::open_with_limits(path, lexicon::store_limits())?.0)
}
pub fn command(a: &[String]) -> Result<String, CliError> {
    match a.first().map(String::as_str) {
        Some("dictionary-import") if a.len() == 4 => {
            let rows = table(
                &a[3],
                &["source", "event", "time", "sense", "relation", "object"],
                lexicon::MAX_FACTS,
            )?;
            let mut identities = BTreeMap::new();
            let mut observations = Vec::new();
            for (i, r) in rows.iter().enumerate() {
                let relation = match r[4].as_str() {
                    "word" => Relation::Word,
                    "definition" => Relation::Definition,
                    "is-a" => Relation::IsA,
                    "not-a" => Relation::NotA,
                    _ => return Err(invalid(i + 2, "dictionary relation")),
                };
                let (key, value) = lexicon::fact(&a[2], &r[3], relation, &r[5])?;
                let stamp = EpisodeStamp::new(
                    &r[0],
                    &r[1],
                    r[2].parse()
                        .map_err(|_| invalid(i + 2, "dictionary time"))?,
                );
                if r[0].is_empty() || r[1].is_empty() {
                    return Err(invalid(i + 2, "source and event required"));
                }
                let identity = (r[0].clone(), r[1].clone());
                if let Some(previous) =
                    identities.insert(identity, (stamp.clone(), key.clone(), value.clone()))
                {
                    if previous != (stamp, key, value) {
                        return Err(invalid(i + 2, "source event collision"));
                    }
                } else {
                    observations.push((stamp, key, value));
                }
            }
            let (mut store, _) = EpisodeStore::open_with_limits(&a[1], lexicon::store_limits())?;
            LexiconView::snapshot(&store, &a[2], None, None)?;
            let context = format!("lexicon/{}", a[2]);
            let mut keys = BTreeSet::new();
            let mut recorded = BTreeMap::new();
            for id in 1..=store.len() as u64 {
                let episode = store.episode(id)?;
                recorded.insert(
                    (
                        episode.stamp().source.as_str(),
                        episode.stamp().event.as_str(),
                    ),
                    episode,
                );
                if let crate::EpisodeData::Fact { key, .. } = episode.data()
                    && key.context == context
                {
                    keys.insert(key.clone());
                }
            }
            let mut pending = Vec::new();
            for (stamp, key, value) in observations {
                let data = crate::EpisodeData::Fact {
                    key: key.clone(),
                    value: value.clone(),
                    replaces: None,
                };
                if let Some(previous) = recorded.get(&(stamp.source.as_str(), stamp.event.as_str()))
                {
                    if previous.stamp() != &stamp || previous.data() != &data {
                        return Err(crate::EpisodeError::EventCollision.into());
                    }
                    continue;
                }
                store.check_record(&stamp, &data)?;
                keys.insert(key.clone());
                pending.push((stamp, key, value));
            }
            if keys.len() > lexicon::MAX_FACTS
                || store.len().saturating_add(pending.len()) > lexicon::MAX_EPISODES
            {
                return Err(invalid(0, "dictionary capacity"));
            }
            let mut inserted = 0;
            for (stamp, key, value) in pending {
                inserted += usize::from(!store.observe(stamp, key, value)?.duplicate);
            }
            let view = LexiconView::snapshot(&store, &a[2], None, None)?;
            Ok(format!(
                "dictionary imported / new={inserted} / facts={} / journal={}\n",
                view.facts,
                store.len()
            ))
        }
        Some("dictionary-query") if a.len() == 5 => {
            let store = existing(&a[1])?;
            let view = LexiconView::snapshot(&store, &a[2], None, None)?;
            let resolution = view.resolve(&a[3]);
            let mut text = format!(
                "word={} / resolution={resolution:?} / known_at={}\n",
                a[3], view.known_at
            );
            if let Resolution::Unique(sense) = resolution {
                text.push_str(&format!(
                    "claim={sense} is-a {} / {:?}\n",
                    a[4],
                    view.classify(&sense, &a[4], lexicon::MAX_VISITS)
                ));
                if let Some(definition) = view.definition(&sense) {
                    text.push_str(&format!("definition={definition:?}\n"));
                }
            }
            Ok(text)
        }
        Some("language-answer") if a.len() == 6 => {
            let store = existing(&a[1])?;
            let view = LexiconView::snapshot(&store, &a[2], None, None)?;
            let mut examples = BTreeMap::<String, Vec<(String, String, String)>>::new();
            for r in table(&a[3], &["frame", "sentence", "word", "values"], 128)? {
                examples.entry(r[0].clone()).or_default().push((
                    r[1].clone(),
                    r[2].clone(),
                    r[3].clone(),
                ));
            }
            if examples.len() > 4 {
                return Err(invalid(0, "frame count limit"));
            }
            let frames = examples
                .values()
                .map(|e| LanguageFrame::learn(e).map_err(|_| invalid(0, "frame learning failed")))
                .collect::<Result<Vec<_>, _>>()?;
            let mut senses = BTreeSet::new();
            let mut tools = Vec::new();
            for r in table(&a[4], &["sense", "context", "file"], 16)? {
                if !senses.insert(r[0].clone()) {
                    return Err(invalid(0, "duplicate tool sense"));
                }
                let file = Path::new(&a[4])
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(&r[2]);
                tools.push(GroundedTool {
                    sense: r[0].clone(),
                    skill: NumericSkill::open(&file, &r[1])?,
                });
            }
            Ok(match answer(&view, &frames, &tools, &a[5]) {
                Ok(result) => format!(
                    "answer={} / word={} / sense={} / evidence={:?} / tool={:016X} / parsers={:016X},{:016X}\n",
                    result.value,
                    result.word,
                    result.sense,
                    result.evidence,
                    result.tool,
                    result.parsers.0,
                    result.parsers.1
                ),
                Err(reason) => format!("refused / {reason:?}\n"),
            })
        }
        _ => Err(CliError::Usage),
    }
}
