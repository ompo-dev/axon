//! Dialogue action semantics and functional facts are declared host interfaces.
//! Both utterance decoding and answer wording come from induced rules.
use super::{Error, StructuralSeed, Term};
use crate::{BeliefStatus, EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue, FactKey};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
const CONTEXT: &str = "structural/dialogue-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reply {
    pub status: String,
    pub text: Option<String>,
}
pub struct Dialogue {
    store: EpisodeStore,
    focus: Vec<String>,
}
fn storage(e: impl std::fmt::Display) -> Error {
    Error::Storage(e.to_string())
}
fn symbol(t: &Term) -> Result<&str, Error> {
    if let Term::Symbol(s) = t {
        Ok(s)
    } else {
        Err(Error::Unsupported)
    }
}
impl Dialogue {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let (store, _) = EpisodeStore::open(path).map_err(storage)?;
        let focus = store
            .traces(CONTEXT, "focus")
            .map_err(storage)?
            .last()
            .and_then(|e| match &e.data() {
                EpisodeData::Trace { fields, .. } => Some(
                    fields
                        .values()
                        .filter_map(|v| {
                            if let EpisodeValue::Text(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        Ok(Self { store, focus })
    }
    pub fn episodes(&self) -> usize {
        self.store.len()
    }
    fn fact(&self, who: &str, predicate: &str) -> Result<String, Error> {
        let b = self
            .store
            .belief(&FactKey::new(who, predicate, CONTEXT), None, None)
            .map_err(storage)?;
        match b.status {
            BeliefStatus::Consistent => match &b.evidence[0].value {
                EpisodeValue::Text(s) => Ok(s.clone()),
                _ => Err(Error::Unsupported),
            },
            BeliefStatus::Conflict => Err(Error::Ambiguous),
            _ => Err(Error::Unsupported),
        }
    }
    fn remember_focus(&mut self, owners: Vec<String>) -> Result<(), Error> {
        let next = self.store.next_id();
        self.store
            .record(
                EpisodeStamp::new("dialogue-runtime", format!("focus-{next}"), next),
                EpisodeData::Trace {
                    context: CONTEXT.into(),
                    channel: "focus".into(),
                    fields: owners
                        .iter()
                        .enumerate()
                        .map(|(i, s)| (i.to_string(), EpisodeValue::Text(s.clone())))
                        .collect::<BTreeMap<_, _>>(),
                },
            )
            .map_err(storage)?;
        self.focus = owners;
        Ok(())
    }
    fn grounded_query(&self, t: &Term) -> bool {
        let Term::Tree(v) = t else { return false };
        let Ok(command) = symbol(&v[0]) else {
            return false;
        };
        let (mut who, predicates) = match command {
            "refer" if self.focus.len() == 1 => (self.focus[0].clone(), &v[1..]),
            "ask" | "path" if v.len() >= 3 => {
                let Ok(s) = symbol(&v[1]) else { return false };
                (s.to_owned(), &v[2..])
            }
            _ => return false,
        };
        if predicates.is_empty() || predicates.len() > 3 {
            return false;
        }
        for p in predicates {
            let Ok(p) = symbol(p) else { return false };
            let Ok(next) = self.fact(&who, p) else {
                return false;
            };
            who = next;
        }
        true
    }
    pub fn turn(
        &mut self,
        seed: &StructuralSeed,
        text: &str,
        source: &str,
    ) -> Result<Reply, Error> {
        if text.is_empty() || text.len() > 1024 || source.is_empty() || source.len() > 128 {
            return Err(Error::Invalid("dialogue input bounds"));
        }
        let mut words = vec![Term::Symbol("utterance".into())];
        words.extend(
            text.trim()
                .trim_end_matches(['.', '?', '!'])
                .split_whitespace()
                .map(|s| Term::Symbol(s.to_lowercase())),
        );
        let input = Term::Tree(words);
        let parsed = match seed.transform(&input, 512) {
            Ok(p) if !p.rules.is_empty() => p.output,
            Ok(_) => return Ok(reply("unknown", None)),
            Err(Error::Ambiguous) => {
                let candidates = match seed.interpretations(&input) {
                    Ok(c) => c,
                    Err(Error::Ambiguous) => return Ok(reply("clarify", None)),
                    Err(e) => return Err(e),
                };
                let mut grounded = candidates.into_iter().filter(|t| self.grounded_query(t));
                let Some(first) = grounded.next() else {
                    return Ok(reply("clarify", None));
                };
                if grounded.next().is_some() {
                    return Ok(reply("clarify", None));
                }
                first
            }
            Err(e) => return Err(e),
        };
        let Term::Tree(v) = parsed else {
            return Ok(reply("unknown", None));
        };
        let command = symbol(&v[0])?;
        match command {
            "tell" if v.len() == 4 => {
                let who = symbol(&v[1])?;
                let pred = symbol(&v[2])?;
                let value = symbol(&v[3])?;
                let next = self.store.next_id();
                self.store
                    .observe(
                        EpisodeStamp::new(source, format!("fact-{next}"), next),
                        FactKey::new(who, pred, CONTEXT),
                        EpisodeValue::Text(value.into()),
                    )
                    .map_err(storage)?;
                self.remember_focus(vec![who.into()])?;
                Ok(reply("recorded", None))
            }
            "focus" if v.len() >= 2 && v.len() <= 5 => {
                let owners = v[1..]
                    .iter()
                    .map(|t| symbol(t).map(str::to_owned))
                    .collect::<Result<Vec<_>, _>>()?;
                self.remember_focus(owners)?;
                Ok(reply("context", None))
            }
            "retract" if v.len() == 4 => {
                let who = symbol(&v[1])?;
                let pred = symbol(&v[2])?;
                let value = symbol(&v[3])?;
                let b = self
                    .store
                    .belief(&FactKey::new(who, pred, CONTEXT), None, None)
                    .map_err(storage)?;
                let targets = b
                    .evidence
                    .iter()
                    .filter(|e| e.source == source && e.value == EpisodeValue::Text(value.into()))
                    .map(|e| e.episode_id)
                    .collect::<Vec<_>>();
                if targets.is_empty() {
                    return Ok(reply("unknown", None));
                }
                for id in targets {
                    let next = self.store.next_id();
                    self.store
                        .retract(id, source, format!("retract-{next}"))
                        .map_err(storage)?;
                }
                Ok(reply("retracted", None))
            }
            "ask" | "refer" | "path" => {
                let who = if command == "refer" {
                    if self.focus.len() != 1 {
                        return Ok(reply("clarify", None));
                    }
                    self.focus[0].clone()
                } else {
                    if v.len() < 3 {
                        return Ok(reply("unknown", None));
                    }
                    symbol(&v[1])?.into()
                };
                let predicates = if command == "refer" { &v[1..] } else { &v[2..] };
                if predicates.is_empty() || predicates.len() > 3 {
                    return Ok(reply("unknown", None));
                }
                let mut value = who.clone();
                let mut owner = who.clone();
                for predicate in predicates {
                    owner = value.clone();
                    value = match self.fact(&value, symbol(predicate)?) {
                        Ok(s) => s,
                        Err(Error::Ambiguous) => return Ok(reply("conflict", None)),
                        Err(Error::Unsupported) => return Ok(reply("unknown", None)),
                        Err(e) => return Err(e),
                    };
                }
                let pred = symbol(predicates.last().expect("nonempty"))?;
                let rendered = seed.transform(
                    &Term::Tree(vec![
                        Term::Symbol("reply".into()),
                        Term::Symbol(owner),
                        Term::Symbol(pred.into()),
                        Term::Symbol(value),
                    ]),
                    512,
                )?;
                if let Term::Tree(words) = rendered.output
                    && words.first() == Some(&Term::Symbol("say".into()))
                {
                    return Ok(reply(
                        "answer",
                        Some(
                            words[1..]
                                .iter()
                                .map(symbol)
                                .collect::<Result<Vec<_>, _>>()?
                                .join(" "),
                        ),
                    ));
                }
                Ok(reply("unknown", None))
            }
            _ => Ok(reply("unknown", None)),
        }
    }
}
fn reply(status: &str, text: Option<String>) -> Reply {
    Reply {
        status: status.into(),
        text,
    }
}
