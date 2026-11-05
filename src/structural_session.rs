//! Personal grounding is separate from inherited rules and never becomes code.
use super::{
    Error, StructuralSeed, Term,
    agent::{Agent, Budget, Environment, Event, valid_name},
    algorithm::Value as DataValue,
    language::DEFAULT_WORK,
};
use crate::{BeliefStatus, EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue, FactKey};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

const CONTEXT: &str = "structural/quantity-session-v1";
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quantity {
    pub value: i64,
    pub unit: String,
}
impl Quantity {
    fn validate(&self) -> Result<(), Error> {
        if self.unit.is_empty()
            || self.unit.len() > 32
            || !self
                .unit
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_/".contains(&b))
        {
            return Err(Error::Invalid("quantity unit"));
        }
        Ok(())
    }
}
/// Host-authorized immutable data. Its unit is supplied, not inferred from cell text.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredData {
    pub value: DataValue,
    pub unit: String,
}
impl MeasuredData {
    pub fn validate(&self) -> Result<(), Error> {
        self.value.validate()?;
        Quantity {
            value: 0,
            unit: self.unit.clone(),
        }
        .validate()
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reply {
    pub status: String,
    pub value: Option<Quantity>,
    pub evidence: Vec<u64>,
    pub meanings: usize,
    pub work: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<String>,
    /// Candidate plans are exposed only when the session cannot choose safely.
    /// They are structural evidence for clarification, never an authorization.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Decision {
    pub plans: Vec<Term>,
    pub grounded: usize,
    pub evidence: Vec<u64>,
    pub work: usize,
    pub error: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Key {
    who: String,
    predicate: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Value {
    plan: Term,
    unit: String,
    inputs: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Prepared {
    Remember(Key, Quantity),
    Forget(Key),
    Focus(Vec<String>),
    Read(Value),
}

pub struct Session {
    agent: Agent,
    store: EpisodeStore,
    use_memory: bool,
    use_composition: bool,
}
fn storage(e: impl std::fmt::Display) -> Error {
    Error::Storage(e.to_string())
}
fn sym(t: &Term) -> Result<&str, Error> {
    if let Term::Symbol(s) = t {
        Ok(s)
    } else {
        Err(Error::Unsupported)
    }
}
fn key(t: &Term) -> Result<Key, Error> {
    let Term::Tree(v) = t else {
        return Err(Error::Unsupported);
    };
    if v.len() != 3 || sym(&v[0])? != "ref" {
        return Err(Error::Unsupported);
    }
    Ok(Key {
        who: sym(&v[1])?.into(),
        predicate: sym(&v[2])?.into(),
    })
}
fn quantity(t: &Term) -> Result<Quantity, Error> {
    let Term::Tree(v) = t else {
        return Err(Error::Unsupported);
    };
    if v.len() != 3 || sym(&v[0])? != "quantity" {
        return Err(Error::Unsupported);
    }
    let Term::Integer(value) = v[1] else {
        return Err(Error::Unsupported);
    };
    let q = Quantity {
        value,
        unit: sym(&v[2])?.into(),
    };
    q.validate()?;
    Ok(q)
}
impl Session {
    pub fn open(seed: StructuralSeed, path: &Path, budget: Budget) -> Result<Self, Error> {
        let (store, _) = EpisodeStore::open(path).map_err(storage)?;
        Ok(Self {
            agent: Agent::new(seed, budget)?,
            store,
            use_memory: true,
            use_composition: true,
        })
    }
    /// Ablations disable only use of evidence/composition, not safeguards.
    pub fn set_controls(&mut self, memory: bool, composition: bool) {
        self.use_memory = memory;
        self.use_composition = composition;
    }
    pub fn seed(&self) -> &StructuralSeed {
        self.agent.seed()
    }
    pub fn into_seed(self) -> StructuralSeed {
        self.agent.into_seed()
    }
    pub fn episodes(&self) -> usize {
        self.store.len()
    }
    pub fn queries(&self) -> usize {
        self.agent.statistics.queries
    }
    fn focus(&self) -> Result<Vec<String>, Error> {
        if !self.use_memory {
            return Ok(vec![]);
        }
        Ok(self
            .store
            .traces(CONTEXT, "focus")
            .map_err(storage)?
            .last()
            .and_then(|e| match e.data() {
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
            .unwrap_or_default())
    }
    fn read(&self, key: &Key, evidence: &mut Vec<u64>) -> Result<Quantity, Error> {
        if !self.use_memory {
            return Err(Error::Unsupported);
        }
        let belief = self
            .store
            .belief(&FactKey::new(&key.who, &key.predicate, CONTEXT), None, None)
            .map_err(storage)?;
        evidence.extend(belief.evidence.iter().map(|e| e.episode_id));
        match belief.status {
            BeliefStatus::Conflict => Err(Error::Invalid("session source conflict")),
            BeliefStatus::Missing => Err(Error::Unsupported),
            BeliefStatus::Consistent => {
                let EpisodeValue::Text(raw) = &belief.evidence[0].value else {
                    return Err(Error::Invalid("quantity memory type"));
                };
                let q: Quantity = serde_json::from_str(raw).map_err(storage)?;
                q.validate()?;
                Ok(q)
            }
        }
    }
    fn resolve(
        &self,
        t: &Term,
        evidence: &mut Vec<u64>,
        data: &BTreeMap<String, MeasuredData>,
        depth: usize,
    ) -> Result<Value, Error> {
        if depth > 32 {
            return Err(Error::Budget);
        }
        let Term::Tree(v) = t else {
            return Err(Error::Unsupported);
        };
        match sym(&v[0])? {
            "quantity" => {
                let q = quantity(t)?;
                Ok(Value {
                    plan: Term::Integer(q.value),
                    unit: q.unit,
                    inputs: vec![],
                })
            }
            "ref" => {
                let q = self.read(&key(t)?, evidence)?;
                Ok(Value {
                    plan: Term::Integer(q.value),
                    unit: q.unit,
                    inputs: vec![],
                })
            }
            "focus-ref" if v.len() == 2 => {
                let focus = self.focus()?;
                if focus.len() != 1 {
                    return Err(Error::Ambiguous);
                }
                let q = self.read(
                    &Key {
                        who: focus[0].clone(),
                        predicate: sym(&v[1])?.into(),
                    },
                    evidence,
                )?;
                Ok(Value {
                    plan: Term::Integer(q.value),
                    unit: q.unit,
                    inputs: vec![],
                })
            }
            "data-quantity" if v.len() == 3 => {
                let name = sym(&v[1])?;
                let input = data
                    .get(name)
                    .ok_or(Error::Invalid("session missing data"))?;
                // One explicitly named data source per quantity prevents implicit unit mixing.
                let (value, inputs) = self.agent.answer_value_plan_with_inputs(
                    &v[2],
                    &BTreeMap::from([(name.to_owned(), input.value.clone())]),
                )?;
                let DataValue::Integer(value) = value else {
                    return Err(Error::Invalid("session scalar data required"));
                };
                if inputs != [name] {
                    return Err(Error::Invalid("session scalar data required"));
                }
                Ok(Value {
                    plan: Term::Integer(value),
                    unit: input.unit.clone(),
                    inputs,
                })
            }
            "measure-call" if (3..=6).contains(&v.len()) => {
                let name = sym(&v[1])?;
                let args = v[2..]
                    .iter()
                    .map(|a| self.resolve(a, evidence, data, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?;
                let unit = args[0].unit.clone();
                if args.iter().any(|a| a.unit != unit) {
                    return Err(Error::Invalid("session unit mismatch"));
                }
                let mut plan = vec![Term::Symbol("invoke".into()), Term::Symbol(name.into())];
                let mut inputs = args
                    .iter()
                    .flat_map(|a| a.inputs.iter().cloned())
                    .collect::<Vec<_>>();
                inputs.sort_unstable();
                inputs.dedup();
                plan.extend(args.into_iter().map(|a| a.plan));
                Ok(Value {
                    plan: Term::Tree(plan),
                    unit,
                    inputs,
                })
            }
            _ => Err(Error::Unsupported),
        }
    }
    fn prepare(
        &self,
        t: &Term,
        evidence: &mut Vec<u64>,
        data: &BTreeMap<String, MeasuredData>,
    ) -> Result<Prepared, Error> {
        let Term::Tree(v) = t else {
            return Err(Error::Unsupported);
        };
        match sym(&v[0])? {
            "remember" if v.len() == 3 => Ok(Prepared::Remember(key(&v[1])?, quantity(&v[2])?)),
            "forget" if v.len() == 2 => Ok(Prepared::Forget(key(&v[1])?)),
            "focus" if (2..=5).contains(&v.len()) => {
                let mut owners = Vec::new();
                for t in &v[1..] {
                    let k = key(t)?;
                    self.read(&k, evidence)?;
                    if !owners.contains(&k.who) {
                        owners.push(k.who);
                    }
                }
                Ok(Prepared::Focus(owners))
            }
            _ => self.resolve(t, evidence, data, 0).map(Prepared::Read),
        }
    }
    pub fn turn(
        &mut self,
        text: &str,
        source: &str,
        environment: &mut impl Environment,
        decide: &mut impl FnMut(&Decision) -> Result<(), Error>,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Reply, Error> {
        self.turn_with_data_inner(
            text,
            source,
            &BTreeMap::new(),
            None,
            environment,
            decide,
            emit,
        )
    }
    pub fn turn_with_data(
        &mut self,
        text: &str,
        source: &str,
        data: &BTreeMap<String, MeasuredData>,
        environment: &mut impl Environment,
        decide: &mut impl FnMut(&Decision) -> Result<(), Error>,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Reply, Error> {
        self.turn_with_data_inner(text, source, data, None, environment, decide, emit)
    }
    /// Re-evaluates the current utterance and executes only a previously exposed
    /// structural alternative after the caller confirms it.
    pub fn turn_with_alternative(
        &mut self,
        text: &str,
        source: &str,
        alternative: &str,
        environment: &mut impl Environment,
        decide: &mut impl FnMut(&Decision) -> Result<(), Error>,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Reply, Error> {
        self.turn_with_data_inner(
            text,
            source,
            &BTreeMap::new(),
            Some(alternative),
            environment,
            decide,
            emit,
        )
    }
    pub fn turn_with_data_alternative(
        &mut self,
        text: &str,
        source: &str,
        data: &BTreeMap<String, MeasuredData>,
        alternative: &str,
        environment: &mut impl Environment,
        decide: &mut impl FnMut(&Decision) -> Result<(), Error>,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Reply, Error> {
        self.turn_with_data_inner(
            text,
            source,
            data,
            Some(alternative),
            environment,
            decide,
            emit,
        )
    }
    fn turn_with_data_inner(
        &mut self,
        text: &str,
        source: &str,
        data: &BTreeMap<String, MeasuredData>,
        alternative: Option<&str>,
        environment: &mut impl Environment,
        decide: &mut impl FnMut(&Decision) -> Result<(), Error>,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Reply, Error> {
        if data.len() > 16 {
            return Err(Error::Budget);
        }
        let mut nodes = 0usize;
        for (name, value) in data {
            if !valid_name(name) {
                return Err(Error::Invalid("session data name"));
            }
            value.validate()?;
            nodes += value.value.nodes();
            if nodes > 8192 {
                return Err(Error::Budget);
            }
        }
        if source.is_empty() || source.len() > 128 || source.chars().any(char::is_control) {
            return Err(Error::Invalid("session source"));
        }
        let report = if self.use_composition {
            self.agent.seed().measure_composition(text, DEFAULT_WORK)
        } else {
            let result = self
                .agent
                .seed()
                .transform(&super::agent::utterance(text)?, 512)
                .map(|p| if p.rules.is_empty() { vec![] } else { vec![p] });
            super::language::Composition {
                visits: result
                    .as_ref()
                    .map_or(0, |v| v.iter().map(|p| p.visits).sum()),
                result,
            }
        };
        let predictions = match report.result {
            Ok(p) => p,
            Err(Error::Unsupported) => vec![],
            Err(e) => {
                decide(&Decision {
                    plans: vec![],
                    grounded: 0,
                    evidence: vec![],
                    work: report.visits,
                    error: Some(e.to_string()),
                })?;
                return Err(e);
            }
        };
        let selected = if let Some(candidate) = alternative {
            if candidate.is_empty() || candidate.len() > 4096 {
                return Err(Error::Invalid("session alternative"));
            }
            let mut matches = predictions.iter().filter(|prediction| {
                serde_json::to_string(&prediction.output)
                    .map(|encoded| encoded == candidate)
                    .unwrap_or(false)
            });
            let selected = matches.next();
            if selected.is_none() || matches.next().is_some() {
                return Err(Error::Invalid("session alternative"));
            }
            selected
        } else {
            None
        };
        let active_predictions: Vec<_> = selected
            .into_iter()
            .chain(if alternative.is_none() {
                predictions.iter().collect::<Vec<_>>().into_iter()
            } else {
                Vec::new().into_iter()
            })
            .collect();
        let mut prepared: Vec<(Prepared, Vec<u64>)> = Vec::new();
        let mut evidence = Vec::new();
        let mut issue = None;
        for p in &active_predictions {
            let mut proof = Vec::new();
            let result = self.prepare(&p.output, &mut proof, data);
            evidence.extend(proof.iter().copied());
            match result {
                Ok(p) => {
                    if let Some((_, previous)) = prepared.iter_mut().find(|(old, _)| *old == p) {
                        previous.extend(proof);
                    } else {
                        prepared.push((p, proof));
                    }
                }
                Err(Error::Unsupported) => (),
                Err(Error::Ambiguous) => issue = issue.or(Some("clarify")),
                Err(Error::Invalid("session source conflict")) => issue = Some("conflict"),
                Err(Error::Invalid("session unit mismatch")) => {
                    if issue != Some("conflict") {
                        issue = Some("unit-mismatch");
                    }
                }
                Err(Error::Invalid("session missing data")) => {
                    issue = issue.or(Some("missing-data"))
                }
                Err(Error::Invalid("session scalar data required")) => {
                    issue = issue.or(Some("clarify-data"))
                }
                Err(e) => return Err(e),
            }
        }
        evidence.sort_unstable();
        evidence.dedup();
        decide(&Decision {
            plans: active_predictions.iter().map(|p| p.output.clone()).collect(),
            grounded: prepared.len(),
            evidence: evidence.clone(),
            work: report.visits,
            error: None,
        })?;
        let mut reply = Reply {
            status: "unknown".into(),
            value: None,
            evidence,
            meanings: active_predictions.len(),
            work: report.visits,
            inputs: vec![],
            alternatives: if alternative.is_none() && predictions.len() > 1 {
                predictions
                    .iter()
                    .map(|prediction| {
                        serde_json::to_string(&prediction.output)
                            .map_err(|e| Error::Storage(e.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            },
        };
        if let Some(status) = issue {
            reply.status = status.into();
            return Ok(reply);
        }
        if prepared.len() > 1 {
            reply.status = "clarify".into();
            return Ok(reply);
        }
        let Some((prepared, proof)) = prepared.pop() else {
            return Ok(reply);
        };
        reply.evidence = proof;
        let next = self.store.next_id();
        match prepared {
            Prepared::Remember(k, q) => {
                let raw = serde_json::to_string(&q).map_err(storage)?;
                let key = FactKey::new(k.who, k.predicate, CONTEXT);
                let belief = self.store.belief(&key, None, None).map_err(storage)?;
                let own = belief
                    .evidence
                    .iter()
                    .filter(|e| e.source == source)
                    .collect::<Vec<_>>();
                if own.len() > 1 {
                    return Err(Error::Ambiguous);
                }
                let receipt = if let Some(old) = own.first() {
                    self.store
                        .revise(
                            old.episode_id,
                            source,
                            format!("quantity-{next}"),
                            EpisodeValue::Text(raw),
                        )
                        .map_err(storage)?
                } else {
                    self.store
                        .observe(
                            EpisodeStamp::new(source, format!("quantity-{next}"), next),
                            key,
                            EpisodeValue::Text(raw),
                        )
                        .map_err(storage)?
                };
                reply.status = "recorded".into();
                reply.value = Some(q);
                reply.evidence.push(receipt.id);
            }
            Prepared::Forget(k) => {
                let b = self
                    .store
                    .belief(&FactKey::new(k.who, k.predicate, CONTEXT), None, None)
                    .map_err(storage)?;
                let own = b
                    .evidence
                    .iter()
                    .filter(|e| e.source == source)
                    .collect::<Vec<_>>();
                if own.len() != 1 {
                    return Ok(reply);
                }
                let receipt = self
                    .store
                    .retract(own[0].episode_id, source, format!("forget-{next}"))
                    .map_err(storage)?;
                reply.status = "retracted".into();
                reply.evidence.extend([own[0].episode_id, receipt.id]);
            }
            Prepared::Focus(owners) => {
                self.store
                    .record(
                        EpisodeStamp::new(source, format!("focus-{next}"), next),
                        EpisodeData::Trace {
                            context: CONTEXT.into(),
                            channel: "focus".into(),
                            fields: owners
                                .into_iter()
                                .enumerate()
                                .map(|(i, who)| (i.to_string(), EpisodeValue::Text(who)))
                                .collect::<BTreeMap<_, _>>(),
                        },
                    )
                    .map_err(storage)?;
                reply.status = "context".into();
            }
            Prepared::Read(v) => {
                let value = self.agent.answer_plan(&v.plan, environment, emit)?;
                reply.status = "answer".into();
                reply.inputs = v.inputs;
                reply.value = Some(Quantity {
                    value,
                    unit: v.unit,
                });
            }
        }
        reply.evidence.sort_unstable();
        reply.evidence.dedup();
        Ok(reply)
    }
}
