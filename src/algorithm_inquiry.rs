//! Pool-based questions use unlabeled inputs; admission remains a separate agent gate.
use super::{Error, Example, Program, SearchBudget, Value, search};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InquiryStrategy {
    Disagreement,
    Passive,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InquiryPlan {
    pub inputs: Vec<Value>,
    pub max_queries: usize,
    pub max_hypotheses: usize,
    pub strategy: InquiryStrategy,
}

#[derive(Clone, Debug, Serialize)]
pub struct InquiryEvent {
    pub phase: String,
    pub input: Option<Value>,
    pub predictions: Vec<Value>,
    pub actual: Option<Value>,
    pub hypotheses: usize,
    pub survivors: usize,
    pub disagreement_pairs: usize,
    pub queries: usize,
    pub candidates: u64,
    pub work: u64,
}

#[derive(Debug, Serialize)]
pub struct InquiryReceipt {
    pub examples: Vec<Example>,
    pub strategy: InquiryStrategy,
    pub queries: usize,
    pub hypotheses: usize,
    pub survivors: usize,
    pub candidates: u64,
    pub work: u64,
    pub templates: usize,
    pub mining_work: u64,
    pub sample_budget_exhausted: bool,
    pub stop: String,
}

impl InquiryPlan {
    pub fn validate(&self, examples: &[Example]) -> Result<(), Error> {
        if !(3..=24).contains(&examples.len())
            || self.inputs.is_empty()
            || self.inputs.len() > 64
            || !(1..=4).contains(&self.max_queries)
            || self.max_queries > self.inputs.len()
            || self.max_queries + examples.len() > 24
            || !(2..=16).contains(&self.max_hypotheses)
            || self.inputs.iter().map(Value::nodes).sum::<usize>() > 8192
        {
            return Err(Error::Invalid("algorithm inquiry bounds"));
        }
        let mut inputs = BTreeSet::new();
        for e in examples {
            e.input.validate()?;
            e.output.validate()?;
            if !inputs.insert(&e.input) {
                return Err(Error::Invalid("duplicate algorithm example"));
            }
        }
        for input in &self.inputs {
            input.validate()?;
            if !inputs.insert(input) {
                return Err(Error::Invalid("algorithm inquiry overlaps teaching"));
            }
        }
        Ok(())
    }
}

pub fn inquire(
    examples: &[Example],
    library: &[Program],
    budget: SearchBudget,
    plan: &InquiryPlan,
    oracle: &mut impl FnMut(&Value) -> Result<Value, Error>,
    emit: &mut impl FnMut(&InquiryEvent) -> Result<(), Error>,
) -> Result<InquiryReceipt, Error> {
    plan.validate(examples)?;
    budget.policy.validate()?;
    if library.len() > 32
        || budget.candidates < 2
        || budget.candidates > 100000
        || budget.work < 2
        || budget.work > 100_000_000
    {
        return Err(Error::Invalid("algorithm inquiry search bounds"));
    }
    let mut receipt = InquiryReceipt {
        examples: examples.to_vec(),
        strategy: plan.strategy,
        queries: 0,
        hypotheses: 0,
        survivors: 0,
        candidates: 0,
        work: 0,
        templates: 0,
        mining_work: 0,
        sample_budget_exhausted: false,
        stop: "query-cap".into(),
    };
    let mut sample = Vec::new();
    let mut survivors = Vec::new();
    let mut queried = BTreeSet::new();
    let mut meter = super::Meter::new(budget.work / 2);
    let result = (|| -> Result<(), Error> {
        for round in 0..plan.max_queries {
            if plan.strategy == InquiryStrategy::Disagreement {
                let candidates = (budget.candidates / 2)
                    .saturating_sub(receipt.candidates as usize)
                    / (plan.max_queries - round);
                let work = meter.limit.saturating_sub(meter.work) / 2;
                if candidates == 0 || work == 0 {
                    receipt.stop = "inquiry-budget".into();
                    break;
                }
                // Rebuild after every answer: a survivor of a finite sample is not certainty.
                let found = search::hypotheses(
                    &receipt.examples,
                    library,
                    SearchBudget {
                        candidates,
                        work,
                        ..budget
                    },
                    &plan.inputs,
                    plan.max_hypotheses,
                )
                .map_err(|f| f.error)?;
                receipt.candidates += found.candidates;
                meter.charge(found.work)?;
                receipt.templates += found.templates;
                receipt.mining_work += found.mining_work;
                receipt.sample_budget_exhausted |= found.budget_exhausted;
                sample = found.hypotheses;
                receipt.hypotheses = sample.len();
                survivors = (0..sample.len()).collect();
            }
            let mut best = None;
            let mut best_pairs = 0;
            for (index, input) in plan.inputs.iter().enumerate() {
                if queried.contains(&index) {
                    continue;
                }
                let mut pairs = 0;
                for (i, &a) in survivors.iter().enumerate() {
                    for &b in &survivors[i + 1..] {
                        meter.charge((1 + sample[a].1[index].nodes()) as u64)?;
                        if sample[a].1[index] != sample[b].1[index] {
                            pairs += 1;
                        }
                    }
                }
                meter.charge(input.nodes() as u64)?;
                if best.is_none() || pairs > best_pairs {
                    best = Some(index);
                    best_pairs = pairs;
                }
                if plan.strategy == InquiryStrategy::Passive {
                    break;
                }
            }
            if plan.strategy == InquiryStrategy::Disagreement && best_pairs == 0 {
                receipt.stop = if survivors.is_empty() {
                    "sample-empty"
                } else {
                    "sample-consensus"
                }
                .into();
                break;
            }
            let Some(index) = best else { break };
            let input = &plan.inputs[index];
            let mut predictions = BTreeSet::new();
            for &member in &survivors {
                meter.charge(sample[member].1[index].nodes() as u64)?;
                predictions.insert(sample[member].1[index].clone());
            }
            emit(&InquiryEvent {
                phase: "query_selected".into(),
                input: Some(input.clone()),
                predictions: predictions.into_iter().collect(),
                actual: None,
                hypotheses: sample.len(),
                survivors: survivors.len(),
                disagreement_pairs: best_pairs,
                queries: receipt.queries,
                candidates: receipt.candidates,
                work: meter.work,
            })?;
            // The callback can see this one selected input, not the committee or admission labels.
            let actual = oracle(input)?;
            actual.validate()?;
            meter.charge(actual.nodes() as u64)?;
            survivors.retain(|&member| sample[member].1[index] == actual);
            queried.insert(index);
            receipt.queries += 1;
            emit(&InquiryEvent {
                phase: "query_observed".into(),
                input: Some(input.clone()),
                predictions: Vec::new(),
                actual: Some(actual.clone()),
                hypotheses: sample.len(),
                survivors: survivors.len(),
                disagreement_pairs: best_pairs,
                queries: receipt.queries,
                candidates: receipt.candidates,
                work: meter.work,
            })?;
            receipt.examples.push(Example {
                input: input.clone(),
                output: actual,
            });
        }
        Ok(())
    })();
    receipt.work = meter.work;
    receipt.survivors = survivors.len();
    emit(&InquiryEvent {
        phase: if result.is_ok() {
            "inquiry_finished"
        } else {
            "inquiry_failed"
        }
        .into(),
        input: None,
        predictions: Vec::new(),
        actual: None,
        hypotheses: receipt.hypotheses,
        survivors: receipt.survivors,
        disagreement_pairs: 0,
        queries: receipt.queries,
        candidates: receipt.candidates,
        work: receipt.work,
    })?;
    result?;
    Ok(receipt)
}
