use super::*;
use crate::{
    live_store::checksum,
    text_skill::{publish_new, read_bounded},
};
use std::{collections::BTreeSet, path::Path};
const MAGIC: &[u8; 8] = b"AXCS0002";
const MAX_FILE: u64 = 2097152;

impl ContinuousAgent {
    pub fn save_checkpoint_new(&self, path: &Path) -> Result<(), NumericError> {
        publish_new(path, &self.to_bytes()).map_err(Into::into)
    }
    pub fn open_checkpoint(
        path: &Path,
        context: &str,
        library: Option<u64>,
    ) -> Result<Self, NumericError> {
        let agent = Self::from_bytes(&read_bounded(path, MAX_FILE)?)?;
        if agent.context != context || agent.library != library {
            return Err(NumericError::Context);
        }
        Ok(agent)
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        b.extend((self.context.len() as u32).to_le_bytes());
        b.extend(self.context.as_bytes());
        b.push(self.arity as u8);
        option_u64(&mut b, self.library);
        let p = self.policy;
        for n in [
            p.training_window,
            p.min_training,
            p.validation_samples,
            p.validation_distinct,
            p.drift_window,
            p.drift_errors,
            p.pending_capacity,
            p.recent_capacity,
            p.archive_models,
            p.archive_bytes,
        ] {
            b.extend((n as u32).to_le_bytes());
        }
        for n in [
            p.max_label_delay,
            p.retry_stride,
            p.search.max_work,
            p.development.max_work,
            p.development.mining.max_visits,
        ] {
            b.extend(n.to_le_bytes());
        }
        for n in [
            p.search.max_cost,
            p.search.max_terms,
            p.development.max_attempts,
            p.development.mining.max_patterns,
            p.development.mining.max_modules,
            p.development.mining.max_body,
        ] {
            b.extend((n as u32).to_le_bytes());
        }
        b.extend(p.tolerance.absolute.to_le_bytes());
        b.extend(p.tolerance.relative.to_le_bytes());
        for n in [
            self.epoch,
            self.last_id,
            self.watermark,
            self.job_token,
            self.next_job_at,
        ] {
            b.extend(n.to_le_bytes());
        }
        b.extend(self.failures.to_le_bytes());
        b.push(u8::from(self.ready));
        let s = self.statistics;
        for n in [
            s.observations,
            s.predictions,
            s.feedback,
            s.scored,
            s.correct,
            s.expired,
            s.late_for_learning,
            s.drifts,
            s.jobs,
            s.failed_jobs,
            s.stale_jobs,
            s.rejected_candidates,
            s.promotions,
            s.reuses,
            s.search_work,
            s.archive_work,
        ] {
            b.extend(n.to_le_bytes());
        }
        option_u64(&mut b, self.temporal_target.map(|t| t as u64));
        option_u64(&mut b, self.temporal_last);
        model(&mut b, self.active.as_ref());
        b.push(u8::from(self.candidate.is_some()));
        if let Some(c) = &self.candidate {
            model(&mut b, Some(&c.model));
            b.extend(c.token.to_le_bytes());
            b.extend(c.through.to_le_bytes());
            b.extend((c.correct as u32).to_le_bytes());
            b.push(u8::from(c.reused));
            b.extend((c.distinct.len() as u32).to_le_bytes());
            for input in &c.distinct {
                input_bytes(&mut b, input);
            }
        }
        b.extend((self.pending.len() as u32).to_le_bytes());
        for p in &self.pending {
            pending_bytes(&mut b, p);
        }
        b.extend((self.recent.len() as u32).to_le_bytes());
        for receipt in &self.recent {
            pending_bytes(&mut b, &receipt.pending);
            option_number(&mut b, receipt.label);
            b.push(u8::from(receipt.expired));
        }
        b.extend((self.window.len() as u32).to_le_bytes());
        for e in &self.window {
            b.extend(e.sequence.to_le_bytes());
            input_bytes(&mut b, &e.input);
            b.extend(e.output.to_le_bytes());
        }
        b.extend((self.errors.len() as u32).to_le_bytes());
        for e in &self.errors {
            b.push(u8::from(*e));
        }
        b.extend((self.archive.len() as u32).to_le_bytes());
        for m in &self.archive {
            b.extend(m.identity.to_le_bytes());
            blob(&mut b, &m.bytes);
        }
        b.extend(checksum(&b).to_le_bytes());
        b
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NumericError> {
        if bytes.len() < 16
            || bytes.len() as u64 > MAX_FILE
            || (&bytes[..8] != MAGIC && &bytes[..8] != b"AXCS0001")
        {
            return Err(NumericError::Invalid("continuous checkpoint header"));
        }
        let end = bytes.len() - 8;
        if checksum(&bytes[..end])
            != u64::from_le_bytes(bytes[end..].try_into().expect("eight bytes"))
        {
            return Err(NumericError::Invalid("continuous checkpoint checksum"));
        }
        let mut r = Reader(&bytes[8..end]);
        let context = r.text()?;
        let arity = r.byte()? as usize;
        let library = read_option_u64(&mut r)?;
        let mut p = ContinuousPolicy {
            training_window: r.count(32)?,
            min_training: r.count(32)?,
            validation_samples: r.count(16)?,
            validation_distinct: r.count(16)?,
            drift_window: r.count(32)?,
            drift_errors: r.count(32)?,
            pending_capacity: r.count(1024)?,
            recent_capacity: r.count(1024)?,
            archive_models: r.count(64)?,
            archive_bytes: r.count(1048576)?,
            max_label_delay: r.u64()?,
            retry_stride: r.u64()?,
            ..ContinuousPolicy::default()
        };
        p.search.max_work = r.u64()?;
        p.development.max_work = r.u64()?;
        p.development.mining.max_visits = r.u64()?;
        p.search.max_cost = r.count(12)?;
        p.search.max_terms = r.count(200000)?;
        p.development.max_attempts = r.count(16)?;
        p.development.mining.max_patterns = r.count(16384)?;
        p.development.mining.max_modules = r.count(16)?;
        p.development.mining.max_body = r.count(64)?;
        p.tolerance = NumericTolerance {
            absolute: r.number()?,
            relative: r.number()?,
        };
        let mut a = Self::new(context, arity, library, p)?;
        a.epoch = r.u64()?;
        a.last_id = r.u64()?;
        a.watermark = r.u64()?;
        a.job_token = r.u64()?;
        a.next_job_at = r.u64()?;
        a.failures = r.u32()?;
        a.ready = boolean(&mut r)?;
        a.statistics = ContinuousStatistics {
            observations: r.u64()?,
            predictions: r.u64()?,
            feedback: r.u64()?,
            scored: r.u64()?,
            correct: r.u64()?,
            expired: r.u64()?,
            late_for_learning: r.u64()?,
            drifts: r.u64()?,
            jobs: r.u64()?,
            failed_jobs: r.u64()?,
            stale_jobs: r.u64()?,
            rejected_candidates: r.u64()?,
            promotions: r.u64()?,
            reuses: r.u64()?,
            search_work: r.u64()?,
            archive_work: r.u64()?,
        };
        if &bytes[..8] == MAGIC {
            a.temporal_target = read_option_u64(&mut r)?
                .map(|v| {
                    usize::try_from(v).map_err(|_| NumericError::Invalid("temporal target index"))
                })
                .transpose()?;
            a.temporal_last = read_option_u64(&mut r)?;
        }
        a.active = read_model(&mut r)?;
        if boolean(&mut r)? {
            let model = read_model(&mut r)?.ok_or(NumericError::Invalid("missing candidate"))?;
            let token = r.u64()?;
            let through = r.u64()?;
            let correct = r.count(p.validation_samples)?;
            let reused = boolean(&mut r)?;
            let mut distinct = Vec::with_capacity(p.validation_distinct);
            for _ in 0..r.count(p.validation_distinct)? {
                distinct.push(read_input(&mut r)?);
            }
            a.candidate = Some(Candidate {
                model,
                token,
                through,
                correct,
                distinct,
                reused,
            });
        }
        for _ in 0..r.count(p.pending_capacity)? {
            a.pending.push(read_pending(&mut r)?);
        }
        for _ in 0..r.count(p.recent_capacity)? {
            a.recent.push_back(Receipt {
                pending: read_pending(&mut r)?,
                label: read_option_number(&mut r)?,
                expired: boolean(&mut r)?,
            });
        }
        for _ in 0..r.count(p.training_window)? {
            a.window.push_back(Labeled {
                sequence: r.u64()?,
                input: read_input(&mut r)?,
                output: r.number()?,
            });
        }
        for _ in 0..r.count(p.drift_window)? {
            a.errors.push_back(boolean(&mut r)?);
        }
        for _ in 0..r.count(p.archive_models)? {
            let identity = r.u64()?;
            let bytes = read_blob(&mut r)?;
            a.archive.push(Archived { identity, bytes });
        }
        if !r.0.is_empty() {
            return Err(NumericError::Invalid("continuous checkpoint trailing data"));
        }
        // In-flight jobs are deliberately not resumed: their results need a new live owner/token.
        a.audit()?;
        Ok(a)
    }
    fn audit(&self) -> Result<(), NumericError> {
        let invalid = || NumericError::Invalid("continuous checkpoint state invariant");
        let s = self.statistics;
        if s.correct > s.scored
            || s.scored > s.feedback
            || s.feedback > s.observations
            || s.predictions > s.observations
            || s.expired > s.observations
            || self.watermark > s.observations
            || self.epoch != s.drifts.checked_add(s.promotions).ok_or_else(invalid)?
            || s.drifts > s.promotions
            || s.reuses > s.promotions
            || self.job_token != s.jobs
            || self.archived_bytes() > self.policy.archive_bytes
            || (self.ready && (self.active.is_none() || self.candidate.is_some()))
            || (s.observations == 0) != (self.last_id == 0)
        {
            return Err(invalid());
        }
        let mut archive_ids = BTreeSet::new();
        let mut models = Vec::new();
        for item in &self.archive {
            let skill = NumericSkill::from_bytes(&item.bytes)?;
            if skill.fingerprint() != item.identity || !archive_ids.insert(item.identity) {
                return Err(invalid());
            }
            self.audit_model(&skill)?;
            models.push(Model {
                skill,
                identity: item.identity,
            });
        }
        if let Some(active) = &self.active {
            self.audit_model(&active.skill)?;
            if !archive_ids.contains(&active.identity) {
                return Err(invalid());
            }
        }
        if let Some(c) = &self.candidate {
            self.audit_model(&c.model.skill)?;
            if c.token == 0
                || c.token > self.job_token
                || c.through > self.watermark
                || c.distinct.len() > c.correct
                || (c.reused && !archive_ids.contains(&c.model.identity))
            {
                return Err(invalid());
            }
            for (i, input) in c.distinct.iter().enumerate() {
                self.audit_input(input)?;
                if c.distinct[..i].iter().any(|x| same_input(x, input)) {
                    return Err(invalid());
                }
            }
        }
        let mut ids = BTreeSet::new();
        if self.temporal_target.is_some() != self.temporal_last.is_some()
            || self.temporal_target.is_some_and(|t| t >= self.arity)
            || self.temporal_last.is_some_and(|id| {
                id != self.last_id || self.pending.len() != 1 || self.pending[0].prediction.id != id
            })
        {
            return Err(invalid());
        }
        let mut sequences = BTreeSet::new();
        for pending in self
            .pending
            .iter()
            .chain(self.recent.iter().map(|r| &r.pending))
        {
            let pred = pending.prediction;
            self.audit_input(&pending.input)?;
            if pred.id == 0
                || pred.id > self.last_id
                || pred.sequence == 0
                || pred.sequence > s.observations
                || pred.epoch > self.epoch
                || pred.duplicate
                || !ids.insert(pred.id)
                || !sequences.insert(pred.sequence)
                || pending
                    .candidate_token
                    .is_some_and(|t| t == 0 || t > self.job_token)
            {
                return Err(invalid());
            }
            if let Some(identity) = pred.model {
                let model = models
                    .iter()
                    .find(|m| m.identity == identity)
                    .ok_or_else(invalid)?;
                let result = model.skill.program().evaluate(&pending.input[..self.arity]);
                if !same_value(result.as_ref().ok().copied(), pred.value)
                    || pending.guarded != result.is_err()
                {
                    return Err(invalid());
                }
            } else if pred.value.is_some() || pending.guarded {
                return Err(invalid());
            }
            if let Some(candidate) = &self.candidate
                && pending.candidate_token == Some(candidate.token)
                && !same_value(
                    candidate
                        .model
                        .skill
                        .program()
                        .evaluate(&pending.input[..self.arity])
                        .ok(),
                    pending.candidate_value,
                )
            {
                return Err(invalid());
            }
        }
        let mut previous = 0;
        for e in &self.window {
            self.audit_input(&e.input)?;
            if e.sequence <= previous || e.sequence > self.watermark {
                return Err(invalid());
            }
            previous = e.sequence;
        }
        if !self.ready && !self.errors.is_empty() {
            return Err(invalid());
        }
        Ok(())
    }
    fn audit_model(&self, skill: &NumericSkill) -> Result<(), NumericError> {
        if skill.context() != self.context
            || skill.program().arity() != self.arity
            || skill.tolerance() != self.policy.tolerance
        {
            return Err(NumericError::Invalid("checkpoint model contract"));
        }
        Ok(())
    }
    fn audit_input(&self, input: &[f64; 4]) -> Result<(), NumericError> {
        if input[self.arity..].iter().any(|v| v.to_bits() != 0) {
            return Err(NumericError::Invalid("checkpoint input padding"));
        }
        Ok(())
    }
}
fn same_value(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
        (None, None) => true,
        _ => false,
    }
}
fn input_bytes(b: &mut Vec<u8>, x: &[f64; 4]) {
    for n in x {
        b.extend(n.to_le_bytes());
    }
}
fn read_input(r: &mut Reader<'_>) -> Result<[f64; 4], NumericError> {
    Ok([r.number()?, r.number()?, r.number()?, r.number()?])
}
fn option_u64(b: &mut Vec<u8>, v: Option<u64>) {
    b.push(u8::from(v.is_some()));
    if let Some(v) = v {
        b.extend(v.to_le_bytes());
    }
}
fn read_option_u64(r: &mut Reader<'_>) -> Result<Option<u64>, NumericError> {
    if boolean(r)? {
        Ok(Some(r.u64()?))
    } else {
        Ok(None)
    }
}
fn option_number(b: &mut Vec<u8>, v: Option<f64>) {
    b.push(u8::from(v.is_some()));
    if let Some(v) = v {
        b.extend(v.to_le_bytes());
    }
}
fn read_option_number(r: &mut Reader<'_>) -> Result<Option<f64>, NumericError> {
    if boolean(r)? {
        Ok(Some(r.number()?))
    } else {
        Ok(None)
    }
}
fn boolean(r: &mut Reader<'_>) -> Result<bool, NumericError> {
    match r.byte()? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(NumericError::Invalid("checkpoint Boolean")),
    }
}
fn blob(b: &mut Vec<u8>, bytes: &[u8]) {
    b.extend((bytes.len() as u32).to_le_bytes());
    b.extend(bytes);
}
fn read_blob(r: &mut Reader<'_>) -> Result<Vec<u8>, NumericError> {
    let n = r.count(32768)?;
    Ok(r.take(n)?.to_vec())
}
fn model(b: &mut Vec<u8>, m: Option<&Model>) {
    b.push(u8::from(m.is_some()));
    if let Some(m) = m {
        blob(b, &m.skill.to_bytes());
    }
}
fn read_model(r: &mut Reader<'_>) -> Result<Option<Model>, NumericError> {
    if boolean(r)? {
        Ok(Some(Model::new(NumericSkill::from_bytes(&read_blob(r)?)?)))
    } else {
        Ok(None)
    }
}
fn pending_bytes(b: &mut Vec<u8>, p: &Pending) {
    let v = p.prediction;
    for n in [v.id, v.sequence, v.epoch] {
        b.extend(n.to_le_bytes());
    }
    option_u64(b, v.model);
    option_number(b, v.value);
    b.push(u8::from(v.duplicate));
    input_bytes(b, &p.input);
    b.push(u8::from(p.guarded));
    option_u64(b, p.candidate_token);
    option_number(b, p.candidate_value);
}
fn read_pending(r: &mut Reader<'_>) -> Result<Pending, NumericError> {
    let prediction = Prediction {
        id: r.u64()?,
        sequence: r.u64()?,
        epoch: r.u64()?,
        model: read_option_u64(r)?,
        value: read_option_number(r)?,
        duplicate: boolean(r)?,
    };
    Ok(Pending {
        prediction,
        input: read_input(r)?,
        guarded: boolean(r)?,
        candidate_token: read_option_u64(r)?,
        candidate_value: read_option_number(r)?,
    })
}
