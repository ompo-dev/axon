//! One outstanding prediction; bounded acquisition followed by future-only admission.
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReactiveStatistics {
    pub observations: u64,
    pub labels: u64,
    pub scored: u64,
    pub correct: u64,
    pub searches: u64,
    pub failed_searches: u64,
    pub rejected_candidates: u64,
    pub promotions: u64,
    pub drifts: u64,
    pub search_work: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReactivePhase {
    Collecting,
    Validating,
    Active,
    Suspended,
}
impl ReactivePhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Collecting => "collecting",
            Self::Validating => "validating",
            Self::Active => "active",
            Self::Suspended => "suspended",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReactivePrediction {
    pub id: u64,
    pub value: Option<i64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReactiveFeedback {
    Accepted,
    Duplicate,
    Candidate,
    Rejected,
    Promoted,
    Drift,
    SearchFailed,
}
impl ReactiveFeedback {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Duplicate => "duplicate",
            Self::Candidate => "candidate",
            Self::Rejected => "rejected",
            Self::Promoted => "promoted",
            Self::Drift => "drift",
            Self::SearchFailed => "search_failed",
        }
    }
}
struct Model {
    skill: ReactiveSkill,
    circuit: ReactiveCircuit,
    cursor: ReactiveCursor,
}
impl Model {
    fn new(skill: ReactiveSkill) -> Result<Self, ReactiveError> {
        let circuit = skill.compile()?;
        let cursor = circuit.cursor();
        Ok(Self {
            skill,
            circuit,
            cursor,
        })
    }
    fn step(&mut self, input: &[i64]) -> Result<i64, ReactiveError> {
        self.circuit
            .step(&mut self.cursor, self.skill.context(), input)
    }
}
struct Pending {
    id: u64,
    input: [i64; 2],
    prediction: Option<i64>,
    shadow: Option<i64>,
}
pub struct ReactiveAgent {
    context: String,
    arity: usize,
    budget: ReactiveBudget,
    library: Vec<ReactiveSkill>,
    events: Vec<ReactiveEvent>,
    active: Option<Model>,
    shadow: Option<Model>,
    pending: Option<Pending>,
    validation: usize,
    distinct: Vec<[i64; 2]>,
    last_id: u64,
    last_label: Option<(u64, i64)>,
    next_search: u64,
    suspended: bool,
    statistics: ReactiveStatistics,
}
impl ReactiveAgent {
    pub fn new(
        context: &str,
        arity: usize,
        library: Vec<ReactiveSkill>,
        budget: ReactiveBudget,
    ) -> Result<Self, ReactiveError> {
        identifier(context)?;
        budget.validate()?;
        if !(1..=2).contains(&arity)
            || library.len() > budget.max_tools
            || library.iter().any(|s| s.arity() != arity)
        {
            return Err(ReactiveError::Invalid("agent schema/library"));
        }
        Ok(Self {
            context: context.into(),
            arity,
            budget,
            library,
            events: Vec::with_capacity(budget.max_ticks),
            active: None,
            shadow: None,
            pending: None,
            validation: 0,
            distinct: Vec::with_capacity(2),
            last_id: 0,
            last_label: None,
            next_search: 8,
            suspended: false,
            statistics: ReactiveStatistics::default(),
        })
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn statistics(&self) -> ReactiveStatistics {
        self.statistics
    }
    pub fn phase(&self) -> ReactivePhase {
        if self.suspended {
            ReactivePhase::Suspended
        } else if self.active.is_some() {
            ReactivePhase::Active
        } else if self.shadow.is_some() {
            ReactivePhase::Validating
        } else {
            ReactivePhase::Collecting
        }
    }
    pub fn skill(&self) -> Option<&ReactiveSkill> {
        self.active.as_ref().map(|m| &m.skill)
    }
    pub fn cursor(&self) -> Option<&ReactiveCursor> {
        self.active.as_ref().map(|m| &m.cursor)
    }
    pub fn candidate(&self) -> Option<&ReactiveSkill> {
        self.shadow.as_ref().map(|m| &m.skill)
    }
    pub fn validation_progress(&self) -> (usize, usize) {
        (self.validation, self.distinct.len())
    }
    /// A pure proposal over previously observed inputs; the host must authorize any intervention.
    pub fn suggest_probe(&self) -> Option<Vec<i64>> {
        self.probe_choices(None)
    }
    /// Explore an explicitly supplied integer domain without acting on the environment.
    pub fn suggest_probe_with_bounds(
        &self,
        bounds: &[(i64, i64)],
    ) -> Result<Option<Vec<i64>>, ReactiveError> {
        if bounds.len() != self.arity || bounds.iter().any(|(lo, hi)| lo > hi) {
            return Err(ReactiveError::Invalid("probe bounds/schema"));
        }
        Ok(self.probe_choices(Some(bounds)))
    }
    fn probe_choices(&self, bounds: Option<&[(i64, i64)]>) -> Option<Vec<i64>> {
        if self.active.is_some()
            || self.pending.is_some()
            || self.suspended
            || self.events.len() < 8
        {
            return None;
        }
        let last = &self.events.last()?.input;
        let mut choices = Vec::<Vec<i64>>::new();
        for e in &self.events {
            if !choices.contains(&e.input)
                && choices.len() < 16
                && bounds.is_none_or(|b| {
                    e.input
                        .iter()
                        .zip(b)
                        .all(|(n, (lo, hi))| (lo..=hi).contains(&n))
                })
            {
                choices.push(e.input.clone());
            }
        }
        if let Some(bounds) = bounds {
            // Boundary and coordinate probes break correlations between input magnitude and memory.
            let anchor: Vec<_> = last
                .iter()
                .zip(bounds)
                .map(|(n, (lo, hi))| (*n).clamp(*lo, *hi))
                .collect();
            for (dimension, (lo, hi)) in bounds.iter().enumerate() {
                for n in [*lo, *hi, 0, -1, 1] {
                    if (*lo..=*hi).contains(&n) {
                        let mut input = anchor.clone();
                        input[dimension] = n;
                        if !choices.contains(&input) {
                            choices.push(input);
                        }
                    }
                }
            }
        }
        choices.into_iter().min_by_key(|next| {
            let pairs = self
                .events
                .windows(2)
                .filter(|p| {
                    p[0].input.as_slice() == last.as_slice()
                        && p[1].input.as_slice() == next.as_slice()
                })
                .count();
            let occurrences = self
                .events
                .iter()
                .filter(|e| e.input.as_slice() == next.as_slice())
                .count();
            (pairs, occurrences)
        })
    }
    pub fn observe(&mut self, id: u64, input: &[i64]) -> Result<ReactivePrediction, ReactiveError> {
        if self.suspended {
            return Err(ReactiveError::Suspended);
        }
        if input.len() != self.arity || id <= self.last_id || self.pending.is_some() {
            return Err(ReactiveError::Invalid(
                "ordered input; one pending prediction",
            ));
        }
        if self.active.is_none() && self.events.len() >= self.budget.max_ticks {
            return Err(ReactiveError::Invalid(
                "acquisition episode limit; reset/new experience required",
            ));
        }
        let mut values = [0; 2];
        values[..self.arity].copy_from_slice(input);
        let value = if let Some(model) = &mut self.active {
            Some(model.step(input)?)
        } else {
            None
        };
        let shadow = self.shadow.as_mut().and_then(|m| m.step(input).ok());
        self.pending = Some(Pending {
            id,
            input: values,
            prediction: value,
            shadow,
        });
        self.last_id = id;
        self.statistics.observations += 1;
        Ok(ReactivePrediction { id, value })
    }
    pub fn feedback(&mut self, id: u64, actual: i64) -> Result<ReactiveFeedback, ReactiveError> {
        if let Some((old, label)) = self.last_label
            && old == id
        {
            return if label == actual {
                Ok(ReactiveFeedback::Duplicate)
            } else {
                Err(ReactiveError::Invalid("feedback collision"))
            };
        }
        if self.pending.as_ref().is_none_or(|p| p.id != id) {
            return Err(ReactiveError::Invalid(
                "feedback must match pending prediction",
            ));
        }
        let pending = self.pending.take().expect("checked");
        self.last_label = Some((id, actual));
        self.statistics.labels += 1;
        if let Some(predicted) = pending.prediction {
            self.statistics.scored += 1;
            self.statistics.correct += u64::from(predicted == actual);
            if predicted != actual {
                self.suspended = true;
                self.statistics.drifts += 1;
                if let Some(model) = &mut self.active {
                    let _ = model.circuit.feedback(&mut model.cursor, predicted, actual);
                }
                return Ok(ReactiveFeedback::Drift);
            }
            return Ok(ReactiveFeedback::Accepted);
        }
        self.events.push(ReactiveEvent::new(
            pending.input[..self.arity].to_vec(),
            actual,
        ));
        let mut status = ReactiveFeedback::Accepted;
        if self.shadow.is_some() {
            if pending.shadow != Some(actual) {
                self.shadow = None;
                self.validation = 0;
                self.distinct.clear();
                self.statistics.rejected_candidates += 1;
                self.next_search = self.statistics.labels.saturating_add(8);
                status = ReactiveFeedback::Rejected;
            } else {
                self.validation += 1;
                if self.distinct.len() < 2 && !self.distinct.contains(&pending.input) {
                    self.distinct.push(pending.input);
                }
                if self.validation >= 8 && self.distinct.len() >= 2 {
                    self.active = self.shadow.take();
                    self.statistics.promotions += 1;
                    return Ok(ReactiveFeedback::Promoted);
                }
            }
        }
        if self.shadow.is_none()
            && self.statistics.labels >= self.next_search
            && self.events.len() >= 8
        {
            self.statistics.searches += 1;
            let tools: Vec<_> = self.library.iter().collect();
            match ReactiveSkill::learn(
                "acquired",
                &self.context,
                vec![ReactiveTrace::new(self.events.clone())],
                &tools,
                self.budget,
            ) {
                Ok(skill) => {
                    self.statistics.search_work = self
                        .statistics
                        .search_work
                        .saturating_add(skill.search().work);
                    let mut model = Model::new(skill)?;
                    for e in &self.events {
                        model.step(&e.input)?;
                    }
                    self.shadow = Some(model);
                    self.validation = 0;
                    self.distinct.clear();
                    status = ReactiveFeedback::Candidate;
                }
                Err(ReactiveError::Budget(s) | ReactiveError::NoProgram(s)) => {
                    self.statistics.search_work =
                        self.statistics.search_work.saturating_add(s.work);
                    self.statistics.failed_searches += 1;
                    self.next_search = self.statistics.labels.saturating_add(8);
                    status = ReactiveFeedback::SearchFailed;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(status)
    }
    pub fn checkpoint_new(&self, path: &Path) -> Result<(), ReactiveError> {
        if self.pending.is_some() || self.suspended {
            return Err(ReactiveError::Invalid(
                "checkpoint requires an acknowledged active boundary",
            ));
        }
        let model = self
            .active
            .as_ref()
            .ok_or(ReactiveError::Invalid("no admitted model"))?;
        let mut payload = Vec::new();
        for n in [
            self.last_id,
            self.statistics.observations,
            self.statistics.labels,
            self.statistics.scored,
            self.statistics.correct,
            self.statistics.searches,
            self.statistics.failed_searches,
            self.statistics.rejected_candidates,
            self.statistics.promotions,
            self.statistics.drifts,
            self.statistics.search_work,
        ] {
            payload.extend_from_slice(&n.to_le_bytes());
        }
        let (id, label) = self
            .last_label
            .ok_or(ReactiveError::Invalid("missing feedback boundary"))?;
        payload.extend_from_slice(&id.to_le_bytes());
        payload.extend_from_slice(&(label as u64).to_le_bytes());
        if !self.events.is_empty() {
            payload.extend_from_slice(&(self.events.len() as u32).to_le_bytes());
            for e in &self.events {
                payload.extend_from_slice(&(e.input.len() as u32).to_le_bytes());
                for n in &e.input {
                    payload.extend_from_slice(&n.to_le_bytes());
                }
                payload.extend_from_slice(&e.output.to_le_bytes());
            }
        }
        for b in [model.skill.to_bytes(), model.cursor.to_bytes()] {
            payload.extend_from_slice(&(b.len() as u32).to_le_bytes());
            payload.extend_from_slice(&b);
        }
        publish_new(
            path,
            &storage::envelope(
                if self.events.is_empty() {
                    b"AXRA0001"
                } else {
                    b"AXRA0002"
                },
                &payload,
            ),
        )?;
        Ok(())
    }
    pub fn resume(context: &str, path: &Path) -> Result<Self, ReactiveError> {
        let bytes = read_bounded(path, MAX_FILE + 4096)?;
        let evidence = bytes.starts_with(b"AXRA0002");
        let mut r = storage::Reader(storage::unwrap(
            &bytes,
            if evidence { b"AXRA0002" } else { b"AXRA0001" },
            MAX_FILE as usize + 4096,
        )?);
        let last_id = r.u64()?;
        let statistics = ReactiveStatistics {
            observations: r.u64()?,
            labels: r.u64()?,
            scored: r.u64()?,
            correct: r.u64()?,
            searches: r.u64()?,
            failed_searches: r.u64()?,
            rejected_candidates: r.u64()?,
            promotions: r.u64()?,
            drifts: r.u64()?,
            search_work: r.u64()?,
        };
        let last_label = Some((r.u64()?, r.u64()? as i64));
        let mut recorded = Vec::new();
        // Keep admission evidence separate from the prefix used to synthesize the program.
        if evidence {
            let n = r.count(64)?;
            // Each event carries its arity so malformed snapshots cannot shift the capsule boundary.
            for _ in 0..n {
                let dims = r.count(2)?;
                let mut input = Vec::new();
                for _ in 0..dims {
                    input.push(r.u64()? as i64);
                }
                recorded.push(ReactiveEvent::new(input, r.u64()? as i64));
            }
        }
        let n = r.count(MAX_FILE as usize)?;
        let skill = ReactiveSkill::from_bytes(r.take(n)?)?;
        if skill.context() != context {
            return Err(ReactiveError::Context);
        }
        if evidence {
            let prefix: Vec<_> = skill.traces.iter().flat_map(|t| t.events.iter()).collect();
            if skill.traces.len() != 1
                || recorded.len() < prefix.len() + 8
                || recorded.iter().any(|e| e.input.len() != skill.arity)
                || prefix.iter().zip(&recorded).any(|(a, b)| **a != *b)
                || recorded[prefix.len()..]
                    .iter()
                    .map(|e| &e.input)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    < 2
            {
                return Err(ReactiveError::Invalid(
                    "admission evidence prefix/diversity",
                ));
            }
            let circuit = skill.compile()?;
            let mut replay = circuit.cursor();
            for e in &recorded {
                if circuit.step(&mut replay, context, &e.input)? != e.output {
                    return Err(ReactiveError::Invalid("admission evidence replay"));
                }
            }
        }
        let mut model = Model::new(skill)?;
        let n = r.count(512)?;
        model.cursor = ReactiveCursor::from_bytes(r.take(n)?, &model.circuit, context)?;
        if !r.0.is_empty()
            || statistics.observations != statistics.labels
            || statistics.observations != model.cursor.ticks()
            || statistics.correct > statistics.scored
            || statistics.scored > statistics.labels
            || statistics.observations == 0
            || statistics.promotions != 1
            || statistics.searches == 0
            || statistics.failed_searches >= statistics.searches
            || statistics.rejected_candidates >= statistics.searches
            || statistics.drifts != 0
            || model.cursor.suspended()
            || last_id < statistics.observations
            || last_label.map(|p| p.0) != Some(last_id)
        {
            return Err(ReactiveError::Invalid("agent counters/boundary"));
        }
        let mut agent = Self::new(
            context,
            model.skill.arity(),
            Vec::new(),
            ReactiveBudget::default(),
        )?;
        agent.last_id = last_id;
        agent.last_label = last_label;
        agent.statistics = statistics;
        agent.events = recorded;
        agent.active = Some(model);
        Ok(agent)
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.context.capacity()
            + self.events.capacity() * size_of::<ReactiveEvent>()
            + self
                .events
                .iter()
                .map(|e| e.input.capacity() * size_of::<i64>())
                .sum::<usize>()
            + self.library.capacity() * size_of::<ReactiveSkill>()
            + self
                .library
                .iter()
                .map(|s| s.retained_bytes() - size_of::<ReactiveSkill>())
                .sum::<usize>()
            + self.distinct.capacity() * size_of::<[i64; 2]>()
            + [&self.active, &self.shadow]
                .iter()
                .map(|m| {
                    m.as_ref().map_or(0, |m| {
                        m.skill.retained_bytes()
                            + m.circuit.retained_bytes()
                            + m.cursor.retained_bytes()
                    })
                })
                .sum::<usize>()
    }
}
