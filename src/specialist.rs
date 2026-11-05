//! Bounded empirical specialization, separate from exact semantic certificates.
//! The caller owns label quality, independent sampling and fallback execution.

use crate::{Capability, CapabilityGate, GateFailure};
use std::{
    fs::OpenOptions,
    io::{self, Read, Write},
    path::Path,
};

pub const FEATURES: usize = 4;
pub const MAX_SPLIT_SAMPLES: usize = 65_536;
pub const MAX_DRIFT_WINDOW: usize = 256;
const COMPOSITION_QUANTILES: usize = 8;
const CHECKPOINT_MAGIC_V1: &[u8; 8] = b"AXSpc001";
const CHECKPOINT_MAGIC: &[u8; 8] = b"AXSpc002";
const CHECKPOINT_BYTES_V1: usize = 612;
const CHECKPOINT_BYTES: usize = 670;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probability(f64);

impl Probability {
    pub fn new(value: f64) -> Option<Self> {
        (value.is_finite() && (0.0..=1.0).contains(&value)).then_some(Self(value))
    }

    pub fn get(self) -> f64 {
        self.0
    }

    pub fn label(self) -> bool {
        self.0 >= 0.5
    }

    pub fn confidence(self) -> f64 {
        self.0.max(1.0 - self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Observation {
    pub id: u64,
    pub features: [i64; FEATURES],
    pub label: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapsuleId {
    /// Caller-defined task/feature-schema revision, not a hardware identifier.
    pub context: u64,
    pub version: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoundryConfig {
    pub min_leaf: usize,
    pub min_audit_accepted: usize,
    /// Current bounded grammar supports one or two numeric predicates.
    pub max_predicates: usize,
    pub min_confidence: f64,
    pub min_coverage: f64,
    pub max_error: f64,
    pub min_brier_gain: f64,
    pub complexity_penalty: f64,
    pub drift_window: usize,
    pub max_window_error: f64,
}

impl Default for FoundryConfig {
    fn default() -> Self {
        Self {
            min_leaf: 16,
            min_audit_accepted: 64,
            max_predicates: 2,
            min_confidence: 0.8,
            min_coverage: 0.9,
            max_error: 0.15,
            min_brier_gain: 0.01,
            complexity_penalty: 0.01,
            drift_window: 64,
            max_window_error: 0.25,
        }
    }
}

impl FoundryConfig {
    fn valid(self) -> bool {
        (1..=MAX_SPLIT_SAMPLES / 2).contains(&self.min_leaf)
            && (1..=MAX_SPLIT_SAMPLES).contains(&self.min_audit_accepted)
            && (1..=2).contains(&self.max_predicates)
            && (1..=MAX_DRIFT_WINDOW).contains(&self.drift_window)
            && [
                self.min_confidence,
                self.min_coverage,
                self.max_error,
                self.min_brier_gain,
                self.complexity_penalty,
                self.max_window_error,
            ]
            .into_iter()
            .all(|x| Probability::new(x).is_some())
            && self.min_confidence >= 0.5
            && self.min_coverage > 0.0
            && self.min_brier_gain > 0.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FoundryError {
    InvalidConfig,
    InvalidVersion,
    TooFewSamples,
    TooManySamples,
    DuplicateObservation(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleForm {
    Constant,
    Threshold,
    PairTable,
}

/// A constant, one threshold, or a two-predicate four-cell table.
/// Private fields preserve feature bounds and keep the runtime allocation-free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryRule {
    feature: Option<usize>,
    threshold: i64,
    second_feature: Option<usize>,
    second_threshold: i64,
    cells: [Probability; 4],
}

impl BinaryRule {
    fn constant(probability: Probability) -> Self {
        Self {
            feature: None,
            threshold: 0,
            second_feature: None,
            second_threshold: 0,
            cells: [probability; 4],
        }
    }

    fn stump(feature: usize, threshold: i64, low: Probability, high: Probability) -> Self {
        Self {
            feature: Some(feature),
            threshold,
            second_feature: None,
            second_threshold: 0,
            cells: [low, high, low, high],
        }
    }

    fn pair(first: (usize, i64), second: (usize, i64), cells: [Probability; 4]) -> Self {
        Self {
            feature: Some(first.0),
            threshold: first.1,
            second_feature: Some(second.0),
            second_threshold: second.1,
            cells,
        }
    }

    pub fn form(self) -> RuleForm {
        match (self.feature, self.second_feature) {
            (None, None) => RuleForm::Constant,
            (Some(_), None) => RuleForm::Threshold,
            (Some(_), Some(_)) => RuleForm::PairTable,
            (None, Some(_)) => unreachable!("validated rules cannot skip the first predicate"),
        }
    }

    pub fn complexity(self) -> usize {
        match self.form() {
            RuleForm::Constant => 0,
            RuleForm::Threshold => 1,
            RuleForm::PairTable => 2,
        }
    }

    pub fn split(self) -> Option<(usize, i64)> {
        self.feature.map(|feature| (feature, self.threshold))
    }

    pub fn second_split(self) -> Option<(usize, i64)> {
        self.second_feature
            .map(|feature| (feature, self.second_threshold))
    }

    /// Probabilities for a single-threshold rule. Composed tables use `cell_probabilities`.
    pub fn probabilities(self) -> Option<(Probability, Probability)> {
        (self.second_feature.is_none()).then_some((self.cells[0], self.cells[1]))
    }

    pub fn cell_probabilities(self) -> [Probability; 4] {
        self.cells
    }

    fn valid_structure(self) -> bool {
        match (self.feature, self.second_feature) {
            (None, None) => self.cells.iter().all(|cell| *cell == self.cells[0]),
            (Some(_), None) => self.cells[0] == self.cells[2] && self.cells[1] == self.cells[3],
            (Some(first), Some(second)) => first != second,
            (None, Some(_)) => false,
        }
    }

    /// Unguarded model evaluation for diagnostics, NOT permission to act.
    pub fn estimate(self, features: &[i64; FEATURES]) -> Probability {
        let first = usize::from(
            self.feature
                .is_some_and(|feature| features[feature] > self.threshold),
        );
        let second = usize::from(
            self.second_feature
                .is_some_and(|feature| features[feature] > self.second_threshold),
        );
        let index = if self.second_feature.is_some() {
            first * 2 + second
        } else {
            first
        };
        self.cells[index]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmpiricalMetrics {
    pub samples: usize,
    pub accepted: usize,
    pub errors: usize,
    /// Brier and ten-bin ECE cover ALL samples, including abstentions.
    pub brier: f64,
    pub baseline_brier: f64,
    pub ece: f64,
}

impl EmpiricalMetrics {
    pub fn coverage(self) -> f64 {
        self.accepted as f64 / self.samples as f64
    }

    /// No accepted samples means undefined risk, not zero risk.
    pub fn selective_error(self) -> Option<f64> {
        (self.accepted > 0).then(|| self.errors as f64 / self.accepted as f64)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Promotion {
    Accepted,
    NoGain,
    InsufficientCoverage,
    ExcessiveError,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmpiricalEvidence {
    pub id: CapsuleId,
    pub training_samples: usize,
    pub rule: BinaryRule,
    pub selection: EmpiricalMetrics,
    pub audit: EmpiricalMetrics,
    pub promotion: Promotion,
}

#[derive(Debug)]
pub struct FoundryOutcome {
    pub evidence: EmpiricalEvidence,
    pub capsule: Option<SpecialistCapsule>,
}

type Domain = [(i64, i64); FEATURES];

/// Fit on train, choose family on selection, approve once on audit.
/// IDs must be globally distinct; this does not detect relabelled copies or group leakage.
pub fn train_specialist(
    id: CapsuleId,
    config: FoundryConfig,
    train: &[Observation],
    selection: &[Observation],
    audit: &[Observation],
) -> Result<FoundryOutcome, FoundryError> {
    if !config.valid() {
        return Err(FoundryError::InvalidConfig);
    }
    if id.version == 0 {
        return Err(FoundryError::InvalidVersion);
    }
    if [train, selection, audit]
        .iter()
        .any(|s| s.len() > MAX_SPLIT_SAMPLES)
    {
        return Err(FoundryError::TooManySamples);
    }
    if train.len() < (2 * config.min_leaf).max(32)
        || selection.len() < 32
        || audit.len() < config.min_audit_accepted.max(32)
    {
        return Err(FoundryError::TooFewSamples);
    }
    let mut ids: Vec<_> = train
        .iter()
        .chain(selection)
        .chain(audit)
        .map(|s| s.id)
        .collect();
    ids.sort_unstable();
    if let Some(pair) = ids.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(FoundryError::DuplicateObservation(pair[0]));
    }
    let domain = std::array::from_fn(|f| {
        (
            train.iter().map(|s| s.features[f]).min().unwrap(),
            train.iter().map(|s| s.features[f]).max().unwrap(),
        )
    });
    let prior = smoothed(train.iter().filter(|s| s.label).count(), train.len());
    let baseline = BinaryRule::constant(prior);
    let mut rule = baseline;
    let mut best_score =
        metrics(baseline, baseline, domain, config.min_confidence, selection).brier;
    let stump = fit_stump(train, config.min_leaf);
    for candidate in [
        Some(stump),
        (config.max_predicates == 2)
            .then(|| fit_pair(train, config.min_leaf))
            .flatten(),
    ]
    .into_iter()
    .flatten()
    {
        let score = metrics(
            candidate,
            baseline,
            domain,
            config.min_confidence,
            selection,
        )
        .brier
            + config.complexity_penalty * candidate.complexity() as f64;
        if score < best_score {
            best_score = score;
            rule = candidate;
        }
    }
    let selection_metrics = metrics(rule, baseline, domain, config.min_confidence, selection);
    let audit_metrics = metrics(rule, baseline, domain, config.min_confidence, audit);
    let promotion = if rule.complexity() == 0
        || audit_metrics.baseline_brier - audit_metrics.brier < config.min_brier_gain
    {
        Promotion::NoGain
    } else if audit_metrics.accepted < config.min_audit_accepted
        || audit_metrics.coverage() < config.min_coverage
    {
        Promotion::InsufficientCoverage
    } else if audit_metrics.selective_error().unwrap() > config.max_error {
        Promotion::ExcessiveError
    } else {
        Promotion::Accepted
    };
    let evidence = EmpiricalEvidence {
        id,
        training_samples: train.len(),
        rule,
        selection: selection_metrics,
        audit: audit_metrics,
        promotion,
    };
    let capsule = (promotion == Promotion::Accepted).then_some(SpecialistCapsule {
        evidence,
        domain,
        baseline,
        config,
        suspended: false,
        last_event: 0,
        errors: [false; MAX_DRIFT_WINDOW],
        seen: 0,
        cursor: 0,
        window_errors: 0,
    });
    Ok(FoundryOutcome { evidence, capsule })
}

fn smoothed(positive: usize, total: usize) -> Probability {
    // Laplace smoothing is an estimate, not a calibration guarantee.
    Probability((positive as f64 + 1.0) / (total as f64 + 2.0))
}

fn loss(positive: usize, total: usize, p: Probability) -> f64 {
    positive as f64 * (1.0 - p.0).powi(2) + (total - positive) as f64 * p.0.powi(2)
}

fn fit_stump(train: &[Observation], min_leaf: usize) -> BinaryRule {
    let positives = train.iter().filter(|s| s.label).count();
    let prior = smoothed(positives, train.len());
    let mut best = BinaryRule::constant(prior);
    let mut best_loss = loss(positives, train.len(), prior);
    let mut sorted: Vec<_> = train.iter().collect();
    for feature in 0..FEATURES {
        sorted.sort_unstable_by_key(|s| s.features[feature]);
        let mut left_positive = 0;
        for left_count in 1..train.len() {
            left_positive += usize::from(sorted[left_count - 1].label);
            let threshold = sorted[left_count - 1].features[feature];
            if left_count < min_leaf
                || train.len() - left_count < min_leaf
                || threshold == sorted[left_count].features[feature]
            {
                continue;
            }
            let right_positive = positives - left_positive;
            let low = smoothed(left_positive, left_count);
            let high = smoothed(right_positive, train.len() - left_count);
            let candidate_loss = loss(left_positive, left_count, low)
                + loss(right_positive, train.len() - left_count, high);
            if candidate_loss < best_loss {
                best_loss = candidate_loss;
                best = BinaryRule::stump(feature, threshold, low, high);
            }
        }
    }
    best
}

#[derive(Clone, Copy)]
struct Predicate {
    feature: usize,
    threshold: i64,
}

fn quantile_predicates(train: &[Observation]) -> Vec<Predicate> {
    let mut predicates = Vec::with_capacity(FEATURES * (COMPOSITION_QUANTILES - 1));
    let mut values = Vec::with_capacity(train.len());
    for feature in 0..FEATURES {
        values.clear();
        values.extend(train.iter().map(|sample| sample.features[feature]));
        values.sort_unstable();
        values.dedup();
        for quantile in 1..COMPOSITION_QUANTILES {
            let boundary = quantile * values.len() / COMPOSITION_QUANTILES;
            if boundary == 0 || boundary >= values.len() {
                continue;
            }
            let threshold = values[boundary - 1];
            if predicates.last().is_none_or(|last: &Predicate| {
                last.feature != feature || last.threshold != threshold
            }) {
                predicates.push(Predicate { feature, threshold });
            }
        }
    }
    predicates
}

fn fit_pair(train: &[Observation], min_leaf: usize) -> Option<BinaryRule> {
    if train.len() < min_leaf.saturating_mul(4) {
        return None;
    }
    let predicates = quantile_predicates(train);
    let mut best: Option<(f64, BinaryRule)> = None;
    for (first_index, first) in predicates.iter().enumerate() {
        for second in &predicates[first_index + 1..] {
            if first.feature == second.feature {
                continue;
            }
            let mut totals = [0_usize; 4];
            let mut positives = [0_usize; 4];
            for sample in train {
                let cell = usize::from(sample.features[first.feature] > first.threshold) * 2
                    + usize::from(sample.features[second.feature] > second.threshold);
                totals[cell] += 1;
                positives[cell] += usize::from(sample.label);
            }
            if totals.iter().any(|&total| total < min_leaf) {
                continue;
            }
            let cells = std::array::from_fn(|cell| smoothed(positives[cell], totals[cell]));
            let candidate_loss = (0..4)
                .map(|cell| loss(positives[cell], totals[cell], cells[cell]))
                .sum();
            if best.is_none_or(|(best_loss, _)| candidate_loss < best_loss) {
                best = Some((
                    candidate_loss,
                    BinaryRule::pair(
                        (first.feature, first.threshold),
                        (second.feature, second.threshold),
                        cells,
                    ),
                ));
            }
        }
    }
    best.map(|(_, rule)| rule)
}

fn in_domain(domain: Domain, features: &[i64; FEATURES]) -> bool {
    domain
        .iter()
        .zip(features)
        .all(|(&(low, high), x)| (low..=high).contains(x))
}

fn metrics(
    rule: BinaryRule,
    baseline: BinaryRule,
    domain: Domain,
    confidence: f64,
    samples: &[Observation],
) -> EmpiricalMetrics {
    let mut result = EmpiricalMetrics {
        samples: samples.len(),
        accepted: 0,
        errors: 0,
        brier: 0.0,
        baseline_brier: 0.0,
        ece: 0.0,
    };
    let mut bins = [(0_usize, 0.0, 0.0); 10];
    for sample in samples {
        let p = rule.estimate(&sample.features);
        let correct = p.label() == sample.label;
        result.brier += (p.0 - f64::from(sample.label)).powi(2);
        result.baseline_brier +=
            (baseline.estimate(&sample.features).0 - f64::from(sample.label)).powi(2);
        let bin = &mut bins[((p.confidence() * 10.0) as usize).min(9)];
        bin.0 += 1;
        bin.1 += p.confidence();
        bin.2 += f64::from(correct);
        if in_domain(domain, &sample.features) && p.confidence() >= confidence {
            result.accepted += 1;
            result.errors += usize::from(!correct);
        }
    }
    result.brier /= samples.len() as f64;
    result.baseline_brier /= samples.len() as f64;
    result.ece = bins.iter().map(|b| (b.1 - b.2).abs()).sum::<f64>() / samples.len() as f64;
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Abstention {
    Suspended,
    ContextMismatch,
    OutsideDomain,
    LowConfidence,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub id: CapsuleId,
    pub probability: Probability,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecisionFailure {
    Gate(GateFailure),
    Abstain(Abstention),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedbackError {
    OutOfOrder,
    Abstain(Abstention),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Feedback {
    pub correct: bool,
    pub window_samples: usize,
    pub window_errors: usize,
    pub suspended: bool,
}

/// Fixed-size runtime. Feedback monitors a frozen model; retraining creates a new capsule.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecialistCapsule {
    evidence: EmpiricalEvidence,
    domain: Domain,
    baseline: BinaryRule,
    config: FoundryConfig,
    suspended: bool,
    last_event: u64,
    errors: [bool; MAX_DRIFT_WINDOW],
    seen: usize,
    cursor: usize,
    window_errors: usize,
}

impl SpecialistCapsule {
    pub const fn checkpoint_bytes() -> usize {
        CHECKPOINT_BYTES
    }

    pub fn evidence(&self) -> EmpiricalEvidence {
        self.evidence
    }

    pub fn domain(&self) -> Domain {
        self.domain
    }

    pub fn suspended(&self) -> bool {
        self.suspended
    }

    /// Saves the complete frozen model and its drift-monitor state without replacing a file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut bytes = Vec::with_capacity(CHECKPOINT_BYTES);
        bytes.extend_from_slice(CHECKPOINT_MAGIC);
        push_usize(&mut bytes, self.evidence.training_samples);
        push_capsule_id(&mut bytes, self.evidence.id);
        push_rule(&mut bytes, self.evidence.rule);
        push_metrics(&mut bytes, self.evidence.selection);
        push_metrics(&mut bytes, self.evidence.audit);
        bytes.push(promotion_byte(self.evidence.promotion));
        for (low, high) in self.domain {
            bytes.extend_from_slice(&low.to_le_bytes());
            bytes.extend_from_slice(&high.to_le_bytes());
        }
        push_rule(&mut bytes, self.baseline);
        push_config(&mut bytes, self.config);
        bytes.push(u8::from(self.suspended));
        bytes.extend_from_slice(&self.last_event.to_le_bytes());
        bytes.extend(self.errors.map(u8::from));
        push_usize(&mut bytes, self.seen);
        push_usize(&mut bytes, self.cursor);
        push_usize(&mut bytes, self.window_errors);
        debug_assert_eq!(bytes.len(), CHECKPOINT_BYTES - 8);
        bytes.extend_from_slice(&checkpoint_checksum(&bytes).to_le_bytes());
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&bytes)?;
        file.sync_all()
    }

    pub fn load(path: impl AsRef<Path>, expected: CapsuleId) -> io::Result<Self> {
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid specialist capsule");
        let mut bytes = Vec::with_capacity(CHECKPOINT_BYTES + 1);
        std::fs::File::open(path)?
            .take((CHECKPOINT_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        let legacy = match bytes.len() {
            CHECKPOINT_BYTES if bytes.get(..8) == Some(CHECKPOINT_MAGIC) => false,
            CHECKPOINT_BYTES_V1 if bytes.get(..8) == Some(CHECKPOINT_MAGIC_V1) => true,
            _ => return Err(invalid()),
        };
        let record_bytes = bytes.len();
        if checkpoint_checksum(&bytes[..record_bytes - 8])
            != u64::from_le_bytes(
                bytes[record_bytes - 8..]
                    .try_into()
                    .expect("fixed checksum field"),
            )
        {
            return Err(invalid());
        }
        let mut reader = ByteReader::new(&bytes[8..record_bytes - 8]);
        let training_samples = reader.usize().ok_or_else(invalid)?;
        let id = reader.capsule_id().ok_or_else(invalid)?;
        let rule = if legacy {
            reader.legacy_rule().ok_or_else(invalid)?
        } else {
            reader.rule().ok_or_else(invalid)?
        };
        let selection = reader.metrics().ok_or_else(invalid)?;
        let audit = reader.metrics().ok_or_else(invalid)?;
        let promotion = reader.promotion().ok_or_else(invalid)?;
        let mut domain = [(0, 0); FEATURES];
        for bounds in &mut domain {
            *bounds = (
                reader.i64().ok_or_else(invalid)?,
                reader.i64().ok_or_else(invalid)?,
            );
        }
        let baseline = if legacy {
            reader.legacy_rule().ok_or_else(invalid)?
        } else {
            reader.rule().ok_or_else(invalid)?
        };
        let config = if legacy {
            reader.legacy_config().ok_or_else(invalid)?
        } else {
            reader.config().ok_or_else(invalid)?
        };
        let suspended = reader.boolean().ok_or_else(invalid)?;
        let last_event = reader.u64().ok_or_else(invalid)?;
        let mut errors = [false; MAX_DRIFT_WINDOW];
        for error in &mut errors {
            *error = reader.boolean().ok_or_else(invalid)?;
        }
        let seen = reader.usize().ok_or_else(invalid)?;
        let cursor = reader.usize().ok_or_else(invalid)?;
        let window_errors = reader.usize().ok_or_else(invalid)?;
        let capsule = Self {
            evidence: EmpiricalEvidence {
                id,
                training_samples,
                rule,
                selection,
                audit,
                promotion,
            },
            domain,
            baseline,
            config,
            suspended,
            last_event,
            errors,
            seen,
            cursor,
            window_errors,
        };
        if !reader.finished() || id != expected || !capsule.valid_checkpoint() {
            return Err(invalid());
        }
        Ok(capsule)
    }

    fn valid_checkpoint(&self) -> bool {
        if !self.config.valid() {
            return false;
        }
        let audit = self.evidence.audit;
        let active_errors = self.errors[..self.config.drift_window]
            .iter()
            .filter(|&&error| error)
            .count();
        self.evidence.id.version != 0
            && self.evidence.training_samples >= (2 * self.config.min_leaf).max(32)
            && self.evidence.rule.valid_structure()
            && (1..=self.config.max_predicates).contains(&self.evidence.rule.complexity())
            && self.evidence.promotion == Promotion::Accepted
            && valid_metrics(self.evidence.selection)
            && valid_metrics(audit)
            && self.domain.iter().all(|(low, high)| low <= high)
            && self.baseline.form() == RuleForm::Constant
            && self.baseline.valid_structure()
            && audit.accepted >= self.config.min_audit_accepted
            && audit.coverage() >= self.config.min_coverage
            && audit
                .selective_error()
                .is_some_and(|error| error <= self.config.max_error)
            && audit.baseline_brier - audit.brier >= self.config.min_brier_gain
            && self.seen <= self.config.drift_window
            && self.cursor < self.config.drift_window
            && self.window_errors == active_errors
            && self.errors[self.config.drift_window..]
                .iter()
                .all(|error| !error)
            && self.seen == self.last_event.min(self.config.drift_window as u64) as usize
            && self.cursor == (self.last_event % self.config.drift_window as u64) as usize
            && (self.seen == self.config.drift_window
                || self.errors[self.seen..self.config.drift_window]
                    .iter()
                    .all(|error| !error))
            && (self.seen == self.config.drift_window
                && self.window_errors as f64 / self.seen as f64 > self.config.max_window_error)
                == self.suspended
    }

    pub fn predict(
        &self,
        context: u64,
        features: &[i64; FEATURES],
    ) -> Result<Prediction, Abstention> {
        if self.suspended {
            return Err(Abstention::Suspended);
        }
        if context != self.evidence.id.context {
            return Err(Abstention::ContextMismatch);
        }
        if !in_domain(self.domain, features) {
            return Err(Abstention::OutsideDomain);
        }
        let probability = self.evidence.rule.estimate(features);
        if probability.confidence() < self.config.min_confidence {
            return Err(Abstention::LowConfidence);
        }
        Ok(Prediction {
            id: self.evidence.id,
            probability,
        })
    }

    /// Checks existing authority independently of statistical confidence; executes no effect.
    pub fn decide(
        &self,
        context: u64,
        features: &[i64; FEATURES],
        capability: &Capability,
        gate: &CapabilityGate,
    ) -> Result<Prediction, DecisionFailure> {
        gate.evaluate(capability).map_err(DecisionFailure::Gate)?;
        self.predict(context, features)
            .map_err(DecisionFailure::Abstain)
    }

    pub fn evaluate(&self, samples: &[Observation]) -> Result<EmpiricalMetrics, FoundryError> {
        if samples.is_empty() {
            return Err(FoundryError::TooFewSamples);
        }
        if samples.len() > MAX_SPLIT_SAMPLES {
            return Err(FoundryError::TooManySamples);
        }
        // Frozen-model diagnostics, not a new promotion or a change to audit evidence.
        Ok(metrics(
            self.evidence.rule,
            self.baseline,
            self.domain,
            self.config.min_confidence,
            samples,
        ))
    }

    /// Ordered test-then-monitor feedback. Rejected events cannot alter the monitor.
    /// A fixed-window alarm is a heuristic, NOT ADWIN or a statistical risk bound.
    pub fn observe_labeled(
        &mut self,
        event: u64,
        context: u64,
        features: &[i64; FEATURES],
        label: bool,
    ) -> Result<Feedback, FeedbackError> {
        if self.last_event.checked_add(1) != Some(event) {
            return Err(FeedbackError::OutOfOrder);
        }
        let prediction = self
            .predict(context, features)
            .map_err(FeedbackError::Abstain)?;
        let wrong = prediction.probability.label() != label;
        self.window_errors -= usize::from(self.errors[self.cursor]);
        self.errors[self.cursor] = wrong;
        self.window_errors += usize::from(wrong);
        self.cursor = (self.cursor + 1) % self.config.drift_window;
        self.seen = (self.seen + 1).min(self.config.drift_window);
        self.last_event = event;
        if self.seen == self.config.drift_window
            && self.window_errors as f64 / self.seen as f64 > self.config.max_window_error
        {
            self.suspended = true;
        }
        Ok(Feedback {
            correct: !wrong,
            window_samples: self.seen,
            window_errors: self.window_errors,
            suspended: self.suspended,
        })
    }
}

fn valid_metrics(metrics: EmpiricalMetrics) -> bool {
    metrics.samples > 0
        && metrics.accepted <= metrics.samples
        && metrics.errors <= metrics.accepted
        && [metrics.brier, metrics.baseline_brier, metrics.ece]
            .into_iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
}

fn push_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn push_capsule_id(bytes: &mut Vec<u8>, id: CapsuleId) {
    bytes.extend_from_slice(&id.context.to_le_bytes());
    bytes.extend_from_slice(&id.version.to_le_bytes());
}

fn push_rule(bytes: &mut Vec<u8>, rule: BinaryRule) {
    bytes.push(rule.feature.map_or(u8::MAX, |feature| feature as u8));
    bytes.extend_from_slice(&rule.threshold.to_le_bytes());
    bytes.push(rule.second_feature.map_or(u8::MAX, |feature| feature as u8));
    bytes.extend_from_slice(&rule.second_threshold.to_le_bytes());
    for probability in rule.cells {
        bytes.extend_from_slice(&probability.0.to_bits().to_le_bytes());
    }
}

fn push_metrics(bytes: &mut Vec<u8>, metrics: EmpiricalMetrics) {
    push_usize(bytes, metrics.samples);
    push_usize(bytes, metrics.accepted);
    push_usize(bytes, metrics.errors);
    bytes.extend_from_slice(&metrics.brier.to_bits().to_le_bytes());
    bytes.extend_from_slice(&metrics.baseline_brier.to_bits().to_le_bytes());
    bytes.extend_from_slice(&metrics.ece.to_bits().to_le_bytes());
}

fn promotion_byte(promotion: Promotion) -> u8 {
    match promotion {
        Promotion::Accepted => 0,
        Promotion::NoGain => 1,
        Promotion::InsufficientCoverage => 2,
        Promotion::ExcessiveError => 3,
    }
}

fn push_config(bytes: &mut Vec<u8>, config: FoundryConfig) {
    push_usize(bytes, config.min_leaf);
    push_usize(bytes, config.min_audit_accepted);
    push_usize(bytes, config.max_predicates);
    for value in [
        config.min_confidence,
        config.min_coverage,
        config.max_error,
        config.min_brier_gain,
        config.complexity_penalty,
    ] {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    push_usize(bytes, config.drift_window);
    bytes.extend_from_slice(&config.max_window_error.to_bits().to_le_bytes());
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.offset.checked_add(N)?;
        let value = self.bytes.get(self.offset..end)?.try_into().ok()?;
        self.offset = end;
        Some(value)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take::<1>()?[0])
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take()?))
    }

    fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take()?))
    }

    fn usize(&mut self) -> Option<usize> {
        usize::try_from(self.u64()?).ok()
    }

    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_bits(self.u64()?))
    }

    fn boolean(&mut self) -> Option<bool> {
        match self.u8()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    fn capsule_id(&mut self) -> Option<CapsuleId> {
        Some(CapsuleId {
            context: self.u64()?,
            version: self.u64()?,
        })
    }

    fn probability(&mut self) -> Option<Probability> {
        Probability::new(self.f64()?)
    }

    fn rule(&mut self) -> Option<BinaryRule> {
        let feature = self.feature()?;
        let threshold = self.i64()?;
        let second_feature = self.feature()?;
        let second_threshold = self.i64()?;
        let rule = BinaryRule {
            feature,
            threshold,
            second_feature,
            second_threshold,
            cells: [
                self.probability()?,
                self.probability()?,
                self.probability()?,
                self.probability()?,
            ],
        };
        rule.valid_structure().then_some(rule)
    }

    fn legacy_rule(&mut self) -> Option<BinaryRule> {
        let feature = self.feature()?;
        let threshold = self.i64()?;
        let low = self.probability()?;
        let high = self.probability()?;
        let rule = match feature {
            Some(feature) => BinaryRule::stump(feature, threshold, low, high),
            None if low == high => BinaryRule::constant(low),
            None => return None,
        };
        Some(rule)
    }

    fn feature(&mut self) -> Option<Option<usize>> {
        match self.u8()? {
            u8::MAX => Some(None),
            value if usize::from(value) < FEATURES => Some(Some(usize::from(value))),
            _ => None,
        }
    }

    fn metrics(&mut self) -> Option<EmpiricalMetrics> {
        Some(EmpiricalMetrics {
            samples: self.usize()?,
            accepted: self.usize()?,
            errors: self.usize()?,
            brier: self.f64()?,
            baseline_brier: self.f64()?,
            ece: self.f64()?,
        })
    }

    fn promotion(&mut self) -> Option<Promotion> {
        match self.u8()? {
            0 => Some(Promotion::Accepted),
            1 => Some(Promotion::NoGain),
            2 => Some(Promotion::InsufficientCoverage),
            3 => Some(Promotion::ExcessiveError),
            _ => None,
        }
    }

    fn config(&mut self) -> Option<FoundryConfig> {
        Some(FoundryConfig {
            min_leaf: self.usize()?,
            min_audit_accepted: self.usize()?,
            max_predicates: self.usize()?,
            min_confidence: self.f64()?,
            min_coverage: self.f64()?,
            max_error: self.f64()?,
            min_brier_gain: self.f64()?,
            complexity_penalty: self.f64()?,
            drift_window: self.usize()?,
            max_window_error: self.f64()?,
        })
    }

    fn legacy_config(&mut self) -> Option<FoundryConfig> {
        Some(FoundryConfig {
            min_leaf: self.usize()?,
            min_audit_accepted: self.usize()?,
            max_predicates: 1,
            min_confidence: self.f64()?,
            min_coverage: self.f64()?,
            max_error: self.f64()?,
            min_brier_gain: self.f64()?,
            complexity_penalty: self.f64()?,
            drift_window: self.usize()?,
            max_window_error: self.f64()?,
        })
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn checkpoint_checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(start: u64, count: usize) -> Vec<Observation> {
        (0..count)
            .map(|index| {
                let x = (index % 128) as i64 - 64;
                Observation {
                    id: start + index as u64,
                    features: [x.rem_euclid(3), 0, x, 0],
                    label: x >= 0,
                }
            })
            .collect()
    }

    fn push_legacy_rule(bytes: &mut Vec<u8>, rule: BinaryRule) {
        bytes.push(rule.feature.map_or(u8::MAX, |feature| feature as u8));
        bytes.extend_from_slice(&rule.threshold.to_le_bytes());
        let (low, high) = rule
            .probabilities()
            .expect("v1 supported only constant and threshold rules");
        bytes.extend_from_slice(&low.0.to_bits().to_le_bytes());
        bytes.extend_from_slice(&high.0.to_bits().to_le_bytes());
    }

    fn legacy_bytes(capsule: &SpecialistCapsule) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(CHECKPOINT_BYTES_V1);
        bytes.extend_from_slice(CHECKPOINT_MAGIC_V1);
        push_usize(&mut bytes, capsule.evidence.training_samples);
        push_capsule_id(&mut bytes, capsule.evidence.id);
        push_legacy_rule(&mut bytes, capsule.evidence.rule);
        push_metrics(&mut bytes, capsule.evidence.selection);
        push_metrics(&mut bytes, capsule.evidence.audit);
        bytes.push(promotion_byte(capsule.evidence.promotion));
        for (low, high) in capsule.domain {
            bytes.extend_from_slice(&low.to_le_bytes());
            bytes.extend_from_slice(&high.to_le_bytes());
        }
        push_legacy_rule(&mut bytes, capsule.baseline);
        push_usize(&mut bytes, capsule.config.min_leaf);
        push_usize(&mut bytes, capsule.config.min_audit_accepted);
        for value in [
            capsule.config.min_confidence,
            capsule.config.min_coverage,
            capsule.config.max_error,
            capsule.config.min_brier_gain,
            capsule.config.complexity_penalty,
        ] {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        push_usize(&mut bytes, capsule.config.drift_window);
        bytes.extend_from_slice(&capsule.config.max_window_error.to_bits().to_le_bytes());
        bytes.push(u8::from(capsule.suspended));
        bytes.extend_from_slice(&capsule.last_event.to_le_bytes());
        bytes.extend(capsule.errors.map(u8::from));
        push_usize(&mut bytes, capsule.seen);
        push_usize(&mut bytes, capsule.cursor);
        push_usize(&mut bytes, capsule.window_errors);
        assert_eq!(bytes.len(), CHECKPOINT_BYTES_V1 - 8);
        bytes.extend_from_slice(&checkpoint_checksum(&bytes).to_le_bytes());
        bytes
    }

    #[test]
    fn v1_checkpoint_migrates_without_changing_its_frozen_rule() {
        let config = FoundryConfig {
            max_predicates: 1,
            ..FoundryConfig::default()
        };
        let id = CapsuleId {
            context: 991,
            version: 7,
        };
        let capsule = train_specialist(id, config, &rows(0, 256), &rows(256, 128), &rows(384, 128))
            .unwrap()
            .capsule
            .unwrap();
        let bytes = legacy_bytes(&capsule);
        assert_eq!(bytes.len(), CHECKPOINT_BYTES_V1);
        let path = std::env::temp_dir().join(format!(
            "axon-specialist-v1-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(SpecialistCapsule::load(&path, id).unwrap(), capsule);
        std::fs::remove_file(path).unwrap();
    }
}
