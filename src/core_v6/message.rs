//! Mensagens relevantes e scheduler hierárquico determinístico.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::core_v5::CostVector;

use super::ids::FactorId;
use super::metrics::CognitiveMetrics;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerTier {
    Local,
    Regional,
    Global,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MessagePayload {
    Residual(String),
    Retrieval(String),
    Teaching(Vec<i16>),
    Control(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CognitiveMessage {
    pub from: FactorId,
    pub to: FactorId,
    pub payload: MessagePayload,
    pub residual_milliunits: u32,
    pub goal_milliunits: u32,
    pub uncertainty_milliunits: u32,
    pub information_gain_milliunits: u32,
    pub timestamp: u64,
    pub provenance: Vec<String>,
    pub cost: CostVector,
}

impl CognitiveMessage {
    pub fn priority_per_cost(&self) -> f64 {
        let numerator = self.residual_milliunits as f64
            + self.goal_milliunits as f64
            + self.uncertainty_milliunits as f64
            + self.information_gain_milliunits as f64;
        let denominator = 1.0
            + self.cost.compute_ops as f64
            + self.cost.bytes_moved as f64
            + self.cost.elapsed_ns as f64;
        numerator / denominator
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueuedMessage(CognitiveMessage);

impl Ord for QueuedMessage {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .priority_per_cost()
            .total_cmp(&other.0.priority_per_cost())
            .then_with(|| other.0.timestamp.cmp(&self.0.timestamp))
            .then_with(|| other.0.from.cmp(&self.0.from))
            .then_with(|| other.0.to.cmp(&self.0.to))
    }
}

impl PartialOrd for QueuedMessage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MessageDisposition {
    Suppressed,
    Enqueued(SchedulerTier),
}

#[derive(Clone, Debug)]
pub struct MessageScheduler {
    local_threshold: f64,
    regional_threshold: f64,
    regional: BinaryHeap<QueuedMessage>,
    global: BinaryHeap<QueuedMessage>,
}

impl MessageScheduler {
    pub fn new(local_threshold: f64, regional_threshold: f64) -> Self {
        Self {
            local_threshold,
            regional_threshold,
            regional: BinaryHeap::new(),
            global: BinaryHeap::new(),
        }
    }

    pub fn submit(
        &mut self,
        message: CognitiveMessage,
        metrics: &mut CognitiveMetrics,
    ) -> MessageDisposition {
        let priority = message.priority_per_cost();
        if priority < self.local_threshold {
            metrics.messages_suppressed = metrics.messages_suppressed.saturating_add(1);
            return MessageDisposition::Suppressed;
        }
        if priority < self.regional_threshold {
            self.regional.push(QueuedMessage(message));
            MessageDisposition::Enqueued(SchedulerTier::Regional)
        } else {
            self.global.push(QueuedMessage(message));
            MessageDisposition::Enqueued(SchedulerTier::Global)
        }
    }

    pub fn peek(&self) -> Option<&CognitiveMessage> {
        self.global
            .peek()
            .or_else(|| self.regional.peek())
            .map(|queued| &queued.0)
    }

    pub fn next(&mut self, metrics: &mut CognitiveMetrics) -> Option<CognitiveMessage> {
        let message = self.global.pop().or_else(|| self.regional.pop())?.0;
        metrics.messages_processed = metrics.messages_processed.saturating_add(1);
        metrics.bytes_read = metrics.bytes_read.saturating_add(message.cost.bytes_moved);
        Some(message)
    }
}

impl Default for MessageScheduler {
    fn default() -> Self {
        Self::new(0.05, 0.20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(residual: u32, bytes: u64) -> CognitiveMessage {
        CognitiveMessage {
            from: FactorId(1),
            to: FactorId(2),
            payload: MessagePayload::Residual("delta".to_string()),
            residual_milliunits: residual,
            goal_milliunits: 0,
            uncertainty_milliunits: 0,
            information_gain_milliunits: 0,
            timestamp: 1,
            provenance: vec!["test".to_string()],
            cost: CostVector::declared(1, bytes, 0, 0, 1),
        }
    }

    #[test]
    fn irrelevant_change_dies_at_the_local_threshold() {
        let mut scheduler = MessageScheduler::default();
        let mut metrics = CognitiveMetrics::default();

        assert_eq!(
            scheduler.submit(message(1, 100), &mut metrics),
            MessageDisposition::Suppressed
        );
        assert_eq!(metrics.messages_suppressed, 1);
        assert!(scheduler.next(&mut metrics).is_none());
    }

    #[test]
    fn important_message_reaches_global_queue_before_regional_work() {
        let mut scheduler = MessageScheduler::default();
        let mut metrics = CognitiveMetrics::default();
        scheduler.submit(message(30, 100), &mut metrics);
        scheduler.submit(message(1000, 1), &mut metrics);

        assert_eq!(
            scheduler.next(&mut metrics).unwrap().residual_milliunits,
            1000
        );
    }
}
