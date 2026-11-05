use axon_uic::{BeliefStatus, EpisodeStamp, EpisodeStore, EpisodeValue, FactKey};
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct Event {
    pub id: u64,
    pub source: String,
    pub at: u64,
    pub key: FactKey,
    pub value: Option<i64>,
    pub replaces: Option<u64>,
}

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn key(&mut self) -> FactKey {
        FactKey::new(
            format!("entity-{}", self.next() % 32),
            "level",
            format!("site-{}", self.next() % 2),
        )
    }
}

pub fn generate(store: &mut EpisodeStore, count: usize, rng: &mut Rng) -> Vec<Event> {
    assert!(store.is_empty());
    let mut events = Vec::<Event>::new();
    let mut replaced = BTreeSet::new();
    for i in 0..count {
        let roll = rng.next() % 10;
        let candidates = events
            .iter()
            .filter(|event| event.value.is_some() && !replaced.contains(&event.id))
            .collect::<Vec<_>>();
        let target = if roll < 3 && !candidates.is_empty() {
            Some(candidates[rng.next() as usize % candidates.len()])
        } else {
            None
        };
        let label = format!("stream-{i}");
        let value = (rng.next() % 5) as i64;
        let event = if let Some(target) = target {
            let retract = roll == 0;
            let receipt = if retract {
                store.retract(target.id, &target.source, &label).unwrap()
            } else {
                store
                    .revise(
                        target.id,
                        &target.source,
                        &label,
                        EpisodeValue::Integer(value),
                    )
                    .unwrap()
            };
            replaced.insert(target.id);
            Event {
                id: receipt.id,
                source: target.source.clone(),
                at: target.at,
                key: target.key.clone(),
                value: (!retract).then_some(value),
                replaces: Some(target.id),
            }
        } else {
            let key = rng.key();
            let source = format!("sensor-{}", rng.next() % 4);
            let at = rng.next() % 80;
            let receipt = store
                .observe(
                    EpisodeStamp::new(&source, label, at),
                    key.clone(),
                    EpisodeValue::Integer(value),
                )
                .unwrap();
            Event {
                id: receipt.id,
                source,
                at,
                key,
                value: Some(value),
                replaces: None,
            }
        };
        events.push(event);
    }
    events
}

// Deliberately scans the independent input log: no production postings, summaries or helpers.
pub fn reference(
    events: &[Event],
    key: &FactKey,
    at: u64,
    known: u64,
) -> (BeliefStatus, Vec<(u64, String, u64, i64)>) {
    let visible = events
        .iter()
        .filter(|event| event.id <= known && event.at <= at && &event.key == key)
        .collect::<Vec<_>>();
    let current = visible
        .iter()
        .filter(|event| !visible.iter().any(|other| other.replaces == Some(event.id)))
        .collect::<Vec<_>>();
    let mut evidence = current
        .iter()
        .filter(|event| {
            !current
                .iter()
                .any(|other| other.source == event.source && other.at > event.at)
        })
        .filter_map(|event| {
            event
                .value
                .map(|value| (event.id, event.source.clone(), event.at, value))
        })
        .collect::<Vec<_>>();
    evidence.sort();
    let values = evidence.iter().map(|e| e.3).collect::<BTreeSet<_>>();
    let status = match values.len() {
        0 => BeliefStatus::Missing,
        1 => BeliefStatus::Consistent,
        _ => BeliefStatus::Conflict,
    };
    (status, evidence)
}

pub fn compare(store: &EpisodeStore, events: &[Event], key: &FactKey, at: u64, known: u64) {
    let expected = reference(events, key, at, known);
    let actual = store.belief(key, Some(at), Some(known)).unwrap();
    let mut evidence = actual
        .evidence
        .iter()
        .map(|e| {
            let EpisodeValue::Integer(value) = e.value else {
                panic!("fixture only has integer observations");
            };
            (e.episode_id, e.source.clone(), e.effective_at, value)
        })
        .collect::<Vec<_>>();
    evidence.sort();
    assert_eq!(
        (actual.status, evidence),
        expected,
        "key={key:?} at={at} known={known}"
    );
}
