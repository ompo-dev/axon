//! Separate lab truth and adapters; not available to the learning API.
use crate::{behavior::BehaviorError, behavior_lab::Rng, factor::*};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::PathBuf,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Cycles,
    Workflow,
    Coupled,
}
#[derive(Clone)]
pub struct Environment {
    kind: Kind,
    dimensions: usize,
    bounds: [usize; MAX_FACTORS],
    initial: [usize; MAX_FACTORS],
    state: [usize; MAX_FACTORS],
    actions: Vec<u8>,
    sensors: Vec<usize>,
    tokens: Vec<Vec<u16>>,
    publications: Option<Vec<(Vec<u8>, PathBuf)>>,
}
impl Environment {
    pub fn random(kind: Kind, dimensions: usize, seed: u64) -> Result<Self, BehaviorError> {
        let actions = if kind == Kind::Workflow {
            3 * dimensions
        } else {
            dimensions
        };
        if !(2..=MAX_FACTORS).contains(&dimensions)
            || actions > 16
            || (kind == Kind::Coupled && dimensions != 2)
        {
            return Err(BehaviorError::Invalid("factor lab dimensions"));
        }
        let mut rng = Rng::new(seed);
        let mut bounds = [1; MAX_FACTORS];
        let mut initial = [0; MAX_FACTORS];
        let mut tokens = Vec::new();
        for i in 0..dimensions {
            bounds[i] = match kind {
                Kind::Workflow => 4,
                Kind::Coupled => 3,
                Kind::Cycles => 3 + rng.pick(2),
            };
            initial[i] = rng.pick(bounds[i]);
            let mut seen = BTreeSet::new();
            let mut values = Vec::new();
            for _ in 0..bounds[i] {
                let token = loop {
                    let t = rng.next_u64() as u16;
                    if seen.insert(t) {
                        break t;
                    }
                };
                values.push(token);
            }
            if kind == Kind::Workflow {
                values[1] = values[0];
                values[2] = values[0];
            }
            tokens.push(values);
        }
        let mut mapping = (0..actions as u8).collect::<Vec<_>>();
        for i in (1..mapping.len()).rev() {
            mapping.swap(i, rng.pick(i + 1));
        }
        let mut sensors = (0..dimensions).collect::<Vec<_>>();
        for i in (1..dimensions).rev() {
            sensors.swap(i, rng.pick(i + 1));
        }
        Ok(Self {
            kind,
            dimensions,
            bounds,
            initial,
            state: initial,
            actions: mapping,
            sensors,
            tokens,
            publications: None,
        })
    }
    pub fn zero_initial(mut self) -> Self {
        self.initial = [0; MAX_FACTORS];
        self.state = self.initial;
        self
    }
    pub fn publishing(mut self, files: Vec<(Vec<u8>, PathBuf)>) -> Result<Self, BehaviorError> {
        if self.kind != Kind::Workflow
            || files.len() != self.dimensions
            || self.initial != [0; MAX_FACTORS]
            || files.iter().any(|(bytes, _)| {
                bytes.len() < 20
                    || bytes.len() > 65536
                    || !bytes.starts_with(b"# ")
                    || bytes.contains(&0)
                    || std::str::from_utf8(bytes).is_err()
            })
        {
            return Err(BehaviorError::Invalid("factor publication payload"));
        }
        self.publications = Some(files);
        Ok(self)
    }
    pub fn actions(&self) -> u8 {
        self.actions.len() as u8
    }
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn states(&self) -> usize {
        self.bounds[..self.dimensions].iter().product()
    }
    pub fn initial_state(&self) -> u64 {
        self.encode(&self.initial)
    }
    fn encode(&self, state: &[usize; MAX_FACTORS]) -> u64 {
        let mut value = 0_u64;
        let mut scale = 1;
        for (i, &s) in state.iter().enumerate().take(self.dimensions) {
            value += s as u64 * scale;
            scale *= self.bounds[i] as u64;
        }
        value
    }
    fn decode(&self, mut value: u64) -> [usize; MAX_FACTORS] {
        let mut state = [0; MAX_FACTORS];
        for (i, s) in state.iter_mut().enumerate().take(self.dimensions) {
            *s = (value % self.bounds[i] as u64) as usize;
            value /= self.bounds[i] as u64;
        }
        state
    }
    fn update(&self, state: &mut [usize; MAX_FACTORS], action: u8) -> Result<(), BehaviorError> {
        let a = *self
            .actions
            .get(action as usize)
            .ok_or(BehaviorError::Invalid("factor lab action"))? as usize;
        match self.kind {
            Kind::Workflow => {
                let i = a / 3;
                match a % 3 {
                    0 => state[i] = 1,
                    1 if state[i] > 0 => state[i] = 2,
                    2 if state[i] == 2 => state[i] = 3,
                    _ => {}
                }
            }
            Kind::Cycles => state[a] = (state[a] + 1) % self.bounds[a],
            Kind::Coupled => {
                if a == 0 || state[0] == 2 {
                    state[a] = (state[a] + 1) % self.bounds[a];
                }
            }
        }
        Ok(())
    }
    fn frame(&self, state: &[usize; MAX_FACTORS]) -> Frame {
        let mut values = [0; MAX_FACTORS];
        for (sensor, &body) in self.sensors.iter().enumerate() {
            values[sensor] = self.tokens[body][state[body]];
        }
        Frame::new(&values[..self.dimensions]).expect("valid lab frame")
    }
    pub fn transition(&self, state: u64, action: u8) -> Result<(u64, Frame), BehaviorError> {
        if state >= self.states() as u64 {
            return Err(BehaviorError::Invalid("factor lab state"));
        }
        let mut state = self.decode(state);
        self.update(&mut state, action)?;
        Ok((self.encode(&state), self.frame(&state)))
    }
    /// Goal semantics are supplied by the adapter, not inferred from token names.
    pub fn goal_tokens(&self) -> Vec<(usize, u16)> {
        self.sensors
            .iter()
            .enumerate()
            .map(|(sensor, &body)| (sensor, self.tokens[body][self.bounds[body] - 1]))
            .collect()
    }
    pub fn target_tokens(&self, state: u64) -> Vec<(usize, u16)> {
        self.frame(&self.decode(state))
            .values()
            .iter()
            .copied()
            .enumerate()
            .collect()
    }
}
impl FrameResettable for Environment {
    fn reset(&mut self) -> Result<(), BehaviorError> {
        self.state = self.initial;
        Ok(())
    }
    fn step(&mut self, action: u8) -> Result<Frame, BehaviorError> {
        let mut next = self.state;
        self.update(&mut next, action)?;
        if let Some(files) = &self.publications {
            for (i, (payload, path)) in files.iter().enumerate() {
                if self.state[i] == 2 && next[i] == 3 {
                    crate::text_skill::publish_new(path, payload)?;
                }
            }
        }
        self.state = next;
        Ok(self.frame(&self.state))
    }
}
pub struct Audit {
    pub checked_edges: usize,
    pub reachable_pairs: usize,
    pub counterexample: Option<Vec<u8>>,
}
/// Whole reachable product audit, after learning; it never supplies refinement witnesses.
pub fn audit(
    model: &FactorModel,
    truth: &Environment,
    max_pairs: usize,
) -> Result<Audit, FactorError> {
    if model.actions() != truth.actions()
        || model.dimensions() != truth.dimensions()
        || max_pairs == 0
        || max_pairs > 65536
    {
        return Err(FactorError::Invalid("factor audit configuration"));
    }
    let root = (truth.initial_state(), [0; MAX_FACTORS]);
    let mut seen = BTreeSet::from([root]);
    let mut queue = VecDeque::from([(root, Vec::new())]);
    let mut edges = 0;
    while let Some(((state, learned), word)) = queue.pop_front() {
        for action in 0..truth.actions() {
            let (next, actual) = truth.transition(state, action)?;
            let (predicted_next, predicted) = model.transition(&learned, action)?;
            edges += 1;
            let mut extension = word.clone();
            extension.push(action);
            if actual != predicted {
                return Ok(Audit {
                    checked_edges: edges,
                    reachable_pairs: seen.len(),
                    counterexample: Some(extension),
                });
            }
            let pair = (next, predicted_next);
            if !seen.contains(&pair) {
                if seen.len() >= max_pairs {
                    return Err(FactorError::Limit("factor audit pairs"));
                }
                seen.insert(pair);
                queue.push_back((pair, extension));
            }
        }
    }
    Ok(Audit {
        checked_edges: edges,
        reachable_pairs: seen.len(),
        counterexample: None,
    })
}
/// Monolithic comparison encodes observed tuples, never true state IDs.
pub struct Flatten<'a, O> {
    oracle: &'a mut O,
    codes: BTreeMap<Frame, u16>,
    frames: Vec<Frame>,
}
impl<'a, O: FrameOracle> Flatten<'a, O> {
    pub fn new(oracle: &'a mut O) -> Self {
        Self {
            oracle,
            codes: BTreeMap::new(),
            frames: Vec::new(),
        }
    }
    pub fn frame(&self, token: u16) -> Option<Frame> {
        self.frames.get(token as usize).copied()
    }
}
impl<O: FrameOracle> crate::behavior::SequenceOracle for Flatten<'_, O> {
    fn query(&mut self, word: &[u8]) -> Result<crate::behavior::Reply, BehaviorError> {
        let frames = self.oracle.query(word)?;
        if frames.len() != word.len() {
            return Err(BehaviorError::Invalid("flatten reply"));
        }
        let mut reply = Vec::new();
        for frame in frames {
            let token = if let Some(&token) = self.codes.get(&frame) {
                token
            } else {
                if self.frames.len() > u16::MAX as usize {
                    return Err(BehaviorError::Limit("tuple alphabet"));
                }
                let token = self.frames.len() as u16;
                self.frames.push(frame);
                self.codes.insert(frame, token);
                token
            };
            reply.push(token);
        }
        Ok(reply)
    }
}
