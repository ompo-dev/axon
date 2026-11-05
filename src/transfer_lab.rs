//! Controlled categorical environments. Physical robotics and natural language are not modeled.
use crate::behavior::{BehaviorError, Resettable};
use crate::behavior_lab::Rng;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Family {
    Grid,
    Protocol,
    Pipeline,
}
impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Self::Grid => "discrete-control",
            Self::Protocol => "symbolic-protocol",
            Self::Pipeline => "document-workflow",
        }
    }
}
#[derive(Clone)]
pub struct Environment {
    pub family: Family,
    pub extent: usize,
    initial: usize,
    state: usize,
    payload: Vec<u8>,
    publication: Option<PathBuf>,
}
impl Environment {
    pub fn new(family: Family, extent: usize, initial: usize) -> Self {
        assert!((2..=5).contains(&extent));
        let env = Self {
            family,
            extent,
            initial,
            state: initial,
            payload: b"# AXON transfer\n\nA document published through an adapted interface.\n"
                .to_vec(),
            publication: None,
        };
        assert!(initial < env.states());
        env
    }
    pub fn publishing(payload: Vec<u8>, path: PathBuf) -> Self {
        Self {
            family: Family::Pipeline,
            extent: 2,
            initial: 0,
            state: 0,
            payload,
            publication: Some(path),
        }
    }
    pub fn states(&self) -> usize {
        match self.family {
            Family::Grid => self.extent * self.extent,
            Family::Protocol => self.extent + 1,
            Family::Pipeline => 5,
        }
    }
    pub fn initial(&self) -> usize {
        self.initial
    }
    pub fn transition(&self, state: usize, action: u8) -> Result<(usize, u16), BehaviorError> {
        if state >= self.states() || action >= 4 {
            return Err(BehaviorError::Invalid("lab interface"));
        }
        Ok(match self.family {
            Family::Grid => {
                let x = state % self.extent;
                let y = state / self.extent;
                let (x, y) = match action {
                    0 => (x, y.saturating_sub(1)),
                    1 => ((x + 1).min(self.extent - 1), y),
                    2 => (x, (y + 1).min(self.extent - 1)),
                    _ => (x.saturating_sub(1), y),
                };
                let next = y * self.extent + x;
                (next, 1000 + next as u16)
            }
            Family::Protocol => match action {
                0 if state < self.extent => (state + 1, 100),
                1 if state > 0 => (state - 1, 100),
                2 if state == 0 => (state, 201),
                3 => (state, 100),
                _ => (state, 409),
            },
            Family::Pipeline => match action {
                0 => (1, 100),
                1 if state > 0 => (2, 100),
                2 if state == 2 => (3, 100),
                3 if state >= 3 => (4, 201),
                3 => (state, 409),
                _ => (state, 100),
            },
        })
    }
    pub fn symbols(&self) -> Vec<u16> {
        let mut symbols = BTreeSet::new();
        for state in 0..self.states() {
            for action in 0..4 {
                symbols.insert(self.transition(state, action).expect("valid").1);
            }
        }
        symbols.into_iter().collect()
    }
}
impl Resettable for Environment {
    fn reset(&mut self) -> Result<(), BehaviorError> {
        self.state = self.initial;
        Ok(())
    }
    fn step(&mut self, action: u8) -> Result<u16, BehaviorError> {
        let (next, output) = self.transition(self.state, action)?;
        if self.family == Family::Pipeline
            && action == 3
            && self.state == 3
            && let Some(path) = &self.publication
        {
            if self.payload.len() < 20
                || self.payload.len() > 65_536
                || !self.payload.starts_with(b"# ")
                || self.payload.contains(&0)
                || std::str::from_utf8(&self.payload).is_err()
            {
                return Err(BehaviorError::Invalid("publication payload"));
            }
            crate::text_skill::publish_new(path, &self.payload)?;
        }
        self.state = next;
        Ok(output)
    }
}

#[derive(Clone)]
pub struct Rewired {
    pub environment: Environment,
    actions: [u8; 4],
    outputs: BTreeMap<u16, u16>,
}
impl Rewired {
    pub fn random(environment: Environment, rng: &mut Rng) -> Self {
        let mut actions = [0, 1, 2, 3];
        for i in (1..4).rev() {
            actions.swap(i, rng.pick(i + 1));
        }
        let mut used = BTreeSet::new();
        let mut outputs = BTreeMap::new();
        for source in environment.symbols() {
            let target = loop {
                let n = rng.next_u64() as u16;
                if used.insert(n) {
                    break n;
                }
            };
            outputs.insert(source, target);
        }
        Self {
            environment,
            actions,
            outputs,
        }
    }
    pub fn output_for(&self, source: u16) -> u16 {
        self.outputs[&source]
    }
    pub fn action_map(&self) -> [u8; 4] {
        self.actions
    }
    pub fn transition(&self, state: usize, action: u8) -> Result<(usize, u16), BehaviorError> {
        let mapped = *self
            .actions
            .get(action as usize)
            .ok_or(BehaviorError::Invalid("rewired action"))?;
        let (next, out) = self.environment.transition(state, mapped)?;
        Ok((next, self.outputs[&out]))
    }
}
impl Resettable for Rewired {
    fn reset(&mut self) -> Result<(), BehaviorError> {
        self.environment.reset()
    }
    fn step(&mut self, action: u8) -> Result<u16, BehaviorError> {
        let mapped = *self
            .actions
            .get(action as usize)
            .ok_or(BehaviorError::Invalid("rewired action"))?;
        let out = self.environment.step(mapped)?;
        Ok(self.outputs[&out])
    }
}
