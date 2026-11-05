use axon_uic::world::WorldShape;
use std::collections::VecDeque;
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn pick(&mut self, n: u8) -> u8 {
        (self.next() % u64::from(n)) as u8
    }
}
#[derive(Clone, Debug)]
pub struct HiddenControl {
    positive: Vec<u8>,
    negative: Vec<u8>,
    effect: HiddenEffect,
}
#[derive(Clone, Debug)]
enum HiddenEffect {
    Toggle(Vec<u8>),
    Assign(u8, bool),
    Nothing,
}
pub struct HiddenWorld {
    pub shape: WorldShape,
    controls: Vec<HiddenControl>,
}
impl HiddenWorld {
    pub fn generate(rng: &mut Rng) -> Self {
        let bits = 3 + rng.pick(4);
        let actions = 2 + rng.pick(7);
        let shape = WorldShape::new(bits, actions).unwrap();
        let mut controls = Vec::new();
        for _ in 0..actions {
            let mut pool: Vec<u8> = (0..bits).collect();
            let mut positive = Vec::new();
            let mut negative = Vec::new();
            for _ in 0..rng.pick(3) {
                let index = rng.pick(pool.len() as u8) as usize;
                let bit = pool.remove(index);
                if rng.pick(2) == 0 {
                    positive.push(bit);
                } else {
                    negative.push(bit);
                }
            }
            let effect = match rng.pick(5) {
                0 => HiddenEffect::Nothing,
                1 => HiddenEffect::Assign(rng.pick(bits), false),
                2 => HiddenEffect::Assign(rng.pick(bits), true),
                _ => {
                    let first = rng.pick(bits);
                    let mut list = vec![first];
                    if rng.pick(2) == 1 {
                        let second = (first + 1 + rng.pick(bits - 1)) % bits;
                        list.push(second);
                    }
                    HiddenEffect::Toggle(list)
                }
            };
            controls.push(HiddenControl {
                positive,
                negative,
                effect,
            });
        }
        Self { shape, controls }
    }
    // Oracle uses Boolean arrays, not the learner's mask rule evaluator.
    pub fn step(&self, state: u8, action: u8) -> u8 {
        let mut bits: Vec<bool> = (0..self.shape.bits())
            .map(|b| state & (1 << b) != 0)
            .collect();
        let control = &self.controls[action as usize];
        if control.positive.iter().all(|&b| bits[b as usize])
            && control.negative.iter().all(|&b| !bits[b as usize])
        {
            match &control.effect {
                HiddenEffect::Toggle(list) => {
                    for &b in list {
                        bits[b as usize] = !bits[b as usize];
                    }
                }
                HiddenEffect::Assign(b, value) => bits[*b as usize] = *value,
                HiddenEffect::Nothing => (),
            }
        }
        bits.iter()
            .enumerate()
            .fold(0, |n, (b, &v)| n | (u8::from(v) << b))
    }
    pub fn shortest(&self, start: u8, target: u8) -> Option<usize> {
        let mut queue = VecDeque::from([(start, 0)]);
        let mut visited = vec![false; self.shape.states() as usize];
        visited[start as usize] = true;
        while let Some((state, depth)) = queue.pop_front() {
            if state == target {
                return Some(depth);
            }
            for action in 0..self.shape.actions() {
                let next = self.step(state, action);
                if !visited[next as usize] {
                    visited[next as usize] = true;
                    queue.push_back((next, depth + 1));
                }
            }
        }
        None
    }
}
