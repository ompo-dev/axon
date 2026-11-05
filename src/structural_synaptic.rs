//! Bounded synaptic state, adapting the public BDH-GPU inference equations.
//! Mathematical port of Pathway BDH (MIT), Copyright 2025 Pathway Technology, Inc.
//! Pinned source and full notice: evals/fixtures/bdh-reference-v1/{bdh.py,LICENSE.md}.
//! Vector interface/quantization/transactional state/delta writes are AXON extensions.
use super::{
    Error,
    factorized::{Quantized, random},
};
use crate::live_store::checksum;
use serde::{Deserialize, Serialize};
use std::{io::Cursor, mem::size_of};

pub const CONTEXT: &str = "axon/bdh-synaptic-v1";
pub const MODULE: &str = "bdh-synaptic-v1";
const MAGIC: &[u8; 8] = b"AXSB0001";
const STATE_MAGIC: &[u8; 8] = b"AXSS0001";
const MAX_BYTES: usize = 196608;
const MAX_POSITION: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Position {
    None,
    Rotary,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Rule {
    Frozen,
    Hebbian,
    Delta,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Sequence,
    Association,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Genes {
    pub seed: u64,
    pub dimensions: usize,
    pub neurons: usize,
    pub layers: usize,
    pub position: Position,
    pub rule: Rule,
    pub decay: f32,
    pub learning_rate: f32,
}
impl Genes {
    fn validate(self) -> Result<(), Error> {
        if self.seed == 0
            || !(4..=32).contains(&self.dimensions)
            || !(16..=256).contains(&self.neurons)
            || !self.neurons.is_multiple_of(2)
            || !(1..=4).contains(&self.layers)
            || !self.decay.is_finite()
            || !(0.0 < self.decay && self.decay <= 1.0)
            || !self.learning_rate.is_finite()
            || !(0.0 < self.learning_rate && self.learning_rate <= 1.0)
        {
            return Err(Error::Invalid("synaptic genes"));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct Model {
    genes: Genes,
    encoder: Quantized,
    encoder_v: Quantized,
    decoder: Quantized,
    identity: u64,
}
#[derive(Clone, Debug)]
pub struct State {
    model: u64,
    scope: String,
    mode: Mode,
    position: usize,
    values: Vec<f32>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    revision: u8,
    genes: Genes,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateHeader {
    revision: u8,
    model: u64,
    scope: String,
    mode: Mode,
    position: usize,
    slots: usize,
}
#[derive(Serialize)]
pub struct Transition {
    pub output: Vec<f32>,
    pub position: usize,
    pub neurons_active: Vec<usize>,
    pub gates_active: Vec<usize>,
    pub synapses_written: usize,
    pub work: usize,
    pub live_bytes: usize,
    pub state_bytes: usize,
}
fn vector(v: &[f32], d: usize) -> Result<(), Error> {
    if v.len() != d || v.iter().any(|x| !x.is_finite() || x.abs() > 1e6) {
        return Err(Error::Invalid("synaptic vector"));
    }
    Ok(())
}
fn finite(v: &[f32]) -> Result<(), Error> {
    if v.iter().any(|x| !x.is_finite() || x.abs() > 1e9) {
        return Err(Error::Invalid("synaptic state bounds"));
    }
    Ok(())
}
fn normalize(v: &mut [f32], work: &mut usize) {
    let mean = v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len() as f64;
    let var = v
        .iter()
        .map(|x| (f64::from(*x) - mean).powi(2))
        .sum::<f64>()
        / v.len() as f64;
    let scale = (var + 1e-5).sqrt();
    for x in v.iter_mut() {
        *x = ((f64::from(*x) - mean) / scale) as f32;
    }
    *work += 6 * v.len();
}
fn project(v: &[f32], matrix: &Quantized, out: usize, work: &mut usize) -> Vec<f32> {
    let mut result = vec![0.0; out];
    for (i, &x) in v.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        for (j, y) in result.iter_mut().enumerate() {
            *y += x * (f32::from(matrix.values[i * out + j]) * matrix.scales[i]);
            *work += 3;
        }
    }
    result
}
fn read(values: &[f32], key: &[f32], d: usize, work: &mut usize) -> Vec<f32> {
    let mut out = vec![0.0; d];
    for (i, &k) in key.iter().enumerate() {
        if k == 0.0 {
            continue;
        }
        for (j, x) in out.iter_mut().enumerate() {
            *x += k * values[i * d + j];
            *work += 2;
        }
    }
    out
}
fn write(values: &mut [f32], key: &[f32], value: &[f32], rate: f32, work: &mut usize) -> usize {
    let d = value.len();
    let mut changed = 0;
    for (i, &k) in key.iter().enumerate() {
        if k == 0.0 {
            continue;
        }
        for (j, &v) in value.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let before = values[i * d + j];
            values[i * d + j] += rate * k * v;
            *work += 3;
            changed += usize::from(before != values[i * d + j]);
        }
    }
    changed
}
fn rotary(v: &mut [f32], position: usize, work: &mut usize) {
    let n = v.len();
    for i in (0..n).step_by(2) {
        let angle = (position as f64 / 65536f64.powf(i as f64 / n as f64)) % std::f64::consts::TAU;
        let (c, s) = (angle.cos() as f32, angle.sin() as f32);
        let (a, b) = (v[i], v[i + 1]);
        v[i] = a * c - b * s;
        v[i + 1] = b * c + a * s;
        *work += 12;
    }
}
fn scope_ok(scope: &str) -> bool {
    !scope.is_empty()
        && scope.len() <= 96
        && scope
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-./".contains(&b))
}
fn framed(magic: &[u8; 8], header: &impl Serialize) -> Result<Vec<u8>, Error> {
    let h = serde_json::to_vec(header).map_err(|e| Error::Storage(e.to_string()))?;
    let mut out = magic.to_vec();
    out.extend_from_slice(&(h.len() as u32).to_le_bytes());
    out.extend(h);
    Ok(out)
}
fn finish(mut bytes: Vec<u8>) -> Result<Vec<u8>, Error> {
    bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
    if bytes.len() > MAX_BYTES {
        return Err(Error::Budget);
    }
    Ok(bytes)
}
fn body<'a>(bytes: &'a [u8], magic: &[u8; 8]) -> Result<(&'a [u8], usize), Error> {
    if !(20..=MAX_BYTES).contains(&bytes.len()) || bytes.get(..8) != Some(magic) {
        return Err(Error::Invalid("synaptic format"));
    }
    let (b, hash) = bytes.split_at(bytes.len() - 8);
    if checksum(b).to_le_bytes() != hash {
        return Err(Error::Invalid("synaptic checksum"));
    }
    let end = 12usize
        .checked_add(u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize)
        .filter(|n| *n <= b.len())
        .ok_or(Error::Invalid("synaptic header"))?;
    Ok((b, end))
}
impl State {
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>() + self.scope.capacity() + self.values.capacity() * 4
    }
    pub fn position(&self) -> usize {
        self.position
    }
}
impl Model {
    pub fn new(genes: Genes) -> Result<Self, Error> {
        genes.validate()?;
        let (d, n) = (genes.dimensions, genes.neurons);
        let mut rng = genes.seed;
        let mut make = |rows, cols| {
            let v = (0..rows * cols)
                .map(|_| {
                    let u = ((random(&mut rng) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
                    let v = ((random(&mut rng) >> 11) as f64) / ((1u64 << 53) as f64);
                    ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos() * 0.02) as f32
                })
                .collect::<Vec<_>>();
            Quantized::encode(&v, cols)
        };
        let mut model = Self {
            genes,
            encoder: make(d, n),
            encoder_v: make(d, n),
            decoder: make(n, d),
            identity: 0,
        };
        model.identity = checksum(&model.to_bytes()?);
        Ok(model)
    }
    pub fn genes(&self) -> Genes {
        self.genes
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.encoder.retained_bytes()
            + self.encoder_v.retained_bytes()
            + self.decoder.retained_bytes()
    }
    pub fn reference_parameters(&self) -> [Vec<f32>; 3] {
        [
            (&self.encoder, self.genes.neurons),
            (&self.encoder_v, self.genes.neurons),
            (&self.decoder, self.genes.dimensions),
        ]
        .map(|(m, width)| {
            m.values
                .iter()
                .enumerate()
                .map(|(i, x)| f32::from(*x) * m.scales[i / width])
                .collect()
        })
    }
    pub fn begin(&self, scope: &str, mode: Mode) -> Result<State, Error> {
        if !scope_ok(scope) {
            return Err(Error::Invalid("synaptic scope"));
        }
        if (mode == Mode::Sequence
            && (self.genes.rule != Rule::Hebbian
                || self.genes.decay != 1.0
                || self.genes.learning_rate != 1.0))
            || (mode == Mode::Association && self.genes.position != Position::None)
        {
            return Err(Error::Invalid("synaptic mode/genes"));
        }
        let layers = if mode == Mode::Sequence {
            self.genes.layers
        } else {
            1
        };
        Ok(State {
            model: self.identity,
            scope: scope.into(),
            mode,
            position: 0,
            values: vec![0.0; layers * self.genes.neurons * self.genes.dimensions],
        })
    }
    fn check(&self, state: &State, mode: Mode) -> Result<(), Error> {
        let layers = if mode == Mode::Sequence {
            self.genes.layers
        } else {
            1
        };
        if state.model != self.identity
            || state.mode != mode
            || state.values.len() != layers * self.genes.neurons * self.genes.dimensions
            || state.position > MAX_POSITION
            || !scope_ok(&state.scope)
        {
            return Err(Error::Invalid("synaptic state identity"));
        }
        Ok(())
    }
    fn budget(
        &self,
        state: &State,
        writing: bool,
        max_work: usize,
        max_live: usize,
    ) -> Result<usize, Error> {
        let live = state.retained_bytes()
            + if writing { state.values.len() * 4 } else { 0 }
            + (5 * self.genes.neurons + 7 * self.genes.dimensions) * 4
            + size_of::<Transition>()
            + self.genes.layers * 2 * size_of::<usize>();
        if max_work == 0 || max_work > 2_000_000 || live > max_live {
            return Err(Error::Budget);
        }
        Ok(live)
    }
    pub fn step(
        &self,
        state: &mut State,
        input: &[f32],
        max_work: usize,
        max_live: usize,
    ) -> Result<Transition, Error> {
        self.check(state, Mode::Sequence)?;
        vector(input, self.genes.dimensions)?;
        if state.position == MAX_POSITION {
            return Err(Error::Budget);
        }
        let live = self.budget(state, true, max_work, max_live)?;
        let (d, n) = (self.genes.dimensions, self.genes.neurons);
        let mut pending = state.values.clone();
        let mut work = pending.len();
        let mut x = input.to_vec();
        normalize(&mut x, &mut work);
        let mut active = Vec::new();
        let mut gates = Vec::new();
        let mut written = 0;
        for layer in 0..self.genes.layers {
            let mut sparse = project(&x, &self.encoder, n, &mut work);
            for a in &mut sparse {
                *a = a.max(0.0);
            }
            active.push(sparse.iter().filter(|x| **x != 0.0).count());
            let mut key = sparse.clone();
            if self.genes.position == Position::Rotary {
                rotary(&mut key, state.position, &mut work);
            }
            let memory = &mut pending[layer * n * d..(layer + 1) * n * d];
            let mut y = read(memory, &key, d, &mut work);
            written += write(memory, &key, &x, 1.0, &mut work);
            normalize(&mut y, &mut work);
            let mut gate = project(&y, &self.encoder_v, n, &mut work);
            for (i, g) in gate.iter_mut().enumerate() {
                *g = g.max(0.0) * sparse[i];
                work += 2;
            }
            gates.push(gate.iter().filter(|x| **x != 0.0).count());
            let mut y = project(&gate, &self.decoder, d, &mut work);
            normalize(&mut y, &mut work);
            for (i, a) in x.iter_mut().enumerate() {
                *a += y[i];
                work += 1;
            }
            normalize(&mut x, &mut work);
        }
        finite(&pending)?;
        finite(&x)?;
        if work > max_work {
            return Err(Error::Budget);
        }
        state.values = pending;
        let position = state.position;
        state.position += 1;
        Ok(Transition {
            output: x,
            position,
            neurons_active: active,
            gates_active: gates,
            synapses_written: written,
            work,
            live_bytes: live,
            state_bytes: state.retained_bytes(),
        })
    }
    fn key(&self, input: &[f32], work: &mut usize) -> Result<Vec<f32>, Error> {
        vector(input, self.genes.dimensions)?;
        let mut x = input.to_vec();
        normalize(&mut x, work);
        let mut key = project(&x, &self.encoder, self.genes.neurons, work);
        for k in &mut key {
            *k = k.max(0.0);
        }
        let norm = key.iter().map(|k| k * k).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= 1e-12 {
            return Err(Error::Unsupported);
        }
        for k in &mut key {
            *k /= norm;
        }
        *work += 4 * key.len();
        Ok(key)
    }
    pub fn query(
        &self,
        state: &State,
        key: &[f32],
        max_work: usize,
        max_live: usize,
    ) -> Result<Transition, Error> {
        self.check(state, Mode::Association)?;
        let live = self.budget(state, false, max_work, max_live)?;
        let mut work = 0;
        let k = self.key(key, &mut work)?;
        let output = read(&state.values, &k, self.genes.dimensions, &mut work);
        finite(&output)?;
        if work > max_work {
            return Err(Error::Budget);
        }
        Ok(Transition {
            output,
            position: state.position,
            neurons_active: vec![k.iter().filter(|x| **x != 0.0).count()],
            gates_active: vec![],
            synapses_written: 0,
            work,
            live_bytes: live,
            state_bytes: state.retained_bytes(),
        })
    }
    pub fn associate(
        &self,
        state: &mut State,
        key: &[f32],
        value: &[f32],
        max_work: usize,
        max_live: usize,
    ) -> Result<Transition, Error> {
        self.check(state, Mode::Association)?;
        vector(value, self.genes.dimensions)?;
        if state.position == MAX_POSITION {
            return Err(Error::Budget);
        }
        let live = self.budget(state, true, max_work, max_live)?;
        let mut work = 0;
        let k = self.key(key, &mut work)?;
        let mut pending = state.values.clone();
        work += pending.len();
        if self.genes.rule != Rule::Frozen && self.genes.decay != 1.0 {
            for x in &mut pending {
                *x *= self.genes.decay;
                work += 1;
            }
        }
        let output = read(&pending, &k, self.genes.dimensions, &mut work);
        let update = if self.genes.rule == Rule::Delta {
            value
                .iter()
                .zip(&output)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>()
        } else {
            value.to_vec()
        };
        if self.genes.rule == Rule::Delta {
            work += update.len();
        }
        let written = if self.genes.rule == Rule::Frozen {
            0
        } else {
            write(
                &mut pending,
                &k,
                &update,
                self.genes.learning_rate,
                &mut work,
            )
        };
        finite(&pending)?;
        finite(&output)?;
        if work > max_work {
            return Err(Error::Budget);
        }
        state.values = pending;
        let position = state.position;
        state.position += 1;
        Ok(Transition {
            output,
            position,
            neurons_active: vec![k.iter().filter(|x| **x != 0.0).count()],
            gates_active: vec![],
            synapses_written: written,
            work,
            live_bytes: live,
            state_bytes: state.retained_bytes(),
        })
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut out = framed(
            MAGIC,
            &Header {
                revision: 1,
                genes: self.genes,
            },
        )?;
        self.encoder.append(&mut out);
        self.encoder_v.append(&mut out);
        self.decoder.append(&mut out);
        finish(out)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let (b, end) = body(bytes, MAGIC)?;
        let h: Header =
            serde_json::from_slice(&b[12..end]).map_err(|e| Error::Storage(e.to_string()))?;
        h.genes.validate()?;
        if h.revision != 1 {
            return Err(Error::Invalid("synaptic revision"));
        }
        let (d, n) = (h.genes.dimensions, h.genes.neurons);
        let mut c = Cursor::new(b);
        c.set_position(end as u64);
        let result = Self {
            genes: h.genes,
            encoder: Quantized::read(&mut c, d, n)?,
            encoder_v: Quantized::read(&mut c, d, n)?,
            decoder: Quantized::read(&mut c, n, d)?,
            identity: checksum(bytes),
        };
        if c.position() != b.len() as u64 {
            return Err(Error::Invalid("synaptic trailing bytes"));
        }
        Ok(result)
    }
    pub fn snapshot(&self, state: &State) -> Result<Vec<u8>, Error> {
        self.check(state, state.mode)?;
        let mut out = framed(
            STATE_MAGIC,
            &StateHeader {
                revision: 1,
                model: state.model,
                scope: state.scope.clone(),
                mode: state.mode,
                position: state.position,
                slots: state.values.len(),
            },
        )?;
        for x in &state.values {
            out.extend_from_slice(&x.to_le_bytes());
        }
        finish(out)
    }
    pub fn restore(&self, scope: &str, mode: Mode, bytes: &[u8]) -> Result<State, Error> {
        let (b, end) = body(bytes, STATE_MAGIC)?;
        let h: StateHeader =
            serde_json::from_slice(&b[12..end]).map_err(|e| Error::Storage(e.to_string()))?;
        let mut result = self.begin(scope, mode)?;
        if h.revision != 1
            || h.model != self.identity
            || h.scope != scope
            || h.mode != mode
            || h.slots != result.values.len()
            || h.position > MAX_POSITION
            || h.slots.checked_mul(4).and_then(|n| end.checked_add(n)) != Some(b.len())
        {
            return Err(Error::Invalid("synaptic checkpoint identity/shape"));
        }
        for (i, word) in b[end..].as_chunks::<4>().0.iter().enumerate() {
            result.values[i] = f32::from_le_bytes(*word);
        }
        finite(&result.values)?;
        result.position = h.position;
        Ok(result)
    }
}

#[cfg(test)]
#[path = "structural_synaptic_tests.rs"]
mod tests;
