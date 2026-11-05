use super::*;

const SKILL_MAGIC: &[u8; 8] = b"AXRP0001";
const CURSOR_MAGIC: &[u8; 8] = b"AXRC0001";
fn u64b(b: &mut Vec<u8>, n: u64) {
    b.extend_from_slice(&n.to_le_bytes());
}
fn count(b: &mut Vec<u8>, n: usize) {
    b.extend_from_slice(&(n as u32).to_le_bytes());
}
fn text(b: &mut Vec<u8>, s: &str) {
    count(b, s.len());
    b.extend_from_slice(s.as_bytes());
}
pub(super) fn envelope(magic: &[u8; 8], payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(payload.len() + 24);
    b.extend_from_slice(magic);
    let len = payload.len() as u32;
    b.extend_from_slice(&len.to_le_bytes());
    b.extend_from_slice(&(!len).to_le_bytes());
    b.extend_from_slice(payload);
    let hash = checksum(&b);
    u64b(&mut b, hash);
    b
}
pub(super) fn unwrap<'a>(
    b: &'a [u8],
    magic: &[u8; 8],
    max: usize,
) -> Result<&'a [u8], ReactiveError> {
    if b.len() < 24 || b.len() > max || &b[..8] != magic {
        return Err(ReactiveError::Invalid("header"));
    }
    let len = u32::from_le_bytes(b[8..12].try_into().expect("fixed"));
    if !len != u32::from_le_bytes(b[12..16].try_into().expect("fixed"))
        || len as usize != b.len() - 24
        || checksum(&b[..b.len() - 8])
            != u64::from_le_bytes(b[b.len() - 8..].try_into().expect("fixed"))
    {
        return Err(ReactiveError::Invalid("length or checksum"));
    }
    Ok(&b[16..b.len() - 8])
}
fn encode_expr(b: &mut Vec<u8>, e: &Expr) {
    match e {
        Expr::Input(i) => {
            b.push(0);
            b.push(*i);
        }
        Expr::State => b.push(1),
        Expr::Tool(i) => {
            b.push(2);
            b.push(*i);
        }
        Expr::Constant(n) => {
            b.push(3);
            u64b(b, *n as u64);
        }
        Expr::Add(a, c) | Expr::Subtract(a, c) | Expr::Multiply(a, c) => {
            b.push(match e {
                Expr::Add(..) => 4,
                Expr::Subtract(..) => 5,
                _ => 6,
            });
            encode_expr(b, a);
            encode_expr(b, c);
        }
        Expr::IfLess(a, c, t, f) => {
            b.push(7);
            for e in [a, c, t, f] {
                encode_expr(b, e);
            }
        }
    }
}
impl ReactiveSkill {
    fn payload(&self) -> Vec<u8> {
        let mut b = Vec::new();
        self.encode(&mut b);
        b
    }
    fn encode(&self, b: &mut Vec<u8>) {
        text(b, &self.name);
        text(b, &self.context);
        b.push(self.arity as u8);
        b.push(u8::from(self.after_update));
        u64b(b, self.initial as u64);
        b.push(u8::from(self.update.is_some()));
        if let Some(e) = &self.update {
            encode_expr(b, e);
        }
        encode_expr(b, &self.output);
        count(b, self.tools.len());
        for tool in &self.tools {
            tool.encode(b);
        }
        count(b, self.traces.len());
        for trace in &self.traces {
            count(b, trace.events.len());
            for e in &trace.events {
                for n in &e.input {
                    u64b(b, *n as u64);
                }
                u64b(b, e.output as u64);
            }
        }
        for n in [
            self.search.candidates,
            self.search.work,
            self.search.guarded_candidates,
            self.search.refused_tools,
            self.search.elapsed_ns,
        ] {
            u64b(b, n);
        }
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.payload())
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        envelope(SKILL_MAGIC, &self.payload())
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self, ReactiveError> {
        let mut r = Reader(unwrap(b, SKILL_MAGIC, MAX_FILE as usize)?);
        let mut cells = 0;
        let skill = r.skill(0, &mut cells)?;
        if !r.0.is_empty() {
            return Err(ReactiveError::Invalid("trailing payload"));
        }
        skill.validate()?;
        Ok(skill)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), ReactiveError> {
        self.validate()?;
        let b = self.to_bytes();
        if b.len() as u64 > MAX_FILE {
            return Err(ReactiveError::Invalid("capsule size"));
        }
        publish_new(path, &b)?;
        Ok(())
    }
    pub fn open(path: &Path, context: &str) -> Result<Self, ReactiveError> {
        let skill = Self::from_bytes(&read_bounded(path, MAX_FILE)?)?;
        if skill.context() != context {
            return Err(ReactiveError::Context);
        }
        Ok(skill)
    }
}
impl ReactiveCursor {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        u64b(&mut b, self.identity);
        u64b(&mut b, self.ticks);
        b.push(u8::from(self.suspended));
        for n in self.states {
            u64b(&mut b, n as u64);
        }
        envelope(CURSOR_MAGIC, &b)
    }
    pub fn from_bytes(
        b: &[u8],
        circuit: &ReactiveCircuit,
        context: &str,
    ) -> Result<Self, ReactiveError> {
        if context != circuit.context {
            return Err(ReactiveError::Context);
        }
        let mut r = Reader(unwrap(b, CURSOR_MAGIC, 512)?);
        let identity = r.u64()?;
        let ticks = r.u64()?;
        let suspended = r.boolean()?;
        let mut states = [0; MAX_CELLS];
        for value in &mut states {
            *value = r.u64()? as i64;
        }
        if !r.0.is_empty() {
            return Err(ReactiveError::Invalid("cursor tail"));
        }
        if identity != circuit.identity {
            return Err(ReactiveError::Identity);
        }
        if states[circuit.cells.len()..].iter().any(|n| *n != 0)
            || circuit.cells.iter().enumerate().any(|(i, c)| {
                (c.update.is_none() && states[i] != 0) || (ticks == 0 && states[i] != c.initial)
            })
        {
            return Err(ReactiveError::Invalid("cursor state shape"));
        }
        Ok(Self {
            identity,
            ticks,
            states,
            suspended,
        })
    }
    pub fn save_new(&self, path: &Path) -> Result<(), ReactiveError> {
        publish_new(path, &self.to_bytes())?;
        Ok(())
    }
    pub fn open(
        path: &Path,
        circuit: &ReactiveCircuit,
        context: &str,
    ) -> Result<Self, ReactiveError> {
        Self::from_bytes(&read_bounded(path, 512)?, circuit, context)
    }
}
pub(super) struct Reader<'a>(pub(super) &'a [u8]);
impl Reader<'_> {
    pub(super) fn take(&mut self, n: usize) -> Result<&[u8], ReactiveError> {
        if n > self.0.len() {
            return Err(ReactiveError::Invalid("truncated payload"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn byte(&mut self) -> Result<u8, ReactiveError> {
        Ok(self.take(1)?[0])
    }
    fn boolean(&mut self) -> Result<bool, ReactiveError> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(ReactiveError::Invalid("boolean")),
        }
    }
    pub(super) fn u64(&mut self) -> Result<u64, ReactiveError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("fixed")))
    }
    pub(super) fn count(&mut self, max: usize) -> Result<usize, ReactiveError> {
        let n = u32::from_le_bytes(self.take(4)?.try_into().expect("fixed")) as usize;
        if n > max {
            return Err(ReactiveError::Invalid("count quota"));
        }
        Ok(n)
    }
    fn text(&mut self) -> Result<String, ReactiveError> {
        let n = self.count(256)?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| ReactiveError::Invalid("UTF-8"))
    }
    fn expr(&mut self, depth: usize, nodes: &mut usize) -> Result<Arc<Expr>, ReactiveError> {
        *nodes += 1;
        if depth > 16 || *nodes > MAX_EXPR {
            return Err(ReactiveError::Invalid("expression quota"));
        }
        let tag = self.byte()?;
        let e = match tag {
            0 => Expr::Input(self.byte()?),
            1 => Expr::State,
            2 => Expr::Tool(self.byte()?),
            3 => Expr::Constant(self.u64()? as i64),
            4..=6 => {
                let a = self.expr(depth + 1, nodes)?;
                let b = self.expr(depth + 1, nodes)?;
                match tag {
                    4 => Expr::Add(a, b),
                    5 => Expr::Subtract(a, b),
                    _ => Expr::Multiply(a, b),
                }
            }
            7 => Expr::IfLess(
                self.expr(depth + 1, nodes)?,
                self.expr(depth + 1, nodes)?,
                self.expr(depth + 1, nodes)?,
                self.expr(depth + 1, nodes)?,
            ),
            _ => return Err(ReactiveError::Invalid("instruction tag")),
        };
        Ok(Arc::new(e))
    }
    fn skill(&mut self, depth: usize, cells: &mut usize) -> Result<ReactiveSkill, ReactiveError> {
        *cells += 1;
        if depth > MAX_DEPTH || *cells > MAX_CELLS {
            return Err(ReactiveError::Invalid("tool graph quota"));
        }
        let name = self.text()?;
        let context = self.text()?;
        let arity = self.byte()? as usize;
        if !(1..=2).contains(&arity) {
            return Err(ReactiveError::Invalid("arity"));
        }
        let after_update = self.boolean()?;
        let initial = self.u64()? as i64;
        let update = if self.boolean()? {
            Some(self.expr(0, &mut 0)?)
        } else {
            None
        };
        let output = self.expr(0, &mut 0)?;
        let n = self.count(4)?;
        let mut tools = Vec::with_capacity(n);
        for _ in 0..n {
            tools.push(self.skill(depth + 1, cells)?);
        }
        let n = self.count(16)?;
        let mut traces = Vec::with_capacity(n);
        let mut total = 0;
        for _ in 0..n {
            let ticks = self.count(64)?;
            total += ticks;
            if total > 256 {
                return Err(ReactiveError::Invalid("sample quota"));
            }
            let mut events = Vec::with_capacity(ticks);
            for _ in 0..ticks {
                let mut input = Vec::with_capacity(arity);
                for _ in 0..arity {
                    input.push(self.u64()? as i64);
                }
                events.push(ReactiveEvent {
                    input,
                    output: self.u64()? as i64,
                });
            }
            traces.push(ReactiveTrace { events });
        }
        let search = ReactiveSearch {
            candidates: self.u64()?,
            work: self.u64()?,
            guarded_candidates: self.u64()?,
            refused_tools: self.u64()?,
            elapsed_ns: self.u64()?,
        };
        if search.candidates == 0
            || search.candidates > 200_000
            || search.work > 100_000_000
            || search.refused_tools > 4
            || search.guarded_candidates > search.candidates
        {
            return Err(ReactiveError::Invalid("search counters"));
        }
        Ok(ReactiveSkill {
            name,
            context,
            arity,
            after_update,
            initial,
            update,
            output,
            tools,
            traces,
            search,
        })
    }
}
