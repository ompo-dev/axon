//! Pure collection programs. Fold is an iteration primitive, not a task-specific solver.
use super::{Error, Term};
use serde::{Deserialize, Serialize};

#[path = "algorithm_search.rs"]
mod search;
pub(crate) use search::learn_libraries;
pub use search::{Example, Policy, SearchBudget, SearchReceipt, learn};

#[path = "algorithm_polynomial.rs"]
mod polynomial;
pub use polynomial::{Representation, learn_polynomial};

pub(crate) fn learn_represented(
    examples: &[Example],
    libraries: &[Vec<Program>],
    budget: SearchBudget,
    representation: Representation,
) -> search::LibrarySearch {
    match representation {
        Representation::Enumerate => learn_libraries(examples, libraries, budget),
        Representation::Polynomial => search::LibrarySearch {
            result: learn_polynomial(examples, budget),
            windows: 0,
        },
    }
}

#[path = "algorithm_inquiry.rs"]
mod inquiry;
pub use inquiry::{InquiryEvent, InquiryPlan, InquiryReceipt, InquiryStrategy, inquire};

#[path = "algorithm_abstraction.rs"]
mod abstraction;

pub const MAX_ITEMS: usize = 2048;
const MAX_VALUE_NODES: usize = 3072;
pub const DEFAULT_FUEL: u64 = 2_000_000;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Integer(i64),
    Boolean(bool),
    Text(String),
    List(Vec<Value>),
}
impl Value {
    pub fn nodes(&self) -> usize {
        1 + match self {
            Self::List(v) => v.iter().map(Self::nodes).sum(),
            _ => 0,
        }
    }
    pub fn validate(&self) -> Result<(), Error> {
        fn walk(v: &Value, depth: usize, n: &mut usize, bytes: &mut usize) -> Result<(), Error> {
            *n += 1;
            *bytes += 16;
            if depth > 16 || *n > MAX_VALUE_NODES {
                return Err(Error::Budget);
            }
            match v {
                Value::Text(s) => {
                    if s.is_empty() || s.len() > 256 || s.chars().any(char::is_control) {
                        return Err(Error::Invalid("algorithm text bounds"));
                    }
                    *bytes += s.len();
                }
                Value::List(v) => {
                    if v.len() > MAX_ITEMS {
                        return Err(Error::Budget);
                    }
                    for x in v {
                        walk(x, depth + 1, n, bytes)?;
                    }
                }
                _ => (),
            }
            if *bytes > 131072 {
                return Err(Error::Budget);
            }
            Ok(())
        }
        walk(self, 0, &mut 0, &mut 0)
    }
    pub fn to_term(&self) -> Result<Term, Error> {
        self.encode_term(false)
    }
    pub fn to_compact_term(&self) -> Result<Term, Error> {
        self.encode_term(true)
    }
    pub(super) fn to_transport_term(&self) -> Result<Term, Error> {
        match self.to_term() {
            Err(Error::Budget) => self.to_compact_term(),
            result => result,
        }
    }
    fn encode_term(&self, compact: bool) -> Result<Term, Error> {
        self.validate()?;
        fn page(v: &[Value], compact: bool) -> Result<Term, Error> {
            if v.len() <= 30 {
                return Ok(Term::Tree(
                    std::iter::once(Term::Symbol("page".into()))
                        .chain(
                            v.iter()
                                .map(|v| v.encode_term(compact))
                                .collect::<Result<Vec<_>, _>>()?,
                        )
                        .collect(),
                ));
            }
            let width = v.len().div_ceil(30);
            Ok(Term::Tree(
                std::iter::once(Term::Symbol("branch".into()))
                    .chain(
                        v.chunks(width)
                            .map(|v| page(v, compact))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                    .collect(),
            ))
        }
        let t = match self {
            Self::Integer(n) => Term::Integer(*n),
            Self::Text(s) => Term::Symbol(s.clone()),
            Self::Boolean(b) => Term::Tree(vec![
                Term::Symbol("boolean".into()),
                Term::Symbol(b.to_string()),
            ]),
            Self::List(v) if v.is_empty() => Term::Tree(vec![
                Term::Symbol("sequence".into()),
                Term::Symbol("empty".into()),
            ]),
            Self::List(v) if compact && v.len() <= 30 => Term::Tree(
                std::iter::once(Term::Symbol("items".into()))
                    .chain(
                        v.iter()
                            .map(|v| v.encode_term(true))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                    .collect(),
            ),
            Self::List(v) => Term::Tree(vec![Term::Symbol("sequence".into()), page(v, compact)?]),
        };
        t.validate()?;
        Ok(t)
    }
    pub fn from_term(t: &Term) -> Result<Self, Error> {
        t.validate()?;
        fn flatten(t: &Term, out: &mut Vec<Value>) -> Result<(), Error> {
            let Term::Tree(v) = t else {
                return Err(Error::Unsupported);
            };
            match v.first() {
                Some(Term::Symbol(s)) if s == "page" => {
                    for t in &v[1..] {
                        out.push(Value::from_term(t)?);
                    }
                }
                Some(Term::Symbol(s)) if s == "branch" => {
                    for t in &v[1..] {
                        flatten(t, out)?;
                    }
                }
                _ => return Err(Error::Unsupported),
            }
            if out.len() > MAX_ITEMS {
                return Err(Error::Budget);
            }
            Ok(())
        }
        let value = match t {
            Term::Integer(n) => Self::Integer(*n),
            Term::Symbol(s) => Self::Text(s.clone()),
            Term::Tree(v) if v.first() == Some(&Term::Symbol("items".into())) => Self::List(
                v[1..]
                    .iter()
                    .map(Self::from_term)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Term::Tree(v) if v.len() == 2 && v[0] == Term::Symbol("sequence".into()) => {
                let mut out = Vec::new();
                if v[1] != Term::Symbol("empty".into()) {
                    flatten(&v[1], &mut out)?;
                }
                Self::List(out)
            }
            Term::Tree(v) if v.len() == 2 && v[0] == Term::Symbol("boolean".into()) => {
                match &v[1] {
                    Term::Symbol(s) if s == "true" => Self::Boolean(true),
                    Term::Symbol(s) if s == "false" => Self::Boolean(false),
                    _ => return Err(Error::Unsupported),
                }
            }
            _ => return Err(Error::Unsupported),
        };
        value.validate()?;
        Ok(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Expr {
    Input,
    Item(u8),
    Acc(u8),
    Literal(Value),
    Empty,
    Get(Box<Expr>, usize),
    List(Vec<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Subtract(Box<Expr>, Box<Expr>),
    Multiply(Box<Expr>, Box<Expr>),
    Remainder(Box<Expr>, Box<Expr>),
    Equal(Box<Expr>, Box<Expr>),
    Less(Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Push(Box<Expr>, Box<Expr>),
    Fold(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Box<Expr>),
}
impl Expr {
    pub fn nodes(&self) -> usize {
        1 + match self {
            Self::Get(a, _) => a.nodes(),
            Self::List(v) => v.iter().map(Self::nodes).sum(),
            Self::Add(a, b)
            | Self::Subtract(a, b)
            | Self::Multiply(a, b)
            | Self::Remainder(a, b)
            | Self::Equal(a, b)
            | Self::Less(a, b)
            | Self::Push(a, b)
            | Self::Call(a, b) => a.nodes() + b.nodes(),
            Self::If(a, b, c) | Self::Fold(a, b, c) => a.nodes() + b.nodes() + c.nodes(),
            _ => 0,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Expr", into = "Expr")]
pub struct Program {
    expression: Expr,
}
impl From<Program> for Expr {
    fn from(p: Program) -> Self {
        p.expression
    }
}
impl TryFrom<Expr> for Program {
    type Error = Error;
    fn try_from(e: Expr) -> Result<Self, Error> {
        Self::new(e)
    }
}
impl Program {
    pub(super) fn validate_at_scope(expression: &Expr, scope: u8) -> Result<(), Error> {
        fn check(e: &Expr, scope: u8, depth: usize, n: &mut usize) -> Result<(), Error> {
            *n += 1;
            if depth > 32 || *n > 256 || scope > 4 {
                return Err(Error::Budget);
            }
            match e {
                Expr::Item(i) | Expr::Acc(i) if *i >= scope => {
                    return Err(Error::Invalid("unbound iteration variable"));
                }
                Expr::Literal(Value::List(_)) => {
                    return Err(Error::Invalid("program cannot retain example collections"));
                }
                Expr::Literal(v) => v.validate()?,
                Expr::Get(a, i) => {
                    if *i > 31 {
                        return Err(Error::Invalid("field index"));
                    }
                    check(a, scope, depth + 1, n)?;
                }
                Expr::List(v) => {
                    if v.len() > 4 {
                        return Err(Error::Invalid("constructor arity"));
                    }
                    for x in v {
                        check(x, scope, depth + 1, n)?;
                    }
                }
                Expr::Call(body, arg) => {
                    check(body, 0, depth + 1, n)?;
                    check(arg, scope, depth + 1, n)?;
                }
                Expr::Fold(seq, init, body) => {
                    check(seq, scope, depth + 1, n)?;
                    check(init, scope, depth + 1, n)?;
                    check(body, scope + 1, depth + 1, n)?;
                }
                Expr::If(a, b, c) => {
                    for x in [a, b, c] {
                        check(x, scope, depth + 1, n)?;
                    }
                }
                Expr::Add(a, b)
                | Expr::Subtract(a, b)
                | Expr::Multiply(a, b)
                | Expr::Remainder(a, b)
                | Expr::Equal(a, b)
                | Expr::Less(a, b)
                | Expr::Push(a, b) => {
                    check(a, scope, depth + 1, n)?;
                    check(b, scope, depth + 1, n)?;
                }
                _ => (),
            }
            Ok(())
        }
        check(expression, scope, 0, &mut 0)
    }
    pub fn new(expression: Expr) -> Result<Self, Error> {
        Self::validate_at_scope(&expression, 0)?;
        Ok(Self { expression })
    }
    pub fn expression(&self) -> &Expr {
        &self.expression
    }
    pub(super) fn captured_calls(&self) -> Vec<&Expr> {
        fn walk<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
            match e {
                Expr::Call(body, arg) => {
                    out.push(body.as_ref());
                    walk(body, out);
                    walk(arg, out);
                }
                Expr::Get(a, _) => walk(a, out),
                Expr::List(v) => {
                    for e in v {
                        walk(e, out);
                    }
                }
                Expr::Fold(a, b, c) | Expr::If(a, b, c) => {
                    for e in [a, b, c] {
                        walk(e, out);
                    }
                }
                Expr::Add(a, b)
                | Expr::Subtract(a, b)
                | Expr::Multiply(a, b)
                | Expr::Remainder(a, b)
                | Expr::Equal(a, b)
                | Expr::Less(a, b)
                | Expr::Push(a, b) => {
                    walk(a, out);
                    walk(b, out);
                }
                _ => (),
            }
        }
        let mut out = Vec::new();
        walk(&self.expression, &mut out);
        out
    }
    pub fn evaluate(&self, input: &Value, fuel: u64) -> Result<Evaluation, Error> {
        input.validate()?;
        if fuel == 0 || fuel > 100_000_000 {
            return Err(Error::Invalid("algorithm fuel"));
        }
        let mut meter = Meter::new(fuel);
        let value = self.run(input, &mut meter)?;
        Ok(Evaluation {
            value,
            work: meter.work,
        })
    }
    pub(super) fn run(&self, input: &Value, meter: &mut Meter) -> Result<Value, Error> {
        let v = eval(&self.expression, input, &[], meter)?;
        v.validate()?;
        Ok(v)
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.expression.nodes() * std::mem::size_of::<Expr>()
            + serde_json::to_vec(self).map_or(0, |b| b.len())
    }
}
#[derive(Debug)]
pub struct Evaluation {
    pub value: Value,
    pub work: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Binding {
    pub slot: usize,
    pub program: Program,
}
impl Binding {
    pub fn evaluate(&self, values: &[Option<Term>]) -> Result<Term, Error> {
        let t = values
            .get(self.slot)
            .and_then(Option::as_ref)
            .ok_or(Error::Invalid("algorithm slot"))?;
        self.program
            .evaluate(&Value::from_term(t)?, DEFAULT_FUEL)?
            .value
            .to_transport_term()
    }
}
pub(super) struct Meter {
    pub work: u64,
    limit: u64,
}
impl Meter {
    pub fn new(limit: u64) -> Self {
        Self { work: 0, limit }
    }
    pub fn charge(&mut self, n: u64) -> Result<(), Error> {
        if n > self.limit.saturating_sub(self.work) {
            self.work = self.limit;
            return Err(Error::Budget);
        }
        self.work += n;
        Ok(())
    }
}
fn copied(v: &Value, m: &mut Meter) -> Result<Value, Error> {
    m.charge(v.nodes() as u64)?;
    Ok(v.clone())
}
type Frame<'a> = (&'a Value, &'a Value);
fn read<'a>(
    e: &Expr,
    input: &'a Value,
    frames: &[Frame<'a>],
    m: &mut Meter,
) -> Result<Option<&'a Value>, Error> {
    m.charge(1)?;
    Ok(match e {
        Expr::Input => Some(input),
        Expr::Acc(i) => frames.iter().rev().nth(*i as usize).map(|f| f.0),
        Expr::Item(i) => frames.iter().rev().nth(*i as usize).map(|f| f.1),
        Expr::Get(a, i) => match read(a, input, frames, m)? {
            Some(Value::List(v)) => v.get(*i),
            _ => None,
        },
        _ => None,
    })
}
fn step(
    e: &Expr,
    input: &Value,
    frames: &[Frame<'_>],
    mut acc: Value,
    acc_nodes: &mut usize,
    item: &Value,
    m: &mut Meter,
) -> Result<Value, Error> {
    let mut local = frames.to_vec();
    local.push((&acc, item));
    match e {
        Expr::If(c, a, b) => {
            m.charge(1)?;
            let Value::Boolean(yes) = eval(c, input, &local, m)? else {
                return Err(Error::Unsupported);
            };
            step(
                if yes { a } else { b },
                input,
                frames,
                acc,
                acc_nodes,
                item,
                m,
            )
        }
        Expr::Push(a, b) if **a == Expr::Acc(0) => {
            m.charge(2)?;
            let v = eval(b, input, &local, m)?;
            let nodes = v.nodes();
            m.charge(nodes as u64)?;
            let Value::List(out) = &mut acc else {
                return Err(Error::Unsupported);
            };
            if out.len() >= MAX_ITEMS || *acc_nodes + nodes > MAX_VALUE_NODES {
                return Err(Error::Budget);
            }
            out.push(v);
            *acc_nodes += nodes;
            Ok(acc)
        }
        Expr::Acc(0) => {
            m.charge(1)?;
            Ok(acc)
        }
        _ => {
            let v = eval(e, input, &local, m)?;
            let n = v.nodes();
            m.charge(n as u64)?;
            if n > MAX_VALUE_NODES {
                return Err(Error::Budget);
            }
            *acc_nodes = n;
            Ok(v)
        }
    }
}
fn eval(e: &Expr, input: &Value, frames: &[Frame<'_>], m: &mut Meter) -> Result<Value, Error> {
    m.charge(1)?;
    match e {
        Expr::Input | Expr::Item(_) | Expr::Acc(_) | Expr::Get(_, _) => {
            if let Some(v) = read(e, input, frames, m)? {
                return copied(v, m);
            }
            if let Expr::Get(a, i) = e
                && let Value::List(v) = eval(a, input, frames, m)?
            {
                return v
                    .get(*i)
                    .ok_or(Error::Unsupported)
                    .and_then(|v| copied(v, m));
            }
            Err(Error::Unsupported)
        }
        Expr::Empty => Ok(Value::List(vec![])),
        Expr::Literal(v) => copied(v, m),
        Expr::List(v) => {
            let value = Value::List(
                v.iter()
                    .map(|x| eval(x, input, frames, m))
                    .collect::<Result<_, _>>()?,
            );
            value.validate()?;
            Ok(value)
        }
        Expr::If(c, a, b) => {
            let Value::Boolean(yes) = eval(c, input, frames, m)? else {
                return Err(Error::Unsupported);
            };
            eval(if yes { a } else { b }, input, frames, m)
        }
        Expr::Fold(seq, init, body) => {
            let Value::List(items) = eval(seq, input, frames, m)? else {
                return Err(Error::Unsupported);
            };
            let mut acc = eval(init, input, frames, m)?;
            let mut nodes = acc.nodes();
            for item in &items {
                m.charge(1)?;
                acc = step(body, input, frames, acc, &mut nodes, item, m)?;
            }
            Ok(acc)
        }
        Expr::Call(body, arg) => {
            let arg = eval(arg, input, frames, m)?;
            arg.validate()?;
            eval(body, &arg, &[], m)
        }
        Expr::Push(a, b) => {
            let Value::List(mut out) = eval(a, input, frames, m)? else {
                return Err(Error::Unsupported);
            };
            if out.len() >= MAX_ITEMS {
                return Err(Error::Budget);
            }
            out.push(eval(b, input, frames, m)?);
            Ok(Value::List(out))
        }
        _ => {
            let (a, b) = match e {
                Expr::Add(a, b)
                | Expr::Subtract(a, b)
                | Expr::Multiply(a, b)
                | Expr::Remainder(a, b)
                | Expr::Equal(a, b)
                | Expr::Less(a, b) => (a, b),
                _ => return Err(Error::Unsupported),
            };
            let a = eval(a, input, frames, m)?;
            let b = eval(b, input, frames, m)?;
            if matches!(e, Expr::Equal(..)) {
                m.charge((a.nodes() + b.nodes()) as u64)?;
                return Ok(Value::Boolean(a == b));
            }
            if matches!(e, Expr::Less(..)) {
                return match (a, b) {
                    (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a < b)),
                    (Value::Text(a), Value::Text(b)) => {
                        m.charge((a.len() + b.len()) as u64)?;
                        Ok(Value::Boolean(a < b))
                    }
                    _ => Err(Error::Unsupported),
                };
            }
            let (Value::Integer(a), Value::Integer(b)) = (a, b) else {
                return Err(Error::Unsupported);
            };
            let n = match e {
                Expr::Add(..) => a.checked_add(b),
                Expr::Subtract(..) => a.checked_sub(b),
                Expr::Multiply(..) => a.checked_mul(b),
                Expr::Remainder(..) => a.checked_rem(b),
                _ => None,
            };
            n.map(Value::Integer).ok_or(Error::Unsupported)
        }
    }
}
