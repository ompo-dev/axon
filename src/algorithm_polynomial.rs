//! Opt-in, bounded exact interpolation. A fitted hypothesis is not a proof of its domain.
//! The host provides the degree/arity prior; examples provide every coefficient.
use super::search::SearchFailure;
use super::{Error, Example, Expr, Meter, Program, SearchBudget, SearchReceipt, Value};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Representation {
    #[default]
    Enumerate,
    Polynomial,
}
impl Representation {
    pub fn is_default(&self) -> bool {
        *self == Self::Enumerate
    }
}

// All rational arithmetic is checked i128. Overflow refuses the search; no floats.
#[derive(Clone, Copy)]
struct Q {
    n: i128,
    d: i128,
}
impl Q {
    fn integer(n: i128) -> Self {
        Self { n, d: 1 }
    }
    fn new(mut n: i128, mut d: i128, meter: &mut Meter) -> Result<Self, Error> {
        meter.charge(1)?;
        if d == 0 {
            return Err(Error::Invalid("polynomial zero denominator"));
        }
        if d < 0 {
            n = n.checked_neg().ok_or(Error::Budget)?;
            d = d.checked_neg().ok_or(Error::Budget)?;
        }
        let mut a = n.checked_abs().ok_or(Error::Budget)?;
        let mut b = d;
        while b != 0 {
            meter.charge(1)?;
            (a, b) = (b, a % b);
        }
        Ok(Self { n: n / a, d: d / a })
    }
    fn sub(self, rhs: Self, meter: &mut Meter) -> Result<Self, Error> {
        meter.charge(3)?;
        Self::new(
            self.n
                .checked_mul(rhs.d)
                .and_then(|a| rhs.n.checked_mul(self.d).and_then(|b| a.checked_sub(b)))
                .ok_or(Error::Budget)?,
            self.d.checked_mul(rhs.d).ok_or(Error::Budget)?,
            meter,
        )
    }
    fn mul(self, rhs: Self, meter: &mut Meter) -> Result<Self, Error> {
        meter.charge(2)?;
        Self::new(
            self.n.checked_mul(rhs.n).ok_or(Error::Budget)?,
            self.d.checked_mul(rhs.d).ok_or(Error::Budget)?,
            meter,
        )
    }
    fn div(self, rhs: Self, meter: &mut Meter) -> Result<Self, Error> {
        meter.charge(2)?;
        Self::new(
            self.n.checked_mul(rhs.d).ok_or(Error::Budget)?,
            self.d.checked_mul(rhs.n).ok_or(Error::Budget)?,
            meter,
        )
    }
}

fn coordinates(v: &Value) -> Result<Vec<i64>, Error> {
    let coords = match v {
        Value::Integer(x) => vec![*x],
        Value::List(v) if v.len() == 2 => v
            .iter()
            .map(|v| match v {
                Value::Integer(x) => Ok(*x),
                _ => Err(Error::Unsupported),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(Error::Unsupported),
    };
    if coords.iter().any(|x| x.unsigned_abs() > 4096) {
        return Err(Error::Invalid("polynomial training coordinate bounds"));
    }
    Ok(coords)
}

fn monomials(arity: usize, degree: usize) -> Vec<(usize, usize)> {
    (0..=degree)
        .flat_map(|d| (0..=if arity == 2 { d } else { 0 }).map(move |y| (d - y, y)))
        .collect()
}

// Reduced row echelon form; free columns are explicitly chosen as zero.
// Rank-deficient teaching therefore needs independent counterexamples/admission.
fn solve(
    mut rows: Vec<Vec<Q>>,
    columns: usize,
    meter: &mut Meter,
) -> Result<Option<Vec<i64>>, Error> {
    let mut pivots = Vec::new();
    for col in 0..columns {
        let mut selected = None;
        for (r, row) in rows.iter().enumerate().skip(pivots.len()) {
            meter.charge(1)?;
            if row[col].n != 0 {
                selected = Some(r);
                break;
            }
        }
        let Some(r) = selected else { continue };
        let pivot = pivots.len();
        rows.swap(pivot, r);
        let divisor = rows[pivot][col];
        for cell in rows[pivot].iter_mut().skip(col) {
            *cell = cell.div(divisor, meter)?;
        }
        let normalized = rows[pivot].clone();
        meter.charge(normalized.len() as u64)?;
        for (r, row) in rows.iter_mut().enumerate() {
            meter.charge(1)?;
            if r == pivot || row[col].n == 0 {
                continue;
            }
            let factor = row[col];
            for c in col..=columns {
                row[c] = row[c].sub(factor.mul(normalized[c], meter)?, meter)?;
            }
        }
        pivots.push(col);
    }
    for row in &rows {
        meter.charge(row.len() as u64)?;
        if row[..columns].iter().all(|v| v.n == 0) && row[columns].n != 0 {
            return Ok(None);
        }
    }
    let mut coefficients = vec![0; columns];
    for (r, col) in pivots.into_iter().enumerate() {
        meter.charge(1)?;
        let v = rows[r][columns];
        if v.d != 1 {
            return Ok(None);
        }
        coefficients[col] = i64::try_from(v.n).map_err(|_| Error::Budget)?;
    }
    Ok(Some(coefficients))
}

fn literal(n: i64) -> Expr {
    Expr::Literal(Value::Integer(n))
}
fn is_zero(e: &Expr) -> bool {
    matches!(e, Expr::Literal(Value::Integer(0)))
}
fn horner(coefficients: impl DoubleEndedIterator<Item = Expr>, variable: &Expr) -> Expr {
    let mut result = literal(0);
    for coefficient in coefficients.rev() {
        result = if is_zero(&result) {
            coefficient
        } else {
            let product = Expr::Multiply(Box::new(result), Box::new(variable.clone()));
            if is_zero(&coefficient) {
                product
            } else {
                Expr::Add(Box::new(product), Box::new(coefficient))
            }
        };
    }
    result
}
fn compile(arity: usize, terms: &[(usize, usize)], coefficients: &[i64]) -> Result<Program, Error> {
    let mut c = [[0; 4]; 4];
    for (&(x, y), &coefficient) in terms.iter().zip(coefficients) {
        c[x][y] = coefficient;
    }
    let x = if arity == 1 {
        Expr::Input
    } else {
        Expr::Get(Box::new(Expr::Input), 0)
    };
    let y = Expr::Get(Box::new(Expr::Input), 1);
    let powers = c.iter().map(|row| {
        if arity == 1 {
            literal(row[0])
        } else {
            horner(row.iter().map(|n| literal(*n)), &y)
        }
    });
    Program::new(horner(powers, &x))
}

/// Integer-coefficient total degree <=3, one scalar or exactly two integers.
/// No library, derivatives, target formula or future labels are consumed.
pub fn learn_polynomial(
    examples: &[Example],
    budget: SearchBudget,
) -> Result<SearchReceipt, SearchFailure> {
    let mut meter = Meter::new(budget.work);
    let mut candidates = 0;
    let result = (|| -> Result<Program, Error> {
        budget.policy.validate()?;
        if !(3..=32).contains(&examples.len())
            || !(1..=100000).contains(&budget.candidates)
            || !(1..=100_000_000).contains(&budget.work)
        {
            return Err(Error::Invalid("polynomial search bounds"));
        }
        let mut points = Vec::new();
        for (i, e) in examples.iter().enumerate() {
            meter.charge((i + 1) as u64)?;
            let point = coordinates(&e.input)?;
            if points
                .first()
                .is_some_and(|p: &Vec<i64>| p.len() != point.len())
                || examples[..i].iter().any(|old| old.input == e.input)
            {
                return Err(Error::Invalid("polynomial teaching shape or duplicate"));
            }
            if !matches!(e.output, Value::Integer(_)) {
                return Err(Error::Unsupported);
            }
            points.push(point);
        }
        let arity = points[0].len();
        for degree in 0..=3 {
            if candidates >= budget.candidates as u64 {
                return Err(Error::Budget);
            }
            candidates += 1;
            let terms = monomials(arity, degree);
            let mut rows = Vec::new();
            for (point, e) in points.iter().zip(examples) {
                let mut row = Vec::new();
                for &(x, y) in &terms {
                    meter.charge((x + y + 1) as u64)?;
                    let a = i128::from(point[0]).pow(x as u32);
                    let b = if arity == 2 {
                        i128::from(point[1]).pow(y as u32)
                    } else {
                        1
                    };
                    row.push(Q::integer(a * b)); // <=4096^3, established above.
                }
                let Value::Integer(output) = e.output else {
                    unreachable!()
                };
                row.push(Q::integer(i128::from(output)));
                rows.push(row);
            }
            if let Some(coefficients) = solve(rows, terms.len(), &mut meter)? {
                meter.charge(256)?; // bounded AST construction/validation allowance
                let program = compile(arity, &terms, &coefficients)?;
                for e in examples {
                    if program.run(&e.input, &mut meter)? != e.output {
                        return Err(Error::Unsupported);
                    }
                }
                return Ok(program);
            }
        }
        Err(Error::Unsupported)
    })();
    result
        .map(|program| SearchReceipt {
            program,
            candidates,
            work: meter.work,
            inherited: false,
            abstracted: false,
            templates: 0,
            mining_work: 0,
        })
        .map_err(|error| SearchFailure {
            error,
            candidates,
            work: meter.work,
            templates: 0,
            mining_work: 0,
        })
}
