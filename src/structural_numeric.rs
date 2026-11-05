//! Reuse the numerical foundry; retain only its validated pure program, not examples.
use super::{Error, Term};
use crate::numeric::{
    NumericBudget, NumericExample, NumericProgram, NumericSkill, NumericTolerance,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug)]
pub(super) struct Calculation {
    pub slots: Vec<usize>,
    pub(super) program: NumericProgram,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    slots: Vec<usize>,
    program: Vec<u8>,
}
impl Serialize for Calculation {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        Wire {
            slots: self.slots.clone(),
            program: self.program.encode_program(),
        }
        .serialize(s)
    }
}
impl<'de> Deserialize<'de> for Calculation {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        let program =
            NumericProgram::decode_program(&w.program).map_err(serde::de::Error::custom)?;
        if w.slots.len() != program.arity()
            || w.slots.iter().any(|i| *i >= 64)
            || w.slots
                .iter()
                .enumerate()
                .any(|(j, i)| w.slots[..j].contains(i))
        {
            return Err(serde::de::Error::custom("calculation slot signature"));
        }
        Ok(Self {
            slots: w.slots,
            program,
        })
    }
}
impl Calculation {
    pub fn learn(rows: &[Term], holes: &[Vec<Term>], work: &mut u64) -> Result<Self, Error> {
        let slots: Vec<_> = holes
            .iter()
            .enumerate()
            .filter(|(_, h)| {
                h.iter()
                    .all(|t| matches!(t,Term::Integer(n) if n.unsigned_abs() <= 1u64 << 53))
            })
            .map(|(i, _)| i)
            .collect();
        if slots.is_empty()
            || slots.len() > 4
            || rows
                .iter()
                .any(|t| !matches!(t,Term::Integer(n) if n.unsigned_abs() <= 1u64 << 53))
            || *work == 0
        {
            return Err(Error::Unsupported);
        }
        let examples = rows
            .iter()
            .enumerate()
            .map(|(j, t)| {
                let Term::Integer(y) = t else { unreachable!() };
                NumericExample::new(
                    slots
                        .iter()
                        .map(|i| {
                            let Term::Integer(n) = holes[*i][j] else {
                                unreachable!()
                            };
                            n as f64
                        })
                        .collect::<Vec<_>>(),
                    *y as f64,
                )
            })
            .collect();
        let skill = NumericSkill::learn(
            "derived",
            "structural/integer-v1",
            examples,
            &[],
            NumericBudget {
                max_cost: 8,
                max_terms: 30000,
                max_work: *work,
            },
            NumericTolerance {
                absolute: 0.0,
                relative: 0.0,
            },
        )
        .map_err(|_| Error::Unsupported)?;
        *work = work.saturating_sub(skill.search().scalar_work);
        Ok(Self {
            slots,
            program: skill.program().clone(),
        })
    }
    pub fn evaluate(&self, values: &[Option<Term>]) -> Result<Term, Error> {
        let mut input = Vec::new();
        for slot in &self.slots {
            let Some(Some(Term::Integer(n))) = values.get(*slot) else {
                return Err(Error::Invalid("calculation binding"));
            };
            input.push(*n);
        }
        self.program
            .evaluate_integer(&input)
            .map(Term::Integer)
            .map_err(|_| Error::Unsupported)
    }
    pub fn expression(&self) -> String {
        self.program.expression()
    }
    pub fn retained_bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<usize>() + self.program.instructions() * 16
    }
}
