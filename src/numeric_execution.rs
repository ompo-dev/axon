use super::{MAX_OPS, NumericError, NumericProgram, Op, finite, root};
use std::{collections::HashMap, mem::size_of};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Instruction {
    Input(u8),
    Constant(u64),
    Sqrt(u16),
    Binary(u8, u16, u16),
}

/// Structural sharing only: no algebraic rewrites or floating-point reassociation.
#[derive(Clone, Debug)]
pub struct NumericExecution {
    arity: usize,
    instructions: Vec<Instruction>,
    result: u16,
    original_instructions: usize,
}
impl NumericExecution {
    pub(super) fn compile(program: &NumericProgram) -> Self {
        let mut instructions = Vec::new();
        let mut known = HashMap::new();
        let mut stack = Vec::new();
        for op in &program.ops {
            let instruction = match *op {
                Op::Input(i) => Instruction::Input(i),
                Op::Constant(n) => Instruction::Constant(n.to_bits()),
                Op::Sqrt => Instruction::Sqrt(stack.pop().expect("validated")),
                _ => {
                    let b = stack.pop().expect("validated");
                    let a = stack.pop().expect("validated");
                    let tag = match op {
                        Op::Add => 2,
                        Op::Subtract => 3,
                        Op::Multiply => 4,
                        _ => 5,
                    };
                    Instruction::Binary(tag, a, b)
                }
            };
            let id = *known.entry(instruction).or_insert_with(|| {
                let id = instructions.len() as u16;
                instructions.push(instruction);
                id
            });
            stack.push(id);
        }
        Self {
            arity: program.arity,
            instructions,
            result: stack[0],
            original_instructions: program.instructions(),
        }
    }
    pub fn instructions(&self) -> usize {
        self.instructions.len()
    }
    pub fn eliminated_instructions(&self) -> usize {
        self.original_instructions - self.instructions.len()
    }
    pub fn scratch_bytes(&self) -> usize {
        size_of::<[f64; MAX_OPS]>()
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>() + self.instructions.capacity() * size_of::<Instruction>()
    }
    pub fn evaluate(&self, input: &[f64]) -> Result<f64, NumericError> {
        if input.len() != self.arity || input.iter().any(|n| !n.is_finite()) {
            return Err(NumericError::Domain("input arity or finite values"));
        }
        let mut values = [0.0; MAX_OPS];
        // First-occurrence order preserves the original guard order as well as f64 bits.
        for (i, instruction) in self.instructions.iter().enumerate() {
            values[i] = match *instruction {
                Instruction::Input(j) => input[j as usize],
                Instruction::Constant(bits) => f64::from_bits(bits),
                Instruction::Sqrt(a) => root(values[a as usize])?,
                Instruction::Binary(tag, a, b) => {
                    let op = match tag {
                        2 => Op::Add,
                        3 => Op::Subtract,
                        4 => Op::Multiply,
                        _ => Op::Divide,
                    };
                    op.binary(values[a as usize], values[b as usize])?
                }
            };
        }
        finite(values[self.result as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn same(program: &NumericProgram, input: &[f64]) {
        let execution = program.compile_execution();
        match (program.evaluate(input), execution.evaluate(input)) {
            (Ok(a), Ok(b)) => assert_eq!(a.to_bits(), b.to_bits()),
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
            (a, b) => panic!("changed evaluation: {a:?} != {b:?}"),
        }
    }
    #[test]
    fn repeated_subexpressions_are_shared_without_reordering_or_algebraic_rewrites() {
        let program = NumericProgram {
            arity: 2,
            ops: vec![
                Op::Input(0),
                Op::Input(1),
                Op::Multiply,
                Op::Input(0),
                Op::Input(1),
                Op::Multiply,
                Op::Add,
            ],
        };
        program.validate().unwrap();
        assert_eq!(program.compile_execution().instructions(), 4);
        for a in [-0.0, 0.0, 1e-300, 1e300, -7.3, f64::MAX, f64::MIN_POSITIVE] {
            for b in [-0.0, 0.0, 1e-300, 1e300, -5.25] {
                same(&program, &[a, b]);
            }
        }
    }
    #[test]
    fn ieee_edge_cases_and_first_domain_failure_are_preserved() {
        let programs = [
            NumericProgram {
                arity: 2,
                ops: vec![
                    Op::Input(0),
                    Op::Input(0),
                    Op::Multiply,
                    Op::Input(1),
                    Op::Input(1),
                    Op::Multiply,
                    Op::Add,
                    Op::Sqrt,
                ],
            },
            NumericProgram {
                arity: 2,
                ops: vec![
                    Op::Input(0),
                    Op::Sqrt,
                    Op::Constant(1.0),
                    Op::Input(1),
                    Op::Divide,
                    Op::Add,
                ],
            },
            NumericProgram {
                arity: 2,
                ops: vec![
                    Op::Input(0),
                    Op::Input(1),
                    Op::Subtract,
                    Op::Input(1),
                    Op::Input(0),
                    Op::Subtract,
                    Op::Multiply,
                ],
            },
        ];
        let mut bits = 0x1928b5973f122e55_u64;
        for _ in 0..30000 {
            bits ^= bits << 13;
            bits ^= bits >> 7;
            bits ^= bits << 17;
            let a = f64::from_bits(bits);
            bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1);
            let b = f64::from_bits(bits);
            for program in &programs {
                same(program, &[a, b]);
            }
        }
        same(&programs[1], &[-1.0, 0.0]);
        assert_eq!(
            programs[1]
                .compile_execution()
                .evaluate(&[-1.0, 0.0])
                .unwrap_err()
                .to_string(),
            "numerical domain guard: negative square root"
        );
    }
    #[test]
    fn nonassociative_arithmetic_and_signed_zero_constants_are_not_folded() {
        let program = NumericProgram {
            arity: 3,
            ops: vec![Op::Input(0), Op::Input(1), Op::Input(2), Op::Add, Op::Add],
        };
        same(&program, &[1e16, -1e16, 1.0]);
        assert_eq!(
            program
                .compile_execution()
                .evaluate(&[1e16, -1e16, 1.0])
                .unwrap(),
            0.0
        );
        let zero = NumericProgram {
            arity: 1,
            ops: vec![Op::Constant(-0.0), Op::Constant(0.0), Op::Add],
        };
        assert_eq!(zero.compile_execution().instructions(), 3);
        same(&zero, &[-0.0]);
    }
}
