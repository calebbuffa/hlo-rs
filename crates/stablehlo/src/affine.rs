//! Structured MLIR affine and semi-affine expressions, maps and integer sets.

use std::fmt;

mod evaluation;

/// The five binary expression kinds represented by MLIR's affine IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AffineBinaryOp {
    Add,
    Mul,
    FloorDiv,
    CeilDiv,
    Mod,
}

/// Dimensions and symbols use independent, zero-based positional namespaces.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AffineExpr {
    Dimension(usize),
    Symbol(usize),
    Constant(i64),
    Binary {
        op: AffineBinaryOp,
        lhs: Box<AffineExpr>,
        rhs: Box<AffineExpr>,
    },
}

/// A map's domain has `dimensions` dimensions and `symbols` parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AffineMap {
    pub dimensions: usize,
    pub symbols: usize,
    pub results: Vec<AffineExpr>,
}

/// A normalized constraint: `expression == 0` or `expression >= 0`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AffineConstraint {
    pub expression: AffineExpr,
    pub equality: bool,
}

/// A conjunction of constraints. An empty conjunction denotes the whole domain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntegerSet {
    pub dimensions: usize,
    pub symbols: usize,
    pub constraints: Vec<AffineConstraint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AffineError {
    DimensionOutOfBounds {
        position: usize,
        dimensions: usize,
    },
    SymbolOutOfBounds {
        position: usize,
        symbols: usize,
    },
    NonAffineProduct,
    NonSymbolicDivisor {
        op: AffineBinaryOp,
    },
    NestingLimit,
    ArgumentCount {
        symbols: bool,
        expected: usize,
        actual: usize,
    },
    DivisionByZero,
    ArithmeticOverflow,
}

impl AffineExpr {
    pub fn binary(op: AffineBinaryOp, lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Validates bindings and semi-affine operand restrictions.
    ///
    /// Like MLIR's parser, multiplication requires one dimension-free operand;
    /// division and modulo require a dimension-free right operand. Zero or
    /// negative divisors can be represented; evaluation diagnoses zero divisors.
    pub fn verify(&self, dimensions: usize, symbols: usize) -> Result<(), AffineError> {
        enum Task<'a> {
            Visit(&'a AffineExpr, usize),
            Combine(AffineBinaryOp),
        }
        let mut tasks = vec![Task::Visit(self, 0)];
        let mut dimension_free = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                Task::Visit(expression, depth) => {
                    if depth >= 128 {
                        return Err(AffineError::NestingLimit);
                    }
                    match expression {
                        Self::Dimension(position) => {
                            if *position >= dimensions {
                                return Err(AffineError::DimensionOutOfBounds {
                                    position: *position,
                                    dimensions,
                                });
                            }
                            dimension_free.push(false);
                        }
                        Self::Symbol(position) => {
                            if *position >= symbols {
                                return Err(AffineError::SymbolOutOfBounds {
                                    position: *position,
                                    symbols,
                                });
                            }
                            dimension_free.push(true);
                        }
                        Self::Constant(_) => dimension_free.push(true),
                        Self::Binary { op, lhs, rhs } => {
                            tasks.push(Task::Combine(*op));
                            tasks.push(Task::Visit(rhs, depth + 1));
                            tasks.push(Task::Visit(lhs, depth + 1));
                        }
                    }
                }
                Task::Combine(op) => {
                    let right = dimension_free.pop().expect("right affine child verified");
                    let left = dimension_free.pop().expect("left affine child verified");
                    match op {
                        AffineBinaryOp::Mul if !left && !right => {
                            return Err(AffineError::NonAffineProduct);
                        }
                        AffineBinaryOp::FloorDiv
                        | AffineBinaryOp::CeilDiv
                        | AffineBinaryOp::Mod
                            if !right =>
                        {
                            return Err(AffineError::NonSymbolicDivisor { op });
                        }
                        _ => (),
                    }
                    dimension_free.push(left && right);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn node_count(&self) -> usize {
        let mut pending = vec![self];
        let mut count = 0;
        while let Some(expression) = pending.pop() {
            count += 1;
            if let Self::Binary { lhs, rhs, .. } = expression {
                pending.push(lhs);
                pending.push(rhs);
            }
        }
        count
    }
}

impl AffineMap {
    pub fn verify(&self) -> Result<(), AffineError> {
        for expression in &self.results {
            expression.verify(self.dimensions, self.symbols)?;
        }
        Ok(())
    }
}

impl IntegerSet {
    pub fn verify(&self) -> Result<(), AffineError> {
        for constraint in &self.constraints {
            constraint
                .expression
                .verify(self.dimensions, self.symbols)?;
        }
        Ok(())
    }
}

impl fmt::Display for AffineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOutOfBounds {
                position,
                dimensions,
            } => write!(
                f,
                "affine dimension {position} is outside domain rank {dimensions}"
            ),
            Self::SymbolOutOfBounds { position, symbols } => write!(
                f,
                "affine symbol {position} is outside symbol count {symbols}"
            ),
            Self::NonAffineProduct => {
                write!(f, "affine multiplication requires a dimension-free operand")
            }
            Self::NonSymbolicDivisor { op } => {
                write!(f, "affine {op:?} requires a dimension-free right operand")
            }
            Self::NestingLimit => write!(f, "affine expression nesting limit exceeded"),
            Self::ArgumentCount {
                symbols,
                expected,
                actual,
            } => write!(
                f,
                "expected {expected} affine {} values, got {actual}",
                if *symbols { "symbol" } else { "dimension" }
            ),
            Self::DivisionByZero => write!(f, "affine expression division by zero"),
            Self::ArithmeticOverflow => write!(f, "affine expression arithmetic overflow"),
        }
    }
}

impl std::error::Error for AffineError {}
