//! Affine expressions use flat native attribute records for verification and evaluation.

use super::*;
use crate::Diagnostic;
use pliron::derive::pliron_attr;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AffineBinaryOp {
    Add = 3,
    Mul = 4,
    FloorDiv = 5,
    CeilDiv = 6,
    Mod = 7,
}
#[pliron_attr(
    name = "stablehlo.affine_node",
    format = "`<` $kind `,` $value `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AffineNode {
    pub kind: u8,
    pub value: i64,
}
#[pliron_attr(
    name = "stablehlo.affine_expr",
    format = "`[` vec($nodes, CharSpace(`,`)) `]`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AffineExpr {
    pub nodes: Vec<AffineNode>,
}
impl AffineExpr {
    #[allow(non_snake_case)]
    pub fn Dimension(position: usize) -> Self {
        Self {
            nodes: vec![AffineNode {
                kind: 0,
                value: position as i64,
            }],
        }
    }
    #[allow(non_snake_case)]
    pub fn Symbol(position: usize) -> Self {
        Self {
            nodes: vec![AffineNode {
                kind: 1,
                value: position as i64,
            }],
        }
    }
    #[allow(non_snake_case)]
    pub fn Constant(value: i64) -> Self {
        Self {
            nodes: vec![AffineNode { kind: 2, value }],
        }
    }
    pub fn binary(op: AffineBinaryOp, mut lhs: Self, rhs: Self) -> Self {
        lhs.nodes.extend(rhs.nodes);
        lhs.nodes.push(AffineNode {
            kind: op as u8,
            value: 0,
        });
        lhs
    }
    pub fn validate(&self, dimensions: usize, symbols: usize) -> Result<()> {
        let mut values = Vec::<(bool, usize)>::new();
        for node in &self.nodes {
            match node.kind {
                0 => {
                    let index =
                        usize::try_from(node.value).map_err(|_| Diagnostic::MalformedExpression)?;
                    if index >= dimensions {
                        return Err(Diagnostic::DimensionOutOfBounds {
                            position: index,
                            dimensions,
                        }
                        .into());
                    }
                    values.push((false, 1));
                }
                1 => {
                    let index =
                        usize::try_from(node.value).map_err(|_| Diagnostic::MalformedExpression)?;
                    if index >= symbols {
                        return Err(Diagnostic::SymbolOutOfBounds {
                            position: index,
                            symbols,
                        }
                        .into());
                    }
                    values.push((true, 1));
                }
                2 => values.push((true, 1)),
                3..=7 => {
                    let (right, rd) = values
                        .pop()
                        .ok_or_else(|| Diagnostic::MalformedExpression.into_error())?;
                    let (left, ld) = values
                        .pop()
                        .ok_or_else(|| Diagnostic::MalformedExpression.into_error())?;
                    if node.kind == 4 && !left && !right {
                        return Err(Diagnostic::NonAffineProduct.into());
                    }
                    if node.kind >= 5 && !right {
                        return Err(Diagnostic::NonSymbolicDivisor {
                            op: match node.kind {
                                5 => AffineBinaryOp::FloorDiv,
                                6 => AffineBinaryOp::CeilDiv,
                                _ => AffineBinaryOp::Mod,
                            },
                        }
                        .into());
                    }
                    let depth = ld.max(rd) + 1;
                    if depth > 128 {
                        return Err(Diagnostic::NestingLimit.into());
                    }
                    values.push((left && right, depth));
                }
                _ => return Err(Diagnostic::MalformedExpression.into()),
            }
        }
        if values.len() != 1 {
            return Err(Diagnostic::MalformedExpression.into());
        }
        Ok(())
    }
    pub fn evaluate_at(&self, dimensions: &[i64], symbols: &[i64]) -> Result<i128> {
        self.validate(dimensions.len(), symbols.len())?;
        let mut values = Vec::<i128>::new();
        for node in &self.nodes {
            match node.kind {
                0 => values.push(i128::from(dimensions[node.value as usize])),
                1 => values.push(i128::from(symbols[node.value as usize])),
                2 => values.push(i128::from(node.value)),
                _ => {
                    let rhs = values
                        .pop()
                        .ok_or_else(|| Diagnostic::MalformedExpression.into_error())?;
                    let lhs = values
                        .pop()
                        .ok_or_else(|| Diagnostic::MalformedExpression.into_error())?;
                    let value = match node.kind {
                        3 => lhs.checked_add(rhs),
                        4 => lhs.checked_mul(rhs),
                        5..=7 => {
                            if rhs == 0 {
                                return Err(Diagnostic::DivisionByZero.into());
                            }
                            let quotient = lhs
                                .checked_div(rhs)
                                .ok_or_else(|| Diagnostic::ArithmeticOverflow.into_error())?;
                            let remainder = lhs
                                .checked_rem(rhs)
                                .ok_or_else(|| Diagnostic::ArithmeticOverflow.into_error())?;
                            let opposite = remainder != 0 && (remainder < 0) != (rhs < 0);
                            match node.kind {
                                5 if opposite => quotient.checked_sub(1),
                                6 if remainder != 0 && !opposite => quotient.checked_add(1),
                                7 if opposite => remainder.checked_add(rhs),
                                7 => Some(remainder),
                                _ => Some(quotient),
                            }
                        }
                        _ => return Err(Diagnostic::MalformedExpression.into()),
                    }
                    .ok_or_else(|| Diagnostic::ArithmeticOverflow.into_error())?;
                    values.push(value);
                }
            }
        }
        values
            .pop()
            .ok_or_else(|| Diagnostic::MalformedExpression.into_error())
    }
}
impl Verify for AffineExpr {
    fn verify(&self, _: &Context) -> Result<()> {
        self.validate(usize::MAX, usize::MAX)
    }
}
#[pliron_attr(
    name = "stablehlo.affine_map",
    format = "`<` $dimensions `,` $symbols `,` `[` vec($results, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AffineMap {
    pub dimensions: usize,
    pub symbols: usize,
    pub results: Vec<AffineExpr>,
}
impl AffineMap {
    pub fn validate(&self) -> Result<()> {
        for result in &self.results {
            result.validate(self.dimensions, self.symbols)?;
        }
        Ok(())
    }
    pub fn evaluate(&self, dimensions: &[i64], symbols: &[i64]) -> Result<Vec<i64>> {
        self.validate()?;
        arguments(self.dimensions, dimensions, false)?;
        arguments(self.symbols, symbols, true)?;
        self.results
            .iter()
            .map(|e| {
                i64::try_from(e.evaluate_at(dimensions, symbols)?)
                    .map_err(|_| Diagnostic::ArithmeticOverflow.into_error())
            })
            .collect()
    }
}
impl Verify for AffineMap {
    fn verify(&self, _: &Context) -> Result<()> {
        self.validate()
    }
}
#[pliron_attr(
    name = "stablehlo.affine_constraint",
    format = "`<` $expression `,` $equality `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AffineConstraint {
    pub expression: AffineExpr,
    pub equality: bool,
}
#[pliron_attr(
    name = "stablehlo.integer_set",
    format = "`<` $dimensions `,` $symbols `,` `[` vec($constraints, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntegerSet {
    pub dimensions: usize,
    pub symbols: usize,
    pub constraints: Vec<AffineConstraint>,
}
impl IntegerSet {
    pub fn validate(&self) -> Result<()> {
        for c in &self.constraints {
            c.expression.validate(self.dimensions, self.symbols)?;
        }
        Ok(())
    }
    pub fn contains(&self, dimensions: &[i64], symbols: &[i64]) -> Result<bool> {
        self.validate()?;
        arguments(self.dimensions, dimensions, false)?;
        arguments(self.symbols, symbols, true)?;
        let mut contains = true;
        for c in &self.constraints {
            let value = c.expression.evaluate_at(dimensions, symbols)?;
            contains &= if c.equality { value == 0 } else { value >= 0 };
        }
        Ok(contains)
    }
}
impl Verify for IntegerSet {
    fn verify(&self, _: &Context) -> Result<()> {
        self.validate()
    }
}
fn arguments(expected: usize, actual: &[i64], symbols: bool) -> Result<()> {
    if expected != actual.len() {
        return Err(Diagnostic::ArgumentCount {
            symbols,
            expected,
            actual: actual.len(),
        }
        .into());
    }
    Ok(())
}
