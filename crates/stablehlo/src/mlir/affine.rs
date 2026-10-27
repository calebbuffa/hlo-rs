use std::collections::HashMap;

use super::{TextError, TextErrorKind, cursor::Cursor};
use crate::{
    attributes::{AffineBinaryOp, AffineConstraint, AffineExpr, AffineMap, IntegerSet},
    pliron::attribute::AttrObj,
};

fn constant(expression: &AffineExpr) -> Option<i64> {
    match expression.nodes.as_slice() {
        [node] if node.kind == 2 => Some(node.value),
        _ => None,
    }
}

fn binary_parts(expression: &AffineExpr) -> Option<(AffineBinaryOp, AffineExpr, AffineExpr)> {
    let last = expression.nodes.last()?;
    let op = match last.kind {
        3 => AffineBinaryOp::Add,
        4 => AffineBinaryOp::Mul,
        5 => AffineBinaryOp::FloorDiv,
        6 => AffineBinaryOp::CeilDiv,
        7 => AffineBinaryOp::Mod,
        _ => return None,
    };
    let mut required = 1_usize;
    let end = expression.nodes.len() - 1;
    for start in (0..end).rev() {
        if expression.nodes[start].kind >= 3 {
            required += 1;
        } else {
            required -= 1;
        }
        if required == 0 {
            return Some((
                op,
                AffineExpr {
                    nodes: expression.nodes[..start].to_vec(),
                },
                AffineExpr {
                    nodes: expression.nodes[start..end].to_vec(),
                },
            ));
        }
    }
    None
}

fn negate(expression: AffineExpr) -> AffineExpr {
    if let Some(value) = constant(&expression).and_then(i64::checked_neg) {
        AffineExpr::constant(value)
    } else {
        normalized_binary(AffineBinaryOp::Mul, expression, AffineExpr::constant(-1))
    }
}

fn subtract(lhs: AffineExpr, rhs: AffineExpr) -> AffineExpr {
    let negative = if let Some(value) = constant(&rhs) {
        AffineExpr::constant(value.wrapping_neg())
    } else {
        negate(rhs)
    };
    normalized_binary(AffineBinaryOp::Add, lhs, negative)
}

fn normalized_binary(
    mut op: AffineBinaryOp,
    mut lhs: AffineExpr,
    mut rhs: AffineExpr,
) -> AffineExpr {
    use AffineBinaryOp::{Add, CeilDiv, FloorDiv, Mod, Mul};
    if matches!(op, Add | Mul) && constant(&lhs).is_some() && constant(&rhs).is_none() {
        std::mem::swap(&mut lhs, &mut rhs);
    }
    while let Some((inner_op, inner, coefficient)) = binary_parts(&lhs) {
        let (Some(coefficient), Some(divisor)) = (constant(&coefficient), constant(&rhs)) else {
            break;
        };
        let rewrite = match (op, inner_op) {
            (Add, Add) => coefficient.checked_add(divisor).map(|value| (Add, value)),
            (Mul, Mul) => coefficient.checked_mul(divisor).map(|value| (Mul, value)),
            (FloorDiv | CeilDiv, Mul) if divisor != 0 => {
                let (coefficient, divisor) = (i128::from(coefficient), i128::from(divisor));
                (coefficient % divisor == 0)
                    .then(|| i64::try_from(coefficient / divisor).ok())
                    .flatten()
                    .map(|value| (Mul, value))
            }
            (Mod, Mul) if divisor > 0 && coefficient % divisor == 0 => {
                return AffineExpr::constant(0);
            }
            (Mod, Mod) if divisor > 0 && coefficient > 0 && coefficient % divisor == 0 => {
                Some((Mod, divisor))
            }
            _ => None,
        };
        let Some((next_op, next_constant)) = rewrite else {
            break;
        };
        lhs = inner;
        rhs = AffineExpr::constant(next_constant);
        op = next_op;
    }
    if let (Some(left), Some(right)) = (constant(&lhs), constant(&rhs)) {
        let (left, right) = (i128::from(left), i128::from(right));
        let value = match op {
            Add => Some(left + right),
            Mul => Some(left * right),
            FloorDiv | CeilDiv if right != 0 => {
                let quotient = left / right;
                let remainder = left % right;
                let opposite = remainder != 0 && (remainder < 0) != (right < 0);
                Some(match op {
                    FloorDiv if opposite => quotient - 1,
                    CeilDiv if remainder != 0 && !opposite => quotient + 1,
                    _ => quotient,
                })
            }
            Mod if right > 0 => Some(left.rem_euclid(right)),
            _ => None,
        };
        if let Some(value) = value.and_then(|value| i64::try_from(value).ok()) {
            return AffineExpr::constant(value);
        }
    }
    match (op, constant(&lhs), constant(&rhs)) {
        (Add, _, Some(0)) | (Mul | FloorDiv | CeilDiv, _, Some(1)) => lhs,
        (Add, Some(0), _) | (Mul, Some(1), _) => rhs,
        (Mul, _, Some(0)) | (Mul, Some(0), _) | (Mod, _, Some(1)) => AffineExpr::constant(0),
        _ => AffineExpr::binary(op, lhs, rhs),
    }
}

impl Cursor<'_> {
    pub(super) fn affine_attribute(
        &mut self,
        is_set: bool,
        depth: usize,
    ) -> Result<AttrObj, TextError> {
        self.expect("<")?;
        let mut names = HashMap::new();
        let dimensions = self.affine_bindings(&mut names, false)?;
        let symbols = if self.starts("[") {
            self.affine_bindings(&mut names, true)?
        } else {
            0
        };
        self.expect(if is_set { ":" } else { "->" })?;
        self.expect("(")?;
        let mut pending = Vec::<AffineExpr>::new();
        let mut equalities = Vec::new();
        if !self.take(")") {
            loop {
                let lhs = self.affine_expression(&names, dimensions, symbols, depth + 1)?;
                pending.push(lhs);
                if is_set {
                    let (equality, reverse) = if self.take(">") {
                        self.expect("=")?;
                        (false, false)
                    } else if self.take("<") {
                        self.expect("=")?;
                        (false, true)
                    } else {
                        self.expect("=")?;
                        self.expect("=")?;
                        (true, false)
                    };
                    let rhs = self.affine_expression(&names, dimensions, symbols, depth + 1)?;
                    let lhs = pending
                        .pop()
                        .ok_or_else(|| self.error(TextErrorKind::Expected("affine expression")))?;
                    let expression = if reverse {
                        subtract(rhs, lhs)
                    } else {
                        subtract(lhs, rhs)
                    };
                    pending.push(expression);
                    pending
                        .last()
                        .ok_or_else(|| self.error(TextErrorKind::Expected("affine expression")))?
                        .validate(dimensions, symbols)
                        .map_err(|e| self.affine_error(e))?;
                    equalities.push(equality);
                }
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        self.expect(">")?;
        let results = std::mem::take(&mut pending);
        Ok(if is_set {
            let mut constraints = results
                .into_iter()
                .zip(equalities)
                .map(|(expression, equality)| AffineConstraint {
                    expression,
                    equality,
                })
                .collect::<Vec<_>>();
            if constraints.is_empty() {
                constraints.push(AffineConstraint {
                    expression: AffineExpr::constant(0),
                    equality: true,
                });
            }
            Box::new(IntegerSet {
                dimensions,
                symbols,
                constraints,
            }) as AttrObj
        } else {
            Box::new(AffineMap {
                dimensions,
                symbols,
                results,
            }) as AttrObj
        })
    }

    fn affine_bindings(
        &mut self,
        names: &mut HashMap<String, AffineExpr>,
        symbol: bool,
    ) -> Result<usize, TextError> {
        self.expect(if symbol { "[" } else { "(" })?;
        let end = if symbol { "]" } else { ")" };
        let mut count = 0;
        if !self.take(end) {
            loop {
                let name = self.word()?.to_owned();
                let expression = if symbol {
                    AffineExpr::symbol(count)
                } else {
                    AffineExpr::dimension(count)
                };
                if names.insert(name.clone(), expression).is_some() {
                    return Err(self.error(TextErrorKind::DuplicateField(name)));
                }
                count += 1;
                if self.take(end) {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(count)
    }

    pub(super) fn affine_expression(
        &mut self,
        names: &HashMap<String, AffineExpr>,
        dimensions: usize,
        symbols: usize,
        depth: usize,
    ) -> Result<AffineExpr, TextError> {
        self.depth(depth)?;
        let mut pending = Vec::<AffineExpr>::new();
        let values = &mut pending;
        let mut operators = Vec::new();
        let mut expecting_operand = true;
        loop {
            self.space();
            if expecting_operand {
                if self.take("(") {
                    operators.push(AffineParseOperator::Parenthesis);
                    self.depth(depth + operator_nesting(&operators))?;
                    continue;
                }
                if self.take("-") {
                    operators.push(AffineParseOperator::Negate);
                    self.depth(depth + operator_nesting(&operators))?;
                    continue;
                }
                let value = if self
                    .input
                    .as_bytes()
                    .get(self.offset)
                    .is_some_and(u8::is_ascii_digit)
                {
                    let value = self.integer()?;
                    // MLIR accepts 2^63 as INT64_MIN, including printer-emitted
                    // subtraction of its positive magnitude.
                    let value = if value == 1_i128 << 63 {
                        i64::MIN
                    } else {
                        i64::try_from(value)
                            .map_err(|_| self.error(TextErrorKind::InvalidNumber))?
                    };
                    AffineExpr::constant(value)
                } else {
                    let name = self.word()?;
                    names.get(name).cloned().ok_or_else(|| {
                        self.error(TextErrorKind::Expected("declared affine identifier"))
                    })?
                };
                values.push(value);
                expecting_operand = false;
                reduce_affine_negations(values, &mut operators, dimensions, symbols, self)?;
                continue;
            }

            if self.input[self.offset..].starts_with(")") {
                if operators
                    .iter()
                    .any(|operator| matches!(operator, AffineParseOperator::Parenthesis))
                {
                    self.offset += 1;
                    reduce_affine_parenthesis(values, &mut operators, dimensions, symbols, self)?;
                    continue;
                }
                break;
            }

            let Some((token, op, precedence, subtraction)) = affine_binary_operator(self) else {
                break;
            };
            self.offset += token.len();
            while matches!(
                operators.last(),
                Some(AffineParseOperator::Binary {
                    precedence: top_precedence,
                    ..
                }) if *top_precedence >= precedence
            ) {
                reduce_affine_binary(values, &mut operators, dimensions, symbols, self)?;
            }
            operators.push(AffineParseOperator::Binary {
                op,
                precedence,
                subtraction,
            });
            expecting_operand = true;
        }

        if expecting_operand {
            return Err(self.error(TextErrorKind::Expected("affine expression operand")));
        }
        if operators
            .iter()
            .any(|operator| matches!(operator, AffineParseOperator::Parenthesis))
        {
            return Err(self.error(TextErrorKind::Expected(")")));
        }
        while !operators.is_empty() {
            match operators.last() {
                Some(AffineParseOperator::Negate) => {
                    reduce_affine_negations(values, &mut operators, dimensions, symbols, self)?
                }
                Some(AffineParseOperator::Binary { .. }) => {
                    reduce_affine_binary(values, &mut operators, dimensions, symbols, self)?
                }
                Some(AffineParseOperator::Parenthesis) => {
                    return Err(self.error(TextErrorKind::Expected(")")));
                }
                None => break,
            }
        }
        if values.len() != 1 {
            return Err(self.error(TextErrorKind::Expected("affine expression")));
        }
        values
            .last()
            .ok_or_else(|| self.error(TextErrorKind::Expected("affine expression")))?
            .validate(dimensions, symbols)
            .map_err(|error| self.affine_error(error))?;
        values
            .pop()
            .ok_or_else(|| self.error(TextErrorKind::Expected("affine expression")))
    }

    fn affine_error(&self, error: crate::pliron::result::Error) -> TextError {
        self.error(TextErrorKind::InvalidAttribute(error.to_string()))
    }
}

#[derive(Clone, Copy)]
enum AffineParseOperator {
    Parenthesis,
    Negate,
    Binary {
        op: AffineBinaryOp,
        precedence: u8,
        subtraction: bool,
    },
}

fn operator_nesting(operators: &[AffineParseOperator]) -> usize {
    operators
        .iter()
        .filter(|operator| {
            matches!(
                operator,
                AffineParseOperator::Parenthesis | AffineParseOperator::Negate
            )
        })
        .count()
}

fn affine_binary_operator(cursor: &Cursor<'_>) -> Option<(&'static str, AffineBinaryOp, u8, bool)> {
    if cursor.input[cursor.offset..].starts_with("+") {
        Some(("+", AffineBinaryOp::Add, 1, false))
    } else if cursor.input[cursor.offset..].starts_with("-") {
        Some(("-", AffineBinaryOp::Add, 1, true))
    } else if cursor.input[cursor.offset..].starts_with("*") {
        Some(("*", AffineBinaryOp::Mul, 2, false))
    } else {
        let rest = &cursor.input[cursor.offset..];
        let keyword = rest
            .split(|c: char| !c.is_ascii_alphanumeric() && !matches!(c, '_' | '.' | '$'))
            .next()
            .unwrap_or("");
        match keyword {
            "floordiv" => Some(("floordiv", AffineBinaryOp::FloorDiv, 2, false)),
            "ceildiv" => Some(("ceildiv", AffineBinaryOp::CeilDiv, 2, false)),
            "mod" => Some(("mod", AffineBinaryOp::Mod, 2, false)),
            _ => None,
        }
    }
}

fn reduce_affine_negations(
    values: &mut Vec<AffineExpr>,
    operators: &mut Vec<AffineParseOperator>,
    dimensions: usize,
    symbols: usize,
    cursor: &Cursor<'_>,
) -> Result<(), TextError> {
    while matches!(operators.last(), Some(AffineParseOperator::Negate)) {
        operators.pop();
        let value = values
            .pop()
            .ok_or_else(|| cursor.error(TextErrorKind::Expected("affine expression operand")))?;
        let value = negate(value);
        values.push(value);
        values
            .last()
            .ok_or_else(|| cursor.error(TextErrorKind::Expected("affine expression")))?
            .validate(dimensions, symbols)
            .map_err(|error| cursor.affine_error(error))?;
    }
    Ok(())
}

fn reduce_affine_parenthesis(
    values: &mut Vec<AffineExpr>,
    operators: &mut Vec<AffineParseOperator>,
    dimensions: usize,
    symbols: usize,
    cursor: &Cursor<'_>,
) -> Result<(), TextError> {
    while !matches!(operators.last(), Some(AffineParseOperator::Parenthesis)) {
        match operators.last() {
            Some(AffineParseOperator::Negate) => {
                reduce_affine_negations(values, operators, dimensions, symbols, cursor)?
            }
            Some(AffineParseOperator::Binary { .. }) => {
                reduce_affine_binary(values, operators, dimensions, symbols, cursor)?
            }
            Some(AffineParseOperator::Parenthesis) => break,
            None => return Err(cursor.error(TextErrorKind::Expected("("))),
        }
    }
    operators.pop();
    reduce_affine_negations(values, operators, dimensions, symbols, cursor)
}

fn reduce_affine_binary(
    values: &mut Vec<AffineExpr>,
    operators: &mut Vec<AffineParseOperator>,
    dimensions: usize,
    symbols: usize,
    cursor: &Cursor<'_>,
) -> Result<(), TextError> {
    let Some(AffineParseOperator::Binary {
        op, subtraction, ..
    }) = operators.pop()
    else {
        return Err(cursor.error(TextErrorKind::Expected("affine binary operator")));
    };
    let rhs = values
        .pop()
        .ok_or_else(|| cursor.error(TextErrorKind::Expected("affine expression operand")))?;
    let Some(lhs) = values.pop() else {
        values.push(rhs);
        return Err(cursor.error(TextErrorKind::Expected("affine expression operand")));
    };
    let value = if subtraction {
        subtract(lhs, rhs)
    } else {
        normalized_binary(op, lhs, rhs)
    };
    values.push(value);
    values
        .last()
        .ok_or_else(|| cursor.error(TextErrorKind::Expected("affine expression")))?
        .validate(dimensions, symbols)
        .map_err(|error| cursor.affine_error(error))?;
    Ok(())
}
