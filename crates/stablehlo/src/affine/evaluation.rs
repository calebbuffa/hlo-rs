use super::{AffineBinaryOp, AffineError, AffineExpr, AffineMap, IntegerSet};

fn arguments(expected: usize, values: &[i64], symbols: bool) -> Result<(), AffineError> {
    if values.len() != expected {
        return Err(AffineError::ArgumentCount {
            symbols,
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

impl AffineExpr {
    fn evaluate_at(&self, dimensions: &[i64], symbols: &[i64]) -> Result<i128, AffineError> {
        let overflow = AffineError::ArithmeticOverflow;
        let mut tasks = vec![(self, false)];
        let mut values = Vec::new();
        while let Some((expression, ready)) = tasks.pop() {
            match expression {
                Self::Constant(value) => values.push(i128::from(*value)),
                Self::Dimension(position) => values.push(i128::from(dimensions[*position])),
                Self::Symbol(position) => values.push(i128::from(symbols[*position])),
                Self::Binary { lhs, rhs, .. } if !ready => {
                    tasks.push((expression, true));
                    tasks.push((rhs, false));
                    tasks.push((lhs, false));
                }
                Self::Binary { op, .. } => {
                    let rhs = values.pop().expect("right affine value evaluated");
                    let lhs = values.pop().expect("left affine value evaluated");
                    let value = match op {
                        AffineBinaryOp::Add => lhs.checked_add(rhs).ok_or_else(|| overflow.clone()),
                        AffineBinaryOp::Mul => lhs.checked_mul(rhs).ok_or_else(|| overflow.clone()),
                        AffineBinaryOp::FloorDiv
                        | AffineBinaryOp::CeilDiv
                        | AffineBinaryOp::Mod => {
                            if rhs == 0 {
                                return Err(AffineError::DivisionByZero);
                            }
                            let quotient = lhs.checked_div(rhs).ok_or(overflow.clone())?;
                            let remainder = lhs.checked_rem(rhs).ok_or(overflow.clone())?;
                            let opposite = remainder != 0 && (remainder < 0) != (rhs < 0);
                            match op {
                                AffineBinaryOp::FloorDiv if opposite => {
                                    quotient.checked_sub(1).ok_or_else(|| overflow.clone())
                                }
                                AffineBinaryOp::CeilDiv if remainder != 0 && !opposite => {
                                    quotient.checked_add(1).ok_or_else(|| overflow.clone())
                                }
                                AffineBinaryOp::Mod if opposite => {
                                    remainder.checked_add(rhs).ok_or_else(|| overflow.clone())
                                }
                                AffineBinaryOp::Mod => Ok(remainder),
                                _ => Ok(quotient),
                            }
                        }
                    }?;
                    values.push(value);
                }
            }
        }
        Ok(values.pop().expect("affine expression evaluated"))
    }
}

impl AffineMap {
    /// Evaluates with checked i128 intermediates and i64 result coordinates.
    ///
    /// Overflow and undefined division return errors; neither wraps nor clamps.
    /// This is map arithmetic, not an interpreter for StableHLO operations.
    pub fn evaluate(&self, dimensions: &[i64], symbols: &[i64]) -> Result<Vec<i64>, AffineError> {
        self.verify()?;
        arguments(self.dimensions, dimensions, false)?;
        arguments(self.symbols, symbols, true)?;
        self.results
            .iter()
            .map(|expression| {
                i64::try_from(expression.evaluate_at(dimensions, symbols)?)
                    .map_err(|_| AffineError::ArithmeticOverflow)
            })
            .collect()
    }
}

impl IntegerSet {
    /// Tests the conjunction using checked i128 expression arithmetic.
    ///
    /// Every constraint is evaluated, even after a false constraint, so an
    /// undefined expression is never hidden by short-circuiting.
    pub fn contains(&self, dimensions: &[i64], symbols: &[i64]) -> Result<bool, AffineError> {
        self.verify()?;
        arguments(self.dimensions, dimensions, false)?;
        arguments(self.symbols, symbols, true)?;
        let mut contains = true;
        for constraint in &self.constraints {
            let value = constraint.expression.evaluate_at(dimensions, symbols)?;
            contains &= if constraint.equality {
                value == 0
            } else {
                value >= 0
            };
        }
        Ok(contains)
    }
}
