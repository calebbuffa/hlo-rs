use crate::{AffineBinaryOp, AffineConstraint, AffineExpr, AffineMap, IntegerSet};

use super::*;

impl PayloadDecoder<'_, '_> {
    fn affine_expr(&self, cursor: &mut Cursor<'_>) -> Result<AffineExpr, BytecodeError> {
        let mut work = vec![None];
        let mut operands = Vec::new();
        while let Some(item) = work.pop() {
            if work.len() > 128 {
                return Err(cursor.error(Kind::NestingLimit));
            }
            self.charge(std::mem::size_of::<AffineExpr>(), cursor)?;
            if let Some(operator) = item {
                let rhs = operands
                    .pop()
                    .ok_or_else(|| cursor.error(Kind::InvalidPayload("affine right operand")))?;
                let lhs = operands
                    .pop()
                    .ok_or_else(|| cursor.error(Kind::InvalidPayload("affine left operand")))?;
                operands.push(AffineExpr::binary(operator, lhs, rhs));
                continue;
            }
            match cursor.varint()? {
                tag @ (0 | 1) => {
                    let position = usize::try_from(cursor.varint()?)
                        .map_err(|_| cursor.error(Kind::ResourceLimit))?;
                    operands.push(if tag == 0 {
                        AffineExpr::Dimension(position)
                    } else {
                        AffineExpr::Symbol(position)
                    });
                }
                2 => operands.push(AffineExpr::Constant(Self::signed(cursor)?)),
                tag @ 3..=7 => {
                    work.push(Some(match tag {
                        3 => AffineBinaryOp::Add,
                        4 => AffineBinaryOp::Mul,
                        5 => AffineBinaryOp::Mod,
                        6 => AffineBinaryOp::FloorDiv,
                        _ => AffineBinaryOp::CeilDiv,
                    }));
                    work.extend([None, None]);
                }
                _ => return Err(cursor.error(Kind::InvalidPayload("affine expression code"))),
            }
        }
        if operands.len() != 1 {
            return Err(cursor.error(Kind::InvalidPayload("single affine expression")));
        }
        Ok(operands.pop().unwrap())
    }

    pub(super) fn affine_attribute(
        &self,
        cursor: &mut Cursor<'_>,
        set: bool,
    ) -> Result<Attribute, BytecodeError> {
        if set {
            let dimensions =
                usize::try_from(cursor.varint()?).map_err(|_| cursor.error(Kind::ResourceLimit))?;
            let symbols =
                usize::try_from(cursor.varint()?).map_err(|_| cursor.error(Kind::ResourceLimit))?;
            self.charge(
                dimensions.saturating_add(symbols).saturating_mul(16),
                cursor,
            )?;
            let expressions = (0..cursor.count()?)
                .map(|_| self.affine_expr(cursor))
                .collect::<Result<Vec<_>, _>>()?;
            let constraints = expressions
                .into_iter()
                .map(|expression| {
                    Ok(AffineConstraint {
                        expression,
                        equality: cursor.byte()? != 0,
                    })
                })
                .collect::<Result<_, BytecodeError>>()?;
            return Ok(Attribute::IntegerSet(IntegerSet {
                dimensions,
                symbols,
                constraints,
            }));
        }
        let code = cursor.varint()?;
        let dimensions =
            usize::try_from(code >> 2).map_err(|_| cursor.error(Kind::ResourceLimit))?;
        self.charge(
            dimensions.saturating_mul(std::mem::size_of::<AffineExpr>()),
            cursor,
        )?;
        let (symbols, results) = match code & 3 {
            0 => (0, (0..dimensions).map(AffineExpr::Dimension).collect()),
            tag @ (1 | 2) => {
                let count = if tag == 1 {
                    dimensions
                } else {
                    cursor.count()?
                };
                self.charge(
                    count.saturating_mul(std::mem::size_of::<AffineExpr>()),
                    cursor,
                )?;
                let results = (0..count)
                    .map(|_| {
                        usize::try_from(cursor.varint()?)
                            .map(AffineExpr::Dimension)
                            .map_err(|_| cursor.error(Kind::ResourceLimit))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if tag == 1 {
                    let positions = results.iter().collect::<std::collections::HashSet<_>>();
                    if positions.len() != dimensions {
                        return Err(
                            cursor.error(Kind::InvalidPayload("unique permutation dimensions"))
                        );
                    }
                }
                (0, results)
            }
            _ => {
                let symbols = usize::try_from(cursor.varint()?)
                    .map_err(|_| cursor.error(Kind::ResourceLimit))?;
                self.charge(symbols.saturating_mul(16), cursor)?;
                let results = (0..cursor.count()?)
                    .map(|_| self.affine_expr(cursor))
                    .collect::<Result<_, _>>()?;
                (symbols, results)
            }
        };
        Ok(Attribute::AffineMap(AffineMap {
            dimensions,
            symbols,
            results,
        }))
    }
}
