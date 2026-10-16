use std::collections::HashMap;

use super::{TextError, TextErrorKind, cursor::Cursor};
use crate::{AffineBinaryOp, AffineConstraint, AffineExpr, AffineMap, Attribute, IntegerSet};

impl Cursor<'_> {
    pub(super) fn affine_attribute(
        &mut self,
        is_set: bool,
        depth: usize,
    ) -> Result<Attribute, TextError> {
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
        let mut results = Vec::new();
        let mut constraints = Vec::new();
        if !self.take(")") {
            loop {
                let lhs = self.affine_expression(&names, dimensions, symbols, depth + 1)?;
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
                    let expression = if reverse {
                        subtract(rhs, lhs)
                    } else {
                        subtract(lhs, rhs)
                    };
                    expression
                        .verify(dimensions, symbols)
                        .map_err(|e| self.affine_error(e))?;
                    constraints.push(AffineConstraint {
                        expression,
                        equality,
                    });
                } else {
                    results.push(lhs);
                }
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        self.expect(">")?;
        Ok(if is_set {
            if constraints.is_empty() {
                constraints.push(AffineConstraint {
                    expression: AffineExpr::Constant(0),
                    equality: true,
                });
            }
            Attribute::IntegerSet(IntegerSet {
                dimensions,
                symbols,
                constraints,
            })
        } else {
            Attribute::AffineMap(AffineMap {
                dimensions,
                symbols,
                results,
            })
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
                    AffineExpr::Symbol(count)
                } else {
                    AffineExpr::Dimension(count)
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
        let mut values = Vec::new();
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
                    AffineExpr::Constant(value)
                } else {
                    let name = self.word()?;
                    names.get(name).cloned().ok_or_else(|| {
                        self.error(TextErrorKind::Expected("declared affine identifier"))
                    })?
                };
                values.push(value);
                expecting_operand = false;
                reduce_affine_negations(&mut values, &mut operators, dimensions, symbols, self)?;
                continue;
            }

            if self.input[self.offset..].starts_with(")") {
                if operators
                    .iter()
                    .any(|operator| matches!(operator, AffineParseOperator::Parenthesis))
                {
                    self.offset += 1;
                    reduce_affine_parenthesis(
                        &mut values,
                        &mut operators,
                        dimensions,
                        symbols,
                        self,
                    )?;
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
                reduce_affine_binary(&mut values, &mut operators, dimensions, symbols, self)?;
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
                    reduce_affine_negations(&mut values, &mut operators, dimensions, symbols, self)?
                }
                Some(AffineParseOperator::Binary { .. }) => {
                    reduce_affine_binary(&mut values, &mut operators, dimensions, symbols, self)?
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
        let result = values.pop().expect("one affine expression value");
        result
            .verify(dimensions, symbols)
            .map_err(|error| self.affine_error(error))?;
        Ok(result)
    }

    fn affine_error(&self, error: crate::AffineError) -> TextError {
        self.error(TextErrorKind::InvalidAttribute(
            crate::AttributeValidationError::Affine(error),
        ))
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
        value
            .verify(dimensions, symbols)
            .map_err(|error| cursor.affine_error(error))?;
        values.push(value);
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
    let lhs = values
        .pop()
        .ok_or_else(|| cursor.error(TextErrorKind::Expected("affine expression operand")))?;
    let value = if subtraction {
        subtract(lhs, rhs)
    } else {
        AffineExpr::binary(op, lhs, rhs)
    };
    value
        .verify(dimensions, symbols)
        .map_err(|error| cursor.affine_error(error))?;
    values.push(value);
    Ok(())
}

fn negate(expression: AffineExpr) -> AffineExpr {
    match expression {
        AffineExpr::Constant(value) => AffineExpr::Constant(value.wrapping_neg()),
        expression => AffineExpr::binary(AffineBinaryOp::Mul, expression, AffineExpr::Constant(-1)),
    }
}

fn subtract(lhs: AffineExpr, rhs: AffineExpr) -> AffineExpr {
    if rhs == AffineExpr::Constant(0) {
        lhs
    } else {
        AffineExpr::binary(AffineBinaryOp::Add, lhs, negate(rhs))
    }
}

pub(super) fn expression(expression: &AffineExpr, depth: usize) -> Result<String, TextError> {
    enum Event<'a> {
        Expression(&'a AffineExpr, usize),
        Text(&'static str),
    }
    let cursor = Cursor::new("");
    let mut events = vec![Event::Expression(expression, depth)];
    let mut output = String::new();
    while let Some(event) = events.pop() {
        match event {
            Event::Text(text) => output.push_str(text),
            Event::Expression(expression, depth) => {
                cursor.depth(depth)?;
                match expression {
                    AffineExpr::Dimension(position) => {
                        output.push_str(&format!("d{position}"));
                    }
                    AffineExpr::Symbol(position) => {
                        output.push_str(&format!("s{position}"));
                    }
                    AffineExpr::Constant(value) => {
                        output.push_str(&value.to_string());
                    }
                    AffineExpr::Binary { op, lhs, rhs } => {
                        let operator = match op {
                            AffineBinaryOp::Add => " + ",
                            AffineBinaryOp::Mul => " * ",
                            AffineBinaryOp::FloorDiv => " floordiv ",
                            AffineBinaryOp::CeilDiv => " ceildiv ",
                            AffineBinaryOp::Mod => " mod ",
                        };
                        events.push(Event::Text(")"));
                        events.push(Event::Expression(rhs, depth + 2));
                        events.push(Event::Text(operator));
                        events.push(Event::Expression(lhs, depth + 1));
                        events.push(Event::Text("("));
                    }
                }
            }
        }
    }
    Ok(output)
}

fn domain(dimensions: usize, symbols: usize) -> Result<String, TextError> {
    if dimensions.saturating_add(symbols) > 1_048_576 {
        return Err(Cursor::new("").error(TextErrorKind::ResourceLimit));
    }
    let dimensions = (0..dimensions)
        .map(|i| format!("d{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let symbols = if symbols == 0 {
        String::new()
    } else {
        format!(
            "[{}]",
            (0..symbols)
                .map(|i| format!("s{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    Ok(format!("({dimensions}){symbols}"))
}

pub(super) fn print_map(map: &AffineMap, depth: usize) -> Result<String, TextError> {
    Ok(format!(
        "affine_map<{} -> ({})>",
        domain(map.dimensions, map.symbols)?,
        map.results
            .iter()
            .map(|e| expression(e, depth + 1))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ")
    ))
}

pub(super) fn print_set(set: &IntegerSet, depth: usize) -> Result<String, TextError> {
    Ok(format!(
        "affine_set<{} : ({})>",
        domain(set.dimensions, set.symbols)?,
        set.constraints
            .iter()
            .map(|c| Ok(format!(
                "{} {} 0",
                expression(&c.expression, depth + 1)?,
                if c.equality { "==" } else { ">=" }
            )))
            .collect::<Result<Vec<_>, TextError>>()?
            .join(", ")
    ))
}
