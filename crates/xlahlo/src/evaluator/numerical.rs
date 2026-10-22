//! Literal elementwise evaluation through the shared qualified scalar semantics.

use super::*;

impl Session<'_> {
    pub(super) fn elementwise(
        &self,
        operation: Ptr<Operation>,
        op: Opcode,
        target: TypeHandle,
        values: &[Literal],
    ) -> Result<Literal> {
        if shape::is_dynamic(self.ctx, target)
            || values
                .iter()
                .any(|value| shape::is_dynamic(self.ctx, value.shape))
        {
            return Err(unsupported_value(
                op,
                "dynamic numerical arrays are not implemented",
            ));
        }
        let output_element = shape::element_type(self.ctx, target)?;
        let input_elements = values
            .iter()
            .map(|value| shape::element_type(self.ctx, value.shape))
            .collect::<Result<Vec<_>>>()?;
        if op == Opcode::BitcastConvert
            && input_elements[0].storage_bit_width() != output_element.storage_bit_width()
        {
            return Err(unsupported_value(
                op,
                "tensor bitcast changes element storage width",
            ));
        }
        let probe = input_elements
            .into_iter()
            .map(|element| Scalar::from_bits(element, 0))
            .collect::<Result<Vec<_>>>()?;
        let scalar_result = self.scalar(operation, op, output_element, &probe)?;
        if scalar_result.element() != output_element {
            return Err(malformed(
                "scalar kernel returned an incompatible result format",
            ));
        }
        let dimensions = shape::dimensions(self.ctx, target)?.to_vec();
        let mut inputs = Vec::with_capacity(values.len());
        for value in values {
            let rank = shape::rank(self.ctx, value.shape)?;
            let shape = shape::with_layout(self.ctx, value.shape, Layout::descending(rank))?;
            let literal = literal_semantics::relayout(self.ctx, value, shape)?;
            let element = shape::element_type(self.ctx, literal.shape)?;
            let scalars = literal
                .bits(self.ctx)?
                .into_iter()
                .map(|bits| Scalar::from_bits(element, bits))
                .collect::<Result<Vec<_>>>()?;
            inputs.push(scalars);
        }
        let count = usize::try_from(shape::element_count(self.ctx, target)?)
            .map_err(|_| malformed("result element count exceeds host addressing"))?;
        for (literal, input) in values.iter().zip(&inputs) {
            if !(op == Opcode::Clamp && shape::rank(self.ctx, literal.shape)? == 0)
                && input.len() != count
            {
                return Err(malformed(
                    "elementwise operand element count differs from result",
                ));
            }
        }
        let mut output = Vec::with_capacity(count);
        for index in 0..count {
            let mut args = Vec::with_capacity(inputs.len());
            for (input, literal) in inputs.iter().zip(values) {
                let scalar_broadcast =
                    op == Opcode::Clamp && shape::rank(self.ctx, literal.shape)? == 0;
                let offset = if scalar_broadcast { 0 } else { index };
                args.push(
                    *input
                        .get(offset)
                        .ok_or_else(|| malformed("elementwise operand element count mismatch"))?,
                );
            }
            let result = self.scalar(operation, op, output_element, &args)?;
            if result.element() != output_element {
                return Err(malformed("scalar kernel returned an incompatible format"));
            }
            output.push(result.bits());
        }
        let row_major = shape::array(self.ctx, output_element, dimensions)?;
        let result = Literal::from_bits(self.ctx, row_major, &output)?;
        literal_semantics::relayout(self.ctx, &result, target)
    }

    pub(super) fn scalar(
        &self,
        operation: Ptr<Operation>,
        op: Opcode,
        output_element: crate::types::ElementType,
        args: &[Scalar],
    ) -> Result<Scalar> {
        let result = (|| -> Result<Scalar> {
            Ok(match op {
                Opcode::Abs
                | Opcode::Ceil
                | Opcode::Clz
                | Opcode::Floor
                | Opcode::Imag
                | Opcode::IsFinite
                | Opcode::Negate
                | Opcode::Not
                | Opcode::PopulationCount
                | Opcode::Real
                | Opcode::RoundNearestAfz
                | Opcode::Sign => scalar_semantics::unary(op, args[0])?,
                Opcode::RoundNearestEven => scalar_semantics::rounding_unary(
                    op,
                    args[0],
                    self.float_context
                        .ok_or_else(|| {
                            failure(EvaluationError::MissingFloatContext { opcode: op })
                        })?
                        .rounding,
                )?,
                Opcode::Compare => scalar_semantics::compare(
                    attr::<Direction>(self.ctx, operation, "hlo_direction")?,
                    attr::<ComparisonOrder>(self.ctx, operation, "hlo_order")?,
                    args[0],
                    args[1],
                )?,
                Opcode::Convert => scalar_semantics::convert(args[0], output_element)?,
                Opcode::BitcastConvert => {
                    scalar_semantics::bitcast_convert(args[0], output_element)?
                }
                Opcode::Clamp => scalar_semantics::clamp(args[0], args[1], args[2])?,
                _ => return scalar_semantics::binary(op, args[0], args[1]),
            })
        })();
        result.map_err(|error: pliron::result::Error| {
            if let Some(crate::Diagnostic::Unsupported { message }) =
                error.err.downcast_ref::<crate::Diagnostic>()
            {
                unsupported_value(op, message)
            } else {
                error
            }
        })
    }
}
