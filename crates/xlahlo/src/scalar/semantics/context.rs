// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from OpenXLA's HLO evaluator's floating-point context operations.

use super::{Scalar, numerical_result, same_format, unsupported};
use crate::{
    attributes::ReducePrecision, inference::elementwise::unary_result_element, opcode::Opcode,
    scalar::rounding::RoundingMode,
};
use pliron::result::Result;

/// Explicit semantic evaluation context, not an observation of ambient FP state.
///
/// No thread rounding register is read or changed. Callers requiring the pinned
/// evaluator's ambient-mode behavior must supply their established mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationRoundingMode {
    /// Rounds to the nearest representable value, breaking ties to even.
    NearestEven,
    /// Discards the fractional part toward zero.
    TowardZero,
    /// Rounds toward positive infinity.
    TowardPositive,
    /// Rounds toward negative infinity.
    TowardNegative,
}

fn numerical_rounding_mode(mode: EvaluationRoundingMode) -> RoundingMode {
    match mode {
        EvaluationRoundingMode::NearestEven => RoundingMode::NearestEven,
        EvaluationRoundingMode::TowardZero => RoundingMode::TowardZero,
        EvaluationRoundingMode::TowardPositive => RoundingMode::TowardPositive,
        EvaluationRoundingMode::TowardNegative => RoundingMode::TowardNegative,
    }
}

/// Explicit simulated floating environment, not observed ambient thread state.
///
/// No default context is supplied and no FP control register is read or changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationFloatContext {
    /// Simulated rounding mode; never inferred from the host thread.
    pub rounding: EvaluationRoundingMode,
    /// Whether subnormal results are flushed to signed zero.
    pub flush_to_zero: bool,
    /// Whether subnormal inputs are treated as signed zero.
    pub denormals_are_zero: bool,
}

fn numerical_float_context(
    context: EvaluationFloatContext,
) -> super::super::remainder::RemainderContext {
    super::super::remainder::RemainderContext {
        rounding: numerical_rounding_mode(context.rounding),
        flush_to_zero: context.flush_to_zero,
        denormals_are_zero: context.denormals_are_zero,
    }
}

/// Compute real square root in an explicit simulated pinned context.
///
/// Only nearest-even with FTZ and DAZ disabled is supported; other contexts
/// return InvalidShape before arithmetic. This is a bounded API restriction,
/// not an upstream sqrt restriction. Narrow formats use F32 computation and
/// final encoding; F64 uses F64 precision. Finite results use exact integer
/// square roots, with no host math or ambient FP state access.
///
/// NaNs are quieted with sign/payload retained at compute width; negative
/// nonzero operands, including negative infinity, produce negative canonical
/// NaN. Signed zero and positive infinity are preserved before final encoding.
/// Complex sqrt is supported upstream but is unimplemented here. Ordinary
/// [`super::unary`] and HLO evaluator Sqrt dispatch remain unsupported.
pub fn sqrt_with_context(operand: Scalar, context: EvaluationFloatContext) -> Result<Scalar> {
    numerical_result(
        super::super::sqrt::sqrt(
            operand.element,
            operand.bits,
            numerical_float_context(context),
        )?,
        operand.element,
    )
}

/// Compute real truncation-quotient remainder in a simulated pinned context.
///
/// Only nearest-even with FTZ and DAZ disabled is supported; other contexts
/// return InvalidShape. This is our bounded API restriction, not a claim that
/// upstream fmod rejects those environments. Both operands must have the same
/// real format. Narrow formats compute in F32; F64 computes in F64.
/// Exceptional bit policies follow the recorded MSVC/UCRT reference.
/// This is neither ambient FP evaluation nor IEEE nearest-quotient remainder.
/// Ordinary [`super::binary`] does not admit real Remainder; integer behavior is
/// unchanged. No HLO evaluator dispatch is enabled.
pub fn remainder_with_context(
    lhs: Scalar,
    rhs: Scalar,
    context: EvaluationFloatContext,
) -> Result<Scalar> {
    same_format(lhs, rhs)?;
    numerical_result(
        super::super::remainder::remainder(
            lhs.element,
            lhs.bits,
            rhs.bits,
            numerical_float_context(context),
        )?,
        lhs.element,
    )
}

/// Apply one of four real integral-rounding operations in an explicit context.
///
/// RoundNearestEven requires NearestEven; another mode returns InvalidShape.
/// RoundNearestAfz, Ceil and Floor are independent of the supplied mode.
/// Narrow formats compute in F32 and convert back; F64 computes in F64.
/// NaN payload/sign policy follows the recorded pinned host, not a portable
/// ambient-libm contract. This neither observes nor changes thread FP state
/// and does not enable rounding in the HLO evaluator.
pub fn rounding_unary(
    opcode: Opcode,
    operand: Scalar,
    mode: EvaluationRoundingMode,
) -> Result<Scalar> {
    if !matches!(
        opcode,
        Opcode::RoundNearestAfz | Opcode::RoundNearestEven | Opcode::Ceil | Opcode::Floor
    ) {
        return Err(unsupported(opcode, operand.element));
    }
    if !operand.element.is_float() {
        return Err(unsupported(opcode, operand.element));
    }
    let expected = unary_result_element(opcode, operand.element)?;
    let mode = numerical_rounding_mode(mode);
    numerical_result(
        super::super::rounding::unary(opcode, operand.element, operand.bits, mode)?,
        expected,
    )
}

/// Reduce real floating precision using the pinned typed-compute semantics.
///
/// Exponent width must be at least one and mantissa width nonnegative; neither
/// has an upper bound. Narrow formats compute in F32 and convert back, so even
/// oversized widths can canonicalize their NaN encodings. This is not a raw
/// identity shortcut or an evaluator dispatch API.
pub fn reduce_precision(operand: Scalar, precision: &ReducePrecision) -> Result<Scalar> {
    numerical_result(
        super::super::precision::reduce_precision(
            operand.element,
            operand.bits,
            precision.exponent_bits,
            precision.mantissa_bits,
        )?,
        operand.element,
    )
}
