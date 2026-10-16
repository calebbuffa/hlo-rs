use super::{InferenceError, bound, bounds, bounds_encoding};
use crate::{ElementType, TensorType, Type};

/// Fourier transform direction and real/complex domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FftType {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

/// Infers Fourier transform result dimensions, element type and bounds.
///
/// Uses the pinned `inferFftOp` rules, preserving batch bounds and dropping
/// the final-axis bound for real/complex conversion. Zero-length IRFFT follows
/// specification C5 (a zero-length operand), rather than the C++ helper's
/// additional acceptance of a one-element operand for a zero-length result.
pub fn infer_fft_type(
    operand: &Type,
    fft_type: FftType,
    fft_length: &[i64],
) -> Result<Type, InferenceError> {
    operand
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    let Type::Tensor(tensor) = operand else {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    };
    if !(1..=3).contains(&fft_length.len()) || fft_length.len() > tensor.rank() {
        return Err(InferenceError::InvalidFftRank {
            operand_rank: tensor.rank(),
            fft_rank: fft_length.len(),
        });
    }
    let lengths = fft_length
        .iter()
        .enumerate()
        .map(|(dimension, &length)| {
            u64::try_from(length)
                .map_err(|_| InferenceError::InvalidFftLength { dimension, length })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let element = match (fft_type, tensor.element()) {
        (FftType::Rfft, ElementType::F32) => ElementType::C64,
        (FftType::Rfft, ElementType::F64) => ElementType::C128,
        (FftType::Irfft, ElementType::C64) => ElementType::F32,
        (FftType::Irfft, ElementType::C128) => ElementType::F64,
        (FftType::Fft | FftType::Ifft, ElementType::C64 | ElementType::C128) => tensor.element(),
        _ => return Err(InferenceError::IncompatibleTypes { index: 0 }),
    };
    let mut sizes = tensor.dimensions().to_vec();
    let real_conversion = matches!(fft_type, FftType::Rfft | FftType::Irfft);
    if real_conversion {
        let start = sizes.len() - lengths.len();
        for (offset, &length) in lengths.iter().enumerate() {
            let axis = start + offset;
            let last = offset + 1 == lengths.len();
            let expected = if fft_type == FftType::Irfft && last && length != 0 {
                length / 2 + 1
            } else {
                length
            };
            if let Some(actual) = sizes[axis]
                && actual != expected
            {
                return Err(InferenceError::DimensionMismatch {
                    dimension: axis,
                    lhs: actual,
                    rhs: expected,
                });
            }
            if let Some(bound) = bound(operand, axis)
                && expected > bound
            {
                return Err(InferenceError::BoundExceeded {
                    dimension: axis,
                    size: expected,
                    bound,
                });
            }
            if last {
                if fft_type == FftType::Irfft {
                    sizes[axis] = Some(length);
                } else if length != 0 {
                    sizes[axis] = Some(length / 2 + 1);
                }
            }
        }
    }
    let encoding = if bounds(operand).is_some_and(|bounds| !bounds.is_empty()) {
        let mut output_bounds = (0..tensor.rank())
            .map(|axis| bound(operand, axis))
            .collect::<Vec<_>>();
        if real_conversion {
            output_bounds[tensor.rank() - 1] = None;
        }
        bounds_encoding(output_bounds)
    } else {
        tensor.encoding().cloned()
    };
    let mut result = TensorType::new(element, sizes);
    if let Some(encoding) = encoding {
        result = result.with_encoding(encoding);
    }
    let result = Type::Tensor(result);
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    Ok(result)
}
