use super::{ElementType, TypeError};

/// Uniform quantized element metadata, independent of a tensor's rank or shape.
///
/// The scale bit patterns are retained for exact equality and hashing. Tensor
/// axis/rank and parameter-count relationships are checked when used in a tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QuantizedElementType {
    storage_type: ElementType,
    storage_min: Option<i128>,
    storage_max: Option<i128>,
    expressed_type: ElementType,
    quantization_dimension: Option<usize>,
    scales: Vec<u64>,
    zero_points: Vec<i128>,
}

impl QuantizedElementType {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        storage_type: ElementType,
        storage_min: Option<i128>,
        storage_max: Option<i128>,
        expressed_type: ElementType,
        quantization_dimension: Option<usize>,
        scales: Vec<f64>,
        zero_points: Vec<i128>,
    ) -> Self {
        Self {
            storage_type,
            storage_min,
            storage_max,
            expressed_type,
            quantization_dimension,
            scales: scales.into_iter().map(f64::to_bits).collect(),
            zero_points,
        }
    }

    pub fn storage_type(&self) -> ElementType {
        self.storage_type
    }

    pub fn storage_range(&self) -> (Option<i128>, Option<i128>) {
        (self.storage_min, self.storage_max)
    }

    pub fn expressed_type(&self) -> ElementType {
        self.expressed_type
    }

    pub fn quantization_dimension(&self) -> Option<usize> {
        self.quantization_dimension
    }

    pub fn scales(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.scales.iter().copied().map(f64::from_bits)
    }

    pub fn zero_points(&self) -> &[i128] {
        &self.zero_points
    }

    /// Checks the element constraints in the pinned spec's Quantized Types section.
    pub fn verify(&self) -> Result<(), TypeError> {
        let (type_min, type_max) = self
            .storage_type
            .integer_bounds()
            .ok_or(TypeError::InvalidQuantizedStorageType(self.storage_type))?;
        let (smallest, largest) = scale_range(self.expressed_type).ok_or(
            TypeError::InvalidQuantizedExpressedType(self.expressed_type),
        )?;
        let storage_min = self.storage_min.unwrap_or(type_min);
        let storage_max = self.storage_max.unwrap_or(type_max);
        if storage_min < type_min || storage_max > type_max || storage_min >= storage_max {
            return Err(TypeError::InvalidQuantizedStorageRange);
        }
        if self.scales.is_empty() || self.scales.len() != self.zero_points.len() {
            return Err(TypeError::MismatchedQuantizationParameters {
                scales: self.scales.len(),
                zero_points: self.zero_points.len(),
            });
        }
        if self
            .scales()
            .any(|scale| !scale.is_finite() || scale <= 0.0)
        {
            return Err(TypeError::InvalidQuantizationScale);
        }
        if self
            .scales()
            .any(|scale| scale < smallest || scale > largest)
        {
            return Err(TypeError::QuantizationScaleOutOfRange(self.expressed_type));
        }
        if self
            .zero_points
            .iter()
            .any(|point| *point < storage_min || *point > storage_max)
        {
            return Err(TypeError::QuantizationZeroPointOutOfRange);
        }

        // LLVM APFloat.cpp's pinned semantics and makeSmallest/makeLargest rules.
        // Scales remain f64, as in QuantTypes.cpp: range checks do not round them.
        fn scale_range(element: ElementType) -> Option<(f64, f64)> {
            use ElementType::*;
            let (min_exponent, max_exponent, precision, nan_uses_largest) = match element {
                F4E2M1Fn => (0, 2, 2, false),
                F6E2M3Fn => (0, 2, 4, false),
                F6E3M2Fn => (-2, 4, 3, false),
                F8E3M4 => (-2, 3, 5, false),
                F8E4M3 => (-6, 7, 4, false),
                F8E4M3Fn => (-6, 8, 4, true),
                F8E4M3FnUz => (-7, 7, 4, false),
                F8E4M3B11FnUz => (-10, 4, 4, false),
                F8E5M2 => (-14, 15, 3, false),
                F8E5M2FnUz => (-15, 15, 3, false),
                F8E8M0FnUz => (-127, 127, 1, false),
                F16 => (-14, 15, 11, false),
                Bf16 => (-126, 127, 8, false),
                Tf32 => (-126, 127, 11, false),
                F32 => (-126, 127, 24, false),
                F64 => return Some((f64::from_bits(1), f64::MAX)),
                _ => return None,
            };
            let smallest = 2.0_f64.powi(min_exponent - precision + 1);
            let last_significand = if nan_uses_largest { 2.0 } else { 1.0 };
            let largest =
                (2.0 - last_significand * 2.0_f64.powi(1 - precision)) * 2.0_f64.powi(max_exponent);
            Some((smallest, largest))
        }
        if self.quantization_dimension.is_none() && self.scales.len() != 1 {
            return Err(TypeError::PerTensorQuantizationRequiresOneScale);
        }
        Ok(())
    }
}
