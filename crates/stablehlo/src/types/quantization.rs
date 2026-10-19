use super::*;
use crate::attributes::Signed128Attr;
use pliron::{location::Location, verify_err};
/// Native uniform quantized scalar metadata, shared by every tensor rank.
#[pliron_type(
    name = "stablehlo.uniform_quantized",
    format = "`<` $storage `,` $expressed `,` $axis `,` `[` vec($storage_min, CharSpace(`,`)) `]` `,` `[` vec($storage_max, CharSpace(`,`)) `]` `,` `[` vec($scale_bits, CharSpace(`,`)) `]` `,` `[` vec($zero_points, CharSpace(`,`)) `]` `>`",
    generate_get = false
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantizedElementType {
    pub storage: TypeHandle,
    pub expressed: TypeHandle,
    pub axis: i64,
    pub storage_min: Vec<Signed128Attr>,
    pub storage_max: Vec<Signed128Attr>,
    pub scale_bits: Vec<u64>,
    pub zero_points: Vec<Signed128Attr>,
}
/// Parameter data for an interned uniform quantized element type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UniformQuantizationParameters {
    /// Optional quantization axis; `None` represents per-tensor quantization.
    pub axis: i64,
    /// Explicit storage minimum, omitted when it matches the storage type default.
    pub storage_min: Vec<Signed128Attr>,
    /// Explicit storage maximum, omitted when it matches the storage type default.
    pub storage_max: Vec<Signed128Attr>,
    /// Exact bit patterns of the scales.
    pub scale_bits: Vec<u64>,
    /// Quantized zero points.
    pub zero_points: Vec<Signed128Attr>,
}

/// User-facing construction data for a uniform quantized element type.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformQuantizedSpec {
    /// Integer storage element type.
    pub storage_type: ElementType,
    /// Optional storage minimum.
    pub storage_min: Option<i128>,
    /// Optional storage maximum.
    pub storage_max: Option<i128>,
    /// Floating-point expressed element type.
    pub expressed_type: ElementType,
    /// Optional quantization dimension.
    pub quantization_dimension: Option<usize>,
    /// Scale factors, preserving their exact bits when interned.
    pub scales: Vec<f64>,
    /// Quantized zero points.
    pub zero_points: Vec<i128>,
}

impl QuantizedElementType {
    /// Intern a uniform quantized element type from native component types and parameters.
    pub fn get(
        ctx: &Context,
        storage: TypeHandle,
        expressed: TypeHandle,
        parameters: UniformQuantizationParameters,
    ) -> TypedHandle<Self> {
        Self::instantiate(
            Self {
                storage,
                expressed,
                axis: parameters.axis,
                storage_min: parameters.storage_min,
                storage_max: parameters.storage_max,
                scale_bits: parameters.scale_bits,
                zero_points: parameters.zero_points,
            },
            ctx,
        )
    }
    /// Intern a normalized uniform quantized element type.
    pub fn intern(ctx: &Context, spec: UniformQuantizedSpec) -> TypeHandle {
        let UniformQuantizedSpec {
            storage_type,
            storage_min,
            storage_max,
            expressed_type,
            quantization_dimension,
            scales,
            zero_points,
        } = spec;
        let defaults = storage_bounds(storage_type);
        let minimum = storage_min
            .filter(|&v| defaults.is_none_or(|d| v != d.0))
            .into_iter()
            .map(Signed128Attr::from)
            .collect();
        let maximum = storage_max
            .filter(|&v| defaults.is_none_or(|d| v != d.1))
            .into_iter()
            .map(Signed128Attr::from)
            .collect();
        Self::get(
            ctx,
            storage_type.native(ctx),
            expressed_type.native(ctx),
            UniformQuantizationParameters {
                axis: quantization_dimension.map_or(-1, |d| i64::try_from(d).unwrap_or(i64::MAX)),
                storage_min: minimum,
                storage_max: maximum,
                scale_bits: scales.into_iter().map(f64::to_bits).collect(),
                zero_points: zero_points.into_iter().map(Signed128Attr::from).collect(),
            },
        )
        .into()
    }
    pub fn storage_type(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.storage)
    }
    pub fn expressed_type(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.expressed)
    }
    pub fn storage_range(&self) -> (Option<i128>, Option<i128>) {
        (
            self.storage_min.first().copied().map(Signed128Attr::value),
            self.storage_max.first().copied().map(Signed128Attr::value),
        )
    }
    pub fn quantization_dimension(&self) -> Option<usize> {
        usize::try_from(self.axis).ok()
    }
    pub fn scales(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.scale_bits.iter().copied().map(f64::from_bits)
    }
    pub fn zero_points(&self) -> impl ExactSizeIterator<Item = i128> + '_ {
        self.zero_points.iter().copied().map(Signed128Attr::value)
    }
    pub fn zero_points_fit_storage(&self, ctx: &Context) -> bool {
        let Some((minimum, maximum)) = self.storage_type(ctx).and_then(storage_bounds) else {
            return false;
        };
        let minimum = self
            .storage_min
            .first()
            .copied()
            .map(Signed128Attr::value)
            .unwrap_or(minimum);
        let maximum = self
            .storage_max
            .first()
            .copied()
            .map(Signed128Attr::value)
            .unwrap_or(maximum);
        self.zero_points
            .iter()
            .all(|v| (minimum..=maximum).contains(&v.value()))
    }
}
impl Verify for QuantizedElementType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let Some(storage) = self.storage_type(ctx) else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantization type",
                    reason: "invalid quantized storage type"
                }
            );
        };
        let Some((minimum, maximum)) = storage_bounds(storage)
            .filter(|_| storage.bit_width().is_some_and(|w| (1..=32).contains(&w)))
        else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantization type",
                    reason: "quantized storage must have 1..32 representable bits"
                }
            );
        };
        let Some((smallest, largest)) = self.expressed_type(ctx).and_then(scale_range) else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantization type",
                    reason: "invalid quantized expressed type"
                }
            );
        };
        let actual_min = self
            .storage_min
            .first()
            .copied()
            .map(Signed128Attr::value)
            .unwrap_or(minimum);
        let actual_max = self
            .storage_max
            .first()
            .copied()
            .map(Signed128Attr::value)
            .unwrap_or(maximum);
        if self.storage_min.len() > 1
            || self.storage_max.len() > 1
            || actual_min < minimum
            || actual_max > maximum
            || actual_min >= actual_max
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantization type",
                    reason: "invalid quantized storage range"
                }
            );
        }
        if self.axis < -1
            || self.axis > i64::from(i32::MAX)
            || self.scale_bits.is_empty()
            || self.scale_bits.len() != self.zero_points.len()
            || (self.axis == -1 && self.scale_bits.len() != 1)
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantization type",
                    reason: "invalid quantization axis or parameter cardinality"
                }
            );
        }
        // Unordered range comparisons preserve NaN payloads by design.
        if self
            .scales()
            .any(|s| s.is_infinite() || s <= 0.0 || s < smallest || s > largest)
            || self
                .zero_points
                .iter()
                .any(|p| i64::try_from(p.value()).is_err())
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantization type",
                    reason: "quantization scale/zero-point is outside the supported range"
                }
            );
        }
        Ok(())
    }
}
pub(super) fn storage_bounds(storage: ElementType) -> Option<(i128, i128)> {
    match storage {
        ElementType::I1 => Some((-1, 0)),
        ElementType::F4E2M1Fn => Some((-6, 6)),
        ElementType::F8E4M3Fn => Some((-448, 448)),
        ElementType::F8E5M2 => Some((-57344, 57344)),
        _ => storage.integer_bounds(),
    }
}
pub(super) fn scale_range(element: ElementType) -> Option<(f64, f64)> {
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
