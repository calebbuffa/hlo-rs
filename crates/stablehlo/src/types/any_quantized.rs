use super::*;
use crate::attributes::Signed128Attr;
use pliron::{location::Location, verify_err};
#[pliron_type(
    name = "stablehlo.any_quantized",
    format = "`<` $storage `,` `[` vec($expressed, CharSpace(`,`)) `]` `,` `[` vec($storage_min, CharSpace(`,`)) `]` `,` `[` vec($storage_max, CharSpace(`,`)) `]` `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Quantized element metadata with optional expressed type and storage bounds.
pub struct AnyQuantizedType {
    /// Primitive storage type for quantized values.
    pub storage: TypeHandle,
    /// Optional expressed floating-point type, represented by zero or one handle.
    pub expressed: Vec<TypeHandle>,
    /// Optional lower storage bound, omitted when equal to the storage default.
    pub storage_min: Vec<Signed128Attr>,
    /// Optional upper storage bound, omitted when equal to the storage default.
    pub storage_max: Vec<Signed128Attr>,
}
impl AnyQuantizedType {
    /// Interns an unspecified quantized type, omitting explicit bounds equal to defaults.
    pub fn intern(
        ctx: &Context,
        storage: ElementType,
        minimum: Option<i128>,
        maximum: Option<i128>,
        expressed: Option<ElementType>,
    ) -> TypeHandle {
        let defaults = quantization::storage_bounds(storage);
        Self::get(
            ctx,
            storage.native(ctx),
            expressed.into_iter().map(|e| e.native(ctx)).collect(),
            minimum
                .filter(|&v| defaults.is_none_or(|d| v != d.0))
                .into_iter()
                .map(Signed128Attr::from)
                .collect(),
            maximum
                .filter(|&v| defaults.is_none_or(|d| v != d.1))
                .into_iter()
                .map(Signed128Attr::from)
                .collect(),
        )
        .into()
    }
    /// Returns the primitive storage element type, when recognized.
    pub fn storage_type(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.storage)
    }
    /// Returns explicit storage bounds; `None` components use the storage type defaults.
    pub fn storage_range(&self) -> (Option<i128>, Option<i128>) {
        (
            self.storage_min.first().copied().map(Signed128Attr::value),
            self.storage_max.first().copied().map(Signed128Attr::value),
        )
    }
    /// Returns the optional primitive expressed element type.
    pub fn expressed_type(&self, ctx: &Context) -> Option<ElementType> {
        self.expressed
            .first()
            .and_then(|&t| primitive_element(ctx, t))
    }
}
impl Verify for AnyQuantizedType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let Some((minimum, maximum)) = self
            .storage_type(ctx)
            .filter(|e| e.bit_width().is_some_and(|w| (1..=32).contains(&w)))
            .and_then(quantization::storage_bounds)
        else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantized type",
                    reason: "invalid unspecified quantized storage type"
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
            || self.expressed.len() > 1
            || actual_min < minimum
            || actual_max > maximum
            || actual_min >= actual_max
            || (!self.expressed.is_empty()
                && self.expressed_type(ctx).is_none_or(|e| !e.is_float()))
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantized type",
                    reason: "invalid unspecified quantized metadata"
                }
            );
        }
        Ok(())
    }
}
