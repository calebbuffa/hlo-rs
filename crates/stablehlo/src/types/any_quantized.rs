use super::*;
use crate::attributes::Signed128Attr;
use pliron::{location::Location, verify_err};
#[pliron_type(
    name = "stablehlo.any_quantized",
    format = "`<` $storage `,` `[` vec($expressed, CharSpace(`,`)) `]` `,` `[` vec($storage_min, CharSpace(`,`)) `]` `,` `[` vec($storage_max, CharSpace(`,`)) `]` `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AnyQuantizedType {
    pub storage: TypeHandle,
    pub expressed: Vec<TypeHandle>,
    pub storage_min: Vec<Signed128Attr>,
    pub storage_max: Vec<Signed128Attr>,
}
impl AnyQuantizedType {
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
    pub fn storage_type(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.storage)
    }
    pub fn storage_range(&self) -> (Option<i128>, Option<i128>) {
        (
            self.storage_min.first().copied().map(Signed128Attr::value),
            self.storage_max.first().copied().map(Signed128Attr::value),
        )
    }
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
