use super::*;
use pliron::{location::Location, verify_err};
#[pliron_type(
    name = "stablehlo.quantile",
    format = "`<` $storage `,` $quantile `,` `[` vec($quantile_bits, CharSpace(`,`)) `]` `,` `[` vec($range, CharSpace(`,`)) `]` `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantileType {
    pub storage: TypeHandle,
    pub quantile: TypeHandle,
    pub quantile_bits: Vec<u64>,
    pub range: Vec<i64>,
}
impl QuantileType {
    pub fn intern(
        ctx: &Context,
        storage: ElementType,
        quantile: ElementType,
        quantiles: Vec<f64>,
        range: Option<(i64, i64)>,
    ) -> TypeHandle {
        Self::get(
            ctx,
            storage.native(ctx),
            quantile.native(ctx),
            quantiles.into_iter().map(f64::to_bits).collect(),
            range.map_or_else(Vec::new, |(a, b)| vec![a, b]),
        )
        .into()
    }
    pub fn storage_type(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.storage)
    }
    pub fn quantile_type(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.quantile)
    }
    pub fn quantiles(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.quantile_bits.iter().copied().map(f64::from_bits)
    }
    pub fn storage_range(&self) -> Option<(i64, i64)> {
        (self.range.len() == 2).then(|| (self.range[0], self.range[1]))
    }
}
impl Verify for QuantileType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let storage = self.storage_type(ctx);
        let quantile = self.quantile_type(ctx);
        if storage.is_none_or(|e| !e.is_integer() && !e.is_float())
            || quantile.is_none_or(|e| !e.is_integer() && !e.is_float())
            || self.quantile_bits.is_empty()
            || self.quantiles().any(|v| !v.is_finite())
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantile type",
                    reason: "invalid quantile type or lookup table"
                }
            );
        }
        let expected = match self.range.as_slice() {
            [minimum, maximum] if minimum < maximum => {
                i128::from(*maximum) - i128::from(*minimum) + 1
            }
            [] => {
                let width = storage.and_then(ElementType::bit_width).unwrap_or(128);
                let Some(count) = 1_i128.checked_shl(width) else {
                    return verify_err!(
                        Location::Unknown,
                        crate::Diagnostic::InvalidType {
                            ty: "quantile type",
                            reason: "quantile table cardinality exceeds i128"
                        }
                    );
                };
                count
            }
            _ => {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidType {
                        ty: "quantile type",
                        reason: "invalid quantile storage range"
                    }
                );
            }
        };
        if self.quantile_bits.len() as i128 != expected {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "quantile type",
                    reason: "quantile lookup-table cardinality mismatch"
                }
            );
        }
        Ok(())
    }
}
