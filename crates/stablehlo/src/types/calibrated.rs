use super::*;
use pliron::{location::Location, verify_err};
#[pliron_type(
    name = "stablehlo.calibrated",
    format = "`<` $expressed `,` $minimum_bits `,` $maximum_bits `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CalibratedQuantizedType {
    pub expressed: TypeHandle,
    pub minimum_bits: u64,
    pub maximum_bits: u64,
}
impl CalibratedQuantizedType {
    pub fn intern(ctx: &Context, expressed: ElementType, minimum: f64, maximum: f64) -> TypeHandle {
        Self::get(
            ctx,
            expressed.native(ctx),
            minimum.to_bits(),
            maximum.to_bits(),
        )
        .into()
    }
    pub fn expressed_type(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.expressed)
    }
    pub fn minimum(&self) -> f64 {
        f64::from_bits(self.minimum_bits)
    }
    pub fn maximum(&self) -> f64 {
        f64::from_bits(self.maximum_bits)
    }
}
impl Verify for CalibratedQuantizedType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.expressed_type(ctx).is_none_or(|e| !e.is_float())
            || self.maximum() <= self.minimum()
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "calibrated type",
                    reason: "invalid calibrated expressed type/range"
                }
            );
        }
        Ok(())
    }
}
