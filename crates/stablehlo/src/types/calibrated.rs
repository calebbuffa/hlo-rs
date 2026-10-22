use super::*;
use pliron::{location::Location, verify_err};
#[pliron_type(
    name = "stablehlo.calibrated",
    format = "`<` $expressed `,` $minimum_bits `,` $maximum_bits `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Floating-point value type calibrated to an inclusive numeric range.
pub struct CalibratedQuantizedType {
    /// Floating-point element type of the calibrated values.
    pub expressed: TypeHandle,
    /// Exact IEEE-754 bits of the inclusive lower calibration bound.
    pub minimum_bits: u64,
    /// Exact IEEE-754 bits of the inclusive upper calibration bound.
    pub maximum_bits: u64,
}
impl CalibratedQuantizedType {
    /// Interns a calibrated type, preserving the exact bound bit patterns.
    pub fn intern(ctx: &Context, expressed: ElementType, minimum: f64, maximum: f64) -> TypeHandle {
        Self::get(
            ctx,
            expressed.native(ctx),
            minimum.to_bits(),
            maximum.to_bits(),
        )
        .into()
    }
    /// Returns the primitive expressed element type, when recognized.
    pub fn expressed_type(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.expressed)
    }
    /// Decodes the inclusive lower calibration bound.
    pub fn minimum(&self) -> f64 {
        f64::from_bits(self.minimum_bits)
    }
    /// Decodes the inclusive upper calibration bound.
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
