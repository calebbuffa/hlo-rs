use super::*;
#[pliron_type(
    name = "stablehlo.future",
    format = "`<` vec($elements, CharSpace(`,`)) `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Deferred collection of ranked tensor values.
pub struct FutureType {
    /// Ranked tensor result types represented by this future.
    pub elements: Vec<TypeHandle>,
}
impl Verify for FutureType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for &element in &self.elements {
            let ty = element.deref(ctx);
            if ty.downcast_ref::<TensorType>().is_none() {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidType {
                        ty: "shaped type",
                        reason: "future element must be a ranked tensor"
                    }
                );
            }
            ty.verify(ctx)?;
        }
        Ok(())
    }
}
macro_rules! unit_type {
    ($name:ident,$mnemonic:literal) => {
        #[doc = concat!("Unit marker type for the `", $mnemonic, "` type.")]
        #[pliron_type(name=$mnemonic,format,generate_get=true,verifier="succ")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name;
    };
}
unit_type!(StringType, "stablehlo.string");
unit_type!(WitnessType, "stablehlo.witness");
