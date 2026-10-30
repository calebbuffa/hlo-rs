use super::*;
use pliron::{builtin::attributes::StringAttr, derive::pliron_attr};
#[pliron_attr(
    name = "stablehlo.enum",
    format = "`<` $dialect `,` $name `,` $value `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Named dialect enum value stored as text.
pub struct EnumAttr {
    /// Dialect that defines the enumeration.
    pub dialect: StringAttr,
    /// Enumeration name.
    pub name: StringAttr,
    /// Selected enumeration spelling.
    pub value: StringAttr,
}
impl Verify for EnumAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.dialect.as_str() == "stablehlo"
            && !enums::is_valid(self.name.as_str(), self.value.as_str())
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "unknown StableHLO enum spelling"
                }
            );
        }
        Ok(())
    }
}
/// Packages precision values in the vector representation used by operation attributes.
pub fn precision_config(values: Vec<Precision>) -> VecAttr {
    VecAttr::new(
        values
            .into_iter()
            .map(|precision| Box::new(precision) as AttrObj)
            .collect(),
    )
}
