use super::*;
use pliron::{builtin::attributes::StringAttr, derive::pliron_attr};
use std::collections::HashSet;

/// Native operation storage for dialect-qualified discardable attributes.
pub const DISCARDABLE_ATTRIBUTES: pliron::identifier::Identifier =
    pliron::identifier::Identifier::new(crate::attributes::keys::DISCARDABLE_ATTRIBUTES);

/// String-keyed source attributes, whose names need not be pliron identifiers.
///
/// MLIR dictionary names and dialect-qualified discardable operation attributes
/// cannot be stored directly in pliron's identifier-keyed `AttributeDict`.
#[pliron_attr(
    name = "stablehlo.named_attributes",
    format = "`<` `[` vec($names, CharSpace(`,`)) `]` `,` $values `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamedAttributesAttr {
    /// Attribute names, stored independently of pliron's identifier restrictions.
    pub names: Vec<StringAttr>,
    /// Values corresponding to `names` by position.
    pub values: VecAttr,
}

impl NamedAttributesAttr {
    /// Builds a container from name/value pairs without normalizing the names.
    pub fn new(entries: impl IntoIterator<Item = (String, AttrObj)>) -> Self {
        let (names, values): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .map(|(name, value)| (StringAttr::from(name), value))
            .unzip();
        Self {
            names,
            values: VecAttr::new(values),
        }
    }

    /// Iterates paired names and values in insertion order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &AttrObj)> {
        self.names
            .iter()
            .map(StringAttr::as_str)
            .zip(&self.values.0)
    }

    /// Checks that names and values are paired and that names are nonempty and unique.
    pub fn validate_names(&self) -> Result<()> {
        let mut names = HashSet::new();
        if self.names.len() != self.values.0.len()
            || self
                .names
                .iter()
                .any(|name| name.as_str().is_empty() || !names.insert(name.as_str()))
        {
            return verify_err!(
                Location::Unknown,
                Diagnostic::InvalidAttribute {
                    attribute: "named attributes",
                    reason: "names must be nonempty, unique, and paired with values"
                }
            );
        }
        Ok(())
    }
}

impl Verify for NamedAttributesAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.validate_names()?;
        for value in &self.values.0 {
            verify_attribute(ctx, value)?;
        }
        Ok(())
    }
}
