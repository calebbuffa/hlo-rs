//! Attributes owned by `iree_linalg_ext`.
//!
//! Operation attributes reuse [`pliron_linalg::attributes`] (indexing maps,
//! iterator types, and integer arrays) and [`pliron_mlir_builtin::attributes`]
//! (scalars and affine maps).

use crate::Diagnostic;
use pliron::{
    attribute::{AttrObj, Attribute},
    derive::pliron_attr,
    result::Result,
};

/// Marks an `scf.forall` loop dimension as a parallel partial reduction
/// produced by split-reduction tiling: `#iree_linalg_ext.split_reduction_mapping<N>`.
#[pliron_attr(
    name = "iree_linalg_ext.split_reduction_mapping",
    format = "`<` $dimension `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SplitReductionMappingAttr {
    /// Mapping id, which is also the relative index.
    pub dimension: i64,
}

impl SplitReductionMappingAttr {
    /// Creates the attribute.
    pub const fn new(dimension: i64) -> Self {
        Self { dimension }
    }

    /// Verifies a `mapping` list as upstream's `verifyAttrList`: every entry
    /// is a split-reduction mapping, none repeats, and the dimensions are
    /// exactly `0..len` in some order.
    ///
    /// # Errors
    ///
    /// Returns an error describing the first violation.
    pub fn verify_list(attrs: &[AttrObj]) -> Result<()> {
        let fail = |reason: String| -> pliron::result::Error {
            Diagnostic::InvalidAttribute {
                attribute: "split_reduction_mapping",
                reason,
            }
            .into()
        };
        let mut dims = attrs
            .iter()
            .map(|a| {
                a.downcast_ref::<Self>().map(|m| m.dimension).ok_or_else(|| {
                    fail("expected all the mapping attribute to be of `SplitReductionMappingAttr` type".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        dims.sort_unstable();
        if dims.windows(2).any(|w| w[0] == w[1]) {
            return Err(fail("Illegal to repeat mapping specification".into()));
        }
        for (index, dim) in (0_i64..).zip(&dims) {
            if *dim != index {
                return Err(fail(format!(
                    "missing dimension {index} in mapping attribute list"
                )));
            }
        }
        Ok(())
    }
}

/// Returns whether `attr` is a [`SplitReductionMappingAttr`].
pub fn is_split_reduction_mapping(attr: &dyn Attribute) -> bool {
    attr.downcast_ref::<SplitReductionMappingAttr>().is_some()
}
