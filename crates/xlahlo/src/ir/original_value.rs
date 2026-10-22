//! Instruction provenance as a normalized last-write-wins tuple tree.
use crate::Diagnostic;

use crate::shape;
use pliron::{
    common_traits::Verify, context::Context, derive::pliron_attr, result::Result,
    r#type::TypeHandle,
};
use std::collections::{BTreeMap, BTreeSet};

#[pliron_attr(
    name = "xlahlo.original_array",
    format = "`<` $instruction_name `,` `[` vec($shape_index, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Source instruction and tuple index identifying an original array.
pub struct OriginalArray {
    /// Name of the source instruction associated with the array.
    pub instruction_name: String,
    /// Nonnegative tuple path to this array in the source instruction's shape.
    pub shape_index: Vec<i64>,
}
#[pliron_attr(
    name = "xlahlo.original_value_element",
    format = "`<` `[` vec($shape_index, CharSpace(`,`)) `]` `,` `[` vec($original_array, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// A tuple-tree path and its optional original-array provenance.
pub struct OriginalValueElement {
    /// Tuple path at which this provenance entry is applied.
    pub shape_index: Vec<i64>,
    /// Zero or one source-array records; later overlapping entries override
    /// earlier entries when the value is normalized.
    pub original_array: Vec<OriginalArray>,
}
#[pliron_attr(
    name = "xlahlo.original_value",
    format = "`<` `[` vec($elements, CharSpace(`,`)) `]` `,` $is_synthetic_call `,` `[` vec($call_hierarchy, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Instruction provenance with last-write-wins tree entries and optional call hierarchy.
pub struct OriginalValue {
    /// Ordered tuple-tree writes; overlapping entries use last-write-wins
    /// semantics when interpreted through [`Self::leaves`].
    pub elements: Vec<OriginalValueElement>,
    /// Whether this value is marked as a synthetic call result.
    pub is_synthetic_call: bool,
    /// Optional single call-hierarchy marker; an empty string represents a
    /// present hierarchy for a synthetic call.
    pub call_hierarchy: Vec<String>,
}
impl Verify for OriginalValue {
    fn verify(&self, _: &Context) -> Result<()> {
        self.leaf_refs().map(|_| ())
    }
}
impl OriginalValue {
    /// Whether a present empty call hierarchy marks a synthetic call.
    pub fn is_synthetic_call(&self) -> bool {
        self.call_hierarchy.first().is_some_and(String::is_empty)
    }
    /// Materialize normalized leaf provenance, removing overwritten ancestors and descendants.
    pub fn leaves(&self) -> Result<BTreeMap<Vec<i64>, Option<OriginalArray>>> {
        Ok(self
            .leaf_refs()?
            .into_iter()
            .map(|(index, original)| (index.to_vec(), original.cloned()))
            .collect())
    }
    fn leaf_refs(&self) -> Result<BTreeMap<&[i64], Option<&OriginalArray>>> {
        if self.call_hierarchy.len() > 1 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "original_value",
                message: "multiple call hierarchies".into(),
            }));
        }
        let mut leaves = BTreeMap::<&[i64], Option<&OriginalArray>>::new();
        if !self.is_synthetic_call {
            for element in &self.elements {
                if element.original_array.len() > 1
                    || element.shape_index.iter().any(|&axis| axis < 0)
                    || element
                        .original_array
                        .iter()
                        .any(|array| array.shape_index.iter().any(|&axis| axis < 0))
                {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "original_value",
                        message: "invalid original-value shape index or array cardinality".into(),
                    }));
                }
                leaves.retain(|index, _| {
                    !index.starts_with(&element.shape_index)
                        && !element.shape_index.starts_with(index)
                });
                leaves.insert(&element.shape_index, element.original_array.first());
            }
        }
        Ok(leaves)
    }
    /// Compare call hierarchy and normalized provenance without copying array records.
    ///
    /// Both records are validated even when their hierarchies differ or identify
    /// synthetic calls, so malformed raw attributes never compare successfully.
    pub fn equal(&self, other: &Self) -> Result<bool> {
        let leaves = self.leaf_refs()?;
        let other_leaves = other.leaf_refs()?;
        Ok(self.call_hierarchy == other.call_hierarchy
            && (self.is_synthetic_call() || leaves == other_leaves))
    }
    /// Construct a canonical owned record containing only effective tree leaves.
    pub fn normalized(&self) -> Result<Self> {
        let leaves = self.leaves()?;
        Ok(Self {
            elements: if self.is_synthetic_call() {
                Vec::new()
            } else {
                leaves
                    .into_iter()
                    .map(|(shape_index, original_array)| OriginalValueElement {
                        shape_index,
                        original_array: original_array.into_iter().collect(),
                    })
                    .collect()
            },
            is_synthetic_call: self.is_synthetic_call(),
            call_hierarchy: self.call_hierarchy.clone(),
        })
    }
    /// Require one provenance entry for every non-tuple leaf of a result shape.
    pub fn verify_shape(&self, ctx: &Context, shape: TypeHandle) -> Result<()> {
        let leaves = self.leaf_refs()?;
        if self.is_synthetic_call() {
            return Ok(());
        }
        let mut expected = BTreeSet::new();
        let mut pending = vec![(shape, Vec::new())];
        while let Some((shape, index)) = pending.pop() {
            if shape::is_tuple(ctx, shape) {
                for (axis, &child) in shape::children(ctx, shape).iter().enumerate() {
                    let mut child_index = index.clone();
                    child_index.push(axis as i64);
                    pending.push((child, child_index));
                }
            } else {
                expected.insert(index);
            }
        }
        if expected.len() != leaves.len()
            || expected
                .iter()
                .any(|index| !leaves.contains_key(index.as_slice()))
        {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "original_value",
                message: "original-value leaf indices differ from shape".into(),
            }));
        }
        Ok(())
    }
}
