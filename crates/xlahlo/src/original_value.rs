use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Error, PrimitiveType, Result, Shape,
    proto::xla_data::{
        OriginalArrayProto, OriginalValueElementProto, OriginalValueProto, ShapeProto,
    },
};

/// Original-value metadata, retaining its wire representation separately from
/// the last-write-wins tuple-tree semantics used by pinned XLA.
#[derive(Clone, Debug, PartialEq)]
pub struct OriginalValue {
    wire: OriginalValueProto,
    leaves: BTreeMap<Vec<i64>, Option<OriginalArrayProto>>,
}

impl OriginalValue {
    pub fn from_proto(wire: OriginalValueProto) -> Result<Self> {
        let mut leaves: BTreeMap<Vec<i64>, Option<OriginalArrayProto>> = BTreeMap::new();
        if !wire.is_synthetic_call {
            for element in &wire.elements {
                if element.shape_index.iter().any(|&axis| axis < 0)
                    || element
                        .original_array
                        .as_ref()
                        .is_some_and(|array| array.shape_index.iter().any(|&axis| axis < 0))
                {
                    return Err(Error::new(
                        "original-value",
                        "negative original-value shape index",
                    ));
                }
                leaves.retain(|index, _| {
                    !index.starts_with(&element.shape_index)
                        && !element.shape_index.starts_with(index)
                });
                let array = element
                    .original_array
                    .as_ref()
                    .map(|array| OriginalArrayProto {
                        instruction_name: array.instruction_name.clone(),
                        shape_index: array.shape_index.clone(),
                        ..Default::default()
                    });
                leaves.insert(element.shape_index.clone(), array);
            }
        }
        Ok(Self { wire, leaves })
    }

    pub fn as_proto(&self) -> &OriginalValueProto {
        &self.wire
    }

    pub fn into_proto(self) -> OriginalValueProto {
        self.wire
    }

    /// The deprecated boolean controls imported tree construction, not this
    /// predicate: XLA recognizes synthetic calls by a present empty hierarchy.
    pub fn is_synthetic_call(&self) -> bool {
        self.wire.call_hierarchy.as_deref() == Some("")
    }

    pub fn leaves(&self) -> impl Iterator<Item = (&[i64], Option<&OriginalArrayProto>)> {
        self.leaves
            .iter()
            .map(|(index, array)| (index.as_slice(), array.as_ref()))
    }

    /// Export the semantic tuple tree as XLA would, omitting unknown wire fields.
    /// Use `as_proto` or `into_proto` for lossless wire retention.
    pub fn normalized_proto(&self) -> OriginalValueProto {
        OriginalValueProto {
            elements: if self.is_synthetic_call() {
                Vec::new()
            } else {
                self.leaves
                    .iter()
                    .map(|(index, array)| OriginalValueElementProto {
                        shape_index: index.clone(),
                        original_array: array.clone().into(),
                        ..Default::default()
                    })
                    .collect()
            },
            is_synthetic_call: self.is_synthetic_call(),
            call_hierarchy: self.wire.call_hierarchy.clone(),
            ..Default::default()
        }
    }

    /// Semantic equality; `PartialEq` also compares the retained wire fields.
    pub fn equal(&self, other: &Self) -> bool {
        self.wire.call_hierarchy == other.wire.call_hierarchy
            && (self.is_synthetic_call() || self.leaves == other.leaves)
    }

    /// Check leaf indices, not reference names or whether a leaf has a value.
    /// Empty tuples have no leaves; buffers, tokens and opaque shapes do.
    pub fn verify_shape(&self, shape: &Shape) -> Result<()> {
        if self.is_synthetic_call() {
            return Ok(());
        }
        let mut expected = BTreeSet::new();
        let mut pending: Vec<(&ShapeProto, Vec<i64>)> = vec![(shape.as_proto(), Vec::new())];
        while let Some((shape, index)) = pending.pop() {
            if shape.element_type == PrimitiveType::TUPLE.into() {
                for (axis, child) in shape.tuple_shapes.iter().enumerate() {
                    let mut child_index = index.clone();
                    child_index
                        .push(i64::try_from(axis).map_err(|_| {
                            Error::new("original-value", "tuple index exceeds i64")
                        })?);
                    pending.push((child, child_index));
                }
            } else {
                expected.insert(index);
            }
        }
        if expected != self.leaves.keys().cloned().collect() {
            return Err(Error::new(
                "original-value",
                "original-value leaf indices differ from instruction shape",
            ));
        }
        Ok(())
    }
}
