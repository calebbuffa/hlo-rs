//! Canonical builtin representations of StableHLO's ODS attribute aliases.

use super::stablehlo::{ArrayOfLayout, ArrayOfTiling, IndexTensor, Layout, Precision, Tiling};
use crate::{Attribute, ElementType, ScalarValue, TensorType, Type};

impl Attribute {
    pub fn precision_config(values: Vec<Precision>) -> Self {
        Self::Array(values.into_iter().map(Self::from).collect())
    }
}

impl From<Precision> for Attribute {
    fn from(value: Precision) -> Self {
        Self::Enum {
            dialect: "stablehlo".into(),
            name: "precision".into(),
            value: match value {
                Precision::Default => "DEFAULT",
                Precision::High => "HIGH",
                Precision::Highest => "HIGHEST",
            }
            .into(),
        }
    }
}

impl From<IndexTensor> for Attribute {
    fn from(value: IndexTensor) -> Self {
        Self::DenseElements {
            ty: Box::new(
                TensorType::static_shape(ElementType::Index, [value.0.len() as u64]).into(),
            ),
            values: value
                .0
                .into_iter()
                .map(|value| ScalarValue::Integer(i128::from(value)))
                .collect(),
        }
    }
}

impl From<Layout> for Attribute {
    fn from(value: Layout) -> Self {
        IndexTensor(value.0).into()
    }
}

impl From<ArrayOfLayout> for Attribute {
    fn from(value: ArrayOfLayout) -> Self {
        Self::Array(value.0.into_iter().map(Self::from).collect())
    }
}

impl From<Tiling> for Attribute {
    fn from(value: Tiling) -> Self {
        Self::Array(value.0.into_iter().map(Self::from).collect())
    }
}

impl From<ArrayOfTiling> for Attribute {
    fn from(value: ArrayOfTiling) -> Self {
        Self::Array(value.0.into_iter().map(Self::from).collect())
    }
}

pub(crate) fn is_index_tensor(attribute: &Attribute) -> bool {
    matches!(attribute, Attribute::DenseElements { ty, .. }
        if matches!(ty.as_ref(), Type::Tensor(tensor)
            if tensor.element() == ElementType::Index && tensor.rank() == 1))
}

pub(crate) fn index_tensor_values(attribute: &Attribute) -> Option<Vec<i64>> {
    if !is_index_tensor(attribute) {
        return None;
    }
    let Attribute::DenseElements { values, .. } = attribute else {
        unreachable!()
    };
    values
        .iter()
        .map(|value| match value {
            ScalarValue::Integer(value) => i64::try_from(*value).ok(),
            _ => None,
        })
        .collect()
}
