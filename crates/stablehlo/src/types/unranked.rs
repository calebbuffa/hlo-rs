use super::{ElementType, QuantizedElementType, Type, TypeError};

/// An MLIR unranked tensor element, with no shape or encoding.
///
/// The pinned dialect permits these tensors at custom-call boundaries, not in
/// operations whose ODS constraints require ranked tensors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnrankedTensorType {
    Plain(ElementType),
    Quantized(QuantizedElementType),
}

impl UnrankedTensorType {
    pub fn verify(&self) -> Result<(), TypeError> {
        match self {
            Self::Plain(_) => Ok(()),
            Self::Quantized(element) => element.verify(),
        }
    }
}

impl From<UnrankedTensorType> for Type {
    fn from(tensor: UnrankedTensorType) -> Self {
        Self::UnrankedTensor(tensor)
    }
}
