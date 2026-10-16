//! Sparse tensor metadata shared by encoding and assembly contracts.

use std::fmt;
mod encoding;
pub use encoding::{
    SparseEncodingError, SparseLevelFormat, SparseLevelType, SparseTensorEncoding,
    infer_sparse_level_to_dimension,
};

/// Per-dimension sparse slice metadata. `None` denotes a dynamic value.
///
/// Offset is nonnegative; size and stride are positive when known. All known
/// values must fit the reference attribute's nonnegative int64 domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SparseTensorDimSlice {
    pub offset: Option<u64>,
    pub size: Option<u64>,
    pub stride: Option<u64>,
}

impl Default for SparseTensorDimSlice {
    fn default() -> Self {
        Self {
            offset: Some(0),
            size: None,
            stride: Some(1),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseSliceError {
    pub field: &'static str,
    pub value: u64,
}

impl SparseTensorDimSlice {
    pub fn verify(&self) -> Result<(), SparseSliceError> {
        for (field, value, positive) in [
            ("offset", self.offset, false),
            ("size", self.size, true),
            ("stride", self.stride, true),
        ] {
            if let Some(value) = value
                && (value > i64::MAX as u64 || (positive && value == 0))
            {
                return Err(SparseSliceError { field, value });
            }
        }
        Ok(())
    }

    pub fn is_completely_dynamic(&self) -> bool {
        self.offset.is_none() && self.size.is_none() && self.stride.is_none()
    }
}

impl fmt::Display for SparseSliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sparse slice {} has invalid int64 value {}",
            self.field, self.value
        )
    }
}

impl std::error::Error for SparseSliceError {}
