use super::{TextError, TextErrorKind, cursor::Cursor};
use crate::SparseTensorDimSlice;

impl Cursor<'_> {
    pub(super) fn sparse_slice(&mut self) -> Result<SparseTensorDimSlice, TextError> {
        self.expect("(")?;
        let offset = self.sparse_slice_value()?;
        self.expect(",")?;
        let size = self.sparse_slice_value()?;
        self.expect(",")?;
        let stride = self.sparse_slice_value()?;
        self.expect(")")?;
        Ok(SparseTensorDimSlice {
            offset,
            size,
            stride,
        })
    }

    fn sparse_slice_value(&mut self) -> Result<Option<u64>, TextError> {
        if self.take("?") {
            return Ok(None);
        }
        u64::try_from(self.i64()?).map(Some).map_err(|_| {
            self.error(TextErrorKind::Expected(
                "nonnegative int64 sparse slice value or '?'",
            ))
        })
    }
}

pub(super) fn print_slice(slice: &SparseTensorDimSlice) -> String {
    let value = |v: Option<u64>| v.map_or_else(|| "?".into(), |v| v.to_string());
    format!(
        "#sparse_tensor<slice({}, {}, {})>",
        value(slice.offset),
        value(slice.size),
        value(slice.stride)
    )
}
