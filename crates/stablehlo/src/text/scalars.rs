use crate::{Attribute, ElementType, ScalarValue, Type};

use super::{TextError, TextErrorKind, cursor::Cursor};

impl Cursor<'_> {
    pub(super) fn scalar_attribute(&mut self, depth: usize) -> Result<Attribute, TextError> {
        let token = self.number_token()?;
        let ty = if self.take(":") {
            let Type::Element(element) = self.ty(depth + 1)? else {
                return Err(self.error(TextErrorKind::Expected("integer, index or floating type")));
            };
            element
        } else if !token.trim_start_matches('-').starts_with("0x")
            && token.contains(['.', 'e', 'E'])
        {
            ElementType::F64
        } else {
            ElementType::I64
        };
        if ty.is_float() {
            let bits = super::floats::parse(token, ty).map_err(|kind| self.error(kind))?;
            return Ok(Attribute::Float { ty, bits });
        }
        let value = super::literals::parse_scalar(token, ty).map_err(|kind| self.error(kind))?;
        match value {
            ScalarValue::Integer(value) => Ok(Attribute::Integer { ty, value }),
            ScalarValue::Bool(value) => Ok(Attribute::Bool(value)),
            _ => Err(self.error(TextErrorKind::Expected("integer, index or floating type"))),
        }
    }
}
