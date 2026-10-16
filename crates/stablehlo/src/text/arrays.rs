use crate::{Attribute, ElementType, ScalarValue, Type};

use super::{TextError, TextErrorKind, cursor::Cursor};

impl Cursor<'_> {
    pub(super) fn dense_array_attribute(&mut self, depth: usize) -> Result<Attribute, TextError> {
        self.expect("<")?;
        let Type::Element(element) = self.ty(depth + 1)? else {
            return Err(self.error(TextErrorKind::Expected("numeric dense array element type")));
        };
        if !crate::attribute::dense_array_element_type(element) {
            return Err(self.error(TextErrorKind::Expected(
                "i1 or byte-aligned numeric element",
            )));
        }
        let mut values = Vec::new();
        if !self.take(">") {
            self.expect(":")?;
            loop {
                let token = self.number_token()?;
                if element == ElementType::I1 && !matches!(token, "true" | "false") {
                    return Err(self.error(TextErrorKind::Expected("true or false")));
                }
                // The pinned dense-array parser calls buildAttributeAPInt
                // directly, permitting unsigned negative two's-complement bits.
                let value = if element != ElementType::I1 && element.is_integer() {
                    super::literals::integer_literal(token, element, true)
                } else {
                    super::literals::parse_scalar(token, element)
                };
                values.push(value.map_err(|kind| self.error(kind))?);
                if self.take(">") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(Attribute::DenseArray { element, values })
    }
}

pub(super) fn print(
    element: ElementType,
    values: &[ScalarValue],
    depth: usize,
) -> Result<String, TextError> {
    let element_name = super::types::print(&Type::Element(element), depth + 1)?;
    if values.is_empty() {
        return Ok(format!("array<{element_name}>"));
    }
    let values = values
        .iter()
        .map(|value| super::literals::print_scalar(value, element))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    Ok(format!("array<{element_name}: {values}>"))
}
