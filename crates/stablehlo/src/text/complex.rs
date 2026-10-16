use super::{TextError, TextErrorKind, cursor::Cursor};
use crate::{Attribute, ElementType};
use rustc_apfloat::{
    Float, FloatConvert, Round,
    ieee::{Double, Single},
};

impl Cursor<'_> {
    pub(super) fn complex_attribute(&mut self, depth: usize) -> Result<Attribute, TextError> {
        self.expect("<")?;
        self.expect(":")?;
        let component = self.element(depth + 1)?;
        let ty = match component {
            ElementType::F32 => ElementType::C64,
            ElementType::F64 => ElementType::C128,
            _ => {
                return Err(self.error(TextErrorKind::Expected(
                    "f32 or f64 complex attribute component",
                )));
            }
        };
        let token = self.number_token()?;
        let real = parse_component(token, component).map_err(|kind| self.error(kind))?;
        self.expect(",")?;
        let token = self.number_token()?;
        let imaginary = parse_component(token, component).map_err(|kind| self.error(kind))?;
        self.expect(">")?;
        Ok(Attribute::Complex {
            ty,
            real,
            imaginary,
        })
    }
}

fn parse_component(token: &str, component: ElementType) -> Result<u64, TextErrorKind> {
    // NumberAttr parses a double before converting to its component semantics.
    let bits = super::floats::parse(token, ElementType::F64)?;
    if component == ElementType::F64 {
        return Ok(bits);
    }
    let value: Single = Double::from_bits(u128::from(bits))
        .convert_r(Round::NearestTiesToEven, &mut false)
        .value;
    Ok(value.to_bits() as u64)
}

fn print_component(bits: u64, component: ElementType) -> Result<String, TextError> {
    if component == ElementType::F32 {
        let value: Double = Single::from_bits(u128::from(bits))
            .convert_r(Round::NearestTiesToEven, &mut false)
            .value;
        let literal = format!("0x{:016X}", value.to_bits());
        if parse_component(&literal, component) != Ok(bits) {
            return Err(TextError {
                offset: 0,
                kind: TextErrorKind::Unsupported("lossless signaling-NaN complex assembly"),
            });
        }
        return Ok(literal);
    }
    super::literals::print_scalar(&crate::ScalarValue::FloatBits(u128::from(bits)), component)
}

pub(super) fn print(ty: ElementType, real: u64, imaginary: u64) -> Result<String, TextError> {
    let (component, name) = match ty {
        ElementType::C64 => (ElementType::F32, "f32"),
        ElementType::C128 => (ElementType::F64, "f64"),
        _ => {
            return Err(TextError {
                offset: 0,
                kind: TextErrorKind::InvalidAttribute(
                    crate::AttributeValidationError::InvalidScalar { ty },
                ),
            });
        }
    };
    Ok(format!(
        "#complex.number<:{name} {}, {}>",
        print_component(real, component)?,
        print_component(imaginary, component)?
    ))
}
