use super::{TextError, TextErrorKind, cursor::Cursor, floats};
use crate::{
    attributes::{ComplexAttr, ScalarBitsAttr},
    pliron::{attribute::AttrObj, context::Context},
    types::{ElementType, primitive_element},
};

impl Cursor<'_> {
    pub(super) fn complex_attribute(
        &mut self,
        ctx: &Context,
        generic: bool,
        depth: usize,
    ) -> Result<AttrObj, TextError> {
        self.depth(depth)?;
        if generic {
            self.expect("<")?;
            if !self.keyword("number") {
                return Err(self.error(TextErrorKind::Expected("complex number attribute")));
            }
        }
        self.expect("<")?;
        self.expect(":")?;
        let ty = self.ty(ctx, depth + 1)?;
        let component = primitive_element(ctx, ty)
            .filter(|element| element.is_float())
            .ok_or_else(|| {
                self.error(TextErrorKind::Expected(
                    "floating complex attribute component",
                ))
            })?;
        let complex = ElementType::complex(component)
            .ok_or_else(|| self.error(TextErrorKind::Expected("complex attribute type")))?;
        let real = self.complex_component(component)?;
        self.expect(",")?;
        let imaginary = self.complex_component(component)?;
        self.expect(">")?;
        if generic {
            self.expect(">")?;
        }
        Ok(Box::new(ComplexAttr {
            ty: complex.native(ctx),
            real: ScalarBitsAttr {
                ty,
                words: vec![real],
            },
            imaginary: ScalarBitsAttr {
                ty,
                words: vec![imaginary],
            },
        }))
    }

    fn complex_component(&mut self, component: ElementType) -> Result<u64, TextError> {
        let token = self.number()?;
        let double = floats::parse(&token, ElementType::F64).map_err(|kind| self.error(kind))?;
        floats::narrow_f64(double, component).map_err(|kind| self.error(kind))
    }
}
