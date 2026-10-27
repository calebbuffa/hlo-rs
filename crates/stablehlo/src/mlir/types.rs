use super::{TextError, TextErrorKind, cursor::Cursor};
use crate::{
    attributes::{AffineMap, StridedLayoutAttr},
    pliron::{builtin::types::Signedness, context::Context, r#type::TypeHandle},
    types::*,
};

/// Parses and verifies one complete MLIR type, returning its interned handle.
///
/// Unsupported syntax fails explicitly. The parser never substitutes an opaque
/// type, discards tensor encodings, or interprets pliron text as MLIR.
/// Leading and trailing whitespace are accepted; trailing entities are not.
///
/// # Errors
///
/// Returns [`TextError`] for unsupported syntax, malformed types, trailing input,
/// excessive nesting, or type verification failures.
pub fn parse_type(ctx: &Context, input: &str) -> Result<TypeHandle, TextError> {
    let mut cursor = Cursor::new(input);
    let ty = cursor.ty(ctx, 0)?;
    cursor.finish()?;
    ty.deref(ctx)
        .verify(ctx)
        .map_err(|error| cursor.error(TextErrorKind::InvalidType(error.to_string())))?;
    Ok(ty)
}

impl Cursor<'_> {
    pub(super) fn ty(&mut self, ctx: &Context, depth: usize) -> Result<TypeHandle, TextError> {
        if depth >= 128 {
            return Err(self.error(TextErrorKind::NestingLimit));
        }
        if self.take("(") {
            let inputs = self.type_list(ctx, depth + 1, ")")?;
            self.expect("->")?;
            let outputs = if self.take("(") {
                self.type_list(ctx, depth + 1, ")")?
            } else {
                vec![self.ty(ctx, depth + 1)?]
            };
            return Ok(FunctionType::intern(ctx, inputs, outputs));
        }
        if self.keyword("tuple") {
            self.expect("<")?;
            let elements = self.type_list(ctx, depth + 1, ">")?;
            return Ok(TupleType::get(ctx, elements).into());
        }
        if self.keyword("complex") {
            self.expect("<")?;
            let component = self.ty(ctx, depth + 1)?;
            self.expect(">")?;
            return Ok(ComplexType::get(ctx, component).into());
        }
        if self.keyword("!stablehlo.token") {
            return Ok(TokenType::get(ctx).into());
        }
        if self.keyword("!stablehlo.future") {
            self.expect("<")?;
            return Ok(FutureType::get(ctx, self.type_list(ctx, depth + 1, ">")?).into());
        }
        if self.keyword("!shape.witness") {
            return Ok(WitnessType::get(ctx).into());
        }
        if self.keyword("!tf_type.string") {
            return Ok(StringType::get(ctx).into());
        }
        if self.keyword("none") {
            return Ok(NoneType::get(ctx).into());
        }
        if self.keyword("!quant.uniform") {
            return self.uniform_quantized(ctx, depth + 1);
        }
        if self.keyword("!quant.calibrated") {
            self.expect("<")?;
            let expressed = self.ty(ctx, depth + 1)?;
            self.expect("<")?;
            let minimum = self.number()?;
            self.expect(":")?;
            let maximum = self.number()?;
            self.expect(">")?;
            self.expect(">")?;
            let minimum = super::floats::parse(&minimum, ElementType::F64)
                .map_err(|kind| self.error(kind))?;
            let maximum = super::floats::parse(&maximum, ElementType::F64)
                .map_err(|kind| self.error(kind))?;
            return Ok(CalibratedQuantizedType::get(ctx, expressed, minimum, maximum).into());
        }
        if self.keyword("!quant.any") {
            self.expect("<")?;
            let (storage, minimum, maximum) = self.quantized_storage(ctx, depth + 1)?;
            let expressed = if self.take(":") {
                let ty = self.ty(ctx, depth + 1)?;
                Some(primitive_element(ctx, ty).ok_or_else(|| {
                    self.error(TextErrorKind::Expected("primitive expressed type"))
                })?)
            } else {
                None
            };
            self.expect(">")?;
            return Ok(AnyQuantizedType::intern(
                ctx, storage, minimum, maximum, expressed,
            ));
        }
        if self.keyword("tensor") {
            self.expect("<")?;
            if self.take("*") {
                self.expect("x")?;
                let element = self.ty(ctx, depth + 1)?;
                self.expect(">")?;
                return Ok(UnrankedTensorType::get(ctx, element).into());
            }
            let mut shape = Vec::new();
            while let Some(dimension) = self.dimension()? {
                shape.push(dimension);
                self.expect("x")?;
            }
            let element = self.ty(ctx, depth + 1)?;
            let tensor = TensorType::get(ctx, shape, element);
            let ty = if self.take(",") {
                let encoding = self.attribute(ctx, depth + 1)?;
                tensor.deref(ctx).with_encoding(ctx, encoding)
            } else {
                tensor.into()
            };
            self.expect(">")?;
            return Ok(ty);
        }
        if self.keyword("memref") {
            self.expect("<")?;
            let unranked = self.take("*");
            let mut dimensions = Vec::new();
            if unranked {
                self.expect("x")?;
            } else {
                while let Some(dimension) = self.dimension()? {
                    dimensions.push(dimension);
                    self.expect("x")?;
                }
            }
            let element = self.ty(ctx, depth + 1)?;
            let mut ty = if unranked {
                UnrankedBufferType::without_memory_space(ctx, element)
            } else {
                BufferType::without_layout(ctx, element, dimensions)
            };
            let mut layout_seen = false;
            let mut space_seen = false;
            while self.take(",") {
                let attribute = self.attribute(ctx, depth + 1)?;
                let is_layout = attribute.downcast_ref::<AffineMap>().is_some()
                    || attribute.downcast_ref::<StridedLayoutAttr>().is_some();
                if is_layout {
                    if unranked || layout_seen || space_seen {
                        return Err(self.error(TextErrorKind::Expected(
                            "one ranked buffer layout before memory space",
                        )));
                    }
                    layout_seen = true;
                    let borrowed = ty.deref(ctx);
                    let buffer = borrowed
                        .downcast_ref::<BufferType>()
                        .ok_or_else(|| self.error(TextErrorKind::Expected("ranked buffer")))?;
                    ty = buffer.with_layout(ctx, attribute);
                } else {
                    if space_seen {
                        return Err(
                            self.error(TextErrorKind::DuplicateField("memory space".into()))
                        );
                    }
                    space_seen = true;
                    let borrowed = ty.deref(ctx);
                    ty = if let Some(buffer) = borrowed.downcast_ref::<BufferType>() {
                        buffer.with_memory_space(ctx, attribute)
                    } else {
                        let buffer =
                            borrowed
                                .downcast_ref::<UnrankedBufferType>()
                                .ok_or_else(|| {
                                    self.error(TextErrorKind::Expected("unranked buffer"))
                                })?;
                        buffer.with_memory_space(ctx, attribute)
                    };
                }
            }
            self.expect(">")?;
            return Ok(ty);
        }
        if self.keyword("vector") {
            self.expect("<")?;
            let mut dimensions = Vec::new();
            let mut scalable = Vec::new();
            loop {
                let bracket = self.take("[");
                let Some(dimension) = self.dimension()? else {
                    if bracket {
                        return Err(self.error(TextErrorKind::Expected("vector dimension")));
                    }
                    break;
                };
                dimensions.push(dimension);
                scalable.push(bracket);
                if bracket {
                    self.expect("]")?;
                }
                self.expect("x")?;
            }
            let element = self.ty(ctx, depth + 1)?;
            self.expect(">")?;
            return Ok(VectorType::get(ctx, element, dimensions, scalable).into());
        }
        let name = self.word()?;
        element_from_name(name)
            .map(|element| element.native(ctx))
            .ok_or_else(|| self.error(TextErrorKind::Unsupported("type spelling")))
    }

    fn type_list(
        &mut self,
        ctx: &Context,
        depth: usize,
        end: &'static str,
    ) -> Result<Vec<TypeHandle>, TextError> {
        let mut types = Vec::new();
        if self.take(end) {
            return Ok(types);
        }
        loop {
            types.push(self.ty(ctx, depth)?);
            if self.take(end) {
                return Ok(types);
            }
            self.expect(",")?;
        }
    }

    fn quantized_storage(
        &mut self,
        ctx: &Context,
        depth: usize,
    ) -> Result<(ElementType, Option<i128>, Option<i128>), TextError> {
        self.space();
        let start = self.offset;
        let name = self.word()?;
        let element =
            if let Some(digits) = name.strip_prefix('u').filter(|name| !name.starts_with('i')) {
                ElementType::integer(
                    digits
                        .parse()
                        .map_err(|_| self.error(TextErrorKind::InvalidNumber))?,
                    Signedness::Unsigned,
                )
                .ok_or_else(|| self.error(TextErrorKind::InvalidNumber))?
            } else {
                self.offset = start;
                let ty = self.ty(ctx, depth)?;
                primitive_element(ctx, ty)
                    .ok_or_else(|| self.error(TextErrorKind::Expected("primitive storage type")))?
            };
        let (minimum, maximum) = if self.take("<") {
            let minimum = self.integer()?;
            self.expect(":")?;
            let maximum = self.integer()?;
            self.expect(">")?;
            (Some(minimum), Some(maximum))
        } else {
            (None, None)
        };
        Ok((element, minimum, maximum))
    }

    fn uniform_quantized(&mut self, ctx: &Context, depth: usize) -> Result<TypeHandle, TextError> {
        self.expect("<")?;
        let (storage_type, storage_min, storage_max) = self.quantized_storage(ctx, depth)?;
        self.expect(":")?;
        let expressed = self.ty(ctx, depth)?;
        let expressed_type = primitive_element(ctx, expressed)
            .ok_or_else(|| self.error(TextErrorKind::Expected("primitive expressed type")))?;
        let quantization_dimension = if self.take(":") {
            Some(
                usize::try_from(self.integer()?)
                    .map_err(|_| self.error(TextErrorKind::InvalidNumber))?,
            )
        } else {
            None
        };
        self.expect(",")?;
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        let per_axis = quantization_dimension.is_some();
        if per_axis {
            self.expect("{")?;
        }
        if !per_axis || !self.take("}") {
            loop {
                let token = self.number()?;
                let bits = super::floats::parse(&token, ElementType::F64)
                    .map_err(|kind| self.error(kind))?;
                scales.push(f64::from_bits(bits));
                zero_points.push(if self.take(":") { self.integer()? } else { 0 });
                if !per_axis || self.take("}") {
                    break;
                }
                self.expect(",")?;
            }
        }
        self.expect(">")?;
        Ok(QuantizedElementType::intern(
            ctx,
            UniformQuantizedSpec {
                storage_type,
                storage_min,
                storage_max,
                expressed_type,
                quantization_dimension,
                scales,
                zero_points,
            },
        ))
    }
}

fn element_from_name(name: &str) -> Option<ElementType> {
    use ElementType::*;
    Some(match name {
        "index" => Index,
        "f4E2M1FN" => F4E2M1Fn,
        "f6E2M3FN" => F6E2M3Fn,
        "f6E3M2FN" => F6E3M2Fn,
        "f16" => F16,
        "bf16" => Bf16,
        "f32" => F32,
        "f64" => F64,
        "tf32" => Tf32,
        "f8E3M4" => F8E3M4,
        "f8E4M3" => F8E4M3,
        "f8E4M3FN" => F8E4M3Fn,
        "f8E4M3FNUZ" => F8E4M3FnUz,
        "f8E4M3B11FNUZ" => F8E4M3B11FnUz,
        "f8E5M2" => F8E5M2,
        "f8E5M2FNUZ" => F8E5M2FnUz,
        "f8E8M0FNU" => F8E8M0FnUz,
        _ => {
            let (digits, signedness) = if let Some(digits) = name.strip_prefix("si") {
                (digits, Signedness::Signed)
            } else if let Some(digits) = name.strip_prefix("ui") {
                (digits, Signedness::Unsigned)
            } else if let Some(digits) = name.strip_prefix('i') {
                (digits, Signedness::Signless)
            } else {
                return None;
            };
            if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            ElementType::integer(digits.parse().ok()?, signedness)?
        }
    })
}
