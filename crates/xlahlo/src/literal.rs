use crate::{Dimension, Error, PrimitiveType, Result, Shape, proto::xla_data::LiteralProto};

/// A validated literal retaining its original wire fields and floating bits.
#[derive(Clone, Debug)]
pub struct Literal {
    shape: Shape,
    wire: LiteralProto,
}

macro_rules! byte_fields {
    ($ty:expr, $wire:expr, $action:ident $(, $arg:expr)?) => {{
        use PrimitiveType::*;
        match $ty {
            S1 => $action!($wire.s1s $(, $arg)?), S2 => $action!($wire.s2s $(, $arg)?),
            S4 => $action!($wire.s4s $(, $arg)?), S8 => $action!($wire.s8s $(, $arg)?),
            S16 => $action!($wire.s16s $(, $arg)?), U1 => $action!($wire.u1s $(, $arg)?),
            U2 => $action!($wire.u2s $(, $arg)?), U4 => $action!($wire.u4s $(, $arg)?),
            U8 => $action!($wire.u8s $(, $arg)?), U16 => $action!($wire.u16s $(, $arg)?),
            F16 => $action!($wire.f16s $(, $arg)?), BF16 => $action!($wire.bf16s $(, $arg)?),
            F4E2M1FN => $action!($wire.f4e2m1fns $(, $arg)?),
            F6E2M3FN => $action!($wire.f6e2m3fns $(, $arg)?),
            F6E3M2FN => $action!($wire.f6e3m2fns $(, $arg)?),
            F8E3M4 => $action!($wire.f8e3m4s $(, $arg)?),
            F8E4M3 => $action!($wire.f8e4m3s $(, $arg)?),
            F8E4M3FN => $action!($wire.f8e4m3fns $(, $arg)?),
            F8E4M3FNUZ => $action!($wire.f8e4m3fnuzs $(, $arg)?),
            F8E4M3B11FNUZ => $action!($wire.f8e4m3b11fnuzs $(, $arg)?),
            F8E5M2 => $action!($wire.f8e5m2s $(, $arg)?),
            F8E5M2FNUZ => $action!($wire.f8e5m2fnuzs $(, $arg)?),
            F8E8M0FNU => $action!($wire.f8e8m0fnus $(, $arg)?),
            _ => return Err(Error::new("literal", "not a byte-backed primitive")),
        }
    }};
}

macro_rules! set_bytes {
    ($field:expr, $value:expr) => {
        $field = $value
    };
}
macro_rules! get_bytes {
    ($field:expr) => {
        &$field
    };
}

impl Literal {
    /// Little-endian physical storage, with one byte per sub-byte element.
    /// Unused bits in sub-byte storage are retained in the wire representation.
    pub fn from_storage_bytes(shape: Shape, bytes: &[u8]) -> Result<Self> {
        let element = shape.element_type();
        let width = element
            .bit_width()
            .ok_or_else(|| Error::new("literal", "storage bytes require array shape"))?;
        let byte_width = width.div_ceil(8) as usize;
        let count = usize::try_from(shape.element_count()?)
            .map_err(|_| Error::new("overflow", "literal element count"))?;
        if count.checked_mul(byte_width) != Some(bytes.len()) {
            return Err(Error::new("literal", "storage byte count mismatch"));
        }
        if element == PrimitiveType::PRED && bytes.iter().any(|&byte| byte > 1) {
            return Err(Error::new(
                "literal",
                "predicate storage must be zero or one",
            ));
        }
        let bits = bytes
            .chunks_exact(byte_width)
            .map(|bytes| {
                let mut word = [0; 16];
                word[..bytes.len()].copy_from_slice(bytes);
                let value = u128::from_le_bytes(word);
                if width == 128 {
                    value
                } else {
                    value & ((1_u128 << width) - 1)
                }
            })
            .collect::<Vec<_>>();
        let mut literal = Self::from_bits(shape, &bits)?;
        if width < 8 && element != PrimitiveType::PRED {
            byte_fields!(element, literal.wire, set_bytes, bytes.to_vec());
        }
        Ok(literal)
    }

    /// Elements are in physical minor-to-major order. One u128 contains the
    /// exact element bit pattern; complex values pack real in the low bits.
    pub fn from_bits(shape: Shape, bits: &[u128]) -> Result<Self> {
        if shape.is_dynamic() {
            return Err(Error::new(
                "unsupported",
                "use from_proto for dynamic literal sizes",
            ));
        }
        let count = usize::try_from(shape.element_count()?)
            .map_err(|_| Error::new("overflow", "literal count"))?;
        if count != bits.len() {
            return Err(Error::new("literal", "literal element count mismatch"));
        }
        let width = shape
            .element_type()
            .bit_width()
            .ok_or_else(|| Error::new("literal", "expected array"))?;
        if width < 128 && bits.iter().any(|&n| n >> width != 0) {
            return Err(Error::new("literal", "element bits exceed primitive width"));
        }
        let shape = if shape.layout().is_none() {
            shape.with_default_layout()?
        } else {
            shape
        };
        let mut wire = LiteralProto {
            shape: Some(shape.as_proto().clone()).into(),
            ..Default::default()
        };
        use PrimitiveType::*;
        match shape.element_type() {
            PRED => wire.preds = bits.iter().map(|&n| n != 0).collect(),
            S32 => wire.s32s = bits.iter().map(|&n| n as i32).collect(),
            U32 => wire.u32s = bits.iter().map(|&n| n as u32).collect(),
            S64 => wire.s64s = bits.iter().map(|&n| n as i64).collect(),
            U64 => wire.u64s = bits.iter().map(|&n| n as u64).collect(),
            F32 => wire.f32s = bits.iter().map(|&n| f32::from_bits(n as u32)).collect(),
            F64 => wire.f64s = bits.iter().map(|&n| f64::from_bits(n as u64)).collect(),
            C64 => {
                wire.c64s = bits
                    .iter()
                    .flat_map(|&n| [f32::from_bits(n as u32), f32::from_bits((n >> 32) as u32)])
                    .collect()
            }
            C128 => {
                wire.c128s = bits
                    .iter()
                    .flat_map(|&n| [f64::from_bits(n as u64), f64::from_bits((n >> 64) as u64)])
                    .collect()
            }
            ty => {
                let bytes: Vec<u8> = bits
                    .iter()
                    .flat_map(|n| n.to_le_bytes().into_iter().take(width.div_ceil(8) as usize))
                    .collect();
                byte_fields!(ty, wire, set_bytes, bytes);
            }
        }
        Self::from_proto(wire)
    }

    pub fn tuple(elements: Vec<Self>) -> Result<Self> {
        let shape = Shape::tuple(elements.iter().map(|v| v.shape.clone()).collect())?;
        Self::from_proto(LiteralProto {
            shape: Some(shape.into_proto()).into(),
            tuple_literals: elements.into_iter().map(Self::into_proto).collect(),
            ..Default::default()
        })
    }

    pub fn token() -> Self {
        let shape = Shape::token();
        Self {
            wire: LiteralProto {
                shape: Some(shape.as_proto().clone()).into(),
                ..Default::default()
            },
            shape,
        }
    }

    pub fn from_proto(wire: LiteralProto) -> Result<Self> {
        let shape = Shape::from_proto(
            wire.shape
                .as_ref()
                .ok_or_else(|| Error::new("literal", "missing literal shape"))?
                .clone(),
        )?;
        Self::validate_piece(&wire, &shape, 0)?;
        Ok(Self { shape, wire })
    }

    fn validate_piece(wire: &LiteralProto, shape: &Shape, depth: usize) -> Result<()> {
        let mut pending = vec![(wire, shape.clone(), depth)];
        while let Some((wire, shape, depth)) = pending.pop() {
            if depth >= 128 {
                return Err(Error::new("resource-limit", "literal nesting limit"));
            }
            if shape.is_tuple() {
                if shape.as_proto().tuple_shapes.len() != wire.tuple_literals.len() {
                    return Err(Error::new("literal", "tuple literal arity mismatch"));
                }
                for (index, child) in wire.tuple_literals.iter().enumerate() {
                    pending.push((child, shape.subshape(&[index])?, depth + 1));
                }
            } else if shape.element_type() != PrimitiveType::TOKEN {
                shape.rank()?;
                if shape.layout().is_none() {
                    return Err(Error::new("literal", "literal requires physical layout"));
                }
                let declared = Shape::from_proto(
                    wire.shape
                        .as_ref()
                        .ok_or_else(|| Error::new("literal", "missing array literal shape"))?
                        .clone(),
                )?;
                if !declared.equal_including_layout(&shape) {
                    return Err(Error::new(
                        "literal",
                        "array literal shape differs from enclosing shape",
                    ));
                }
                if shape.is_dynamic() {
                    if wire.dynamic_sizes.len() != shape.rank()? {
                        return Err(Error::new("literal", "dynamic size cardinality"));
                    }
                    for (axis, &size) in wire.dynamic_sizes.iter().enumerate() {
                        if matches!(shape.dimension(axis)?, Dimension::Bounded(_))
                            && (size < 0 || i64::from(size) > shape.dimensions()[axis])
                        {
                            return Err(Error::new("literal", "dynamic size out of bounds"));
                        }
                    }
                }
                let count = usize::try_from(shape.element_count()?)
                    .map_err(|_| Error::new("overflow", "literal size"))?;
                if Self::bits_from_wire(&shape, wire)?.len() != count {
                    return Err(Error::new("literal", "payload length mismatch"));
                }
            }
        }
        Ok(())
    }

    /// Extract a subtree, materializing its root shape from the enclosing
    /// literal. Native tuple/token payload shapes are otherwise ignored.
    /// The original literal's raw wire fields remain unchanged.
    pub fn sub_literal(&self, index: &[usize]) -> Result<Self> {
        let shape = self.shape.subshape(index)?;
        let mut current = &self.wire;
        for &axis in index {
            current = current
                .tuple_literals
                .get(axis)
                .ok_or_else(|| Error::new("literal", "missing tuple literal element"))?;
        }
        let mut wire = current.clone();
        wire.shape = Some(shape.into_proto()).into();
        Self::from_proto(wire)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn as_proto(&self) -> &LiteralProto {
        &self.wire
    }
    pub fn into_proto(self) -> LiteralProto {
        self.wire
    }

    pub fn bits(&self) -> Result<Vec<u128>> {
        Self::bits_from_wire(&self.shape, &self.wire)
    }

    fn bits_from_wire(shape: &Shape, wire: &LiteralProto) -> Result<Vec<u128>> {
        use PrimitiveType::*;
        Ok(match shape.element_type() {
            PRED => wire.preds.iter().map(|&n| u128::from(n)).collect(),
            S32 => wire.s32s.iter().map(|&n| u128::from(n as u32)).collect(),
            U32 => wire.u32s.iter().map(|&n| u128::from(n)).collect(),
            S64 => wire.s64s.iter().map(|&n| u128::from(n as u64)).collect(),
            U64 => wire.u64s.iter().map(|&n| u128::from(n)).collect(),
            F32 => wire.f32s.iter().map(|n| u128::from(n.to_bits())).collect(),
            F64 => wire.f64s.iter().map(|n| u128::from(n.to_bits())).collect(),
            C64 => {
                if !wire.c64s.len().is_multiple_of(2) {
                    return Err(Error::new("literal", "odd complex component count"));
                }
                wire.c64s
                    .chunks_exact(2)
                    .map(|n| u128::from(n[0].to_bits()) | (u128::from(n[1].to_bits()) << 32))
                    .collect()
            }
            C128 => {
                if !wire.c128s.len().is_multiple_of(2) {
                    return Err(Error::new("literal", "odd complex component count"));
                }
                wire.c128s
                    .chunks_exact(2)
                    .map(|n| u128::from(n[0].to_bits()) | (u128::from(n[1].to_bits()) << 64))
                    .collect()
            }
            ty => {
                let width = ty
                    .bit_width()
                    .ok_or_else(|| Error::new("literal", "expected array literal"))?;
                let bytes = byte_fields!(ty, wire, get_bytes);
                let byte_width = width.div_ceil(8) as usize;
                if !bytes.len().is_multiple_of(byte_width) {
                    return Err(Error::new("literal", "partial element bytes"));
                }
                bytes
                    .chunks_exact(byte_width)
                    .map(|bytes| {
                        let mut word = [0; 16];
                        word[..bytes.len()].copy_from_slice(bytes);
                        u128::from_le_bytes(word) & ((1_u128 << width) - 1)
                    })
                    .collect()
            }
        })
    }
}
