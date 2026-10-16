use super::*;
use crate::{BufferLayout, BufferType};

impl PayloadDecoder<'_, '_> {
    pub(super) fn builtin_type(
        &self,
        cursor: &mut Cursor<'_>,
        depth: usize,
    ) -> Result<String, BytecodeError> {
        Ok(match cursor.varint()? {
            0 => {
                let code = cursor.varint()?;
                let prefix = match code & 3 {
                    0 => "i",
                    1 => "si",
                    2 => "ui",
                    _ => return Err(cursor.error(Kind::InvalidPayload("integer signedness"))),
                };
                format!("{prefix}{}", code >> 2)
            }
            1 => "index".into(),
            2 => format!(
                "({}) -> ({})",
                self.types(cursor, depth)?.join(", "),
                self.types(cursor, depth)?.join(", ")
            ),
            3 => "bf16".into(),
            4 => "f16".into(),
            5 => "f32".into(),
            6 => "f64".into(),
            9 => format!("complex<{}>", self.type_syntax(cursor, depth)?),
            tag @ (10 | 11) => {
                let memory_space = if tag == 11 {
                    Some(self.attr_ref(cursor, depth)?)
                } else {
                    None
                };
                let dimensions = Self::signed_list(cursor)?
                    .into_iter()
                    .map(|size| {
                        if size == i64::MIN {
                            Ok(None)
                        } else if size >= 0 {
                            Ok(Some(size as u64))
                        } else {
                            Err(cursor.error(Kind::InvalidPayload(
                                "nonnegative memref dimension or dynamic sentinel",
                            )))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let element = self.type_ref(cursor, depth)?;
                let Type::Element(element) = element else {
                    return Err(cursor.error(Kind::InvalidPayload("memref element type")));
                };
                let layout = match self.attr_ref(cursor, depth)? {
                    Attribute::StridedLayout { offset, strides } => {
                        BufferLayout::Strided { offset, strides }
                    }
                    Attribute::AffineMap(map) => BufferLayout::Affine(map),
                    _ => {
                        return Err(cursor
                            .error(Kind::InvalidPayload("affine map or strided memref layout")));
                    }
                };
                let mut buffer = BufferType::new(element, dimensions).with_layout(layout);
                if let Some(memory_space) = memory_space {
                    buffer = buffer.with_memory_space(memory_space);
                }
                text::print_type(&Type::Buffer(buffer))
                    .map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))?
            }
            tag @ (13 | 14) => {
                let encoding = if tag == 14 {
                    Some(self.attr_ref(cursor, depth)?)
                } else {
                    None
                };
                let dimensions = Self::signed_list(cursor)?
                    .into_iter()
                    .map(|size| {
                        if size == i64::MIN {
                            Ok("?".into())
                        } else if size >= 0 {
                            Ok(size.to_string())
                        } else {
                            Err(cursor.error(Kind::InvalidPayload(
                                "nonnegative tensor dimension or dynamic sentinel",
                            )))
                        }
                    })
                    .collect::<Result<Vec<String>, _>>()?;
                let shape = if dimensions.is_empty() {
                    String::new()
                } else {
                    format!("{}x", dimensions.join("x"))
                };
                let element = self.type_syntax(cursor, depth)?;
                let encoding = encoding
                    .map(|value| text::print_attribute(&value))
                    .transpose()
                    .map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))?;
                format!(
                    "tensor<{shape}{element}{}>",
                    encoding
                        .map(|value| format!(", {value}"))
                        .unwrap_or_default()
                )
            }
            15 => format!("tuple<{}>", self.types(cursor, depth)?.join(", ")),
            18 => format!("tensor<*x{}>", self.type_syntax(cursor, depth)?),
            21 => "tf32".into(),
            22 => "f8E5M2".into(),
            23 => "f8E4M3".into(),
            24 => "f8E4M3FN".into(),
            25 => "f8E5M2FNUZ".into(),
            26 => "f8E4M3FNUZ".into(),
            27 => "f8E4M3B11FNUZ".into(),
            28 => "f8E3M4".into(),
            29 => "f4E2M1FN".into(),
            30 => "f6E2M3FN".into(),
            31 => "f6E3M2FN".into(),
            32 => "f8E8M0FNU".into(),
            _ => return Err(cursor.error(Kind::Unsupported("builtin type code"))),
        })
    }

    pub(super) fn builtin_attribute(
        &self,
        cursor: &mut Cursor<'_>,
        depth: usize,
    ) -> Result<Attribute, BytecodeError> {
        Ok(match cursor.varint()? {
            0 => Attribute::Array(self.attrs(cursor, depth)?),
            1 => {
                let count = cursor.count()?;
                let mut values = BTreeMap::new();
                for _ in 0..count {
                    let name = String::from_utf8(self.string_attr(cursor, depth)?)
                        .map_err(|_| cursor.error(Kind::InvalidUtf8))?;
                    if values.insert(name, self.attr_ref(cursor, depth)?).is_some() {
                        return Err(cursor.error(Kind::InvalidPayload("unique dictionary keys")));
                    }
                }
                Attribute::Dictionary(values)
            }
            tag @ (2 | 3) => {
                let value = self.string(cursor)?;
                if tag == 3 {
                    Attribute::TypedString {
                        value,
                        ty: Box::new(self.type_ref(cursor, depth)?),
                    }
                } else {
                    match String::from_utf8(value) {
                        Ok(value) => Attribute::String(value),
                        Err(error) => Attribute::Bytes(error.into_bytes()),
                    }
                }
            }
            tag @ (4 | 5) => {
                let mut refs = vec![
                    String::from_utf8(self.string_attr(cursor, depth)?)
                        .map_err(|_| cursor.error(Kind::InvalidUtf8))?,
                ];
                if tag == 5 {
                    for _ in 0..cursor.count()? {
                        match self.attr_ref(cursor, depth)? {
                            Attribute::SymbolRef(mut nested) if nested.len() == 1 => {
                                refs.append(&mut nested)
                            }
                            _ => {
                                return Err(
                                    cursor.error(Kind::InvalidPayload("flat symbol reference"))
                                );
                            }
                        }
                    }
                }
                Attribute::SymbolRef(refs)
            }
            6 => Attribute::Type(Box::new(self.type_ref(cursor, depth)?)),
            7 => Attribute::Unit,
            8 => {
                let element = Self::element(self.type_ref(cursor, depth)?, cursor)?;
                let width = if element == ElementType::Index {
                    64
                } else {
                    element
                        .bit_width()
                        .filter(|_| element.is_integer())
                        .ok_or_else(|| {
                            cursor.error(Kind::InvalidPayload("integer attribute type"))
                        })?
                };
                let bits = Self::bits(cursor, width)?;
                if element == ElementType::I1 {
                    Attribute::Bool(bits != 0)
                } else {
                    let value = if matches!(
                        element,
                        ElementType::Ui2
                            | ElementType::Ui4
                            | ElementType::Ui8
                            | ElementType::Ui16
                            | ElementType::Ui32
                            | ElementType::Ui64
                    ) {
                        i128::from(bits)
                    } else {
                        i128::from((bits << (64 - width)) as i64 >> (64 - width))
                    };
                    Attribute::Integer { ty: element, value }
                }
            }
            9 => {
                let element = Self::element(self.type_ref(cursor, depth)?, cursor)?;
                let width = element
                    .bit_width()
                    .filter(|_| element.is_float())
                    .ok_or_else(|| cursor.error(Kind::InvalidPayload("floating attribute type")))?;
                if width > 64 {
                    return Err(cursor.error(Kind::InvalidPayload("floating attribute type")));
                }
                Attribute::Float {
                    ty: element,
                    bits: Self::bits(cursor, width)?,
                }
            }
            10 => Attribute::Location(Box::new(Location::CallSite {
                callee: Box::new(self.location(cursor, depth)?),
                caller: Box::new(self.location(cursor, depth)?),
            })),
            11 => {
                let file = self.string_attr(cursor, depth)?;
                let line = u32::try_from(cursor.varint()?)
                    .map_err(|_| cursor.error(Kind::InvalidPayload("32-bit source line")))?;
                let column = u32::try_from(cursor.varint()?)
                    .map_err(|_| cursor.error(Kind::InvalidPayload("32-bit source column")))?;
                Attribute::Location(Box::new(Location::FileLineCol { file, line, column }))
            }
            tag @ (12 | 13) => {
                let locations = (0..cursor.count()?)
                    .map(|_| self.location(cursor, depth))
                    .collect::<Result<_, _>>()?;
                let metadata = if tag == 13 {
                    Some(Box::new(self.attr_ref(cursor, depth)?))
                } else {
                    None
                };
                Attribute::Location(Box::new(Location::Fused {
                    locations,
                    metadata,
                }))
            }
            14 => {
                let name = self.string_attr(cursor, depth)?;
                let child = self.location(cursor, depth)?;
                Attribute::Location(Box::new(Location::Name {
                    name,
                    child: if child == Location::Unknown {
                        None
                    } else {
                        Some(Box::new(child))
                    },
                }))
            }
            15 => Attribute::Location(Box::new(Location::Unknown)),
            17 => {
                let element = Self::element(self.type_ref(cursor, depth)?, cursor)?;
                let size = usize::try_from(cursor.varint()?)
                    .map_err(|_| cursor.error(Kind::ResourceLimit))?;
                let length = usize::try_from(cursor.varint()?)
                    .map_err(|_| cursor.error(Kind::ResourceLimit))?;
                let bytes = cursor.bytes(length)?;
                let width = element
                    .bit_width()
                    .filter(|_| {
                        element.is_integer()
                            || matches!(element, ElementType::F32 | ElementType::F64)
                    })
                    .ok_or_else(|| {
                        cursor.error(Kind::InvalidPayload("dense array element type"))
                    })?;
                if size.checked_mul(width.div_ceil(8) as usize) != Some(bytes.len()) {
                    return Err(cursor.error(Kind::InvalidPayload("dense array byte count")));
                }
                let element_syntax = text::print_type(&Type::Element(element))
                    .map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))?;
                let ty = text::parse_type(&format!("tensor<{size}x{element_syntax}>"))
                    .map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))?;
                let value = self.dense(ty, bytes, cursor, false)?;
                let Attribute::DenseElements { values, .. } = value else {
                    unreachable!()
                };
                Attribute::DenseArray { element, values }
            }
            18 => {
                let ty = self.type_ref(cursor, depth)?;
                let length = usize::try_from(cursor.varint()?)
                    .map_err(|_| cursor.error(Kind::ResourceLimit))?;
                let bytes = cursor.bytes(length)?;
                self.dense(ty, bytes, cursor, true)?
            }
            20 => {
                let ty = self.type_ref(cursor, depth)?;
                let indices = self.attr_ref(cursor, depth)?;
                let values = self.attr_ref(cursor, depth)?;
                let Attribute::DenseElements {
                    ty: index_type,
                    values: index_values,
                } = indices
                else {
                    return Err(cursor.error(Kind::InvalidPayload("dense integer sparse indices")));
                };
                let Type::Tensor(index_type) = *index_type else {
                    return Err(cursor.error(Kind::InvalidPayload("rank-two sparse indices")));
                };
                if index_type.rank() != 2 || index_type.element() != ElementType::I64 {
                    return Err(cursor.error(Kind::InvalidPayload("rank-two i64 sparse indices")));
                }
                let rank = index_type.dimensions()[1].ok_or_else(|| {
                    cursor.error(Kind::InvalidPayload("static sparse index rank"))
                })?;
                let rank = usize::try_from(rank).map_err(|_| cursor.error(Kind::ResourceLimit))?;
                let count = usize::try_from(index_type.dimensions()[0].ok_or_else(|| {
                    cursor.error(Kind::InvalidPayload("static sparse entry count"))
                })?)
                .map_err(|_| cursor.error(Kind::ResourceLimit))?;
                let mut indexes = index_values.into_iter().map(|value| match value {
                    crate::ScalarValue::Integer(value) => i64::try_from(value)
                        .map_err(|_| cursor.error(Kind::InvalidPayload("i64 sparse index"))),
                    _ => Err(cursor.error(Kind::InvalidPayload("integer sparse index"))),
                });
                let indices = (0..count)
                    .map(|_| {
                        (0..rank)
                            .map(|_| {
                                indexes.next().ok_or_else(|| {
                                    cursor.error(Kind::InvalidPayload("complete sparse indices"))
                                })?
                            })
                            .collect()
                    })
                    .collect::<Result<Vec<Vec<i64>>, BytecodeError>>()?;
                let Attribute::DenseElements { values, .. } = values else {
                    return Err(cursor.error(Kind::InvalidPayload("dense sparse values")));
                };
                Attribute::SparseElements {
                    ty: Box::new(ty),
                    indices,
                    values,
                }
            }
            22 => {
                let file = self.string_attr(cursor, depth)?;
                let count = cursor.count()?;
                if count > 4 {
                    return Err(cursor.error(Kind::InvalidPayload("file location range length")));
                }
                let values = (0..count)
                    .map(|_| {
                        u32::try_from(cursor.varint()?).map_err(|_| {
                            cursor.error(Kind::InvalidPayload("32-bit source coordinate"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let location = match values.as_slice() {
                    [] => Location::FileLineCol {
                        file,
                        line: 0,
                        column: 0,
                    },
                    [line] => Location::FileLine { file, line: *line },
                    [line, column] => Location::FileLineCol {
                        file,
                        line: *line,
                        column: *column,
                    },
                    [line, column, end_column] => Location::FileRange {
                        file,
                        start_line: *line,
                        start_column: *column,
                        end_line: None,
                        end_column: *end_column,
                    },
                    [line, column, end_line, end_column] => Location::FileRange {
                        file,
                        start_line: *line,
                        start_column: *column,
                        end_line: Some(*end_line),
                        end_column: *end_column,
                    },
                    _ => unreachable!(),
                };
                Attribute::Location(Box::new(location))
            }
            tag @ (23 | 24) => self.affine_attribute(cursor, tag == 24)?,
            25 => {
                let offset = Self::signed(cursor)?;
                let strides = Self::signed_list(cursor)?;
                Attribute::StridedLayout {
                    offset: (offset != i64::MIN).then_some(offset),
                    strides: strides
                        .into_iter()
                        .map(|stride| (stride != i64::MIN).then_some(stride))
                        .collect(),
                }
            }
            _ => return Err(cursor.error(Kind::Unsupported("builtin attribute code"))),
        })
    }

    fn dense(
        &self,
        ty: Type,
        bytes: &[u8],
        cursor: &Cursor<'_>,
        packed_bools: bool,
    ) -> Result<Attribute, BytecodeError> {
        let (dimensions, element) = match &ty {
            Type::Tensor(ty) => (ty.dimensions(), ty.element()),
            Type::QuantizedTensor(ty) => (ty.dimensions(), ty.storage_type()),
            _ => return Err(cursor.error(Kind::InvalidPayload("ranked dense tensor type"))),
        };
        if dimensions.iter().any(Option::is_none) {
            return Err(cursor.error(Kind::InvalidPayload("static dense shape")));
        }
        let count = if dimensions.contains(&Some(0)) {
            0
        } else {
            dimensions
                .iter()
                .try_fold(1_usize, |count, dimension| {
                    let dimension = dimension.and_then(|value| usize::try_from(value).ok())?;
                    count.checked_mul(dimension)
                })
                .ok_or_else(|| cursor.error(Kind::InvalidPayload("static dense shape")))?
        };
        self.charge(
            count.saturating_mul(std::mem::size_of::<crate::ScalarValue>()),
            cursor,
        )?;
        let unpacked;
        let bytes = if packed_bools && element == ElementType::I1 {
            if bytes == [0xff] {
                &[1][..]
            } else if bytes.len() == count.div_ceil(8) && bytes.len() != count {
                unpacked = (0..count)
                    .map(|index| (bytes[index / 8] >> (index % 8)) & 1)
                    .collect::<Vec<_>>();
                &unpacked
            } else {
                bytes
            }
        } else {
            bytes
        };
        self.charge(bytes.len().saturating_mul(2), cursor)?;
        let hex = bytes
            .iter()
            .map(|byte| format!("{byte:02X}"))
            .collect::<String>();
        let syntax =
            text::print_type(&ty).map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))?;
        text::parse_attribute(&format!("dense<\"0x{hex}\"> : {syntax}"))
            .map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))
    }
}
