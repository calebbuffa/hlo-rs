use super::{TextError, cursor::Cursor};
use crate::{Attribute, BufferLayout, BufferType, Type};

impl Cursor<'_> {
    pub(super) fn buffer_type(&mut self, depth: usize) -> Result<Type, TextError> {
        self.expect("<")?;
        let dimensions = self.dimensions()?;
        let mut buffer = BufferType::new(self.element(depth + 1)?, dimensions);
        if self.take(",") {
            if self.take("strided") {
                self.expect("<")?;
                self.expect("[")?;
                let mut strides = Vec::new();
                if !self.take("]") {
                    loop {
                        strides.push(if self.take("?") {
                            None
                        } else {
                            Some(self.i64()?)
                        });
                        if self.take("]") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                let offset = if self.take(",") {
                    self.expect("offset")?;
                    self.expect(":")?;
                    if self.take("?") {
                        None
                    } else {
                        Some(self.i64()?)
                    }
                } else {
                    Some(0)
                };
                self.expect(">")?;
                buffer = buffer.with_layout(BufferLayout::Strided { offset, strides });
                if self.take(",") {
                    buffer = buffer.with_memory_space(self.attribute(depth + 1)?);
                }
            } else {
                match self.attribute(depth + 1)? {
                    Attribute::AffineMap(map) => {
                        buffer = buffer.with_layout(BufferLayout::Affine(map));
                        if self.take(",") {
                            buffer = buffer.with_memory_space(self.attribute(depth + 1)?);
                        }
                    }
                    memory_space => buffer = buffer.with_memory_space(memory_space),
                }
            }
        }
        self.expect(">")?;
        Ok(Type::Buffer(buffer))
    }
}
