//! File-scoped MLIR aliases resolve into the canonical type/attribute model.

use std::collections::HashMap;

use super::{TextError, TextErrorKind, cursor::Cursor};
use crate::{Attribute, Type};

#[derive(Clone, Copy)]
pub(super) struct Alias {
    start: usize,
    cost: usize,
}

#[derive(Default)]
pub(super) struct Aliases {
    attributes: HashMap<String, Alias>,
    types: HashMap<String, Alias>,
    expanded: usize,
}

impl Cursor<'_> {
    pub(super) fn alias_definitions(&mut self) -> Result<(), TextError> {
        while self.starts("#") || self.starts("!") {
            let is_type = self.take("!");
            if !is_type {
                self.expect("#")?;
            }
            let name = self.alias_name()?.to_owned();
            if name.contains('.') {
                return Err(self.error(TextErrorKind::Expected(
                    "alias name without a reserved dialect namespace",
                )));
            }
            let aliases = if is_type {
                &self.aliases.types
            } else {
                &self.aliases.attributes
            };
            if aliases.contains_key(&name) {
                return Err(self.error(TextErrorKind::DuplicateField(name)));
            }
            self.expect("=")?;
            self.space();
            let start = self.position();
            let cost = if is_type {
                let ty = self.ty(0)?;
                ty.verify()
                    .map_err(|error| self.error(TextErrorKind::InvalidType(error)))?;
                type_cost(&ty)
            } else {
                attribute_cost(&self.attribute(0)?)
            };
            let alias = Alias {
                start,
                cost: cost.saturating_add(self.position() - start),
            };
            let aliases = if is_type {
                &mut self.aliases.types
            } else {
                &mut self.aliases.attributes
            };
            aliases.insert(name, alias);
        }
        Ok(())
    }

    pub(super) fn type_alias(&mut self, depth: usize) -> Result<Type, TextError> {
        self.expect("!")?;
        let name = self.alias_name()?.to_owned();
        let alias = self
            .aliases
            .types
            .get(&name)
            .copied()
            .ok_or_else(|| self.error(TextErrorKind::UnknownAlias(format!("!{name}"))))?;
        self.expand_alias(alias, depth, true, |cursor, depth| cursor.ty(depth))
    }

    pub(super) fn attribute_alias(&mut self, depth: usize) -> Result<Attribute, TextError> {
        self.expect("#")?;
        let name = self.alias_name()?.to_owned();
        let alias = self
            .aliases
            .attributes
            .get(&name)
            .copied()
            .ok_or_else(|| self.error(TextErrorKind::UnknownAlias(format!("#{name}"))))?;
        self.expand_alias(alias, depth, false, |cursor, depth| cursor.attribute(depth))
    }

    fn expand_alias<T>(
        &mut self,
        mut alias: Alias,
        depth: usize,
        is_type: bool,
        parse: impl FnOnce(&mut Self, usize) -> Result<T, TextError>,
    ) -> Result<T, TextError> {
        let saved = self.offset;
        let prefix = if is_type { "!" } else { "#" };
        // Follow direct reference chains without adding a parser stack frame
        // for every link. Each link still consumes depth and expansion budget.
        let value = (|| {
            let mut alias_depth = 0usize;
            loop {
                self.aliases.expanded = self.aliases.expanded.saturating_add(alias.cost);
                if self.aliases.expanded > 1_048_576 {
                    return Err(self.error(TextErrorKind::ResourceLimit));
                }
                self.offset = alias.start;
                if depth.saturating_add(alias_depth) >= 128 {
                    return Err(self.error(TextErrorKind::NestingLimit));
                }
                if !self.take(prefix) {
                    break;
                }
                alias_depth += 1;
                let name = self.alias_name()?.to_owned();
                if name.contains('.') || self.starts("<") {
                    self.offset = alias.start;
                    break;
                }
                alias = if is_type {
                    self.aliases.types.get(&name)
                } else {
                    self.aliases.attributes.get(&name)
                }
                .copied()
                .ok_or_else(|| {
                    self.error(TextErrorKind::UnknownAlias(format!("{prefix}{name}")))
                })?;
            }
            parse(self, depth)
        })();
        self.offset = saved;
        value
    }

    pub(super) fn alias_name(&mut self) -> Result<&str, TextError> {
        let start = self.offset;
        let numeric = self
            .input
            .as_bytes()
            .get(start)
            .is_some_and(u8::is_ascii_digit);
        while self.input.as_bytes().get(self.offset).is_some_and(|byte| {
            if numeric {
                byte.is_ascii_digit()
            } else {
                byte.is_ascii_alphanumeric() || matches!(byte, b'$' | b'.' | b'_' | b'-')
            }
        }) {
            self.offset += 1;
        }
        if start == self.offset {
            return Err(self.error(TextErrorKind::Expected(
                "alias identifier adjacent to its prefix",
            )));
        }
        Ok(&self.input[start..self.offset])
    }
}

fn type_cost(ty: &Type) -> usize {
    1 + match ty {
        Type::Tensor(tensor) => tensor.rank() + tensor.encoding().map_or(0, attribute_cost),
        Type::QuantizedTensor(tensor) => {
            tensor.dimensions().len()
                + tensor.zero_points().len() * 2
                + tensor.encoding().map_or(0, attribute_cost)
        }
        Type::QuantizedElement(element) => element.zero_points().len() * 2,
        Type::UnrankedTensor(tensor) => match tensor {
            crate::UnrankedTensorType::Plain(_) => 0,
            crate::UnrankedTensorType::Quantized(element) => element.zero_points().len() * 2,
        },
        Type::Tuple(types) | Type::Future(types) => types.iter().map(type_cost).sum(),
        Type::Function(function) => function
            .inputs()
            .iter()
            .chain(function.outputs())
            .map(type_cost)
            .sum(),
        Type::Buffer(buffer) => {
            buffer.dimensions().len()
                + buffer.memory_space().map_or(0, attribute_cost)
                + match buffer.layout() {
                    Some(crate::BufferLayout::Affine(map)) => {
                        map.dimensions.saturating_add(map.symbols).saturating_add(
                            map.results.iter().map(crate::AffineExpr::node_count).sum(),
                        )
                    }
                    Some(crate::BufferLayout::Strided { strides, .. }) => strides.len() + 1,
                    Some(crate::BufferLayout::Opaque(syntax)) => syntax.len(),
                    _ => 0,
                }
        }
        Type::Element(_) | Type::Token | Type::String => 0,
    }
}

fn attribute_cost(attribute: &Attribute) -> usize {
    1 + match attribute {
        Attribute::SparseTensorEncoding(encoding) => {
            encoding.levels.len()
                + encoding.dimension_slices.len() * 3
                + encoding
                    .dimension_to_level
                    .results
                    .iter()
                    .map(crate::AffineExpr::node_count)
                    .sum::<usize>()
                + encoding.level_to_dimension.as_ref().map_or(0, |map| {
                    map.results.iter().map(crate::AffineExpr::node_count).sum()
                })
                + encoding.explicit_value.as_deref().map_or(0, attribute_cost)
                + encoding.implicit_value.as_deref().map_or(0, attribute_cost)
        }
        Attribute::AffineMap(map) => map
            .dimensions
            .saturating_add(map.symbols)
            .saturating_add(map.results.iter().map(crate::AffineExpr::node_count).sum()),
        Attribute::IntegerSet(set) => set.dimensions.saturating_add(set.symbols).saturating_add(
            set.constraints
                .iter()
                .map(|c| c.expression.node_count())
                .sum(),
        ),
        Attribute::Location(location) => location_cost(location),
        Attribute::Array(values) => values.iter().map(attribute_cost).sum(),
        Attribute::Dictionary(values) => values
            .iter()
            .map(|(key, value)| key.len() + attribute_cost(value))
            .sum(),
        Attribute::Type(ty) => type_cost(ty),
        Attribute::TypedString { value, ty } => value.len() + type_cost(ty),
        Attribute::DenseElements { ty, values } => type_cost(ty) + values.len(),
        Attribute::SparseElements {
            ty,
            indices,
            values,
        } => type_cost(ty) + values.len() + indices.iter().map(Vec::len).sum::<usize>(),
        Attribute::DenseArray { values, .. } => values.len(),
        Attribute::String(value) => value.len(),
        Attribute::Bytes(value) => value.len(),
        Attribute::SymbolRef(path) => path.iter().map(String::len).sum(),
        Attribute::Enum {
            dialect,
            name,
            value,
        } => dialect.len() + name.len() + value.len(),
        Attribute::Opaque { dialect, syntax } => {
            dialect.as_ref().map_or(0, String::len) + syntax.len()
        }

        Attribute::SparseTensorSlice(_) => 3,
        Attribute::StridedLayout { strides, .. } => strides.len() + 1,
        Attribute::StableHlo(_)
        | Attribute::Unit
        | Attribute::Bool(_)
        | Attribute::Integer { .. }
        | Attribute::Complex { .. }
        | Attribute::Float { .. } => 0,
    }
}

fn location_cost(location: &crate::Location) -> usize {
    use crate::Location;
    1 + match location {
        Location::Unknown => 0,
        Location::FileLine { file, .. }
        | Location::FileLineCol { file, .. }
        | Location::FileRange { file, .. } => file.len(),
        Location::Name { name, child } => {
            name.len() + child.as_ref().map_or(0, |child| location_cost(child))
        }
        Location::CallSite { callee, caller } => location_cost(callee) + location_cost(caller),
        Location::Fused {
            locations,
            metadata,
        } => {
            locations.iter().map(location_cost).sum::<usize>()
                + metadata
                    .as_ref()
                    .map_or(0, |metadata| attribute_cost(metadata))
        }
    }
}
