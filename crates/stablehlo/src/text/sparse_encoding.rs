use super::{TextError, TextErrorKind, cursor::Cursor};
use crate::{
    AffineExpr, AffineMap, Attribute, AttributeValidationError, SparseEncodingError,
    SparseLevelFormat, SparseLevelType, SparseTensorEncoding,
};
use std::collections::{HashMap, HashSet};

impl Cursor<'_> {
    pub(super) fn sparse_encoding(&mut self, depth: usize) -> Result<Attribute, TextError> {
        self.expect("<")?;
        self.expect("{")?;
        let mut fields = HashSet::new();
        let mut encoding = None;
        let mut position_width = 0;
        let mut coordinate_width = 0;
        let mut explicit_value = None;
        let mut implicit_value = None;
        if !self.take("}") {
            loop {
                let name = self.word()?.to_owned();
                if !fields.insert(name.clone()) {
                    return Err(self.error(TextErrorKind::DuplicateField(name)));
                }
                self.expect("=")?;
                match name.as_str() {
                    "map" => encoding = Some(self.sparse_map(depth + 1)?),
                    "posWidth" | "crdWidth" => {
                        let attribute = self.attribute(depth + 1)?;
                        let value = match attribute {
                            Attribute::Integer { value, .. } => value,
                            Attribute::Bool(value) => i128::from(value),
                            _ => {
                                return Err(self.error(TextErrorKind::Expected(
                                    "integer sparse storage width",
                                )));
                            }
                        };
                        let value = u32::try_from(value)
                            .map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
                        if name == "posWidth" {
                            position_width = value;
                        } else {
                            coordinate_width = value;
                        }
                    }
                    "explicitVal" => explicit_value = Some(Box::new(self.attribute(depth + 1)?)),
                    "implicitVal" => implicit_value = Some(Box::new(self.attribute(depth + 1)?)),
                    _ => return Err(self.error(TextErrorKind::Expected("sparse encoding field"))),
                }
                if self.take("}") {
                    break;
                }
                self.expect(",")?;
                if self.take("}") {
                    break;
                }
            }
        }
        self.expect(">")?;
        let mut encoding =
            encoding.ok_or_else(|| self.error(TextErrorKind::Expected("sparse encoding map")))?;
        encoding.position_width = position_width;
        encoding.coordinate_width = coordinate_width;
        encoding.explicit_value = explicit_value;
        encoding.implicit_value = implicit_value;
        encoding
            .verify()
            .map_err(|error| self.sparse_error(error))?;
        Ok(Attribute::SparseTensorEncoding(Box::new(encoding)))
    }

    fn sparse_names(
        &mut self,
        start: &'static str,
        end: &'static str,
        names: &mut HashMap<String, AffineExpr>,
        symbols: bool,
        all: &mut HashSet<String>,
    ) -> Result<Vec<String>, TextError> {
        self.expect(start)?;
        let mut ordered = Vec::new();
        if !self.take(end) {
            loop {
                let name = self.word()?.to_owned();
                if !all.insert(name.clone()) {
                    return Err(self.error(TextErrorKind::DuplicateField(name)));
                }
                let expr = if symbols {
                    AffineExpr::Symbol(ordered.len())
                } else {
                    AffineExpr::Dimension(ordered.len())
                };
                names.insert(name.clone(), expr);
                ordered.push(name);
                if self.take(end) {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(ordered)
    }

    fn sparse_map(&mut self, depth: usize) -> Result<SparseTensorEncoding, TextError> {
        self.depth(depth)?;
        let mut all = HashSet::new();
        let mut dimensions = HashMap::new();
        let mut levels = HashMap::new();
        let symbols = if self.starts("[") {
            self.sparse_names("[", "]", &mut dimensions, true, &mut all)?
                .len()
        } else {
            0
        };
        levels.extend(dimensions.clone());
        let declared = if self.starts("{") {
            self.sparse_names("{", "}", &mut levels, false, &mut all)?
        } else {
            vec![]
        };
        let mut dimension_count = 0;
        let mut inverse = Vec::new();
        let mut slices = Vec::new();
        self.expect("(")?;
        if !self.take(")") {
            loop {
                let name = self.word()?.to_owned();
                if !all.insert(name.clone()) {
                    return Err(self.error(TextErrorKind::DuplicateField(name)));
                }
                dimensions.insert(name, AffineExpr::Dimension(dimension_count));
                dimension_count += 1;
                if self.take("=") {
                    inverse.push(self.affine_expression(
                        &levels,
                        declared.len(),
                        symbols,
                        depth + 1,
                    )?);
                }
                slices.push(if self.take(":") {
                    match self.attribute(depth + 1)? {
                        Attribute::SparseTensorSlice(slice) => Some(slice),
                        _ => {
                            return Err(self.error(TextErrorKind::Expected(
                                "sparse dimension slice attribute",
                            )));
                        }
                    }
                } else {
                    None
                });
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        self.expect("->")?;
        self.expect("(")?;
        let mut results = Vec::new();
        let mut level_types = Vec::new();
        if !self.take(")") {
            loop {
                if !declared.is_empty() {
                    let name = self.word()?;
                    if declared
                        .get(results.len())
                        .is_none_or(|expected| expected != name)
                    {
                        return Err(self.error(TextErrorKind::Expected(
                            "level binding in declaration order",
                        )));
                    }
                    self.expect("=")?;
                }
                results.push(self.affine_expression(
                    &dimensions,
                    dimension_count,
                    symbols,
                    depth + 1,
                )?);
                self.expect(":")?;
                level_types.push(self.sparse_level_type()?);
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        if !declared.is_empty() && declared.len() != results.len() {
            return Err(self.sparse_error(SparseEncodingError::MapRankMismatch));
        }
        let map = AffineMap {
            dimensions: dimension_count,
            symbols,
            results,
        };
        let mut encoding = SparseTensorEncoding::new(level_types, map)
            .map_err(|error| self.sparse_error(error))?;
        if !inverse.is_empty() {
            encoding.level_to_dimension = Some(AffineMap {
                dimensions: encoding.levels.len(),
                symbols,
                results: inverse,
            });
        }
        if slices.iter().any(Option::is_some) {
            encoding.dimension_slices = slices
                .into_iter()
                .map(|slice| slice.unwrap_or_default())
                .collect();
        }
        encoding
            .verify()
            .map_err(|error| self.sparse_error(error))?;
        Ok(encoding)
    }

    fn sparse_level_type(&mut self) -> Result<SparseLevelType, TextError> {
        use SparseLevelFormat::*;
        let format = match self.word()? {
            "dense" => Dense,
            "batch" => Batch,
            "compressed" => Compressed,
            "loose_compressed" => LooseCompressed,
            "singleton" => Singleton,
            "structured" => {
                self.expect("[")?;
                let n = u8::try_from(self.i64()?)
                    .map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
                self.expect(",")?;
                let m = u8::try_from(self.i64()?)
                    .map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
                self.expect("]")?;
                Structured { n, m }
            }
            _ => return Err(self.error(TextErrorKind::Expected("sparse level format"))),
        };
        let mut level = SparseLevelType::new(format);
        if self.take("(") && !self.take(")") {
            loop {
                match self.word()? {
                    "nonunique" => level.nonunique = true,
                    "nonordered" => level.nonordered = true,
                    "soa" => level.soa = true,
                    _ => return Err(self.error(TextErrorKind::Expected("sparse level property"))),
                }
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        level.verify().map_err(|error| self.sparse_error(error))?;
        Ok(level)
    }

    fn sparse_error(&self, error: SparseEncodingError) -> TextError {
        self.error(TextErrorKind::InvalidAttribute(
            AttributeValidationError::SparseEncoding(error),
        ))
    }
}

pub(super) fn print(encoding: &SparseTensorEncoding, depth: usize) -> Result<String, TextError> {
    encoding.verify().map_err(|error| TextError {
        offset: 0,
        kind: TextErrorKind::InvalidAttribute(AttributeValidationError::SparseEncoding(error)),
    })?;
    let map = &encoding.dimension_to_level;
    if map.dimensions.saturating_add(map.symbols) > 1_048_576 {
        return Err(TextError {
            offset: 0,
            kind: TextErrorKind::ResourceLimit,
        });
    }
    let symbols = if map.symbols == 0 {
        String::new()
    } else {
        format!(
            "[{}]",
            (0..map.symbols)
                .map(|i| format!("s{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let dimensions = (0..map.dimensions)
        .map(|i| {
            let slice = encoding
                .dimension_slices
                .get(i)
                .map_or_else(String::new, |slice| {
                    format!(" : {}", super::sparse::print_slice(slice))
                });
            format!("d{i}{slice}")
        })
        .collect::<Vec<_>>()
        .join(", ");
    let levels = map
        .results
        .iter()
        .zip(&encoding.levels)
        .map(|(expression, level)| {
            let format = match level.format {
                SparseLevelFormat::Dense => "dense".into(),
                SparseLevelFormat::Batch => "batch".into(),
                SparseLevelFormat::Compressed => "compressed".into(),
                SparseLevelFormat::LooseCompressed => "loose_compressed".into(),
                SparseLevelFormat::Singleton => "singleton".into(),
                SparseLevelFormat::Structured { n, m } => format!("structured[{n}, {m}]"),
            };
            let properties = [
                ("nonunique", level.nonunique),
                ("nonordered", level.nonordered),
                ("soa", level.soa),
            ]
            .into_iter()
            .filter_map(|(name, enabled)| enabled.then_some(name))
            .collect::<Vec<_>>();
            let properties = if properties.is_empty() {
                String::new()
            } else {
                format!("({})", properties.join(", "))
            };
            Ok(format!(
                "{} : {format}{properties}",
                super::affine::expression(expression, depth + 1)?
            ))
        })
        .collect::<Result<Vec<_>, TextError>>()?
        .join(", ");
    let mut fields = vec![format!("map = {symbols}({dimensions}) -> ({levels})")];
    if encoding.position_width != 0 {
        fields.push(format!("posWidth = {}", encoding.position_width));
    }
    if encoding.coordinate_width != 0 {
        fields.push(format!("crdWidth = {}", encoding.coordinate_width));
    }
    for (name, value) in [
        ("explicitVal", encoding.explicit_value.as_deref()),
        ("implicitVal", encoding.implicit_value.as_deref()),
    ] {
        if let Some(value) = value {
            fields.push(format!(
                "{name} = {}",
                super::attributes::print(value, depth + 1)?
            ));
        }
    }
    Ok(format!(
        "#sparse_tensor.encoding<{{ {} }}>",
        fields.join(", ")
    ))
}
