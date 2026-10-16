use super::{
    Attribute, Attributes, Cursor, ParsedOperation, StableHloAttribute, TextError, TextErrorKind,
    Type,
};
use crate::{ElementType, ScalarValue, TensorType};

impl Cursor<'_> {
    pub(super) fn custom_convolution(
        &mut self,
        offset: usize,
        name: String,
        results: Vec<String>,
        depth: usize,
    ) -> Result<ParsedOperation, TextError> {
        self.expect("(")?;
        let operands = self.value_list(")")?;
        self.expect("dim_numbers")?;
        self.expect("=")?;
        let dimensions = self.convolution_dimension_numbers()?;
        let mut attributes = Attributes::new();
        attributes.insert(
            "dimension_numbers".into(),
            Attribute::StableHlo(Box::new(StableHloAttribute::ConvDimensionNumbers(
                dimensions,
            ))),
        );
        self.expect(",")?;
        self.expect("window")?;
        self.expect("=")?;
        self.expect("{")?;
        while !self.take("}") {
            let keyword = self.word()?;
            self.expect("=")?;
            let (name, value) = match keyword {
                "pad" => {
                    self.expect("[")?;
                    let mut values = Vec::new();
                    if !self.take("]") {
                        loop {
                            let pair = self.int_list()?;
                            if pair.len() != 2 {
                                return Err(self.error(TextErrorKind::Expected(
                                    "window padding pair containing exactly two integers",
                                )));
                            }
                            values.extend(
                                pair.into_iter()
                                    .map(|value| ScalarValue::Integer(i128::from(value))),
                            );
                            if self.take("]") {
                                break;
                            }
                            self.expect(",")?;
                        }
                    }
                    let rows = u64::try_from(values.len() / 2)
                        .map_err(|_| self.error(TextErrorKind::ResourceLimit))?;
                    (
                        "padding",
                        Attribute::DenseElements {
                            ty: Box::new(
                                TensorType::static_shape(ElementType::I64, [rows, 2]).into(),
                            ),
                            values,
                        },
                    )
                }
                "reverse" => (
                    "window_reversal",
                    Attribute::bools(
                        self.int_list()?
                            .into_iter()
                            .map(|value| value != 0)
                            .collect(),
                    ),
                ),
                "stride" | "lhs_dilate" | "rhs_dilate" => {
                    let name = match keyword {
                        "stride" => "window_strides",
                        "lhs_dilate" => "lhs_dilation",
                        _ => "rhs_dilation",
                    };
                    (name, Attribute::integers(self.int_list()?))
                }
                _ => {
                    return Err(self.error(TextErrorKind::Expected("convolution window attribute")));
                }
            };
            if attributes.insert(name.into(), value).is_some() {
                return Err(self.error(TextErrorKind::DuplicateField(name.into())));
            }
            if self.take("}") {
                break;
            }
            self.expect(",")?;
        }
        self.custom_dictionary(&mut attributes, depth)?;
        self.expect(":")?;
        let Type::Function(signature) = self.ty(depth + 1)? else {
            return Err(self.error(TextErrorKind::Expected("convolution function type")));
        };
        if signature.inputs().len() != operands.len() || signature.outputs().len() != results.len()
        {
            return Err(self.error(TextErrorKind::Expected(
                "convolution operand/result type counts",
            )));
        }
        Ok(ParsedOperation {
            location: self.trailing_location(depth)?,
            offset,
            name,
            results,
            operands,
            attributes,
            signature,
            regions: vec![],
        })
    }
}
