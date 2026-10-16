use super::super::ParsedBlock;
use super::{
    Attributes, Cursor, FunctionType, ParsedOperation, StableHloOp, TensorType, TextError,
    TextErrorKind, Type,
};
use crate::{Attribute, QuantizedTensorType};

impl Cursor<'_> {
    pub(super) fn custom_composite(
        &mut self,
        offset: usize,
        name: String,
        results: Vec<String>,
        depth: usize,
    ) -> Result<ParsedOperation, TextError> {
        let mut attributes = Attributes::new();
        attributes.insert("name".into(), Attribute::String(self.string()?));
        let operands = self.custom_operands()?;
        let mut regions = Vec::new();
        if self.take("(") {
            loop {
                regions.push(self.generic_region(depth + 1)?);
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        self.custom_dictionary(&mut attributes, depth)?;
        self.expect(":")?;
        let Type::Function(signature) = self.ty(depth + 1)? else {
            return Err(self.error(TextErrorKind::Expected("composite function type")));
        };
        if signature.inputs().len() != operands.len() || signature.outputs().len() != results.len()
        {
            return Err(self.error(TextErrorKind::Expected("composite operand/result counts")));
        }
        Ok(ParsedOperation {
            location: self.trailing_location(depth)?,
            offset,
            name,
            results,
            operands,
            attributes,
            signature,
            regions,
        })
    }

    pub(super) fn custom_while(
        &mut self,
        offset: usize,
        name: String,
        results: Vec<String>,
        depth: usize,
    ) -> Result<ParsedOperation, TextError> {
        self.expect("(")?;
        let mut names = Vec::new();
        let mut operands = Vec::new();
        if !self.take(")") {
            loop {
                names.push(self.prefixed_name("%")?);
                self.expect("=")?;
                operands.push(self.value_reference()?);
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        let mut inputs = Vec::new();
        if !operands.is_empty() {
            self.expect(":")?;
            loop {
                inputs.push(self.ty(depth + 1)?);
                if !self.take(",") {
                    break;
                }
            }
        }
        if inputs.len() != operands.len() || results.len() != operands.len() {
            return Err(self.error(TextErrorKind::Expected(
                "while iteration arguments, inputs, types and results with equal counts",
            )));
        }
        let mut attributes = Attributes::new();
        if self.take("attributes") {
            self.custom_dictionary(&mut attributes, depth)?;
        }
        let arguments = names
            .into_iter()
            .zip(inputs.iter().cloned())
            .collect::<Vec<_>>();
        self.expect("cond")?;
        let condition = self.custom_argument_region(arguments.clone(), depth)?;
        self.expect("do")?;
        let body = self.custom_argument_region(arguments, depth)?;
        Ok(ParsedOperation {
            location: self.trailing_location(depth)?,
            offset,
            name,
            results,
            operands,
            attributes,
            signature: FunctionType::new(inputs.clone(), inputs),
            regions: vec![condition, body],
        })
    }

    pub(super) fn custom_reduce(
        &mut self,
        offset: usize,
        name: String,
        results: Vec<String>,
        depth: usize,
    ) -> Result<ParsedOperation, TextError> {
        let mut operands = Vec::new();
        let mut initial = Vec::new();
        while self.take("(") {
            operands.push(self.value_reference()?);
            self.expect("init")?;
            self.expect(":")?;
            initial.push(self.value_reference()?);
            self.expect(")")?;
            if !self.take(",") {
                break;
            }
        }
        operands.extend(initial);
        let inner = if self.take("applies") {
            let name = if self.starts("\"") {
                self.string()?
            } else {
                self.word()?.into()
            };
            let op = StableHloOp::ALL
                .iter()
                .copied()
                .find(|op| {
                    name.strip_prefix("stablehlo.") == Some(op.mnemonic())
                        && matches!(
                            op,
                            StableHloOp::Add
                                | StableHloOp::Multiply
                                | StableHloOp::Maximum
                                | StableHloOp::Minimum
                                | StableHloOp::And
                                | StableHloOp::Or
                                | StableHloOp::Xor
                        )
                })
                .ok_or_else(|| {
                    self.error(TextErrorKind::Expected(
                        "commutative binary StableHLO operation with one result and no regions",
                    ))
                })?;
            Some(op)
        } else {
            None
        };
        self.expect("across")?;
        let mut attributes = Attributes::new();
        self.custom_array_field(&mut attributes, "dimensions", "dimensions", depth)?;
        self.custom_dictionary(&mut attributes, depth)?;
        self.expect(":")?;
        let Type::Function(signature) = self.ty(depth + 1)? else {
            return Err(self.error(TextErrorKind::Expected("reduce function type")));
        };
        if signature.inputs().len() != operands.len() || signature.outputs().len() != results.len()
        {
            return Err(self.error(TextErrorKind::Expected(
                "reduce signature matching operand and result counts",
            )));
        }
        let location = if inner.is_some() {
            self.trailing_location(depth)?
        } else {
            None
        };
        let region = if let Some(op) = inner {
            let ty = scalar_type(
                signature
                    .inputs()
                    .first()
                    .ok_or_else(|| self.error(TextErrorKind::Expected("reduce input type")))?,
            )
            .ok_or_else(|| self.error(TextErrorKind::Expected("reduce tensor input")))?;
            let lhs = format!(":reduce{offset}.lhs");
            let rhs = format!(":reduce{offset}.rhs");
            let output = format!(":reduce{offset}.result#0");
            vec![ParsedBlock {
                argument_locations: vec![location.clone(), location.clone()],
                arguments: vec![(lhs.clone(), ty.clone()), (rhs.clone(), ty.clone())],
                operations: vec![
                    ParsedOperation {
                        location: location.clone(),
                        offset,
                        name: format!("stablehlo.{}", op.mnemonic()),
                        results: vec![output.clone()],
                        operands: vec![format!("{lhs}#0"), format!("{rhs}#0")],
                        regions: vec![],
                        attributes: Attributes::new(),
                        signature: FunctionType::new(
                            vec![ty.clone(), ty.clone()],
                            vec![ty.clone()],
                        ),
                    },
                    ParsedOperation {
                        location: location.clone(),
                        offset,
                        name: "stablehlo.return".into(),
                        results: vec![],
                        operands: vec![output],
                        regions: vec![],
                        attributes: Attributes::new(),
                        signature: FunctionType::new(vec![ty], vec![]),
                    },
                ],
            }]
        } else {
            self.expect("reducer")?;
            let mut arguments = Vec::new();
            let mut initial = Vec::new();
            let mut argument_locations = Vec::new();
            let mut initial_locations = Vec::new();
            while self.take("(") {
                let name = self.prefixed_name("%")?;
                self.expect(":")?;
                arguments.push((name, self.ty(depth + 1)?));
                argument_locations.push(self.trailing_location(depth + 1)?);
                self.expect(",")?;
                let name = self.prefixed_name("%")?;
                self.expect(":")?;
                initial.push((name, self.ty(depth + 1)?));
                initial_locations.push(self.trailing_location(depth + 1)?);
                self.expect(")")?;
            }
            arguments.extend(initial);
            argument_locations.extend(initial_locations);
            let mut region = self.custom_argument_region(arguments, depth)?;
            region[0].argument_locations = argument_locations;
            region
        };
        Ok(ParsedOperation {
            location: if location.is_some() {
                location
            } else {
                self.trailing_location(depth)?
            },
            offset,
            name,
            results,
            operands,
            attributes,
            signature,
            regions: vec![region],
        })
    }

    fn custom_argument_region(
        &mut self,
        arguments: Vec<(String, Type)>,
        depth: usize,
    ) -> Result<Vec<ParsedBlock>, TextError> {
        let mut region = self.generic_region(depth + 1)?;
        if region.is_empty() {
            region.push(ParsedBlock {
                argument_locations: vec![],
                arguments: vec![],
                operations: vec![],
            });
        }
        if region.len() != 1 || !region[0].arguments.is_empty() {
            return Err(self.error(TextErrorKind::Expected(
                "one region block with arguments supplied by the operation",
            )));
        }
        region[0].arguments = arguments;
        Ok(region)
    }
}

fn scalar_type(ty: &Type) -> Option<Type> {
    match ty {
        Type::Tensor(tensor) => Some(TensorType::scalar(tensor.element()).into()),
        Type::QuantizedTensor(tensor) => Some(Type::QuantizedTensor(
            QuantizedTensorType::from_element(vec![], tensor.element().clone()),
        )),
        _ => None,
    }
}
