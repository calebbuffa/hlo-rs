use super::{
    Attributes, Cursor, FunctionType, ParsedOperation, StableHloOp, TextError, TextErrorKind, Type,
    dictionary,
};
use crate::{Attribute, DotDimensionNumbers, ElementType, StableHloAttribute, TensorType};

mod convolution;
mod regions;

#[derive(Clone, Copy)]
enum Signature {
    Same,
    Functional,
    Result,
    Tuple,
    Complex,
    Select,
}

impl Cursor<'_> {
    pub(super) fn custom_stablehlo_operation(
        &mut self,
        offset: usize,
        results: Vec<String>,
        depth: usize,
    ) -> Result<ParsedOperation, TextError> {
        let name = self.word()?.to_owned();
        let op = StableHloOp::ALL
            .iter()
            .copied()
            .find(|op| name.strip_prefix("stablehlo.") == Some(op.mnemonic()))
            .ok_or_else(|| self.error(TextErrorKind::Expected("known StableHLO operation")))?;
        match op {
            StableHloOp::Constant => self.custom_constant(offset, name, results, depth),
            StableHloOp::OptimizationBarrier => self.custom_barrier(offset, name, results, depth),
            StableHloOp::While => self.custom_while(offset, name, results, depth),
            StableHloOp::Reduce => self.custom_reduce(offset, name, results, depth),
            StableHloOp::Composite => self.custom_composite(offset, name, results, depth),
            StableHloOp::Convolution => self.custom_convolution(offset, name, results, depth),
            _ => self.custom_flat_operation(offset, op, name, results, depth),
        }
    }

    fn custom_flat_operation(
        &mut self,
        offset: usize,
        op: StableHloOp,
        name: String,
        results: Vec<String>,
        depth: usize,
    ) -> Result<ParsedOperation, TextError> {
        let mut attributes = Attributes::new();
        let mut operands = Vec::new();
        let signature = match op {
            StableHloOp::Constant
            | StableHloOp::OptimizationBarrier
            | StableHloOp::While
            | StableHloOp::Reduce
            | StableHloOp::Composite
            | StableHloOp::Convolution => {
                unreachable!("specialized operations are handled by custom assembly dispatch");
            }
            StableHloOp::Abs
            | StableHloOp::Cbrt
            | StableHloOp::Ceil
            | StableHloOp::Convert
            | StableHloOp::CountLeadingZeros
            | StableHloOp::Cosine
            | StableHloOp::Exponential
            | StableHloOp::ExponentialMinusOne
            | StableHloOp::Floor
            | StableHloOp::Imag
            | StableHloOp::Log
            | StableHloOp::LogPlusOne
            | StableHloOp::Logistic
            | StableHloOp::Not
            | StableHloOp::Negate
            | StableHloOp::PopulationCount
            | StableHloOp::Real
            | StableHloOp::RoundNearestAfz
            | StableHloOp::RoundNearestEven
            | StableHloOp::Rsqrt
            | StableHloOp::Sign
            | StableHloOp::Sine
            | StableHloOp::Sqrt
            | StableHloOp::Tan
            | StableHloOp::Tanh
            | StableHloOp::UniformQuantize
            | StableHloOp::UniformDequantize
            | StableHloOp::Add
            | StableHloOp::Atan2
            | StableHloOp::Divide
            | StableHloOp::Maximum
            | StableHloOp::Minimum
            | StableHloOp::Multiply
            | StableHloOp::Power
            | StableHloOp::Remainder
            | StableHloOp::ShiftLeft
            | StableHloOp::ShiftRightArithmetic
            | StableHloOp::ShiftRightLogical
            | StableHloOp::Subtract
            | StableHloOp::And
            | StableHloOp::Or
            | StableHloOp::Xor
            | StableHloOp::AfterAll
            | StableHloOp::Clamp => {
                operands = self.custom_operands()?;
                Signature::Same
            }
            StableHloOp::IsFinite
            | StableHloOp::DynamicUpdateSlice
            | StableHloOp::BitcastConvert
            | StableHloOp::Reshape
            | StableHloOp::DynamicReshape
            | StableHloOp::RealDynamicSlice
            | StableHloOp::DynamicPad => {
                operands = self.custom_operands()?;
                Signature::Functional
            }
            StableHloOp::CreateToken | StableHloOp::ReplicaId | StableHloOp::PartitionId => {
                Signature::Result
            }
            StableHloOp::Tuple => {
                operands = self.custom_operands()?;
                Signature::Tuple
            }
            StableHloOp::Complex => {
                operands = self.custom_operands()?;
                Signature::Complex
            }
            StableHloOp::Select => {
                operands = self.custom_operands()?;
                Signature::Select
            }
            StableHloOp::Iota => {
                self.custom_integer_field(&mut attributes, "dim", "iota_dimension")?;
                Signature::Result
            }
            StableHloOp::DynamicIota
            | StableHloOp::GetDimensionSize
            | StableHloOp::SetDimensionSize => {
                operands = self.custom_operands()?;
                self.expect(",")?;
                self.custom_integer_field(
                    &mut attributes,
                    "dim",
                    if op == StableHloOp::DynamicIota {
                        "iota_dimension"
                    } else {
                        "dimension"
                    },
                )?;
                Signature::Functional
            }
            StableHloOp::Concatenate => {
                // VariadicOperandWithAttribute includes a trailing comma for each input.
                while self.starts("%") {
                    operands.push(self.value_reference()?);
                    self.expect(",")?;
                }
                self.custom_integer_field(&mut attributes, "dim", "dimension")?;
                Signature::Functional
            }
            StableHloOp::Broadcast
            | StableHloOp::BroadcastInDim
            | StableHloOp::DynamicBroadcastInDim
            | StableHloOp::Reverse
            | StableHloOp::Transpose
            | StableHloOp::DynamicSlice => {
                operands = self.custom_operands()?;
                self.expect(",")?;
                let (keyword, attribute) = match op {
                    StableHloOp::Broadcast => ("sizes", "broadcast_sizes"),
                    StableHloOp::BroadcastInDim | StableHloOp::DynamicBroadcastInDim => {
                        ("dims", "broadcast_dimensions")
                    }
                    StableHloOp::Reverse => ("dims", "dimensions"),
                    StableHloOp::Transpose => ("dims", "permutation"),
                    _ => ("sizes", "slice_sizes"),
                };
                self.custom_array_field(&mut attributes, keyword, attribute, depth)?;
                if op == StableHloOp::Reverse {
                    Signature::Same
                } else {
                    Signature::Functional
                }
            }
            StableHloOp::Pad => {
                operands = self.custom_operands()?;
                for (keyword, name) in [
                    ("low", "edge_padding_low"),
                    ("high", "edge_padding_high"),
                    ("interior", "interior_padding"),
                ] {
                    self.expect(",")?;
                    self.custom_array_field(&mut attributes, keyword, name, depth)?;
                }
                Signature::Functional
            }
            StableHloOp::Slice => {
                operands.push(self.value_reference()?);
                self.custom_slice_ranges(&mut attributes)?;
                Signature::Functional
            }
            StableHloOp::CustomCall => {
                attributes.insert(
                    "call_target_name".into(),
                    Attribute::String(self.function_symbol()?),
                );
                self.expect("(")?;
                operands = self.value_list(")")?;
                Signature::Functional
            }
            StableHloOp::Dot | StableHloOp::DotGeneral => {
                operands = self.custom_operands()?;
                if op == StableHloOp::DotGeneral {
                    self.expect(",")?;
                    let mut dimensions = DotDimensionNumbers::default();
                    if self.take("batching_dims") {
                        self.expect("=")?;
                        dimensions.lhs_batching_dimensions = self.int_list()?;
                        self.expect("x")?;
                        dimensions.rhs_batching_dimensions = self.int_list()?;
                        self.expect(",")?;
                    }
                    self.expect("contracting_dims")?;
                    self.expect("=")?;
                    dimensions.lhs_contracting_dimensions = self.int_list()?;
                    self.expect("x")?;
                    dimensions.rhs_contracting_dimensions = self.int_list()?;
                    attributes.insert(
                        "dot_dimension_numbers".into(),
                        Attribute::StableHlo(Box::new(StableHloAttribute::DotDimensionNumbers(
                            dimensions,
                        ))),
                    );
                }
                if self.take(",") {
                    if op == StableHloOp::DotGeneral && self.take("algorithm") {
                        self.expect("=")?;
                        attributes.insert(
                            "algorithm".into(),
                            Attribute::StableHlo(Box::new(
                                self.dialect_attribute("dot_algorithm", depth + 1)?,
                            )),
                        );
                    } else {
                        self.custom_precision(&mut attributes, depth)?;
                        if op == StableHloOp::DotGeneral && self.take(",") {
                            self.expect("algorithm")?;
                            self.expect("=")?;
                            attributes.insert(
                                "algorithm".into(),
                                Attribute::StableHlo(Box::new(
                                    self.dialect_attribute("dot_algorithm", depth + 1)?,
                                )),
                            );
                        }
                    }
                }
                Signature::Functional
            }
            StableHloOp::Compare => {
                attributes.insert(
                    "comparison_direction".into(),
                    self.custom_enum("comparison_direction", depth)?,
                );
                self.expect(",")?;
                operands.push(self.value_reference()?);
                self.expect(",")?;
                operands.push(self.value_reference()?);
                if self.take(",") {
                    attributes.insert(
                        "compare_type".into(),
                        self.custom_enum("comparison_type", depth)?,
                    );
                }
                Signature::Functional
            }
            StableHloOp::Einsum | StableHloOp::UnaryEinsum => {
                operands = self.custom_operands()?;
                self.expect(",")?;
                self.expect("config")?;
                self.expect("=")?;
                attributes.insert("einsum_config".into(), Attribute::String(self.string()?));
                Signature::Functional
            }
            StableHloOp::Fft => {
                operands = self.custom_operands()?;
                self.expect(",")?;
                self.expect("type")?;
                self.expect("=")?;
                attributes.insert("fft_type".into(), self.custom_enum("fft_type", depth)?);
                self.expect(",")?;
                self.custom_array_field(&mut attributes, "length", "fft_length", depth)?;
                Signature::Functional
            }
            StableHloOp::Rng | StableHloOp::RngBitGenerator => {
                operands = self.custom_operands()?;
                self.expect(",")?;
                let (keyword, category) = if op == StableHloOp::Rng {
                    ("distribution", "rng_distribution")
                } else {
                    ("algorithm", "rng_algorithm")
                };
                self.expect(keyword)?;
                self.expect("=")?;
                attributes.insert(category.into(), self.custom_enum(category, depth)?);
                Signature::Functional
            }
            StableHloOp::GetTupleElement => {
                operands.push(self.value_reference()?);
                self.expect("[")?;
                let index = i32::try_from(self.i64()?)
                    .map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
                attributes.insert("index".into(), Attribute::i32(index));
                self.expect("]")?;
                Signature::Functional
            }
            StableHloOp::Cholesky => {
                operands.push(self.value_reference()?);
                if self.take(",") {
                    self.expect("lower")?;
                    self.expect("=")?;
                    let lower = match self.word()? {
                        "true" => true,
                        "false" => false,
                        _ => return Err(self.error(TextErrorKind::Expected("boolean lower flag"))),
                    };
                    attributes.insert("lower".into(), Attribute::Bool(lower));
                }
                Signature::Same
            }
            StableHloOp::ReducePrecision => {
                operands.push(self.value_reference()?);
                self.expect(",")?;
                self.expect("format")?;
                self.expect("=")?;
                let format = self.word()?;
                let (exponent, mantissa) = format
                    .strip_prefix('e')
                    .and_then(|format| format.split_once('m'))
                    .filter(|(exponent, mantissa)| {
                        !exponent.is_empty()
                            && !mantissa.is_empty()
                            && exponent.bytes().all(|byte| byte.is_ascii_digit())
                            && mantissa.bytes().all(|byte| byte.is_ascii_digit())
                    })
                    .ok_or_else(|| {
                        self.error(TextErrorKind::Expected("exponent/mantissa format eNmN"))
                    })?;
                for (name, value) in [("exponent_bits", exponent), ("mantissa_bits", mantissa)] {
                    attributes.insert(
                        name.into(),
                        Attribute::i32(
                            value
                                .parse::<i32>()
                                .map_err(|_| self.error(TextErrorKind::InvalidNumber))?,
                        ),
                    );
                }
                Signature::Same
            }
            StableHloOp::Return => {
                return Err(self.error(TextErrorKind::Expected("return without result bindings")));
            }
            StableHloOp::Infeed
            | StableHloOp::Outfeed
            | StableHloOp::Send
            | StableHloOp::Recv
            | StableHloOp::If
            | StableHloOp::Case
            | StableHloOp::AllGather
            | StableHloOp::AllReduce
            | StableHloOp::ReduceScatter
            | StableHloOp::AllToAll
            | StableHloOp::AsyncStart
            | StableHloOp::AsyncDone
            | StableHloOp::BatchNormGrad
            | StableHloOp::BatchNormInference
            | StableHloOp::BatchNormTraining
            | StableHloOp::CollectiveBroadcast
            | StableHloOp::CollectiveReduce
            | StableHloOp::CollectivePermute
            | StableHloOp::CrossReplicaSum
            | StableHloOp::Gather
            | StableHloOp::Map
            | StableHloOp::Scatter
            | StableHloOp::SelectAndScatter
            | StableHloOp::Sort
            | StableHloOp::TriangularSolve
            | StableHloOp::ReduceWindow
            | StableHloOp::TorchIndexSelect
            | StableHloOp::DynamicGather
            | StableHloOp::DynamicConv => {
                return Err(
                    self.error(TextErrorKind::Expected("quoted generic operation assembly"))
                );
            }
        };
        self.custom_dictionary(&mut attributes, depth)?;
        self.expect(":")?;
        let signature = self.custom_signature(signature, operands.len(), depth)?;
        if operands.len() != signature.inputs().len() || results.len() != signature.outputs().len()
        {
            return Err(self.error(TextErrorKind::Expected(
                "signature matching operand/result counts",
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

    fn custom_operands(&mut self) -> Result<Vec<String>, TextError> {
        let mut operands = Vec::new();
        while self.starts("%") {
            operands.push(self.value_reference()?);
            if !self.starts(",") {
                break;
            }
            let before_comma = self.offset;
            self.expect(",")?;
            if !self.starts("%") {
                self.offset = before_comma;
                break;
            }
        }
        Ok(operands)
    }

    fn custom_dictionary(
        &mut self,
        attributes: &mut Attributes,
        depth: usize,
    ) -> Result<(), TextError> {
        if self.starts("{") {
            for (name, value) in dictionary(self.attribute(depth + 1)?)? {
                if attributes.insert(name.clone(), value).is_some() {
                    return Err(self.error(TextErrorKind::DuplicateField(name)));
                }
            }
        }
        Ok(())
    }

    fn custom_integer_field(
        &mut self,
        attributes: &mut Attributes,
        keyword: &'static str,
        name: &str,
    ) -> Result<(), TextError> {
        self.expect(keyword)?;
        self.expect("=")?;
        attributes.insert(name.into(), Attribute::integer(self.i64()?));
        Ok(())
    }

    fn custom_array_field(
        &mut self,
        attributes: &mut Attributes,
        keyword: &'static str,
        name: &str,
        depth: usize,
    ) -> Result<(), TextError> {
        self.expect(keyword)?;
        self.expect("=")?;
        let attribute = if self.starts("[") {
            Attribute::integers(self.int_list()?)
        } else {
            let attribute = self.attribute(depth + 1)?;
            if !matches!(
                attribute,
                Attribute::DenseArray {
                    element: ElementType::I64,
                    ..
                }
            ) {
                return Err(self.error(TextErrorKind::Expected("dense i64 array")));
            }
            attribute
        };
        attributes.insert(name.into(), attribute);
        Ok(())
    }

    fn custom_enum(
        &mut self,
        category: &'static str,
        depth: usize,
    ) -> Result<Attribute, TextError> {
        if self.starts("#") {
            let attribute = self.attribute(depth + 1)?;
            if !matches!(&attribute, Attribute::Enum { dialect, name, .. } if dialect == "stablehlo" && name == category)
            {
                return Err(self.error(TextErrorKind::Expected("matching StableHLO enum category")));
            }
            return Ok(attribute);
        }
        let value = self.word()?;
        if !crate::attribute::enums::values(category).is_some_and(|values| values.contains(&value))
        {
            return Err(self.error(TextErrorKind::Expected("valid StableHLO enum value")));
        }
        Ok(Attribute::Enum {
            dialect: "stablehlo".into(),
            name: category.into(),
            value: value.into(),
        })
    }

    fn custom_precision(
        &mut self,
        attributes: &mut Attributes,
        depth: usize,
    ) -> Result<(), TextError> {
        self.expect("precision")?;
        self.expect("=")?;
        self.expect("[")?;
        let mut precision = Vec::new();
        if !self.take("]") {
            loop {
                precision.push(self.custom_enum("precision", depth)?);
                if self.take("]") {
                    break;
                }
                self.expect(",")?;
            }
        }
        attributes.insert("precision_config".into(), Attribute::Array(precision));
        Ok(())
    }

    fn custom_slice_ranges(&mut self, attributes: &mut Attributes) -> Result<(), TextError> {
        self.expect("[")?;
        let (mut start, mut limit, mut stride) = (Vec::new(), Vec::new(), Vec::new());
        if !self.take("]") {
            loop {
                start.push(self.i64()?);
                self.expect(":")?;
                limit.push(self.i64()?);
                stride.push(if self.take(":") { self.i64()? } else { 1 });
                if self.take("]") {
                    break;
                }
                self.expect(",")?;
            }
        }
        for (name, values) in [
            ("start_indices", start),
            ("limit_indices", limit),
            ("strides", stride),
        ] {
            attributes.insert(name.into(), Attribute::integers(values));
        }
        Ok(())
    }

    fn custom_signature(
        &mut self,
        style: Signature,
        operands: usize,
        depth: usize,
    ) -> Result<FunctionType, TextError> {
        let ty = self.ty(depth + 1)?;
        if matches!(
            style,
            Signature::Functional | Signature::Same | Signature::Complex | Signature::Select
        ) && let Type::Function(signature) = ty
        {
            if !matches!(style, Signature::Functional) && signature.outputs().len() != 1 {
                return Err(self.error(TextErrorKind::Expected("single result function type")));
            }
            return Ok(signature);
        }
        let inputs = match style {
            Signature::Same => vec![ty.clone(); operands],
            Signature::Result => vec![],
            Signature::Tuple => {
                let Type::Tuple(elements) = &ty else {
                    return Err(self.error(TextErrorKind::Expected("tuple result type")));
                };
                elements.clone()
            }
            Signature::Complex => {
                let Type::Tensor(tensor) = &ty else {
                    return Err(
                        self.error(TextErrorKind::Expected("tensor with complex element type"))
                    );
                };
                let element = match tensor.element() {
                    ElementType::C64 => ElementType::F32,
                    ElementType::C128 => ElementType::F64,
                    _ => {
                        return Err(
                            self.error(TextErrorKind::Expected("tensor with complex element type"))
                        );
                    }
                };
                let mut real = TensorType::new(element, tensor.dimensions().to_vec());
                if let Some(encoding) = tensor.encoding() {
                    real = real.with_encoding(encoding.clone());
                }
                vec![Type::Tensor(real); 2]
            }
            Signature::Select => {
                self.expect(",")?;
                let selected = self.ty(depth + 1)?;
                return Ok(FunctionType::new(
                    vec![ty, selected.clone(), selected.clone()],
                    vec![selected],
                ));
            }
            Signature::Functional => {
                return Err(self.error(TextErrorKind::Expected("operation function type")));
            }
        };
        Ok(FunctionType::new(inputs, vec![ty]))
    }

    fn custom_barrier(
        &mut self,
        offset: usize,
        name: String,
        results: Vec<String>,
        depth: usize,
    ) -> Result<ParsedOperation, TextError> {
        let mut attributes = Attributes::new();
        self.custom_dictionary(&mut attributes, depth)?;
        let operands = if self.take("(") {
            self.expect(")")?;
            vec![]
        } else {
            let operands = self.custom_operands()?;
            if operands.is_empty() {
                return Err(self.error(TextErrorKind::Expected("barrier operands or ()")));
            }
            operands
        };
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
        if operands.len() != inputs.len() || results.len() != operands.len() {
            return Err(self.error(TextErrorKind::Expected("pairwise operand and result types")));
        }
        Ok(ParsedOperation {
            location: self.trailing_location(depth)?,
            offset,
            name,
            results,
            operands,
            attributes,
            signature: FunctionType::new(inputs.clone(), inputs),
            regions: vec![],
        })
    }

    fn custom_constant(
        &mut self,
        offset: usize,
        name: String,
        results: Vec<String>,
        depth: usize,
    ) -> Result<ParsedOperation, TextError> {
        if self.starts("(") {
            return self.generic_operation_body(offset, name, results, depth);
        }
        let mut attributes = Attributes::new();
        self.custom_dictionary(&mut attributes, depth)?;
        let value = self.attribute(depth + 1)?;
        let ty = match &value {
            Attribute::DenseElements { ty, .. } | Attribute::SparseElements { ty, .. } => {
                ty.as_ref().clone()
            }
            _ => return Err(self.error(TextErrorKind::Expected("constant elements attribute"))),
        };
        if attributes.insert("value".into(), value).is_some() {
            return Err(self.error(TextErrorKind::DuplicateField("value".into())));
        }
        if results.len() != 1 {
            return Err(self.error(TextErrorKind::Expected("single constant result")));
        }
        Ok(ParsedOperation {
            location: self.trailing_location(depth)?,
            offset,
            name,
            results,
            operands: vec![],
            regions: vec![],
            attributes,
            signature: FunctionType::new(vec![], vec![ty]),
        })
    }
}
