use super::{module, module_attributes, printer};
use crate::{attributes::*, ops::*, types::*};
use pliron::{
    attribute::{AttrObj, AttributeDict},
    builtin::attributes::{BoolAttr, IntegerAttr, StringAttr, VecAttr},
    common_traits::{Named, Verify},
    context::{Context, Ptr},
    derive::{attr_interface_impl, op_interface_impl, type_interface_impl},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    printable::State,
    result::Result,
    r#type::{Type, Typed},
};
use pliron_mlir_interop::{
    ToMlirAttr, ToMlirOp, ToMlirType,
    printers::{GenericOp, print_op, print_op_location, print_type},
};
use std::fmt;

macro_rules! types {
    ($($ty:ty),* $(,)?) => {$(
        #[type_interface_impl]
        impl ToMlirType for $ty {
            fn to_mlir(&self, ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
                write!(f, "{}", printer::dialect_type_text(ctx, self.get_self_handle(ctx))?)?;
                Ok(())
            }
        }
    )*};
}

types!(
    TokenType,
    FutureType,
    StringType,
    WitnessType,
    QuantizedElementType,
    AnyQuantizedType,
    CalibratedQuantizedType,
    QuantileType,
    SubchannelQuantizedType,
);

macro_rules! attributes {
    ($($ty:ty),* $(,)?) => {$(
        #[attr_interface_impl]
        impl ToMlirAttr for $ty {
            fn to_mlir(&self, ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
                write!(f, "{}", printer::dialect_attribute_text(ctx, self)?)?;
                Ok(())
            }
        }
    )*};
}

attributes!(
    EnumAttr,
    DotDimensionNumbersAttr,
    GatherDimensionNumbersAttr,
    ScatterDimensionNumbersAttr,
    ConvDimensionNumbersAttr,
    DotAlgorithmAttr,
    OutputOperandAliasAttr,
    ChannelHandleAttr,
    TypeExtensionsAttr,
    ResultAccuracyAttr,
    Precision,
    ComparisonDirection,
    ComparisonKind,
    FftKind,
    RngAlgorithmAttr,
    RngDistributionAttr,
    ResultAccuracyMode,
    SubAxisInfoAttr,
    AxisRefAttr,
    MeshAxisAttr,
    MeshAttr,
    MeshReferenceAttr,
    ReplicaGroupMeshAxesAttr,
    LayoutAttr,
    ArrayOfLayoutAttr,
    TilingAttr,
    ArrayOfTilingAttr,
    crate::sparse::SparseTensorEncoding,
);

macro_rules! operations {
    ($($ty:ty),* $(,)?) => {
    #[cfg(test)]
    pub(super) fn catalog_names(ctx: &mut Context) -> std::collections::BTreeSet<String> {
        let mut names = std::collections::BTreeSet::new();
        $(
            let op = Operation::new(ctx, <$ty>::get_concrete_op_info(), vec![], vec![], vec![], 0);
            let concrete = Operation::get_op_dyn(op, ctx);
            assert!(pliron::op::op_cast::<dyn ToMlirOp>(&*concrete).is_some());
            names.insert(concrete.get_opid().to_string());
        )*
        names
    }
    $(
        #[op_interface_impl]
        impl ToMlirOp for $ty {
            fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
                print_operation(ctx, self.get_operation(), state, f)
            }
        }
    )*};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_native_operation_has_a_framework_translation() {
        let ctx = &mut Context::new();
        let mut translated = catalog_names(ctx);
        translated.insert("stablehlo.module".to_owned());
        translated.insert("stablehlo.func".to_owned());
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("ops");
        let mut declared = std::collections::BTreeSet::new();
        for file in std::fs::read_dir(directory).unwrap() {
            let path = file.unwrap().path();
            if path.extension().is_none_or(|ext| ext != "rs") {
                continue;
            }
            let source = std::fs::read_to_string(path).unwrap();
            for declaration in source.split("#[pliron_op(").skip(1) {
                let name = declaration.split('"').nth(1).unwrap();
                declared.insert(name.to_owned());
                let (header, _) = declaration.split_once("pub struct ").unwrap();
                for field in header
                    .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .filter(|field| field.starts_with("stablehlo_"))
                {
                    assert!(
                        module_attributes::is_native_field(name, field),
                        "unmapped field {name}: {field}"
                    );
                }
            }
        }
        assert_eq!(
            declared, translated,
            "the exporter must cover the entire native operation catalog"
        );
    }
}

operations!(
    AllGatherOp,
    AllReduceOp,
    ReduceScatterOp,
    AllToAllOp,
    CollectiveBroadcastOp,
    CollectiveReduceOp,
    CollectivePermuteOp,
    CrossReplicaSumOp,
    DotGeneralOp,
    DotOp,
    EinsumOp,
    UnaryEinsumOp,
    ConvolutionOp,
    DynamicConvOp,
    FftOp,
    CholeskyOp,
    TriangularSolveOp,
    BatchNormInferenceOp,
    BatchNormTrainingOp,
    BatchNormGradOp,
    RngOp,
    RngBitGeneratorOp,
    ConstantOp,
    DynamicIotaOp,
    ReturnOp,
    CallOp,
    WhileOp,
    IfOp,
    CaseOp,
    TupleOp,
    GetTupleElementOp,
    CompositeOp,
    CustomCallOp,
    OptimizationBarrierOp,
    CreateTokenOp,
    AfterAllOp,
    ReplicaIdOp,
    PartitionIdOp,
    InfeedOp,
    OutfeedOp,
    SendOp,
    RecvOp,
    AsyncStartOp,
    AsyncDoneOp,
    ConvertOp,
    SelectOp,
    CompareOp,
    ClampOp,
    IotaOp,
    AbsOp,
    CbrtOp,
    CeilOp,
    CountLeadingZerosOp,
    CosineOp,
    ExponentialOp,
    ExponentialMinusOneOp,
    Exp2Op,
    FloorOp,
    ImagOp,
    IsFiniteOp,
    LogOp,
    LogPlusOneOp,
    Log2Op,
    LogisticOp,
    NotOp,
    NegateOp,
    PopulationCountOp,
    RealOp,
    RoundNearestAfzOp,
    RoundNearestEvenOp,
    RsqrtOp,
    SignOp,
    SineOp,
    SqrtOp,
    TanOp,
    TanhOp,
    AddOp,
    Atan2Op,
    ComplexOp,
    DivideOp,
    MaximumOp,
    MinimumOp,
    MultiplyOp,
    PowerOp,
    RemainderOp,
    ShiftLeftOp,
    ShiftRightArithmeticOp,
    ShiftRightLogicalOp,
    SubtractOp,
    AndOp,
    OrOp,
    XorOp,
    BitcastConvertOp,
    ReducePrecisionOp,
    UniformQuantizeOp,
    UniformDequantizeOp,
    GatherOp,
    DynamicGatherOp,
    ScatterOp,
    TorchIndexSelectOp,
    ReduceOp,
    ReduceWindowOp,
    SelectAndScatterOp,
    SortOp,
    MapOp,
    BroadcastInDimOp,
    ReshapeOp,
    TransposeOp,
    BroadcastOp,
    ReverseOp,
    DynamicBroadcastInDimOp,
    DynamicReshapeOp,
    GetDimensionSizeOp,
    SetDimensionSizeOp,
    SliceOp,
    ConcatenateOp,
    DynamicSliceOp,
    DynamicUpdateSliceOp,
    RealDynamicSliceOp,
    PadOp,
    DynamicPadOp,
);

fn invalid(reason: impl Into<String>) -> pliron::result::Error {
    pliron::input_error_noloc!(pliron_mlir_interop::Error::Untranslatable(reason.into()))
}

fn assembly_error(error: super::TextError) -> pliron::result::Error {
    pliron::input_error_noloc!(error)
}

fn print_operation(
    ctx: &Context,
    op: Ptr<Operation>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let concrete = Operation::get_op_dyn(op, ctx);
    concrete.verify(ctx)?;
    let raw = op.deref(ctx);
    super::location::validate(ctx, raw.loc())?;
    let native_name = concrete.get_opid().to_string();
    if concrete.is::<CompositeOp>() && raw.regions().next().is_some() {
        return Err(invalid("MLIR composite operations cannot contain regions"));
    }
    if matches!(
        native_name.as_str(),
        "stablehlo.infeed" | "stablehlo.outfeed" | "stablehlo.send" | "stablehlo.recv"
    ) && raw.operands().chain(raw.results()).any(|value| {
        value
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<TupleType>()
            .is_some()
    }) {
        return Err(invalid(
            "MLIR communication operations require flat tensor/token values, not native tuple-valued transfers",
        ));
    }
    let mut name = native_name.as_str();
    let mut attrs = raw.attributes.clone();
    if let Some(constant) = concrete.downcast_ref::<ConstantOp>() {
        let value = module::constant_attribute(ctx, constant, raw.get_result(0).get_type(ctx))
            .map_err(assembly_error)?;
        for key in [
            "stablehlo_constant_bits",
            "stablehlo_constant_splat",
            "stablehlo_constant_sparse",
        ] {
            attrs.0.remove(&Identifier::new(key));
        }
        insert(&mut attrs, "value", value)?;
    } else if concrete.is::<ReturnOp>() {
        if raw
            .get_parent_block()
            .and_then(|block| block.deref(ctx).get_parent_region())
            .is_some_and(|region| {
                Operation::get_op_dyn(region.deref(ctx).get_parent_op(), ctx).is::<FuncOp>()
            })
        {
            name = "func.return";
        }
    } else if let Some(call) = concrete.downcast_ref::<CallOp>() {
        name = "func.call";
        let callee = call
            .get_attr_stablehlo_call_callee(ctx)
            .ok_or_else(|| invalid("missing callee"))?;
        attrs.0.remove(&Identifier::new("stablehlo_call_callee"));
        insert(
            &mut attrs,
            "callee",
            Box::new(SymbolRefAttr {
                path: vec![callee.clone()],
            }),
        )?;
    } else {
        module_attributes::source_attributes(ctx, &native_name, &mut attrs)
            .map_err(assembly_error)?;
        if concrete.is::<CrossReplicaSumOp>() {
            name = "stablehlo.cross-replica-sum";
        }
    }
    let mut generic = GenericOp::new(op, name);
    let mut entries = attrs.0.iter().collect::<Vec<_>>();
    entries.sort_by(|(a, _), (b, _)| a.as_ref().cmp(b.as_ref()));
    for (key, attr) in entries {
        if *key == DISCARDABLE_ATTRIBUTES {
            let dict = attr
                .downcast_ref::<NamedAttributesAttr>()
                .ok_or_else(|| invalid("invalid discardable dictionary"))?;
            dict.validate_names()?;
            for (key, value) in dict.entries() {
                if !key.contains('.') || attrs.0.keys().any(|name| name.as_ref() == key) {
                    return Err(invalid("invalid or colliding discardable attribute"));
                }
                generic = generic.attr_raw(key, printer::mlir_attribute(ctx, &**value)?);
            }
        } else {
            if !module_attributes::is_source_field(&native_name, key.as_ref()) {
                return Err(invalid(format!(
                    "unmapped attribute {key} on {native_name}"
                )));
            }
            generic = generic.prop_raw(key.as_ref(), printer::mlir_attribute(ctx, &**attr)?);
        }
    }
    generic.print(ctx, state, f)
}

fn insert(attrs: &mut AttributeDict, name: &'static str, value: AttrObj) -> Result<()> {
    if attrs.0.insert(Identifier::new(name), value).is_some() {
        return Err(invalid(format!("attribute collision: {name}")));
    }
    Ok(())
}

fn integers(
    attrs: &mut AttributeDict,
    key: &'static str,
    count: usize,
) -> Result<Option<Vec<u64>>> {
    let Some(value) = attrs.0.remove(&Identifier::new(key)) else {
        return Ok(None);
    };
    let array = value
        .downcast_ref::<VecAttr>()
        .ok_or_else(|| invalid(format!("invalid {key}")))?;
    let values = array
        .0
        .iter()
        .map(|value| {
            value
                .downcast_ref::<IntegerAttr>()
                .ok_or_else(|| invalid(format!("invalid integer in {key}")))?
                .value()
                .to_string_decimal(false)
                .parse::<u64>()
                .map_err(|error| invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() != count {
        return Err(invalid(format!("expected {count} fields in {key}")));
    }
    Ok(Some(values))
}

pub(super) fn unpack_attributes(
    ctx: &Context,
    name: &str,
    attrs: &mut AttributeDict,
) -> std::result::Result<(), super::TextError> {
    unpack(ctx, name, attrs).map_err(|error| super::TextError {
        offset: 0,
        kind: super::TextErrorKind::InvalidAttribute(error.to_string()),
    })
}

fn unpack(ctx: &Context, name: &str, attrs: &mut AttributeDict) -> Result<()> {
    let scalar = |bits| {
        Box::new(ScalarBitsAttr {
            ty: ElementType::I64.native(ctx),
            words: vec![bits],
        }) as AttrObj
    };
    for key in [
        "stablehlo_compute_conv_groups",
        "stablehlo_compute_dynamic_conv_groups",
    ] {
        if let Some(values) = integers(attrs, key, 2)? {
            insert(attrs, "feature_group_count", scalar(values[0]))?;
            insert(attrs, "batch_group_count", scalar(values[1]))?;
        }
    }
    for key in [
        "stablehlo_compute_conv_reversal",
        "stablehlo_compute_dynamic_conv_reversal",
    ] {
        if let Some(value) = attrs.0.remove(&Identifier::new(key)) {
            let array = value
                .downcast_ref::<VecAttr>()
                .ok_or_else(|| invalid("invalid window reversal"))?;
            let values = array
                .0
                .iter()
                .map(|value| {
                    let value = value
                        .downcast_ref::<IntegerAttr>()
                        .ok_or_else(|| invalid("invalid reversal flag"))?;
                    let value = value.value().to_string_decimal(false);
                    match value.as_str() {
                        "0" | "1" => Ok(Box::new(BoolAttr::new(value == "1")) as AttrObj),
                        _ => Err(invalid("invalid reversal flag")),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            insert(
                attrs,
                "window_reversal",
                Box::new(DenseArrayAttr {
                    element: ElementType::I1.native(ctx),
                    values: VecAttr::new(values),
                }),
            )?;
        }
    }
    for key in [
        "stablehlo_compute_bn_options",
        "stablehlo_compute_bn_training_options",
        "stablehlo_compute_bn_grad_options",
    ] {
        if let Some(values) = integers(attrs, key, 2)? {
            insert(attrs, "feature_index", scalar(values[0]))?;
            let epsilon = f64::from_bits(values[1]) as f32;
            insert(
                attrs,
                "epsilon",
                Box::new(ScalarBitsAttr {
                    ty: ElementType::F32.native(ctx),
                    words: vec![u64::from(epsilon.to_bits())],
                }),
            )?;
        }
    }
    if let Some(values) = integers(attrs, "stablehlo_compute_triangular_flags", 4)? {
        for (index, key) in ["left_side", "lower", "unit_diagonal"].iter().enumerate() {
            if values[index] > 1 {
                return Err(invalid("invalid triangular solve flag"));
            }
            insert(attrs, key, Box::new(BoolAttr::new(values[index] != 0)))?;
        }
        let mode = match values[3] {
            0 => "NO_TRANSPOSE",
            1 => "TRANSPOSE",
            2 => "ADJOINT",
            _ => return Err(invalid("invalid transpose mode")),
        };
        insert(
            attrs,
            "transpose_a",
            Box::new(OpaqueAttr {
                dialect: StringAttr::new("stablehlo".into()),
                syntax: StringAttr::new(format!("transpose {mode}")),
            }),
        )?;
    }
    if name == "stablehlo.composite" {
        let key = Identifier::new("stablehlo_composite_decomposition");
        if let Some(value) = attrs.0.get_mut(&key) {
            let symbol = value
                .downcast_ref::<StringAttr>()
                .ok_or_else(|| invalid("invalid decomposition symbol"))?;
            *value = Box::new(SymbolRefAttr {
                path: vec![symbol.clone()],
            });
        }
    }
    if name == "stablehlo.all_to_all"
        && let Some(flag) = attrs.0.remove(&Identifier::new(
            "stablehlo_all_to_all_use_global_device_ids",
        ))
        && flag
            .downcast_ref::<BoolAttr>()
            .is_none_or(|flag| bool::from(flag.clone()))
    {
        return Err(invalid(
            "all_to_all has no MLIR use_global_device_ids field",
        ));
    }
    Ok(())
}

#[op_interface_impl]
impl ToMlirOp for ModuleOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        self.verify(ctx)?;
        let raw = self.get_operation().deref(ctx);
        super::location::validate(ctx, raw.loc())?;
        module_attributes::check_source_fields("builtin.module", &raw.attributes)
            .map_err(assembly_error)?;
        write!(f, "module")?;
        if !raw.attributes.0.is_empty() {
            write!(
                f,
                " attributes {}",
                module::operation_dictionary_text(ctx, &raw.attributes).map_err(assembly_error)?
            )?;
        }
        writeln!(f, " {{")?;
        let block = self
            .body_block(ctx)
            .ok_or_else(|| invalid("missing module body"))?;
        for op in block.deref(ctx).iter(ctx) {
            print_op(ctx, op, state, f)?;
            writeln!(f)?;
        }
        write!(f, "}}")?;
        print_op_location(ctx, raw.loc(), state, f)
    }
}

#[op_interface_impl]
impl ToMlirOp for FuncOp {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        self.verify(ctx)?;
        let raw = self.get_operation().deref(ctx);
        super::location::validate(ctx, raw.loc())?;
        let signature = self
            .signature(ctx)
            .ok_or_else(|| invalid("missing function signature"))?;
        let signature = signature.deref(ctx);
        let signature = signature
            .downcast_ref::<FunctionType>()
            .ok_or_else(|| invalid("invalid function signature"))?;
        let name = self
            .name(ctx)
            .ok_or_else(|| invalid("missing function name"))?;
        write!(f, "func.func @{}(", module::symbol(&name))?;
        for (index, arg) in self.arguments(ctx).iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "%{}: ", arg.unique_name(ctx))?;
            print_type(ctx, arg.get_type(ctx), state, f)?;
        }

        write!(f, ") -> (")?;
        pliron_mlir_interop::printers::print_type_list(
            ctx,
            signature.outputs.iter().copied(),
            state,
            f,
        )?;
        write!(f, ")")?;
        let mut attrs = raw.attributes.clone();
        attrs.0.remove(&Identifier::new("stablehlo_function_name"));
        attrs
            .0
            .remove(&Identifier::new("stablehlo_function_signature"));
        module_attributes::check_source_fields("func.func", &attrs).map_err(assembly_error)?;
        if !attrs.0.is_empty() {
            write!(
                f,
                " attributes {}",
                module::operation_dictionary_text(ctx, &attrs).map_err(assembly_error)?
            )?;
        }
        writeln!(f, " {{")?;
        let blocks = raw.get_region(0).deref(ctx).iter(ctx).collect::<Vec<_>>();
        for (index, block) in blocks.iter().enumerate() {
            if index > 0 {
                write!(f, "^{}(", block.deref(ctx).unique_name(ctx))?;
                for (index, arg) in block.deref(ctx).arguments().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}: ", arg.unique_name(ctx))?;
                    print_type(ctx, arg.get_type(ctx), state, f)?;
                }
                writeln!(f, "):")?;
            }
            for op in block.deref(ctx).iter(ctx) {
                print_op(ctx, op, state, f)?;
                writeln!(f)?;
            }
        }
        write!(f, "}}")?;
        print_op_location(ctx, raw.loc(), state, f)
    }
}
