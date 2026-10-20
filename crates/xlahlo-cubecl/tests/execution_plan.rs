use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use cubecl_core::compute::KernelBuilder;
use cubecl_ir::{
    AddressType, Builtin, ElemType,
    attributes::{FloatAttr, IndexAttr},
    dialect::{
        branch::{IfOp, YieldOp},
        cmp::{IEqualOp, UGreaterThanOp, ULessThanOp},
        general::{BuiltinAttr, CastOp, ReadBuiltinOp, SelectOp},
        math::{IAddOp, IMulOp, ISubOp, UDivOp, URemOp},
        memory::{IndexOp, LoadOp, StoreOp},
    },
    settings::{Dim3, ExecutionMode, KernelSettings},
};
use cubecl_runtime::{
    id::KernelId,
    kernel::{KernelDefinition, KernelMetadata},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IntegerAttr,
        ops::{ConstantOp as CubeConstantOp, ModuleOp},
        types::FunctionType,
    },
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::match_rewrite::{MatchRewrite, MatchRewriter, RewriterOrder, apply_match_rewrite},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    value::Value,
};
use xlahlo::{
    attributes::{
        Algorithm, Axes, DotDimensionNumbers, FusionKind, GatherDimensionNumbers, Index,
        IndexFlags, OperandPrecision, Precision, PrecisionConfig, Slice,
    },
    ops::{
        ComputationOp, ReturnOp,
        control::FusionOp,
        control::{ConstantOp, ParameterOp, TupleOp},
        elementwise::{AddOp, MinimumOp, MultiplyOp},
        movement::{BroadcastOp, GatherOp, ReshapeOp, ReverseOp, SliceOp, TransposeOp},
        numerical::DotOp,
    },
    shape::{Layout, Shape},
    types::ElementType,
};
use xlahlo_cubecl::{
    BufferAccess, BufferDescriptor, BufferRole, CubeclTargetPass, IndexTransformKind,
    KernelBinding, LaunchGeometry, ModuleCompiler, OperationLoweringRequest,
    OperationLoweringStrategy, OutputIndexMap, PlannedKernel, PlannedOutput,
};

// Evaluate the scalar expression feeding a generated store, not a device runtime.
fn evaluate_materialization(
    context: &Context,
    value: Value,
    position: usize,
    source: &[u128],
) -> u128 {
    let operation = value.defining_op().expect("materialization expression");
    if let Some(constant) = Operation::get_op::<CubeConstantOp>(operation, context) {
        let attr = constant
            .get_attr_builtin_constant_value(context)
            .expect("materialization constant value");
        if let Some(index) = attr.downcast_ref::<IndexAttr>() {
            return index.0 as u128;
        }
        if let Some(float) = attr.downcast_ref::<FloatAttr>() {
            return float.val.to_bits();
        }
        return attr
            .downcast_ref::<IntegerAttr>()
            .expect("supported materialization constant")
            .value()
            .to_u64() as u128;
    }
    if Operation::is_op::<ReadBuiltinOp>(operation, context) {
        return position as u128;
    }
    let operand = |index| {
        evaluate_materialization(
            context,
            operation.deref(context).get_operand(index),
            position,
            source,
        )
    };
    if Operation::is_op::<CastOp>(operation, context) {
        return operand(0);
    }
    if Operation::is_op::<SelectOp>(operation, context) {
        return if operand(0) != 0 {
            operand(1)
        } else {
            operand(2)
        };
    }
    if Operation::is_op::<LoadOp>(operation, context) {
        let pointer = operation
            .deref(context)
            .get_operand(0)
            .defining_op()
            .unwrap();
        assert!(Operation::is_op::<IndexOp>(pointer, context));
        let index = evaluate_materialization(
            context,
            pointer.deref(context).get_operand(1),
            position,
            source,
        );
        return source[usize::try_from(index).unwrap()];
    }
    if Operation::is_op::<IAddOp>(operation, context) {
        operand(0) + operand(1)
    } else if Operation::is_op::<IMulOp>(operation, context) {
        operand(0) * operand(1)
    } else if Operation::is_op::<ISubOp>(operation, context) {
        operand(0) - operand(1)
    } else if Operation::is_op::<UDivOp>(operation, context) {
        operand(0) / operand(1)
    } else if Operation::is_op::<URemOp>(operation, context) {
        operand(0) % operand(1)
    } else if Operation::is_op::<IEqualOp>(operation, context) {
        u128::from(operand(0) == operand(1))
    } else if Operation::is_op::<ULessThanOp>(operation, context) {
        u128::from(operand(0) < operand(1))
    } else if Operation::is_op::<UGreaterThanOp>(operation, context) {
        u128::from(operand(0) > operand(1))
    } else {
        panic!("unexpected materialization operation");
    }
}

fn assert_materialized_values(
    kernel: &xlahlo_cubecl::KernelStep,
    source: &[u128],
    expected: &[u128],
) {
    let definition = kernel.define_checked().unwrap();
    let context = definition.body.ctx();
    let mut pending = vec![definition.body.state().module.get_operation()];
    let mut stores = Vec::new();
    let mut guards = Vec::new();
    while let Some(operation) = pending.pop() {
        if Operation::is_op::<StoreOp>(operation, context) {
            stores.push(operation);
        }
        if Operation::is_op::<IfOp>(operation, context) {
            guards.push(operation);
        }
        for region in operation.deref(context).regions() {
            for block in region.deref(context).iter(context) {
                pending.extend(block.deref(context).iter(context));
            }
        }
    }
    let [store] = stores.as_slice() else {
        panic!("expected one materialization store");
    };
    let [guard] = guards.as_slice() else {
        panic!("expected one materialization bounds guard");
    };
    let condition = guard.deref(context).get_operand(0);
    assert_eq!(
        evaluate_materialization(context, condition, expected.len(), source),
        0
    );
    assert_eq!(
        evaluate_materialization(context, condition, expected.len() + 256, source),
        0
    );
    let pointer = store.deref(context).get_operand(0).defining_op().unwrap();
    assert!(Operation::is_op::<IndexOp>(pointer, context));
    for (position, &bits) in expected.iter().enumerate() {
        assert_eq!(
            evaluate_materialization(context, condition, position, source),
            1
        );
        assert_eq!(
            evaluate_materialization(
                context,
                pointer.deref(context).get_operand(1),
                position,
                source,
            ),
            position as u128
        );
        assert_eq!(
            evaluate_materialization(
                context,
                store.deref(context).get_operand(1),
                position,
                source,
            ),
            bits,
            "materialized element {position}"
        );
    }
}

#[derive(Clone, Copy)]
struct ExternalGemmKernel;

impl KernelMetadata for ExternalGemmKernel {
    fn name(&self) -> &'static str {
        "external_gemm"
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }

    fn address_type(&self) -> ElemType {
        AddressType::U32.unsigned_type()
    }
}

impl PlannedKernel for ExternalGemmKernel {
    fn define_checked(&self) -> Result<KernelDefinition, String> {
        let settings = KernelSettings::new(
            Dim3::new_1d(128),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
        .kernel_name(self.name());
        let mut builder = KernelBuilder::new(settings);
        let buffer_type = ElemType::Float(cubecl_ir::FloatKind::F32).to_type(builder.ctx());
        for _ in 0..3 {
            builder.buffer(buffer_type);
        }
        builder.register(&cubecl_ir::dialect::branch::ReturnOp::new(
            builder.ctx_mut(),
        ));
        Ok(builder.build())
    }
}

struct CompositionalAddStrategy;

impl OperationLoweringStrategy for CompositionalAddStrategy {
    fn name(&self) -> &'static str {
        "test-add"
    }

    fn priority(&self) -> i32 {
        0
    }

    fn matches(&self, context: &Context, operation: Ptr<Operation>) -> Result<bool, String> {
        Ok(Operation::get_op::<AddOp>(operation, context).is_some())
    }

    fn lower(&self, request: OperationLoweringRequest<'_>) -> Result<(), xlahlo_cubecl::PlanError> {
        let elements = request.result_descriptors[0].num_elements()?;
        request.plan.add_kernel(
            "test-add",
            Arc::new(ExternalGemmKernel),
            vec![
                KernelBinding {
                    buffer: request.operands[0],
                    access: BufferAccess::Read,
                },
                KernelBinding {
                    buffer: request.operands[1],
                    access: BufferAccess::Read,
                },
                KernelBinding {
                    buffer: request.results[0],
                    access: BufferAccess::Write,
                },
            ],
            LaunchGeometry::for_elements(elements, [128, 1, 1])?,
            false,
        )
    }
}

struct NoopRewrite;

impl MatchRewrite for NoopRewrite {
    fn r#match(&mut self, _context: &Context, _op: Ptr<Operation>) -> bool {
        false
    }

    fn rewrite(
        &mut self,
        _context: &mut Context,
        _rewriter: &mut MatchRewriter,
        _op: Ptr<Operation>,
    ) -> pliron::result::Result<()> {
        Ok(())
    }
}

struct PlironPass(Arc<AtomicUsize>);

impl CubeclTargetPass for PlironPass {
    fn name(&self) -> &'static str {
        "integration-noop-rewrite"
    }

    fn priority(&self) -> i32 {
        10
    }

    fn cache_key(&self) -> Vec<u8> {
        b"integration-noop-rewrite-v1".to_vec()
    }

    fn run(&self, context: &mut Context, root: Ptr<Operation>) -> Result<(), String> {
        apply_match_rewrite(context, &mut NoopRewrite, RewriterOrder::default(), root)
            .map_err(|error| error.to_string())?;
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn dot_module(context: &mut Context) -> Ptr<Operation> {
    xlahlo::register(context);
    let lhs_type = Shape::array(context, ElementType::F32, vec![2, 4]).unwrap();
    let rhs_type = Shape::array(context, ElementType::F32, vec![4, 3]).unwrap();
    let output_type = Shape::array(context, ElementType::F32, vec![2, 3]).unwrap();
    let signature = FunctionType::get(context, vec![lhs_type, rhs_type], vec![output_type]).into();
    let module = ModuleOp::new(context, Identifier::new("custom_gemm"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let lhs = ParameterOp::new(context, lhs_type, vec![], Index(0));
    let rhs = ParameterOp::new(context, rhs_type, vec![], Index(1));
    lhs.get_operation().insert_at_back(block, context);
    rhs.get_operation().insert_at_back(block, context);
    let dimensions = DotDimensionNumbers {
        lhs_contracting_dimensions: Axes(vec![1]),
        rhs_contracting_dimensions: Axes(vec![0]),
        lhs_batch_dimensions: Axes(vec![]),
        rhs_batch_dimensions: Axes(vec![]),
    };
    let precision = PrecisionConfig {
        operand_precision: OperandPrecision(vec![Precision::Default, Precision::Default]),
        algorithm: Algorithm::Unset,
    };
    let dot = DotOp::new(
        context,
        output_type,
        vec![lhs.result(context), rhs.result(context)],
        dimensions,
        precision,
    );
    dot.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, dot.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn dot_then_add_module(context: &mut Context) -> Ptr<Operation> {
    xlahlo::register(context);
    let lhs_type = Shape::array(context, ElementType::F32, vec![2, 4]).unwrap();
    let rhs_type = Shape::array(context, ElementType::F32, vec![4, 3]).unwrap();
    let output_type = Shape::array(context, ElementType::F32, vec![2, 3]).unwrap();
    let signature = FunctionType::get(
        context,
        vec![lhs_type, rhs_type, output_type],
        vec![output_type],
    )
    .into();
    let module = ModuleOp::new(context, Identifier::new("custom_gemm_composed"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let lhs = ParameterOp::new(context, lhs_type, vec![], Index(0));
    let rhs = ParameterOp::new(context, rhs_type, vec![], Index(1));
    let addend = ParameterOp::new(context, output_type, vec![], Index(2));
    lhs.get_operation().insert_at_back(block, context);
    rhs.get_operation().insert_at_back(block, context);
    addend.get_operation().insert_at_back(block, context);
    let dimensions = DotDimensionNumbers {
        lhs_contracting_dimensions: Axes(vec![1]),
        rhs_contracting_dimensions: Axes(vec![0]),
        lhs_batch_dimensions: Axes(vec![]),
        rhs_batch_dimensions: Axes(vec![]),
    };
    let precision = PrecisionConfig {
        operand_precision: OperandPrecision(vec![Precision::Default, Precision::Default]),
        algorithm: Algorithm::Unset,
    };
    let dot = DotOp::new(
        context,
        output_type,
        vec![lhs.result(context), rhs.result(context)],
        dimensions,
        precision,
    );
    dot.get_operation().insert_at_back(block, context);
    let add = AddOp::new(
        context,
        output_type,
        vec![dot.result(context), addend.result(context)],
    );
    add.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, add.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

struct TunedGemmStrategy;

impl OperationLoweringStrategy for TunedGemmStrategy {
    fn name(&self) -> &'static str {
        "integration-tuned-gemm"
    }

    fn priority(&self) -> i32 {
        100
    }

    fn matches(&self, context: &Context, operation: Ptr<Operation>) -> Result<bool, String> {
        Ok(Operation::get_op::<DotOp>(operation, context).is_some())
    }

    fn lower(&self, request: OperationLoweringRequest<'_>) -> Result<(), xlahlo_cubecl::PlanError> {
        let output_elements = request.result_descriptors[0].num_elements()?;
        let bindings = request
            .operands
            .iter()
            .copied()
            .map(|buffer| KernelBinding {
                buffer,
                access: BufferAccess::Read,
            })
            .chain(request.results.iter().copied().map(|buffer| KernelBinding {
                buffer,
                access: BufferAccess::Write,
            }))
            .collect();
        request.plan.add_kernel(
            "tuned-gemm",
            Arc::new(ExternalGemmKernel),
            bindings,
            LaunchGeometry::for_elements(output_elements, [128, 1, 1])?,
            false,
        )
    }
}

fn add_module(context: &mut Context) -> Ptr<Operation> {
    typed_add_module(context, ElementType::F32)
}

fn typed_add_module(context: &mut Context, element_type: ElementType) -> Ptr<Operation> {
    xlahlo::register(context);
    let input_type = Shape::array(context, element_type, vec![8]).unwrap();
    let signature =
        FunctionType::get(context, vec![input_type, input_type], vec![input_type]).into();
    let module = ModuleOp::new(context, Identifier::new("default_add"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let lhs = ParameterOp::new(context, input_type, vec![], Index(0));
    let rhs = ParameterOp::new(context, input_type, vec![], Index(1));
    lhs.get_operation().insert_at_back(block, context);
    rhs.get_operation().insert_at_back(block, context);
    let add = AddOp::new(
        context,
        input_type,
        vec![lhs.result(context), rhs.result(context)],
    );
    add.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, add.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn identity_module(context: &mut Context) -> Ptr<Operation> {
    xlahlo::register(context);
    let input_type = Shape::array(context, ElementType::F32, vec![8]).unwrap();
    let signature = FunctionType::get(context, vec![input_type], vec![input_type]).into();
    let module = ModuleOp::new(context, Identifier::new("default_identity"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let input = ParameterOp::new(context, input_type, vec![], Index(0));
    input.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, input.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn tuple_output_module(context: &mut Context, compute_first: bool) -> Ptr<Operation> {
    xlahlo::register(context);
    let input_type = Shape::array(context, ElementType::F32, vec![8]).unwrap();
    let tuple_type = Shape::tuple(context, vec![input_type, input_type]).unwrap();
    let signature =
        FunctionType::get(context, vec![input_type, input_type], vec![tuple_type]).into();
    let module = ModuleOp::new(context, Identifier::new("tuple_outputs"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let lhs = ParameterOp::new(context, input_type, vec![], Index(0));
    let rhs = ParameterOp::new(context, input_type, vec![], Index(1));
    lhs.get_operation().insert_at_back(block, context);
    rhs.get_operation().insert_at_back(block, context);
    let first = if compute_first {
        let add = AddOp::new(
            context,
            input_type,
            vec![lhs.result(context), rhs.result(context)],
        );
        add.get_operation().insert_at_back(block, context);
        add.result(context)
    } else {
        lhs.result(context)
    };
    let tuple = TupleOp::new(context, tuple_type, vec![first, rhs.result(context)]);
    tuple.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, tuple.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn unit_result_module(context: &mut Context) -> Ptr<Operation> {
    xlahlo::register(context);
    let unit_type = Shape::tuple(context, vec![]).unwrap();
    let signature = FunctionType::get(context, vec![], vec![unit_type]).into();
    let module = ModuleOp::new(context, Identifier::new("unit_result"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let tuple = TupleOp::new(context, unit_type, vec![]);
    tuple.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, tuple.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn constant_output_module(context: &mut Context) -> Ptr<Operation> {
    xlahlo::register(context);
    let output_type = Shape::array(context, ElementType::F32, vec![3]).unwrap();
    let signature = FunctionType::get(context, vec![], vec![output_type]).into();
    let module = ModuleOp::new(context, Identifier::new("constant_output"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let literal =
        xlahlo::literal::Literal::from_bits(context, output_type, &[0x3f80_0000, 0, 0x4000_0000])
            .unwrap();
    let constant = ConstantOp::new(context, output_type, vec![], literal);
    constant.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, constant.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn scalar_broadcast_add_module(context: &mut Context, constant: bool) -> Ptr<Operation> {
    xlahlo::register(context);
    let scalar_type = Shape::scalar(context, ElementType::F32).unwrap();
    let output_type = Shape::array(context, ElementType::F32, vec![8]).unwrap();
    let parameter_types = if constant {
        vec![output_type]
    } else {
        vec![scalar_type, output_type]
    };
    let signature = FunctionType::get(context, parameter_types, vec![output_type]).into();
    let module = ModuleOp::new(context, Identifier::new("scalar_broadcast_add"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let scalar = if constant {
        let literal =
            xlahlo::literal::Literal::from_bits(context, scalar_type, &[0x3f80_0000]).unwrap();
        let scalar = ConstantOp::new(context, scalar_type, vec![], literal);
        scalar.get_operation().insert_at_back(block, context);
        scalar.result(context)
    } else {
        let scalar = ParameterOp::new(context, scalar_type, vec![], Index(0));
        scalar.get_operation().insert_at_back(block, context);
        scalar.result(context)
    };
    let input_index = if constant { 0 } else { 1 };
    let input = ParameterOp::new(context, output_type, vec![], Index(input_index as u32));
    input.get_operation().insert_at_back(block, context);
    let broadcast = BroadcastOp::new(context, output_type, vec![scalar], Axes(vec![]));
    broadcast.get_operation().insert_at_back(block, context);
    let add = AddOp::new(
        context,
        output_type,
        vec![broadcast.result(context), input.result(context)],
    );
    add.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, add.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn broadcast_add_module(
    context: &mut Context,
    source_shape: Vec<i64>,
    result_shape: Vec<i64>,
    dimensions: Vec<i64>,
    constant_bits: Option<Vec<u128>>,
) -> Ptr<Operation> {
    xlahlo::register(context);
    let source_type = Shape::array(context, ElementType::F32, source_shape).unwrap();
    let result_type = Shape::array(context, ElementType::F32, result_shape).unwrap();
    broadcast_typed_module(
        context,
        source_type,
        result_type,
        dimensions,
        constant_bits,
        true,
    )
}

fn broadcast_typed_module(
    context: &mut Context,
    source_type: pliron::r#type::TypeHandle,
    result_type: pliron::r#type::TypeHandle,
    dimensions: Vec<i64>,
    constant_bits: Option<Vec<u128>>,
    add: bool,
) -> Ptr<Operation> {
    let has_constant = constant_bits.is_some();
    let mut parameter_types = Vec::new();
    if constant_bits.is_none() {
        parameter_types.push(source_type);
    }
    if add {
        parameter_types.push(result_type);
    }
    let signature = FunctionType::get(context, parameter_types, vec![result_type]).into();
    let module = ModuleOp::new(context, Identifier::new("broadcast_add"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let source = if let Some(bits) = constant_bits {
        let literal = xlahlo::literal::Literal::from_bits(context, source_type, &bits).unwrap();
        let constant = ConstantOp::new(context, source_type, vec![], literal);
        constant.get_operation().insert_at_back(block, context);
        constant.result(context)
    } else {
        let source = ParameterOp::new(context, source_type, vec![], Index(0));
        source.get_operation().insert_at_back(block, context);
        source.result(context)
    };
    let addend = if add {
        let addend_index = if has_constant { 0 } else { 1 };
        let addend = ParameterOp::new(context, result_type, vec![], Index(addend_index));
        addend.get_operation().insert_at_back(block, context);
        Some(addend.result(context))
    } else {
        None
    };
    let broadcast = BroadcastOp::new(context, result_type, vec![source], Axes(dimensions));
    broadcast.get_operation().insert_at_back(block, context);
    let result = if let Some(addend) = addend {
        let add = AddOp::new(
            context,
            result_type,
            vec![broadcast.result(context), addend],
        );
        add.get_operation().insert_at_back(block, context);
        add.result(context)
    } else {
        broadcast.result(context)
    };
    ReturnOp::new(context, result)
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn vector_constant_add_module(context: &mut Context, bits: &[u128]) -> Ptr<Operation> {
    xlahlo::register(context);
    let input_type = Shape::array(context, ElementType::F32, vec![2, 2]).unwrap();
    let signature = FunctionType::get(context, vec![input_type], vec![input_type]).into();
    let module = ModuleOp::new(context, Identifier::new("vector_constant_add"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let input = ParameterOp::new(context, input_type, vec![], Index(0));
    input.get_operation().insert_at_back(block, context);
    let literal = xlahlo::literal::Literal::from_bits(context, input_type, bits).unwrap();
    let constant = ConstantOp::new(context, input_type, vec![], literal);
    constant.get_operation().insert_at_back(block, context);
    let add = AddOp::new(
        context,
        input_type,
        vec![input.result(context), constant.result(context)],
    );
    add.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, add.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn low_precision_constant_add_module(context: &mut Context, uses_constant: bool) -> Ptr<Operation> {
    xlahlo::register(context);
    let scalar_type = Shape::scalar(context, ElementType::F16).unwrap();
    let output_type = Shape::array(context, ElementType::F16, vec![8]).unwrap();
    let signature = FunctionType::get(context, vec![output_type], vec![output_type]).into();
    let module = ModuleOp::new(context, Identifier::new("low_precision_constant_add"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let input = ParameterOp::new(context, output_type, vec![], Index(0));
    input.get_operation().insert_at_back(block, context);
    let literal = xlahlo::literal::Literal::from_bits(context, scalar_type, &[0]).unwrap();
    let constant = ConstantOp::new(context, scalar_type, vec![], literal);
    constant.get_operation().insert_at_back(block, context);
    let rhs = if uses_constant {
        let broadcast = BroadcastOp::new(
            context,
            output_type,
            vec![constant.result(context)],
            Axes(vec![]),
        );
        broadcast.get_operation().insert_at_back(block, context);
        broadcast.result(context)
    } else {
        input.result(context)
    };
    let add = AddOp::new(context, output_type, vec![input.result(context), rhs]);
    add.get_operation().insert_at_back(block, context);
    ReturnOp::new(context, add.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn fused_arithmetic_module(context: &mut Context) -> Ptr<Operation> {
    xlahlo::register(context);
    let input_type = Shape::array(context, ElementType::F32, vec![8]).unwrap();
    let signature =
        FunctionType::get(context, vec![input_type, input_type], vec![input_type]).into();
    let module = ModuleOp::new(context, Identifier::new("fused_arithmetic"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let lhs = ParameterOp::new(context, input_type, vec![], Index(0));
    let rhs = ParameterOp::new(context, input_type, vec![], Index(1));
    lhs.get_operation().insert_at_back(block, context);
    rhs.get_operation().insert_at_back(block, context);
    let fusion = FusionOp::new(
        context,
        input_type,
        vec![lhs.result(context), rhs.result(context)],
        FusionKind::Loop,
    );
    fusion.get_operation().insert_at_back(block, context);
    let body = BasicBlock::new(context, None, vec![input_type, input_type]);
    body.insert_at_back(fusion.body(context), context);
    let arguments: Vec<_> = body.deref(context).arguments().collect();
    let add = AddOp::new(context, input_type, vec![arguments[0], arguments[1]]);
    add.get_operation().insert_at_back(body, context);
    let multiply = MultiplyOp::new(context, input_type, vec![add.result(context), arguments[1]]);
    multiply.get_operation().insert_at_back(body, context);
    ReturnOp::new(context, multiply.result(context))
        .get_operation()
        .insert_at_back(body, context);
    ReturnOp::new(context, fusion.result(context))
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

fn contains_dot(context: &Context, root: Ptr<Operation>) -> Result<bool, String> {
    let mut pending = vec![root];
    while let Some(op) = pending.pop() {
        if Operation::get_op::<DotOp>(op, context).is_some() {
            return Ok(true);
        }
        let ir = op.deref(context);
        for region in ir.regions() {
            for block in region.deref(context).iter(context) {
                pending.extend(block.deref(context).iter(context));
            }
        }
    }
    Ok(false)
}

#[test]
fn external_gemm_strategy_builds_a_verified_plan_and_runs_pliron_passes() {
    let mut context = Context::new();
    let module = dot_module(&mut context);
    xlahlo::verification::verify(&context, module).unwrap();

    let fallback_invoked = Arc::new(AtomicBool::new(false));
    let pass_runs = Arc::new(AtomicUsize::new(0));
    let mut builder = ModuleCompiler::builder();
    let fallback = fallback_invoked.clone();
    builder
        .register_strategy(
            "generic-fallback",
            0,
            |_, _| Ok(true),
            move |_, _, _| {
                fallback.store(true, Ordering::SeqCst);
                Err(xlahlo_cubecl::PlanError::SizeOverflow)
            },
        )
        .unwrap();
    builder
        .register_strategy("external-gemm", 100, contains_dot, |_, _, plan| {
            let lhs = plan.add_buffer(
                BufferRole::Input(0),
                BufferDescriptor::dense_row_major(ElementType::F32, vec![2, 4]),
            );
            let rhs = plan.add_buffer(
                BufferRole::Input(1),
                BufferDescriptor::dense_row_major(ElementType::F32, vec![4, 3]),
            );
            let output = plan.add_buffer(
                BufferRole::Output(0),
                BufferDescriptor::dense_row_major(ElementType::F32, vec![2, 3]),
            );
            plan.add_kernel(
                "tuned-matmul",
                Arc::new(ExternalGemmKernel),
                vec![
                    KernelBinding {
                        buffer: lhs,
                        access: BufferAccess::Read,
                    },
                    KernelBinding {
                        buffer: rhs,
                        access: BufferAccess::Read,
                    },
                    KernelBinding {
                        buffer: output,
                        access: BufferAccess::Write,
                    },
                ],
                LaunchGeometry::for_elements(6, [128, 1, 1])?,
                false,
            )?;
            plan.add_output(PlannedOutput::Buffer(output));
            Ok(())
        })
        .unwrap();
    builder
        .register_target_pass(Arc::new(PlironPass(pass_runs.clone())))
        .unwrap();
    let compiler = builder.build();

    let plan = compiler.compile(&context, module).unwrap();
    assert!(!fallback_invoked.load(Ordering::SeqCst));
    assert_eq!(plan.kernels().len(), 1);
    assert_eq!(plan.kernels()[0].name(), "tuned-matmul");
    assert_eq!(plan.kernels()[0].bindings().len(), 3);
    assert_eq!(plan.kernels()[0].bindings()[0].buffer.index(), 0);
    assert_eq!(plan.kernels()[0].bindings()[1].buffer.index(), 1);
    assert_eq!(plan.kernels()[0].bindings()[2].buffer.index(), 2);
    assert_eq!(plan.kernels()[0].launch().cube_count, Some([1, 1, 1]));
    assert_eq!(
        plan.buffers()[0].lifetime,
        Some(xlahlo_cubecl::BufferLifetime {
            first_kernel: 0,
            last_kernel: 0,
            retained_for_output: false,
        })
    );
    assert!(matches!(
        plan.outputs(),
        [PlannedOutput::Buffer(output)] if output.index() == 2
    ));

    plan.kernels()[0].kernel().define();
    assert_eq!(pass_runs.load(Ordering::SeqCst), 2);
}

#[test]
fn operation_strategy_composes_tuned_gemm_with_generic_followup() {
    let mut context = Context::new();
    let module = dot_then_add_module(&mut context);
    let mut builder = ModuleCompiler::builder();
    builder
        .register_operation_strategy(Arc::new(TunedGemmStrategy))
        .unwrap();
    let plan = builder.build().compile(&context, module).unwrap();

    assert_eq!(plan.buffers().len(), 5);
    assert_eq!(plan.kernels().len(), 2);
    assert_eq!(plan.kernels()[0].name(), "tuned-gemm");
    assert_eq!(plan.kernels()[1].bindings().len(), 3);
    assert_eq!(plan.kernels()[0].bindings()[2].buffer.index(), 3);
    assert_eq!(plan.kernels()[1].bindings()[0].buffer.index(), 3);
    assert_eq!(plan.kernels()[1].bindings()[1].buffer.index(), 2);
    assert_eq!(plan.kernels()[1].bindings()[2].buffer.index(), 4);
    assert!(matches!(
        plan.outputs(),
        [PlannedOutput::Buffer(output)] if output.index() == 4
    ));
}

#[test]
fn default_compiler_lowers_elementwise_ir_to_a_checked_kernel() {
    let mut context = Context::new();
    let module = add_module(&mut context);
    let compiler = ModuleCompiler::builder().build();

    let plan = compiler.compile(&context, module).unwrap();

    assert_eq!(plan.buffers().len(), 3);
    assert_eq!(plan.kernels().len(), 1);
    assert_eq!(plan.kernels()[0].bindings().len(), 3);
    assert_eq!(plan.kernels()[0].launch().cube_count, Some([1, 1, 1]));
    assert!(matches!(
        plan.outputs(),
        [PlannedOutput::Buffer(output)] if output.index() == 2
    ));
    plan.kernels()[0].define_checked().unwrap();
}

#[test]
fn default_compiler_lowers_cubecl_float_integer_and_u1_arithmetic() {
    for element_type in [
        ElementType::F4E2M1FN,
        ElementType::F8E4M3FN,
        ElementType::F16,
        ElementType::F32,
        ElementType::F64,
        ElementType::S32,
        ElementType::U8,
        ElementType::U1,
    ] {
        let mut context = Context::new();
        let module = typed_add_module(&mut context, element_type);
        let compiler = ModuleCompiler::builder().build();

        let plan = compiler.compile(&context, module).unwrap();

        assert_eq!(plan.kernels().len(), 1, "element type {element_type:?}");
        plan.kernels()[0].define_checked().unwrap();
        match element_type {
            ElementType::F4E2M1FN => {
                assert_eq!(plan.buffers()[0].descriptor.byte_size().unwrap(), 8);
            }
            ElementType::U1 => {
                assert_eq!(plan.buffers()[0].descriptor.byte_size().unwrap(), 8);
            }
            _ => {}
        }
    }
}

#[test]
fn default_compiler_inlines_loop_fusion_regions_before_cubecl_lowering() {
    let mut context = Context::new();
    let module = fused_arithmetic_module(&mut context);
    let compiler = ModuleCompiler::builder().build();

    let plan = compiler.compile(&context, module).unwrap();

    assert_eq!(plan.kernels().len(), 1);
    plan.kernels()[0].define_checked().unwrap();
}

#[test]
fn default_compiler_returns_identity_outputs_as_aliases_without_dispatch() {
    let mut context = Context::new();
    let module = identity_module(&mut context);
    let compiler = ModuleCompiler::builder().build();

    let plan = compiler.compile(&context, module).unwrap();

    assert_eq!(plan.buffers().len(), 1);
    assert!(plan.kernels().is_empty());
    assert!(matches!(
        plan.outputs(),
        [PlannedOutput::Alias(input)] if input.index() == 0
    ));
    assert!(!plan.has_device_work());
}

#[test]
fn default_compiler_flattens_tuple_results_and_materializes_only_computed_outputs() {
    let mut context = Context::new();
    let module = tuple_output_module(&mut context, true);
    let compiler = ModuleCompiler::builder().build();

    let plan = compiler.compile(&context, module).unwrap();

    assert_eq!(plan.kernels().len(), 1);
    assert_eq!(plan.outputs().len(), 2);
    assert!(matches!(
        plan.outputs(),
        [PlannedOutput::Buffer(output), PlannedOutput::Alias(input)]
            if output.index() == 2 && input.index() == 1
    ));
    plan.kernels()[0].define_checked().unwrap();
}

#[test]
fn default_compiler_treats_empty_tuple_results_as_no_work() {
    let mut context = Context::new();
    let module = unit_result_module(&mut context);
    let compiler = ModuleCompiler::builder().build();

    let plan = compiler.compile(&context, module).unwrap();

    assert!(plan.buffers().is_empty());
    assert!(plan.kernels().is_empty());
    assert!(plan.outputs().is_empty());
    assert!(!plan.has_device_work());
}

#[test]
fn default_compiler_returns_constant_tensors_as_device_free_static_outputs() {
    let mut context = Context::new();
    let module = constant_output_module(&mut context);
    let plan = ModuleCompiler::builder()
        .build()
        .compile(&context, module)
        .unwrap();

    assert!(plan.buffers().is_empty());
    assert!(plan.kernels().is_empty());
    assert!(!plan.has_device_work());
    assert!(matches!(
        plan.outputs(),
        [PlannedOutput::Static(value)]
            if value.descriptor == BufferDescriptor::dense_row_major(ElementType::F32, vec![3])
                && value.bytes == [0, 0, 0x80, 0x3f, 0, 0, 0, 0, 0, 0, 0, 0x40]
    ));
}

#[test]
fn default_compiler_lowers_scalar_constant_and_parameter_broadcasts() {
    for constant in [true, false] {
        let mut context = Context::new();
        let module = scalar_broadcast_add_module(&mut context, constant);
        let plan = ModuleCompiler::builder()
            .build()
            .compile(&context, module)
            .unwrap();

        assert_eq!(plan.kernels().len(), 1);
        assert_eq!(
            plan.kernels()[0].bindings().len(),
            if constant { 2 } else { 3 }
        );
        plan.kernels()[0].define_checked().unwrap();
    }
}

#[test]
fn scalar_constant_dispatch_tracks_low_precision_arithmetic_use() {
    for (uses_constant, expected_kernel) in [(true, "xlahlo-module"), (false, "xlahlo-op-2")] {
        let mut context = Context::new();
        let module = low_precision_constant_add_module(&mut context, uses_constant);
        let plan = ModuleCompiler::builder()
            .build()
            .compile(&context, module)
            .unwrap();

        assert_eq!(plan.kernels().len(), 1);
        assert_eq!(plan.kernels()[0].name(), expected_kernel);
        plan.kernels()[0].define_checked().unwrap();
    }
}

#[test]
fn compositional_lowering_fuses_scalar_constant_broadcast_materialization() {
    let mut context = Context::new();
    let module = scalar_broadcast_add_module(&mut context, true);
    let mut builder = ModuleCompiler::builder();
    builder
        .register_operation_strategy(Arc::new(CompositionalAddStrategy))
        .unwrap();
    let plan = builder.build().compile(&context, module).unwrap();

    assert_eq!(
        plan.kernels()
            .iter()
            .map(|kernel| kernel.name())
            .collect::<Vec<_>>(),
        ["xlahlo-broadcast-2", "test-add"]
    );
    assert_eq!(plan.buffers().len(), 3);
    assert_eq!(plan.kernels()[0].bindings().len(), 1);
    assert_eq!(plan.kernels()[1].bindings().len(), 3);
    for kernel in plan.kernels() {
        kernel.define_checked().unwrap();
    }
}

#[test]
fn compositional_lowering_broadcasts_runtime_scalar_buffers() {
    let mut context = Context::new();
    let module = scalar_broadcast_add_module(&mut context, false);
    let mut builder = ModuleCompiler::builder();
    builder
        .register_operation_strategy(Arc::new(CompositionalAddStrategy))
        .unwrap();
    let plan = builder.build().compile(&context, module).unwrap();

    assert_eq!(
        plan.kernels()
            .iter()
            .map(|kernel| kernel.name())
            .collect::<Vec<_>>(),
        ["xlahlo-broadcast-2", "test-add"]
    );
    assert_eq!(plan.kernels()[0].bindings().len(), 2);
    assert_eq!(plan.kernels()[1].bindings().len(), 3);
    for kernel in plan.kernels() {
        kernel.define_checked().unwrap();
    }
}

#[test]
fn compositional_lowering_materializes_non_splat_constants() {
    let bits = [0x8000_0000, 0x7fc0_1234, 0xff80_0000, 0x3f80_0000];
    for custom_add in [false, true] {
        let mut context = Context::new();
        let module = vector_constant_add_module(&mut context, &bits);
        let mut builder = ModuleCompiler::builder();
        if custom_add {
            builder
                .register_operation_strategy(Arc::new(CompositionalAddStrategy))
                .unwrap();
        }
        let plan = builder.build().compile(&context, module).unwrap();
        assert_eq!(
            plan.kernels()
                .iter()
                .map(|kernel| kernel.name())
                .collect::<Vec<_>>(),
            [
                "xlahlo-constant-1",
                if custom_add {
                    "test-add"
                } else {
                    "xlahlo-op-2"
                }
            ]
        );
        assert_eq!(plan.buffers()[1].role, BufferRole::Temporary);
        assert_eq!(plan.buffers()[1].descriptor.shape, [2, 2]);
        assert_eq!(plan.kernels()[0].bindings()[0].access, BufferAccess::Write);
        assert_eq!(
            plan.kernels()[1].bindings()[1].buffer,
            plan.kernels()[0].bindings()[0].buffer
        );
        assert_materialized_values(&plan.kernels()[0], &[], &bits);
    }
}

#[test]
fn default_compiler_broadcasts_explicit_axes_and_preserves_literal_order() {
    let cases = [
        (vec![2], vec![2, 3], vec![0], vec![0, 0, 0, 1, 1, 1]),
        (vec![2], vec![3, 2], vec![1], vec![0, 1, 0, 1, 0, 1]),
        (
            vec![2, 3],
            vec![3, 4, 2],
            vec![2, 0],
            vec![
                0, 3, 0, 3, 0, 3, 0, 3, 1, 4, 1, 4, 1, 4, 1, 4, 2, 5, 2, 5, 2, 5, 2, 5,
            ],
        ),
        (vec![2, 3], vec![3, 2], vec![1, 0], vec![0, 3, 1, 4, 2, 5]),
        (
            vec![2, 1],
            vec![2, 3, 4],
            vec![0, 2],
            [vec![0; 12], vec![1; 12]].concat(),
        ),
        (vec![1], vec![0, 3], vec![0], vec![]),
        (vec![0], vec![0, 3], vec![0], vec![]),
        (vec![0], vec![2, 0, 3], vec![1], vec![]),
        (vec![1, 0], vec![2, 0, 3], vec![0, 1], vec![]),
    ];
    for (source_shape, result_shape, dimensions, expected_indices) in cases {
        let elements = source_shape.iter().product::<i64>() as usize;
        let bits: Vec<_> = (0..elements)
            .map(|index| u128::from(((index + 1) as f32).to_bits()))
            .collect();
        let expected: Vec<_> = expected_indices.iter().map(|&index| bits[index]).collect();
        for constant in [false, true] {
            let mut context = Context::new();
            let module = broadcast_add_module(
                &mut context,
                source_shape.clone(),
                result_shape.clone(),
                dimensions.clone(),
                constant.then(|| bits.clone()),
            );
            let plan = ModuleCompiler::builder()
                .build()
                .compile(&context, module)
                .unwrap();
            assert_eq!(plan.kernels().len(), 2);
            let broadcast = &plan.kernels()[0];
            assert_eq!(broadcast.name(), "xlahlo-broadcast-2");
            assert_eq!(broadcast.bindings().len(), if constant { 1 } else { 2 });
            let output = broadcast.bindings().last().unwrap();
            assert_eq!(output.access, BufferAccess::Write);
            assert_eq!(
                plan.buffers()[output.buffer.index()].descriptor.shape,
                result_shape
            );
            assert_eq!(plan.kernels()[1].bindings()[0].buffer, output.buffer);
            assert_eq!(plan.has_device_work(), !expected.is_empty());
            assert_eq!(broadcast.launch().cube_count.is_none(), expected.is_empty());
            assert_materialized_values(broadcast, &bits, &expected);
        }
    }
}

#[test]
fn default_compiler_rejects_invalid_broadcast_mappings_before_lowering() {
    for (source, result, dimensions, message) in [
        (
            vec![2],
            vec![2, 3],
            vec![],
            "broadcast mapping rank mismatch",
        ),
        (vec![2], vec![2, 3], vec![-1], "axis"),
        (vec![2], vec![2, 3], vec![2], "axis"),
        (vec![2, 2], vec![2, 3], vec![0, 0], "axis"),
        (
            vec![2],
            vec![3, 3],
            vec![0],
            "broadcast mapped dimensions differ",
        ),
        (
            vec![0],
            vec![2, 3],
            vec![0],
            "broadcast mapped dimensions differ",
        ),
    ] {
        let mut context = Context::new();
        let module = broadcast_add_module(&mut context, source, result, dimensions, None);
        let Err(error) = ModuleCompiler::builder().build().compile(&context, module) else {
            panic!("invalid broadcast must fail verification");
        };
        assert!(
            matches!(&error, xlahlo_cubecl::ModuleCompileError::Verification(_)),
            "{error}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }
}

#[test]
fn default_compiler_preserves_broadcast_storage_element_types() {
    for (element, bits) in [
        (ElementType::Pred, vec![0, 1]),
        (ElementType::U1, vec![1, 0]),
        (ElementType::S32, vec![0xffff_ffff, 7]),
        (ElementType::U64, vec![u64::MAX as u128, 9]),
        (ElementType::F16, vec![0x8000, 0x3c00]),
        (ElementType::BF16, vec![0x8000, 0x3f80]),
        (
            ElementType::F64,
            vec![0x8000_0000_0000_0000, 0x3ff0_0000_0000_0000],
        ),
    ] {
        for constant in [false, true] {
            let mut context = Context::new();
            xlahlo::register(&mut context);
            let source = Shape::array(&context, element, vec![2]).unwrap();
            let result = Shape::array(&context, element, vec![2, 3]).unwrap();
            let module = broadcast_typed_module(
                &mut context,
                source,
                result,
                vec![0],
                constant.then(|| bits.clone()),
                false,
            );
            let plan = ModuleCompiler::builder()
                .build()
                .compile(&context, module)
                .unwrap();
            let broadcast = &plan.kernels()[0];
            let output = broadcast.bindings().last().unwrap().buffer;
            assert_eq!(
                plan.buffers()[output.index()].descriptor.element_type,
                element
            );
            assert_materialized_values(
                broadcast,
                &bits,
                &[bits[0], bits[0], bits[0], bits[1], bits[1], bits[1]],
            );
        }
    }
}

#[test]
fn default_compiler_rejects_broadcast_element_changes_and_non_row_major_layouts() {
    for non_row_major in [false, true] {
        let mut context = Context::new();
        xlahlo::register(&mut context);
        let source = Shape::array(&context, ElementType::F32, vec![2, 3]).unwrap();
        let source = if non_row_major {
            let mut layout = Layout::descending(2);
            layout.minor_to_major.reverse();
            Shape::with_layout(&context, source, layout).unwrap()
        } else {
            source
        };
        let result = Shape::array(
            &context,
            if non_row_major {
                ElementType::F32
            } else {
                ElementType::S32
            },
            vec![2, 3, 4],
        )
        .unwrap();
        let module = broadcast_typed_module(&mut context, source, result, vec![0, 1], None, false);
        let Err(error) = ModuleCompiler::builder().build().compile(&context, module) else {
            panic!("invalid broadcast contract must be rejected");
        };
        if non_row_major {
            assert!(matches!(
                error,
                xlahlo_cubecl::ModuleCompileError::Unsupported { feature }
                    if feature == "non-dense row-major tensor layout"
            ));
        } else {
            assert!(matches!(
                &error,
                xlahlo_cubecl::ModuleCompileError::Verification(_)
            ));
            assert!(
                error.to_string().contains("broadcast element types differ"),
                "{error}"
            );
        }
    }
}

#[test]
fn compositional_lowering_broadcasts_non_scalar_tensors_with_permuted_rank_expansion() {
    let mut context = Context::new();
    let module = broadcast_add_module(&mut context, vec![2, 3], vec![3, 4, 2], vec![2, 0], None);
    let mut builder = ModuleCompiler::builder();
    builder
        .register_operation_strategy(Arc::new(CompositionalAddStrategy))
        .unwrap();
    let plan = builder.build().compile(&context, module).unwrap();

    assert_eq!(
        plan.kernels()
            .iter()
            .map(|kernel| kernel.name())
            .collect::<Vec<_>>(),
        ["xlahlo-broadcast-2", "test-add"]
    );
    assert_eq!(plan.kernels()[0].bindings().len(), 2);
    for kernel in plan.kernels() {
        kernel.define_checked().unwrap();
    }
}

#[test]
fn compositional_lowering_broadcasts_non_leading_and_unit_axes() {
    let mut context = Context::new();
    let module = broadcast_add_module(&mut context, vec![2, 1], vec![2, 3, 4], vec![0, 2], None);
    let mut builder = ModuleCompiler::builder();
    builder
        .register_operation_strategy(Arc::new(CompositionalAddStrategy))
        .unwrap();
    let plan = builder.build().compile(&context, module).unwrap();

    assert_eq!(plan.kernels()[0].name(), "xlahlo-broadcast-2");
    assert_eq!(plan.kernels()[0].bindings().len(), 2);
    plan.kernels()[0].define_checked().unwrap();
}

#[test]
fn compositional_lowering_inlines_dense_constant_broadcast_before_elementwise() {
    let mut context = Context::new();
    let module = broadcast_add_module(
        &mut context,
        vec![2, 2],
        vec![3, 2, 2],
        vec![1, 2],
        Some(vec![0x3f80_0000, 0x4000_0000, 0x4040_0000, 0x4080_0000]),
    );
    let mut builder = ModuleCompiler::builder();
    builder
        .register_operation_strategy(Arc::new(CompositionalAddStrategy))
        .unwrap();
    let plan = builder.build().compile(&context, module).unwrap();

    assert_eq!(
        plan.kernels()
            .iter()
            .map(|kernel| kernel.name())
            .collect::<Vec<_>>(),
        ["xlahlo-broadcast-2", "test-add"]
    );
    assert_eq!(plan.kernels()[0].bindings().len(), 1);
    for kernel in plan.kernels() {
        kernel.define_checked().unwrap();
    }
}

#[test]
fn compositional_lowering_materializes_large_constants_before_broadcast() {
    let bits: Vec<_> = (0..9).map(|index| 0x3f80_0000 + index).collect();
    for custom_add in [false, true] {
        let mut context = Context::new();
        let module = broadcast_add_module(
            &mut context,
            vec![9],
            vec![2, 9],
            vec![1],
            Some(bits.clone()),
        );
        let mut builder = ModuleCompiler::builder();
        if custom_add {
            builder
                .register_operation_strategy(Arc::new(CompositionalAddStrategy))
                .unwrap();
        }
        let plan = builder.build().compile(&context, module).unwrap();
        assert_eq!(
            plan.kernels()
                .iter()
                .map(|kernel| kernel.name())
                .collect::<Vec<_>>(),
            [
                "xlahlo-constant-0",
                "xlahlo-broadcast-2",
                if custom_add {
                    "test-add"
                } else {
                    "xlahlo-op-3"
                }
            ]
        );
        assert_materialized_values(&plan.kernels()[0], &[], &bits);
        assert_materialized_values(&plan.kernels()[1], &bits, &bits.repeat(2));
        assert_eq!(
            plan.kernels()[0].bindings()[0].buffer,
            plan.kernels()[1].bindings()[0].buffer
        );
    }
}

#[test]
fn default_compiler_reports_unsupported_operations_with_a_typed_error() {
    let mut context = Context::new();
    let module = dot_module(&mut context);
    let compiler = ModuleCompiler::builder().build();

    let Err(error) = compiler.compile(&context, module) else {
        panic!("dot must be rejected by the default elementwise lowering");
    };

    assert!(matches!(
        error,
        xlahlo_cubecl::ModuleCompileError::Unsupported { .. }
    ));
}

#[derive(Clone, Copy)]
enum ViewUse {
    Root,
    Add,
    Minimum,
    AddAndView,
}

fn static_view_module(
    context: &mut Context,
    source: BufferDescriptor,
    transforms: &[(Vec<i64>, IndexTransformKind)],
    usage: ViewUse,
    computed_base: bool,
) -> Ptr<Operation> {
    xlahlo::register(context);
    let input_type = Shape::array(context, source.element_type, source.shape).unwrap();
    let output_shape = &transforms.last().unwrap().0;
    let output_type = Shape::array(context, source.element_type, output_shape.clone()).unwrap();
    let return_type = if matches!(usage, ViewUse::AddAndView) {
        Shape::tuple(context, vec![output_type, output_type]).unwrap()
    } else {
        output_type
    };
    let signature = FunctionType::get(context, vec![input_type], vec![return_type]).into();
    let module = ModuleOp::new(context, Identifier::new("static_views"));
    xlahlo::ir::set_entry(context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(context)
        .get_region(0)
        .deref(context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, context);
    let block = BasicBlock::new(context, None, vec![]);
    block.insert_at_back(computation.body(context), context);
    let input = ParameterOp::new(context, input_type, vec![], Index(0));
    input.get_operation().insert_at_back(block, context);
    let mut value = input.result(context);
    if computed_base {
        let add = AddOp::new(context, input_type, vec![value, value]);
        add.get_operation().insert_at_back(block, context);
        value = add.result(context);
    }
    for (shape, transform) in transforms {
        let ty = Shape::array(context, source.element_type, shape.clone()).unwrap();
        let axes = |axes: &[usize]| Axes(axes.iter().map(|&axis| axis as i64).collect());
        let operation = match transform {
            IndexTransformKind::Reshape => ReshapeOp::new(context, ty, vec![value]).get_operation(),
            IndexTransformKind::Transpose(permutation) => {
                TransposeOp::new(context, ty, vec![value], axes(permutation)).get_operation()
            }
            IndexTransformKind::Reverse(dimensions) => {
                ReverseOp::new(context, ty, vec![value], axes(dimensions)).get_operation()
            }
            IndexTransformKind::Slice { starts, strides } => {
                let limits = starts
                    .iter()
                    .zip(strides)
                    .zip(shape)
                    .map(|((&start, &stride), &extent)| {
                        if extent == 0 {
                            start
                        } else {
                            start + (extent as usize - 1) * stride + 1
                        }
                    })
                    .collect::<Vec<_>>();
                SliceOp::new(
                    context,
                    ty,
                    vec![value],
                    Slice {
                        starts: axes(starts),
                        limits: axes(&limits),
                        strides: axes(strides),
                    },
                )
                .get_operation()
            }
            IndexTransformKind::Broadcast(dimensions) => {
                BroadcastOp::new(context, ty, vec![value], axes(dimensions)).get_operation()
            }
        };
        operation.insert_at_back(block, context);
        value = operation.deref(context).get_result(0);
    }
    let view = value;
    match usage {
        ViewUse::Root => {}
        ViewUse::Add | ViewUse::AddAndView => {
            let add = AddOp::new(context, output_type, vec![value, value]);
            add.get_operation().insert_at_back(block, context);
            value = add.result(context);
        }
        ViewUse::Minimum => {
            let minimum = MinimumOp::new(context, output_type, vec![value, value]);
            minimum.get_operation().insert_at_back(block, context);
            value = minimum.result(context);
        }
    }
    if matches!(usage, ViewUse::AddAndView) {
        let tuple = TupleOp::new(context, return_type, vec![value, view]);
        tuple.get_operation().insert_at_back(block, context);
        value = tuple.result(context);
    }
    ReturnOp::new(context, value)
        .get_operation()
        .insert_at_back(block, context);
    module.get_operation()
}

#[test]
fn static_view_roots_alias_storage_without_kernels_or_allocations() {
    for element_type in [
        ElementType::F32,
        ElementType::F8E4M3FN,
        ElementType::U1,
        ElementType::U64,
    ] {
        let mut context = Context::new();
        let module = static_view_module(
            &mut context,
            BufferDescriptor::dense_row_major(element_type, vec![2, 3, 4]),
            &[(vec![3, 4, 2], IndexTransformKind::Transpose(vec![1, 2, 0]))],
            ViewUse::Root,
            false,
        );
        let plan = ModuleCompiler::builder()
            .build()
            .compile(&context, module)
            .unwrap();
        assert!(plan.kernels().is_empty());
        assert_eq!(plan.buffers().len(), 1);
        assert!(!plan.has_device_work());
        let [PlannedOutput::View(view)] = plan.outputs() else {
            panic!("root view")
        };
        assert_eq!(view.buffer.index(), 0);
        assert_eq!(view.descriptor.element_type, element_type);
        assert_eq!(view.descriptor.shape, [3, 4, 2]);
        assert_eq!(
            view.map,
            OutputIndexMap::Affine {
                offset: 0,
                strides: vec![4, 1, 12]
            }
        );
    }
}

#[test]
fn static_view_chain_composes_signed_slice_broadcast_and_reshape_addresses() {
    let transforms = [
        (
            vec![2, 2],
            IndexTransformKind::Slice {
                starts: vec![0, 1],
                strides: vec![2, 2],
            },
        ),
        (vec![2, 2], IndexTransformKind::Reverse(vec![1])),
        (vec![2, 2], IndexTransformKind::Transpose(vec![1, 0])),
        (vec![2, 1, 2], IndexTransformKind::Reshape),
        (vec![2, 3, 2], IndexTransformKind::Broadcast(vec![0, 1, 2])),
    ];
    for usage in [ViewUse::Root, ViewUse::Add] {
        let mut context = Context::new();
        let module = static_view_module(
            &mut context,
            BufferDescriptor::dense_row_major(ElementType::U32, vec![3, 4]),
            &transforms,
            usage,
            false,
        );
        let plan = ModuleCompiler::builder()
            .build()
            .compile(&context, module)
            .unwrap();
        if matches!(usage, ViewUse::Root) {
            assert_eq!(plan.buffers().len(), 1);
            assert!(plan.kernels().is_empty());
            let [PlannedOutput::View(view)] = plan.outputs() else {
                panic!("root view")
            };
            assert_eq!(
                view.map,
                OutputIndexMap::Affine {
                    offset: 3,
                    strides: vec![-2, 0, 8]
                }
            );
        } else {
            assert_eq!(plan.buffers().len(), 2);
            assert_eq!(plan.kernels().len(), 1);
            assert_eq!(plan.kernels()[0].bindings()[0].buffer.index(), 0);
            assert_eq!(plan.kernels()[0].bindings()[1].buffer.index(), 0);
            assert_materialized_values(
                &plan.kernels()[0],
                &(0..12).collect::<Vec<_>>(),
                &[6, 22, 6, 22, 6, 22, 2, 18, 2, 18, 2, 18],
            );
        }
    }
}

#[test]
fn static_nonaffine_reshape_keeps_exact_root_map_and_fuses_consumer_loads() {
    let transforms = [
        (vec![3, 2], IndexTransformKind::Transpose(vec![1, 0])),
        (vec![6], IndexTransformKind::Reshape),
    ];
    for usage in [ViewUse::Root, ViewUse::Add, ViewUse::Minimum] {
        let mut context = Context::new();
        let module = static_view_module(
            &mut context,
            BufferDescriptor::dense_row_major(ElementType::U32, vec![2, 3]),
            &transforms,
            usage,
            false,
        );
        let plan = ModuleCompiler::builder()
            .build()
            .compile(&context, module)
            .unwrap();
        if matches!(usage, ViewUse::Root) {
            assert!(plan.kernels().is_empty());
            assert_eq!(plan.buffers().len(), 1);
            let [PlannedOutput::View(view)] = plan.outputs() else {
                panic!("root view")
            };
            let OutputIndexMap::Composed {
                storage_shape,
                transforms,
            } = &view.map
            else {
                panic!("non-affine reshape must not be described as affine");
            };
            assert_eq!(storage_shape, &[2, 3]);
            assert_eq!(transforms.len(), 2);
            assert_eq!(
                transforms[0].kind,
                IndexTransformKind::Transpose(vec![1, 0])
            );
            assert_eq!(transforms[1].kind, IndexTransformKind::Reshape);
        } else {
            assert_eq!(plan.buffers().len(), 2);
            assert_eq!(plan.kernels().len(), 1);
            if matches!(usage, ViewUse::Add) {
                assert_materialized_values(
                    &plan.kernels()[0],
                    &[0, 1, 2, 3, 4, 5],
                    &[0, 6, 2, 8, 4, 10],
                );
            } else {
                assert_materialized_values(
                    &plan.kernels()[0],
                    &[0, 1, 2, 3, 4, 5],
                    &[0, 3, 1, 4, 2, 5],
                );
            }
        }
    }
}

#[test]
fn static_reshape_preserves_contiguous_reversed_and_scalar_aliases() {
    let cases = [
        (
            vec![2, 3],
            vec![(vec![3, 2], IndexTransformKind::Reshape)],
            0,
            vec![2, 1],
        ),
        (
            vec![2, 3],
            vec![
                (vec![2, 3], IndexTransformKind::Reverse(vec![0, 1])),
                (vec![6], IndexTransformKind::Reshape),
            ],
            5,
            vec![-1],
        ),
        (
            vec![],
            vec![
                (vec![1], IndexTransformKind::Reshape),
                (vec![3, 4], IndexTransformKind::Broadcast(vec![1])),
            ],
            0,
            vec![0, 0],
        ),
    ];
    for (shape, transforms, offset, strides) in cases {
        let mut context = Context::new();
        let module = static_view_module(
            &mut context,
            BufferDescriptor::dense_row_major(ElementType::U32, shape),
            &transforms,
            ViewUse::Root,
            false,
        );
        let plan = ModuleCompiler::builder()
            .build()
            .compile(&context, module)
            .unwrap();
        assert_eq!(plan.buffers().len(), 1);
        assert!(plan.kernels().is_empty());
        let [PlannedOutput::View(view)] = plan.outputs() else {
            panic!("root view")
        };
        assert_eq!(view.map, OutputIndexMap::Affine { offset, strides });
    }
}

#[test]
fn static_empty_views_do_not_dereference_or_allocate_view_storage() {
    let transforms = [
        (vec![3, 0], IndexTransformKind::Transpose(vec![1, 0])),
        (vec![3, 0], IndexTransformKind::Reverse(vec![1])),
        (vec![0], IndexTransformKind::Reshape),
    ];
    for usage in [ViewUse::Root, ViewUse::Add] {
        let mut context = Context::new();
        let module = static_view_module(
            &mut context,
            BufferDescriptor::dense_row_major(ElementType::U32, vec![0, 3]),
            &transforms,
            usage,
            false,
        );
        let plan = ModuleCompiler::builder()
            .build()
            .compile(&context, module)
            .unwrap();
        assert!(!plan.has_device_work());
        if matches!(usage, ViewUse::Root) {
            assert_eq!(plan.buffers().len(), 1);
            assert!(plan.kernels().is_empty());
            let [PlannedOutput::View(view)] = plan.outputs() else {
                panic!("root view")
            };
            assert_eq!(
                view.map,
                OutputIndexMap::Affine {
                    offset: 0,
                    strides: vec![1]
                }
            );
        } else {
            assert_eq!(plan.buffers().len(), 2);
            assert_eq!(plan.kernels().len(), 1);
            assert_materialized_values(&plan.kernels()[0], &[], &[]);
        }
    }
}

#[test]
fn static_view_outputs_retain_computed_leaf_storage_through_consumers() {
    let mut context = Context::new();
    let module = static_view_module(
        &mut context,
        BufferDescriptor::dense_row_major(ElementType::U32, vec![2, 3]),
        &[(vec![3, 2], IndexTransformKind::Transpose(vec![1, 0]))],
        ViewUse::AddAndView,
        true,
    );
    let plan = ModuleCompiler::builder()
        .build()
        .compile(&context, module)
        .unwrap();
    assert_eq!(plan.buffers().len(), 3);
    assert_eq!(plan.kernels().len(), 2);
    let [PlannedOutput::Buffer(output), PlannedOutput::View(view)] = plan.outputs() else {
        panic!("computed result and leaf-backed view");
    };
    assert_eq!(output.index(), 2);
    assert_eq!(view.buffer.index(), 1);
    assert_eq!(plan.buffers()[1].role, BufferRole::Temporary);
    assert_eq!(
        plan.buffers()[1].lifetime,
        Some(xlahlo_cubecl::BufferLifetime {
            first_kernel: 0,
            last_kernel: 1,
            retained_for_output: true,
        })
    );
    assert_materialized_values(
        &plan.kernels()[0],
        &[0, 1, 2, 3, 4, 5],
        &[0, 2, 4, 6, 8, 10],
    );
    assert_materialized_values(
        &plan.kernels()[1],
        &[0, 2, 4, 6, 8, 10],
        &[0, 12, 4, 16, 8, 20],
    );
}

#[test]
fn static_view_operands_reject_dense_only_custom_strategies_without_copying() {
    let mut context = Context::new();
    let module = static_view_module(
        &mut context,
        BufferDescriptor::dense_row_major(ElementType::U32, vec![2, 3]),
        &[(vec![3, 2], IndexTransformKind::Transpose(vec![1, 0]))],
        ViewUse::Add,
        false,
    );
    let mut builder = ModuleCompiler::builder();
    builder
        .register_operation_strategy(Arc::new(CompositionalAddStrategy))
        .unwrap();
    let Err(error) = builder.build().compile(&context, module) else {
        panic!("dense-only custom strategy must not receive mapped storage");
    };
    assert!(matches!(
        error,
        xlahlo_cubecl::ModuleCompileError::Plan(
            xlahlo_cubecl::PlanError::IndexedOperandsUnsupported { .. }
        )
    ));
}

struct IndexedAddStrategy;

impl OperationLoweringStrategy for IndexedAddStrategy {
    fn name(&self) -> &'static str {
        "indexed-test-add"
    }
    fn priority(&self) -> i32 {
        100
    }
    fn supports_indexed_operands(&self) -> bool {
        true
    }
    fn matches(&self, context: &Context, operation: Ptr<Operation>) -> Result<bool, String> {
        Ok(Operation::is_op::<AddOp>(operation, context))
    }
    fn lower(&self, request: OperationLoweringRequest<'_>) -> Result<(), xlahlo_cubecl::PlanError> {
        assert!(
            request
                .operand_views
                .iter()
                .all(|view| !view.transforms.is_empty())
        );
        let elements = request.result_descriptors[0].num_elements()?;
        request.plan.add_kernel(
            "indexed-test-add",
            Arc::new(IndexedTestKernel {
                operands: request.operand_views.to_vec(),
                elements,
            }),
            vec![
                KernelBinding {
                    buffer: request.operands[0],
                    access: BufferAccess::Read,
                },
                KernelBinding {
                    buffer: request.operands[1],
                    access: BufferAccess::Read,
                },
                KernelBinding {
                    buffer: request.results[0],
                    access: BufferAccess::Write,
                },
            ],
            LaunchGeometry::for_elements(elements, [128, 1, 1])?,
            false,
        )
    }
}

struct IndexedTestKernel {
    operands: Vec<xlahlo_cubecl::OperandView>,
    elements: usize,
}

impl KernelMetadata for IndexedTestKernel {
    fn name(&self) -> &'static str {
        "indexed_test_add"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.operands.clone(), self.elements))
    }
    fn address_type(&self) -> ElemType {
        AddressType::U32.unsigned_type()
    }
}

impl PlannedKernel for IndexedTestKernel {
    fn define_checked(&self) -> Result<KernelDefinition, String> {
        let settings = KernelSettings::new(
            Dim3::new_1d(128),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
        .kernel_name(self.name());
        let mut builder = KernelBuilder::new(settings);
        let ty = ElemType::UInt(cubecl_ir::UIntKind::U32).to_type(builder.ctx());
        let lhs = builder.buffer(ty);
        let rhs = builder.buffer(ty);
        let output = builder.buffer(ty);
        let index_type = ElemType::Index.to_type(builder.ctx());
        let position = builder.register_with_result(&ReadBuiltinOp::new(
            builder.ctx_mut(),
            index_type,
            BuiltinAttr(Builtin::AbsolutePos),
        ));
        let count = builder.const_usize(self.elements);
        let predicate =
            builder.register_with_result(&ULessThanOp::new(builder.ctx_mut(), position, count));
        let guard = IfOp::new(builder.ctx_mut(), predicate);
        builder.register(&guard);
        let body = builder.child(cubecl_ir::OpInserter::new_at_block_end(
            guard.then_block(builder.ctx()),
        ));
        let load = |buffer, view: &xlahlo_cubecl::OperandView| -> Result<Value, String> {
            let index = view
                .emit_index(&body, position)
                .map_err(|error| error.to_string())?;
            let pointer =
                body.register_with_result(&IndexOp::new(body.ctx_mut(), buffer, index, None));
            Ok(body.register_with_result(&LoadOp::new(body.ctx_mut(), pointer)))
        };
        let left = load(lhs, &self.operands[0])?;
        let right = load(rhs, &self.operands[1])?;
        let result = body.register_with_result(&IAddOp::new(body.ctx_mut(), left, right));
        let pointer =
            body.register_with_result(&IndexOp::new(body.ctx_mut(), output, position, None));
        body.register(&StoreOp::new(body.ctx_mut(), pointer, result));
        body.register(&YieldOp::new(body.ctx_mut()));
        let otherwise = builder.child(cubecl_ir::OpInserter::new_at_block_end(
            guard.else_block(builder.ctx()),
        ));
        otherwise.register(&YieldOp::new(otherwise.ctx_mut()));
        builder.register(&cubecl_ir::dialect::branch::ReturnOp::new(
            builder.ctx_mut(),
        ));
        Ok(builder.build())
    }
}

#[test]
fn static_view_indexed_custom_strategy_uses_the_shared_index_emitter() {
    let mut context = Context::new();
    let module = static_view_module(
        &mut context,
        BufferDescriptor::dense_row_major(ElementType::U32, vec![2, 3]),
        &[
            (vec![3, 2], IndexTransformKind::Transpose(vec![1, 0])),
            (vec![6], IndexTransformKind::Reshape),
        ],
        ViewUse::Add,
        false,
    );
    let mut builder = ModuleCompiler::builder();
    builder
        .register_operation_strategy(Arc::new(IndexedAddStrategy))
        .unwrap();
    let plan = builder.build().compile(&context, module).unwrap();
    assert_eq!(plan.kernels().len(), 1);
    assert_eq!(plan.buffers().len(), 2);
    assert_materialized_values(
        &plan.kernels()[0],
        &[0, 1, 2, 3, 4, 5],
        &[0, 6, 2, 8, 4, 10],
    );
}

#[test]
fn static_views_fuse_into_both_gather_operand_and_index_loads() {
    let mut context = Context::new();
    xlahlo::register(&mut context);
    let source_type = Shape::array(&context, ElementType::U32, vec![2, 3]).unwrap();
    let transposed_type = Shape::array(&context, ElementType::U32, vec![3, 2]).unwrap();
    let flat_type = Shape::array(&context, ElementType::U32, vec![6]).unwrap();
    let indices_type = Shape::array(&context, ElementType::U32, vec![3]).unwrap();
    let signature = FunctionType::get(
        &context,
        vec![source_type, indices_type],
        vec![indices_type],
    )
    .into();
    let module = ModuleOp::new(&mut context, Identifier::new("gather_views"));
    xlahlo::ir::set_entry(&context, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(&context)
        .get_region(0)
        .deref(&context)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(&mut context, Identifier::new("main"), signature);
    computation
        .get_operation()
        .insert_at_back(container, &context);
    let block = BasicBlock::new(&mut context, None, vec![]);
    block.insert_at_back(computation.body(&context), &context);
    let source = ParameterOp::new(&mut context, source_type, vec![], Index(0));
    let indices = ParameterOp::new(&mut context, indices_type, vec![], Index(1));
    source.get_operation().insert_at_back(block, &context);
    indices.get_operation().insert_at_back(block, &context);
    let source_value = source.result(&context);
    let indices_value = indices.result(&context);
    let transpose = TransposeOp::new(
        &mut context,
        transposed_type,
        vec![source_value],
        Axes(vec![1, 0]),
    );
    transpose.get_operation().insert_at_back(block, &context);
    let transpose_value = transpose.result(&context);
    let reshape = ReshapeOp::new(&mut context, flat_type, vec![transpose_value]);
    reshape.get_operation().insert_at_back(block, &context);
    let reverse = ReverseOp::new(
        &mut context,
        indices_type,
        vec![indices_value],
        Axes(vec![0]),
    );
    reverse.get_operation().insert_at_back(block, &context);
    let reshape_value = reshape.result(&context);
    let reverse_value = reverse.result(&context);
    let gather = GatherOp::new(
        &mut context,
        indices_type,
        vec![reshape_value, reverse_value],
        GatherDimensionNumbers {
            offset_dims: Axes(vec![]),
            collapsed_slice_dims: Axes(vec![0]),
            start_index_map: Axes(vec![0]),
            index_vector_dim: 1,
            operand_batching_dims: Axes(vec![]),
            start_indices_batching_dims: Axes(vec![]),
        },
        Axes(vec![1]),
        IndexFlags {
            sorted: false,
            unique: false,
        },
    );
    gather.get_operation().insert_at_back(block, &context);
    let gather_value = gather.result(&context);
    ReturnOp::new(&mut context, gather_value)
        .get_operation()
        .insert_at_back(block, &context);
    let plan = ModuleCompiler::builder()
        .build()
        .compile(&context, module.get_operation())
        .unwrap();
    assert_eq!(plan.kernels().len(), 1);
    assert_eq!(plan.buffers().len(), 3);
    // The two input buffers intentionally share these first three test values.
    assert_materialized_values(&plan.kernels()[0], &[0, 1, 2, 3, 4, 5], &[1, 3, 0]);
}

#[test]
fn static_nonselfinverse_transpose_and_singleton_maps_have_exact_numeric_indices() {
    let mut context = Context::new();
    let module = static_view_module(
        &mut context,
        BufferDescriptor::dense_row_major(ElementType::U32, vec![2, 3, 4]),
        &[(vec![3, 4, 2], IndexTransformKind::Transpose(vec![1, 2, 0]))],
        ViewUse::Add,
        false,
    );
    let plan = ModuleCompiler::builder()
        .build()
        .compile(&context, module)
        .unwrap();
    assert_eq!(plan.kernels().len(), 1);
    assert_eq!(plan.buffers().len(), 2);
    let expected: Vec<u128> = (0..3)
        .flat_map(|a| (0..4).flat_map(move |b| (0..2).map(move |c| 2 * (c * 12 + a * 4 + b))))
        .collect();
    assert_materialized_values(&plan.kernels()[0], &(0..24).collect::<Vec<_>>(), &expected);

    let mut context = Context::new();
    let module = static_view_module(
        &mut context,
        BufferDescriptor::dense_row_major(ElementType::U32, vec![2, 1, 3]),
        &[
            (vec![3, 2, 1], IndexTransformKind::Transpose(vec![2, 0, 1])),
            (vec![3, 2], IndexTransformKind::Reshape),
        ],
        ViewUse::Root,
        false,
    );
    let plan = ModuleCompiler::builder()
        .build()
        .compile(&context, module)
        .unwrap();
    assert!(plan.kernels().is_empty());
    assert_eq!(plan.buffers().len(), 1);
    let [PlannedOutput::View(view)] = plan.outputs() else {
        panic!("singleton root view")
    };
    assert_eq!(
        view.map,
        OutputIndexMap::Affine {
            offset: 0,
            strides: vec![1, 3]
        }
    );
}
