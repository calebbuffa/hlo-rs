use pliron::{
    basic_block::BasicBlock,
    builtin::{ops::ModuleOp, types::FunctionType},
    context::{Context, Ptr},
    identifier::Identifier,
    op::Op,
    operation::Operation,
};
use xlahlo::{
    attributes::{Axes, GatherDimensionNumbers, Index, IndexFlags},
    ops::{ComputationOp, ReturnOp, control::ParameterOp, movement::GatherOp},
    shape,
    types::ElementType,
};
use xlahlo_cubecl::{BufferRole, ModuleCompiler};

#[allow(clippy::too_many_arguments)]
fn gather_module(
    ctx: &mut Context,
    element_type: ElementType,
    operand_shape: Vec<i64>,
    indices_shape: Vec<i64>,
    output_shape: Vec<i64>,
    index_type: ElementType,
    dimensions: GatherDimensionNumbers,
    slice_sizes: Vec<i64>,
) -> Ptr<Operation> {
    xlahlo::register(ctx);
    let operand_type = shape::array(ctx, element_type, operand_shape).unwrap();
    let indices_type = shape::array(ctx, index_type, indices_shape).unwrap();
    let output_type = shape::array(ctx, element_type, output_shape).unwrap();
    let signature =
        FunctionType::get(ctx, vec![operand_type, indices_type], vec![output_type]).into();
    let module = ModuleOp::new(ctx, Identifier::new("gather"));
    xlahlo::ir::set_entry(ctx, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block()
        .unwrap();
    let computation = ComputationOp::new(ctx, Identifier::new("main"), signature);
    computation.get_operation().insert_at_back(container, ctx);
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(computation.body(ctx), ctx);
    let operand = ParameterOp::new(ctx, operand_type, vec![], Index(0));
    let indices = ParameterOp::new(ctx, indices_type, vec![], Index(1));
    operand.get_operation().insert_at_back(block, ctx);
    indices.get_operation().insert_at_back(block, ctx);
    let gather = GatherOp::new(
        ctx,
        output_type,
        vec![operand.result(ctx), indices.result(ctx)],
        dimensions,
        Axes(slice_sizes),
        IndexFlags {
            sorted: false,
            unique: false,
        },
    );
    gather.get_operation().insert_at_back(block, ctx);
    ReturnOp::new(ctx, gather.result(ctx))
        .get_operation()
        .insert_at_back(block, ctx);
    module.get_operation()
}

fn compile_gather_plan(module: Ptr<Operation>, ctx: &Context, output_shape: &[i64]) {
    xlahlo::verification::verify(ctx, module).unwrap();
    let plan = ModuleCompiler::builder()
        .build()
        .compile(ctx, module)
        .unwrap();
    assert_eq!(plan.buffers().len(), 3);
    assert_eq!(plan.buffers()[0].role, BufferRole::Input(0));
    assert_eq!(plan.buffers()[1].role, BufferRole::Input(1));
    assert_eq!(plan.buffers()[2].role, BufferRole::Output(0));
    assert_eq!(plan.buffers()[2].descriptor.shape.as_slice(), output_shape);
    assert_eq!(plan.kernels().len(), 1);
    assert_eq!(plan.kernels()[0].bindings().len(), 3);
    assert_eq!(
        plan.kernels()[0].launch().cube_count.is_none(),
        output_shape.iter().product::<i64>() == 0
    );
    assert_eq!(
        plan.has_device_work(),
        plan.kernels()[0].launch().cube_count.is_some()
    );
}

#[test]
fn generic_gather_lowers_index_vectors_batching_and_arbitrary_windows() {
    let cases = [
        (
            vec![5, 6, 7],
            vec![2, 1],
            vec![2, 2, 3],
            GatherDimensionNumbers {
                offset_dims: Axes(vec![1, 2]),
                collapsed_slice_dims: Axes(vec![2]),
                start_index_map: Axes(vec![2]),
                index_vector_dim: 1,
                operand_batching_dims: Axes(vec![]),
                start_indices_batching_dims: Axes(vec![]),
            },
            vec![2, 3, 1],
        ),
        (
            vec![3, 5, 4],
            vec![3, 2, 1],
            vec![3, 2, 4],
            GatherDimensionNumbers {
                offset_dims: Axes(vec![2]),
                collapsed_slice_dims: Axes(vec![1]),
                start_index_map: Axes(vec![1]),
                index_vector_dim: 2,
                operand_batching_dims: Axes(vec![0]),
                start_indices_batching_dims: Axes(vec![0]),
            },
            vec![1, 1, 4],
        ),
    ];
    for (operand, indices, output, dimensions, slices) in cases {
        for element_type in [ElementType::F32, ElementType::S32] {
            for index_type in [ElementType::S32, ElementType::U32] {
                let mut ctx = Context::new();
                let module = gather_module(
                    &mut ctx,
                    element_type,
                    operand.clone(),
                    indices.clone(),
                    output.clone(),
                    index_type,
                    dimensions.clone(),
                    slices.clone(),
                );
                compile_gather_plan(module, &ctx, &output);
            }
        }
    }
}

#[test]
fn generic_gather_lowers_implicit_index_vectors_for_empty_and_nonempty_outputs() {
    for (indices_shape, output_shape) in [(vec![3], vec![3, 4]), (vec![0], vec![0, 4])] {
        let mut ctx = Context::new();
        let module = gather_module(
            &mut ctx,
            ElementType::F32,
            vec![5, 4],
            indices_shape.clone(),
            output_shape.clone(),
            ElementType::S32,
            GatherDimensionNumbers {
                offset_dims: Axes(vec![1]),
                collapsed_slice_dims: Axes(vec![0]),
                start_index_map: Axes(vec![0]),
                index_vector_dim: 1,
                operand_batching_dims: Axes(vec![]),
                start_indices_batching_dims: Axes(vec![]),
            },
            vec![1, 4],
        );
        compile_gather_plan(module, &ctx, &output_shape);
    }
}
