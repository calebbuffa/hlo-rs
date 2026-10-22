use crate::control_support::*;
use xlahlo::{
    attributes::{Axes, GatherDimensionNumbers, IndexFlags, ScatterDimensionNumbers},
    ops::{
        control::{AsyncDoneOp, AsyncStartOp},
        movement::{GatherOp, ScatterOp},
    },
    pliron::{builtin::attributes::StringAttr, op::Op, operation::Operation},
    shape,
    types::ElementType,
    verification,
};

#[test]
fn gather_identity_uses_sortedness_but_not_scatter_only_uniqueness() {
    let mut ctx = context();
    let input = shape(&ctx, ElementType::F32, &[5, 4]);
    let indices = shape(&ctx, ElementType::S32, &[3, 1]);
    let output = shape(&ctx, ElementType::F32, &[3, 4]);
    let dimensions = GatherDimensionNumbers {
        offset_dims: Axes(vec![1]),
        collapsed_slice_dims: Axes(vec![0]),
        operand_batching_dims: Axes(vec![]),
        start_indices_batching_dims: Axes(vec![]),
        start_index_map: Axes(vec![0]),
        index_vector_dim: 1,
    };
    let mut computations = vec![];
    let mut gathers = vec![];
    for name in ["first", "second"] {
        let (comp, block, args) = computation(&mut ctx, name, vec![input, indices], output);
        let gather = GatherOp::new(
            &mut ctx,
            output,
            args,
            dimensions.clone(),
            Axes(vec![1, 4]),
            IndexFlags {
                sorted: false,
                unique: false,
            },
        );
        gather.get_operation().insert_at_back(block, &ctx);
        let value = gather.result(&ctx);
        ret(&mut ctx, block, value);
        computations.push(comp);
        gathers.push(gather);
    }
    let (main, block, args) = computation(&mut ctx, "main", vec![input, indices], output);
    let async_context = shape::tuple(
        &ctx,
        vec![
            shape::tuple(&ctx, vec![input, indices]).unwrap(),
            output,
            shape(&ctx, ElementType::U32, &[]),
        ],
    )
    .unwrap();
    let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
    start.get_operation().insert_at_back(block, &ctx);
    let value = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, output, vec![value]);
    attr(
        &ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, &ctx);
    let value = done.result(&ctx);
    ret(&mut ctx, block, value);
    computations.push(main);
    let root = module(&mut ctx, &computations, "main").get_operation();
    attr(
        &ctx,
        gathers[0].get_operation(),
        "hlo_dimension_numbers",
        StringAttr::new("not-the-opcode-dimension-attribute".into()),
    );
    attr(
        &ctx,
        gathers[1].get_operation(),
        "hlo_dimension_numbers",
        Axes(vec![99]),
    );
    for sorted_a in [false, true] {
        for sorted_b in [false, true] {
            for unique_a in [false, true] {
                for unique_b in [false, true] {
                    for (op, sorted, unique) in [
                        (gathers[0], sorted_a, unique_a),
                        (gathers[1], sorted_b, unique_b),
                    ] {
                        attr(
                            &ctx,
                            op.get_operation(),
                            "hlo_flags",
                            IndexFlags { sorted, unique },
                        );
                        Operation::get_op_dyn(op.get_operation(), &ctx)
                            .verify(&ctx)
                            .unwrap();
                    }
                    assert_eq!(
                        verification::verify(&ctx, root).is_ok(),
                        sorted_a == sorted_b
                    );
                }
            }
        }
    }
}

#[test]
fn scatter_identity_retains_both_sortedness_and_uniqueness() {
    let mut ctx = context();
    let input = shape(&ctx, ElementType::F32, &[5]);
    let indices = shape(&ctx, ElementType::S32, &[3, 1]);
    let updates = shape(&ctx, ElementType::F32, &[3]);
    let scalar = shape(&ctx, ElementType::F32, &[]);
    let (reducer, block, args) = computation(&mut ctx, "reducer", vec![scalar, scalar], scalar);
    ret(&mut ctx, block, args[1]);
    let dimensions = ScatterDimensionNumbers {
        update_window_dims: Axes(vec![]),
        inserted_window_dims: Axes(vec![0]),
        input_batching_dims: Axes(vec![]),
        scatter_indices_batching_dims: Axes(vec![]),
        scatter_dims_to_operand_dims: Axes(vec![0]),
        index_vector_dim: 1,
    };
    let mut computations = vec![reducer];
    let mut scatters = vec![];
    for name in ["first", "second"] {
        let (comp, block, args) = computation(&mut ctx, name, vec![input, indices, updates], input);
        let scatter = ScatterOp::new(
            &mut ctx,
            input,
            args,
            dimensions.clone(),
            callees(&["reducer"]),
            IndexFlags {
                sorted: false,
                unique: false,
            },
        );
        scatter.get_operation().insert_at_back(block, &ctx);
        let value = scatter.result(&ctx);
        ret(&mut ctx, block, value);
        computations.push(comp);
        scatters.push(scatter);
    }
    let (main, block, args) = computation(&mut ctx, "main", vec![input, indices, updates], input);
    let async_context = shape::tuple(
        &ctx,
        vec![
            shape::tuple(&ctx, vec![input, indices, updates]).unwrap(),
            input,
            shape(&ctx, ElementType::U32, &[]),
        ],
    )
    .unwrap();
    let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
    start.get_operation().insert_at_back(block, &ctx);
    let value = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, input, vec![value]);
    attr(
        &ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, &ctx);
    let value = done.result(&ctx);
    ret(&mut ctx, block, value);
    computations.push(main);
    let root = module(&mut ctx, &computations, "main").get_operation();
    attr(
        &ctx,
        scatters[0].get_operation(),
        "hlo_dimension_numbers",
        StringAttr::new("not-the-opcode-dimension-attribute".into()),
    );
    attr(
        &ctx,
        scatters[1].get_operation(),
        "hlo_dimension_numbers",
        Axes(vec![99]),
    );
    for sorted_a in [false, true] {
        for sorted_b in [false, true] {
            for unique_a in [false, true] {
                for unique_b in [false, true] {
                    for (op, sorted, unique) in [
                        (scatters[0], sorted_a, unique_a),
                        (scatters[1], sorted_b, unique_b),
                    ] {
                        attr(
                            &ctx,
                            op.get_operation(),
                            "hlo_flags",
                            IndexFlags { sorted, unique },
                        );
                        Operation::get_op_dyn(op.get_operation(), &ctx)
                            .verify(&ctx)
                            .unwrap();
                    }
                    assert_eq!(
                        verification::verify(&ctx, root).is_ok(),
                        sorted_a == sorted_b && unique_a == unique_b
                    );
                }
            }
        }
    }
}
