use super::*;
use clienthlo::attributes::{Precision, RaggedDotDimensionNumbersAttr, RaggedDotDimensions};

#[test]
fn scan_lowering_handles_directions_shapes_and_nested_chlo() {
    for reverse in [false, true] {
        for shape in [vec![0, 3], vec![4, 3], vec![-1, 3], vec![4, -1]] {
            for element in [ElementType::F32, ElementType::C64, ElementType::I32] {
                let mut ctx = Context::new();
                let ty = tensor(&ctx, &shape, element);
                let carry = tensor(&ctx, &[shape[1]], element);
                let (module, fun) = function(&mut ctx, vec![ty, carry], vec![ty, carry]);
                let args = fun.arguments(&ctx);
                let scan = ScanOp::new_inferred(
                    &mut ctx,
                    vec![args[0]],
                    vec![args[1]],
                    0,
                    None,
                    reverse,
                    Some(true),
                    |ctx, block| {
                        let args: Vec<_> = block.deref(ctx).arguments().collect();
                        let add = BroadcastAddOp::new_inferred(ctx, args[0], args[1], None)?;
                        let value = add.get_operation().deref(ctx).get_result(0);
                        add.get_operation().insert_at_back(block, ctx);
                        ReturnOp::new_many(ctx, vec![value, value])
                            .get_operation()
                            .insert_at_back(block, ctx);
                        Ok(())
                    },
                )
                .unwrap();
                attach(&mut ctx, &fun, scan);
                check(&mut ctx, &module);
            }
        }
    }
}

#[test]
fn scan_without_inputs_and_carry_only_scan_export() {
    for outputs in [false, true] {
        let mut ctx = Context::new();
        let scalar = tensor(&ctx, &[], ElementType::F64);
        let values = tensor(&ctx, &[3], ElementType::F64);
        let result_types = if outputs {
            vec![values, scalar]
        } else {
            vec![scalar]
        };
        let arguments = if outputs {
            vec![scalar]
        } else {
            vec![values, scalar]
        };
        let (module, fun) = function(&mut ctx, arguments, result_types);
        let args = fun.arguments(&ctx);
        let arg = *args.last().unwrap();
        let inputs = if outputs { vec![] } else { vec![args[0]] };
        let scan = ScanOp::new_inferred(
            &mut ctx,
            inputs,
            vec![arg],
            0,
            Some(3),
            false,
            None,
            |ctx, block| {
                let x = *block
                    .deref(ctx)
                    .arguments()
                    .collect::<Vec<_>>()
                    .last()
                    .unwrap();
                let square = SquareOp::new_inferred(ctx, x)?;
                let value = square.get_operation().deref(ctx).get_result(0);
                square.get_operation().insert_at_back(block, ctx);
                let results = if outputs {
                    vec![value, value]
                } else {
                    vec![value]
                };
                ReturnOp::new_many(ctx, results)
                    .get_operation()
                    .insert_at_back(block, ctx);
                Ok(())
            },
        )
        .unwrap();
        attach(&mut ctx, &fun, scan);
        check(&mut ctx, &module);
    }
}

#[test]
fn ragged_dot_all_modes_and_dynamic_geometry_export() {
    for dynamic in [false, true] {
        for count in [0, 2] {
            for mode in 1..=3 {
                let mut ctx = Context::new();
                let m = if dynamic { -1 } else { 4 };
                let k = if dynamic { -1 } else { 3 };
                let batch = if dynamic { -1 } else { 2 };
                let (lhs_shape, rhs_shape, group_shape, d) = match mode {
                    1 => (
                        vec![batch, m, k],
                        vec![batch, count, k, 5],
                        vec![batch, count],
                        RaggedDotDimensions {
                            lhs_batching_dimensions: vec![0],
                            rhs_batching_dimensions: vec![0],
                            lhs_contracting_dimensions: vec![2],
                            rhs_contracting_dimensions: vec![2],
                            lhs_ragged_dimensions: vec![1],
                            rhs_group_dimensions: vec![1],
                        },
                    ),
                    2 => (
                        vec![batch, m, k],
                        vec![batch, k, 5],
                        vec![batch, count],
                        RaggedDotDimensions {
                            lhs_batching_dimensions: vec![0],
                            rhs_batching_dimensions: vec![0],
                            lhs_contracting_dimensions: vec![2],
                            rhs_contracting_dimensions: vec![1],
                            lhs_ragged_dimensions: vec![2],
                            rhs_group_dimensions: vec![],
                        },
                    ),
                    _ => (
                        vec![batch, m, k],
                        vec![batch, k, 5],
                        vec![count],
                        RaggedDotDimensions {
                            lhs_batching_dimensions: vec![0],
                            rhs_batching_dimensions: vec![0],
                            lhs_contracting_dimensions: vec![2],
                            rhs_contracting_dimensions: vec![1],
                            lhs_ragged_dimensions: vec![0],
                            rhs_group_dimensions: vec![],
                        },
                    ),
                };
                let lhs = tensor(&ctx, &lhs_shape, ElementType::F32);
                let rhs = tensor(&ctx, &rhs_shape, ElementType::F32);
                let groups = tensor(&ctx, &group_shape, ElementType::I32);
                let result_shape = if mode == 2 {
                    vec![count, batch, m, 5]
                } else {
                    vec![batch, m, 5]
                };
                let result_type = tensor(&ctx, &result_shape, ElementType::F32);
                let (module, fun) = function(&mut ctx, vec![lhs, rhs, groups], vec![result_type]);
                let args = fun.arguments(&ctx);
                let op = RaggedDotOp::new_inferred(
                    &mut ctx,
                    args[0],
                    args[1],
                    args[2],
                    RaggedDotOptions {
                        dimension_numbers: RaggedDotDimensionNumbersAttr::new(d),
                        precision_config: Some(vec![Precision::High, Precision::Highest]),
                    },
                )
                .unwrap();
                attach(&mut ctx, &fun, op);
                check(&mut ctx, &module);
            }
        }
    }
}
