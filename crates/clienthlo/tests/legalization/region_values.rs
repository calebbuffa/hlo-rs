use super::tensors::{Tensor, execute};
use super::*;
use clienthlo::attributes::{RaggedDotDimensionNumbersAttr, RaggedDotDimensions};

#[test]
fn emitted_scan_values_match_forward_and_reverse_recurrences() {
    for reverse in [false, true] {
        for dynamic in [false, true] {
            let mut ctx = Context::new();
            let ty = tensor(&ctx, &[2, if dynamic { -1 } else { 3 }], ElementType::F64);
            let carry = tensor(&ctx, &[2], ElementType::F64);
            let (module, fun) = function(&mut ctx, vec![ty, carry], vec![ty, carry]);
            let args = fun.arguments(&ctx);
            let scan = ScanOp::new_inferred(
                &mut ctx,
                vec![args[0]],
                vec![args[1]],
                1,
                None,
                reverse,
                None,
                |ctx, block| {
                    let args: Vec<_> = block.deref(ctx).arguments().collect();
                    let op = BroadcastAddOp::new_inferred(ctx, args[0], args[1], None)?;
                    let value = op.get_operation().deref(ctx).get_result(0);
                    op.get_operation().insert_at_back(block, ctx);
                    ReturnOp::new_many(ctx, vec![value, value])
                        .get_operation()
                        .insert_at_back(block, ctx);
                    Ok(())
                },
            )
            .unwrap();
            attach(&mut ctx, &fun, scan);
            check(&mut ctx, &module);
            let input = Tensor::new(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
            let initial = Tensor::new(&[2], vec![0., 0.]);
            let actual = execute(&ctx, &fun, &[input, initial]);
            let expected = if reverse {
                vec![6., 5., 3., 15., 11., 6.]
            } else {
                vec![1., 3., 6., 4., 9., 15.]
            };
            assert_eq!(
                actual,
                [
                    Tensor::new(&[2, 3], expected),
                    Tensor::new(&[2], vec![6., 15.])
                ]
            );
            if dynamic {
                let result = execute(
                    &ctx,
                    &fun,
                    &[
                        Tensor::new(&[2, 0], vec![]),
                        Tensor::new(&[2], vec![3., 4.]),
                    ],
                );
                assert_eq!(
                    result,
                    [
                        Tensor::new(&[2, 0], vec![]),
                        Tensor::new(&[2], vec![3., 4.])
                    ]
                );
            }
        }
    }
}

#[test]
fn emitted_ragged_dot_values_match_grouped_products() {
    for dynamic in [false, true] {
        for mode in 1..=3 {
            let mut ctx = Context::new();
            let (left, right, group, expected, dimensions) = match mode {
                1 => (
                    Tensor::new(&[3, 2], vec![1., 2., 3., 4., 5., 6.]),
                    Tensor::new(&[2, 2, 1], vec![10., 20., 100., 200.]),
                    Tensor::new(&[2], vec![1., 2.]),
                    Tensor::new(&[3, 1], vec![50., 1100., 1700.]),
                    RaggedDotDimensions {
                        lhs_batching_dimensions: vec![],
                        rhs_batching_dimensions: vec![],
                        lhs_contracting_dimensions: vec![1],
                        rhs_contracting_dimensions: vec![1],
                        lhs_ragged_dimensions: vec![0],
                        rhs_group_dimensions: vec![0],
                    },
                ),
                2 => (
                    Tensor::new(&[1, 3], vec![1., 2., 3.]),
                    Tensor::new(&[3, 1], vec![10., 20., 30.]),
                    Tensor::new(&[2], vec![1., 2.]),
                    Tensor::new(&[2, 1, 1], vec![10., 130.]),
                    RaggedDotDimensions {
                        lhs_batching_dimensions: vec![],
                        rhs_batching_dimensions: vec![],
                        lhs_contracting_dimensions: vec![1],
                        rhs_contracting_dimensions: vec![0],
                        lhs_ragged_dimensions: vec![1],
                        rhs_group_dimensions: vec![],
                    },
                ),
                _ => (
                    Tensor::new(&[2, 1, 2], vec![1., 2., 3., 4.]),
                    Tensor::new(&[2, 2, 1], vec![10., 20., 30., 40.]),
                    Tensor::new(&[2], vec![1., 1.]),
                    Tensor::new(&[2, 1, 1], vec![50., 250.]),
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
            let shape = |value: &Tensor, group_axis: Option<usize>| {
                value
                    .shape
                    .iter()
                    .enumerate()
                    .map(|(i, &size)| {
                        if dynamic && Some(i) != group_axis {
                            -1
                        } else {
                            size as i64
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let lhs = tensor(&ctx, &shape(&left, None), ElementType::F64);
            let rhs_group = dimensions.rhs_group_dimensions.first().map(|&i| i as usize);
            let rhs = tensor(&ctx, &shape(&right, rhs_group), ElementType::F64);
            let gs = tensor(&ctx, &[2], ElementType::I32);
            let output = tensor(
                &ctx,
                &shape(&expected, if mode == 2 { Some(0) } else { None }),
                ElementType::F64,
            );
            let (module, fun) = function(&mut ctx, vec![lhs, rhs, gs], vec![output]);
            let args = fun.arguments(&ctx);
            let op = RaggedDotOp::new(
                &mut ctx,
                args[0],
                args[1],
                args[2],
                output,
                RaggedDotOptions {
                    dimension_numbers: RaggedDotDimensionNumbersAttr::new(dimensions),
                    precision_config: None,
                },
            );
            attach(&mut ctx, &fun, op);
            check(&mut ctx, &module);
            assert_eq!(execute(&ctx, &fun, &[left, right, group]), [expected]);
        }
    }
}

#[test]
fn dynamic_scan_outputs_keep_their_own_static_dimensions() {
    let mut ctx = Context::new();
    let input = tensor(&ctx, &[-1], ElementType::F64);
    let carry = tensor(&ctx, &[2], ElementType::F64);
    let output = tensor(&ctx, &[-1, 2], ElementType::F64);
    let (module, fun) = function(&mut ctx, vec![input, carry], vec![output, carry]);
    let args = fun.arguments(&ctx);
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![args[0]],
        vec![args[1]],
        0,
        None,
        false,
        None,
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
    assert_eq!(
        execute(
            &ctx,
            &fun,
            &[
                Tensor::new(&[3], vec![1., 2., 3.]),
                Tensor::new(&[2], vec![10., 20.])
            ]
        ),
        [
            Tensor::new(&[3, 2], vec![11., 21., 13., 23., 16., 26.]),
            Tensor::new(&[2], vec![16., 26.])
        ]
    );
    assert_eq!(
        execute(
            &ctx,
            &fun,
            &[Tensor::new(&[0], vec![]), Tensor::new(&[2], vec![10., 20.])]
        ),
        [
            Tensor::new(&[0, 2], vec![]),
            Tensor::new(&[2], vec![10., 20.])
        ]
    );
}

#[test]
fn ragged_group_sizes_broadcast_over_the_actual_batch_axis() {
    let mut ctx = Context::new();
    let lhs = tensor(&ctx, &[3, 2, 1], ElementType::F64);
    let rhs = tensor(&ctx, &[2, 2, 1, 1], ElementType::F64);
    let groups = tensor(&ctx, &[2, 2], ElementType::I32);
    let out = tensor(&ctx, &[2, 3, 1], ElementType::F64);
    let (module, fun) = function(&mut ctx, vec![lhs, rhs, groups], vec![out]);
    let args = fun.arguments(&ctx);
    let op = RaggedDotOp::new(
        &mut ctx,
        args[0],
        args[1],
        args[2],
        out,
        RaggedDotOptions {
            dimension_numbers: RaggedDotDimensionNumbersAttr::new(RaggedDotDimensions {
                lhs_batching_dimensions: vec![1],
                rhs_batching_dimensions: vec![1],
                lhs_contracting_dimensions: vec![2],
                rhs_contracting_dimensions: vec![2],
                lhs_ragged_dimensions: vec![0],
                rhs_group_dimensions: vec![0],
            }),
            precision_config: None,
        },
    );
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
    assert_eq!(
        execute(
            &ctx,
            &fun,
            &[
                Tensor::new(&[3, 2, 1], vec![1., 2., 3., 4., 5., 6.]),
                Tensor::new(&[2, 2, 1, 1], vec![10., 20., 100., 200.]),
                Tensor::new(&[2, 2], vec![1., 2., 2., 1.]),
            ]
        ),
        [Tensor::new(
            &[2, 3, 1],
            vec![10., 300., 500., 40., 80., 1200.]
        )]
    );
}
