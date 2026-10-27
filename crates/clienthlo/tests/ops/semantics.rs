use clienthlo::ops::{ConstantLikeOp, ConstantOp, ErfOp, ScanOp};
use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::attributes::{BoolAttr, IntegerAttr, StringAttr, VecAttr},
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};
use stablehlo::{
    attributes::{
        ComplexAttr, DenseElementsAttr, IndexTensorAttr, ScalarBitsAttr, SparseElementsAttr,
        TypeExtensionsAttr,
    },
    ops::ReturnOp,
    types::{ElementType, TensorType, tensor},
};

fn value(ctx: &mut Context, shape: &[i64], element: ElementType) -> Value {
    let ty = tensor(ctx, shape, element);
    BasicBlock::new(ctx, None, vec![ty])
        .deref(ctx)
        .arguments()
        .next()
        .unwrap()
}

fn bounded(ctx: &mut Context, shape: &[i64], bounds: &[i64]) -> Value {
    let ty = tensor(ctx, shape, ElementType::F32);
    let ty = ty
        .deref(ctx)
        .downcast_ref::<TensorType>()
        .unwrap()
        .with_encoding(
            ctx,
            Box::new(TypeExtensionsAttr {
                bounds: bounds.to_vec(),
            }),
        );
    BasicBlock::new(ctx, None, vec![ty])
        .deref(ctx)
        .arguments()
        .next()
        .unwrap()
}

fn returns(ctx: &mut Context, block: Ptr<BasicBlock>, values: Vec<Value>) -> Result<()> {
    ReturnOp::new_many(ctx, values)
        .get_operation()
        .insert_at_back(block, ctx);
    Ok(())
}

fn return_arguments(ctx: &mut Context, block: Ptr<BasicBlock>) -> Result<()> {
    let args = block.deref(ctx).arguments().collect();
    returns(ctx, block, args)
}

fn return_result(ctx: &mut Context, block: Ptr<BasicBlock>, op: &impl Op) -> Result<()> {
    let result = op.get_operation().deref(ctx).get_result(0);
    returns(ctx, block, vec![result])
}

fn result_types(ctx: &Context, op: &impl Op) -> Vec<TypeHandle> {
    op.get_operation().deref(ctx).result_types().collect()
}

#[test]
fn constants_store_only_typed_value_and_preserve_wide_integer_words() {
    let mut ctx = Context::new();
    let element = ElementType::integer(257, pliron::builtin::types::Signedness::Signless).unwrap();
    let words = vec![u64::MAX, 17, 23, 91, 1];
    let ty = tensor(&ctx, &[2], element);
    let op = ConstantOp::new(&mut ctx, ty, words.clone(), true);
    op.value(&ctx).unwrap().verify(&ctx).unwrap();
    assert!(op.verify(&ctx).is_err());
    assert_eq!(op.bits(&ctx), Some(words.clone()));
    assert_eq!(op.is_splat(&ctx), Some(true));
    let value = op.value(&ctx).unwrap();
    let dense = value.downcast_ref::<DenseElementsAttr>().unwrap();
    assert_eq!(dense.ty, result_types(&ctx, &op)[0]);
    assert_eq!(
        dense.values.0[0]
            .downcast_ref::<ScalarBitsAttr>()
            .unwrap()
            .words,
        words
    );
    assert_eq!(op.get_operation().deref(&ctx).attributes.0.len(), 1);
    assert!(ConstantOp::new_inferred(&mut ctx, vec![2], element, vec![0; 4], true).is_err());
    assert!(
        ConstantOp::new_inferred(&mut ctx, vec![2], element, vec![0, 0, 0, 0, 2], true).is_err()
    );
}

#[test]
fn constant_accepts_native_integer_float_complex_dense_and_sparse_elements() {
    let mut ctx = Context::new();
    let int_element = ElementType::I64;
    let int_ty = int_element.native(&ctx);
    let integer = IntegerAttr::new(
        int_ty
            .deref(&ctx)
            .downcast_ref::<pliron::builtin::types::IntegerType>()
            .map(|_| {
                pliron::builtin::types::IntegerType::get(
                    &ctx,
                    64,
                    pliron::builtin::types::Signedness::Signless,
                )
            })
            .unwrap(),
        APInt::from_u64(7, bw(64)),
    );
    let ty = tensor(&ctx, &[2], int_element);
    let dense = DenseElementsAttr::splat(ty, Box::new(integer));
    ConstantOp::from_value(&mut ctx, Box::new(dense))
        .unwrap()
        .verify(&ctx)
        .unwrap();

    let ty = tensor(&ctx, &[2], ElementType::F32);
    let dense = DenseElementsAttr::new(
        ty,
        VecAttr::new(vec![
            Box::new(ScalarBitsAttr::f32_bits(&ctx, f32::NAN.to_bits())),
            Box::new(ScalarBitsAttr::f32_bits(&ctx, (-0.0f32).to_bits())),
        ]),
    );
    let op = ConstantOp::from_value(&mut ctx, Box::new(dense.clone())).unwrap();
    assert_eq!(
        op.value(&ctx).unwrap().downcast_ref::<DenseElementsAttr>(),
        Some(&dense)
    );
    let complex = ComplexAttr {
        ty: ElementType::C128.native(&ctx),
        real: ScalarBitsAttr::float_bits(&ctx, 1f64.to_bits()),
        imaginary: ScalarBitsAttr::float_bits(&ctx, (-2f64).to_bits()),
    };
    let complex_ty = tensor(&ctx, &[4], ElementType::C128);
    let complex_op = ConstantOp::from_value(
        &mut ctx,
        Box::new(DenseElementsAttr::splat(
            complex_ty,
            Box::new(complex.clone()),
        )),
    )
    .unwrap();
    complex_op.verify(&ctx).unwrap();
    let sparse = SparseElementsAttr {
        ty: complex_ty,
        indices: vec![IndexTensorAttr { values: vec![3] }],
        values: VecAttr::new(vec![Box::new(complex)]),
    };
    let sparse_op = ConstantOp::from_value(&mut ctx, Box::new(sparse.clone())).unwrap();
    sparse_op.verify(&ctx).unwrap();
    assert_eq!(
        sparse_op
            .value(&ctx)
            .unwrap()
            .downcast_ref::<SparseElementsAttr>(),
        Some(&sparse)
    );
    assert_eq!(sparse_op.bits(&ctx), None);
    assert_eq!(sparse_op.is_splat(&ctx), None);
}

#[test]
fn constant_rejects_bad_elements_and_result_types() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let wrong_ty = tensor(&ctx, &[3], ElementType::F32);
    let scalar: AttrObj = Box::new(ScalarBitsAttr::f32_bits(&ctx, 0));
    let dense = DenseElementsAttr::splat(ty, scalar.clone());
    let op = ConstantOp::new_with_value(&mut ctx, wrong_ty, Box::new(dense));
    assert!(op.verify(&ctx).is_err());
    assert!(ConstantOp::from_value(&mut ctx, scalar.clone()).is_err());
    let bad_count = DenseElementsAttr::new(ty, VecAttr::new(vec![scalar.clone()]));
    assert!(ConstantOp::from_value(&mut ctx, Box::new(bad_count)).is_err());
    let sparse = SparseElementsAttr {
        ty,
        indices: vec![IndexTensorAttr { values: vec![2] }],
        values: VecAttr::new(vec![scalar]),
    };
    assert!(ConstantOp::from_value(&mut ctx, Box::new(sparse)).is_err());
    let empty = tensor(&ctx, &[0], ElementType::F32);
    assert!(ConstantOp::new_inferred(&mut ctx, vec![0], ElementType::F32, vec![0], true).is_err());
    ConstantOp::from_value(
        &mut ctx,
        Box::new(DenseElementsAttr::new(empty, VecAttr::new(vec![]))),
    )
    .unwrap()
    .verify(&ctx)
    .unwrap();
    assert!(ConstantOp::new_inferred(&mut ctx, vec![-1], ElementType::F32, vec![0], true).is_err());
}

#[test]
fn constant_like_accepts_typed_float_integer_and_complex_scalars() {
    let mut ctx = Context::new();
    let operand = value(&mut ctx, &[2, 3], ElementType::I32);
    let float = ScalarBitsAttr::f32_bits(&ctx, f32::INFINITY.to_bits());
    let float_op = ConstantLikeOp::from_value(&mut ctx, Box::new(float.clone()), operand).unwrap();
    assert_eq!(
        result_types(&ctx, &float_op),
        vec![tensor(&ctx, &[2, 3], ElementType::F32)]
    );
    assert_eq!(
        float_op
            .value(&ctx)
            .unwrap()
            .downcast_ref::<ScalarBitsAttr>(),
        Some(&float)
    );
    let complex = ComplexAttr {
        ty: ElementType::C64.native(&ctx),
        real: ScalarBitsAttr::f32_bits(&ctx, 1f32.to_bits()),
        imaginary: ScalarBitsAttr::f32_bits(&ctx, 2f32.to_bits()),
    };
    ConstantLikeOp::from_value(&mut ctx, Box::new(complex), operand)
        .unwrap()
        .verify(&ctx)
        .unwrap();
    let element = ElementType::integer(129, pliron::builtin::types::Signedness::Unsigned).unwrap();
    let wide = ScalarBitsAttr {
        ty: element.native(&ctx),
        words: vec![1, 2, 1],
    };
    wide.verify(&ctx).unwrap();
    assert!(ConstantLikeOp::from_value(&mut ctx, Box::new(wide), operand).is_err());
    assert!(ConstantLikeOp::new_inferred(&mut ctx, vec![1], element, operand).is_err());
    assert!(ConstantLikeOp::new_inferred(&mut ctx, vec![1, 2, 2], element, operand).is_err());
    let bool_op = ConstantLikeOp::from_value(&mut ctx, Box::new(BoolAttr::new(true)), operand);
    assert!(bool_op.is_ok());
}

#[test]
fn constant_like_preserves_bounds_and_rejects_non_scalar_or_wrong_result() {
    let mut ctx = Context::new();
    let operand = bounded(&mut ctx, &[-1, 3], &[7, -1]);
    let scalar: AttrObj = Box::new(ScalarBitsAttr::f32_bits(&ctx, 0));
    let op = ConstantLikeOp::from_value(&mut ctx, scalar.clone(), operand).unwrap();
    assert_eq!(result_types(&ctx, &op), vec![operand.get_type(&ctx)]);
    let ty = tensor(&ctx, &[2], ElementType::F32);
    assert!(
        ConstantLikeOp::from_value(
            &mut ctx,
            Box::new(DenseElementsAttr::splat(ty, scalar.clone())),
            operand,
        )
        .is_err()
    );
    assert!(
        ConstantLikeOp::from_value(&mut ctx, Box::new(StringAttr::new("bad".into())), operand)
            .is_err()
    );
    let wrong = tensor(&ctx, &[-1, 3], ElementType::I32);
    assert!(
        ConstantLikeOp::new_with_value(&mut ctx, scalar, operand, wrong)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn scan_infers_output_count_and_types_from_completed_body() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[4, 3], ElementType::F32);
    let init = value(&mut ctx, &[], ElementType::I32);
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![init],
        0,
        None,
        true,
        Some(false),
        |ctx, block| {
            let args: Vec<_> = block.deref(ctx).arguments().collect();
            returns(ctx, block, vec![args[0], args[0], args[1]])
        },
    )
    .unwrap();
    scan.verify(&ctx).unwrap();
    assert_eq!(
        result_types(&ctx, &scan),
        vec![
            tensor(&ctx, &[4, 3], ElementType::F32),
            tensor(&ctx, &[4, 3], ElementType::F32),
            tensor(&ctx, &[], ElementType::I32),
        ]
    );
    assert_eq!(scan.is_reverse(&ctx), Some(true));
    assert_eq!(scan.is_associative(&ctx), Some(false));
    assert_eq!(
        scan.get_attr_chlo_scan_result_segment_sizes(&ctx)
            .unwrap()
            .0
            .len(),
        2
    );
}

#[test]
fn scan_carries_come_from_body_not_init_types() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[4, 3], ElementType::F32);
    let init = value(&mut ctx, &[], ElementType::F32);
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![init],
        0,
        None,
        false,
        None,
        |ctx, block| {
            let ty = tensor(ctx, &[2], ElementType::I32);
            let c = ConstantOp::new(ctx, ty, vec![7], true);
            c.get_operation().insert_at_back(block, ctx);
            return_result(ctx, block, &c)
        },
    )
    .unwrap();
    assert_eq!(
        result_types(&ctx, &scan),
        vec![tensor(&ctx, &[2], ElementType::I32)]
    );
    scan.verify(&ctx).unwrap();
}

#[test]
fn scan_without_inputs_reinserts_explicit_size_and_supports_dynamic_size() {
    let mut ctx = Context::new();
    for size in [Some(5), None, Some(0)] {
        let scan = ScanOp::new_inferred(
            &mut ctx,
            vec![],
            vec![],
            1,
            size,
            false,
            None,
            |ctx, block| {
                let ty = tensor(ctx, &[3], ElementType::F32);
                let c = ConstantOp::new(ctx, ty, vec![0], true);
                c.get_operation().insert_at_back(block, ctx);
                return_result(ctx, block, &c)
            },
        )
        .unwrap();
        let dimension = size.map_or(-1, |n| n as i64);
        assert_eq!(
            result_types(&ctx, &scan),
            vec![tensor(&ctx, &[3, dimension], ElementType::F32)]
        );
        scan.verify(&ctx).unwrap();
    }
}

#[test]
fn scan_preserves_and_reinserts_body_and_input_bounds() {
    let mut ctx = Context::new();
    let first = bounded(&mut ctx, &[-1, -1], &[12, 7]);
    let second = bounded(&mut ctx, &[-1, -1], &[9, 5]);
    let carry = bounded(&mut ctx, &[-1], &[6]);
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![first, second],
        vec![carry],
        0,
        None,
        false,
        None,
        |ctx, block| {
            let args: Vec<_> = block.deref(ctx).arguments().collect();
            assert_eq!(
                args[0]
                    .get_type(ctx)
                    .deref(ctx)
                    .downcast_ref::<TensorType>()
                    .unwrap()
                    .bounds(),
                Some(&[7][..])
            );
            returns(ctx, block, vec![args[0], args[2]])
        },
    )
    .unwrap();
    let types = result_types(&ctx, &scan);
    let output = types[0].deref(&ctx);
    let output = output.downcast_ref::<TensorType>().unwrap();
    assert_eq!(output.dimensions, vec![-1, -1]);
    assert_eq!(output.bounds(), Some(&[9, 7][..]));
    assert_eq!(types[1], carry.get_type(&ctx));
    scan.verify(&ctx).unwrap();
}

#[test]
fn scan_borrows_dimension_bound_when_body_result_has_no_encoding() {
    let mut ctx = Context::new();
    let input = bounded(&mut ctx, &[-1, 3], &[13, -1]);
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![],
        0,
        None,
        false,
        None,
        |ctx, block| {
            let ty = tensor(ctx, &[2], ElementType::F32);
            let c = ConstantOp::new(ctx, ty, vec![0], true);
            c.get_operation().insert_at_back(block, ctx);
            return_result(ctx, block, &c)
        },
    )
    .unwrap();
    let ty = result_types(&ctx, &scan)[0].deref(&ctx);
    let ty = ty.downcast_ref::<TensorType>().unwrap();
    assert_eq!(ty.dimensions, vec![-1, 2]);
    assert_eq!(ty.bounds(), Some(&[13, -1][..]));
}

#[test]
fn scan_rejects_dimension_mismatch_and_overflow_without_panicking() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    let other = value(&mut ctx, &[4, 3], ElementType::F32);
    for (inputs, dimension, size) in [
        (vec![input], 2, None),
        (vec![input], u64::MAX, None),
        (vec![input], 0, Some(u64::MAX)),
        (vec![input], 0, Some(4)),
        (vec![input, other], 0, None),
    ] {
        assert!(
            ScanOp::new_inferred(
                &mut ctx,
                inputs,
                vec![],
                dimension,
                size,
                false,
                None,
                return_arguments,
            )
            .is_err()
        );
    }
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![],
            vec![],
            2,
            Some(4),
            false,
            None,
            |ctx, block| {
                let ty = tensor(ctx, &[], ElementType::F32);
                let c = ConstantOp::new(ctx, ty, vec![0], true);
                c.get_operation().insert_at_back(block, ctx);
                return_result(ctx, block, &c)
            },
        )
        .is_err()
    );
}

#[test]
fn scan_rejects_empty_body_missing_carries_and_absent_inputs_and_outputs() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    let init = value(&mut ctx, &[], ElementType::F32);
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![],
            0,
            None,
            false,
            None,
            |_, _| Ok(()),
        )
        .is_err()
    );
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![init],
            0,
            None,
            false,
            None,
            |ctx, block| returns(ctx, block, vec![]),
        )
        .is_err()
    );
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![],
            vec![init],
            0,
            Some(3),
            false,
            None,
            return_arguments,
        )
        .is_err()
    );
}

#[test]
fn scan_rejects_captures_in_return_and_nested_body_operations() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    let external = value(&mut ctx, &[3], ElementType::F32);
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![],
            0,
            None,
            false,
            None,
            |ctx, block| returns(ctx, block, vec![external]),
        )
        .is_err()
    );
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![],
            0,
            None,
            false,
            None,
            |ctx, block| {
                let erf = ErfOp::new_inferred(ctx, external)?;
                erf.get_operation().insert_at_back(block, ctx);
                return_result(ctx, block, &erf)
            },
        )
        .is_err()
    );
}

#[test]
fn scan_checks_body_signature_operations_and_final_stored_result_independently() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![],
            0,
            None,
            false,
            None,
            |ctx, block| {
                let arg = block.deref(ctx).arguments().next().unwrap();
                arg.set_type(ctx, tensor(ctx, &[4], ElementType::F32));
                returns(ctx, block, vec![arg])
            },
        )
        .is_err()
    );
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![],
            0,
            None,
            false,
            None,
            |ctx, block| {
                let arg = block.deref(ctx).arguments().next().unwrap();
                let invalid_ty = tensor(ctx, &[3], ElementType::I32);
                let erf = ErfOp::new(ctx, arg, invalid_ty);
                erf.get_operation().insert_at_back(block, ctx);
                return_result(ctx, block, &erf)
            },
        )
        .is_err()
    );
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![],
        0,
        None,
        false,
        None,
        return_arguments,
    )
    .unwrap();
    let result = scan.get_operation().deref(&ctx).get_result(0);
    result.set_type(&ctx, tensor(&ctx, &[2, 4], ElementType::F32));
    assert!(scan.verify(&ctx).is_err());
}

#[test]
fn scan_rejects_bad_segments_options_and_multiple_blocks() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![],
        0,
        None,
        false,
        None,
        return_arguments,
    )
    .unwrap();
    scan.set_attr_chlo_scan_operand_segment_sizes(&ctx, VecAttr::new(vec![]));
    assert!(scan.verify(&ctx).is_err());
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![],
        0,
        None,
        false,
        None,
        return_arguments,
    )
    .unwrap();
    scan.get_operation().deref_mut(&ctx).attributes.set(
        pliron::ident!("chlo_scan_is_reverse"),
        StringAttr::new("false".into()),
    );
    assert!(scan.verify(&ctx).is_err());
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![],
        0,
        None,
        false,
        None,
        return_arguments,
    )
    .unwrap();
    let block = BasicBlock::new(&mut ctx, None, vec![]);
    block.insert_at_back(scan.get_operation().deref(&ctx).get_region(0), &ctx);
    assert!(scan.verify(&ctx).is_err());
}

fn quantized(ctx: &Context, axis: Option<usize>) -> TypeHandle {
    stablehlo::types::QuantizedElementType::intern(
        ctx,
        stablehlo::types::UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: Some(-8),
            storage_max: Some(7),
            expressed_type: ElementType::F32,
            quantization_dimension: axis,
            scales: if axis.is_some() {
                vec![0.25, 0.5]
            } else {
                vec![0.25]
            },
            zero_points: if axis.is_some() { vec![0, 0] } else { vec![0] },
        },
    )
}

#[test]
fn constants_accept_quantized_dense_sparse_storage_and_validate_domain() {
    let mut ctx = Context::new();
    let q = quantized(&ctx, None);
    let ty: TypeHandle = TensorType::get(&ctx, vec![2], q).into();
    let scalar = ScalarBitsAttr {
        ty: ElementType::I8.native(&ctx),
        words: vec![7],
    };
    let dense = DenseElementsAttr::splat(ty, Box::new(scalar.clone()));
    ConstantOp::from_value(&mut ctx, Box::new(dense))
        .unwrap()
        .verify(&ctx)
        .unwrap();
    ConstantOp::new(&mut ctx, ty, vec![7], true)
        .verify(&ctx)
        .unwrap();
    let sparse = SparseElementsAttr {
        ty,
        indices: vec![IndexTensorAttr { values: vec![1] }],
        values: VecAttr::new(vec![Box::new(scalar)]),
    };
    ConstantOp::from_value(&mut ctx, Box::new(sparse))
        .unwrap()
        .verify(&ctx)
        .unwrap();
    assert!(
        ConstantOp::new(&mut ctx, ty, vec![8], true)
            .verify(&ctx)
            .is_err()
    );
    let per_axis = quantized(&ctx, Some(0));
    let ty: TypeHandle = TensorType::get(&ctx, vec![2], per_axis).into();
    assert!(
        ConstantOp::new(&mut ctx, ty, vec![7], true)
            .verify(&ctx)
            .is_err()
    );
    for element in [
        ElementType::Index,
        ElementType::Si32,
        ElementType::Ui1,
        ElementType::Tf32,
        ElementType::complex(ElementType::F16).unwrap(),
    ] {
        let ty = tensor(&ctx, &[2], element);
        assert!(
            ConstantOp::new(&mut ctx, ty, vec![0], true)
                .verify(&ctx)
                .is_err()
        );
    }
}

#[test]
fn scan_accepts_quantized_inputs_carries_and_preserves_native_types() {
    let mut ctx = Context::new();
    for axis in [None, Some(0)] {
        let q = quantized(&ctx, axis);
        let input_ty: TypeHandle = TensorType::get(&ctx, vec![2, 4], q).into();
        let init_ty: TypeHandle = TensorType::get(&ctx, vec![2], q).into();
        let source = BasicBlock::new(&mut ctx, None, vec![input_ty, init_ty]);
        let args: Vec<_> = source.deref(&ctx).arguments().collect();
        let scan = ScanOp::new_inferred(
            &mut ctx,
            vec![args[0]],
            vec![args[1]],
            1,
            None,
            false,
            None,
            return_arguments,
        )
        .unwrap();
        assert_eq!(result_types(&ctx, &scan), vec![input_ty, init_ty]);
        scan.verify(&ctx).unwrap();
    }
}

#[test]
fn constant_like_checks_bounded_and_unranked_compatibility() {
    use pliron::r#type::Type;
    use stablehlo::types::UnrankedTensorType;
    let mut ctx = Context::new();
    let operand = bounded(&mut ctx, &[-1], &[5]);
    let scalar: AttrObj = Box::new(ScalarBitsAttr::f32_bits(&ctx, 0));
    let oversize = tensor(&ctx, &[6], ElementType::F32);
    assert!(
        ConstantLikeOp::new_with_value(&mut ctx, scalar.clone(), operand, oversize)
            .verify(&ctx)
            .is_err()
    );
    let compatible = tensor(&ctx, &[4], ElementType::F32);
    ConstantLikeOp::new_with_value(&mut ctx, scalar.clone(), operand, compatible)
        .verify(&ctx)
        .unwrap();
    let unranked: TypeHandle = UnrankedTensorType::instantiate(
        UnrankedTensorType {
            element: ElementType::F32.native(&ctx),
        },
        &ctx,
    )
    .into();
    ConstantLikeOp::new_with_value(&mut ctx, scalar.clone(), operand, unranked)
        .verify(&ctx)
        .unwrap();
    let source = BasicBlock::new(&mut ctx, None, vec![unranked]);
    let input = source.deref(&ctx).arguments().next().unwrap();
    let inferred = ConstantLikeOp::from_value(&mut ctx, scalar.clone(), input).unwrap();
    assert_eq!(result_types(&ctx, &inferred), vec![unranked]);
    ConstantLikeOp::new_with_value(&mut ctx, scalar, input, compatible)
        .verify(&ctx)
        .unwrap();
}

#[test]
fn scan_defaults_reverse_to_false_and_requires_signless_nonnegative_i64() {
    use pliron::builtin::types::{IntegerType, Signedness};
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![],
        0,
        None,
        false,
        None,
        return_arguments,
    )
    .unwrap();
    let dimension = scan.get_attr_chlo_scan_dimension(&ctx).unwrap();
    assert_eq!(
        dimension.get_type().deref(&ctx).signedness(),
        Signedness::Signless
    );
    drop(dimension);
    scan.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&pliron::ident!("chlo_scan_is_reverse"));
    assert_eq!(scan.is_reverse(&ctx), Some(false));
    scan.verify(&ctx).unwrap();
    let wrong = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    scan.set_attr_chlo_scan_dimension(&ctx, IntegerAttr::new(wrong, APInt::from_u64(0, bw(64))));
    assert!(scan.verify(&ctx).is_err());
    let signless = IntegerType::get(&ctx, 64, Signedness::Signless);
    scan.set_attr_chlo_scan_dimension(
        &ctx,
        IntegerAttr::new(signless, APInt::from_i64(-1, bw(64))),
    );
    assert!(scan.verify(&ctx).is_err());
}

#[test]
fn scan_rejects_forward_references_and_sibling_region_scope_leaks() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![],
            0,
            None,
            false,
            None,
            |ctx, block| {
                let ty = tensor(ctx, &[3], ElementType::F32);
                let constant = ConstantOp::new(ctx, ty, vec![0], true);
                let late = constant.get_operation().deref(ctx).get_result(0);
                let early = ErfOp::new_inferred(ctx, late)?;
                early.get_operation().insert_at_back(block, ctx);
                constant.get_operation().insert_at_back(block, ctx);
                return_result(ctx, block, &early)
            },
        )
        .is_err()
    );
    assert!(
        ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![],
            0,
            None,
            false,
            None,
            |ctx, block| {
                let first = stablehlo::ops::ModuleOp::new(ctx);
                first.get_operation().insert_at_back(block, ctx);
                let first_body = first.body_block(ctx).unwrap();
                let ty = tensor(ctx, &[3], ElementType::F32);
                let constant = ConstantOp::new(ctx, ty, vec![0], true);
                constant.get_operation().insert_at_back(first_body, ctx);
                returns(ctx, first_body, vec![])?;
                let escaped = constant.get_operation().deref(ctx).get_result(0);
                let second = stablehlo::ops::ModuleOp::new(ctx);
                second.get_operation().insert_at_back(block, ctx);
                let second_body = second.body_block(ctx).unwrap();
                let illegal = ErfOp::new_inferred(ctx, escaped)?;
                illegal.get_operation().insert_at_back(second_body, ctx);
                returns(ctx, second_body, vec![])?;
                return_arguments(ctx, block)
            },
        )
        .is_err()
    );
}

#[test]
fn native_folding_interface_returns_canonical_values_and_static_splats() {
    use pliron::{op::op_cast, opts::constants::ConstFoldInterface};
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let constant = ConstantOp::new(&mut ctx, ty, vec![1f32.to_bits().into()], true);
    let fold = op_cast::<dyn ConstFoldInterface>(&constant).unwrap();
    let folded = fold.check_fold(&ctx, &[]);
    assert_eq!(folded[0], constant.value(&ctx));
    for shape in [&[2, 3][..], &[0][..], &[-1][..]] {
        let input = value(&mut ctx, shape, ElementType::I32);
        let scalar: AttrObj = Box::new(ScalarBitsAttr::f32_bits(&ctx, 1f32.to_bits()));
        let like = ConstantLikeOp::from_value(&mut ctx, scalar, input).unwrap();
        let fold = op_cast::<dyn ConstFoldInterface>(&like).unwrap();
        let folded = fold.check_fold(&ctx, &[None]);
        if shape.contains(&-1) {
            assert!(folded[0].is_none());
        } else {
            let attr = folded[0].as_ref().unwrap();
            let dense = attr.downcast_ref::<DenseElementsAttr>().unwrap();
            dense.verify(&ctx).unwrap();
            assert_eq!(dense.is_splat, !shape.contains(&0));
        }
    }
}

#[test]
fn constant_like_native_folding_materializes_and_replaces_uses() {
    use pliron::{
        irbuild::{
            inserter::{Inserter, OpInsertionPoint},
            listener::DummyListener,
            rewriter::IRRewriter,
        },
        opts::constants::ConstFoldInterface,
    };
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let block = BasicBlock::new(&mut ctx, None, vec![ty]);
    let input = block.deref(&ctx).arguments().next().unwrap();
    let scalar: AttrObj = Box::new(ScalarBitsAttr::f32_bits(&ctx, 1f32.to_bits()));
    let like = ConstantLikeOp::from_value(&mut ctx, scalar, input).unwrap();
    like.get_operation().insert_at_back(block, &ctx);
    let old_result = like.get_operation().deref(&ctx).get_result(0);
    let consumer = ReturnOp::new(&mut ctx, old_result);
    consumer.get_operation().insert_at_back(block, &ctx);
    let mut rewriter = IRRewriter::<DummyListener>::default();
    rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(like.get_operation()));
    assert_eq!(
        like.fold_in_place(&mut ctx, &[None], &mut rewriter),
        pliron::irbuild::IRStatus::Changed,
    );
    let replacement = consumer.get_operation().deref(&ctx).get_operand(0);
    assert_ne!(replacement, old_result);
    let replacement = replacement.defining_op().unwrap();
    let replacement = pliron::operation::Operation::get_op_dyn(replacement, &ctx);
    replacement
        .downcast_ref::<ConstantOp>()
        .unwrap()
        .verify(&ctx)
        .unwrap();
}

#[test]
fn scan_explicit_results_follow_upstream_shape_only_compatibility() {
    use pliron::r#type::Type;
    use stablehlo::types::UnrankedTensorType;
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    for result in [
        tensor(&ctx, &[2, 3], ElementType::I32),
        UnrankedTensorType::instantiate(
            UnrankedTensorType {
                element: ElementType::F32.native(&ctx),
            },
            &ctx,
        )
        .into(),
    ] {
        let scan = ScanOp::new(
            &mut ctx,
            vec![input],
            vec![],
            vec![result],
            0,
            None,
            false,
            None,
        );
        let block = scan.entry_block(&ctx).unwrap();
        return_arguments(&mut ctx, block).unwrap();
        scan.verify(&ctx).unwrap();
    }
}

#[test]
fn scan_body_quantized_arguments_allow_different_scales_but_not_storage_ranges() {
    use pliron::r#type::Type;
    use stablehlo::types::QuantizedElementType;
    let mut ctx = Context::new();
    let q = quantized(&ctx, None);
    let input_ty: TypeHandle = TensorType::get(&ctx, vec![2, 3], q).into();
    let source = BasicBlock::new(&mut ctx, None, vec![input_ty]);
    let input = source.deref(&ctx).arguments().next().unwrap();
    for (scale, minimum, valid) in [(0.5f64, -8i128, true), (0.25, -7, false)] {
        let mut changed = q
            .deref(&ctx)
            .downcast_ref::<QuantizedElementType>()
            .unwrap()
            .clone();
        changed.scale_bits = vec![scale.to_bits()];
        changed.storage_min = vec![minimum.into()];
        let changed: TypeHandle = QuantizedElementType::instantiate(changed, &ctx).into();
        let argument_ty: TypeHandle = TensorType::get(&ctx, vec![3], changed).into();
        let result = ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![],
            0,
            None,
            false,
            None,
            |ctx, block| {
                let arg = block.deref(ctx).arguments().next().unwrap();
                arg.set_type(ctx, argument_ty);
                returns(ctx, block, vec![arg])
            },
        );
        assert_eq!(result.is_ok(), valid);
    }
}

#[test]
fn scan_rejects_non_hlo_native_tensor_domains() {
    let mut ctx = Context::new();
    for element in [
        ElementType::Index,
        ElementType::Si32,
        ElementType::Ui1,
        ElementType::Tf32,
        ElementType::complex(ElementType::F16).unwrap(),
    ] {
        let input = value(&mut ctx, &[2, 3], element);
        assert!(
            ScanOp::new_inferred(
                &mut ctx,
                vec![input],
                vec![],
                0,
                None,
                false,
                None,
                return_arguments,
            )
            .is_err()
        );
    }
}

#[test]
fn native_float_attributes_normalize_losslessly_in_elements_and_folded_splats() {
    use pliron::{
        builtin::attributes::{FPDoubleAttr, FPSingleAttr},
        opts::constants::ConstFoldInterface,
    };
    let mut ctx = Context::new();
    let single = FPSingleAttr::from(-0.0f32);
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let dense = DenseElementsAttr::splat(ty, Box::new(single.clone()));
    let constant = ConstantOp::from_value(&mut ctx, Box::new(dense)).unwrap();
    assert_eq!(
        constant.bits(&ctx),
        Some(vec![u64::from((-0.0f32).to_bits())])
    );
    let input = value(&mut ctx, &[2], ElementType::I32);
    let like = ConstantLikeOp::from_value(&mut ctx, Box::new(single), input).unwrap();
    let folded = like.check_fold(&ctx, &[None]);
    let dense = folded[0]
        .as_ref()
        .unwrap()
        .downcast_ref::<DenseElementsAttr>()
        .unwrap();
    assert_eq!(
        dense.values.0[0]
            .downcast_ref::<ScalarBitsAttr>()
            .unwrap()
            .words,
        vec![u64::from((-0.0f32).to_bits())]
    );
    let bits = 0x7ff8_0000_0000_0042;
    let double = FPDoubleAttr::from(f64::from_bits(bits));
    let ty = tensor(&ctx, &[3], ElementType::F64);
    let sparse = SparseElementsAttr {
        ty,
        indices: vec![IndexTensorAttr { values: vec![1] }],
        values: VecAttr::new(vec![Box::new(double)]),
    };
    let constant = ConstantOp::from_value(&mut ctx, Box::new(sparse)).unwrap();
    let attr = constant.value(&ctx).unwrap();
    let sparse = attr.downcast_ref::<SparseElementsAttr>().unwrap();
    assert_eq!(
        sparse.values.0[0]
            .downcast_ref::<ScalarBitsAttr>()
            .unwrap()
            .words,
        vec![bits]
    );
    constant.verify(&ctx).unwrap();
}

#[test]
fn scan_drops_vacuous_bounds_and_static_dimensions_do_not_get_bounds() {
    let mut ctx = Context::new();
    let input = bounded(&mut ctx, &[-1, 3], &[8, -1]);
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![],
        0,
        Some(5),
        false,
        None,
        |ctx, block| {
            let arg = block.deref(ctx).arguments().next().unwrap();
            assert!(
                inference_shape(ctx, arg.get_type(ctx))
                    .encoding
                    .0
                    .is_empty()
            );
            returns(ctx, block, vec![arg])
        },
    )
    .unwrap();
    let ty = result_types(&ctx, &scan)[0];
    let result = inference_shape(&ctx, ty);
    assert_eq!(result.dimensions, Some(vec![5, 3]));
    assert!(result.encoding.0.is_empty());
}

fn inference_shape(ctx: &Context, ty: TypeHandle) -> super::TensorMetadata {
    super::tensor_metadata(ctx, ty)
}

#[test]
fn scan_and_constant_native_assembly_preserve_body_values_types_and_options() {
    use pliron::{
        builtin::op_interfaces::SingleBlockRegionInterface, irfmt::parsers::spaced,
        linked_list::ContainsLinkedList, parsable::parse_from_str, printable::Printable,
    };
    let mut ctx = Context::new();
    let module = pliron::builtin::ops::ModuleOp::new(&mut ctx, pliron::ident!("scan_test"));
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![],
        vec![],
        0,
        Some(5),
        true,
        Some(false),
        |ctx, block| {
            let ty = tensor(ctx, &[], ElementType::F32);
            let constant = ConstantOp::new(ctx, ty, vec![1f32.to_bits().into()], true);
            constant.get_operation().insert_at_back(block, ctx);
            let input = constant.get_operation().deref(ctx).get_result(0);
            let like = ConstantLikeOp::from_value(
                ctx,
                Box::new(ScalarBitsAttr {
                    ty: ElementType::F32.native(ctx),
                    words: vec![2f32.to_bits().into()],
                }),
                input,
            )?;
            like.get_operation().insert_at_back(block, ctx);
            return_result(ctx, block, &like)
        },
    )
    .unwrap();
    scan.get_operation()
        .insert_at_back(module.get_body(&ctx, 0), &ctx);
    let text = module.get_operation().disp(&ctx).to_string();
    let mut parsed_ctx = Context::new();
    let parsed = parse_from_str(
        spaced(pliron::operation::Operation::top_level_parser()),
        &mut parsed_ctx,
        &text,
    )
    .unwrap_or_else(|error| panic!("scan assembly failed to parse:\n{text}\n{error}"));
    let region = parsed.deref(&parsed_ctx).get_region(0);
    let body = region.deref(&parsed_ctx).iter(&parsed_ctx).next().unwrap();
    let parsed_op = body.deref(&parsed_ctx).iter(&parsed_ctx).next().unwrap();
    let dynamic = pliron::operation::Operation::get_op_dyn(parsed_op, &parsed_ctx);
    let parsed_scan = dynamic.downcast_ref::<ScanOp>().unwrap();
    parsed_scan.verify(&parsed_ctx).unwrap();
    assert_eq!(parsed_scan.scan_dim_size(&parsed_ctx), Some(5));
    assert_eq!(parsed_scan.is_reverse(&parsed_ctx), Some(true));
    assert_eq!(parsed_scan.is_associative(&parsed_ctx), Some(false));
    assert_eq!(
        result_types(&parsed_ctx, parsed_scan),
        vec![tensor(&parsed_ctx, &[5], ElementType::F32)]
    );
    let printed = parsed.disp(&parsed_ctx).to_string();
    let mut second_ctx = Context::new();
    let second = parse_from_str(
        spaced(pliron::operation::Operation::top_level_parser()),
        &mut second_ctx,
        &printed,
    )
    .unwrap();
    let region = second.deref(&second_ctx).get_region(0);
    let body = region.deref(&second_ctx).iter(&second_ctx).next().unwrap();
    let second_op = body.deref(&second_ctx).iter(&second_ctx).next().unwrap();
    pliron::operation::Operation::get_op_dyn(second_op, &second_ctx)
        .downcast_ref::<ScanOp>()
        .unwrap()
        .verify(&second_ctx)
        .unwrap();
}

#[test]
fn constant_like_typed_quantized_attribute_uses_exact_value_type() {
    use pliron::builtin::attributes::TypeAttr;
    use pliron::r#type::Type;
    use stablehlo::types::QuantizedElementType;
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2], ElementType::F32);
    let q = quantized(&ctx, None);
    let op = ConstantLikeOp::from_value(&mut ctx, Box::new(TypeAttr::new(q)), input).unwrap();
    let ty: TypeHandle = TensorType::get(&ctx, vec![2], q).into();
    assert_eq!(result_types(&ctx, &op), vec![ty]);
    op.verify(&ctx).unwrap();
    let mut changed = q
        .deref(&ctx)
        .downcast_ref::<QuantizedElementType>()
        .unwrap()
        .clone();
    changed.scale_bits = vec![0.5f64.to_bits()];
    let changed: TypeHandle = QuantizedElementType::instantiate(changed, &ctx).into();
    let compatible_but_not_equal: TypeHandle = TensorType::get(&ctx, vec![2], changed).into();
    let invalid = ConstantLikeOp::new_with_value(
        &mut ctx,
        Box::new(TypeAttr::new(q)),
        input,
        compatible_but_not_equal,
    );
    assert!(invalid.verify(&ctx).is_err());
}

#[test]
fn scan_allows_compatible_unranked_body_arguments_but_requires_ranked_operands_and_returns() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[3, 2], ElementType::F32);
    let init = value(&mut ctx, &[2], ElementType::F32);
    let unranked =
        stablehlo::types::UnrankedTensorType::get(&ctx, ElementType::F32.native(&ctx)).into();
    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![init],
        0,
        None,
        false,
        None,
        |ctx, block| {
            let args: Vec<_> = block.deref(ctx).arguments().collect();
            for arg in args {
                arg.set_type(ctx, unranked);
            }
            let ty = tensor(ctx, &[2], ElementType::F32);
            let constant = ConstantOp::new(ctx, ty, vec![0], true);
            constant.get_operation().insert_at_back(block, ctx);
            let result = constant.get_operation().deref(ctx).get_result(0);
            returns(ctx, block, vec![result, result])
        },
    )
    .unwrap();
    scan.verify(&ctx).unwrap();
    assert_eq!(
        result_types(&ctx, &scan),
        vec![
            tensor(&ctx, &[3, 2], ElementType::F32),
            tensor(&ctx, &[2], ElementType::F32),
        ]
    );
    let arg = scan
        .entry_block(&ctx)
        .unwrap()
        .deref(&ctx)
        .arguments()
        .next()
        .unwrap();
    let wrong_element =
        stablehlo::types::UnrankedTensorType::get(&ctx, ElementType::I32.native(&ctx)).into();
    arg.set_type(&ctx, wrong_element);
    assert!(scan.verify(&ctx).is_err());
    arg.set_type(&ctx, unranked);
    scan.verify(&ctx).unwrap();

    let input_ty = input.get_type(&ctx);
    input.set_type(&ctx, unranked);
    assert!(scan.verify(&ctx).is_err());
    input.set_type(&ctx, input_ty);
    let init_ty = init.get_type(&ctx);
    init.set_type(&ctx, unranked);
    assert!(scan.verify(&ctx).is_err());
    init.set_type(&ctx, init_ty);
    scan.verify(&ctx).unwrap();

    let error = ScanOp::new_inferred(
        &mut ctx,
        vec![input],
        vec![],
        0,
        None,
        false,
        None,
        |ctx, block| {
            let arg = block.deref(ctx).arguments().next().unwrap();
            arg.set_type(ctx, unranked);
            returns(ctx, block, vec![arg])
        },
    );
    assert!(error.is_err());
}
