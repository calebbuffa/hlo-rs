use pliron::{
    common_traits::Verify,
    context::Context,
    op::Op,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use stablehlo::{attributes::*, ops::*, types::*};

fn value(ctx: &mut Context, ty: TypeHandle) -> Value {
    ConstantOp::new(ctx, ty, vec![0], true)
        .get_operation()
        .deref(ctx)
        .get_result(0)
}
fn shaped_value(ctx: &mut Context, shape: &[i64], element: ElementType) -> Value {
    let ty = tensor(ctx, shape, element);
    value(ctx, ty)
}

#[test]
fn broadcast_and_transpose_infer_shapes_and_reject_invalid_axes() {
    let mut ctx = Context::new();
    let source = shaped_value(&mut ctx, &[2, 3], ElementType::F32);
    let broadcast = BroadcastOp::new_inferred(&mut ctx, source, vec![4]).unwrap();
    assert_eq!(
        tensor_parts(
            &ctx,
            broadcast
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx)
        )
        .unwrap()
        .0,
        vec![4, 2, 3]
    );
    assert!(broadcast.verify(&ctx).is_ok());
    let transposed = TransposeOp::new_inferred(&mut ctx, source, vec![1, 0]).unwrap();
    assert_eq!(
        tensor_parts(
            &ctx,
            transposed
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx)
        )
        .unwrap()
        .0,
        vec![3, 2]
    );
    assert!(transposed.verify(&ctx).is_ok());
    assert!(TransposeOp::new_inferred(&mut ctx, source, vec![0, 0]).is_err());
}

#[test]
fn broadcast_in_dim_and_reshape_check_shapes() {
    let mut ctx = Context::new();
    let source = shaped_value(&mut ctx, &[2, 1], ElementType::I32);
    let incompatible_result = tensor(&ctx, &[4, 4, 3], ElementType::I32);
    let broadcast = BroadcastInDimOp::new(&mut ctx, source, incompatible_result, vec![0, 1]);
    assert!(broadcast.verify(&ctx).is_err());
    let reshape = ReshapeOp::new_inferred(&mut ctx, source, &[3]).unwrap();
    assert!(reshape.verify(&ctx).is_err());
}

#[test]
fn slice_and_dynamic_slice_enforce_bounds_and_sizes() {
    let mut ctx = Context::new();
    let source = shaped_value(&mut ctx, &[8, 6], ElementType::F32);
    let slice =
        SliceOp::new_inferred(&mut ctx, source, vec![1, 0], vec![8, 6], vec![2, 1]).unwrap();
    assert_eq!(
        tensor_parts(
            &ctx,
            slice
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx)
        )
        .unwrap()
        .0,
        vec![4, 6]
    );
    assert!(slice.verify(&ctx).is_ok());
    assert!(SliceOp::new_inferred(&mut ctx, source, vec![0, 0], vec![9, 6], vec![1, 1]).is_err());
    let idx = shaped_value(&mut ctx, &[], ElementType::I32);
    let dyn_slice =
        DynamicSliceOp::new_inferred(&mut ctx, source, vec![idx, idx], vec![4, 3]).unwrap();
    assert!(dyn_slice.verify(&ctx).is_ok());
}

#[test]
fn pad_concat_and_reverse_cover_shape_rules() {
    let mut ctx = Context::new();
    let left = shaped_value(&mut ctx, &[2, 3], ElementType::F32);
    let right = shaped_value(&mut ctx, &[4, 3], ElementType::F32);
    let concat = ConcatenateOp::new_inferred(&mut ctx, vec![left, right], 0).unwrap();
    assert_eq!(
        tensor_parts(
            &ctx,
            concat
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx)
        )
        .unwrap()
        .0,
        vec![6, 3]
    );
    assert!(concat.verify(&ctx).is_ok());
    let pad_value = shaped_value(&mut ctx, &[], ElementType::F32);
    let pad = PadOp::new_inferred(
        &mut ctx,
        left,
        pad_value,
        vec![1, 0],
        vec![2, 0],
        vec![0, 1],
    )
    .unwrap();
    assert_eq!(
        tensor_parts(
            &ctx,
            pad.get_operation().deref(&ctx).get_result(0).get_type(&ctx)
        )
        .unwrap()
        .0,
        vec![5, 5]
    );
    assert!(pad.verify(&ctx).is_ok());
    let reverse = ReverseOp::new_inferred(&mut ctx, left, vec![1]).unwrap();
    assert!(reverse.verify(&ctx).is_ok());
    assert!(
        ReverseOp::new_inferred(&mut ctx, left, vec![1, 1])
            .unwrap()
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn dynamic_update_and_real_dynamic_slice_preserve_tensor_contracts() {
    let mut ctx = Context::new();
    let source = shaped_value(&mut ctx, &[8, 6], ElementType::F32);
    let update = shaped_value(&mut ctx, &[2, 3], ElementType::F32);
    let idx = shaped_value(&mut ctx, &[], ElementType::I32);
    let upd = DynamicUpdateSliceOp::new_inferred(&mut ctx, source, update, vec![idx, idx]).unwrap();
    assert!(upd.verify(&ctx).is_ok());
    let start = shaped_value(&mut ctx, &[2], ElementType::I32);
    let limit = shaped_value(&mut ctx, &[2], ElementType::I32);
    let stride = shaped_value(&mut ctx, &[2], ElementType::I32);
    let dyn_slice =
        RealDynamicSliceOp::new_inferred(&mut ctx, source, start, limit, stride).unwrap();
    assert!(dyn_slice.verify(&ctx).is_ok());
}

#[test]
fn dimension_size_ops_validate_dimension_and_i32_size() {
    let mut ctx = Context::new();
    let source = shaped_value(&mut ctx, &[4, 7], ElementType::F32);
    let get = GetDimensionSizeOp::new_inferred(&mut ctx, source, 1).unwrap();
    assert!(get.verify(&ctx).is_ok());
    assert!(GetDimensionSizeOp::new_inferred(&mut ctx, source, 2).is_err());
    let size = shaped_value(&mut ctx, &[], ElementType::I32);
    let set = SetDimensionSizeOp::new_inferred(&mut ctx, source, size, 1).unwrap();
    assert_eq!(
        set.get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx)
            .deref(&ctx)
            .downcast_ref::<TensorType>()
            .unwrap()
            .dimensions(),
        &[4, -1]
    );
    assert!(set.verify(&ctx).is_ok());
}

#[test]
fn gather_uses_dimension_numbers_for_result_shape() {
    let mut ctx = Context::new();
    let operand = shaped_value(&mut ctx, &[5, 7], ElementType::F32);
    let indices = shaped_value(&mut ctx, &[3, 1], ElementType::I32);
    let dims = GatherDimensionNumbersAttr {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        operand_batching_dims: vec![],
        start_indices_batching_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
    };
    let gather = GatherOp::new_inferred(&mut ctx, operand, indices, dims, vec![1, 7]).unwrap();
    assert_eq!(
        tensor_parts(
            &ctx,
            gather
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx)
        )
        .unwrap()
        .0,
        vec![3, 7]
    );
    assert!(gather.verify(&ctx).is_ok());
}

#[test]
fn torch_index_select_replaces_the_selected_axis() {
    let mut ctx = Context::new();
    let source = shaped_value(&mut ctx, &[2, 4, 3], ElementType::F32);
    let indices = shaped_value(&mut ctx, &[6], ElementType::I32);
    let op = TorchIndexSelectOp::new_inferred(&mut ctx, source, indices, 1).unwrap();
    assert_eq!(
        tensor_parts(
            &ctx,
            op.get_operation().deref(&ctx).get_result(0).get_type(&ctx)
        )
        .unwrap()
        .0,
        vec![2, 6, 3]
    );
    assert!(op.verify(&ctx).is_ok());
}
#[test]
fn dynamic_broadcast_reshape_pad_and_gather_build_native_ops() {
    let mut ctx = Context::new();
    let source = shaped_value(&mut ctx, &[2, 1], ElementType::F32);
    let output_dims = shaped_value(&mut ctx, &[3], ElementType::I64);
    let dynamic_broadcast =
        DynamicBroadcastInDimOp::new_inferred(&mut ctx, source, output_dims, vec![0, 2]).unwrap();
    assert!(dynamic_broadcast.verify(&ctx).is_ok());

    let output_shape = shaped_value(&mut ctx, &[2], ElementType::I64);
    let reshape = DynamicReshapeOp::new_inferred(&mut ctx, source, output_shape).unwrap();
    assert!(reshape.verify(&ctx).is_ok());

    let pad_value = shaped_value(&mut ctx, &[], ElementType::F32);
    let low = shaped_value(&mut ctx, &[2], ElementType::I64);
    let high = shaped_value(&mut ctx, &[2], ElementType::I64);
    let interior = shaped_value(&mut ctx, &[2], ElementType::I64);
    let dynamic_pad =
        DynamicPadOp::new_inferred(&mut ctx, source, pad_value, low, high, interior).unwrap();
    assert!(dynamic_pad.verify(&ctx).is_ok());

    let indices = shaped_value(&mut ctx, &[3, 1], ElementType::I32);
    let size_values = shaped_value(&mut ctx, &[2], ElementType::I64);
    let dims = GatherDimensionNumbersAttr {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        operand_batching_dims: vec![],
        start_indices_batching_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
    };
    let dynamic_gather =
        DynamicGatherOp::new_inferred(&mut ctx, source, indices, size_values, dims.clone())
            .unwrap();
    assert!(dynamic_gather.verify(&ctx).is_ok());
    let oversized = tensor(&ctx, &[3, 2], ElementType::F32);
    let invalid = DynamicGatherOp::new(&mut ctx, source, indices, size_values, oversized, dims);
    assert!(invalid.verify(&ctx).is_err());
}

#[test]
fn scatter_checks_native_dimension_numbers_and_result_arity() {
    let mut ctx = Context::new();
    let input = shaped_value(&mut ctx, &[4], ElementType::F32);
    let updates = shaped_value(&mut ctx, &[2], ElementType::F32);
    let indices = shaped_value(&mut ctx, &[2, 1], ElementType::I32);
    let dims = ScatterDimensionNumbersAttr {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        input_batching_dims: vec![],
        scatter_indices_batching_dims: vec![],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let scatter =
        ScatterOp::new_inferred(&mut ctx, vec![input], indices, vec![updates], dims.clone())
            .unwrap();
    assert!(scatter.verify(&ctx).is_ok());
    let invalid = ScatterDimensionNumbersAttr {
        index_vector_dim: 3,
        ..dims
    };
    assert!(
        ScatterOp::new_inferred(&mut ctx, vec![input], indices, vec![updates], invalid).is_err()
    );
}

#[test]
fn torch_index_select_preserves_batch_axes_and_inserts_all_index_axes() {
    let mut ctx = Context::new();
    let input = shaped_value(&mut ctx, &[2, 3, 4], ElementType::F32);
    let indices = shaped_value(&mut ctx, &[2, 5, 6], ElementType::I32);
    let op =
        TorchIndexSelectOp::new_inferred_with_batch_dims(&mut ctx, input, indices, 1, 1).unwrap();
    let result = op.get_operation().deref(&ctx).get_result(0).get_type(&ctx);
    assert_eq!(tensor_parts(&ctx, result).unwrap().0, vec![2, 5, 6, 4]);
    assert_eq!(op.dimension(&ctx), Some(1));
    assert_eq!(op.batch_dims(&ctx), Some(1));
    assert!(op.verify(&ctx).is_ok());

    let invalid_result = tensor(&ctx, &[5, 6, 4], ElementType::F32);
    let invalid =
        TorchIndexSelectOp::new_with_batch_dims(&mut ctx, input, indices, invalid_result, 1, 1);
    assert!(invalid.verify(&ctx).is_err());
}

#[test]
fn dynamic_pad_requires_matching_integer_padding_vectors() {
    let mut ctx = Context::new();
    let input = shaped_value(&mut ctx, &[2, 3], ElementType::F32);
    let padding = shaped_value(&mut ctx, &[], ElementType::F32);
    let low = shaped_value(&mut ctx, &[2], ElementType::I32);
    let high = shaped_value(&mut ctx, &[2], ElementType::I32);
    let interior = shaped_value(&mut ctx, &[2], ElementType::I64);
    let op = DynamicPadOp::new_inferred(&mut ctx, input, padding, low, high, interior).unwrap();
    assert!(op.verify(&ctx).is_err());
}
