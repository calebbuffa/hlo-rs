use pliron_mlir_builtin::{
    attributes::{DenseElementsAttr, ScalarBitsAttr},
    pliron::{common_traits::Verify, context::Context},
    types::{ElementType, TensorType, VectorType},
};

#[test]
fn tensor_shapes_and_exact_constant_payloads_verify_without_hlo() {
    let ctx = &Context::new();
    let ty = TensorType::static_shape(ctx, ElementType::F32, [2, 3]);
    ty.deref(ctx).verify(ctx).unwrap();
    let constant =
        DenseElementsAttr::splat(ty, Box::new(ScalarBitsAttr::f32_bits(ctx, 0x8000_0000)));
    constant.verify(ctx).unwrap();
    let invalid = TensorType::static_shape(ctx, ElementType::F32, [-2]);
    let error = invalid.deref(ctx).verify(ctx).unwrap_err();
    assert!(matches!(
        pliron_mlir_builtin::Diagnostic::from_error(&error),
        Some(pliron_mlir_builtin::Diagnostic::InvalidType { .. })
    ));
}

#[test]
fn fixed_and_scalable_vectors_preserve_shape_constraints() {
    let ctx = &Context::new();
    let valid = VectorType::new_scalable(ctx, ElementType::I32, vec![4, 8], vec![false, true]);
    valid.deref(ctx).verify(ctx).unwrap();
    let invalid = VectorType::new_scalable(ctx, ElementType::I32, vec![4, 8], vec![true]);
    assert!(invalid.deref(ctx).verify(ctx).is_err());
}

#[cfg(feature = "mlir-interop")]
#[test]
fn builtin_printing_has_no_hlo_dependency() {
    let ctx = &Context::new();
    let ty = TensorType::static_shape(ctx, ElementType::F32, [2, -1]);
    assert_eq!(
        pliron_mlir_builtin::mlir::print_type(ctx, ty).unwrap(),
        "tensor<2x?xf32>"
    );
    assert_eq!(
        pliron_mlir_builtin::mlir::print_attribute(
            ctx,
            &ScalarBitsAttr::f32_bits(ctx, 0x8000_0000)
        )
        .unwrap(),
        "0x80000000 : f32"
    );
}
