use std::any::TypeId;

use pliron::{context::Context, printable::Printable};
use stablehlo::{
    attributes::TypeExtensionsAttr,
    types::{ElementType, TensorType},
};

#[test]
fn compatibility_paths_reexport_identical_registered_types() {
    assert_eq!(
        TypeId::of::<TensorType>(),
        TypeId::of::<pliron_mlir_builtin::types::TensorType>()
    );
    assert_eq!(
        TypeId::of::<stablehlo::attributes::DenseElementsAttr>(),
        TypeId::of::<pliron_mlir_builtin::attributes::DenseElementsAttr>()
    );
    assert_eq!(
        TypeId::of::<stablehlo::integer::IntegerValue>(),
        TypeId::of::<pliron_mlir_builtin::integer::IntegerValue>()
    );
    let ctx = &Context::new();
    let old = TensorType::static_shape(ctx, ElementType::F32, [4, -1]);
    let shared =
        pliron_mlir_builtin::types::TensorType::static_shape(ctx, ElementType::F32, [4, -1]);
    assert_eq!(old, shared);
    assert!(old.disp(ctx).to_string().contains("stablehlo.tensor"));
}

#[test]
fn shared_tensors_enforce_stablehlo_bounds_without_reverse_dependencies() {
    let ctx = &Context::new();
    let base = TensorType::get(ctx, vec![4, -1], ElementType::F32.native(ctx));
    let valid = base.deref(ctx).with_encoding(
        ctx,
        Box::new(TypeExtensionsAttr {
            bounds: vec![-1, 16],
        }),
    );
    valid.deref(ctx).verify(ctx).unwrap();
    let tensor = valid.deref(ctx);
    assert_eq!(
        tensor.downcast_ref::<TensorType>().unwrap().bounds(),
        Some(&[-1, 16][..])
    );
    let invalid = base.deref(ctx).with_encoding(
        ctx,
        Box::new(TypeExtensionsAttr {
            bounds: vec![4, 16],
        }),
    );
    assert!(invalid.deref(ctx).verify(ctx).is_err());
}
