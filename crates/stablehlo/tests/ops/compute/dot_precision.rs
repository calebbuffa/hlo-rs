use stablehlo::{
    attributes::{DotDimensionNumbersAttr, Precision},
    ops::{ConstantOp, DotGeneralOp, DotOp},
    pliron::{
        builtin::attributes::{StringAttr, VecAttr},
        common_traits::Verify,
        context::Context,
        identifier::Identifier,
        op::Op,
    },
    types::{ElementType, tensor},
};

#[test]
fn omitted_dot_precision_is_default_but_malformed_present_values_are_not() {
    let mut ctx = Context::new();
    let lhs_type = tensor(&ctx, &[2, 3], ElementType::F32);
    let rhs_type = tensor(&ctx, &[3, 4], ElementType::F32);
    let result_type = tensor(&ctx, &[2, 4], ElementType::F32);
    let lhs = ConstantOp::new(&mut ctx, lhs_type, vec![0], true);
    let rhs = ConstantOp::new(&mut ctx, rhs_type, vec![0], true);
    let lhs = lhs.get_operation().deref(&ctx).get_result(0);
    let rhs = rhs.get_operation().deref(&ctx).get_result(0);
    let dot = DotOp::new(&mut ctx, lhs, rhs, result_type, vec![]);
    let general = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result_type,
        DotDimensionNumbersAttr {
            lhs_batching_dimensions: vec![],
            rhs_batching_dimensions: vec![],
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![0],
        },
    )
    .with_precision(&ctx, vec![Precision::High, Precision::Highest]);
    assert_eq!(
        general.precision(&ctx),
        Some(vec![Precision::High, Precision::Highest])
    );
    assert_eq!(
        general.dimension_numbers(&ctx).as_deref(),
        Some(&DotDimensionNumbersAttr {
            lhs_batching_dimensions: vec![],
            rhs_batching_dimensions: vec![],
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![0],
        })
    );
    for (operation, key, verifier) in [
        (
            dot.get_operation(),
            "stablehlo_compute_dot_precision",
            &dot as &dyn Verify,
        ),
        (
            general.get_operation(),
            "stablehlo_dot_precision",
            &general as &dyn Verify,
        ),
    ] {
        let key = Identifier::new(key);
        operation.deref_mut(&ctx).attributes.0.remove(&key);
        verifier.verify(&ctx).unwrap();
        assert!(!operation.deref(&ctx).attributes.0.contains_key(&key));
        operation
            .deref_mut(&ctx)
            .attributes
            .set(key.clone(), StringAttr::new("DEFAULT".into()));
        assert!(verifier.verify(&ctx).is_err());
        operation.deref_mut(&ctx).attributes.set(
            key.clone(),
            VecAttr::new(vec![Box::new(StringAttr::new("DEFAULT".into()))]),
        );
        assert!(verifier.verify(&ctx).is_err());
        operation.deref_mut(&ctx).attributes.0.remove(&key);
        verifier.verify(&ctx).unwrap();
    }
}
