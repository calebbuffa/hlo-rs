use super::*;

#[test]
fn all_real_math_decompositions_export() {
    macro_rules! unary {
        ($($op:ident $(=> $extra:expr)?),* $(,)?) => {$(
            for element in [ElementType::F16, ElementType::Bf16, ElementType::F32, ElementType::F64] {
                for shape in [vec![], vec![2, 3], vec![-1, 3]] {
                    let mut ctx = Context::new();
                    let ty = tensor(&ctx, &shape, element);
                    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
                    let x = fun.arguments(&ctx)[0];
                    let op = $op::new_inferred(&mut ctx, x).unwrap();
                    attach(&mut ctx, &fun, op);
                    check(&mut ctx, &module);
                }
            }
        )*};
    }
    unary!(
        AsinOp,
        AcosOp,
        AsinhOp,
        AcoshOp,
        AtanOp,
        AtanhOp,
        BesselI1eOp,
        CoshOp,
        SinhOp,
        ErfOp,
        ErfcOp,
        ErfInvOp,
        LgammaOp,
        DigammaOp
    );
    macro_rules! binary {
        ($($op:ident $(=> $extra:expr)?),* $(,)?) => {$(
            for element in [ElementType::F16, ElementType::Bf16, ElementType::F32, ElementType::F64] {
                let mut ctx = Context::new();
                let ty = tensor(&ctx, &[-1, 3], element);
                let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
                let x = fun.arguments(&ctx);
                let op = $op::new_inferred(&mut ctx, x[0], x[1] $(, $extra)?).unwrap();
                attach(&mut ctx, &fun, op);
                check(&mut ctx, &module);
            }
        )*};
    }
    binary!(PolygammaOp, ZetaOp, BroadcastPolygammaOp => None, BroadcastZetaOp => None);
}

#[test]
fn all_complex_math_decompositions_export() {
    macro_rules! unary {
        ($($op:ident),* $(,)?) => {$(
            for element in [ElementType::C64, ElementType::C128] {
                for shape in [vec![], vec![2, 3], vec![-1, 3]] {
                    let mut ctx = Context::new();
                    let ty = tensor(&ctx, &shape, element);
                    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
                    let x = fun.arguments(&ctx)[0];
                    let op = $op::new_inferred(&mut ctx, x).unwrap();
                    attach(&mut ctx, &fun, op);
                    check(&mut ctx, &module);
                }
            }
        )*};
    }
    unary!(
        AsinAcosKernelOp,
        AsinOp,
        AcosOp,
        AsinhOp,
        AcoshOp,
        AtanOp,
        AtanhOp,
        SquareOp,
        ConjOp,
        TanOp,
        SinhOp,
        CoshOp
    );
}

#[test]
fn dynamic_top_k_exports() {
    for shape in [vec![-1], vec![-1, 4], vec![2, -1]] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &shape, ElementType::F32);
        let mut output = shape.clone();
        *output.last_mut().unwrap() = 2;
        let values = tensor(&ctx, &output, ElementType::F32);
        let indices = tensor(&ctx, &output, ElementType::I32);
        let (module, fun) = function(&mut ctx, vec![ty], vec![values, indices]);
        let x = fun.arguments(&ctx)[0];
        let op = TopKOp::new_inferred(
            &mut ctx,
            x,
            TopKOptions {
                k: 2,
                is_stable: true,
            },
        )
        .unwrap();
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
    }
}

#[test]
fn bounded_and_quantized_target_domains_are_not_rejected_globally() {
    use stablehlo::{
        attributes::TypeExtensionsAttr,
        types::{QuantizedElementType, TensorType, UniformQuantizedSpec},
    };
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[-1, 3], ElementType::F32);
    let ty = ty
        .deref(&ctx)
        .downcast_ref::<TensorType>()
        .unwrap()
        .with_encoding(
            &ctx,
            Box::new(TypeExtensionsAttr {
                bounds: vec![8, -1],
            }),
        );
    let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
    let args = fun.arguments(&ctx);
    let op = BroadcastAddOp::new(&mut ctx, args[0], args[1], ty, None);
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);

    let mut ctx = Context::new();
    let element = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: None,
            storage_max: None,
            expressed_type: ElementType::F32,
            quantization_dimension: None,
            scales: vec![0.25],
            zero_points: vec![3],
        },
    );
    let ty = TensorType::get(&ctx, vec![2, 3], element).into();
    let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
    let args = fun.arguments(&ctx);
    let op = BroadcastAddOp::new_inferred(&mut ctx, args[0], args[1], None).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
}
