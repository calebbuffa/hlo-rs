use super::evaluate::execute;
use super::*;

#[test]
fn float32_special_functions_keep_float32_accuracy() {
    macro_rules! unary {
        ($op:ident, $reference:expr) => {{
            let mut ctx = Context::new();
            let ty = tensor(&ctx, &[], ElementType::F32);
            let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
            let x = fun.arguments(&ctx)[0];
            let op = $op::new_inferred(&mut ctx, x).unwrap();
            attach(&mut ctx, &fun, op);
            check(&mut ctx, &module);
            for i in -100..=100 {
                let x = i as f32 / 10.0;
                let expected: f64 = ($reference)(f64::from(x));
                let actual = f64::from(f32::from_bits(
                    execute(&ctx, &fun, &[u64::from(x.to_bits())])[0] as u32,
                ));
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else if expected.is_infinite() {
                    assert_eq!(actual, expected);
                } else {
                    assert!(
                        (actual - expected).abs() <= 4e-6 * expected.abs().max(1.0),
                        "{}({x}): {actual:e} vs {expected:e}",
                        stringify!($op)
                    );
                }
            }
        }};
    }
    unary!(ErfOp, libm::erf);
    unary!(ErfcOp, libm::erfc);
    unary!(LgammaOp, libm::lgamma);
    unary!(AsinhOp, libm::asinh);
    unary!(AcoshOp, libm::acosh);
    unary!(SinhOp, libm::sinh);
    unary!(CoshOp, libm::cosh);
}

#[test]
fn low_precision_upstream_math_domains_export() {
    macro_rules! test_ops {
        ($($op:ident),* $(,)?) => {$(
            for element in [
                ElementType::F4E2M1Fn, ElementType::F6E2M3Fn, ElementType::F6E3M2Fn,
                ElementType::F8E3M4, ElementType::F8E4M3, ElementType::F8E4M3Fn,
                ElementType::F8E4M3FnUz, ElementType::F8E4M3B11FnUz,
                ElementType::F8E5M2, ElementType::F8E5M2FnUz, ElementType::F8E8M0FnUz,
            ] {
                let mut ctx = Context::new();
                let ty = tensor(&ctx, &[], element);
                let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
                let arg = fun.arguments(&ctx)[0];
                let op = $op::new_inferred(&mut ctx, arg).unwrap();
                attach(&mut ctx, &fun, op);
                check(&mut ctx, &module);
            }

        )*};
    }
    test_ops!(
        LgammaOp, DigammaOp, ErfInvOp, SinhOp, CoshOp, AsinOp, AcosOp, AsinhOp, AcoshOp
    );
}

#[test]
fn float8_next_after_handles_unsigned_exponents_and_unsigned_zero() {
    for element in [
        ElementType::F8E4M3FnUz,
        ElementType::F8E4M3B11FnUz,
        ElementType::F8E5M2FnUz,
        ElementType::F8E8M0FnUz,
    ] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[], element);
        let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
        let args = fun.arguments(&ctx);
        let op = NextAfterOp::new_inferred(&mut ctx, args[0], args[1]).unwrap();
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
        if element == ElementType::F8E8M0FnUz {
            for (x, y, expected) in [
                (127, 129, 128),
                (129, 127, 128),
                (0, 254, 1),
                (254, 0, 253),
                (128, 128, 128),
                (255, 127, 255),
            ] {
                assert_eq!(execute(&ctx, &fun, &[x, y]), [expected]);
            }
        } else {
            for (x, y, expected) in [
                (129, 0, 0),
                (1, 0, 0),
                (0, 129, 129),
                (0, 1, 1),
                (127, 0, 126),
                (255, 0, 254),
                (129, 1, 0),
                (128, 1, 128),
            ] {
                assert_eq!(execute(&ctx, &fun, &[x, y]), [expected]);
            }
        }
    }
}

#[test]
fn narrow_next_after_domains_fail_before_rewriting_neighbors() {
    for element in [ElementType::F6E2M3Fn, ElementType::F6E3M2Fn] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[], element);
        let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
        let args = fun.arguments(&ctx);
        let square = SquareOp::new_inferred(&mut ctx, args[0]).unwrap();
        square
            .get_operation()
            .insert_at_back(fun.entry_block(&ctx).unwrap(), &ctx);
        let op = NextAfterOp::new_inferred(&mut ctx, args[0], args[1]).unwrap();
        attach(&mut ctx, &fun, op);
        let error = run(&mut ctx, module.get_operation()).err().unwrap();
        assert!(matches!(Diagnostic::from_error(&error),
            Some(Diagnostic::UnsupportedLowering { operation, .. }) if operation == "chlo.next_after"));
        assert!(Operation::get_op::<SquareOp>(square.get_operation(), &ctx).is_some());
    }
}

#[test]
fn remaining_narrow_next_after_domains_export() {
    for element in [
        ElementType::F4E2M1Fn,
        ElementType::F8E3M4,
        ElementType::F8E4M3,
        ElementType::F8E4M3Fn,
        ElementType::F8E5M2,
    ] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[], element);
        let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
        let args = fun.arguments(&ctx);
        let op = NextAfterOp::new_inferred(&mut ctx, args[0], args[1]).unwrap();
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
    }
}

#[test]
fn float32_erfinv_composes_with_an_independent_erf() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[], ElementType::F32);
    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
    let x = fun.arguments(&ctx)[0];
    let op = ErfInvOp::new_inferred(&mut ctx, x).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
    for i in -99..=99 {
        let x = i as f32 / 100.0;
        let inverse = f32::from_bits(execute(&ctx, &fun, &[u64::from(x.to_bits())])[0] as u32);
        assert!((libm::erf(f64::from(inverse)) - f64::from(x)).abs() < 3e-7);
    }
}
