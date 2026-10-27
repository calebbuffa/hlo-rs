use super::*;
use clienthlo::pliron::r#type::Typed;
use clienthlo::stablehlo::types::primitive_element;
use num_complex::Complex64 as C;
use std::collections::HashMap;

fn execute(ctx: &Context, fun: &FuncOp, input: C) -> C {
    let mut values = HashMap::from([(fun.arguments(ctx)[0], input)]);
    for op in fun.entry_block(ctx).unwrap().deref(ctx).iter(ctx) {
        let ir = op.deref(ctx);
        let name = Operation::get_opid(op, ctx).to_string();
        let operands: Vec<C> = ir.operands().map(|value| values[&value]).collect();
        if name == "stablehlo.return" {
            return operands[0];
        }
        let result = ir.get_result(0);
        let ty = result.get_type(ctx);
        let object = ty.deref(ctx);
        let tensor = object
            .downcast_ref::<stablehlo::types::TensorType>()
            .unwrap();
        let complex = primitive_element(ctx, tensor.element) == Some(ElementType::C128);
        let a = operands.first().copied().unwrap_or_default();
        let b = operands.get(1).copied().unwrap_or_default();
        let real = |v| C::new(v, 0.0);
        let value = match name.as_str() {
            "stablehlo.constant" => {
                let bits = Operation::get_op::<stablehlo::ops::ConstantOp>(op, ctx)
                    .unwrap()
                    .bits(ctx)
                    .unwrap();
                C::new(
                    f64::from_bits(bits[0]),
                    if complex {
                        f64::from_bits(bits[1])
                    } else {
                        0.0
                    },
                )
            }
            "stablehlo.convert" | "stablehlo.broadcast_in_dim" => a,
            "stablehlo.real" => real(a.re),
            "stablehlo.imag" => real(a.im),
            "stablehlo.complex" => C::new(a.re, b.re),
            "stablehlo.add" => {
                if complex {
                    a + b
                } else {
                    real(a.re + b.re)
                }
            }
            "stablehlo.subtract" => {
                if complex {
                    a - b
                } else {
                    real(a.re - b.re)
                }
            }
            "stablehlo.multiply" => {
                if complex {
                    a * b
                } else {
                    real(a.re * b.re)
                }
            }
            "stablehlo.divide" => {
                if complex {
                    a / b
                } else {
                    real(a.re / b.re)
                }
            }
            "stablehlo.negate" => -a,
            "stablehlo.abs" => real(a.re.abs()),
            "stablehlo.maximum" => real(if a.re.is_nan() || b.re.is_nan() {
                f64::NAN
            } else {
                a.re.max(b.re)
            }),
            "stablehlo.minimum" => real(if a.re.is_nan() || b.re.is_nan() {
                f64::NAN
            } else {
                a.re.min(b.re)
            }),
            "stablehlo.sqrt" => {
                if complex {
                    a.sqrt()
                } else {
                    real(a.re.sqrt())
                }
            }
            "stablehlo.log" => {
                if complex {
                    a.ln()
                } else {
                    real(a.re.ln())
                }
            }
            "stablehlo.log_plus_one" => real(a.re.ln_1p()),
            "stablehlo.exponential" => {
                if complex {
                    a.exp()
                } else {
                    real(a.re.exp())
                }
            }
            "stablehlo.atan2" => real(a.re.atan2(b.re)),
            "stablehlo.and" => real(f64::from(a.re != 0.0 && b.re != 0.0)),
            "stablehlo.or" => real(f64::from(a.re != 0.0 || b.re != 0.0)),
            "stablehlo.not" => real(f64::from(a.re == 0.0)),
            "stablehlo.select" => {
                if a.re != 0.0 {
                    operands[1]
                } else {
                    operands[2]
                }
            }
            "stablehlo.compare" => {
                use stablehlo::attributes::ComparisonDirection::*;
                let direction = Operation::get_op::<stablehlo::ops::CompareOp>(op, ctx)
                    .unwrap()
                    .direction(ctx)
                    .unwrap();
                real(f64::from(match direction {
                    Eq => a.re == b.re,
                    Ne => a.re != b.re,
                    Lt => a.re < b.re,
                    Le => a.re <= b.re,
                    Gt => a.re > b.re,
                    Ge => a.re >= b.re,
                }))
            }
            other => panic!("missing complex evaluator operation {other}"),
        };
        values.insert(result, value);
    }
    panic!("missing return")
}

#[test]
fn emitted_complex_inverse_functions_match_independent_references() {
    macro_rules! unary {
        ($op:ident, $reference:expr) => {{
            let mut ctx = Context::new();
            let ty = tensor(&ctx, &[], ElementType::C128);
            let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
            let x = fun.arguments(&ctx)[0];
            let op = $op::new_inferred(&mut ctx, x).unwrap();
            attach(&mut ctx, &fun, op);
            check(&mut ctx, &module);
            for re in [-10.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0] {
                for im in [-10.0, -2.0, -0.5, -1e-100, 1e-100, 0.5, 2.0, 10.0] {
                    let input = C::new(re, im);
                    let expected: C = ($reference)(input);
                    let actual = execute(&ctx, &fun, input);
                    for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
                        assert!(
                            (a - e).abs() <= 2e-13 * e.abs().max(1.0),
                            "{}({input}): got {actual}, expected {expected}",
                            stringify!($op)
                        );
                    }
                }
            }
        }};
    }
    unary!(AsinOp, C::asin);
    unary!(AcosOp, C::acos);
    unary!(AsinhOp, |z: C| {
        // num-complex chooses the opposite side of this cut for +0-yi.
        // CHLO's pinned decomposition selects the right-half-plane limit.
        if z.re == 0.0 && z.im.abs() > 1.0 {
            C::new(
                libm::acosh(z.im.abs()),
                z.im.signum() * std::f64::consts::FRAC_PI_2,
            )
        } else {
            z.asinh()
        }
    });
    unary!(AcoshOp, C::acosh);
    unary!(AtanOp, |z: C| {
        let value = z.atan();
        if z.re == 0.0 && z.im.abs() > 1.0 {
            C::new(std::f64::consts::FRAC_PI_2, value.im)
        } else {
            value
        }
    });
    unary!(AtanhOp, C::atanh);
}

#[test]
fn emitted_complex_inverse_functions_avoid_overflow_at_extreme_magnitudes() {
    let quarter_pi = std::f64::consts::FRAC_PI_4;
    let magnitude = f64::MAX.ln() + (2.0_f64 * 2.0_f64.sqrt()).ln();
    macro_rules! unary {
        ($op:ident, $reference:expr) => {{
            let mut ctx = Context::new();
            let ty = tensor(&ctx, &[], ElementType::C128);
            let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
            let x = fun.arguments(&ctx)[0];
            let op = $op::new_inferred(&mut ctx, x).unwrap();
            attach(&mut ctx, &fun, op);
            check(&mut ctx, &module);
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    let input = C::new(sx * f64::MAX, sy * f64::MAX);
                    let reference: &dyn Fn(f64, f64) -> C = &$reference;
                    let expected = reference(sx, sy);
                    let actual = execute(&ctx, &fun, input);
                    for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
                        assert!(
                            (a - e).abs() <= 2e-13 * e.abs().max(1.0),
                            "{}({input}): got {actual}, expected {expected}",
                            stringify!($op)
                        );
                    }
                }
            }
        }};
    }
    unary!(AsinOp, |sx, sy| C::new(sx * quarter_pi, sy * magnitude));
    unary!(AcosOp, |sx, sy| C::new(
        if sx > 0.0 {
            quarter_pi
        } else {
            3.0 * quarter_pi
        },
        -sy * magnitude
    ));
    unary!(AsinhOp, |sx, sy| C::new(sx * magnitude, sy * quarter_pi));
    unary!(AcoshOp, |sx, sy| C::new(
        magnitude,
        sy * if sx > 0.0 {
            quarter_pi
        } else {
            3.0 * quarter_pi
        }
    ));
    unary!(AtanOp, |sx, _sy| C::new(sx * 2.0 * quarter_pi, 0.0));
    unary!(AtanhOp, |_sx, sy| C::new(0.0, sy * 2.0 * quarter_pi));
}

#[test]
fn complex_square_preserves_equal_magnitude_cancellation() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[], ElementType::C128);
    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
    let x = fun.arguments(&ctx)[0];
    let op = SquareOp::new_inferred(&mut ctx, x).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
    for (input, expected) in [
        (C::new(f64::MAX, f64::MAX), C::new(0.0, f64::INFINITY)),
        (
            C::new(f64::INFINITY, -f64::INFINITY),
            C::new(0.0, -f64::INFINITY),
        ),
        (C::new(3.0, 4.0), C::new(-7.0, 24.0)),
    ] {
        assert_eq!(execute(&ctx, &fun, input), expected);
    }
}
