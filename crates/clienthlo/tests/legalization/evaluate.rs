use super::*;
use clienthlo::{
    pliron::r#type::Typed,
    stablehlo::{attributes::ComparisonDirection, types::primitive_element},
};
use std::collections::HashMap;

fn signed(bits: u64, width: u32) -> i128 {
    ((u128::from(bits) << (128 - width)) as i128) >> (128 - width)
}

fn decode_float(element: ElementType, bits: u64) -> f64 {
    match element {
        ElementType::F64 => return f64::from_bits(bits),
        ElementType::F32 => return f64::from(f32::from_bits(bits as u32)),
        ElementType::F8E8M0FnUz => {
            return if bits == 255 {
                f64::NAN
            } else {
                2.0_f64.powi(bits as i32 - 127)
            };
        }
        _ => {}
    }
    let (mantissa, bias) = match element {
        ElementType::F8E4M3FnUz => (3, 8),
        ElementType::F8E4M3B11FnUz => (3, 11),
        ElementType::F8E5M2FnUz => (2, 16),
        _ => panic!("missing test float decoder for {element:?}"),
    };
    if bits == 128 {
        return f64::NAN;
    }
    let sign = if bits & 128 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits & 127) >> mantissa) as i32;
    let fraction = (bits & ((1 << mantissa) - 1)) as f64 / f64::from(1 << mantissa);
    sign * if exponent == 0 {
        fraction * 2.0_f64.powi(1 - bias)
    } else {
        (1.0 + fraction) * 2.0_f64.powi(exponent - bias)
    }
}

fn encode_float(element: ElementType, value: f64) -> u64 {
    match element {
        ElementType::F64 => return value.to_bits(),
        ElementType::F32 => return u64::from((value as f32).to_bits()),
        _ => {}
    }
    if value.is_nan() {
        return if element == ElementType::F8E8M0FnUz {
            255
        } else {
            128
        };
    }
    (0..256)
        .filter(|&bits| !decode_float(element, bits).is_nan())
        .min_by(|&a, &b| {
            (decode_float(element, a) - value)
                .abs()
                .total_cmp(&(decode_float(element, b) - value).abs())
                .then((a & 1).cmp(&(b & 1)))
        })
        .unwrap()
}

pub(super) fn execute(ctx: &Context, fun: &FuncOp, inputs: &[u64]) -> Vec<u64> {
    let args = fun.arguments(ctx);
    let mut values: HashMap<_, _> = args.into_iter().zip(inputs.iter().copied()).collect();
    for op in fun.entry_block(ctx).unwrap().deref(ctx).iter(ctx) {
        let ir = op.deref(ctx);
        let operands: Vec<_> = ir.operands().collect();
        let bits: Vec<u64> = operands.iter().map(|operand| values[operand]).collect();
        let name = Operation::get_opid(op, ctx).to_string();
        if name == "stablehlo.return" {
            return bits;
        }
        let result = ir.get_result(0);
        let (_, result_element) = {
            let ty = result.get_type(ctx);
            let object = ty.deref(ctx);
            let tensor = object
                .downcast_ref::<stablehlo::types::TensorType>()
                .unwrap();
            (
                tensor.dimensions.clone(),
                primitive_element(ctx, tensor.element).unwrap(),
            )
        };
        let operand_element = operands.first().map(|operand| {
            let ty = operand.get_type(ctx);
            let object = ty.deref(ctx);
            primitive_element(
                ctx,
                object
                    .downcast_ref::<stablehlo::types::TensorType>()
                    .unwrap()
                    .element,
            )
            .unwrap()
        });
        let width = result_element.bit_width().unwrap();
        let mask = u64::MAX >> (64 - width);
        let float = operand_element.is_some_and(ElementType::is_float);
        let decode = |bits| decode_float(operand_element.unwrap(), bits);
        let x = || decode(bits[0]);
        let y = || decode(bits[1]);
        let value = match name.as_str() {
            "stablehlo.constant" => Operation::get_op::<stablehlo::ops::ConstantOp>(op, ctx)
                .unwrap()
                .bits(ctx)
                .unwrap()[0],
            "stablehlo.convert" => {
                let source = operand_element.unwrap();
                if source.is_float() {
                    encode_float(result_element, x())
                } else {
                    let (width, signedness) = source.integer_parts().unwrap();
                    if signedness == clienthlo::pliron::builtin::types::Signedness::Unsigned {
                        bits[0]
                    } else {
                        signed(bits[0], width) as u64
                    }
                }
            }
            "stablehlo.bitcast_convert" | "stablehlo.broadcast_in_dim" => bits[0],
            "stablehlo.add" if float => (x() + y()).to_bits(),
            "stablehlo.subtract" if float => (x() - y()).to_bits(),
            "stablehlo.multiply" if float => (x() * y()).to_bits(),
            "stablehlo.divide" if float => (x() / y()).to_bits(),
            "stablehlo.power" => x().powf(y()).to_bits(),
            "stablehlo.remainder" if float => (x() % y()).to_bits(),
            "stablehlo.maximum" if float => (if x().is_nan() || y().is_nan() {
                f64::NAN
            } else {
                x().max(y())
            })
            .to_bits(),
            "stablehlo.minimum" if float => (if x().is_nan() || y().is_nan() {
                f64::NAN
            } else {
                x().min(y())
            })
            .to_bits(),
            "stablehlo.atan2" => x().atan2(y()).to_bits(),
            "stablehlo.log_plus_one" => x().ln_1p().to_bits(),
            "stablehlo.log" => x().ln().to_bits(),
            "stablehlo.sqrt" => x().sqrt().to_bits(),
            "stablehlo.exponential" => x().exp().to_bits(),
            "stablehlo.exponential_minus_one" => x().exp_m1().to_bits(),
            "stablehlo.sine" => x().sin().to_bits(),
            "stablehlo.cosine" => x().cos().to_bits(),
            "stablehlo.floor" => x().floor().to_bits(),
            "stablehlo.sign" => (if x() == 0.0 { x() } else { x().signum() }).to_bits(),
            "stablehlo.not" => !bits[0],
            "stablehlo.negate" => (-x()).to_bits(),
            "stablehlo.abs" => x().abs().to_bits(),
            "stablehlo.add" => bits[0].wrapping_add(bits[1]),
            "stablehlo.subtract" => bits[0].wrapping_sub(bits[1]),
            "stablehlo.multiply" => bits[0].wrapping_mul(bits[1]),
            "stablehlo.and" => bits[0] & bits[1],
            "stablehlo.or" => bits[0] | bits[1],
            "stablehlo.shift_right_logical" => bits[0] >> bits[1],
            "stablehlo.shift_right_arithmetic" => {
                let width = operand_element.unwrap().bit_width().unwrap();
                (signed(bits[0], width) >> bits[1]) as u64
            }

            "stablehlo.compare" => {
                let comparison = Operation::get_op::<stablehlo::ops::CompareOp>(op, ctx).unwrap();
                let relation = if float {
                    x().partial_cmp(&y())
                } else {
                    let (width, signedness) = operand_element.unwrap().integer_parts().unwrap();
                    Some(
                        if signedness == clienthlo::pliron::builtin::types::Signedness::Unsigned {
                            bits[0].cmp(&bits[1])
                        } else {
                            signed(bits[0], width).cmp(&signed(bits[1], width))
                        },
                    )
                };
                use std::cmp::Ordering::*;
                u64::from(match comparison.direction(ctx).unwrap() {
                    ComparisonDirection::Eq => relation == Some(Equal),
                    ComparisonDirection::Ne => relation != Some(Equal),
                    ComparisonDirection::Gt => relation == Some(Greater),
                    ComparisonDirection::Ge => matches!(relation, Some(Greater | Equal)),
                    ComparisonDirection::Lt => relation == Some(Less),
                    ComparisonDirection::Le => matches!(relation, Some(Less | Equal)),
                })
            }
            "stablehlo.select" => {
                if bits[0] != 0 {
                    bits[1]
                } else {
                    bits[2]
                }
            }
            other => panic!("missing scalar evaluator for {other}"),
        };
        let value = if result_element.is_float()
            && !matches!(
                name.as_str(),
                "stablehlo.constant"
                    | "stablehlo.convert"
                    | "stablehlo.select"
                    | "stablehlo.broadcast_in_dim"
                    | "stablehlo.bitcast_convert"
            ) {
            encode_float(result_element, f64::from_bits(value))
        } else {
            value
        };
        values.insert(result, value & mask);
    }
    panic!("function has no return");
}

#[test]
fn emitted_real_math_matches_independent_references() {
    macro_rules! unary {
        ($op:ident, $reference:expr, $cases:expr, $tolerance:expr) => {{
            let mut ctx = Context::new();
            let ty = tensor(&ctx, &[], ElementType::F64);
            let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
            let arg = fun.arguments(&ctx)[0];
            let op = $op::new_inferred(&mut ctx, arg).unwrap();
            attach(&mut ctx, &fun, op);
            check(&mut ctx, &module);
            for x in $cases {
                let expected: f64 = ($reference)(x);
                let actual = f64::from_bits(execute(&ctx, &fun, &[x.to_bits()])[0]);
                assert_close(
                    actual,
                    expected,
                    $tolerance,
                    &format!("{}({x})", stringify!($op)),
                );
            }
        }};
    }
    let all = [
        -f64::INFINITY,
        -f64::MAX,
        -100.0,
        -10.0,
        -2.0,
        -1.0,
        -0.5,
        -1e-100,
        -0.0,
        0.0,
        1e-100,
        0.5,
        1.0,
        2.0,
        10.0,
        100.0,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ];
    unary!(AsinOp, f64::asin, all, 3e-15);
    unary!(AcosOp, f64::acos, all, 3e-15);
    unary!(AsinhOp, libm::asinh, all, 3e-15);
    unary!(AcoshOp, libm::acosh, all, 3e-15);
    unary!(AtanOp, f64::atan, all, 3e-15);
    unary!(AtanhOp, f64::atanh, all, 3e-15);
    unary!(SinhOp, f64::sinh, all, 3e-14);
    unary!(CoshOp, f64::cosh, all, 3e-14);
    unary!(ErfOp, libm::erf, all, 5e-15);
    unary!(ErfcOp, libm::erfc, all, 5e-15);
    unary!(LgammaOp, libm::lgamma, all, 2e-14);
    // Validate inverse accuracy by composition, independently of its polynomial.
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[], ElementType::F64);
    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
    let arg = fun.arguments(&ctx)[0];
    let op = ErfInvOp::new_inferred(&mut ctx, arg).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
    for x in [
        -1.0_f64,
        -0.999999999999,
        -0.99,
        -0.5,
        -0.0,
        0.0,
        0.5,
        0.99,
        0.999999999999,
        1.0,
    ] {
        let inverse = f64::from_bits(execute(&ctx, &fun, &[x.to_bits()])[0]);
        assert_close(libm::erf(inverse), x, 2e-15, "erf(erf_inv(x))");
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}: expected NaN, got {actual}");
    } else if expected.is_infinite() {
        assert_eq!(actual, expected, "{label}");
    } else {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{label}: got {actual:e}, expected {expected:e}"
        );
    }
}

#[test]
fn emitted_gamma_zeta_and_bessel_match_known_values() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[], ElementType::F64);
    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
    let x = fun.arguments(&ctx)[0];
    let op = DigammaOp::new_inferred(&mut ctx, x).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
    let euler = 0.5772156649015329;
    for (x, expected) in [
        (1.0_f64, -euler),
        (2.0, 1.0 - euler),
        (0.5, -euler - 2.0 * 2.0_f64.ln()),
        (0.0, f64::NEG_INFINITY),
        (-1.0, f64::NAN),
    ] {
        assert_close(
            f64::from_bits(execute(&ctx, &fun, &[x.to_bits()])[0]),
            expected,
            2e-14,
            "digamma",
        );
    }
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[], ElementType::F64);
    let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
    let args = fun.arguments(&ctx);
    let op = ZetaOp::new_inferred(&mut ctx, args[0], args[1]).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
    for (s, q, expected) in [
        (2.0_f64, 1.0_f64, std::f64::consts::PI.powi(2) / 6.0),
        (2.0, 2.0, std::f64::consts::PI.powi(2) / 6.0 - 1.0),
        (4.0, 1.0, std::f64::consts::PI.powi(4) / 90.0),
        (2.0, f64::INFINITY, 0.0),
        (1.0, 1.0, f64::INFINITY),
        (0.5, 1.0, f64::NAN),
        (2.0, 0.0, f64::INFINITY),
    ] {
        assert_close(
            f64::from_bits(execute(&ctx, &fun, &[s.to_bits(), q.to_bits()])[0]),
            expected,
            3e-14,
            "zeta",
        );
    }

    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[], ElementType::F64);
    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
    let x = fun.arguments(&ctx)[0];
    let op = BesselI1eOp::new_inferred(&mut ctx, x).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
    for x in [
        -32.0_f64, -10.0, -8.0, -1.0, -1e-100, -0.0, 0.0, 1e-100, 1.0, 8.0, 10.0, 32.0,
    ] {
        let mut term = x / 2.0;
        let mut sum = term;
        for k in 1..160 {
            term *= (x * x / 4.0) / f64::from(k * (k + 1));
            sum += term;
        }
        let expected = sum * (-x.abs()).exp();
        let actual = f64::from_bits(execute(&ctx, &fun, &[x.to_bits()])[0]);
        assert_close(actual, expected, 3e-15, "bessel_i1e");
    }

    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[], ElementType::F64);
    let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
    let args = fun.arguments(&ctx);
    let op = PolygammaOp::new_inferred(&mut ctx, args[0], args[1]).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
    for (n, x, expected) in [
        (0.0_f64, 1.0_f64, -euler),
        (1.0, 1.0, std::f64::consts::PI.powi(2) / 6.0),
        (2.0, 1.0, -2.4041138063191885),
        (3.0, 1.0, std::f64::consts::PI.powi(4) / 15.0),
        (-1.0, 1.0, f64::NAN),
        (0.5, 1.0, f64::NAN),
    ] {
        let actual = f64::from_bits(execute(&ctx, &fun, &[n.to_bits(), x.to_bits()])[0]);
        assert_close(actual, expected, 2e-14, "polygamma");
    }
}
#[test]
fn emitted_mulhi_matches_full_width_integer_products() {
    for element in [
        ElementType::I2,
        ElementType::I4,
        ElementType::Ui4,
        ElementType::I8,
        ElementType::Ui8,
        ElementType::I16,
        ElementType::Ui16,
        ElementType::I32,
        ElementType::Ui32,
        ElementType::I64,
        ElementType::Ui64,
    ] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[], element);
        let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
        let args = fun.arguments(&ctx);
        let op = MulhiOp::new_inferred(&mut ctx, args[0], args[1]).unwrap();
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
        let (width, signedness) = element.integer_parts().unwrap();
        let mask = u64::MAX >> (64 - width);
        let mut state = 0x791c_011e_9119_ab41_u64;
        let mut cases = vec![0, 1, mask, mask >> 1, 1u64 << (width - 1)];
        for _ in 0..32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            cases.push(state & mask);
        }
        for &a in &cases {
            for &b in &cases {
                let expected =
                    if signedness == clienthlo::pliron::builtin::types::Signedness::Unsigned {
                        ((u128::from(a) * u128::from(b)) >> width) as u64
                    } else {
                        ((signed(a, width) * signed(b, width)) >> width) as u64
                    } & mask;
                assert_eq!(
                    execute(&ctx, &fun, &[a, b]),
                    [expected],
                    "{element:?}: {a:x} * {b:x}"
                );
            }
        }
    }
}

#[test]
fn emitted_next_after_matches_ieee_neighbors() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[], ElementType::F64);
    let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
    let args = fun.arguments(&ctx);
    let op = NextAfterOp::new_inferred(&mut ctx, args[0], args[1]).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    for x in values {
        for y in values {
            let actual = f64::from_bits(execute(&ctx, &fun, &[x.to_bits(), y.to_bits()])[0]);
            let expected = if x.is_nan() || y.is_nan() {
                f64::NAN
            } else if x == y {
                y
            } else if x < y {
                x.next_up()
            } else {
                x.next_down()
            };
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{x:?} -> {y:?}");
            }
        }
    }
}
