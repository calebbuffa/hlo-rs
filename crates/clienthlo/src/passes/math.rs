//! Elemental decompositions, following the pinned upstream CHLO algorithms.
use super::{
    builder::{Builder, ranked},
    unsupported,
};
use crate::Result;
use pliron::{r#type::Typed, value::Value};
use stablehlo::{
    attributes::ComparisonDirection as D,
    types::{ElementType, primitive_element, tensor},
};

// Evaluate each subtree before borrowing the builder for its parent.
macro_rules! expr {
    ($b:ident, $like:ident; ($op:ident $a:tt $c:tt $d:tt)) => {{
        let a = expr!($b, $like; $a)?;
        let c = expr!($b, $like; $c)?;
        let d = expr!($b, $like; $d)?;
        $b.$op(a, c, d)
    }};
    ($b:ident, $like:ident; ($op:ident $a:tt $c:tt)) => {{
        let a = expr!($b, $like; $a)?;
        let c = expr!($b, $like; $c)?;
        $b.$op(a, c)
    }};
    ($b:ident, $like:ident; ($op:ident $a:tt)) => {{
        let a = expr!($b, $like; $a)?;
        $b.$op(a)
    }};
    ($b:ident, $like:ident; $value:literal) => { $b.float($value, $like) };
    ($b:ident, $like:ident; $value:ident) => { Ok::<Value, pliron::result::Error>($value) };
}

macro_rules! relations {
    ($($method:ident => $direction:ident),* $(,)?) => {$(
        fn $method(&mut self, a: Value, b: Value) -> Result<Value> {
            self.compare(a, b, D::$direction)
        }
    )*};
}

impl Builder<'_> {
    relations!(lt => Lt, le => Le, eq => Eq, ne => Ne, ge => Ge, gt => Gt);
    fn real(&mut self, z: Value) -> Result<Value> {
        self.component(z, false)
    }
    fn imag(&mut self, z: Value) -> Result<Value> {
        self.component(z, true)
    }
}

fn element(b: &Builder<'_>, x: Value) -> Result<ElementType> {
    ranked(b.ctx, x.get_type(b.ctx))
        .and_then(|(_, e)| primitive_element(b.ctx, e))
        .ok_or_else(|| {
            unsupported(
                b.ctx,
                b.source,
                "elemental decomposition requires primitive ranked tensors",
            )
        })
}

fn largest(b: &Builder<'_>, x: Value) -> Result<f64> {
    Ok(match element(b, x)? {
        ElementType::F64 => f64::MAX,
        ElementType::F32 => f64::from(f32::MAX),
        ElementType::F16 => 65504.0,
        ElementType::Bf16 => f64::from(f32::from_bits(0x7f7f0000)),
        _ => {
            return Err(unsupported(
                b.ctx,
                b.source,
                "expected IEEE floating elements",
            ));
        }
    })
}

pub(super) fn lower(b: &mut Builder<'_>, name: &str, values: &[Value]) -> Result<Value> {
    let original = values[0].get_type(b.ctx);
    let input_element = element(b, values[0])?;
    let needs_upcast = input_element.is_float()
        && input_element.bit_width().is_some_and(|width| width < 32)
        && matches!(
            name,
            "bessel_i1e"
                | "cosh"
                | "sinh"
                | "erf"
                | "erfc"
                | "erf_inv"
                | "lgamma"
                | "digamma"
                | "polygamma"
                | "zeta"
                | "asin"
                | "acos"
                | "asinh"
                | "acosh"
        );
    let values = if needs_upcast {
        values
            .iter()
            .map(|&x| {
                let (shape, _) = ranked(b.ctx, x.get_type(b.ctx)).ok_or_else(|| {
                    unsupported(b.ctx, b.source, "upcast requires ranked tensors")
                })?;
                let ty = tensor(b.ctx, &shape, ElementType::F32);
                b.convert(x, ty)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        values.to_vec()
    };
    let x = values[0];
    let complex = element(b, x)?.complex_component().is_some();
    let result = match name {
        "asin" | "acos" | "asinh" | "acosh" | "_asin_acos_kernel" if complex => {
            inverse_complex(b, name, x)?
        }
        "atan" | "atanh" if complex => {
            if name == "atan" {
                let r = b.real(x)?;
                let i = expr!(b, r; (neg (imag x)))?;
                let z = b.complex(i, r)?;
                let w = complex_atanh(b, z)?;
                expr!(b, r; (complex (imag w) (neg (real w))))?
            } else {
                complex_atanh(b, x)?
            }
        }
        "asin" => expr!(b, x; (atan2 x (sqrt (mul (sub 1.0 x) (add 1.0 x)))))?,
        "acos" => expr!(b, x; (atan2 (sqrt (mul (sub 1.0 x) (add 1.0 x))) x))?,
        "asinh" => {
            let a = b.abs(x)?;
            let limit = largest(b, x)?.sqrt();
            let limit = b.float(limit, x)?;
            expr!(b, x; (mul (sign x) (select (ge a limit)
                (add (log 2.0) (log a))
                (log1p (add a (div (mul a a) (add 1.0 (sqrt (add 1.0 (mul a a))))))))))?
        }
        "acosh" => {
            let limit = b.float(largest(b, x)? / 2.0, x)?;
            let a = expr!(b, x; (sqrt (sub x 1.0)))?;
            expr!(b, x; (select (ge x limit) (add (log 2.0) (log x))
                (log1p (mul a (add (sqrt (add x 1.0)) a)))))?
        }
        "cosh" | "sinh" => {
            let half_log = expr!(b, x; (log 0.5))?;
            let positive = expr!(b, x; (exp (add x half_log)))?;
            let negative = expr!(b, x; (exp (sub half_log x)))?;
            if name == "cosh" {
                b.add(positive, negative)?
            } else {
                let large = b.sub(positive, negative)?;
                if complex {
                    large
                } else {
                    let e = b.expm1(x)?;
                    expr!(b, x; (select (lt (abs x) 1.0)
                        (mul 0.5 (add e (div e (add e 1.0)))) large))?
                }
            }
        }
        "lgamma" => gamma(b, x, false)?,
        "digamma" => gamma(b, x, true)?,
        "zeta" => zeta(b, x, values[1])?,
        "polygamma" => {
            let n = x;
            let x = values[1];
            let next = expr!(b, x; (add n 1.0))?;
            let g = gamma(b, next, false)?;
            let z = zeta(b, next, x)?;
            let general = expr!(b, x; (mul (mul (sub (mul 2.0 (rem n 2.0)) 1.0) (exp g)) z))?;
            let d = gamma(b, x, true)?;
            let nan = b.float(f64::NAN, x)?;
            expr!(b, x; (select (or (ne n (floor n)) (lt n 0.0)) nan
                (select (eq n 0.0) d general)))?
        }
        "erf" | "erfc" => erf(b, x, name == "erfc")?,
        "erf_inv" => erf_inv(b, x)?,
        "bessel_i1e" => bessel(b, x)?,
        _ => {
            return Err(unsupported(
                b.ctx,
                b.source,
                "unknown elemental decomposition",
            ));
        }
    };
    if needs_upcast {
        b.convert(result, original)
    } else {
        Ok(result)
    }
}

fn hypot(b: &mut Builder<'_>, x: Value, y: Value) -> Result<Value> {
    let high = b.max(x, y)?;
    let low = b.min(x, y)?;
    let ratio = expr!(b, x; (mul (div low high) (div low high)))?;
    let s = expr!(b, x; (sqrt (add 1.0 ratio)))?;
    expr!(b, x; (select (eq high low) (mul (sqrt 2.0) high)
        (select (and (eq s 1.0) (gt ratio 0.0))
            (add high (div (mul high ratio) 2.0)) (mul high s))))
}

fn inverse_kernel(b: &mut Builder<'_>, z: Value) -> Result<(Value, Value)> {
    let x = expr!(b, z; (abs (real z)))?;
    let y = expr!(b, x; (abs (imag z)))?;
    let safe_max = b.float(largest(b, x)?.sqrt() / 8.0, x)?;
    let xp1 = expr!(b, x; (add x 1.0))?;
    let xm1 = expr!(b, x; (sub x 1.0))?;
    let abs_xm1 = b.abs(xm1)?;
    let r = hypot(b, xp1, y)?;
    let s = hypot(b, abs_xm1, y)?;
    let a = expr!(b, x; (mul 0.5 (add r s)))?;
    let half_apx = expr!(b, x; (mul 0.5 (add a x)))?;
    let rpxp1 = b.add(r, xp1)?;
    let smxm1 = b.sub(s, xm1)?;
    let spxm1 = b.add(s, xm1)?;
    let yy = b.mul(y, y)?;
    let real = expr!(b, x; (select (ge (max x y) safe_max) y
        (select (le x 1.0)
            (sqrt (mul half_apx (add (div yy rpxp1) smxm1)))
            (mul y (sqrt (add (div half_apx rpxp1) (div half_apx spxm1)))))))?;
    let safe_opt = expr!(b, x; (select (lt x (mul safe_max 1000000000000.0))
        (mul safe_max 0.000001) (mul safe_max 100.0)))?;
    let y_large = b.ge(y, safe_opt)?;
    let high = b.select(y_large, y, x)?;
    let limit = b.select(y_large, safe_opt, safe_max)?;
    let inf = b.float(f64::INFINITY, x)?;
    let ratio = expr!(b, x; (select (and y_large (not (eq y inf))) (div x y) 0.0))?;
    let large = expr!(b, x; (add (add (log 2.0) (log high))
        (mul 0.5 (log1p (mul ratio ratio)))))?;
    let min_normal = match element(b, x)? {
        ElementType::F64 => f64::MIN_POSITIVE,
        ElementType::F32 | ElementType::Bf16 => f64::from(f32::MIN_POSITIVE),
        _ => 0.00006103515625,
    };
    let tiny = b.float(min_normal.sqrt() * 4.0, x)?;
    let tiny_region = expr!(b, x; (and (lt y tiny) (lt x 1.0)))?;
    let ap1 = expr!(b, x; (add a 1.0))?;
    let half_yy_over_r = expr!(b, x; (div (mul 0.5 yy) rpxp1))?;
    let am1 = expr!(b, x; (select tiny_region (neg (div (mul xp1 xm1) ap1))
        (select (ge x 1.0) (add half_yy_over_r (mul 0.5 spxm1))
            (select (le a 1.5) (add half_yy_over_r (div (mul 0.5 yy) smxm1))
                (sub a 1.0)))))?;
    let sq = expr!(b, x; (sqrt (mul am1 ap1)))?;
    let imag = expr!(b, x; (select (ge high limit) large
        (select tiny_region (div y sq) (log1p (add am1 sq)))))?;
    Ok((real, imag))
}

fn inverse_complex(b: &mut Builder<'_>, name: &str, z: Value) -> Result<Value> {
    let x = b.real(z)?;
    let y = b.imag(z)?;
    let input = if name == "asinh" {
        expr!(b, x; (complex (neg y) x))?
    } else {
        z
    };
    let (r, i) = inverse_kernel(b, input)?;
    match name {
        "_asin_acos_kernel" => b.complex(r, i),
        "asin" => expr!(b, x; (complex (atan2 x r) (select (lt y 0.0) (neg i) i))),
        "acos" => expr!(b, x; (complex (atan2 r x) (select (lt y 0.0) i (neg i)))),
        "acosh" => {
            let angle = b.atan2(r, x)?;
            expr!(b, x; (complex i (select (lt y 0.0) (neg angle) angle)))
        }
        "asinh" => expr!(b, x; (complex (select (lt x 0.0) (neg i) i) (atan2 y r))),
        _ => Err(unsupported(b.ctx, b.source, "unknown complex inverse")),
    }
}

fn complex_atanh(b: &mut Builder<'_>, z: Value) -> Result<Value> {
    let x = b.real(z)?;
    let y = b.imag(z)?;
    let ax = b.abs(x)?;
    let ay = b.abs(y)?;
    let inv_epsilon = if element(b, x)? == ElementType::F64 {
        9007199254740994.0
    } else {
        16777218.0
    };
    let inverse = b.float(inv_epsilon, x)?;
    let limit = b.float(inv_epsilon * inv_epsilon, x)?;
    let safe = expr!(b, x; (and (lt ax limit) (lt ay limit)))?;
    let one_minus_ax = expr!(b, x; (sub 1.0 ax))?;
    let yy = b.mul(y, y)?;
    let inf = b.float(f64::INFINITY, x)?;
    let quotient = expr!(b, x; (select safe
        (div ax (add (mul one_minus_ax one_minus_ax) yy))
        (select (lt (mul ay inverse) ax) (div 1.0 ax)
            (select (or (eq ax inf) (eq ay inf)) 0.0
                (div (div 1.0 (add (div ax y) (div y ax))) y)))))?;
    let real = expr!(b, x; (mul (mul (select (ge x 0.0) 1.0 (neg 1.0))
        (log1p (mul 4.0 quotient))) 0.25))?;
    let pi = b.float(std::f64::consts::PI, x)?;
    let imaginary = expr!(b, x; (mul (select safe
        (atan2 (add y y) (sub (mul one_minus_ax (add 1.0 ax)) yy))
        (mul (select (ge y 0.0) 1.0 (neg 1.0)) pi)) 0.5))?;
    b.complex(real, imaginary)
}

const LANCZOS: [f64; 8] = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
];

fn gamma(b: &mut Builder<'_>, x: Value, derivative: bool) -> Result<Value> {
    let reflect = expr!(b, x; (lt x 0.5))?;
    let z = expr!(b, x; (select reflect (neg x) (sub x 1.0)))?;
    let mut a = b.float(0.9999999999998099, x)?;
    let mut prime = b.float(0.0, x)?;
    for (i, coefficient) in LANCZOS.into_iter().enumerate() {
        let coefficient = b.float(coefficient, x)?;
        let index = b.float((i + 1) as f64, x)?;
        let denominator = b.add(z, index)?;
        a = expr!(b, x; (add a (div coefficient denominator)))?;
        if derivative {
            prime = expr!(b, x; (sub prime (div coefficient (mul denominator denominator))))?;
        }
    }
    let log_base = b.float(7.5_f64.ln(), x)?;
    let log_t = expr!(b, x; (add log_base (log1p (div z 7.5))))?;
    let t = expr!(b, x; (add z 7.5))?;
    let pi = b.float(std::f64::consts::PI, x)?;
    if derivative {
        let base = expr!(b, x; (sub (add log_t (div prime a)) (div 7.0 t)))?;
        let reduced = expr!(b, x; (add x (abs (floor (add x 0.5)))))?;
        let angle = b.mul(pi, reduced)?;
        let reflection = expr!(b, x; (sub base (div (mul pi (cos angle)) (sin angle))))?;
        let result = b.select(reflect, reflection, base)?;
        let nan = b.float(f64::NAN, x)?;
        let negative_inf = b.float(f64::NEG_INFINITY, x)?;
        expr!(b, x; (select (eq x 0.0) negative_inf
            (select (and (lt x 0.0) (eq x (floor x))) nan result)))
    } else {
        let normalization = b.float((2.0 * std::f64::consts::PI).ln() / 2.0, x)?;
        let base = expr!(b, x; (add (add normalization
            (mul (sub (add z 0.5) (div t log_t)) log_t)) (log a)))?;
        let fraction = expr!(b, x; (sub (abs x) (floor (abs x))))?;
        let fraction = expr!(b, x; (select (lt 0.5 fraction) (sub 1.0 fraction) fraction))?;
        let denominator = expr!(b, x; (log (sin (mul pi fraction))))?;
        let log_pi = b.float(std::f64::consts::PI.ln(), x)?;
        let reflection = expr!(b, x; (sub (sub log_pi denominator) base))?;
        let infinity = b.float(f64::INFINITY, x)?;
        let reflection =
            expr!(b, x; (select (eq (abs denominator) infinity) (neg denominator) reflection))?;
        expr!(b, x; (select (eq (abs x) infinity) infinity (select reflect reflection base)))
    }
}

fn zeta(b: &mut Builder<'_>, x: Value, q: Value) -> Result<Value> {
    let negative = b.neg(x)?;
    let mut acc = q;
    let mut sum = b.pow(q, negative)?;
    for _ in 0..9 {
        acc = expr!(b, x; (add acc 1.0))?;
        let term = b.pow(acc, negative)?;
        sum = b.add(sum, term)?;
    }
    acc = expr!(b, x; (add acc 1.0))?;
    let power = b.pow(acc, negative)?;
    let correction = expr!(b, x; (div (pow acc (neg (sub x 1.0))) (sub x 1.0)))?;
    let reciprocal_square = expr!(b, x; (div 1.0 (mul acc acc)))?;
    let mut horner = b.float(0.0, x)?;
    let coefficients = [
        -7.166165256175667e18,
        1.8152105401943546e17,
        -4_597_978_722_407_473.0,
        1.1646782814350067e14,
        -2.950130727918164e12,
        7.47242496e10,
        -1.8924375803183792e9,
        47900160.0,
        -1209600.0,
        30240.0,
        -720.0,
    ];
    for (i, coefficient) in coefficients.into_iter().enumerate() {
        let left = b.float((22 - 2 * i) as f64, x)?;
        let right = b.float((21 - 2 * i) as f64, x)?;
        let coefficient = b.float(1.0 / coefficient, x)?;
        horner = expr!(b, x; (mul (mul (add x left) (add x right))
            (mul reciprocal_square (add horner coefficient))))?;
    }
    let twelfth = b.float(1.0 / 12.0, x)?;
    let accurate = expr!(b, x; (add (add sum correction)
        (mul power (add 0.5 (mul (div x acc) (add twelfth horner))))))?;
    let smallest = b.float(
        if element(b, x)? == ElementType::F64 {
            f64::from_bits(1)
        } else {
            f64::from(f32::from_bits(1))
        },
        x,
    )?;
    let output = expr!(b, x; (select (lt (abs power) (mul (abs sum) smallest)) sum accurate))?;
    let nan = b.float(f64::NAN, x)?;
    let inf = b.float(f64::INFINITY, x)?;
    let output = expr!(b, x; (select (lt x 1.0) nan output))?;
    let output = expr!(b, x; (select (and (le q 0.0) (ne x (floor x))) nan output))?;
    let output = expr!(b, x; (select (and (le q 0.0) (eq q (floor q)))
        (select (and (eq x (floor x)) (eq (rem x 2.0) 0.0)) inf nan) output))?;
    expr!(b, x; (select (eq x 1.0) inf output))
}

fn polynomial(b: &mut Builder<'_>, x: Value, coefficients: &[f64]) -> Result<Value> {
    let mut value = b.float(0.0, x)?;
    for &coefficient in coefficients {
        let coefficient = b.float(coefficient, x)?;
        value = expr!(b, x; (add (mul value x) coefficient))?;
    }
    Ok(value)
}

fn erf(b: &mut Builder<'_>, x: Value, complement: bool) -> Result<Value> {
    if element(b, x)? == ElementType::F32 {
        return erf32(b, x, complement);
    }
    let square = b.mul(x, x)?;
    let numerator = polynomial(
        b,
        square,
        &[
            9.604973739870516,
            90.02601972038427,
            2232.005345946843,
            7003.325141128051,
            55592.3013010395,
        ],
    )?;
    let denominator = polynomial(
        b,
        square,
        &[
            1.0,
            33.56171416475031,
            521.3579497801527,
            4594.323829709801,
            22_629.000_061_389_09,
            49267.39426086359,
        ],
    )?;
    let small = expr!(b, x; (div (mul x numerator) denominator))?;
    let magnitude = b.abs(x)?;
    let p = polynomial(
        b,
        magnitude,
        &[
            2.461969814735305e-10,
            0.5641895648310688,
            7.463210564422699,
            48.63719709856814,
            196.5208329560771,
            526.4451949954773,
            934.5285271719576,
            1027.5518868951572,
            557.5353353693993,
        ],
    )?;
    let q = polynomial(
        b,
        magnitude,
        &[
            1.0,
            13.228195115474499,
            86.70721408859897,
            354.9377788878199,
            975.7085017432055,
            1823.9091668790973,
            2246.3376081871097,
            1656.6630919416135,
            557.5353408177277,
        ],
    )?;
    let r = polynomial(
        b,
        magnitude,
        &[
            0.5641895835477551,
            1.275366707599781,
            5.019050422511805,
            6.160210979930536,
            7.4097426995044895,
            2.9788666537210024,
        ],
    )?;
    let s = polynomial(
        b,
        magnitude,
        &[
            1.0,
            2.2605286322011726,
            9.396035249380015,
            12.048953980809666,
            17.08144507475659,
            9.608968090632859,
            3.369076451000815,
        ],
    )?;
    let ratio = expr!(b, x; (select (lt magnitude 8.0) (div p q) (div r s)))?;
    let max_log = b.float(
        if element(b, x)? == ElementType::F64 {
            709.782712893384
        } else {
            88.72283905206835
        },
        x,
    )?;
    let tail = expr!(b, x; (select (gt square max_log) 0.0 (mul (exp (neg square)) ratio)))?;
    let tail = expr!(b, x; (select (lt x 0.0) (sub 2.0 tail) tail))?;
    if complement {
        expr!(b, x; (select (lt magnitude 1.0) (sub 1.0 small) tail))
    } else {
        expr!(b, x; (select (lt magnitude 1.0) small (sub 1.0 tail)))
    }
}

fn erf_inv(b: &mut Builder<'_>, x: Value) -> Result<Value> {
    let w = expr!(b, x; (neg (log1p (mul x (neg x)))))?;
    if element(b, x)? == ElementType::F32 {
        let small = expr!(b, x; (sub w 2.5))?;
        let large = expr!(b, x; (sub (sqrt w) 3.0))?;
        let small = polynomial(
            b,
            small,
            &[
                2.81022636e-8,
                3.43273939e-7,
                -3.5233877e-6,
                -4.39150654e-6,
                0.00021858087,
                -0.00125372503,
                -0.00417768164,
                0.246640727,
                1.50140941,
            ],
        )?;
        let large = polynomial(
            b,
            large,
            &[
                -0.000200214257,
                0.000100950558,
                0.00134934322,
                -0.00367342844,
                0.00573950773,
                -0.0076224613,
                0.00943887047,
                1.00167406,
                2.83297682,
            ],
        )?;
        let inf = b.float(f64::INFINITY, x)?;
        return expr!(b, x; (select (eq (abs x) 1.0) (mul x inf)
            (mul x (select (lt w 5.0) small large))));
    }
    let low = expr!(b, x; (sub w 3.125))?;
    let mid = expr!(b, x; (sub (sqrt w) 3.25))?;
    let high = expr!(b, x; (sub (sqrt w) 5.0))?;
    let a = polynomial(
        b,
        low,
        &[
            -3.644_412_064_017_82e-21,
            -1.6850591381820166e-19,
            1.28584807152564e-18,
            1.1157877678025181e-17,
            -1.333171662854621e-16,
            2.0972767875968562e-17,
            6.637638134358324e-15,
            -4.054566272975207e-14,
            -8.151934197605472e-14,
            2.6335093153082323e-12,
            -1.2975133253453532e-11,
            -5.415412054294628e-11,
            1.0512122733215323e-9,
            -4.112633980346984e-9,
            -2.9070369957882005e-8,
            4.2347877827932404e-7,
            -1.3654692000834679e-6,
            -1.3882523362786469e-5,
            0.00018673420803405714,
            -0.000740702534166267,
            -0.006033670871430149,
            0.24015818242558962,
            1.6536545626831027,
        ],
    )?;
    let c = polynomial(
        b,
        mid,
        &[
            2.2137376921775787e-9,
            9.075656193888539e-8,
            -2.7517406297064545e-7,
            1.8239629214389228e-8,
            1.5027403968909828e-6,
            -4.013867526981546e-6,
            2.9234449089955446e-6,
            1.2475304481671779e-5,
            -4.7318229009055734e-5,
            6.828485145957318e-5,
            2.4031110387097894e-5,
            -0.0003550375203628475,
            0.0009532893797373805,
            -0.0016882755560235047,
            0.002491442096107851,
            -0.003751208507569241,
            0.005370914553590064,
            1.0052589676941592,
            3.0838856104922208,
        ],
    )?;
    let d = polynomial(
        b,
        high,
        &[
            -2.7109920616438573e-11,
            -2.555641816996525e-10,
            1.5076572693500548e-9,
            -3.789465440126737e-9,
            7.61570120807834e-9,
            -1.496002662714924e-8,
            2.914795345090108e-8,
            -6.771199775845234e-8,
            2.2900482228026655e-7,
            -9.9298272942317e-7,
            4.526062597223154e-6,
            -1.968177810553167e-5,
            7.599527703001776e-5,
            -0.00021503011930044477,
            -0.00013871931833623122,
            1.0103004648645344,
            4.849906401408584,
        ],
    )?;
    let inf = b.float(f64::INFINITY, x)?;
    expr!(b, x; (select (eq (abs x) 1.0) (mul x inf)
        (mul x (select (lt w 6.25) a (select (lt w 16.0) c d)))))
}

fn chebyshev(b: &mut Builder<'_>, x: Value, coefficients: &[f64]) -> Result<Value> {
    let mut b0 = b.float(0.0, x)?;
    let mut b1 = b0;
    let mut b2 = b0;
    for &coefficient in coefficients {
        b2 = b1;
        b1 = b0;
        let coefficient = b.float(coefficient, x)?;
        b0 = expr!(b, x; (add (sub (mul x b1) b2) coefficient))?;
    }
    expr!(b, x; (mul 0.5 (sub b0 b2)))
}

fn bessel(b: &mut Builder<'_>, x: Value) -> Result<Value> {
    let z = b.abs(x)?;
    let a = expr!(b, x; (sub (mul 0.5 z) 2.0))?;
    let c = expr!(b, x; (sub (div 32.0 z) 2.0))?;
    let (a_start, c_start) = if element(b, x)? == ElementType::F64 {
        (0, 0)
    } else {
        (12, 18)
    };
    let a = chebyshev(
        b,
        a,
        &[
            2.7779141127610464e-18,
            -2.111421214358166e-17,
            1.5536319577362005e-16,
            -1.1055969477353863e-15,
            7.600684294735407e-15,
            -5.042185504727912e-14,
            3.223793365945575e-13,
            -1.983974397764944e-12,
            1.1736186298890902e-11,
            -6.663489723502028e-11,
            3.625590281552117e-10,
            -1.8872497517228293e-9,
            9.381537386495772e-9,
            -4.445059128796328e-8,
            2.0032947535521353e-7,
            -8.568720264695455e-7,
            3.4702513081376785e-6,
            -1.3273163656039436e-5,
            4.781565107550054e-5,
            -0.00016176081582589674,
            0.0005122859561685758,
            -0.0015135724506312531,
            0.004156422944312888,
            -0.010564084894626198,
            0.024726449030626517,
            -0.052945981208095,
            0.1026436586898471,
            -0.17641651835783406,
            0.25258718644363365,
        ][a_start..],
    )?;
    let c = chebyshev(
        b,
        c,
        &[
            7.517296310842105e-18,
            4.414348323071708e-18,
            -4.6503053684893586e-17,
            -3.209525921993424e-17,
            2.96262899764595e-16,
            3.308202310920928e-16,
            -1.8803547755107825e-15,
            -3.814403072437008e-15,
            1.0420276984128803e-14,
            4.272440016711951e-14,
            -2.1015418427726643e-14,
            -4.083551111092197e-13,
            -7.198551776245909e-13,
            2.0356285441470895e-12,
            1.412_580_743_661_378e-11,
            3.2526035830154884e-11,
            -1.8974958123505412e-11,
            -5.589743462196584e-10,
            -3.835380385964237e-9,
            -2.6314688468895195e-8,
            -2.512236237870209e-7,
            -3.882564808877691e-6,
            -0.00011058893876262372,
            -0.009761097491361468,
            0.7785762350182801,
        ][c_start..],
    )?;
    expr!(b, x; (mul (sign x) (select (le z 8.0) (mul z a) (div c (sqrt z)))))
}

fn erf32(b: &mut Builder<'_>, x: Value, complement: bool) -> Result<Value> {
    if !complement {
        let low = b.float(-4.0, x)?;
        let clamped = expr!(b, x; (max low (min x 4.0)))?;
        let square = b.mul(clamped, clamped)?;
        let alpha = polynomial(
            b,
            square,
            &[
                -2.72614225801306e-10,
                2.77068142495902e-8,
                -2.10102402082508e-6,
                -5.69250639462346e-5,
                -7.34990630326855e-4,
                -2.95459980854025e-3,
                -1.60960333262415e-2,
            ],
        )?;
        let beta = polynomial(
            b,
            square,
            &[
                -1.45660718464996e-5,
                -2.13374055278905e-4,
                -1.68282697438203e-3,
                -7.37332916720468e-3,
                -1.42647390514189e-2,
            ],
        )?;
        let negative_one = b.float(-1.0, x)?;
        return expr!(b, x; (max negative_one (min (div (mul clamped alpha) beta) 1.0)));
    }
    let square = b.mul(x, x)?;
    let small = polynomial(
        b,
        square,
        &[
            7.853861353153693e-5,
            -8.010193625184903e-4,
            5.188327685732524e-3,
            -2.685381193529856e-2,
            1.128358514861418e-1,
            -3.7612625824233e-1,
            1.128_379_165_726_71,
        ],
    )?;
    let reciprocal = expr!(b, x; (div 1.0 square))?;
    let p = polynomial(
        b,
        reciprocal,
        &[
            2.326819970068386e-2,
            -1.387039388740657e-1,
            3.687424674597105e-1,
            -5.824733027278666e-1,
            6.210004621745983e-1,
            -4.944515323274145e-1,
            3.404879937665872e-1,
            -2.741127028184656e-1,
            5.638259427386472e-1,
        ],
    )?;
    let r = polynomial(
        b,
        reciprocal,
        &[
            -1.047766399936249e1,
            1.297719955372516e1,
            -7.495518717768503,
            2.921019019210786,
            -1.0152652792027,
            4.218463358204948e-1,
            -2.820767439740514e-1,
            5.641895067754075e-1,
        ],
    )?;
    let magnitude = b.abs(x)?;
    let tail = expr!(b, x; (mul (mul (exp (neg square)) (div 1.0 magnitude))
        (select (lt magnitude 2.0) p r)))?;
    let limit = b.float(88.72283905206835, x)?;
    let tail = expr!(b, x; (select (gt square limit) 0.0 tail))?;
    let tail = expr!(b, x; (select (lt x 0.0) (sub 2.0 tail) tail))?;
    expr!(b, x; (select (lt magnitude 1.0) (sub 1.0 (mul x small)) tail))
}
