use std::{
    io::Write,
    process::{Command, Stdio},
};
use xlahlo::{Literal, PrimitiveType as P, text};

#[test]
#[ignore = "requires XLAHLO_NUMERIC_ORACLE built with tools/numeric_oracle.py"]
fn scalar_conversions_match_pinned_native_dependency_types() {
    let executable = std::env::var_os("XLAHLO_NUMERIC_ORACLE").expect("XLAHLO_NUMERIC_ORACLE");
    let types = [
        P::F16,
        P::BF16,
        P::F32,
        P::F64,
        P::F8E3M4,
        P::F8E4M3,
        P::F8E5M2,
        P::F8E4M3FN,
        P::F8E4M3FNUZ,
        P::F8E4M3B11FNUZ,
        P::F8E5M2FNUZ,
        P::F8E8M0FNU,
        P::F4E2M1FN,
        P::F6E2M3FN,
        P::F6E3M2FN,
    ];
    let mut values = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from(f32::MAX),
        65519.0,
        65520.0,
        1.0004882812500002,
        1.0039062500000002,
        1.0625000000000002,
        3.4028235e38,
        449.0,
        465.0,
        2_f64.powi(-127),
    ];
    for ty in [P::F16, P::BF16] {
        let fraction = if ty == P::F16 { 10 } else { 7 };
        for index in 0..32 {
            let midpoint = 1.0 + (f64::from(index) + 0.5) * 2_f64.powi(-fraction);
            values.extend([
                midpoint,
                f64::from_bits(midpoint.to_bits() - 1),
                f64::from_bits(midpoint.to_bits() + 1),
            ]);
        }
    }
    let boundary = 2_f64.powi(-127).to_bits();
    values.extend([f64::from_bits(boundary - 1), f64::from_bits(boundary + 1)]);
    let mut state = 0x7834_7245_a67b_0192_u64;
    for _ in 0..1024 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let exponent = 873 + (state % 301);
        values.push(f64::from_bits(
            (state & 0x800f_ffff_ffff_ffff) | (exponent << 52),
        ));
    }
    let cases = types
        .into_iter()
        .flat_map(|ty| values.iter().copied().map(move |value| (ty, value)))
        .collect::<Vec<_>>();
    let input = cases
        .iter()
        .map(|(ty, value)| format!("{} {:x}\n", ty.hlo_name(), value.to_bits()))
        .collect::<String>();
    let mut child = Command::new(executable)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = String::from_utf8(output.stdout).unwrap();
    let lines = output.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), cases.len());
    for ((ty, value), line) in cases.into_iter().zip(lines) {
        let (bits, finite) = line.split_once(' ').unwrap();
        let expected = u128::from_str_radix(bits, 16).unwrap();
        let mut scalar = value.to_string();
        if value.is_finite() && !scalar.contains(['.', 'e', 'E']) {
            scalar.push_str(".0");
        }
        let source = format!(
            "HloModule m ENTRY main {{ ROOT c = {}[] constant({scalar}) }}",
            ty.hlo_name()
        );
        let parsed = text::parse_module(&source);
        if value.is_finite() && finite == "0" {
            assert!(parsed.is_err(), "native conversion must overflow: {source}");
        } else {
            let parsed = parsed.unwrap_or_else(|error| panic!("{source}: {error}"));
            let literal = Literal::from_proto(
                parsed.as_proto().computations[0].instructions[0]
                    .literal
                    .as_ref()
                    .unwrap()
                    .clone(),
            )
            .unwrap();
            assert_eq!(
                literal.bits().unwrap(),
                [expected],
                "{source}; native={line}"
            );
        }
    }
}
