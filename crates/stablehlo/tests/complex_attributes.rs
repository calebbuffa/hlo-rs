use stablehlo::text::{parse_attribute, print_attribute};
use stablehlo::{Attribute, ElementType};

#[test]
fn complex_numeric_attributes_preserve_component_types_bits_and_nonfinite_payloads() {
    for ty in [ElementType::C64, ElementType::C128] {
        let bits = if ty == ElementType::C64 {
            vec![
                0, 0x80000000, 0x3f800000, 0x7f800000, 0xff800000, 0x7fc12345,
            ]
        } else {
            vec![
                0,
                0x8000000000000000,
                0x3ff0000000000000,
                0x7ff0000000000000,
                0xfff0000000000000,
                0x7ff8000000012345,
            ]
        };
        for &real in &bits {
            for &imaginary in &bits {
                let value = Attribute::Complex {
                    ty,
                    real,
                    imaginary,
                };
                value.verify().unwrap();
                let emitted = print_attribute(&value).unwrap();
                assert_eq!(parse_attribute(&emitted).unwrap(), value, "{emitted}");
            }
        }
    }
    assert_eq!(
        parse_attribute("#complex.number<:f32 1.5, -2.5>").unwrap(),
        Attribute::Complex {
            ty: ElementType::C64,
            real: u64::from(1.5_f32.to_bits()),
            imaginary: u64::from((-2.5_f32).to_bits())
        }
    );
}

#[test]
fn complex_attributes_reject_wrong_domains_invalid_widths_and_malformed_syntax() {
    for value in [
        Attribute::Complex {
            ty: ElementType::F64,
            real: 0,
            imaginary: 0,
        },
        Attribute::Complex {
            ty: ElementType::C64,
            real: 1 << 32,
            imaginary: 0,
        },
        Attribute::Complex {
            ty: ElementType::C64,
            real: 0,
            imaginary: 1 << 32,
        },
    ] {
        assert!(value.verify().is_err());
        assert!(print_attribute(&value).is_err());
    }
    for source in [
        "#complex.number<:i32 1, 2>",
        "#complex.number<:f16 1.0, 2.0>",
        "#complex.number<:f32 1.0>",
        "#complex.number<:f32 1.0, 2.0, 3.0>",
        "#complex.number<:f32 0x10000000000000000, 0.0>",
        "#complex.number<:f32 1.0, 2.0> trailing",
    ] {
        assert!(parse_attribute(source).is_err(), "{source}");
    }
}

#[test]
fn complex_parser_matches_reference_intermediate_double_rounding_and_hex_semantics() {
    let expected = Attribute::Complex {
        ty: ElementType::C64,
        real: u64::from(1.0_f32.to_bits()),
        imaginary: 0,
    };
    for source in [
        "#complex.number<:f32 1.0000000596046447753906250000000000000001, 0.0>",
        "#complex<number<:f32 1.0, 0x100000000>>",
    ] {
        assert_eq!(parse_attribute(source).unwrap(), expected, "{source}");
    }
    let value = parse_attribute("#complex.number<:f32 0x7FF82468A0000000, 0.0>").unwrap();
    assert_eq!(
        value,
        Attribute::Complex {
            ty: ElementType::C64,
            real: 0x7fc12345,
            imaginary: 0,
        }
    );
    assert_eq!(
        parse_attribute(&print_attribute(&value).unwrap()).unwrap(),
        value
    );
}

#[test]
fn complex_printing_does_not_silently_quiet_signaling_nan_payloads() {
    let value = Attribute::Complex {
        ty: ElementType::C64,
        real: 0x7f812345,
        imaginary: 0,
    };
    value.verify().unwrap();
    assert!(print_attribute(&value).is_err());
}
