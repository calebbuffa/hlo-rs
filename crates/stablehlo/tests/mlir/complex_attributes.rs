use stablehlo::mlir::assembly::{parse_attribute, print_attribute};
use stablehlo::{
    attributes::{ComplexAttr, ScalarBitsAttr},
    pliron::context::Context,
    types::ElementType as E,
};

#[test]
fn complex_number_forms_parse_double_before_component_conversion() {
    let ctx = Context::new();
    for source in [
        "#complex.number<:f32 0x8000000000000000, 0x7FF8024680000000>",
        "#complex<number<:f32 0x8000000000000000, 0x7FF8024680000000>>",
    ] {
        let attribute = parse_attribute(&ctx, source).unwrap();
        let complex = attribute.downcast_ref::<ComplexAttr>().unwrap();
        assert_eq!(complex.real.words, [0x8000_0000]);
        assert_eq!(complex.imaginary.words, [0x7fc0_1234]);
        let printed = print_attribute(&ctx, &*attribute).unwrap();
        let fresh = Context::new();
        let restored = parse_attribute(&fresh, &printed).unwrap();
        assert_eq!(print_attribute(&fresh, &*restored).unwrap(), printed);
    }
    let attribute = parse_attribute(
        &ctx,
        "#complex.number<:f16 1.00048828125000000000001, -0.0>",
    )
    .unwrap();
    let complex = attribute.downcast_ref::<ComplexAttr>().unwrap();
    assert_eq!(complex.real.words, [0x3c00]);
    assert_eq!(complex.imaginary.words, [0x8000]);
    let direct = parse_attribute(&ctx, "1.00048828125000000000001 : f16").unwrap();
    assert_eq!(
        direct.downcast_ref::<ScalarBitsAttr>().unwrap().words,
        [0x3c01]
    );
    let attribute = parse_attribute(
        &ctx,
        "#complex.number<:f32 0x7FF0000020000000, 0xFFF0000020000000>",
    )
    .unwrap();
    let complex = attribute.downcast_ref::<ComplexAttr>().unwrap();
    assert_eq!(complex.real.words, [0x7fc0_0001]);
    assert_eq!(complex.imaginary.words, [0xffc0_0001]);
    let attribute = parse_attribute(
        &ctx,
        "#complex.number<:f64 0x7FF0000000000001, 0xFFF0000000000001>",
    )
    .unwrap();
    let complex = attribute.downcast_ref::<ComplexAttr>().unwrap();
    assert_eq!(complex.real.words, [0x7ff0_0000_0000_0001]);
    assert_eq!(complex.imaginary.words, [0xfff0_0000_0000_0001]);
}

#[test]
fn all_small_component_encodings_roundtrip_or_report_the_exact_signaling_nan_boundary() {
    let ctx = Context::new();
    for (element, ieee) in [
        (E::F4E2M1Fn, None),
        (E::F6E2M3Fn, None),
        (E::F6E3M2Fn, None),
        (E::F8E3M4, Some((3, 4))),
        (E::F8E4M3, Some((4, 3))),
        (E::F8E5M2, Some((5, 2))),
        (E::F8E4M3Fn, None),
        (E::F8E4M3FnUz, None),
        (E::F8E4M3B11FnUz, None),
        (E::F8E5M2FnUz, None),
        (E::F8E8M0FnUz, None),
    ] {
        let width = element.bit_width().unwrap();
        for bits in 0..1_u64 << width {
            let complex = ComplexAttr {
                ty: E::complex(element).unwrap().native(&ctx),
                real: ScalarBitsAttr {
                    ty: element.native(&ctx),
                    words: vec![bits],
                },
                imaginary: ScalarBitsAttr {
                    ty: element.native(&ctx),
                    words: vec![0],
                },
            };
            let signaling = ieee.is_some_and(|(exponent, fraction)| {
                let payload = bits & ((1 << fraction) - 1);
                (bits >> fraction) & ((1 << exponent) - 1) == (1 << exponent) - 1
                    && payload != 0
                    && payload & (1 << (fraction - 1)) == 0
            });
            let printed = print_attribute(&ctx, &complex);
            if signaling {
                assert!(printed.is_err(), "{element:?}: {bits:#x}");
            } else {
                let printed = printed.unwrap();
                let restored = parse_attribute(&ctx, &printed)
                    .unwrap_or_else(|error| panic!("{element:?}: {bits:#x}: {printed}: {error}"));
                assert_eq!(
                    restored.downcast_ref::<ComplexAttr>().unwrap(),
                    &complex,
                    "{printed}"
                );
            }
        }
    }
}

#[test]
fn larger_components_roundtrip_payloads_and_reject_unrepresentable_source_categories() {
    let ctx = Context::new();
    for (element, patterns) in [
        (E::F16, vec![0, 1, 0x8000, 0x3c00, 0x7bff, 0x7c00, 0x7e01]),
        (E::Bf16, vec![0, 1, 0x8000, 0x3f80, 0x7f7f, 0x7f80, 0x7fc1]),
        (
            E::Tf32,
            vec![0, 1, 0x40000, 0x1fc00, 0x3fbff, 0x3fc00, 0x3fe01],
        ),
        (
            E::F32,
            vec![0, 1, 0x8000_0000, 0x3f80_0000, 0x7f7f_ffff, 0x7fc0_1234],
        ),
        (
            E::F64,
            vec![
                0,
                1,
                0x8000_0000_0000_0000,
                0x7ff0_0000_0000_0001,
                0x7ff8_0000_0000_0017,
            ],
        ),
    ] {
        for bits in patterns {
            let complex = ComplexAttr {
                ty: E::complex(element).unwrap().native(&ctx),
                real: ScalarBitsAttr {
                    ty: element.native(&ctx),
                    words: vec![bits],
                },
                imaginary: ScalarBitsAttr {
                    ty: element.native(&ctx),
                    words: vec![bits],
                },
            };
            let printed = print_attribute(&ctx, &complex).unwrap();
            let restored = parse_attribute(&ctx, &printed).unwrap();
            assert_eq!(
                restored.downcast_ref::<ComplexAttr>().unwrap(),
                &complex,
                "{printed}"
            );
        }
    }
    for source in [
        "#complex.number<:i32 1.0, 2.0>",
        "#complex.number<:complex<f32> 1.0, 2.0>",
        "#complex.number<:f32 1.0>",
        "#complex.number<:f32 1.0, 2.0> trailing",
        "#complex<other<:f32 1.0, 2.0>>",
        "#complex.number<:f4E2M1FN 0x7FF0000000000000, 0.0>",
        "#complex.number<:f6E2M3FN 0x7FF8000000000000, 0.0>",
        "#complex.number<:f8E8M0FNU -1.0, 1.0>",
    ] {
        assert!(parse_attribute(&ctx, source).is_err(), "{source}");
    }
    for (source, bits) in [
        ("#complex.number<:f32 0x7F800001, 0x0>", 0),
        ("#complex.number<:f4E2M1FN -0.0, 0.0>", 8),
        ("#complex.number<:f6E2M3FN -0.0, 0.0>", 32),
        ("#complex.number<:f6E3M2FN -0.0, 0.0>", 32),
        ("#complex.number<:f8E4M3FNUZ -0.0, 0.0>", 0),
        ("#complex.number<:f8E8M0FNU -0.0, 0.0>", 0),
        ("#complex.number<:f8E8M0FNU 1.0e-300, 0.0>", 0),
        ("#complex.number<:f8E8M0FNU 0x7FF0000000000000, 0.0>", 255),
    ] {
        let parsed = parse_attribute(&ctx, source).unwrap();
        assert_eq!(
            parsed.downcast_ref::<ComplexAttr>().unwrap().real.words,
            [bits],
            "{source}"
        );
    }
}
