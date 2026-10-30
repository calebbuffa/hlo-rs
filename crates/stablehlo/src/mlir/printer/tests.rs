use super::*;
use crate::{RecoveryCategory, integer::IntegerValue};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{
        DictAttr, FPDoubleAttr, FPHalfAttr, FPSingleAttr, IntegerAttr, StringAttr,
        TypeAttr as PlironTypeAttr, UnitAttr, VecAttr,
    },
    builtin::types::IntegerType,
    combine::{Parser, eof},
    identifier::Identifier,
    location::Source,
    parsable::{self, Parsable, state_stream_from_iterator},
    printable::Printable,
    utils::{
        apfloat::{BFloat, Double, Float, Float8E4M3FN, Float8E5M2, FloatConvert, Half, Single},
        apint::{APInt, bw},
    },
};

fn dict(values: Vec<(&'static str, AttrObj)>) -> DictAttr {
    DictAttr::new(
        values
            .into_iter()
            .map(|(key, value)| (Identifier::new(key), value))
            .collect(),
    )
}

fn one(value: impl Attribute + 'static) -> DictAttr {
    dict(vec![("value", Box::new(value))])
}

#[test]
fn empty_and_nested_dictionaries_are_sorted_with_mlir_array_type_elision() {
    let ctx = Context::new();
    assert_eq!(mlir_dictionary(&ctx, &dict(vec![])).unwrap(), "{}");
    let inner = dict(vec![
        ("z", Box::new(UnitAttr::new())),
        (
            "a",
            Box::new(
                ScalarBitsAttr::integer(&ctx, ElementType::I64, IntegerValue::from(-1_i64))
                    .unwrap(),
            ),
        ),
    ]);
    let source = dict(vec![
        (
            "z",
            Box::new(VecAttr::new(vec![
                Box::new(
                    ScalarBitsAttr::integer(&ctx, ElementType::I64, IntegerValue::from(12_i64))
                        .unwrap(),
                ),
                Box::new(
                    ScalarBitsAttr::integer(&ctx, ElementType::Ui64, IntegerValue::from(12_i64))
                        .unwrap(),
                ),
                Box::new(UnitAttr::new()),
            ])),
        ),
        ("a", Box::new(inner)),
    ]);
    assert_eq!(
        mlir_dictionary(&ctx, &source).unwrap(),
        "{a = {a = -1 : i64, z}, z = [12, 12 : ui64, unit]}"
    );
    let mut entries = source
        .0
        .0
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Vec<_>>();
    entries.reverse();
    let reverse = DictAttr::new(entries);
    assert_eq!(
        mlir_dictionary(&ctx, &source).unwrap(),
        mlir_dictionary(&ctx, &reverse).unwrap()
    );
}

#[test]
fn byte_strings_follow_llvm_escapes_including_quotes_nuls_and_utf8() {
    let ctx = Context::new();
    let source = one(BytesAttr {
        value: vec![0, b'"', b'\\', b'\n', b'\r', b'\t', 0x7f, 0xff, b'A'],
    });
    assert_eq!(
        mlir_dictionary(&ctx, &source).unwrap(),
        r#"{value = "\00\22\\\0A\0D\09\7F\FFA"}"#
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(StringAttr::from("caf\u{e9}"))).unwrap(),
        r#"{value = "caf\C3\A9"}"#
    );
    assert_eq!(keyword("abc_12.$"), "abc_12.$");
    assert_eq!(keyword("0name"), "\"0name\"");
    assert_eq!(keyword("with-dash"), "\"with-dash\"");
}

#[test]
fn integer_widths_and_signedness_preserve_values_beyond_u128() {
    let ctx = Context::new();
    for (element, words, expected) in [
        (ElementType::I1, vec![1], "true"),
        (ElementType::Si1, vec![1], "-1 : si1"),
        (ElementType::Ui1, vec![1], "1 : ui1"),
        (ElementType::I8, vec![255], "-1 : i8"),
        (ElementType::Ui8, vec![255], "255 : ui8"),
        (ElementType::Si64, vec![u64::MAX], "-1 : si64"),
        (
            ElementType::Ui64,
            vec![u64::MAX],
            "18446744073709551615 : ui64",
        ),
        (
            ElementType::integer(129, Signedness::Unsigned).unwrap(),
            vec![0, 0, 1],
            "340282366920938463463374607431768211456 : ui129",
        ),
    ] {
        let value = ScalarBitsAttr {
            ty: element.native(&ctx),
            words,
        };
        assert_eq!(
            mlir_dictionary(&ctx, &one(value)).unwrap(),
            format!("{{value = {expected}}}")
        );
    }
    let value = IntegerAttr::new(
        IntegerType::get(&ctx, 129, Signedness::Unsigned),
        APInt::from_str("340282366920938463463374607431768211456", 129, 10).unwrap(),
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(value)).unwrap(),
        "{value = 340282366920938463463374607431768211456 : ui129}"
    );
}

#[test]
fn all_float_formats_use_raw_hex_without_decimal_conversion() {
    let ctx = Context::new();
    for element in [
        ElementType::F4E2M1Fn,
        ElementType::F6E2M3Fn,
        ElementType::F6E3M2Fn,
        ElementType::F16,
        ElementType::Bf16,
        ElementType::F32,
        ElementType::F64,
        ElementType::Tf32,
        ElementType::F8E3M4,
        ElementType::F8E4M3,
        ElementType::F8E4M3Fn,
        ElementType::F8E4M3FnUz,
        ElementType::F8E4M3B11FnUz,
        ElementType::F8E5M2,
        ElementType::F8E5M2FnUz,
        ElementType::F8E8M0FnUz,
    ] {
        let width = element.bit_width().unwrap();
        let mask = u64::MAX >> (64 - width);
        for bits in [0, 1, 1 << (width - 1), mask] {
            let source = one(ScalarBitsAttr {
                ty: element.native(&ctx),
                words: vec![bits],
            });
            let text = mlir_dictionary(&ctx, &source).unwrap();
            let literal = text
                .strip_prefix("{value = 0x")
                .unwrap()
                .split_once(" : ")
                .unwrap()
                .0;
            // Pinned Parser.cpp parseFloatFromIntegerLiteral checks active bits,
            // then constructs APFloat(semantics, APInt), without numeric conversion.
            let parsed = u64::from_str_radix(literal, 16).unwrap();
            assert_eq!(parsed, bits);
            assert_eq!(parsed & !mask, 0);
        }
    }
    for bits in [
        0,
        1,
        0x8000_0000,
        0x7f80_0000,
        0x7fc0_1234,
        0x7f80_0001,
        0xffc0_1234,
    ] {
        let text = mlir_dictionary(&ctx, &one(ScalarBitsAttr::f32_bits(&ctx, bits))).unwrap();
        assert_eq!(text, format!("{{value = 0x{bits:X} : f32}}"));
    }
    for bits in [
        0,
        1,
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff8_1234_5678_9abc,
        0xfff8_1234_5678_9abc,
    ] {
        assert_eq!(
            mlir_dictionary(&ctx, &one(ScalarBitsAttr::float_bits(&ctx, bits))).unwrap(),
            format!("{{value = 0x{bits:X} : f64}}")
        );
    }
}

#[test]
fn native_pliron_float_payloads_are_preserved() {
    let ctx = Context::new();
    assert_eq!(
        mlir_dictionary(&ctx, &one(FPSingleAttr(Single::from_bits(0x8000_0000)))).unwrap(),
        "{value = 0x80000000 : f32}"
    );
    assert_eq!(
        mlir_dictionary(
            &ctx,
            &one(FPDoubleAttr(Double::from_bits(0x7ff8_1234_5678_9abc)))
        )
        .unwrap(),
        "{value = 0x7FF8123456789ABC : f64}"
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(FPHalfAttr(Half::from_bits(0x8001)))).unwrap(),
        "{value = 0x8001 : f16}"
    );
}

#[test]
fn double_parsed_components_widen_exactly_instead_of_reinterpreting_small_bits() {
    let ctx = Context::new();
    fn check<F: Float + FloatConvert<Double>>(ctx: &Context, element: ElementType, bits: u64)
    where
        Double: FloatConvert<F>,
    {
        let original = F::from_bits(u128::from(bits));
        let scalar = ScalarBitsAttr {
            ty: element.native(ctx),
            words: vec![bits],
        };
        if original.is_signaling() {
            assert!(double_bits(ctx, &scalar).is_err());
        } else {
            let widened = Double::from_bits(u128::from(double_bits(ctx, &scalar).unwrap()));
            let roundtrip: F = widened.convert(&mut false).value;
            assert_eq!(
                roundtrip.to_bits(),
                original.to_bits(),
                "{element:?} 0x{bits:X}"
            );
        }
    }
    for bits in 0..256 {
        check::<Float8E5M2>(&ctx, ElementType::F8E5M2, bits);
        check::<Float8E4M3FN>(&ctx, ElementType::F8E4M3Fn, bits);
    }
    for bits in [0, 1, 0x8000, 0x3c00, 0x7c00, 0x7e01, 0x7c01, 0xfc00, 0xfe01] {
        check::<Half>(&ctx, ElementType::F16, bits);
    }
    for bits in [0, 1, 0x8000, 0x3f80, 0x7f80, 0x7fc1, 0x7f81, 0xff80, 0xffc1] {
        check::<BFloat>(&ctx, ElementType::Bf16, bits);
    }
    for (element, bits, expected) in [
        (ElementType::F4E2M1Fn, 7, 6.0_f64),
        (ElementType::F6E2M3Fn, 31, 7.5),
        (ElementType::F6E3M2Fn, 31, 28.0),
        (ElementType::Tf32, 127 << 10, 1.0),
        (ElementType::F8E4M3FnUz, 8 << 3, 1.0),
        (ElementType::F8E4M3B11FnUz, 11 << 3, 1.0),
        (ElementType::F8E5M2FnUz, 16 << 2, 1.0),
        (ElementType::F8E8M0FnUz, 127, 1.0),
        (ElementType::F8E8M0FnUz, 0, 2_f64.powi(-127)),
    ] {
        assert_eq!(
            double_bits(
                &ctx,
                &ScalarBitsAttr {
                    ty: element.native(&ctx),
                    words: vec![bits]
                }
            )
            .unwrap(),
            expected.to_bits()
        );
    }
}

#[test]
fn complex_scalar_bit_containers_split_components_without_losing_high_limbs() {
    let ctx = Context::new();
    let bits = ScalarBitsAttr {
        ty: ElementType::C64.native(&ctx),
        words: vec![(0x7fc0_1234_u64 << 32) | 0x8000_0000],
    };
    assert_eq!(
        mlir_dictionary(&ctx, &one(bits.clone())).unwrap(),
        "{value = #complex.number<:f32 0x8000000000000000, 0x7FF8024680000000>}"
    );
    let dense = DenseElementsAttr::splat(tensor(&ctx, &[2], ElementType::C64), Box::new(bits));
    assert_eq!(
        mlir_dictionary(&ctx, &one(dense)).unwrap(),
        "{value = dense<(0x80000000,0x7FC01234)> : tensor<2xcomplex<f32>>}"
    );
    let element =
        ElementType::complex(ElementType::integer(65, Signedness::Unsigned).unwrap()).unwrap();
    let bits = ScalarBitsAttr {
        ty: element.native(&ctx),
        words: vec![0, 3, 2],
    };
    let dense = DenseElementsAttr::splat(tensor(&ctx, &[1], element), Box::new(bits.clone()));
    assert_eq!(
        mlir_dictionary(&ctx, &one(dense)).unwrap(),
        "{value = dense<(18446744073709551616,18446744073709551617)> : tensor<1xcomplex<ui65>>}"
    );
    let error = mlir_dictionary(&ctx, &one(bits)).unwrap_err();
    assert_eq!(recovery(&error), RecoveryCategory::Unsupported);
}

#[test]
fn quantized_type_attributes_preserve_parameter_bits_and_storage_syntax() {
    let ctx = Context::new();
    let uniform = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: Some(-127),
            storage_max: None,
            expressed_type: ElementType::F32,
            quantization_dimension: None,
            scales: vec![0.5],
            zero_points: vec![2],
        },
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(PlironTypeAttr::new(uniform))).unwrap(),
        "{value = !quant.uniform<i8<-127:127>:f32, 0x3FE0000000000000:2>}"
    );
    let quantized_dense = DenseElementsAttr::splat(
        TensorType::get(&ctx, vec![2], uniform).into(),
        Box::new(ScalarBitsAttr {
            ty: ElementType::I8.native(&ctx),
            words: vec![2],
        }),
    );
    let error = mlir_dictionary(&ctx, &one(quantized_dense)).unwrap_err();
    assert_eq!(recovery(&error), RecoveryCategory::Unsupported);
    let quantized_ty = TensorType::get(&ctx, vec![2], uniform).into();
    let sparse = SparseElementsAttr {
        ty: quantized_ty,
        indices: vec![IndexTensorAttr { values: vec![0] }],
        values: VecAttr::new(vec![Box::new(ScalarBitsAttr {
            ty: ElementType::I8.native(&ctx),
            words: vec![2],
        })]),
    };
    let dense = DenseElementsAttr::splat(
        quantized_ty,
        Box::new(ScalarBitsAttr {
            ty: ElementType::I8.native(&ctx),
            words: vec![2],
        }),
    );
    for value in [Box::new(dense) as AttrObj, Box::new(sparse)] {
        value.verify(&ctx).unwrap();
        let source = dict(vec![
            ("a", Box::new(PlironTypeAttr::new(uniform))),
            ("nested", Box::new(dict(vec![("payload", value)]))),
        ]);
        let before = source.clone();
        let error = mlir_dictionary(&ctx, &source).unwrap_err();
        let diagnostic = pliron_mlir_builtin::Diagnostic::from_error(&error).unwrap();
        assert_eq!(
            diagnostic.recovery_category(),
            pliron_mlir_builtin::RecoveryCategory::Unsupported
        );
        assert!(
            matches!(diagnostic, pliron_mlir_builtin::Diagnostic::UnsupportedSerialization { reason, .. }
            if reason.contains("numeric elements grammar"))
        );
        assert_eq!(source, before);
        assert_eq!(
            mlir_dictionary(&ctx, &one(PlironTypeAttr::new(uniform))).unwrap(),
            "{value = !quant.uniform<i8<-127:127>:f32, 0x3FE0000000000000:2>}"
        );
    }
    let any = AnyQuantizedType::intern(&ctx, ElementType::Ui8, None, None, None);
    assert_eq!(
        mlir_dictionary(&ctx, &one(PlironTypeAttr::new(any))).unwrap(),
        "{value = !quant.any<u8>}"
    );
    let per_axis = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::Ui8,
            storage_min: None,
            storage_max: None,
            expressed_type: ElementType::F32,
            quantization_dimension: Some(0),
            scales: vec![0.5, 1.0],
            zero_points: vec![0, 2],
        },
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(PlironTypeAttr::new(per_axis))).unwrap(),
        "{value = !quant.uniform<u8:f32:0, {0x3FE0000000000000, 0x3FF0000000000000:2}>}"
    );
    let calibrated = CalibratedQuantizedType::intern(&ctx, ElementType::F32, -1.0, 2.0);
    assert_eq!(
        mlir_dictionary(&ctx, &one(PlironTypeAttr::new(calibrated))).unwrap(),
        "{value = !quant.calibrated<f32<0xBFF0000000000000:0x4000000000000000>>}"
    );
    let quantile = QuantileType::intern(
        &ctx,
        ElementType::I2,
        ElementType::F32,
        vec![0.0, 1.0, 2.0, 3.0],
        None,
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(PlironTypeAttr::new(quantile))).unwrap(),
        "{value = !quant.quantile<i2:f32, {0x0, 0x3FF0000000000000, 0x4000000000000000, 0x4008000000000000}>}"
    );
    let subchannel = SubchannelQuantizedType::intern(
        &ctx,
        SubchannelQuantizedSpec {
            storage: ElementType::I8,
            range: (None, None),
            expressed: ElementType::F32,
            quantized_dimensions: vec![0],
            block_sizes: vec![2],
            parameter_shape: vec![2],
            scales: vec![0x3f00_0000],
            zero_points: vec![0],
        },
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(PlironTypeAttr::new(subchannel))).unwrap(),
        "{value = !quant.uniform<i8:f32:{0:2}, {0x3FE0000000000000, 0x3FE0000000000000}>}"
    );
}

#[test]
fn dense_arrays_and_shaped_elements_use_mlir_literals() {
    let ctx = Context::new();
    let ty = tensor(&ctx, &[2, 2], ElementType::I64);
    let source =
        DenseElementsAttr::new(ty, DenseArrayAttr::integers(&ctx, vec![1, 2, 3, 4]).values);
    assert_eq!(
        mlir_dictionary(&ctx, &one(source)).unwrap(),
        "{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}"
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(DenseArrayAttr::integers(&ctx, vec![-1, 0]))).unwrap(),
        "{value = array<i64: -1, 0>}"
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(DenseArrayAttr::bools(&ctx, vec![true, false]))).unwrap(),
        "{value = array<i1: true, false>}"
    );
    assert_eq!(
        mlir_dictionary(
            &ctx,
            &one(DenseArrayAttr::floats_bits(
                &ctx,
                vec![0x8000_0000_0000_0000]
            ))
        )
        .unwrap(),
        "{value = array<f64: 0x8000000000000000>}"
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(DenseArrayAttr::integers(&ctx, vec![]))).unwrap(),
        "{value = array<i64>}"
    );
    let splat = DenseElementsAttr::splat(
        ty,
        Box::new(ScalarBitsAttr {
            ty: ElementType::I64.native(&ctx),
            words: vec![u64::MAX],
        }),
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(splat)).unwrap(),
        "{value = dense<-1> : tensor<2x2xi64>}"
    );
    let empty =
        DenseElementsAttr::new(tensor(&ctx, &[0, 2], ElementType::I8), VecAttr::new(vec![]));
    assert_eq!(
        mlir_dictionary(&ctx, &one(empty)).unwrap(),
        "{value = dense<> : tensor<0x2xi8>}"
    );
}

#[test]
fn dense_complex_pairs_are_not_coerced_to_other_attribute_types() {
    let ctx = Context::new();
    let complex = ComplexAttr {
        ty: ElementType::C64.native(&ctx),
        real: ScalarBitsAttr::f32_bits(&ctx, 0x8000_0000),
        imaginary: ScalarBitsAttr::f32_bits(&ctx, 0x7fc0_1234),
    };
    let dense = DenseElementsAttr::splat(
        tensor(&ctx, &[2], ElementType::C64),
        Box::new(complex.clone()),
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(dense)).unwrap(),
        "{value = dense<(0x80000000,0x7FC01234)> : tensor<2xcomplex<f32>>}"
    );
    assert_eq!(
        mlir_dictionary(&ctx, &one(complex)).unwrap(),
        "{value = #complex.number<:f32 0x8000000000000000, 0x7FF8024680000000>}"
    );
    let signaling = ComplexAttr {
        ty: ElementType::C64.native(&ctx),
        real: ScalarBitsAttr::f32_bits(&ctx, 0x7f80_0001),
        imaginary: ScalarBitsAttr::f32_bits(&ctx, 0),
    };
    let error = mlir_dictionary(&ctx, &one(signaling)).unwrap_err();
    assert_eq!(recovery(&error), RecoveryCategory::Unsupported);
}

#[test]
fn sparse_elements_and_symbols_use_mlir_syntax() {
    let ctx = Context::new();
    let value = SparseElementsAttr {
        ty: tensor(&ctx, &[3, 4], ElementType::I64),
        indices: vec![
            IndexTensorAttr { values: vec![0, 1] },
            IndexTensorAttr { values: vec![2, 3] },
        ],
        values: DenseArrayAttr::integers(&ctx, vec![5, -6]).values,
    };
    assert_eq!(
        mlir_dictionary(&ctx, &one(value)).unwrap(),
        "{value = sparse<[[0, 1], [2, 3]], [5, -6]> : tensor<3x4xi64>}"
    );
    assert_eq!(
        mlir_dictionary(
            &ctx,
            &one(SymbolRefAttr {
                path: vec![StringAttr::from("root"), StringAttr::from("with space")],
            })
        )
        .unwrap(),
        "{value = @root::@\"with space\"}"
    );
    assert!(mlir_dictionary(&ctx, &one(SymbolRefAttr { path: vec![] })).is_err());
}

#[test]
fn type_attributes_are_native_mlir_types_not_pliron_handles() {
    let ctx = Context::new();
    let ty = TensorType::get(&ctx, vec![-1, 2], ElementType::F16.native(&ctx)).into();
    assert_eq!(
        mlir_dictionary(&ctx, &one(crate::attributes::TypeAttr::new(ty))).unwrap(),
        "{value = tensor<?x2xf16>}"
    );
    let vector = VectorType::new_scalable(&ctx, ElementType::I32, vec![2, 4], vec![false, true]);
    assert_eq!(
        mlir_dictionary(&ctx, &one(PlironTypeAttr::new(vector))).unwrap(),
        "{value = vector<2x[4]xi32>}"
    );
    let function = FunctionType::get(&ctx, vec![ty], vec![TokenType::get(&ctx).into()]).into();
    assert_eq!(
        mlir_dictionary(&ctx, &one(PlironTypeAttr::new(function))).unwrap(),
        "{value = (tensor<?x2xf16>) -> !stablehlo.token}"
    );
}

#[test]
fn affine_and_strided_layouts_have_explicit_mlir_syntax() {
    let ctx = Context::new();
    let map = AffineMap {
        dimensions: 1,
        symbols: 1,
        results: vec![AffineExpr::binary(
            AffineBinaryOp::Add,
            AffineExpr::dimension(0),
            AffineExpr::symbol(0),
        )],
    };
    assert_eq!(
        mlir_dictionary(&ctx, &one(map)).unwrap(),
        "{value = affine_map<(d0)[s0] -> ((d0 + s0))>}"
    );
    assert_eq!(
        mlir_dictionary(
            &ctx,
            &one(StridedLayoutAttr {
                offset: vec![],
                strides: vec![4, i64::MIN]
            })
        )
        .unwrap(),
        "{value = strided<[4, ?], offset: ?>}"
    );
}

#[test]
fn opaque_enums_and_tensor_bounds_use_pinned_dialect_syntax() {
    let ctx = Context::new();
    assert_eq!(
        mlir_dictionary(
            &ctx,
            &one(OpaqueAttr {
                dialect: StringAttr::from("test"),
                syntax: StringAttr::from("thing<[1], (i32) -> i64>"),
            })
        )
        .unwrap(),
        "{value = #test.thing<[1], (i32) -> i64>}"
    );
    assert_eq!(
        mlir_dictionary(
            &ctx,
            &one(OpaqueAttr {
                dialect: StringAttr::from("test"),
                syntax: StringAttr::from("\"opaque data\""),
            })
        )
        .unwrap(),
        "{value = #test<\"opaque data\">}"
    );
    assert_eq!(
        mlir_dictionary(
            &ctx,
            &one(OpaqueAttr {
                dialect: StringAttr::from("test"),
                syntax: StringAttr::from("thing<a><b>"),
            })
        )
        .unwrap(),
        "{value = #test<thing<a><b>>}"
    );
    for syntax in [">", "foo<]> ", "\"bad\\q\"", "\"unterminated", "\0"] {
        assert!(
            mlir_dictionary(
                &ctx,
                &one(OpaqueAttr {
                    dialect: StringAttr::from("test"),
                    syntax: StringAttr::from(syntax),
                })
            )
            .is_err()
        );
    }
    assert_eq!(
        mlir_dictionary(&ctx, &one(Precision::Highest)).unwrap(),
        "{value = #stablehlo<precision HIGHEST>}"
    );
    let ty = TensorType::get(&ctx, vec![-1, 2], ElementType::F32.native(&ctx))
        .deref(&ctx)
        .with_encoding(
            &ctx,
            Box::new(TypeExtensionsAttr {
                bounds: vec![4, -1],
            }),
        );
    assert_eq!(
        mlir_dictionary(&ctx, &one(PlironTypeAttr::new(ty))).unwrap(),
        "{value = tensor<?x2xf32, #stablehlo.type_extensions<bounds = [4, ?]>>}"
    );
}

#[test]
fn malformed_values_fail_with_typed_recovery_and_do_not_mutate_source() {
    let ctx = Context::new();
    let mut source = one(ScalarBitsAttr {
        ty: ElementType::I8.native(&ctx),
        words: vec![256],
    });
    let before = source.clone();
    let error = mlir_dictionary(&ctx, &source).unwrap_err();
    assert_eq!(recovery(&error), RecoveryCategory::InvalidInput);
    assert_eq!(source, before);
    source.insert(
        &Identifier::new("value"),
        Box::new(ScalarBitsAttr {
            ty: ElementType::I8.native(&ctx),
            words: vec![255],
        }),
    );
    assert_eq!(mlir_dictionary(&ctx, &source).unwrap(), "{value = -1 : i8}");
    let malformed = DenseArrayAttr {
        element: ElementType::I8.native(&ctx),
        values: VecAttr::new(vec![Box::new(ScalarBitsAttr::f32_bits(&ctx, 0))]),
    };
    assert!(mlir_dictionary(&ctx, &one(malformed)).is_err());
    let invalid_integer = IntegerAttr::new(
        IntegerType::get(&ctx, 8, Signedness::Signless),
        APInt::from_u64(1, bw(16)),
    );
    assert!(mlir_dictionary(&ctx, &one(invalid_integer)).is_err());
}

#[test]
fn source_native_assembly_roundtrip_in_fresh_context_keeps_serialization() {
    let ctx = Context::new();
    let source = dict(vec![
        ("text", Box::new(StringAttr::from("abc"))),
        (
            "bits",
            Box::new(ScalarBitsAttr::f32_bits(&ctx, 0x7fc0_1234)),
        ),
    ]);
    let expected = mlir_dictionary(&ctx, &source).unwrap();
    let native = source.disp(&ctx).to_string();
    let mut fresh = Context::new();
    let stream = state_stream_from_iterator(
        native.chars(),
        parsable::State::new(&mut fresh, Source::InMemory),
    );
    let parsed = DictAttr::parser(()).skip(eof()).parse(stream).unwrap().0;
    assert_eq!(mlir_dictionary(&fresh, &parsed).unwrap(), expected);
    assert_eq!(mlir_dictionary(&ctx, &source).unwrap(), expected);
}

fn recovery(error: &pliron::result::Error) -> RecoveryCategory {
    if let Some(diagnostic) = Diagnostic::from_error(error) {
        return diagnostic.recovery_category();
    }
    match pliron_mlir_builtin::Diagnostic::from_error(error)
        .expect("serialization failures must carry a structured diagnostic")
        .recovery_category()
    {
        pliron_mlir_builtin::RecoveryCategory::InvalidInput => RecoveryCategory::InvalidInput,
        pliron_mlir_builtin::RecoveryCategory::Unsupported => RecoveryCategory::Unsupported,
        category => panic!("unexpected builtin recovery category: {category:?}"),
    }
}
