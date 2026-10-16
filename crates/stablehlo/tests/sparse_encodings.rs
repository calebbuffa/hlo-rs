use stablehlo::text::{
    parse_attribute, parse_module, parse_type, print_attribute, print_module, print_type,
};
use stablehlo::{
    AffineBinaryOp as Op, AffineExpr as Expr, AffineMap, Attribute, SparseLevelFormat as Format,
    SparseLevelType as Level, SparseTensorEncoding, infer_sparse_level_to_dimension,
};

fn encoding(source: &str) -> SparseTensorEncoding {
    let Attribute::SparseTensorEncoding(encoding) = parse_attribute(source).unwrap() else {
        panic!("expected encoding")
    };
    *encoding
}

#[test]
fn representative_formats_widths_values_slices_and_explicit_reverse_maps_roundtrip() {
    for source in [
        "#sparse_tensor.encoding<{map=(i)->(i:compressed)}>",
        "#sparse_tensor<encoding<{map=(i)->(i:compressed)}>>",
        "#sparse_tensor.encoding<{map=(i,j)->(i:dense,j:compressed)}>",
        "#sparse_tensor.encoding<{map=(i,j)->(j:compressed,i:loose_compressed),posWidth=32,crdWidth=8}>",
        "#sparse_tensor.encoding<{map=(i,j)->(i:compressed(nonunique),j:singleton)}>",
        "#sparse_tensor.encoding<{map=(i,j)->(i:compressed(nonunique,nonordered),j:singleton(soa))}>",
        "#sparse_tensor.encoding<{map=(i,j,k)->(i:batch,j:dense,k:compressed)}>",
        "#sparse_tensor.encoding<{map=(i,j)->(i:dense,j:structured[2,4])}>",
        "#sparse_tensor.encoding<{map=(i,j)->(i floordiv 2:dense,j floordiv 3:compressed,i mod 2:dense,j mod 3:dense)}>",
        "#sparse_tensor.encoding<{map={ib,jb,ii,jj}(i=ib*2+ii,j=jb*3+jj)->(ib=i floordiv 2:dense,jb=j floordiv 3:compressed,ii=i mod 2:dense,jj=j mod 3:dense)}>",
        "#sparse_tensor.encoding<{map=[c](i,j)->(c*3*i:dense,i:dense,j:compressed)}>",
        "#sparse_tensor.encoding<{map=(i:#sparse_tensor<slice(0,4,1)>,j:#sparse_tensor<slice(0,8,?)>)->(i:dense,j:compressed)}>",
        "#sparse_tensor.encoding<{map=(i:#sparse_tensor<slice(0,4,1)>,j)->(i:dense,j:compressed)}>",
        "#sparse_tensor.encoding<{map=(i)->(i:compressed),explicitVal=1:i32,implicitVal=0:i32}>",
        "#sparse_tensor.encoding<{posWidth=64,crdWidth=16,map=(i)->(i:compressed),}>",
        "#sparse_tensor.encoding<{map=(i)->(i:compressed(nonordered,nonordered))}>",
    ] {
        let value = encoding(source);
        let value = Attribute::SparseTensorEncoding(Box::new(value));
        let emitted = print_attribute(&value).unwrap();
        assert_eq!(
            parse_attribute(&emitted).unwrap(),
            value,
            "{source}\n{emitted}"
        );
    }
}

#[test]
fn permutations_and_blocked_inverse_maps_recover_original_coordinates() {
    for source in [
        "#sparse_tensor.encoding<{map=(i,j)->(j:compressed,i:compressed)}>",
        "#sparse_tensor.encoding<{map=(i,j)->(i floordiv 2:dense,j floordiv 3:compressed,i mod 2:dense,j mod 3:dense)}>",
        "#sparse_tensor.encoding<{map=(i,j)->(j floordiv 4:dense,i:compressed,j mod 4:dense)}>",
    ] {
        let value = encoding(source);
        let inverse = value.level_to_dimension.as_ref().unwrap();
        for i in 0..17 {
            for j in 0..17 {
                let level = value.dimension_to_level.evaluate(&[i, j], &[]).unwrap();
                assert_eq!(inverse.evaluate(&level, &[]).unwrap(), [i, j]);
            }
        }
    }
    let symbolic =
        encoding("#sparse_tensor.encoding<{map=[c](i,j)->(c*3*i:dense,i:dense,j:compressed)}>");
    assert!(symbolic.level_to_dimension.is_none());
}

#[test]
fn all_width_combinations_and_supported_primary_types_are_validated() {
    for pos in [0, 8, 16, 32, 64] {
        for crd in [0, 8, 16, 32, 64] {
            for element in [
                "f16",
                "bf16",
                "f32",
                "f64",
                "i8",
                "i16",
                "i32",
                "i64",
                "si8",
                "si16",
                "si32",
                "si64",
                "ui8",
                "ui16",
                "ui32",
                "ui64",
                "complex<f32>",
                "complex<f64>",
            ] {
                let source = format!(
                    "tensor<3x4x{element}, #sparse_tensor.encoding<{{map=(i,j)->(i:dense,j:compressed),posWidth={pos},crdWidth={crd}}}>>"
                );
                let ty = parse_type(&source).unwrap();
                assert_eq!(parse_type(&print_type(&ty).unwrap()).unwrap(), ty);
            }
        }
    }
    for element in [
        "i1",
        "i2",
        "i4",
        "index",
        "f8E5M2",
        "f6E3M2FN",
        "tf32",
        "!quant.uniform<i8:f32,0.5:0>",
    ] {
        let source =
            format!("tensor<4x{element},#sparse_tensor.encoding<{{map=(i)->(i:compressed)}}>>");
        assert!(parse_type(&source).is_err(), "{source}");
    }
}

#[test]
fn tensor_rank_and_explicit_implicit_element_contracts_are_checked() {
    for (element, explicit, implicit) in [
        ("f32", "1.0:f32", "-0.0:f32"),
        ("f64", "1.0:f64", "0.0:f64"),
        ("i32", "1:i32", "0:i32"),
        (
            "complex<f32>",
            "#complex.number<:f32 1.0, 2.0>",
            "#complex.number<:f32 -0.0, 0.0>",
        ),
        (
            "complex<f64>",
            "#complex.number<:f64 1.0, -2.0>",
            "#complex.number<:f64 0.0, -0.0>",
        ),
    ] {
        let source = format!(
            "tensor<4x{element},#sparse_tensor.encoding<{{map=(i)->(i:compressed),explicitVal={explicit},implicitVal={implicit}}}>>"
        );
        let ty = parse_type(&source).unwrap();
        assert_eq!(parse_type(&print_type(&ty).unwrap()).unwrap(), ty);
    }
    for source in [
        "tensor<f32,#sparse_tensor.encoding<{map=(i)->(i:compressed)}>>",
        "tensor<4x5xf32,#sparse_tensor.encoding<{map=(i)->(i:compressed)}>>",
        "tensor<4xf32,#sparse_tensor.encoding<{map=(i)->(i:compressed),explicitVal=1:i32}>>",
        "tensor<4xf32,#sparse_tensor.encoding<{map=(i)->(i:compressed),implicitVal=0:f64}>>",
        "tensor<4xf32,#sparse_tensor.encoding<{map=(i)->(i:compressed),implicitVal=1.0:f32}>>",
        "tensor<4xcomplex<f32>,#sparse_tensor.encoding<{map=(i)->(i:compressed),implicitVal=#complex.number<:f32 0.0, 1.0>}>>",
        "tensor<4xcomplex<f64>,#sparse_tensor.encoding<{map=(i)->(i:compressed),implicitVal=#complex.number<:f64 1.0, 0.0>}>>",
    ] {
        assert!(parse_type(source).is_err(), "{source}");
    }
}

#[test]
fn invalid_maps_formats_properties_sequences_and_fields_are_rejected() {
    for body in [
        "",
        "map=()->()",
        "map=(i)->(i:unknown)",
        "map=(i)->(i:dense(nonunique))",
        "map=(i)->(i:batch(nonordered))",
        "map=(i)->(i:compressed(soa))",
        "map=(i)->(i:compressed(high))",
        "map=(i)->(i:structured[4,2])",
        "map=(i)->(i:structured[256,256])",
        "map=(i)->(i:structured[-1,2])",
        "map=(i)->(i:structured[2,4](nonunique))",
        "map=(i,j)->(i:singleton,j:compressed)",
        "map=(i,j)->(i:dense,j:singleton)",
        "map=(i,j,k)->(i:compressed,j:singleton(soa),k:singleton)",
        "map=(i,j)->(i:compressed,j:batch)",
        "map=(i,j)->(i:compressed(nonunique),j:dense)",
        "map=(i,j)->(i:structured[2,4],j:dense)",
        "map=(i,j)->(i:compressed,j:structured[2,4])",
        "map=(i,j)->(i:dense)",
        "map=(i)->(i+1:compressed)",
        "map=(i,j)->(i:compressed,i:compressed)",
        "map=(i)->(i mod 2:dense,i floordiv 2:compressed)",
        "map=(i)->(i floordiv 2:dense,i mod 3:compressed)",
        "map=(i)->(i floordiv 2:dense)",
        "map=(i)->(i floordiv 2:dense,i mod 2:structured[1,4])",
        "map=(i:#sparse_tensor<slice(0,2,1)>)->(i floordiv 2:dense,i mod 2:compressed)",
        "map={l0,l1}(i=l0)->(l1=i:dense,l0=i:compressed)",
        "map={l}(i=l+1)->(l=i:compressed)",
        "map=[s]{l}(i=l)->(l=i*s:compressed)",
        "map=(i,i)->(i:compressed)",
        "map=[i](i)->(i:compressed)",
        "map=(i)->(j:compressed)",
        "map=(i)->(i:compressed),posWidth=7",
        "map=(i)->(i:compressed),crdWidth=128",
        "map=(i)->(i:compressed),posWidth=-1",
        "map=(i)->(i:compressed),posWidth=\"32\"",
        "map=(i)->(i:compressed),posWidth=8,posWidth=16",
        "map=(i)->(i:compressed),unknown=0",
        "map=(i)->(i:compressed),explicitVal=\"1\"",
        "map=(i)->(i:compressed),implicitVal=[0]",
    ] {
        let source = format!("#sparse_tensor.encoding<{{{body}}}>");
        assert!(parse_attribute(&source).is_err(), "{source}");
    }
}

#[test]
fn programmatic_inverse_and_metadata_validation_cannot_be_bypassed() {
    let map = AffineMap {
        dimensions: 1,
        symbols: 0,
        results: vec![Expr::Dimension(0)],
    };
    let mut value = SparseTensorEncoding::new(vec![Level::new(Format::Compressed)], map).unwrap();
    value.coordinate_width = 3;
    assert!(value.verify().is_err());
    value.coordinate_width = 8;
    let inverse = value.level_to_dimension.take();
    assert!(value.verify().is_err());
    assert!(print_attribute(&Attribute::SparseTensorEncoding(Box::new(value.clone()))).is_err());
    value.level_to_dimension = inverse;
    value.level_to_dimension.as_mut().unwrap().results[0] = Expr::Constant(7);
    assert!(value.verify().is_err());
    assert!(print_attribute(&Attribute::SparseTensorEncoding(Box::new(value))).is_err());
    let invalid = AffineMap {
        dimensions: 1,
        symbols: 0,
        results: vec![Expr::Dimension(1)],
    };
    assert!(infer_sparse_level_to_dimension(&invalid).is_err());
    let nonlinear = AffineMap {
        dimensions: 1,
        symbols: 0,
        results: vec![Expr::binary(Op::Add, Expr::Dimension(0), Expr::Constant(1))],
    };
    assert!(
        infer_sparse_level_to_dimension(&nonlinear)
            .unwrap()
            .is_none()
    );
}

#[test]
fn sparse_and_dense_tensor_operations_share_metadata_without_requiring_equal_encodings() {
    let source = r#"
        #SV = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>
        func.func @mix(%arg0: tensor<1xf32>, %arg1: tensor<1xf32, #SV>) {
            %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32, #SV>) -> tensor<1xf32, #SV>
            %1 = "stablehlo.add"(%arg1, %arg0) : (tensor<1xf32, #SV>, tensor<1xf32>) -> tensor<1xf32>
            func.return
        }
    "#;
    let module = parse_module(source).unwrap();
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
}

#[test]
fn generated_block_maps_with_permuted_levels_recover_coordinates_exactly() {
    for factor in 1..=8 {
        for ordering in [[0, 1, 2], [1, 0, 2], [0, 2, 1]] {
            // The quotient must precede its remainder in the reference format.
            let original = [
                Expr::binary(Op::FloorDiv, Expr::Dimension(0), Expr::Constant(factor)),
                Expr::Dimension(1),
                Expr::binary(Op::Mod, Expr::Dimension(0), Expr::Constant(factor)),
            ];
            let map = AffineMap {
                dimensions: 2,
                symbols: 0,
                results: ordering.into_iter().map(|i| original[i].clone()).collect(),
            };
            let inverse = infer_sparse_level_to_dimension(&map).unwrap().unwrap();
            for i in 0..=23 {
                for j in 0..=7 {
                    assert_eq!(
                        inverse
                            .evaluate(&map.evaluate(&[i, j], &[]).unwrap(), &[])
                            .unwrap(),
                        [i, j]
                    );
                }
            }
        }
    }
}
