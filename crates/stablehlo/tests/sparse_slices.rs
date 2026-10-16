use stablehlo::text::{parse_attribute, parse_module, print_attribute, print_module};
use stablehlo::{Attribute, SparseTensorDimSlice};

#[test]
fn default_slice_and_completely_dynamic_slice_match_reference_metadata() {
    let slice = SparseTensorDimSlice::default();
    assert_eq!(
        slice,
        SparseTensorDimSlice {
            offset: Some(0),
            size: None,
            stride: Some(1)
        }
    );
    assert!(!slice.is_completely_dynamic());
    assert_eq!(
        parse_attribute("#sparse_tensor<slice(0, ?, 1)>").unwrap(),
        Attribute::SparseTensorSlice(slice)
    );
    let dynamic = SparseTensorDimSlice {
        offset: None,
        size: None,
        stride: None,
    };
    assert!(dynamic.is_completely_dynamic());
    assert_eq!(
        parse_attribute("#sparse_tensor<slice(?, ?, ?)>").unwrap(),
        Attribute::SparseTensorSlice(dynamic)
    );
}

#[test]
fn generated_known_and_dynamic_components_roundtrip_without_sentinel_confusion() {
    for offset in [None, Some(0), Some(1), Some(i64::MAX as u64)] {
        for size in [None, Some(1), Some(2), Some(i64::MAX as u64)] {
            for stride in [None, Some(1), Some(3), Some(i64::MAX as u64)] {
                let slice = SparseTensorDimSlice {
                    offset,
                    size,
                    stride,
                };
                let attribute = Attribute::SparseTensorSlice(slice);
                attribute.verify().unwrap();
                assert_eq!(
                    parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
                    attribute
                );
            }
        }
    }
}

#[test]
fn invalid_static_values_fail_validation_and_printing() {
    for slice in [
        SparseTensorDimSlice {
            offset: Some(i64::MAX as u64 + 1),
            ..Default::default()
        },
        SparseTensorDimSlice {
            size: Some(0),
            ..Default::default()
        },
        SparseTensorDimSlice {
            size: Some(u64::MAX),
            ..Default::default()
        },
        SparseTensorDimSlice {
            stride: Some(0),
            ..Default::default()
        },
        SparseTensorDimSlice {
            stride: Some(u64::MAX),
            ..Default::default()
        },
    ] {
        assert!(slice.verify().is_err());
        let attribute = Attribute::SparseTensorSlice(slice);
        assert!(attribute.verify().is_err());
        assert!(print_attribute(&attribute).is_err());
    }
}

#[test]
fn negative_numeric_sentinels_zero_sizes_and_malformed_assembly_are_rejected() {
    for source in [
        "#sparse_tensor<slice(-1, ?, 1)>",
        "#sparse_tensor<slice(0, -1, 1)>",
        "#sparse_tensor<slice(0, ?, -1)>",
        "#sparse_tensor<slice(0, 0, 1)>",
        "#sparse_tensor<slice(0, 1, 0)>",
        "#sparse_tensor<slice(9223372036854775808, ?, 1)>",
        "#sparse_tensor<slice(0, 9223372036854775808, 1)>",
        "#sparse_tensor<slice(0, ?, 9223372036854775808)>",
        "#sparse_tensor<slice(0, 1)>",
        "#sparse_tensor<slice(0, 1, 1, 2)>",
        "#sparse_tensor<slice(0, 1, 1)> trailing",
        "#sparse_tensor<slice(0, 1, 1)",
        "#sparse_tensor<slice<(0, 1, 1)>>",
        "#sparse_tensor<slice(0, *, 1)>",
    ] {
        assert!(parse_attribute(source).is_err(), "{source}");
    }
}

#[test]
fn slice_metadata_survives_aliases_nested_attributes_and_locations() {
    let source = r#"
        #slice = #sparse_tensor<slice(0, 4, ?)>
        #metadata = {first = #slice, all = [#slice, #sparse_tensor<slice(?, ?, ?)>]}
        module attributes {test.slices = #metadata} {
            func.func @identity(%arg: tensor<f32>) -> tensor<f32> {
                return %arg : tensor<f32> loc(fused<#metadata>["slice origin"])
            }
        }
    "#;
    let module = parse_module(source).unwrap();
    let emitted = print_module(&module).unwrap();
    assert_eq!(parse_module(&emitted).unwrap(), module);
    assert!(emitted.contains("#sparse_tensor<slice(0, 4, ?)>"));
}

#[test]
fn sparse_slice_parser_handles_comments_whitespace_and_hexadecimal_int64_values() {
    let value =
        parse_attribute("#sparse_tensor<slice( // offset\n 0x2, 0x10, // stride\n ?)>").unwrap();
    assert_eq!(
        value,
        Attribute::SparseTensorSlice(SparseTensorDimSlice {
            offset: Some(2),
            size: Some(16),
            stride: None,
        })
    );
}
