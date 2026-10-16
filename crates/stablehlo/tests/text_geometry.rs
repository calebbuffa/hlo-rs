use stablehlo::{
    Attribute, ConvDimensionNumbers, Mesh, MeshReference, ReplicaGroupMeshAxes, StableHloAttribute,
    text::{parse_attribute, print_attribute},
};

fn round_trip(input: &str) -> Attribute {
    let attribute = parse_attribute(input).unwrap_or_else(|error| panic!("{input}: {error}"));
    let output = print_attribute(&attribute).unwrap();
    assert_eq!(parse_attribute(&output).unwrap(), attribute, "{output}");
    attribute
}

#[test]
fn compressed_convolution_dimensions_preserve_spatial_order() {
    let attribute = round_trip("#stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>");
    assert_eq!(
        attribute,
        Attribute::StableHlo(Box::new(StableHloAttribute::ConvDimensionNumbers(
            ConvDimensionNumbers {
                input_batch_dimension: 0,
                input_feature_dimension: 3,
                input_spatial_dimensions: vec![1, 2],
                kernel_input_feature_dimension: 2,
                kernel_output_feature_dimension: 3,
                kernel_spatial_dimensions: vec![0, 1],
                output_batch_dimension: 0,
                output_feature_dimension: 3,
                output_spatial_dimensions: vec![1, 2],
            }
        )))
    );
    for input in [
        "#stablehlo.conv<[b, f]x[i, o]->[f, b]>",
        "#stablehlo.conv<[2, b, 0, f, 1] x [o, 1, i, 2, 0] -> [f, b, 1, 0, 2]>",
        "#stablehlo.conv<[b, 0, f]x[o, i, 0]->[b, 0, f]>",
    ] {
        round_trip(input);
    }
}

#[test]
fn raw_convolution_syntax_is_lossless_and_supports_omissions() {
    // StablehloOps.cpp: parseConvolutionDimensionsRaw shares parseStruct.
    round_trip(
        "#stablehlo.conv<raw input_batch_dimension = 42, output_spatial_dimensions = [5, 3]>",
    );
    round_trip("#stablehlo.conv<raw>");
    round_trip(
        "#stablehlo.conv<raw input_spatial_dimensions = [1, 0], kernel_input_feature_dimension = 7,>",
    );
    for input in [
        "#stablehlo.conv<raw bad = 1>",
        "#stablehlo.conv<raw input_batch_dimension = 0, input_batch_dimension = 1>",
        "#stablehlo.conv<raw input_spatial_dimensions = [-1]>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn malformed_compressed_convolution_is_rejected_without_large_allocations() {
    for dimensions in [
        "[b, b, f]",
        "[b, 0, 0, f]",
        "[b, 1, f]",
        "[b, -1, f]",
        "[b, 9223372036854775807, f]",
        "[b, 0]",
        "[f, 0]",
        "[b, i, f]",
        "[]",
        "[b, f,]",
    ] {
        let input = format!("#stablehlo.conv<{dimensions}x[i, o]->[b, f]>");
        assert!(parse_attribute(&input).is_err(), "{input}");
    }
}

#[test]
fn accuracy_syntax_defaults_and_tolerances_follow_reference() {
    // AssemblyFormat.h: atol, rtol and ulps are optional but ordered.
    for input in [
        "#stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<DEFAULT>>",
        "#stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>",
        "#stablehlo.result_accuracy<atol = 1e-5, rtol = 2e-4, ulps = 3, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo.result_accuracy<rtol = 0.001, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo.result_accuracy<ulps = 2, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
    ] {
        round_trip(input);
    }
    for input in [
        "#stablehlo.result_accuracy<>",
        "#stablehlo.result_accuracy<atol = -1.0, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo.result_accuracy<ulps = -1, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo.result_accuracy<ulps = 1, mode = #stablehlo.result_accuracy_mode<DEFAULT>>",
        "#stablehlo.result_accuracy<rtol = 1.0, atol = 1.0, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<UNKNOWN>>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn mesh_axes_and_subaxes_follow_ods_struct_syntax() {
    for input in [
        "#stablehlo<sub_axis_info(1)2>",
        "#stablehlo.axis_ref<name = \"x\">",
        "#stablehlo.axis_ref<name = \"y\", sub_axis_info = (1)2>",
        "#stablehlo.axis_ref<sub_axis_info = (3)2, name = \"foo\">",
        "#stablehlo.mesh_axis<name = \"x\", size = 2>",
        "#stablehlo.mesh_axis<size = 4, name = \"y\">",
    ] {
        round_trip(input);
    }
    for input in [
        "#stablehlo.sub_axis_info(1)2",
        "#stablehlo<sub_axis_info(0)2>",
        "#stablehlo.sub_axis_info(0)2",
        "#stablehlo.axis_ref<>",
        "#stablehlo.axis_ref<name = \"\">",
        "#stablehlo.axis_ref<name = \"x\", sub_axis_info = (1)0>",
        "#stablehlo.mesh_axis<name = \"x\">",
        "#stablehlo.mesh_axis<size = 2>",
        "#stablehlo.mesh_axis<name = \"x\", size = 0>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn inline_meshes_and_symbolic_replica_groups_round_trip() {
    let empty = round_trip("#stablehlo.mesh<axes = []>");
    assert_eq!(
        empty,
        Attribute::StableHlo(Box::new(StableHloAttribute::Mesh(Mesh {
            axes: vec![],
            device_ids: None,
        })))
    );
    // Fixture form from stablehlo/tests/interpret/all_gather.mlir.
    round_trip(
        "#stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = \"x\", size = 2>, #stablehlo.mesh_axis<name = \"y\", size = 4>]>, axes = [#stablehlo.axis_ref<name = \"y\", sub_axis_info = (1)2>]>",
    );
    let symbolic = round_trip("#stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = []>");
    assert_eq!(
        symbolic,
        Attribute::StableHlo(Box::new(StableHloAttribute::ReplicaGroupMeshAxes(
            ReplicaGroupMeshAxes {
                mesh: MeshReference::Symbol("mesh".into()),
                axes: vec![],
            }
        )))
    );
    round_trip(
        "#stablehlo.replica_group_mesh_axes<axes = [#stablehlo.axis_ref<name = \"x\">], mesh = @\"mesh name\">",
    );
}

#[test]
fn mesh_device_lists_preserve_permutations_and_optional_empty_ids() {
    for input in [
        "#stablehlo.mesh<axes = [], device_ids = dense<[]> : tensor<0xi64>>",
        "#stablehlo.mesh<axes = [], device_ids = dense<0> : tensor<1xi64>>",
        "#stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = \"x\", size = 4>], device_ids = dense<[2, 0, 3, 1]> : tensor<4xi64>>",
    ] {
        round_trip(input);
    }
    for input in [
        "#stablehlo.mesh<axes = [], device_ids = dense<[0]> : tensor<2xi64>>",
        "#stablehlo.mesh<axes = [], device_ids = dense<[0]> : tensor<1xi32>>",
        "#stablehlo.mesh<axes = [], device_ids = dense<0> : tensor<9223372036854775807xi64>>",
        "#stablehlo.mesh<axes = [], device_ids = dense<[0]> : tensor<?xi64>>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn mesh_fields_reject_wrong_types_missing_values_and_duplicate_axes() {
    for input in [
        "#stablehlo.replica_group_mesh_axes<mesh = @mesh>",
        "#stablehlo.replica_group_mesh_axes<axes = []>",
        "#stablehlo.replica_group_mesh_axes<mesh = @mesh::@nested, axes = []>",
        "#stablehlo.replica_group_mesh_axes<mesh = @mesh, axes = [1]>",
        "#stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = []>, axes = [#stablehlo.axis_ref<name = \"x\">]>",
        "#stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = \"x\", size = 2>, #stablehlo.mesh_axis<name = \"x\", size = 4>]>",
        "#stablehlo.mesh<axes = [#stablehlo.axis_ref<name = \"x\">]>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}
