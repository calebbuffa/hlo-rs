use super::{assembly, module_attributes, translation};
use crate::{
    attributes::*,
    ops::ConstantOp,
    pliron::{
        attribute::{AttrObj, attr_cast},
        builtin::attributes::StringAttr,
        context::Context,
        op::Op,
    },
    types::*,
};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    sync::OnceLock,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn inherits(record: &Value, class: &str) -> bool {
    record["!superclasses"]
        .as_array()
        .is_some_and(|classes| classes.iter().any(|name| name == class))
}

fn definitions() -> Option<&'static Value> {
    static DEFINITIONS: OnceLock<Option<Value>> = OnceLock::new();
    DEFINITIONS.get_or_init(|| {
        let root = root();
        let upstream = root.join("extern").join("stablehlo");
        let llvm = std::env::var_os("STABLEHLO_LLVM_SOURCE")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("target").join("mlir-validation"));
        let tool = std::env::var_os("STABLEHLO_TABLEGEN").map(PathBuf::from).unwrap_or_else(|| {
            llvm.join("build").join("bin").join(if cfg!(windows) {
                "llvm-min-tblgen.exe"
            } else {
                "llvm-min-tblgen"
            })
        });
        if std::env::var_os("STABLEHLO_TABLEGEN").is_none() && !tool.is_file() {
            eprintln!("upstream resolved-schema checks require STABLEHLO_TABLEGEN and STABLEHLO_LLVM_SOURCE; only the source catalog guard runs without TableGen");
            return None;
        }
        let output = Command::new(tool)
            .arg("--dump-json")
            .arg("-I").arg(&upstream)
            .arg("-I").arg(llvm.join("mlir").join("include"))
            .arg("-I").arg(llvm.join("llvm").join("include"))
            .arg(upstream.join("stablehlo").join("dialect").join("StablehloOps.td"))
            .output().expect("run upstream TableGen");
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        Some(serde_json::from_slice(&output.stdout).expect("resolved upstream schema"))
    }).as_ref()
}

fn native_verify(source: &str) {
    let Some(tool) = std::env::var_os("STABLEHLO_OPT") else {
        return;
    };
    let mut child = Command::new(tool)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch upstream verifier");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{source}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn check_attribute(ctx: &Context, value: &AttrObj) {
    assert!(attr_cast::<dyn pliron_mlir_interop::ToMlirAttr>(&**value).is_some());
    let text = assembly::print_attribute(ctx, &**value).unwrap();
    native_verify(&format!("module attributes {{test.value = {text}}} {{}}"));
}

#[test]
fn upstream_operation_catalog_is_complete() {
    let path = root()
        .join("extern")
        .join("stablehlo")
        .join("stablehlo")
        .join("dialect")
        .join("StablehloOps.td");
    if !path.is_file() {
        eprintln!("upstream source catalog check requires the extern/stablehlo checkout");
        return;
    }
    let source = std::fs::read_to_string(path).unwrap();
    let native = translation::catalog_names(&mut Context::new());
    let mut upstream = BTreeSet::new();
    for definition in source.split("def StableHLO_").skip(1) {
        let (name, body) = definition.split_once(':').unwrap();
        if name.trim().ends_with("Op") {
            let mnemonic = body.split('"').nth(1).unwrap();
            let name = format!("stablehlo.{}", mnemonic.replace('-', "_"));
            assert!(
                native.contains(&name),
                "missing upstream operation: {mnemonic}"
            );
            upstream.insert(name);
        }
    }
    assert!(!upstream.is_empty());
    assert_eq!(
        native
            .difference(&upstream)
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["stablehlo.call"]
    );
}

#[test]
fn every_upstream_property_has_an_export_mapping() {
    let Some(definitions) = definitions() else {
        return;
    };
    let mut checked = 0;
    for record in definitions
        .as_object()
        .unwrap()
        .values()
        .filter(|record| inherits(record, "StableHLO_Op"))
    {
        let name = format!(
            "stablehlo.{}",
            record["opName"].as_str().unwrap().replace('-', "_")
        );
        for argument in record["arguments"]["args"].as_array().unwrap() {
            let constraint = &definitions[argument[0]["def"].as_str().unwrap()];
            if !inherits(constraint, "Attr") {
                continue;
            }
            let key = argument[1].as_str().unwrap();
            assert!(
                module_attributes::is_source_field(&name, key),
                "missing upstream field: {name}.{key}"
            );
            if constraint["isOptional"] == 1 || !constraint["defaultValue"].is_null() {
                assert_ne!(
                    module_attributes::source_field_required(&name, key),
                    Some(true),
                    "upstream optional field made mandatory: {name}.{key}"
                );
            }
            checked += 1;
        }
    }
    assert!(
        checked > 100,
        "upstream schema must include inherited properties"
    );
}

fn attribute_examples(ctx: &Context) -> Vec<(&'static str, AttrObj)> {
    let mut values = vec![];
    for (name, source) in [
        (
            "dot",
            "#stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>",
        ),
        (
            "gather",
            "#stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], operand_batching_dims = [2], start_indices_batching_dims = [0], start_index_map = [0], index_vector_dim = 1>",
        ),
        (
            "scatter",
            "#stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], input_batching_dims = [2], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>",
        ),
        ("conv", "#stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>"),
        (
            "dot_algorithm",
            "#stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>",
        ),
        (
            "output_operand_alias",
            "#stablehlo.output_operand_alias<output_tuple_indices = [0, 1], operand_index = 2, operand_tuple_indices = [3]>",
        ),
        (
            "channel_handle",
            "#stablehlo.channel_handle<handle = -1, type = 1>",
        ),
        (
            "type_extensions",
            "#stablehlo.type_extensions<bounds = [8, ?]>",
        ),
        (
            "result_accuracy",
            "#stablehlo.result_accuracy<atol = 0.000001, rtol = 0.000002, ulps = 3, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        ),
    ] {
        values.push((name, assembly::parse_attribute(ctx, source).unwrap()));
    }
    let axis = MeshAxisAttr {
        name: StringAttr::new("x".into()),
        size: 4,
    };
    let sub = SubAxisInfoAttr {
        pre_size: 2,
        size: 2,
    };
    let reference = AxisRefAttr {
        name: StringAttr::new("x".into()),
        sub_axis_info: vec![sub.clone()],
    };
    let mesh = MeshAttr {
        axes: vec![axis.clone()],
        device_ids: vec![IndexTensorAttr {
            values: vec![0, 1, 2, 3],
        }],
    };
    values.extend([
        ("sub_axis_info", Box::new(sub) as AttrObj),
        ("axis_ref", Box::new(reference.clone())),
        ("mesh_axis", Box::new(axis)),
        ("mesh", Box::new(mesh.clone())),
        (
            "replica_group_mesh_axes",
            Box::new(ReplicaGroupMeshAxesAttr {
                mesh: MeshReferenceAttr {
                    inline: vec![mesh],
                    symbol: vec![],
                },
                axes: vec![reference],
            }),
        ),
    ]);
    values
}

#[test]
fn every_upstream_attribute_and_enum_value_exports() {
    let ctx = Context::new();
    let mut covered = BTreeSet::new();
    for (name, value) in attribute_examples(&ctx) {
        check_attribute(&ctx, &value);
        covered.insert(name.to_owned());
    }
    let Some(definitions) = definitions() else {
        return;
    };
    for record in definitions.as_object().unwrap().values().filter(|record| {
        inherits(record, "AttrDef") && record["dialect"]["def"] == "StableHLO_Dialect"
    }) {
        let name = record["mnemonic"].as_str().unwrap();
        if let Some(enumeration) = record["enum"]["def"].as_str() {
            for case in definitions[enumeration]["enumerants"].as_array().unwrap() {
                let case = &definitions[case["def"].as_str().unwrap()];
                let value: AttrObj = Box::new(EnumAttr {
                    dialect: StringAttr::new("stablehlo".into()),
                    name: StringAttr::new(name.into()),
                    value: StringAttr::new(case["str"].as_str().unwrap().into()),
                });
                check_attribute(&ctx, &value);
            }
            covered.insert(name.to_owned());
        }
        assert!(
            covered.contains(name),
            "no native attribute export fixture for {name}"
        );
    }
}

#[test]
fn empty_optional_mesh_device_ids_use_the_default_assignment() {
    let ctx = Context::new();
    let mesh: AttrObj = Box::new(MeshAttr {
        axes: vec![MeshAxisAttr {
            name: StringAttr::new("x".into()),
            size: 2,
        }],
        device_ids: vec![IndexTensorAttr { values: vec![] }],
    });
    check_attribute(&ctx, &mesh);
    assert!(
        !assembly::print_attribute(&ctx, &*mesh)
            .unwrap()
            .contains("device_ids")
    );
}

#[test]
fn upstream_primitive_and_dialect_types_export() {
    use ElementType::*;
    let elements = [
        I1,
        I2,
        I4,
        I8,
        I16,
        I32,
        I64,
        Ui2,
        Ui4,
        Ui8,
        Ui16,
        Ui32,
        Ui64,
        F4E2M1Fn,
        F6E2M3Fn,
        F6E3M2Fn,
        F8E3M4,
        F8E4M3,
        F8E4M3Fn,
        F8E4M3FnUz,
        F8E4M3B11FnUz,
        F8E5M2,
        F8E5M2FnUz,
        F8E8M0FnUz,
        F16,
        Bf16,
        F32,
        F64,
        C64,
        C128,
    ];
    let mut ctx = Context::new();
    let mut float_names = BTreeSet::new();
    for element in elements {
        let ty = tensor(&ctx, &[2], element);
        let words = vec![0; element.bit_width().unwrap().div_ceil(64) as usize];
        let op = ConstantOp::new(&mut ctx, ty, words, true);
        let text = String::try_from(assembly::MlirPrinter::new(&ctx, &op.get_operation())).unwrap();
        native_verify(&format!(
            "module {{func.func @test() {{ {text}\nreturn }} }}"
        ));
        if element.is_float() {
            float_names.insert(
                assembly::print_type(&ctx, element.native(&ctx))
                    .unwrap()
                    .to_ascii_lowercase(),
            );
        }
    }
    let token = TokenType::get(&ctx).into();
    let future = FutureType::get(&ctx, vec![tensor(&ctx, &[2], F32)]).into();
    for ty in [token, future] {
        let text = assembly::print_type(&ctx, ty).unwrap();
        native_verify(&format!("module {{func.func private @test({text})}}"));
    }
    let Some(definitions) = definitions() else {
        return;
    };
    let upstream_floats = definitions["HLO_Float"]["allowedTypes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|ty| ty["def"].as_str().unwrap().to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        float_names, upstream_floats,
        "all upstream float formats need native export fixtures"
    );
    let types = definitions
        .as_object()
        .unwrap()
        .values()
        .filter(|record| inherits(record, "StableHLO_Type"))
        .map(|record| record["mnemonic"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(types, BTreeSet::from(["future", "token"]));
}

#[test]
fn upstream_container_and_quantized_types_export() {
    let ctx = Context::new();
    for source in [
        "tensor<f32>",
        "tensor<0x2xf32>",
        "tensor<?x3xf32, #stablehlo.type_extensions<bounds = [8, ?]>>",
        "tensor<*xf32>",
        "memref<2x?xf32>",
        "memref<*xf32>",
        "tuple<tensor<2xf32>, tuple<!stablehlo.token, memref<?xi32>>>",
        "tensor<2x!quant.uniform<i8<-127:127>:f32, 0.5:1>>",
        "tensor<2x!quant.uniform<i8:f32:0, {0.5:0, 0.25:1}>>",
        "!stablehlo.future<tensor<2xf32>, tensor<?xi32>>",
    ] {
        let ty = assembly::parse_type(&ctx, source).unwrap();
        let text = assembly::print_type(&ctx, ty).unwrap();
        native_verify(&format!("module {{func.func private @test({text})}}"));
    }
}
