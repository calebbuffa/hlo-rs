use clienthlo::{attributes::*, ops::*, pliron::op::Op};
use serde_json::Value;
use std::{collections::BTreeSet, path::PathBuf, process::Command, sync::OnceLock};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn definitions() -> Option<&'static Value> {
    static SCHEMA: OnceLock<Option<Value>> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        let llvm = std::env::var_os("STABLEHLO_LLVM_SOURCE").map(PathBuf::from)
            .unwrap_or_else(|| root().join("target").join("mlir-validation"));
        let tool = std::env::var_os("STABLEHLO_TABLEGEN").map(PathBuf::from)
            .unwrap_or_else(|| llvm.join("build").join("bin").join(if cfg!(windows) { "llvm-min-tblgen.exe" } else { "llvm-min-tblgen" }));
        if std::env::var_os("STABLEHLO_TABLEGEN").is_none() && !tool.is_file() {
            eprintln!("CHLO resolved-schema checks require STABLEHLO_TABLEGEN and STABLEHLO_LLVM_SOURCE");
            return None;
        }
        let upstream = root().join("extern").join("stablehlo");
        let output = Command::new(tool).arg("--dump-json")
            .arg("-I").arg(&upstream)
            .arg("-I").arg(llvm.join("mlir").join("include"))
            .arg("-I").arg(llvm.join("llvm").join("include"))
            .arg(upstream.join("stablehlo").join("dialect").join("ChloOps.td"))
            .output().expect("run pinned upstream TableGen");
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        Some(serde_json::from_slice(&output.stdout).unwrap())
    }).as_ref()
}

fn inherits(record: &Value, class: &str) -> bool {
    record["!superclasses"]
        .as_array()
        .is_some_and(|classes| classes.iter().any(|name| name == class))
}

fn native_operations() -> BTreeSet<String> {
    macro_rules! catalog {
        ($($op:ty),* $(,)?) => { BTreeSet::from([$(
            <$op>::get_opid_static().to_string()
        ),*]) };
    }
    catalog!(
        AcosOp,
        AcoshOp,
        AsinAcosKernelOp,
        AsinOp,
        AsinhOp,
        AtanOp,
        AtanhOp,
        BesselI1eOp,
        BroadcastAddOp,
        BroadcastAndOp,
        BroadcastAtan2Op,
        BroadcastCompareOp,
        BroadcastComplexOp,
        BroadcastDivideOp,
        BroadcastMaximumOp,
        BroadcastMinimumOp,
        BroadcastMultiplyOp,
        BroadcastNextAfterOp,
        BroadcastOrOp,
        BroadcastPolygammaOp,
        BroadcastPowerOp,
        BroadcastRemainderOp,
        BroadcastSelectOp,
        BroadcastShiftLeftOp,
        BroadcastShiftRightArithmeticOp,
        BroadcastShiftRightLogicalOp,
        BroadcastSubtractOp,
        BroadcastXorOp,
        BroadcastZetaOp,
        ConjOp,
        ConstantLikeOp,
        ConstantOp,
        CoshOp,
        DigammaOp,
        ErfInvOp,
        ErfOp,
        ErfcOp,
        IsInfOp,
        IsNegInfOp,
        IsPosInfOp,
        LgammaOp,
        MulhiOp,
        NextAfterOp,
        PolygammaOp,
        RaggedDotOp,
        ScanOp,
        SinhOp,
        SquareOp,
        TanOp,
        TopKOp,
        ZetaOp,
    )
}

#[test]
fn every_upstream_operation_has_a_compiled_rust_definition() {
    let native = native_operations();
    let path = root()
        .join("extern")
        .join("stablehlo")
        .join("stablehlo")
        .join("dialect")
        .join("ChloOps.td");
    if !path.is_file() {
        eprintln!("source catalog guard requires extern/stablehlo");
        return;
    }
    let source = std::fs::read_to_string(path).unwrap();
    let upstream = source
        .split("def CHLO_")
        .skip(1)
        .filter_map(|record| {
            let (name, body) = record.split_once(':')?;
            name.trim()
                .ends_with("Op")
                .then(|| format!("chlo.{}", body.split('"').nth(1).unwrap()))
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(native, upstream);
    if let Some(definitions) = definitions() {
        let resolved = definitions
            .as_object()
            .unwrap()
            .values()
            .filter(|record| inherits(record, "CHLO_Op"))
            .map(|record| format!("chlo.{}", record["opName"].as_str().unwrap()))
            .collect::<BTreeSet<_>>();
        assert_eq!(native, resolved);
    }
}

#[test]
fn every_resolved_upstream_property_has_a_native_storage_key() {
    let Some(definitions) = definitions() else {
        return;
    };
    let source = include_str!("../src/ops/definitions.rs");
    let mut fields = BTreeSet::new();
    for record in definitions
        .as_object()
        .unwrap()
        .values()
        .filter(|record| inherits(record, "CHLO_Op"))
    {
        let name = record["opName"].as_str().unwrap();
        for argument in record["arguments"]["args"].as_array().unwrap() {
            let mut constraint = &definitions[argument[0]["def"].as_str().unwrap()];
            if let Some(inner) = constraint["constraint"]["def"].as_str() {
                constraint = &definitions[inner];
            }
            if !inherits(constraint, "Attr") {
                continue;
            }
            let key = argument[1].as_str().unwrap();
            let native_key = match (name, key) {
                (_, "broadcast_dimensions") => format!("chlo_{name}_dimensions"),
                ("broadcast_compare", "comparison_direction") => "chlo_comparison_direction".into(),
                ("broadcast_compare", "compare_type") => "chlo_comparison_type".into(),
                ("constant" | "constant_like", "value") => "value".into(),
                ("ragged_dot", "ragged_dot_dimension_numbers") => {
                    "chlo_ragged_dot_dimension_numbers".into()
                }
                ("ragged_dot", "precision_config") => "chlo_precision_config".into(),
                ("scan", "scan_dim_size") => "chlo_scan_dim_size".into(),
                _ => format!("chlo_{name}_{key}"),
            };
            assert!(
                source.contains(&native_key),
                "missing storage for chlo.{name}.{key}: {native_key}"
            );
            fields.insert((name, key));
        }
    }
    assert_eq!(
        fields.len(),
        32,
        "review upstream property inventory changes"
    );
}

#[test]
fn upstream_attributes_parameters_and_enum_encodings_match() {
    let Some(definitions) = definitions() else {
        return;
    };
    let mut attrs = BTreeSet::new();
    let mut cases = 0;
    for record in
        definitions.as_object().unwrap().values().filter(|record| {
            inherits(record, "AttrDef") && record["dialect"]["def"] == "CHLO_Dialect"
        })
    {
        let name = record["mnemonic"].as_str().unwrap();
        attrs.insert(name);
        if let Some(enumeration) = record["enum"]["def"].as_str() {
            for case in definitions[enumeration]["enumerants"].as_array().unwrap() {
                let case = &definitions[case["def"].as_str().unwrap()];
                let spelling = case["str"].as_str().unwrap();
                let number = match name {
                    "comparison_direction" => {
                        u64::from(spelling.parse::<ComparisonDirection>().unwrap())
                    }
                    "comparison_type" => u64::from(spelling.parse::<ComparisonKind>().unwrap()),
                    "precision" => u64::from(spelling.parse::<Precision>().unwrap()),
                    _ => panic!("missing enum: {name}"),
                };
                assert_eq!(number, case["value"].as_u64().unwrap(), "{name}.{spelling}");
                cases += 1;
            }
        } else {
            assert_eq!(
                name, "ragged_dot",
                "new upstream attribute requires a Rust implementation"
            );
            let parameters = record["parameters"]["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|arg| arg[1].as_str().unwrap())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                parameters,
                BTreeSet::from([
                    "lhsBatchingDimensions",
                    "rhsBatchingDimensions",
                    "lhsContractingDimensions",
                    "rhsContractingDimensions",
                    "lhsRaggedDimensions",
                    "rhsGroupDimensions",
                ])
            );
            let source = include_str!("../src/attributes/mod.rs");
            for field in [
                "lhs_batching_dimensions",
                "rhs_batching_dimensions",
                "lhs_contracting_dimensions",
                "rhs_contracting_dimensions",
                "lhs_ragged_dimensions",
                "rhs_group_dimensions",
            ] {
                assert!(source.contains(&format!("pub {field}: Vec<i64>")));
            }
        }
    }
    assert_eq!(
        attrs,
        BTreeSet::from([
            "comparison_direction",
            "comparison_type",
            "precision",
            "ragged_dot"
        ])
    );
    assert_eq!(cases, 15);
    let types = definitions
        .as_object()
        .unwrap()
        .values()
        .filter(|record| inherits(record, "TypeDef") && record["dialect"]["def"] == "CHLO_Dialect")
        .map(|record| record["mnemonic"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    assert!(
        types.is_empty(),
        "new CHLO-specific types need Rust representations: {types:?}"
    );
}
