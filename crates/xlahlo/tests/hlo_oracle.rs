use std::{
    io::Write,
    process::{Command, Stdio},
};
use xlahlo::{
    Literal, Shape, inference, interchange,
    proto::hlo::{
        HloInstructionProto, HloModuleProto, HloScheduleProto,
        hlo_schedule_proto::InstructionSequence,
    },
    proto::xla_data::ShapeProto,
    text,
};

fn native(source: &str, mode: &str) -> std::process::Output {
    native_bytes(source.as_bytes(), mode)
}

fn native_bytes(source: &[u8], mode: &str) -> std::process::Output {
    let executable = std::env::var_os("XLAHLO_HLO_ORACLE").expect("XLAHLO_HLO_ORACLE");
    let mut child = Command::new(executable)
        .arg(mode)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let source = source.to_owned();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&source));
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap().unwrap();
    output
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn single_instruction_assembly_matches_pinned_cpp() {
    for source in [
        "f32[2] add(f32[2], f32[2])",
        "sum = f32[2] add(f32[2] lhs, f32[2] rhs)",
        "ROOT result = negate(f32[2])",
        "f32[2] add(f32[2] negate(f32[2]), f32[2])",
    ] {
        let module = text::parse_instruction(source).unwrap();
        module.verify(Default::default()).unwrap();
        let output = native(source, "parse");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        let rust = module.as_proto();
        assert_eq!(rust.name, cpp.name, "{source}");
        assert_eq!(
            rust.entry_computation_id, cpp.entry_computation_id,
            "{source}"
        );
        assert_eq!(rust.computations.len(), cpp.computations.len(), "{source}");
        let rust_entry = rust
            .computations
            .iter()
            .find(|computation| computation.id == rust.entry_computation_id)
            .unwrap();
        let cpp_entry = cpp
            .computations
            .iter()
            .find(|computation| computation.id == cpp.entry_computation_id)
            .unwrap();
        assert_eq!(rust_entry.name, cpp_entry.name, "{source}");
        assert_eq!(
            rust_entry.instructions.len(),
            cpp_entry.instructions.len(),
            "{source}: Rust {:#?}; native {:#?}",
            rust_entry.instructions,
            cpp_entry.instructions
        );
        assert_eq!(
            rust_entry.root_id,
            cpp_entry.root_id,
            "{source}: Rust instructions {:?}; native instructions {:?}",
            rust_entry
                .instructions
                .iter()
                .map(|instruction| (
                    &instruction.name,
                    &instruction.opcode,
                    &instruction.operand_ids
                ))
                .collect::<Vec<_>>(),
            cpp_entry
                .instructions
                .iter()
                .map(|instruction| (
                    &instruction.name,
                    &instruction.opcode,
                    &instruction.operand_ids
                ))
                .collect::<Vec<_>>()
        );
        for (rust_instruction, cpp_instruction) in
            rust_entry.instructions.iter().zip(&cpp_entry.instructions)
        {
            assert_eq!(rust_instruction.name, cpp_instruction.name, "{source}");
            assert_eq!(rust_instruction.opcode, cpp_instruction.opcode, "{source}");
            assert_eq!(
                rust_instruction.operand_ids, cpp_instruction.operand_ids,
                "{source}"
            );
            assert_eq!(rust_instruction.shape, cpp_instruction.shape, "{source}");
        }
    }
    for source in [
        "f32[2] add(lhs, rhs)",
        "same = f32[2] add(f32[2] same, f32[2])",
        "f32[] negate(f32[]) extra",
    ] {
        assert!(text::parse_instruction(source).is_err(), "{source}");
        let output = native(source, "parse");
        assert_eq!(
            output.status.code(),
            Some(2),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn iota_type_domains_match_pinned_cpp() {
    for (element_type, expected_success) in
        [("s32", true), ("f32", true), ("c64", true), ("pred", false)]
    {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT i = {element_type}[3] iota(), iota_dimension=0 }}"
        );
        let rust = text::parse_module(&source)
            .unwrap()
            .verify(Default::default())
            .is_ok();
        let native = native(&source, "verify").status.success();
        assert_eq!(rust, expected_success, "Rust verification: {source}");
        assert_eq!(native, expected_success, "native verification: {source}");
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn mixed_precision_verification_policies_match_pinned_cpp() {
    let cases = [
        (
            "HloModule m ENTRY main { a = f32[2] parameter(0) b = f64[2] parameter(1) ROOT r = bf16[2] add(a,b) }",
            false,
            true,
        ),
        (
            "HloModule m ENTRY main { a = f32[2] parameter(0) b = bf16[2] parameter(1) ROOT r = (f32[2],bf16[2]) tuple(a,b) }",
            true,
            true,
        ),
        (
            "HloModule m ENTRY main { a = (f32[2],bf16[2]) parameter(0) ROOT r = f32[2] get-tuple-element(a), index=0 }",
            true,
            true,
        ),
        (
            "HloModule m f { ROOT p = f64[] parameter(0) } ENTRY main { p = f32[2] parameter(0) ROOT r = bf16[2] map(p), to_apply=f }",
            false,
            true,
        ),
        (
            "HloModule m f { p = f64[] parameter(0) q = f32[] parameter(1) ROOT r = f64[] copy(p) } ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0) ROOT r = bf16[] reduce(p,z), dimensions={0}, to_apply=f }",
            false,
            true,
        ),
        (
            "HloModule m ENTRY main { a = f32[2] parameter(0) b = bf16[2] parameter(1) ROOT r = f32[2] custom-call(a,b), custom_call_target=\"mixed\" }",
            true,
            true,
        ),
    ];
    for (source, default_expected, mixed_expected) in cases {
        let module = text::parse_module(source).unwrap();
        for (mode, allow_mixed, expected) in [
            ("verify", false, default_expected),
            ("verify-mixed", true, mixed_expected),
        ] {
            let rust = module
                .verify(xlahlo::VerifyOptions {
                    allow_mixed_precision: allow_mixed,
                    ..Default::default()
                })
                .is_ok();
            let output = native(source, mode);
            assert_eq!(rust, expected, "Rust mode {mode}, source {source}");
            assert_eq!(
                output.status.success(),
                expected,
                "native mode {mode}, source {source}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn replica_group_verification_policy_matches_pinned_cpp() {
    let reducer = "sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT add = f32[] add(a,b) }";
    let cases = [
        (
            "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT g = f32[4] all-gather(p), dimensions={0}, replica_groups={{0,0}} }".to_owned(),
            false,
        ),
        (
            "HloModule m, replica_count=3 ENTRY main { p = f32[2] parameter(0) ROOT g = f32[4] all-gather(p), dimensions={0}, replica_groups={{0,1}} }".to_owned(),
            false,
        ),
        (
            format!(
                "HloModule m {reducer} ENTRY main {{ p = f32[2] parameter(0) ROOT r = f32[2] all-reduce(p), to_apply=sum, replica_groups={{{{0,0}}}} }}"
            ),
            false,
        ),
    ];
    for (source, expected_default) in cases {
        let module = text::parse_module(&source).unwrap();
        let rust_default = module.verify(Default::default()).is_ok();
        let rust_unchecked = module
            .verify(xlahlo::VerifyOptions {
                check_replica_groups: false,
                ..Default::default()
            })
            .is_ok();
        let native_default = native(&source, "verify");
        let native_unchecked = native(&source, "verify-no-groups");
        assert_eq!(rust_default, expected_default, "{source}");
        assert!(rust_unchecked, "{source}");
        assert_eq!(
            native_default.status.success(),
            expected_default,
            "{source}: {}",
            String::from_utf8_lossy(&native_default.stderr)
        );
        assert!(
            native_unchecked.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&native_unchecked.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn dead_and_fusion_schedule_sequences_match_pinned_cpp_policy() {
    let source = r#"HloModule m
fused { p = f32[] parameter(0) ROOT n = f32[] negate(p) }
dead { p = f32[] parameter(0) ROOT n = f32[] negate(p) }
ENTRY main { p = f32[] parameter(0) ROOT f = f32[] fusion(p), kind=kLoop, calls=fused }
"#;
    let module = text::parse_module(source).unwrap();
    let mut wire = module.to_proto_with_program_shapes().unwrap();
    let mut schedule = HloScheduleProto::new();
    for computation in &wire.computations {
        schedule.sequences.insert(
            computation.id,
            InstructionSequence {
                instruction_ids: computation.instructions.iter().map(|i| i.id).collect(),
                ..Default::default()
            },
        );
    }
    let dead_id = wire
        .computations
        .iter()
        .find(|computation| computation.name == "dead")
        .unwrap()
        .id;
    let fused_id = wire
        .computations
        .iter()
        .find(|computation| computation.name == "fused")
        .unwrap()
        .id;
    schedule
        .sequences
        .get_mut(&dead_id)
        .unwrap()
        .instruction_ids
        .clear();
    schedule
        .sequences
        .get_mut(&fused_id)
        .unwrap()
        .instruction_ids
        .reverse();
    wire.schedule = Some(schedule).into();

    let rust = xlahlo::Module::from_proto(wire.clone())
        .unwrap()
        .verify(Default::default())
        .is_ok();
    let native = native_bytes(&interchange::encode(&wire).unwrap(), "verify-proto");
    assert!(rust, "Rust schedule verification failed");
    assert!(
        native.status.success(),
        "native schedule verification failed: {}",
        String::from_utf8_lossy(&native.stderr)
    );
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn shortform_constant_storage_matches_pinned_cpp() {
    let source = "HloModule m ENTRY main { \
        a = pred[1] constant({...}) \
        b = u8[5] constant({...}) \
        c = f16[3] constant({...}) \
        d = f32[2] constant({...}) \
        ROOT e = c64[1] constant({...}) \
    }";
    let rust = text::parse_module(source).unwrap();
    let output = native(source, "parse");
    assert!(
        output.status.success(),
        "{source}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
    let rust_entry = rust
        .as_proto()
        .computations
        .iter()
        .find(|computation| computation.id == rust.as_proto().entry_computation_id)
        .unwrap();
    let cpp_entry = cpp
        .computations
        .iter()
        .find(|computation| computation.id == cpp.entry_computation_id)
        .unwrap();
    for name in ["a", "b", "c", "d", "e"] {
        let rust_literal = Literal::from_proto(
            rust_entry
                .instructions
                .iter()
                .find(|instruction| instruction.name == name)
                .unwrap()
                .literal
                .as_ref()
                .unwrap()
                .clone(),
        )
        .unwrap();
        let cpp_literal = Literal::from_proto(
            cpp_entry
                .instructions
                .iter()
                .find(|instruction| instruction.name == name)
                .unwrap()
                .literal
                .as_ref()
                .unwrap()
                .clone(),
        )
        .unwrap();
        compare_literals(&rust_literal, &cpp_literal);
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn bounded_dynamic_constant_text_normalization_matches_pinned_cpp() {
    let source = "HloModule m ENTRY main { ROOT c = s32[<=3] constant({1,2,3}) }";
    let output = native(source, "parse");
    assert!(
        output.status.success(),
        "{source}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
    let constant = root(&cpp);
    assert_eq!(constant.opcode, "constant");
    let shape = constant.shape.as_ref().unwrap();
    assert_eq!(shape.dimensions, [3]);
    assert_eq!(shape.is_dynamic_dimension, [false]);
    let literal = constant.literal.as_ref().unwrap();
    assert_eq!(literal.shape.as_ref().unwrap(), shape);
    assert_eq!(literal.s32s, [1, 2, 3]);
    assert!(literal.dynamic_sizes.is_empty());
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE and the extern/xla submodule"]
fn upstream_positive_parser_pool_round_trips() {
    let checkout = std::path::PathBuf::from(env!("XLA_SOURCE_DIR"));
    let source = std::fs::read_to_string(
        checkout
            .join("xla")
            .join("hlo")
            .join("parser")
            .join("hlo_parser_test.cc"),
    )
    .expect("extern/xla/xla/hlo/parser/hlo_parser_test.cc");
    let pool = source
        .split_once("std::vector<TestData> CreateTestCases() {")
        .expect("upstream fixture pool")
        .1
        .split_once("// clang-format on")
        .expect("upstream fixture pool end")
        .0;
    let mut remaining = pool;
    let mut cases = 0;
    let mut valid = 0;
    while let Some((_, after)) = remaining.split_once("R\"") {
        let (delimiter, after) = after.split_once('(').expect("C++ raw string delimiter");
        let end = format!("){delimiter}\"");
        let (text, after) = after.split_once(&end).expect("C++ raw string end");
        remaining = after;
        if !text.trim_start().starts_with("HloModule") {
            continue;
        }
        if text.contains("constant(") && text.contains("f6e") {
            continue;
        }
        if text
            .split(|ch: char| !ch.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .any(|part| part.parse::<u64>().is_ok_and(|number| number > 1_000_000))
        {
            continue;
        }
        let module = match text::parse_module_with_options(
            text,
            text::ParseOptions {
                max_literal_elements: 10_000,
                ..Default::default()
            },
        ) {
            Ok(module) => module,
            Err(error) => panic!("upstream positive failed to parse: {text}\n{error}"),
        };
        cases += 1;
        let rust_valid = module.verify(Default::default()).is_ok();
        let output = native(text, "verify");
        assert_eq!(
            output.status.code(),
            Some(if rust_valid { 0 } else { 3 }),
            "{text}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if !rust_valid {
            continue;
        }
        valid += 1;
        let printed = text::print_module(&module).unwrap();
        let output = native(&printed, "verify");
        assert!(
            output.status.success(),
            "{printed}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = interchange::encode(&module.to_proto_with_program_shapes().unwrap()).unwrap();
        let output = native_bytes(&bytes, "verify-proto");
        assert!(
            output.status.success(),
            "{text}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let wire = interchange::decode(&output.stdout, Default::default()).unwrap();
        xlahlo::Module::from_proto_with_config(wire, module.config())
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
    assert!(
        cases > 0,
        "no bounded upstream parser fixtures were exercised"
    );
    assert!(
        valid > 0,
        "no valid upstream parser fixtures were exercised"
    );
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn recovery_tables_and_module_metadata_match_pinned_cpp() {
    for source in [
        r#"HloModule m, origin_recovery_table={{"z" {2}}:{"new" {1}} {"a"}:{"first"} {"a"}:{"ignored"}}
 ENTRY main { ROOT c = s32[] constant(1) }"#,
        r#"HloModule m, origin_recovery_table={{"old"}:{"new"}, "HloModule n, frontend_attributes={foo=\"bar\"}, backend_config=\"\\000\\377\" ENTRY main { ROOT p = s32[2] parameter(0) }"}
 ENTRY main { ROOT c = s32[] constant(1) }"#,
        r#"HloModule m, origin_recovery_table={{"a"}:{"b"}, "HloModule n, origin_recovery_table={{\"c\"}:{\"d\"}} ENTRY main { ROOT c = s32[] constant(2) }"}
 ENTRY main { ROOT c = s32[] constant(1) }"#,
        r#"HloModule m, origin_recovery_table={}, frontend_attributes={x="/*payload*/"}, backend_config={"a":"//payload"}
 ENTRY main { ROOT c = s32[] constant(1) }"#,
        r#"HloModule m, origin_recovery_table={{}:{}}
 ENTRY main { ROOT c = s32[] constant(1) }"#,
        r#"HloModule m, origin_recovery_table={{"a"}:{"b"},{"c"}:{"d"}}
 ENTRY main { ROOT c = s32[] constant(1) }"#,
    ] {
        let rust = text::parse_module(source);
        let output = native(source, "parse");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{source}: {rust:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let Ok(module) = rust else { continue };
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        assert_eq!(
            module.as_proto().frontend_attributes,
            cpp.frontend_attributes
        );
        assert_eq!(module.as_proto().backend_config, cpp.backend_config);
        let left = &module
            .as_proto()
            .original_value_recovery_table
            .get_or_default()
            .entries;
        let right = &cpp.original_value_recovery_table.get_or_default().entries;
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert_eq!(left.old_original_array, right.old_original_array);
            assert_eq!(left.new_original_array, right.new_original_array);
            assert_eq!(
                left.recovery_module.is_some(),
                right.recovery_module.is_some()
            );
            if let (Some(left), Some(right)) = (
                left.recovery_module.as_ref(),
                right.recovery_module.as_ref(),
            ) {
                assert_eq!(left.frontend_attributes, right.frontend_attributes);
                assert_eq!(left.backend_config, right.backend_config);
                assert_eq!(left.host_program_shape, right.host_program_shape);
            }
        }
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        let output = native(&printed, "verify");
        assert!(
            output.status.success(),
            "{printed}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = interchange::encode(&module.to_proto_with_program_shapes().unwrap()).unwrap();
        let output = native_bytes(&bytes, "verify-proto");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn owned_module_configuration_matches_pinned_cpp() {
    for source in [
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0) }",
        "HloModule m, replica_count=2, num_partitions=3 ENTRY main { ROOT p = f32[] parameter(0) }",
        "HloModule m, replica_count=1, num_partitions=1 ENTRY main { ROOT p = f32[] parameter(0) }",
        "HloModule m, replica_count=2 ENTRY main { p = f32[1] parameter(0) ROOT g = f32[2] all-gather(p), dimensions={0}, replica_groups={{0,1}} }",
        "HloModule m, replica_count=2 ENTRY main { p = f32[1] parameter(0) ROOT g = f32[3] all-gather(p), dimensions={0}, replica_groups={{0,1,2}} }",
        "HloModule m, num_partitions=2 ENTRY main { ROOT p = f32[] parameter(0), sharding={maximal device=3} }",
    ] {
        let module = text::parse_module(source).unwrap();
        let config = module.config();
        let output = native(source, "parse-module-config");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!(
                "{} {} {}\n",
                config.replica_count, config.num_partitions, config.use_spmd_partitioning
            ),
            "{source}"
        );
        let verified = module.verify(Default::default()).is_ok();
        let output = native(source, "verify");
        assert_eq!(
            output.status.code(),
            Some(if verified { 0 } else { 3 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let printed = text::print_module(&module).unwrap();
        let output = native(&printed, "verify");
        assert_eq!(
            output.status.code(),
            Some(if verified { 0 } else { 3 }),
            "{printed}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output = native(source, "parse");
        assert!(output.status.success());
        let wire = interchange::decode(&output.stdout, Default::default()).unwrap();
        let imported = xlahlo::Module::from_proto_with_config(wire, config).unwrap();
        assert_eq!(
            imported.verify(Default::default()).is_ok(),
            verified,
            "{source}"
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn owned_module_header_flags_match_pinned_cpp() {
    for attributes in [
        "",
        ", alias_passthrough_params=true",
        ", alias_passthrough_params=false",
        ", allow_spmd_sharding_propagation_to_parameters=true",
        ", allow_spmd_sharding_propagation_to_output={}",
        ", allow_spmd_sharding_propagation_to_parameters={false}",
        ", allow_spmd_sharding_propagation_to_parameters={true,false,true}, allow_spmd_sharding_propagation_to_output={false,true}",
        ", replica_count=2, num_partitions=3, alias_passthrough_params=true, allow_spmd_sharding_propagation_to_output=false",
    ] {
        let source =
            format!("HloModule m{attributes} ENTRY main {{ ROOT p = f32[] parameter(0) }}");
        let module = text::parse_module(&source).unwrap();
        let config = module.config();
        let expected = format!(
            "{}\n{}\n{}\n",
            config.alias_passthrough_params,
            config
                .allow_spmd_sharding_propagation_to_parameters
                .iter()
                .map(|b| format!("{b} "))
                .collect::<String>(),
            config
                .allow_spmd_sharding_propagation_to_output
                .iter()
                .map(|b| format!("{b} "))
                .collect::<String>()
        );
        for text in [&source, &text::print_module(&module).unwrap()] {
            let output = native(text, "parse-module-options");
            assert!(
                output.status.success(),
                "{text}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                String::from_utf8(output.stdout).unwrap(),
                expected,
                "{text}"
            );
            assert!(native(text, "verify").status.success(), "{text}");
        }
        assert_eq!(
            text::parse_module(&text::print_module(&module).unwrap())
                .unwrap()
                .config(),
            config
        );
    }
    for attributes in [
        "alias_passthrough_params={true}",
        "allow_spmd_sharding_propagation_to_parameters={true,}",
        "allow_spmd_sharding_propagation_to_output=1",
        "allow_spmd_sharding_propagation_to_output={true,false",
    ] {
        let source =
            format!("HloModule m, {attributes} ENTRY main {{ ROOT p = f32[] parameter(0) }}");
        assert!(text::parse_module(&source).is_err(), "{source}");
        assert_eq!(
            native(&source, "parse-module-options").status.code(),
            Some(2),
            "{source}"
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn omitted_result_shapes_match_pinned_cpp() {
    let mut bodies = vec![
        "a = after-all() ROOT r = add-dependency(a,a)".to_owned(),
        "ROOT r = replica-id()".to_owned(),
        "ROOT r = partition-id()".to_owned(),
        "p = c64[2] parameter(0) ROOT r = abs(p)".to_owned(),
        "p = f32[2,3]{0,1} parameter(0) n = negate(f32[2,3] p) t = tuple(n,p) g = get-tuple-element(t), index=0 ROOT r = transpose(g), dimensions={1,0}".to_owned(),
        "p = f32[<=2,3] parameter(0) ROOT r = negate(p)".to_owned(),
        "p = f32[2,2] parameter(0) ROOT r = cholesky(p), lower=true".to_owned(),
        "p = c64[2] parameter(0) ROOT r = fft(p), fft_type=FFT, fft_length={2}".to_owned(),
        "p = f32[2,3] parameter(0) ROOT r = topk(p), k=2, largest=false".to_owned(),
        "a = f32[2,3] parameter(0) b = f32[3,4] parameter(1) ROOT r = dot(a,b), lhs_contracting_dims={1}, rhs_contracting_dims={0}".to_owned(),
        "a = f32[1,5,2] parameter(0) b = f32[3,2,4] parameter(1) ROOT r = convolution(a,b), window={size=3}, dim_labels=b0f_0io->b0f".to_owned(),
        "a = f32[4] parameter(0) i = s32[2,1] parameter(1) ROOT r = gather(a,i), offset_dims={}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1}".to_owned(),
        "a = c64[3,3] parameter(0) b = c64[3,5] parameter(1) ROOT r = triangular-solve(a,b), left_side=true, lower=false, unit_diagonal=true, transpose_a=ADJOINT".to_owned(),
        "a = f32[2,3] parameter(0) ROOT r = shuffle(a), dimensions={0,1}, mode=rotate, shifts={-1,99}".to_owned(),
        "p = pred[] parameter(0) a = f32[2] parameter(1) ROOT r = select(p,a,a)".to_owned(),
        "a = f32[2] parameter(0) z = f32[] constant(0) ROOT r = pad(a,z), padding=1_1".to_owned(),
        "a = f32[3] parameter(0) s = s32[] constant(2) ROOT r = set-dimension-size(a,s), dimensions={0}".to_owned(),
        "a = f32[] parameter(0) ROOT r = broadcast(a)".to_owned(),
        "a = f32[2] parameter(0) ROOT r = broadcast(a), dimensions={1}".to_owned(),
        "a = f32[2,3] parameter(0) ROOT r = broadcast(a), dimensions={2,3}".to_owned(),
        "a = f32[2] parameter(0) ROOT r = broadcast(a), dimensions={}".to_owned(),
        "p = f32[2] parameter(0) t = after-all() s = (f32[2],s32[],token[]) send(p,t), channel_id=1 g = get-tuple-element(s), index=1 ROOT r = copy(u32[] g)".to_owned(),
        "p = f32[2] parameter(0) n = negate(p) t = after-all() ROOT r = token[] outfeed(n,t)".to_owned(),
        "l = f32[2,3,4] parameter(0) r = f32[5,2,4,6] parameter(1) g = s32[2,5] parameter(2) ROOT d = ragged-dot(l,r,g), lhs_batch_dims={0}, rhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_contracting_dims={2}, lhs_ragged_dims={1}, rhs_group_dims={0}".to_owned(),
        "l = f32[8,8] parameter(0) r = f32[8,8] parameter(1) a = f32[8,8] parameter(2) b = f32[8,8] parameter(3) ROOT d = scaled-dot(l,r,a,b), lhs_contracting_dims={1}, rhs_contracting_dims={0}".to_owned(),
        "l = f32[8,8] parameter(0) r = f32[8,6] parameter(1) a = f32[8,4] parameter(2) b = f32[8,6] parameter(3) ROOT d = scaled-dot(l,r,a,b), lhs_contracting_dims={1}, rhs_contracting_dims={0}".to_owned(),
    ];
    for (opcode, extras) in [
        ("batch-norm-training", ""),
        ("batch-norm-inference", ",v,v"),
        ("batch-norm-grad", ",v,a"),
    ] {
        bodies.push(format!("a = f32[2,3] parameter(0) v = f32[3] parameter(1) ROOT r = {opcode}(a,v,v{extras}), epsilon=0.01, feature_index=1"));
    }
    for expression in [
        "abs(p)",
        "add(p,p)",
        "is-finite(p)",
        "compare(p,p), direction=LT",
        "complex(p,p)",
        "clamp(p,p,p)",
        "concatenate(p,p), dimensions={0}",
        "reverse(p), dimensions={0}",
        "get-dimension-size(p), dimensions={0}",
    ] {
        bodies.push(format!("p = f32[2,3] parameter(0) ROOT r = {expression}"));
    }
    let mut sources = bodies
        .into_iter()
        .map(|body| format!("HloModule m ENTRY main {{ {body} }}"))
        .collect::<Vec<_>>();
    sources.extend([
        "HloModule m neg { p = f32[] parameter(0) ROOT n = negate(p) } ENTRY main { p = f32[] parameter(0) ROOT c = call(p), to_apply=neg }",
        "HloModule m neg { p = f32[] parameter(0) ROOT n = negate(p) } ENTRY main { p = f32[2] parameter(0) ROOT c = map(p), dimensions={0}, to_apply=neg }",
        "HloModule m sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT c = add(a,b) } ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0) ROOT c = reduce(p,z), dimensions={0}, to_apply=sum }",
        "HloModule m sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT c = add(a,b) } ENTRY main { p = f32[] parameter(0) z = f32[] constant(0) ROOT c = reduce-window(p,z), to_apply=sum }",
        "HloModule m body { p = f32[] parameter(0) ROOT n = negate(p) } cond { p = f32[] parameter(0) ROOT c = pred[] constant(false) } ENTRY main { p = f32[] parameter(0) ROOT w = while(p), body=body, condition=cond }",
        "HloModule m neg { p = f32[] parameter(0) ROOT n = negate(p) } id { ROOT p = f32[] parameter(0) } ENTRY main { p = pred[] parameter(0) x = f32[] parameter(1) ROOT c = conditional(p,x,x), true_computation=neg, false_computation=id }",
        "HloModule m cmp { a = f32[] parameter(0) b = f32[] parameter(1) ROOT c = compare(a,b), direction=LT } ENTRY main { p = f32[3] parameter(0) ROOT r = sort(p), dimensions={0}, to_apply=cmp }",
        "HloModule m body { p = f32[] parameter(0) ROOT n = copy(p) } ENTRY main { p = f32[3] parameter(0) ROOT r = scan(p), dimensions={0}, num_carries=0, to_apply=body }",
        "HloModule m body { x = f32[3] parameter(0) c = f32[3] parameter(1) y = add(x,c) ROOT r = tuple(y,y) } ENTRY main { x = f32[4,3] parameter(0) c = f32[3] parameter(1) ROOT r = scan(x,c), dimensions={0}, num_carries=1, to_apply=body }",
        "HloModule m body { p = f32[2,3]{0,1} parameter(0) ROOT n = copy(p) } ENTRY main { p = f32[2,4,3] parameter(0) ROOT r = scan(p), dimensions={1}, num_carries=0, to_apply=body }",
        "HloModule m body { p = f32[<=3] parameter(0) ROOT n = copy(p) } ENTRY main { p = f32[<=4,3] parameter(0) ROOT r = scan(p), dimensions={0}, num_carries=0, to_apply=body }",
    ].into_iter().map(str::to_owned));
    for source in sources {
        let module = text::parse_module(&source).unwrap();
        let output = native(&source, "parse");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let wire: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        let computation = wire
            .computations
            .iter()
            .find(|c| c.id == wire.entry_computation_id)
            .unwrap();
        let root = computation
            .instructions
            .iter()
            .find(|i| i.id == computation.root_id)
            .unwrap();
        let native_shape = xlahlo::Shape::from_proto(root.shape.as_ref().unwrap().clone()).unwrap();
        let rust_shape = module
            .shape(module.root(module.entry().unwrap()).unwrap())
            .unwrap();
        assert!(
            rust_shape.equal_including_layout(&native_shape),
            "{source}: Rust {rust_shape:?}, native {native_shape:?}"
        );
        let verified = module.verify(Default::default()).is_ok();
        let output = native(&source, "verify");
        assert_eq!(
            output.status.code(),
            Some(if verified { 0 } else { 3 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let printed = text::print_module(&module).unwrap();
        let output = native(&printed, "verify");
        assert_eq!(
            output.status.code(),
            Some(if verified { 0 } else { 3 }),
            "{printed}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = interchange::encode(&module.to_proto_with_program_shapes().unwrap()).unwrap();
        let output = native_bytes(&bytes, "verify-proto");
        assert_eq!(
            output.status.code(),
            Some(if verified { 0 } else { 3 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    for body in [
        "ROOT p = parameter(0)",
        "ROOT c = constant(1)",
        "p = f32[2] parameter(0) ROOT n = negate()",
        "p = pred[2] parameter(0) ROOT n = negate(p)",
        "p = f32[2] parameter(0) ROOT n = transpose(p), dimensions={1}",
        "p = f32[2] parameter(0) ROOT n = reshape(p)",
        "p = f32[2] parameter(0) ROOT n = opt-barrier(p)",
        "p = f32[2] parameter(0) ROOT n = broadcast(p)",
        "p = f32[2] parameter(0) ROOT n = f32[2] broadcast(p)",
        "l = f32[4,8] parameter(0) r = f32[8,6] parameter(1) a = f16[2,2] parameter(2) b = s32[4,3] parameter(3) ROOT d = scaled-dot(l,r,a,b), lhs_contracting_dims={1}, rhs_contracting_dims={0}",
        "p = f32[2] parameter(0) t = token[] after-all() ROOT n = outfeed(p,t)",
    ] {
        let source = format!("HloModule m ENTRY main {{ {body} }}");
        assert!(text::parse_module(&source).is_err(), "{source}");
        assert_eq!(native(&source, "parse").status.code(), Some(2), "{source}");
    }
    for source in [
        "HloModule m body { p = f32[] parameter(0) ROOT n = copy(p) } ENTRY main { p = f32[3] parameter(0) ROOT r = map(p), to_apply=body }",
        "HloModule m body { ROOT c = f32[] constant(1) } ENTRY main { ROOT r = scan(), dimensions={0}, num_carries=0, to_apply=body }",
        "HloModule m cmp { a = f32[] parameter(0) b = f32[] parameter(1) c = f32[] parameter(2) d = f32[] parameter(3) ROOT r = compare(a,b), direction=LT } ENTRY main { a = f32[2] parameter(0) b = f32[3] parameter(1) ROOT r = sort(a,b), dimensions={0}, to_apply=cmp }",
    ] {
        assert!(text::parse_module(source).is_err(), "{source}");
        assert_eq!(native(source, "parse").status.code(), Some(2), "{source}");
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn nested_instruction_assembly_matches_pinned_cpp() {
    for source in [
        "HloModule m, is_scheduled=true ENTRY main { p = f32[2] parameter(0) ROOT r = add(sqrt(p), f32[2] negate(copy(p))) }",
        "HloModule m ENTRY main { ROOT r = add(f32[] constant(2), f32[] constant(3)) }",
        "HloModule m ENTRY main { ROOT r = copy(f32[] parameter(0)) }",
        "HloModule m ENTRY main { p = (f32[],f32[]) parameter(0) ROOT r = add(copy(p#0), negate(p#1)) }",
        "HloModule m ENTRY main { p = f32[] parameter(0) a = copy(copy(p)) ROOT r = add(a,copy.anon) }",
        "HloModule m ENTRY main { p = f32[] parameter(0) copy.anon = copy(p) ROOT r = add(copy(p),copy.anon) }",
        "HloModule m helper { ROOT r = copy(f32[] parameter(0)) } ENTRY main { ROOT r = copy(f32[] parameter(0)) }",
        "HloModule m ENTRY main { ROOT r = f32[2] broadcast(f32[] constant(1)) }",
        "HloModule m ENTRY main { t = after-all() ROOT r = token[] outfeed(negate(f32[] constant(1)),t) }",
        "HloModule m ENTRY main { ROOT r = tuple(s32[] constant(1), after-all()) }",
    ] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        let output = native(source, "verify");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let wire: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        let expected = wire
            .computations
            .iter()
            .find(|c| c.id == wire.entry_computation_id)
            .unwrap();
        let actual = module
            .as_proto()
            .computations
            .iter()
            .find(|c| c.id == module.as_proto().entry_computation_id)
            .unwrap();
        let graph = |c: &xlahlo::proto::hlo::HloComputationProto| {
            let mut graph = c
                .instructions
                .iter()
                .map(|i| {
                    (
                        i.name.clone(),
                        i.opcode.clone(),
                        i.operand_ids
                            .iter()
                            .map(|id| {
                                c.instructions
                                    .iter()
                                    .find(|p| p.id == *id)
                                    .unwrap()
                                    .name
                                    .clone()
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            graph.sort();
            graph
        };
        assert_eq!(graph(actual), graph(expected), "{source}");
        assert!(
            module
                .shape(module.root(module.entry().unwrap()).unwrap())
                .unwrap()
                .equal_including_layout(
                    &xlahlo::Shape::from_proto(root(&wire).shape.as_ref().unwrap().clone())
                        .unwrap()
                ),
            "{source}"
        );
        let printed = text::print_module(&module).unwrap();
        assert!(native(&printed, "verify").status.success(), "{printed}");
        assert!(
            native_bytes(
                &interchange::encode(&module.to_proto_with_program_shapes().unwrap()).unwrap(),
                "verify-proto"
            )
            .status
            .success(),
            "{source}"
        );
    }
    for source in [
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = copy(transpose(p), dimensions={0}) }",
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = copy(get-tuple-element(p), index=0) }",
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = add(copy(p),) }",
        "HloModule m ENTRY main { ROOT r = copy(constant(1)) }",
        "HloModule m ENTRY main { p = f32[] parameter(0) a = copy(copy(p)) copy.anon = negate(p) ROOT r = add(a,copy.anon) }",
    ] {
        assert!(text::parse_module(source).is_err(), "{source}");
        assert_eq!(native(source, "parse").status.code(), Some(2), "{source}");
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn headerless_computation_assembly_matches_pinned_cpp() {
    for source in [
        "ENTRY main { ROOT p = f32[] parameter(0) }",
        "main { ROOT p = f32[] parameter(0) }",
        "helper { ROOT p = f32[] parameter(0) } main { x = f32[] parameter(0) ROOT r = call(x), to_apply=helper }",
        "ENTRY first { ROOT x = f32[] constant(0) } last { ROOT x = f32[] constant(1) }",
        "/*headerless*/ ENTRY main (p: f32[]) -> f32[] { ROOT p = f32[] parameter(0) }",
    ] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        let output = native(source, "verify");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let wire: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        assert_eq!(module.as_proto().name, wire.name, "{source}");
        assert_eq!(
            module.as_proto().entry_computation_name,
            wire.entry_computation_name,
            "{source}"
        );
        assert_eq!(
            module.as_proto().computations.len(),
            wire.computations.len(),
            "{source}"
        );
        assert!(
            module
                .shape(module.root(module.entry().unwrap()).unwrap())
                .unwrap()
                .equal_including_layout(
                    &xlahlo::Shape::from_proto(root(&wire).shape.as_ref().unwrap().clone())
                        .unwrap()
                ),
            "{source}"
        );
        let printed = text::print_module(&module).unwrap();
        assert!(native(&printed, "verify").status.success(), "{printed}");
        assert!(
            native_bytes(
                &interchange::encode(&module.to_proto_with_program_shapes().unwrap()).unwrap(),
                "verify-proto"
            )
            .status
            .success(),
            "{source}"
        );
    }
    for source in [
        "main, replica_count=2 { ROOT p = f32[] parameter(0) }",
        "HloModule",
        "",
        "ENTRY main { ROOT x = f32[] constant(0) } ENTRY last { ROOT x = f32[] constant(1) }",
    ] {
        assert!(text::parse_module(source).is_err(), "{source}");
        assert_eq!(native(source, "parse").status.code(), Some(2), "{source}");
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn compact_tuple_operands_match_pinned_cpp() {
    fn projections(
        computation: &xlahlo::proto::hlo::HloComputationProto,
    ) -> Vec<(String, i64, String)> {
        let mut result = computation
            .instructions
            .iter()
            .filter(|i| i.opcode == "get-tuple-element")
            .map(|i| {
                let operand = computation
                    .instructions
                    .iter()
                    .find(|p| i.operand_ids == [p.id])
                    .unwrap();
                (i.name.clone(), i.tuple_index, operand.name.clone())
            })
            .collect::<Vec<_>>();
        result.sort();
        result
    }
    for source in [
        "HloModule m, is_scheduled=true ENTRY main { p = (f32[2],(s32[],f32[])) parameter(0) a = copy(f32[2] p#0) b = add(p#0,p#00) c = negate(p#1#0) ROOT r = tuple(a,b,c) }",
        "HloModule m ENTRY main { p = ((f32[2]{0}),token[]) parameter(0) ROOT r = copy(f32[2] p#0#0) }",
        "HloModule m ENTRY main { p = (f32[]) parameter(0) p_0 = f32[] constant(1) ROOT r = add(p#0,p_0) }",
        "HloModule m ENTRY main { p = f32[2] parameter(0) t = after-all() s = (f32[2],s32[],token[]) send(p,t), channel_id=1 ROOT r = copy(u32[] s#1) }",
    ] {
        let module = text::parse_module(source).unwrap();
        let output = native(source, "parse");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let wire: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        let expected = wire
            .computations
            .iter()
            .find(|c| c.id == wire.entry_computation_id)
            .unwrap();
        let actual = module
            .as_proto()
            .computations
            .iter()
            .find(|c| c.id == module.as_proto().entry_computation_id)
            .unwrap();
        assert_eq!(projections(actual), projections(expected), "{source}");
        let native_root = expected
            .instructions
            .iter()
            .find(|i| i.id == expected.root_id)
            .unwrap();
        assert!(
            module
                .shape(module.root(module.entry().unwrap()).unwrap())
                .unwrap()
                .equal_including_layout(
                    &xlahlo::Shape::from_proto(native_root.shape.as_ref().unwrap().clone())
                        .unwrap()
                ),
            "{source}"
        );
        let verified = module.verify(Default::default()).is_ok();
        for text in [source, &text::print_module(&module).unwrap()] {
            let output = native(text, "verify");
            assert_eq!(
                output.status.code(),
                Some(if verified { 0 } else { 3 }),
                "{text}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let output = native_bytes(
            &interchange::encode(&module.to_proto_with_program_shapes().unwrap()).unwrap(),
            "verify-proto",
        );
        assert_eq!(
            output.status.code(),
            Some(if verified { 0 } else { 3 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    for body in [
        "p = (f32[]) parameter(0) ROOT r = copy(p#-1)",
        "p = (f32[]) parameter(0) ROOT r = copy(p#1)",
        "p = (f32[]) parameter(0) ROOT r = copy(p#0#0)",
        "p = (f32[]) parameter(0) ROOT r = copy(f64[] p#0)",
        "ROOT r = f32[] copy(p#0) p = (f32[]) parameter(0)",
        "p = (f32[]) parameter(0) ROOT r = copy(p#)",
    ] {
        let source = format!("HloModule m ENTRY main {{ {body} }}");
        assert!(text::parse_module(&source).is_err(), "{source}");
        assert_eq!(native(&source, "parse").status.code(), Some(2), "{source}");
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn parser_constructor_regressions_match_pinned_cpp() {
    let mut sources = vec![
        "HloModule m ENTRY main { c = s32[] constant(1) ROOT s = s32[] slice(c), slice={} }".into(),
        "HloModule m ENTRY main (ignored: s32[7]) -> f32[2,3]{1,0} { ROOT p = f32[2,3]{0,1} parameter(0) }".into(),
        "HloModule m ENTRY main () -> s32[2] { ROOT p = s32[3] parameter(0) }".into(),
        "HloModule m ENTRY main { p = f32[1,2,1] parameter(0) k = f32[1,1,1] parameter(1) ROOT c = f32[1,2,1] custom-call(p,k), custom_call_target=\"DynamicConvolutionForward\", window={size=1}, dim_labels=b0f_0io->b0f, operand_precision={high,default} }".into(),
        "/*module*/HloModule m ENTRY main { ROOT c = (f32[2,0],/*index=1*/s32[]) constant(({/*i0=0*/{},/*i0=1*/{}},7/*value*/)) }/*eof*/".to_owned(),
        "HloModule m sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT s = f32[] add(a,b) } ENTRY main { x = f32[] constant(3) z = f32[] constant(0) ROOT r = f32[] reduce-window(x,z), to_apply=sum }".into(),
        "HloModule m sum { a = f32[] parameter(0) b = f32[] parameter(1) c = f32[] parameter(2) d = f32[] parameter(3) x = f32[] add(a,c) y = f32[] add(b,d) ROOT r = (f32[],f32[]) tuple(x,y) } ENTRY main { x = f32[] constant(3) z = f32[] constant(0) ROOT r = (f32[],f32[]) reduce-window(x,x,z,z), to_apply=sum }".into(),
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0), literal=f32[] 1 }".into(),
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0) } /*unterminated".into(),
        "HloModule m, debug_attributes={ {\"z\" {1}}:(), {\"c\"}:({log_mode=default,callback_id=3,partitioned=true},{op_id=7}), {\"a\"}:{callback_id=1,callback_id=2} } ENTRY main { ROOT c = f32[] constant(1) }".into(),
        "HloModule m, debug_attributes={ {\"c\"}:{log_mode=fusion_debugger,op_id=7}, {\"\" {1}}:{callback_id=0} } ENTRY main { ROOT c = f32[] constant(1) }".into(),
        "HloModule m, debug_attributes={ {\"c\"}:{partitioned=true}, {\"c\"}:{op_id=-1} } ENTRY main { ROOT c = f32[] constant(1) }".into(),
        "HloModule m, debug_attributes={ {\"c\"}:{callback_id=1}, {\"c\"}:() } ENTRY main { ROOT c = f32[] constant(1) }".into(),
        "HloModule m, debug_attributes={ {\"c\"}:{log_mode=none} } ENTRY main { ROOT c = f32[] constant(1) }".into(),
    ];
    for literal in [
        "f32[] 0.1",
        "s32[2]{0} {1,2}",
        "(s32[] 7,f32[1]{0} {2})",
        "()",
        "((s32[] 1),f32[] 2)",
    ] {
        sources.push(format!("HloModule m ENTRY main {{ ROOT c = f32[3] custom-call(), custom_call_target=\"/*literal*/\", literal={literal} }}"));
    }
    for source in sources {
        let rust = text::parse_module(&source);
        let output = native(&source, "parse");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{source}: Rust {rust:?}, C++ {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let Ok(module) = rust else { continue };
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        assert_eq!(
            module.as_proto().debug_attributes,
            cpp.debug_attributes,
            "{source}"
        );
        let entry = cpp
            .computations
            .iter()
            .find(|c| c.id == cpp.entry_computation_id)
            .unwrap();
        let cpp_root = entry
            .instructions
            .iter()
            .find(|i| i.id == entry.root_id)
            .unwrap();
        let root = module
            .instruction(module.root(module.entry().unwrap()).unwrap())
            .unwrap();
        assert_eq!(root.window, cpp_root.window, "{source}");
        assert_eq!(
            root.convolution_dimension_numbers, cpp_root.convolution_dimension_numbers,
            "{source}"
        );
        let wire = module.to_proto_with_program_shapes().unwrap();
        let program = wire
            .computations
            .iter()
            .find(|c| c.id == wire.entry_computation_id)
            .unwrap()
            .program_shape
            .as_ref()
            .unwrap();
        let cpp_program = entry.program_shape.as_ref().unwrap();
        assert_eq!(program.parameters, cpp_program.parameters, "{source}");
        assert_eq!(program.result, cpp_program.result, "{source}");
        assert_eq!(
            interchange::encode(root.literal.get_or_default()).unwrap(),
            interchange::encode(cpp_root.literal.get_or_default()).unwrap(),
            "{source}"
        );
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        let output = native(&printed, "verify");
        assert!(
            output.status.success(),
            "{printed}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = interchange::encode(&module.to_proto_with_program_shapes().unwrap()).unwrap();
        let output = native_bytes(&bytes, "verify-proto");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn standalone_collective_assembly_matches_pinned_cpp() {
    for (shape, groups, verified) in [
        (2, "[1,1]<=[1]", true),
        (4, "mesh[\"x\"=2] {\"x\"}", true),
        (4, "mesh[\"x\"=4] {\"x\":(1)2}", true),
        (6, "mesh[\"x\"=2] {\"x\"}", false),
        (2, "single_device[device_id=7] {}", false),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ p = f32[2] parameter(0) ROOT g = f32[{shape}] all-gather(p), dimensions={{0}}, replica_groups={groups} }}"
        );
        let module = text::parse_module(&source).unwrap();
        assert_eq!(
            module.verify(Default::default()).is_ok(),
            verified,
            "{source}"
        );
        let printed = text::print_module(&module).unwrap();
        let output = native(&printed, "verify");
        assert_eq!(
            output.status.code(),
            Some(if verified { 0 } else { 3 }),
            "{printed}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = interchange::encode(&module.to_proto_with_program_shapes().unwrap()).unwrap();
        let output = native_bytes(&bytes, "verify-proto");
        assert_eq!(
            output.status.code(),
            Some(if verified { 0 } else { 3 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    for source in [
        "{}",
        "{{}}",
        "{{0,1},{2,3}}",
        "{{-1,4,4},{7}}",
        "{{18446744073709551615}}",
        "[2,3]<=[3,2]T(1,0)",
        "[2,3]<=[6]",
        "",
        "{0,1}",
        "{{1,}}",
        "{{1}} extra",
        "[2,3]<=[5]",
        "[2,3]<=[3,2]T(0,2)",
        "[2,3]<=[3,2]",
        "[2,3]<=[6]T(0)",
        "mesh[\"x\"=2,\"y\"=3] {\"x\"}",
        "mesh[\"x\"=2,\"y\"=3] {\"y\",\"x\"}",
        "mesh[\"x\"=8] {\"x\":(1)2}",
        "mesh[\"x\"=16] {\"x\":(8)2,\"x\":(1)2}",
        "mesh[\"x\"=2,\"y\"=2], device_ids=(3,2,1,0) {\"x\"}",
        "mesh[\"x\"=2,\"y\"=2], device_ids=([2,2]T(1,0)) {\"x\"}",
        "single_device[device_id=0] {}",
        "mesh[\"x\"=2] {}",
    ] {
        let rust = text::parse_replica_groups(source);
        let output = native(source, "parse-replica-groups");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{source}: Rust {rust:?}, native {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Ok(groups) = rust {
            let cpp: xlahlo::proto::xla_data::CollectiveDeviceListProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(groups, cpp.replica_groups, "{source}");
            let printed = text::print_replica_groups(&groups).unwrap();
            let output = native(&printed, "parse-replica-groups");
            assert!(output.status.success(), "{printed}");
            let cpp: xlahlo::proto::xla_data::CollectiveDeviceListProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(groups, cpp.replica_groups, "{source}");
        }
        let rust = text::parse_collective_device_list(source);
        let output = native(source, "parse-device-list");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{source}: Rust {rust:?}, native {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Ok(list) = rust {
            let cpp: HloInstructionProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(Some(list.clone()), cpp.replica_group_list, "{source}");
            let output = native(
                &text::print_collective_device_list(&list).unwrap(),
                "parse-device-list",
            );
            assert!(
                output.status.success(),
                "{source}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let cpp: HloInstructionProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(Some(list), cpp.replica_group_list, "{source}");
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn generic_async_sugar_matches_pinned_cpp() {
    let unary = "HloModule m ENTRY main { p = f32[2] parameter(0) s = ((f32[2]),f32[2],u32[]) negate-start(p) u = ((f32[2]),f32[2],u32[]) negate-update(s) ROOT d = f32[2] negate-done(u) }";
    let cases = vec![
        unary.to_owned(),
        unary.replace("negate-done", "abs-done"),
        unary.replace("negate-update", "abs-update"),
        unary.replace("negate-update", "copy-update"),
        unary.replace("negate-start(p)", "negate-start(p), async_execution_thread=\"worker\", metadata={op_name=\"start\"}, sharding={replicated}")
            .replace("negate-done(u)", "negate-done(u), metadata={op_name=\"done\"}, statistics={index=3,a=-0.0}"),
        unary.replace("negate", "exponential"),
        unary.replace("negate-done(u)", "negate-done(u), dimensions={0}"),
        unary.replace("negate-update(s)", "negate-update(s), calls=main"),
        "HloModule m ENTRY main { s = ((),f32[],u32[]) constant-start()(1) ROOT d = f32[] constant-done(s) }".into(),
        "HloModule m ENTRY main { p = f32[2] parameter(0) s = ((f32[2]),f32[2],u32[]) negate-start(p), output_to_operand_aliasing={{1}:(0,{})} ROOT d = f32[2] negate-done(s) }".into(),
        "HloModule m ENTRY main { p = f32[2] parameter(0) s = ((f32[2]),f32[2],u32[]) custom-call-start(p), custom_call_target=\"test\", output_to_operand_aliasing={{}:(0,{})} ROOT d = f32[2] custom-call-done(s) }".into(),
        "HloModule m ENTRY main { p = f32[2] parameter(0) s = ((f32[2]),f32[2],u32[]) all-gather-start(p), dimensions={0}, replica_groups={{0}} ROOT d = f32[2] all-gather-done(s) }".into(),
        "HloModule m ENTRY main { p = f32[2] parameter(0) s = ((f32[2],f32[2]),f32[2],u32[]) add-start(p,p) ROOT d = f32[2] add-done(s) }".into(),
        "HloModule m ENTRY main { p = f32[2,3] parameter(0) q = f32[3,4] parameter(1) s = ((f32[2,3],f32[3,4]),f32[2,4],u32[]) dot-start(p,q), lhs_contracting_dims={1}, rhs_contracting_dims={0} ROOT d = f32[2,4] dot-done(s) }".into(),
        "HloModule m f { ROOT p = f32[2] parameter(0) } ENTRY main { p = f32[2] parameter(0) s = ((),(),u32[]) call-start(), to_apply=f ROOT d = f32[2] call-done(s) }".into(),
        "HloModule m f { ROOT p = f32[2] parameter(0) } ENTRY main { p = f32[2] parameter(0) s = ((f32[2]),f32[2],u32[]) call-start(p), to_apply=f ROOT d = f32[2] call-done(s) }".into(),
        "HloModule m f { ROOT p = f32[2] parameter(0) } ENTRY main { p = f32[2] parameter(0) s = ((f32[2]),f32[2],u32[]) fusion-start(p), kind=kLoop, calls=f ROOT d = f32[2] fusion-done(s) }".into(),
        "HloModule m ENTRY main { p = f32[2] parameter(0) s = ((f32[2]),f32[2],u32[]) custom-call-start(p), custom_call_target=\"test\", backend_config=\"cfg\" ROOT d = f32[2] custom-call-done(s) }".into(),
        "HloModule m ENTRY main { p = f32[2] parameter(0) s = ((),(),u32[]) negate-start(p) u = ((f32[2]),f32[2],u32[]) negate-update(s) ROOT d = f32[2] negate-done(u) }".into(),
        "HloModule m ENTRY main { p = f32[2] parameter(0) s = ((f32[2]),f32[2],u32[]) negate-start(p) ROOT d = f32[3] negate-done(s) }".into(),
    ];
    fn wrapped<'a>(
        module: &'a HloModuleProto,
        start: &HloInstructionProto,
    ) -> &'a xlahlo::proto::hlo::HloComputationProto {
        module
            .computations
            .iter()
            .find(|c| c.id == start.called_computation_ids[0])
            .unwrap()
    }
    for source in cases {
        let rust = text::parse_module(&source);
        let output = native(&source, "parse");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{source}: Rust {rust:?}, C++ {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let Ok(module) = rust else { continue };
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        let entry = module
            .as_proto()
            .computations
            .iter()
            .find(|c| c.id == module.as_proto().entry_computation_id)
            .unwrap();
        let other = cpp
            .computations
            .iter()
            .find(|c| c.id == cpp.entry_computation_id)
            .unwrap();
        for (left, right) in entry.instructions.iter().zip(&other.instructions) {
            assert_eq!(left.opcode, right.opcode, "{source}");
            if left.opcode != "async-start" {
                continue;
            }
            assert_eq!(
                left.async_execution_thread,
                if right.async_execution_thread == "main" {
                    ""
                } else {
                    &right.async_execution_thread
                },
                "{source}"
            );
            let a = wrapped(module.as_proto(), left);
            let b = wrapped(&cpp, right);
            let a_root = a.instructions.iter().find(|i| i.id == a.root_id).unwrap();
            let b_root = b.instructions.iter().find(|i| i.id == b.root_id).unwrap();
            assert_eq!(a_root.opcode, b_root.opcode, "{source}");
            assert!(
                Shape::from_proto(a_root.shape.get_or_default().clone())
                    .unwrap()
                    .equal_including_layout(
                        &Shape::from_proto(b_root.shape.get_or_default().clone()).unwrap()
                    ),
                "{source}"
            );
            assert_eq!(
                a_root.dot_dimension_numbers, b_root.dot_dimension_numbers,
                "{source}"
            );
            assert_eq!(a_root.precision_config, b_root.precision_config, "{source}");
            assert_eq!(a_root.backend_config, b_root.backend_config, "{source}");
            assert_eq!(
                left.output_operand_aliasing, right.output_operand_aliasing,
                "{source}"
            );
            assert_eq!(
                a_root.output_operand_aliasing, b_root.output_operand_aliasing,
                "{source}"
            );
            if let (Some(a), Some(b)) = (a_root.literal.as_ref(), b_root.literal.as_ref()) {
                assert_eq!(
                    Literal::from_proto(a.clone()).unwrap().bits().unwrap(),
                    Literal::from_proto(b.clone()).unwrap().bits().unwrap(),
                    "{source}"
                );
            } else {
                assert_eq!(
                    a_root.literal.is_some(),
                    b_root.literal.is_some(),
                    "{source}"
                );
            }
            assert_eq!(
                a_root.metadata.get_or_default().op_name,
                b_root.metadata.get_or_default().op_name,
                "{source}"
            );
            let params = |body: &xlahlo::proto::hlo::HloComputationProto| {
                let mut parameters = body
                    .instructions
                    .iter()
                    .filter(|i| i.opcode == "parameter")
                    .collect::<Vec<_>>();
                parameters.sort_by_key(|i| i.parameter_number);
                parameters
                    .into_iter()
                    .map(|i| Shape::from_proto(i.shape.get_or_default().clone()).unwrap())
                    .collect::<Vec<_>>()
            };
            let (a, b) = (params(a), params(b));
            assert_eq!(a.len(), b.len(), "{source}");
            for (a, b) in a.iter().zip(&b) {
                assert!(a.equal_including_layout(b), "{source}")
            }
        }
        let verified = module.verify(Default::default()).is_ok();
        let output = native(&source, "verify");
        assert_eq!(
            output.status.code(),
            Some(if verified { 0 } else { 3 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let printed = text::print_module(&module).unwrap();
        let output = native(&printed, "parse");
        assert!(
            output.status.success(),
            "{printed}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            text::parse_module(&printed)
                .unwrap()
                .verify(Default::default())
                .is_ok(),
            verified,
            "{source}"
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn standalone_geometry_assembly_matches_pinned_cpp() {
    use xlahlo::proto::xla_data::{LayoutProto, PaddingConfig};
    for source in [
        "{}",
        "{f32[2], (s32[],token[]), b(f32[3])}",
        "{f32[<=4]{0:S(5)}}",
        "",
        "f32[2],s32[]",
        "{f32[2],}",
        "{} trailing",
    ] {
        let rust = text::parse_shape_list(source);
        let output = native(source, "parse-shape-list");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Ok(shapes) = rust {
            let cpp: ShapeProto = interchange::decode(&output.stdout, Default::default()).unwrap();
            assert!(
                Shape::tuple(shapes.clone())
                    .unwrap()
                    .equal_including_layout(&Shape::from_proto(cpp).unwrap()),
                "{source}"
            );
            let printed = text::print_shape_list(&shapes).unwrap();
            assert!(
                native(&printed, "parse-shape-list").status.success(),
                "{printed}"
            );
        }
    }
    for source in [
        "{}",
        "{1,0}",
        "{7,7,-1}",
        "{18446744073709551615}",
        "{1,0:T(2,*)L(8)E(4)S(5)SC(0:1,2)M(16)}",
        "",
        "{0,}",
        "{0} trailing",
    ] {
        let rust = text::parse_layout(source);
        let output = native(source, "parse-layout");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Ok(layout) = rust {
            let cpp: LayoutProto = interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(layout, cpp, "{source}");
            let printed = text::print_layout(&layout).unwrap();
            let output = native(&printed, "parse-layout");
            assert!(output.status.success(), "{printed}");
            let cpp: LayoutProto = interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(layout, cpp, "{printed}");
        }
    }
    for source in [
        "0_0",
        "-1_2_3x4_-5_0",
        "0_0_-1",
        "-9223372036854775808_9223372036854775807",
        "",
        "1",
        "0_0_0_0",
        "0_0x",
        "0_0 trailing",
        "18446744073709551615_0",
        "0_0.5",
        "+1_0",
    ] {
        let rust = text::parse_padding_config(source);
        let output = native(source, "parse-padding");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Ok(padding) = rust {
            let cpp: PaddingConfig =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(padding, cpp, "{source}");
            let printed = text::print_padding_config(&padding).unwrap();
            let output = native(&printed, "parse-padding");
            assert!(output.status.success(), "{printed}");
            let cpp: PaddingConfig =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(padding, cpp, "{printed}");
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn buffer_outer_layout_import_matches_pinned_cpp() {
    for change in ["minor-order", "alignment", "tile", "physical-shape"] {
        let mut wire =
            text::parse_module("HloModule m ENTRY main { ROOT p = b(f32[2,3]) parameter(0) }")
                .unwrap()
                .to_proto_with_program_shapes()
                .unwrap();
        let shape = wire.computations[0].instructions[0]
            .shape
            .mut_or_insert_default();
        shape.layout = text::parse_shape("f32[2,3]{0,1}")
            .unwrap()
            .as_proto()
            .layout
            .clone();
        let inner = shape.tuple_shapes[0].layout.mut_or_insert_default();
        match change {
            "minor-order" => inner.minor_to_major = vec![0, 0],
            "alignment" => inner.tail_padding_alignment_in_elements = -1,
            "tile" => inner.tiles.push(xlahlo::proto::xla_data::TileProto {
                dimensions: vec![-1],
                ..Default::default()
            }),
            _ => {
                inner.physical_shape = Some(ShapeProto {
                    element_type: xlahlo::PrimitiveType::BUFFER.into(),
                    ..Default::default()
                })
                .into()
            }
        }
        let rust = xlahlo::Module::from_proto(wire.clone());
        let output = native_bytes(&interchange::encode(&wire).unwrap(), "import-proto");
        assert_eq!(rust.is_ok(), change == "minor-order", "{change}");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{change}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    for outer in [
        "f32[2,3]{0,1}",
        "f32[2,3]{1,0:S(5)}",
        "f32[2,3]{1,0:T(2,4)}",
    ] {
        let mut wire = text::parse_module(
            "HloModule m ENTRY main { ROOT p = b(f32[2,3]{1,0:S(5)}) parameter(0) }",
        )
        .unwrap()
        .to_proto_with_program_shapes()
        .unwrap();
        let shape = wire.computations[0].instructions[0]
            .shape
            .mut_or_insert_default();
        shape.layout = text::parse_shape(outer).unwrap().as_proto().layout.clone();
        let expected_wire = shape.clone();
        let module = xlahlo::Module::from_proto(wire.clone()).unwrap();
        let actual = module
            .shape(module.root(module.entry().unwrap()).unwrap())
            .unwrap();
        let output = native_bytes(&interchange::encode(&wire).unwrap(), "import-proto");
        assert!(
            output.status.success(),
            "{outer}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        let expected = Shape::from_proto(root(&cpp).shape.get_or_default().clone()).unwrap();
        assert!(actual.equal_including_layout(&expected), "{outer}");
        assert_eq!(actual.as_proto(), &expected_wire);
        assert_eq!(
            actual.array_view().unwrap().as_proto().layout,
            expected.array_view().unwrap().as_proto().layout
        );
        assert_eq!(actual.byte_size(8).unwrap(), expected.byte_size(8).unwrap());
        assert_eq!(
            actual.array_data_size().unwrap(),
            expected.array_data_size().unwrap()
        );
        assert!(text::print_shape(&actual).is_err());
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn zero_tail_alignment_import_matches_pinned_cpp() {
    let mut wire = text::parse_module("HloModule m ENTRY main { ROOT p = f32[2]{0} parameter(0) }")
        .unwrap()
        .to_proto_with_program_shapes()
        .unwrap();
    wire.computations[0].instructions[0]
        .shape
        .mut_or_insert_default()
        .layout
        .mut_or_insert_default()
        .tail_padding_alignment_in_elements = 0;

    let rust = xlahlo::Module::from_proto(wire.clone()).unwrap();
    let actual = rust
        .shape(rust.root(rust.entry().unwrap()).unwrap())
        .unwrap();
    assert_eq!(
        actual.layout().unwrap().tail_padding_alignment_in_elements,
        1
    );

    let output = native_bytes(&interchange::encode(&wire).unwrap(), "import-proto");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
    assert_eq!(
        root(&cpp)
            .shape
            .get_or_default()
            .layout
            .as_ref()
            .unwrap()
            .tail_padding_alignment_in_elements,
        1
    );
    let expected = Shape::from_proto(root(&cpp).shape.get_or_default().clone()).unwrap();
    assert!(actual.equal_including_layout(&expected));
    assert_eq!(actual.as_proto(), expected.as_proto());
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn category_specific_shape_import_matches_pinned_cpp() {
    use xlahlo::{
        Module,
        proto::xla_data::{LayoutProto, ShapeProto},
    };

    for (source, kind) in [
        ("f32[2]", "array"),
        ("(f32[2])", "tuple"),
        ("token[]", "token"),
        ("opaque[]", "opaque"),
        ("b(f32[2])", "buffer"),
    ] {
        let base = text::parse_module(&format!(
            "HloModule m ENTRY main {{ ROOT p = {source} parameter(0) }}"
        ))
        .unwrap()
        .to_proto_with_program_shapes()
        .unwrap();
        for change in ["dimensions", "dynamic-flags", "tuple-payload", "layout"] {
            let mut wire = base.clone();
            let shape = wire.computations[0].instructions[0].shape.as_mut().unwrap();
            match change {
                "dimensions" => shape.dimensions = vec![-7, i64::MIN],
                "dynamic-flags" => shape.is_dynamic_dimension = vec![true, false, true],
                "tuple-payload" => shape.tuple_shapes.push(ShapeProto::new()),
                "layout" => {
                    shape.layout = Some(LayoutProto {
                        memory_space: 5,
                        ..Default::default()
                    })
                    .into()
                }

                _ => unreachable!(),
            }
            let accepted = match kind {
                "array" => matches!(change, "tuple-payload" | "layout"),
                "tuple" => matches!(change, "dimensions" | "dynamic-flags"),
                "buffer" => matches!(change, "dimensions" | "dynamic-flags" | "layout"),
                _ => change != "layout",
            };
            let rust = Module::from_proto(wire.clone());
            assert_eq!(rust.is_ok(), accepted, "{kind} {change}: {rust:?}");
            let output = native_bytes(&interchange::encode(&wire).unwrap(), "import-proto");
            assert_eq!(
                output.status.code(),
                Some(if accepted { 0 } else { 2 }),
                "{kind} {change}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            if let Ok(module) = rust {
                let cpp: HloModuleProto =
                    interchange::decode(&output.stdout, Default::default()).unwrap();
                assert_eq!(
                    module.as_proto().debug_attributes,
                    cpp.debug_attributes,
                    "{source}"
                );
                let actual =
                    Shape::from_proto(root(module.as_proto()).shape.as_ref().unwrap().clone())
                        .unwrap();
                let expected =
                    Shape::from_proto(root(&cpp).shape.as_ref().unwrap().clone()).unwrap();
                assert!(actual.equal_including_layout(&expected), "{kind} {change}");
                assert!(
                    text::print_module(&module).is_err() || kind == "array" && change == "layout"
                );
            }
        }
    }
    let mut wire = text::parse_module("HloModule m ENTRY main { ROOT c = f32[2] constant({1,2}) }")
        .unwrap()
        .to_proto_with_program_shapes()
        .unwrap();
    wire.computations[0].instructions[0]
        .literal
        .as_mut()
        .unwrap()
        .shape
        .as_mut()
        .unwrap()
        .tuple_shapes
        .push(Shape::opaque().into_proto());
    let rust = Module::from_proto(wire.clone()).unwrap();
    rust.verify(Default::default()).unwrap();
    let output = native_bytes(&interchange::encode(&wire).unwrap(), "verify-proto");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn standalone_attribute_assembly_matches_pinned_cpp() {
    use protobuf::MessageFull;
    use xlahlo::proto::xla_data::{
        FrontendAttributes, OriginalValueProto, ParameterReplication, StatisticsViz,
    };

    fn check<T: MessageFull>(
        inputs: &[&str],
        mode: &str,
        parse: fn(&str) -> xlahlo::Result<T>,
        print: fn(&T) -> xlahlo::Result<String>,
        equal: fn(&T, &T) -> bool,
    ) {
        for &source in inputs {
            let rust = parse(source);
            let output = native(source, mode);
            assert_eq!(
                output.status.code(),
                Some(if rust.is_ok() { 0 } else { 2 }),
                "{mode} {source}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            if let Ok(value) = rust {
                let other: T = interchange::decode(&output.stdout, Default::default()).unwrap();
                assert!(
                    equal(&value, &other),
                    "{mode} {source}: {value:?} != {other:?}"
                );
                let printed = print(&value).unwrap();
                let output = native(&printed, mode);
                assert!(output.status.success(), "{mode} {printed}");
                let other: T = interchange::decode(&output.stdout, Default::default()).unwrap();
                assert!(equal(&value, &other), "{mode} printed {printed}");
                assert!(equal(&value, &parse(&printed).unwrap()));
                let trailing = format!("{source} trailing");
                assert!(parse(&trailing).is_err());
                assert_eq!(native(&trailing, mode).status.code(), Some(2));
            }
        }
    }

    check::<OriginalValueProto>(
        &[
            "{}",
            r#"{{"p"{0}}}"#,
            r#"{({"p"},{},{"q"{1}}),["parent"]}"#,
            "{[synthetic_call]}",
            "{",
            "{} trailing",
        ],
        "parse-original-value",
        text::parse_original_value,
        text::print_original_value,
        |a, b| a == b,
    );
    check::<FrontendAttributes>(
        &[
            "{}",
            r#"{b="two",a="one",a="last"}"#,
            r#"{a={"nested":{"text":"} // literal","array":[1,2]}}}"#,
            r#"{a={"unfinished":1}"#,
            r#"{a=1}"#,
            "{} trailing",
        ],
        "parse-frontend-attributes",
        text::parse_frontend_attributes,
        text::print_frontend_attributes,
        |a, b| a == b,
    );
    check::<StatisticsViz>(
        &[
            "{}",
            "{ignored=-3,a=nan(0x123),a=-0.0,b=1e999}",
            "{ignored=18446744073709551615,a=18446744073709551615}",
            "{ignored=0,a=inf}",
            "{} trailing",
        ],
        "parse-statistics",
        text::parse_statistics,
        text::print_statistics,
        |a, b| interchange::encode(a).unwrap() == interchange::encode(b).unwrap(),
    );
    check::<ParameterReplication>(
        &[
            "{}",
            "{true}",
            "{false,true,false}",
            "{true,}",
            "{1}",
            "{} trailing",
        ],
        "parse-parameter-replication",
        text::parse_parameter_replication,
        text::print_parameter_replication,
        |a, b| a == b,
    );
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn contextual_literal_import_matches_pinned_cpp() {
    use xlahlo::{Instruction, Module, Opcode, PrimitiveType as P, proto::xla_data::LiteralProto};

    fn module(literal: Literal) -> HloModuleProto {
        let mut module = Module::new("m");
        let main = module.add_computation("main").unwrap();
        let instruction = Instruction::new("c", Opcode::Constant, literal.shape().clone())
            .with_fields(|wire| wire.literal = Some(literal.into_proto()).into())
            .unwrap();
        let root = module.add_instruction(main, instruction, &[], &[]).unwrap();
        module.set_root(main, root).unwrap();
        module.set_entry(main).unwrap();
        module.to_proto_with_program_shapes().unwrap()
    }

    fn changed(base: &HloModuleProto, edit: impl FnOnce(&mut LiteralProto)) -> HloModuleProto {
        let mut wire = base.clone();
        edit(
            wire.computations[0].instructions[0]
                .literal
                .as_mut()
                .unwrap(),
        );
        wire
    }

    let array =
        || Literal::from_bits(Shape::array(P::F32, vec![2, 2]).unwrap(), &[0, 1, 2, 3]).unwrap();
    let dense = module(array());
    let tuple = module(
        Literal::tuple(vec![
            array(),
            Literal::tuple(vec![array(), Literal::token()]).unwrap(),
        ])
        .unwrap(),
    );
    let token = module(Literal::token());
    let mut cases = vec![
        ("array", dense.clone(), true),
        ("tuple", tuple.clone(), true),
        ("token", token.clone(), true),
        (
            "array-tuple-payload",
            changed(&dense, |p| p.tuple_literals.push(LiteralProto::new())),
            true,
        ),
        (
            "array-sparse-indices",
            changed(&dense, |p| p.sparse_indices.push(7)),
            true,
        ),
        (
            "array-other-payload",
            changed(&dense, |p| p.f64s.push(9.0)),
            true,
        ),
        (
            "array-unknown-field",
            changed(&dense, |p| {
                p.special_fields.mut_unknown_fields().add_varint(999, 7)
            }),
            true,
        ),
        (
            "array-missing-shape",
            changed(&dense, |p| p.shape.clear()),
            false,
        ),
        (
            "array-short-payload",
            changed(&dense, |p| {
                p.f32s.pop();
            }),
            false,
        ),
        (
            "nested-missing-shape",
            changed(&tuple, |p| p.tuple_literals[1].shape.clear()),
            true,
        ),
        (
            "nested-ignored-shape",
            changed(&tuple, |p| {
                p.tuple_literals[1].shape = Some(Shape::token().into_proto()).into()
            }),
            true,
        ),
        (
            "nested-invalid-ignored-shape",
            changed(&tuple, |p| {
                p.tuple_literals[1].shape = Some(Default::default()).into()
            }),
            true,
        ),
        (
            "nested-missing-element",
            changed(&tuple, |p| {
                p.tuple_literals[1].tuple_literals.pop();
            }),
            false,
        ),
        (
            "leaf-missing-shape",
            changed(&tuple, |p| p.tuple_literals[0].shape.clear()),
            false,
        ),
        (
            "leaf-missing-layout",
            changed(&tuple, |p| {
                p.tuple_literals[0].shape.as_mut().unwrap().layout.clear()
            }),
            false,
        ),
        (
            "leaf-layout-permutation",
            changed(&tuple, |p| {
                p.tuple_literals[0]
                    .shape
                    .as_mut()
                    .unwrap()
                    .layout
                    .as_mut()
                    .unwrap()
                    .minor_to_major = vec![0, 1]
            }),
            false,
        ),
        (
            "leaf-memory-space",
            changed(&tuple, |p| {
                p.tuple_literals[0]
                    .shape
                    .as_mut()
                    .unwrap()
                    .layout
                    .as_mut()
                    .unwrap()
                    .memory_space = 5
            }),
            false,
        ),
        (
            "leaf-dynamic-flags",
            changed(&tuple, |p| {
                p.tuple_literals[0]
                    .shape
                    .as_mut()
                    .unwrap()
                    .is_dynamic_dimension = vec![true, false]
            }),
            false,
        ),
        (
            "token-missing-child-shape",
            changed(&tuple, |p| {
                p.tuple_literals[1].tuple_literals[1].shape.clear()
            }),
            true,
        ),
        (
            "token-ignored-child-shape",
            changed(&tuple, |p| {
                p.tuple_literals[1].tuple_literals[1].shape = Some(Default::default()).into()
            }),
            true,
        ),
        (
            "token-extra-payload",
            changed(&token, |p| {
                p.f32s.push(9.0);
                p.tuple_literals.push(LiteralProto::new());
            }),
            true,
        ),
        (
            "opaque",
            changed(&token, |p| {
                p.shape = Some(Shape::opaque().into_proto()).into()
            }),
            false,
        ),
    ];
    let dynamic_shape = Shape::dynamic_array(P::S32, vec![xlahlo::Dimension::Bounded(3)])
        .unwrap()
        .with_default_layout()
        .unwrap();
    let dynamic = module(
        Literal::from_proto(LiteralProto {
            shape: Some(dynamic_shape.into_proto()).into(),
            s32s: vec![1, 2, 3],
            dynamic_sizes: vec![3],
            ..Default::default()
        })
        .unwrap(),
    );
    for (name, sizes, accepted) in [
        ("dynamic-missing-size", vec![], false),
        ("dynamic-negative-size", vec![-1], false),
        ("dynamic-empty", vec![0], true),
        ("dynamic-full", vec![3], true),
        ("dynamic-oversized", vec![4], false),
        ("dynamic-extra-size", vec![1, 2], false),
    ] {
        cases.push((
            name,
            changed(&dynamic, |p| p.dynamic_sizes = sizes),
            accepted,
        ));
    }
    for (name, wire, accepted) in cases {
        let original = root(&wire).literal.as_ref().unwrap();
        let rust =
            Module::from_proto(wire.clone()).and_then(|m| m.verify(Default::default()).map(|()| m));
        assert_eq!(rust.is_ok(), accepted, "{name}: {rust:?}");
        let output = native_bytes(&interchange::encode(&wire).unwrap(), "verify-proto");
        assert_eq!(
            output.status.code(),
            Some(if accepted { 0 } else { 2 }),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Ok(module) = rust {
            let imported = root(module.as_proto()).literal.as_ref().unwrap();
            assert_eq!(
                interchange::encode(imported).unwrap(),
                interchange::encode(original).unwrap(),
                "{name}"
            );
            let cpp: HloModuleProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            compare_literals(
                &Literal::from_proto(imported.clone()).unwrap(),
                &Literal::from_proto(root(&cpp).literal.as_ref().unwrap().clone()).unwrap(),
            );
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn common_integer_attribute_tokens_match_pinned_cpp() {
    for token in [
        "18446744073709551615",
        "18446744073709551614",
        "9223372036854775808",
        "18446744073709551616",
        "-9223372036854775808",
        "-9223372036854775809",
        "2147483647",
        "2147483648",
        "-2147483648",
        "-2147483649",
        "+1",
    ] {
        for source in [
            format!(
                "HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), metadata={{source_line={token}}} }}"
            ),
            format!(
                "HloModule m ENTRY main {{ p = f32[2] parameter(0) ROOT g = f32[2] all-gather(p), dimensions={{0}}, replica_groups={{{{0}}}}, channel_id={token} }}"
            ),
        ] {
            let rust = text::parse_module(&source);
            let output = native(&source, "verify");
            assert_eq!(
                output.status.code(),
                Some(if rust.is_ok() { 0 } else { 2 }),
                "{source}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            if let Ok(module) = rust {
                let cpp: HloModuleProto =
                    interchange::decode(&output.stdout, Default::default()).unwrap();
                let root = module
                    .instruction(module.root(module.entry().unwrap()).unwrap())
                    .unwrap();
                let entry = cpp
                    .computations
                    .iter()
                    .find(|c| c.id == cpp.entry_computation_id)
                    .unwrap();
                let other = entry
                    .instructions
                    .iter()
                    .find(|i| i.id == entry.root_id)
                    .unwrap();
                assert_eq!(root.channel_id, other.channel_id);
                assert_eq!(
                    root.metadata.get_or_default().source_line,
                    other.metadata.get_or_default().source_line
                );
                let printed = text::print_module(&module).unwrap();
                assert!(native(&printed, "verify").status.success(), "{printed}");
            }
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn parameter_replication_text_matches_pinned_cpp() {
    for (shape, replication) in [
        ("f32[]", "{true}"),
        ("token[]", "{false}"),
        ("()", "{}"),
        ("((),())", "{}"),
        ("(f32[],(s32[],()),token[])", "{true,false,true}"),
        ("b(f32[2])", "{false}"),
        ("f32[]", "{}"),
        ("f32[]", "{true,false}"),
        ("f32[]", "{1}"),
        ("f32[]", "{true,}"),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = {shape} parameter(0), parameter_replication={replication} }}"
        );
        let rust = text::parse_module(&source);
        let output = native(&source, "parse");
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Ok(module) = rust {
            let cpp: HloModuleProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            let actual = &cpp.computations[0].instructions[0]
                .parameter_replication
                .get_or_default()
                .replicated_at_leaf_buffers;
            let root = module.root(module.entry().unwrap()).unwrap();
            assert_eq!(
                &module
                    .instruction(root)
                    .unwrap()
                    .parameter_replication
                    .as_ref()
                    .unwrap()
                    .replicated_at_leaf_buffers,
                actual
            );
            let printed = text::print_module(&module).unwrap();
            assert!(native(&printed, "parse").status.success(), "{printed}");
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn statistics_grammar_and_double_storage_match_pinned_cpp() {
    for statistics in [
        "{}",
        "{ignored=17}",
        "{ignored=-3,a=1,a=-0.0,b=1e20}",
        "{ignored=18446744073709551615,a=18446744073709551615}",
        "{visualizing_index=999,a=-7.0,b=150.0}",
        "{visualizing_index=0,a=nan(0x1),b=-nan(0x8000000000001),c=nan,d=1e999,e=-1e999}",
        "{a=0,b=inf}",
        "{a=0,b=-inf}",
        "{a=0,b=.5}",
        "{a=0,b=+1}",
        "{a=0,b=18446744073709551616}",
        "{a=0,}",
        "{a=0 b=1}",
        "{a=0.5}",
        "{a=0,\"quoted\"=1}",
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), statistics={statistics} }}"
        );
        let output = native(&source, "verify");
        let rust = text::parse_module(&source);
        assert_eq!(
            output.status.code(),
            Some(if rust.is_ok() { 0 } else { 2 }),
            "{statistics}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Ok(module) = rust {
            let cpp: HloModuleProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            let view = |wire: &HloModuleProto| {
                let c = wire
                    .computations
                    .iter()
                    .find(|c| c.id == wire.entry_computation_id)
                    .unwrap();
                let s = c
                    .instructions
                    .iter()
                    .find(|i| i.id == c.root_id)
                    .unwrap()
                    .statistics_viz
                    .get_or_default();
                (
                    s.stat_index_to_visualize,
                    s.statistics
                        .iter()
                        .map(|s| (s.stat_name.clone(), s.stat_val.to_bits()))
                        .collect::<Vec<_>>(),
                )
            };
            assert_eq!(
                view(&module.to_proto().unwrap()),
                view(&cpp),
                "{statistics}"
            );
            let printed = text::print_module(&module).unwrap();
            let output = native(&printed, "verify");
            assert!(
                output.status.success(),
                "{printed}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let cpp: HloModuleProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(
                view(&module.to_proto().unwrap()),
                view(&cpp),
                "{statistics}"
            );
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn shared_async_metadata_and_original_values_match_pinned_cpp() {
    let source = "HloModule m work { p = f32[] parameter(0) ROOT n = f32[] negate(p) } ENTRY main { p = f32[] parameter(0) s = ((f32[]),f32[]) async-start(p), calls=work, metadata={op_name=\"start\"}, statistics={ignored=0,time=1} u = ((f32[]),f32[]) async-update(s), metadata={op_name=\"update\"} ROOT d = f32[] async-done(u), metadata={op_name=\"done\"}, origin={{\"last\"}}, statistics={ignored=1,time=nan(0x1),cost=-0.0} }";
    for source in [source.to_owned(), source.replace("op_name=\"done\"", "")] {
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let output = native(&source, "verify");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        let view = |wire: &HloModuleProto| {
            wire.computations
                .iter()
                .flat_map(|c| &c.instructions)
                .filter(|i| {
                    matches!(
                        i.opcode.as_str(),
                        "negate" | "async-start" | "async-update" | "async-done"
                    )
                })
                .map(|i| {
                    (
                        i.opcode.clone(),
                        i.metadata.get_or_default().clone(),
                        i.original_value.clone(),
                        (
                            i.statistics_viz.get_or_default().stat_index_to_visualize,
                            i.statistics_viz
                                .get_or_default()
                                .statistics
                                .iter()
                                .map(|s| (s.stat_name.clone(), s.stat_val.to_bits()))
                                .collect::<Vec<_>>(),
                        ),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(view(&module.to_proto().unwrap()), view(&cpp), "{source}");
        let printed = text::print_module(&module).unwrap();
        let output = native(&printed, "verify");
        assert!(
            output.status.success(),
            "{printed}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn recursive_unused_operand_cleanup_matches_pinned_cpp() {
    let base = include_str!("fixtures/recursive_removal.hlo");
    for source in [
        base.to_owned(),
        base.replace("copy(p)", "add(p,shared)"),
        base.replace("negate(p)", "negate(p), control-predecessors={p}"),
        base.replace("copy(p)", "copy(p), control-predecessors={shared}"),
        "HloModule m ENTRY main { p = f32[] parameter(0) q = f32[] parameter(1) dead = f32[] negate(q) ROOT r = f32[] copy(p) }".to_owned(),
        "HloModule m ENTRY main { p = f32[] parameter(0) e = f32[] custom-call(p), custom_call_target=\"effect\", custom_call_has_side_effect=true n = f32[] negate(e) dead = f32[] add(n,n) ROOT r = f32[] copy(p) }".to_owned(),
        "HloModule m ENTRY main { t = token[] after-all() e = (f32[],token[]) infeed(t) g = token[] get-tuple-element(e), index=1 dead = token[] after-all(g) ROOT r = token[] after-all() }".to_owned(),
        "HloModule m ENTRY main { t = token[] after-all() dead = token[] after-all(t) ROOT r = token[] after-all() }".to_owned(),
    ] {
        let mut rust = text::parse_module(&source).unwrap();
        let dead = rust.instructions(rust.entry().unwrap()).unwrap().into_iter()
            .find(|&id| rust.instruction(id).unwrap().name == "dead").unwrap();
        rust.remove_instruction_and_unused_operands(dead).unwrap();
        rust.verify(Default::default()).unwrap();
        let output = native(&source, "remove-unused-operands");
        assert!(output.status.success(), "{source}: {}", String::from_utf8_lossy(&output.stderr));
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        let view = |wire: &HloModuleProto| {
            let names = wire.computations.iter().flat_map(|c| &c.instructions)
                .map(|i| (i.id, i.name.clone())).collect::<std::collections::BTreeMap<_, _>>();
            wire.computations.iter().map(|c| {
                (c.name.clone(), (names[&c.root_id].clone(),
                    c.instructions.iter().map(|i| (
                        i.name.clone(), (i.opcode.clone(), i.shape.clone(),
                            i.operand_ids.iter().map(|id| names[id].clone()).collect::<Vec<_>>(),
                            i.control_predecessor_ids.iter().map(|id| names[id].clone()).collect::<Vec<_>>())
                    )).collect::<std::collections::BTreeMap<_, _>>()))
            }).collect::<std::collections::BTreeMap<_, _>>()
        };
        assert_eq!(view(&rust.to_proto().unwrap()), view(&cpp), "{source}");
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn optional_channels_verification_and_effects_match_pinned_cpp() {
    let reducer = "sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT r = f32[] add(a,b) }";
    let cases = [
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT c = f32[2] collective-permute(p), source_target_pairs={{0,0}}CHANNEL }".to_owned(),
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT g = f32[2] all-gather(p), dimensions={0}, replica_groups={{0}}CHANNEL }".to_owned(),
        format!("HloModule m {reducer} ENTRY main {{ p = f32[2] parameter(0) ROOT r = f32[2] all-reduce(p), to_apply=sum, replica_groups={{{{0}}}}CHANNEL }}"),
        format!("HloModule m {reducer} ENTRY main {{ p = f32[2] parameter(0) s = f32[2] all-reduce-start(p), to_apply=sum, replica_groups={{{{0}}}}CHANNEL ROOT d = f32[2] all-reduce-done(s) }}"),
        "HloModule m ENTRY main { p = f32[2] parameter(0) t = token[] after-all() s = () send(p,t)CHANNEL control = () custom-call(s), custom_call_target=\"control_dep\" ROOT d = token[] send-done(s)CHANNEL }".to_owned(),
        "HloModule m body { p = f32[2] parameter(0) c = f32[2] collective-permute(p), source_target_pairs={{0,0}}CHANNEL ROOT r = f32[2] copy(p) } ENTRY main { x = f32[2] parameter(0) ROOT f = f32[2] fusion(x), kind=kLoop, calls=body }".to_owned(),
        "HloModule m ENTRY main { a = f32[2] parameter(0) b = f32[3] parameter(1) t = token[] after-all() s0 = () send(a,t), is_host_transfer=trueCHANNEL send1 = () send(b,t), is_host_transfer=trueCHANNEL d0 = token[] send-done(s0), is_host_transfer=trueCHANNEL d1 = token[] send-done(send1), is_host_transfer=trueCHANNEL ROOT r = (token[],token[]) tuple(d0,d1) }".to_owned(),
    ];
    for base in cases {
        for channel in [None, Some(-1), Some(0), Some(3)] {
            let attribute = channel.map_or_else(String::new, |id| format!(", channel_id={id}"));
            let source = base.replace("CHANNEL", &attribute);
            let module = text::parse_module(&source).unwrap();
            let output = native(&source, "verify");
            assert_eq!(
                output.status.code(),
                Some(if module.verify(Default::default()).is_ok() {
                    0
                } else {
                    3
                }),
                "{source}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let output = native(&source, "effects");
            assert!(
                output.status.success(),
                "{source}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let effect = module
                .instruction_has_side_effect(
                    module.root(module.entry().unwrap()).unwrap(),
                    Default::default(),
                )
                .unwrap();
            assert_eq!(
                String::from_utf8(output.stdout).unwrap().trim(),
                if effect { "1" } else { "0" },
                "{source}"
            );
            let printed = text::print_module(&module).unwrap();
            let output = native(&printed, "verify");
            assert_eq!(
                output.status.code(),
                Some(if module.verify(Default::default()).is_ok() {
                    0
                } else {
                    3
                }),
                "{printed}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn coordinated_parameter_removal_matches_pinned_cpp() {
    for (source, entry) in [
        (
            "HloModule m body { a = f32[] parameter(0) ROOT b = f32[] parameter(1), metadata={op_name=\"kept\"} } ENTRY main { x = f32[] parameter(0) y = f32[] parameter(1) ROOT c = f32[] call(x,y), to_apply=body }",
            false,
        ),
        (
            "HloModule m body { a = f32[] parameter(0) ROOT b = f32[] parameter(1) } ENTRY main { x = f32[] parameter(0) y = f32[] parameter(1) ROOT c = f32[] fusion(x,y), kind=kLoop, calls=body }",
            false,
        ),
        (
            "HloModule m body { a = f32[] parameter(0) ROOT b = f32[] parameter(1) } ENTRY main { x = f32[] parameter(0) y = f32[] parameter(1) s = ((f32[],f32[]),f32[],s32[]) async-start(x,y), calls=body ROOT d = f32[] async-done(s) }",
            false,
        ),
        (
            "HloModule m ENTRY main { a = f32[] parameter(0) ROOT b = f32[] parameter(1) }",
            true,
        ),
        (
            "HloModule m body { a = f32[] parameter(0) b = f32[] parameter(1) c = f32[] parameter(2) ROOT n = f32[] add(a,c) } ENTRY main { x = f32[] parameter(0) y = f32[] parameter(1) z = f32[] parameter(2) ROOT c1 = f32[] call(x,y,z), to_apply=body }",
            false,
        ),
        (
            "HloModule m body { a = f32[] parameter(0) ROOT b = f32[] parameter(1) } ENTRY main { x = f32[] parameter(0) y = f32[] parameter(1) c1 = f32[] call(x,y), to_apply=body c2 = f32[] call(y,y), to_apply=body ROOT r = (f32[],f32[]) tuple(c1,c2) }",
            false,
        ),
    ] {
        let mut rust = text::parse_module(source).unwrap();
        let owner = rust.computations().next().unwrap();
        let parameter = rust
            .instructions(owner)
            .unwrap()
            .into_iter()
            .find(|&id| {
                let i = rust.instruction(id).unwrap();
                i.opcode == "parameter"
                    && rust.users(id).unwrap().is_empty()
                    && rust.root(owner).unwrap() != id
            })
            .unwrap();
        rust.remove_unused_parameter(parameter, entry).unwrap();
        rust.verify(Default::default()).unwrap();
        let output = native(
            source,
            if entry {
                "remove-entry-parameter"
            } else {
                "remove-parameter"
            },
        );
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        xlahlo::Module::from_proto(cpp.clone())
            .unwrap()
            .verify(Default::default())
            .unwrap();
        let view = |wire: &HloModuleProto| {
            let names = wire
                .computations
                .iter()
                .flat_map(|c| &c.instructions)
                .map(|i| (i.id, i.name.clone()))
                .collect::<std::collections::BTreeMap<_, _>>();
            wire.computations
                .iter()
                .map(|c| {
                    let instructions = c
                        .instructions
                        .iter()
                        .map(|i| {
                            (
                                i.name.clone(),
                                (
                                    i.opcode.clone(),
                                    i.parameter_number,
                                    i.operand_ids
                                        .iter()
                                        .map(|id| names[id].clone())
                                        .collect::<Vec<_>>(),
                                    i.shape.clone(),
                                    i.metadata.as_ref().cloned().unwrap_or_default(),
                                ),
                            )
                        })
                        .collect::<std::collections::BTreeMap<_, _>>();
                    (c.name.clone(), (names[&c.root_id].clone(), instructions))
                })
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        assert_eq!(view(&rust.to_proto().unwrap()), view(&cpp), "{source}");
    }
    let source = "HloModule m body { a = f32[] parameter(0) ROOT b = f32[] constant(0) } ENTRY main { x = f32[2] parameter(0) ROOT r = f32[2] map(x), to_apply=body }";
    let output = native(source, "remove-parameter");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn binary_import_and_default_upgrades_match_pinned_cpp() {
    for source in [
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT n = f32[2] negate(p) }",
        "HloModule m body { ROOT p = f32[] parameter(0) } ENTRY main { p = f32[2,3] parameter(0) ROOT r = f32[2,3] map(p), to_apply=body }",
        "HloModule m ENTRY main { p = f32[] parameter(0) ROOT r = pred[] compare(p,p), direction=EQ, type=FLOAT }",
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0), metadata={metadata_payload=\"bytes\"} }",
        "HloModule m body { ROOT p = f32[] parameter(0) } ENTRY main { p = f32[] parameter(0) ROOT c = f32[] call(p), to_apply=body }",
        "HloModule m, entry_computation_layout={(f32[], f32[])->f32[]} ENTRY main { a = f32[] parameter(0) b = f32[] parameter(1) ROOT r = f32[] add(a,b) }",
    ] {
        let module = text::parse_module(source).unwrap();
        let mut wire = module.to_proto_with_program_shapes().unwrap();
        if root(&wire).opcode == "call" {
            wire.computations
                .iter_mut()
                .find(|c| c.id == wire.entry_computation_id)
                .unwrap()
                .instructions
                .iter_mut()
                .find(|i| i.opcode == "call")
                .unwrap()
                .output_operand_aliasing
                .push(Default::default());
        }
        let encoded = interchange::encode(&wire).unwrap();
        let output = native_bytes(&encoded, "verify-proto");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        xlahlo::Module::from_proto(cpp.clone())
            .unwrap()
            .verify(Default::default())
            .unwrap();
        assert_eq!(root(&wire).opcode, root(&cpp).opcode);
        assert_eq!(root(&wire).dimensions, root(&cpp).dimensions);
        assert_eq!(root(&wire).comparison_order, root(&cpp).comparison_order);
        if root(&wire).opcode == "call" {
            // The pinned call serializer omits aliases even though import and
            // verification use them. Rust raw export deliberately retains them.
            assert!(root(&cpp).output_operand_aliasing.is_empty());
            let mut invalid = wire.clone();
            invalid
                .computations
                .iter_mut()
                .find(|c| c.id == invalid.entry_computation_id)
                .unwrap()
                .instructions
                .iter_mut()
                .find(|i| i.opcode == "call")
                .unwrap()
                .output_operand_aliasing[0]
                .operand_index = 99;
            let output = native_bytes(&interchange::encode(&invalid).unwrap(), "verify-proto");
            assert_eq!(
                output.status.code(),
                Some(3),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                xlahlo::Module::from_proto(invalid)
                    .unwrap()
                    .verify(Default::default())
                    .is_err()
            );
        } else {
            assert_eq!(
                root(&wire).output_operand_aliasing,
                root(&cpp).output_operand_aliasing
            );
        }
        if root(&wire)
            .metadata
            .as_ref()
            .is_some_and(|m| m.metadata_payload.is_some())
        {
            assert_eq!(cpp.payloads, [b"bytes".to_vec()]);
            assert_eq!(
                root(&cpp)
                    .metadata
                    .as_ref()
                    .unwrap()
                    .metadata_payload
                    .as_ref()
                    .unwrap()
                    .id(),
                0
            );
            for id in [-1, 0, 1, i64::MAX] {
                let mut indexed = wire.clone();
                indexed.payloads = vec![b"bytes".to_vec()];
                indexed.computations[0].instructions[0]
                    .metadata
                    .as_mut()
                    .unwrap()
                    .metadata_payload
                    .as_mut()
                    .unwrap()
                    .set_id(id);
                let output = native_bytes(&interchange::encode(&indexed).unwrap(), "import-proto");
                assert_eq!(
                    output.status.code(),
                    Some(if id == 0 { 0 } else { 2 }),
                    "payload {id}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(xlahlo::Module::from_proto(indexed).is_ok(), id == 0);
            }
        }
        if root(&wire).opcode == "compare" {
            let mut legacy = wire.clone();
            let root = legacy
                .computations
                .iter_mut()
                .flat_map(|c| &mut c.instructions)
                .find(|i| i.opcode == "compare")
                .unwrap();
            root.comparison_order.clear();
            root.comparison_type = "FLOAT".into();
            let output = native_bytes(&interchange::encode(&legacy).unwrap(), "verify-proto");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let cpp: HloModuleProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(self::root(&cpp).comparison_order, "PARTIAL");
            assert!(self::root(&cpp).comparison_type.is_empty());
        }
        for source in [
            "HloModule m a { ROOT p = f32[] parameter(0) } b { ROOT p = f32[] parameter(0) } ENTRY main { ROOT p = f32[] parameter(0) }",
            "HloModule m a { ROOT p.01 = f32[] parameter(0) } b { ROOT p = f32[] parameter(0) } c { ROOT p.0 = f32[] parameter(0) } ENTRY main { ROOT p.-1 = f32[] parameter(0) }",
        ] {
            let rust = text::parse_module(source)
                .unwrap()
                .to_proto_with_program_shapes()
                .unwrap();
            let output = native(source, "verify");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let cpp: HloModuleProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            let names = |wire: &HloModuleProto| {
                wire.computations
                    .iter()
                    .map(|c| {
                        (
                            c.name.clone(),
                            c.instructions
                                .iter()
                                .map(|i| i.name.clone())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<std::collections::BTreeMap<_, _>>()
            };
            assert_eq!(names(&rust), names(&cpp));
            let output = native_bytes(&interchange::encode(&rust).unwrap(), "verify-proto");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let module =
            text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0) }").unwrap();
        let output = native_bytes(
            &interchange::encode(&module.to_proto().unwrap()).unwrap(),
            "import-proto",
        );
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("program shape"));
        let mut module = xlahlo::Module::new("constructed");
        let main = module.add_computation("main").unwrap();
        let shape = Shape::array(xlahlo::PrimitiveType::F32, vec![2]).unwrap();
        let parameter = module
            .add_instruction(
                main,
                xlahlo::Instruction::new("f32", xlahlo::Opcode::Parameter, shape.clone()),
                &[],
                &[],
            )
            .unwrap();
        let result = module
            .add_instruction(
                main,
                xlahlo::Instruction::new("__reserved", xlahlo::Opcode::Negate, shape),
                &[parameter],
                &[],
            )
            .unwrap();
        module.set_root(main, result).unwrap();
        module.set_entry(main).unwrap();
        module.verify(Default::default()).unwrap();
        let output = native_bytes(
            &interchange::encode(&module.to_proto_with_program_shapes().unwrap()).unwrap(),
            "verify-proto",
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn stack_frame_parsing_and_canonicalization_match_pinned_cpp() {
    let source = include_str!("fixtures/stack_frames.hlo");
    for source in [
        source.to_owned(),
        source.replace("stack_frame_id=3", "stack_frame_id=99"),
        source.replace("file_location_id=1", "file_location_id=-1"),
        source.replace("file_name_id=2", "file_name_id=0"),
        source.replace("function_name_id=1", "function_name_id=99"),
        source.replace(
            "file_location_id=1 parent_frame_id=1",
            "file_location_id=1 parent_frame_id=3",
        ),
        source
            .replace("stack_frame_id=3", "stack_frame_id=0")
            .replace("stack_frame_id=2", "stack_frame_id=0"),
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0), metadata={stack_frame_id=-3} }"
            .into(),
    ] {
        let rust = text::parse_module(&source).unwrap();
        for source in [source, text::print_module(&rust).unwrap()] {
            let output = native(&source, "parse");
            assert!(
                output.status.success(),
                "{source}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let cpp: HloModuleProto =
                interchange::decode(&output.stdout, Default::default()).unwrap();
            assert_eq!(
                rust.as_proto().stack_frame_index,
                cpp.stack_frame_index,
                "{source}"
            );
            assert_eq!(
                root(rust.as_proto())
                    .metadata
                    .as_ref()
                    .unwrap()
                    .stack_frame_id,
                root(&cpp).metadata.as_ref().unwrap().stack_frame_id,
                "{source}"
            );
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn constructor_defaults_and_domain_sides_match_pinned_cpp_wire() {
    let compare = |source: &str| {
        let module = text::parse_module(source).unwrap();
        let expected = if module.verify(Default::default()).is_ok() {
            0
        } else {
            3
        };
        for source in [source.to_owned(), text::print_module(&module).unwrap()] {
            let cpp = native(&source, "verify");
            assert_eq!(
                cpp.status.code(),
                Some(expected),
                "{source}: {}",
                String::from_utf8_lossy(&cpp.stderr)
            );
            let cpp = native(&source, "parse");
            assert!(
                cpp.status.success(),
                "{source}: {}",
                String::from_utf8_lossy(&cpp.stderr)
            );
            let cpp: HloModuleProto = interchange::decode(&cpp.stdout, Default::default()).unwrap();
            let (rust, cpp) = (root(module.as_proto()), root(&cpp));
            assert_eq!(rust.comparison_order, cpp.comparison_order, "{source}");
            assert_eq!(rust.comparison_type, cpp.comparison_type, "{source}");
            assert_eq!(rust.dimensions, cpp.dimensions, "{source}");
            assert_eq!(rust.sharding, cpp.sharding, "{source}");
            assert_eq!(
                rust.domain_entry_sharding, cpp.domain_entry_sharding,
                "{source}"
            );
            assert_eq!(
                rust.domain_exit_sharding, cpp.domain_exit_sharding,
                "{source}"
            );
        }
    };
    for ty in ["pred", "s4", "u32", "f32", "c64"] {
        for attribute in [
            "",
            ", order=TOTAL",
            ", order=WEAK",
            ", type=FLOAT",
            ", type=UNSIGNED",
        ] {
            compare(&format!(
                "HloModule m ENTRY main {{ p = {ty}[] parameter(0) ROOT c = pred[] compare(p,p), direction=EQ{attribute} }}"
            ));
        }
    }
    for dimensions in ["", ", dimensions={}", ", dimensions={1}"] {
        compare(&format!(
            "HloModule m body {{ ROOT p = f32[] parameter(0) }} ENTRY main {{ p = f32[2,3] parameter(0) ROOT r = f32[2,3] map(p), to_apply=body{dimensions} }}"
        ));
    }
    compare(
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = f32[2] domain(p), domain={kind=\"sharding\", entry={manual}, exit={mesh['x'=2], [{'x'}]}} }",
    );
    for sharding in [
        "{replicated}",
        "{devices=[2]0,1}",
        "{manual shard_like 7}",
        "{manual shard_as 7}",
    ] {
        compare(&format!(
            "HloModule m ENTRY main {{ ROOT p = f32[2] parameter(0), sharding={sharding} }}"
        ));
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn parser_names_attributes_and_async_bodies_match_pinned_cpp() {
    let compare = |source: &str, expected| {
        let rust = match text::parse_module(source) {
            Err(_) => 2,
            Ok(module) => {
                if module.verify(Default::default()).is_ok() {
                    0
                } else {
                    3
                }
            }
        };
        assert_eq!(rust, expected, "{source}");
        let cpp = native(source, "verify");
        assert_eq!(
            cpp.status.code(),
            Some(expected),
            "{source}: {}",
            String::from_utf8_lossy(&cpp.stderr)
        );
        if expected == 0 {
            let printed = text::print_module(&text::parse_module(source).unwrap()).unwrap();
            let cpp = native(&printed, "verify");
            assert!(
                cpp.status.success(),
                "{printed}: {}",
                String::from_utf8_lossy(&cpp.stderr)
            );
        }
    };
    for name in [
        "pred",
        "token",
        "s1",
        "nan",
        "true",
        "inf",
        "ROOT",
        "maximal",
        "FileNames",
    ] {
        compare(
            &format!("HloModule m ENTRY main {{ ROOT {name} = f32[] parameter(0) }}"),
            2,
        );
        compare(
            &format!("HloModule %{name} ENTRY %{name} {{ ROOT %{name} = f32[] parameter(0) }}"),
            if matches!(name, "pred" | "token" | "s1") {
                2
            } else {
                0
            },
        );
    }
    for (attribute, expected) in [
        ("dimensions={0}", 0),
        ("dimension=0", 2),
        ("dimensions={}", 2),
        ("dimensions={0,1}", 2),
        ("", 2),
    ] {
        let suffix = if attribute.is_empty() {
            String::new()
        } else {
            format!(", {attribute}")
        };
        compare(
            &format!(
                "HloModule m ENTRY main {{ p = f32[3] parameter(0) ROOT s = s32[] get-dimension-size(p){suffix} }}"
            ),
            expected,
        );
        compare(
            &format!(
                "HloModule m ENTRY main {{ p = f32[3] parameter(0) s = s32[] parameter(1) ROOT d = f32[<=3] set-dimension-size(p,s){suffix} }}"
            ),
            expected,
        );
    }
    compare(
        "HloModule m body { ROOT p = f32[] parameter(0) } ENTRY main { p = f32[] parameter(0) ROOT r = f32[] call(p), to_apply=body, output_to_operand_aliasing={{}: (0,{})} }",
        2,
    );
    let source = "HloModule m work { p = f32[] parameter(0) n = f32[] negate(p) ROOT r = f32[] negate(n) } ENTRY main { p = f32[] parameter(0) s = ((f32[]),f32[]) async-start(p), calls=work ROOT d = f32[] async-done(s) }";
    compare(source, 3);
    compare(
        &source.replace(
            "calls=work",
            "calls=work, async_execution_thread=\"worker\"",
        ),
        0,
    );
    compare(
        &source.replace(
            "calls=work",
            "calls=work, frontend_attributes={_collectives_group=\"false\"}",
        ),
        0,
    );
    let barrier = "HloModule m work { p = (f32[]) parameter(0) g = f32[] get-tuple-element(p), index=0 ROOT r = f32[] negate(g) } ENTRY main { p = (f32[]) parameter(0) s = (((f32[])),f32[]) async-start(p), calls=work ROOT d = f32[] async-done(s) }";
    compare(barrier, 0);
}

fn compare_core_inference(source: &str, rust: xlahlo::Result<Shape>) {
    text::parse_module(source).unwrap();
    let output = native(source, "infer-core");
    assert_eq!(
        output.status.code(),
        Some(if rust.is_ok() { 0 } else { 4 }),
        "{source}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    if let Ok(rust) = rust {
        let cpp: ShapeProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        assert!(
            rust.equal_including_layout(&Shape::from_proto(cpp).unwrap()),
            "{source}"
        );
    }
}

fn root(module: &HloModuleProto) -> &HloInstructionProto {
    let entry = module
        .computations
        .iter()
        .find(|computation| computation.id == module.entry_computation_id)
        .unwrap();
    entry
        .instructions
        .iter()
        .find(|instruction| instruction.id == entry.root_id)
        .unwrap()
}

fn compare_literals(left: &Literal, right: &Literal) {
    assert!(left.shape().equal_including_layout(right.shape()));
    if left.shape().is_tuple() {
        assert_eq!(
            left.as_proto().tuple_literals.len(),
            right.as_proto().tuple_literals.len()
        );
        for index in 0..left.as_proto().tuple_literals.len() {
            compare_literals(
                &left.sub_literal(&[index]).unwrap(),
                &right.sub_literal(&[index]).unwrap(),
            );
        }
    } else if left.shape().element_type() != xlahlo::PrimitiveType::TOKEN {
        assert_eq!(left.bits().unwrap(), right.bits().unwrap());
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn scalar_tuple_and_nan_literals_match_pinned_cpp_parser() {
    for (shape, value) in [
        ("pred[]", "2"),
        ("u64[]", "-1"),
        ("s64[]", "18446744073709551615"),
        ("f16[]", "1.0004882812500002"),
        ("bf16[]", "1.0039062500000002"),
        ("f32[]", "-nan(0x400007)"),
        ("f64[]", "nan(0x1)"),
        ("c64[]", "(nan(0x400007),3)"),
        ("c128[]", "(-nan(0x8000000000007),5)"),
        ("c64[]", "(2,nan(0x8000020000000))"),
        ("c128[]", "(2,nan(0x1))"),
        ("f8e3m4[]", "1.5"),
        ("f8e4m3[]", "1.0625000000000002"),
        ("f8e5m2[]", "nan(0x1)"),
        ("f8e4m3fn[]", "-nan"),
        ("f8e4m3fnuz[]", "-0.0"),
        ("f8e4m3b11fnuz[]", "inf"),
        ("f8e5m2fnuz[]", "nan"),
        ("f8e8m0fnu[]", "1.5"),
        ("f4e2m1fn[]", "nan"),
        ("(f32[],(s32[2],pred[]))", "(1.5,({2,3},false))"),
    ] {
        let source = format!("HloModule m ENTRY main {{ ROOT c = {shape} constant({value}) }}");
        let rust = text::parse_module(&source).unwrap();
        let output = native(&source, "parse");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        compare_literals(
            &Literal::from_proto(root(rust.as_proto()).literal.as_ref().unwrap().clone()).unwrap(),
            &Literal::from_proto(root(&cpp).literal.as_ref().unwrap().clone()).unwrap(),
        );
        let printed = text::print_module(&rust).unwrap();
        let output = native(&printed, "parse");
        assert!(
            output.status.success(),
            "{printed}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let printed: HloModuleProto =
            interchange::decode(&output.stdout, Default::default()).unwrap();
        compare_literals(
            &Literal::from_proto(root(&cpp).literal.as_ref().unwrap().clone()).unwrap(),
            &Literal::from_proto(root(&printed).literal.as_ref().unwrap().clone()).unwrap(),
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn six_bit_literals_match_native_storage_without_using_broken_cpp_serialization() {
    for ty in ["f6e2m3fn", "f6e3m2fn"] {
        for value in [
            "0.0", "-0.0", "1.5", "1e300", "-1e300", "nan", "inf", "-inf",
        ] {
            let source = format!("HloModule m ENTRY main {{ ROOT c = {ty}[] constant({value}) }}");
            let module = text::parse_module(&source).unwrap();
            let literal =
                Literal::from_proto(root(module.as_proto()).literal.as_ref().unwrap().clone())
                    .unwrap();
            let expected = [u8::try_from(literal.bits().unwrap()[0]).unwrap()];
            for source in [source, text::print_module(&module).unwrap()] {
                let output = native(&source, "scalar-bytes");
                assert!(
                    output.status.success(),
                    "{source}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(output.stdout, expected, "{source}");
            }
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn numeric_rejections_match_pinned_cpp_parser() {
    for (shape, value) in [
        ("s8[]", "128"),
        ("u8[]", "-1"),
        ("f16[]", "65520"),
        ("bf16[]", "nan(0x80)"),
        ("f32[]", "1e40"),
        ("f64[]", "1e400"),
        ("f64[]", "nan(0x0)"),
        ("f8e4m3fn[]", "465"),
        ("f8e8m0fnu[]", "0"),
        ("c64[]", "(3.4028235e38,0)"),
    ] {
        let source = format!("HloModule m ENTRY main {{ ROOT c = {shape} constant({value}) }}");
        assert!(text::parse_module(&source).is_err(), "{source}");
        let output = native(&source, "parse");
        assert_eq!(
            output.status.code(),
            Some(2),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn basic_shape_verification_matches_pinned_cpp_verifier() {
    for (result_shape, valid) in [("f32[2]", true), ("f32[3]", false)] {
        let source = format!(
            "HloModule m ENTRY main {{ p = f32[2] parameter(0) ROOT n = {result_shape} negate(p) }}"
        );
        let rust = text::parse_module(&source).unwrap();
        assert_eq!(rust.verify(Default::default()).is_ok(), valid);
        let output = native(&source, "verify");
        assert_eq!(
            output.status.code(),
            Some(if valid { 0 } else { 3 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn bounded_reshape_inference_and_verification_match_pinned_cpp() {
    for (input, result, infers, verifies, parses) in [
        ("f32[<=6]", "f32[2,3]", true, true, true),
        ("f32[2,<=3]", "f32[6]", true, true, true),
        ("f32[<=2,<=3]", "f32[6]", true, true, true),
        ("f32[<=1]", "f32[1,1]", true, true, true),
        ("f32[<=1]", "f32[]", true, true, true),
        ("f32[<=0]", "f32[0]", true, true, true),
        ("f32[2,<=3]", "f32[3,2]", false, true, true),
        ("f32[2,<=1,3]", "f32[2,1,3]", false, true, true),
        ("f32[<=6]", "f32[7]", false, false, false),
        ("b(f32[<=6])", "f32[2,3]", false, false, true),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ p = {input} parameter(0) ROOT r = {result} reshape(p) }}"
        );
        let input = text::parse_shape(input).unwrap();
        let result = text::parse_shape(result).unwrap();
        let rust = inference::infer_reshape(&input, result.dimensions(), None);
        assert_eq!(rust.is_ok(), infers, "{source}");
        let output = native(&source, "infer-reshape");
        assert_eq!(
            output.status.code(),
            Some(if !parses {
                2
            } else if infers {
                0
            } else {
                4
            }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Ok(rust) = rust {
            let cpp: ShapeProto = interchange::decode(&output.stdout, Default::default()).unwrap();
            assert!(
                rust.equal_including_layout(&Shape::from_proto(cpp).unwrap()),
                "{source}"
            );
        }

        let rust = text::parse_module(&source);
        assert_eq!(rust.is_ok(), parses, "{source}");
        if let Ok(rust) = rust {
            assert_eq!(
                rust.verify(Default::default()).is_ok(),
                verifies,
                "{source}"
            );
        }
        let output = native(&source, "verify");
        assert_eq!(
            output.status.code(),
            Some(if !parses {
                2
            } else if verifies {
                0
            } else {
                3
            }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn core_inference_buffer_entry_guards_match_pinned_cpp() {
    for input in ["f32[2,3]", "b(f32[2,3])"] {
        let shape = text::parse_shape(input).unwrap();
        let size = text::parse_shape("s32[]").unwrap();
        let update = text::parse_shape("f32[1,2]").unwrap();
        let cases = [
            (
                "f32[3,2]",
                "transpose(p), dimensions={1,0}",
                inference::infer_transpose(&shape, &[1, 0]),
            ),
            (
                input,
                "reduce-precision(p), exponent_bits=8, mantissa_bits=10",
                inference::infer_reduce_precision(&shape, 8, 10),
            ),
            (
                "s32[]",
                "get-dimension-size(p), dimensions={1}",
                inference::infer_get_dimension_size(&shape, 1),
            ),
            (
                input,
                "set-dimension-size(p,s), dimensions={0}",
                inference::infer_set_dimension_size(&shape, &size, 0),
            ),
            (
                input,
                "dynamic-update-slice(p,q,s,s)",
                inference::infer_dynamic_update_slice(
                    &shape,
                    &update,
                    &[size.clone(), size.clone()],
                ),
            ),
            (
                "f32[2,3]",
                "dynamic-reshape(p,s,s)",
                inference::infer_dynamic_reshape(
                    &shape,
                    &[size.clone(), size.clone()],
                    vec![xlahlo::Dimension::Static(2), xlahlo::Dimension::Static(3)],
                ),
            ),
        ];
        for (result, operation, rust) in cases {
            let source = format!(
                "HloModule m ENTRY main {{ p = {input} parameter(0) s = s32[] parameter(1) q = f32[1,2] parameter(2) ROOT r = {result} {operation} }}"
            );
            compare_core_inference(&source, rust);
        }
    }
    for input in [
        "f32[8,16]{1,0:T(4,8)SC(0:1,2)(1:1)}",
        "f32[<=8,16]{1,0:T(4,8)SC(0:1,2)(1:1)}",
    ] {
        let shape = text::parse_shape(input).unwrap();
        compare_core_inference(
            &format!(
                "HloModule m ENTRY main {{ p = {input} parameter(0) ROOT r = f32[16,8] transpose(p), dimensions={{1,0}} }}"
            ),
            inference::infer_transpose(&shape, &[1, 0]),
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn matrix_and_complex_fft_buffer_inference_match_pinned_cpp() {
    use xlahlo::proto::xla_data::{
        FftType, TriangularSolveOptions, triangular_solve_options::Transpose,
    };
    let options = TriangularSolveOptions {
        left_side: true,
        transpose_a: Transpose::NO_TRANSPOSE.into(),
        ..Default::default()
    };
    for input in ["f32[3,3]", "b(f32[3,3])"] {
        let a = text::parse_shape(input).unwrap();
        compare_core_inference(
            &format!(
                "HloModule m ENTRY main {{ p = {input} parameter(0) ROOT r = {input} cholesky(p), lower=true }}"
            ),
            inference::infer_cholesky(&a),
        );
        for rhs in ["f32[3,4]", "b(f32[3,4])"] {
            let b = text::parse_shape(rhs).unwrap();
            compare_core_inference(
                &format!(
                    "HloModule m ENTRY main {{ p = {input} parameter(0) q = {rhs} parameter(1) ROOT r = {rhs} triangular-solve(p,q), left_side=true, lower=true, unit_diagonal=false, transpose_a=NO_TRANSPOSE }}"
                ),
                inference::infer_triangular_solve(&a, &b, &options),
            );
        }
    }
    for input in ["c64[2,8]", "b(c64[2,8])"] {
        for (name, kind) in [("FFT", FftType::FFT), ("IFFT", FftType::IFFT)] {
            compare_core_inference(
                &format!(
                    "HloModule m ENTRY main {{ p = {input} parameter(0) ROOT r = {input} fft(p), fft_type={name}, fft_length={{8}} }}"
                ),
                inference::infer_fft(&text::parse_shape(input).unwrap(), kind, &[8]),
            );
        }
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn original_value_metadata_matches_pinned_cpp_parser_and_verifier() {
    for (shape, original, verifies) in [
        ("f32[]", r#"{{"unknown" {2,1}}}"#, true),
        ("f32[]", "{{}}", true),
        ("f32[]", "{()}", false),
        ("(f32[],s32[])", r#"{({"a"}, {})}"#, true),
        ("((),f32[])", r#"{((), {"a"})}"#, true),
        ("f32[]", "{[synthetic_call]}", true),
        ("f32[]", r#"{{"discarded"},[""]}"#, true),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = {shape} parameter(0), origin={original} }}"
        );
        let rust = text::parse_module(&source).unwrap();
        let output = native(&source, "parse");
        assert!(
            output.status.success(),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cpp: HloModuleProto = interchange::decode(&output.stdout, Default::default()).unwrap();
        assert_eq!(
            root(rust.as_proto()).original_value,
            root(&cpp).original_value,
            "{source}"
        );
        assert_eq!(
            rust.verify(Default::default()).is_ok(),
            verifies,
            "{source}"
        );
        let output = native(&source, "verify");
        assert_eq!(
            output.status.code(),
            Some(if verifies { 0 } else { 3 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires XLAHLO_HLO_ORACLE built with tools/hlo_oracle.py"]
fn optional_scheduled_deadlock_policy_matches_pinned_cpp() {
    for (send_pairs, recv_pairs, interruption, valid) in [
        ("{{0,1}}", "{{0,1}}", "", true),
        ("{{0,1}}", "{{2,3}}", "", false),
        ("{{0,1}}", "{{0, 1}}", "", false),
        ("{{0,1},{1,0}}", "{{0,1},{1,0}}", "", false),
        ("[2,2]<=[4]", "[2,2]<=[4]", "", true),
        (
            "mesh[\\\"x\\\"=2] {\\\"x\\\"}",
            "mesh[\\\"x\\\"=2] {\\\"x\\\"}",
            "",
            true,
        ),
        (
            "{{0,1}}",
            "{{0,1}}",
            "g = f32[1] all-gather(p), dimensions={0}, replica_groups={}",
            false,
        ),
    ] {
        let source = format!(
            "HloModule m, is_scheduled=true ENTRY main {{ p = f32[1] parameter(0) t = token[] after-all() s = () send(p,t), channel_id=1, frontend_attributes={{_xla_send_recv_source_target_pairs=\"{send_pairs}\"}} sd = token[] send-done(s), channel_id=1 {interruption} r = (f32[1]) recv(t), channel_id=2, frontend_attributes={{_xla_send_recv_source_target_pairs=\"{recv_pairs}\"}} rd = (f32[1],token[]) recv-done(r), channel_id=2 ROOT result = f32[1] copy(p) }}"
        );
        let rust = text::parse_module(&source).unwrap();
        assert_eq!(
            rust.verify(xlahlo::VerifyOptions {
                check_for_collective_deadlocks: true,
                ..Default::default()
            })
            .is_ok(),
            valid,
            "{source}"
        );
        let output = native(&source, "verify-deadlocks");
        assert_eq!(
            output.status.code(),
            Some(if valid { 0 } else { 3 }),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
