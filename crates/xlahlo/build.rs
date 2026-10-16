use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("xlahlo must remain under the workspace crates directory");
    println!("cargo:rerun-if-env-changed=XLA_SOURCE_DIR");
    let xla_root = env::var_os("XLA_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("extern").join("xla"));
    let xla_root = xla_root.canonicalize().unwrap_or_else(|error| {
        panic!(
            "cannot resolve OpenXLA source at {}: {error}",
            xla_root.display()
        )
    });
    if xla_root.file_name().is_none_or(|name| name != "xla")
        || xla_root
            .parent()
            .and_then(Path::file_name)
            .is_none_or(|name| name != "extern")
    {
        panic!("XLA_SOURCE_DIR must point to the extern/xla submodule");
    }
    if !xla_root.join("xla").is_dir() {
        panic!(
            "OpenXLA source is missing at {}; initialize the extern/xla submodule",
            xla_root.display()
        );
    }

    let opcode_path = xla_root
        .join("xla")
        .join("hlo")
        .join("ir")
        .join("hlo_opcode.h");
    let opcode_source = fs::read_to_string(&opcode_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", opcode_path.display()));
    let opcode_list = generate_opcode_list(&opcode_source);

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));
    fs::write(out_dir.join("opcode_list.rs"), opcode_list).expect("write generated opcode list");

    let proto_root = xla_root.join("xla");
    let proto_inputs = [
        proto_root.join("xla_data.proto"),
        proto_root.join("service").join("metrics.proto"),
        proto_root.join("service").join("hlo.proto"),
    ];
    for input in &proto_inputs {
        if !input.is_file() {
            panic!("required OpenXLA schema is missing: {}", input.display());
        }
        println!("cargo:rerun-if-changed={}", input.display());
    }
    println!("cargo:rerun-if-changed={}", opcode_path.display());

    let revision = git(&xla_root, &["rev-parse", "HEAD"]);
    println!("cargo:rustc-env=XLA_SOURCE_COMMIT={revision}");
    println!("cargo:rustc-env=XLA_SOURCE_DIR={}", xla_root.display());
    let git_head = git(&xla_root, &["rev-parse", "--git-path", "HEAD"]);
    let git_head = PathBuf::from(git_head);
    let git_head = if git_head.is_absolute() {
        git_head
    } else {
        xla_root.join(git_head)
    };
    if let Ok(git_head) = git_head.canonicalize() {
        println!("cargo:rerun-if-changed={}", git_head.display());
    }

    let generated_proto = out_dir.join("proto");
    fs::create_dir_all(&generated_proto).expect("create generated protobuf directory");
    protobuf_codegen::Codegen::new()
        .pure()
        .include(&xla_root)
        .inputs(proto_inputs)
        .out_dir(&generated_proto)
        .run_from_script();

    let mut module_source = String::new();
    for module in ["hlo", "metrics", "xla_data"] {
        let path = generated_proto.join(format!("{module}.rs"));
        if !path.is_file() {
            panic!("protobuf generator did not create {}", path.display());
        }
        module_source.push_str(&format!(
            "#[path = {:?}]\npub mod {module};\n",
            path.to_string_lossy()
        ));
    }
    fs::write(generated_proto.join("mod.rs"), module_source)
        .expect("write generated protobuf module declarations");
}

fn git(repository: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("failed to run Git for {}: {error}", repository.display()));
    if !output.status.success() {
        panic!(
            "Git {} failed in {}: {}",
            args.join(" "),
            repository.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout)
        .expect("Git output is UTF-8")
        .trim()
        .to_owned()
}

fn generate_opcode_list(source: &str) -> String {
    let mut entries = Vec::new();
    let mut variants = HashSet::new();
    let mut names = HashSet::new();

    for line in source.lines() {
        let Some(start) = line.find("V(") else {
            continue;
        };
        let after = &line[start + 2..];
        let Some(end) = after.find(')') else {
            panic!("malformed opcode entry: {line}");
        };
        let fields: Vec<_> = after[..end].splitn(3, ',').map(str::trim).collect();
        if fields.len() != 3 {
            panic!("malformed opcode entry: {line}");
        }
        let variant = fields[0]
            .strip_prefix('k')
            .unwrap_or_else(|| panic!("unexpected opcode variant: {}", fields[0]));
        let name = fields[1]
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .unwrap_or_else(|| panic!("unexpected opcode name: {}", fields[1]));
        let arity = if fields[2] == "kHloOpcodeIsVariadic" {
            "None".to_owned()
        } else {
            format!("Some({})", fields[2])
        };
        if !variants.insert(variant) || !names.insert(name) {
            panic!("duplicate OpenXLA opcode entry: {line}");
        }
        entries.push(format!("    {variant} => (\"{name}\", {arity}),\n"));
    }
    if entries.is_empty() {
        panic!("no opcode entries found in the OpenXLA source");
    }

    let mut generated = String::from("// Generated from extern/xla; do not edit.\nopcodes! {\n");
    for entry in entries {
        generated.push_str(&entry);
    }
    generated.push_str("}\n");
    generated
}
