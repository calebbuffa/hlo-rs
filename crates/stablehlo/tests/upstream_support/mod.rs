use std::{
    env,
    fmt::Write as _,
    io::Write as _,
    process::{Command, Stdio},
    thread,
};

pub enum UpstreamResult {
    Accepted(String),
    Rejected(String),
}

fn encode(source: &str) -> String {
    encode_bytes(source.as_bytes())
}

fn decode(hex: &str) -> String {
    assert!(hex.len().is_multiple_of(2), "truncated upstream response");
    let bytes = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    String::from_utf8(bytes).expect("upstream response must be UTF-8")
}

#[allow(dead_code)]
pub fn verify(modules: &[String]) -> Vec<UpstreamResult> {
    invoke(modules, "--verify")
}

#[allow(dead_code)]
pub fn verify_native(modules: &[String]) -> Vec<UpstreamResult> {
    invoke_native(modules, "--verify")
}

#[allow(dead_code)]
pub fn verify_native_bytecode(modules: &[Vec<u8>]) -> Vec<UpstreamResult> {
    run(
        native_command("--verify-generic"),
        modules.iter().map(|bytes| encode_bytes(bytes)),
        NATIVE_VERSION,
    )
}

#[allow(dead_code)]
pub fn serialize_native(modules: &[(String, String)]) -> Vec<UpstreamResult> {
    let requests = modules
        .iter()
        .map(|(version, source)| {
            assert!(!version.contains('\t') && !version.contains('\n'));
            format!("{version}\t{source}")
        })
        .collect::<Vec<_>>();
    invoke_native(&requests, "--serialize")
}

#[allow(dead_code)]
pub fn emit_native_bytecode(modules: &[(u64, String)]) -> Vec<UpstreamResult> {
    let requests = modules
        .iter()
        .map(|(version, source)| format!("{version}\t{source}"))
        .collect::<Vec<_>>();
    invoke_native(&requests, "--bytecode")
}

#[allow(dead_code)]
pub fn deserialize_native(artifacts: &[Vec<u8>]) -> Vec<UpstreamResult> {
    run(
        native_command("--deserialize"),
        artifacts.iter().map(|bytes| encode_bytes(bytes)),
        NATIVE_VERSION,
    )
}

#[allow(dead_code)]
pub fn serialize(modules: &[(String, String)]) -> Vec<UpstreamResult> {
    let requests = modules
        .iter()
        .map(|(version, source)| {
            assert!(!version.contains('\t') && !version.contains('\n'));
            format!("{version}\t{source}")
        })
        .collect::<Vec<_>>();
    invoke(&requests, "--serialize")
}

pub(crate) fn invoke(modules: &[String], mode: &str) -> Vec<UpstreamResult> {
    let python = env::var_os("STABLEHLO_UPSTREAM_PYTHON").expect(
        "set STABLEHLO_UPSTREAM_PYTHON to Python with tests/upstream/requirements.txt installed",
    );
    let mut command = Command::new(python);
    command.args([
        "-B",
        "-c",
        include_str!("../upstream/verify_assembly.py"),
        mode,
    ]);
    run(
        command,
        modules.iter().map(|source| encode(source)),
        "VERSION\t0.11.2\t1.20.0",
    )
}

const NATIVE_VERSION: &str = concat!(
    "VERSION\tnative\t1.22.0\t",
    "25605c9f48a9aec305876d60d26b028398d298d2\t",
    "2b35fb9283a44aae92cb72a41d077ee8de2be7ad"
);

fn native_command(mode: &str) -> Command {
    let executable = env::var_os("STABLEHLO_UPSTREAM_NATIVE")
        .expect("set STABLEHLO_UPSTREAM_NATIVE to the pinned native validator executable");
    let mut command = Command::new(executable);
    command.arg(mode);
    command
}

fn invoke_native(modules: &[String], mode: &str) -> Vec<UpstreamResult> {
    run(
        native_command(mode),
        modules.iter().map(|source| encode(source)),
        NATIVE_VERSION,
    )
}

fn encode_bytes(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut hex, "{byte:02x}").unwrap();
    }
    hex
}

fn run(
    mut command: Command,
    requests: impl Iterator<Item = String>,
    expected_version: &str,
) -> Vec<UpstreamResult> {
    let requests = requests.collect::<Vec<_>>();
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("cannot launch independent upstream verifier");
    let mut input = child.stdin.take().unwrap();
    let payload = requests
        .iter()
        .map(|request| request.clone() + "\n")
        .collect::<String>();
    let writer = thread::spawn(move || input.write_all(payload.as_bytes()));
    let output = child
        .wait_with_output()
        .expect("cannot read independent verifier results");
    writer
        .join()
        .expect("upstream input writer panicked")
        .expect("cannot write upstream verifier input");
    assert!(
        output.status.success(),
        "independent verifier process failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = String::from_utf8(output.stdout).unwrap();
    let mut lines = output.lines();
    assert_eq!(lines.next(), Some(expected_version));
    let results = lines
        .enumerate()
        .map(|(expected_index, line)| {
            let mut fields = line.splitn(3, '\t');
            let index: usize = fields.next().unwrap().parse().unwrap();
            assert_eq!(index, expected_index, "out-of-order upstream response");
            let status = fields.next().expect("missing upstream status");
            let text = decode(fields.next().expect("missing upstream payload"));
            match status {
                "OK" => UpstreamResult::Accepted(text),
                "ERROR" => UpstreamResult::Rejected(text),
                _ => panic!("unknown upstream result {status}"),
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        results.len(),
        requests.len(),
        "incomplete independent verifier results"
    );
    results
}
