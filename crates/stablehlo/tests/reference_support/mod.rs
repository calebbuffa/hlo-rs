use std::{env, fs, path::PathBuf, process::Command};

const REVISION: &str = "25605c9f48a9aec305876d60d26b028398d298d2";

pub fn pinned_operation_fixtures() -> String {
    let reference = PathBuf::from(
        env::var_os("STABLEHLO_REFERENCE_DIR")
            .expect("set STABLEHLO_REFERENCE_DIR to the pinned StableHLO reference checkout"),
    );
    let revision = Command::new("git")
        .args(["--no-pager", "-C"])
        .arg(&reference)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("git must be available to verify the reference revision");
    assert!(
        revision.status.success(),
        "cannot read reference revision: {}",
        String::from_utf8_lossy(&revision.stderr)
    );
    assert_eq!(
        String::from_utf8(revision.stdout).unwrap().trim(),
        REVISION,
        "fixture expectations must be audited before changing the reference revision"
    );
    let source = fs::read_to_string(
        reference
            .join("stablehlo")
            .join("tests")
            .join("ops_stablehlo.mlir"),
    )
    .expect("the pinned ops_stablehlo.mlir fixture must be readable");
    let pinned = Command::new("git")
        .args(["--no-pager", "-C"])
        .arg(&reference)
        .args([
            "show",
            &format!("{REVISION}:stablehlo/tests/ops_stablehlo.mlir"),
        ])
        .output()
        .expect("git must be available to verify the fixture contents");
    assert!(pinned.status.success(), "cannot read pinned fixture blob");
    let pinned = String::from_utf8(pinned.stdout).unwrap();
    assert!(
        source.replace("\r\n", "\n") == pinned.replace("\r\n", "\n"),
        "the fixture has local changes relative to the pinned revision"
    );
    source
}
