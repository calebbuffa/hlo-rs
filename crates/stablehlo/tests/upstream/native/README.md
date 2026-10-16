# Pinned native reference validator

This is a **test-only** driver for the unmodified StableHLO reference at
`25605c9f48a9aec305876d60d26b028398d298d2` (1.22.0), using its LLVM pin
`2b35fb9283a44aae92cb72a41d077ee8de2be7ad`. It is not an implementation of
StableHLO in Rust and is not linked into the Rust crate.

It explicitly registers StableHLO, VHLO, Func, Arith, Shape, Tensor, Quant,
SparseTensor and Complex. Unregistered dialects remain forbidden.
The driver supports parse/verify/print, native MLIR bytecode writing, portable
serialization and portable deserialization. It does not execute programs; execution audits continue to
use the separately pinned jaxlib adapter.

## Windows build

Requirements: CMake, Ninja, clang-cl, Visual Studio C++ tools/Windows SDK,
Git, curl and tar. The reference checkout must be at the specified revision
without tracked local changes. The source archive SHA256 is
`B61C85C97CDE1C3E96EA7CFD09737C24578D6D0656C392D79CE402DA6904B828`.

From the Rust crate root:

```powershell
.\tests\upstream\native\build.ps1 -ReferenceDir ..\..\extern\stablehlo
$env:STABLEHLO_REFERENCE_DIR = (Resolve-Path ..\..\extern\stablehlo).Path
$env:STABLEHLO_UPSTREAM_NATIVE = (Resolve-Path ..\..\target\reference-1.22\build\bin\stablehlo-rust-validator.exe).Path
cargo test --test upstream_interop native -- --ignored --nocapture
cargo test --test portable_versions matching_native -- --ignored --nocapture
```

Use `-VcVars64` for a different Visual Studio installation and `-Jobs` to
adjust the default three concurrent compiler jobs. Only the validator target's
dependency closure is built; this is still a substantial MLIR build. Sources
and the verified LLVM source is extracted under the workspace's
`extern\llvm-project-2b35fb9283a44aae92cb72a41d077ee8de2be7ad`.
Archives and build outputs stay in `target\reference-1.22`; neither location is
packaged with the crate. The default reference path is `extern\stablehlo` at the workspace
root; it must be at the exact revision above. If the submodule is at a different
revision, deliberately align the `extern\stablehlo` submodule before building
and set `STABLEHLO_REFERENCE_DIR` to that checkout. All upstream source paths
must remain under the workspace's `extern` directory. No automatic revision
change is performed.

Native artifacts retained from the standalone crate are still under the crate's
`target\reference-1.22`. The retained validator executable may be used directly
with `STABLEHLO_UPSTREAM_NATIVE`, but its old CMake/Ninja cache contains absolute
paths. Fresh builds use the workspace target directory to avoid reusing that
stale cache; do not rebuild from the migrated cache.
The script downloads and verifies the archive before extracting LLVM, MLIR,
libc utility headers and CMake/third-party build support. It does not install
dependencies, modify the reference checkout or change the global environment.
The external CMake project disables shared precompiled-header reuse for the
CHLO legalization source on Windows: otherwise the MLIR precompiled `<cmath>`
precedes StableHLO's `_USE_MATH_DEFINES`, leaving `M_PI` unavailable.
Do not edit these extracted sources; the extraction stamp assumes they remain
unmodified. The CMake checks verify the reference checkout and its dependency
pin; the build script verifies the actual LLVM archive.

## Protocol

`--version` reports the dialect version and both pinned revisions.
Other modes read one hex-encoded byte string per stdin line and begin stdout
with the same version header. Each response is
`index<TAB>OK|ERROR<TAB>hex-encoded-UTF8-payload`. Diagnostics are captured per
request. Protocol and process failures are fatal, not successful responses.

| Mode | Request bytes | Successful response payload |
| --- | --- | --- |
| `--verify` | MLIR UTF-8 text | Verified printed assembly |
| `--verify-generic` | MLIR text or bytecode | Verified generic assembly, preserving explicit defaults |
| `--bytecode` | `format-version<TAB>MLIR text` | Hex-encoded native MLIR bytecode (versions 0-6) |
| `--serialize` | `target-version<TAB>MLIR text` | Hex-encoded portable artifact |
| `--deserialize` | Raw artifact bytes | Verified printed assembly |

Generic verification includes debug locations, preserving both explicit source
provenance and locations assigned by the native parser. Native bytecode mode
uses the reference's default compressed dialect payloads and operation
properties; it does not use Rust's textual-fallback writer.

The extra hex layer for serialization matches the historical Python adapter.
The shared Rust test harness asserts version/revision identity, result count,
ordering, process success and response encoding. Native audits do not reuse
the older tool's registration or version exceptions.

These audits are additional evidence, not proof of complete Rust conformance.
Portable conversion here runs in upstream C++; it does not supply the missing
Rust VHLO/portable-artifact implementation.
