# HLO compatibility

`xlahlo` implements an evolving subset of OpenXLA HLO. It is a Rust data model,
text and protobuf interface, and validator—not an optimizer or execution
engine. The crate does not claim complete compatibility with OpenXLA.

## Current scope

The implementation includes:

- HLO shapes, layouts, literals, computations, instructions, and modules.
- Protobuf import/export, graph editing, inference, and semantic verification.
- HLO text parsing and printing for supported instructions and attributes.
- Rust tests for graph, shape, literal, text, wire-format, and verification
  behavior.

The optional C++ oracle compares selected parser, verifier, inference, and
serialization cases with the current OpenXLA checkout. These tests provide
independent evidence for those cases only; they are not a run of the full XLA
test suite.

## Known limitations

Operation constraints, inference rules, text syntax, and import normalization
are not complete. Unsupported or lossy text constructs return errors rather
than being silently rewritten. Successful protobuf import is a structural
check, not proof that a module passes semantic verification. Protobuf unknown
fields are retained where the wire representation permits it.

The generated schema and opcode catalog follow `extern/xla`. Updating that
submodule automatically changes the Rust wire types and opcode list on the
next Cargo build. Semantic behavior is Rust code and still requires review and
tests when upstream changes.

## Checks

Run from the `hlo-rs` workspace root:

```powershell
cargo test --locked -p xlahlo
cargo clippy --locked -p xlahlo --all-targets --all-features -- -D warnings
```

The native HLO oracle is optional and requires Bazel, a C++ compiler, and the
initialized `extern/xla` submodule:

```powershell
python crates/xlahlo/tools/hlo_oracle.py
```

The oracle test binary contains ignored differential tests; set
`XLAHLO_HLO_ORACLE` to the resulting executable and run the relevant ignored
tests to compare Rust behavior with XLA. The Rust test suite does not require
the native oracle.
