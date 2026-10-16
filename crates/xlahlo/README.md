# xlahlo

`xlahlo` is a Rust implementation of the OpenXLA HLO model and its text and
protobuf representations. It provides HLO data structures, parsing, printing,
and validation; optimization and execution belong to downstream projects such
as `nanoxla`.

The crate is under active development and does not claim complete OpenXLA
conformance. See [HLO compatibility](docs/CONFORMANCE.md) for its scope and
known limitations.

## Use

Build a module from the typed model, then validate or serialize it:

```rust
use xlahlo::{text, Instruction, Module, Opcode, PrimitiveType, Shape};

let shape = Shape::array(PrimitiveType::F32, vec![2])?;
let mut module = Module::new("example");
let entry = module.add_computation("entry")?;
let x = module.add_instruction(
    entry,
    Instruction::new("x", Opcode::Parameter, shape.clone()),
    &[],
    &[],
)?;
let y = module.add_instruction(
    entry,
    Instruction::new("y", Opcode::Parameter, shape.clone())
        .with_fields(|instruction| instruction.parameter_number = 1)?,
    &[],
    &[],
)?;
let sum = module.add_instruction(
    entry,
    Instruction::new("sum", Opcode::Add, shape),
    &[x, y],
    &[],
)?;
module.set_root(entry, sum)?;
module.set_entry(entry)?;
module.verify(Default::default())?;

let text = text::print_module(&module)?;
# Ok::<(), xlahlo::Error>(())
```

Consult the crate API for supported builders and operation-specific
verification requirements. Parsing and printing report unsupported or
unrepresentable constructs explicitly rather than silently changing them.

## Model and interoperability

- Shapes represent arrays, tuples, layouts, and dynamic dimensions. Literals
  retain typed values and their physical representations.
- A module can be imported from or exported to the generated HLO protobuf
  types. Protobuf data not modeled by the Rust API is retained where possible.
- `Module::verify` checks semantic invariants. Protobuf import also performs
  structural checks; successful import alone does not imply a valid HLO
  program.
- `ModuleConfig` is crate-side metadata and is not encoded in
  `HloModuleProto`. Preserve it across conversions with the configuration-aware
  APIs.
- Nested parsing and validation use bounded worklists to avoid input-driven
  call-stack growth.

## Workspace checks

Run these commands from the `hlo-rs` workspace root:

```powershell
cargo fmt --manifest-path crates/xlahlo/Cargo.toml --check
cargo test --locked -p xlahlo
cargo clippy --locked -p xlahlo --all-targets --all-features -- -D warnings
```

## OpenXLA source and tools

Initialize the `extern/xla` submodule before building. Cargo generates the
protobuf bindings and opcode catalog directly from that checkout, and
`XLA_SOURCE_COMMIT` identifies the revision used by the build. No copied schema
or manually refreshed revision manifest is required.

The optional native oracles use the same checkout. Their build artifacts live
under `target`; numeric-oracle dependency sources are extracted under `extern`
at the revisions declared by XLA.

```powershell
git submodule update --init extern/xla
python crates/xlahlo/tools/hlo_oracle.py
python crates/xlahlo/tools/numeric_oracle.py
```

Extracted Cargo packages do not contain the submodule. Set `XLA_SOURCE_DIR` to
the workspace's `extern/xla` checkout when building or verifying a package.

## License

Apache-2.0 or MIT.
