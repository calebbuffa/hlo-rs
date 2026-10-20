# xlahlo-cubecl

This crate verifies xlahlo modules, lowers supported operations to CubeCL IR,
and produces caller-executable `ExecutionPlan`s. Nanoxla owns StableHLO import
and target-independent optimization; this crate does not define a second
optimization IR.

Add `xlahlo-cubecl` and `xlahlo` to your dependencies. Build and verify an
xlahlo module, then compile it with `ModuleCompiler::builder().build().compile`.
The returned plan describes buffers, kernels, and launches; it does not
allocate device memory or execute kernels.

## License

Apache-2.0. See the workspace `LICENSE-APACHE`.
