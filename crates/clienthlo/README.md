# CHLO

`clienthlo` is a pliron implementation of OpenXLA's CHLO (client HLO)
dialect. CHLO adds client-oriented, higher-level tensor operations alongside
StableHLO operations in StableHLO functions. It augments StableHLO and legalizes
to StableHLO/XLA HLO; it does not replace StableHLO.

All operations use the `chlo.` dialect namespace. Typed operation wrappers are
re-exported from `clienthlo::ops`; operation verifiers check operand domains,
attributes, shape inference, and result types.

| Operations | Family |
| --- | --- |
| `broadcast_add`, `broadcast_atan2`, `broadcast_divide`, `broadcast_maximum`, `broadcast_minimum`, `broadcast_multiply`, `broadcast_next_after`, `broadcast_polygamma`, `broadcast_power`, `broadcast_remainder`, `broadcast_shift_left`, `broadcast_shift_right_arithmetic`, `broadcast_shift_right_logical`, `broadcast_subtract`, `broadcast_zeta`, `broadcast_and`, `broadcast_or`, `broadcast_xor` | Broadcasting binary elementwise |
| `broadcast_complex`, `broadcast_compare`, `broadcast_select` | Broadcasting type conversion, compare, and selection |
| `next_after`, `polygamma`, `zeta`, `mulhi` | Non-broadcasting binary elementwise |
| `_asin_acos_kernel`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `bessel_i1e`, `conj`, `cosh`, `sinh`, `tan`, `digamma`, `erf`, `erf_inv`, `erfc`, `is_inf`, `is_neg_inf`, `is_pos_inf`, `lgamma`, `square` | Unary elementwise |
| `constant`, `constant_like`, `top_k`, `ragged_dot`, `scan` | Constants, compute, and control |

Inferred constructors derive result types from operands and attributes, using
upstream's per-operation inference rules (including relaxed broadcast fallback).
Use the explicit `new` constructor when a result type is already available,
then verify the completed operation.

```rust
use clienthlo::ops::{BroadcastAddOp, ErfOp};
use pliron::{context::Context, op::Op, r#type::Typed};
use stablehlo::{ops::ConstantOp, types::{ElementType, tensor}};

let mut ctx = Context::new();
let ty = tensor(&ctx, &[4], ElementType::F32);
let x = ConstantOp::new(&mut ctx, ty, vec![0x3f80_0000], true);
let x = x.get_operation().deref(&ctx).get_result(0);
let erf = ErfOp::new_inferred(&mut ctx, x)?;
let sum = BroadcastAddOp::new_inferred(&mut ctx, x, x, None)?;
assert_eq!(erf.result(&ctx).get_type(&ctx), sum.get_operation().deref(&ctx).get_result(0).get_type(&ctx));
# Ok::<(), pliron::result::Error>(())
```

`clienthlo::{Diagnostic, RecoveryCategory}` expose recoverable verifier and
builder diagnostics. The crate re-exports `pliron` for downstream users.

## License

Apache-2.0. See the workspace `LICENSE-APACHE`.
