//! Native StableHLO dialect for the pliron intermediate representation.
//!
//! Public APIs are organized under [`attributes`], [`inference`], [`integer`],
//! [`ops`], [`sparse`], [`types`], and [`verification`].

#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub use pliron;

pub mod attributes;
mod error;
pub use error::{Diagnostic, RecoveryCategory};
pub mod inference;
pub mod integer;
pub mod ops;
pub mod sparse;
pub mod types;
pub mod verification;

#[cfg(test)]
mod tests {
    use crate::{ops::*, types::*};
    use pliron::{common_traits::Verify, context::Context, op::Op, r#type::Typed};

    #[test]
    fn context_registers_stablehlo_entities_without_key_collisions() {
        let _ctx = Context::new();
    }

    #[test]
    fn tensor_type_accepts_explicit_dtype_set() {
        let ctx = Context::default();
        for ty in [
            ElementType::F16,
            ElementType::Bf16,
            ElementType::F32,
            ElementType::F64,
            ElementType::I1,
            ElementType::I8,
            ElementType::I16,
            ElementType::I32,
            ElementType::I64,
            ElementType::Ui8,
            ElementType::Ui32,
        ] {
            assert_eq!(
                tensor_parts(&ctx, tensor(&ctx, &[2, 3], ty)),
                Some((vec![2, 3], ty))
            );
        }
    }

    #[test]
    fn dense_constants_validate_bit_pattern_count() {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[2], ElementType::F32);
        let splat = ConstantOp::new(&mut ctx, ty, vec![0x8000_0000], true);
        assert_eq!(splat.bits(&ctx), Some(vec![0x8000_0000]));
        assert!(splat.verify(&ctx).is_ok());
        let bad = ConstantOp::new(&mut ctx, ty, vec![1, 2, 3], false);
        assert!(bad.verify(&ctx).is_err());
    }

    #[test]
    fn inferred_add_preserves_identical_tensor_type() {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[4], ElementType::F32);
        let a = ConstantOp::new(&mut ctx, ty, vec![1], true);
        let b = ConstantOp::new(&mut ctx, ty, vec![2], true);
        let av = a.get_operation().deref(&ctx).get_result(0);
        let bv = b.get_operation().deref(&ctx).get_result(0);
        let add = AddOp::new_inferred(&mut ctx, av, bv);
        assert!(add.is_ok(), "matching tensors should infer add");
        let Some(add) = add.ok() else {
            return;
        };
        assert_eq!(
            tensor_parts(
                &ctx,
                add.get_operation().deref(&ctx).get_result(0).get_type(&ctx)
            ),
            Some((vec![4], ElementType::F32))
        );
    }
}
