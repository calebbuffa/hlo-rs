use std::ptr;
use xlahlo::{
    Diagnostic,
    literal::Literal,
    pliron::{common_traits::Verify, context::Context},
    shape::{self, Dimension, Layout},
    types::ElementType as P,
};

#[test]
fn raw_literal_payloads_are_checked_at_verification_and_query_boundaries() {
    let ctx = Context::new();
    let scalar = Literal::from_bits(&ctx, shape::scalar(&ctx, P::U8).unwrap(), &[7]).unwrap();
    let tuple = Literal::tuple(&ctx, vec![scalar.clone()]).unwrap();
    let token = Literal::token(&ctx);
    let mut malformed = Vec::new();
    for literal in [&tuple, &token] {
        let mut bad = literal.clone();
        bad.storage.push(1);
        malformed.push(bad);
        let mut bad = literal.clone();
        bad.dynamic_sizes.push(1);
        malformed.push(bad);
    }
    let mut bad = token.clone();
    bad.tuple_literals.push(scalar.clone());
    malformed.push(bad);
    let mut bad = scalar.clone();
    bad.dynamic_sizes.push(1);
    malformed.push(bad);
    let mut bad = scalar.clone();
    bad.storage.clear();
    malformed.push(bad);
    let mut bad = scalar.clone();
    bad.tuple_literals.push(token.clone());
    malformed.push(bad);
    let mut bad = scalar.clone();
    bad.storage.push(8);
    malformed.push(bad);
    let mut bad = tuple.clone();
    bad.tuple_literals.clear();
    malformed.push(bad);
    let mut bad = scalar.clone();
    bad.storage[0] = 2;
    bad.shape = shape::with_default_layout(&ctx, shape::scalar(&ctx, P::Pred).unwrap()).unwrap();
    malformed.push(bad);

    for bad in malformed {
        for error in [
            bad.verify(&ctx).unwrap_err(),
            bad.bits(&ctx).unwrap_err(),
            bad.equal_logical_bits(&ctx, &bad).unwrap_err(),
            bad.sub_literal(&ctx, &[]).unwrap_err(),
        ] {
            assert!(matches!(
                error.err.downcast_ref::<Diagnostic>(),
                Some(Diagnostic::InvalidShape { .. })
            ));
        }
    }
    assert_eq!(scalar.bits(&ctx).unwrap(), [7]);
    assert!(tuple.verify(&ctx).is_ok());
    assert!(token.verify(&ctx).is_ok());
    assert!(tuple.bits(&ctx).is_err());
    assert!(token.bits(&ctx).is_err());
}

#[test]
fn every_f16_and_bf16_encoding_roundtrips_without_numeric_conversion() {
    let ctx = Context::new();
    let patterns: Vec<u128> = (0..=u16::MAX).map(u128::from).collect();
    let bytes: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    for element in [P::F16, P::BF16] {
        let shape = shape::array(&ctx, element, vec![patterns.len() as i64]).unwrap();
        let literal = Literal::from_bits(&ctx, shape, &patterns).unwrap();
        assert_eq!(literal.storage, bytes);
        assert_eq!(literal.bits(&ctx).unwrap(), patterns);
        let restored = Literal::from_storage_bytes(&ctx, shape, bytes.as_slice()).unwrap();
        assert_eq!(restored.bits(&ctx).unwrap(), patterns);
        assert!(literal.equal_logical_bits(&ctx, &restored).unwrap());
    }
}

#[test]
fn owned_storage_moves_and_tuple_subviews_borrow_without_payload_copies() {
    let ctx = Context::new();
    let shape = shape::array(&ctx, P::U8, vec![3]).unwrap();
    let storage = vec![7, 8, 9];
    let storage_ptr = storage.as_ptr();
    let literal = Literal::from_storage_bytes(&ctx, shape, storage).unwrap();
    assert!(ptr::eq(literal.storage.as_ptr(), storage_ptr));
    let tuple = Literal::tuple(&ctx, vec![literal]).unwrap();
    let child = tuple.sub_literal(&ctx, &[0]).unwrap();
    assert!(ptr::eq(child, &tuple.tuple_literals[0]));
    assert!(ptr::eq(child.storage.as_ptr(), storage_ptr));
    assert!(ptr::eq(tuple.sub_literal(&ctx, &[]).unwrap(), &tuple));
}

#[test]
fn bit_literals_preserve_every_array_primitive_including_nan_and_signed_zero() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    for &ty in P::ALL {
        let width = ty.bit_width();
        let patterns = if width == 128 {
            vec![0, 1, u128::MAX, 1u128 << 127]
        } else {
            vec![0, 1, (1u128 << width) - 1, 1u128 << (width - 1)]
        };
        let shape = shape::array(&ctx, ty, vec![patterns.len() as i64]).unwrap();
        let literal = Literal::from_bits(&ctx, shape, &patterns).unwrap();
        assert_eq!(literal.bits(&ctx).unwrap(), patterns, "{ty:?}");
        literal.verify(&ctx).unwrap();
    }
    for ty in [
        P::S1,
        P::S2,
        P::S4,
        P::U1,
        P::U2,
        P::U4,
        P::F4E2M1FN,
        P::F6E2M3FN,
        P::F6E3M2FN,
    ] {
        let values: Vec<_> = (0..1u128 << ty.bit_width()).collect();
        let shape = shape::array(&ctx, ty, vec![values.len() as i64]).unwrap();
        let literal = Literal::from_bits(&ctx, shape, &values).unwrap();
        assert_eq!(literal.bits(&ctx).unwrap(), values);
    }
    assert!(Literal::from_bits(&ctx, shape::scalar(&ctx, P::U8).unwrap(), &[256]).is_err());
    assert!(Literal::from_bits(&ctx, shape::scalar(&ctx, P::F32).unwrap(), &[]).is_err());
}

fn matrix(ctx: &Context, order: Vec<i64>, bits: &[u128]) -> Literal {
    let shape = shape::with_layout(
        ctx,
        shape::array(ctx, P::F32, vec![2, 2]).unwrap(),
        Layout {
            minor_to_major: order,
            ..Layout::default()
        },
    )
    .unwrap();
    Literal::from_bits(ctx, shape, bits).unwrap()
}

#[test]
fn logical_bit_equality_is_layout_independent_and_preserves_nan_and_zero_bits() {
    let ctx = Context::new();
    let bits = [0x7fc00001, 0x80000000, 0x7f800000, 0x40400000];
    let left = matrix(&ctx, vec![1, 0], &bits);
    let right = matrix(&ctx, vec![0, 1], &[bits[0], bits[2], bits[1], bits[3]]);
    assert!(left.equal_logical_bits(&ctx, &right).unwrap());
    for bits in [
        [0x7fc00002, bits[2], bits[1], bits[3]],
        [bits[0], bits[2], 0, bits[3]],
    ] {
        assert!(
            !left
                .equal_logical_bits(&ctx, &matrix(&ctx, vec![0, 1], &bits))
                .unwrap()
        );
    }
    let left = Literal::tuple(&ctx, vec![left, Literal::token(&ctx)]).unwrap();
    let right = Literal::tuple(&ctx, vec![right, Literal::token(&ctx)]).unwrap();
    assert!(left.equal_logical_bits(&ctx, &right).unwrap());
    assert_eq!(left.sub_literal(&ctx, &[1]).unwrap(), &Literal::token(&ctx));
    assert!(left.sub_literal(&ctx, &[2]).is_err());
}

fn dynamic(ctx: &Context, bound: i64, active: i64, order: Vec<i64>, bits: &[u32]) -> Literal {
    let shape = shape::dynamic_array(
        ctx,
        P::S32,
        vec![Dimension::Bounded(bound), Dimension::Static(2)],
    )
    .unwrap();
    let shape = shape::with_layout(
        ctx,
        shape,
        Layout {
            minor_to_major: order,
            ..Layout::default()
        },
    )
    .unwrap();
    let bytes = bits.iter().flat_map(|n| n.to_le_bytes()).collect();
    Literal::from_dynamic_storage_bytes(ctx, shape, bytes, vec![active, 2]).unwrap()
}

#[test]
fn logical_bit_equality_uses_only_active_dynamic_elements() {
    let ctx = Context::new();
    let a = dynamic(&ctx, 2, 1, vec![1, 0], &[1, 2, 3, 4]);
    let b = dynamic(&ctx, 3, 1, vec![0, 1], &[1, 9, 7, 2, 8, 6]);
    assert!(a.equal_logical_bits(&ctx, &b).unwrap());
    let c = dynamic(&ctx, 2, 2, vec![1, 0], &[1, 2, 3, 4]);
    assert!(!a.equal_logical_bits(&ctx, &c).unwrap());
    let a = dynamic(&ctx, 2, 0, vec![1, 0], &[1, 2, 3, 4]);
    let b = dynamic(&ctx, 3, 0, vec![0, 1], &[9, 7, 5, 8, 6, 4]);
    assert!(a.equal_logical_bits(&ctx, &b).unwrap());
}

#[test]
fn logical_bit_equality_ignores_unused_subbyte_storage_bits() {
    let ctx = Context::new();
    let shape = shape::array(&ctx, P::U4, vec![2]).unwrap();
    let a = Literal::from_storage_bytes(&ctx, shape, [0xf1, 0xa2]).unwrap();
    let b = Literal::from_storage_bytes(&ctx, shape, [0x01, 0x02]).unwrap();
    let c = Literal::from_storage_bytes(&ctx, shape, [0x01, 0x03]).unwrap();
    assert_eq!(a.storage, [0xf1, 0xa2]);
    assert!(a.equal_logical_bits(&ctx, &b).unwrap());
    assert!(!a.equal_logical_bits(&ctx, &c).unwrap());
}

#[test]
fn payload_validation_checks_predicates_dynamic_bounds_and_layout_normalization() {
    let ctx = Context::new();
    let pred = shape::scalar(&ctx, P::Pred).unwrap();
    assert!(Literal::from_storage_bytes(&ctx, pred, [2]).is_err());
    let packed = shape::with_layout(
        &ctx,
        shape::array(&ctx, P::U4, vec![2]).unwrap(),
        Layout {
            minor_to_major: vec![0],
            element_size_in_bits: 4,
            ..Layout::default()
        },
    )
    .unwrap();
    let literal = Literal::from_storage_bytes(&ctx, packed, [15, 3]).unwrap();
    assert_eq!(
        shape::layout(&ctx, literal.shape())
            .unwrap()
            .unwrap()
            .element_size_in_bits,
        0
    );
    let dynamic = shape::dynamic_array(&ctx, P::U8, vec![Dimension::Bounded(3)]).unwrap();
    assert!(Literal::from_bits(&ctx, dynamic, &[0, 1, 2]).is_err());
    assert!(Literal::from_dynamic_storage_bytes(&ctx, dynamic, vec![0, 1, 2], vec![]).is_err());
    assert!(Literal::from_dynamic_storage_bytes(&ctx, dynamic, vec![0, 1, 2], vec![4]).is_err());
    assert!(Literal::from_dynamic_storage_bytes(&ctx, dynamic, vec![0, 1, 2], vec![-1]).is_err());
    let unbounded = shape::dynamic_array(&ctx, P::U8, vec![Dimension::Unbounded]).unwrap();
    assert!(Literal::from_dynamic_storage_bytes(&ctx, unbounded, vec![], vec![0]).is_err());
}
