use stablehlo::mlir::assembly::{parse_attribute, print_attribute};
use stablehlo::{
    attributes::{AffineMap, IntegerSet},
    pliron::context::Context,
};

#[test]
fn recovered_iterative_affine_grammar_roundtrips_to_flat_native_records() {
    for source in [
        "affine_map<(d0, d1)[s0] -> (d0 + d1 * s0, d1 floordiv 4, d0 mod 3)>",
        "affine_map<() -> ()>",
        "affine_map<(d0) -> (-d0, -(d0 + 1), d0 ceildiv 2)>",
        "affine_map<(d0) -> ((d0 * 4) floordiv 2, (d0 + 3) + 5)>",
        "affine_map<() -> (-9223372036854775808)>",
        "affine_set<(d0)[s0] : (d0 >= 0, d0 <= s0, d0 == 3)>",
        "affine_set<() : ()>",
    ] {
        let ctx = Context::new();
        let value =
            parse_attribute(&ctx, source).unwrap_or_else(|error| panic!("{source}: {error}"));
        let printed = print_attribute(&ctx, &*value).unwrap();
        let fresh = Context::new();
        let reparsed =
            parse_attribute(&fresh, &printed).unwrap_or_else(|error| panic!("{printed}: {error}"));
        assert_eq!(print_attribute(&fresh, &*reparsed).unwrap(), printed);
    }
}

#[test]
fn affine_arithmetic_normalization_preserves_values() {
    let ctx = Context::new();
    let attribute = parse_attribute(&ctx,
        "affine_map<(d0) -> ((d0 * 4) floordiv 2, (d0 + 3) + 5, -7 floordiv 3, -7 ceildiv 3, -7 mod 3)>").unwrap();
    let map = attribute.downcast_ref::<AffineMap>().unwrap();
    for value in -100..100 {
        assert_eq!(
            map.evaluate(&[value], &[]).unwrap(),
            [value * 2, value + 8, -3, -2, 2]
        );
    }
    let attribute = parse_attribute(&ctx, "affine_set<(d0)[s0] : (d0 >= 0, d0 <= s0)>").unwrap();
    let set = attribute.downcast_ref::<IntegerSet>().unwrap();
    assert!(set.contains(&[2], &[4]).unwrap());
    assert!(!set.contains(&[-1], &[4]).unwrap());
    assert!(!set.contains(&[5], &[4]).unwrap());
}

#[test]
fn malformed_nonlinear_and_deep_expressions_are_rejected() {
    let ctx = Context::new();
    for source in [
        "affine_map<(d0, d0) -> (d0)>",
        "affine_map<(d0) -> (d1)>",
        "affine_map<(d0, d1) -> (d0 * d1)>",
        "affine_map<(d0, d1) -> (d0 floordiv d1)>",
        "affine_map<(d0) -> (d0 +)>",
        "affine_map<(d0) -> ((d0)>",
        "affine_set<(d0) : (d0 = 0)>",
    ] {
        assert!(parse_attribute(&ctx, source).is_err(), "{source}");
    }
    let source = format!(
        "affine_map<(d0) -> ({}d0{})>",
        "(".repeat(200),
        ")".repeat(200)
    );
    assert!(parse_attribute(&ctx, &source).is_err());
    assert!(parse_attribute(&ctx, "affine_map<(d0) -> (d0)>").is_ok());
}
