use xlahlo::proto::xla_data::{PaddingConfig, padding_config::PaddingConfigDimension};
use xlahlo::{Dimension as D, PrimitiveType as P, Shape, inference, text};

#[test]
fn dimension_size_text_requires_the_native_single_axis_list() {
    for root in [
        "s32[] get-dimension-size(p)",
        "f32[<=3] set-dimension-size(p,s)",
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ p = f32[3] parameter(0) s = s32[] parameter(1) ROOT d = {root}, dimensions={{0}} }}"
        );
        let module = text::parse_module(&source).unwrap();
        assert!(
            text::print_module(&module)
                .unwrap()
                .contains("dimensions={0}")
        );
        for attribute in ["dimension=0", "dimensions={}", "dimensions={0,1}"] {
            assert!(text::parse_module(&source.replace("dimensions={0}", attribute)).is_err());
        }
        assert!(text::parse_module(&source.replace(", dimensions={0}", "")).is_err());
        let mut wire = module.to_proto().unwrap();
        wire.computations[0]
            .instructions
            .iter_mut()
            .find(|i| i.name == "d")
            .unwrap()
            .dimensions
            .clear();
        assert!(text::print_module(&xlahlo::Module::from_proto(wire).unwrap()).is_err());
    }
}

fn pad(low: i64, high: i64, interior: i64) -> PaddingConfig {
    PaddingConfig {
        dimensions: vec![PaddingConfigDimension {
            edge_padding_low: low,
            edge_padding_high: high,
            interior_padding: interior,
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn small_padding_geometry_includes_negative_edges_zero_extents_and_dynamics() {
    let value = Shape::scalar(P::F32).unwrap();
    for size in 0..5 {
        let input = Shape::array(P::F32, vec![size]).unwrap();
        for low in -3..4 {
            for high in -3..4 {
                for interior in 0..4 {
                    let expected = size + low + high + (size - 1).max(0) * interior;
                    let result = inference::infer_pad(&input, &value, &pad(low, high, interior));
                    if expected < 0 {
                        assert!(result.is_err());
                    } else {
                        assert_eq!(result.unwrap().dimensions(), [expected]);
                    }
                }
            }
        }
    }
    let input = Shape::dynamic_array(P::F32, vec![D::Bounded(4)]).unwrap();
    assert_eq!(
        inference::infer_pad(&input, &value, &pad(1, 2, 1))
            .unwrap()
            .dimension(0)
            .unwrap(),
        D::Bounded(10)
    );
    assert!(inference::infer_pad(&input, &value, &pad(0, 0, -1)).is_err());
    assert!(inference::infer_pad(&input, &value, &pad(i64::MAX, 0, 0)).is_err());
    let unbounded = Shape::dynamic_array(P::F32, vec![D::Unbounded]).unwrap();
    assert_eq!(
        inference::infer_pad(&unbounded, &value, &pad(1, 2, 1))
            .unwrap()
            .dimension(0)
            .unwrap(),
        D::Unbounded
    );
}

#[test]
fn concatenate_sums_bounds_and_merges_nonconcatenated_dimensions() {
    for a in 0..5 {
        for b in 0..5 {
            let lhs = Shape::array(P::F32, vec![2, a]).unwrap();
            let rhs = Shape::array(P::F32, vec![2, b]).unwrap();
            assert_eq!(
                inference::infer_concatenate(&[lhs.clone(), rhs], 1)
                    .unwrap()
                    .dimensions(),
                [2, a + b]
            );
            assert!(inference::infer_concatenate(&[lhs], 2).is_err());
        }
    }
    let a = Shape::dynamic_array(P::F32, vec![D::Unbounded, D::Bounded(3)]).unwrap();
    let b = Shape::array(P::F32, vec![2, 4]).unwrap();
    let result = inference::infer_concatenate(&[a, b], 1).unwrap();
    assert_eq!(result.dimension(0).unwrap(), D::Static(2));
    assert_eq!(result.dimension(1).unwrap(), D::Bounded(7));
    assert!(inference::infer_concatenate(&[], 0).is_err());
    let huge = Shape::array(P::U8, vec![i64::MAX]).unwrap();
    assert!(inference::infer_concatenate(&[huge.clone(), huge], 0).is_err());
}

#[test]
fn select_and_clamp_accept_only_their_upstream_broadcast_forms() {
    let a = Shape::array(P::F32, vec![2, 3]).unwrap();
    let scalar = Shape::scalar(P::F32).unwrap();
    let pred = Shape::scalar(P::PRED).unwrap();
    assert!(inference::infer_select(&pred, &a, &a).is_ok());
    assert!(inference::infer_select(&a, &a, &a).is_err());
    assert!(inference::infer_select(&pred, &a, &scalar).is_err());
    assert!(inference::infer_clamp(&scalar, &a, &scalar).is_ok());
    assert!(inference::infer_clamp(&a, &a, &a).is_ok());
    assert!(
        inference::infer_clamp(&Shape::array(P::F32, vec![1, 3]).unwrap(), &a, &scalar).is_err()
    );
    let dynamic = Shape::dynamic_array(P::F32, vec![D::Bounded(2), D::Static(3)]).unwrap();
    assert_eq!(
        inference::infer_select(&pred, &a, &dynamic)
            .unwrap()
            .dimension(0)
            .unwrap(),
        D::Bounded(2)
    );
}

#[test]
fn comparison_padding_concatenate_and_ternary_assembly_round_trip() {
    let source = r#"
HloModule arrays
ENTRY main {
  p = f32[2] parameter(0)
  zero = f32[] constant(0)
  lo = f32[] constant(-1)
  hi = f32[] constant(1)
  clamped = f32[2] clamp(lo,p,hi)
  cmp = pred[2] compare(p,clamped), direction=LE, order=PARTIAL
  selected = f32[2] select(cmp,p,clamped)
  padded = f32[5] pad(selected,zero), padding=1_1_1
  ROOT concat = f32[7] concatenate(p,padded), dimensions={0}
}
"#;
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for invalid in [
        source.replace("direction=LE", "direction=INVALID"),
        source.replace("order=PARTIAL", "order=INVALID"),
        source.replace("padding=1_1_1", "padding=1_1_-1"),
    ] {
        assert!(text::parse_module(&invalid).is_err(), "{invalid}");
    }
    for invalid in [
        source.replace("dimensions={0}", "dimensions={1}"),
        source.replace("pred[2] compare", "f32[2] compare"),
    ] {
        assert!(
            text::parse_module(&invalid)
                .unwrap()
                .verify(Default::default())
                .is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn dimension_size_dynamic_slice_and_iota_have_upstream_attribute_spellings() {
    for source in [
        "HloModule m ENTRY main { ROOT i = s32[3] iota(), iota_dimension=0 }",
        "HloModule m ENTRY main { p = f32[3] parameter(0) ROOT size = s32[] get-dimension-size(p), dimensions={0} }",
        "HloModule m ENTRY main { p = f32[3] parameter(0) s = s32[] parameter(1) ROOT d = f32[<=3] set-dimension-size(p,s), dimensions={0} }",
        "HloModule m ENTRY main { p = f32[3] parameter(0) s = s32[] parameter(1) ROOT d = f32[2] dynamic-slice(p,s), dynamic_slice_sizes={2} }",
    ] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
}

#[test]
fn iota_accepts_integral_floating_and_complex_arrays_but_not_predicate_arrays() {
    for element_type in ["s32", "f32", "c64"] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT i = {element_type}[3] iota(), iota_dimension=0 }}"
        );
        text::parse_module(&source)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }

    let predicate =
        text::parse_module("HloModule m ENTRY main { ROOT i = pred[3] iota(), iota_dimension=0 }")
            .unwrap();
    assert_eq!(
        predicate.verify(Default::default()).unwrap_err().category,
        "iota"
    );
}

#[test]
fn precision_and_stochastic_conversion_validate_their_numeric_contracts() {
    let input = Shape::array(P::F32, vec![2]).unwrap();
    for exponent in -1..3 {
        for mantissa in -1..3 {
            assert_eq!(
                inference::infer_reduce_precision(&input, exponent, mantissa).is_ok(),
                exponent >= 1 && mantissa >= 0
            );
        }
    }
    let random = Shape::array(P::U32, vec![2]).unwrap();
    assert_eq!(
        inference::infer_stochastic_convert(&input, &random, P::F16)
            .unwrap()
            .element_type(),
        P::F16
    );
    for invalid in [
        Shape::array(P::S32, vec![2]).unwrap(),
        Shape::array(P::U64, vec![2]).unwrap(),
        Shape::array(P::U32, vec![3]).unwrap(),
        Shape::scalar(P::U32).unwrap(),
    ] {
        assert!(inference::infer_stochastic_convert(&input, &invalid, P::F16).is_err());
    }
    for source in [
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = f32[2] reduce-precision(p), exponent_bits=8, mantissa_bits=10 }",
        "HloModule m ENTRY main { p = f32[2] parameter(0) r = u32[2] parameter(1) ROOT c = f16[2] stochastic-convert(p,r) }",
    ] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
}
