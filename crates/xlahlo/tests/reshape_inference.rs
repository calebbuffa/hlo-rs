use xlahlo::{Dimension as D, Shape, inference, text};

fn shape(source: &str) -> Shape {
    text::parse_shape(source).unwrap()
}

#[test]
fn bounded_reshape_propagates_factor_groups_and_inferred_dimension_ties() {
    for (input, dimensions, inferred, expected) in [
        ("f32[<=6]", vec![2, 3], None, "f32[<=2,3]"),
        ("f32[<=6]", vec![2, 3], Some(1), "f32[2,<=3]"),
        ("f32[2,<=3]", vec![6], None, "f32[<=6]"),
        ("f32[<=2,<=3]", vec![6], None, "f32[<=6]"),
        ("f32[<=1]", vec![1, 1], None, "f32[1,<=1]"),
        ("f32[<=1]", vec![], None, "f32[]"),
        ("f32[1,<=4]", vec![4, 1], None, "f32[<=4,1]"),
        ("f32[<=2,3]", vec![2, 3], None, "f32[<=2,3]"),
        ("f32[<=0]", vec![0], None, "f32[<=0]"),
        ("f32[<=0]", vec![0, 2], None, "f32[<=0,2]"),
        ("f32[2,<=3]", vec![6], Some(usize::MAX), "f32[<=6]"),
        ("f32[6]", vec![2, 3], Some(usize::MAX), "f32[2,3]"),
        ("f32[<=1]", vec![1, 1], Some(usize::MAX), "f32[1,<=1]"),
        ("f32[<=6,5]", vec![2, 3, 5], Some(2), "f32[2,3,<=5]"),
        ("f32[0,<=6]", vec![0, 3, 2], None, "f32[0,<=3,2]"),
    ] {
        let actual = inference::infer_reshape(&shape(input), &dimensions, inferred).unwrap();
        assert!(
            actual.equal_including_layout(&shape(expected)),
            "{input} -> {dimensions:?}: {actual:?}"
        );
    }
    assert!(inference::infer_reshape(&shape("f32[<=6]"), &[2, 3], Some(3)).is_err());
    assert!(inference::infer_reshape(&shape("f32[2,<=3]"), &[3, 2], None).is_err());
    assert!(inference::infer_reshape(&shape("f32[2,<=1,3]"), &[2, 1, 3], None).is_err());
    assert!(inference::infer_reshape(&shape("f32[0,<=4]"), &[0, 2], None).is_err());
    assert!(inference::infer_static_reshape(&shape("f32[<=6]"), &[7]).is_err());
}

#[test]
fn unbounded_reshape_follows_the_separate_upstream_inference_branch() {
    let output = inference::infer_static_reshape(&shape("f32[?,4]"), &[3]).unwrap();
    assert!(output.equal_including_layout(&shape("f32[3]")));
    assert!(inference::infer_static_reshape(&shape("f32[?,<=4]"), &[3]).is_err());
    assert!(inference::infer_static_reshape(&shape("f32[?]"), &[i64::MIN]).is_err());
}

#[test]
fn reshape_verification_does_not_require_inferred_dynamic_flags_or_supported_inference_groups() {
    for (input, output, valid) in [
        ("f32[<=6]", "f32[2,3]", true),
        ("f32[<=6]", "f32[2,<=3]", true),
        ("f32[2,<=3]", "f32[3,2]", true),
        ("f32[2,<=1,3]", "f32[2,1,3]", true),
        ("f32[<=6]", "f32[<=7]", false),
        ("f32[<=6]", "f64[6]", false),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ p = {input} parameter(0) ROOT r = f32[6] reshape(p) }}"
        );
        let mut wire = text::parse_module(&source).unwrap().to_proto().unwrap();
        wire.computations[0]
            .instructions
            .iter_mut()
            .find(|i| i.name == "r")
            .unwrap()
            .shape = Some(shape(output).into_proto()).into();
        let module = xlahlo::Module::from_proto(wire).unwrap();
        assert_eq!(module.verify(Default::default()).is_ok(), valid, "{source}");
    }
}

#[test]
fn reshape_parser_checks_constructor_counts_without_requiring_inference_success() {
    for (input, output, valid) in [
        ("f32[6]", "f32[7]", false),
        ("f32[?,4]", "f32[3]", true),
        ("f32[6]", "f32[?,6]", true),
        ("f32[6]", "f32[?,7]", false),
        ("b(f32[6])", "b(f32[2,3])", true),
        ("f32[6]", "(f32[6])", false),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ p = {input} parameter(0) ROOT r = {output} reshape(p) }}"
        );
        assert_eq!(text::parse_module(&source).is_ok(), valid, "{source}");
    }
}

#[test]
fn transpose_preserves_physical_tiles_while_permuting_logical_axes() {
    let input = shape("f32[8,16]{1,0:T(4,8)SC(0:1,2)(1:1)}");
    let output = inference::infer_transpose(&input, &[1, 0]).unwrap();
    assert_eq!(output.dimensions(), [16, 8]);
    assert_eq!(output.layout().unwrap().minor_to_major, [0, 1]);
    assert_eq!(
        output.layout().unwrap().tiles,
        input.layout().unwrap().tiles
    );
    assert_eq!(
        output.layout().unwrap().split_configs,
        input.layout().unwrap().split_configs
    );
    let inverse = inference::infer_transpose(&output, &[1, 0]).unwrap();
    assert!(inverse.equal_including_layout(&input));
}

#[test]
fn reshape_and_transpose_reject_buffers_at_the_array_only_inference_guard() {
    let input = shape("b(f32[<=6]{0:T(2)})");
    assert!(inference::infer_reshape(&input, &[2, 3], None).is_err());
    let input = shape("b(f32[8,16]{1,0:T(4,8)SC(0:1,2)})");
    assert!(inference::infer_transpose(&input, &[1, 0]).is_err());
    let source =
        "HloModule m ENTRY main { p = b(f32[<=6]) parameter(0) ROOT r = f32[2,3] reshape(p) }";
    assert_eq!(
        text::parse_module(source)
            .unwrap()
            .verify(Default::default())
            .unwrap_err()
            .category,
        "buffer"
    );
}

#[test]
fn every_dynamic_axis_survives_small_single_group_merges_and_splits() {
    for first in 2..=8 {
        for second in 2..=8 {
            for axis in 0..2 {
                let mut dimensions = vec![D::Static(first), D::Static(second)];
                dimensions[axis] = D::Bounded(if axis == 0 { first } else { second });
                let input = Shape::dynamic_array(xlahlo::PrimitiveType::F32, dimensions).unwrap();
                let merged = inference::infer_reshape(&input, &[first * second], None).unwrap();
                assert_eq!(merged.dimension(0).unwrap(), D::Bounded(first * second));
            }
            let input =
                Shape::dynamic_array(xlahlo::PrimitiveType::F32, vec![D::Bounded(first * second)])
                    .unwrap();
            for inferred in [None, Some(0), Some(1)] {
                let split = inference::infer_reshape(&input, &[first, second], inferred).unwrap();
                let dynamic = inferred.unwrap_or(0);
                assert!(matches!(split.dimension(dynamic).unwrap(), D::Bounded(_)));
                assert!(matches!(
                    split.dimension(1 - dynamic).unwrap(),
                    D::Static(_)
                ));
            }
        }
    }
}
