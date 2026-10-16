use xlahlo::proto::xla_data::{Window, WindowDimension};
use xlahlo::{Dimension as D, PrimitiveType as P, Shape, inference, text};

fn window(size: i64, stride: i64, low: i64, high: i64, base: i64, kernel: i64) -> Window {
    Window {
        dimensions: vec![WindowDimension {
            size,
            stride,
            padding_low: low,
            padding_high: high,
            base_dilation: base,
            window_dilation: kernel,
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn window_geometry_matches_enumerated_small_sliding_windows() {
    for size in 0..8 {
        let input = Shape::array(P::F32, vec![size]).unwrap();
        for kernel in 1..5 {
            for low in -1..3 {
                for high in -1..3 {
                    for base_dilation in 1..3 {
                        for kernel_dilation in 1..3 {
                            for stride in 1..4 {
                                let config = window(
                                    kernel,
                                    stride,
                                    low,
                                    high,
                                    base_dilation,
                                    kernel_dilation,
                                );
                                let base = if size == 0 {
                                    0
                                } else {
                                    (size - 1) * base_dilation + 1
                                };
                                let padded = base + low + high;
                                let kernel = (kernel - 1) * kernel_dilation + 1;
                                let result =
                                    inference::infer_window_output(&input, &config, P::F32);
                                if padded < 0 {
                                    assert!(result.is_err());
                                } else {
                                    let count = (0..=padded)
                                        .filter(|position| {
                                            position % stride == 0 && position + kernel <= padded
                                        })
                                        .count()
                                        as i64;
                                    assert_eq!(result.unwrap().dimensions(), [count]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn window_attributes_round_trip_with_upstream_defaults() {
    let parsed = text::parse_window("size=2x3 pad=-1_2x0_1 rhs_reversal=1x0").unwrap();
    assert_eq!(parsed.dimensions[0].stride, 1);
    assert_eq!(parsed.dimensions[1].base_dilation, 1);
    assert!(parsed.dimensions[0].window_reversal);
    assert_eq!(
        text::parse_window(&text::print_window(&parsed).unwrap()).unwrap(),
        parsed
    );
    assert!(text::parse_window("size=2x3 stride=2").is_err());
    assert!(text::parse_window("size=2 pad=1_2_3").is_err());
    assert!(text::parse_window("size=2 unexpected=1").is_err());
    assert!(text::parse_window("").unwrap().dimensions.is_empty());
}

#[test]
fn window_inference_preserves_dynamic_bounds_and_rejects_overflow_and_bad_factors() {
    let input = Shape::dynamic_array(P::F32, vec![D::Bounded(5)]).unwrap();
    assert_eq!(
        inference::infer_window_output(&input, &window(3, 1, 0, 0, 1, 1), P::F32)
            .unwrap()
            .dimension(0)
            .unwrap(),
        D::Bounded(3)
    );
    let unbounded = Shape::dynamic_array(P::F32, vec![D::Unbounded]).unwrap();
    assert_eq!(
        inference::infer_window_output(&unbounded, &window(3, 1, 0, 0, 1, 1), P::F32)
            .unwrap()
            .dimension(0)
            .unwrap(),
        D::Unbounded
    );
    for config in [
        window(0, 1, 0, 0, 1, 1),
        window(1, 0, 0, 0, 1, 1),
        window(1, 1, 0, 0, 0, 1),
        window(1, 1, 0, 0, 1, 0),
        window(i64::MAX, 1, 0, 0, 1, 2),
        window(1, 1, i64::MAX, 0, 1, 1),
    ] {
        assert!(inference::infer_window_output(&input, &config, P::F32).is_err());
    }
    let mut unknown = window(3, 1, 0, 0, 1, 1);
    unknown.dimensions[0]
        .special_fields
        .mut_unknown_fields()
        .add_varint(999, 1);
    assert_eq!(
        text::print_window(&unknown).unwrap_err().category,
        "unsupported"
    );
}

#[test]
fn reduce_window_shares_reducer_checks_and_round_trips_complete_window_attributes() {
    let source = r#"
HloModule windowed
sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT s = f32[] add(a,b) }
ENTRY main { a = f32[5] parameter(0) initial = f32[] constant(0)
  ROOT result = f32[3] reduce-window(a,initial), window={size=3}, to_apply=sum }
"#;
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for invalid in [
        source.replace("window={size=3}", "window={size=3 stride=0}"),
        source.replace("window={size=3}", "window={size=3x3}"),
        source.replace("reduce-window(a,initial)", "reduce-window(a)"),
        source.replace("f32[3] reduce-window", "f32[4] reduce-window"),
    ] {
        assert!(
            text::parse_module(&invalid)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let reducer = module
        .computations()
        .find(|&id| module.computation(id).unwrap().name == "sum")
        .unwrap();
    let mut program = module.program_shape(reducer).unwrap();
    program.result = Some(
        Shape::tuple(vec![Shape::scalar(P::F32).unwrap()])
            .unwrap()
            .into_proto(),
    )
    .into();
    let result = inference::infer_reduce_window(
        &[
            Shape::array(P::F32, vec![5]).unwrap(),
            Shape::scalar(P::F32).unwrap(),
        ],
        &window(3, 1, 0, 0, 1, 1),
        &program,
    )
    .unwrap();
    assert!(result.is_tuple());
    assert_eq!(result.subshape(&[0]).unwrap().dimensions(), [3]);
}
