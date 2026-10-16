use xlahlo::{Dimension as D, PrimitiveType as P, Shape, VerifyOptions, inference, text};

#[test]
fn scan_checks_sliced_parameters_carries_outputs_and_layout_policy() {
    let source = r#"
HloModule scans
body {
  x = f32[3] parameter(0)
  carry = f32[3] parameter(1)
  sum = f32[3] add(x,carry)
  ROOT result = (f32[3],f32[3]) tuple(sum,sum)
}
ENTRY main {
  x = f32[4,3] parameter(0)
  initial = f32[3] parameter(1)
  ROOT s = (f32[4,3],f32[3]) scan(x,initial), dimensions={0}, num_carries=1, to_apply=body, is_reverse=true
}
"#;
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    assert!(
        module
            .verify(VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            })
            .is_err()
    );
    let associative = text::parse_module(
        &source.replace("is_reverse=true", "is_reverse=true, is_associative=true"),
    )
    .unwrap();
    associative
        .verify(VerifyOptions {
            layout_sensitive: true,
            ..Default::default()
        })
        .unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("is_associative=false"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for source in [
        source.replace("num_carries=1", "num_carries=3"),
        source.replace("num_carries=1", "num_carries=-1"),
        source.replace("dimensions={0}", "dimensions={2}"),
        source.replace("dimensions={0}", "dimensions={}"),
        source.replace("x = f32[3] parameter(0)", "x = f32[2] parameter(0)"),
        source.replace("s = (f32[4,3],f32[3])", "s = (f32[5,3],f32[3])"),
        source.replace("initial = f32[3]", "initial = f32[2]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let body_id = associative
        .computations()
        .into_iter()
        .find(|&id| associative.computation(id).unwrap().name == "body")
        .unwrap();
    let signature = associative.program_shape(body_id).unwrap();
    let result = inference::infer_scan(
        &[
            Shape::array(P::F32, vec![4, 3]).unwrap(),
            Shape::array(P::F32, vec![3]).unwrap(),
        ],
        &signature,
        1,
        0,
        None,
    )
    .unwrap();
    assert_eq!(result.as_proto().tuple_shapes[0].dimensions, [4, 3]);
}

#[test]
fn scans_can_have_no_inputs_or_no_outputs() {
    for source in [
        "HloModule m body { c = f32[] parameter(0) ROOT r = (f32[],f32[]) tuple(c,c) } ENTRY main { c = f32[] parameter(0) ROOT s = (f32[5],f32[]) scan(c), dimensions={0}, num_carries=1, to_apply=body }",
        "HloModule m body { x = f32[] parameter(0) c = f32[] parameter(1) ROOT r = f32[] add(x,c) } ENTRY main { x = f32[5] parameter(0) c = f32[] parameter(1) ROOT s = f32[] scan(x,c), dimensions={0}, num_carries=1, to_apply=body }",
        "HloModule m body { ROOT r = f32[] constant(1) } ENTRY main { ROOT s = f32[5] scan(), dimensions={0}, num_carries=0, to_apply=body }",
    ] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
        let body_id = module
            .computations()
            .into_iter()
            .find(|&id| module.computation(id).unwrap().name == "body")
            .unwrap();
        let body = module.program_shape(body_id).unwrap();
        let args = body
            .parameters
            .iter()
            .cloned()
            .map(Shape::from_proto)
            .collect::<xlahlo::Result<Vec<_>>>()
            .unwrap();
        if args.len() <= 1 {
            let carries = args.len() as i64;
            let output =
                inference::infer_scan(&args, &body, carries, 0, Some(D::Static(5))).unwrap();
            assert_eq!(
                if output.is_tuple() {
                    output.as_proto().tuple_shapes[0].dimensions.clone()
                } else {
                    output.dimensions().to_vec()
                },
                [5]
            );
            assert!(inference::infer_scan(&args, &body, carries, 0, None).is_err());
        }
    }
}

#[test]
fn shuffle_rotate_preserves_shape_and_validates_attribute_cardinalities() {
    let source = "HloModule m ENTRY main { a = f32[2,3] parameter(0) ROOT s = f32[2,3] shuffle(a), dimensions={0,1}, mode=rotate, shifts={-1,99} }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for source in [
        source.replace("dimensions={0,1}", "dimensions={0,0}"),
        source.replace("dimensions={0,1}", "dimensions={}"),
        source.replace("shifts={-1,99}", "shifts={-1}"),
        source.replace("dimensions={0,1}", "dimensions={0,2}"),
        source.replace("s = f32[2,3]", "s = f32[3,2]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    assert!(text::parse_module(&source.replace("mode=rotate", "mode=unknown")).is_err());
}

#[test]
fn select_and_scatter_checks_both_callees_and_window_source_shape() {
    let source = r#"
HloModule select_scatter
select {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = pred[] compare(a,b), direction=GE
}
scatter {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] add(a,b)
}
ENTRY main {
  a = f32[5] parameter(0)
  source = f32[2] parameter(1)
  initial = f32[] parameter(2)
  ROOT r = f32[5] select-and-scatter(a,source,initial), window={size=2 stride=2}, select=select, scatter=scatter
}
"#;
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for source in [
        source.replace("source = f32[2]", "source = f32[3]"),
        source.replace("initial = f32[]", "initial = s32[]"),
        source.replace("ROOT r = pred[] compare", "ROOT r = f32[] compare"),
        source.replace("size=2 stride=2", "size=2 stride=1"),
        source.replace("select=select", "select=scatter"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
}
