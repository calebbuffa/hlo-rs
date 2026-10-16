use xlahlo::{Dimension as D, PrimitiveType as P, Shape, inference, text};

const REDUCE: &str = r#"
HloModule reduction
sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a,b)
}
ENTRY main {
  input = f32[2,3] parameter(0)
  initial = f32[] constant(0)
  ROOT sum = f32[3] reduce(input,initial), dimensions={0}, to_apply=sum
}
"#;

#[test]
fn scalar_reduction_supports_partial_total_and_empty_axes_and_round_trips() {
    for source in [
        REDUCE.to_owned(),
        REDUCE
            .replace("f32[3] reduce", "f32[] reduce")
            .replace("dimensions={0}", "dimensions={0,1}"),
        REDUCE
            .replace("f32[3] reduce", "f32[2,3] reduce")
            .replace("dimensions={0}", "dimensions={}"),
    ] {
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        assert!(printed.contains("dimensions="));
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
}

#[test]
fn reduction_rejects_malformed_axes_cardinality_and_reducer_contracts() {
    for source in [
        REDUCE.replace("dimensions={0}", "dimensions={0,0}"),
        REDUCE.replace("dimensions={0}", "dimensions={2}"),
        REDUCE.replace("reduce(input,initial)", "reduce(input)"),
        REDUCE.replace("reduce(input,initial)", "reduce()"),
        REDUCE.replace("b = f32[] parameter(1)", "b = s32[] parameter(1)"),
        REDUCE.replace(
            "initial = f32[] constant(0)",
            "initial = f32[1] constant({0})",
        ),
        REDUCE.replace("f32[3] reduce", "f32[4] reduce"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn variadic_reduction_has_accumulators_first_and_heterogeneous_tuple_results() {
    let source = r#"
HloModule variadic
sum {
  a = f32[] parameter(0)
  b = s32[] parameter(1)
  c = f32[] parameter(2)
  d = s32[] parameter(3)
  f = f32[] add(a,c)
  s = s32[] add(b,d)
  ROOT tuple = (f32[],s32[]) tuple(f,s)
}
ENTRY main {
  a = f32[2,3] parameter(0)
  b = s32[2,3] parameter(1)
  zero_f = f32[] constant(0)
  zero_s = s32[] constant(0)
  ROOT sum = (f32[3],s32[3]) reduce(a,b,zero_f,zero_s), dimensions={0}, to_apply=sum
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
        source.replace("b = s32[2,3]", "b = s32[2,4]"),
        source.replace("reduce(a,b,zero_f,zero_s)", "reduce(a,b,zero_s,zero_f)"),
        source.replace("parameter(2)", "parameter(4)"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
}

#[test]
fn reduction_inference_preserves_unreduced_first_operand_bounds() {
    let module = text::parse_module(REDUCE).unwrap();
    let reducer = module
        .computations()
        .find(|&id| module.computation(id).unwrap().name == "sum")
        .unwrap();
    let program = module.program_shape(reducer).unwrap();
    let input = Shape::dynamic_array(P::F32, vec![D::Bounded(2), D::Static(3)]).unwrap();
    let result =
        inference::infer_reduce(&[input, Shape::scalar(P::F32).unwrap()], &[1], &program).unwrap();
    assert_eq!(result.dimension(0).unwrap(), D::Bounded(2));
}

#[test]
fn map_supports_scalar_broadcasts_and_heterogeneous_equal_rank_inputs() {
    let scalar = r#"
HloModule mapped
sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT s = f32[] add(a,b) }
ENTRY main { a = f32[<=3] parameter(0) b = f32[] parameter(1) ROOT result = f32[<=3] map(a,b), to_apply=sum }
"#;
    let heterogeneous = r#"
HloModule mapped
check { a = f32[] parameter(0) b = s32[] parameter(1) ROOT p = pred[] is-finite(a) }
ENTRY main { a = f32[3] parameter(0) b = s32[3] parameter(1) ROOT result = pred[3] map(a,b), to_apply=check }
"#;
    for source in [scalar, heterogeneous] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
    for source in [
        scalar.replace("map(a,b)", "map(a)"),
        scalar.replace(
            "b = f32[] parameter(1) ROOT result",
            "b = f32[4] parameter(1) ROOT result",
        ),
        heterogeneous.replace("b = s32[3]", "b = s32[]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let module = text::parse_module(scalar).unwrap();
    let callee = module
        .computations()
        .find(|&id| module.computation(id).unwrap().name == "sum")
        .unwrap();
    let program = module.program_shape(callee).unwrap();
    let arguments = [
        Shape::array(P::F32, vec![2, 3]).unwrap(),
        Shape::scalar(P::F32).unwrap(),
    ];
    assert!(inference::infer_map(&arguments, &program, &[1, 0]).is_err());
    assert!(inference::infer_map(&arguments, &program, &[0]).is_err());
}
