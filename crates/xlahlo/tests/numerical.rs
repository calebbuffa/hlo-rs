use xlahlo::proto::xla_data::{DotDimensionNumbers, FftType as F};
use xlahlo::{Dimension as D, PrimitiveType as P, Shape, inference, text};

#[test]
fn sort_comparator_pairs_are_interleaved_by_operand_and_result_can_be_tuple() {
    let source = r#"
HloModule sorted
less {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = s32[] parameter(2)
  d = s32[] parameter(3)
  ROOT lt = pred[] compare(a,b), direction=LT
}
ENTRY main {
  keys = f32[2,3] parameter(0)
  values = s32[2,3] parameter(1)
  ROOT sorted = (f32[2,3],s32[2,3]) sort(keys,values), dimensions={1}, to_apply=less, is_stable=true
}
"#;
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("is_stable=true"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for source in [
        source.replace("values = s32[2,3]", "values = s32[2,4]"),
        source.replace("b = f32[] parameter", "b = s32[] parameter"),
        source.replace("dimensions={1}", "dimensions={2}"),
        source.replace("sort(keys,values)", "sort()"),
        source.replace("pred[] compare", "f32[] compare"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let single = "HloModule m less { a = s32[] parameter(0) b = s32[] parameter(1) ROOT c = pred[] compare(a,b), direction=LT } ENTRY main { a = s32[3] parameter(0) ROOT s = s32[3] sort(a), dimensions={0}, to_apply=less }";
    text::parse_module(single)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn fft_shapes_preserve_layout_and_distinguish_real_transform_geometry() {
    let input = Shape::array(P::F32, vec![2, 4, 6])
        .unwrap()
        .with_default_layout()
        .unwrap();
    let complex = inference::infer_fft(&input, F::RFFT, &[4, 6]).unwrap();
    assert_eq!(complex.dimensions(), [2, 4, 4]);
    assert_eq!(complex.element_type(), P::C64);
    assert_eq!(complex.layout(), input.layout());
    assert_eq!(
        inference::infer_fft(&complex, F::IRFFT, &[4, 6]).unwrap(),
        input
    );
    for kind in [F::FFT, F::IFFT] {
        // Upstream does not require FFT/IFFT lengths to match input extents.
        assert_eq!(inference::infer_fft(&complex, kind, &[1]).unwrap(), complex);
        assert!(inference::infer_fft(&input, kind, &[6]).is_err());
    }
    for lengths in [vec![], vec![1, 2, 3, 4], vec![5, 6], vec![4, 7]] {
        assert!(inference::infer_fft(&input, F::RFFT, &lengths).is_err());
    }
    let zero = Shape::array(P::F64, vec![0]).unwrap();
    let zero_complex = inference::infer_fft(&zero, F::RFFT, &[0]).unwrap();
    assert_eq!(zero_complex.dimensions(), [0]);
    assert_eq!(
        inference::infer_fft(&zero_complex, F::IRFFT, &[0]).unwrap(),
        zero
    );
    let dynamic = Shape::dynamic_array(P::F32, vec![D::Bounded(6)]).unwrap();
    assert_eq!(
        inference::infer_fft(&dynamic, F::RFFT, &[6])
            .unwrap()
            .dimension(0)
            .unwrap(),
        D::Bounded(4)
    );
    let unbounded = Shape::dynamic_array(P::F32, vec![D::Unbounded]).unwrap();
    assert_eq!(
        inference::infer_fft(&unbounded, F::RFFT, &[6])
            .unwrap()
            .dimension(0)
            .unwrap(),
        D::Static(4)
    );
}

#[test]
fn fft_assembly_round_trips_all_four_kinds_and_rejects_unknown_enums() {
    for (input, output, kind, length) in [
        ("c64[4]", "c64[4]", "FFT", 4),
        ("c128[4]", "c128[4]", "IFFT", 4),
        ("f32[6]", "c64[4]", "RFFT", 6),
        ("c128[4]", "f64[6]", "IRFFT", 6),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ p = {input} parameter(0) ROOT f = {output} fft(p), fft_type={kind}, fft_length={{{length}}} }}"
        );
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
        let mut wire = module.to_proto().unwrap();
        wire.computations[0]
            .instructions
            .iter_mut()
            .find(|i| i.opcode == "fft")
            .unwrap()
            .fft_type = protobuf::EnumOrUnknown::from_i32(999);
        assert!(
            xlahlo::Module::from_proto(wire)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
}

#[test]
fn dot_contraction_geometry_includes_scalar_outer_batched_and_dynamic_cases() {
    for m in 0..4 {
        for k in 0..4 {
            for n in 0..4 {
                let left = Shape::array(P::F32, vec![m, k]).unwrap();
                let right = Shape::array(P::F32, vec![k, n]).unwrap();
                let dimensions = DotDimensionNumbers {
                    lhs_contracting_dimensions: vec![1],
                    rhs_contracting_dimensions: vec![0],
                    ..Default::default()
                };
                let expected = Shape::array(P::F32, vec![m, n])
                    .unwrap()
                    .with_default_layout()
                    .unwrap();
                assert_eq!(
                    inference::infer_dot(&left, &right, &dimensions, None).unwrap(),
                    expected
                );
            }
        }
    }
    let left =
        Shape::dynamic_array(P::F32, vec![D::Bounded(2), D::Static(3), D::Static(4)]).unwrap();
    let right = Shape::array(P::F32, vec![2, 4, 5]).unwrap();
    let dimensions = DotDimensionNumbers {
        lhs_batch_dimensions: vec![0],
        rhs_batch_dimensions: vec![0],
        lhs_contracting_dimensions: vec![2],
        rhs_contracting_dimensions: vec![1],
        ..Default::default()
    };
    let result = inference::infer_dot(&left, &right, &dimensions, Some(P::F64)).unwrap();
    assert_eq!(result.dimensions(), [2, 3, 5]);
    assert_eq!(result.dimension(0).unwrap(), D::Bounded(2));
    assert_eq!(result.element_type(), P::F64);
    let scalar = Shape::scalar(P::F32).unwrap();
    assert_eq!(
        inference::infer_dot(&scalar, &scalar, &Default::default(), None)
            .unwrap()
            .rank()
            .unwrap(),
        0
    );
    assert_eq!(
        inference::infer_dot(&left, &scalar, &Default::default(), None)
            .unwrap()
            .dimensions(),
        left.dimensions()
    );
    for dimensions in [
        DotDimensionNumbers {
            lhs_contracting_dimensions: vec![2],
            ..Default::default()
        },
        DotDimensionNumbers {
            lhs_contracting_dimensions: vec![2],
            rhs_contracting_dimensions: vec![0],
            ..Default::default()
        },
        DotDimensionNumbers {
            lhs_contracting_dimensions: vec![2],
            rhs_contracting_dimensions: vec![1],
            lhs_batch_dimensions: vec![2],
            rhs_batch_dimensions: vec![0],
            ..Default::default()
        },
        DotDimensionNumbers {
            lhs_contracting_dimensions: vec![99],
            rhs_contracting_dimensions: vec![1],
            ..Default::default()
        },
    ] {
        assert!(inference::infer_dot(&left, &right, &dimensions, None).is_err());
    }
}

#[test]
fn dot_text_retains_dimension_numbers_precision_and_algorithm() {
    let source = r#"
HloModule dot
ENTRY main {
  a = f32[2,3,4] parameter(0)
  b = f32[2,4,5] parameter(1)
  ROOT d = f32[2,3,5] dot(a,b), lhs_contracting_dims={2}, rhs_contracting_dims={1},
    lhs_batch_dims={0}, rhs_batch_dims={0}, operand_precision={high,highest}, algorithm=dot_f32_f32_f32
}
"#;
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("operand_precision={high,highest}"));
    assert!(printed.contains("algorithm=dot_f32_f32_f32"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for invalid in [
        source.replace("rhs_contracting_dims={1}", "rhs_contracting_dims={2}"),
        source.replace("lhs_batch_dims={0}", "lhs_batch_dims={2}"),
        source.replace("dot(a,b)", "dot(a)"),
        source.replace("f32[2,3,5] dot", "f32[2,5,3] dot"),
    ] {
        assert!(
            text::parse_module(&invalid)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let outer = "HloModule outer ENTRY main { a = f32[2] parameter(0) b = f32[3] parameter(1) ROOT d = f32[2,3] dot(a,b) }";
    let module = text::parse_module(outer).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("operand_precision={default,default}"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn decompositions_preserve_shapes_and_require_square_compatible_matrices() {
    use xlahlo::proto::xla_data::{TriangularSolveOptions, triangular_solve_options::Transpose};
    let a = Shape::array(P::C128, vec![2, 3, 3]).unwrap();
    let b = Shape::array(P::C128, vec![2, 3, 5]).unwrap();
    let mut options = TriangularSolveOptions {
        left_side: true,
        lower: true,
        unit_diagonal: true,
        transpose_a: Transpose::ADJOINT.into(),
        ..Default::default()
    };
    assert_eq!(inference::infer_cholesky(&a).unwrap(), a);
    assert_eq!(
        inference::infer_triangular_solve(&a, &b, &options).unwrap(),
        b
    );
    options.left_side = false;
    assert!(inference::infer_triangular_solve(&a, &b, &options).is_err());
    let right = Shape::array(P::C128, vec![2, 5, 3]).unwrap();
    assert_eq!(
        inference::infer_triangular_solve(&a, &right, &options).unwrap(),
        right
    );
    for transpose in [
        protobuf::EnumOrUnknown::from(Transpose::TRANSPOSE_INVALID),
        protobuf::EnumOrUnknown::from_i32(99),
    ] {
        options.transpose_a = transpose;
        assert!(inference::infer_triangular_solve(&a, &right, &options).is_err());
    }
    for invalid in [
        Shape::array(P::S32, vec![3, 3]).unwrap(),
        Shape::array(P::F32, vec![3]).unwrap(),
        Shape::array(P::F32, vec![3, 4]).unwrap(),
        Shape::tuple(vec![a.clone()]).unwrap(),
    ] {
        assert!(inference::infer_cholesky(&invalid).is_err());
    }
    for source in [
        "HloModule m ENTRY main { a = f64[3,3]{0,1} parameter(0) ROOT c = f64[3,3]{0,1} cholesky(a), lower=true }",
        "HloModule m ENTRY main { a = c64[3,3] parameter(0) b = c64[3,5] parameter(1) ROOT s = c64[3,5] triangular-solve(a,b), left_side=true, lower=false, unit_diagonal=true, transpose_a=ADJOINT }",
    ] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        text::parse_module(&text::print_module(&module).unwrap())
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
}

#[test]
fn top_k_exhaustive_small_shapes_and_text_defaults() {
    for batch in 0..4 {
        for n in 0..6 {
            for k in 0..=n {
                let operand =
                    Shape::dynamic_array(P::F32, vec![D::Bounded(batch), D::Bounded(n)]).unwrap();
                let output = inference::infer_top_k(&operand, k).unwrap();
                let values = Shape::dynamic_array(P::F32, vec![D::Bounded(batch), D::Static(k)])
                    .unwrap()
                    .with_default_layout()
                    .unwrap();
                assert_eq!(
                    output,
                    Shape::tuple(vec![
                        values.clone(),
                        values.with_element_type(P::S32).unwrap()
                    ])
                    .unwrap()
                );
            }
            assert!(
                inference::infer_top_k(&Shape::array(P::F32, vec![n]).unwrap(), n + 1).is_err()
            );
        }
    }
    assert!(inference::infer_top_k(&Shape::scalar(P::F32).unwrap(), 0).is_err());
    assert!(inference::infer_top_k(&Shape::array(P::F32, vec![3]).unwrap(), -1).is_err());
    let source =
        "HloModule m ENTRY main { a = f32[5] parameter(0) ROOT t = (f32[2],s32[2]) topk(a), k=2 }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("largest=true"));
    assert!(printed.contains("is_stable=true"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    let module = text::parse_module(&format!(
        "{}, largest=false, is_stable=false }}",
        source.trim_end_matches('}')
    ))
    .unwrap();
    module.verify(Default::default()).unwrap();
    assert!(
        text::print_module(&module)
            .unwrap()
            .contains("largest=false")
    );
}

#[test]
fn batch_norm_training_inference_and_grad_share_feature_geometry() {
    use xlahlo::Opcode as O;
    let input =
        Shape::dynamic_array(P::F32, vec![D::Static(2), D::Bounded(3), D::Static(4)]).unwrap();
    let vector = Shape::array(P::F32, vec![3]).unwrap();
    let feature = Shape::dynamic_array(P::F32, vec![D::Bounded(3)])
        .unwrap()
        .with_default_layout()
        .unwrap();
    let expected = Shape::tuple(vec![input.clone(), feature.clone(), feature]).unwrap();
    let training = vec![input.clone(), vector.clone(), vector.clone()];
    assert_eq!(
        inference::infer_batch_norm(O::BatchNormTraining, &training, 1).unwrap(),
        expected
    );
    let params = vec![
        input.clone(),
        vector.clone(),
        vector.clone(),
        vector.clone(),
        vector.clone(),
    ];
    assert_eq!(
        inference::infer_batch_norm(O::BatchNormInference, &params, 1).unwrap(),
        input
    );
    let mut grad = params.clone();
    grad[4] = input.clone();
    assert_eq!(
        inference::infer_batch_norm(O::BatchNormGrad, &grad, 1).unwrap(),
        expected
    );
    for axis in [-1, 3] {
        assert!(inference::infer_batch_norm(O::BatchNormTraining, &training, axis).is_err());
    }
    for bad in [
        Shape::array(P::F32, vec![4]).unwrap(),
        Shape::scalar(P::F32).unwrap(),
        Shape::array(P::S32, vec![3]).unwrap(),
    ] {
        let mut params = training.clone();
        params[1] = bad;
        assert!(inference::infer_batch_norm(O::BatchNormTraining, &params, 1).is_err());
    }
    grad[4] = Shape::array(P::F32, vec![2, 3, 5]).unwrap();
    assert!(inference::infer_batch_norm(O::BatchNormGrad, &grad, 1).is_err());
    for (opcode, extra, output) in [
        ("batch-norm-training", "", "(f32[2,3],f32[3],f32[3])"),
        ("batch-norm-inference", ",v,v", "f32[2,3]"),
        ("batch-norm-grad", ",v,a", "(f32[2,3],f32[3],f32[3])"),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ a = f32[2,3] parameter(0) v = f32[3] parameter(1) ROOT n = {output} {opcode}(a,v,v{extra}), epsilon=0.001, feature_index=1 }}"
        );
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
        let mixed = source.replace("v = f32[3]", "v = f64[3]");
        assert!(
            text::parse_module(&mixed)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
}
