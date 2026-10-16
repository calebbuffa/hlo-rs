use xlahlo::proto::xla_data::{
    DotDimensionNumbers, RaggedDotDimensionNumbers, SparsityConfig,
    sparsity_config::TensorSparsityConfig,
};
use xlahlo::{Dimension as D, PrimitiveType as P, Shape, VerifyOptions, inference, text};

const RAGGED: &str = "HloModule m ENTRY main { l = f32[2,3,4] parameter(0) r = f32[5,2,4,6] parameter(1) g = s32[2,5] parameter(2) ROOT d = f32[2,3,6] ragged-dot(l,r,g), lhs_batch_dims={0}, rhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_contracting_dims={2}, lhs_ragged_dims={1}, rhs_group_dims={0} }";
const SPARSE: &str = "HloModule m ENTRY main { l = f32[2,4] parameter(0) r = f32[8,3] parameter(1) m = u16[2,1] parameter(2) ROOT d = f32[2,3] dot(l,r,m), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity_config={lhs={sparsity=2x4 dimension=1 stride=1 idx=2}} }";
const SCALED: &str = "HloModule m ENTRY main { l = f32[4,8] parameter(0) r = f32[8,6] parameter(1) a = f16[2,2] parameter(2) b = s32[4,3] parameter(3) ROOT d = f32[4,6] scaled-dot(l,r,a,b), lhs_contracting_dims={1}, rhs_contracting_dims={0} }";

fn verify(source: &str) {
    let options = VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    };
    let module = text::parse_module(source).unwrap();
    module.verify(options).unwrap();
    text::parse_module(&text::print_module(&module).unwrap())
        .unwrap()
        .verify(options)
        .unwrap();
}

fn dimensions() -> DotDimensionNumbers {
    DotDimensionNumbers {
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![0],
        ..Default::default()
    }
}

#[test]
fn all_three_ragged_modes_round_trip() {
    verify(RAGGED);
    verify(
        &RAGGED
            .replace("r = f32[5,2,4,6]", "r = f32[2,4,6]")
            .replace("rhs_batch_dims={1}", "rhs_batch_dims={0}")
            .replace("rhs_contracting_dims={2}", "rhs_contracting_dims={1}")
            .replace("lhs_ragged_dims={1}", "lhs_ragged_dims={2}")
            .replace("rhs_group_dims={0}", "rhs_group_dims={}")
            .replace("d = f32[2,3,6]", "d = f32[5,2,3,6]"),
    );
    verify(
        &RAGGED
            .replace("r = f32[5,2,4,6]", "r = f32[2,4,6]")
            .replace("g = s32[2,5]", "g = f64[5]")
            .replace("rhs_batch_dims={1}", "rhs_batch_dims={0}")
            .replace("rhs_contracting_dims={2}", "rhs_contracting_dims={1}")
            .replace("lhs_ragged_dims={1}", "lhs_ragged_dims={0}")
            .replace("rhs_group_dims={0}", "rhs_group_dims={}"),
    );
    verify(&RAGGED.replace("g = s32[2,5]", "g = f64[5]"));
}

#[test]
fn small_ragged_geometries_match_all_three_mode_formulas() {
    for b in 0..4 {
        for m in 0..4 {
            for k in 0..4 {
                for g in 0..4 {
                    let lhs = Shape::array(P::F32, vec![b, m, k]).unwrap();
                    for (ragged, rhs_dims, rhs_batch, rhs_contract, groups, rhs_groups, expected) in [
                        (
                            1,
                            vec![g, b, k, 2],
                            1,
                            2,
                            vec![b, g],
                            vec![0],
                            vec![b, m, 2],
                        ),
                        (2, vec![b, k, 2], 0, 1, vec![b, g], vec![], vec![g, b, m, 2]),
                        (0, vec![b, k, 2], 0, 1, vec![g], vec![], vec![b, m, 2]),
                    ] {
                        let config = RaggedDotDimensionNumbers {
                            dot_dimension_numbers: Some(DotDimensionNumbers {
                                lhs_batch_dimensions: vec![0],
                                rhs_batch_dimensions: vec![rhs_batch],
                                lhs_contracting_dimensions: vec![2],
                                rhs_contracting_dimensions: vec![rhs_contract],
                                ..Default::default()
                            })
                            .into(),
                            lhs_ragged_dimensions: vec![ragged],
                            rhs_group_dimensions: rhs_groups,
                            ..Default::default()
                        };
                        let rhs = Shape::array(P::F32, rhs_dims).unwrap();
                        let groups = Shape::array(P::S32, groups).unwrap();
                        let result =
                            inference::infer_ragged_dot(&lhs, &rhs, &groups, &config, None)
                                .unwrap();
                        assert_eq!(result.dimensions(), expected);
                    }
                }
            }
        }
    }
}

#[test]
fn ragged_group_shape_and_dimension_constraints_reject_single_changes() {
    for source in [
        RAGGED.replace("lhs_ragged_dims={1}", "lhs_ragged_dims={}"),
        RAGGED.replace("lhs_ragged_dims={1}", "lhs_ragged_dims={1,2}"),
        RAGGED.replace("lhs_ragged_dims={1}", "lhs_ragged_dims={3}"),
        RAGGED.replace("rhs_group_dims={0}", "rhs_group_dims={1}"),
        RAGGED.replace("rhs_group_dims={0}", "rhs_group_dims={0,0}"),
        RAGGED.replace("rhs_group_dims={0}", "rhs_group_dims={}"),
        RAGGED.replace("g = s32[2,5]", "g = s32[]"),
        RAGGED.replace("g = s32[2,5]", "g = s32[3,5]"),
        RAGGED.replace("g = s32[2,5]", "g = s32[2,6]"),
        RAGGED.replace("g = s32[2,5]", "g = s32[1,2,5]"),
        RAGGED.replace("lhs_contracting_dims={2}", "lhs_contracting_dims={0}"),
        RAGGED.replace("d = f32[2,3,6]", "d = f32[2,5,3,6]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err(),
            "{source}"
        );
    }
    assert!(text::parse_module(&RAGGED.replace("ragged-dot(l,r,g)", "ragged-dot(l,r)")).is_err());
}

#[test]
fn ragged_dynamic_group_count_and_nested_noncontracting_prefix_are_preserved() {
    let mut dot = dimensions();
    dot.lhs_batch_dimensions = vec![0];
    dot.rhs_batch_dimensions = vec![0];
    dot.lhs_contracting_dimensions = vec![2];
    dot.rhs_contracting_dimensions = vec![1];
    let config = RaggedDotDimensionNumbers {
        dot_dimension_numbers: Some(dot).into(),
        lhs_ragged_dimensions: vec![2],
        ..Default::default()
    };
    let result = inference::infer_ragged_dot(
        &Shape::array(P::F32, vec![2, 3, 4]).unwrap(),
        &Shape::array(P::F32, vec![2, 4, 6]).unwrap(),
        &Shape::dynamic_array(P::S32, vec![D::Static(2), D::Bounded(5)]).unwrap(),
        &config,
        None,
    )
    .unwrap();
    assert_eq!(result.dimension(0).unwrap(), D::Bounded(5));
    assert_eq!(result.dimensions(), [5, 2, 3, 6]);
    verify(
        &RAGGED
            .replace("l = f32[2,3,4]", "l = f32[2,3,7,4]")
            .replace("lhs_contracting_dims={2}", "lhs_contracting_dims={3}")
            .replace("lhs_ragged_dims={1}", "lhs_ragged_dims={2}")
            .replace("g = s32[2,5]", "g = s32[2,3,5]")
            .replace("d = f32[2,3,6]", "d = f32[2,3,7,6]"),
    );
}

#[test]
fn excluded_ragged_group_axis_does_not_cause_spurious_output_overflow() {
    let lhs = Shape::array(P::PRED, vec![1, 0]).unwrap();
    let rhs = Shape::array(P::PRED, vec![i64::MAX, 0, i64::MAX]).unwrap();
    let groups = Shape::array(P::PRED, vec![i64::MAX]).unwrap();
    let config = RaggedDotDimensionNumbers {
        dot_dimension_numbers: Some(DotDimensionNumbers {
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![1],
            ..Default::default()
        })
        .into(),
        lhs_ragged_dimensions: vec![0],
        rhs_group_dimensions: vec![0],
        ..Default::default()
    };
    assert_eq!(
        inference::infer_ragged_dot(&lhs, &rhs, &groups, &config, None)
            .unwrap()
            .dimensions(),
        [1, i64::MAX]
    );
}

#[test]
fn sparse_dot_expands_only_configured_contracting_dimensions() {
    verify(SPARSE);
    verify(
        &SPARSE
            .replace("l = f32[2,4]", "l = f32[2,8]")
            .replace("r = f32[8,3]", "r = f32[4,3]")
            .replace("2x4", "4x2"),
    );
    verify(
        &SPARSE
            .replace("r = f32[8,3]", "r = f32[4,3]")
            .replace("2x4 dimension=1 stride=1", "0x0 dimension=-1 stride=-3"),
    );
    let lhs = Shape::array(P::F32, vec![2, 4]).unwrap();
    let rhs = Shape::array(P::F32, vec![8, 3]).unwrap();
    let config = SparsityConfig {
        lhs: Some(TensorSparsityConfig {
            num_non_zero: 2,
            block_size: 4,
            dimension: 1,
            ..Default::default()
        })
        .into(),
        ..Default::default()
    };
    assert_eq!(
        inference::infer_dot_with_sparsity(&lhs, &rhs, &dimensions(), None, &config)
            .unwrap()
            .dimensions(),
        [2, 3]
    );
    let buffer = Shape::buffer(lhs).unwrap();
    assert_eq!(
        inference::infer_dot_with_sparsity(&buffer, &rhs, &dimensions(), None, &config)
            .unwrap()
            .dimensions(),
        [2, 3]
    );
}

#[test]
fn sparse_geometry_and_auxiliary_operand_references_are_checked() {
    for source in [
        SPARSE.replace("2x4", "0x4"),
        SPARSE.replace("2x4", "2x0"),
        SPARSE.replace("2x4", "3x4"),
        SPARSE.replace("idx=2", "idx=1"),
        SPARSE.replace("idx=2", "idx=3"),
        SPARSE.replace("idx=2", "idx=-1"),
        SPARSE.replace("r = f32[8,3]", "r = f32[7,3]"),
        SPARSE.replace("dot(l,r,m)", "dot(l,r,m,m)"),
        SPARSE.replace("u16[2,1]", "token[]"),
        SPARSE.replace(
            "stride=1 idx=2",
            "stride=1 idx=2} rhs={sparsity=2x4 dimension=0 idx=2",
        ),
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
fn block_scaling_tracks_all_scales_and_optional_zero_points_without_inventing_geometry() {
    let source = SPARSE.replace("l = f32[2,4]", "l = f32[2,8]").replace(
        "sparsity_config={lhs={sparsity=2x4 dimension=1 stride=1 idx=2}}",
        "block_scaling_config={lhs={scale_idx=2 strides=3x7 steps=-1x0}}",
    );
    verify(&source);
    let with_zero = source
        .replace("dot(l,r,m)", "dot(l,r,m,m)")
        .replace("scale_idx=2", "scale_idx=2 zero_idx=3");
    verify(&with_zero);
    for source in [
        with_zero.replace("zero_idx=3", "zero_idx=2"),
        with_zero.replace("zero_idx=3", "zero_idx=4"),
        source.replace("scale_idx=2", "scale_idx=0"),
        source.replace("dot(l,r,m)", "dot(l,r,m,m)"),
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
fn scaled_dot_geometry_and_precision_round_trip() {
    verify(SCALED);
    verify(&SCALED.replace(
        "rhs_contracting_dims={0}",
        "rhs_contracting_dims={0}, operand_precision={high,highest}, algorithm=dot_f32_f32_f32",
    ));
    for source in [
        SCALED.replace("a = f16[2,2]", "a = f16[3,2]"),
        SCALED.replace("a = f16[2,2]", "a = f16[0,2]"),
        SCALED.replace("a = f16[2,2]", "a = f16[2]"),
        SCALED.replace("a = f16[2,2]", "a = f16[1,2,2]"),
        SCALED.replace("lhs_contracting_dims={1}", "lhs_contracting_dims={}"),
        SCALED.replace("r = f32[8,6]", "r = f32[9,6]"),
        SCALED.replace("d = f32[4,6]", "d = f32[4,7]"),
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
fn bf16_dummy_scales_follow_the_pinned_shape_only_exception() {
    let source = SCALED
        .replace("f32", "bf16")
        .replace("a = f16[2,2]", "a = bf16[]")
        .replace("b = s32[4,3]", "b = bf16[1,1,1]");
    verify(&source);
    assert!(
        text::parse_module(&source.replace("b = bf16[1,1,1]", "b = bf16[]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    assert!(
        text::parse_module(&source.replace("a = bf16[]", "a = f16[]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}

#[test]
fn sparse_and_block_scaled_convolution_share_auxiliary_operand_contracts() {
    let source = SPARSE
        .replace(" dot(l,r,m)", " convolution(l,r,m)")
        .replace(
            "lhs_contracting_dims={1}, rhs_contracting_dims={0}",
            "dim_labels=bf_io->bf",
        );
    verify(&source);
    verify(
        &source
            .replace("l = f32[2,4]", "l = f32[2,8]")
            .replace("r = f32[8,3]", "r = f32[4,3]")
            .replace(
                "lhs={sparsity=2x4 dimension=1",
                "rhs={sparsity=2x4 dimension=0",
            ),
    );
    verify(&source.replace("l = f32[2,4]", "l = f32[2,8]").replace(
        "sparsity_config={lhs={sparsity=2x4 dimension=1 stride=1 idx=2}}",
        "block_scaling_config={lhs={scale_idx=2 strides=4x4 steps=1x1}}",
    ));
    for source in [
        source.replace("idx=2", "idx=1"),
        source.replace("2x4", "0x4"),
        source.replace("convolution(l,r,m)", "convolution(l,r,m,m)"),
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
