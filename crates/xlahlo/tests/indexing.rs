use xlahlo::proto::xla_data::{
    GatherDimensionNumbers as G, ProgramShapeProto, ScatterDimensionNumbers as S,
};
use xlahlo::{Dimension as D, PrimitiveType as P, Shape, inference, text};

fn update_signature(elements: &[P]) -> ProgramShapeProto {
    let scalars: Vec<_> = elements
        .iter()
        .map(|&element| Shape::scalar(element).unwrap())
        .collect();
    let result = if scalars.len() == 1 {
        scalars[0].clone()
    } else {
        Shape::tuple(scalars.clone()).unwrap()
    };
    ProgramShapeProto {
        parameters: scalars
            .iter()
            .chain(&scalars)
            .map(|shape| shape.as_proto().clone())
            .collect(),
        result: Some(result.into_proto()).into(),
        ..Default::default()
    }
}

#[test]
fn gather_slice_window_and_index_batch_geometry() {
    for rows in 0..5 {
        for cols in 0..5 {
            let input = Shape::array(P::F32, vec![rows, cols]).unwrap();
            for slice in 0..=cols {
                let indices = Shape::array(P::S32, vec![3, 1]).unwrap();
                let dims = G {
                    offset_dims: vec![1],
                    collapsed_slice_dims: vec![0],
                    start_index_map: vec![0],
                    index_vector_dim: 1,
                    ..Default::default()
                };
                let result =
                    inference::infer_gather(&input, &indices, &dims, &[rows.min(1), slice])
                        .unwrap();
                assert_eq!(
                    result,
                    Shape::array(P::F32, vec![3, slice])
                        .unwrap()
                        .with_default_layout()
                        .unwrap()
                );
                let dims = G {
                    offset_dims: vec![0],
                    ..dims
                };
                assert_eq!(
                    inference::infer_gather(&input, &indices, &dims, &[rows.min(1), slice])
                        .unwrap()
                        .dimensions(),
                    [slice, 3]
                );
            }
        }
    }
    let input = Shape::dynamic_array(P::F32, vec![D::Static(5), D::Bounded(4)]).unwrap();
    let indices = Shape::dynamic_array(P::U32, vec![D::Bounded(3)]).unwrap();
    let dims = G {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        ..Default::default()
    };
    let result = inference::infer_gather(&input, &indices, &dims, &[1, 4]).unwrap();
    assert_eq!(result.dimension(0).unwrap(), D::Bounded(3));
    assert_eq!(result.dimension(1).unwrap(), D::Bounded(4));
    let result = inference::infer_gather(&input, &indices, &dims, &[1, 2]).unwrap();
    assert_eq!(result.dimension(1).unwrap(), D::Static(2));
    let batched = G {
        offset_dims: vec![2],
        operand_batching_dims: vec![0],
        start_indices_batching_dims: vec![0],
        start_index_map: vec![1],
        index_vector_dim: 2,
        ..Default::default()
    };
    let indices = Shape::array(P::S32, vec![5, 3, 1]).unwrap();
    assert_eq!(
        inference::infer_gather(&input, &indices, &batched, &[1, 2])
            .unwrap()
            .dimensions(),
        [5, 3, 2]
    );
}

#[test]
fn gather_rejects_invalid_axes_maps_types_and_slice_bounds() {
    let input = Shape::array(P::F32, vec![5, 4]).unwrap();
    let indices = Shape::array(P::S32, vec![3, 1]).unwrap();
    let dims = G {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        ..Default::default()
    };
    for invalid in [
        G {
            offset_dims: vec![2],
            ..dims.clone()
        },
        G {
            collapsed_slice_dims: vec![1, 0],
            ..dims.clone()
        },
        G {
            start_index_map: vec![0, 0],
            ..dims.clone()
        },
        G {
            start_index_map: vec![-1],
            ..dims.clone()
        },
        G {
            index_vector_dim: -1,
            ..dims.clone()
        },
        G {
            index_vector_dim: 3,
            ..dims.clone()
        },
        G {
            operand_batching_dims: vec![0],
            ..dims.clone()
        },
        G {
            offset_dims: vec![0, 0],
            ..dims.clone()
        },
    ] {
        assert!(inference::infer_gather(&input, &indices, &invalid, &[1, 4]).is_err());
    }
    for sizes in [vec![2, 4], vec![1, 5], vec![-1, 4], vec![1]] {
        assert!(inference::infer_gather(&input, &indices, &dims, &sizes).is_err());
    }
    assert!(
        inference::infer_gather(
            &input,
            &indices.with_element_type(P::F32).unwrap(),
            &dims,
            &[1, 4]
        )
        .is_err()
    );
}

#[test]
fn scatter_geometry_and_variadic_reducer_contracts() {
    let dims = S {
        update_window_dims: vec![1],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
        ..Default::default()
    };
    for rows in 0..4 {
        for cols in 0..4 {
            for update_cols in 0..=cols {
                let input = Shape::array(P::F32, vec![rows, cols]).unwrap();
                let indices = Shape::array(P::S32, vec![3, 1]).unwrap();
                let updates = Shape::array(P::F32, vec![3, update_cols]).unwrap();
                assert_eq!(
                    inference::infer_scatter(
                        &[input.clone(), indices, updates],
                        &dims,
                        &update_signature(&[P::F32])
                    )
                    .unwrap(),
                    input
                );
            }
        }
    }
    let f = Shape::array(P::F32, vec![5, 4]).unwrap();
    let i = f.with_element_type(P::S32).unwrap();
    let indices = Shape::array(P::S32, vec![3, 1]).unwrap();
    let uf = Shape::array(P::F32, vec![3, 4]).unwrap();
    let ui = uf.with_element_type(P::S32).unwrap();
    let args = vec![f.clone(), i.clone(), indices, uf, ui];
    assert_eq!(
        inference::infer_scatter(&args, &dims, &update_signature(&[P::F32, P::S32])).unwrap(),
        Shape::tuple(vec![f, i]).unwrap()
    );
    assert!(inference::infer_scatter(&args, &dims, &update_signature(&[P::F32])).is_err());
    let mut bad = args.clone();
    bad[4] = Shape::array(P::S32, vec![3, 5]).unwrap();
    assert!(inference::infer_scatter(&bad, &dims, &update_signature(&[P::F32, P::S32])).is_err());
    let mut bad = args.clone();
    bad[4] = Shape::array(P::S32, vec![2, 4]).unwrap();
    assert!(inference::infer_scatter(&bad, &dims, &update_signature(&[P::F32, P::S32])).is_err());
    for dims in [
        S {
            update_window_dims: vec![2],
            ..dims.clone()
        },
        S {
            inserted_window_dims: vec![2],
            ..dims.clone()
        },
        S {
            scatter_dims_to_operand_dims: vec![0, 0],
            ..dims.clone()
        },
        S {
            index_vector_dim: 3,
            ..dims.clone()
        },
        S {
            input_batching_dims: vec![0],
            ..dims.clone()
        },
    ] {
        assert!(
            inference::infer_scatter(&args, &dims, &update_signature(&[P::F32, P::S32])).is_err()
        );
    }
    for args in [
        vec![],
        args[..1].to_vec(),
        args[..2].to_vec(),
        args[..4].to_vec(),
    ] {
        assert!(inference::infer_scatter(&args, &dims, &update_signature(&[P::F32])).is_err());
    }
}

#[test]
fn gather_and_scatter_text_preserve_batching_and_hint_fields() {
    let gather = "HloModule m ENTRY main { a = f32[5,4] parameter(0) i = s32[3,1] parameter(1) ROOT g = f32[3,4] gather(a,i), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,4}, indices_are_sorted=true }";
    let scatter = "HloModule m add { a = f32[] parameter(0) b = f32[] parameter(1) ROOT s = f32[] add(a,b) } ENTRY main { a = f32[5,4] parameter(0) i = s32[3,1] parameter(1) u = f32[3,4] parameter(2) ROOT s = f32[5,4] scatter(a,i,u), to_apply=add, update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, indices_are_sorted=true, unique_indices=true }";
    let batched = "HloModule m ENTRY main { a = f32[5,4] parameter(0) i = s32[5,3,1] parameter(1) ROOT g = f32[5,3,2] gather(a,i), offset_dims={2}, collapsed_slice_dims={}, start_index_map={1}, index_vector_dim=2, slice_sizes={1,2}, operand_batching_dims={0}, start_indices_batching_dims={0} }";
    for source in [gather, scatter, batched] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        let reparsed = text::parse_module(&printed).unwrap();
        reparsed.verify(Default::default()).unwrap();
        assert_eq!(text::print_module(&reparsed).unwrap(), printed);
    }
    let batched_scatter = scatter
        .replace("f32[5,4]", "f32[5,6,4]")
        .replace("s32[3,1]", "s32[5,3,1]")
        .replace("f32[3,4]", "f32[5,3,4]")
        .replace("update_window_dims={1}", "update_window_dims={2}")
        .replace("inserted_window_dims={0}", "inserted_window_dims={1}")
        .replace(
            "scatter_dims_to_operand_dims={0}",
            "scatter_dims_to_operand_dims={1}",
        )
        .replace(
            "index_vector_dim=1",
            "index_vector_dim=2, input_batching_dims={0}, scatter_indices_batching_dims={0}",
        );
    let module = text::parse_module(&batched_scatter).unwrap();
    module.verify(Default::default()).unwrap();
    text::parse_module(&text::print_module(&module).unwrap())
        .unwrap()
        .verify(Default::default())
        .unwrap();
}
