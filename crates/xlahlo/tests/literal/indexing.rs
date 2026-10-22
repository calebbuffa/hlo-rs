//! High-rank indexing and default-layout inference regressions.
use xlahlo::{
    attributes::{Axes, GatherDimensionNumbers, ScatterDimensionNumbers},
    inference,
    pliron::context::Context,
    types::{
        ElementType::{F32, S32},
        TensorType, tensor,
    },
};

#[test]
fn high_rank_gather_and_scatter_preserve_axis_partitions() {
    let ctx = Context::new();
    const RANK: usize = 32_768;
    let input = TensorType::get(&ctx, F32, vec![1; RANK]).unwrap();
    let axes = (0..RANK as i64).collect::<Vec<_>>();
    let even_axes = (0..RANK as i64).step_by(2).collect::<Vec<_>>();
    let half_axes = (0..(RANK / 2) as i64).collect::<Vec<_>>();
    for (collapsed, batching) in [
        (vec![], vec![]),
        (even_axes.clone(), vec![]),
        (vec![], even_axes.clone()),
    ] {
        let offset_dims = if collapsed.is_empty() && batching.is_empty() {
            axes.clone()
        } else {
            half_axes.clone()
        };
        let indices = TensorType::get(&ctx, S32, vec![1; batching.len() + 1]).unwrap();
        let result = inference::movement::infer_gather(
            &ctx,
            input,
            indices,
            &GatherDimensionNumbers {
                offset_dims: Axes(offset_dims.clone()),
                collapsed_slice_dims: Axes(collapsed),
                start_index_map: Axes(vec![1]),
                index_vector_dim: batching.len() as i64,
                start_indices_batching_dims: Axes((0..batching.len() as i64).rev().collect()),
                operand_batching_dims: Axes(batching.clone()),
            },
            &Axes(vec![1; RANK]),
        )
        .unwrap();
        assert_eq!(
            tensor(&ctx, result).unwrap().dimensions,
            vec![1; offset_dims.len() + batching.len()]
        );
    }
    for (inserted, batching) in [
        (vec![], vec![]),
        (even_axes.clone(), vec![]),
        (vec![], even_axes),
    ] {
        let update_window_dims = if inserted.is_empty() && batching.is_empty() {
            axes.clone()
        } else {
            half_axes.clone()
        };
        let indices = TensorType::get(&ctx, S32, vec![1; batching.len() + 1]).unwrap();
        let updates = TensorType::get(
            &ctx,
            F32,
            vec![1; update_window_dims.len() + batching.len()],
        )
        .unwrap();
        let result = inference::movement::infer_scatter(
            &ctx,
            input,
            indices,
            updates,
            &ScatterDimensionNumbers {
                update_window_dims: Axes(update_window_dims),
                inserted_window_dims: Axes(inserted),
                scatter_dims_to_operand_dims: Axes(vec![1]),
                index_vector_dim: batching.len() as i64,
                scatter_indices_batching_dims: Axes((0..batching.len() as i64).rev().collect()),
                input_batching_dims: Axes(batching),
            },
        )
        .unwrap();
        assert_eq!(result, input);
    }
}

#[test]
fn gather_inference_uses_default_layout_instead_of_copying_operand_layouts() {
    let ctx = Context::new();
    let input = TensorType::with_layout(&ctx, F32, vec![5, 4], vec![0, 1]).unwrap();
    let indices = TensorType::with_layout(&ctx, S32, vec![3, 1], vec![0, 1]).unwrap();
    let dimensions = GatherDimensionNumbers {
        offset_dims: Axes(vec![1]),
        collapsed_slice_dims: Axes(vec![0]),
        start_index_map: Axes(vec![0]),
        index_vector_dim: 1,
        operand_batching_dims: Axes(vec![]),
        start_indices_batching_dims: Axes(vec![]),
    };
    let result =
        inference::movement::infer_gather(&ctx, input, indices, &dimensions, &Axes(vec![1, 4]))
            .unwrap();
    let result = tensor(&ctx, result).unwrap();
    assert_eq!(result.dimensions, [3, 4]);
    assert_eq!(result.layout[0].minor_to_major, [1, 0]);
}
