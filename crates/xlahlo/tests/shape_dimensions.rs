use xlahlo::{
    Diagnostic,
    pliron::{
        context::Context,
        parsable::{Parsable, parse_from_str},
        printable::Printable,
        r#type::{Type, TypeHandle},
    },
    shape::{DimLevelType, Dimension, Layout, Shape, SplitConfig, Tile, UNBOUNDED_SIZE},
    types::{ElementType, TensorType, tensor},
};

#[test]
fn changing_dynamic_flags_preserves_bounds_formats_and_buffer_wrapping() {
    let ctx = Context::new();
    for &element in ElementType::ALL {
        let array = Shape::dynamic_array(
            &ctx,
            element,
            vec![
                Dimension::Static(0),
                Dimension::Bounded(3),
                Dimension::Unbounded,
            ],
        )
        .unwrap();
        for buffered in [false, true] {
            let source = if buffered {
                Shape::buffer(&ctx, array).unwrap()
            } else {
                array
            };
            for (axis, dynamic, expected) in [
                (0, false, Dimension::Static(0)),
                (0, true, Dimension::Bounded(0)),
                (1, false, Dimension::Static(3)),
                (1, true, Dimension::Bounded(3)),
                (2, true, Dimension::Unbounded),
            ] {
                let result = Shape::set_dynamic_dimension(&ctx, source, axis, dynamic).unwrap();
                let result_array = Shape::array_view(&ctx, result).unwrap();
                assert_eq!(Shape::is_array(&ctx, result), !buffered);
                assert_eq!(Shape::element_type(&ctx, result_array).unwrap(), element);
                assert_eq!(
                    Shape::dimension(&ctx, result_array, axis).unwrap(),
                    expected
                );
                for other in (0..3).filter(|&other| other != axis) {
                    assert_eq!(
                        Shape::dimension(&ctx, result_array, other).unwrap(),
                        Shape::dimension(&ctx, array, other).unwrap(),
                    );
                }
                assert_eq!(
                    Shape::set_dynamic_dimension(&ctx, result, axis, dynamic).unwrap(),
                    result,
                );
            }
            assert!(Shape::set_dynamic_dimension(&ctx, source, 2, false).is_err());
            assert_eq!(
                tensor(&ctx, array).unwrap().dynamic_dimensions,
                vec![false, true, true],
            );
        }
    }
}

#[test]
fn changing_dynamic_flags_preserves_layout_and_dimension_size_inference() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let layout = Layout {
        minor_to_major: vec![0, 1],
        split_configs: vec![SplitConfig {
            dimension: 1,
            split_indices: vec![1],
        }],
        tiles: vec![Tile {
            dimensions: vec![2, 4],
        }],
        tail_padding_alignment_in_elements: 16,
        element_size_in_bits: 4,
        memory_space: 5,
        dynamic_shape_metadata_prefix_bytes: 32,
        dim_level_types: vec![DimLevelType::Dense; 2],
        dim_unique: vec![true; 2],
        dim_ordered: vec![false; 2],
        ..Layout::default()
    };
    let array = Shape::array(&ctx, ElementType::BF16, vec![4, 8]).unwrap();
    let array = Shape::with_layout(&ctx, array, layout.clone()).unwrap();
    let size = Shape::scalar(&ctx, ElementType::S32).unwrap();
    for buffered in [false, true] {
        let source = if buffered {
            Shape::buffer(&ctx, array).unwrap()
        } else {
            array
        };
        let result = Shape::set_dynamic_dimension(&ctx, source, 1, true).unwrap();
        assert_eq!(*Shape::layout(&ctx, result).unwrap().unwrap(), layout);
        assert_eq!(
            xlahlo::inference::movement::infer_set_dimension_size(&ctx, source, size, 1).unwrap(),
            result,
        );
        assert_eq!(
            Shape::set_dynamic_dimension(&ctx, result, 1, false).unwrap(),
            source,
        );
        let text = result.disp(&ctx).to_string();
        let mut fresh = Context::new();
        xlahlo::register(&mut fresh);
        let parsed = parse_from_str(TypeHandle::parser(()), &mut fresh, &text).unwrap();
        parsed.deref(&fresh).verify(&fresh).unwrap();
        assert_eq!(parsed.disp(&fresh).to_string(), text);
    }
}

#[test]
fn invalid_dynamic_flag_updates_are_typed_and_never_repair_invalid_sources() {
    let ctx = Context::new();
    let array = Shape::array(&ctx, ElementType::F32, vec![2]).unwrap();
    for axis in [1, usize::MAX] {
        let error = Shape::set_dynamic_dimension(&ctx, array, axis, true).unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidShape { .. }),
        ));
    }
    for source in [
        Shape::scalar(&ctx, ElementType::F32).unwrap(),
        Shape::token(&ctx),
        Shape::opaque(&ctx),
        Shape::tuple(&ctx, vec![array]).unwrap(),
    ] {
        assert!(Shape::set_dynamic_dimension(&ctx, source, 0, true).is_err());
    }
    let raw: TypeHandle = TensorType::instantiate(
        TensorType {
            element: ElementType::F32,
            dimensions: vec![UNBOUNDED_SIZE],
            dynamic_dimensions: vec![false],
            layout: vec![],
        },
        &ctx,
    )
    .into();
    assert!(Shape::set_dynamic_dimension(&ctx, raw, 0, true).is_err());
    assert_eq!(&*Shape::dimensions(&ctx, array).unwrap(), &[2]);
    assert!(
        !raw.deref(&ctx)
            .downcast_ref::<TensorType>()
            .unwrap()
            .dynamic_dimensions[0]
    );
}

#[test]
fn replacing_bounds_preserves_dynamic_flags_formats_and_buffer_wrapping() {
    let ctx = Context::new();
    for &element in ElementType::ALL {
        let array = Shape::dynamic_array(
            &ctx,
            element,
            vec![
                Dimension::Static(2),
                Dimension::Bounded(3),
                Dimension::Unbounded,
            ],
        )
        .unwrap();
        for buffered in [false, true] {
            let source = if buffered {
                Shape::buffer(&ctx, array).unwrap()
            } else {
                array
            };
            for (axis, bound, expected) in [
                (0, 0, Dimension::Static(0)),
                (0, 12, Dimension::Static(12)),
                (1, 0, Dimension::Bounded(0)),
                (1, 6, Dimension::Bounded(6)),
                (1, UNBOUNDED_SIZE, Dimension::Unbounded),
                (2, 7, Dimension::Bounded(7)),
                (2, UNBOUNDED_SIZE, Dimension::Unbounded),
            ] {
                let result = Shape::set_dimension(&ctx, source, axis, bound).unwrap();
                let result_array = Shape::array_view(&ctx, result).unwrap();
                assert_eq!(Shape::is_array(&ctx, result), !buffered);
                assert_eq!(Shape::element_type(&ctx, result_array).unwrap(), element);
                assert_eq!(
                    Shape::dimension(&ctx, result_array, axis).unwrap(),
                    expected
                );
                for other in (0..3).filter(|&other| other != axis) {
                    assert_eq!(
                        Shape::dimension(&ctx, result_array, other).unwrap(),
                        Shape::dimension(&ctx, array, other).unwrap(),
                    );
                }
                assert!(Shape::layout(&ctx, result).unwrap().is_none());
                assert_eq!(
                    Shape::set_dimension(&ctx, result, axis, bound).unwrap(),
                    result
                );
                assert_eq!(
                    &*Shape::dimensions(&ctx, array).unwrap(),
                    &[2, 3, UNBOUNDED_SIZE]
                );
            }
        }
        let byte = Shape::array(&ctx, ElementType::U8, vec![1]).unwrap();
        let maximum = Shape::set_dimension(&ctx, byte, 0, i64::MAX).unwrap();
        assert_eq!(
            Shape::dimension(&ctx, maximum, 0).unwrap(),
            Dimension::Static(i64::MAX)
        );
    }
}

#[test]
fn replacing_bounds_preserves_complete_layout_and_follows_axis_deletion() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let layout = Layout {
        minor_to_major: vec![2, 0, 1],
        split_configs: vec![SplitConfig {
            dimension: 1,
            split_indices: vec![1],
        }],
        tiles: vec![Tile {
            dimensions: vec![2, 4],
        }],
        tail_padding_alignment_in_elements: 16,
        element_size_in_bits: 4,
        memory_space: 5,
        dynamic_shape_metadata_prefix_bytes: 32,
        dim_level_types: vec![DimLevelType::Dense; 3],
        dim_unique: vec![true; 3],
        dim_ordered: vec![false; 3],
        ..Layout::default()
    };
    let source = Shape::dynamic_array(
        &ctx,
        ElementType::BF16,
        vec![
            Dimension::Bounded(2),
            Dimension::Static(3),
            Dimension::Static(4),
        ],
    )
    .unwrap();
    let source = Shape::with_layout(&ctx, source, layout.clone()).unwrap();
    for buffered in [false, true] {
        let source = if buffered {
            Shape::buffer(&ctx, source).unwrap()
        } else {
            source
        };
        let result = Shape::set_dimension(&ctx, source, 0, 6).unwrap();
        assert_eq!(*Shape::layout(&ctx, result).unwrap().unwrap(), layout);
        assert!(Shape::set_dimension(&ctx, source, 0, 1).is_err());
        let mut fresh = Context::new();
        xlahlo::register(&mut fresh);
        let parsed = parse_from_str(
            TypeHandle::parser(()),
            &mut fresh,
            &result.disp(&ctx).to_string(),
        )
        .unwrap();
        assert_eq!(
            Shape::dimension(&fresh, Shape::array_view(&fresh, parsed).unwrap(), 0).unwrap(),
            Dimension::Bounded(6)
        );
        assert_eq!(
            parsed.disp(&fresh).to_string(),
            result.disp(&ctx).to_string()
        );

        let deleted = Shape::delete_dimension(&ctx, source, 1).unwrap();
        let retained_layout = Shape::layout(&ctx, deleted).unwrap().unwrap().clone();
        let merged = Shape::set_dimension(&ctx, deleted, 0, 6).unwrap();
        let merged_array = Shape::array_view(&ctx, merged).unwrap();
        assert_eq!(
            Shape::dimension(&ctx, merged_array, 0).unwrap(),
            Dimension::Bounded(6)
        );
        assert_eq!(
            Shape::dimension(&ctx, merged_array, 1).unwrap(),
            Dimension::Static(4)
        );
        assert_eq!(
            *Shape::layout(&ctx, merged).unwrap().unwrap(),
            retained_layout
        );
        assert_eq!(Shape::is_array(&ctx, merged), !buffered);
        assert_eq!(*Shape::layout(&ctx, source).unwrap().unwrap(), layout);
    }
}

#[test]
fn invalid_bound_updates_fail_without_repairing_or_mutating_source_types() {
    let ctx = Context::new();
    let source = Shape::array(&ctx, ElementType::F32, vec![2, 3]).unwrap();
    for (axis, bound) in [
        (0, -1),
        (0, UNBOUNDED_SIZE),
        (1, i64::MIN),
        (0, i64::MAX),
        (2, 4),
        (usize::MAX, 4),
    ] {
        let error = Shape::set_dimension(&ctx, source, axis, bound).unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidShape { .. })
        ));
        assert_eq!(&*Shape::dimensions(&ctx, source).unwrap(), &[2, 3]);
        assert_eq!(
            Shape::dimension(&ctx, Shape::set_dimension(&ctx, source, 0, 6).unwrap(), 0).unwrap(),
            Dimension::Static(6)
        );
    }
    let scalar = Shape::scalar(&ctx, ElementType::F32).unwrap();
    for shape in [
        scalar,
        Shape::token(&ctx),
        Shape::opaque(&ctx),
        Shape::tuple(&ctx, vec![source]).unwrap(),
    ] {
        assert!(Shape::set_dimension(&ctx, shape, 0, 1).is_err());
    }
    for raw in [
        TensorType {
            element: ElementType::F32,
            dimensions: vec![2, 3],
            dynamic_dimensions: vec![false],
            layout: vec![],
        },
        TensorType {
            element: ElementType::F32,
            dimensions: vec![-1, 3],
            dynamic_dimensions: vec![false, false],
            layout: vec![],
        },
        TensorType {
            element: ElementType::F32,
            dimensions: vec![2, 3],
            dynamic_dimensions: vec![false, false],
            layout: vec![Layout {
                minor_to_major: vec![0, 0],
                ..Layout::default()
            }],
        },
    ] {
        let shape = TensorType::instantiate(raw.clone(), &ctx).into();
        assert!(Shape::set_dimension(&ctx, shape, 0, 2).is_err());
        assert_eq!(
            *shape.deref(&ctx).downcast_ref::<TensorType>().unwrap(),
            raw
        );
    }
}

#[test]
fn deleting_dimensions_preserves_dynamic_classification_without_adding_layouts() {
    let ctx = Context::new();
    let source = Shape::dynamic_array(
        &ctx,
        ElementType::F32,
        vec![
            Dimension::Static(5),
            Dimension::Bounded(7),
            Dimension::Unbounded,
            Dimension::Static(0),
        ],
    )
    .unwrap();
    let result = Shape::delete_dimensions(&ctx, source, &[3, 0]).unwrap();
    assert_eq!(
        Shape::dimension(&ctx, result, 0).unwrap(),
        Dimension::Bounded(7)
    );
    assert_eq!(
        Shape::dimension(&ctx, result, 1).unwrap(),
        Dimension::Unbounded
    );
    assert!(Shape::layout(&ctx, result).unwrap().is_none());
    assert_eq!(Shape::element_type(&ctx, result).unwrap(), ElementType::F32);
    assert_eq!(Shape::delete_dimensions(&ctx, source, &[]).unwrap(), source);
    let scalar = Shape::delete_dimensions(&ctx, source, &[2, 0, 3, 1]).unwrap();
    assert_eq!(Shape::rank(&ctx, scalar).unwrap(), 0);
    assert!(!Shape::is_dynamic(&ctx, scalar));
    assert_eq!(Shape::rank(&ctx, source).unwrap(), 4);
}

#[test]
fn layout_deletion_remaps_physical_splits_but_preserves_other_metadata() {
    let ctx = Context::new();
    let physical = Shape::array(&ctx, ElementType::U8, vec![8]).unwrap();
    let mut layout = Layout {
        minor_to_major: vec![2, 0, 1],
        split_configs: vec![
            SplitConfig {
                dimension: 0,
                split_indices: vec![10],
            },
            SplitConfig {
                dimension: 1,
                split_indices: vec![20],
            },
            SplitConfig {
                dimension: 2,
                split_indices: vec![30],
            },
        ],
        tiles: vec![Tile {
            dimensions: vec![2, 4, 8],
        }],
        tail_padding_alignment_in_elements: 16,
        element_size_in_bits: 4,
        memory_space: 5,
        index_primitive_type: vec![ElementType::U32],
        pointer_primitive_type: vec![ElementType::U64],
        physical_shape: vec![physical],
        dynamic_shape_metadata_prefix_bytes: 32,
        dim_level_types: vec![DimLevelType::Dense, DimLevelType::Compressed],
        dim_unique: vec![true, false, true],
        dim_ordered: vec![false, true],
    };
    let mut expected = layout.clone();
    expected.minor_to_major = vec![1, 0];
    expected.split_configs = vec![
        SplitConfig {
            dimension: 0,
            split_indices: vec![10],
        },
        SplitConfig {
            dimension: 1,
            split_indices: vec![30],
        },
    ];
    layout.delete_dimension(0).unwrap();
    assert_eq!(layout, expected);
}

#[test]
fn dimension_deletion_matches_all_rank_four_layout_permutations_and_axis_subsets() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let array = Shape::array(&ctx, ElementType::F32, vec![5, 7, 9, 11]).unwrap();
    let original_dimensions = [5, 7, 9, 11];
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let order = [a, b, c, d];
                    if order
                        .iter()
                        .enumerate()
                        .any(|(i, axis)| order[..i].contains(axis))
                    {
                        continue;
                    }
                    let layout = Layout {
                        minor_to_major: order.to_vec(),
                        split_configs: (0..4)
                            .map(|axis| SplitConfig {
                                dimension: axis,
                                split_indices: vec![axis + 1],
                            })
                            .collect(),
                        tiles: vec![Tile {
                            dimensions: vec![2, 4],
                        }],
                        memory_space: 5,
                        dynamic_shape_metadata_prefix_bytes: 16,
                        dim_level_types: vec![DimLevelType::Dense; 4],
                        dim_unique: vec![true; 4],
                        dim_ordered: vec![false; 4],
                        ..Layout::default()
                    };
                    let source = Shape::with_layout(&ctx, array, layout.clone()).unwrap();
                    for mask in 0..16 {
                        let kept = (0..4)
                            .filter(|axis| mask & (1 << axis) == 0)
                            .collect::<Vec<_>>();
                        let deleted = (0..4)
                            .rev()
                            .filter(|axis| !kept.contains(axis))
                            .collect::<Vec<_>>();
                        let result = Shape::delete_dimensions(&ctx, source, &deleted).unwrap();
                        let expected_dimensions = kept
                            .iter()
                            .map(|&axis| original_dimensions[axis as usize])
                            .collect::<Vec<_>>();
                        assert_eq!(
                            &*Shape::dimensions(&ctx, result).unwrap(),
                            expected_dimensions
                        );
                        let mut expected = layout.clone();
                        expected.minor_to_major = order
                            .iter()
                            .filter_map(|axis| kept.iter().position(|kept| axis == kept))
                            .map(|axis| axis as i64)
                            .collect();
                        let retained_physical = order
                            .iter()
                            .rev()
                            .enumerate()
                            .filter_map(|(physical, axis)| {
                                kept.contains(axis).then_some(physical as i64)
                            })
                            .collect::<Vec<_>>();
                        expected.split_configs = layout
                            .split_configs
                            .iter()
                            .filter_map(|split| {
                                retained_physical
                                    .iter()
                                    .position(|&physical| physical == split.dimension)
                                    .map(|physical| SplitConfig {
                                        dimension: physical as i64,
                                        split_indices: split.split_indices.clone(),
                                    })
                            })
                            .collect();
                        assert_eq!(*Shape::layout(&ctx, result).unwrap().unwrap(), expected);
                        result.deref(&ctx).verify(&ctx).unwrap();
                        let text = result.disp(&ctx).to_string();
                        let parsed =
                            parse_from_str(TypeHandle::parser(()), &mut ctx, &text).unwrap();
                        assert_eq!(parsed, result);
                        assert_eq!(*Shape::layout(&ctx, source).unwrap().unwrap(), layout);
                    }
                }
            }
        }
    }
}

#[test]
fn deleting_buffer_axes_preserves_buffer_identity_and_array_layout() {
    let ctx = Context::new();
    let array = Shape::array(&ctx, ElementType::BF16, vec![2, 3, 4]).unwrap();
    let array = Shape::with_layout(
        &ctx,
        array,
        Layout {
            minor_to_major: vec![0, 2, 1],
            memory_space: 5,
            ..Layout::default()
        },
    )
    .unwrap();
    let source = Shape::buffer(&ctx, array).unwrap();
    let result = Shape::delete_dimension(&ctx, source, 1).unwrap();
    assert!(!Shape::is_array(&ctx, result));
    assert!(Shape::is_array_or_buffer(&ctx, result));
    let result_array = Shape::array_view(&ctx, result).unwrap();
    assert_eq!(&*Shape::dimensions(&ctx, result_array).unwrap(), &[2, 4]);
    let layout = Shape::layout(&ctx, result).unwrap().unwrap();
    assert_eq!(layout.minor_to_major, vec![0, 1]);
    assert_eq!(layout.memory_space, 5);
    assert_eq!(Shape::delete_dimensions(&ctx, source, &[]).unwrap(), source);
    assert!(Shape::filter_dimensions(&ctx, source, |_| true).is_err());
}

#[test]
fn filtering_visits_original_axes_once_in_order() {
    let ctx = Context::new();
    let source = Shape::array(&ctx, ElementType::F32, vec![1, 4, 1, 7]).unwrap();
    let mut visited = Vec::new();
    let result = Shape::filter_dimensions(&ctx, source, |axis| {
        visited.push(axis);
        Shape::dimension(&ctx, source, axis as usize).unwrap() != Dimension::Static(1)
    })
    .unwrap();
    assert_eq!(visited, vec![0, 1, 2, 3]);
    assert_eq!(&*Shape::dimensions(&ctx, result).unwrap(), &[4, 7]);
    let scalar = Shape::scalar(&ctx, ElementType::F32).unwrap();
    assert_eq!(
        Shape::filter_dimensions(&ctx, scalar, |_| panic!("scalar has no axes")).unwrap(),
        scalar
    );
}

#[test]
fn invalid_deletions_are_typed_errors_and_leave_inputs_unchanged() {
    let ctx = Context::new();
    let source = Shape::array(&ctx, ElementType::F32, vec![2, 3]).unwrap();
    for dimensions in [vec![-1], vec![2], vec![0, 0], vec![i64::MAX]] {
        let error = Shape::delete_dimensions(&ctx, source, &dimensions).unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidShape { .. })
        ));
        assert_eq!(&*Shape::dimensions(&ctx, source).unwrap(), &[2, 3]);
    }
    for shape in [
        Shape::token(&ctx),
        Shape::opaque(&ctx),
        Shape::tuple(&ctx, vec![source]).unwrap(),
    ] {
        assert!(Shape::delete_dimensions(&ctx, shape, &[]).is_err());
        assert!(Shape::filter_dimensions(&ctx, shape, |_| true).is_err());
    }
    for mut layout in [
        Layout::descending(2),
        Layout {
            minor_to_major: vec![0, 0],
            ..Layout::default()
        },
        Layout {
            minor_to_major: vec![0, 3],
            ..Layout::default()
        },
    ] {
        let before = layout.clone();
        assert!(layout.delete_dimension(-1).is_err());
        assert_eq!(layout, before);
        assert!(layout.delete_dimension(2).is_err());
        assert_eq!(layout, before);
    }
    let mut duplicate_layout = Layout {
        minor_to_major: vec![0, 0],
        ..Layout::default()
    };
    let before = duplicate_layout.clone();
    assert!(duplicate_layout.delete_dimension(0).is_err());
    assert_eq!(duplicate_layout, before);
    let incomplete_layout = Shape::with_layout(
        &ctx,
        source,
        Layout {
            memory_space: 5,
            ..Layout::default()
        },
    )
    .unwrap();
    assert!(Shape::delete_dimension(&ctx, incomplete_layout, 0).is_err());
    assert_eq!(
        Shape::delete_dimensions(&ctx, incomplete_layout, &[]).unwrap(),
        incomplete_layout,
    );
    let raw = TensorType::instantiate(
        TensorType {
            element: ElementType::F32,
            dimensions: vec![2, 3],
            dynamic_dimensions: vec![false],
            layout: vec![],
        },
        &ctx,
    )
    .into();
    assert!(Shape::delete_dimension(&ctx, raw, 0).is_err());
    assert!(tensor(&ctx, source).is_ok());
}
