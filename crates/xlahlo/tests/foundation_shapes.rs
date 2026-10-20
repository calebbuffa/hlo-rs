use std::ptr;
use xlahlo::{
    Diagnostic,
    pliron::{common_traits::Verify, context::Context, result::ErrorKind, r#type::Type},
    primitive::PrimitiveType,
    shape::{
        COMBINE_DIMENSION, Dimension, Layout, MATCH_DIMENSION, Shape, SplitConfig, Tile,
        UNBOUNDED_SIZE,
    },
    types::{BufferType, ElementType as P, TensorType, TupleType, tensor},
};

#[test]
fn raw_malformed_array_queries_return_typed_framework_errors() {
    let ctx = Context::new();
    let valid = TensorType {
        element: P::F32,
        dimensions: vec![2],
        dynamic_dimensions: vec![false],
        layout: vec![Layout::descending(1)],
    };
    let mut invalid = vec![];
    let mut bad = valid.clone();
    bad.dynamic_dimensions.clear();
    invalid.push(bad);
    let mut bad = valid.clone();
    bad.dimensions[0] = -2;
    invalid.push(bad);
    let mut bad = valid.clone();
    bad.layout[0].minor_to_major[0] = 99;
    invalid.push(bad);
    let mut bad = valid.clone();
    bad.layout[0].minor_to_major[0] = -1;
    invalid.push(bad);
    let mut bad = valid;
    bad.layout.push(Layout::descending(1));
    invalid.push(bad);

    for raw in invalid {
        let ty = TensorType::instantiate(raw, &ctx).into();
        let error = tensor(&ctx, ty).unwrap_err();
        assert!(matches!(error.kind, ErrorKind::VerificationFailed));
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidShape { .. })
        ));
        assert!(Shape::dimension(&ctx, ty, 0).is_err());
        assert!(Shape::array_data_size(&ctx, ty).is_err());
    }
    let error = Shape::array(&ctx, P::F32, vec![-1]).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidShape { .. })
    ));
    assert!(Shape::array(&ctx, P::F32, vec![2]).is_ok());
}

#[test]
fn shape_queries_borrow_the_single_interned_representation() {
    let ctx = Context::new();
    let array = TensorType::get(&ctx, P::F32, vec![2, 3]).unwrap();
    let array_type = tensor(&ctx, array).unwrap();
    let dimensions = Shape::dimensions(&ctx, array).unwrap();
    assert!(ptr::eq(dimensions.as_ptr(), array_type.dimensions.as_ptr()));
    let layout = Shape::layout(&ctx, array).unwrap().unwrap();
    assert_eq!(array_type.layout.len(), 1);
    assert!(ptr::eq(&*layout, &array_type.layout[0]));
    assert_eq!(layout.minor_to_major, [1, 0]);

    let tuple = Shape::tuple(&ctx, vec![array, Shape::token(&ctx)]).unwrap();
    let tuple_type = tuple.deref(&ctx);
    let tuple_type = tuple_type.downcast_ref::<TupleType>().unwrap();
    let children = Shape::children(&ctx, tuple);
    assert!(ptr::eq(children.as_ptr(), tuple_type.elements.as_ptr()));
    assert_eq!(&*children, [array, Shape::token(&ctx)]);

    let buffer = Shape::buffer(&ctx, array).unwrap();
    let buffer_type = buffer.deref(&ctx);
    let buffer_type = buffer_type.downcast_ref::<BufferType>().unwrap();
    let children = Shape::children(&ctx, buffer);
    assert!(ptr::eq(children.as_ptr(), &buffer_type.array));
    assert!(Shape::children(&ctx, array).is_empty());
}

#[test]
fn every_primitive_has_the_correct_category() {
    let ctx = Context::new();
    for &ty in P::ALL {
        assert_eq!(P::from_hlo_name(&ty.hlo_name()).unwrap(), ty);
        let shape = Shape::array(&ctx, ty, vec![2, 0, 3]).unwrap();
        assert_eq!(Shape::element_count(&ctx, shape).unwrap(), 0);
        assert_eq!(ty.is_integer(), PrimitiveType::from(ty).is_integral());
        assert_eq!(ty.is_float(), PrimitiveType::from(ty).is_floating());
    }

    assert_eq!(P::C128.bit_width(), 128);
    assert_eq!(P::Pred.storage_bit_width(), 8);
    assert_eq!(P::F6E2M3FN.storage_bit_width(), 8);
    assert_eq!(P::U4.storage_bit_width(), 4);
    assert!(P::from_hlo_name("F32").is_err());
}

#[test]
fn every_primitive_has_the_correct_category_and_wire_number() {
    let ctx = Context::new();
    for &ty in PrimitiveType::VALUES {
        assert_eq!(PrimitiveType::from_i32(ty.value()), Some(ty));
        if ty == PrimitiveType::PRIMITIVE_TYPE_INVALID {
            continue;
        }
        assert_eq!(PrimitiveType::from_hlo_name(&ty.hlo_name()).unwrap(), ty);
        assert_eq!(ty.is_array_element(), ty.bit_width().is_some());
        if ty.is_array_element() {
            let shape = Shape::array(&ctx, ty, vec![2, 0, 3]).unwrap();
            assert_eq!(Shape::element_count(&ctx, shape).unwrap(), 0);
        } else {
            assert!(Shape::scalar(&ctx, ty).is_err());
        }
    }
    assert_eq!(PrimitiveType::BUFFER.value(), 34);
    assert_eq!(PrimitiveType::F6E2M3FN.value(), 36);
    assert_eq!(PrimitiveType::C128.bit_width(), Some(128));
    assert!(PrimitiveType::from_i32(999).is_none());
    for invalid in [
        PrimitiveType::PRIMITIVE_TYPE_INVALID,
        PrimitiveType::TUPLE,
        PrimitiveType::TOKEN,
        PrimitiveType::OPAQUE_TYPE,
        PrimitiveType::BUFFER,
    ] {
        assert!(invalid.higher_precision(PrimitiveType::F32).is_err());
        assert!(PrimitiveType::F32.higher_precision(invalid).is_err());
    }
}

#[test]
fn dynamic_dimensions_preserve_bounds_and_the_upstream_sentinel() {
    let ctx = Context::new();
    let shape = Shape::dynamic_array(
        &ctx,
        P::F32,
        vec![
            Dimension::Static(2),
            Dimension::Bounded(7),
            Dimension::Unbounded,
        ],
    )
    .unwrap();
    assert_eq!(&*Shape::dimensions(&ctx, shape).unwrap(), [2, 7, i64::MIN]);
    assert_eq!(UNBOUNDED_SIZE, i64::MIN);
    assert_eq!(
        Shape::dimension(&ctx, shape, 1).unwrap(),
        Dimension::Bounded(7)
    );
    assert_eq!(
        Shape::dimension(&ctx, shape, 2).unwrap(),
        Dimension::Unbounded
    );
    assert!(Shape::is_dynamic(&ctx, shape));
    assert!(Shape::element_count(&ctx, shape).is_err());
    assert!(Shape::array(&ctx, P::F32, vec![UNBOUNDED_SIZE]).is_err());
    assert!(Shape::dynamic_array(&ctx, P::F32, vec![Dimension::Bounded(-1)]).is_err());
    assert!(Shape::dynamic_array(&ctx, P::F32, vec![Dimension::Bounded(UNBOUNDED_SIZE)]).is_err());
    let static_shape = Shape::array(&ctx, P::F32, vec![7]).unwrap();
    let bounded = Shape::dynamic_array(&ctx, P::F32, vec![Dimension::Bounded(7)]).unwrap();
    assert!(Shape::compatible(&ctx, static_shape, bounded));
    assert!(!Shape::equal_ignoring_layout(&ctx, static_shape, bounded));
    assert!(!Shape::compatible(
        &ctx,
        static_shape,
        Shape::array(&ctx, P::F32, vec![6]).unwrap()
    ));
}

#[test]
fn malformed_shapes_are_errors_instead_of_invalid_canonical_values() {
    let ctx = Context::new();
    assert!(Shape::array(&ctx, P::U8, vec![i64::MAX, 2]).is_err());
    assert!(Shape::array(&ctx, P::F64, vec![i64::MAX]).is_err());
    assert!(Shape::array(&ctx, P::F32, vec![-1]).is_err());
    let scalar = Shape::scalar(&ctx, P::F64).unwrap();
    let tuple = Shape::tuple(
        &ctx,
        vec![
            Shape::token(&ctx),
            Shape::tuple(&ctx, vec![scalar]).unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(
        Shape::element_type(&ctx, Shape::subshape(&ctx, tuple, &[1, 0]).unwrap()).unwrap(),
        P::F64
    );
    assert!(Shape::subshape(&ctx, tuple, &[0, 0]).is_err());
    assert!(Shape::subshape(&ctx, tuple, &[2]).is_err());
    assert!(Shape::buffer(&ctx, tuple).is_err());
    assert!(Shape::rank(&ctx, Shape::opaque(&ctx)).is_err());
    assert_eq!(Shape::leaf_count(&ctx, tuple), 2);
    let mut invalid = xlahlo::types::tensor(&ctx, scalar).unwrap().clone();
    invalid.dynamic_dimensions = vec![true];
    assert!(invalid.verify(&ctx).is_err());
    let mut nested = scalar;
    for _ in 0..127 {
        nested = Shape::tuple(&ctx, vec![nested]).unwrap();
    }
    assert!(Shape::tuple(&ctx, vec![nested]).is_err());
}

#[test]
fn layout_validation_checks_permutation_tiles_and_physical_splits() {
    let ctx = Context::new();
    let shape = Shape::array(&ctx, P::F32, vec![4, 8]).unwrap();
    for permutation in [vec![1, 0], vec![0, 1]] {
        assert!(
            Shape::with_layout(
                &ctx,
                shape,
                Layout {
                    minor_to_major: permutation,
                    ..Layout::default()
                }
            )
            .is_ok()
        );
    }
    for permutation in [vec![0, 0], vec![2, 1], vec![-1, 0], vec![1]] {
        assert!(
            Shape::with_layout(
                &ctx,
                shape,
                Layout {
                    minor_to_major: permutation,
                    ..Layout::default()
                }
            )
            .is_err()
        );
    }
    for tile in [vec![0], vec![-2], vec![]] {
        assert!(
            Shape::with_layout(
                &ctx,
                shape,
                Layout {
                    minor_to_major: vec![1, 0],
                    tiles: vec![Tile { dimensions: tile }],
                    ..Layout::default()
                }
            )
            .is_err()
        );
    }
    let layout = Layout {
        minor_to_major: vec![1, 0],
        tiles: vec![Tile {
            dimensions: vec![COMBINE_DIMENSION, 8],
        }],
        ..Layout::default()
    };
    assert!(Shape::with_layout(&ctx, shape, layout).is_ok());
    for indices in [vec![1, 3], vec![3, 1], vec![0], vec![4]] {
        let layout = Layout {
            minor_to_major: vec![1, 0],
            split_configs: vec![SplitConfig {
                dimension: 0,
                split_indices: indices.clone(),
            }],
            ..Layout::default()
        };
        assert_eq!(
            Shape::with_layout(&ctx, shape, layout).is_ok(),
            indices == [1, 3]
        );
    }
    assert!(
        Shape::with_layout(
            &ctx,
            shape,
            Layout {
                minor_to_major: vec![1, 0],
                index_primitive_type: vec![P::U32],
                ..Layout::default()
            }
        )
        .is_err()
    );
    let memory_only = Shape::with_layout(
        &ctx,
        shape,
        Layout {
            memory_space: 5,
            ..Layout::default()
        },
    )
    .unwrap();
    assert!(Shape::array_data_size(&ctx, memory_only).is_err());
}

#[test]
fn layout_alignment_must_be_positive_except_for_auto_memory_layouts() {
    let ctx = Context::new();
    let shape = Shape::array(&ctx, P::F32, vec![4]).unwrap();
    for alignment in [i64::MIN, -1, 0, 1, 2, i64::MAX] {
        let layout = Layout {
            tail_padding_alignment_in_elements: alignment,
            ..Layout::descending(1)
        };
        let result = Shape::with_layout(&ctx, shape, layout.clone());
        assert_eq!(result.is_ok(), alignment >= 0);
        if alignment == 0 {
            assert_eq!(
                Shape::layout(&ctx, result.unwrap())
                    .unwrap()
                    .unwrap()
                    .tail_padding_alignment_in_elements,
                1
            );
        }
        let validation = xlahlo::shape::validate_layout(&layout, &[4]);
        assert_eq!(validation.is_ok(), alignment > 0);
        assert_eq!(
            TensorType {
                element: P::F32,
                dimensions: vec![4],
                dynamic_dimensions: vec![false],
                layout: vec![layout],
            }
            .verify(&ctx)
            .is_ok(),
            alignment > 0
        );
        if alignment <= 0 {
            let error = validation.unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<Diagnostic>(),
                Some(Diagnostic::InvalidShape { .. })
            ));
        }
    }
    let auto = Layout {
        memory_space: 5,
        tail_padding_alignment_in_elements: 0,
        ..Layout::default()
    };
    assert!(xlahlo::shape::validate_layout(&auto, &[4]).is_ok());
    assert!(
        TensorType {
            element: P::F32,
            dimensions: vec![4],
            dynamic_dimensions: vec![false],
            layout: vec![auto.clone()],
        }
        .verify(&ctx)
        .is_ok()
    );
    assert!(Shape::with_layout(&ctx, shape, auto).is_ok());
}

#[test]
fn layout_presence_and_exact_comparison_follow_pinned_semantics() {
    let ctx = Context::new();
    let a = Shape::scalar(&ctx, P::F32).unwrap();
    assert!(Shape::layout(&ctx, a).unwrap().is_none());
    let b = Shape::with_default_layout(&ctx, a).unwrap();
    assert!(Shape::layout(&ctx, b).unwrap().is_some());
    assert!(Shape::equal_ignoring_layout(&ctx, a, b));
    assert!(!Shape::equal_including_layout(&ctx, a, b));
    let mut layout = (*Shape::layout(&ctx, b).unwrap().unwrap()).clone();
    layout.dynamic_shape_metadata_prefix_bytes = 99;
    layout.dim_unique = vec![false];
    assert!(xlahlo::shape::layouts_equal(
        &ctx,
        &Shape::layout(&ctx, b).unwrap().unwrap(),
        &layout,
    ));
    let c = Shape::with_layout(&ctx, b, layout).unwrap();
    assert!(Shape::equal_including_layout(&ctx, b, c));
    assert_ne!(b, c);
    assert!(!Shape::compatible(
        &ctx,
        Shape::token(&ctx),
        Shape::opaque(&ctx)
    ));
    assert!(Shape::equal_including_layout(
        &ctx,
        Shape::buffer(&ctx, b).unwrap(),
        Shape::buffer(&ctx, c).unwrap()
    ));
    assert!(Shape::equal_ignoring_layout_and_fp_precision(
        &ctx,
        a,
        Shape::scalar(&ctx, P::F64).unwrap()
    ));
}

#[test]
fn byte_sizes_include_packed_elements_tuple_tables_and_buffers() {
    let ctx = Context::new();
    let shape = Shape::array(&ctx, P::U4, vec![3]).unwrap();
    assert_eq!(Shape::byte_size(&ctx, shape, -1).unwrap(), 3);
    let packed = Shape::with_layout(
        &ctx,
        shape,
        Layout {
            minor_to_major: vec![0],
            element_size_in_bits: 4,
            ..Layout::default()
        },
    )
    .unwrap();
    assert_eq!(Shape::byte_size(&ctx, packed, -1).unwrap(), 2);
    assert_eq!(
        Shape::byte_size(&ctx, Shape::buffer(&ctx, packed).unwrap(), -1).unwrap(),
        2
    );
    assert_eq!(Shape::byte_size(&ctx, Shape::token(&ctx), -1).unwrap(), 0);
    assert_eq!(Shape::byte_size(&ctx, Shape::opaque(&ctx), 8).unwrap(), 8);
    let tuple = Shape::tuple(&ctx, vec![shape, Shape::token(&ctx)]).unwrap();
    assert_eq!(Shape::byte_size(&ctx, tuple, 8).unwrap(), 16);
    assert!(Shape::byte_size(&ctx, tuple, -1).is_err());
}

#[test]
fn small_tiled_data_sizes_agree_with_block_and_in_block_coordinates() {
    let ctx = Context::new();
    for rows in 1_i64..=6 {
        for cols in 1..=6 {
            for tile_rows in 1..=4 {
                for tile_cols in 1..=4 {
                    for transpose in [false, true] {
                        let (major, minor) = if transpose {
                            (cols, rows)
                        } else {
                            (rows, cols)
                        };
                        let shape = Shape::with_layout(
                            &ctx,
                            Shape::array(&ctx, P::U8, vec![rows, cols]).unwrap(),
                            Layout {
                                minor_to_major: if transpose { vec![0, 1] } else { vec![1, 0] },
                                tiles: vec![Tile {
                                    dimensions: vec![tile_rows, tile_cols],
                                }],
                                ..Layout::default()
                            },
                        )
                        .unwrap();
                        let block = (major - 1) / tile_rows * ((minor + tile_cols - 1) / tile_cols)
                            + (minor - 1) / tile_cols;
                        let in_block =
                            (major - 1) % tile_rows * tile_cols + (minor - 1) % tile_cols;
                        assert_eq!(
                            Shape::array_data_size(&ctx, shape).unwrap(),
                            block * tile_rows * tile_cols + in_block + 1
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn data_size_distinguishes_outer_tile_gaps_from_trailing_padding() {
    let ctx = Context::new();
    for (dimensions, tiles, bits, expected) in [
        (vec![], vec![], 0, 4),
        (vec![3], vec![vec![8]], 0, 12),
        (vec![3, 3], vec![], 0, 36),
        (vec![3, 3], vec![vec![2, 2]], 0, 52),
        (vec![3, 3], vec![vec![2, 2], vec![8, 8]], 0, 52),
        (vec![16, 256], vec![vec![MATCH_DIMENSION, 128]], 0, 16384),
        (
            vec![3, 256],
            vec![vec![MATCH_DIMENSION, 128], vec![2, 1]],
            0,
            3584,
        ),
        (vec![64, 128], vec![], 4, 4096),
        (
            vec![9216, 6144],
            vec![vec![64, 128], vec![8, 1]],
            4,
            28311552,
        ),
    ] {
        let shape = Shape::with_layout(
            &ctx,
            Shape::array(&ctx, P::F32, dimensions.clone()).unwrap(),
            Layout {
                minor_to_major: (0..dimensions.len() as i64).rev().collect(),
                tiles: tiles
                    .into_iter()
                    .map(|dimensions| Tile { dimensions })
                    .collect(),
                element_size_in_bits: bits,
                ..Layout::default()
            },
        )
        .unwrap();
        assert_eq!(Shape::array_data_size(&ctx, shape).unwrap(), expected);
    }
    for dimensions in [vec![0], vec![0, 3], vec![3, 0], vec![2, 0, 3]] {
        let ty = TensorType::get(&ctx, P::F32, dimensions).unwrap();
        assert_eq!(Shape::array_data_size(&ctx, ty).unwrap(), 0);
    }
}

#[test]
fn reshape_bitcast_checks_dense_strides_not_tiles_or_padding() {
    let ctx = Context::new();
    let a = TensorType::get(&ctx, P::F32, vec![2, 3]).unwrap();
    let b = TensorType::get(&ctx, P::F32, vec![6]).unwrap();
    assert!(Shape::reshape_is_bitcast(&ctx, a, b, false).unwrap());
    let transposed = TensorType::with_layout(&ctx, P::F32, vec![2, 3], vec![0, 1]).unwrap();
    assert!(!Shape::reshape_is_bitcast(&ctx, transposed, b, false).unwrap());
    let u = TensorType::get(&ctx, P::U32, vec![6]).unwrap();
    assert!(!Shape::reshape_is_bitcast(&ctx, a, u, false).unwrap());
    assert!(Shape::reshape_is_bitcast(&ctx, a, u, true).unwrap());
    assert!(
        Shape::reshape_is_bitcast(&ctx, Shape::array(&ctx, P::F32, vec![6]).unwrap(), b, false)
            .is_err()
    );
}

#[test]
fn floating_precision_prioritizes_range_then_significand_then_storage() {
    let types = [
        P::F4E2M1FN,
        P::F6E2M3FN,
        P::F8E3M4,
        P::F6E3M2FN,
        P::F8E4M3B11FNUZ,
        P::F8E4M3,
        P::F8E4M3FN,
        P::F8E5M2,
        P::F16,
        P::F8E8M0FNU,
        P::BF16,
        P::F32,
        P::F64,
    ];
    for (i, &a) in types.iter().enumerate() {
        for (j, &b) in types.iter().enumerate() {
            assert_eq!(a.higher_precision(b).unwrap(), types[i.max(j)]);
        }
    }
    for (a, b) in [
        (P::F8E4M3, P::F8E4M3FNUZ),
        (P::F8E5M2, P::F8E5M2FNUZ),
        (P::Pred, P::U1),
    ] {
        assert!(a.higher_precision(b).is_err());
        assert!(b.higher_precision(a).is_err());
        assert_eq!(a.higher_precision(a).unwrap(), a);
    }
}

#[test]
fn all_array_primitive_pairs_are_symmetric_and_idempotent() {
    for &a in P::ALL {
        assert_eq!(a.higher_precision(a).unwrap(), a);
        for &b in P::ALL {
            assert_eq!(a.higher_precision(b).ok(), b.higher_precision(a).ok());
        }
    }
}

#[test]
fn complex_signedness_and_integer_width_use_upstream_lexicographic_priority() {
    for (a, b, expected) in [
        (P::C64, P::F64, P::C64),
        (P::C128, P::C64, P::C128),
        (P::S8, P::U8, P::S8),
        (P::S8, P::U16, P::U16),
        (P::S1, P::Pred, P::S1),
        (P::F4E2M1FN, P::S64, P::F4E2M1FN),
        (P::F8E8M0FNU, P::F16, P::F8E8M0FNU),
    ] {
        assert_eq!(a.higher_precision(b).unwrap(), expected);
        assert_eq!(b.higher_precision(a).unwrap(), expected);
    }
}
