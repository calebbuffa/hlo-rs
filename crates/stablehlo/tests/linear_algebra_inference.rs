use stablehlo::text::parse_type;
use stablehlo::{
    Attribute, Attributes, ElementType, InferenceError, StableHloAttribute, StableHloOp,
    TensorType, Type, TypeExtensions, infer_cholesky_type, infer_triangular_solve_type,
};

mod common;
use common::computation;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

#[test]
fn bounded_dynamic_matrix_witnesses_match_extent_intersection() {
    let witnesses = [
        (Some(0), None),
        (Some(1), None),
        (Some(2), None),
        (None, None),
        (None, Some(0)),
        (None, Some(1)),
        (None, Some(2)),
    ];
    let matrix = |axes: &[(Option<u64>, Option<u64>)]| {
        let dimensions = axes.iter().map(|&(size, _)| size).collect::<Vec<_>>();
        let bounds = axes
            .iter()
            .map(|&(_, bound)| bound.map_or(-1, |bound| bound as i64))
            .collect::<Vec<_>>();
        let mut tensor = TensorType::new(ElementType::F32, dimensions);
        if bounds.iter().any(|&bound| bound >= 0) {
            tensor = tensor.with_encoding(Attribute::StableHlo(Box::new(
                StableHloAttribute::TypeExtensions(TypeExtensions { bounds }),
            )));
        }
        Type::from(tensor)
    };
    for a_row in witnesses {
        for a_col in witnesses {
            for b_common in witnesses {
                for result_common in witnesses {
                    let valid = (0..=2).any(|size| {
                        [a_row, a_col, b_common, result_common].into_iter().all(
                            |(extent, bound)| {
                                extent.is_none_or(|extent| extent == size)
                                    && bound.is_none_or(|bound| size <= bound)
                            },
                        )
                    });
                    let a = matrix(&[a_row, a_col]);
                    for left in [true, false] {
                        let free = (Some(2), None);
                        let b = matrix(&if left {
                            [b_common, free]
                        } else {
                            [free, b_common]
                        });
                        let output = matrix(&if left {
                            [result_common, free]
                        } else {
                            [free, result_common]
                        });
                        assert_eq!(
                            computation(
                                StableHloOp::TriangularSolve,
                                vec![a.clone(), b],
                                vec![output],
                                solve_attributes(left)
                            )
                            .is_ok(),
                            valid,
                            "{a_row:?} {a_col:?} {b_common:?} {result_common:?} left={left}"
                        );
                    }
                    let output = matrix(&[b_common, result_common]);
                    assert_eq!(
                        computation(
                            StableHloOp::Cholesky,
                            vec![a],
                            vec![output],
                            Attributes::new()
                        )
                        .is_ok(),
                        valid,
                        "{a_row:?} {a_col:?} {b_common:?} {result_common:?}"
                    );
                }
            }
        }
    }
}

fn solve_attributes(left_side: bool) -> Attributes {
    [
        ("left_side".into(), Attribute::Bool(left_side)),
        ("lower".into(), Attribute::Bool(true)),
        ("unit_diagonal".into(), Attribute::Bool(false)),
        (
            "transpose_a".into(),
            Attribute::Enum {
                dialect: "stablehlo".into(),
                name: "transpose".into(),
                value: "NO_TRANSPOSE".into(),
            },
        ),
    ]
    .into()
}

fn cholesky(input: &str, output: &str) -> bool {
    computation(
        StableHloOp::Cholesky,
        vec![ty(input)],
        vec![ty(output)],
        Attributes::new(),
    )
    .is_ok()
}

fn solve(a: &str, b: &str, output: &str, left: bool) -> bool {
    computation(
        StableHloOp::TriangularSolve,
        vec![ty(a), ty(b)],
        vec![ty(output)],
        solve_attributes(left),
    )
    .is_ok()
}

#[test]
fn cholesky_retains_shapes_bounds_and_quantization_without_refining_them() {
    for source in [
        "tensor<0x0xf32>",
        "tensor<2x3x3xcomplex<f64>>",
        "tensor<?x3xf32, #stablehlo.bounds<8, ?>>",
        "tensor<?x?x?xf64, #stablehlo.bounds<4, 5, 6>>",
        "tensor<2x2x!quant.uniform<i8:f32, 0.5:1>>",
    ] {
        let input = ty(source);
        assert_eq!(infer_cholesky_type(&input).unwrap(), input);
        assert!(cholesky(source, source));
    }
    assert!(cholesky(
        "tensor<2x2x!quant.uniform<i8:f32, 0.5:1>>",
        "tensor<2x2x!quant.uniform<i8:f32, 0.25:2>>"
    ));
}

#[test]
fn matrix_helpers_reject_invalid_domains_ranks_and_bounds() {
    for source in ["tensor<f32>", "tensor<2xf32>"] {
        assert!(matches!(
            infer_cholesky_type(&ty(source)),
            Err(InferenceError::InvalidMatrixRank { .. })
        ));
    }
    for source in [
        "tensor<2x3xf32>",
        "tensor<?x4xf32, #stablehlo.bounds<3, ?>>",
        "tensor<4x?xf32, #stablehlo.bounds<?, 3>>",
        "tensor<2x2xi32>",
        "tensor<*xf32>",
        "!stablehlo.token",
        "tensor<2x2x!quant.uniform<i8:f32:0, {0.25, 0.5}>>",
    ] {
        assert!(infer_cholesky_type(&ty(source)).is_err(), "{source}");
    }
    for source in ["tensor<2x2xi32>", "tensor<2x2x!quant.uniform<i8:f32, 0.5>>"] {
        assert!(infer_triangular_solve_type(&ty(source), &ty(source), true).is_err());
    }
    assert!(
        infer_triangular_solve_type(&ty("tensor<2x2xf32>"), &ty("tensor<2x2xf64>"), true).is_err()
    );
}

#[test]
fn cholesky_checks_square_witnesses_across_dynamic_declared_results() {
    assert!(cholesky("tensor<?x?xf32>", "tensor<3x3xf32>"));
    assert!(cholesky("tensor<3x3xf32>", "tensor<?x?xf32>"));
    for (input, output) in [
        ("tensor<?x?xf32>", "tensor<3x4xf32>"),
        ("tensor<3x?xf32>", "tensor<?x4xf32>"),
        ("tensor<?x4xf32>", "tensor<3x?xf32>"),
        (
            "tensor<?x?xf32, #stablehlo.bounds<3, 5>>",
            "tensor<4x?xf32>",
        ),
        (
            "tensor<4x?xf32>",
            "tensor<?x?xf32, #stablehlo.bounds<3, 5>>",
        ),
        ("tensor<3x3xf32>", "tensor<3x3xf64>"),
    ] {
        assert!(!cholesky(input, output), "{input} -> {output}");
    }
}

#[test]
fn triangular_solve_preserves_rhs_metadata_for_both_sides() {
    let a = ty("tensor<?x3x3xf32, #stablehlo.bounds<4, ?, ?>>");
    for (left, source) in [
        (true, "tensor<?x?x2xf32, #stablehlo.bounds<5, 6, ?>>"),
        (false, "tensor<?x2x?xf32, #stablehlo.bounds<5, ?, 6>>"),
    ] {
        let b = ty(source);
        assert_eq!(infer_triangular_solve_type(&a, &b, left).unwrap(), b);
        assert!(
            computation(
                StableHloOp::TriangularSolve,
                vec![a.clone(), b.clone()],
                vec![b],
                solve_attributes(left)
            )
            .is_ok()
        );
    }
    assert_eq!(
        infer_triangular_solve_type(
            &ty("tensor<3x3xcomplex<f64>>"),
            &ty("tensor<3x2xcomplex<f64>>"),
            true
        )
        .unwrap(),
        ty("tensor<3x2xcomplex<f64>>")
    );
}

#[test]
fn triangular_solve_combines_all_common_extent_and_batch_witnesses() {
    for (a, b, left) in [
        ("tensor<3x?xf32>", "tensor<4x2xf32>", true),
        ("tensor<3x?xf32>", "tensor<2x4xf32>", false),
        (
            "tensor<?x?xf32, #stablehlo.bounds<3, 8>>",
            "tensor<4x2xf32>",
            true,
        ),
        ("tensor<2x3x3xf32>", "tensor<3x3x2xf32>", true),
        (
            "tensor<?x3x3xf32, #stablehlo.bounds<2, ?, ?>>",
            "tensor<3x3x2xf32>",
            true,
        ),
        (
            "tensor<3x3xf32>",
            "tensor<?x2xf32, #stablehlo.bounds<2, ?>>",
            true,
        ),
        ("tensor<3x3xf32>", "tensor<1x3x2xf32>", true),
    ] {
        assert!(
            infer_triangular_solve_type(&ty(a), &ty(b), left).is_err(),
            "{a}, {b}"
        );
        assert!(!solve(a, b, b, left));
    }
}

#[test]
fn triangular_solve_declared_results_cannot_hide_matrix_or_batch_contradictions() {
    for (a, b, output, left) in [
        (
            "tensor<3x3xf32>",
            "tensor<?x2xf32>",
            "tensor<4x2xf32>",
            true,
        ),
        (
            "tensor<3x?xf32>",
            "tensor<?x2xf32>",
            "tensor<4x2xf32>",
            true,
        ),
        (
            "tensor<3x3xf32>",
            "tensor<2x?xf32>",
            "tensor<2x4xf32>",
            false,
        ),
        (
            "tensor<2x3x3xf32>",
            "tensor<?x3x2xf32>",
            "tensor<4x3x2xf32>",
            true,
        ),
        (
            "tensor<3x3xf32>",
            "tensor<?x2xf32>",
            "tensor<?x2xf32, #stablehlo.bounds<2, ?>>",
            true,
        ),
        (
            "tensor<?x?xf32, #stablehlo.bounds<3, 3>>",
            "tensor<?x2xf32>",
            "tensor<4x2xf32>",
            true,
        ),
        (
            "tensor<3x3xf32>",
            "tensor<?x2xf32, #stablehlo.bounds<3, ?>>",
            "tensor<4x2xf32>",
            true,
        ),
    ] {
        assert!(!solve(a, b, output, left), "{a}, {b} -> {output}");
    }
    assert!(solve(
        "tensor<3x3xf32>",
        "tensor<3x2xf32>",
        "tensor<?x2xf32>",
        true
    ));
    assert!(solve(
        "tensor<3x3xf32>",
        "tensor<?x2xf32>",
        "tensor<3x2xf32>",
        true
    ));
}

#[test]
fn small_static_matrix_geometry_matches_equal_extent_equations() {
    let matrix = |x, y| Type::from(TensorType::static_shape(ElementType::F32, [x, y]));
    for rows in 0..4 {
        for cols in 0..4 {
            let a = matrix(rows, cols);
            assert_eq!(infer_cholesky_type(&a).is_ok(), rows == cols);
            for b_rows in 0..4 {
                for b_cols in 0..4 {
                    let b = matrix(b_rows, b_cols);
                    for left in [true, false] {
                        let valid = rows == cols && cols == if left { b_rows } else { b_cols };
                        assert_eq!(infer_triangular_solve_type(&a, &b, left).is_ok(), valid);
                        assert_eq!(
                            computation(
                                StableHloOp::TriangularSolve,
                                vec![a.clone(), b.clone()],
                                vec![b.clone()],
                                solve_attributes(left)
                            )
                            .is_ok(),
                            valid
                        );
                    }
                }
            }
        }
    }
}
