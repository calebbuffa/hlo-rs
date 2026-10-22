use super::super::*;

#[test]
fn dot_and_convolution_default_precision_share_audited_ordering() {
    let ctx = context();
    for (left, right, expected) in [
        (P::F16, P::BF16, P::BF16),
        (P::C64, P::F64, P::C64),
        (P::F8E4M3FN, P::F8E4M3, P::F8E4M3FN),
        (P::S8, P::U16, P::U16),
    ] {
        let lhs = a(&ctx, left, &[2, 3]);
        let rhs = a(&ctx, right, &[3, 4]);
        let dot =
            inference::infer_dot(&ctx, lhs, rhs, &dot_dims(&[1], &[0], &[], &[]), None).unwrap();
        assert_eq!(shape::element_type(&ctx, dot).unwrap(), expected);
        let d = ConvolutionDimensionNumbers {
            input_batch_dimension: 0,
            input_feature_dimension: 1,
            input_spatial_dimensions: Axes(vec![]),
            kernel_input_feature_dimension: 0,
            kernel_output_feature_dimension: 1,
            kernel_spatial_dimensions: Axes(vec![]),
            output_batch_dimension: 0,
            output_feature_dimension: 1,
            output_spatial_dimensions: Axes(vec![]),
        };
        let conv =
            inference::infer_convolution(&ctx, lhs, rhs, 1, 1, &Window(vec![]), &d, None).unwrap();
        assert_eq!(shape::element_type(&ctx, conv).unwrap(), expected);
    }
}
