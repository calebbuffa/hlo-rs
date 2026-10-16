use super::InferenceError;

pub(super) fn output_extent(
    size: u64,
    window: u64,
    stride: i64,
    base_dilation: i64,
    window_dilation: i64,
    padding: [i128; 2],
) -> i128 {
    let dilated = |size: u64, dilation: i64| {
        if size == 0 {
            0
        } else {
            (i128::from(size) - 1) * i128::from(dilation) + 1
        }
    };
    let padded = padding[0] + dilated(size, base_dilation) + padding[1];
    let window = dilated(window, window_dilation);
    if padded <= 0 || padded < window {
        0
    } else {
        (padded - window) / i128::from(stride) + 1
    }
}

pub(crate) fn output_shape(
    input: &[Option<u64>],
    window: &[Option<u64>],
    strides: &[i64],
    base_dilations: &[i64],
    window_dilations: &[i64],
    padding: &[Option<[i128; 2]>],
) -> Result<Vec<Option<u64>>, InferenceError> {
    let rank = input.len();
    for (attribute, length) in [
        ("window_dimensions", window.len()),
        ("window_strides", strides.len()),
        ("base_dilations", base_dilations.len()),
        ("window_dilations", window_dilations.len()),
        ("padding", padding.len()),
    ] {
        if length != rank {
            return Err(InferenceError::AttributeLengthMismatch {
                attribute,
                expected: rank,
                actual: length,
            });
        }
    }
    for (attribute, values) in [
        ("window_strides", strides),
        ("base_dilations", base_dilations),
        ("window_dilations", window_dilations),
    ] {
        for (dimension, &value) in values.iter().enumerate() {
            if value <= 0 {
                return Err(InferenceError::NonPositiveWindowValue {
                    attribute,
                    dimension,
                    value,
                });
            }
        }
    }
    input
        .iter()
        .zip(window)
        .enumerate()
        .map(|(axis, (&size, &window))| {
            if padding[axis].is_some_and(|p| p.iter().any(|&v| i64::try_from(v).is_err())) {
                return Err(InferenceError::InvalidWindowPadding { dimension: axis });
            }
            let (Some(size), Some(window), Some(padding)) = (size, window, padding[axis]) else {
                return Ok(None);
            };
            let output = output_extent(
                size,
                window,
                strides[axis],
                base_dilations[axis],
                window_dilations[axis],
                padding,
            );
            u64::try_from(output)
                .ok()
                .filter(|&size| size <= i64::MAX as u64)
                .map(Some)
                .ok_or(InferenceError::DimensionOverflow { dimension: axis })
        })
        .collect()
}
