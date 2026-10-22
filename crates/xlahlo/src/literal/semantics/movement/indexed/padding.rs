// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Modified: Rust static, exact-format byte copying with checked coordinates.
// See the crate-level LICENSE for the full Apache-2.0 terms.
//
//! Static subset of OpenXLA's `hlo_evaluator.cc` HandlePad.

use super::static_array;
use crate::{
    attributes::PaddingConfig,
    inference::movement::infer_pad,
    literal::Literal,
    literal::semantics::{geometry, invalid, read_element, visit_indices},
    shape,
};
use pliron::{context::Context, result::Result};

/// Pad a static array with an exact-format scalar, returning default layout.
///
/// Interior padding precedes edge cropping. All source and padding bytes are
/// preserved; mixed-precision conversion and dynamic runtime sizes are excluded.
pub fn pad(
    ctx: &Context,
    source: &Literal,
    value: &Literal,
    config: &PaddingConfig,
) -> Result<Literal> {
    static_array(ctx, source)?;
    static_array(ctx, value)?;
    if shape::element_type(ctx, source.shape)? != shape::element_type(ctx, value.shape)? {
        return Err(invalid(
            "raw literal pad requires identical element formats",
        ));
    }
    let target = infer_pad(ctx, source.shape, value.shape, config)?;
    let scalar = read_element(ctx, value, &[])?;
    let bytes = usize::try_from(shape::byte_size(ctx, target, 0)?)
        .map_err(|_| invalid("literal pad exceeds host addressing"))?;
    let mut storage = vec![0; bytes];
    for element in storage.chunks_exact_mut(scalar.len()) {
        element.copy_from_slice(scalar);
    }
    let mut result = Literal::from_storage_bytes(ctx, target, storage)?;
    let input = geometry(ctx, source)?;
    let output = geometry(ctx, &result)?;
    let mut destination = vec![0; config.0.len()];
    visit_indices(input.dimensions(), |index| {
        for (axis, (&index, padding)) in index.iter().zip(&config.0).enumerate() {
            let index = i128::try_from(index).map_err(|_| invalid("literal pad index overflow"))?;
            let coordinate = index
                .checked_mul(i128::from(padding.interior_padding) + 1)
                .and_then(|offset| offset.checked_add(i128::from(padding.edge_padding_low)))
                .ok_or_else(|| invalid("literal pad index overflow"))?;
            if coordinate < 0 || coordinate >= output.dimensions()[axis] as i128 {
                return Ok(());
            }
            destination[axis] =
                usize::try_from(coordinate).map_err(|_| invalid("literal pad index overflow"))?;
        }
        let from = input.byte_offset(index)?;
        let to = output.byte_offset(&destination)?;
        let width = input.element_bytes();
        result.storage[to..to + width].copy_from_slice(&source.storage[from..from + width]);
        Ok(())
    })?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attributes::PaddingConfigDimension,
        shape::{Dimension, Layout},
        types::ElementType,
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        ctx
    }

    fn config(axes: &[(i64, i64, i64)]) -> PaddingConfig {
        PaddingConfig(
            axes.iter()
                .map(|&(low, high, interior)| PaddingConfigDimension {
                    edge_padding_low: low,
                    edge_padding_high: high,
                    interior_padding: interior,
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn all_32_formats_preserve_source_and_padding_bytes_across_layouts() {
        let ctx = context();
        for &element in ElementType::ALL {
            let shape = shape::with_layout(
                &ctx,
                shape::array(&ctx, element, vec![2, 3]).unwrap(),
                Layout {
                    minor_to_major: vec![0, 1],
                    memory_space: 7,
                    ..Default::default()
                },
            )
            .unwrap();
            let width = element.bit_width().div_ceil(8) as usize;
            let source = Literal::from_storage_bytes(
                &ctx,
                shape,
                (0..6 * width)
                    .map(|index| {
                        if element == ElementType::Pred {
                            (index % 2) as u8
                        } else {
                            (0x80 + index) as u8
                        }
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let value = Literal::from_storage_bytes(
                &ctx,
                shape::scalar(&ctx, element).unwrap(),
                (0..width)
                    .map(|index| {
                        if element == ElementType::Pred {
                            1
                        } else {
                            (0xf0 + index) as u8
                        }
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let saved_source = source.clone();
            let saved_value = value.clone();
            let output = pad(&ctx, &source, &value, &config(&[(1, 0, 1), (-1, 2, 2)])).unwrap();
            assert_eq!(&*shape::dimensions(&ctx, output.shape).unwrap(), &[4, 8]);
            assert_eq!(
                shape::layout(&ctx, output.shape).unwrap().as_deref(),
                Some(&Layout::descending(2))
            );
            for i in 0..4 {
                for j in 0..8 {
                    let expected = if matches!(i, 1 | 3) && matches!(j, 2 | 5) {
                        read_element(&ctx, &source, &[(i - 1) / 2, (j + 1) / 3]).unwrap()
                    } else {
                        read_element(&ctx, &value, &[]).unwrap()
                    };
                    assert_eq!(
                        read_element(&ctx, &output, &[i, j]).unwrap(),
                        expected,
                        "{element:?} [{i},{j}]"
                    );
                }
            }
            assert_eq!(source, saved_source);
            assert_eq!(value, saved_value);
        }
    }

    #[test]
    fn negative_edges_crop_after_interior_padding_not_before_it() {
        let ctx = context();
        let source = Literal::from_bits(
            &ctx,
            shape::array(&ctx, ElementType::U8, vec![3]).unwrap(),
            &[10, 20, 30],
        )
        .unwrap();
        let value = Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::U8).unwrap(), &[255])
            .unwrap();
        assert_eq!(
            pad(&ctx, &source, &value, &config(&[(-1, -1, 1)]))
                .unwrap()
                .bits(&ctx)
                .unwrap(),
            [255, 20, 255]
        );
        assert_eq!(
            pad(&ctx, &source, &value, &config(&[(-2, -2, 1)]))
                .unwrap()
                .bits(&ctx)
                .unwrap(),
            [20]
        );
    }

    #[test]
    fn scalar_and_zero_extents_return_owned_storage_without_phantom_elements() {
        let ctx = context();
        let scalar = Literal::from_bits(
            &ctx,
            shape::scalar(&ctx, ElementType::C128).unwrap(),
            &[u128::MAX],
        )
        .unwrap();
        let value = Literal::from_bits(&ctx, scalar.shape, &[0]).unwrap();
        let mut output = pad(&ctx, &scalar, &value, &config(&[])).unwrap();
        assert_eq!(output, scalar);
        output.storage[0] = 0;
        assert_ne!(output, scalar);
        let empty = Literal::from_bits(
            &ctx,
            shape::array(&ctx, ElementType::C128, vec![0, 3]).unwrap(),
            &[],
        )
        .unwrap();
        let output = pad(
            &ctx,
            &empty,
            &scalar,
            &config(&[(1, 1, i64::MAX), (0, 0, 0)]),
        )
        .unwrap();
        assert_eq!(output.bits(&ctx).unwrap(), [u128::MAX; 6]);
        let output = pad(&ctx, &empty, &scalar, &config(&[(0, 0, 0), (0, 0, 0)])).unwrap();
        assert!(output.storage.is_empty());
    }

    #[test]
    fn extreme_edge_and_interior_coordinates_use_checked_wide_arithmetic() {
        let ctx = context();
        let source = Literal::from_bits(
            &ctx,
            shape::array(&ctx, ElementType::U8, vec![3]).unwrap(),
            &[1, 2, 3],
        )
        .unwrap();
        let value =
            Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::U8).unwrap(), &[9]).unwrap();
        let output = pad(&ctx, &source, &value, &config(&[(i64::MIN, i64::MAX, 1)])).unwrap();
        assert_eq!(output.bits(&ctx).unwrap(), [9; 4]);
        let one = Literal::from_bits(
            &ctx,
            shape::array(&ctx, ElementType::U8, vec![1]).unwrap(),
            &[7],
        )
        .unwrap();
        assert_eq!(
            pad(&ctx, &one, &value, &config(&[(0, 0, i64::MAX)]))
                .unwrap()
                .bits(&ctx)
                .unwrap(),
            [7]
        );
        assert!(pad(&ctx, &source, &value, &config(&[(0, 0, i64::MAX)])).is_err());
    }

    #[test]
    fn malformed_configs_and_values_fail_without_mutating_either_input() {
        let ctx = context();
        let source = Literal::from_bits(
            &ctx,
            shape::array(&ctx, ElementType::F32, vec![2]).unwrap(),
            &[0x7f80_1234, 0x8000_0000],
        )
        .unwrap();
        let value = Literal::from_bits(
            &ctx,
            shape::scalar(&ctx, ElementType::F32).unwrap(),
            &[0xff80_2345],
        )
        .unwrap();
        let saved_source = source.clone();
        let saved_value = value.clone();
        for invalid in [
            config(&[]),
            config(&[(0, 0, -1)]),
            config(&[(-3, 0, 0)]),
            config(&[(0, i64::MAX, 0)]),
        ] {
            assert!(pad(&ctx, &source, &value, &invalid).is_err());
        }
        let different =
            Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::F64).unwrap(), &[0]).unwrap();
        assert!(pad(&ctx, &source, &different, &config(&[(0, 0, 0)])).is_err());
        assert!(pad(&ctx, &source, &source, &config(&[(0, 0, 0)])).is_err());
        assert_eq!(source, saved_source);
        assert_eq!(value, saved_value);
    }

    #[test]
    fn dynamic_inputs_and_nonarrays_have_no_invented_runtime_size_policy() {
        let ctx = context();
        let shape =
            shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(2)]).unwrap();
        let source = Literal::from_dynamic_storage_bytes(&ctx, shape, vec![1, 2], vec![1]).unwrap();
        let value =
            Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::U8).unwrap(), &[0]).unwrap();
        assert!(pad(&ctx, &source, &value, &config(&[(1, 1, 0)])).is_err());
        let tuple = Literal::tuple(&ctx, vec![value.clone()]).unwrap();
        assert!(pad(&ctx, &tuple, &value, &config(&[])).is_err());
        let token = Literal::token(&ctx);
        assert!(pad(&ctx, &token, &value, &config(&[])).is_err());
    }
}
