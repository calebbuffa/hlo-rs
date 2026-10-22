// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Modified: Rust static, exact-format logical byte copying.
// See the crate-level LICENSE for the full Apache-2.0 terms.
//
//! Static subset of OpenXLA's `hlo_evaluator.cc` HandleSelect.

use super::static_array;
use crate::{
    inference::elementwise::infer_select,
    literal::Literal,
    literal::semantics::{geometry, invalid, visit_indices},
    shape,
};
use pliron::{context::Context, result::Result};

/// Select complete raw elements using a scalar or matching-shape predicate.
///
/// The inferred layout is retained. All arrays must be static and both data
/// inputs must have the exact same format; no numerical conversion is performed.
pub fn select(
    ctx: &Context,
    predicate: &Literal,
    on_true: &Literal,
    on_false: &Literal,
) -> Result<Literal> {
    for input in [predicate, on_true, on_false] {
        static_array(ctx, input)?;
    }
    if shape::element_type(ctx, on_true.shape)? != shape::element_type(ctx, on_false.shape)? {
        return Err(invalid(
            "raw literal select requires identical element formats",
        ));
    }
    let target = infer_select(ctx, predicate.shape, on_true.shape, on_false.shape)?;
    let bytes = usize::try_from(shape::byte_size(ctx, target, 0)?)
        .map_err(|_| invalid("literal select exceeds host addressing"))?;
    let mut result = Literal::from_storage_bytes(ctx, target, vec![0; bytes])?;
    let true_geometry = geometry(ctx, on_true)?;
    let false_geometry = geometry(ctx, on_false)?;
    let predicate_geometry = geometry(ctx, predicate)?;
    let output = geometry(ctx, &result)?;
    let scalar_predicate = predicate_geometry.dimensions().is_empty();
    visit_indices(output.dimensions(), |index| {
        let predicate_index = if scalar_predicate { &[][..] } else { index };
        let condition = predicate.storage[predicate_geometry.byte_offset(predicate_index)?] != 0;
        let (input, layout) = if condition {
            (on_true, &true_geometry)
        } else {
            (on_false, &false_geometry)
        };
        let from = layout.byte_offset(index)?;
        let to = output.byte_offset(index)?;
        let width = output.element_bytes();
        result.storage[to..to + width].copy_from_slice(&input.storage[from..from + width]);
        Ok(())
    })?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        literal::semantics::read_element,
        shape::{Dimension, Layout},
        types::ElementType,
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        ctx
    }

    fn input(ctx: &Context, element: ElementType, descending: bool, seed: usize) -> Literal {
        let shape = shape::array(ctx, element, vec![2, 3]).unwrap();
        let shape = if descending {
            shape
        } else {
            shape::with_layout(
                ctx,
                shape,
                Layout {
                    minor_to_major: vec![0, 1],
                    memory_space: 7,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let width = element.bit_width().div_ceil(8) as usize;
        let bytes = (0..6 * width)
            .map(|index| {
                if element == ElementType::Pred {
                    ((index + seed) % 2) as u8
                } else {
                    (seed + index) as u8
                }
            })
            .collect::<Vec<_>>();
        Literal::from_storage_bytes(ctx, shape, bytes).unwrap()
    }

    #[test]
    fn array_predicate_selects_exact_bytes_for_all_32_formats_and_distinct_layouts() {
        let ctx = context();
        let predicate = input(&ctx, ElementType::Pred, false, 0);
        for &element in ElementType::ALL {
            let on_true = input(&ctx, element, true, 0x80);
            let on_false = input(&ctx, element, false, 0x21);
            let saved = (predicate.clone(), on_true.clone(), on_false.clone());
            let mut output = select(&ctx, &predicate, &on_true, &on_false).unwrap();
            let inferred =
                infer_select(&ctx, predicate.shape, on_true.shape, on_false.shape).unwrap();
            assert_eq!(output.shape, inferred);
            for i in 0..2 {
                for j in 0..3 {
                    let selected = if read_element(&ctx, &predicate, &[i, j]).unwrap()[0] != 0 {
                        &on_true
                    } else {
                        &on_false
                    };
                    assert_eq!(
                        read_element(&ctx, &output, &[i, j]).unwrap(),
                        read_element(&ctx, selected, &[i, j]).unwrap(),
                        "{element:?} [{i},{j}]"
                    );
                }
            }
            output.storage[0] ^= 1;
            assert_eq!((predicate.clone(), on_true, on_false), saved);
        }
    }

    #[test]
    fn scalar_predicate_selects_all_raw_bytes_without_canonicalizing_padding_or_nans() {
        let ctx = context();
        for &element in ElementType::ALL {
            let on_true = input(&ctx, element, true, 0x80);
            let on_false = input(&ctx, element, false, 0x21);
            for condition in [0, 1] {
                let predicate = Literal::from_bits(
                    &ctx,
                    shape::scalar(&ctx, ElementType::Pred).unwrap(),
                    &[condition],
                )
                .unwrap();
                let selected = if condition == 1 { &on_true } else { &on_false };
                let output = select(&ctx, &predicate, &on_true, &on_false).unwrap();
                for i in 0..2 {
                    for j in 0..3 {
                        assert_eq!(
                            read_element(&ctx, &output, &[i, j]).unwrap(),
                            read_element(&ctx, selected, &[i, j]).unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_and_empty_results_do_not_read_phantom_array_elements() {
        let ctx = context();
        let predicate =
            Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::Pred).unwrap(), &[0])
                .unwrap();
        let scalar = Literal::from_bits(
            &ctx,
            shape::scalar(&ctx, ElementType::C128).unwrap(),
            &[u128::MAX],
        )
        .unwrap();
        assert_eq!(select(&ctx, &predicate, &scalar, &scalar).unwrap(), scalar);
        for scalar_predicate in [true, false] {
            let empty = Literal::from_bits(
                &ctx,
                shape::array(&ctx, ElementType::C128, vec![0, 3]).unwrap(),
                &[],
            )
            .unwrap();
            let empty_predicate = Literal::from_bits(
                &ctx,
                shape::array(&ctx, ElementType::Pred, vec![0, 3]).unwrap(),
                &[],
            )
            .unwrap();
            assert!(
                select(
                    &ctx,
                    if scalar_predicate {
                        &predicate
                    } else {
                        &empty_predicate
                    },
                    &empty,
                    &empty,
                )
                .unwrap()
                .storage
                .is_empty()
            );
        }
    }

    #[test]
    fn signaling_nans_signed_zero_complex_components_and_padding_are_never_canonicalized() {
        let ctx = context();
        for (element, bits) in [
            (ElementType::S1, [0xffu128, 0x80]),
            (ElementType::F6E3M2FN, [0xf9, 0xc0]),
            (ElementType::F8E5M2, [0x7d, 0x80]),
            (ElementType::F16, [0x7c01, 0x8000]),
            (ElementType::BF16, [0x7f81, 0x8000]),
            (ElementType::F32, [0x7f80_1234, 0x8000_0000]),
            (
                ElementType::F64,
                [0x7ff0_0000_0000_1234, 0x8000_0000_0000_0000],
            ),
            (
                ElementType::C64,
                [0x8000_0000_7f80_1234, 0x7f80_4321_ff80_1234],
            ),
            (
                ElementType::C128,
                [
                    (0x8000_0000_0000_0000u128 << 64) | 0x7ff0_0000_0000_1234,
                    (0x7ff0_0000_0000_4321u128 << 64) | 0xfff0_0000_0000_1234,
                ],
            ),
        ] {
            let width = element.bit_width().div_ceil(8) as usize;
            let shape = shape::array(&ctx, element, vec![2]).unwrap();
            let make = |values: [u128; 2]| {
                Literal::from_storage_bytes(
                    &ctx,
                    shape,
                    values
                        .into_iter()
                        .flat_map(|bits| bits.to_le_bytes().into_iter().take(width))
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            };
            let on_true = make(bits);
            let on_false = make([bits[1], bits[0]]);
            for conditions in [[1, 1], [0, 0], [1, 0], [0, 1]] {
                let predicate = Literal::from_bits(
                    &ctx,
                    shape::array(&ctx, ElementType::Pred, vec![2]).unwrap(),
                    &conditions,
                )
                .unwrap();
                let output = select(&ctx, &predicate, &on_true, &on_false).unwrap();
                for (index, condition) in conditions.into_iter().enumerate() {
                    assert_eq!(
                        read_element(&ctx, &output, &[index]).unwrap(),
                        read_element(
                            &ctx,
                            if condition == 1 { &on_true } else { &on_false },
                            &[index]
                        )
                        .unwrap(),
                        "{element:?} {conditions:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn mismatched_types_shapes_predicates_and_dynamic_inputs_are_explicit_errors() {
        let ctx = context();
        let predicate = input(&ctx, ElementType::Pred, false, 0);
        let on_true = input(&ctx, ElementType::U8, true, 0x80);
        let on_false = input(&ctx, ElementType::U8, false, 0x21);
        assert!(select(&ctx, &on_true, &on_true, &on_false).is_err());
        let other = input(&ctx, ElementType::S8, true, 0x80);
        assert!(select(&ctx, &predicate, &on_true, &other).is_err());
        let scalar =
            Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::U8).unwrap(), &[1]).unwrap();
        assert!(select(&ctx, &predicate, &on_true, &scalar).is_err());
        let short_predicate = Literal::from_bits(
            &ctx,
            shape::array(&ctx, ElementType::Pred, vec![2]).unwrap(),
            &[1, 0],
        )
        .unwrap();
        assert!(select(&ctx, &short_predicate, &on_true, &on_false).is_err());
        let shape = shape::dynamic_array(
            &ctx,
            ElementType::U8,
            vec![Dimension::Bounded(2), Dimension::Static(3)],
        )
        .unwrap();
        let dynamic =
            Literal::from_dynamic_storage_bytes(&ctx, shape, vec![0; 6], vec![1, 3]).unwrap();
        assert!(select(&ctx, &predicate, &dynamic, &on_false).is_err());
    }
}
