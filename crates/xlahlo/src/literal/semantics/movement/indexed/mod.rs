//! Pinned `hlo_evaluator.cc` Reverse/Slice/Concatenate byte-copy algorithms.

use crate::{
    attributes::{Axes, Slice},
    inference::movement::{infer_concatenate, infer_reverse, infer_slice},
    literal::Literal,
    literal::semantics::{geometry, invalid, offset, visit_indices},
    shape,
};
use pliron::{common_traits::Verify, context::Context, result::Result, r#type::TypeHandle};

mod padding;
pub use padding::pad;
mod selection;
pub use selection::select;

fn static_array(ctx: &Context, source: &Literal) -> Result<()> {
    source.verify(ctx)?;
    if !shape::is_array(ctx, source.shape) {
        return Err(invalid("indexed literal movement requires an array"));
    }
    if shape::is_dynamic(ctx, source.shape) {
        return Err(invalid(
            "dynamic indexed literal movement runtime-size policy is not implemented",
        ));
    }
    Ok(())
}

fn copy_indices(
    ctx: &Context,
    source: &Literal,
    target: TypeHandle,
    mut source_index: impl FnMut(&[usize]) -> Result<Vec<usize>>,
) -> Result<Literal> {
    let bytes = usize::try_from(shape::byte_size(ctx, target, 0)?)
        .map_err(|_| invalid("indexed literal movement exceeds host addressing"))?;
    let mut result = Literal::from_storage_bytes(ctx, target, vec![0; bytes])?;
    let input = geometry(ctx, source)?;
    let output = geometry(ctx, &result)?;
    visit_indices(output.dimensions(), |index| {
        let from = input.byte_offset(&source_index(index)?)?;
        let to = offset(index, output.strides(), output.element_bytes())?;
        let width = input.element_bytes();
        result.storage[to..to + width].copy_from_slice(&source.storage[from..from + width]);
        Ok(())
    })?;
    Ok(result)
}

/// Reverse selected static array axes, retaining the source's layout.
///
/// Every element byte is copied unchanged, including sub-byte padding and NaN
/// payloads. Dynamic input is rejected rather than inventing output runtime sizes.
pub fn reverse(ctx: &Context, source: &Literal, dimensions: &Axes) -> Result<Literal> {
    static_array(ctx, source)?;
    let target = infer_reverse(ctx, source.shape, dimensions)?;
    let bounds = geometry(ctx, source)?.dimensions().to_vec();
    let axes = dimensions
        .0
        .iter()
        .map(|&axis| usize::try_from(axis).map_err(|_| invalid("negative literal reverse axis")))
        .collect::<Result<Vec<_>>>()?;
    copy_indices(ctx, source, target, |index| {
        let mut from = index.to_vec();
        for &axis in &axes {
            from[axis] = bounds[axis] - 1 - index[axis];
        }
        Ok(from)
    })
}

/// Slice a static array using starts, exclusive limits and positive strides.
///
/// Native shape inference validates the geometry and selects default output
/// layout. Copying follows logical indices, not source physical storage order.
pub fn slice(ctx: &Context, source: &Literal, spec: &Slice) -> Result<Literal> {
    static_array(ctx, source)?;
    let target = infer_slice(ctx, source.shape, spec)?;
    let starts = spec
        .starts
        .0
        .iter()
        .map(|&value| usize::try_from(value).map_err(|_| invalid("invalid literal slice start")))
        .collect::<Result<Vec<_>>>()?;
    let strides = spec
        .strides
        .0
        .iter()
        .map(|&value| usize::try_from(value).map_err(|_| invalid("invalid literal slice stride")))
        .collect::<Result<Vec<_>>>()?;
    copy_indices(ctx, source, target, |index| {
        index
            .iter()
            .zip(&starts)
            .zip(&strides)
            .map(|((&index, &start), &stride)| {
                index
                    .checked_mul(stride)
                    .and_then(|offset| start.checked_add(offset))
                    .ok_or_else(|| invalid("literal slice index overflow"))
            })
            .collect()
    })
}

/// Concatenate static arrays of the exact same element format in operand order.
///
/// The output uses default layout, matching pinned HandleConcatenate. No
/// numerical conversion is performed for mixed-precision floating operands.
pub fn concatenate(ctx: &Context, sources: &[&Literal], axis: usize) -> Result<Literal> {
    let first = sources
        .first()
        .ok_or_else(|| invalid("literal concatenate requires an operand"))?;
    let mut shapes = Vec::with_capacity(sources.len());
    for &source in sources {
        static_array(ctx, source)?;
        if shape::element_type(ctx, source.shape)? != shape::element_type(ctx, first.shape)? {
            return Err(invalid(
                "raw literal concatenate requires identical element formats",
            ));
        }
        shapes.push(source.shape);
    }
    let inferred = infer_concatenate(ctx, &shapes, axis)?;
    let target = shape::with_default_layout(ctx, inferred)?;
    let bytes = usize::try_from(shape::byte_size(ctx, target, 0)?)
        .map_err(|_| invalid("literal concatenate exceeds host addressing"))?;
    let mut result = Literal::from_storage_bytes(ctx, target, vec![0; bytes])?;
    let output = geometry(ctx, &result)?;
    let mut axis_offset = 0usize;
    for &source in sources {
        let input = geometry(ctx, source)?;
        visit_indices(input.dimensions(), |index| {
            let mut destination = index.to_vec();
            destination[axis] = destination[axis]
                .checked_add(axis_offset)
                .ok_or_else(|| invalid("literal concatenate index overflow"))?;
            let from = input.byte_offset(index)?;
            let to = output.byte_offset(&destination)?;
            let width = input.element_bytes();
            result.storage[to..to + width].copy_from_slice(&source.storage[from..from + width]);
            Ok(())
        })?;
        axis_offset = axis_offset
            .checked_add(input.dimensions()[axis])
            .ok_or_else(|| invalid("literal concatenate extent overflow"))?;
    }
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

    fn source(ctx: &Context, element: ElementType) -> Literal {
        let shape = shape::array(ctx, element, vec![3, 4]).unwrap();
        let shape = shape::with_layout(
            ctx,
            shape,
            Layout {
                minor_to_major: vec![0, 1],
                memory_space: 7,
                ..Default::default()
            },
        )
        .unwrap();
        let width = element.bit_width().div_ceil(8) as usize;
        let bytes = (0..12)
            .flat_map(|index| {
                (0..width).map(move |byte| {
                    if element == ElementType::Pred {
                        (index % 2) as u8
                    } else {
                        (index * 17 + byte + 0x80) as u8
                    }
                })
            })
            .collect::<Vec<_>>();
        Literal::from_storage_bytes(ctx, shape, bytes).unwrap()
    }

    fn spec(starts: &[i64], limits: &[i64], strides: &[i64]) -> Slice {
        Slice {
            starts: Axes(starts.to_vec()),
            limits: Axes(limits.to_vec()),
            strides: Axes(strides.to_vec()),
        }
    }

    #[test]
    fn reverse_all_32_formats_preserves_exact_bytes_and_source_layout() {
        let ctx = context();
        for &element in ElementType::ALL {
            let source = source(&ctx, element);
            let saved = source.clone();
            let output = reverse(&ctx, &source, &Axes(vec![0, 1])).unwrap();
            assert_eq!(output.shape, source.shape);
            for i in 0..3 {
                for j in 0..4 {
                    assert_eq!(
                        read_element(&ctx, &output, &[i, j]).unwrap(),
                        read_element(&ctx, &source, &[2 - i, 3 - j]).unwrap(),
                        "{element:?}"
                    );
                }
            }
            assert_eq!(reverse(&ctx, &output, &Axes(vec![0, 1])).unwrap(), source);
            assert_eq!(source, saved);
        }
    }

    #[test]
    fn reverse_one_axis_and_no_axes_have_distinct_owned_storage() {
        let ctx = context();
        let source = source(&ctx, ElementType::U8);
        let mut copied = reverse(&ctx, &source, &Axes(vec![])).unwrap();
        assert_eq!(copied, source);
        copied.storage[0] ^= 0xff;
        assert_ne!(copied.storage, source.storage);
        let output = reverse(&ctx, &source, &Axes(vec![1])).unwrap();
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(
                    read_element(&ctx, &output, &[i, j]).unwrap(),
                    read_element(&ctx, &source, &[i, 3 - j]).unwrap()
                );
            }
        }
    }

    #[test]
    fn reverse_scalar_and_zero_extents_do_not_underflow_indices() {
        let ctx = context();
        let scalar = Literal::from_bits(
            &ctx,
            shape::scalar(&ctx, ElementType::C128).unwrap(),
            &[u128::MAX],
        )
        .unwrap();
        assert_eq!(reverse(&ctx, &scalar, &Axes(vec![])).unwrap(), scalar);
        for dimensions in [vec![0, 3], vec![3, 0], vec![0, 0]] {
            let empty = Literal::from_bits(
                &ctx,
                shape::array(&ctx, ElementType::F32, dimensions).unwrap(),
                &[],
            )
            .unwrap();
            assert_eq!(reverse(&ctx, &empty, &Axes(vec![0, 1])).unwrap(), empty);
        }
    }

    #[test]
    fn strided_slice_all_32_formats_copies_logical_bytes_into_default_layout() {
        let ctx = context();
        for &element in ElementType::ALL {
            let source = source(&ctx, element);
            let saved = source.clone();
            let output = slice(&ctx, &source, &spec(&[0, 1], &[3, 4], &[2, 2])).unwrap();
            assert_eq!(&*shape::dimensions(&ctx, output.shape).unwrap(), &[2, 2]);
            assert_eq!(
                shape::layout(&ctx, output.shape)
                    .unwrap()
                    .unwrap()
                    .minor_to_major,
                vec![1, 0]
            );
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(
                        read_element(&ctx, &output, &[i, j]).unwrap(),
                        read_element(&ctx, &source, &[i * 2, 1 + j * 2]).unwrap(),
                        "{element:?}"
                    );
                }
            }
            assert_eq!(source, saved);
        }
    }

    #[test]
    fn slice_scalar_empty_domain_and_oversized_positive_stride() {
        let ctx = context();
        let scalar = Literal::from_bits(
            &ctx,
            shape::scalar(&ctx, ElementType::C128).unwrap(),
            &[u128::MAX],
        )
        .unwrap();
        assert_eq!(slice(&ctx, &scalar, &spec(&[], &[], &[])).unwrap(), scalar);
        let source = source(&ctx, ElementType::U8);
        let empty = slice(&ctx, &source, &spec(&[3, 1], &[3, 4], &[1, 1])).unwrap();
        assert_eq!(&*shape::dimensions(&ctx, empty.shape).unwrap(), &[0, 3]);
        assert!(empty.storage.is_empty());
        let one = slice(
            &ctx,
            &source,
            &spec(&[1, 2], &[3, 4], &[i64::MAX, i64::MAX]),
        )
        .unwrap();
        assert_eq!(&*shape::dimensions(&ctx, one.shape).unwrap(), &[1, 1]);
        assert_eq!(
            read_element(&ctx, &one, &[0, 0]).unwrap(),
            read_element(&ctx, &source, &[1, 2]).unwrap()
        );
    }

    #[test]
    fn reverse_rejects_duplicate_negative_and_out_of_range_axes() {
        let ctx = context();
        let source = source(&ctx, ElementType::F32);
        let saved = source.clone();
        for axes in [vec![0, 0], vec![-1], vec![2], vec![i64::MAX]] {
            assert!(reverse(&ctx, &source, &Axes(axes)).is_err());
        }
        assert_eq!(source, saved);
    }

    #[test]
    fn slice_rejects_invalid_geometry_without_mutating_source() {
        let ctx = context();
        let source = source(&ctx, ElementType::F32);
        let saved = source.clone();
        for geometry in [
            spec(&[0], &[3], &[1]),
            spec(&[-1, 0], &[3, 4], &[1, 1]),
            spec(&[0, 0], &[4, 4], &[1, 1]),
            spec(&[2, 0], &[1, 4], &[1, 1]),
            spec(&[0, 0], &[3, 4], &[0, 1]),
            spec(&[0, 0], &[3, 4], &[-1, 1]),
        ] {
            assert!(slice(&ctx, &source, &geometry).is_err());
        }
        assert_eq!(source, saved);
    }

    #[test]
    fn indexed_movement_rejects_dynamic_and_nonarray_inputs_explicitly() {
        let ctx = context();
        let shape =
            shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(3)]).unwrap();
        let dynamic =
            Literal::from_dynamic_storage_bytes(&ctx, shape, vec![1, 2, 3], vec![1]).unwrap();
        let token = Literal::token(&ctx);
        let tuple = Literal::tuple(&ctx, vec![]).unwrap();
        for source in [&dynamic, &token, &tuple] {
            assert!(reverse(&ctx, source, &Axes(vec![])).is_err());
            assert!(slice(&ctx, source, &spec(&[], &[], &[])).is_err());
        }
        let mut malformed = source(&ctx, ElementType::S32);
        malformed.storage.pop();
        assert!(reverse(&ctx, &malformed, &Axes(vec![0])).is_err());
        assert!(slice(&ctx, &malformed, &spec(&[0, 0], &[3, 4], &[1, 1])).is_err());
    }

    #[test]
    fn concatenate_all_32_formats_preserves_operand_order_and_bytes_across_layouts() {
        let ctx = context();
        for &element in ElementType::ALL {
            let source = source(&ctx, element);
            let saved = source.clone();
            let row_shape = shape::with_default_layout(&ctx, source.shape).unwrap();
            let row = crate::literal::semantics::relayout(&ctx, &source, row_shape).unwrap();
            for axis in [0, 1] {
                let output = concatenate(&ctx, &[&source, &row], axis).unwrap();
                let expected = if axis == 0 { vec![6, 4] } else { vec![3, 8] };
                assert_eq!(&*shape::dimensions(&ctx, output.shape).unwrap(), &expected);
                assert_eq!(
                    shape::layout(&ctx, output.shape)
                        .unwrap()
                        .unwrap()
                        .minor_to_major,
                    vec![1, 0]
                );
                for i in 0..expected[0] as usize {
                    for j in 0..expected[1] as usize {
                        assert_eq!(
                            read_element(&ctx, &output, &[i, j]).unwrap(),
                            read_element(&ctx, &source, &[i % 3, j % 4]).unwrap(),
                            "{element:?}, axis {axis}"
                        );
                    }
                }
            }
            assert_eq!(source, saved);
        }
    }

    #[test]
    fn concatenate_zero_extents_and_single_operand_are_owned() {
        let ctx = context();
        let source = source(&ctx, ElementType::U8);
        let empty = slice(&ctx, &source, &spec(&[0, 0], &[0, 4], &[1, 1])).unwrap();
        let output = concatenate(&ctx, &[&empty, &source, &empty], 0).unwrap();
        assert!(output.equal_logical_bits(&ctx, &source).unwrap());
        let mut copied = concatenate(&ctx, &[&source], 0).unwrap();
        let saved = source.storage.clone();
        copied.storage[0] ^= 0xff;
        assert_eq!(source.storage, saved);
        let output = concatenate(&ctx, &[&empty, &empty], 0).unwrap();
        assert_eq!(&*shape::dimensions(&ctx, output.shape).unwrap(), &[0, 4]);
        assert!(output.storage.is_empty());
    }

    #[test]
    fn concatenate_rejects_mixed_formats_geometry_and_invalid_axes() {
        let ctx = context();
        let other = source(&ctx, ElementType::F64);
        let source = source(&ctx, ElementType::F32);
        assert!(concatenate(&ctx, &[&source, &other], 0).is_err());
        assert!(concatenate(&ctx, &[], 0).is_err());
        assert!(concatenate(&ctx, &[&source], 2).is_err());
        assert!(concatenate(&ctx, &[&source], usize::MAX).is_err());
        let mismatch = slice(&ctx, &source, &spec(&[0, 0], &[2, 3], &[1, 1])).unwrap();
        assert!(concatenate(&ctx, &[&source, &mismatch], 0).is_err());
        assert!(concatenate(&ctx, &[&source, &mismatch], 1).is_err());
        let scalar =
            Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::F32).unwrap(), &[0]).unwrap();
        assert!(concatenate(&ctx, &[&scalar], 0).is_err());
    }

    #[test]
    fn concatenate_rejects_dynamic_malformed_and_nonarray_operands() {
        let ctx = context();
        let shape =
            shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(3)]).unwrap();
        let dynamic =
            Literal::from_dynamic_storage_bytes(&ctx, shape, vec![1, 2, 3], vec![1]).unwrap();
        assert!(concatenate(&ctx, &[&dynamic], 0).is_err());
        assert!(concatenate(&ctx, &[&Literal::token(&ctx)], 0).is_err());
        assert!(concatenate(&ctx, &[&Literal::tuple(&ctx, vec![]).unwrap()], 0).is_err());
        let mut malformed = source(&ctx, ElementType::S32);
        malformed.storage.pop();
        assert!(concatenate(&ctx, &[&malformed], 0).is_err());
    }
}
