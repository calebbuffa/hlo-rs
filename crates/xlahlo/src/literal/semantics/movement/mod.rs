//! Raw literal movement, grounded in pinned `literal.cc` Broadcast/Reshape/Transpose.
//!
//! These operations preserve element bytes, including inactive bounded storage,
//! and never decode numeric values. Evaluator postprocessing may relayout their
//! results to an instruction's declared layout separately.

use super::{geometry, invalid, offset, relayout, visit_indices};
use crate::{
    literal::Literal,
    shape::{self, Dimension, Layout},
};
use pliron::{common_traits::Verify, context::Context, result::Result, r#type::TypeHandle};

mod indexed;
pub use indexed::{concatenate, pad, reverse, select, slice};

/// Reshape a static array in logical row-major order, returning default layout.
///
/// Dynamic reshape is explicitly unsupported by the pinned Literal operation.
pub fn reshape(ctx: &Context, source: &Literal, dimensions: &[i64]) -> Result<Literal> {
    source.verify(ctx)?;
    if !shape::is_array(ctx, source.shape) {
        return Err(invalid("literal reshape requires an array"));
    }
    if shape::is_dynamic(ctx, source.shape) {
        return Err(invalid("dynamic literal reshape is not implemented"));
    }
    let element = shape::element_type(ctx, source.shape)?;
    let target = shape::array(ctx, element, dimensions.to_vec())?;
    if shape::element_count(ctx, source.shape)? != shape::element_count(ctx, target)? {
        return Err(invalid("literal reshape changes the number of elements"));
    }
    let contiguous = shape::with_layout(
        ctx,
        source.shape,
        Layout::descending(shape::rank(ctx, source.shape)?),
    )?;
    let source = relayout(ctx, source, contiguous)?;
    Literal::from_storage_bytes(ctx, target, source.storage)
}

/// Permute array axes while preserving the physical allocation byte-for-byte.
///
/// The minor order is permuted inversely, so logical transposition requires no
/// element copies. Dynamic flags and runtime sizes follow their source axes.
pub fn transpose(ctx: &Context, source: &Literal, permutation: &[usize]) -> Result<Literal> {
    source.verify(ctx)?;
    if !shape::is_array(ctx, source.shape) {
        return Err(invalid("literal transpose requires an array"));
    }
    let rank = shape::rank(ctx, source.shape)?;
    if permutation.len() != rank {
        return Err(invalid("literal transpose permutation rank mismatch"));
    }
    let mut inverse = vec![usize::MAX; rank];
    for (target, &axis) in permutation.iter().enumerate() {
        let entry = inverse
            .get_mut(axis)
            .ok_or_else(|| invalid("literal transpose axis out of range"))?;
        if *entry != usize::MAX {
            return Err(invalid("literal transpose repeats an axis"));
        }
        *entry = target;
    }
    let dimensions = permutation
        .iter()
        .map(|&axis| shape::dimension(ctx, source.shape, axis))
        .collect::<Result<Vec<_>>>()?;
    let target = shape::dynamic_array(ctx, shape::element_type(ctx, source.shape)?, dimensions)?;
    let mut layout = shape::layout(ctx, source.shape)?
        .ok_or_else(|| invalid("literal transpose requires a layout"))?
        .clone();
    for axis in &mut layout.minor_to_major {
        *axis = inverse[*axis as usize] as i64;
    }
    let target = shape::with_layout(ctx, target, layout)?;
    let dynamic_sizes = if shape::is_dynamic(ctx, source.shape) {
        permutation
            .iter()
            .map(|&axis| source.dynamic_sizes[axis])
            .collect()
    } else {
        Vec::new()
    };
    Literal::from_dynamic_storage_bytes(ctx, target, source.storage.clone(), dynamic_sizes)
}

/// Broadcast source axes into ordered target axes, copying all bounded storage.
///
/// Unmapped dynamic output axes start at their bounds. Mapped dynamic input
/// axes carry their runtime sizes and mark the corresponding output dynamic,
/// matching pinned `SetDynamicSize` even when the requested target is static.
pub fn broadcast(
    ctx: &Context,
    source: &Literal,
    mut target: TypeHandle,
    dimensions: &[usize],
) -> Result<Literal> {
    source.verify(ctx)?;
    target.deref(ctx).verify(ctx)?;
    if !shape::is_array(ctx, source.shape) || !shape::is_array(ctx, target) {
        return Err(invalid("literal broadcast requires array shapes"));
    }
    if shape::element_type(ctx, source.shape)? != shape::element_type(ctx, target)? {
        return Err(invalid("literal broadcast changes element format"));
    }
    let source_bounds = shape::dimensions(ctx, source.shape)?.to_vec();
    let target_bounds = shape::dimensions(ctx, target)?.to_vec();
    if dimensions.len() != source_bounds.len()
        || dimensions.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid(
            "literal broadcast dimensions must be ordered source axes",
        ));
    }
    for (source_axis, &target_axis) in dimensions.iter().enumerate() {
        if target_bounds.get(target_axis) != Some(&source_bounds[source_axis]) {
            return Err(invalid("literal broadcast mapped dimensions disagree"));
        }
        if matches!(
            shape::dimension(ctx, source.shape, source_axis)?,
            Dimension::Bounded(_)
        ) {
            target = shape::with_dynamic_dimension(ctx, target, target_axis, true)?;
        }
    }
    let mut dynamic_sizes = if shape::is_dynamic(ctx, target) {
        target_bounds.clone()
    } else {
        Vec::new()
    };
    if !dynamic_sizes.is_empty() {
        for (axis, &output_axis) in dimensions.iter().enumerate() {
            if matches!(
                shape::dimension(ctx, source.shape, axis)?,
                Dimension::Bounded(_)
            ) {
                dynamic_sizes[output_axis] = source.dynamic_sizes[axis];
            }
        }
    }
    if shape::layout(ctx, target)?.is_none() {
        target = shape::with_default_layout(ctx, target)?;
    }
    let mut layout = shape::layout(ctx, target)?
        .ok_or_else(|| invalid("literal broadcast requires a layout"))?
        .clone();
    layout.element_size_in_bits = 0;
    target = shape::with_layout(ctx, target, layout)?;
    let bytes = usize::try_from(shape::byte_size(ctx, target, 0)?)
        .map_err(|_| invalid("literal broadcast storage exceeds host addressing"))?;
    let mut result =
        Literal::from_dynamic_storage_bytes(ctx, target, vec![0; bytes], dynamic_sizes)?;
    let input_geometry = geometry(ctx, source)?;
    let output_geometry = geometry(ctx, &result)?;
    let bounds = target_bounds
        .into_iter()
        .map(|bound| usize::try_from(bound).map_err(|_| invalid("unbounded literal broadcast")))
        .collect::<Result<Vec<_>>>()?;
    let mut input_index = vec![0; dimensions.len()];
    visit_indices(&bounds, |index| {
        for (axis, &output_axis) in dimensions.iter().enumerate() {
            input_index[axis] = index[output_axis];
        }
        let start = offset(
            &input_index,
            input_geometry.strides(),
            input_geometry.element_bytes(),
        )?;
        let end = offset(
            index,
            output_geometry.strides(),
            output_geometry.element_bytes(),
        )?;
        let width = input_geometry.element_bytes();
        result.storage[end..end + width].copy_from_slice(&source.storage[start..start + width]);
        Ok(())
    })?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{literal::semantics::read_element, types::ElementType};

    fn context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        ctx
    }

    fn shape(ctx: &Context, element: ElementType, dims: &[i64], order: &[i64]) -> TypeHandle {
        let shape = shape::array(ctx, element, dims.to_vec()).unwrap();
        shape::with_layout(
            ctx,
            shape,
            Layout {
                minor_to_major: order.to_vec(),
                ..Layout::descending(dims.len())
            },
        )
        .unwrap()
    }

    fn source(ctx: &Context, element: ElementType) -> Literal {
        let shape = shape(ctx, element, &[2, 3], &[0, 1]);
        let width = element.bit_width().div_ceil(8) as usize;
        let bytes = (0..6)
            .flat_map(|index| {
                (0..width).map(move |byte| {
                    if element == ElementType::Pred {
                        (index % 2) as u8
                    } else {
                        (index * 17 + byte + 1) as u8
                    }
                })
            })
            .collect::<Vec<_>>();
        Literal::from_storage_bytes(ctx, shape, bytes).unwrap()
    }

    fn dynamic(ctx: &Context) -> Literal {
        let shape = shape::dynamic_array(
            ctx,
            ElementType::U8,
            vec![Dimension::Bounded(3), Dimension::Static(2)],
        )
        .unwrap();
        Literal::from_dynamic_storage_bytes(ctx, shape, vec![1, 2, 3, 4, 5, 6], vec![1, 2]).unwrap()
    }

    #[test]
    fn transpose_all_32_formats_preserves_exact_storage_and_logical_values() {
        let ctx = context();
        for &element in ElementType::ALL {
            let source = source(&ctx, element);
            let saved = source.clone();
            let result = transpose(&ctx, &source, &[1, 0]).unwrap();
            assert_eq!(result.storage, source.storage, "{element:?}");
            assert_eq!(&*shape::dimensions(&ctx, result.shape).unwrap(), &[3, 2]);
            assert_eq!(
                shape::layout(&ctx, result.shape)
                    .unwrap()
                    .unwrap()
                    .minor_to_major,
                vec![1, 0]
            );
            for i in 0..2 {
                for j in 0..3 {
                    assert_eq!(
                        read_element(&ctx, &result, &[j, i]).unwrap(),
                        read_element(&ctx, &source, &[i, j]).unwrap()
                    );
                }
            }
            let roundtrip = transpose(&ctx, &result, &[1, 0]).unwrap();
            assert_eq!(roundtrip, source);
            assert_eq!(source, saved);
        }
    }

    #[test]
    fn transpose_dynamic_flags_sizes_and_inactive_bytes_follow_axes() {
        let ctx = context();
        let source = dynamic(&ctx);
        let result = transpose(&ctx, &source, &[1, 0]).unwrap();
        assert_eq!(result.dynamic_sizes, vec![2, 1]);
        assert_eq!(
            shape::dimension(&ctx, result.shape, 0).unwrap(),
            Dimension::Static(2)
        );
        assert_eq!(
            shape::dimension(&ctx, result.shape, 1).unwrap(),
            Dimension::Bounded(3)
        );
        assert_eq!(result.storage, source.storage);
        assert_eq!(read_element(&ctx, &result, &[1, 0]).unwrap(), &[2]);
        assert!(read_element(&ctx, &result, &[0, 1]).is_err());
    }

    #[test]
    fn transpose_rank_zero_and_empty_arrays_are_owned() {
        let ctx = context();
        let scalar = Literal::from_bits(
            &ctx,
            shape::scalar(&ctx, ElementType::C128).unwrap(),
            &[u128::MAX],
        )
        .unwrap();
        assert_eq!(transpose(&ctx, &scalar, &[]).unwrap(), scalar);
        let empty = Literal::from_bits(
            &ctx,
            shape::array(&ctx, ElementType::U8, vec![0, 2]).unwrap(),
            &[],
        )
        .unwrap();
        let result = transpose(&ctx, &empty, &[1, 0]).unwrap();
        assert!(result.storage.is_empty());
        assert_eq!(&*shape::dimensions(&ctx, result.shape).unwrap(), &[2, 0]);
    }

    #[test]
    fn transpose_rejects_malformed_permutations_and_nonarrays_without_mutation() {
        let ctx = context();
        let source = source(&ctx, ElementType::S32);
        let saved = source.clone();
        for permutation in [vec![], vec![0], vec![0, 0], vec![2, 0], vec![usize::MAX, 1]] {
            assert!(transpose(&ctx, &source, &permutation).is_err());
        }
        assert!(transpose(&ctx, &Literal::token(&ctx), &[]).is_err());
        assert_eq!(source, saved);
    }

    #[test]
    fn reshape_all_32_formats_uses_logical_order_not_physical_order() {
        let ctx = context();
        for &element in ElementType::ALL {
            let source = source(&ctx, element);
            let saved = source.clone();
            let result = reshape(&ctx, &source, &[6]).unwrap();
            for i in 0..2 {
                for j in 0..3 {
                    assert_eq!(
                        read_element(&ctx, &result, &[i * 3 + j]).unwrap(),
                        read_element(&ctx, &source, &[i, j]).unwrap(),
                        "{element:?}"
                    );
                }
            }
            assert_eq!(source, saved);
        }
    }

    #[test]
    fn reshape_rank_four_and_scalar_roundtrip() {
        let ctx = context();
        let source = source(&ctx, ElementType::U8);
        let four = reshape(&ctx, &source, &[1, 2, 1, 3]).unwrap();
        assert_eq!(read_element(&ctx, &four, &[0, 1, 0, 2]).unwrap(), &[86]);
        let scalar =
            Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[17])
                .unwrap();
        let expanded = reshape(&ctx, &scalar, &[1, 1, 1]).unwrap();
        assert_eq!(reshape(&ctx, &expanded, &[]).unwrap(), scalar);
    }

    #[test]
    fn reshape_zero_element_arrays_preserve_empty_domain() {
        let ctx = context();
        let empty = Literal::from_bits(
            &ctx,
            shape::array(&ctx, ElementType::F32, vec![2, 0]).unwrap(),
            &[],
        )
        .unwrap();
        assert!(
            reshape(&ctx, &empty, &[0, 7, 4])
                .unwrap()
                .storage
                .is_empty()
        );
        assert!(reshape(&ctx, &empty, &[]).is_err());
    }

    #[test]
    fn reshape_rejects_dynamic_mismatched_negative_and_nonarray_shapes() {
        let ctx = context();
        assert!(reshape(&ctx, &dynamic(&ctx), &[6]).is_err());
        let source = source(&ctx, ElementType::U8);
        for dimensions in [vec![5], vec![-1, -6], vec![i64::MAX, 2]] {
            assert!(reshape(&ctx, &source, &dimensions).is_err());
        }
        assert!(reshape(&ctx, &Literal::token(&ctx), &[]).is_err());
    }

    #[test]
    fn broadcast_all_32_formats_preserves_raw_payloads_and_target_layout() {
        let ctx = context();
        for &element in ElementType::ALL {
            let source = source(&ctx, element);
            let target = shape(&ctx, element, &[2, 4, 3], &[0, 2, 1]);
            let result = broadcast(&ctx, &source, target, &[0, 2]).unwrap();
            assert_eq!(result.shape, target);
            for i in 0..2 {
                for j in 0..4 {
                    for k in 0..3 {
                        assert_eq!(
                            read_element(&ctx, &result, &[i, j, k]).unwrap(),
                            read_element(&ctx, &source, &[i, k]).unwrap(),
                            "{element:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn broadcast_scalar_fills_unmapped_dynamic_dimensions_to_bounds() {
        let ctx = context();
        let scalar =
            Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::U8).unwrap(), &[23]).unwrap();
        let target = shape::dynamic_array(
            &ctx,
            ElementType::U8,
            vec![Dimension::Bounded(2), Dimension::Bounded(3)],
        )
        .unwrap();
        let result = broadcast(&ctx, &scalar, target, &[]).unwrap();
        assert_eq!(result.dynamic_sizes, vec![2, 3]);
        assert_eq!(result.storage, vec![23; 6]);
    }

    #[test]
    fn broadcast_dynamic_source_marks_static_target_and_copies_inactive_storage() {
        let ctx = context();
        let source = dynamic(&ctx);
        let target = shape::array(&ctx, ElementType::U8, vec![3, 4, 2]).unwrap();
        let result = broadcast(&ctx, &source, target, &[0, 2]).unwrap();
        assert_eq!(result.dynamic_sizes, vec![1, 4, 2]);
        assert_eq!(
            shape::dimension(&ctx, result.shape, 0).unwrap(),
            Dimension::Bounded(3)
        );
        assert_eq!(
            result.storage,
            vec![
                1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 5, 6, 5, 6, 5, 6, 5, 6
            ]
        );
        assert!(read_element(&ctx, &result, &[1, 0, 0]).is_err());
    }

    #[test]
    fn broadcast_empty_domain_and_identity_are_exact() {
        let ctx = context();
        let source = source(&ctx, ElementType::U8);
        let identity = broadcast(&ctx, &source, source.shape, &[0, 1]).unwrap();
        assert_eq!(identity, source);
        let target = shape::array(&ctx, ElementType::U8, vec![0, 2, 3]).unwrap();
        assert!(
            broadcast(&ctx, &source, target, &[1, 2])
                .unwrap()
                .storage
                .is_empty()
        );
        let empty = Literal::from_bits(
            &ctx,
            shape::array(&ctx, ElementType::U8, vec![0]).unwrap(),
            &[],
        )
        .unwrap();
        let target = shape::array(&ctx, ElementType::U8, vec![2, 0, 3]).unwrap();
        assert!(
            broadcast(&ctx, &empty, target, &[1])
                .unwrap()
                .storage
                .is_empty()
        );
    }

    #[test]
    fn broadcast_rejects_wrong_format_rank_order_dimensions_and_nonarrays() {
        let ctx = context();
        let source = source(&ctx, ElementType::U8);
        let target = shape::array(&ctx, ElementType::U8, vec![2, 4, 3]).unwrap();
        for axes in [vec![], vec![0], vec![0, 0], vec![2, 0], vec![0, 3]] {
            assert!(broadcast(&ctx, &source, target, &axes).is_err());
        }
        let wrong_type = shape::array(&ctx, ElementType::S8, vec![2, 4, 3]).unwrap();
        assert!(broadcast(&ctx, &source, wrong_type, &[0, 2]).is_err());
        assert!(broadcast(&ctx, &source, shape::token(&ctx), &[]).is_err());
        assert!(broadcast(&ctx, &Literal::token(&ctx), target, &[]).is_err());
    }

    #[test]
    fn movement_rejects_malformed_literals_before_access() {
        let ctx = context();
        let mut source = source(&ctx, ElementType::U8);
        source.storage.pop();
        assert!(transpose(&ctx, &source, &[1, 0]).is_err());
        assert!(reshape(&ctx, &source, &[6]).is_err());
        assert!(broadcast(&ctx, &source, source.shape, &[0, 1]).is_err());
    }
}
