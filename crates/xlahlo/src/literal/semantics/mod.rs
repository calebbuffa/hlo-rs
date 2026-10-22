//! Logical access to owned native literals, without numerical evaluation.
//!
//! Based on OpenXLA's `literal.cc` (GetFirstInteger, Piece::CopyFrom,
//! Relayout) and `literal_util.cc` (LiteralAsScalarInt64). Native literal arrays
//! use little-endian, minor-to-major dense storage, one byte per sub-byte value.
//! Raw access and copying preserve every storage bit, including unused sub-byte
//! bits, NaN payloads and complex components; no floating-point decoding occurs.
use pliron::{common_traits::Verify, context::Context, result::Result, r#type::TypeHandle};

use crate::{
    Diagnostic,
    literal::Literal,
    shape,
    types::{ElementType, TokenType},
};

pub mod movement;

fn invalid(message: &str) -> pliron::result::Error {
    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
        message: message.into(),
    })
}

/// Checked active logical dimensions and allocation-bound physical strides.
///
/// Offsets address the native literal byte encoding, not a backend allocation.
/// Dynamic bounds determine strides; runtime sizes determine accessible indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralGeometry {
    dimensions: Vec<usize>,
    strides: Vec<usize>,
    element_bytes: usize,
}

impl LiteralGeometry {
    /// Runtime dimensions, or static extents for non-dynamic axes.
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Physical strides in elements, computed from allocation bounds.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Bytes per native literal element; sub-byte values occupy one byte.
    pub fn element_bytes(&self) -> usize {
        self.element_bytes
    }

    /// Checked byte offset for a logical coordinate in the active domain.
    pub fn byte_offset(&self, index: &[usize]) -> Result<usize> {
        if index.len() != self.dimensions.len() {
            return Err(invalid("literal index rank mismatch"));
        }
        if index.iter().zip(&self.dimensions).any(|(&i, &d)| i >= d) {
            return Err(invalid("literal index outside active dimensions"));
        }
        offset(index, &self.strides, self.element_bytes)
    }
}

/// Validate an array and reuse the literal's single geometry implementation.
pub fn geometry(ctx: &Context, literal: &Literal) -> Result<LiteralGeometry> {
    literal.verify(ctx)?;
    if !shape::is_array(ctx, literal.shape) {
        return Err(invalid("logical element access requires an array literal"));
    }
    let (dimensions, strides) = literal.geometry(ctx)?;
    Ok(LiteralGeometry {
        dimensions,
        strides,
        element_bytes: shape::element_type(ctx, literal.shape)?
            .bit_width()
            .div_ceil(8) as usize,
    })
}

fn offset(index: &[usize], strides: &[usize], bytes: usize) -> Result<usize> {
    index
        .iter()
        .zip(strides)
        .try_fold(0usize, |sum, (&i, &stride)| {
            i.checked_mul(stride).and_then(|term| sum.checked_add(term))
        })
        .and_then(|elements| elements.checked_mul(bytes))
        .ok_or_else(|| invalid("literal byte offset overflow"))
}

/// Borrow the exact element bytes at an active logical coordinate.
pub fn read_element<'a>(ctx: &Context, literal: &'a Literal, index: &[usize]) -> Result<&'a [u8]> {
    let geometry = geometry(ctx, literal)?;
    let start = geometry.byte_offset(index)?;
    let end = start
        .checked_add(geometry.element_bytes)
        .ok_or_else(|| invalid("literal byte range overflow"))?;
    literal
        .storage
        .get(start..end)
        .ok_or_else(|| invalid("literal element storage is missing"))
}

/// Write exact element bytes without changing inactive elements or runtime sizes.
///
/// Width and predicate encodings are checked before mutation. Other sub-byte
/// encodings retain their unused high bits, matching native Literal storage.
pub fn write_element(
    ctx: &Context,
    literal: &mut Literal,
    index: &[usize],
    bytes: &[u8],
) -> Result<()> {
    let geometry = geometry(ctx, literal)?;
    let start = geometry.byte_offset(index)?;
    if bytes.len() != geometry.element_bytes {
        return Err(invalid("literal element byte width mismatch"));
    }
    if shape::element_type(ctx, literal.shape)? == ElementType::Pred && bytes[0] > 1 {
        return Err(invalid("predicate storage must be zero or one"));
    }
    let end = start
        .checked_add(bytes.len())
        .ok_or_else(|| invalid("literal byte range overflow"))?;
    let target = literal
        .storage
        .get_mut(start..end)
        .ok_or_else(|| invalid("literal element storage is missing"))?;
    target.copy_from_slice(bytes);
    Ok(())
}

/// Copy one active logical element between arrays of the same numeric format.
///
/// Array ranks and layouts may differ. Exact bytes, including unused sub-byte
/// bits, are retained; no conversion, broadcasting, or mutation on error occurs.
pub fn copy_element(
    ctx: &Context,
    destination: &mut Literal,
    destination_index: &[usize],
    source: &Literal,
    source_index: &[usize],
) -> Result<()> {
    if shape::element_type(ctx, destination.shape)? != shape::element_type(ctx, source.shape)? {
        return Err(invalid(
            "literal element copy requires the same numeric format",
        ));
    }
    let bytes = read_element(ctx, source, source_index)?;
    write_element(ctx, destination, destination_index, bytes)
}

/// Copy a compatible source subtree into a destination, preserving destination types.
///
/// Like pinned `Piece::CopyFrom` with `only_dynamic_bound=false`, this copies
/// the entire bounded allocation, not only active dynamic elements. Differing
/// minor orders reorder raw bytes by logical coordinates. Runtime sizes are
/// copied when both leaves are dynamic; a dynamic destination receiving a static
/// source retains its existing runtime sizes. Tuple/token structure is retained.
/// Both trees are checked first; an error leaves the destination unchanged.
pub fn copy_sub_literal(
    ctx: &Context,
    destination: &mut Literal,
    destination_index: &[usize],
    source: &Literal,
    source_index: &[usize],
) -> Result<()> {
    destination.verify(ctx)?;
    source.verify(ctx)?;
    let source = source.sub_literal(ctx, source_index)?;
    let destination_type = shape::subshape(ctx, destination.shape, destination_index)?;
    if !shape::compatible(ctx, destination_type, source.shape) {
        return Err(invalid("literal copy subshapes are incompatible"));
    }
    let mut staged = destination.clone();
    let mut target = &mut staged;
    for &index in destination_index {
        target = target
            .tuple_literals
            .get_mut(index)
            .ok_or_else(|| invalid("literal destination tuple child is missing"))?;
    }
    copy_tree(ctx, target, source)?;
    staged.verify(ctx)?;
    *destination = staged;
    Ok(())
}

fn copy_tree(ctx: &Context, destination: &mut Literal, source: &Literal) -> Result<()> {
    if shape::is_tuple(ctx, destination.shape) {
        for (target, input) in destination
            .tuple_literals
            .iter_mut()
            .zip(&source.tuple_literals)
        {
            copy_tree(ctx, target, input)?;
        }
        return Ok(());
    }
    if destination.shape.deref(ctx).is::<TokenType>() {
        return Ok(());
    }
    let target_geometry = geometry(ctx, destination)?;
    let source_geometry = geometry(ctx, source)?;
    let bounds = shape::dimensions(ctx, source.shape)?
        .iter()
        .map(|&size| usize::try_from(size).map_err(|_| invalid("invalid literal allocation bound")))
        .collect::<Result<Vec<_>>>()?;
    let bytes = source_geometry.element_bytes;
    if target_geometry.strides == source_geometry.strides {
        destination.storage.copy_from_slice(&source.storage);
    } else {
        visit_indices(&bounds, |index| {
            let from = offset(index, &source_geometry.strides, bytes)?;
            let to = offset(index, &target_geometry.strides, bytes)?;
            destination.storage[to..to + bytes]
                .copy_from_slice(&source.storage[from..from + bytes]);
            Ok(())
        })?;
    }
    if shape::is_dynamic(ctx, destination.shape) && shape::is_dynamic(ctx, source.shape) {
        destination.dynamic_sizes.clone_from(&source.dynamic_sizes);
    }
    Ok(())
}

fn visit_indices(
    dimensions: &[usize],
    mut visit: impl FnMut(&[usize]) -> Result<()>,
) -> Result<()> {
    if dimensions.contains(&0) {
        return Ok(());
    }
    let mut index = vec![0; dimensions.len()];
    loop {
        visit(&index)?;
        let mut axis = dimensions.len();
        loop {
            if axis == 0 {
                return Ok(());
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < dimensions[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// Return an independently owned literal with the target tree's physical order.
///
/// Element formats, allocation bounds and dynamic flags must match. Target
/// layouts follow the existing Literal constructor contract: absent layouts get
/// default order, and element-size packing is normalized to zero for literals.
/// Runtime sizes and all bounded storage bytes, including inactive elements, are
/// preserved. This is a logical relayout, not a physical bitcast or conversion.
pub fn relayout(ctx: &Context, source: &Literal, target: TypeHandle) -> Result<Literal> {
    source.verify(ctx)?;
    target.deref(ctx).verify(ctx)?;
    if !shape::equal_ignoring_layout(ctx, source.shape, target) {
        return Err(invalid(
            "literal relayout changes element type, bounds or dynamic flags",
        ));
    }
    let mut result = retarget(ctx, source, target)?;
    result.verify(ctx)?;
    copy_tree(ctx, &mut result, source)?;
    result.verify(ctx)?;
    Ok(result)
}

fn retarget(ctx: &Context, source: &Literal, target: TypeHandle) -> Result<Literal> {
    if shape::is_tuple(ctx, target) {
        let children = shape::children(ctx, target).to_vec();
        let elements = source
            .tuple_literals
            .iter()
            .zip(children)
            .map(|(child, ty)| retarget(ctx, child, ty))
            .collect::<Result<_>>()?;
        return Literal::tuple(ctx, elements);
    }
    if target.deref(ctx).is::<TokenType>() {
        return Ok(Literal::token(ctx));
    }
    if !shape::is_array(ctx, target) {
        return Err(invalid("literal relayout requires array, tuple or token"));
    }
    Literal::from_dynamic_storage_bytes(
        ctx,
        target,
        source.storage.clone(),
        source.dynamic_sizes.clone(),
    )
}

fn scalar_encoding(ctx: &Context, literal: &Literal) -> Result<(ElementType, u128)> {
    let geometry = geometry(ctx, literal)?;
    if shape::dimensions(ctx, literal.shape)?
        .iter()
        .any(|&bound| bound != 1)
        || geometry.dimensions.iter().any(|&size| size != 1)
    {
        return Err(invalid("literal is not an effective scalar"));
    }
    let bytes = read_element(ctx, literal, &vec![0; geometry.dimensions.len()])?;
    let mut encoding = [0u8; 16];
    encoding[..bytes.len()].copy_from_slice(bytes);
    let element = shape::element_type(ctx, literal.shape)?;
    let width = element.bit_width();
    let bits = u128::from_le_bytes(encoding);
    Ok((
        element,
        if width == 128 {
            bits
        } else {
            bits & ((1u128 << width) - 1)
        },
    ))
}

/// Extract a checked effective-scalar integer, excluding predicates.
///
/// Narrow signed formats are sign extended after masking unused storage bits.
/// Unsigned64 values above `i64::MAX` are errors, not wrapped negative values.
pub fn scalar_integer(ctx: &Context, literal: &Literal) -> Result<i64> {
    let (element, bits) = scalar_encoding(ctx, literal)?;
    if !element.is_integer() {
        return Err(invalid(
            "scalar integer extraction requires an integer format",
        ));
    }
    let width = element.bit_width();
    let value = if element.is_signed() && bits & (1u128 << (width - 1)) != 0 {
        bits as i128 - (1i128 << width)
    } else {
        bits as i128
    };
    i64::try_from(value).map_err(|_| invalid("scalar integer does not fit i64"))
}

/// Extract a checked effective-scalar predicate, without numeric coercion.
pub fn scalar_predicate(ctx: &Context, literal: &Literal) -> Result<bool> {
    let (element, bits) = scalar_encoding(ctx, literal)?;
    if element != ElementType::Pred {
        return Err(invalid("scalar predicate extraction requires pred"));
    }
    Ok(bits == 1)
}

/// Construct a rank-zero integer literal with checked mathematical range.
///
/// `i128` admits every signed/unsigned native integer value, including U64::MAX.
/// Unlike a C++ narrowing cast, out-of-range values are errors; callers requiring
/// modular arithmetic must compute the intended encoding separately.
pub fn integer_scalar(ctx: &Context, element: ElementType, value: i128) -> Result<Literal> {
    if !element.is_integer() {
        return Err(invalid(
            "integer scalar construction requires an integer format",
        ));
    }
    let width = element.bit_width();
    let (minimum, maximum) = if element.is_signed() {
        (-(1i128 << (width - 1)), (1i128 << (width - 1)) - 1)
    } else {
        (0, (1i128 << width) - 1)
    };
    if value < minimum || value > maximum {
        return Err(invalid("integer scalar value is outside its format range"));
    }
    let bits = (value as u128) & ((1u128 << width) - 1);
    let ty = shape::scalar(ctx, element)?;
    Literal::from_bits(ctx, ty, &[bits])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{Dimension, Layout};

    fn context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        ctx
    }

    fn shape(ctx: &Context, element: ElementType, dimensions: &[i64], order: &[i64]) -> TypeHandle {
        let ty = shape::array(ctx, element, dimensions.to_vec()).unwrap();
        shape::with_layout(
            ctx,
            ty,
            Layout {
                minor_to_major: order.to_vec(),
                ..Layout::descending(dimensions.len())
            },
        )
        .unwrap()
    }

    fn matrix(ctx: &Context, element: ElementType, order: &[i64]) -> Literal {
        let ty = shape(ctx, element, &[2, 3], order);
        let width = element.bit_width().div_ceil(8) as usize;
        let storage = (0..6)
            .flat_map(|index| {
                (0..width).map(move |byte| {
                    if element == ElementType::Pred {
                        (index % 2) as u8
                    } else {
                        (index * 23 + byte + 1) as u8
                    }
                })
            })
            .collect::<Vec<_>>();
        Literal::from_storage_bytes(ctx, ty, storage).unwrap()
    }

    fn dynamic(ctx: &Context, order: &[i64], sizes: &[i64], storage: Vec<u8>) -> Literal {
        let ty = shape::dynamic_array(
            ctx,
            ElementType::U8,
            vec![Dimension::Bounded(3), Dimension::Static(2)],
        )
        .unwrap();
        let ty = shape::with_layout(
            ctx,
            ty,
            Layout {
                minor_to_major: order.to_vec(),
                ..Layout::descending(2)
            },
        )
        .unwrap();
        Literal::from_dynamic_storage_bytes(ctx, ty, storage, sizes.to_vec()).unwrap()
    }

    fn ranges(element: ElementType) -> (i128, i128) {
        let width = element.bit_width();
        if element.is_signed() {
            (-(1i128 << (width - 1)), (1i128 << (width - 1)) - 1)
        } else {
            (0, (1i128 << width) - 1)
        }
    }

    #[test]
    fn rank_zero_geometry_has_one_accessible_element() {
        let ctx = context();
        let literal = integer_scalar(&ctx, ElementType::S32, 17).unwrap();
        let geometry = geometry(&ctx, &literal).unwrap();
        assert_eq!(geometry.dimensions(), &[]);
        assert_eq!(geometry.strides(), &[]);
        assert_eq!(geometry.element_bytes(), 4);
        assert_eq!(geometry.byte_offset(&[]).unwrap(), 0);
        assert_eq!(read_element(&ctx, &literal, &[]).unwrap(), &[17, 0, 0, 0]);
    }

    #[test]
    fn physical_offsets_follow_layout_strides_and_element_width() {
        let ctx = context();
        for &element in ElementType::ALL {
            for (order, strides) in [([1, 0], [3, 1]), ([0, 1], [1, 2])] {
                let literal = matrix(&ctx, element, &order);
                let geometry = geometry(&ctx, &literal).unwrap();
                assert_eq!(geometry.dimensions(), &[2, 3]);
                assert_eq!(geometry.strides(), &strides);
                for i in 0..2 {
                    for j in 0..3 {
                        assert_eq!(
                            geometry.byte_offset(&[i, j]).unwrap(),
                            (i * strides[0] + j * strides[1])
                                * element.bit_width().div_ceil(8) as usize,
                            "{element:?}, order={order:?}, index={i},{j}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_dynamic_geometry_uses_allocation_strides() {
        let ctx = context();
        let literal = dynamic(&ctx, &[0, 1], &[2, 2], vec![1, 2, 3, 4, 5, 6]);
        let geometry = geometry(&ctx, &literal).unwrap();
        assert_eq!(geometry.dimensions(), &[2, 2]);
        assert_eq!(geometry.strides(), &[1, 3]);
        assert_eq!(read_element(&ctx, &literal, &[1, 1]).unwrap(), &[5]);
        assert!(read_element(&ctx, &literal, &[2, 1]).is_err());
    }

    #[test]
    fn empty_static_and_dynamic_domains_have_no_accessible_elements() {
        let ctx = context();
        let static_type = shape(&ctx, ElementType::C128, &[0, 3], &[1, 0]);
        for literal in [
            dynamic(&ctx, &[1, 0], &[0, 2], vec![9; 6]),
            Literal::from_storage_bytes(&ctx, static_type, vec![]).unwrap(),
        ] {
            assert!(geometry(&ctx, &literal).is_ok());
            assert!(read_element(&ctx, &literal, &[0, 0]).is_err());
        }
    }

    #[test]
    fn tuple_and_token_do_not_have_array_geometry() {
        let ctx = context();
        let token = Literal::token(&ctx);
        let tuple = Literal::tuple(&ctx, vec![token.clone()]).unwrap();
        assert!(geometry(&ctx, &token).is_err());
        assert!(geometry(&ctx, &tuple).is_err());
        assert!(read_element(&ctx, &tuple, &[]).is_err());
    }

    #[test]
    fn malformed_payload_is_rejected_before_indexing() {
        let ctx = context();
        let mut literal = integer_scalar(&ctx, ElementType::S64, 0).unwrap();
        literal.storage.pop();
        assert!(geometry(&ctx, &literal).is_err());
        assert!(read_element(&ctx, &literal, &[]).is_err());
        let before = literal.clone();
        assert!(write_element(&ctx, &mut literal, &[], &[0; 8]).is_err());
        assert_eq!(literal, before);
    }

    #[test]
    fn index_rank_and_extent_failures_are_checked() {
        let ctx = context();
        let literal = matrix(&ctx, ElementType::U8, &[1, 0]);
        for index in [
            vec![],
            vec![0],
            vec![0, 0, 0],
            vec![2, 0],
            vec![0, 3],
            vec![usize::MAX, 0],
        ] {
            assert!(read_element(&ctx, &literal, &index).is_err());
        }
    }

    #[test]
    fn all_32_raw_formats_support_exact_logical_read_write() {
        let ctx = context();
        assert_eq!(ElementType::ALL.len(), 32);
        for &element in ElementType::ALL {
            let mut literal = matrix(&ctx, element, &[0, 1]);
            let before = literal.clone();
            let bytes = if element == ElementType::Pred {
                vec![1]
            } else {
                vec![0xa5; element.bit_width().div_ceil(8) as usize]
            };
            write_element(&ctx, &mut literal, &[1, 1], &bytes).unwrap();
            assert_eq!(read_element(&ctx, &literal, &[1, 1]).unwrap(), bytes);
            for index in [[0, 0], [1, 0], [0, 1], [0, 2], [1, 2]] {
                assert_eq!(
                    read_element(&ctx, &literal, &index).unwrap(),
                    read_element(&ctx, &before, &index).unwrap()
                );
            }
            literal.verify(&ctx).unwrap();
        }
    }

    #[test]
    fn write_wrong_width_is_atomic() {
        let ctx = context();
        let mut literal = integer_scalar(&ctx, ElementType::S32, 5).unwrap();
        let before = literal.clone();
        for bytes in [vec![], vec![1], vec![1; 5]] {
            assert!(write_element(&ctx, &mut literal, &[], &bytes).is_err());
            assert_eq!(literal, before);
        }
    }

    #[test]
    fn invalid_predicate_write_is_atomic() {
        let ctx = context();
        let ty = shape::scalar(&ctx, ElementType::Pred).unwrap();
        let mut literal = Literal::from_bits(&ctx, ty, &[0]).unwrap();
        let before = literal.clone();
        assert!(write_element(&ctx, &mut literal, &[], &[2]).is_err());
        assert_eq!(literal, before);
        write_element(&ctx, &mut literal, &[], &[1]).unwrap();
        assert!(scalar_predicate(&ctx, &literal).unwrap());
    }

    #[test]
    fn subbyte_raw_writes_preserve_unused_high_bits() {
        let ctx = context();
        for element in [
            ElementType::S1,
            ElementType::U2,
            ElementType::S4,
            ElementType::F4E2M1FN,
            ElementType::F6E2M3FN,
            ElementType::F6E3M2FN,
        ] {
            let ty = shape::scalar(&ctx, element).unwrap();
            let mut literal = Literal::from_storage_bytes(&ctx, ty, vec![0]).unwrap();
            write_element(&ctx, &mut literal, &[], &[0xff]).unwrap();
            assert_eq!(read_element(&ctx, &literal, &[]).unwrap(), &[0xff]);
            assert_eq!(
                literal.bits(&ctx).unwrap(),
                vec![(1u128 << element.bit_width()) - 1]
            );
        }
    }

    #[test]
    fn dynamic_write_does_not_touch_inactive_storage_or_sizes() {
        let ctx = context();
        let mut literal = dynamic(&ctx, &[0, 1], &[1, 2], vec![0, 0xfa, 0xfb, 0, 0xfc, 0xfd]);
        write_element(&ctx, &mut literal, &[0, 1], &[7]).unwrap();
        assert_eq!(literal.storage, vec![0, 0xfa, 0xfb, 7, 0xfc, 0xfd]);
        assert_eq!(literal.dynamic_sizes, vec![1, 2]);
        let before = literal.clone();
        assert!(write_element(&ctx, &mut literal, &[1, 0], &[8]).is_err());
        assert_eq!(literal, before);
    }

    #[test]
    fn all_32_raw_formats_relayout_round_trip_without_numeric_conversion() {
        let ctx = context();
        for &element in ElementType::ALL {
            let source = matrix(&ctx, element, &[1, 0]);
            let target = shape(&ctx, element, &[2, 3], &[0, 1]);
            let result = relayout(&ctx, &source, target).unwrap();
            assert!(source.equal_logical_bits(&ctx, &result).unwrap());
            for i in 0..2 {
                for j in 0..3 {
                    assert_eq!(
                        read_element(&ctx, &source, &[i, j]).unwrap(),
                        read_element(&ctx, &result, &[i, j]).unwrap()
                    );
                }
            }
            assert_eq!(relayout(&ctx, &result, source.shape).unwrap(), source);
        }
    }

    #[test]
    fn bounded_dynamic_relayout_preserves_inactive_bytes_and_runtime_sizes() {
        let ctx = context();
        let source = dynamic(&ctx, &[1, 0], &[1, 2], vec![1, 2, 0xa3, 0xa4, 0xa5, 0xa6]);
        let target = dynamic(&ctx, &[0, 1], &[3, 2], vec![0; 6]).shape;
        let result = relayout(&ctx, &source, target).unwrap();
        assert_eq!(result.storage, vec![1, 0xa3, 0xa5, 2, 0xa4, 0xa6]);
        assert_eq!(result.dynamic_sizes, vec![1, 2]);
        assert_eq!(relayout(&ctx, &result, source.shape).unwrap(), source);
    }

    #[test]
    fn copying_dynamic_layouts_transfers_full_storage_and_runtime_sizes() {
        let ctx = context();
        for rows in 0..=3 {
            let source = dynamic(&ctx, &[1, 0], &[rows, 2], vec![1, 2, 3, 4, 5, 6]);
            let mut destination = dynamic(&ctx, &[0, 1], &[3, 2], vec![0; 6]);
            let ty = destination.shape;
            copy_sub_literal(&ctx, &mut destination, &[], &source, &[]).unwrap();
            assert_eq!(destination.shape, ty);
            assert_eq!(destination.dynamic_sizes, vec![rows, 2]);
            assert_eq!(destination.storage, vec![1, 3, 5, 2, 4, 6]);
        }
    }

    #[test]
    fn upstream_copy_from_and_to_zero_element() {
        let ctx = context();
        let source = Literal::from_storage_bytes(
            &ctx,
            shape(&ctx, ElementType::U8, &[0, 2], &[1, 0]),
            vec![],
        )
        .unwrap();
        let mut destination = Literal::from_storage_bytes(
            &ctx,
            shape(&ctx, ElementType::U8, &[0, 2], &[0, 1]),
            vec![],
        )
        .unwrap();
        copy_sub_literal(&ctx, &mut destination, &[], &source, &[]).unwrap();
        assert!(destination.storage.is_empty());
    }

    #[test]
    fn upstream_copy_from_nil_shape_and_tokens() {
        let ctx = context();
        let token = Literal::token(&ctx);
        let source = Literal::tuple(
            &ctx,
            vec![token.clone(), Literal::tuple(&ctx, vec![]).unwrap()],
        )
        .unwrap();
        let mut destination = source.clone();
        copy_sub_literal(&ctx, &mut destination, &[], &source, &[]).unwrap();
        assert_eq!(destination, source);
        assert_eq!(relayout(&ctx, &source, source.shape).unwrap(), source);
    }

    #[test]
    fn upstream_copy_from_tuples_preserves_other_subtrees() {
        let ctx = context();
        let matrix = matrix(&ctx, ElementType::C128, &[1, 0]);
        let inner = Literal::tuple(
            &ctx,
            vec![
                integer_scalar(&ctx, ElementType::S32, 42).unwrap(),
                Literal::token(&ctx),
            ],
        )
        .unwrap();
        let mut destination = Literal::tuple(&ctx, vec![matrix.clone(), inner]).unwrap();
        let replacement = Literal::tuple(
            &ctx,
            vec![
                integer_scalar(&ctx, ElementType::S32, -5).unwrap(),
                Literal::token(&ctx),
            ],
        )
        .unwrap();
        copy_sub_literal(&ctx, &mut destination, &[1], &replacement, &[]).unwrap();
        assert_eq!(destination.tuple_literals[0], matrix);
        assert_eq!(
            scalar_integer(&ctx, destination.sub_literal(&ctx, &[1, 0]).unwrap()).unwrap(),
            -5
        );
    }

    #[test]
    fn nested_tuple_relayout_preserves_target_types_and_token() {
        let ctx = context();
        let array = matrix(&ctx, ElementType::F32, &[1, 0]);
        let source = Literal::tuple(
            &ctx,
            vec![Literal::tuple(&ctx, vec![array.clone(), Literal::token(&ctx)]).unwrap()],
        )
        .unwrap();
        let array_target = shape(&ctx, ElementType::F32, &[2, 3], &[0, 1]);
        let inner_target = shape::tuple(&ctx, vec![array_target, shape::token(&ctx)]).unwrap();
        let target = shape::tuple(&ctx, vec![inner_target]).unwrap();
        let result = relayout(&ctx, &source, target).unwrap();
        assert_eq!(result.shape, target);
        assert_eq!(
            result.sub_literal(&ctx, &[0, 0]).unwrap().shape,
            array_target
        );
        assert!(source.equal_logical_bits(&ctx, &result).unwrap());
    }

    #[test]
    fn copying_scalars_and_subtrees_preserves_value_without_aliasing_storage() {
        let ctx = context();
        for &element in ElementType::ALL {
            let scalar_type = shape::scalar(&ctx, element).unwrap();
            let scalar = Literal::from_bits(&ctx, scalar_type, &[1]).unwrap();
            let zero = Literal::from_bits(&ctx, scalar_type, &[0]).unwrap();
            for nested in [false, true] {
                let mut source = if nested {
                    Literal::tuple(&ctx, vec![scalar.clone()]).unwrap()
                } else {
                    scalar.clone()
                };
                let mut destination = if nested {
                    Literal::tuple(&ctx, vec![zero.clone()]).unwrap()
                } else {
                    zero.clone()
                };
                let path: &[usize] = if nested { &[0] } else { &[] };
                copy_sub_literal(&ctx, &mut destination, path, &source, path).unwrap();
                assert_eq!(destination, source);
                let source_leaf = if nested {
                    &mut source.tuple_literals[0]
                } else {
                    &mut source
                };
                source_leaf.storage[0] = 0;
                assert_ne!(destination, source, "{element:?}, nested={nested}");
            }
        }
    }

    #[test]
    fn rejected_subtree_copies_leave_destination_unchanged() {
        let ctx = context();
        let scalar = integer_scalar(&ctx, ElementType::U8, 7).unwrap();
        let mut malformed = scalar.clone();
        malformed.storage.clear();
        let matrix = matrix(&ctx, ElementType::U8, &[1, 0]);
        for (source, target, source_path, destination_path) in [
            (&scalar, &matrix, &[][..], &[][..]),
            (&scalar, &scalar, &[][..], &[0][..]),
            (&scalar, &scalar, &[0][..], &[][..]),
            (&malformed, &scalar, &[][..], &[][..]),
        ] {
            let mut destination = target.clone();
            assert!(
                copy_sub_literal(
                    &ctx,
                    &mut destination,
                    destination_path,
                    source,
                    source_path
                )
                .is_err()
            );
            assert_eq!(&destination, target);
        }
    }

    #[test]
    fn relayout_rejects_element_dimension_and_dynamic_flag_changes() {
        let ctx = context();
        let source = matrix(&ctx, ElementType::U8, &[1, 0]);
        for target in [
            shape(&ctx, ElementType::S8, &[2, 3], &[1, 0]),
            shape(&ctx, ElementType::U8, &[3, 2], &[1, 0]),
            shape::with_default_layout(
                &ctx,
                shape::dynamic_array(
                    &ctx,
                    ElementType::U8,
                    vec![Dimension::Bounded(2), Dimension::Static(3)],
                )
                .unwrap(),
            )
            .unwrap(),
        ] {
            assert!(relayout(&ctx, &source, target).is_err());
        }
    }

    #[test]
    fn relayout_normalizes_absent_layout_and_subbyte_packing_like_literal_constructor() {
        let ctx = context();
        let source = matrix(&ctx, ElementType::S4, &[1, 0]);
        let no_layout = shape::array(&ctx, ElementType::S4, vec![2, 3]).unwrap();
        assert_eq!(relayout(&ctx, &source, no_layout).unwrap(), source);
        let packed = shape::with_layout(
            &ctx,
            source.shape,
            Layout {
                element_size_in_bits: 4,
                ..Layout::descending(2)
            },
        )
        .unwrap();
        assert_eq!(relayout(&ctx, &source, packed).unwrap(), source);
    }

    #[test]
    fn integer_scalar_admission_and_extraction_follow_format_ranges() {
        let ctx = context();
        for &element in ElementType::ALL {
            if !element.is_integer() {
                assert!(integer_scalar(&ctx, element, 0).is_err());
                let ty = shape::scalar(&ctx, element).unwrap();
                let literal = Literal::from_bits(&ctx, ty, &[0]).unwrap();
                assert!(scalar_integer(&ctx, &literal).is_err());
                continue;
            }
            let (minimum, maximum) = ranges(element);
            assert!(integer_scalar(&ctx, element, minimum - 1).is_err());
            assert!(integer_scalar(&ctx, element, maximum + 1).is_err());
            let mut samples = vec![minimum, 0, maximum];
            if element == ElementType::U64 {
                samples.extend([i64::MAX as i128, i64::MAX as i128 + 1]);
            }
            for value in samples {
                let literal = integer_scalar(&ctx, element, value).unwrap();
                assert_eq!(shape::rank(&ctx, literal.shape).unwrap(), 0);
                assert_eq!(shape::element_type(&ctx, literal.shape).unwrap(), element);
                assert_eq!(
                    scalar_integer(&ctx, &literal).ok(),
                    i64::try_from(value).ok(),
                    "{element:?}, value={value}"
                );
            }
        }
    }

    #[test]
    fn narrow_signed_extraction_masks_padding_then_sign_extends() {
        let ctx = context();
        for element in [ElementType::S1, ElementType::S2, ElementType::S4] {
            let ty = shape::scalar(&ctx, element).unwrap();
            let literal = Literal::from_storage_bytes(&ctx, ty, vec![0xff]).unwrap();
            assert_eq!(scalar_integer(&ctx, &literal).unwrap(), -1);
        }
    }

    #[test]
    fn scalar_extraction_requires_unit_bounds_and_an_active_element() {
        let ctx = context();
        let ty = shape(&ctx, ElementType::S32, &[1, 1, 1], &[0, 2, 1]);
        let literal = Literal::from_bits(&ctx, ty, &[17]).unwrap();
        assert_eq!(scalar_integer(&ctx, &literal).unwrap(), 17);
        let non_scalar = matrix(&ctx, ElementType::S32, &[1, 0]);
        assert!(scalar_integer(&ctx, &non_scalar).is_err());
        for bound in [1, 3] {
            let ty = shape::dynamic_array(&ctx, ElementType::S8, vec![Dimension::Bounded(bound)])
                .unwrap();
            for active in 0..=bound {
                let literal = Literal::from_dynamic_storage_bytes(
                    &ctx,
                    ty,
                    vec![0xfe; bound as usize],
                    vec![active],
                )
                .unwrap();
                assert_eq!(
                    scalar_integer(&ctx, &literal).ok(),
                    (bound == 1 && active == 1).then_some(-2),
                    "bound={bound}, active={active}"
                );
            }
        }
    }

    #[test]
    fn predicates_extract_without_integer_coercion() {
        let ctx = context();
        let ty = shape(&ctx, ElementType::Pred, &[1, 1], &[1, 0]);
        for bits in [0, 1] {
            let literal = Literal::from_bits(&ctx, ty, &[bits]).unwrap();
            assert_eq!(scalar_predicate(&ctx, &literal).unwrap(), bits == 1);
            assert!(scalar_integer(&ctx, &literal).is_err());
        }
        let integer = integer_scalar(&ctx, ElementType::U8, 1).unwrap();
        assert!(scalar_predicate(&ctx, &integer).is_err());
    }

    #[test]
    fn malformed_predicate_and_tuple_scalar_extraction_fail() {
        let ctx = context();
        let ty = shape::scalar(&ctx, ElementType::Pred).unwrap();
        let mut literal = Literal::from_bits(&ctx, ty, &[1]).unwrap();
        literal.storage[0] = 2;
        assert!(scalar_predicate(&ctx, &literal).is_err());
        let tuple = Literal::tuple(
            &ctx,
            vec![integer_scalar(&ctx, ElementType::S8, 1).unwrap()],
        )
        .unwrap();
        assert!(scalar_integer(&ctx, &tuple).is_err());
        assert!(scalar_predicate(&ctx, &Literal::token(&ctx)).is_err());
    }

    #[test]
    fn all_32_formats_copy_one_element_between_different_layouts() {
        let ctx = context();
        for &element in ElementType::ALL {
            let source = matrix(&ctx, element, &[1, 0]);
            let mut destination = matrix(&ctx, element, &[0, 1]);
            let before = destination.clone();
            copy_element(&ctx, &mut destination, &[1, 2], &source, &[0, 1]).unwrap();
            assert_eq!(
                read_element(&ctx, &destination, &[1, 2]).unwrap(),
                read_element(&ctx, &source, &[0, 1]).unwrap()
            );
            assert_eq!(
                read_element(&ctx, &destination, &[0, 0]).unwrap(),
                read_element(&ctx, &before, &[0, 0]).unwrap()
            );
        }
    }

    #[test]
    fn element_copy_allows_different_array_ranks() {
        let ctx = context();
        let source = matrix(&ctx, ElementType::U8, &[1, 0]);
        let mut destination = integer_scalar(&ctx, ElementType::U8, 0).unwrap();
        copy_element(&ctx, &mut destination, &[], &source, &[1, 2]).unwrap();
        assert_eq!(read_element(&ctx, &destination, &[]).unwrap(), &[116]);
    }

    #[test]
    fn element_copy_rejects_equal_width_but_different_formats() {
        let ctx = context();
        let source = integer_scalar(&ctx, ElementType::U32, 7).unwrap();
        let mut destination = integer_scalar(&ctx, ElementType::S32, 0).unwrap();
        let before = destination.clone();
        assert!(copy_element(&ctx, &mut destination, &[], &source, &[]).is_err());
        assert_eq!(destination, before);
    }

    #[test]
    fn element_copy_active_bound_failures_leave_destination_unchanged() {
        let ctx = context();
        let source = dynamic(&ctx, &[0, 1], &[1, 2], vec![1, 2, 3, 4, 5, 6]);
        let mut destination = source.clone();
        let before = destination.clone();
        assert!(copy_element(&ctx, &mut destination, &[0, 0], &source, &[2, 0]).is_err());
        assert_eq!(destination, before);
        assert!(copy_element(&ctx, &mut destination, &[1, 0], &source, &[0, 0]).is_err());
        assert_eq!(destination, before);
    }

    #[test]
    fn copy_from_static_source_retains_dynamic_destination_runtime_sizes() {
        let ctx = context();
        let source = Literal::from_bits(
            &ctx,
            shape(&ctx, ElementType::U8, &[3, 2], &[1, 0]),
            &[1, 2, 3, 4, 5, 6],
        )
        .unwrap();
        let mut destination = dynamic(&ctx, &[0, 1], &[1, 2], vec![0; 6]);
        copy_sub_literal(&ctx, &mut destination, &[], &source, &[]).unwrap();
        assert_eq!(destination.storage, vec![1, 3, 5, 2, 4, 6]);
        assert_eq!(destination.dynamic_sizes, vec![1, 2]);
    }

    #[test]
    fn copy_from_dynamic_source_to_static_destination_copies_full_bounds() {
        let ctx = context();
        let source = dynamic(&ctx, &[0, 1], &[1, 2], vec![1, 2, 3, 4, 5, 6]);
        let mut destination = Literal::from_bits(
            &ctx,
            shape(&ctx, ElementType::U8, &[3, 2], &[1, 0]),
            &[0; 6],
        )
        .unwrap();
        copy_sub_literal(&ctx, &mut destination, &[], &source, &[]).unwrap();
        assert_eq!(destination.storage, vec![1, 4, 2, 5, 3, 6]);
        assert!(destination.dynamic_sizes.is_empty());
    }

    #[test]
    fn rank_four_relayout_matches_all_logical_elements() {
        let ctx = context();
        let source_type = shape(&ctx, ElementType::U8, &[2, 2, 3, 3], &[2, 0, 3, 1]);
        let source = Literal::from_bits(&ctx, source_type, &(0..36).collect::<Vec<_>>()).unwrap();
        let target = shape(&ctx, ElementType::U8, &[2, 2, 3, 3], &[1, 3, 0, 2]);
        let result = relayout(&ctx, &source, target).unwrap();
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..3 {
                    for d in 0..3 {
                        let index = [a, b, c, d];
                        assert_eq!(
                            read_element(&ctx, &source, &index).unwrap(),
                            read_element(&ctx, &result, &index).unwrap()
                        );
                    }
                }
            }
        }
        assert_eq!(relayout(&ctx, &result, source_type).unwrap(), source);
    }

    #[test]
    fn relayout_retains_nonordering_physical_layout_metadata() {
        let ctx = context();
        let source = matrix(&ctx, ElementType::U8, &[1, 0]);
        let target = shape::with_layout(
            &ctx,
            source.shape,
            Layout {
                minor_to_major: vec![0, 1],
                memory_space: 7,
                tail_padding_alignment_in_elements: 8,
                ..Layout::descending(2)
            },
        )
        .unwrap();
        let result = relayout(&ctx, &source, target).unwrap();
        assert_eq!(result.shape, target);
        assert!(source.equal_logical_bits(&ctx, &result).unwrap());
    }

    #[test]
    fn later_dynamic_copy_failure_rolls_back_earlier_tuple_child() {
        let ctx = context();
        let source_type = shape::dynamic_array(
            &ctx,
            ElementType::U8,
            vec![Dimension::Static(3), Dimension::Bounded(2)],
        )
        .unwrap();
        // Native validation ignores the size entry on a static axis. Transferring
        // it to a bounded dynamic axis must fail without publishing earlier copies.
        let source_array =
            Literal::from_dynamic_storage_bytes(&ctx, source_type, vec![9; 6], vec![99, 1])
                .unwrap();
        let source = Literal::tuple(
            &ctx,
            vec![
                integer_scalar(&ctx, ElementType::S8, 7).unwrap(),
                source_array,
            ],
        )
        .unwrap();
        let mut destination = Literal::tuple(
            &ctx,
            vec![
                integer_scalar(&ctx, ElementType::S8, 0).unwrap(),
                dynamic(&ctx, &[1, 0], &[2, 2], vec![0; 6]),
            ],
        )
        .unwrap();
        let before = destination.clone();
        assert!(copy_sub_literal(&ctx, &mut destination, &[], &source, &[]).is_err());
        assert_eq!(destination, before);
    }

    #[test]
    fn upstream_copy_between_same_tuple_uses_owned_source_projection() {
        let ctx = context();
        let mut tuple = Literal::tuple(
            &ctx,
            vec![
                integer_scalar(&ctx, ElementType::S32, -2).unwrap(),
                integer_scalar(&ctx, ElementType::S32, 4).unwrap(),
            ],
        )
        .unwrap();
        let source = tuple.sub_literal(&ctx, &[0]).unwrap().clone();
        copy_sub_literal(&ctx, &mut tuple, &[1], &source, &[]).unwrap();
        assert_eq!(
            scalar_integer(&ctx, tuple.sub_literal(&ctx, &[0]).unwrap()).unwrap(),
            -2
        );
        assert_eq!(
            scalar_integer(&ctx, tuple.sub_literal(&ctx, &[1]).unwrap()).unwrap(),
            -2
        );
    }
}
