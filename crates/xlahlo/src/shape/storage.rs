use super::MATCH_DIMENSION;
use crate::{
    Diagnostic,
    types::{OpaqueType, TokenType, TupleType},
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

/// Follow checked tuple indices, returning the root for an empty index.
pub fn subshape(ctx: &Context, mut shape: TypeHandle, index: &[usize]) -> Result<TypeHandle> {
    for &axis in index {
        let next = shape
            .deref(ctx)
            .downcast_ref::<TupleType>()
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "index traverses non-tuple shape".into(),
                })
            })?
            .elements
            .get(axis)
            .copied()
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "tuple index out of range".into(),
                })
            })?;
        shape = next;
    }
    Ok(shape)
}

/// Return bounded byte storage size, using `pointer_size` for tuple entries.
pub fn byte_size(ctx: &Context, shape: TypeHandle, pointer_size: i64) -> Result<i64> {
    let ty = shape.deref(ctx);
    if ty.is::<TupleType>() || ty.is::<OpaqueType>() {
        if pointer_size <= 0 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "tuple/opaque size requires positive pointer size".into(),
            }));
        }
        let count = ty
            .downcast_ref::<TupleType>()
            .map_or(1, |t| t.elements.len() as i64);
        return count.checked_mul(pointer_size).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "pointer table byte size overflow".into(),
            })
        });
    }
    if ty.is::<TokenType>() {
        return Ok(0);
    }
    let array = super::array_view(ctx, shape)?;
    let count = super::element_count(ctx, array)?;
    let size = super::layout(ctx, array)?.map_or(0, |l| l.element_size_in_bits);
    if size != 0 {
        let bits = count.checked_mul(size).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "packed array bit size overflow".into(),
            })
        })?;
        Ok(bits / 8 + i64::from(bits % 8 != 0))
    } else {
        count
            .checked_mul(i64::from(
                super::element_type(ctx, array)?.bit_width().div_ceil(8),
            ))
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "array byte size overflow".into(),
                })
            })
    }
}

/// Test dense stride compatibility without moving elements; tiles and padding are ignored.
pub fn reshape_is_bitcast(
    ctx: &Context,
    input: TypeHandle,
    output: TypeHandle,
    ignore_element_type: bool,
) -> Result<bool> {
    let input = super::array_view(ctx, input)?;
    let output = super::array_view(ctx, output)?;
    fn strides(ctx: &Context, shape: TypeHandle) -> Result<Vec<i64>> {
        let dimensions = super::dimensions(ctx, shape)?;
        let layout = super::layout(ctx, shape)?.ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "reshape bitcast requires layouts".into(),
            })
        })?;
        if layout.minor_to_major.len() != dimensions.len() {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "reshape bitcast requires complete minor order".into(),
            }));
        }
        let mut result = vec![0; dimensions.len()];
        let mut stride = 1i64;
        for &axis in &layout.minor_to_major {
            result[axis as usize] = stride;
            stride = stride
                .checked_mul(dimensions[axis as usize])
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "reshape stride overflow".into(),
                    })
                })?;
        }
        Ok(result)
    }
    if super::layout(ctx, input)?.is_none() || super::layout(ctx, output)?.is_none() {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "reshape bitcast requires layouts".into(),
        }));
    }
    if !ignore_element_type && super::element_type(ctx, input)? != super::element_type(ctx, output)?
    {
        return Ok(false);
    }
    let count = super::element_count(ctx, input)?;
    if count != super::element_count(ctx, output)? {
        return Ok(false);
    }
    if count == 0 {
        return Ok(true);
    }
    let input_strides = strides(ctx, input)?;
    let output_strides = strides(ctx, output)?;
    fn units(
        input: &[i64],
        output: &[i64],
        input_strides: &[i64],
        output_strides: &[i64],
    ) -> Result<bool> {
        let mut logical_stride = 1i64;
        for axis in (0..input.len()).rev() {
            if input[axis] > 1 {
                let mut remaining = logical_stride;
                let mut physical = 0i64;
                for target in (0..output.len()).rev() {
                    physical = physical
                        .checked_add(
                            (remaining % output[target])
                                .checked_mul(output_strides[target])
                                .ok_or_else(|| {
                                    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                                        message: "reshape physical index overflow".into(),
                                    })
                                })?,
                        )
                        .ok_or_else(|| {
                            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                                message: "reshape physical index overflow".into(),
                            })
                        })?;
                    remaining /= output[target];
                }
                if physical != input_strides[axis] {
                    return Ok(false);
                }
            }
            logical_stride = logical_stride.checked_mul(input[axis]).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "reshape logical stride overflow".into(),
                })
            })?;
        }
        Ok(true)
    }
    let input = super::dimensions(ctx, input)?;
    let output = super::dimensions(ctx, output)?;
    Ok(units(&input, &output, &input_strides, &output_strides)?
        && units(&output, &input, &output_strides, &input_strides)?)
}

/// Pinned ArrayDataSize: through the last logical element, without trailing
/// padding. Only the outer tile indexes; the second resolves MATCH packing.
/// Return tiled array data bytes, excluding tail-alignment padding.
pub fn array_data_size(ctx: &Context, shape: TypeHandle) -> Result<i64> {
    let shape = super::array_view(ctx, shape)?;
    let dimensions = super::dimensions(ctx, shape)?;
    let rank = dimensions.len();
    super::element_count(ctx, shape)?;
    let layout = super::layout(ctx, shape)?.ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "array data size requires layout".into(),
        })
    })?;
    if layout.minor_to_major.len() != rank {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "array data size requires full minor order".into(),
        }));
    }
    let overflow = || {
        pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "array data size overflow".into(),
        })
    };
    let mul = |a: i64, b: i64| a.checked_mul(b).ok_or_else(overflow);
    let add = |a: i64, b: i64| a.checked_add(b).ok_or_else(overflow);
    let divide = |a: i64, b: i64| a.checked_div(b).ok_or_else(overflow);
    let remainder = |a: i64, b: i64| a.checked_rem(b).ok_or_else(overflow);
    let ceil =
        |a: i64, b: i64| -> Result<i64> { add(divide(a, b)?, i64::from(remainder(a, b)? > 0)) };
    let count = if rank == 0 {
        1
    } else if rank == 1 {
        dimensions[0]
    } else {
        let tiles = layout
            .tiles
            .first()
            .map_or(&[][..], |tile| tile.dimensions.as_slice());
        let sub_tiles = layout
            .tiles
            .get(1)
            .map_or(&[][..], |tile| tile.dimensions.as_slice());
        let resolve = |axis: usize, bound: i64| -> Result<i64> {
            let sub_axis = axis as isize + sub_tiles.len() as isize - tiles.len() as isize;
            let packing = if sub_axis >= 0 && (sub_axis as usize) < sub_tiles.len() {
                sub_tiles[sub_axis as usize].max(1)
            } else if axis.checked_add(2) == Some(tiles.len()) && !sub_tiles.is_empty() {
                sub_tiles[0].max(1)
            } else {
                1
            };
            mul(ceil(bound.max(1), packing)?, packing)
        };
        let mut outer = 1;
        for (axis, &tile) in tiles.iter().enumerate() {
            let size = if tile == MATCH_DIMENSION {
                let minor = tiles.len() - 1 - axis;
                if minor >= rank {
                    continue;
                }
                resolve(axis, dimensions[layout.minor_to_major[minor] as usize])?
            } else {
                tile
            };
            outer = mul(outer, size)?;
        }
        let (mut inner, mut index) = (1, 0);
        for (minor, &logical) in layout.minor_to_major.iter().enumerate() {
            let bound = dimensions[logical as usize];
            let coordinate = bound.checked_sub(1).ok_or_else(overflow)?;
            if minor < tiles.len() {
                let axis = tiles.len() - 1 - minor;
                let tile = if tiles[axis] == MATCH_DIMENSION {
                    resolve(axis, bound)?
                } else {
                    tiles[axis]
                };
                index = add(
                    index,
                    add(
                        mul(outer, divide(coordinate, tile)?)?,
                        mul(inner, remainder(coordinate, tile)?)?,
                    )?,
                )?;
                outer = mul(outer, ceil(bound, tile)?)?;
                inner = mul(inner, tile)?;
            } else {
                index = add(index, mul(coordinate, outer)?)?;
                outer = mul(outer, bound)?;
            }
        }
        add(index, 1)?
    };
    if count < 0 {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "array data size is negative".into(),
        }));
    }
    if layout.element_size_in_bits != 0 {
        ceil(mul(count, layout.element_size_in_bits)?, 8)
    } else {
        mul(
            count,
            i64::from(super::element_type(ctx, shape)?.bit_width().div_ceil(8)),
        )
    }
}
