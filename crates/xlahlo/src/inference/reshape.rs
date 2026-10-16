use crate::{Dimension, Error, Result, Shape};

fn product(dimensions: &[i64]) -> Result<i64> {
    dimensions.iter().try_fold(1_i64, |value, &dimension| {
        value
            .checked_mul(dimension)
            .ok_or_else(|| Error::new("overflow", "reshape factor product exceeds i64"))
    })
}

fn common_factors(left: &[i64], right: &[i64]) -> Result<Vec<(usize, usize)>> {
    if left == right {
        return Ok((0..=left.len()).map(|index| (index, index)).collect());
    }
    let mut bounds = Vec::new();
    let (mut i, mut j) = (0, 0);
    let mut previous = None;
    while i < left.len() && j < right.len() && left[i] == right[j] {
        bounds.push((i, j));
        previous = Some((i, j));
        i += 1;
        j += 1;
    }
    let left_tail = product(&left[i..])?;
    let right_tail = product(&right[j..])?;
    if left_tail != right_tail {
        return Ok(vec![(0, 0), (left.len(), right.len())]);
    }
    if left_tail == 0 {
        bounds.extend([(i, j), (left.len(), right.len())]);
        return Ok(bounds);
    }
    let (mut left_size, mut right_size) = (1_i64, 1_i64);
    loop {
        if left_size == right_size && previous != Some((i, j)) {
            bounds.push((i, j));
            previous = Some((i, j));
            continue;
        }
        if i == left.len() && j == right.len() {
            break;
        }
        let advance_left = left_size < right_size
            || (i < left.len()
                && (j == right.len() || (left_size == right_size && left[i] <= right[j])));
        let advance_right = right_size < left_size
            || (j < right.len()
                && (i == left.len() || (right_size == left_size && right[j] <= left[i])));
        if advance_left {
            left_size = left_size
                .checked_mul(left[i])
                .ok_or_else(|| Error::new("overflow", "reshape factor product exceeds i64"))?;
            i += 1;
        }
        if advance_right {
            right_size = right_size
                .checked_mul(right[j])
                .ok_or_else(|| Error::new("overflow", "reshape factor product exceeds i64"))?;
            j += 1;
        }
    }
    Ok(bounds)
}

/// Infer an HLO reshape with statically specified output bounds. An optional
/// inferred dimension resolves ambiguous dynamic splits, as in pinned XLA.
pub fn infer_reshape(
    operand: &Shape,
    dimensions: &[i64],
    inferred_dimension: Option<usize>,
) -> Result<Shape> {
    let rank = operand.rank()?;
    let input = (0..rank)
        .map(|index| operand.dimension(index))
        .collect::<Result<Vec<_>>>()?;
    let output =
        Shape::array(operand.element_type(), dimensions.to_vec())?.with_default_layout()?;
    if input.contains(&Dimension::Unbounded) {
        if input
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Bounded(_)))
        {
            return Err(Error::new(
                "reshape",
                "mixed bounded and unbounded reshape input",
            ));
        }
        return Ok(output);
    }
    if operand.element_count()? != output.element_count()? {
        return Err(Error::new("reshape", "reshape element counts differ"));
    }
    let factors = common_factors(operand.dimensions(), dimensions)?;
    let mut dynamic = vec![false; dimensions.len()];
    for (index, dimension) in input.iter().enumerate() {
        if !matches!(dimension, Dimension::Bounded(_)) {
            continue;
        }
        let group = factors
            .windows(2)
            .find(|group| index >= group[0].0 && index < group[1].0)
            .ok_or_else(|| Error::new("reshape", "dynamic input dimension has no factor group"))?;
        let ((input_start, output_start), (input_end, output_end)) = (group[0], group[1]);
        if input_end - input_start > 1 && output_end - output_start > 1 {
            return Err(Error::new(
                "unsupported",
                "dynamic reshape dimension is both combined and split",
            ));
        }
        let mut start = output_start;
        let mut end = output_end;
        for &(input, output) in &factors {
            if input == input_start {
                start = start.min(output);
            }
            if input == input_end {
                end = end.max(output);
            }
        }
        let mut target = None;
        if operand.dimensions()[index] == 1 && !dimensions.is_empty() {
            if index == 0 {
                target = Some(0);
            }
            if index == rank - 1 {
                target = Some(dimensions.len() - 1);
            }
            if target.is_none() {
                return Err(Error::new(
                    "unsupported",
                    "interior dynamic unit reshape dimension",
                ));
            }
        }
        if target.is_none() {
            target = if end - start == 1 {
                Some(start)
            } else if end - start > 1 {
                inferred_dimension
            } else {
                None
            };
        }
        if target.is_none() {
            target = (start..end).find(|&axis| dimensions[axis] != 1);
        }
        if let Some(target) = target {
            *dynamic.get_mut(target).ok_or_else(|| {
                Error::new("reshape", "inferred reshape dimension out of range")
            })? = true;
        }
    }
    let mut wire = output.into_proto();
    wire.is_dynamic_dimension = dynamic;
    Shape::from_proto(wire)
}

pub fn infer_static_reshape(operand: &Shape, dimensions: &[i64]) -> Result<Shape> {
    infer_reshape(operand, dimensions, None)
}
