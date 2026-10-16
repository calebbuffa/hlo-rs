use crate::{Error, PrimitiveType, Result, Shape, UNBOUNDED_SIZE};

fn one_or_tuple(shapes: &[Shape]) -> Result<Shape> {
    if shapes.len() == 1 {
        Ok(shapes[0].clone())
    } else {
        Shape::tuple(shapes.to_vec())
    }
}

fn dimension(shape: &Shape, axis: i64) -> Result<usize> {
    let axis = usize::try_from(axis)
        .map_err(|_| Error::new("collective", "negative collective dimension"))?;
    if axis >= shape.rank()? {
        return Err(Error::new(
            "collective",
            "collective dimension out of range",
        ));
    }
    Ok(axis)
}

fn change_dimension(shape: &Shape, axis: usize, bound: i64, dynamic: bool) -> Result<Shape> {
    let mut wire = shape.clone().into_proto();
    wire.is_dynamic_dimension
        .resize(wire.dimensions.len(), false);
    wire.dimensions[axis] = bound;
    wire.is_dynamic_dimension[axis] = dynamic;
    Shape::from_proto(wire)
}

pub fn infer_all_gather(operands: &[Shape], axis: i64, shard_count: i64) -> Result<Shape> {
    if axis < 0 || shard_count <= 0 {
        return Err(Error::new(
            "collective",
            "all-gather requires nonnegative axis and positive shard count",
        ));
    }
    let outputs = operands
        .iter()
        .map(|operand| {
            let axis = dimension(operand, axis)?;
            let bound = operand.dimensions()[axis];
            let unbounded = bound == UNBOUNDED_SIZE;
            let bound = if unbounded {
                bound
            } else {
                bound
                    .checked_mul(shard_count)
                    .ok_or_else(|| Error::new("overflow", "all-gather dimension overflow"))?
            };
            change_dimension(operand, axis, bound, unbounded)
        })
        .collect::<Result<Vec<_>>>()?;
    one_or_tuple(&outputs)
}

pub fn infer_all_gather_start(operands: &[Shape], axis: i64, shard_count: i64) -> Result<Shape> {
    Shape::tuple(vec![
        one_or_tuple(operands)?,
        infer_all_gather(operands, axis, shard_count)?,
    ])
}

pub fn infer_all_gather_done(context: &Shape) -> Result<Shape> {
    context.subshape(&[1])
}

pub fn infer_all_reduce(operands: &[Shape]) -> Result<Shape> {
    for operand in operands {
        operand.rank()?;
    }
    one_or_tuple(operands)
}

pub fn infer_all_reduce_start(operands: &[Shape]) -> Result<Shape> {
    infer_all_reduce(operands)
}

pub fn infer_all_reduce_done(context: &Shape) -> Result<Shape> {
    Ok(context.clone())
}

pub fn infer_reduce_scatter(operands: &[Shape], axis: i64, shard_count: i64) -> Result<Shape> {
    if axis < 0 || shard_count <= 0 {
        return Err(Error::new(
            "collective",
            "reduce-scatter requires nonnegative axis and positive shard count",
        ));
    }
    let outputs = operands
        .iter()
        .map(|operand| {
            let axis = dimension(operand, axis)?;
            let bound = operand.dimensions()[axis];
            if bound % shard_count != 0 {
                return Err(Error::new(
                    "collective",
                    "reduce-scatter dimension must be divisible by shard count",
                ));
            }
            let dynamic = operand
                .as_proto()
                .is_dynamic_dimension
                .get(axis)
                .copied()
                .unwrap_or(false);
            change_dimension(
                operand,
                axis,
                if dynamic {
                    UNBOUNDED_SIZE
                } else {
                    bound / shard_count
                },
                dynamic,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    one_or_tuple(&outputs)
}

pub fn infer_all_to_all(
    operand: &Shape,
    split_axis: i64,
    concat_axis: i64,
    split_count: i64,
) -> Result<Shape> {
    if split_count <= 0 {
        return Err(Error::new(
            "collective",
            "all-to-all requires positive split count",
        ));
    }
    let split = dimension(operand, split_axis)?;
    let concat = dimension(operand, concat_axis)?;
    if operand
        .dimensions()
        .iter()
        .enumerate()
        .any(|(axis, &bound)| {
            bound != UNBOUNDED_SIZE
                && operand.as_proto().is_dynamic_dimension.get(axis) == Some(&true)
        })
    {
        return Err(Error::new(
            "collective",
            "all-to-all does not support bounded dynamic shapes",
        ));
    }
    let mut dimensions = (0..operand.rank()?)
        .map(|axis| operand.dimension(axis))
        .collect::<Result<Vec<_>>>()?;
    if operand.dimensions()[split] != UNBOUNDED_SIZE {
        if operand.dimensions()[split] % split_count != 0 {
            return Err(Error::new(
                "collective",
                "all-to-all split dimension must be divisible by split count",
            ));
        }
        dimensions[split] = crate::Dimension::Static(operand.dimensions()[split] / split_count);
    }
    if dimensions[concat] != crate::Dimension::Unbounded {
        let bound = match dimensions[concat] {
            crate::Dimension::Static(bound) => bound,
            _ => {
                return Err(Error::new(
                    "collective",
                    "unexpected bounded all-to-all dimension",
                ));
            }
        };
        dimensions[concat] = crate::Dimension::Static(
            bound
                .checked_mul(split_count)
                .ok_or_else(|| Error::new("overflow", "all-to-all concatenation overflow"))?,
        );
    }
    Shape::dynamic_array(operand.element_type(), dimensions)?.with_default_layout()
}

pub fn infer_all_to_all_tuple(operands: &[Shape]) -> Result<Shape> {
    let first = operands
        .first()
        .ok_or_else(|| Error::new("collective", "tuple all-to-all requires operands"))?;
    fn normalized(shape: &Shape) -> Result<Shape> {
        let mut wire = shape.clone().into_proto();
        let mut pending = vec![&mut wire];
        while let Some(wire) = pending.pop() {
            if wire
                .element_type
                .enum_value()
                .is_ok_and(PrimitiveType::is_array_element)
            {
                wire.is_dynamic_dimension
                    .resize(wire.dimensions.len(), false);
            }
            if let Some(layout) = wire.layout.as_mut() {
                layout.memory_space = 0;
            }
            if matches!(
                wire.element_type.enum_value(),
                Ok(PrimitiveType::TUPLE | PrimitiveType::BUFFER)
            ) {
                pending.extend(&mut wire.tuple_shapes);
            }
        }
        Shape::from_proto(wire)
    }
    let first = normalized(first)?;
    fn unbounded(shape: &crate::proto::xla_data::ShapeProto) -> bool {
        let mut pending = vec![shape];
        while let Some(shape) = pending.pop() {
            if shape.logical_dimensions().contains(&UNBOUNDED_SIZE) {
                return true;
            }
            pending.extend(shape.logical_children());
        }
        false
    }
    for operand in operands {
        if unbounded(operand.as_proto()) {
            return Err(Error::new(
                "collective",
                "tuple all-to-all does not support unbounded dynamics",
            ));
        }
        if !normalized(operand)?.equal_including_layout(&first) {
            return Err(Error::new(
                "collective",
                "tuple all-to-all operands differ apart from memory space",
            ));
        }
    }
    Shape::tuple(operands.to_vec())
}

pub fn infer_collective_broadcast(operands: &[Shape], has_dynamic_root: bool) -> Result<Shape> {
    if has_dynamic_root {
        let (root, data) = operands
            .split_last()
            .filter(|(_, data)| !data.is_empty())
            .ok_or_else(|| {
                Error::new(
                    "collective",
                    "dynamic-root broadcast requires data and root operands",
                )
            })?;
        if root.element_type() != PrimitiveType::S32
            || root.rank()? != 1
            || root.element_count()? != data.len() as i64
        {
            return Err(Error::new(
                "collective",
                "dynamic-root broadcast requires S32 vector with one root per data operand",
            ));
        }
        return one_or_tuple(data);
    }
    let first = operands
        .first()
        .ok_or_else(|| Error::new("collective", "broadcast requires an operand"))?;
    first.rank()?;
    Ok(first.clone())
}

pub fn infer_collective_reduce(operands: &[Shape], has_dynamic_root: bool) -> Result<Shape> {
    let data = if has_dynamic_root {
        let (root, data) = operands
            .split_last()
            .filter(|(_, data)| !data.is_empty())
            .ok_or_else(|| {
                Error::new(
                    "collective",
                    "dynamic-root reduce requires data and root operands",
                )
            })?;
        let expected =
            Shape::array(PrimitiveType::S32, vec![data.len() as i64])?.with_default_layout()?;
        if !root.equal_including_layout(&expected) {
            return Err(Error::new(
                "collective",
                "dynamic-root reduce requires default-layout S32 vector with one root per data operand",
            ));
        }
        data
    } else {
        operands
    };
    infer_all_reduce(data)
}

pub fn infer_ragged_all_to_all(operands: &[Shape]) -> Result<Shape> {
    let output = operands
        .get(1)
        .ok_or_else(|| Error::new("collective", "ragged all-to-all requires an output operand"))?;
    output.rank()?;
    Ok(output.clone())
}

pub fn infer_collective_permute(operands: &[Shape], inplace: bool) -> Result<Shape> {
    if inplace {
        let [_, output, _, _] = operands else {
            return Err(Error::new(
                "collective",
                "in-place permute requires four operands",
            ));
        };
        Ok(output.clone())
    } else {
        infer_all_reduce(operands)
    }
}

pub fn infer_collective_permute_start(
    operands: &[Shape],
    contexts: &[Shape],
    inplace: bool,
) -> Result<Shape> {
    let mut shapes = if inplace {
        let [input, output, _, _] = operands else {
            return Err(Error::new(
                "collective",
                "in-place permute start requires four operands",
            ));
        };
        vec![input.clone(), output.clone()]
    } else {
        if operands.len() == 1 {
            operands[0].rank()?;
        }
        let shape = one_or_tuple(operands)?;
        vec![shape.clone(), shape]
    };
    shapes.extend_from_slice(contexts);
    Shape::tuple(shapes)
}

pub fn infer_collective_permute_done(context: &Shape) -> Result<Shape> {
    context.subshape(&[1])
}
