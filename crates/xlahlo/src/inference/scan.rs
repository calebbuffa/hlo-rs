use crate::proto::xla_data::ProgramShapeProto;
use crate::{Dimension, Error, Result, Shape};

fn elements(shape: &Shape) -> Result<Vec<Shape>> {
    if shape.is_tuple() {
        shape
            .as_proto()
            .tuple_shapes
            .iter()
            .cloned()
            .map(Shape::from_proto)
            .collect()
    } else {
        Ok(vec![shape.clone()])
    }
}

fn remove_axis(shape: &Shape, axis: usize) -> Result<Shape> {
    let rank = shape.rank()?;
    if axis >= rank {
        return Err(Error::new(
            "dimensions",
            "scan axis exceeds input/output rank",
        ));
    }
    Shape::dynamic_array(
        shape.element_type(),
        (0..rank)
            .filter(|&i| i != axis)
            .map(|i| shape.dimension(i))
            .collect::<Result<Vec<_>>>()?,
    )
}

fn same(a: &Shape, b: &Shape, strict_dynamic: bool, allow_mixed_precision: bool) -> bool {
    if strict_dynamic {
        if allow_mixed_precision {
            a.equal_ignoring_layout_and_fp_precision(b)
        } else {
            a.equal_ignoring_layout(b)
        }
    } else {
        a.compatible(b)
    }
}

pub(crate) fn verify_scan(
    operands: &[Shape],
    result: &Shape,
    body: &ProgramShapeProto,
    num_carries: i64,
    dimension: i64,
    strict_dynamic: bool,
    allow_mixed_precision: bool,
) -> Result<()> {
    let carries =
        usize::try_from(num_carries).map_err(|_| Error::new("scan", "negative carry count"))?;
    let axis =
        usize::try_from(dimension).map_err(|_| Error::new("dimensions", "negative scan axis"))?;
    let root = Shape::from_proto(
        body.result
            .as_ref()
            .ok_or_else(|| Error::new("callee", "scan body missing root"))?
            .clone(),
    )?;
    let roots = elements(&root)?;
    let results = elements(result)?;
    if carries > operands.len()
        || carries > results.len()
        || (carries == operands.len() && carries == results.len())
        || body.parameters.len() != operands.len()
        || roots.len() != results.len()
    {
        return Err(Error::new(
            "scan",
            "scan requires matching operand/parameter, root/result and carry counts and at least one input or output",
        ));
    }
    let inputs = operands.len() - carries;
    let outputs = results.len() - carries;
    let size_source = if inputs > 0 {
        &operands[0]
    } else {
        &results[0]
    };
    let size = size_source.dimension(axis)?;
    let size_bound = match size {
        Dimension::Static(n) | Dimension::Bounded(n) => n,
        Dimension::Unbounded => crate::UNBOUNDED_SIZE,
    };
    let parameters = body
        .parameters
        .iter()
        .cloned()
        .map(Shape::from_proto)
        .collect::<Result<Vec<_>>>()?;
    for (index, (operand, parameter)) in operands.iter().zip(&parameters).enumerate() {
        let expected = if index < inputs {
            let slice = remove_axis(operand, axis)?;
            if operand.dimensions()[axis] != size_bound {
                return Err(Error::new(
                    "shape",
                    "scan inputs have different scan dimension extents",
                ));
            }
            slice
        } else {
            operand.clone()
        };
        if !same(&expected, parameter, strict_dynamic, allow_mixed_precision) {
            return Err(Error::new(
                "callee",
                "scan operand slice/carry differs from body parameter",
            ));
        }
    }
    for index in 0..carries {
        if !same(
            &parameters[index + inputs],
            &roots[index + outputs],
            strict_dynamic,
            allow_mixed_precision,
        ) {
            return Err(Error::new(
                "callee",
                "scan body carry parameter and root differ",
            ));
        }
    }
    for (index, (root, result)) in roots.iter().zip(&results).enumerate() {
        let expected = if index < outputs {
            let slice = remove_axis(result, axis)?;
            if result.dimensions()[axis] != size_bound {
                return Err(Error::new(
                    "shape",
                    "scan outputs have different scan dimension extents",
                ));
            }
            slice
        } else {
            result.clone()
        };
        if !same(root, &expected, strict_dynamic, allow_mixed_precision) {
            return Err(Error::new(
                "callee",
                "scan result slice/carry differs from body root",
            ));
        }
    }
    Ok(())
}

/// `scan_size` supplies the output iteration extent when there are no scanned inputs.
pub fn infer_scan(
    operands: &[Shape],
    body: &ProgramShapeProto,
    num_carries: i64,
    dimension: i64,
    scan_size: Option<Dimension>,
) -> Result<Shape> {
    let carries =
        usize::try_from(num_carries).map_err(|_| Error::new("scan", "negative carry count"))?;
    let axis =
        usize::try_from(dimension).map_err(|_| Error::new("dimensions", "negative scan axis"))?;
    let root = Shape::from_proto(
        body.result
            .as_ref()
            .ok_or_else(|| Error::new("callee", "scan body missing root"))?
            .clone(),
    )?;
    let roots = elements(&root)?;
    if carries > operands.len() || carries > roots.len() {
        return Err(Error::new("scan", "carry count exceeds operand/root count"));
    }
    let size = if operands.len() > carries {
        operands[0].dimension(axis)?
    } else {
        scan_size.ok_or_else(|| {
            Error::new(
                "scan",
                "scan without inputs requires an explicit iteration extent",
            )
        })?
    };
    if scan_size.is_some_and(|requested| requested != size) {
        return Err(Error::new(
            "scan",
            "explicit scan extent differs from input extent",
        ));
    }
    let outputs = roots.len() - carries;
    let results = roots
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            if index >= outputs {
                return Ok(shape.clone());
            }
            let rank = shape.rank()?;
            if axis > rank {
                return Err(Error::new("dimensions", "scan axis exceeds result rank"));
            }
            let mut dimensions = (0..rank)
                .map(|axis| shape.dimension(axis))
                .collect::<Result<Vec<_>>>()?;
            dimensions.insert(axis, size);
            Shape::dynamic_array(shape.element_type(), dimensions)?.with_default_layout()
        })
        .collect::<Result<Vec<_>>>()?;
    let result = if root.is_tuple() {
        Shape::tuple(results)?
    } else {
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::new("scan", "scan missing result"))?
    };
    verify_scan(
        operands,
        &result,
        body,
        num_carries,
        dimension,
        false,
        false,
    )?;
    Ok(result)
}
