use super::*;
use crate::{IndexTransform, IndexTransformKind, OperandView};
use xlahlo::ops::movement::{ReverseOp, SliceOp, TransposeOp};

pub(super) fn transform(
    context: &Context,
    operation: Ptr<Operation>,
    source: &BufferDescriptor,
    result: &BufferDescriptor,
) -> Result<IndexTransform, PlanError> {
    if source.element_type != result.element_type {
        return Err(PlanError::UnsupportedFeature(
            "view changes element type".into(),
        ));
    }
    let operation_ref = Operation::get_op_dyn(operation, context);
    let hlo = pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref)
        .ok_or_else(|| PlanError::LoweringFailure("view has no xlahlo opcode".into()))?;
    let axes = |values: &[i64], rank: usize| -> Result<Vec<usize>, PlanError> {
        let mut seen = vec![false; rank];
        values
            .iter()
            .map(|&axis| {
                let axis = usize::try_from(axis).map_err(|_| PlanError::InvalidShape)?;
                if axis >= rank || std::mem::replace(&mut seen[axis], true) {
                    return Err(PlanError::LoweringFailure(
                        "view axes are out of range or repeated".into(),
                    ));
                }
                Ok(axis)
            })
            .collect()
    };
    let kind = match hlo.opcode() {
        Opcode::Reshape => {
            if source.num_elements()? != result.num_elements()? {
                return Err(PlanError::LoweringFailure(
                    "reshape changes element count".into(),
                ));
            }
            IndexTransformKind::Reshape
        }
        Opcode::Transpose => {
            let transpose = Operation::get_op::<TransposeOp>(operation, context)
                .ok_or_else(|| PlanError::LoweringFailure("expected transpose".into()))?;
            let dimensions = transpose
                .dimensions(context)
                .ok_or_else(|| PlanError::LoweringFailure("transpose dimensions missing".into()))?;
            let permutation = axes(&dimensions.0, source.shape.len())?;
            if permutation.len() != source.shape.len()
                || result.shape.len() != permutation.len()
                || permutation
                    .iter()
                    .enumerate()
                    .any(|(axis, &input)| result.shape[axis] != source.shape[input])
            {
                return Err(PlanError::LoweringFailure("invalid transpose shape".into()));
            }
            IndexTransformKind::Transpose(permutation)
        }
        Opcode::Reverse => {
            let reverse = Operation::get_op::<ReverseOp>(operation, context)
                .ok_or_else(|| PlanError::LoweringFailure("expected reverse".into()))?;
            let dimensions = reverse
                .dimensions(context)
                .ok_or_else(|| PlanError::LoweringFailure("reverse dimensions missing".into()))?;
            IndexTransformKind::Reverse(axes(&dimensions.0, source.shape.len())?)
        }
        Opcode::Slice => {
            let slice = Operation::get_op::<SliceOp>(operation, context)
                .ok_or_else(|| PlanError::LoweringFailure("expected slice".into()))?;
            let geometry = slice
                .geometry(context)
                .ok_or_else(|| PlanError::LoweringFailure("slice geometry missing".into()))?;
            let starts = geometry
                .starts
                .0
                .iter()
                .map(|&value| usize::try_from(value).map_err(|_| PlanError::InvalidShape))
                .collect::<Result<_, _>>()?;
            let strides = geometry
                .strides
                .0
                .iter()
                .map(|&value| usize::try_from(value).map_err(|_| PlanError::InvalidShape))
                .collect::<Result<_, _>>()?;
            IndexTransformKind::Slice { starts, strides }
        }
        Opcode::Broadcast => {
            let broadcast = Operation::get_op::<BroadcastOp>(operation, context)
                .ok_or_else(|| PlanError::LoweringFailure("expected broadcast".into()))?;
            let dimensions = broadcast
                .dimensions(context)
                .ok_or_else(|| PlanError::LoweringFailure("broadcast dimensions missing".into()))?;
            let dimensions = axes(&dimensions.0, result.shape.len())?;
            if dimensions.len() != source.shape.len() {
                return Err(PlanError::LoweringFailure(
                    "broadcast mapping rank mismatch".into(),
                ));
            }
            IndexTransformKind::Broadcast(dimensions)
        }
        opcode => {
            return Err(PlanError::UnsupportedFeature(format!(
                "view operation {opcode}"
            )));
        }
    };
    let transform = IndexTransform {
        source_shape: source.shape.clone(),
        result_shape: result.shape.clone(),
        kind,
    };
    transform.validate(source.element_type)?;
    Ok(transform)
}

pub(crate) fn source_position(
    scope: &Scope,
    mut position: Value,
    view: &OperandView,
) -> Result<Value, PlanError> {
    let empty = view.descriptor.num_elements()? == 0;
    let mut shape = view
        .transforms
        .first()
        .map(|transform| transform.source_shape.as_slice())
        .unwrap_or(&view.descriptor.shape);
    for transform in &view.transforms {
        transform.validate(view.descriptor.element_type)?;
        if shape != transform.source_shape {
            return Err(PlanError::LoweringFailure(
                "operand view transform source mismatch".into(),
            ));
        }
        let source =
            BufferDescriptor::dense_row_major(view.descriptor.element_type, shape.to_vec());
        if !empty && source.num_elements()? > u32::MAX as usize {
            return Err(PlanError::UnsupportedFeature(
                "indexed operand requires 64-bit kernel addressing".into(),
            ));
        }
        shape = &transform.result_shape;
    }
    if shape != view.descriptor.shape {
        return Err(PlanError::LoweringFailure(
            "operand view logical shape mismatch".into(),
        ));
    }
    if empty {
        return Ok(scope.const_usize(0));
    }
    for transform in view.transforms.iter().rev() {
        if matches!(transform.kind, IndexTransformKind::Reshape) {
            continue;
        }
        let result_strides = crate::view::dense_strides(&transform.result_shape)?;
        let source_strides = crate::view::dense_strides(&transform.source_shape)?;
        let empty = transform.result_shape.contains(&0);
        let mut coordinates = Vec::with_capacity(transform.result_shape.len());
        for (axis, &extent) in transform.result_shape.iter().enumerate() {
            let coordinate = if empty || extent <= 1 {
                scope.const_usize(0)
            } else {
                let divisor = scope.const_usize(
                    usize::try_from(result_strides[axis]).map_err(|_| PlanError::SizeOverflow)?,
                );
                let linear =
                    scope.register_with_result(&UDivOp::new(scope.ctx_mut(), position, divisor));
                let extent = scope
                    .const_usize(usize::try_from(extent).map_err(|_| PlanError::InvalidShape)?);
                scope.register_with_result(&URemOp::new(scope.ctx_mut(), linear, extent))
            };
            coordinates.push(coordinate);
        }
        if let IndexTransformKind::Strided { offset, strides } = &transform.kind {
            position = scope.const_usize(*offset);
            for ((coordinate, stride), _) in coordinates
                .iter()
                .zip(strides)
                .zip(&transform.result_shape)
                .filter(|((_, stride), extent)| **stride > 0 && **extent > 1)
            {
                let scale = scope
                    .const_usize(usize::try_from(*stride).map_err(|_| PlanError::SizeOverflow)?);
                let term =
                    scope.register_with_result(&IMulOp::new(scope.ctx_mut(), *coordinate, scale));
                position =
                    scope.register_with_result(&IAddOp::new(scope.ctx_mut(), position, term));
            }
            for ((coordinate, stride), _) in coordinates
                .iter()
                .zip(strides)
                .zip(&transform.result_shape)
                .filter(|((_, stride), extent)| **stride < 0 && **extent > 1)
            {
                let scale = scope.const_usize(
                    usize::try_from(stride.unsigned_abs()).map_err(|_| PlanError::SizeOverflow)?,
                );
                let term =
                    scope.register_with_result(&IMulOp::new(scope.ctx_mut(), *coordinate, scale));
                position =
                    scope.register_with_result(&ISubOp::new(scope.ctx_mut(), position, term));
            }
            continue;
        }
        let mut source_coordinates = match &transform.kind {
            IndexTransformKind::Transpose(permutation) => {
                let mut source = vec![scope.const_usize(0); transform.source_shape.len()];
                for (output, &input) in permutation.iter().enumerate() {
                    source[input] = coordinates[output];
                }
                source
            }
            IndexTransformKind::Broadcast(dimensions) => dimensions
                .iter()
                .enumerate()
                .map(|(input, &output)| {
                    if transform.source_shape[input] == 1 {
                        scope.const_usize(0)
                    } else {
                        coordinates[output]
                    }
                })
                .collect(),
            _ => coordinates,
        };
        match &transform.kind {
            IndexTransformKind::Reverse(axes) => {
                for &axis in axes {
                    let last = scope.const_usize(
                        usize::try_from((transform.source_shape[axis] - 1).max(0))
                            .map_err(|_| PlanError::InvalidShape)?,
                    );
                    source_coordinates[axis] = scope.register_with_result(&ISubOp::new(
                        scope.ctx_mut(),
                        last,
                        source_coordinates[axis],
                    ));
                }
            }
            IndexTransformKind::Slice { starts, strides } => {
                for axis in 0..source_coordinates.len() {
                    let stride = scope.const_usize(strides[axis]);
                    let term = scope.register_with_result(&IMulOp::new(
                        scope.ctx_mut(),
                        source_coordinates[axis],
                        stride,
                    ));
                    let start = scope.const_usize(starts[axis]);
                    source_coordinates[axis] =
                        scope.register_with_result(&IAddOp::new(scope.ctx_mut(), start, term));
                }
            }
            _ => {}
        }
        position = scope.const_usize(0);
        for (coordinate, stride) in source_coordinates.into_iter().zip(source_strides) {
            let stride =
                scope.const_usize(usize::try_from(stride).map_err(|_| PlanError::SizeOverflow)?);
            let term =
                scope.register_with_result(&IMulOp::new(scope.ctx_mut(), coordinate, stride));
            position = scope.register_with_result(&IAddOp::new(scope.ctx_mut(), position, term));
        }
    }
    Ok(position)
}
