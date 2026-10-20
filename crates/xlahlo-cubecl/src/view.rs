use crate::{BufferDescriptor, BufferId, PlanError};

#[cfg(test)]
mod tests;

/// A logical tensor backed by an existing plan allocation.
///
/// Transforms are stored from storage toward the logical value. Consumers
/// apply them in reverse order to map logical row-major positions to storage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperandView {
    pub buffer: BufferId,
    pub descriptor: BufferDescriptor,
    pub transforms: Vec<IndexTransform>,
}

/// Static xlahlo coordinate transformation. No caller layout metadata is used.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndexTransform {
    pub source_shape: Vec<i64>,
    pub result_shape: Vec<i64>,
    pub kind: IndexTransformKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IndexTransformKind {
    Reshape,
    /// Output axis -> source axis.
    Transpose(Vec<usize>),
    Reverse(Vec<usize>),
    Slice {
        starts: Vec<usize>,
        strides: Vec<usize>,
    },
    /// Source axis -> output axis.
    Broadcast(Vec<usize>),
}

/// A logical output view over a retained physical allocation.
///
/// Affine offsets and signed strides are in elements from the start of `buffer`.
/// The caller retains that physical buffer; it must not rebase it and apply
/// `offset` again. A negative stride counts backward from the logical origin.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OutputView {
    pub buffer: BufferId,
    pub descriptor: BufferDescriptor,
    pub map: OutputIndexMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutputIndexMap {
    Affine {
        offset: usize,
        strides: Vec<i64>,
    },
    /// Exact logical indexing when affine strides cannot represent the view.
    /// Transforms run from storage toward the result; address evaluation applies
    /// them in reverse order to row-major logical coordinates.
    Composed {
        storage_shape: Vec<i64>,
        transforms: Vec<IndexTransform>,
    },
}

pub(crate) fn dense_strides(shape: &[i64]) -> Result<Vec<i64>, PlanError> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1i64;
    for axis in (0..shape.len()).rev() {
        if shape[axis] < 0 {
            return Err(PlanError::InvalidShape);
        }
        strides[axis] = stride;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or(PlanError::SizeOverflow)?;
    }
    Ok(strides)
}

impl OperandView {
    /// Emit a checked physical element index for a row-major logical position.
    /// Strategies opting into indexed operands can use this before `IndexOp`.
    pub fn emit_index(
        &self,
        scope: &cubecl_ir::Scope,
        position: pliron::value::Value,
    ) -> Result<pliron::value::Value, PlanError> {
        crate::lowering::views::source_position(scope, position, self)
    }

    pub(crate) fn dense(buffer: BufferId, descriptor: BufferDescriptor) -> Self {
        Self {
            buffer,
            descriptor,
            transforms: Vec::new(),
        }
    }

    pub(crate) fn output(&self, storage: &BufferDescriptor) -> Result<OutputView, PlanError> {
        let map = match self.affine_map(storage)? {
            Some(map) => map,
            None => OutputIndexMap::Composed {
                storage_shape: storage.shape.clone(),
                transforms: self.transforms.clone(),
            },
        };
        let output = OutputView {
            buffer: self.buffer,
            descriptor: self.descriptor.clone(),
            map,
        };
        output.validate(storage)?;
        Ok(output)
    }

    fn affine_map(&self, storage: &BufferDescriptor) -> Result<Option<OutputIndexMap>, PlanError> {
        let mut shape = storage.shape.clone();
        let mut strides = dense_strides(&shape)?;
        let mut offset = 0i128;
        for transform in &self.transforms {
            transform.validate(storage.element_type)?;
            if transform.source_shape != shape {
                return Err(PlanError::LoweringFailure(
                    "view transform source shape mismatch".into(),
                ));
            }
            match &transform.kind {
                IndexTransformKind::Reshape => {
                    let count =
                        BufferDescriptor::dense_row_major(storage.element_type, shape.clone())
                            .num_elements()?;
                    let next_count = BufferDescriptor::dense_row_major(
                        storage.element_type,
                        transform.result_shape.clone(),
                    )
                    .num_elements()?;
                    if count != next_count {
                        return Err(PlanError::LoweringFailure(
                            "reshape changes element count".into(),
                        ));
                    }
                    if count == 0 {
                        strides = dense_strides(&transform.result_shape)?;
                    } else {
                        let Some(next) =
                            Self::reshape_strides(&shape, &strides, &transform.result_shape)?
                        else {
                            return Ok(None);
                        };
                        strides = next;
                    }
                }
                IndexTransformKind::Transpose(permutation) => {
                    strides = permutation.iter().map(|&axis| strides[axis]).collect();
                }
                IndexTransformKind::Reverse(axes) => {
                    for &axis in axes {
                        if shape[axis] > 0 {
                            offset = offset
                                .checked_add(
                                    i128::from(shape[axis] - 1) * i128::from(strides[axis]),
                                )
                                .ok_or(PlanError::SizeOverflow)?;
                        }
                        strides[axis] =
                            strides[axis].checked_neg().ok_or(PlanError::SizeOverflow)?;
                    }
                }
                IndexTransformKind::Slice {
                    starts,
                    strides: steps,
                } => {
                    for axis in 0..shape.len() {
                        offset = offset
                            .checked_add(starts[axis] as i128 * i128::from(strides[axis]))
                            .ok_or(PlanError::SizeOverflow)?;
                        strides[axis] = strides[axis]
                            .checked_mul(
                                i64::try_from(steps[axis]).map_err(|_| PlanError::SizeOverflow)?,
                            )
                            .ok_or(PlanError::SizeOverflow)?;
                    }
                }
                IndexTransformKind::Broadcast(dimensions) => {
                    let mut next = vec![0; transform.result_shape.len()];
                    for (source_axis, &result_axis) in dimensions.iter().enumerate() {
                        if shape[source_axis] != 1 {
                            next[result_axis] = strides[source_axis];
                        }
                    }
                    strides = next;
                }
            }
            shape = transform.result_shape.clone();
        }
        let empty = self.descriptor.num_elements()? == 0;
        Ok(Some(OutputIndexMap::Affine {
            offset: if empty {
                0
            } else {
                usize::try_from(offset).map_err(|_| PlanError::SizeOverflow)?
            },
            strides,
        }))
    }

    fn reshape_strides(
        shape: &[i64],
        strides: &[i64],
        result: &[i64],
    ) -> Result<Option<Vec<i64>>, PlanError> {
        if shape.is_empty() {
            return Ok(Some(dense_strides(result)?));
        }
        let mut next = vec![0; result.len()];
        let mut output_axis = result.len();
        let mut chunk_stride = strides[shape.len() - 1];
        let mut chunk_elements = 1i64;
        let mut output_elements = 1i64;
        for axis in (0..shape.len()).rev() {
            chunk_elements = chunk_elements
                .checked_mul(shape[axis])
                .ok_or(PlanError::SizeOverflow)?;
            let boundary = axis == 0
                || (shape[axis - 1] != 1
                    && strides[axis - 1]
                        != chunk_elements
                            .checked_mul(chunk_stride)
                            .ok_or(PlanError::SizeOverflow)?);
            if boundary {
                while output_axis > 0
                    && (output_elements < chunk_elements || result[output_axis - 1] == 1)
                {
                    output_axis -= 1;
                    next[output_axis] = output_elements
                        .checked_mul(chunk_stride)
                        .ok_or(PlanError::SizeOverflow)?;
                    output_elements = output_elements
                        .checked_mul(result[output_axis])
                        .ok_or(PlanError::SizeOverflow)?;
                }
                if output_elements != chunk_elements {
                    return Ok(None);
                }
                if axis > 0 {
                    chunk_stride = strides[axis - 1];
                    chunk_elements = 1;
                    output_elements = 1;
                }
            }
        }
        Ok((output_axis == 0).then_some(next))
    }
}

impl OutputView {
    pub(crate) fn validate(&self, storage: &BufferDescriptor) -> Result<(), PlanError> {
        if self.descriptor.element_type != storage.element_type {
            return Err(PlanError::LoweringFailure(
                "output view type or stride rank mismatch".into(),
            ));
        }
        let (offset, strides) = match &self.map {
            OutputIndexMap::Affine { offset, strides } => (*offset, strides),
            OutputIndexMap::Composed {
                storage_shape,
                transforms,
            } => {
                if storage_shape != &storage.shape {
                    return Err(PlanError::LoweringFailure(
                        "output view physical shape mismatch".into(),
                    ));
                }
                let mut shape = storage_shape;
                for transform in transforms {
                    if &transform.source_shape != shape {
                        return Err(PlanError::LoweringFailure(
                            "output view transform source mismatch".into(),
                        ));
                    }
                    transform.validate(storage.element_type)?;
                    shape = &transform.result_shape;
                }
                if shape != &self.descriptor.shape {
                    return Err(PlanError::LoweringFailure(
                        "output view logical shape mismatch".into(),
                    ));
                }
                return Ok(());
            }
        };
        if strides.len() != self.descriptor.shape.len() {
            return Err(PlanError::LoweringFailure(
                "output view stride rank mismatch".into(),
            ));
        }
        let count = self.descriptor.num_elements()?;
        let storage_count = storage.num_elements()?;
        if count == 0 {
            if offset > storage_count {
                return Err(PlanError::LoweringFailure(
                    "empty output view origin exceeds storage".into(),
                ));
            }
            return Ok(());
        }
        let mut low = offset as i128;
        let mut high = low;
        for (&extent, &stride) in self.descriptor.shape.iter().zip(strides) {
            let reach = i128::from(extent - 1) * i128::from(stride);
            low = low
                .checked_add(reach.min(0))
                .ok_or(PlanError::SizeOverflow)?;
            high = high
                .checked_add(reach.max(0))
                .ok_or(PlanError::SizeOverflow)?;
        }

        if low < 0 || high >= storage_count as i128 {
            return Err(PlanError::LoweringFailure(
                "output view reaches outside storage".into(),
            ));
        }
        Ok(())
    }
}

impl IndexTransform {
    pub(crate) fn validate(
        &self,
        element_type: xlahlo::types::ElementType,
    ) -> Result<(), PlanError> {
        let source = BufferDescriptor::dense_row_major(element_type, self.source_shape.clone());
        let result = BufferDescriptor::dense_row_major(element_type, self.result_shape.clone());
        let source_count = source.num_elements()?;
        let result_count = result.num_elements()?;
        let invalid = || PlanError::LoweringFailure("invalid logical view geometry".into());
        let unique = |axes: &[usize], rank: usize| {
            let mut seen = vec![false; rank];
            axes.iter()
                .all(|&axis| axis < rank && !std::mem::replace(&mut seen[axis], true))
        };
        match &self.kind {
            IndexTransformKind::Reshape if source_count == result_count => {}
            IndexTransformKind::Transpose(axes)
                if axes.len() == self.source_shape.len()
                    && axes.len() == self.result_shape.len()
                    && unique(axes, self.source_shape.len())
                    && axes.iter().enumerate().all(|(output, &input)| {
                        self.result_shape[output] == self.source_shape[input]
                    }) => {}
            IndexTransformKind::Reverse(axes)
                if self.source_shape == self.result_shape
                    && unique(axes, self.source_shape.len()) => {}
            IndexTransformKind::Broadcast(axes)
                if axes.len() == self.source_shape.len()
                    && unique(axes, self.result_shape.len())
                    && axes.iter().enumerate().all(|(input, &output)| {
                        self.source_shape[input] == 1
                            || self.source_shape[input] == self.result_shape[output]
                    }) => {}
            IndexTransformKind::Slice { starts, strides }
                if starts.len() == self.source_shape.len()
                    && strides.len() == starts.len()
                    && self.result_shape.len() == starts.len() =>
            {
                for axis in 0..starts.len() {
                    let extent = usize::try_from(self.source_shape[axis])
                        .map_err(|_| PlanError::InvalidShape)?;
                    let count = usize::try_from(self.result_shape[axis])
                        .map_err(|_| PlanError::InvalidShape)?;
                    if strides[axis] == 0 || starts[axis] > extent {
                        return Err(invalid());
                    }
                    if count > 0
                        && starts[axis]
                            .checked_add(
                                (count - 1)
                                    .checked_mul(strides[axis])
                                    .ok_or(PlanError::SizeOverflow)?,
                            )
                            .ok_or(PlanError::SizeOverflow)?
                            >= extent
                    {
                        return Err(invalid());
                    }
                }
            }
            _ => return Err(invalid()),
        }
        Ok(())
    }
}
