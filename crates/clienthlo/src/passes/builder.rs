use crate::Result;
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{BoolAttr, IntegerAttr},
    context::{Context, Ptr},
    irbuild::{dialect_conversion::DialectConversionRewriter, inserter::Inserter},
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use stablehlo::{
    attributes::{ComparisonDirection, ComplexAttr, DenseElementsAttr, ScalarBitsAttr},
    ops::*,
    types::{ElementType, TensorType, primitive_element, tensor},
};

pub(super) fn ranked(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let object = ty.deref(ctx);
    let tensor = object.downcast_ref::<TensorType>()?;
    Some((tensor.dimensions.clone(), tensor.element))
}

pub(super) fn scalar_words(ctx: &Context, value: &AttrObj) -> Option<Vec<u64>> {
    let value = crate::ops::constant_element_value(ctx, value);
    if let Some(bits) = value.downcast_ref::<ScalarBitsAttr>() {
        return Some(bits.words.clone());
    }
    if let Some(integer) = value.downcast_ref::<IntegerAttr>() {
        return Some(vec![integer.value().to_u64()]);
    }
    if let Some(boolean) = value.downcast_ref::<BoolAttr>() {
        return Some(vec![u64::from(bool::from(boolean.clone()))]);
    }
    if let Some(complex) = value.downcast_ref::<ComplexAttr>() {
        return match primitive_element(ctx, complex.ty)? {
            ElementType::C64 => Some(vec![
                complex.real.words[0] | (complex.imaginary.words[0] << 32),
            ]),
            ElementType::C128 => Some(vec![complex.real.words[0], complex.imaginary.words[0]]),
            _ => None,
        };
    }
    None
}

pub(super) struct Builder<'a> {
    pub(super) ctx: &'a mut Context,
    pub(super) rewriter: &'a mut DialectConversionRewriter,
    pub(super) source: Ptr<Operation>,
}

macro_rules! binary {
    ($($method:ident => $op:ident),* $(,)?) => {$(
        pub(super) fn $method(&mut self, lhs: Value, rhs: Value) -> Result<Value> {
            let ty = lhs.get_type(self.ctx);
            let op = $op::new(self.ctx, lhs, rhs, ty);
            self.emit(op)
        }
    )*};
}

macro_rules! unary {
    ($($method:ident => $op:ident),* $(,)?) => {$(
        pub(super) fn $method(&mut self, value: Value) -> Result<Value> {
            let ty = value.get_type(self.ctx);
            let op = $op::new(self.ctx, value, ty);
            self.emit(op)
        }
    )*};
}

impl Builder<'_> {
    pub(super) fn emit(&mut self, op: impl Op) -> Result<Value> {
        let operation = op.get_operation();
        self.insert(op)?;
        Ok(operation.deref(self.ctx).get_result(0))
    }

    pub(super) fn insert(&mut self, op: impl Op) -> Result<()> {
        op.verify(self.ctx)?;
        let operation = op.get_operation();
        let location = self.source.deref(self.ctx).loc().clone();
        operation.deref_mut(self.ctx).set_loc(location);
        self.rewriter.insert_operation(self.ctx, operation);
        Ok(())
    }

    pub(super) fn in_block<T>(
        &mut self,
        block: Ptr<pliron::basic_block::BasicBlock>,
        build: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let previous = self.rewriter.get_insertion_point();
        self.rewriter.set_insertion_point_to_block_end(block);
        let result = build(self);
        self.rewriter.set_insertion_point(previous);
        result
    }

    pub(super) fn zero(&mut self, ty: TypeHandle, shape: Option<Value>) -> Result<Value> {
        let (_, element) = ranked(self.ctx, ty).ok_or_else(|| {
            super::unsupported(self.ctx, self.source, "zero requires a ranked tensor")
        })?;
        let scalar = TensorType::get(self.ctx, vec![], element).into();
        let limbs = primitive_element(self.ctx, element)
            .and_then(ElementType::bit_width)
            .map_or(1, |bits| bits.div_ceil(64));
        let bits = if let Some(q) = element
            .deref(self.ctx)
            .downcast_ref::<stablehlo::types::QuantizedElementType>()
        {
            if q.quantization_dimension().is_some() {
                return Err(super::unsupported(
                    self.ctx,
                    self.source,
                    "a scalar initializer cannot represent per-axis quantization",
                ));
            }
            let point = q.zero_points().next().ok_or_else(|| {
                super::unsupported(
                    self.ctx,
                    self.source,
                    "quantized initializer has no zero point",
                )
            })?;
            let width = q
                .storage_type(self.ctx)
                .and_then(ElementType::bit_width)
                .ok_or_else(|| {
                    super::unsupported(
                        self.ctx,
                        self.source,
                        "quantized initializer has no storage width",
                    )
                })?;
            vec![(point as u64) & (u64::MAX >> (64 - width))]
        } else {
            vec![0; limbs as usize]
        };
        let zero = ConstantOp::new(self.ctx, scalar, bits, true);
        let zero = self.emit(zero)?;
        self.broadcast(zero, ty, vec![], shape)
    }

    /// Extract one position and remove that axis, including runtime-shaped tensors.
    pub(super) fn slice_axis(&mut self, input: Value, axis: usize, index: Value) -> Result<Value> {
        let (mut shape, element) = ranked(self.ctx, input.get_type(self.ctx)).ok_or_else(|| {
            super::unsupported(self.ctx, self.source, "slice requires a ranked tensor")
        })?;
        let zero = self.integer(0, ElementType::I64)?;
        let one = self.integer(1, ElementType::I64)?;
        let mut starts = vec![zero; shape.len()];
        starts[axis] = index;
        let mut limits = Vec::new();
        let mut result_sizes = Vec::new();
        for (dim, &extent) in shape.iter().enumerate() {
            if dim == axis {
                limits.push(self.add(index, one)?);
            } else {
                let size = self.dimension(input, dim, extent)?;
                limits.push(size);
                result_sizes.push(size);
            }
        }
        shape[axis] = 1;
        let ty = TensorType::get(self.ctx, shape.clone(), element).into();
        let start = self.shape_vector(starts)?;
        let limit = self.shape_vector(limits)?;
        let strides = self.shape_vector(vec![one; shape.len()])?;
        let slice = RealDynamicSliceOp::new(self.ctx, input, start, limit, strides, ty);
        let slice = self.emit(slice)?;
        shape.remove(axis);
        let ty = TensorType::get(self.ctx, shape.clone(), element).into();
        self.reshape(slice, ty, result_sizes)
    }

    pub(super) fn reshape(
        &mut self,
        value: Value,
        result: TypeHandle,
        sizes: Vec<Value>,
    ) -> Result<Value> {
        let (shape, _) = ranked(self.ctx, result).ok_or_else(|| {
            super::unsupported(self.ctx, self.source, "reshape requires ranked tensors")
        })?;
        if shape.iter().any(|&d| d < 0) {
            let sizes = self.shape_vector(sizes)?;
            let op = DynamicReshapeOp::new(self.ctx, value, sizes, result);
            self.emit(op)
        } else {
            let op = ReshapeOp::new(self.ctx, value, result);
            self.emit(op)
        }
    }

    binary!(
        add => AddOp, sub => SubtractOp, mul => MultiplyOp,
        atan2 => Atan2Op, and => AndOp, or => OrOp, shr => ShiftRightLogicalOp,
        sar => ShiftRightArithmeticOp,
        div => DivideOp, pow => PowerOp, rem => RemainderOp,
        max => MaximumOp, min => MinimumOp,
    );
    unary!(neg => NegateOp, abs => AbsOp, log1p => LogPlusOneOp,
        log => LogOp, exp => ExponentialOp, expm1 => ExponentialMinusOneOp,
        sqrt => SqrtOp, floor => FloorOp, sin => SineOp, cos => CosineOp,
        sign => SignOp, not => NotOp);

    pub(super) fn component(&mut self, value: Value, imaginary: bool) -> Result<Value> {
        let (shape, element) = ranked(self.ctx, value.get_type(self.ctx)).ok_or_else(|| {
            super::unsupported(
                self.ctx,
                self.source,
                "complex components require ranked tensors",
            )
        })?;
        let component = primitive_element(self.ctx, element)
            .and_then(ElementType::complex_component)
            .ok_or_else(|| {
                super::unsupported(self.ctx, self.source, "expected complex elements")
            })?;
        let ty = tensor(self.ctx, &shape, component);
        if imaginary {
            let op = ImagOp::new(self.ctx, value, ty);
            self.emit(op)
        } else {
            let op = RealOp::new(self.ctx, value, ty);
            self.emit(op)
        }
    }

    pub(super) fn complex(&mut self, real: Value, imaginary: Value) -> Result<Value> {
        let op = ComplexOp::new_inferred(self.ctx, real, imaginary)?;
        self.emit(op)
    }

    pub(super) fn convert(&mut self, value: Value, ty: TypeHandle) -> Result<Value> {
        if value.get_type(self.ctx) == ty {
            return Ok(value);
        }
        let op = ConvertOp::new(self.ctx, value, ty);
        self.emit(op)
    }

    pub(super) fn bitcast(&mut self, value: Value, ty: TypeHandle) -> Result<Value> {
        let op = BitcastConvertOp::new(self.ctx, value, ty);
        self.emit(op)
    }

    pub(super) fn compare(
        &mut self,
        lhs: Value,
        rhs: Value,
        direction: ComparisonDirection,
    ) -> Result<Value> {
        let op = CompareOp::new_inferred(self.ctx, lhs, rhs, direction, None)?;
        self.emit(op)
    }

    pub(super) fn select(&mut self, condition: Value, yes: Value, no: Value) -> Result<Value> {
        let op = SelectOp::new_inferred(self.ctx, condition, yes, no)?;
        self.emit(op)
    }

    pub(super) fn integer(&mut self, value: i64, element: ElementType) -> Result<Value> {
        let ty = tensor(self.ctx, &[], element);
        let width = element.bit_width().ok_or_else(|| {
            super::unsupported(
                self.ctx,
                self.source,
                "integer constant needs a primitive integer type",
            )
        })?;
        let mask = u64::MAX >> (64 - width);
        let op = ConstantOp::new(self.ctx, ty, vec![(value as u64) & mask], true);
        self.emit(op)
    }

    pub(super) fn float(&mut self, value: f64, like: Value) -> Result<Value> {
        let element = ranked(self.ctx, like.get_type(self.ctx))
            .map(|(_, element)| element)
            .ok_or_else(|| {
                super::unsupported(
                    self.ctx,
                    self.source,
                    "floating constants need ranked tensors",
                )
            })?;
        let ty = tensor(self.ctx, &[], ElementType::F64);
        let op = ConstantOp::new(self.ctx, ty, vec![value.to_bits()], true);
        let scalar = self.emit(op)?;
        let primitive = primitive_element(self.ctx, element);
        let component = primitive.and_then(ElementType::complex_component);
        let scalar_element = component.map_or(element, |component| component.native(self.ctx));
        let ty = TensorType::get(self.ctx, vec![], scalar_element).into();
        let scalar = self.convert(scalar, ty)?;
        let scalar = if component.is_some() {
            let zero = ConstantOp::new(self.ctx, ty, vec![0], true);
            let zero = self.emit(zero)?;
            self.complex(scalar, zero)?
        } else {
            scalar
        };
        self.broadcast_like(scalar, like)
    }

    pub(super) fn shape(&mut self, value: Value) -> Result<Value> {
        let (dimensions, _) = ranked(self.ctx, value.get_type(self.ctx)).ok_or_else(|| {
            super::unsupported(
                self.ctx,
                self.source,
                "runtime shape requires a ranked tensor",
            )
        })?;
        let mut sizes = Vec::new();
        for (axis, extent) in dimensions.into_iter().enumerate() {
            let size = self.dimension(value, axis, extent)?;
            sizes.push(size);
        }
        self.shape_vector(sizes)
    }

    pub(super) fn dimension(&mut self, value: Value, axis: usize, extent: i64) -> Result<Value> {
        if extent >= 0 {
            return self.integer(extent, ElementType::I64);
        }
        let ty = tensor(self.ctx, &[], ElementType::I32);
        let op = GetDimensionSizeOp::new(self.ctx, value, ty, axis);
        let size = self.emit(op)?;
        let ty = tensor(self.ctx, &[], ElementType::I64);
        self.convert(size, ty)
    }

    pub(super) fn shape_vector(&mut self, sizes: Vec<Value>) -> Result<Value> {
        if sizes.is_empty() {
            let ty = tensor(self.ctx, &[0], ElementType::I64);
            let op = ConstantOp::new(self.ctx, ty, vec![], false);
            return self.emit(op);
        }
        let mut vectors = Vec::new();
        for size in sizes {
            let ty = tensor(self.ctx, &[1], ElementType::I64);
            let op = ReshapeOp::new(self.ctx, size, ty);
            vectors.push(self.emit(op)?);
        }
        let ty = tensor(self.ctx, &[vectors.len() as i64], ElementType::I64);
        let op = ConcatenateOp::new(self.ctx, vectors, ty, 0);
        self.emit(op)
    }

    pub(super) fn broadcast(
        &mut self,
        value: Value,
        result: TypeHandle,
        axes: Vec<usize>,
        shape: Option<Value>,
    ) -> Result<Value> {
        if shape.is_none()
            && value.get_type(self.ctx) == result
            && axes.iter().copied().eq(0..axes.len())
        {
            return Ok(value);
        }
        if let Some(shape) = shape {
            let op =
                DynamicBroadcastInDimOp::new(self.ctx, value, shape, result, axes, vec![], vec![]);
            self.emit(op)
        } else {
            let op = BroadcastInDimOp::new(self.ctx, value, result, axes);
            self.emit(op)
        }
    }

    pub(super) fn broadcast_like(&mut self, scalar: Value, like: Value) -> Result<Value> {
        let (dimensions, _) = ranked(self.ctx, like.get_type(self.ctx)).ok_or_else(|| {
            super::unsupported(self.ctx, self.source, "broadcast requires ranked tensors")
        })?;
        let (_, element) = ranked(self.ctx, scalar.get_type(self.ctx)).ok_or_else(|| {
            super::unsupported(self.ctx, self.source, "broadcast requires a scalar tensor")
        })?;
        let ty = TensorType::get(self.ctx, dimensions.clone(), element).into();
        let shape = if dimensions.iter().any(|&extent| extent < 0) {
            Some(self.shape(like)?)
        } else {
            None
        };
        self.broadcast(scalar, ty, vec![], shape)
    }

    pub(super) fn elements(&mut self, result: TypeHandle, value: AttrObj) -> Result<Value> {
        if let Some(dense) = value.downcast_ref::<DenseElementsAttr>() {
            let bits = dense
                .values
                .0
                .iter()
                .map(|value| scalar_words(self.ctx, value))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    super::unsupported(
                        self.ctx,
                        self.source,
                        "constant scalar representation is not supported",
                    )
                })?;
            let op = ConstantOp::new(
                self.ctx,
                result,
                bits.into_iter().flatten().collect(),
                dense.is_splat,
            );
            self.emit(op)
        } else if let Some(sparse) =
            value.downcast_ref::<stablehlo::attributes::SparseElementsAttr>()
        {
            let op = ConstantOp::new_sparse(
                self.ctx,
                result,
                sparse
                    .indices
                    .iter()
                    .map(|index| index.values.clone())
                    .collect(),
                sparse.values.0.clone(),
            );
            self.emit(op)
        } else {
            Err(super::unsupported(
                self.ctx,
                self.source,
                "constant requires dense or sparse elements",
            ))
        }
    }
}
