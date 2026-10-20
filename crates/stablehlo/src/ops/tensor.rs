use crate::verification::invalid;
use crate::{attributes::*, types::*};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{IntegerAttr, VecAttr},
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};

#[pliron_op(name="stablehlo.slice",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_slice_starts:VecAttr,stablehlo_slice_limits:VecAttr,stablehlo_slice_strides:VecAttr))]
pub struct SliceOp;
impl SliceOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        result: TypeHandle,
        starts: Vec<u64>,
        limits: Vec<u64>,
        strides: Vec<u64>,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input],
                vec![],
                0,
            ),
        };
        let a = integer_list(ctx, starts);
        let b = integer_list(ctx, limits);
        let c = integer_list(ctx, strides);
        op.set_attr_stablehlo_slice_starts(ctx, a);
        op.set_attr_stablehlo_slice_limits(ctx, b);
        op.set_attr_stablehlo_slice_strides(ctx, c);
        op
    }
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        starts: Vec<u64>,
        limits: Vec<u64>,
        strides: Vec<u64>,
    ) -> Result<Self> {
        let (shape, element) = crate::ops::builder_option(
            tensor_shape_elem(ctx, input.get_type(ctx)),
            "stablehlo.slice",
            "operand must be a supported ranked tensor",
        )?;
        if starts.len() != shape.len()
            || limits.len() != shape.len()
            || strides.len() != shape.len()
        {
            return crate::ops::builder_option(
                None,
                "stablehlo.slice",
                "builder inputs are invalid",
            );
        }
        let mut output = Vec::with_capacity(shape.len());
        for axis in 0..shape.len() {
            let (start, limit, stride) = (starts[axis], limits[axis], strides[axis]);
            if stride == 0
                || limit < start
                || shape[axis] >= 0 && limit > shape[axis] as u64
                || known_tensor_bound(ctx, input.get_type(ctx), axis)
                    .is_some_and(|bound| limit > bound)
            {
                return crate::ops::builder_option(
                    None,
                    "stablehlo.slice",
                    "builder inputs are invalid",
                );
            }
            let distance = limit - start;
            output.push((distance / stride + u64::from(distance % stride != 0)) as i64);
        }
        let result = TensorType::get(ctx, output, element).into();
        Ok(Self::new(ctx, input, result, starts, limits, strides))
    }
    /// Return the inclusive start index of each dimension.
    pub fn starts(&self, ctx: &Context) -> Option<Vec<u64>> {
        self.get_attr_stablehlo_slice_starts(ctx)
            .as_deref()
            .and_then(read_integer_list)
    }
    /// Return the exclusive limit index of each dimension.
    pub fn limits(&self, ctx: &Context) -> Option<Vec<u64>> {
        self.get_attr_stablehlo_slice_limits(ctx)
            .as_deref()
            .and_then(read_integer_list)
    }
    /// Return the stride of each dimension.
    pub fn strides(&self, ctx: &Context) -> Option<Vec<u64>> {
        self.get_attr_stablehlo_slice_strides(ctx)
            .as_deref()
            .and_then(read_integer_list)
    }
}
impl Verify for SliceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "slice arity mismatch");
        }
        let a = self
            .get_attr_stablehlo_slice_starts(ctx)
            .as_deref()
            .and_then(read_integer_list);
        let b = self
            .get_attr_stablehlo_slice_limits(ctx)
            .as_deref()
            .and_then(read_integer_list);
        let strides = self
            .get_attr_stablehlo_slice_strides(ctx)
            .as_deref()
            .and_then(read_integer_list);
        let (Some((src, e)), Some((dst, de)), Some(a), Some(b), Some(strides)) = (
            tensor_shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            tensor_shape_elem(ctx, ir.get_result(0).get_type(ctx)),
            a,
            b,
            strides,
        ) else {
            return invalid(
                self,
                ctx,
                "slice requires ranked types and typed dimension attributes",
            );
        };
        if e != de
            || a.len() != src.len()
            || b.len() != src.len()
            || strides.len() != src.len()
            || a.iter()
                .zip(&b)
                .zip(&strides)
                .zip(&src)
                .zip(&dst)
                .enumerate()
                .any(|(axis, ((((&lo, &hi), &step), &size), &out))| {
                    step == 0
                        || hi < lo
                        || size >= 0 && hi > size as u64
                        || known_tensor_bound(ctx, ir.get_operand(0).get_type(ctx), axis)
                            .is_some_and(|bound| hi > bound)
                        || out as u64 != (hi - lo) / step + u64::from((hi - lo) % step != 0)
                })
        {
            return invalid(
                self,
                ctx,
                "slice bounds, strides, or inferred shape are invalid",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.concatenate",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_concatenate_dimension:IntegerAttr))]
pub struct ConcatenateOp;
impl ConcatenateOp {
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        result: TypeHandle,
        dimension: usize,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                inputs,
                vec![],
                0,
            ),
        };
        let d = integer_attr(ctx, dimension as u64);
        op.set_attr_stablehlo_concatenate_dimension(ctx, d);
        op
    }
}
impl Verify for ConcatenateOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() < 1 || ir.get_num_results() != 1 {
            return invalid(
                self,
                ctx,
                "concatenate requires input tensors and one result",
            );
        }
        let Some(dimension) = self
            .get_attr_stablehlo_concatenate_dimension(ctx)
            .and_then(|a| a.value().to_string_decimal(false).parse::<usize>().ok())
        else {
            return invalid(self, ctx, "concatenate dimension is missing");
        };
        let inputs = ir
            .operands()
            .map(|value| value.get_type(ctx))
            .collect::<Vec<_>>();
        let Some(expected) = crate::inference::infer_concatenate(ctx, &inputs, dimension).ok()
        else {
            return invalid(
                self,
                ctx,
                "concatenate operand shapes or element types are incompatible",
            );
        };
        if !compatible_shape(ctx, expected, ir.get_result(0).get_type(ctx)) {
            return invalid(
                self,
                ctx,
                "concatenate result type differs from inferred shape",
            );
        }
        Ok(())
    }
}

impl ConcatenateOp {
    /// Return the concatenation dimension.
    pub fn dimension(&self, ctx: &Context) -> Option<usize> {
        self.get_attr_stablehlo_concatenate_dimension(ctx)
            .and_then(|a| a.value().to_string_decimal(false).parse::<usize>().ok())
    }
    pub fn new_inferred(ctx: &mut Context, inputs: Vec<Value>, dimension: usize) -> Result<Self> {
        let tys = inputs.iter().map(|v| v.get_type(ctx)).collect::<Vec<_>>();
        let out = crate::ops::builder_result(
            crate::inference::infer_concatenate(ctx, &tys, dimension),
            "stablehlo.concatenate",
        )?;
        Ok(Self::new(ctx, inputs, out, dimension))
    }
}

#[pliron_op(name="stablehlo.dynamic_slice",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_dynamic_slice_sizes:VecAttr))]
pub struct DynamicSliceOp;
impl DynamicSliceOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        starts: Vec<Value>,
        result: TypeHandle,
        sizes: Vec<i64>,
    ) -> Self {
        let mut ops = vec![input];
        ops.extend(starts);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                ops,
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_dynamic_slice_sizes(ctx, vec_i64(ctx, sizes));
        op
    }
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        starts: Vec<Value>,
        sizes: Vec<i64>,
    ) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_dynamic_slice(ctx, input.get_type(ctx), &sizes),
            "stablehlo.dynamic_slice",
        )?;
        Ok(Self::new(ctx, input, starts, result, sizes))
    }
    pub fn sizes(&self, ctx: &Context) -> Option<Vec<i64>> {
        self.get_attr_stablehlo_dynamic_slice_sizes(ctx)
            .as_deref()
            .and_then(read_i64)
    }
}
impl Verify for DynamicSliceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_results() != 1 || ir.get_num_operands() < 1 {
            return invalid(self, ctx, "dynamic_slice arity mismatch");
        }
        let (Some((shape, elem)), Some((out, oe)), Some(sizes)) = (
            tensor_shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            tensor_shape_elem(ctx, ir.get_result(0).get_type(ctx)),
            self.sizes(ctx),
        ) else {
            return invalid(self, ctx, "dynamic_slice types or sizes are invalid");
        };
        if ir.get_num_operands() != shape.len() + 1
            || sizes.len() != shape.len()
            || out != sizes
            || elem != oe
            || sizes.iter().enumerate().any(|(i, &size)| {
                size < 0
                    || known_tensor_bound(ctx, ir.get_operand(0).get_type(ctx), i)
                        .is_some_and(|bound| size as u64 > bound)
            })
            || ir
                .operands()
                .skip(1)
                .any(|v| !scalar_integer(ctx, v.get_type(ctx)))
            || ir
                .operands()
                .skip(2)
                .any(|v| v.get_type(ctx) != ir.get_operand(1).get_type(ctx))
        {
            return invalid(
                self,
                ctx,
                "dynamic_slice sizes and index operands are invalid",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.dynamic_update_slice",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct DynamicUpdateSliceOp;
impl DynamicUpdateSliceOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        update: Value,
        starts: Vec<Value>,
        result: TypeHandle,
    ) -> Self {
        let mut ops = vec![input, update];
        ops.extend(starts);
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                ops,
                vec![],
                0,
            ),
        }
    }
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        update: Value,
        starts: Vec<Value>,
    ) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_dynamic_update_slice(
                ctx,
                input.get_type(ctx),
                update.get_type(ctx),
            ),
            "stablehlo.dynamic_update_slice",
        )?;
        Ok(Self::new(ctx, input, update, starts, result))
    }
}
impl Verify for DynamicUpdateSliceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_results() != 1 || ir.get_num_operands() < 2 {
            return invalid(self, ctx, "dynamic_update_slice arity mismatch");
        }
        let (Some((shape, elem)), Some((upd, ue)), Some((out, oe))) = (
            tensor_shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            tensor_shape_elem(ctx, ir.get_operand(1).get_type(ctx)),
            tensor_shape_elem(ctx, ir.get_result(0).get_type(ctx)),
        ) else {
            return invalid(self, ctx, "dynamic_update_slice types are invalid");
        };
        if ir.get_num_operands() != shape.len() + 2
            || upd.len() != shape.len()
            || shape != out
            || elem != ue
            || elem != oe
            || upd.iter().enumerate().any(|(i, &size)| {
                (size >= 0
                    && known_tensor_bound(ctx, ir.get_operand(0).get_type(ctx), i)
                        .is_some_and(|bound| size as u64 > bound))
                    || known_tensor_bound(ctx, ir.get_operand(1).get_type(ctx), i)
                        .zip(known_tensor_bound(ctx, ir.get_operand(0).get_type(ctx), i))
                        .is_some_and(|(update_bound, input_bound)| update_bound > input_bound)
            })
            || ir
                .operands()
                .skip(2)
                .any(|v| !scalar_integer(ctx, v.get_type(ctx)))
            || ir
                .operands()
                .skip(3)
                .any(|v| v.get_type(ctx) != ir.get_operand(2).get_type(ctx))
        {
            return invalid(
                self,
                ctx,
                "dynamic_update_slice shape or indices are invalid",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.real_dynamic_slice",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RealDynamicSliceOp;
impl RealDynamicSliceOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        start: Value,
        limit: Value,
        stride: Value,
        result: TypeHandle,
    ) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input, start, limit, stride],
                vec![],
                0,
            ),
        }
    }
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        start: Value,
        limit: Value,
        stride: Value,
    ) -> Result<Self> {
        let out = crate::ops::builder_result(
            crate::inference::infer_real_dynamic_slice_type(
                ctx,
                input.get_type(ctx),
                start.get_type(ctx),
                limit.get_type(ctx),
                stride.get_type(ctx),
            ),
            "stablehlo.real_dynamic_slice",
        )?;
        Ok(Self::new(ctx, input, start, limit, stride, out))
    }
}
impl Verify for RealDynamicSliceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 4 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "real_dynamic_slice arity mismatch");
        }
        let (Some((shape, elem)), Some((result, re))) = (
            tensor_shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            tensor_shape_elem(ctx, ir.get_result(0).get_type(ctx)),
        ) else {
            return invalid(self, ctx, "real_dynamic_slice tensors must be ranked");
        };
        let rank = shape.len();
        let Some((_, index_element)) = tensor_shape_elem(ctx, ir.get_operand(1).get_type(ctx))
        else {
            return invalid(self, ctx, "real_dynamic_slice index tensors must be ranked");
        };
        if result.len() != rank
            || elem != re
            || index_element
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none()
            || ir.operands().skip(1).any(|v| {
                tensor_shape_elem(ctx, v.get_type(ctx))
                    .is_none_or(|(s, e)| s != vec![rank as i64] || e != index_element)
            })
        {
            return invalid(
                self,
                ctx,
                "real_dynamic_slice index vectors must match operand rank",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.pad",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_pad_low:VecAttr,stablehlo_pad_high:VecAttr,stablehlo_pad_interior:VecAttr))]
pub struct PadOp;
impl PadOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        padding: Value,
        result: TypeHandle,
        low: Vec<i64>,
        high: Vec<i64>,
        interior: Vec<i64>,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input, padding],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_pad_low(ctx, vec_i64(ctx, low));
        op.set_attr_stablehlo_pad_high(ctx, vec_i64(ctx, high));
        op.set_attr_stablehlo_pad_interior(ctx, vec_i64(ctx, interior));
        op
    }
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        padding: Value,
        low: Vec<i64>,
        high: Vec<i64>,
        interior: Vec<i64>,
    ) -> Result<Self> {
        let out = crate::ops::builder_result(
            crate::inference::infer_pad(
                ctx,
                input.get_type(ctx),
                padding.get_type(ctx),
                &low,
                &high,
                &interior,
            ),
            "stablehlo.pad",
        )?;
        Ok(Self::new(ctx, input, padding, out, low, high, interior))
    }
}
impl Verify for PadOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "pad arity mismatch");
        }
        let (
            Some((input, elem)),
            Some((pad, pe)),
            Some((out, oe)),
            Some(low),
            Some(high),
            Some(interior),
        ) = (
            tensor_shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            tensor_shape_elem(ctx, ir.get_operand(1).get_type(ctx)),
            tensor_shape_elem(ctx, ir.get_result(0).get_type(ctx)),
            self.get_attr_stablehlo_pad_low(ctx)
                .as_deref()
                .and_then(read_i64),
            self.get_attr_stablehlo_pad_high(ctx)
                .as_deref()
                .and_then(read_i64),
            self.get_attr_stablehlo_pad_interior(ctx)
                .as_deref()
                .and_then(read_i64),
        )
        else {
            return invalid(self, ctx, "pad tensors or attributes are invalid");
        };
        let expected = crate::inference::infer_pad(
            ctx,
            ir.get_operand(0).get_type(ctx),
            ir.get_operand(1).get_type(ctx),
            &low,
            &high,
            &interior,
        )
        .ok();
        if !pad.is_empty()
            || elem != pe
            || elem != oe
            || expected.is_none_or(|x| !compatible_shape(ctx, x, ir.get_result(0).get_type(ctx)))
            || out.len() != input.len()
        {
            return invalid(
                self,
                ctx,
                "pad result shape or padding attributes are invalid",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.dynamic_pad",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct DynamicPadOp;
impl DynamicPadOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        padding: Value,
        low: Value,
        high: Value,
        interior: Value,
        result: TypeHandle,
    ) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input, padding, low, high, interior],
                vec![],
                0,
            ),
        }
    }
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        padding: Value,
        low: Value,
        high: Value,
        interior: Value,
    ) -> Result<Self> {
        let out = crate::ops::builder_result(
            crate::inference::infer_dynamic_pad_type(
                ctx,
                input.get_type(ctx),
                padding.get_type(ctx),
            ),
            "stablehlo.dynamic_pad",
        )?;
        Ok(Self::new(ctx, input, padding, low, high, interior, out))
    }
}
impl Verify for DynamicPadOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 5 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "dynamic_pad arity mismatch");
        }
        let (Some((a, e)), Some((p, pe)), Some((b, be))) = (
            tensor_shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            tensor_shape_elem(ctx, ir.get_operand(1).get_type(ctx)),
            tensor_shape_elem(ctx, ir.get_result(0).get_type(ctx)),
        ) else {
            return invalid(self, ctx, "dynamic_pad tensors must be ranked");
        };
        let vector_type = ir.get_operand(2).get_type(ctx);
        if !p.is_empty()
            || e != pe
            || e != be
            || a.len() != b.len()
            || ir.operands().skip(2).any(|v| {
                tensor_shape_elem(ctx, v.get_type(ctx)).is_none_or(|(shape, element)| {
                    shape != vec![a.len() as i64]
                        || element.deref(ctx).downcast_ref::<IntegerType>().is_none()
                        || v.get_type(ctx) != vector_type
                })
            })
        {
            return invalid(
                self,
                ctx,
                "dynamic_pad padding vectors must match input rank",
            );
        }
        Ok(())
    }
}

fn vec_i64(ctx: &Context, values: impl IntoIterator<Item = i64>) -> VecAttr {
    VecAttr::new(
        values
            .into_iter()
            .map(|v| {
                Box::new(IntegerAttr::new(
                    IntegerType::get(ctx, 64, Signedness::Signed),
                    APInt::from_i64(v, bw(64)),
                )) as AttrObj
            })
            .collect(),
    )
}
fn read_i64(attr: &VecAttr) -> Option<Vec<i64>> {
    attr.0
        .iter()
        .map(|x| {
            x.downcast_ref::<IntegerAttr>()?
                .value()
                .to_string_decimal(true)
                .parse()
                .ok()
        })
        .collect()
}
fn tensor_shape_elem(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let ty = ty.deref(ctx);
    let t = ty.downcast_ref::<TensorType>()?;
    Some((t.dimensions.clone(), t.element))
}
fn scalar_integer(ctx: &Context, ty: TypeHandle) -> bool {
    tensor_shape_elem(ctx, ty)
        .is_some_and(|(s, e)| s.is_empty() && e.deref(ctx).downcast_ref::<IntegerType>().is_some())
}
fn compatible_shape(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    let (Some((as_, ae)), Some((bs, be))) = (tensor_shape_elem(ctx, a), tensor_shape_elem(ctx, b))
    else {
        return false;
    };
    ae == be
        && as_.len() == bs.len()
        && as_
            .iter()
            .zip(bs)
            .all(|(&x, y)| x == y || x == -1 || y == -1)
}

fn known_tensor_bound(ctx: &Context, ty: TypeHandle, axis: usize) -> Option<u64> {
    crate::verification::shape::dimension_bound(ctx, ty, axis)
}
