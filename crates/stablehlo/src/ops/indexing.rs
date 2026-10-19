use crate::verification::invalid;
use crate::{attributes::*, types::*};
use pliron::region::Region;
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{IntegerAttr, VecAttr},
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};

#[pliron_op(name="stablehlo.gather",format="$0 `,` $1 attr($stablehlo_gather_dimension_numbers, $GatherDimensionNumbersAttr) attr($stablehlo_gather_slice_sizes, $VecAttr)",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_gather_dimension_numbers:GatherDimensionNumbersAttr,stablehlo_gather_slice_sizes:VecAttr))]
pub struct GatherOp;
impl GatherOp {
    pub fn new(
        ctx: &mut Context,
        operand: Value,
        indices: Value,
        result: TypeHandle,
        dimensions: GatherDimensionNumbersAttr,
        slice_sizes: Vec<i64>,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![operand, indices],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_gather_dimension_numbers(ctx, dimensions);
        op.set_attr_stablehlo_gather_slice_sizes(ctx, vec_i64(ctx, slice_sizes));
        op
    }
    pub fn new_inferred(
        ctx: &mut Context,
        operand: Value,
        indices: Value,
        dimensions: GatherDimensionNumbersAttr,
        slice_sizes: Vec<i64>,
    ) -> Option<Self> {
        let shape = infer_gather_shape(
            ctx,
            operand.get_type(ctx),
            indices.get_type(ctx),
            &dimensions,
            &slice_sizes,
        )?;
        Some(Self::new(
            ctx,
            operand,
            indices,
            shape,
            dimensions,
            slice_sizes,
        ))
    }
    pub fn dimensions(&self, ctx: &Context) -> Option<GatherDimensionNumbersAttr> {
        self.get_attr_stablehlo_gather_dimension_numbers(ctx)
            .map(|x| (*x).clone())
    }
    pub fn slice_sizes(&self, ctx: &Context) -> Option<Vec<i64>> {
        self.get_attr_stablehlo_gather_slice_sizes(ctx)
            .and_then(|x| read_i64(&x))
    }
}
impl Verify for GatherOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "gather arity mismatch");
        }
        let (Some(d), Some(sizes), Some((_, se)), Some((_, _)), Some((_, oe))) = (
            self.dimensions(ctx),
            self.slice_sizes(ctx),
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_operand(1).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
        ) else {
            return invalid(
                self,
                ctx,
                "gather tensors and typed dimension numbers are required",
            );
        };
        if se != oe
            || !integer_tensor(ctx, ir.get_operand(1).get_type(ctx))
            || infer_gather_shape(
                ctx,
                ir.get_operand(0).get_type(ctx),
                ir.get_operand(1).get_type(ctx),
                &d,
                &sizes,
            )
            .is_none_or(|ty| !compatible_shape(ctx, ty, ir.get_result(0).get_type(ctx)))
        {
            return invalid(
                self,
                ctx,
                "gather dimension numbers, slice sizes or result shape are invalid",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.dynamic_gather",format="$0 `,` $1 `,` $2 attr($stablehlo_dynamic_gather_dimension_numbers, $GatherDimensionNumbersAttr)",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_dynamic_gather_dimension_numbers:GatherDimensionNumbersAttr))]
pub struct DynamicGatherOp;
impl DynamicGatherOp {
    pub fn new(
        ctx: &mut Context,
        operand: Value,
        indices: Value,
        slice_sizes: Value,
        result: TypeHandle,
        dimensions: GatherDimensionNumbersAttr,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![operand, indices, slice_sizes],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_dynamic_gather_dimension_numbers(ctx, dimensions);
        op
    }
    pub fn dimensions(&self, ctx: &Context) -> Option<GatherDimensionNumbersAttr> {
        self.get_attr_stablehlo_dynamic_gather_dimension_numbers(ctx)
            .map(|x| (*x).clone())
    }
    pub fn new_inferred(
        ctx: &mut Context,
        operand: Value,
        indices: Value,
        slice_sizes: Value,
        dimensions: GatherDimensionNumbersAttr,
    ) -> Option<Self> {
        let out = crate::inference::infer_dynamic_gather_type(
            ctx,
            operand.get_type(ctx),
            indices.get_type(ctx),
            slice_sizes.get_type(ctx),
            &dimensions,
        )
        .ok()?;
        Some(Self::new(
            ctx,
            operand,
            indices,
            slice_sizes,
            out,
            dimensions,
        ))
    }
}
impl Verify for DynamicGatherOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 3 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "dynamic_gather arity mismatch");
        }
        let Some(d) = self.dimensions(ctx) else {
            return invalid(self, ctx, "dynamic_gather dimension numbers are required");
        };
        let inferred = crate::inference::infer_dynamic_gather_type(
            ctx,
            ir.get_operand(0).get_type(ctx),
            ir.get_operand(1).get_type(ctx),
            ir.get_operand(2).get_type(ctx),
            &d,
        );
        let geometry = shape_elem(ctx, ir.get_operand(0).get_type(ctx))
            .zip(shape_elem(ctx, ir.get_result(0).get_type(ctx)))
            .is_some_and(|((input_shape, _), (result_shape, _))| {
                let collapsed = d
                    .collapsed_slice_dims
                    .iter()
                    .filter_map(|&axis| usize::try_from(axis).ok())
                    .collect::<Vec<_>>();
                let batching = d
                    .operand_batching_dims
                    .iter()
                    .filter_map(|&axis| usize::try_from(axis).ok())
                    .collect::<Vec<_>>();
                let slice_axes = (0..input_shape.len())
                    .filter(|axis| !collapsed.contains(axis) && !batching.contains(axis))
                    .collect::<Vec<_>>();
                d.offset_dims
                    .iter()
                    .zip(slice_axes)
                    .all(|(&result_axis, input_axis)| {
                        usize::try_from(result_axis)
                            .ok()
                            .and_then(|axis| result_shape.get(axis))
                            .is_some_and(|&size| {
                                size < 0
                                    || crate::verification::shape::dimension_bound(
                                        ctx,
                                        ir.get_operand(0).get_type(ctx),
                                        input_axis,
                                    )
                                    .is_none_or(|bound| size as u64 <= bound)
                            })
                    })
            });
        if !integer_tensor(ctx, ir.get_operand(1).get_type(ctx))
            || inferred.is_err()
            || inferred.is_ok_and(|ty| !compatible_shape(ctx, ty, ir.get_result(0).get_type(ctx)))
            || !geometry
        {
            return invalid(
                self,
                ctx,
                "dynamic_gather slices, dimensions or result shape are invalid",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.scatter",format="operands(CharSpace(`,`)) regions(CharSpace(`,`)) attr($stablehlo_scatter_dimension_numbers, $ScatterDimensionNumbersAttr)",attributes=(stablehlo_scatter_dimension_numbers:ScatterDimensionNumbersAttr))]
pub struct ScatterOp;
impl ScatterOp {
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        indices: Value,
        updates: Vec<Value>,
        dimensions: ScatterDimensionNumbersAttr,
    ) -> Self {
        let results = inputs
            .iter()
            .map(|input| input.get_type(ctx))
            .collect::<Vec<_>>();
        let mut operands = inputs;
        operands.push(indices);
        operands.extend(updates);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                operands,
                vec![],
                1,
            ),
        };
        op.set_attr_stablehlo_scatter_dimension_numbers(ctx, dimensions);
        op
    }
    pub fn new_inferred(
        ctx: &mut Context,
        inputs: Vec<Value>,
        indices: Value,
        updates: Vec<Value>,
        dimensions: ScatterDimensionNumbersAttr,
    ) -> Option<Self> {
        let input_types = inputs
            .iter()
            .map(|value| value.get_type(ctx))
            .collect::<Vec<_>>();
        let update_types = updates
            .iter()
            .map(|value| value.get_type(ctx))
            .collect::<Vec<_>>();
        if !verify_scatter_geometry(
            ctx,
            &input_types,
            indices.get_type(ctx),
            &update_types,
            &dimensions,
        ) {
            return None;
        }
        Some(Self::new(ctx, inputs, indices, updates, dimensions))
    }
    pub fn dimensions(&self, ctx: &Context) -> Option<ScatterDimensionNumbersAttr> {
        self.get_attr_stablehlo_scatter_dimension_numbers(ctx)
            .map(|x| (*x).clone())
    }
    pub fn update_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}

impl Verify for ScatterOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        let Some(d) = self.dimensions(ctx) else {
            return invalid(self, ctx, "scatter dimension numbers are required");
        };
        if ir.get_num_operands() < 3
            || ir.get_num_results() == 0
            || ir.get_num_results() * 2 + 1 != ir.get_num_operands()
            || ir
                .operands()
                .any(|v| shape_elem(ctx, v.get_type(ctx)).is_none())
            || ir
                .results()
                .any(|v| shape_elem(ctx, v.get_type(ctx)).is_none())
        {
            return invalid(
                self,
                ctx,
                "scatter requires ranked inputs, updates, indices and results",
            );
        };
        let input_count = ir.get_num_results();
        let Some((_, _)) = shape_elem(ctx, ir.get_operand(input_count).get_type(ctx)) else {
            return invalid(self, ctx, "scatter indices must be ranked");
        };
        let inputs = (0..input_count)
            .map(|i| ir.get_operand(i).get_type(ctx))
            .collect::<Vec<_>>();
        let updates = (input_count + 1..ir.get_num_operands())
            .map(|i| ir.get_operand(i).get_type(ctx))
            .collect::<Vec<_>>();
        if !integer_tensor(ctx, ir.get_operand(input_count).get_type(ctx))
            || !verify_scatter_geometry(
                ctx,
                &inputs,
                ir.get_operand(input_count).get_type(ctx),
                &updates,
                &d,
            )
        {
            return invalid(
                self,
                ctx,
                "scatter dimension numbers and update shapes are incompatible",
            );
        }
        for (i, input_type) in inputs.iter().enumerate().take(input_count) {
            if !crate::verification::shape::compatible_shape(
                ctx,
                *input_type,
                ir.get_result(i).get_type(ctx),
            ) {
                return invalid(
                    self,
                    ctx,
                    "scatter result types must match input shapes and elements",
                );
            }
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.torch_index_select",format="$0 `,` $1 attr($stablehlo_torch_index_select_dimension, $IntegerAttr) attr($stablehlo_torch_index_select_batch_dims, $IntegerAttr)",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_torch_index_select_dimension:IntegerAttr,stablehlo_torch_index_select_batch_dims:IntegerAttr))]
pub struct TorchIndexSelectOp;
impl TorchIndexSelectOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        indices: Value,
        result: TypeHandle,
        dimension: usize,
    ) -> Self {
        Self::new_with_batch_dims(ctx, input, indices, result, dimension, 0)
    }
    pub fn new_with_batch_dims(
        ctx: &mut Context,
        input: Value,
        indices: Value,
        result: TypeHandle,
        dimension: usize,
        batch_dims: usize,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input, indices],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_torch_index_select_dimension(
            ctx,
            integer_attr(ctx, dimension as u64),
        );
        op.set_attr_stablehlo_torch_index_select_batch_dims(
            ctx,
            integer_attr(ctx, batch_dims as u64),
        );
        op
    }
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        indices: Value,
        dimension: usize,
    ) -> Option<Self> {
        Self::new_inferred_with_batch_dims(ctx, input, indices, dimension, 0)
    }
    pub fn new_inferred_with_batch_dims(
        ctx: &mut Context,
        input: Value,
        indices: Value,
        dimension: usize,
        batch_dims: usize,
    ) -> Option<Self> {
        let (input_shape, element) = shape_elem(ctx, input.get_type(ctx))?;
        let (index_shape, _) = shape_elem(ctx, indices.get_type(ctx))?;
        if batch_dims > dimension
            || dimension >= input_shape.len()
            || batch_dims > index_shape.len()
            || input_shape[..batch_dims] != index_shape[..batch_dims]
        {
            return None;
        }
        let mut output = input_shape[..dimension].to_vec();
        output.extend_from_slice(&index_shape[batch_dims..]);
        output.extend_from_slice(&input_shape[dimension + 1..]);
        Some(Self::new_with_batch_dims(
            ctx,
            input,
            indices,
            TensorType::get(ctx, output, element).into(),
            dimension,
            batch_dims,
        ))
    }
    pub fn dimension(&self, ctx: &Context) -> Option<usize> {
        usize::try_from(
            self.get_attr_stablehlo_torch_index_select_dimension(ctx)?
                .value()
                .to_string_decimal(false)
                .parse::<u64>()
                .ok()?,
        )
        .ok()
    }
    pub fn batch_dims(&self, ctx: &Context) -> Option<usize> {
        usize::try_from(
            self.get_attr_stablehlo_torch_index_select_batch_dims(ctx)?
                .value()
                .to_string_decimal(false)
                .parse::<u64>()
                .ok()?,
        )
        .ok()
    }
}

impl Verify for TorchIndexSelectOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "torch_index_select arity mismatch");
        }
        let (
            Some((input_shape, input_element)),
            Some((index_shape, _)),
            Some((result_shape, result_element)),
            Some(dimension),
            Some(batch_dims),
        ) = (
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_operand(1).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
            self.dimension(ctx),
            self.batch_dims(ctx),
        )
        else {
            return invalid(
                self,
                ctx,
                "torch_index_select tensors or dimensions are invalid",
            );
        };
        if !integer_tensor(ctx, ir.get_operand(1).get_type(ctx))
            || batch_dims > dimension
            || dimension >= input_shape.len()
            || batch_dims > index_shape.len()
            || input_shape[..batch_dims] != index_shape[..batch_dims]
            || input_element != result_element
        {
            return invalid(
                self,
                ctx,
                "torch_index_select dimension or batch dimensions are invalid",
            );
        }
        let mut expected = input_shape[..dimension].to_vec();
        expected.extend_from_slice(&index_shape[batch_dims..]);
        expected.extend_from_slice(&input_shape[dimension + 1..]);
        if expected != result_shape {
            return invalid(self, ctx, "torch_index_select result shape is invalid");
        }
        Ok(())
    }
}

fn infer_gather_shape(
    ctx: &Context,
    operand: TypeHandle,
    indices: TypeHandle,
    d: &GatherDimensionNumbersAttr,
    sizes: &[i64],
) -> Option<TypeHandle> {
    let (input, elem) = shape_elem(ctx, operand)?;
    let (index, _) = shape_elem(ctx, indices)?;
    if sizes.len() != input.len()
        || input.len()
            != d.offset_dims.len() + d.collapsed_slice_dims.len() + d.operand_batching_dims.len()
    {
        return None;
    }
    let vector = usize::try_from(d.index_vector_dim).ok()?;
    if vector > index.len() {
        return None;
    }
    let vec_size = index.get(vector).copied().unwrap_or(1);
    if vec_size >= 0 && vec_size != d.start_index_map.len() as i64 {
        return None;
    }
    let mut collapsed = vec![false; input.len()];
    let mut batch = vec![false; input.len()];
    let mut mapped = vec![false; input.len()];
    let mut index_batch = vec![false; index.len()];
    for &x in &d.collapsed_slice_dims {
        let x = usize::try_from(x).ok()?;
        if x >= input.len() || std::mem::replace(&mut collapsed[x], true) {
            return None;
        }
    }
    for &x in &d.operand_batching_dims {
        let x = usize::try_from(x).ok()?;
        if x >= input.len() || std::mem::replace(&mut batch[x], true) {
            return None;
        }
    }
    for &x in &d.start_index_map {
        let x = usize::try_from(x).ok()?;
        if x >= input.len() || std::mem::replace(&mut mapped[x], true) {
            return None;
        }
    }
    for &x in &d.start_indices_batching_dims {
        let x = usize::try_from(x).ok()?;
        if x >= index.len() || x == vector || std::mem::replace(&mut index_batch[x], true) {
            return None;
        }
    }
    if d.operand_batching_dims.len() != d.start_indices_batching_dims.len()
        || d.operand_batching_dims
            .iter()
            .zip(&d.start_indices_batching_dims)
            .any(|(&a, &b)| {
                input[a as usize] >= 0
                    && index[b as usize] >= 0
                    && input[a as usize] != index[b as usize]
            })
    {
        return None;
    }
    for (i, &s) in sizes.iter().enumerate() {
        if s < 0 || input[i] >= 0 && s > input[i] || (s > 1 && (collapsed[i] || batch[i])) {
            return None;
        }
    }
    let batch_axes = (0..index.len())
        .filter(|&x| x != vector)
        .collect::<Vec<_>>();
    let mut offset = vec![false; batch_axes.len() + d.offset_dims.len()];
    for &x in &d.offset_dims {
        let x = usize::try_from(x).ok()?;
        if x >= offset.len() || std::mem::replace(&mut offset[x], true) {
            return None;
        }
    }
    let slice_axes = (0..input.len())
        .filter(|&x| !collapsed[x] && !batch[x])
        .collect::<Vec<_>>();
    if slice_axes.len() != d.offset_dims.len()
        || batch_axes.len() != offset.len() - d.offset_dims.len()
    {
        return None;
    }
    let mut out = Vec::with_capacity(offset.len());
    let (mut si, mut bi) = (0, 0);
    for is_offset in offset {
        if is_offset {
            out.push(sizes[slice_axes[si]]);
            si += 1
        } else {
            out.push(index[batch_axes[bi]]);
            bi += 1
        }
    }
    Some(TensorType::get(ctx, out, elem).into())
}
fn shape_elem(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let ty = ty.deref(ctx);
    let t = ty.downcast_ref::<TensorType>()?;
    Some((t.dimensions.clone(), t.element))
}
fn compatible_shape(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    let (Some((as_, ae)), Some((bs, be))) = (shape_elem(ctx, a), shape_elem(ctx, b)) else {
        return false;
    };
    ae == be
        && as_.len() == bs.len()
        && as_
            .iter()
            .zip(bs)
            .all(|(&a, b)| a == b || a == -1 || b == -1)
}
fn vec_i64(ctx: &Context, x: impl IntoIterator<Item = i64>) -> VecAttr {
    VecAttr::new(
        x.into_iter()
            .map(|v| {
                Box::new(IntegerAttr::new(
                    IntegerType::get(ctx, 64, Signedness::Signed),
                    APInt::from_i64(v, bw(64)),
                )) as AttrObj
            })
            .collect(),
    )
}
fn read_i64(x: &VecAttr) -> Option<Vec<i64>> {
    x.0.iter()
        .map(|v| {
            v.downcast_ref::<IntegerAttr>()?
                .value()
                .to_string_decimal(true)
                .parse()
                .ok()
        })
        .collect()
}
fn integer_attr(ctx: &Context, v: u64) -> IntegerAttr {
    IntegerAttr::new(
        IntegerType::get(ctx, 64, Signedness::Unsigned),
        APInt::from_u64(v, bw(64)),
    )
}

fn integer_tensor(ctx: &Context, ty: TypeHandle) -> bool {
    shape_elem(ctx, ty)
        .is_some_and(|(_, elem)| elem.deref(ctx).downcast_ref::<IntegerType>().is_some())
}

fn dimensions_compatible(
    ctx: &Context,
    left: TypeHandle,
    left_axis: usize,
    right: TypeHandle,
    right_axis: usize,
) -> bool {
    let (Some((left_shape, _)), Some((right_shape, _))) =
        (shape_elem(ctx, left), shape_elem(ctx, right))
    else {
        return false;
    };
    let (Some(&left_dim), Some(&right_dim)) =
        (left_shape.get(left_axis), right_shape.get(right_axis))
    else {
        return false;
    };
    if left_dim >= 0 && right_dim >= 0 {
        return left_dim == right_dim;
    }
    !(left_dim >= 0
        && crate::verification::shape::dimension_bound(ctx, right, right_axis)
            .is_some_and(|bound| left_dim as u64 > bound)
        || right_dim >= 0
            && crate::verification::shape::dimension_bound(ctx, left, left_axis)
                .is_some_and(|bound| right_dim as u64 > bound))
}

fn verify_scatter_geometry(
    ctx: &Context,
    inputs: &[TypeHandle],
    indices: TypeHandle,
    updates: &[TypeHandle],
    d: &ScatterDimensionNumbersAttr,
) -> bool {
    if inputs.is_empty() || inputs.len() != updates.len() {
        return false;
    }
    let Some((index_shape, _)) = shape_elem(ctx, indices) else {
        return false;
    };
    let Some((first_input, _)) = shape_elem(ctx, inputs[0]) else {
        return false;
    };
    let Some((first_update, _)) = shape_elem(ctx, updates[0]) else {
        return false;
    };
    let axes = |values: &[i64], rank: usize, sorted: bool| -> Option<Vec<usize>> {
        let mut seen = vec![false; rank];
        let mut result = Vec::with_capacity(values.len());
        for &v in values {
            let axis = usize::try_from(v).ok()?;
            if axis >= rank || std::mem::replace(&mut seen[axis], true) {
                return None;
            }
            result.push(axis)
        }
        if sorted && !result.windows(2).all(|w| w[0] < w[1]) {
            return None;
        }
        Some(result)
    };
    let Some(vector) = usize::try_from(d.index_vector_dim)
        .ok()
        .filter(|&x| x <= index_shape.len())
    else {
        return false;
    };
    let (Some(windows), Some(inserted), Some(input_batch), Some(index_batch), Some(index_map)) = (
        axes(&d.update_window_dims, first_update.len(), true),
        axes(&d.inserted_window_dims, first_input.len(), true),
        axes(&d.input_batching_dims, first_input.len(), true),
        axes(&d.scatter_indices_batching_dims, index_shape.len(), false),
        axes(&d.scatter_dims_to_operand_dims, first_input.len(), false),
    ) else {
        return false;
    };
    if windows.len() + inserted.len() + input_batch.len() != first_input.len()
        || input_batch.len() != index_batch.len()
        || input_batch
            .iter()
            .any(|x| inserted.binary_search(x).is_ok() || index_map.contains(x))
        || index_batch.contains(&vector)
    {
        return false;
    }
    let vector_size = index_shape.get(vector).copied().unwrap_or(1);
    if (vector_size >= 0 && vector_size != index_map.len() as i64)
        || crate::verification::shape::dimension_bound(ctx, indices, vector)
            .is_some_and(|bound| bound < index_map.len() as u64)
    {
        return false;
    }
    let scatter_axes = (0..index_shape.len())
        .filter(|&x| x != vector)
        .collect::<Vec<_>>();
    let update_axes = (0..first_update.len())
        .filter(|x| windows.binary_search(x).is_err())
        .collect::<Vec<_>>();
    if scatter_axes.len() != update_axes.len() {
        return false;
    }
    for (input, update) in inputs.iter().zip(updates) {
        let (Some((input_shape, input_elem)), Some((update_shape, update_elem))) =
            (shape_elem(ctx, *input), shape_elem(ctx, *update))
        else {
            return false;
        };
        if input_shape.len() != first_input.len()
            || update_shape.len() != first_update.len()
            || input_elem != update_elem
        {
            return false;
        }
    }
    for (&input_axis, &index_axis) in input_batch.iter().zip(&index_batch) {
        let position = index_axis - usize::from(index_axis > vector);
        let Some(&update_axis) = update_axes.get(position) else {
            return false;
        };
        if !dimensions_compatible(ctx, inputs[0], input_axis, indices, index_axis)
            || !dimensions_compatible(ctx, updates[0], update_axis, indices, index_axis)
            || !dimensions_compatible(ctx, inputs[0], input_axis, updates[0], update_axis)
        {
            return false;
        }
    }
    let input_windows = (0..first_input.len())
        .filter(|x| inserted.binary_search(x).is_err() && input_batch.binary_search(x).is_err())
        .collect::<Vec<_>>();
    for (&update_axis, &input_axis) in windows.iter().zip(&input_windows) {
        let Some((update_shape, _)) = shape_elem(ctx, updates[0]) else {
            return false;
        };
        if update_shape[update_axis] >= 0
            && crate::verification::shape::dimension_bound(ctx, inputs[0], input_axis)
                .is_some_and(|bound| update_shape[update_axis] as u64 > bound)
        {
            return false;
        }
    }
    true
}
