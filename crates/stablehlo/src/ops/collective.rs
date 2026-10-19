use crate::verification::invalid;
use crate::{attributes::*, types::*};
use pliron::{
    builtin::attributes::{BoolAttr, IntegerAttr, VecAttr},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron::{
    builtin::types::IntegerType,
    builtin::types::Signedness,
    region::Region,
    utils::apint::{APInt, bw},
};

fn replica_ids(group: &VecAttr) -> Option<Vec<i64>> {
    group
        .0
        .iter()
        .map(|value| {
            let integer = value.downcast_ref::<IntegerAttr>()?;
            integer.value().to_string_decimal(true).parse::<i64>().ok()
        })
        .collect()
}

fn valid_replica_groups(groups: Option<&VecAttr>, allow_padding: bool, complete: bool) -> bool {
    use std::collections::HashSet;
    let Some(groups) = groups else { return false };
    let mut all = HashSet::new();
    let mut equal_size = None;
    for item in &groups.0 {
        let Some(group) = item.downcast_ref::<VecAttr>().and_then(replica_ids) else {
            return false;
        };
        if group.iter().any(|&id| id < -1) || (!allow_padding && group.iter().any(|&id| id < 0)) {
            return false;
        }
        if group
            .iter()
            .filter(|&&id| id >= 0)
            .any(|id| !all.insert(*id))
        {
            return false;
        }
        let size = group.iter().filter(|&&id| id >= 0).count();
        if !allow_padding {
            if let Some(expected) = equal_size {
                if expected != size {
                    return false;
                }
            } else {
                equal_size = Some(size)
            }
        }
    }
    if all.is_empty() {
        return true;
    }
    !complete || (0..all.len()).all(|id| all.contains(&(id as i64)))
}
fn global_group_attributes_valid(
    groups: Option<&VecAttr>,
    channel: Option<&ChannelHandleAttr>,
    global: Option<&BoolAttr>,
    allow_padding: bool,
    complete: bool,
) -> bool {
    let enabled = global.is_some_and(|v| bool::from(v.clone()));
    valid_replica_groups(groups, allow_padding, complete)
        && (!enabled
            || (groups.is_some_and(|g| !g.0.is_empty()) && channel.is_some_and(|c| c.handle > 0)))
}
fn replica_group_size(groups: Option<&VecAttr>) -> Option<i64> {
    let groups = groups?;
    let first = groups.0.first()?.downcast_ref::<VecAttr>()?;
    let replicas = replica_ids(first)?;
    i64::try_from(replicas.iter().filter(|&&id| id >= 0).count()).ok()
}
fn valid_source_target_pairs(pairs: Option<&VecAttr>) -> bool {
    use std::collections::HashSet;
    let Some(pairs) = pairs else { return false };
    if pairs.0.len() % 2 != 0 {
        return false;
    }
    let (mut sources, mut targets) = (HashSet::new(), HashSet::new());
    for pair in pairs.0.chunks_exact(2) {
        let (Some(source), Some(target)) = (
            pair[0].downcast_ref::<IntegerAttr>(),
            pair[1].downcast_ref::<IntegerAttr>(),
        ) else {
            return false;
        };
        let (Ok(source), Ok(target)) = (
            source.value().to_string_decimal(true).parse::<i64>(),
            target.value().to_string_decimal(true).parse::<i64>(),
        ) else {
            return false;
        };
        if source < 0 || target < 0 || !sources.insert(source) || !targets.insert(target) {
            return false;
        }
    }
    true
}

fn ty_rank(ctx: &Context, ty: TypeHandle) -> Option<usize> {
    let object = ty.deref(ctx);
    Some(object.downcast_ref::<TensorType>()?.rank())
}
fn same_tensor_types(ctx: &Context, ir: &Operation) -> bool {
    ir.operands().zip(ir.results()).all(|(a, b)| {
        crate::inference::compatible_tensor_types(ctx, a.get_type(ctx), b.get_type(ctx))
    })
}
fn same_tensor_shapes(ctx: &Context, ir: &Operation) -> bool {
    ir.operands().zip(ir.results()).all(|(a, b)| {
        let ta = a.get_type(ctx);
        let tb = b.get_type(ctx);
        let oa = ta.deref(ctx);
        let ob = tb.deref(ctx);
        let (Some(a), Some(b)) = (
            oa.downcast_ref::<TensorType>(),
            ob.downcast_ref::<TensorType>(),
        ) else {
            return false;
        };
        a.dimensions.len() == b.dimensions.len()
            && a.dimensions
                .iter()
                .zip(&b.dimensions)
                .all(|(&x, &y)| x == y || x < 0 || y < 0)
    })
}
fn is_ranked(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx).downcast_ref::<TensorType>().is_some()
}
fn all_ranked(ctx: &Context, ir: &Operation) -> bool {
    ir.operands().all(|v| is_ranked(ctx, v.get_type(ctx)))
        && ir.results().all(|v| is_ranked(ctx, v.get_type(ctx)))
}
fn compatible_shapes(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    let (Some((a, ae)), Some((b, be))) = (
        crate::types::tensor_parts(ctx, a),
        crate::types::tensor_parts(ctx, b),
    ) else {
        return false;
    };
    ae == be && a.len() == b.len() && a.iter().zip(&b).all(|(&x, &y)| x == y || x < 0 || y < 0)
}
fn reducer_contract(ctx: &Context, ir: &Operation, check_shape: bool, count: usize) -> bool {
    use pliron::linked_list::ContainsLinkedList;
    if count == 0
        || ir.get_num_operands() < count
        || ir.get_num_results() != count
        || ir.regions().count() != 1
    {
        return false;
    }
    let region = ir.get_region(0);
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    if blocks.len() != 1 {
        return false;
    }
    let block = blocks[0].deref(ctx);
    if block.get_num_arguments() != count.saturating_mul(2) {
        return false;
    }
    let ops: Vec<_> = block.iter(ctx).collect();
    let Some(last) = ops.last() else { return false };
    if Operation::get_op_dyn(*last, ctx)
        .downcast_ref::<crate::ops::ReturnOp>()
        .is_none()
    {
        return false;
    }
    let ret = last.deref(ctx);
    if ret.get_num_operands() != count {
        return false;
    }
    let inputs: Vec<_> = ir.operands().map(|v| v.get_type(ctx)).collect();
    for (index, &input) in inputs.iter().take(count).enumerate() {
        let accumulator = ret.get_operand(index).get_type(ctx);
        if tensor_parts(ctx, accumulator).is_none_or(|(shape, _)| !shape.is_empty())
            || block.get_argument(index).get_type(ctx) != accumulator
            || block.get_argument(count + index).get_type(ctx) != accumulator
        {
            return false;
        }
        let Ok(expected) = crate::inference::infer_all_reduce_types(ctx, &[input], accumulator)
        else {
            return false;
        };
        let Some((expected_shape, expected_element)) = crate::types::tensor_parts(ctx, expected[0])
        else {
            return false;
        };
        let Some((actual_shape, actual_element)) =
            crate::types::tensor_parts(ctx, ir.get_result(index).get_type(ctx))
        else {
            return false;
        };
        if expected_element != actual_element
            || (check_shape
                && !crate::inference::compatible_tensor_types(
                    ctx,
                    expected[0],
                    ir.get_result(index).get_type(ctx),
                ))
            || (!check_shape && expected_shape.len() != actual_shape.len())
        {
            return false;
        }
    }
    true
}

#[pliron_op(name="stablehlo.all_gather",format="operands(CharSpace(`,`)) attr($stablehlo_all_gather_dimension, $IntegerAttr) attr($stablehlo_all_gather_replica_groups, $VecAttr) attr($stablehlo_all_gather_channel_handle, $ChannelHandleAttr) attr($stablehlo_all_gather_use_global_device_ids, $BoolAttr)",attributes=(stablehlo_all_gather_dimension:IntegerAttr,stablehlo_all_gather_replica_groups:VecAttr,stablehlo_all_gather_channel_handle:ChannelHandleAttr,stablehlo_all_gather_use_global_device_ids:BoolAttr))]
pub struct AllGatherOp;
impl AllGatherOp {
    pub fn new(
        ctx: &mut Context,
        operands: Vec<Value>,
        results: Vec<TypeHandle>,
        dimension: u64,
        groups: VecAttr,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                operands,
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_all_gather_dimension(ctx, signed_attr(ctx, dimension));
        op.set_attr_stablehlo_all_gather_replica_groups(ctx, groups);
        op
    }
}
impl Verify for AllGatherOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() == 0
            || ir.get_num_operands() != ir.get_num_results()
            || !all_ranked(ctx, &ir)
            || self.get_attr_stablehlo_all_gather_dimension(ctx).is_none()
            || !global_group_attributes_valid(
                self.get_attr_stablehlo_all_gather_replica_groups(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_all_gather_channel_handle(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_all_gather_use_global_device_ids(ctx)
                    .as_deref(),
                false,
                true,
            )
        {
            return invalid(
                self,
                ctx,
                "all_gather requires paired ranked tensors, dimension, and replica groups",
            );
        }
        let dim = self
            .get_attr_stablehlo_all_gather_dimension(ctx)
            .and_then(|v| v.value().to_string_decimal(true).parse::<usize>().ok());
        let Some(dim) = dim else {
            return invalid(self, ctx, "all_gather dimension is invalid");
        };
        for (input, result) in ir.operands().zip(ir.results()) {
            let (Some((a, ae)), Some((b, be))) = (
                crate::types::tensor_parts(ctx, input.get_type(ctx)),
                crate::types::tensor_parts(ctx, result.get_type(ctx)),
            ) else {
                return invalid(self, ctx, "all_gather requires ranked tensors");
            };
            if a.len() != b.len()
                || dim >= a.len()
                || ae != be
                || a.iter()
                    .zip(&b)
                    .enumerate()
                    .any(|(i, (&x, &y))| i != dim && x >= 0 && y >= 0 && x != y)
                || (a[dim] >= 0 && b[dim] >= 0 && b[dim] < a[dim])
            {
                return invalid(self, ctx, "all_gather result shape is invalid");
            }
        }
        if let Some(size) = replica_group_size(
            self.get_attr_stablehlo_all_gather_replica_groups(ctx)
                .as_deref(),
        ) {
            let inputs: Vec<_> = ir.operands().map(|v| v.get_type(ctx)).collect();
            let Ok(expected) =
                crate::inference::infer_all_gather_types(ctx, &inputs, dim as i64, size)
            else {
                return invalid(self, ctx, "all_gather geometry is invalid");
            };
            if expected
                .iter()
                .copied()
                .zip(ir.results())
                .any(|(e, r)| !crate::inference::compatible_tensor_types(ctx, e, r.get_type(ctx)))
            {
                return invalid(
                    self,
                    ctx,
                    "all_gather result does not match replica group size",
                );
            }
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.all_reduce",format="operands(CharSpace(`,`)) regions(CharSpace(`,`)) attr($stablehlo_all_reduce_replica_groups, $VecAttr) attr($stablehlo_all_reduce_channel_handle, $ChannelHandleAttr) attr($stablehlo_all_reduce_use_global_device_ids, $BoolAttr)",attributes=(stablehlo_all_reduce_replica_groups:VecAttr,stablehlo_all_reduce_channel_handle:ChannelHandleAttr,stablehlo_all_reduce_use_global_device_ids:BoolAttr))]
pub struct AllReduceOp;
impl AllReduceOp {
    pub fn new(
        ctx: &mut Context,
        operands: Vec<Value>,
        results: Vec<TypeHandle>,
        groups: VecAttr,
    ) -> Self {
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
        op.set_attr_stablehlo_all_reduce_replica_groups(ctx, groups);
        op
    }
    pub fn reducer_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}
impl Verify for AllReduceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() == 0
            || ir.get_num_operands() != ir.get_num_results()
            || !all_ranked(ctx, &ir)
            || !reducer_contract(ctx, &ir, true, ir.get_num_operands())
            || !same_tensor_shapes(ctx, &ir)
            || !global_group_attributes_valid(
                self.get_attr_stablehlo_all_reduce_replica_groups(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_all_reduce_channel_handle(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_all_reduce_use_global_device_ids(ctx)
                    .as_deref(),
                true,
                true,
            )
        {
            return invalid(
                self,
                ctx,
                "all_reduce requires paired ranked tensors, reducer, and replica groups",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.reduce_scatter",format="operands(CharSpace(`,`)) attr($stablehlo_reduce_scatter_dimension, $IntegerAttr) attr($stablehlo_reduce_scatter_replica_groups, $VecAttr) attr($stablehlo_reduce_scatter_channel_handle, $ChannelHandleAttr) attr($stablehlo_reduce_scatter_use_global_device_ids, $BoolAttr) regions(CharSpace(`,`))",attributes=(stablehlo_reduce_scatter_dimension:IntegerAttr,stablehlo_reduce_scatter_replica_groups:VecAttr,stablehlo_reduce_scatter_channel_handle:ChannelHandleAttr,stablehlo_reduce_scatter_use_global_device_ids:BoolAttr))]
pub struct ReduceScatterOp;
impl ReduceScatterOp {
    pub fn new(
        ctx: &mut Context,
        operands: Vec<Value>,
        results: Vec<TypeHandle>,
        dimension: u64,
        groups: VecAttr,
    ) -> Self {
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
        op.set_attr_stablehlo_reduce_scatter_dimension(ctx, signed_attr(ctx, dimension));
        op.set_attr_stablehlo_reduce_scatter_replica_groups(ctx, groups);
        op
    }
    pub fn reducer_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}
impl Verify for ReduceScatterOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() == 0
            || ir.get_num_operands() != ir.get_num_results()
            || !all_ranked(ctx, &ir)
            || ir.regions().count() != 1
            || !reducer_contract(ctx, &ir, false, ir.get_num_operands())
            || !global_group_attributes_valid(
                self.get_attr_stablehlo_reduce_scatter_replica_groups(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_reduce_scatter_channel_handle(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_reduce_scatter_use_global_device_ids(ctx)
                    .as_deref(),
                false,
                true,
            )
        {
            return invalid(
                self,
                ctx,
                "reduce_scatter requires paired ranked tensors, reducer and dimensions",
            );
        }
        let Some(dim) = self
            .get_attr_stablehlo_reduce_scatter_dimension(ctx)
            .and_then(|v| v.value().to_string_decimal(true).parse::<usize>().ok())
        else {
            return invalid(self, ctx, "reduce_scatter dimension is invalid");
        };
        if ir
            .operands()
            .any(|v| ty_rank(ctx, v.get_type(ctx)).is_none_or(|rank| dim >= rank))
        {
            return invalid(self, ctx, "reduce_scatter dimension is outside tensor rank");
        }
        if let Some(size) = replica_group_size(
            self.get_attr_stablehlo_reduce_scatter_replica_groups(ctx)
                .as_deref(),
        ) {
            let inputs: Vec<_> = ir.operands().map(|v| v.get_type(ctx)).collect();
            let Ok(expected) =
                crate::inference::infer_reduce_scatter_types(ctx, &inputs, dim as i64, size)
            else {
                return invalid(self, ctx, "reduce_scatter geometry is invalid");
            };
            if expected
                .iter()
                .copied()
                .zip(ir.results())
                .any(|(e, r)| !compatible_shapes(ctx, e, r.get_type(ctx)))
            {
                return invalid(
                    self,
                    ctx,
                    "reduce_scatter result does not match replica group size",
                );
            }
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.all_to_all",format="operands(CharSpace(`,`)) attr($stablehlo_all_to_all_split_dimension, $IntegerAttr) attr($stablehlo_all_to_all_concat_dimension, $IntegerAttr) attr($stablehlo_all_to_all_split_count, $IntegerAttr) attr($stablehlo_all_to_all_replica_groups, $VecAttr) attr($stablehlo_all_to_all_channel_handle, $ChannelHandleAttr) attr($stablehlo_all_to_all_use_global_device_ids, $BoolAttr)",attributes=(stablehlo_all_to_all_split_dimension:IntegerAttr,stablehlo_all_to_all_concat_dimension:IntegerAttr,stablehlo_all_to_all_split_count:IntegerAttr,stablehlo_all_to_all_replica_groups:VecAttr,stablehlo_all_to_all_channel_handle:ChannelHandleAttr,stablehlo_all_to_all_use_global_device_ids:BoolAttr))]
pub struct AllToAllOp;
impl AllToAllOp {
    pub fn new(
        ctx: &mut Context,
        operands: Vec<Value>,
        results: Vec<TypeHandle>,
        split: u64,
        concat: u64,
        count: u64,
        groups: VecAttr,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                operands,
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_all_to_all_split_dimension(ctx, signed_attr(ctx, split));
        op.set_attr_stablehlo_all_to_all_concat_dimension(ctx, signed_attr(ctx, concat));
        op.set_attr_stablehlo_all_to_all_split_count(ctx, signed_attr(ctx, count));
        op.set_attr_stablehlo_all_to_all_replica_groups(ctx, groups);
        op
    }
}
impl Verify for AllToAllOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() == 0
            || ir.get_num_operands() != ir.get_num_results()
            || !all_ranked(ctx, &ir)
            || !global_group_attributes_valid(
                self.get_attr_stablehlo_all_to_all_replica_groups(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_all_to_all_channel_handle(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_all_to_all_use_global_device_ids(ctx)
                    .as_deref(),
                false,
                true,
            )
            || self
                .get_attr_stablehlo_all_to_all_split_count(ctx)
                .is_none_or(|v| {
                    v.value()
                        .to_string_decimal(true)
                        .parse::<i64>()
                        .map_or(true, |n| n <= 0)
                })
        {
            return invalid(
                self,
                ctx,
                "all_to_all requires paired ranked tensors and positive split count",
            );
        }
        let Some(split) = self
            .get_attr_stablehlo_all_to_all_split_dimension(ctx)
            .and_then(|v| v.value().to_string_decimal(true).parse::<usize>().ok())
        else {
            return invalid(self, ctx, "all_to_all split dimension is invalid");
        };
        let Some(concat) = self
            .get_attr_stablehlo_all_to_all_concat_dimension(ctx)
            .and_then(|v| v.value().to_string_decimal(true).parse::<usize>().ok())
        else {
            return invalid(self, ctx, "all_to_all concat dimension is invalid");
        };
        if ir
            .operands()
            .any(|v| ty_rank(ctx, v.get_type(ctx)).is_none_or(|r| split >= r || concat >= r))
        {
            return invalid(self, ctx, "all_to_all dimensions are outside operand rank");
        }
        let count = self
            .get_attr_stablehlo_all_to_all_split_count(ctx)
            .and_then(|v| v.value().to_string_decimal(true).parse::<i64>().ok())
            .unwrap_or(0);
        if replica_group_size(
            self.get_attr_stablehlo_all_to_all_replica_groups(ctx)
                .as_deref(),
        )
        .is_some_and(|size| size != count)
        {
            return invalid(
                self,
                ctx,
                "all_to_all split count must equal replica group size",
            );
        }
        let inputs: Vec<_> = ir.operands().map(|v| v.get_type(ctx)).collect();
        let Ok(expected) = crate::inference::infer_all_to_all_types(
            ctx,
            &inputs,
            split as i64,
            concat as i64,
            count,
        ) else {
            return invalid(self, ctx, "all_to_all split geometry is invalid");
        };
        if expected
            .iter()
            .copied()
            .zip(ir.results())
            .any(|(e, r)| !crate::inference::compatible_tensor_types(ctx, e, r.get_type(ctx)))
        {
            return invalid(
                self,
                ctx,
                "all_to_all result types do not match inferred geometry",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.collective_broadcast",format="operands(CharSpace(`,`)) attr($stablehlo_collective_broadcast_has_dynamic_root, $BoolAttr) attr($stablehlo_collective_broadcast_replica_groups, $VecAttr) attr($stablehlo_collective_broadcast_channel_handle, $ChannelHandleAttr)",attributes=(stablehlo_collective_broadcast_has_dynamic_root:BoolAttr,stablehlo_collective_broadcast_replica_groups:VecAttr,stablehlo_collective_broadcast_channel_handle:ChannelHandleAttr))]
pub struct CollectiveBroadcastOp;
impl CollectiveBroadcastOp {
    pub fn new(
        ctx: &mut Context,
        operands: Vec<Value>,
        results: Vec<TypeHandle>,
        dynamic_root: bool,
        groups: VecAttr,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                operands,
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_collective_broadcast_has_dynamic_root(
            ctx,
            BoolAttr::new(dynamic_root),
        );
        op.set_attr_stablehlo_collective_broadcast_replica_groups(ctx, groups);
        op
    }
}
impl Verify for CollectiveBroadcastOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        let dynamic = self
            .get_attr_stablehlo_collective_broadcast_has_dynamic_root(ctx)
            .is_some_and(|x| bool::from(x.clone()));
        let count = ir.get_num_operands().saturating_sub(usize::from(dynamic));
        if count == 0
            || ir.get_num_results() != count
            || !all_ranked(ctx, &ir)
            || !valid_replica_groups(
                self.get_attr_stablehlo_collective_broadcast_replica_groups(ctx)
                    .as_deref(),
                false,
                false,
            )
        {
            return invalid(
                self,
                ctx,
                "collective_broadcast operands, results, root and groups invalid",
            );
        }
        if dynamic {
            let Some((shape, element)) = crate::types::tensor_parts(
                ctx,
                ir.get_operand(ir.get_num_operands() - 1).get_type(ctx),
            ) else {
                return invalid(self, ctx, "dynamic root must be an i32 vector");
            };
            if shape != vec![count as i64] || element != ElementType::I32 {
                return invalid(
                    self,
                    ctx,
                    "dynamic root must be an i32 vector sized to data operands",
                );
            }
        }
        if ir
            .operands()
            .take(count)
            .zip(ir.results())
            .any(|(input, result)| {
                !crate::inference::compatible_tensor_types(
                    ctx,
                    input.get_type(ctx),
                    result.get_type(ctx),
                )
            })
        {
            return invalid(
                self,
                ctx,
                "collective_broadcast result types must match data operands",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.collective_reduce",format="operands(CharSpace(`,`)) regions(CharSpace(`,`)) attr($stablehlo_collective_reduce_replica_groups, $VecAttr) attr($stablehlo_collective_reduce_has_dynamic_root, $BoolAttr) attr($stablehlo_collective_reduce_channel_handle, $ChannelHandleAttr) attr($stablehlo_collective_reduce_use_global_device_ids, $BoolAttr)",attributes=(stablehlo_collective_reduce_replica_groups:VecAttr,stablehlo_collective_reduce_has_dynamic_root:BoolAttr,stablehlo_collective_reduce_channel_handle:ChannelHandleAttr,stablehlo_collective_reduce_use_global_device_ids:BoolAttr))]
pub struct CollectiveReduceOp;
impl CollectiveReduceOp {
    pub fn new(
        ctx: &mut Context,
        operands: Vec<Value>,
        results: Vec<TypeHandle>,
        groups: VecAttr,
    ) -> Self {
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
        op.set_attr_stablehlo_collective_reduce_replica_groups(ctx, groups);
        op.set_attr_stablehlo_collective_reduce_has_dynamic_root(ctx, BoolAttr::new(false));
        op
    }
}
impl Verify for CollectiveReduceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        let dynamic = self
            .get_attr_stablehlo_collective_reduce_has_dynamic_root(ctx)
            .is_some_and(|v| bool::from(v.clone()));
        let count = ir.get_num_operands().saturating_sub(usize::from(dynamic));
        if count == 0
            || count != ir.get_num_results()
            || !all_ranked(ctx, &ir)
            || !reducer_contract(ctx, &ir, true, count)
            || !same_tensor_shapes(ctx, &ir)
            || !global_group_attributes_valid(
                self.get_attr_stablehlo_collective_reduce_replica_groups(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_collective_reduce_channel_handle(ctx)
                    .as_deref(),
                self.get_attr_stablehlo_collective_reduce_use_global_device_ids(ctx)
                    .as_deref(),
                true,
                true,
            )
        {
            return invalid(
                self,
                ctx,
                "collective_reduce requires paired tensors and reducer",
            );
        }
        if dynamic {
            let Some((shape, element)) =
                crate::types::tensor_parts(ctx, ir.get_operand(count).get_type(ctx))
            else {
                return invalid(self, ctx, "dynamic collective root must be ranked i32");
            };
            if shape != vec![count as i64] || element != ElementType::I32 {
                return invalid(
                    self,
                    ctx,
                    "dynamic collective root must be i32 vector sized to data operands",
                );
            }
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.collective_permute",format="$0 attr($stablehlo_collective_permute_source_target_pairs, $VecAttr) attr($stablehlo_collective_permute_channel_handle, $ChannelHandleAttr)",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_collective_permute_source_target_pairs:VecAttr,stablehlo_collective_permute_channel_handle:ChannelHandleAttr))]
pub struct CollectivePermuteOp;
impl CollectivePermuteOp {
    pub fn new(ctx: &mut Context, input: Value, result: TypeHandle, pairs: VecAttr) -> Self {
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
        op.set_attr_stablehlo_collective_permute_source_target_pairs(ctx, pairs);
        op
    }
}
impl Verify for CollectivePermuteOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1
            || ir.get_num_results() != 1
            || !all_ranked(ctx, &ir)
            || !same_tensor_types(ctx, &ir)
            || !valid_source_target_pairs(
                self.get_attr_stablehlo_collective_permute_source_target_pairs(ctx)
                    .as_deref(),
            )
        {
            return invalid(
                self,
                ctx,
                "collective_permute requires unchanged ranked tensor and source/target pairs",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.cross_replica_sum",format="$0 attr($stablehlo_cross_replica_sum_replica_groups, $VecAttr)",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_cross_replica_sum_replica_groups:VecAttr))]
pub struct CrossReplicaSumOp;
impl CrossReplicaSumOp {
    pub fn new(ctx: &mut Context, input: Value, result: TypeHandle, groups: VecAttr) -> Self {
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
        op.set_attr_stablehlo_cross_replica_sum_replica_groups(ctx, groups);
        op
    }
}
impl Verify for CrossReplicaSumOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1
            || ir.get_num_results() != 1
            || !all_ranked(ctx, &ir)
            || !same_tensor_types(ctx, &ir)
            || !valid_replica_groups(
                self.get_attr_stablehlo_cross_replica_sum_replica_groups(ctx)
                    .as_deref(),
                true,
                true,
            )
        {
            return invalid(
                self,
                ctx,
                "cross_replica_sum preserves a ranked tensor and requires replica groups",
            );
        }
        Ok(())
    }
}

fn signed_attr(ctx: &Context, value: u64) -> IntegerAttr {
    let ty = IntegerType::get(ctx, 64, Signedness::Signless);
    IntegerAttr::new(ty, APInt::from_u64(value, bw(64)))
}
