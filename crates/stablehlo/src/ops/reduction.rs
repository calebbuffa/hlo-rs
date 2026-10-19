use crate::types::*;
use crate::verification::invalid;
use pliron::region::Region;
use pliron::{
    attribute::AttrObj,
    builtin::{
        attributes::{BoolAttr, IntegerAttr, VecAttr},
        types::{IntegerType, Signedness},
    },
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
fn ints(ctx: &Context, v: impl IntoIterator<Item = u64>) -> VecAttr {
    crate::attributes::integer_list(ctx, v)
}
fn read_ints(a: Option<std::cell::Ref<'_, VecAttr>>) -> Option<Vec<u64>> {
    a.as_deref().and_then(crate::attributes::read_integer_list)
}
fn signed_list(ctx: &Context, values: impl IntoIterator<Item = i64>) -> VecAttr {
    VecAttr::new(
        values
            .into_iter()
            .map(|value| {
                Box::new(IntegerAttr::new(
                    IntegerType::get(ctx, 64, Signedness::Signed),
                    APInt::from_i64(value, bw(64)),
                )) as AttrObj
            })
            .collect(),
    )
}
fn read_signed_ints(attr: Option<std::cell::Ref<'_, VecAttr>>) -> Option<Vec<i64>> {
    attr.as_deref().and_then(|attr| {
        attr.0
            .iter()
            .map(|value| {
                value
                    .downcast_ref::<IntegerAttr>()?
                    .value()
                    .to_string_decimal(true)
                    .parse()
                    .ok()
            })
            .collect()
    })
}
fn parts(ctx: &Context, v: Value) -> Option<(Vec<i64>, TypeHandle)> {
    let ty = v.get_type(ctx);
    let d = ty.deref(ctx);
    let t = d.downcast_ref::<TensorType>()?;
    Some((t.dimensions.clone(), t.element))
}
fn parts_ty(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let d = ty.deref(ctx);
    let t = d.downcast_ref::<TensorType>()?;
    Some((t.dimensions.clone(), t.element))
}

fn region_signature(
    ctx: &Context,
    region: Ptr<Region>,
    arguments: &[TypeHandle],
    returns: &[TypeHandle],
) -> bool {
    use pliron::linked_list::ContainsLinkedList;
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    if blocks.len() != 1 {
        return false;
    }
    let block = blocks[0].deref(ctx);
    let args: Vec<_> = block.arguments().map(|v| v.get_type(ctx)).collect();
    if args != arguments {
        return false;
    }
    let ops: Vec<_> = block.iter(ctx).collect();
    let Some(&last) = ops.last() else {
        return false;
    };
    if Operation::get_op_dyn(last, ctx).get_opid().to_string() != "stablehlo.return" {
        return false;
    }
    let ret = last.deref(ctx);
    ret.get_num_results() == 0
        && ret.get_num_operands() == returns.len()
        && ret
            .operands()
            .zip(returns)
            .all(|(v, t)| v.get_type(ctx) == *t)
}
fn compatible_dim(a: i64, b: i64) -> bool {
    a < 0 || b < 0 || a == b
}
fn promotable(ctx: &Context, source: TypeHandle, target: TypeHandle) -> bool {
    let (Some(a), Some(b)) = (
        crate::types::primitive_element(ctx, source),
        crate::types::primitive_element(ctx, target),
    ) else {
        return source == target;
    };
    let same_class = (a.is_integer() && b.is_integer())
        || (a.is_float() && b.is_float())
        || (matches!(a, ElementType::C64 | ElementType::C128)
            && matches!(b, ElementType::C64 | ElementType::C128));
    same_class && matches!((a.bit_width(),b.bit_width()),(Some(x),Some(y)) if x<=y)
}
fn scalar(ctx: &Context, element: TypeHandle) -> TypeHandle {
    TensorType::get(ctx, vec![], element).into()
}
fn region_for_reduce(ctx: &Context, op: &ReduceOp) -> bool {
    let ir = op.get_operation().deref(ctx);
    region_for_reduce_ir(ctx, &ir, op.reducer_region(ctx))
}
fn region_for_reduce_ir(ctx: &Context, ir: &Operation, region: Ptr<Region>) -> bool {
    let n = ir.get_num_results();
    let mut args = Vec::new();
    let mut rets = Vec::new();
    for _ in 0..2 {
        for i in 0..n {
            let Some((_, element)) = parts_ty(ctx, ir.get_result(i).get_type(ctx)) else {
                return false;
            };
            args.push(scalar(ctx, element));
        }
    }
    for i in 0..n {
        let Some((_, element)) = parts_ty(ctx, ir.get_result(i).get_type(ctx)) else {
            return false;
        };
        rets.push(scalar(ctx, element));
    }
    region_signature(ctx, region, &args, &rets)
}

#[pliron_op(name="stablehlo.reduce",format="operands(CharSpace(`,`)) attr($stablehlo_reduce_dimensions, $VecAttr) regions(CharSpace(`,`))",attributes=(stablehlo_reduce_dimensions:VecAttr))]
pub struct ReduceOp;
impl ReduceOp {
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        inits: Vec<Value>,
        results: Vec<TypeHandle>,
        dimensions: Vec<u64>,
    ) -> Self {
        let mut operands = inputs;
        operands.extend(inits);
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
        op.set_attr_stablehlo_reduce_dimensions(ctx, ints(ctx, dimensions));
        op
    }
    pub fn reducer_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}
impl Verify for ReduceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        let n = ir.get_num_results();
        if n == 0 || ir.get_num_operands() != 2 * n {
            return invalid(
                self,
                ctx,
                "reduce requires paired inputs and initial values",
            );
        };
        let Some(dims) = read_ints(self.get_attr_stablehlo_reduce_dimensions(ctx)) else {
            return invalid(self, ctx, "reduce dimensions missing");
        };
        let Some((shape, _)) = parts(ctx, ir.get_operand(0)) else {
            return invalid(self, ctx, "reduce input must be ranked");
        };
        if !crate::verification::compute::valid_reduction_dimensions(shape.len(), &dims) {
            return invalid(self, ctx, "reduce dimensions must be distinct and in range");
        };
        for i in 0..n {
            let (Some((input, ie)), Some((init, ne)), Some((out, oe))) = (
                parts(ctx, ir.get_operand(i)),
                parts(ctx, ir.get_operand(n + i)),
                parts(ctx, ir.get_result(i)),
            ) else {
                return invalid(
                    self,
                    ctx,
                    "reduce inputs, initial values and results must be ranked",
                );
            };
            let reduced = dims.iter().map(|&d| d as usize).collect::<Vec<_>>();
            let expected = shape
                .iter()
                .enumerate()
                .filter_map(|(axis, &d)| (!reduced.contains(&axis)).then_some(d))
                .collect::<Vec<_>>();
            if ie != ne
                || !promotable(ctx, ie, oe)
                || !init.is_empty()
                || input.len() != shape.len()
                || input
                    .iter()
                    .zip(&shape)
                    .any(|(&a, &b)| !compatible_dim(a, b))
                || expected.len() != out.len()
                || expected
                    .iter()
                    .zip(&out)
                    .any(|(&a, &b)| !compatible_dim(a, b))
            {
                return invalid(
                    self,
                    ctx,
                    "reduce initial values and result elements are invalid",
                );
            };
        }
        if !region_for_reduce(ctx, self) {
            return invalid(self, ctx, "reduce reducer region signature mismatch");
        };
        Ok(())
    }
}

#[pliron_op(name="stablehlo.reduce_window",format="operands(CharSpace(`,`)) attr($stablehlo_reduction_window, $VecAttr) attr($stablehlo_reduction_strides, $VecAttr) attr($stablehlo_reduction_base_dilations, $VecAttr) attr($stablehlo_reduction_window_dilations, $VecAttr) attr($stablehlo_reduction_padding, $VecAttr) regions(CharSpace(`,`))",attributes=(stablehlo_reduction_window:VecAttr,stablehlo_reduction_strides:VecAttr,stablehlo_reduction_base_dilations:VecAttr,stablehlo_reduction_window_dilations:VecAttr,stablehlo_reduction_padding:VecAttr))]
pub struct ReduceWindowOp;
impl ReduceWindowOp {
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        inits: Vec<Value>,
        results: Vec<TypeHandle>,
        window: Vec<u64>,
    ) -> Self {
        let mut operands = inputs;
        operands.extend(inits);
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
        op.set_attr_stablehlo_reduction_window(ctx, ints(ctx, window));
        op.set_attr_stablehlo_reduction_strides(ctx, ints(ctx, []));
        op.set_attr_stablehlo_reduction_base_dilations(ctx, ints(ctx, []));
        op.set_attr_stablehlo_reduction_window_dilations(ctx, ints(ctx, []));
        op.set_attr_stablehlo_reduction_padding(ctx, signed_list(ctx, []));
        op
    }
    #[allow(clippy::too_many_arguments)]
    pub fn with_window_attributes(
        ctx: &mut Context,
        inputs: Vec<Value>,
        inits: Vec<Value>,
        results: Vec<TypeHandle>,
        window: Vec<u64>,
        strides: Vec<u64>,
        base_dilations: Vec<u64>,
        window_dilations: Vec<u64>,
        padding: Vec<i64>,
    ) -> Self {
        let op = Self::new(ctx, inputs, inits, results, window);
        op.set_attr_stablehlo_reduction_strides(ctx, ints(ctx, strides));
        op.set_attr_stablehlo_reduction_base_dilations(ctx, ints(ctx, base_dilations));
        op.set_attr_stablehlo_reduction_window_dilations(ctx, ints(ctx, window_dilations));
        op.set_attr_stablehlo_reduction_padding(ctx, signed_list(ctx, padding));
        op
    }
    pub fn reducer_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}
impl Verify for ReduceWindowOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        let n = ir.get_num_results();
        if n == 0
            || ir.get_num_operands() != 2 * n
            || read_ints(self.get_attr_stablehlo_reduction_window(ctx))
                .is_none_or(|w| w.is_empty() || w.contains(&0))
        {
            return invalid(
                self,
                ctx,
                "reduce_window requires paired tensors and positive window dimensions",
            );
        };
        let Some(windows) = read_ints(self.get_attr_stablehlo_reduction_window(ctx)) else {
            return invalid(self, ctx, "reduce_window dimensions missing");
        };
        let Some((input_shape, _)) = parts(ctx, ir.get_operand(0)) else {
            return invalid(self, ctx, "reduce_window input must be ranked");
        };
        if windows.len() != input_shape.len()
            || ir.operands().any(|v| parts(ctx, v).is_none())
            || ir
                .results()
                .any(|v| parts_ty(ctx, v.get_type(ctx)).is_none())
        {
            return invalid(
                self,
                ctx,
                "reduce_window tensor ranks or window dimensions invalid",
            );
        };
        let inputs = (0..n)
            .map(|i| ir.get_operand(i).get_type(ctx))
            .collect::<Vec<_>>();
        let accumulators = (0..n)
            .map(|i| {
                let (_, e) = parts_ty(ctx, ir.get_result(i).get_type(ctx))?;
                Some(scalar(ctx, e))
            })
            .collect::<Option<Vec<_>>>();
        let Some(accumulators) = accumulators else {
            return invalid(self, ctx, "reduce_window result elements invalid");
        };
        let dims = windows
            .iter()
            .map(|&d| i64::try_from(d).ok())
            .collect::<Option<Vec<_>>>();
        let Some(dims) = dims else {
            return invalid(self, ctx, "reduce_window dimension overflow");
        };
        let (Some(strides), Some(base), Some(dilation), Some(padding_flat)) = (
            read_ints(self.get_attr_stablehlo_reduction_strides(ctx)),
            read_ints(self.get_attr_stablehlo_reduction_base_dilations(ctx)),
            read_ints(self.get_attr_stablehlo_reduction_window_dilations(ctx)),
            read_signed_ints(self.get_attr_stablehlo_reduction_padding(ctx)),
        ) else {
            return invalid(self, ctx, "reduce_window attributes are invalid");
        };
        let (Some(strides), Some(base), Some(dilation)) = (
            strides
                .iter()
                .map(|&x| i64::try_from(x).ok())
                .collect::<Option<Vec<_>>>(),
            base.iter()
                .map(|&x| i64::try_from(x).ok())
                .collect::<Option<Vec<_>>>(),
            dilation
                .iter()
                .map(|&x| i64::try_from(x).ok())
                .collect::<Option<Vec<_>>>(),
        ) else {
            return invalid(self, ctx, "reduce_window window attributes overflow");
        };
        let padding = if padding_flat.is_empty() {
            None
        } else if padding_flat.len() == 2 * input_shape.len() {
            Some(
                padding_flat
                    .chunks_exact(2)
                    .map(|pair| [pair[0], pair[1]])
                    .collect::<Vec<_>>(),
            )
        } else {
            return invalid(self, ctx, "reduce_window padding rank mismatch");
        };
        let Some(expected) = crate::inference::infer_reduce_window_types(
            ctx,
            &inputs,
            &accumulators,
            &dims,
            (!strides.is_empty()).then_some(strides.as_slice()),
            (!base.is_empty()).then_some(base.as_slice()),
            (!dilation.is_empty()).then_some(dilation.as_slice()),
            padding.as_deref(),
        )
        .ok() else {
            return invalid(self, ctx, "reduce_window inference failed");
        };
        for i in 0..n {
            let (
                Some((shape, input_element)),
                Some((init_shape, init_element)),
                Some((_, result_element)),
            ) = (
                parts(ctx, ir.get_operand(i)),
                parts(ctx, ir.get_operand(n + i)),
                parts(ctx, ir.get_result(i)),
            )
            else {
                return invalid(self, ctx, "reduce_window operand or result invalid");
            };
            if !init_shape.is_empty()
                || shape.len() != input_shape.len()
                || shape
                    .iter()
                    .zip(&input_shape)
                    .any(|(&a, &b)| !compatible_dim(a, b))
                || input_element != init_element
                || !promotable(ctx, input_element, result_element)
            {
                return invalid(
                    self,
                    ctx,
                    "reduce_window input, init, and result types invalid",
                );
            }
        }
        for (i, ty) in expected.iter().enumerate() {
            let (Some((shape, element)), Some((actual, actual_element))) = (
                parts_ty(ctx, *ty),
                parts_ty(ctx, ir.get_result(i).get_type(ctx)),
            ) else {
                return invalid(self, ctx, "reduce_window result invalid");
            };
            if element != actual_element
                || shape.len() != actual.len()
                || shape
                    .iter()
                    .zip(&actual)
                    .any(|(&a, &b)| a >= 0 && b >= 0 && a != b)
            {
                return invalid(self, ctx, "reduce_window result shape invalid");
            }
        }
        if !region_for_reduce_ir(ctx, &ir, ir.get_region(0)) {
            return invalid(self, ctx, "reduce_window reducer region signature invalid");
        };
        Ok(())
    }
}

#[pliron_op(name="stablehlo.select_and_scatter",format="$0 `,` $1 `,` $2 attr($stablehlo_reduction_select_window, $VecAttr) attr($stablehlo_reduction_select_strides, $VecAttr) attr($stablehlo_reduction_select_padding, $VecAttr) regions(CharSpace(`,`))",attributes=(stablehlo_reduction_select_window:VecAttr,stablehlo_reduction_select_strides:VecAttr,stablehlo_reduction_select_padding:VecAttr))]
pub struct SelectAndScatterOp;
impl SelectAndScatterOp {
    pub fn new(
        ctx: &mut Context,
        input: Value,
        source: Value,
        init: Value,
        result: TypeHandle,
        window: Vec<u64>,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input, source, init],
                vec![],
                2,
            ),
        };
        op.set_attr_stablehlo_reduction_select_window(ctx, ints(ctx, window));
        op.set_attr_stablehlo_reduction_select_strides(ctx, ints(ctx, []));
        op.set_attr_stablehlo_reduction_select_padding(ctx, signed_list(ctx, []));
        op
    }
    #[allow(clippy::too_many_arguments)]
    pub fn with_window_attributes(
        ctx: &mut Context,
        input: Value,
        source: Value,
        init: Value,
        result: TypeHandle,
        window: Vec<u64>,
        strides: Vec<u64>,
        padding: Vec<i64>,
    ) -> Self {
        let op = Self::new(ctx, input, source, init, result, window);
        op.set_attr_stablehlo_reduction_select_strides(ctx, ints(ctx, strides));
        op.set_attr_stablehlo_reduction_select_padding(ctx, signed_list(ctx, padding));
        op
    }
    pub fn select_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
    pub fn scatter_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(1)
    }
}
impl Verify for SelectAndScatterOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 3
            || ir.get_num_results() != 1
            || ir.operands().any(|v| parts(ctx, v).is_none())
            || parts(ctx, ir.get_result(0)).is_none()
            || read_ints(self.get_attr_stablehlo_reduction_select_window(ctx))
                .is_none_or(|w| w.is_empty() || w.contains(&0))
        {
            return invalid(
                self,
                ctx,
                "select_and_scatter requires ranked values and positive window dimensions",
            );
        };
        let Some((_, e)) = parts(ctx, ir.get_operand(0)) else {
            return invalid(self, ctx, "select input type invalid");
        };
        let value_type = scalar(ctx, e);
        let Some((_, result_element)) = parts(ctx, ir.get_result(0)) else {
            return invalid(self, ctx, "select_and_scatter result invalid");
        };
        let accumulator_type = scalar(ctx, result_element);
        if !promotable(ctx, e, result_element) {
            return invalid(
                self,
                ctx,
                "select_and_scatter accumulator is not promotable",
            );
        }
        let pred = scalar(ctx, ElementType::I1.native(ctx));
        if !region_signature(
            ctx,
            self.select_region(ctx),
            &[value_type, value_type],
            &[pred],
        ) || !region_signature(
            ctx,
            self.scatter_region(ctx),
            &[accumulator_type, accumulator_type],
            &[accumulator_type],
        ) {
            return invalid(
                self,
                ctx,
                "select_and_scatter region signatures are invalid",
            );
        };
        let (Some(windows), Some(strides), Some(padding_flat)) = (
            read_ints(self.get_attr_stablehlo_reduction_select_window(ctx)).and_then(|values| {
                values
                    .into_iter()
                    .map(|d| i64::try_from(d).ok())
                    .collect::<Option<Vec<_>>>()
            }),
            read_ints(self.get_attr_stablehlo_reduction_select_strides(ctx)).and_then(|values| {
                values
                    .into_iter()
                    .map(|d| i64::try_from(d).ok())
                    .collect::<Option<Vec<_>>>()
            }),
            read_signed_ints(self.get_attr_stablehlo_reduction_select_padding(ctx)),
        ) else {
            return invalid(self, ctx, "select_and_scatter window attributes invalid");
        };
        let rank = parts(ctx, ir.get_operand(0)).map_or(0, |(shape, _)| shape.len());
        let padding = if padding_flat.is_empty() {
            None
        } else if padding_flat.len() == 2 * rank {
            Some(
                padding_flat
                    .chunks_exact(2)
                    .map(|pair| [pair[0], pair[1]])
                    .collect::<Vec<_>>(),
            )
        } else {
            return invalid(self, ctx, "select_and_scatter padding rank mismatch");
        };
        let expected = crate::inference::infer_select_and_scatter_type(
            ctx,
            ir.get_operand(0).get_type(ctx),
            ir.get_operand(1).get_type(ctx),
            accumulator_type,
            &windows,
            (!strides.is_empty()).then_some(strides.as_slice()),
            padding.as_deref(),
        );
        let Some((expected_shape, expected_element)) =
            expected.ok().and_then(|ty| parts_ty(ctx, ty))
        else {
            return invalid(self, ctx, "select_and_scatter inference failed");
        };
        let Some((actual_shape, actual_element)) = parts_ty(ctx, ir.get_result(0).get_type(ctx))
        else {
            return invalid(self, ctx, "select_and_scatter result invalid");
        };
        if expected_element != actual_element
            || expected_shape.len() != actual_shape.len()
            || expected_shape
                .iter()
                .zip(&actual_shape)
                .any(|(&a, &b)| a >= 0 && b >= 0 && a != b)
        {
            return invalid(self, ctx, "select_and_scatter result shape invalid");
        };
        Ok(())
    }
}

#[pliron_op(name="stablehlo.sort",format="operands(CharSpace(`,`)) attr($stablehlo_reduction_sort_dimension, $IntegerAttr) attr($stablehlo_reduction_sort_stable, $BoolAttr) regions(CharSpace(`,`))",attributes=(stablehlo_reduction_sort_dimension:IntegerAttr,stablehlo_reduction_sort_stable:BoolAttr))]
pub struct SortOp;
impl SortOp {
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        results: Vec<TypeHandle>,
        dimension: u64,
        stable: bool,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                inputs,
                vec![],
                1,
            ),
        };
        op.set_attr_stablehlo_reduction_sort_dimension(
            ctx,
            crate::attributes::integer_attr(ctx, dimension),
        );
        op.set_attr_stablehlo_reduction_sort_stable(ctx, BoolAttr::new(stable));
        op
    }
    pub fn comparator_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}
impl Verify for SortOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() == 0
            || ir.get_num_operands() != ir.get_num_results()
            || ir.operands().any(|v| parts(ctx, v).is_none())
            || ir
                .results()
                .any(|v| parts_ty(ctx, v.get_type(ctx)).is_none())
        {
            return invalid(
                self,
                ctx,
                "sort requires paired ranked input and result tensors",
            );
        };
        let dim = self
            .get_attr_stablehlo_reduction_sort_dimension(ctx)
            .and_then(|a| a.value().to_string_decimal(false).parse::<usize>().ok());
        if dim.is_none_or(|d| {
            ir.get_operand(0)
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<crate::types::TensorType>()
                .is_none_or(|t| d >= t.dimensions.len())
        }) {
            return invalid(self, ctx, "sort dimension out of range");
        };
        let Some((common_shape, _)) = parts(ctx, ir.get_operand(0)) else {
            return invalid(self, ctx, "sort input type invalid");
        };
        if ir.operands().any(|value| {
            parts(ctx, value).is_none_or(|(shape, _)| {
                shape.len() != common_shape.len()
                    || shape
                        .iter()
                        .zip(&common_shape)
                        .any(|(&a, &b)| !compatible_dim(a, b))
            })
        }) {
            return invalid(self, ctx, "sort input shapes are incompatible");
        }
        for i in 0..ir.get_num_operands() {
            let (Some((input, element)), Some((output, out_element))) = (
                parts(ctx, ir.get_operand(i)),
                parts_ty(ctx, ir.get_result(i).get_type(ctx)),
            ) else {
                return invalid(self, ctx, "sort tensor invalid");
            };
            if element != out_element
                || input.len() != output.len()
                || input
                    .iter()
                    .zip(&output)
                    .any(|(&a, &b)| a >= 0 && b >= 0 && a != b)
            {
                return invalid(self, ctx, "sort result type mismatch");
            }
        }
        let mut args = Vec::new();
        for i in 0..ir.get_num_operands() {
            let Some((_, e)) = parts(ctx, ir.get_operand(i)) else {
                return invalid(self, ctx, "sort type invalid");
            };
            args.extend([scalar(ctx, e), scalar(ctx, e)]);
        }
        let pred = scalar(ctx, ElementType::I1.native(ctx));
        if !region_signature(ctx, self.comparator_region(ctx), &args, &[pred]) {
            return invalid(self, ctx, "sort comparator signature invalid");
        };
        Ok(())
    }
}

#[pliron_op(name="stablehlo.map",format="operands(CharSpace(`,`)) attr($stablehlo_reduction_map_dimensions, $VecAttr) regions(CharSpace(`,`))",attributes=(stablehlo_reduction_map_dimensions:VecAttr))]
pub struct MapOp;
impl MapOp {
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        result: TypeHandle,
        dimensions: Vec<u64>,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                inputs,
                vec![],
                1,
            ),
        };
        op.set_attr_stablehlo_reduction_map_dimensions(ctx, ints(ctx, dimensions));
        op
    }
    pub fn computation_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}
impl Verify for MapOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() == 0
            || ir.get_num_results() != 1
            || ir.operands().any(|v| parts(ctx, v).is_none())
            || parts(ctx, ir.get_result(0)).is_none()
        {
            return invalid(
                self,
                ctx,
                "map requires ranked inputs and one ranked result",
            );
        };
        let Some(dims) = read_ints(self.get_attr_stablehlo_reduction_map_dimensions(ctx)) else {
            return invalid(self, ctx, "map dimensions missing");
        };
        let Some((input_shape, _)) = parts(ctx, ir.get_operand(0)) else {
            return invalid(self, ctx, "map input must be ranked");
        };
        let rank = input_shape.len();
        if dims.len() != rank
            || dims
                .iter()
                .enumerate()
                .any(|(i, &d)| usize::try_from(d).ok() != Some(i))
        {
            return invalid(self, ctx, "map dimensions must enumerate all axes");
        };
        let mut args = Vec::new();
        for v in ir.operands() {
            let Some((_, e)) = parts(ctx, v) else {
                return invalid(self, ctx, "map operand invalid");
            };
            args.push(scalar(ctx, e));
        }
        let Some((_, e)) = parts(ctx, ir.get_result(0)) else {
            return invalid(self, ctx, "map result invalid");
        };
        let Some((common, _)) = parts(ctx, ir.get_operand(0)) else {
            return invalid(self, ctx, "map input must be ranked");
        };
        let Some((output, _)) = parts(ctx, ir.get_result(0)) else {
            return invalid(self, ctx, "map result must be ranked");
        };
        if common.len() != output.len()
            || common
                .iter()
                .zip(&output)
                .any(|(&a, &b)| !compatible_dim(a, b))
            || ir.operands().any(|v| {
                parts(ctx, v).is_none_or(|(shape, _)| {
                    shape.len() != common.len()
                        || shape
                            .iter()
                            .zip(&common)
                            .any(|(&a, &b)| !compatible_dim(a, b))
                })
            })
        {
            return invalid(self, ctx, "map input and result shapes are incompatible");
        };
        if !region_signature(ctx, self.computation_region(ctx), &args, &[scalar(ctx, e)]) {
            return invalid(self, ctx, "map computation region signature invalid");
        };
        Ok(())
    }
}
