//! Sorting, scanning, and selection operations.

use super::BodyBuilder;
use crate::support::*;
use pliron::{
    builtin::attributes::{BoolAttr, OperandSegmentSizesAttr, UnitAttr},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use stablehlo::attributes::IndexAttr;

/// Fails unless the region yields exactly one `i1` (any signedness, as MLIR's `isInteger(1)`).
fn yields_i1(ctx: &Context, op: Ptr<Operation>, types: &[TypeHandle]) -> Result<()> {
    ensure(
        ctx,
        op,
        types.len() == 1 && int_width(ctx, types[0]) == Some(1),
        "region must yield exactly one i1",
    )
}

/// Sorts `outputs` in place along `dimension` using a comparator region.
///
/// Each `inputs` operand pairs with an `outputs` operand of identical type.
/// The comparator receives two elements per input (`lhs0, rhs0, lhs1, rhs1, ..`)
/// and yields an `i1` that is true when `lhs` orders before `rhs`.
#[pliron_op(
    name = "iree_linalg_ext.sort",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    attributes = (
        iree_sort_dimension: IndexAttr,
        iree_sort_operand_segment_sizes: OperandSegmentSizesAttr
    )
)]
pub struct SortOp;

impl SortOp {
    /// Allocates the operation with an empty comparator block.
    pub fn new(ctx: &mut Context, inputs: Vec<Value>, outputs: Vec<Value>, dimension: u32) -> Self {
        let results = outputs.iter().map(|v| v.get_type(ctx)).collect();
        let args = inputs
            .iter()
            .flat_map(|&v| {
                let e = element(ctx, v);
                [e, e]
            })
            .collect();
        let sizes = segment_sizes(&[inputs.len(), outputs.len()]);
        let operands = inputs.into_iter().chain(outputs).collect();
        let op = new_with_body(ctx, |op| Self { op }, results, operands, args);
        op.set_attr_iree_sort_dimension(
            ctx,
            IndexAttr {
                value: i64::from(dimension),
            },
        );
        op.set_attr_iree_sort_operand_segment_sizes(ctx, sizes);
        op
    }

    /// Builds the operation with a comparator from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    pub fn build(
        ctx: &mut Context,
        inputs: Vec<Value>,
        outputs: Vec<Value>,
        dimension: u32,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, inputs, outputs, dimension);
        populate(ctx, op, body)
    }

    /// Returns the sorted dimension, or `None` when absent.
    pub fn dimension(&self, ctx: &Context) -> Option<i64> {
        self.get_attr_iree_sort_dimension(ctx).map(|a| a.value)
    }
}

impl Verify for SortOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let sizes = self.get_attr_iree_sort_operand_segment_sizes(ctx);
        let [(_, inputs), (_, outputs)] = segments::<2>(ctx, op, sizes.as_deref())?;
        ensure(ctx, op, inputs > 0, "expected at least one ins operand")?;
        ensure(ctx, op, outputs > 0, "expected at least one outs operand")?;
        ensure(
            ctx,
            op,
            inputs == outputs,
            "expected the same number of ins and outs operands",
        )?;
        verify_outputs(ctx, op, inputs..inputs + outputs)?;
        let types = operand_types(ctx, op);
        let (shape, _) = ranked_operand(ctx, op, types[0])?;
        dimension_in(ctx, op, self.dimension(ctx), shape.len())?;
        let mut args = Vec::new();
        for i in 0..inputs {
            let (dims, element) = ranked_operand(ctx, op, types[i])?;
            let (out_dims, out_element) = ranked_operand(ctx, op, types[inputs + i])?;
            ensure(
                ctx,
                op,
                dims == shape && out_dims == shape && out_element == element,
                "ins and outs must share one shape and pair element types",
            )?;
            args.extend([element, element]);
        }
        let yielded = region_signature(ctx, op, &args)?;
        yields_i1(ctx, op, &yielded)
    }
}

/// Inclusive or exclusive prefix scan of `input` along `dimension`.
///
/// Operands are `input` (ins) and `output`, `accumulator` (outs). The region
/// combines the running value and the next element and yields the new value.
#[pliron_op(
    name = "iree_linalg_ext.scan",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    attributes = (iree_scan_dimension: IndexAttr, iree_scan_inclusive: BoolAttr)
)]
pub struct ScanOp;

impl ScanOp {
    /// Allocates the operation with an empty combiner block.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        output: Value,
        accumulator: Value,
        dimension: u32,
        inclusive: bool,
    ) -> Self {
        let results = vec![output.get_type(ctx), accumulator.get_type(ctx)];
        let e = element(ctx, input);
        let op = new_with_body(
            ctx,
            |op| Self { op },
            results,
            vec![input, output, accumulator],
            vec![e, e],
        );
        op.set_attr_iree_scan_dimension(
            ctx,
            IndexAttr {
                value: i64::from(dimension),
            },
        );
        op.set_attr_iree_scan_inclusive(ctx, BoolAttr::new(inclusive));
        op
    }

    /// Builds the operation with a combiner from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        input: Value,
        output: Value,
        accumulator: Value,
        dimension: u32,
        inclusive: bool,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, input, output, accumulator, dimension, inclusive);
        populate(ctx, op, body)
    }

    /// Returns the scanned dimension, or `None` when absent.
    pub fn dimension(&self, ctx: &Context) -> Option<i64> {
        self.get_attr_iree_scan_dimension(ctx).map(|a| a.value)
    }

    /// Returns whether the scan is inclusive, or `None` when absent.
    pub fn inclusive(&self, ctx: &Context) -> Option<bool> {
        self.get_attr_iree_scan_inclusive(ctx)
            .map(|a| bool::from((*a).clone()))
    }
}

impl Verify for ScanOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        ensure(
            ctx,
            op,
            num_operands(ctx, op) == 3,
            "expected input, output, and accumulator",
        )?;
        ensure(
            ctx,
            op,
            self.inclusive(ctx).is_some(),
            "missing inclusive flag",
        )?;
        verify_outputs(ctx, op, 1..3)?;
        let types = operand_types(ctx, op);
        let (input, element) = ranked_operand(ctx, op, types[0])?;
        let (output, out_element) = ranked_operand(ctx, op, types[1])?;
        let (accumulator, acc_element) = ranked_operand(ctx, op, types[2])?;
        let dim = dimension_in(ctx, op, self.dimension(ctx), input.len())?;
        ensure(
            ctx,
            op,
            acc_element == element,
            "expected input/accumulator element types to be identical",
        )?;
        let mut reduced = input.clone();
        reduced.remove(dim);
        ensure(
            ctx,
            op,
            compatible_shapes(&reduced, &accumulator),
            "incompatible input/accumulator shapes",
        )?;
        ensure(
            ctx,
            op,
            out_element == element,
            "expected input/output element types to be identical",
        )?;
        ensure(
            ctx,
            op,
            compatible_shapes(&input, &output),
            "incompatible input/output shapes",
        )?;
        let yielded = region_signature(ctx, op, &[element, element])?;
        ensure(
            ctx,
            op,
            yielded == [element],
            "region must yield one input element",
        )
    }
}

/// Selects the top `K` elements along `dimension` using a comparator region.
///
/// Operands are `values` and optional `indices` (ins), then `output_values`
/// and `output_indices` (outs). `K` is the output extent of `dimension`. The
/// comparator receives two value elements and yields `i1`.
#[pliron_op(
    name = "iree_linalg_ext.topk",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    attributes = (iree_topk_dimension: IndexAttr)
)]
pub struct TopkOp;

impl TopkOp {
    /// Allocates the operation with an empty comparator block.
    pub fn new(
        ctx: &mut Context,
        values: Value,
        indices: Option<Value>,
        output_values: Value,
        output_indices: Value,
        dimension: u32,
    ) -> Self {
        let results = vec![output_values.get_type(ctx), output_indices.get_type(ctx)];
        let e = element(ctx, values);
        let operands = std::iter::once(values)
            .chain(indices)
            .chain([output_values, output_indices])
            .collect();
        let op = new_with_body(ctx, |op| Self { op }, results, operands, vec![e, e]);
        op.set_attr_iree_topk_dimension(
            ctx,
            IndexAttr {
                value: i64::from(dimension),
            },
        );
        op
    }

    /// Builds the operation with a comparator from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        values: Value,
        indices: Option<Value>,
        output_values: Value,
        output_indices: Value,
        dimension: u32,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(
            ctx,
            values,
            indices,
            output_values,
            output_indices,
            dimension,
        );
        populate(ctx, op, body)
    }

    /// Returns the reduced dimension, or `None` when absent.
    pub fn dimension(&self, ctx: &Context) -> Option<i64> {
        self.get_attr_iree_topk_dimension(ctx).map(|a| a.value)
    }
}

impl Verify for TopkOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let count = num_operands(ctx, op);
        ensure(
            ctx,
            op,
            (3..=4).contains(&count),
            "expected values, optional indices, and two outs",
        )?;
        verify_outputs(ctx, op, count - 2..count)?;
        let types = operand_types(ctx, op);
        let (input, element) = ranked_operand(ctx, op, types[0])?;
        let dim = dimension_in(ctx, op, self.dimension(ctx), input.len())?;
        let (out_values, out_element) = ranked_operand(ctx, op, types[count - 2])?;
        let (out_indices, out_index) = ranked_operand(ctx, op, types[count - 1])?;
        ensure(
            ctx,
            op,
            out_element == element,
            "expected input/output value types to be identical",
        )?;
        if count == 4 {
            let (indices, in_index) = ranked_operand(ctx, op, types[1])?;
            ensure(
                ctx,
                op,
                int_width(ctx, in_index) == Some(32) && int_width(ctx, out_index) == Some(32),
                "expected input/output indices types to be int32",
            )?;
            ensure(
                ctx,
                op,
                indices.len() == out_indices.len(),
                "expected input/output to have the same rank",
            )?;
            ensure(
                ctx,
                op,
                compatible_shapes(&input, &indices),
                "input indices/values shape must match",
            )?;
        }
        ensure(
            ctx,
            op,
            input.len() == out_values.len(),
            "expected input/output to have the same rank",
        )?;
        ensure(
            ctx,
            op,
            compatible_shapes(&out_values, &out_indices),
            "output indices/values shape must match",
        )?;
        let same_except_dim = input
            .iter()
            .zip(&out_values)
            .enumerate()
            .all(|(i, (&a, &b))| i == dim || compatible(a, b));
        ensure(ctx, op, same_except_dim, "incompatible input/output shapes")?;
        let yielded = region_signature(ctx, op, &[element, element])?;
        yields_i1(ctx, op, &yielded)
    }
}

/// Top-`K` selection with optional index tracking and an `is_sorted` hint.
///
/// Operands are `values`, optional `input_indices` (ins), `output_values`,
/// and optional `output_indices` (outs). Input indices require output indices.
#[pliron_op(
    name = "iree_linalg_ext.topk_v2",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    attributes = (
        iree_topk_v2_dimension: IndexAttr,
        iree_topk_v2_is_sorted: UnitAttr,
        iree_topk_v2_operand_segment_sizes: OperandSegmentSizesAttr
    )
)]
pub struct TopkV2Op;

impl TopkV2Op {
    /// Allocates the operation with an empty comparator block.
    pub fn new(
        ctx: &mut Context,
        values: Value,
        input_indices: Option<Value>,
        output_values: Value,
        output_indices: Option<Value>,
        dimension: u32,
        is_sorted: bool,
    ) -> Self {
        let results = std::iter::once(output_values)
            .chain(output_indices)
            .map(|v| v.get_type(ctx))
            .collect();
        let e = element(ctx, values);
        let sizes = segment_sizes(&[
            1,
            usize::from(input_indices.is_some()),
            1,
            usize::from(output_indices.is_some()),
        ]);
        let operands = std::iter::once(values)
            .chain(input_indices)
            .chain(std::iter::once(output_values))
            .chain(output_indices)
            .collect();
        let op = new_with_body(ctx, |op| Self { op }, results, operands, vec![e, e]);
        op.set_attr_iree_topk_v2_dimension(
            ctx,
            IndexAttr {
                value: i64::from(dimension),
            },
        );
        op.set_attr_iree_topk_v2_operand_segment_sizes(ctx, sizes);
        if is_sorted {
            op.set_attr_iree_topk_v2_is_sorted(ctx, UnitAttr::new());
        }
        op
    }

    /// Builds the operation with a comparator from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        values: Value,
        input_indices: Option<Value>,
        output_values: Value,
        output_indices: Option<Value>,
        dimension: u32,
        is_sorted: bool,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(
            ctx,
            values,
            input_indices,
            output_values,
            output_indices,
            dimension,
            is_sorted,
        );
        populate(ctx, op, body)
    }

    /// Returns the selection dimension, or `None` when absent.
    pub fn dimension(&self, ctx: &Context) -> Option<i64> {
        self.get_attr_iree_topk_v2_dimension(ctx).map(|a| a.value)
    }

    /// Returns whether the outputs are sorted.
    pub fn is_sorted(&self, ctx: &Context) -> bool {
        self.get_attr_iree_topk_v2_is_sorted(ctx).is_some()
    }
}

impl Verify for TopkV2Op {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let sizes = self.get_attr_iree_topk_v2_operand_segment_sizes(ctx);
        let [values, input_indices, output_values, output_indices] =
            segments::<4>(ctx, op, sizes.as_deref())?;
        ensure(
            ctx,
            op,
            values.1 == 1 && input_indices.1 <= 1 && output_values.1 == 1 && output_indices.1 <= 1,
            "expected values, optional input indices, output values, optional output indices",
        )?;
        verify_outputs(
            ctx,
            op,
            output_values.0..output_indices.0 + output_indices.1,
        )?;
        let types = operand_types(ctx, op);
        let (input, element) = ranked_operand(ctx, op, types[values.0])?;
        let (output, out_element) = ranked_operand(ctx, op, types[output_values.0])?;
        let dim = dimension_in(ctx, op, self.dimension(ctx), input.len())?;
        ensure(
            ctx,
            op,
            element == out_element,
            "expected input/output value types to be identical",
        )?;
        ensure(
            ctx,
            op,
            input.len() == output.len(),
            "expected input/output to have the same rank",
        )?;
        if input_indices.1 == 1 {
            ensure(
                ctx,
                op,
                output_indices.1 == 1,
                "input indices require output indices to carry provenance",
            )?;
            let (indices, in_index) = ranked_operand(ctx, op, types[input_indices.0])?;
            ensure(
                ctx,
                op,
                int_width(ctx, in_index).is_some(),
                "expected input indices to be integer type",
            )?;
            ensure(
                ctx,
                op,
                compatible_shapes(&input, &indices),
                "input values/indices shape must match",
            )?;
            let (_, out_index) = ranked_operand(ctx, op, types[output_indices.0])?;
            ensure(
                ctx,
                op,
                in_index == out_index,
                "expected input/output indices element types to be identical",
            )?;
        }
        if output_indices.1 == 1 {
            let (indices, out_index) = ranked_operand(ctx, op, types[output_indices.0])?;
            ensure(
                ctx,
                op,
                int_width(ctx, out_index).is_some(),
                "expected output indices to be integer type",
            )?;
            ensure(
                ctx,
                op,
                compatible_shapes(&output, &indices),
                "output values/indices shape must match",
            )?;
        }
        let others_match = input
            .iter()
            .zip(&output)
            .enumerate()
            .all(|(i, (&a, &b))| i == dim || compatible(a, b));
        ensure(ctx, op, others_match, "incompatible input/output shapes")?;
        if input[dim] >= 0 && output[dim] >= 0 {
            ensure(
                ctx,
                op,
                output[dim] != 0,
                "output dimension must be positive",
            )?;
            ensure(
                ctx,
                op,
                output[dim] <= input[dim],
                "output dimension must not exceed input",
            )?;
        }
        let yielded = region_signature(ctx, op, &[element, element])?;
        yields_i1(ctx, op, &yielded)
    }
}

/// Reduces `dimension` to the best value and its index using a comparator.
///
/// Operands are `input_value`, optional `input_index` (ins), `output_value`,
/// `output_index` (outs), and an optional `index_base` (`index`) that offsets
/// implicit indices. The comparator receives two value elements and yields
/// `i1`.
#[pliron_op(
    name = "iree_linalg_ext.arg_compare",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    attributes = (
        iree_arg_compare_dimension: IndexAttr,
        iree_arg_compare_operand_segment_sizes: OperandSegmentSizesAttr
    )
)]
pub struct ArgCompareOp;

impl ArgCompareOp {
    /// Allocates the operation with an empty comparator block.
    pub fn new(
        ctx: &mut Context,
        input_value: Value,
        input_index: Option<Value>,
        output_value: Value,
        output_index: Value,
        index_base: Option<Value>,
        dimension: u32,
    ) -> Self {
        let results = vec![output_value.get_type(ctx), output_index.get_type(ctx)];
        let e = element(ctx, input_value);
        let sizes = segment_sizes(&[
            1,
            usize::from(input_index.is_some()),
            1,
            1,
            usize::from(index_base.is_some()),
        ]);
        let operands = std::iter::once(input_value)
            .chain(input_index)
            .chain([output_value, output_index])
            .chain(index_base)
            .collect();
        let op = new_with_body(ctx, |op| Self { op }, results, operands, vec![e, e]);
        op.set_attr_iree_arg_compare_dimension(
            ctx,
            IndexAttr {
                value: i64::from(dimension),
            },
        );
        op.set_attr_iree_arg_compare_operand_segment_sizes(ctx, sizes);
        op
    }

    /// Builds the operation with a comparator from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        input_value: Value,
        input_index: Option<Value>,
        output_value: Value,
        output_index: Value,
        index_base: Option<Value>,
        dimension: u32,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(
            ctx,
            input_value,
            input_index,
            output_value,
            output_index,
            index_base,
            dimension,
        );
        populate(ctx, op, body)
    }

    /// Returns the reduced dimension, or `None` when absent.
    pub fn dimension(&self, ctx: &Context) -> Option<i64> {
        self.get_attr_iree_arg_compare_dimension(ctx)
            .map(|a| a.value)
    }
}

impl Verify for ArgCompareOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let sizes = self.get_attr_iree_arg_compare_operand_segment_sizes(ctx);
        let [value, index, out_value, out_index, base] = segments::<5>(ctx, op, sizes.as_deref())?;
        ensure(
            ctx,
            op,
            value.1 == 1 && index.1 <= 1 && out_value.1 == 1 && out_index.1 == 1 && base.1 <= 1,
            "expected input value, optional input index, two outs, optional index base",
        )?;
        verify_outputs(ctx, op, out_value.0..out_index.0 + 1)?;
        let types = operand_types(ctx, op);
        let (input, element) = ranked_operand(ctx, op, types[value.0])?;
        let (output, out_element) = ranked_operand(ctx, op, types[out_value.0])?;
        let (output_index, out_index_element) = ranked_operand(ctx, op, types[out_index.0])?;
        if index.1 == 1 {
            let (input_index, in_index_element) = ranked_operand(ctx, op, types[index.0])?;
            ensure(
                ctx,
                op,
                input == input_index,
                "explicit-index mode: value and index inputs must have the same shape",
            )?;
            ensure(
                ctx,
                op,
                is_int_or_index(ctx, in_index_element),
                "explicit-index mode: index input must have integer or index element type",
            )?;
            ensure(
                ctx,
                op,
                in_index_element == out_index_element,
                "explicit-index mode: input and output index element types must match",
            )?;
            ensure(
                ctx,
                op,
                base.1 == 0,
                "index_base must not be used with explicit indices",
            )?;
        }
        if base.1 == 1 {
            ensure(
                ctx,
                op,
                pliron_arith::shaped::is_index(ctx, types[base.0]),
                "index_base must be index",
            )?;
        }
        ensure(
            ctx,
            op,
            element == out_element,
            "input and output value element types must match",
        )?;
        ensure(
            ctx,
            op,
            is_int_or_index(ctx, out_index_element),
            "output index must have integer or index element type",
        )?;
        ensure(
            ctx,
            op,
            compatible_shapes(&output, &output_index),
            "output indices/values shape must match",
        )?;
        let dim = dimension_in(ctx, op, self.dimension(ctx), input.len())?;
        let mut expected = input.clone();
        expected.remove(dim);
        ensure(
            ctx,
            op,
            expected == output,
            "output shape must match input shape with reduction dimension removed",
        )?;
        let yielded = region_signature(ctx, op, &[element, element])?;
        yields_i1(ctx, op, &yielded)
    }
}
