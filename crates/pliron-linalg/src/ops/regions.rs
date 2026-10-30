//! Region-carrying structured operations: `generic`, `map`, `reduce`, and
//! the `yield`/`index` operations used inside their bodies.

use crate::{
    attributes::{I64ArrayAttr, IndexingMapsAttr, IteratorType, IteratorTypesAttr},
    support::*,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::attributes::{OperandSegmentSizesAttr, StringAttr},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_arith::shaped;
use stablehlo::attributes::{AffineMap, IndexAttr};

/// Body of a structured operation: receives the entry block and its arguments
/// and returns the values to yield.
pub type BodyBuilder<'a> =
    dyn FnOnce(&mut Context, Ptr<BasicBlock>, &[Value]) -> Result<Vec<Value>> + 'a;

fn element_of(ctx: &Context, value: &Value) -> TypeHandle {
    let ty = value.get_type(ctx);
    ranked_or_scalar(ctx, ty).map_or(ty, |(_, element)| element)
}

/// Runs `body` on the entry block of `op` and terminates it with [`YieldOp`].
fn populate(ctx: &mut Context, op: Ptr<Operation>, build: Box<BodyBuilder<'_>>) -> Result<()> {
    let block = body(ctx, op)?;
    let args = block_args(ctx, block);
    let yielded = build(ctx, block, &args)?;
    let yield_op = YieldOp::new(ctx, yielded);
    append(ctx, yield_op.get_operation(), block);
    Ok(())
}

/// Elementwise or reduction computation over an iteration space.
///
/// Operands are `ins` followed by `outs`. Each operand has one indexing map
/// from the iteration space to its indices, and the body receives one scalar
/// per operand (the element type of tensors, or the scalar itself) and
/// terminates with [`YieldOp`], yielding one value per output. Results have
/// the `outs` tensor types. Optional `doc` and `library_call` strings are
/// carried through unchanged.
#[pliron_op(
    name = "linalg.generic",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    attributes = (
        linalg_iterator_types: IteratorTypesAttr,
        linalg_doc: StringAttr,
        linalg_library_call: StringAttr
    )
)]
pub struct GenericOp;
crate::pliron_arith::shared_attr!(
    GenericOp,
    super::ATTR_KEY_LINALG_INDEXING_MAPS,
    IndexingMapsAttr,
    get_attr_linalg_indexing_maps,
    set_attr_linalg_indexing_maps
);
crate::pliron_arith::shared_attr!(
    GenericOp,
    super::ATTR_KEY_LINALG_OPERAND_SEGMENT_SIZES,
    OperandSegmentSizesAttr,
    get_attr_linalg_operand_segment_sizes,
    set_attr_linalg_operand_segment_sizes
);

impl GenericOp {
    /// Allocates the operation with an empty entry block.
    ///
    /// Block arguments are the element types of `inputs` then `outputs`.
    /// Populate the block, terminate it with [`YieldOp`], and call `verify`.
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        outputs: Vec<Value>,
        indexing_maps: Vec<AffineMap>,
        iterator_types: Vec<IteratorType>,
    ) -> Self {
        let results: Vec<TypeHandle> = outputs.iter().map(|v| v.get_type(ctx)).collect();
        let args = inputs
            .iter()
            .chain(&outputs)
            .map(|v| element_of(ctx, v))
            .collect();
        let sizes = OperandSegmentSizesAttr(vec![inputs.len() as u32, outputs.len() as u32]);
        let operands = inputs.into_iter().chain(outputs).collect();
        let op = new_with_body(ctx, |op| Self { op }, results, operands, args);
        op.set_attr_linalg_indexing_maps(ctx, IndexingMapsAttr::new(indexing_maps));
        op.set_attr_linalg_iterator_types(ctx, IteratorTypesAttr::new(iterator_types));
        op.set_attr_linalg_operand_segment_sizes(ctx, sizes);
        op
    }

    /// Builds the operation, populating its body with `body` and verifying it.
    ///
    /// `body` receives the entry block and its arguments, inserts scalar
    /// operations, and returns the values to yield.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification. Allocation is not
    /// transactional: a failure can leave allocated IR in `ctx`.
    pub fn build(
        ctx: &mut Context,
        inputs: Vec<Value>,
        outputs: Vec<Value>,
        indexing_maps: Vec<AffineMap>,
        iterator_types: Vec<IteratorType>,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, inputs, outputs, indexing_maps, iterator_types);
        populate(ctx, op.get_operation(), body)?;
        op.verify(ctx)?;
        Ok(op)
    }

    /// Returns the entry block of the body.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation lacks exactly one single-block region.
    pub fn body(&self, ctx: &Context) -> Result<Ptr<BasicBlock>> {
        body(ctx, self.get_operation())
    }

    fn split(&self, ctx: &Context) -> Result<(usize, usize)> {
        let sizes = self.get_attr_linalg_operand_segment_sizes(ctx);
        segments(ctx, self.get_operation(), sizes.as_deref())
    }

    /// Returns the `ins` operands.
    ///
    /// # Errors
    ///
    /// Returns an error when the segment sizes are missing or inconsistent.
    pub fn inputs(&self, ctx: &Context) -> Result<Vec<Value>> {
        let (inputs, _) = self.split(ctx)?;
        Ok(self
            .get_operation()
            .deref(ctx)
            .operands()
            .take(inputs)
            .collect())
    }

    /// Returns the `outs` operands.
    ///
    /// # Errors
    ///
    /// Returns an error when the segment sizes are missing or inconsistent.
    pub fn outputs(&self, ctx: &Context) -> Result<Vec<Value>> {
        let (inputs, _) = self.split(ctx)?;
        Ok(self
            .get_operation()
            .deref(ctx)
            .operands()
            .skip(inputs)
            .collect())
    }

    /// Returns the indexing maps, or an empty list when absent.
    pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
        self.get_attr_linalg_indexing_maps(ctx)
            .map(|a| a.maps.clone())
            .unwrap_or_default()
    }

    /// Returns the iterator types, or an empty list when absent.
    pub fn iterator_types(&self, ctx: &Context) -> Vec<IteratorType> {
        self.get_attr_linalg_iterator_types(ctx)
            .map(|a| a.values.clone())
            .unwrap_or_default()
    }

    /// Sets the documentation string.
    pub fn set_doc(&self, ctx: &Context, doc: String) {
        self.set_attr_linalg_doc(ctx, StringAttr::new(doc));
    }

    /// Returns the documentation string, if any.
    pub fn doc(&self, ctx: &Context) -> Option<String> {
        self.get_attr_linalg_doc(ctx).map(|a| a.as_str().to_owned())
    }

    /// Sets the external library function implementing this operation.
    pub fn set_library_call(&self, ctx: &Context, name: String) {
        self.set_attr_linalg_library_call(ctx, StringAttr::new(name));
    }

    /// Returns the external library function, if any.
    pub fn library_call(&self, ctx: &Context) -> Option<String> {
        self.get_attr_linalg_library_call(ctx)
            .map(|a| a.as_str().to_owned())
    }
}

impl Verify for GenericOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let (inputs, outputs) = self.split(ctx)?;
        if outputs == 0 {
            return Err(fail(ctx, op, "expected at least one outs operand"));
        }
        verify_tensor_outputs(ctx, op, inputs..inputs + outputs)?;
        let loops = self
            .get_attr_linalg_iterator_types(ctx)
            .ok_or_else(|| fail(ctx, op, "missing iterator types"))?
            .values
            .len();
        let mut shapes = Vec::with_capacity(inputs + outputs);
        let mut elements = Vec::with_capacity(inputs + outputs);
        for ty in operand_types(ctx, op) {
            let (dims, element) = ranked_or_scalar(ctx, ty)
                .ok_or_else(|| fail(ctx, op, "operands must be ranked tensors or scalars"))?;
            shapes.push(dims);
            elements.push(element);
        }
        verify_indexing(ctx, op, &self.indexing_maps(ctx), &shapes, loops)?;
        verify_body(ctx, op, &elements, &elements[inputs..])
    }
}

/// Checks block argument types and the `linalg.yield` terminator's types.
fn verify_body(
    ctx: &Context,
    op: Ptr<Operation>,
    args: &[TypeHandle],
    yields: &[TypeHandle],
) -> Result<()> {
    let block = body(ctx, op)?;
    if block_arg_types(ctx, block) != args {
        return Err(fail(
            ctx,
            op,
            "block arguments must be the operand element types",
        ));
    }
    let yielded =
        terminator_operands::<YieldOp>(ctx, op, block, "body must terminate with linalg.yield")?;
    if yielded != yields {
        return Err(fail(
            ctx,
            op,
            "yielded values must have the outs element types",
        ));
    }
    Ok(())
}

/// Elementwise application of a scalar body to same-shaped inputs.
///
/// Operands are the `ins` tensors followed by one `init` tensor of the same
/// shape. The `mapper` body receives one element per input followed by the
/// `init` element, and yields one value of the `init` element type.
#[pliron_op(
    name = "linalg.map",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct MapOp;

impl MapOp {
    /// Allocates the operation with an empty entry block.
    pub fn new(ctx: &mut Context, inputs: Vec<Value>, init: Value) -> Self {
        let results = vec![init.get_type(ctx)];
        let args = inputs
            .iter()
            .chain([&init])
            .map(|v| element_of(ctx, v))
            .collect();
        let operands = inputs.into_iter().chain([init]).collect();
        new_with_body(ctx, |op| Self { op }, results, operands, args)
    }

    /// Builds the operation, populating its body with `body` and verifying it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    pub fn build(
        ctx: &mut Context,
        inputs: Vec<Value>,
        init: Value,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, inputs, init);
        populate(ctx, op.get_operation(), body)?;
        op.verify(ctx)?;
        Ok(op)
    }

    /// Returns the entry block of the `mapper` body.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation lacks exactly one single-block region.
    pub fn body(&self, ctx: &Context) -> Result<Ptr<BasicBlock>> {
        body(ctx, self.get_operation())
    }

    /// Returns the `ins` operands.
    pub fn inputs(&self, ctx: &Context) -> Vec<Value> {
        let ir = self.get_operation().deref(ctx);
        let n = ir.get_num_operands().saturating_sub(1);
        ir.operands().take(n).collect()
    }

    /// Returns the `init` operand.
    pub fn init(&self, ctx: &Context) -> Option<Value> {
        self.get_operation().deref(ctx).operands().last()
    }
}

impl Verify for MapOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let operands = operand_types(ctx, op);
        let Some(inputs) = operands.len().checked_sub(1) else {
            return Err(fail(ctx, op, "expected an init operand"));
        };
        verify_tensor_outputs(ctx, op, inputs..inputs + 1)?;
        let mut elements = Vec::with_capacity(operands.len());
        let mut shapes = Vec::with_capacity(operands.len());
        for &ty in &operands {
            let (dims, element) =
                ranked(ctx, ty).ok_or_else(|| fail(ctx, op, "operands must be ranked tensors"))?;
            shapes.push(dims);
            elements.push(element);
        }
        if shapes[..inputs].iter().any(|s| *s != shapes[inputs]) {
            return Err(fail(ctx, op, "input shapes must equal the init shape"));
        }
        verify_body(ctx, op, &elements, &elements[inputs..])
    }
}

/// Reduction of inputs along `dimensions` into same-count inits.
///
/// Operands are `N` input tensors of one shape followed by `N` init tensors
/// whose shape is the input shape without `dimensions`. The `combiner` body
/// receives the input elements followed by the init elements and yields one
/// value per init.
#[pliron_op(
    name = "linalg.reduce",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig"
)]
pub struct ReduceOp;
crate::pliron_arith::shared_attr!(
    ReduceOp,
    super::ATTR_KEY_LINALG_DIMENSIONS,
    I64ArrayAttr,
    get_attr_linalg_dimensions,
    set_attr_linalg_dimensions
);

impl ReduceOp {
    /// Allocates the operation with an empty entry block.
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        inits: Vec<Value>,
        dimensions: Vec<i64>,
    ) -> Self {
        let results = inits.iter().map(|v| v.get_type(ctx)).collect();
        let args = inputs
            .iter()
            .chain(&inits)
            .map(|v| element_of(ctx, v))
            .collect();
        let operands = inputs.into_iter().chain(inits).collect();
        let op = new_with_body(ctx, |op| Self { op }, results, operands, args);
        op.set_attr_linalg_dimensions(ctx, I64ArrayAttr::new(dimensions));
        op
    }

    /// Builds the operation, populating its body with `body` and verifying it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    pub fn build(
        ctx: &mut Context,
        inputs: Vec<Value>,
        inits: Vec<Value>,
        dimensions: Vec<i64>,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, inputs, inits, dimensions);
        populate(ctx, op.get_operation(), body)?;
        op.verify(ctx)?;
        Ok(op)
    }

    /// Returns the entry block of the `combiner` body.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation lacks exactly one single-block region.
    pub fn body(&self, ctx: &Context) -> Result<Ptr<BasicBlock>> {
        body(ctx, self.get_operation())
    }

    /// Returns the input operands.
    pub fn inputs(&self, ctx: &Context) -> Vec<Value> {
        let ir = self.get_operation().deref(ctx);
        let n = ir.get_num_operands() / 2;
        ir.operands().take(n).collect()
    }

    /// Returns the init operands.
    pub fn inits(&self, ctx: &Context) -> Vec<Value> {
        let ir = self.get_operation().deref(ctx);
        let n = ir.get_num_operands() / 2;
        ir.operands().skip(n).collect()
    }

    /// Returns the reduced dimensions, or an empty list when absent.
    pub fn dimensions(&self, ctx: &Context) -> Vec<i64> {
        self.get_attr_linalg_dimensions(ctx)
            .map(|a| a.values.clone())
            .unwrap_or_default()
    }
}

impl Verify for ReduceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let operands = operand_types(ctx, op);
        if operands.is_empty() || !operands.len().is_multiple_of(2) {
            return Err(fail(
                ctx,
                op,
                "expected equal, nonzero numbers of inputs and inits",
            ));
        }
        let n = operands.len() / 2;
        verify_tensor_outputs(ctx, op, n..2 * n)?;
        let mut shapes = Vec::with_capacity(operands.len());
        let mut elements = Vec::with_capacity(operands.len());
        for &ty in &operands {
            let (dims, element) =
                ranked(ctx, ty).ok_or_else(|| fail(ctx, op, "operands must be ranked tensors"))?;
            shapes.push(dims);
            elements.push(element);
        }
        let (ins, inits) = shapes.split_at(n);
        if ins.iter().any(|s| *s != ins[0]) || inits.iter().any(|s| *s != inits[0]) {
            return Err(fail(
                ctx,
                op,
                "inputs must share one shape and inits another",
            ));
        }
        let dims = self
            .get_attr_linalg_dimensions(ctx)
            .ok_or_else(|| fail(ctx, op, "missing dimensions"))?
            .values
            .clone();
        let rank = ins[0].len() as i64;
        if dims.windows(2).any(|w| w[0] >= w[1]) || dims.iter().any(|&d| d < 0 || d >= rank) {
            return Err(fail(
                ctx,
                op,
                "dimensions must be strictly increasing and in range",
            ));
        }
        let kept: Vec<i64> = ins[0]
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(&(*i as i64)))
            .map(|(_, &d)| d)
            .collect();
        if kept != inits[0] {
            return Err(fail(
                ctx,
                op,
                "init shape must be the input shape without the reduced dimensions",
            ));
        }
        verify_body(ctx, op, &elements, &elements[n..])
    }
}

/// Terminator of a structured-operation body, yielding one value per output.
#[pliron_op(
    name = "linalg.yield",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [
        pliron::builtin::op_interfaces::NResultsInterface<0>,
        pliron::builtin::op_interfaces::IsTerminatorInterface
    ]
)]
pub struct YieldOp;

impl Verify for YieldOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let ir = op.deref(ctx);
        if ir.get_num_results() != 0 || ir.num_regions() != 0 || ir.get_num_successors() != 0 {
            drop(ir);
            return Err(fail(
                ctx,
                op,
                "yield must have no results, regions, or successors",
            ));
        }
        Ok(())
    }
}

impl YieldOp {
    /// Allocates the terminator. The enclosing operation verifies the yielded types.
    pub fn new(ctx: &mut Context, values: Vec<Value>) -> Self {
        Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], values, vec![], 0),
        }
    }
}

/// Returns the loop count of a region-carrying structured operation.
fn loop_count(ctx: &Context, parent: Ptr<Operation>) -> Option<usize> {
    let op = Operation::get_op_dyn(parent, ctx);
    let rank = |v: Option<Value>| {
        v.and_then(|v| ranked(ctx, v.get_type(ctx)))
            .map(|(d, _)| d.len())
    };
    if let Some(generic) = op.downcast_ref::<GenericOp>() {
        Some(generic.iterator_types(ctx).len())
    } else if let Some(map) = op.downcast_ref::<MapOp>() {
        rank(map.init(ctx))
    } else if let Some(reduce) = op.downcast_ref::<ReduceOp>() {
        rank(reduce.inputs(ctx).into_iter().next())
    } else {
        None
    }
}

/// Current index of iteration-space dimension `dim` inside a structured-op body.
#[pliron_op(
    name = "linalg.index",
    format = "attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface],
    attributes = (linalg_index_dim: IndexAttr)
)]
pub struct IndexOp;

impl IndexOp {
    /// Allocates the operation with an `index` result.
    pub fn new(ctx: &mut Context, dim: u32) -> Self {
        let index = shaped::index(ctx);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![index],
                vec![],
                vec![],
                0,
            ),
        };
        op.set_attr_linalg_index_dim(
            ctx,
            IndexAttr {
                value: i64::from(dim),
            },
        );
        op
    }

    /// Returns the queried dimension, or `None` when the attribute is absent.
    pub fn dim(&self, ctx: &Context) -> Option<i64> {
        self.get_attr_linalg_index_dim(ctx).map(|a| a.value)
    }

    /// Returns the result.
    pub fn result(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for IndexOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        {
            let ir = op.deref(ctx);
            if ir.get_num_operands() != 0 || ir.get_num_results() != 1 || ir.num_regions() != 0 {
                return Err(fail(
                    ctx,
                    op,
                    "expected no operands, one result, and no regions",
                ));
            }
        }
        if !shaped::is_index(ctx, self.result(ctx).get_type(ctx)) {
            return Err(fail(ctx, op, "result must be index"));
        }
        let dim = self
            .dim(ctx)
            .ok_or_else(|| fail(ctx, op, "missing dimension"))?;
        if dim < 0 {
            return Err(fail(ctx, op, "dimension must be nonnegative"));
        }
        if let Some(loops) = parent_op(ctx, op).and_then(|p| loop_count(ctx, p))
            && dim >= loops as i64
        {
            return Err(fail(ctx, op, "dimension is outside the iteration space"));
        }
        Ok(())
    }
}
