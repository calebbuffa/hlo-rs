//! Instruction-level classification based on OpenXLA's instruction utilities.
//!
//! `hlo_instruction.cc`: IsOpElementwise/IsElementwiseImpl;
//! `hlo_instructions.cc`: Constant, RNG, Map and Fusion overrides;
//! `hlo_instructions.h`: HloChannelInstruction::ClassOf and its subclasses.
use std::collections::{HashMap, HashSet};

use pliron::{
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    result::Result,
    r#type::Typed,
};

use crate::{
    attributes::{Axes, Callees},
    opcode::Opcode,
    ops::{ComputationOp, HloOpInterface, ReturnOp},
    shape,
    verification::{error, optional_attr, require},
};

#[derive(Clone, Copy)]
enum Elementwise {
    Always,
    Never,
    BitcastConvert,
    Map,
    Fusion,
}

// Exhaustive intentionally: new opcodes require an explicit source-grounded choice.
fn classification(opcode: Opcode) -> (Elementwise, bool) {
    use Opcode::*;
    match opcode {
        Abs | Acos | Acosh | Asin | Asinh | Atanh | RoundNearestAfz | RoundNearestEven | Ceil
        | Clz | Convert | Copy | Cos | Cosh | Erf | Exp | Exp2 | Expm1 | Floor | Imag
        | IsFinite | Log | Log1p | Log2 | Not | Negate | PopulationCount | Real
        | ReducePrecision | Rsqrt | Logistic | Sign | Sin | Sinh | Sqrt | Cbrt | Tan | Tanh
        | Add | Atan2 | Compare | Complex | Divide | Maximum | Minimum | Multiply | Mulhi
        | Power | Remainder | Subtract | And | Or | Xor | ShiftLeft | ShiftRightArithmetic
        | ShiftRightLogical | StochasticConvert | Select | Clamp | Constant | Rng => {
            (Elementwise::Always, false)
        }
        BitcastConvert => (Elementwise::BitcastConvert, false),
        Map => (Elementwise::Map, false),
        Fusion => (Elementwise::Fusion, false),
        AllGather
        | AllGatherStart
        | AllReduce
        | AllReduceStart
        | ReduceScatter
        | CollectiveReduce
        | CollectiveBroadcast
        | AllToAll
        | RaggedAllToAll
        | CollectivePermute
        | CollectivePermuteStart
        | Send
        | SendDone
        | Recv
        | RecvDone => (Elementwise::Never, true),
        AddDependency
        | AfterAll
        | AllGatherDone
        | AllReduceDone
        | AsyncDone
        | AsyncStart
        | AsyncUpdate
        | BatchNormGrad
        | BatchNormInference
        | BatchNormTraining
        | Bitcast
        | Broadcast
        | Call
        | Cholesky
        | CollectivePermuteDone
        | Concatenate
        | Conditional
        | Convolution
        | CopyDone
        | CopyStart
        | CustomCall
        | Domain
        | Dot
        | DynamicReshape
        | DynamicSlice
        | DynamicUpdateSlice
        | Fft
        | Gather
        | GetDimensionSize
        | GetTupleElement
        | Infeed
        | Iota
        | OptimizationBarrier
        | Outfeed
        | Pad
        | Parameter
        | PartitionId
        | RaggedDot
        | Reduce
        | ReduceWindow
        | ReplicaId
        | Reshape
        | Reverse
        | RngBitGenerator
        | RngGetAndUpdateState
        | ScaledDot
        | Scan
        | Scatter
        | SelectAndScatter
        | SetDimensionSize
        | Shuffle
        | Slice
        | Sort
        | TopK
        | Transpose
        | TriangularSolve
        | Tuple
        | While => (Elementwise::Never, false),
    }
}

/// Exact pinned HloChannelInstruction class membership, independent of channel ID.
///
/// Collective starts belong to the class; their done instructions do not.
/// Send/Recv and their done instructions do. Generic async and Copy do not.
pub fn is_channel_instruction(opcode: Opcode) -> bool {
    classification(opcode).1
}

fn opcode(ctx: &Context, operation: Ptr<Operation>) -> Result<Opcode> {
    let dynamic = Operation::get_op_dyn(operation, ctx);
    let hlo = op_cast::<dyn HloOpInterface>(&*dynamic)
        .ok_or_else(|| error("instruction classification requires native HLO"))?;
    let opcode = hlo.opcode();
    let ir = operation.deref(ctx);
    require(
        ir.get_num_results() == 1,
        "classified HLO requires one result",
    )?;
    if let Some(arity) = opcode.arity() {
        require(
            ir.get_num_operands() == arity,
            "classified HLO operand arity mismatch",
        )?;
    }
    Ok(opcode)
}

fn symbols(
    ctx: &Context,
    mut operation: Ptr<Operation>,
) -> Result<HashMap<crate::ir::ComputationName, Ptr<Operation>>> {
    while !Operation::is_op::<ModuleOp>(operation, ctx) {
        operation = operation
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("named classification callee requires an attached module"))?;
    }
    let container = operation
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block()
        .ok_or_else(|| error("classification module has no body"))?;
    let mut symbols = HashMap::new();
    for operation in container.deref(ctx).iter(ctx) {
        let computation = Operation::get_op::<ComputationOp>(operation, ctx)
            .ok_or_else(|| error("classification module contains a non-computation"))?;
        let name = computation
            .get_attr_hlo_computation_name(ctx)
            .ok_or_else(|| error("classification computation has no symbol"))?
            .clone();
        require(
            symbols.insert(name, operation).is_none(),
            "duplicate classification computation symbol",
        )?;
    }
    Ok(symbols)
}

fn children(ctx: &Context, operation: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    operation
        .deref(ctx)
        .regions()
        .flat_map(|region| {
            region
                .deref(ctx)
                .iter(ctx)
                .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn body(ctx: &Context, operation: Ptr<Operation>) -> Result<Vec<Ptr<Operation>>> {
    let regions: Vec<_> = operation.deref(ctx).regions().collect();
    require(regions.len() == 1, "classified body requires one region")?;
    let blocks: Vec<_> = regions[0].deref(ctx).iter(ctx).collect();
    require(blocks.len() == 1, "classified body requires one block")?;
    let instructions: Vec<_> = blocks[0].deref(ctx).iter(ctx).collect();
    require(
        instructions
            .last()
            .is_some_and(|&ret| Operation::is_op::<ReturnOp>(ret, ctx)),
        "classified body requires a return",
    )?;
    Ok(instructions)
}

// Validate referenced bodies without interpreting their contents as elementwise:
// Map's override depends on dimensions, not whether its callee is elementwise.
fn validate_references(
    ctx: &Context,
    operation: Ptr<Operation>,
    symbols: &mut Option<HashMap<crate::ir::ComputationName, Ptr<Operation>>>,
    active: &mut HashSet<Ptr<Operation>>,
    complete: &mut HashSet<Ptr<Operation>>,
) -> Result<()> {
    if complete.contains(&operation) {
        return Ok(());
    }
    require(
        active.insert(operation),
        "recursive instruction classification callee",
    )?;
    if Operation::is_op::<ReturnOp>(operation, ctx) {
        require(
            operation.deref(ctx).get_num_operands() == 1,
            "classified body requires one return value",
        )?;
    } else if Operation::is_op::<ComputationOp>(operation, ctx) {
        body(ctx, operation)?;
    } else {
        let opcode = opcode(ctx, operation)?;
        let callees = optional_attr::<Callees>(ctx, operation, "hlo_callees")?
            .map(|callees| callees.0.clone());
        if opcode == Opcode::Map {
            require(
                callees.as_ref().is_some_and(|callees| callees.len() == 1),
                "classified Map requires one callee",
            )?;
        }
        if opcode == Opcode::Fusion {
            body(ctx, operation)?;
            require(
                callees.as_ref().is_none_or(Vec::is_empty),
                "native Fusion classification requires an inline body, not named callees",
            )?;
        }
        if let Some(callees) = callees {
            if symbols.is_none() {
                *symbols = Some(self::symbols(ctx, operation)?);
            }
            for callee in callees {
                let computation = *symbols
                    .as_ref()
                    .and_then(|symbols| symbols.get(&callee))
                    .ok_or_else(|| error("unresolved classification callee"))?;
                validate_references(ctx, computation, symbols, active, complete)?;
            }
        }
    }
    for child in children(ctx, operation) {
        validate_references(ctx, child, symbols, active, complete)?;
    }
    active.remove(&operation);
    complete.insert(operation);
    Ok(())
}

fn classify(ctx: &Context, operation: Ptr<Operation>) -> Result<bool> {
    let kind = classification(opcode(ctx, operation)?).0;
    match kind {
        Elementwise::Always => Ok(true),
        Elementwise::Never => Ok(false),
        Elementwise::BitcastConvert => {
            let ir = operation.deref(ctx);
            let from = shape::element_type(ctx, ir.get_operand(0).get_type(ctx))?;
            let to = shape::element_type(ctx, ir.get_result(0).get_type(ctx))?;
            Ok(from.storage_bit_width() == to.storage_bit_width())
        }
        Elementwise::Map => {
            let dims = optional_attr::<Axes>(ctx, operation, "hlo_dimensions")?
                .map(|dims| dims.0.clone())
                .unwrap_or_default();
            if dims.is_empty() {
                return Ok(true);
            }
            let result = operation.deref(ctx).get_result(0).get_type(ctx);
            Ok(dims.len() == shape::rank(ctx, result)?
                && dims
                    .iter()
                    .enumerate()
                    .all(|(index, &axis)| usize::try_from(axis) == Ok(index)))
        }
        Elementwise::Fusion => {
            let instructions = body(ctx, operation)?;
            let mut elementwise = true;
            for instruction in instructions
                .into_iter()
                .filter(|&instruction| !Operation::is_op::<ReturnOp>(instruction, ctx))
            {
                if opcode(ctx, instruction)? != Opcode::Parameter {
                    elementwise &= classify(ctx, instruction)?;
                }
            }
            Ok(elementwise)
        }
    }
}

/// Pinned `HloInstruction::IsElementwise()` with no operand-index argument.
///
/// Constant and RNG are elementwise; this is not a purity or fusibility query.
/// BitcastConvert compares primitive storage widths (not layout packing).
/// Map uses empty/full-identity dimensions, without classifying its callee.
/// Fusion checks every instruction, including unused ones, skipping parameters.
/// Inline native fusion arguments represent upstream parameter instructions.
///
/// Read-only: does not infer, evaluate literals, or choose a verifier policy.
/// Non-HLO, malformed consumed structure/metadata, unresolved named callees and
/// recursive callee graphs are errors, not a `false` optimization nonmatch.
pub fn is_elementwise(ctx: &Context, operation: Ptr<Operation>) -> Result<bool> {
    opcode(ctx, operation)?;
    validate_references(
        ctx,
        operation,
        &mut None,
        &mut HashSet::new(),
        &mut HashSet::new(),
    )?;
    classify(ctx, operation)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        attributes::{FusionKind, Index, RandomDistribution},
        literal::Literal,
        ops::{
            control::{CallOp, ConstantOp, FusionOp, ParameterOp, RngOp, TupleOp},
            elementwise::{AddOp, BitcastConvertOp, NegateOp, SelectOp},
            movement::{BroadcastOp, CopyOp, DynamicUpdateSliceOp, ReshapeOp, TransposeOp},
            numerical::MapOp,
        },
        shape::Layout,
        types::ElementType,
        verification::{VerifyOptions, verify, verify_with_options},
    };
    use pliron::identifier::Identifier;
    use pliron::{
        basic_block::BasicBlock,
        builtin::{attributes::StringAttr, types::FunctionType},
        op::Op,
        printable::Printable,
        r#type::TypeHandle,
        value::Value,
    };

    fn array(ctx: &Context, element: ElementType, dims: &[i64]) -> TypeHandle {
        shape::with_default_layout(ctx, shape::array(ctx, element, dims.to_vec()).unwrap()).unwrap()
    }

    fn parameter(ctx: &mut Context, ty: TypeHandle) -> (Ptr<Operation>, Value) {
        let op = ParameterOp::new(ctx, ty, vec![], Index(0));
        (op.get_operation(), op.result(ctx))
    }

    fn check(ctx: &Context, op: Ptr<Operation>, expected: bool) {
        let before = op.disp(ctx).to_string();
        assert_eq!(is_elementwise(ctx, op).unwrap(), expected);
        assert_eq!(is_elementwise(ctx, op).unwrap(), expected);
        assert_eq!(op.disp(ctx).to_string(), before);
    }

    fn dispose(ctx: &mut Context, operations: &[Ptr<Operation>]) {
        for &operation in operations {
            Operation::erase(operation, ctx);
        }
        assert!(ctx.is_ir_empty());
    }

    struct MapFixture {
        module: Ptr<Operation>,
        container: Ptr<BasicBlock>,
        scalar_ret: Ptr<Operation>,
        scalar: Value,
        map: MapOp,
    }

    fn mapped(ctx: &mut Context, dims: &[i64]) -> MapFixture {
        crate::register(ctx);
        let scalar_type = array(ctx, ElementType::F32, &[]);
        let ty = array(ctx, ElementType::F32, dims);
        let module = ModuleOp::new(ctx, Identifier::new("classification")).get_operation();
        crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let container = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let signature = FunctionType::get(ctx, vec![scalar_type], vec![scalar_type]).into();
        let callee = ComputationOp::new(ctx, Identifier::new("scalar"), signature);
        callee.get_operation().insert_at_back(container, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(callee.body(ctx), ctx);
        let (p, scalar) = parameter(ctx, scalar_type);
        p.insert_at_back(block, ctx);
        let scalar_ret = ReturnOp::new(ctx, scalar).get_operation();
        scalar_ret.insert_at_back(block, ctx);
        let signature = FunctionType::get(ctx, vec![ty], vec![ty]).into();
        let main = ComputationOp::new(ctx, Identifier::new("main"), signature);
        main.get_operation().insert_at_back(container, ctx);
        let main_block = BasicBlock::new(ctx, None, vec![]);
        main_block.insert_at_back(main.body(ctx), ctx);
        let (p, value) = parameter(ctx, ty);
        p.insert_at_back(main_block, ctx);
        let map = MapOp::new(ctx, ty, vec![value], Callees(vec!["scalar".into()]));
        map.get_operation().insert_at_back(main_block, ctx);
        let result = map.result(ctx);
        ReturnOp::new(ctx, result)
            .get_operation()
            .insert_at_back(main_block, ctx);
        verify(ctx, module).unwrap();
        MapFixture {
            module,
            container,
            scalar_ret,
            scalar,
            map,
        }
    }

    fn fused(
        ctx: &mut Context,
        ty: TypeHandle,
        input: Value,
        kind: FusionKind,
    ) -> (FusionOp, Ptr<BasicBlock>, Ptr<Operation>, Value) {
        let fusion = FusionOp::new(ctx, ty, vec![input], kind);
        let block = BasicBlock::new(ctx, None, vec![ty]);
        block.insert_at_back(fusion.body(ctx), ctx);
        let argument = block.deref(ctx).get_argument(0);
        let ret = ReturnOp::new(ctx, argument).get_operation();
        ret.insert_at_back(block, ctx);
        (fusion, block, ret, argument)
    }

    #[test]
    fn opcode_catalog_elementwise_baseline_is_complete() {
        let names = "abs acos acosh asin asinh atanh round-nearest-afz round-nearest-even ceil count-leading-zeros convert copy cosine cosh erf exponential exp2 exponential-minus-one floor imag is-finite log log-plus-one log2 not negate popcnt real reduce-precision rsqrt logistic sign sine sinh sqrt cbrt tan tanh add atan2 compare complex divide maximum minimum multiply mulhi power remainder subtract and or xor shift-left shift-right-arithmetic shift-right-logical stochastic-convert select clamp constant rng";
        let expected: HashSet<_> = names.split_whitespace().collect();
        let actual: HashSet<_> = Opcode::ALL
            .iter()
            .filter_map(|&op| {
                matches!(classification(op).0, Elementwise::Always).then_some(op.name())
            })
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn channel_membership_exactly_matches_all_pinned_subclasses() {
        let expected: HashSet<_> = [
            Opcode::AllGather,
            Opcode::AllGatherStart,
            Opcode::AllReduce,
            Opcode::AllReduceStart,
            Opcode::ReduceScatter,
            Opcode::CollectiveReduce,
            Opcode::CollectiveBroadcast,
            Opcode::AllToAll,
            Opcode::RaggedAllToAll,
            Opcode::CollectivePermute,
            Opcode::CollectivePermuteStart,
            Opcode::Send,
            Opcode::SendDone,
            Opcode::Recv,
            Opcode::RecvDone,
        ]
        .into_iter()
        .collect();
        let actual: HashSet<_> = Opcode::ALL
            .iter()
            .copied()
            .filter(|&op| is_channel_instruction(op))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn unary_binary_and_ternary_native_operations_are_elementwise() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[2]);
        let pred = array(ctx, ElementType::Pred, &[2]);
        let (p, value) = parameter(ctx, ty);
        let (q, condition) = parameter(ctx, pred);
        let unary = NegateOp::new(ctx, ty, vec![value]);
        let binary = AddOp::new(ctx, ty, vec![value, value]);
        let select = SelectOp::new(ctx, ty, vec![condition, value, value]);
        for op in [
            unary.get_operation(),
            binary.get_operation(),
            select.get_operation(),
        ] {
            check(ctx, op, true);
        }
        dispose(
            ctx,
            &[
                select.get_operation(),
                binary.get_operation(),
                unary.get_operation(),
                q,
                p,
            ],
        );
    }

    #[test]
    fn copy_is_elementwise_but_reshape_and_transpose_are_not() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[2, 2]);
        let (p, value) = parameter(ctx, ty);
        let copy = CopyOp::new(ctx, ty, vec![value]);
        let reshape = ReshapeOp::new(ctx, ty, vec![value]);
        let transpose = TransposeOp::new(ctx, ty, vec![value], Axes(vec![0, 1]));
        check(ctx, copy.get_operation(), true);
        check(ctx, reshape.get_operation(), false);
        check(ctx, transpose.get_operation(), false);
        dispose(
            ctx,
            &[
                copy.get_operation(),
                reshape.get_operation(),
                transpose.get_operation(),
                p,
            ],
        );
    }

    #[test]
    fn broadcast_and_parameter_are_not_instruction_elementwise() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let scalar = array(ctx, ElementType::F32, &[]);
        let array = array(ctx, ElementType::F32, &[3]);
        let (p, value) = parameter(ctx, scalar);
        let broadcast = BroadcastOp::new(ctx, array, vec![value], Axes(vec![]));
        check(ctx, p, false);
        check(ctx, broadcast.get_operation(), false);
        dispose(ctx, &[broadcast.get_operation(), p]);
    }

    #[test]
    fn dynamic_update_slice_is_false_without_an_operand_index() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::S32, &[]);
        let (p, value) = parameter(ctx, ty);
        let update = DynamicUpdateSliceOp::new(ctx, ty, vec![value, value]);
        check(ctx, update.get_operation(), false);
        dispose(ctx, &[update.get_operation(), p]);
    }

    #[test]
    fn constant_and_rng_override_opcode_classification() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[]);
        let literal = Literal::from_bits(ctx, ty, &[0]).unwrap();
        let c = ConstantOp::new(ctx, ty, vec![], literal);
        let value = c.result(ctx);
        let rng = RngOp::new(ctx, ty, vec![value, value], RandomDistribution::Uniform);
        check(ctx, c.get_operation(), true);
        check(ctx, rng.get_operation(), true);
        assert!(
            crate::analysis::effects::has_local_side_effect(ctx, rng.get_operation(), true)
                .unwrap()
        );
        dispose(ctx, &[rng.get_operation(), c.get_operation()]);
    }

    fn bitcast(
        ctx: &mut Context,
        from: ElementType,
        to: ElementType,
    ) -> (Ptr<Operation>, BitcastConvertOp) {
        crate::register(ctx);
        let from = array(ctx, from, &[]);
        let to = array(ctx, to, &[]);
        let (p, value) = parameter(ctx, from);
        (p, BitcastConvertOp::new(ctx, to, vec![value]))
    }

    #[test]
    fn bitcast_classification_depends_only_on_primitive_storage_width() {
        let ctx = &mut Context::new();
        let storage_width = |element: ElementType| match element {
            ElementType::Pred | ElementType::F6E3M2FN | ElementType::F6E2M3FN => 8,
            _ => element.bit_width(),
        };
        for &from in ElementType::ALL {
            for &to in ElementType::ALL {
                let (p, cast) = bitcast(ctx, from, to);
                let actual = is_elementwise(ctx, cast.get_operation()).unwrap();
                assert_eq!(
                    actual,
                    storage_width(from) == storage_width(to),
                    "{from:?} -> {to:?}"
                );
                dispose(ctx, &[cast.get_operation(), p]);
            }
        }
    }

    #[test]
    fn layout_packing_does_not_override_primitive_storage_width() {
        let ctx = &mut Context::new();
        let (p, cast) = bitcast(ctx, ElementType::S32, ElementType::F32);
        let result = cast.result(ctx);
        let packed = shape::with_layout(
            ctx,
            result.get_type(ctx),
            Layout {
                element_size_in_bits: 16,
                ..Layout::descending(0)
            },
        )
        .unwrap();
        result.set_type(ctx, packed);
        check(ctx, cast.get_operation(), true);
        dispose(ctx, &[cast.get_operation(), p]);
    }

    #[test]
    fn map_is_elementwise_only_for_absent_empty_or_full_identity_dimensions() {
        let ctx = &mut Context::new();
        for dimensions in [vec![], vec![2], vec![2, 3]] {
            let f = mapped(ctx, &dimensions);
            check(ctx, f.map.get_operation(), true);
            for axes in [
                vec![],
                (0..dimensions.len() as i64).collect(),
                vec![0],
                vec![1, 0],
                vec![0, 0],
                vec![-1, 1],
            ] {
                let expected =
                    axes.is_empty() || axes == (0..dimensions.len() as i64).collect::<Vec<_>>();
                f.map
                    .get_operation()
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_dimensions"), Axes(axes.clone()));
                assert_eq!(
                    is_elementwise(ctx, f.map.get_operation()).unwrap(),
                    expected,
                    "shape={dimensions:?}, axes={axes:?}"
                );
            }
            dispose(ctx, &[f.module]);
        }
    }

    #[test]
    fn map_does_not_require_elementwise_callee_instructions() {
        let ctx = &mut Context::new();
        let f = mapped(ctx, &[2]);
        let ty = f.scalar.get_type(ctx);
        let reshape = ReshapeOp::new(ctx, ty, vec![f.scalar]);
        reshape.get_operation().insert_before(ctx, f.scalar_ret);
        Operation::replace_operand(f.scalar_ret, ctx, 0, reshape.result(ctx));
        verify(ctx, f.module).unwrap();
        check(ctx, reshape.get_operation(), false);
        check(ctx, f.map.get_operation(), true);
        dispose(ctx, &[f.module]);
    }

    #[test]
    fn map_mistyped_dimensions_are_errors() {
        let ctx = &mut Context::new();
        let f = mapped(ctx, &[2]);
        f.map
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_dimensions"), Index(0));
        assert!(is_elementwise(ctx, f.map.get_operation()).is_err());
        dispose(ctx, &[f.module]);
    }

    #[test]
    fn map_missing_mistyped_and_empty_callee_lists_are_errors() {
        let ctx = &mut Context::new();
        let f = mapped(ctx, &[2]);
        f.map
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .remove(&Identifier::new("hlo_callees"));
        assert!(is_elementwise(ctx, f.map.get_operation()).is_err());
        f.map
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_callees"), Index(0));
        assert!(is_elementwise(ctx, f.map.get_operation()).is_err());
        f.map
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_callees"), Callees(vec![]));
        assert!(is_elementwise(ctx, f.map.get_operation()).is_err());
        dispose(ctx, &[f.module]);
    }

    #[test]
    fn map_unknown_and_ambiguous_symbols_are_errors() {
        let ctx = &mut Context::new();
        let f = mapped(ctx, &[2]);
        f.map.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_callees"),
            Callees(vec!["missing".into()]),
        );
        assert!(is_elementwise(ctx, f.map.get_operation()).is_err());
        f.map.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_callees"),
            Callees(vec!["scalar".into()]),
        );
        let scalar_type = f.scalar.get_type(ctx);
        let signature = FunctionType::get(ctx, vec![scalar_type], vec![scalar_type]).into();
        let duplicate = ComputationOp::new(ctx, Identifier::new("scalar"), signature);
        duplicate.get_operation().insert_at_back(f.container, ctx);
        assert!(is_elementwise(ctx, f.map.get_operation()).is_err());
        dispose(ctx, &[f.module]);
    }

    #[test]
    fn recursive_map_callee_graph_is_an_error_not_false() {
        let ctx = &mut Context::new();
        let f = mapped(ctx, &[]);
        let ty = f.scalar.get_type(ctx);
        let recursive = MapOp::new(ctx, ty, vec![f.scalar], Callees(vec!["scalar".into()]));
        recursive.get_operation().insert_before(ctx, f.scalar_ret);
        Operation::replace_operand(f.scalar_ret, ctx, 0, recursive.result(ctx));
        assert!(is_elementwise(ctx, f.map.get_operation()).is_err());
        dispose(ctx, &[f.module]);
    }

    #[test]
    fn detached_map_cannot_silently_resolve_named_callees() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[]);
        let (p, value) = parameter(ctx, ty);
        let map = MapOp::new(ctx, ty, vec![value], Callees(vec!["scalar".into()]));
        assert!(is_elementwise(ctx, map.get_operation()).is_err());
        dispose(ctx, &[map.get_operation(), p]);
    }

    #[test]
    fn fusion_identity_and_elementwise_body_are_elementwise_for_all_kinds() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[2]);
        let (p, value) = parameter(ctx, ty);
        for kind in [
            FusionKind::Loop,
            FusionKind::Input,
            FusionKind::Output,
            FusionKind::Custom,
        ] {
            let (fusion, _, ret, arg) = fused(ctx, ty, value, kind);
            check(ctx, fusion.get_operation(), true);
            let add = AddOp::new(ctx, ty, vec![arg, arg]);
            add.get_operation().insert_before(ctx, ret);
            Operation::replace_operand(ret, ctx, 0, add.result(ctx));
            check(ctx, fusion.get_operation(), true);
            Operation::erase(fusion.get_operation(), ctx);
        }
        dispose(ctx, &[p]);
    }

    #[test]
    fn unused_non_elementwise_fusion_instruction_blocks_classification() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[2]);
        let (p, value) = parameter(ctx, ty);
        let (fusion, _, ret, arg) = fused(ctx, ty, value, FusionKind::Loop);
        let reshape = ReshapeOp::new(ctx, ty, vec![arg]);
        reshape.get_operation().insert_before(ctx, ret);
        check(ctx, fusion.get_operation(), false);
        dispose(ctx, &[fusion.get_operation(), p]);
    }

    #[test]
    fn fusion_tuple_root_is_not_elementwise_even_when_children_are() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[2]);
        let (p, value) = parameter(ctx, ty);
        let (fusion, _, ret, arg) = fused(ctx, ty, value, FusionKind::Loop);
        let tuple_type = shape::tuple(ctx, vec![ty]).unwrap();
        let tuple = TupleOp::new(ctx, tuple_type, vec![arg]);
        tuple.get_operation().insert_before(ctx, ret);
        Operation::replace_operand(ret, ctx, 0, tuple.result(ctx));
        let result = fusion.result(ctx);
        result.set_type(ctx, tuple_type);
        check(ctx, fusion.get_operation(), false);
        dispose(ctx, &[fusion.get_operation(), p]);
    }

    #[test]
    fn nested_fusion_classification_recurses_into_every_inline_body() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[2]);
        let (p, value) = parameter(ctx, ty);
        let (outer, _, ret, arg) = fused(ctx, ty, value, FusionKind::Loop);
        let (inner, _, inner_ret, inner_arg) = fused(ctx, ty, arg, FusionKind::Loop);
        inner.get_operation().insert_before(ctx, ret);
        Operation::replace_operand(ret, ctx, 0, inner.result(ctx));
        check(ctx, outer.get_operation(), true);
        let reshape = ReshapeOp::new(ctx, ty, vec![inner_arg]);
        reshape.get_operation().insert_before(ctx, inner_ret);
        check(ctx, outer.get_operation(), false);
        dispose(ctx, &[outer.get_operation(), p]);
    }

    #[test]
    fn fusion_uses_bitcast_storage_width_override() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::S32, &[]);
        let target = array(ctx, ElementType::S16, &[]);
        let (p, value) = parameter(ctx, ty);
        let (fusion, _, ret, arg) = fused(ctx, ty, value, FusionKind::Loop);
        let cast = BitcastConvertOp::new(ctx, target, vec![arg]);
        cast.get_operation().insert_before(ctx, ret);
        check(ctx, fusion.get_operation(), false);
        dispose(ctx, &[fusion.get_operation(), p]);
    }

    #[test]
    fn malformed_later_fusion_instruction_is_not_hidden_by_an_earlier_false() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[2]);
        let (p, value) = parameter(ctx, ty);
        let (fusion, _, ret, arg) = fused(ctx, ty, value, FusionKind::Loop);
        let reshape = ReshapeOp::new(ctx, ty, vec![arg]);
        reshape.get_operation().insert_before(ctx, ret);
        let malformed = AddOp::new(ctx, ty, vec![]);
        malformed.get_operation().insert_before(ctx, ret);
        assert!(is_elementwise(ctx, fusion.get_operation()).is_err());
        dispose(ctx, &[fusion.get_operation(), p]);
    }

    #[test]
    fn fusion_named_callees_cannot_replace_native_inline_ownership() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[]);
        let (p, value) = parameter(ctx, ty);
        let (fusion, _, _, _) = fused(ctx, ty, value, FusionKind::Loop);
        fusion
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_callees"), Callees(vec!["body".into()]));
        assert!(is_elementwise(ctx, fusion.get_operation()).is_err());
        dispose(ctx, &[fusion.get_operation(), p]);
    }

    #[test]
    fn malformed_fusion_body_is_an_explicit_error() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = array(ctx, ElementType::F32, &[]);
        let (p, value) = parameter(ctx, ty);
        let fusion = FusionOp::new(ctx, ty, vec![value], FusionKind::Loop);
        assert!(is_elementwise(ctx, fusion.get_operation()).is_err());
        dispose(ctx, &[fusion.get_operation(), p]);
    }

    #[test]
    fn non_hlo_and_malformed_arity_are_errors() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let module = ModuleOp::new(ctx, Identifier::new("not_hlo")).get_operation();
        assert!(is_elementwise(ctx, module).is_err());
        let ty = array(ctx, ElementType::F32, &[]);
        let add = AddOp::new(ctx, ty, vec![]);
        assert!(is_elementwise(ctx, add.get_operation()).is_err());
        dispose(ctx, &[add.get_operation(), module]);
    }

    #[test]
    fn malformed_referenced_computation_body_is_an_error() {
        let ctx = &mut Context::new();
        let f = mapped(ctx, &[2]);
        Operation::erase(f.scalar_ret, ctx);
        assert!(is_elementwise(ctx, f.map.get_operation()).is_err());
        dispose(ctx, &[f.module]);
    }

    #[test]
    fn shared_callee_diamond_is_not_mistaken_for_recursion() {
        let ctx = &mut Context::new();
        let f = mapped(ctx, &[2]);
        let ty = f.scalar.get_type(ctx);
        let signature = FunctionType::get(ctx, vec![ty], vec![ty]).into();
        let leaf = ComputationOp::new(ctx, Identifier::new("leaf"), signature);
        leaf.get_operation().insert_at_back(f.container, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(leaf.body(ctx), ctx);
        let (p, value) = parameter(ctx, ty);
        p.insert_at_back(block, ctx);
        ReturnOp::new(ctx, value)
            .get_operation()
            .insert_at_back(block, ctx);
        for _ in 0..2 {
            let call = CallOp::new(ctx, ty, vec![f.scalar], Callees(vec!["leaf".into()]));
            call.get_operation().insert_before(ctx, f.scalar_ret);
            check(ctx, call.get_operation(), false);
        }
        check(ctx, f.map.get_operation(), true);
        dispose(ctx, &[f.module]);
    }

    #[test]
    fn non_array_bitcast_element_types_are_errors() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = shape::tuple(ctx, vec![]).unwrap();
        let (p, value) = parameter(ctx, ty);
        let cast = BitcastConvertOp::new(ctx, ty, vec![value]);
        assert!(is_elementwise(ctx, cast.get_operation()).is_err());
        dispose(ctx, &[cast.get_operation(), p]);
    }

    #[test]
    fn classification_preserves_verified_module_and_metadata_exactly() {
        let ctx = &mut Context::new();
        let f = mapped(ctx, &[2]);
        f.map.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("test_metadata"),
            StringAttr::new("opaque".into()),
        );
        let before = f.module.disp(ctx).to_string();
        verify_with_options(
            ctx,
            f.module,
            VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            },
        )
        .unwrap();
        check(ctx, f.map.get_operation(), true);
        assert_eq!(f.module.disp(ctx).to_string(), before);
        verify_with_options(
            ctx,
            f.module,
            VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            },
        )
        .unwrap();
        dispose(ctx, &[f.module]);
    }
}
