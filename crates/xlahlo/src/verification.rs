use crate::{
    Error, InstructionId, Literal, Module, Opcode, PrimitiveType, Result, Shape,
    inference::{
        infer_binary, infer_dynamic_slice, infer_dynamic_update_slice, infer_reverse, infer_slice,
        infer_transpose, infer_unary,
    },
    proto::hlo::HloInstructionProto,
};

pub(crate) mod asynchronous;
mod buffers;
mod callable;
mod channels;
pub(crate) mod collective;
mod dot;
pub use collective::CollectiveContext;
mod deadlocks;
mod layout;
mod metadata;
mod sharding;
pub use sharding::ShardingContext;

/// Only options with implemented behavior are exposed. More upstream verifier
/// policies remain required before full parity.
#[derive(Clone, Copy, Debug)]
pub struct VerifyOptions {
    pub layout_sensitive: bool,
    /// Relax floating precision only where the pinned verifier permits it.
    pub allow_mixed_precision: bool,
    pub supported_unbounded_dynamic_op: Option<fn(&Module, InstructionId) -> bool>,
    pub instruction_can_change_layout: Option<fn(&Module, InstructionId) -> bool>,
    pub verify_instruction_name_unchanged: bool,
    pub verify_no_host_memory_space: bool,
    pub verify_reshape_is_bitcast: bool,
    /// Permit different target allocation sizes for static bitcasts whose
    /// outer-tile data sizes agree, excluding trailing padding.
    pub allow_bitcast_to_have_different_size: bool,
    pub shape_size: Option<fn(&Shape) -> Result<i64>>,
    pub verify_call_nested_computation_thread_name: bool,
    pub verify_broadcast_dimensions_order: bool,
    pub check_for_collective_deadlocks: bool,
    /// Check explicit collective group membership, participant counts, and
    /// uniformity where required, as in the pinned verifier.
    pub check_replica_groups: bool,
    pub collective_context: CollectiveContext,
    pub sharding_context: ShardingContext,
    /// Use the supplied compatibility contexts instead of nondefault owned
    /// module configuration. By default, owned counts/SPMD mode take precedence.
    pub ignore_module_config: bool,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            layout_sensitive: false,
            allow_mixed_precision: false,
            supported_unbounded_dynamic_op: None,
            instruction_can_change_layout: None,
            verify_instruction_name_unchanged: false,
            verify_no_host_memory_space: false,
            verify_reshape_is_bitcast: false,
            allow_bitcast_to_have_different_size: false,
            shape_size: None,
            verify_call_nested_computation_thread_name: false,
            verify_broadcast_dimensions_order: false,
            check_for_collective_deadlocks: false,
            check_replica_groups: true,
            collective_context: CollectiveContext::default(),
            sharding_context: ShardingContext::default(),
            ignore_module_config: false,
        }
    }
}

impl VerifyOptions {
    fn size_of(self, shape: &Shape) -> Result<i64> {
        match self.shape_size {
            Some(size) => size(shape),
            None if has_unbounded(shape.as_proto()) => Ok(crate::UNBOUNDED_SIZE),
            None => shape.byte_size(-1),
        }
    }
}

impl Module {
    pub fn verify(&self, mut options: VerifyOptions) -> Result<()> {
        let config = self.config();
        if !options.ignore_module_config
            && (config.replica_count != 1
                || config.num_partitions != 1
                || config.use_spmd_partitioning)
        {
            options.collective_context = CollectiveContext {
                replica_count: config.replica_count,
                partition_count: config.num_partitions,
            };
            options.sharding_context.use_spmd_partitioning = config.use_spmd_partitioning;
        }
        self.verify_schedule()?;
        asynchronous::verify_pairs(self)?;
        asynchronous::verify(self, options)?;
        channels::verify(self)?;
        collective::verify_module(self)?;
        callable::verify_ownership(self)?;
        for computation_id in self.computations() {
            let computation = self.computation(computation_id)?;
            let program = self.program_shape(computation_id)?;
            if let Some(declared) = computation.program_shape.as_ref() {
                verify_signature(declared, &program)?;
            }
            for id in self.instructions(computation_id)? {
                let instruction = self.instruction(id)?;
                let opcode: Opcode = instruction.opcode.parse()?;
                let operands: Vec<_> = self
                    .operands(id)?
                    .iter()
                    .map(|&id| self.shape(id))
                    .collect::<Result<_>>()?;
                if opcode.arity().is_some_and(|n| operands.len() != n) {
                    return Err(Error::new(
                        "arity",
                        format!("{}: wrong operand count for {opcode}", instruction.name),
                    ));
                }
                let declared = self.shape(id)?;
                let layout_shape = if declared.element_type() == PrimitiveType::BUFFER {
                    declared.array_view()?
                } else {
                    declared.clone()
                };
                if options.layout_sensitive
                    && layout_shape
                        .layout()
                        .is_some_and(|l| l.minor_to_major.len() != layout_shape.dimensions().len())
                {
                    return Err(Error::new("layout","layout-sensitive verification requires minor-to-major rank to match shape rank").context(&instruction.name));
                }
                if has_unbounded(declared.as_proto())
                    && !options
                        .supported_unbounded_dynamic_op
                        .is_some_and(|supported| supported(self, id))
                {
                    return Err(Error::new(
                        "unbounded",
                        "unbounded dynamism is disabled for this instruction",
                    ));
                }
                if !instruction.called_computation_ids.is_empty()
                    && !matches!(
                        opcode,
                        Opcode::Call
                            | Opcode::While
                            | Opcode::Conditional
                            | Opcode::Map
                            | Opcode::Reduce
                            | Opcode::ReduceWindow
                            | Opcode::Sort
                            | Opcode::Scatter
                            | Opcode::SelectAndScatter
                            | Opcode::Scan
                            | Opcode::Fusion
                            | Opcode::CustomCall
                            | Opcode::AllReduce
                            | Opcode::AllReduceStart
                            | Opcode::ReduceScatter
                            | Opcode::CollectiveReduce
                            | Opcode::AsyncStart
                            | Opcode::AsyncUpdate
                            | Opcode::AsyncDone
                    )
                {
                    return Err(Error::new(
                        "unsupported",
                        format!("callee contracts for {opcode}"),
                    ));
                }
                let has_to_apply = matches!(
                    opcode,
                    Opcode::AllReduce
                        | Opcode::AllReduceStart
                        | Opcode::Call
                        | Opcode::Map
                        | Opcode::Reduce
                        | Opcode::ReduceScatter
                        | Opcode::CollectiveReduce
                        | Opcode::ReduceWindow
                        | Opcode::Scatter
                        | Opcode::Sort
                        | Opcode::Scan
                ) || (opcode == Opcode::CustomCall
                    && instruction.called_computation_ids.len() == 1);
                let mut callees = Vec::new();
                for &wire_id in &instruction.called_computation_ids {
                    let callee_id = self.computation_handle(wire_id);
                    if (matches!(opcode, Opcode::While | Opcode::Conditional | Opcode::Fusion)
                        || (options.verify_call_nested_computation_thread_name && has_to_apply))
                        && crate::ir::execution_thread(
                            &self.computation(callee_id)?.execution_thread,
                        ) != crate::ir::execution_thread(&computation.execution_thread)
                    {
                        return Err(Error::new(
                            "execution-thread",
                            "callee execution thread differs from caller",
                        ));
                    }
                    callees.push(self.program_shape(callee_id)?);
                }
                verify_instruction(
                    instruction,
                    opcode,
                    &declared,
                    &operands,
                    &callees,
                    InstructionContext {
                        options,
                        in_fusion: computation.is_fusion_computation,
                        channel: self.channel_id(id)?,
                    },
                )
                .map_err(|e| e.context(&instruction.name))?;
                if opcode == Opcode::Bitcast {
                    layout::verify_bitcast(self, id, options)
                        .map_err(|e| e.context(&instruction.name))?;
                }
                callable::verify_aliases(
                    self,
                    computation_id,
                    instruction,
                    &declared,
                    &operands,
                    options,
                )
                .map_err(|e| e.context(&instruction.name))?;
                metadata::verify_instruction(instruction, &declared, options)
                    .map_err(|e| e.context(&instruction.name))?;
                layout::verify_instruction(self, id, options)
                    .map_err(|e| e.context(&instruction.name))?;
            }
        }
        if let Some(declared) = self.wire.host_program_shape.as_ref() {
            verify_entry_signature(
                declared,
                &self.program_shape(self.entry()?)?,
                options.layout_sensitive,
            )?;
        }
        metadata::verify_aliases_and_donors(self, options)?;
        sharding::verify(self, options)?;
        buffers::verify(self, options)?;
        if options.check_for_collective_deadlocks && self.as_proto().schedule.is_some() {
            deadlocks::verify(self)?;
        }
        Ok(())
    }

    /// Check instruction ordering and coverage for every live nonfusion
    /// computation in each execution thread represented by the schedule.
    pub fn verify_schedule(&self) -> Result<()> {
        self.validate_structure()?;
        if let Some(schedule) = self.wire.schedule.as_ref() {
            let threads = schedule
                .sequences
                .keys()
                .map(|&id| {
                    self.computation(self.computation_handle(id))
                        .map(|c| crate::ir::execution_thread(&c.execution_thread))
                })
                .collect::<Result<std::collections::BTreeSet<_>>>()?;
            let called = self
                .wire
                .computations
                .iter()
                .flat_map(|c| &c.instructions)
                .flat_map(|i| i.called_computation_ids.iter().copied())
                .collect::<std::collections::BTreeSet<_>>();
            for computation in &self.wire.computations {
                // Upstream defines dead by zero callers, not entry reachability.
                let live = computation.id == self.wire.entry_computation_id
                    || called.contains(&computation.id);
                if live
                    && !computation.is_fusion_computation
                    && threads.contains(crate::ir::execution_thread(&computation.execution_thread))
                    && !schedule.sequences.contains_key(&computation.id)
                {
                    return Err(Error::new(
                        "schedule",
                        format!("missing live computation {}", computation.name),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn has_unbounded(wire: &crate::proto::xla_data::ShapeProto) -> bool {
    let mut pending = vec![wire];
    while let Some(shape) = pending.pop() {
        if shape.logical_dimensions().contains(&crate::UNBOUNDED_SIZE) {
            return true;
        }
        pending.extend(shape.logical_children());
    }
    false
}

fn verify_signature(
    declared: &crate::proto::xla_data::ProgramShapeProto,
    actual: &crate::proto::xla_data::ProgramShapeProto,
) -> Result<()> {
    if declared.parameters.len() != actual.parameters.len() {
        return Err(Error::new("signature", "parameter cardinality mismatch"));
    }
    for (a, b) in declared
        .parameters
        .iter()
        .zip(&actual.parameters)
        .chain(declared.result.iter().zip(actual.result.iter()))
    {
        if !Shape::from_proto(a.clone())?.compatible(&Shape::from_proto(b.clone())?) {
            return Err(Error::new("signature", "program signature shape mismatch"));
        }
    }
    if declared.result.is_none() {
        return Err(Error::new("signature", "missing program result"));
    }
    Ok(())
}

fn verify_entry_signature(
    declared: &crate::proto::xla_data::ProgramShapeProto,
    actual: &crate::proto::xla_data::ProgramShapeProto,
    layout_sensitive: bool,
) -> Result<()> {
    if !layout_sensitive {
        return verify_signature(declared, actual);
    }
    if declared.parameters.len() != actual.parameters.len() {
        return Err(Error::new("signature", "parameter cardinality mismatch"));
    }
    let declared_result = declared
        .result
        .as_ref()
        .ok_or_else(|| Error::new("signature", "missing program result"))?;
    let actual_result = actual
        .result
        .as_ref()
        .ok_or_else(|| Error::new("signature", "missing program result"))?;
    for (declared, actual) in declared
        .parameters
        .iter()
        .zip(&actual.parameters)
        .chain(std::iter::once((declared_result, actual_result)))
    {
        let declared = Shape::from_proto(declared.clone())?;
        let actual = Shape::from_proto(actual.clone())?;
        if !same_shapes(
            &declared,
            &actual,
            LayoutComparison::IgnoreEntryLayoutDifferences,
        ) {
            return Err(Error::new(
                "signature",
                "entry computation layout differs from parameter or result",
            ));
        }
    }
    Ok(())
}

struct InstructionContext {
    options: VerifyOptions,
    in_fusion: bool,
    channel: Option<i64>,
}

fn verify_instruction(
    instruction: &HloInstructionProto,
    opcode: Opcode,
    declared: &Shape,
    operands: &[Shape],
    callees: &[crate::proto::xla_data::ProgramShapeProto],
    context: InstructionContext,
) -> Result<()> {
    use Opcode::*;
    let InstructionContext {
        options,
        in_fusion,
        channel,
    } = context;
    if opcode == AllReduce && channel.is_some_and(|id| id <= 0) {
        return Err(Error::new(
            "channel",
            "all-reduce channel ID must be positive when present",
        ));
    }
    if matches!(opcode, AllReduce | AllReduceStart)
        && instruction.channel_id > 0
        && instruction.all_reduce_id > 0
    {
        return Err(Error::new(
            "channel",
            "both channel_id and legacy all_reduce_id are present",
        ));
    }
    // These handlers do not call upstream CheckShape.
    let checks_mixed_operands = !matches!(opcode, Bitcast | RngBitGenerator | Scan);
    if !options.allow_mixed_precision
        && checks_mixed_operands
        && !matches!(
            opcode,
            Call | Conditional
                | Constant
                | Convolution
                | Dot
                | RaggedDot
                | ScaledDot
                | AllReduce
                | AllReduceStart
                | AllReduceDone
                | AllGather
                | AllGatherStart
                | AllGatherDone
                | AsyncDone
                | AsyncUpdate
                | AsyncStart
                | CollectivePermute
                | CollectivePermuteStart
                | CollectivePermuteDone
                | CopyDone
                | CopyStart
                | CustomCall
                | Domain
                | Fusion
                | GetTupleElement
                | OptimizationBarrier
                | Infeed
                | Outfeed
                | Parameter
                | Recv
                | RecvDone
                | ReducePrecision
                | ReduceWindow
                | Send
                | SendDone
                | Sort
                | Tuple
                | While
        )
    {
        let mut precision = None;
        let mut pending = operands
            .iter()
            .map(|operand| operand.as_proto())
            .collect::<Vec<_>>();
        while let Some(shape) = pending.pop() {
            let element = shape
                .element_type
                .enum_value()
                .map_err(|_| Error::new("type-domain", "unknown primitive type"))?;
            if element.is_floating() {
                if precision.is_some_and(|previous| previous != element) {
                    return Err(Error::new(
                        "mixed-precision",
                        "mixed floating point operand precision is disabled by upstream default",
                    ));
                }
                precision = Some(element);
            }
            pending.extend(shape.logical_children());
        }
    }
    if opcode == Opcode::Compare
        && (!matches!(
            instruction.comparison_direction.as_str(),
            "EQ" | "NE" | "GE" | "GT" | "LE" | "LT"
        ) || !matches!(
            instruction.comparison_order.as_str(),
            "" | "TOTAL" | "WEAK" | "PARTIAL"
        ) || (instruction.comparison_order.is_empty()
            && !matches!(
                instruction.comparison_type.as_str(),
                "" | "FLOAT" | "TOTALORDER" | "SIGNED" | "UNSIGNED"
            )))
    {
        return Err(Error::new(
            "comparison",
            "unknown comparison direction, order or legacy type",
        ));
    }
    if opcode == Opcode::Compare
        && (operands[0].element_type().is_integral()
            || operands[0].element_type() == PrimitiveType::PRED)
    {
        let total = if instruction.comparison_order.is_empty() {
            instruction.comparison_type != "FLOAT"
        } else {
            instruction.comparison_order == "TOTAL"
        };
        if !total {
            return Err(Error::new(
                "comparison",
                "integral and predicate comparisons require TOTAL ordering",
            ));
        }
    }
    let unary = matches!(
        opcode,
        Abs | Acos
            | Acosh
            | Asin
            | Asinh
            | Atanh
            | Cbrt
            | Ceil
            | Clz
            | Cos
            | Cosh
            | Erf
            | Exp
            | Exp2
            | Expm1
            | Floor
            | Imag
            | IsFinite
            | Log
            | Log1p
            | Log2
            | Logistic
            | Negate
            | Not
            | PopulationCount
            | Real
            | RoundNearestAfz
            | RoundNearestEven
            | Rsqrt
            | Sign
            | Sin
            | Sinh
            | Sqrt
            | Tan
            | Tanh
    );
    let binary = matches!(
        opcode,
        Add | And
            | Atan2
            | Complex
            | Compare
            | Divide
            | Maximum
            | Minimum
            | Mulhi
            | Multiply
            | Or
            | Power
            | Remainder
            | ShiftLeft
            | ShiftRightArithmetic
            | ShiftRightLogical
            | Subtract
            | Xor
    );
    if unary || binary {
        for operand in operands {
            if !operand
                .with_element_type(declared.element_type())?
                .compatible(declared)
            {
                return Err(Error::new(
                    "broadcast",
                    "implicit elementwise broadcasting is not permitted in raw HLO",
                ));
            }
        }
    }
    let expected = if unary {
        infer_unary(opcode, &operands[0])?
    } else if binary {
        infer_binary(opcode, &operands[0], &operands[1])?
    } else {
        match opcode {
            CustomCall => {
                callable::verify_custom_call(instruction, declared, operands)?;
                declared.clone()
            }
            Fusion => {
                if callees.len() != 1
                    || !matches!(
                        instruction.fusion_kind.as_str(),
                        "kLoop" | "kInput" | "kOutput" | "kCustom"
                    )
                {
                    return Err(Error::new(
                        "fusion",
                        "fusion requires one computation and a recognized fusion kind",
                    ));
                }
                callable::verify_parameters(operands, &callees[0], options, true)?;
                Shape::from_proto(
                    callees[0]
                        .result
                        .as_ref()
                        .ok_or_else(|| Error::new("fusion", "missing fusion result"))?
                        .clone(),
                )?
            }
            Parameter => {
                if instruction.parameter_number < 0 {
                    return Err(Error::new("parameter", "negative parameter number"));
                }
                return Ok(());
            }
            Constant => Literal::from_proto(
                instruction
                    .literal
                    .as_ref()
                    .ok_or_else(|| Error::new("literal", "constant has no literal"))?
                    .clone(),
            )?
            .shape()
            .clone(),
            Tuple => Shape::tuple(operands.to_vec())?,
            GetTupleElement => operands[0].subshape(&[usize::try_from(instruction.tuple_index)
                .map_err(|_| Error::new("tuple-index", "negative tuple index"))?])?,
            AfterAll => {
                if operands
                    .iter()
                    .any(|s| s.element_type() != PrimitiveType::TOKEN)
                {
                    return Err(Error::new("token", "after-all requires tokens"));
                }
                Shape::token()
            }
            OptimizationBarrier => operands[0].clone(),
            Copy | Domain => operands[0].clone(),
            AllGather
            | AllGatherStart
            | AllReduce
            | AllReduceStart
            | ReduceScatter
            | AllToAll
            | RaggedAllToAll
            | CollectiveReduce
            | CollectiveBroadcast
            | CollectivePermute
            | CollectivePermuteStart => {
                if matches!(
                    opcode,
                    AllReduce | AllReduceStart | ReduceScatter | CollectiveReduce
                ) && callees.len() != 1
                {
                    return Err(Error::new(
                        "callee",
                        "collective reduction requires one reducer",
                    ));
                }
                collective::infer(
                    instruction,
                    opcode,
                    declared,
                    operands,
                    options.collective_context,
                    channel.is_some(),
                    options.check_replica_groups,
                )?
            }
            AllGatherDone => crate::inference::infer_all_gather_done(&operands[0])?,
            AllReduceDone => crate::inference::infer_all_reduce_done(&operands[0])?,
            CollectivePermuteDone => crate::inference::infer_collective_permute_done(&operands[0])?,
            Send => Shape::tuple(vec![
                operands[0].clone(),
                Shape::scalar(PrimitiveType::U32)?.with_default_layout()?,
                Shape::token(),
            ])?,
            SendDone => Shape::token(),
            Recv => Shape::tuple(vec![
                declared.subshape(&[0])?,
                Shape::scalar(PrimitiveType::U32)?.with_default_layout()?,
                Shape::token(),
            ])?,
            RecvDone => Shape::tuple(vec![operands[0].subshape(&[0])?, Shape::token()])?,
            CopyStart => Shape::tuple(vec![
                operands[0].clone(),
                operands[0].clone(),
                Shape::scalar(PrimitiveType::U32)?.with_default_layout()?,
            ])?,
            CopyDone => {
                let destination = operands[0].subshape(&[0])?;
                let source = operands[0].subshape(&[1])?;
                let equal = if options.layout_sensitive {
                    same_shapes(
                        &destination,
                        &source,
                        LayoutComparison::IgnoreMemoryAndSplits,
                    )
                } else {
                    destination.compatible(&source)
                };
                if !equal {
                    return Err(Error::new(
                        "copy",
                        "copy context source and destination differ",
                    ));
                }
                destination
            }
            Bitcast => declared.clone(),
            DynamicReshape => {
                let (operand, sizes) = operands
                    .split_first()
                    .ok_or_else(|| Error::new("arity", "dynamic reshape missing input"))?;
                if operand.element_type() != declared.element_type()
                    || sizes.len() != declared.rank()?
                    || sizes
                        .iter()
                        .any(|shape| shape.element_type() != PrimitiveType::S32)
                {
                    return Err(Error::new(
                        "shape",
                        "dynamic reshape requires one S32 size operand per result dimension",
                    ));
                }
                crate::inference::infer_dynamic_reshape(
                    operand,
                    sizes,
                    (0..declared.rank()?)
                        .map(|axis| declared.dimension(axis))
                        .collect::<Result<Vec<_>>>()?,
                )?
            }
            Convert => crate::inference::infer_convert(&operands[0], declared.element_type())?,
            StochasticConvert => crate::inference::infer_stochastic_convert(
                &operands[0],
                &operands[1],
                declared.element_type(),
            )?,
            ReducePrecision => crate::inference::infer_reduce_precision(
                &operands[0],
                instruction.exponent_bits,
                instruction.mantissa_bits,
            )?,
            Concatenate => {
                if instruction.dimensions.len() != 1 {
                    return Err(Error::new(
                        "dimension",
                        "concatenate requires one dimension",
                    ));
                }
                crate::inference::infer_concatenate(operands, instruction.dimensions[0])?
            }
            Pad => crate::inference::infer_pad(
                &operands[0],
                &operands[1],
                instruction
                    .padding_config
                    .as_ref()
                    .ok_or_else(|| Error::new("padding", "missing padding configuration"))?,
            )?,
            Clamp => crate::inference::infer_clamp(&operands[0], &operands[1], &operands[2])?,
            Select => crate::inference::infer_select(&operands[0], &operands[1], &operands[2])?,
            BitcastConvert => {
                crate::inference::infer_bitcast_convert(&operands[0], declared.element_type())?
            }
            GetDimensionSize => {
                if instruction.dimensions.len() != 1 {
                    return Err(Error::new(
                        "dimension",
                        "get-dimension-size requires exactly one dimension",
                    ));
                }
                crate::inference::infer_get_dimension_size(&operands[0], instruction.dimensions[0])?
            }
            SetDimensionSize => {
                if instruction.dimensions.len() != 1 {
                    return Err(Error::new(
                        "dimension",
                        "set-dimension-size requires exactly one dimension",
                    ));
                }
                crate::inference::infer_set_dimension_size(
                    &operands[0],
                    &operands[1],
                    instruction.dimensions[0],
                )?
            }
            Iota => {
                let array = declared.array_view()?;
                let rank = array.rank()?;
                if rank == 0
                    || instruction.dimensions.len() != 1
                    || !(array.element_type().is_integral()
                        || array.element_type().is_floating()
                        || array.element_type().is_complex())
                {
                    return Err(Error::new(
                        "iota",
                        "iota requires a numerical array and one dimension",
                    ));
                }
                crate::inference::axes(rank, &instruction.dimensions, false)?;
                declared.clone()
            }
            Scan => {
                if instruction.dimensions.len() != 1 || callees.len() != 1 {
                    return Err(Error::new(
                        "scan",
                        "scan requires one dimension and one body computation",
                    ));
                }
                if options.layout_sensitive
                    && instruction.is_associative.enum_value()
                        != Ok(crate::proto::hlo::TriState::TRI_STATE_TRUE)
                {
                    return Err(Error::new(
                        "scan",
                        "non-associative scan must be expanded before layout-sensitive verification",
                    ));
                }
                crate::inference::verify_scan(
                    operands,
                    declared,
                    &callees[0],
                    instruction.num_carries,
                    instruction.dimensions[0],
                    options.layout_sensitive,
                    options.allow_mixed_precision,
                )?;
                declared.clone()
            }
            Shuffle => crate::inference::infer_shuffle(
                &operands[0],
                &instruction.dimensions,
                instruction
                    .shuffle_mode
                    .as_ref()
                    .ok_or_else(|| Error::new("shuffle", "missing shuffle mode"))?,
            )?,
            SelectAndScatter => {
                if callees.len() != 2 {
                    return Err(Error::new(
                        "callee",
                        "select-and-scatter requires select and scatter computations",
                    ));
                }
                crate::inference::infer_select_and_scatter(
                    &operands[0],
                    &operands[1],
                    &operands[2],
                    instruction
                        .window
                        .as_ref()
                        .ok_or_else(|| Error::new("window", "missing select-and-scatter window"))?,
                    &callees[0],
                    &callees[1],
                )?
            }
            ReplicaId | PartitionId => Shape::scalar(PrimitiveType::U32)?.with_default_layout()?,
            AddDependency => {
                if operands[1].element_type() != PrimitiveType::TOKEN {
                    return Err(Error::new(
                        "token",
                        "add-dependency requires a token second operand",
                    ));
                }
                operands[0].clone()
            }
            Infeed => {
                if operands[0].element_type() != PrimitiveType::TOKEN {
                    return Err(Error::new("token", "infeed requires a token operand"));
                }
                let data = declared
                    .as_proto()
                    .tuple_shapes
                    .first()
                    .filter(|_| declared.is_tuple())
                    .ok_or_else(|| {
                        Error::new("shape", "infeed requires a nonempty tuple result")
                    })?;
                Shape::tuple(vec![Shape::from_proto(data.clone())?, Shape::token()])?
            }
            Outfeed => {
                if operands[1].element_type() != PrimitiveType::TOKEN {
                    return Err(Error::new(
                        "token",
                        "outfeed requires a token second operand",
                    ));
                }
                let host_shape = Shape::from_proto(
                    instruction
                        .outfeed_shape
                        .as_ref()
                        .ok_or_else(|| Error::new("shape", "missing outfeed host shape"))?
                        .clone(),
                )?;
                if !host_shape.compatible(&operands[0])
                    || (options.layout_sensitive
                        && !same_shapes(&host_shape, &operands[0], LayoutComparison::Exact))
                {
                    return Err(Error::new(
                        "shape",
                        "outfeed host shape differs from data operand",
                    ));
                }
                Shape::token()
            }
            Rng => {
                if operands.len() != 2 || operands[0].rank()? != 0 || operands[1].rank()? != 0 {
                    return Err(Error::new("arity", "rng requires two scalar operands"));
                }
                let element = operands[0].element_type();
                if operands[1].element_type() != element
                    || (declared.element_type() != element
                        && !(options.allow_mixed_precision
                            && declared.element_type().same_ignoring_fp_precision(element)))
                {
                    return Err(Error::new(
                        "type-domain",
                        "rng operand and result element types differ",
                    ));
                }
                use crate::proto::xla_data::RandomDistribution;
                match instruction.distribution.enum_value() {
                    Ok(RandomDistribution::RNG_UNIFORM)
                        if element.is_floating()
                            || element.is_integral()
                            || element == PrimitiveType::PRED => {}
                    Ok(RandomDistribution::RNG_NORMAL) if element.is_floating() => {}
                    _ => {
                        return Err(Error::new(
                            "rng",
                            "invalid random distribution or element type",
                        ));
                    }
                }
                declared.clone()
            }
            RngBitGenerator => {
                if declared.is_tuple()
                    && (declared.as_proto().tuple_shapes.len() != 2
                        || !Shape::from_proto(declared.as_proto().tuple_shapes[0].clone())?
                            .compatible(&operands[0]))
                {
                    return Err(Error::new(
                        "rng",
                        "rng-bit-generator requires two tuple components and matching state shapes",
                    ));
                }
                declared.clone()
            }
            RngGetAndUpdateState => {
                Shape::array(PrimitiveType::U64, vec![2])?.with_default_layout()?
            }
            Gather => crate::inference::infer_gather(
                &operands[0],
                &operands[1],
                instruction
                    .gather_dimension_numbers
                    .as_ref()
                    .ok_or_else(|| Error::new("dimensions", "missing gather dimensions"))?,
                &instruction.gather_slice_sizes,
            )?,
            Scatter => {
                if callees.len() != 1 {
                    return Err(Error::new(
                        "callee",
                        "scatter requires one update computation",
                    ));
                }
                crate::inference::infer_scatter(
                    operands,
                    instruction
                        .scatter_dimension_numbers
                        .as_ref()
                        .ok_or_else(|| Error::new("dimensions", "missing scatter dimensions"))?,
                    &callees[0],
                )?
            }
            Cholesky => crate::inference::infer_cholesky(&operands[0])?,
            TriangularSolve => crate::inference::infer_triangular_solve(
                &operands[0],
                &operands[1],
                instruction
                    .triangular_solve_options
                    .as_ref()
                    .ok_or_else(|| Error::new("transpose", "missing triangular solve options"))?,
            )?,
            TopK => crate::inference::infer_top_k(&operands[0], instruction.k)?,
            BatchNormTraining | BatchNormInference | BatchNormGrad => {
                crate::inference::infer_batch_norm(opcode, operands, instruction.feature_index)?
            }
            Sort => {
                if callees.len() != 1 || instruction.dimensions.len() != 1 {
                    return Err(Error::new(
                        "sort",
                        "sort requires one dimension and one comparator",
                    ));
                }
                crate::inference::infer_sort(operands, instruction.dimensions[0], &callees[0])?
            }
            Fft => crate::inference::infer_fft(
                &operands[0],
                instruction
                    .fft_type
                    .enum_value()
                    .map_err(|_| Error::new("fft", "unknown FFT kind"))?,
                &instruction.fft_length,
            )?,
            Dot => {
                dot::verify_auxiliary_operands(instruction, operands)?;
                let sparsity = instruction
                    .sparsity_config
                    .as_ref()
                    .cloned()
                    .unwrap_or_default();
                crate::inference::infer_dot_with_sparsity(
                    &operands[0],
                    &operands[1],
                    instruction
                        .dot_dimension_numbers
                        .as_ref()
                        .ok_or_else(|| Error::new("dot", "missing dot dimension numbers"))?,
                    Some(declared.element_type()),
                    &sparsity,
                )?
            }
            RaggedDot => {
                if operands.len() != 3 {
                    return Err(Error::new("arity", "ragged-dot requires three operands"));
                }
                crate::inference::infer_ragged_dot(
                    &operands[0],
                    &operands[1],
                    &operands[2],
                    instruction
                        .ragged_dot_dimension_numbers
                        .as_ref()
                        .ok_or_else(|| {
                            Error::new("ragged-dot", "missing ragged dot dimension numbers")
                        })?,
                    Some(declared.element_type()),
                )?
            }
            ScaledDot => {
                if operands.len() != 4 {
                    return Err(Error::new("arity", "scaled-dot requires four operands"));
                }
                crate::inference::infer_scaled_dot(
                    &operands[0],
                    &operands[1],
                    &operands[2],
                    &operands[3],
                    instruction
                        .dot_dimension_numbers
                        .as_ref()
                        .ok_or_else(|| Error::new("scaled-dot", "missing dot dimension numbers"))?,
                    Some(declared.element_type()),
                )?
            }
            Convolution => {
                dot::verify_auxiliary_operands(instruction, operands)?;
                let sparsity = instruction
                    .sparsity_config
                    .as_ref()
                    .cloned()
                    .unwrap_or_default();
                crate::inference::infer_convolution_with_sparsity(
                    &operands[0],
                    &operands[1],
                    instruction.feature_group_count,
                    instruction.batch_group_count,
                    instruction
                        .window
                        .as_ref()
                        .ok_or_else(|| Error::new("window", "missing convolution window"))?,
                    instruction
                        .convolution_dimension_numbers
                        .as_ref()
                        .ok_or_else(|| {
                            Error::new("dimensions", "missing convolution dimensions")
                        })?,
                    Some(declared.element_type()),
                    &sparsity,
                )?
            }
            Map => {
                if callees.len() != 1 {
                    return Err(Error::new("callee", "map requires one computation"));
                }
                let rank = operands
                    .iter()
                    .map(Shape::rank)
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .max()
                    .unwrap_or(0);
                crate::inference::infer_map(
                    operands,
                    &callees[0],
                    &(0..rank as i64).collect::<Vec<_>>(),
                )?
            }
            Reduce => {
                if callees.len() != 1 {
                    return Err(Error::new("callee", "reduce requires one computation"));
                }
                crate::inference::infer_reduce(operands, &instruction.dimensions, &callees[0])?
            }
            ReduceWindow => {
                if callees.len() != 1 {
                    return Err(Error::new(
                        "callee",
                        "reduce-window requires one computation",
                    ));
                }
                crate::inference::infer_reduce_window(
                    operands,
                    instruction
                        .window
                        .as_ref()
                        .ok_or_else(|| Error::new("window", "missing window configuration"))?,
                    &callees[0],
                )?
            }
            Call => {
                if callees.len() != 1 {
                    return Err(Error::new(
                        "callee",
                        "call requires exactly one computation",
                    ));
                }
                if instruction.is_composite {
                    let attributes = &instruction
                        .frontend_attributes
                        .as_ref()
                        .ok_or_else(|| {
                            Error::new("composite", "composite call requires frontend attributes")
                        })?
                        .map;
                    if attributes
                        .get("composite.name")
                        .is_none_or(|name| name.is_empty())
                        || attributes
                            .get("composite.attributes")
                            .is_some_and(|value| value.is_empty())
                        || attributes.get("composite.version").is_some_and(|value| {
                            value
                                .trim()
                                .parse::<i64>()
                                .map_or(true, |version| version < 0)
                        })
                    {
                        return Err(Error::new(
                            "composite",
                            "invalid composite name, attributes or version",
                        ));
                    }
                }
                callable::verify_parameters(operands, &callees[0], options, false)?;
                crate::inference::infer_call(operands, &callees[0])?
            }
            While => {
                if callees.len() != 2 {
                    return Err(Error::new(
                        "callee",
                        "while requires body and condition computations",
                    ));
                }
                for callee in callees {
                    callable::verify_parameters(operands, callee, options, false)?;
                }
                crate::inference::infer_while(&operands[0], &callees[0], &callees[1])?
            }
            Conditional => {
                let (selector, branches) = operands
                    .split_first()
                    .ok_or_else(|| Error::new("arity", "conditional requires a selector"))?;
                let result = crate::inference::infer_conditional(selector, branches, callees)?;
                for (operand, callee) in branches.iter().zip(callees) {
                    callable::verify_parameters(
                        std::slice::from_ref(operand),
                        callee,
                        options,
                        false,
                    )?;
                    let root = Shape::from_proto(
                        callee
                            .result
                            .as_ref()
                            .ok_or_else(|| Error::new("callee", "missing branch result"))?
                            .clone(),
                    )?;
                    if options.layout_sensitive
                        && !same_shapes(declared, &root, LayoutComparison::Exact)
                    {
                        return Err(Error::new(
                            "result-shape",
                            "conditional differs from a branch result",
                        ));
                    }
                }
                result
            }
            Reverse => infer_reverse(&operands[0], &instruction.dimensions)?,
            Broadcast => {
                let operand = &operands[0];
                if operand.element_type() != declared.element_type()
                    || instruction.dimensions.len() != operand.rank()?
                    || (options.verify_broadcast_dimensions_order
                        && !instruction.dimensions.is_sorted())
                {
                    return Err(Error::new(
                        "broadcast",
                        "broadcast element type, dimensions or required ordering mismatch",
                    ));
                }
                for (axis, &target) in instruction.dimensions.iter().enumerate() {
                    if usize::try_from(target)
                        .ok()
                        .and_then(|target| declared.dimensions().get(target))
                        != operand.dimensions().get(axis)
                    {
                        return Err(Error::new(
                            "broadcast",
                            "broadcast mapped dimension is invalid or differs from operand",
                        ));
                    }
                }
                declared.rank()?;
                declared.clone()
            }
            DynamicSlice => {
                let (operand, indices) = operands
                    .split_first()
                    .ok_or_else(|| Error::new("arity", "dynamic-slice missing operand"))?;
                infer_dynamic_slice(operand, indices, &instruction.dynamic_slice_sizes)?
            }
            AsyncStart | AsyncUpdate | AsyncDone => declared.clone(),
            DynamicUpdateSlice => {
                if operands.len() < 2 {
                    return Err(Error::new(
                        "arity",
                        "dynamic-update-slice requires operand and update",
                    ));
                }
                infer_dynamic_update_slice(&operands[0], &operands[1], &operands[2..])?
            }
            Transpose => infer_transpose(&operands[0], &instruction.dimensions)?,
            Slice => {
                let starts: Vec<_> = instruction
                    .slice_dimensions
                    .iter()
                    .map(|d| d.start)
                    .collect();
                let limits: Vec<_> = instruction
                    .slice_dimensions
                    .iter()
                    .map(|d| d.limit)
                    .collect();
                let strides: Vec<_> = instruction
                    .slice_dimensions
                    .iter()
                    .map(|d| d.stride)
                    .collect();
                infer_slice(&operands[0], &starts, &limits, &strides)?
            }
            Reshape => {
                let operand = operands[0].array_view()?;
                let result = declared.array_view()?;
                if operand.element_type() != result.element_type() {
                    return Err(Error::new(
                        "type-domain",
                        "reshape cannot change element type",
                    ));
                }
                if options.verify_reshape_is_bitcast
                    && !in_fusion
                    && !operands[0].reshape_is_bitcast(declared, false)?
                {
                    return Err(Error::new("reshape", "reshape is not a physical bitcast"));
                }
                if operand.element_count()? != result.element_count()? {
                    return Err(Error::new("reshape", "reshape element counts differ"));
                }
                declared.clone()
            }
            _ => {
                return Err(Error::new(
                    "unsupported",
                    format!("semantic verification for {opcode}"),
                ));
            }
        }
    };
    if !options.allow_mixed_precision && matches!(opcode, Map | Reduce | ReduceWindow) {
        for (operand, parameter) in operands.iter().zip(&callees[0].parameters) {
            if operand.element_type()
                != parameter.element_type.enum_value().map_err(|_| {
                    Error::new("type-domain", "unknown callee parameter element type")
                })?
            {
                return Err(Error::new(
                    "mixed-precision",
                    "callee parameter and operand element types differ",
                ));
            }
        }
    }
    let exact = matches!(
        opcode,
        Bitcast
            | Call
            | Conditional
            | Constant
            | CopyDone
            | CopyStart
            | CustomCall
            | GetTupleElement
            | Infeed
            | Outfeed
            | OptimizationBarrier
            | Parameter
            | Recv
            | RecvDone
            | Send
            | SendDone
            | Tuple
            | While
            | DynamicUpdateSlice
            | Fusion
    );
    let equal = if options.layout_sensitive && opcode == Copy {
        match (declared.layout(), expected.layout()) {
            (Some(output), Some(input))
                if output.memory_space != input.memory_space
                    && (output.memory_space == 5 || input.memory_space == 5) =>
            {
                if output.memory_space == 5 {
                    declared.element_type() == expected.element_type()
                } else {
                    same_shapes(declared, &expected, LayoutComparison::IgnoreMemory)
                }
            }
            _ if options.allow_mixed_precision => {
                declared.compatible_ignoring_fp_precision(&expected)
            }
            _ => declared.compatible(&expected),
        }
    } else if exact && options.layout_sensitive {
        let mode = if matches!(opcode, Infeed | Send | Recv | CopyStart) {
            LayoutComparison::MinorOnly
        } else if opcode == DynamicUpdateSlice && in_fusion {
            LayoutComparison::IgnoreMemoryAndTiles
        } else {
            LayoutComparison::Exact
        };
        same_shapes(declared, &expected, mode)
    } else if !exact && options.allow_mixed_precision {
        declared.compatible_ignoring_fp_precision(&expected)
    } else {
        declared.compatible(&expected)
    };
    if !equal {
        return Err(Error::new(
            "result-shape",
            "declared result differs from inferred shape",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LayoutComparison {
    Exact,
    MinorOnly,
    IgnoreMemory,
    IgnoreMemoryAndTiles,
    IgnoreMemoryAndSplits,
    IgnoreEntryLayoutDifferences,
}

fn same_shapes(a: &Shape, b: &Shape, mode: LayoutComparison) -> bool {
    use crate::proto::xla_data::{LayoutProto, PrimitiveType};

    fn layout_equal(a: &LayoutProto, b: &LayoutProto, mode: LayoutComparison) -> bool {
        if mode == LayoutComparison::MinorOnly {
            return a.minor_to_major == b.minor_to_major;
        }
        let (ignore_memory, ignore_tiles, ignore_splits, ignore_tail) = match mode {
            LayoutComparison::Exact => (false, false, false, false),
            LayoutComparison::MinorOnly => unreachable!(),
            LayoutComparison::IgnoreMemory => (true, false, false, false),
            LayoutComparison::IgnoreMemoryAndTiles => (true, true, false, false),
            LayoutComparison::IgnoreMemoryAndSplits => (true, false, true, false),
            LayoutComparison::IgnoreEntryLayoutDifferences => (true, true, false, true),
        };
        a.minor_to_major == b.minor_to_major
            && (ignore_tiles || a.tiles == b.tiles)
            && (ignore_tail
                || a.tail_padding_alignment_in_elements.max(1)
                    == b.tail_padding_alignment_in_elements.max(1))
            && a.element_size_in_bits == b.element_size_in_bits
            && (ignore_memory || a.memory_space == b.memory_space)
            && a.index_primitive_type == b.index_primitive_type
            && a.pointer_primitive_type == b.pointer_primitive_type
            && (ignore_splits || a.split_configs == b.split_configs)
            && matches!(
                (a.physical_shape.as_ref(), b.physical_shape.as_ref()),
                (None, None) | (Some(_), Some(_))
            )
    }

    let mut pending = vec![(a.as_proto().clone(), b.as_proto().clone(), mode)];
    while let Some((left, right, mode)) = pending.pop() {
        if left.element_type != right.element_type
            || left.logical_dimensions() != right.logical_dimensions()
            || left.logical_children().len() != right.logical_children().len()
            || (0..left.logical_dimensions().len()).any(|axis| {
                left.is_dynamic_dimension
                    .get(axis)
                    .copied()
                    .unwrap_or(false)
                    != right
                        .is_dynamic_dimension
                        .get(axis)
                        .copied()
                        .unwrap_or(false)
            })
        {
            return false;
        }
        if left.element_type.enum_value() == Ok(PrimitiveType::BUFFER) {
            pending.push((
                left.array_storage().into_owned(),
                right.array_storage().into_owned(),
                mode,
            ));
            continue;
        }
        match (left.layout.as_ref(), right.layout.as_ref()) {
            (None, None) => {}
            (Some(left_layout), Some(right_layout)) => {
                if !layout_equal(left_layout, right_layout, mode) {
                    return false;
                }
                if let (Some(left_shape), Some(right_shape)) = (
                    left_layout.physical_shape.as_ref(),
                    right_layout.physical_shape.as_ref(),
                ) {
                    pending.push((
                        left_shape.clone(),
                        right_shape.clone(),
                        LayoutComparison::Exact,
                    ));
                }
            }
            _ => return false,
        }
        pending.extend(
            left.logical_children()
                .iter()
                .zip(right.logical_children())
                .map(|(left, right)| (left.clone(), right.clone(), mode)),
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{InstructionContext, VerifyOptions, verify_instruction};
    use crate::{Opcode, PrimitiveType as P, Shape, proto::hlo::HloInstructionProto};

    #[test]
    fn iota_shape_verification_accepts_buffer_arrays_and_checks_inner_type() {
        for element_type in [P::S32, P::F32, P::C64] {
            let shape = Shape::buffer(Shape::array(element_type, [3]).unwrap()).unwrap();
            let instruction = HloInstructionProto {
                dimensions: vec![0],
                ..Default::default()
            };
            verify_instruction(
                &instruction,
                Opcode::Iota,
                &shape,
                &[],
                &[],
                InstructionContext {
                    options: VerifyOptions::default(),
                    in_fusion: false,
                    channel: None,
                },
            )
            .unwrap();
        }

        let shape = Shape::buffer(Shape::array(P::PRED, [3]).unwrap()).unwrap();
        let instruction = HloInstructionProto {
            dimensions: vec![0],
            ..Default::default()
        };
        assert_eq!(
            verify_instruction(
                &instruction,
                Opcode::Iota,
                &shape,
                &[],
                &[],
                InstructionContext {
                    options: VerifyOptions::default(),
                    in_fusion: false,
                    channel: None,
                },
            )
            .unwrap_err()
            .category,
            "iota"
        );
    }
}
