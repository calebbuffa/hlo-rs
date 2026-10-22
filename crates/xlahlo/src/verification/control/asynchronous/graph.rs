use super::*;

pub(super) fn direct(value: Value) -> Option<Ptr<Operation>> {
    match value.defining_entity() {
        DefiningEntity::Op(op) => Some(op),
        _ => None,
    }
}

pub(super) fn target(ctx: &Context, op: Ptr<Operation>) -> Result<Option<Ref<'_, StringAttr>>> {
    optional_attr(ctx, op, crate::attributes::keys::TARGET)
}

pub(super) fn intermediary(ctx: &Context, op: Ptr<Operation>) -> Result<bool> {
    Ok(matches!(
        opcode(ctx, op),
        Some(
            Opcode::Tuple
                | Opcode::GetTupleElement
                | Opcode::OptimizationBarrier
                | Opcode::Copy
                | Opcode::Parameter
                | Opcode::While
        )
    ) || opcode(ctx, op) == Some(Opcode::CustomCall)
        && target(ctx, op)?.is_some_and(|s| {
            matches!(
                s.as_str(),
                "Sharding"
                    | "LocalToGlobalShape"
                    | "GlobalToLocalShape"
                    | "xla.sdy.LocalToGlobalShape"
                    | "xla.sdy.GlobalToLocalShape"
                    | "xla.sdy.FuncResultSharding"
            )
        }))
}

pub(super) type ComputationPair = (Ptr<Operation>, Ptr<Operation>);

pub(super) struct Graph<'a> {
    pub(super) ctx: &'a Context,
    pub(super) payloads: MetadataPayloads,
    pub(super) signatures: &'a HashMap<Ptr<Operation>, Signature>,
    pub(super) owners: HashMap<Ptr<Operation>, Ptr<Operation>>,
    pub(super) roots: HashMap<Ptr<Operation>, Value>,
    pub(super) users: HashMap<Value, Vec<Ptr<Operation>>>,
    pub(super) callees: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>>,
    pub(super) callers: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>>,
    pub(super) instructions: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>>,
    pub(super) producer_cache: RefCell<HashMap<Value, Option<Ptr<Operation>>>>,
    pub(super) wrapped_cache: RefCell<HashMap<Ptr<Operation>, Ptr<Operation>>>,
    pub(super) equality_cache: RefCell<HashMap<ComputationPair, bool>>,
    pub(super) start_cache: RefCell<HashMap<Ptr<Operation>, Ptr<Operation>>>,
}

impl Graph<'_> {
    pub(super) fn backend_config(&self, operation: Ptr<Operation>) -> Result<Bytes> {
        let payload = ir::backend_config_payload(self.ctx, operation)?;
        let legacy = optional_attr::<StringAttr>(
            self.ctx,
            operation,
            crate::attributes::keys::BACKEND_CONFIG,
        )?
        .map_or_else(Bytes::default, |value| {
            Bytes(value.as_str().as_bytes().to_vec())
        });
        ir::resolve_backend_config(payload.as_ref(), &legacy, &self.payloads)
    }

    pub(super) fn computations_equal(
        &self,
        left: Ptr<Operation>,
        right: Ptr<Operation>,
    ) -> Result<bool> {
        if left == right {
            return Ok(true);
        }
        if let Some(&equal) = self.equality_cache.borrow().get(&(left, right)) {
            return Ok(equal);
        }
        let equal = self.computations_equal_uncached(left, right)?;
        self.equality_cache
            .borrow_mut()
            .insert((left, right), equal);
        Ok(equal)
    }

    pub(super) fn computations_equal_uncached(
        &self,
        left: Ptr<Operation>,
        right: Ptr<Operation>,
    ) -> Result<bool> {
        let ctx = self.ctx;
        let mut computations = vec![(left, right)];
        let mut seen_computations = HashSet::new();
        let mut seen_values = HashSet::new();
        while let Some((left, right)) = computations.pop() {
            if left == right || !seen_computations.insert((left, right)) {
                continue;
            }
            if execution_thread(ctx, left, crate::attributes::keys::EXECUTION_THREAD)?
                .as_deref()
                .unwrap_or("main")
                != execution_thread(ctx, right, crate::attributes::keys::EXECUTION_THREAD)?
                    .as_deref()
                    .unwrap_or("main")
            {
                return Ok(false);
            }
            let left_signature = self.signatures.get(&left).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing wrapped computation signature".into()
                })
            })?;
            let right_signature = self.signatures.get(&right).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing wrapped computation signature".into()
                })
            })?;
            if left_signature.0.len() != right_signature.0.len() {
                return Ok(false);
            }
            let left_root = self.roots.get(&left).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing wrapped root".into()
                })
            })?;
            let right_root = self.roots.get(&right).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing wrapped root".into()
                })
            })?;
            let mut values = vec![(*left_root, *right_root)];
            while let Some((left, right)) = values.pop() {
                if left == right || !seen_values.insert((left, right)) {
                    continue;
                }
                if !shape::equal_including_layout(ctx, left.get_type(ctx), right.get_type(ctx)) {
                    return Ok(false);
                }
                if let (DefiningEntity::Block(a), DefiningEntity::Block(b)) =
                    (left.defining_entity(), right.defining_entity())
                {
                    if a.deref(ctx).arguments().position(|v| v == left)
                        != b.deref(ctx).arguments().position(|v| v == right)
                    {
                        return Ok(false);
                    }
                    continue;
                }
                let (Some(a), Some(b)) = (direct(left), direct(right)) else {
                    return Ok(false);
                };
                let (Some(kind), Some(other)) = (opcode(ctx, a), opcode(ctx, b)) else {
                    return Ok(false);
                };
                if kind != other {
                    return Ok(false);
                }
                let (a_data, b_data) = (a.deref(ctx), b.deref(ctx));
                if a_data.get_num_operands() != b_data.get_num_operands() {
                    return Ok(false);
                }
                if matches!(
                    kind,
                    Opcode::Send
                        | Opcode::SendDone
                        | Opcode::Recv
                        | Opcode::RecvDone
                        | Opcode::AllGather
                        | Opcode::AllGatherStart
                        | Opcode::AllReduce
                        | Opcode::AllReduceStart
                        | Opcode::ReduceScatter
                        | Opcode::CollectiveReduce
                        | Opcode::AllToAll
                        | Opcode::RaggedAllToAll
                        | Opcode::CollectiveBroadcast
                        | Opcode::CollectivePermute
                        | Opcode::CollectivePermuteStart
                ) && channel(ctx, a)? != channel(ctx, b)?
                {
                    return Ok(false);
                }
                if self.backend_config(a)? != self.backend_config(b)? {
                    return Ok(false);
                }
                let a_accuracy = optional_attr::<ResultAccuracy>(
                    ctx,
                    a,
                    crate::attributes::keys::RESULT_ACCURACY,
                )?;
                let b_accuracy = optional_attr::<ResultAccuracy>(
                    ctx,
                    b,
                    crate::attributes::keys::RESULT_ACCURACY,
                )?;
                if !a_accuracy
                    .as_deref()
                    .unwrap_or(&ResultAccuracy::Default)
                    .equal(b_accuracy.as_deref().unwrap_or(&ResultAccuracy::Default))
                {
                    return Ok(false);
                }
                if matches!(kind, Opcode::Constant | Opcode::CustomCall) {
                    match (
                        optional_attr::<Literal>(ctx, a, crate::attributes::keys::LITERAL)?,
                        optional_attr::<Literal>(ctx, b, crate::attributes::keys::LITERAL)?,
                    ) {
                        (Some(a), Some(b)) if a.equal_logical_bits(ctx, &b)? => {}
                        (None, None) => {}
                        _ => return Ok(false),
                    }
                }
                if matches!(
                    kind,
                    Opcode::CustomCall | Opcode::Fusion | Opcode::AsyncStart | Opcode::AsyncUpdate
                ) {
                    let a_aliases = optional_attr::<OutputOperandAliases>(
                        ctx,
                        a,
                        crate::attributes::keys::ALIASES,
                    )?;
                    let b_aliases = optional_attr::<OutputOperandAliases>(
                        ctx,
                        b,
                        crate::attributes::keys::ALIASES,
                    )?;
                    if a_aliases
                        .as_deref()
                        .map(|aliases| aliases.0.as_slice())
                        .unwrap_or_default()
                        != b_aliases
                            .as_deref()
                            .map(|aliases| aliases.0.as_slice())
                            .unwrap_or_default()
                    {
                        return Ok(false);
                    }
                }
                if kind == Opcode::CustomCall {
                    if super::custom_call_api_version(ctx, a)?
                        != super::custom_call_api_version(ctx, b)?
                        || super::custom_call_schedule(ctx, a)?
                            != super::custom_call_schedule(ctx, b)?
                        || super::custom_call_padding_kind(ctx, a)?
                            != super::custom_call_padding_kind(ctx, b)?
                    {
                        return Ok(false);
                    }
                    let empty_precision = PrecisionConfig {
                        operand_precision: OperandPrecision(vec![]),
                        algorithm: Algorithm::Unset,
                    };
                    let a_precision = optional_attr::<PrecisionConfig>(
                        ctx,
                        a,
                        crate::attributes::keys::PRECISION,
                    )?;
                    let b_precision = optional_attr::<PrecisionConfig>(
                        ctx,
                        b,
                        crate::attributes::keys::PRECISION,
                    )?;
                    if a_precision.as_deref().unwrap_or(&empty_precision)
                        != b_precision.as_deref().unwrap_or(&empty_precision)
                    {
                        return Ok(false);
                    }
                    for key in [
                        crate::attributes::keys::FEATURE_GROUPS,
                        crate::attributes::keys::BATCH_GROUPS,
                    ] {
                        let a = optional_attr::<NumericalInteger>(ctx, a, key)?;
                        let b = optional_attr::<NumericalInteger>(ctx, b, key)?;
                        if a.as_deref().map_or(1, |groups| groups.0)
                            != b.as_deref().map_or(1, |groups| groups.0)
                        {
                            return Ok(false);
                        }
                    }
                }
                if kind == Opcode::Domain {
                    for key in [
                        crate::attributes::keys::DOMAIN_ENTRY_SHARDING,
                        crate::attributes::keys::DOMAIN_EXIT_SHARDING,
                    ] {
                        let left = optional_attr::<Sharding>(ctx, a, key)?;
                        let right = optional_attr::<Sharding>(ctx, b, key)?;
                        let equal = match (left.as_deref(), right.as_deref()) {
                            (None, None) => true,
                            (Some(left), Some(right)) => left.equal_domain_semantics(right)?,
                            _ => false,
                        };
                        if !equal {
                            return Ok(false);
                        }
                    }
                }
                let keys: &[&'static str] = match kind {
                    Opcode::Parameter
                    | Opcode::GetTupleElement
                    | Opcode::Iota
                    | Opcode::Concatenate
                    | Opcode::GetDimensionSize
                    | Opcode::SetDimensionSize => &[crate::attributes::keys::INDEX],
                    Opcode::Compare => &[
                        crate::attributes::keys::DIRECTION,
                        crate::attributes::keys::ORDER,
                    ],
                    Opcode::ReducePrecision => &[crate::attributes::keys::PRECISION],
                    Opcode::Broadcast
                    | Opcode::Transpose
                    | Opcode::Reverse
                    | Opcode::Reduce
                    | Opcode::Sort => &[crate::attributes::keys::DIMENSIONS],
                    Opcode::Slice => &[crate::attributes::keys::GEOMETRY],
                    Opcode::DynamicSlice => &[crate::attributes::keys::SLICE_SIZES],
                    Opcode::Pad => &["hlo_config"],
                    Opcode::Gather => &[
                        crate::attributes::keys::DIMENSIONS,
                        crate::attributes::keys::SLICE_SIZES,
                    ],
                    Opcode::Scatter => &[
                        crate::attributes::keys::DIMENSIONS,
                        crate::attributes::keys::FLAGS,
                    ],
                    Opcode::Shuffle => &[crate::attributes::keys::DIMENSIONS, "hlo_mode"],
                    Opcode::Dot => &[
                        crate::attributes::keys::DIMENSIONS,
                        crate::attributes::keys::PRECISION,
                    ],
                    Opcode::RaggedDot | Opcode::ScaledDot => &[
                        crate::attributes::keys::DIMENSIONS,
                        crate::attributes::keys::PRECISION,
                    ],
                    Opcode::Convolution => &[
                        crate::attributes::keys::DIMENSIONS,
                        crate::attributes::keys::WINDOW,
                        crate::attributes::keys::FEATURE_GROUPS,
                        crate::attributes::keys::BATCH_GROUPS,
                        "hlo_kind",
                        crate::attributes::keys::PRECISION,
                    ],
                    Opcode::ReduceWindow | Opcode::SelectAndScatter => {
                        &[crate::attributes::keys::WINDOW]
                    }
                    Opcode::Scan => &[
                        crate::attributes::keys::DIMENSIONS,
                        "hlo_num_carries",
                        "hlo_is_associative",
                    ],
                    Opcode::Fft => &["hlo_kind", "hlo_lengths"],
                    Opcode::TopK => &["hlo_k", "hlo_options"],
                    Opcode::TriangularSolve | Opcode::Cholesky => &["hlo_options"],
                    Opcode::Rng => &["hlo_distribution"],
                    Opcode::RngBitGenerator => &["hlo_algorithm"],
                    Opcode::Outfeed => &["hlo_outfeed_shape"],
                    Opcode::CustomCall => &[
                        crate::attributes::keys::TARGET,
                        crate::attributes::keys::WINDOW,
                        crate::attributes::keys::DIMENSIONS,
                    ],
                    Opcode::Fusion => &["hlo_fusion_kind"],
                    Opcode::AllGather
                    | Opcode::AllGatherStart
                    | Opcode::AllToAll
                    | Opcode::ReduceScatter => &[crate::attributes::keys::DIMENSIONS],
                    Opcode::CollectivePermute | Opcode::CollectivePermuteStart => {
                        &["hlo_pairs", "hlo_dynamic_slice_sizes"]
                    }
                    _ => &[],
                };
                for &key in keys {
                    let key = Identifier::new(key);
                    if a_data.attributes.0.get(&key) != b_data.attributes.0.get(&key) {
                        return Ok(false);
                    }
                    if kind == Opcode::CustomCall
                        && flag(ctx, a, crate::attributes::keys::CONSTRAIN_LAYOUT)?
                    {
                        let key = Identifier::new(crate::attributes::keys::OPERAND_LAYOUTS);
                        if a_data.attributes.0.get(&key) != b_data.attributes.0.get(&key) {
                            return Ok(false);
                        }
                    }
                }
                if kind == Opcode::Gather
                    && required::<IndexFlags>(ctx, a, crate::attributes::keys::FLAGS)?.sorted
                        != required::<IndexFlags>(ctx, b, crate::attributes::keys::FLAGS)?.sorted
                {
                    return Ok(false);
                }
                let feed_key = match kind {
                    Opcode::Infeed => Some("hlo_infeed_config"),
                    Opcode::Outfeed => Some("hlo_outfeed_config"),
                    _ => None,
                };
                if let Some(key) = feed_key
                    && feed_config(ctx, a, key)? != feed_config(ctx, b, key)?
                {
                    return Ok(false);
                }
                if matches!(kind, Opcode::Dot | Opcode::Convolution) {
                    let empty_sparsity = SparsityConfig::default();
                    let a_sparsity =
                        optional_attr::<SparsityConfig>(ctx, a, crate::attributes::keys::SPARSITY)?;
                    let b_sparsity =
                        optional_attr::<SparsityConfig>(ctx, b, crate::attributes::keys::SPARSITY)?;
                    if a_sparsity.as_deref().unwrap_or(&empty_sparsity)
                        != b_sparsity.as_deref().unwrap_or(&empty_sparsity)
                    {
                        return Ok(false);
                    }
                    let empty_scaling = BlockScalingConfig::default();
                    let a_scaling = optional_attr::<BlockScalingConfig>(
                        ctx,
                        a,
                        crate::attributes::keys::BLOCK_SCALING,
                    )?;
                    let b_scaling = optional_attr::<BlockScalingConfig>(
                        ctx,
                        b,
                        crate::attributes::keys::BLOCK_SCALING,
                    )?;
                    if a_scaling.as_deref().unwrap_or(&empty_scaling)
                        != b_scaling.as_deref().unwrap_or(&empty_scaling)
                    {
                        return Ok(false);
                    }
                }
                if matches!(
                    kind,
                    Opcode::BatchNormTraining | Opcode::BatchNormInference | Opcode::BatchNormGrad
                ) {
                    let a_options = required::<BatchNormOptions>(ctx, a, "hlo_options")?;
                    let b_options = required::<BatchNormOptions>(ctx, b, "hlo_options")?;
                    if a_options.feature_index != b_options.feature_index
                        || f32::from_bits(a_options.epsilon_bits)
                            != f32::from_bits(b_options.epsilon_bits)
                    {
                        return Ok(false);
                    }
                }
                if matches!(kind, Opcode::Sort | Opcode::TopK)
                    && crate::verification::numerical::sort_stability(ctx, a)?
                        != crate::verification::numerical::sort_stability(ctx, b)?
                {
                    return Ok(false);
                }
                if kind == Opcode::Scan
                    && crate::verification::numerical::scan_direction(ctx, a)?
                        != crate::verification::numerical::scan_direction(ctx, b)?
                {
                    return Ok(false);
                }
                if kind == Opcode::Reshape
                    && crate::verification::movement::reshape_inferred_dimension(ctx, a)?
                        != crate::verification::movement::reshape_inferred_dimension(ctx, b)?
                {
                    return Ok(false);
                }
                if kind == Opcode::CopyStart
                    && copy_prefetch_index(ctx, a)? != copy_prefetch_index(ctx, b)?
                {
                    return Ok(false);
                }
                if kind == Opcode::RngGetAndUpdateState
                    && super::rng_delta(ctx, a)? != super::rng_delta(ctx, b)?
                {
                    return Ok(false);
                }
                let flags: &[&'static str] = match kind {
                    Opcode::CustomCall => &[
                        crate::attributes::keys::HAS_SIDE_EFFECT,
                        crate::attributes::keys::CONSTRAIN_LAYOUT,
                    ],
                    Opcode::Send | Opcode::SendDone | Opcode::Recv | Opcode::RecvDone => {
                        &["hlo_host_transfer"]
                    }
                    Opcode::AllGather
                    | Opcode::AllGatherStart
                    | Opcode::AllReduce
                    | Opcode::AllReduceStart
                    | Opcode::ReduceScatter => &[
                        crate::attributes::keys::USE_GLOBAL_DEVICE_IDS,
                        crate::attributes::keys::CONSTRAIN_LAYOUT,
                    ],
                    Opcode::CollectiveReduce => &[
                        crate::attributes::keys::USE_GLOBAL_DEVICE_IDS,
                        crate::attributes::keys::CONSTRAIN_LAYOUT,
                        "hlo_has_dynamic_root",
                    ],
                    Opcode::AllToAll | Opcode::CollectiveBroadcast => {
                        &[crate::attributes::keys::CONSTRAIN_LAYOUT]
                    }
                    _ => &[],
                };
                for &key in flags {
                    if flag(ctx, a, key)? != flag(ctx, b, key)? {
                        return Ok(false);
                    }
                }
                if matches!(
                    kind,
                    Opcode::Send | Opcode::SendDone | Opcode::Recv | Opcode::RecvDone
                ) && !same_frontend_attributes(ctx, a, b)?
                {
                    return Ok(false);
                }
                if matches!(
                    kind,
                    Opcode::AllGather
                        | Opcode::AllGatherStart
                        | Opcode::AllReduce
                        | Opcode::AllReduceStart
                        | Opcode::AllToAll
                        | Opcode::ReduceScatter
                        | Opcode::RaggedAllToAll
                        | Opcode::CollectiveBroadcast
                        | Opcode::CollectiveReduce
                ) && !super::super::collective::same_groups(ctx, a, b)?
                {
                    return Ok(false);
                }
                let a_callees = self.callees.get(&a).map(Vec::as_slice).unwrap_or_default();
                let b_callees = self.callees.get(&b).map(Vec::as_slice).unwrap_or_default();
                if a_callees.len() != b_callees.len() {
                    return Ok(false);
                }
                computations.extend(a_callees.iter().copied().zip(b_callees.iter().copied()));
                values.extend(a_data.operands().zip(b_data.operands()));
                if a_data.num_regions() != b_data.num_regions() {
                    return Ok(false);
                }
                for (a, b) in a_data.regions().zip(b_data.regions()) {
                    let a = crate::verification::single_block(ctx, a)?;
                    let b = crate::verification::single_block(ctx, b)?;
                    let a = a.deref(ctx).get_terminator(ctx).ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Verification {
                            message: "missing region return".into()
                        })
                    })?;
                    let b = b.deref(ctx).get_terminator(ctx).ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Verification {
                            message: "missing region return".into()
                        })
                    })?;
                    require(
                        a.deref(ctx).get_num_operands() == 1
                            && b.deref(ctx).get_num_operands() == 1,
                        "invalid region return",
                    )?;
                    values.push((a.deref(ctx).get_operand(0), b.deref(ctx).get_operand(0)));
                }
            }
        }
        Ok(true)
    }

    pub(super) fn producer(&self, from: Value) -> Result<Option<Ptr<Operation>>> {
        if let Some(&producer) = self.producer_cache.borrow().get(&from) {
            return Ok(producer);
        }
        let producer = self.producer_uncached(from)?;
        self.producer_cache.borrow_mut().insert(from, producer);
        Ok(producer)
    }

    pub(super) fn producer_uncached(&self, from: Value) -> Result<Option<Ptr<Operation>>> {
        enum Work {
            Visit(Value, Vec<usize>),
            Exit(Ptr<Operation>),
        }
        let ctx = self.ctx;
        let mut pending = vec![Work::Visit(from, Vec::new())];
        let mut visited = HashSet::new();
        let mut active = HashSet::new();
        while let Some(work) = pending.pop() {
            let (value, mut index) = match work {
                Work::Visit(value, index) => (value, index),
                Work::Exit(op) => {
                    active.remove(&op);
                    continue;
                }
            };
            if !visited.insert((value, index.clone())) {
                continue;
            }
            let Some(op) = direct(value) else {
                continue;
            };
            if !active.insert(op) {
                continue;
            }
            pending.push(Work::Exit(op));
            let kind = opcode(ctx, op);
            if matches!(
                kind,
                Some(
                    Opcode::AsyncStart
                        | Opcode::AsyncUpdate
                        | Opcode::AllGatherStart
                        | Opcode::AllReduceStart
                        | Opcode::CollectivePermuteStart
                )
            ) {
                return Ok(Some(op));
            }
            let operands: Vec<_> = op.deref(ctx).operands().collect();
            match kind {
                Some(Opcode::GetTupleElement) => {
                    index.push(
                        required::<Index>(ctx, op, crate::attributes::keys::INDEX)?.0 as usize,
                    );
                    if let Some(&operand) = operands.first() {
                        pending.push(Work::Visit(operand, index));
                    }
                }
                Some(Opcode::Tuple) => {
                    if let Some(position) = index.pop() {
                        if let Some(&operand) = operands.get(position) {
                            pending.push(Work::Visit(operand, index));
                        }
                    } else {
                        pending.extend(operands.into_iter().map(|v| Work::Visit(v, index.clone())));
                    }
                }
                Some(Opcode::OptimizationBarrier | Opcode::Copy | Opcode::Domain) => {
                    if let Some(&operand) = operands.first() {
                        pending.push(Work::Visit(operand, index));
                    }
                }
                Some(Opcode::CustomCall) if intermediary(ctx, op)? => {
                    if let Some(&operand) = operands.first() {
                        pending.push(Work::Visit(operand, index));
                    }
                }
                Some(Opcode::While) => {
                    if let Some(&operand) = operands.first() {
                        pending.push(Work::Visit(operand, index.clone()));
                    }
                    if let Some(callee) = self.callees.get(&op).and_then(|c| c.first())
                        && let Some(&root) = self.roots.get(callee)
                    {
                        pending.push(Work::Visit(root, index));
                    }
                }
                Some(Opcode::Parameter) => {
                    if let Some(owner) = self.owners.get(&op)
                        && let Some(callers) = self.callers.get(owner)
                        && callers.len() == 1
                        && opcode(ctx, callers[0]) == Some(Opcode::While)
                        && self.callees.get(&callers[0]).and_then(|c| c.first()) == Some(owner)
                    {
                        if let Some(&root) = self.roots.get(owner) {
                            pending.push(Work::Visit(root, index.clone()));
                        }
                        if callers[0].deref(ctx).get_num_operands() > 0 {
                            pending.push(Work::Visit(callers[0].deref(ctx).get_operand(0), index));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(None)
    }

    pub(super) fn wrapped(&self, from: Ptr<Operation>) -> Result<Ptr<Operation>> {
        if let Some(&callee) = self.wrapped_cache.borrow().get(&from) {
            return Ok(callee);
        }
        let ctx = self.ctx;
        let mut current = from;
        let mut visited = HashSet::new();
        while visited.insert(current) {
            let cached = self.wrapped_cache.borrow().get(&current).copied();
            if let Some(callee) = cached {
                for op in visited {
                    self.wrapped_cache.borrow_mut().insert(op, callee);
                }
                return Ok(callee);
            }
            require(
                matches!(
                    opcode(ctx, current),
                    Some(Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone)
                ),
                "cannot resolve a wrapped computation",
            )?;
            if let Some(callees) = self.callees.get(&current)
                && !callees.is_empty()
            {
                require(
                    callees.len() == 1,
                    "async instruction has multiple wrapped computations",
                )?;
                let callee = callees[0];
                for op in visited {
                    self.wrapped_cache.borrow_mut().insert(op, callee);
                }
                return Ok(callee);
            }
            let data = current.deref(ctx);
            require(data.get_num_operands() > 0, "missing async predecessor")?;
            current = self.producer(data.get_operand(0))?.ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "cannot resolve a wrapped computation".into()
                })
            })?;
        }
        Err(pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "cycle in async chain".into()
        }))
    }

    pub(super) fn chain_start(&self, from: Ptr<Operation>) -> Result<Ptr<Operation>> {
        let mut current = from;
        let mut seen = HashSet::new();
        while seen.insert(current) {
            let cached = self.start_cache.borrow().get(&current).copied();
            if let Some(start) = cached {
                for op in seen {
                    self.start_cache.borrow_mut().insert(op, start);
                }
                return Ok(start);
            }
            if opcode(self.ctx, current) == Some(Opcode::AsyncStart) {
                for op in seen {
                    self.start_cache.borrow_mut().insert(op, current);
                }
                return Ok(current);
            }
            require(
                matches!(
                    opcode(self.ctx, current),
                    Some(Opcode::AsyncUpdate | Opcode::AsyncDone)
                ),
                "async-done has no chain start",
            )?;
            let data = current.deref(self.ctx);
            require(data.get_num_operands() > 0, "missing async predecessor")?;
            current = self.producer(data.get_operand(0))?.ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "operand does not trace to an async producer".into()
                })
            })?;
        }
        Err(pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "async chain contains a cycle".into()
        }))
    }

    pub(super) fn carried(&self, from: Ptr<Operation>) -> Result<bool> {
        let ctx = self.ctx;
        let mut work = vec![from];
        let mut seen = HashSet::new();
        while let Some(op) = work.pop() {
            if !seen.insert(op) {
                continue;
            }
            let data = op.deref(ctx);
            if data.get_num_results() == 0 {
                continue;
            }
            let result = data.get_result(0);
            if let Some(owner) = self.owners.get(&op)
                && self.roots.get(owner) == Some(&result)
            {
                for &caller in self.callers.get(owner).into_iter().flatten() {
                    if opcode(ctx, caller) == Some(Opcode::While)
                        && self
                            .callees
                            .get(&caller)
                            .and_then(|callees| callees.first())
                            == Some(owner)
                    {
                        return Ok(true);
                    }
                    if matches!(
                        opcode(ctx, caller),
                        Some(Opcode::Call | Opcode::Conditional)
                    ) {
                        work.push(caller);
                    }
                }
            }
            for &user in self.users.get(&result).into_iter().flatten() {
                match opcode(ctx, user) {
                    Some(Opcode::While) => return Ok(true),
                    Some(
                        Opcode::AsyncUpdate
                        | Opcode::Tuple
                        | Opcode::GetTupleElement
                        | Opcode::OptimizationBarrier
                        | Opcode::Domain
                        | Opcode::Call
                        | Opcode::Conditional,
                    ) => work.push(user),
                    _ => {}
                }
            }
        }
        Ok(false)
    }
}

pub(super) fn build_graph<'a>(
    ctx: &'a Context,
    all: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    computations: &HashMap<crate::ir::ComputationName, Ptr<Operation>>,
    signatures: &'a HashMap<Ptr<Operation>, Signature>,
) -> Result<Graph<'a>> {
    let mut graph = Graph {
        ctx,
        payloads: MetadataPayloads::default(),
        signatures,
        owners: HashMap::new(),
        roots: HashMap::new(),
        users: HashMap::new(),
        callees: HashMap::new(),
        callers: HashMap::new(),
        instructions: HashMap::new(),
        producer_cache: RefCell::new(HashMap::new()),
        wrapped_cache: RefCell::new(HashMap::new()),
        equality_cache: RefCell::new(HashMap::new()),
        start_cache: RefCell::new(HashMap::new()),
    };
    for &(op, owner) in all {
        if let Some(owner) = owner {
            graph.owners.insert(op, owner);
            graph.instructions.entry(owner).or_default().push(op);
            if Operation::is_op::<ReturnOp>(op, ctx)
                && op.deref(ctx).get_parent_op(ctx) == Some(owner)
            {
                require(
                    op.deref(ctx).get_num_operands() == 1,
                    "return requires one operand",
                )?;
                graph.roots.insert(owner, op.deref(ctx).get_operand(0));
            }
        }
        let data = op.deref(ctx);
        let mut unique = HashSet::new();
        for operand in data.operands() {
            if unique.insert(operand) {
                graph.users.entry(operand).or_default().push(op);
            }
        }
        if let Some(names) = optional_attr::<Callees>(ctx, op, crate::attributes::keys::CALLEES)? {
            let mut callees = Vec::new();
            for name in &names.0 {
                let callee = *computations.get(name).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Verification {
                        message: "callee not found".into()
                    })
                })?;
                callees.push(callee);
                graph.callers.entry(callee).or_default().push(op);
            }
            graph.callees.insert(op, callees);
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        ops::control::{GetTupleElementOp, ParameterOp},
        types::{ElementType, TensorType},
    };
    use pliron::op::Op;

    #[test]
    fn producer_tracing_terminates_when_a_projection_cycle_grows_shape_indices() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let f = TensorType::get(&ctx, ElementType::F32, vec![]).unwrap();
        let tuple = shape::tuple(&ctx, vec![f]).unwrap();
        let parameter = ParameterOp::new(&mut ctx, tuple, vec![], Index(0));
        let value = parameter.result(&ctx);
        let projection = GetTupleElementOp::new(&mut ctx, f, vec![value], Index(0));
        let value = projection.result(&ctx);
        Operation::replace_operand(projection.get_operation(), &ctx, 0, value);
        let signatures = HashMap::new();
        let graph = Graph {
            ctx: &ctx,
            payloads: MetadataPayloads::default(),
            signatures: &signatures,
            owners: HashMap::new(),
            roots: HashMap::new(),
            users: HashMap::new(),
            callees: HashMap::new(),
            callers: HashMap::new(),
            instructions: HashMap::new(),
            producer_cache: RefCell::new(HashMap::new()),
            wrapped_cache: RefCell::new(HashMap::new()),
            equality_cache: RefCell::new(HashMap::new()),
            start_cache: RefCell::new(HashMap::new()),
        };
        assert!(graph.producer_uncached(value).unwrap().is_none());
    }
}
