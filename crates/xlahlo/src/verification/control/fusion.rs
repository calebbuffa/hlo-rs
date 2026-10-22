use super::*;

pub(crate) fn fusion(ctx: &Context, fusion: &FusionOp, options: VerifyOptions) -> Result<()> {
    super::verify_result_accuracy(ctx, fusion.get_operation(), Opcode::Fusion)?;
    let data = fusion.get_operation().deref(ctx);
    require(
        data.get_num_results() == 1
            && data.num_regions() == 1
            && data.get_num_successors() == 0
            && fusion.get_attr_hlo_fusion_kind(ctx).is_some(),
        "invalid fusion structure",
    )?;
    let block = super::single_block(ctx, fusion.body(ctx))?;
    require(
        block.deref(ctx).get_num_arguments() <= data.get_num_operands(),
        "fusion has more parameters than operands",
    )?;
    require(
        block
            .deref(ctx)
            .arguments()
            .map(|arg| arg.get_type(ctx))
            .zip(data.operand_types(ctx))
            .all(|(a, b)| options.same_shapes(ctx, a, b)),
        "fusion arguments mismatch",
    )?;
    drop(data);
    let result_type = fusion.result(ctx).get_type(ctx);
    require(
        options.same_shapes(ctx, result_type, super::returned(ctx, block)?),
        "fusion return shape mismatch",
    )?;
    let instructions: Vec<_> = block.deref(ctx).iter(ctx).collect();
    let mut used = HashSet::new();
    let mut root = None;
    for &instruction in &instructions {
        let data = instruction.deref(ctx);
        for operand in data.operands() {
            used.insert(operand);
        }
        if Operation::is_op::<ReturnOp>(instruction, ctx) {
            require(
                data.get_num_operands() == 1,
                "fusion return must have one operand",
            )?;
            root = Some(data.get_operand(0));
        }
    }
    let root = root.ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "fusion lacks root".into()
        })
    })?;
    for (index, argument) in block.deref(ctx).arguments().enumerate() {
        require(
            argument == root
                || used.contains(&argument)
                || crate::ir::parameter_has_control_dependency(ctx, block, index)?,
            "unused non-root parameter in fusion",
        )?;
    }
    let root_users = instructions
        .iter()
        .filter(|&&op| !Operation::is_op::<ReturnOp>(op, ctx))
        .any(|&op| op.deref(ctx).operands().any(|operand| operand == root));
    require(!root_users, "fusion root cannot have users")?;
    for &op in &instructions {
        if Operation::is_op::<ReturnOp>(op, ctx) {
            continue;
        }
        let data = op.deref(ctx);
        let live = data
            .results()
            .any(|value| value == root || used.contains(&value));
        require(
            live || has_local_side_effect(ctx, op)?,
            "unused non-root instruction in fusion",
        )?;
    }
    let data = fusion.get_operation().deref(ctx);
    verify_aliases(
        ctx,
        fusion.get_operation(),
        Opcode::Fusion,
        result_type,
        &data.operand_types(ctx).collect::<Vec<_>>(),
        options,
    )?;
    crate::ir::verify_fusion_parameters(ctx, block)
}
