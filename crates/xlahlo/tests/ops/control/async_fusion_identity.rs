use crate::control_support::*;
use xlahlo::{
    attributes::{Axes, FusionKind, OutputOperandAliases, OutputOperandAliasing},
    ir::ExecutionThread,
    ops::{
        control::{AsyncDoneOp, AsyncStartOp, FusionOp},
        elementwise::NegateOp,
    },
    pliron::{
        basic_block::BasicBlock, builtin::attributes::StringAttr, common_traits::Verify, op::Op,
    },
    shape,
    types::ElementType,
    verification,
};

#[test]
fn wrapped_fusion_alias_identity_normalizes_empty_but_preserves_real_aliases() {
    let mut ctx = context();
    let data = shape(&ctx, ElementType::F32, &[4]);
    let mut computations = vec![];
    let mut fusions = vec![];
    for name in ["first", "second"] {
        let (comp, block, args) = computation(&mut ctx, name, vec![data], data);
        attr(
            &ctx,
            comp.get_operation(),
            "hlo_execution_thread",
            ExecutionThread("fusion_worker".into()),
        );
        let fusion = FusionOp::new(&mut ctx, data, args, FusionKind::Loop);
        fusion.get_operation().insert_at_back(block, &ctx);
        let body = BasicBlock::new(&mut ctx, None, vec![data]);
        body.insert_at_back(fusion.body(&ctx), &ctx);
        let input = body.deref(&ctx).get_argument(0);
        let negate = NegateOp::new(&mut ctx, data, vec![input]);
        negate.get_operation().insert_at_back(body, &ctx);
        let value = negate.result(&ctx);
        ret(&mut ctx, body, value);
        let value = fusion.result(&ctx);
        ret(&mut ctx, block, value);
        computations.push(comp);
        fusions.push(fusion);
    }
    let (main, block, args) = computation(&mut ctx, "main", vec![data], data);
    let async_context = shape::tuple(
        &ctx,
        vec![
            shape::tuple(&ctx, vec![data]).unwrap(),
            data,
            shape(&ctx, ElementType::U32, &[]),
        ],
    )
    .unwrap();
    let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
    attr(
        &ctx,
        start.get_operation(),
        "hlo_async_execution_thread",
        StringAttr::new("fusion_worker".into()),
    );
    start.get_operation().insert_at_back(block, &ctx);
    let value = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, data, vec![value]);
    attr(
        &ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, &ctx);
    let value = done.result(&ctx);
    ret(&mut ctx, block, value);
    computations.push(main);
    let root = module(&mut ctx, &computations, "main").get_operation();
    verification::verify(&ctx, root).unwrap();
    attr(
        &ctx,
        fusions[1].get_operation(),
        "hlo_aliases",
        OutputOperandAliases(vec![]),
    );
    verification::verify(&ctx, root).unwrap();
    let aliases = OutputOperandAliases(vec![OutputOperandAliasing {
        output_shape_index: Axes(vec![]),
        operand_index: 0,
        operand_shape_index: Axes(vec![]),
    }]);
    attr(
        &ctx,
        fusions[1].get_operation(),
        "hlo_aliases",
        aliases.clone(),
    );
    fusions[1].verify(&ctx).unwrap();
    assert!(verification::verify(&ctx, root).is_err());
    attr(&ctx, fusions[0].get_operation(), "hlo_aliases", aliases);
    verification::verify(&ctx, root).unwrap();
    attr(
        &ctx,
        fusions[1].get_operation(),
        "hlo_aliases",
        StringAttr::new("invalid".into()),
    );
    assert!(fusions[1].verify(&ctx).is_err());
    assert!(verification::verify(&ctx, root).is_err());
    for fusion in fusions {
        attr(
            &ctx,
            fusion.get_operation(),
            "hlo_aliases",
            OutputOperandAliases(vec![]),
        );
    }
    verification::verify(&ctx, root).unwrap();
}
