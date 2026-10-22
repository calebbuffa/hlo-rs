use crate::control_support::*;
use xlahlo::{
    analysis::effects::{SideEffects, has_local_side_effect},
    attributes::{Axes, Bools, OptionalInteger, ReplicaGroups},
    ops::control::{AllGatherOp, CallOp, CustomCallOp},
    pliron::{
        builtin::attributes::StringAttr,
        op::Op,
        pass::{Analysis, AnalysisManager},
    },
    types::ElementType,
};

#[test]
fn local_effect_query_does_not_recurse_into_callees() {
    let mut ctx = context();
    let ty = shape(&ctx, ElementType::F32, &[1]);
    let (worker, block, _) = computation(&mut ctx, "worker", vec![], ty);
    let effect = CustomCallOp::new(&mut ctx, ty, vec![], StringAttr::new("effect".into()));
    attr(
        &ctx,
        effect.get_operation(),
        "hlo_has_side_effect",
        Bools(vec![true]),
    );
    effect.get_operation().insert_at_back(block, &ctx);
    let output = effect.result(&ctx);
    ret(&mut ctx, block, output);
    let (main, block, _) = computation(&mut ctx, "main", vec![], ty);
    let call = CallOp::new(&mut ctx, ty, vec![], callees(&["worker"]));
    call.get_operation().insert_at_back(block, &ctx);
    let output = call.result(&ctx);
    ret(&mut ctx, block, output);
    let module = module(&mut ctx, &[worker, main], "main");
    for sensitive in [false, true] {
        assert!(has_local_side_effect(&ctx, effect.get_operation(), sensitive).unwrap());
        assert!(!has_local_side_effect(&ctx, call.get_operation(), sensitive).unwrap());
    }
    let effects = SideEffects::compute(
        module.get_operation(),
        &ctx,
        &mut AnalysisManager::default(),
    )
    .unwrap();
    assert!(effects.has_side_effect(call.get_operation()).unwrap());
    assert!(has_local_side_effect(&ctx, module.get_operation(), true).is_err());
    attr(
        &ctx,
        effect.get_operation(),
        "hlo_has_side_effect",
        StringAttr::new("true".into()),
    );
    assert!(has_local_side_effect(&ctx, effect.get_operation(), true).is_err());
}

#[test]
fn local_effect_query_respects_explicit_channel_sensitivity() {
    let mut ctx = context();
    let ty = shape(&ctx, ElementType::F32, &[1]);
    let parameter = param(&mut ctx, ty, 0);
    let input = parameter.result(&ctx);
    let gather = AllGatherOp::new(
        &mut ctx,
        ty,
        vec![input],
        Axes(vec![0]),
        ReplicaGroups::default(),
    );
    for channel in [OptionalInteger::Absent, OptionalInteger::Value(1)] {
        for constrained in [false, true] {
            attr(&ctx, gather.get_operation(), "hlo_channel", channel);
            attr(
                &ctx,
                gather.get_operation(),
                "hlo_constrain_layout",
                Bools(vec![constrained]),
            );
            for sensitive in [false, true] {
                assert_eq!(
                    has_local_side_effect(&ctx, gather.get_operation(), sensitive).unwrap(),
                    constrained || matches!(channel, OptionalInteger::Value(_)) && sensitive,
                );
            }
        }
    }
    attr(
        &ctx,
        gather.get_operation(),
        "hlo_constrain_layout",
        Bools(vec![false]),
    );
    attr(
        &ctx,
        gather.get_operation(),
        "hlo_channel",
        StringAttr::new("invalid".into()),
    );
    assert!(has_local_side_effect(&ctx, gather.get_operation(), true).is_err());
}
