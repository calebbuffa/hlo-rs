use crate::control_support::*;
use xlahlo::{
    Diagnostic,
    attributes::*,
    inference::control as i,
    literal::Literal,
    ops::{HloOpInterface, control::*, elementwise::NegateOp},
    pliron::{
        basic_block::BasicBlock, builtin::attributes::StringAttr, common_traits::Verify, op::Op,
        operation::Operation, r#type::Typed,
    },
    shape::{self, Dimension as D},
    types::ElementType::*,
    verification::{self, VerifyOptions},
};

#[test]
fn while_wire_order_is_body_then_condition() {
    let ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let p = shape(&ctx, Pred, &[]);
    let body = (vec![f], f);
    let condition = (vec![f], p);
    assert_eq!(i::infer_while(&ctx, f, &body, &condition).unwrap(), f);
    assert!(i::infer_while(&ctx, f, &condition, &body).is_err());
}
#[test]
fn while_checks_body_result_against_each_state_shape() {
    let ctx = context();
    let a = shape(&ctx, F32, &[2]);
    let b = shape(&ctx, F32, &[3]);
    let pred = shape(&ctx, Pred, &[]);
    let body = (vec![a], a);
    let condition = (vec![b], pred);
    assert!(i::infer_while(&ctx, a, &body, &condition).is_err());
    assert!(
        i::infer_while(&ctx, a, &(vec![], a), &(vec![], pred))
            .unwrap_err()
            .to_string()
            .contains("condition")
    );
    assert!(
        i::infer_while(&ctx, a, &(vec![], a), &(vec![a], pred))
            .unwrap_err()
            .to_string()
            .contains("body")
    );
    let unbounded = shape::dynamic_array(&ctx, F32, vec![D::Unbounded]).unwrap();
    assert!(
        i::infer_while(
            &ctx,
            shape(&ctx, F32, &[9]),
            &(vec![a], unbounded),
            &condition
        )
        .is_ok()
    );
}
#[test]
fn calls_check_signature_result_and_composite_frontend_attributes() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let (callee, block, args) = computation(&mut ctx, "callee", vec![f], f);
    ret(&mut ctx, block, args[0]);
    let (main, block, args) = computation(&mut ctx, "main", vec![f], f);
    let call = CallOp::new(&mut ctx, f, args, callees(&["callee"]));
    call.get_operation().insert_at_back(block, &ctx);
    let root = call.result(&ctx);
    ret(&mut ctx, block, root);
    let module = module(&mut ctx, &[callee, main], "main");
    let sig = (vec![f], f);
    verification::verify(&ctx, module.get_operation()).unwrap();
    attr(&ctx, call.get_operation(), "hlo_callees", Callees(vec![]));
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    attr(
        &ctx,
        call.get_operation(),
        "hlo_callees",
        callees(&["callee"]),
    );
    attr(
        &ctx,
        call.get_operation(),
        "hlo_is_composite",
        Bools(vec![true]),
    );
    assert!(call.verify(&ctx).is_err());
    attr(
        &ctx,
        call.get_operation(),
        "hlo_frontend_attributes",
        FrontendAttributes::new([
            ("composite.name".into(), "example".into()),
            ("composite.attributes".into(), "{}".into()),
            ("composite.version".into(), "2".into()),
        ]),
    );
    call.verify(&ctx).unwrap();
    attr(
        &ctx,
        call.get_operation(),
        "hlo_frontend_attributes",
        FrontendAttributes::new([
            ("composite.name".into(), "example".into()),
            ("composite.version".into(), "-1".into()),
        ]),
    );
    assert!(call.verify(&ctx).is_err());
    assert!(i::infer_call(&ctx, &[], &sig).is_err());
}
#[test]
fn conditional_selector_cardinality_and_branch_shapes_are_checked() {
    let ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let g = shape(&ctx, F32, &[3]);
    let pred = shape(&ctx, Pred, &[]);
    let index = shape(&ctx, S32, &[]);
    assert_eq!(
        i::infer_conditional(&ctx, pred, &[f, f], &[(vec![f], f), (vec![f], f)]).unwrap(),
        f
    );
    assert!(i::infer_conditional(&ctx, pred, &[f], &[(vec![f], f)]).is_err());
    assert!(i::infer_conditional(&ctx, index, &[f], &[(vec![f], f)]).is_ok());
    assert!(i::infer_conditional(&ctx, pred, &[f, f], &[(vec![f], f), (vec![f], g)]).is_err());
    assert!(i::validate_conditional_selector(&ctx, shape(&ctx, U32, &[]), 1, 1).is_err());
    let dynamic = shape::dynamic_array(&ctx, F32, vec![D::Bounded(2)]).unwrap();
    let merged =
        i::infer_conditional(&ctx, index, &[f, f], &[(vec![f], f), (vec![f], dynamic)]).unwrap();
    assert_eq!(shape::dimension(&ctx, merged, 0).unwrap(), D::Bounded(2));
}
#[test]
fn tuple_literals_parameters_and_get_tuple_element_verify_native_shapes() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let literal = Literal::from_bits(&ctx, f, &[0, 1]).unwrap();
    let constant = ConstantOp::new(&mut ctx, f, vec![], literal);
    constant.verify(&ctx).unwrap();
    let tuple = shape::tuple(&ctx, vec![f, shape::token(&ctx)]).unwrap();
    let tl = Literal::tuple(
        &ctx,
        vec![
            Literal::from_bits(&ctx, f, &[0, 1]).unwrap(),
            Literal::token(&ctx),
        ],
    )
    .unwrap();
    ConstantOp::new(&mut ctx, tuple, vec![], tl)
        .verify(&ctx)
        .unwrap();
    let token_ty = shape::token(&ctx);
    let token = AfterAllOp::new(&mut ctx, token_ty, vec![]);
    let cv = constant.result(&ctx);
    let tv = token.result(&ctx);
    let t = TupleOp::new(&mut ctx, tuple, vec![cv, tv]);
    t.verify(&ctx).unwrap();
    let tuple_value = t.result(&ctx);
    GetTupleElementOp::new(&mut ctx, f, vec![tuple_value], Index(0))
        .verify(&ctx)
        .unwrap();
    assert!(
        GetTupleElementOp::new(&mut ctx, f, vec![tuple_value], Index(2))
            .verify(&ctx)
            .is_err()
    );
    assert!(i::infer_get_tuple_element(&ctx, f, 0).is_err());
    assert!(i::infer_get_tuple_element(&ctx, tuple, -1).is_err());
    let mut bad = constant.literal(&ctx).unwrap().clone();
    bad.storage.pop();
    attr(&ctx, constant.get_operation(), "hlo_literal", bad);
    assert!(constant.verify(&ctx).is_err());
}
#[test]
fn token_operations_check_after_all_results_and_add_dependency_tokens() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[]);
    let token = shape::token(&ctx);
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    // AfterAll requires a token result without restricting operand types.
    let all = AfterAllOp::new(&mut ctx, token, vec![pv]);
    all.verify(&ctx).unwrap();
    let av = all.result(&ctx);
    AddDependencyOp::new(&mut ctx, f, vec![pv, av])
        .verify(&ctx)
        .unwrap();
    assert!(
        AddDependencyOp::new(&mut ctx, f, vec![pv, pv])
            .verify(&ctx)
            .is_err()
    );
    DomainOp::new(&mut ctx, f, vec![pv]).verify(&ctx).unwrap();
    OptimizationBarrierOp::new(&mut ctx, f, vec![pv])
        .verify(&ctx)
        .unwrap();
}
#[test]
fn fusion_owns_its_body_and_checks_internal_liveness_and_root_users() {
    for mode in 0..3 {
        let mut ctx = context();
        let f = shape(&ctx, F32, &[]);
        let p = param(&mut ctx, f, 0);
        let pv = p.result(&ctx);
        let fusion = FusionOp::new(&mut ctx, f, vec![pv], FusionKind::Loop);
        let block = BasicBlock::new(&mut ctx, None, vec![f]);
        block.insert_at_back(fusion.body(&ctx), &ctx);
        let arg = block.deref(&ctx).arguments().next().unwrap();
        let n = NegateOp::new(&mut ctx, f, vec![arg]);
        n.get_operation().insert_at_back(block, &ctx);
        if mode == 1 {
            let unused = NegateOp::new(&mut ctx, f, vec![arg]);
            unused.get_operation().insert_at_back(block, &ctx);
        }
        let root = if mode == 2 { arg } else { n.result(&ctx) };
        ret(&mut ctx, block, root);
        assert_eq!(fusion.verify(&ctx).is_ok(), mode == 0);
    }
}
#[test]
fn fusion_keeps_unused_local_effects_but_not_unused_pure_custom_calls() {
    for effect in [false, true] {
        let mut ctx = context();
        let f = shape(&ctx, S32, &[]);
        let empty = shape::tuple(&ctx, vec![]).unwrap();
        let p = param(&mut ctx, f, 0);
        let pv = p.result(&ctx);
        let fusion = FusionOp::new(&mut ctx, f, vec![pv], FusionKind::Custom);
        let block = BasicBlock::new(&mut ctx, None, vec![f]);
        block.insert_at_back(fusion.body(&ctx), &ctx);
        let call = CustomCallOp::new(&mut ctx, empty, vec![], StringAttr::new("effect".into()));
        attr(
            &ctx,
            call.get_operation(),
            "hlo_has_side_effect",
            Bools(vec![effect]),
        );
        call.get_operation().insert_at_back(block, &ctx);
        let arg = block.deref(&ctx).arguments().next().unwrap();
        ret(&mut ctx, block, arg);
        assert_eq!(fusion.verify(&ctx).is_ok(), effect);
    }
}
#[test]
fn custom_call_layout_constraints_seed_and_aliases_are_checked() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let u = shape(&ctx, U64, &[]);
    let p = param(&mut ctx, f, 0);
    CustomCallOp::new(&mut ctx, u, vec![], StringAttr::new("GetRngSeed".into()))
        .verify(&ctx)
        .unwrap();
    assert!(
        CustomCallOp::new(&mut ctx, f, vec![], StringAttr::new("GetRngSeed".into()))
            .verify(&ctx)
            .is_err()
    );
    let pv = p.result(&ctx);
    let call = CustomCallOp::new(&mut ctx, f, vec![pv], StringAttr::new("backend".into()));
    attr(
        &ctx,
        call.get_operation(),
        "hlo_constrain_layout",
        Bools(vec![true]),
    );
    assert!(call.verify(&ctx).is_err());
    attr(
        &ctx,
        call.get_operation(),
        "hlo_operand_layouts",
        ShapeList(vec![f]),
    );
    call.verify(&ctx).unwrap();
    attr(
        &ctx,
        call.get_operation(),
        "hlo_aliases",
        OutputOperandAliases(vec![OutputOperandAliasing {
            output_shape_index: Axes(vec![]),
            operand_index: 1,
            operand_shape_index: Axes(vec![]),
        }]),
    );
    assert!(call.verify(&ctx).is_err());
}
#[test]
fn pin_unpin_and_create_buffer_require_native_aliases() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let b = shape::buffer(&ctx, f).unwrap();
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    let pin = CustomCallOp::new(&mut ctx, b, vec![pv], StringAttr::new("Pin".into()));
    assert!(pin.verify(&ctx).is_err());
    let alias = OutputOperandAliases(vec![OutputOperandAliasing {
        output_shape_index: Axes(vec![]),
        operand_index: 0,
        operand_shape_index: Axes(vec![]),
    }]);
    attr(&ctx, pin.get_operation(), "hlo_aliases", alias.clone());
    pin.verify(&ctx).unwrap();
    let pinned = pin.result(&ctx);
    let unpin = CustomCallOp::new(&mut ctx, f, vec![pinned], StringAttr::new("Unpin".into()));
    attr(&ctx, unpin.get_operation(), "hlo_aliases", alias);
    unpin.verify(&ctx).unwrap();
    CustomCallOp::new(&mut ctx, b, vec![], StringAttr::new("CreateBuffer".into()))
        .verify(&ctx)
        .unwrap();
    assert!(
        CustomCallOp::new(&mut ctx, f, vec![], StringAttr::new("CreateBuffer".into()))
            .verify(&ctx)
            .is_err()
    );
}
#[test]
fn random_distribution_state_and_replica_identifiers_verify() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[]);
    let out = shape(&ctx, F32, &[2]);
    let s = shape(&ctx, S32, &[]);
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    RngOp::new(&mut ctx, out, vec![pv, pv], RandomDistribution::Normal)
        .verify(&ctx)
        .unwrap();
    let integer = param(&mut ctx, s, 0);
    let iv = integer.result(&ctx);
    assert!(
        RngOp::new(&mut ctx, s, vec![iv, iv], RandomDistribution::Normal)
            .verify(&ctx)
            .is_err()
    );
    RngOp::new(&mut ctx, s, vec![iv, iv], RandomDistribution::Uniform)
        .verify(&ctx)
        .unwrap();
    let state = shape(&ctx, U64, &[2]);
    let p = param(&mut ctx, state, 0);
    let tuple = shape::tuple(&ctx, vec![state, out]).unwrap();
    let state_value = p.result(&ctx);
    RngBitGeneratorOp::new(&mut ctx, tuple, vec![state_value], RandomAlgorithm::Philox)
        .verify(&ctx)
        .unwrap();
    RngGetAndUpdateStateOp::new(&mut ctx, state, vec![])
        .verify(&ctx)
        .unwrap();
    let id = shape(&ctx, U32, &[]);
    ReplicaIdOp::new(&mut ctx, id, vec![]).verify(&ctx).unwrap();
    PartitionIdOp::new(&mut ctx, id, vec![])
        .verify(&ctx)
        .unwrap();
    assert!(ReplicaIdOp::new(&mut ctx, s, vec![]).verify(&ctx).is_err());
    let half = shape(&ctx, F16, &[2]);
    let mixed = RngOp::new(&mut ctx, half, vec![iv, iv], RandomDistribution::Uniform);
    assert!(mixed.verify(&ctx).is_err());
    assert!(mixed.result(&ctx).get_type(&ctx) == half);
    let _ = VerifyOptions::default();
}

#[test]
fn fusion_alias_layout_relaxation_distinguishes_entry_called_and_orphan_computations() {
    for owner in ["entry", "called", "orphan"] {
        let mut ctx = context();
        let f = shape(&ctx, F32, &[2]);
        let mut layout = shape::layout(&ctx, f).unwrap().unwrap().clone();
        layout.memory_space = 5;
        let output = shape::with_layout(&ctx, f, layout).unwrap();
        let name = if owner == "entry" {
            "main"
        } else {
            "fused_owner"
        };
        let (comp, block, args) = computation(&mut ctx, name, vec![f], output);
        let fusion = FusionOp::new(&mut ctx, output, args, FusionKind::Loop);
        attr(
            &ctx,
            fusion.get_operation(),
            "hlo_aliases",
            OutputOperandAliases(vec![OutputOperandAliasing {
                output_shape_index: Axes(vec![]),
                operand_index: 0,
                operand_shape_index: Axes(vec![]),
            }]),
        );
        let body = BasicBlock::new(&mut ctx, None, vec![f]);
        body.insert_at_back(fusion.body(&ctx), &ctx);
        let arg = body.deref(&ctx).arguments().next().unwrap();
        let backend = CustomCallOp::new(
            &mut ctx,
            output,
            vec![arg],
            StringAttr::new("backend".into()),
        );
        backend.get_operation().insert_at_back(body, &ctx);
        let root = backend.result(&ctx);
        ret(&mut ctx, body, root);
        fusion.get_operation().insert_at_back(block, &ctx);
        let root = fusion.result(&ctx);
        ret(&mut ctx, block, root);
        let mut comps = vec![comp];
        if owner != "entry" {
            let result = if owner == "called" { output } else { f };
            let (entry, block, args) = computation(&mut ctx, "main", vec![f], result);
            let root = if owner == "called" {
                let call = CallOp::new(&mut ctx, output, args, callees(&["fused_owner"]));
                call.get_operation().insert_at_back(block, &ctx);
                call.result(&ctx)
            } else {
                args[0]
            };
            ret(&mut ctx, block, root);
            comps.push(entry);
        }
        let module = module(&mut ctx, &comps, "main");
        let result = verification::verify_with_options(
            &ctx,
            module.get_operation(),
            VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            },
        );
        if owner == "orphan" {
            result.unwrap();
        } else {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("aliased output"), "{owner}: {error}");
        }
    }
}

#[test]
fn nested_inline_fusion_aliases_relax_layout_without_relaxing_geometry() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let mut layout = shape::layout(&ctx, f).unwrap().unwrap().clone();
    layout.memory_space = 5;
    let output = shape::with_layout(&ctx, f, layout).unwrap();
    let (main, main_block, args) = computation(&mut ctx, "main", vec![f], output);
    let outer = FusionOp::new(&mut ctx, output, args, FusionKind::Loop);
    outer.get_operation().insert_at_back(main_block, &ctx);
    let root = outer.result(&ctx);
    ret(&mut ctx, main_block, root);
    let block = BasicBlock::new(&mut ctx, None, vec![f]);
    block.insert_at_back(outer.body(&ctx), &ctx);
    let arg = block.deref(&ctx).arguments().next().unwrap();
    let inner = FusionOp::new(&mut ctx, output, vec![arg], FusionKind::Loop);
    inner.get_operation().insert_at_back(block, &ctx);
    let inner_block = BasicBlock::new(&mut ctx, None, vec![f]);
    inner_block.insert_at_back(inner.body(&ctx), &ctx);
    let arg = inner_block.deref(&ctx).arguments().next().unwrap();
    let backend = CustomCallOp::new(
        &mut ctx,
        output,
        vec![arg],
        StringAttr::new("backend".into()),
    );
    backend.get_operation().insert_at_back(inner_block, &ctx);
    let root = backend.result(&ctx);
    ret(&mut ctx, inner_block, root);
    let root = inner.result(&ctx);
    ret(&mut ctx, block, root);
    let module = module(&mut ctx, &[main], "main");
    let aliases = OutputOperandAliases(vec![OutputOperandAliasing {
        output_shape_index: Axes(vec![]),
        operand_index: 0,
        operand_shape_index: Axes(vec![]),
    }]);
    attr(&ctx, inner.get_operation(), "hlo_aliases", aliases.clone());
    let options = VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    };
    verification::verify_with_options(&ctx, module.get_operation(), options).unwrap();
    attr(&ctx, outer.get_operation(), "hlo_aliases", aliases);
    assert!(verification::verify_with_options(&ctx, module.get_operation(), options).is_err());
    attr(
        &ctx,
        outer.get_operation(),
        "hlo_aliases",
        OutputOperandAliases(vec![]),
    );
    attr(
        &ctx,
        inner.get_operation(),
        "hlo_aliases",
        OutputOperandAliases(vec![OutputOperandAliasing {
            output_shape_index: Axes(vec![0]),
            operand_index: 0,
            operand_shape_index: Axes(vec![]),
        }]),
    );
    assert!(verification::verify_with_options(&ctx, module.get_operation(), options).is_err());
}

#[test]
fn tuple_alias_layout_checks_walk_array_leaves() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let tuple = shape::tuple(&ctx, vec![f, shape::token(&ctx)]).unwrap();
    let (callee, block, args) = computation(&mut ctx, "callee", vec![tuple], tuple);
    ret(&mut ctx, block, args[0]);
    let (main, block, args) = computation(&mut ctx, "main", vec![tuple], tuple);
    let pv = args[0];
    let call = CallOp::new(&mut ctx, tuple, vec![pv], callees(&["callee"]));
    call.get_operation().insert_at_back(block, &ctx);
    attr(
        &ctx,
        call.get_operation(),
        "hlo_aliases",
        OutputOperandAliases(vec![OutputOperandAliasing {
            output_shape_index: Axes(vec![]),
            operand_index: 0,
            operand_shape_index: Axes(vec![]),
        }]),
    );
    call.verify_with_options(
        &ctx,
        VerifyOptions {
            layout_sensitive: true,
            ..Default::default()
        },
    )
    .unwrap();
    let fusion = FusionOp::new(&mut ctx, tuple, vec![pv], FusionKind::Loop);
    fusion.get_operation().insert_at_back(block, &ctx);
    let body = BasicBlock::new(&mut ctx, None, vec![tuple]);
    body.insert_at_back(fusion.body(&ctx), &ctx);
    let arg = body.deref(&ctx).arguments().next().unwrap();
    ret(&mut ctx, body, arg);
    let root = fusion.result(&ctx);
    ret(&mut ctx, block, root);
    attr(
        &ctx,
        fusion.get_operation(),
        "hlo_aliases",
        OutputOperandAliases(vec![OutputOperandAliasing {
            output_shape_index: Axes(vec![]),
            operand_index: 0,
            operand_shape_index: Axes(vec![]),
        }]),
    );
    let module = module(&mut ctx, &[main, callee], "main");
    verification::verify_with_options(
        &ctx,
        module.get_operation(),
        VerifyOptions {
            layout_sensitive: true,
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn typed_control_diagnostics_preserve_framework_errors_and_allow_recovery() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let tuple = shape::tuple(&ctx, vec![f]).unwrap();
    let error = i::infer_get_tuple_element(&ctx, tuple, -1).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::Inference { .. })
    ));
    assert_eq!(i::infer_get_tuple_element(&ctx, tuple, 0).unwrap(), f);

    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    let call = CustomCallOp::new(&mut ctx, f, vec![pv], StringAttr::new("backend".into()));
    attr(
        &ctx,
        call.get_operation(),
        "hlo_has_side_effect",
        Bools(vec![]),
    );
    let error = call
        .verify_with_options(&ctx, Default::default())
        .unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "hlo_has_side_effect",
            ..
        })
    ));
    attr(
        &ctx,
        call.get_operation(),
        "hlo_has_side_effect",
        Bools(vec![false]),
    );
    call.verify_with_options(&ctx, Default::default()).unwrap();

    let fusion = FusionOp::new(&mut ctx, f, vec![pv], FusionKind::Loop);
    let error = fusion.verify(&ctx).unwrap_err();
    assert!(error.err.downcast_ref::<Diagnostic>().is_some());
    let block = BasicBlock::new(&mut ctx, None, vec![f]);
    block.insert_at_back(fusion.body(&ctx), &ctx);
    let root = block.deref(&ctx).arguments().next().unwrap();
    ret(&mut ctx, block, root);
    fusion.verify(&ctx).unwrap();
}

#[test]
fn malformed_tuple_projection_cycles_are_rejected_and_graph_can_be_repaired() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[]);
    let tuple = shape::tuple(&ctx, vec![f]).unwrap();
    let nested_tuple = shape::tuple(&ctx, vec![tuple]).unwrap();
    let (main, block, args) = computation(&mut ctx, "main", vec![nested_tuple], f);
    let projection = GetTupleElementOp::new(&mut ctx, tuple, vec![args[0]], Index(0));
    projection.get_operation().insert_at_back(block, &ctx);
    let value = projection.result(&ctx);
    let second = GetTupleElementOp::new(&mut ctx, f, vec![value], Index(0));
    second.get_operation().insert_at_back(block, &ctx);
    let root = second.result(&ctx);
    ret(&mut ctx, block, root);
    let module = module(&mut ctx, &[main], "main");
    Operation::replace_operand(projection.get_operation(), &ctx, 0, root);
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    Operation::replace_operand(projection.get_operation(), &ctx, 0, args[0]);
    verification::verify(&ctx, module.get_operation()).unwrap();
}
