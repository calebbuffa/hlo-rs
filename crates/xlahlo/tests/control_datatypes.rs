#[path = "control_support/mod.rs"]
mod support;

use support::*;
use xlahlo::{
    attributes::{FrontendAttributes, FusionKind, Index},
    inference::control as infer,
    literal::Literal,
    ops::{HloOpInterface, control::*},
    pliron::{
        basic_block::BasicBlock, common_traits::Verify, identifier::Identifier, op::Op,
        r#type::Typed,
    },
    shape::{Dimension, Layout, Shape},
    types::ElementType,
    verification::{self, VerifyOptions},
};

#[test]
fn constants_and_tuple_projections_preserve_every_primitive_encoding() {
    for &element in ElementType::ALL {
        let mut ctx = context();
        let array = Shape::with_layout(
            &ctx,
            shape(&ctx, element, &[2, 2]),
            Layout {
                minor_to_major: vec![0, 1],
                memory_space: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let byte_width = element.bit_width().div_ceil(8) as usize;
        let bytes: Vec<_> = (0..4 * byte_width)
            .map(|index| {
                if element == ElementType::Pred {
                    (index % 2) as u8
                } else {
                    [0xff, 0x80, 0x01, 0x7f, 0xa5][index % 5]
                }
            })
            .collect();
        let literal = Literal::from_storage_bytes(&ctx, array, bytes.clone()).unwrap();
        let mask = u128::MAX >> (128 - element.bit_width());
        let expected_bits: Vec<_> = bytes
            .chunks_exact(byte_width)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u128, |bits, (index, &byte)| {
                        bits | (u128::from(byte) << (index * 8))
                    })
                    & mask
            })
            .collect();
        let tuple_literal = Literal::tuple(&ctx, vec![literal, Literal::token(&ctx)]).unwrap();
        let tuple = tuple_literal.shape();
        let constant = ConstantOp::new(&mut ctx, tuple, vec![], tuple_literal);
        let frontend = FrontendAttributes::new([("payload.encoding".into(), "opaque".into())]);
        attr(
            &ctx,
            constant.get_operation(),
            "hlo_frontend_attributes",
            frontend.clone(),
        );
        constant
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    layout_sensitive: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let value = constant.result(&ctx);
        let projection = GetTupleElementOp::new(&mut ctx, array, vec![value], Index(0));
        projection.verify(&ctx).unwrap();
        let projected = projection.result(&ctx);
        let token = Shape::token(&ctx);
        let token = AfterAllOp::new(&mut ctx, token, vec![]).result(&ctx);
        let rebuilt = TupleOp::new(&mut ctx, tuple, vec![projected, token]);
        rebuilt.verify(&ctx).unwrap();
        let stored = constant.literal(&ctx).unwrap();
        let child = stored.sub_literal(&ctx, &[0]).unwrap();
        assert_eq!(child.storage, bytes, "{element:?}");
        assert_eq!(child.bits(&ctx).unwrap(), expected_bits, "{element:?}");
        assert_eq!(child.shape(), array, "{element:?}");
        assert_eq!(projection.result(&ctx).get_type(&ctx), array);
        assert_eq!(
            infer::infer_tuple(&ctx, &[array, Shape::token(&ctx)]).unwrap(),
            tuple
        );
        assert_eq!(
            constant
                .get_operation()
                .deref(&ctx)
                .attributes
                .get::<FrontendAttributes>(&Identifier::new("hlo_frontend_attributes"))
                .unwrap(),
            &frontend
        );
    }
}

#[test]
fn dynamic_constants_keep_storage_bounds_and_declared_packing_metadata() {
    for &element in ElementType::ALL {
        let mut ctx = context();
        let layout = Layout {
            minor_to_major: vec![0],
            memory_space: 7,
            element_size_in_bits: if element.is_integer() && element.bit_width() < 8
                || element == ElementType::F4E2M1FN
            {
                i64::from(element.bit_width())
            } else {
                0
            },
            ..Default::default()
        };
        let declared = Shape::with_layout(
            &ctx,
            Shape::dynamic_array(&ctx, element, vec![Dimension::Bounded(4)]).unwrap(),
            layout.clone(),
        )
        .unwrap();
        let bytes = vec![1; 4 * element.bit_width().div_ceil(8) as usize];
        let literal =
            Literal::from_dynamic_storage_bytes(&ctx, declared, bytes.clone(), vec![3]).unwrap();
        let constant = ConstantOp::new(&mut ctx, declared, vec![], literal);
        constant
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    layout_sensitive: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let stored = constant.literal(&ctx).unwrap();
        assert_eq!(stored.dynamic_sizes, [3], "{element:?}");
        assert_eq!(stored.storage, bytes, "{element:?}");
        assert_eq!(Shape::element_type(&ctx, stored.shape()).unwrap(), element);
        assert_eq!(
            Shape::dimension(&ctx, stored.shape(), 0).unwrap(),
            Dimension::Bounded(4)
        );
        let stored_layout = Shape::layout(&ctx, stored.shape()).unwrap().unwrap();
        assert_eq!(stored_layout.element_size_in_bits, 0);
        assert_eq!(stored_layout.memory_space, layout.memory_space);
        assert_eq!(
            Shape::layout(&ctx, constant.result(&ctx).get_type(&ctx))
                .unwrap()
                .as_deref(),
            Some(&layout)
        );
    }
}

#[test]
fn callable_graphs_accept_all_primitive_payloads_without_backend_type_gates() {
    for &element in ElementType::ALL {
        let mut ctx = context();
        let array = Shape::with_layout(
            &ctx,
            shape(&ctx, element, &[2]),
            Layout {
                minor_to_major: vec![0],
                memory_space: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let state = Shape::tuple(&ctx, vec![array]).unwrap();
        let pred =
            Shape::with_default_layout(&ctx, Shape::scalar(&ctx, ElementType::Pred).unwrap())
                .unwrap();
        let (callee, block, args) = computation(&mut ctx, "callee", vec![state], state);
        ret(&mut ctx, block, args[0]);
        let (body, block, args) = computation(&mut ctx, "body", vec![state], state);
        ret(&mut ctx, block, args[0]);
        let (condition, block, _) = computation(&mut ctx, "condition", vec![state], pred);
        let literal = Literal::from_bits(&ctx, pred, &[0]).unwrap();
        let stop = ConstantOp::new(&mut ctx, pred, vec![], literal);
        stop.get_operation().insert_at_back(block, &ctx);
        let stop = stop.result(&ctx);
        ret(&mut ctx, block, stop);
        let (yes, block, args) = computation(&mut ctx, "yes", vec![state], state);
        ret(&mut ctx, block, args[0]);
        let (no, block, args) = computation(&mut ctx, "no", vec![state], state);
        ret(&mut ctx, block, args[0]);
        let (main, block, args) = computation(&mut ctx, "main", vec![state, pred], array);
        let call = CallOp::new(&mut ctx, state, vec![args[0]], callees(&["callee"]));
        call.get_operation().insert_at_back(block, &ctx);
        let value = call.result(&ctx);
        let fusion = FusionOp::new(&mut ctx, state, vec![value], FusionKind::Loop);
        fusion.get_operation().insert_at_back(block, &ctx);
        let fused_block = BasicBlock::new(&mut ctx, None, vec![state]);
        fused_block.insert_at_back(fusion.body(&ctx), &ctx);
        let argument = fused_block.deref(&ctx).arguments().next().unwrap();
        ret(&mut ctx, fused_block, argument);
        let value = fusion.result(&ctx);
        let loop_op = WhileOp::new(
            &mut ctx,
            state,
            vec![value],
            callees(&["body", "condition"]),
        );
        loop_op.get_operation().insert_at_back(block, &ctx);
        let value = loop_op.result(&ctx);
        let branch = ConditionalOp::new(
            &mut ctx,
            state,
            vec![args[1], value, value],
            callees(&["yes", "no"]),
        );
        branch.get_operation().insert_at_back(block, &ctx);
        let value = branch.result(&ctx);
        let projection = GetTupleElementOp::new(&mut ctx, array, vec![value], Index(0));
        projection.get_operation().insert_at_back(block, &ctx);
        let root = projection.result(&ctx);
        ret(&mut ctx, block, root);
        let module = module(&mut ctx, &[main, callee, body, condition, yes, no], "main");
        verification::verify_with_options(
            &ctx,
            module.get_operation(),
            VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("{element:?}: {error}"));
        for value in [
            call.result(&ctx),
            fusion.result(&ctx),
            loop_op.result(&ctx),
            branch.result(&ctx),
        ] {
            assert_eq!(value.get_type(&ctx), state, "{element:?}");
        }
        assert_eq!(root.get_type(&ctx), array, "{element:?}");
    }
}

#[test]
fn conditional_dynamic_union_preserves_payload_type_and_layout_for_all_primitives() {
    let ctx = context();
    let selector = Shape::scalar(&ctx, ElementType::S32).unwrap();
    for &element in ElementType::ALL {
        let layout = Layout {
            minor_to_major: vec![0],
            memory_space: 5,
            ..Default::default()
        };
        let static_array =
            Shape::with_layout(&ctx, shape(&ctx, element, &[4]), layout.clone()).unwrap();
        let dynamic_array = Shape::with_layout(
            &ctx,
            Shape::dynamic_array(&ctx, element, vec![Dimension::Bounded(4)]).unwrap(),
            layout.clone(),
        )
        .unwrap();
        let static_tuple = Shape::tuple(&ctx, vec![static_array]).unwrap();
        let dynamic_tuple = Shape::tuple(&ctx, vec![dynamic_array]).unwrap();
        let merged = infer::infer_conditional(
            &ctx,
            selector,
            &[static_tuple, static_tuple],
            &[
                (vec![static_tuple], static_tuple),
                (vec![static_tuple], dynamic_tuple),
            ],
        )
        .unwrap();
        let child = Shape::subshape(&ctx, merged, &[0]).unwrap();
        assert_eq!(Shape::element_type(&ctx, child).unwrap(), element);
        assert_eq!(
            Shape::dimension(&ctx, child, 0).unwrap(),
            Dimension::Bounded(4)
        );
        assert_eq!(
            Shape::layout(&ctx, child).unwrap().as_deref(),
            Some(&layout)
        );
        let wrong = shape(
            &ctx,
            if element == ElementType::F32 {
                ElementType::F64
            } else {
                ElementType::F32
            },
            &[4],
        );
        assert!(infer::infer_call(&ctx, &[static_array], &(vec![wrong], wrong)).is_err());
    }
}
