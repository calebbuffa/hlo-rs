use crate::control_support::*;
use xlahlo::{
    attributes::{Axes, OutputOperandAliases, OutputOperandAliasing},
    ops::control::CustomCallOp,
    pliron::{builtin::attributes::StringAttr, common_traits::Verify, op::Op},
    shape::{self, Dimension},
    types::ElementType,
};

#[test]
fn custom_call_aliases_ignore_dynamic_flags_but_not_dimension_bounds() {
    for nested in [false, true] {
        let mut ctx = context();
        let bounded = shape(&ctx, ElementType::F32, &[4]);
        let dynamic =
            shape::dynamic_array(&ctx, ElementType::F32, vec![Dimension::Bounded(4)]).unwrap();
        let unbounded =
            shape::dynamic_array(&ctx, ElementType::F32, vec![Dimension::Unbounded]).unwrap();
        let wrap = |ty| {
            if nested {
                shape::tuple(&ctx, vec![ty]).unwrap()
            } else {
                ty
            }
        };
        let cases = [
            (wrap(bounded), wrap(dynamic), true),
            (wrap(dynamic), wrap(bounded), true),
            (wrap(unbounded), wrap(unbounded), true),
            (wrap(bounded), wrap(unbounded), false),
            (wrap(unbounded), wrap(bounded), false),
            (
                wrap(bounded),
                wrap(shape(&ctx, ElementType::F32, &[5])),
                false,
            ),
            (
                wrap(bounded),
                wrap(shape(&ctx, ElementType::BF16, &[4])),
                false,
            ),
        ];
        for (input, output, accepted) in cases {
            let parameter = param(&mut ctx, input, 0);
            let value = parameter.result(&ctx);
            let call = CustomCallOp::new(
                &mut ctx,
                output,
                vec![value],
                StringAttr::new("backend".into()),
            );
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
            assert_eq!(call.verify(&ctx).is_ok(), accepted, "nested={nested}");
        }
    }
}
