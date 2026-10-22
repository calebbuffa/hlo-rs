//! Optional comparisons against an externally supplied native HLO verifier.
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
use xlahlo::{
    attributes::{ComparisonOrder, Direction, Index},
    ops::{
        ComputationOp, ReturnOp,
        control::ParameterOp,
        elementwise::{
            AbsOp, AddOp, CeilOp, CompareOp, ExpOp, MaximumOp, MultiplyOp, NegateOp, NotOp,
            PowerOp, ShiftLeftOp, SignOp,
        },
    },
    pliron::{
        basic_block::BasicBlock, builtin::types::FunctionType, context::Context,
        identifier::Identifier, op::Op,
    },
    types::{
        ElementType::{self, *},
        TensorType,
    },
};

fn native_accepts(source: &str) -> bool {
    let executable = PathBuf::from(
        std::env::var_os("XLAHLO_HLO_ORACLE")
            .expect("set XLAHLO_HLO_ORACLE to an external verifier executable"),
    );
    let mut child = Command::new(executable)
        .arg("verify")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start native oracle");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(source.as_bytes())
        .expect("write small native fixture");
    let output = child.wait_with_output().expect("wait for native verdict");
    assert!(
        matches!(output.status.code(), Some(0 | 2 | 3)),
        "oracle process failure: {:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    output.status.success()
}

#[test]
#[ignore = "requires external XLAHLO_HLO_ORACLE executable"]
fn elementwise_primitive_domains_match_native() {
    let elements = [
        (Pred, "pred"),
        (S8, "s8"),
        (U8, "u8"),
        (S32, "s32"),
        (U32, "u32"),
        (F16, "f16"),
        (BF16, "bf16"),
        (F32, "f32"),
        (F64, "f64"),
        (C64, "c64"),
        (C128, "c128"),
    ];
    for (element, name) in elements {
        for opcode in [
            "abs",
            "sign",
            "negate",
            "exponential",
            "ceil",
            "not",
            "add",
            "multiply",
            "power",
            "maximum",
            "shift-left",
        ] {
            let mut ctx = Context::new();
            let input = TensorType::get(&ctx, element, vec![2]).unwrap();
            let result_element = if opcode == "abs" {
                match element {
                    C64 => F32,
                    C128 => F64,
                    _ => element,
                }
            } else {
                element
            };
            let result = TensorType::get(&ctx, result_element, vec![2]).unwrap();
            let signature = FunctionType::get(&ctx, vec![input], vec![result]).into();
            let comp = ComputationOp::new(&mut ctx, Identifier::new("main"), signature);
            let block = BasicBlock::new(&mut ctx, None, vec![]);
            block.insert_at_back(comp.body(&ctx), &ctx);
            let parameter = ParameterOp::new(&mut ctx, input, vec![], Index(0));
            parameter.get_operation().insert_at_back(block, &ctx);
            let p = parameter.result(&ctx);
            let (operation, binary) = match opcode {
                "abs" => (AbsOp::new(&mut ctx, result, vec![p]).get_operation(), false),
                "sign" => (
                    SignOp::new(&mut ctx, result, vec![p]).get_operation(),
                    false,
                ),
                "negate" => (
                    NegateOp::new(&mut ctx, result, vec![p]).get_operation(),
                    false,
                ),
                "exponential" => (ExpOp::new(&mut ctx, result, vec![p]).get_operation(), false),
                "ceil" => (
                    CeilOp::new(&mut ctx, result, vec![p]).get_operation(),
                    false,
                ),
                "not" => (NotOp::new(&mut ctx, result, vec![p]).get_operation(), false),
                "add" => (
                    AddOp::new(&mut ctx, result, vec![p, p]).get_operation(),
                    true,
                ),
                "multiply" => (
                    MultiplyOp::new(&mut ctx, result, vec![p, p]).get_operation(),
                    true,
                ),
                "power" => (
                    PowerOp::new(&mut ctx, result, vec![p, p]).get_operation(),
                    true,
                ),
                "maximum" => (
                    MaximumOp::new(&mut ctx, result, vec![p, p]).get_operation(),
                    true,
                ),
                "shift-left" => (
                    ShiftLeftOp::new(&mut ctx, result, vec![p, p]).get_operation(),
                    true,
                ),
                _ => unreachable!("fixed case list"),
            };
            operation.insert_at_back(block, &ctx);
            let value = operation.deref(&ctx).get_result(0);
            ReturnOp::new(&mut ctx, value)
                .get_operation()
                .insert_at_back(block, &ctx);
            let output_name = match result_element {
                F32 => "f32",
                F64 => "f64",
                _ => name,
            };
            let arguments = if binary { "p,p" } else { "p" };
            let source = format!(
                "HloModule m ENTRY main {{ p = {name}[2] parameter(0) ROOT r = {output_name}[2] {opcode}({arguments}) }}"
            );
            assert_eq!(
                xlahlo::verification::verify(&ctx, comp.get_operation()).is_ok(),
                native_accepts(&source),
                "{source}"
            );
        }
    }
}

#[test]
#[ignore = "requires external XLAHLO_HLO_ORACLE executable"]
fn comparison_orders_match_native() {
    for (element, name) in [
        (ElementType::Pred, "pred"),
        (ElementType::S32, "s32"),
        (ElementType::F32, "f32"),
        (ElementType::C64, "c64"),
    ] {
        for (order, order_name) in [
            (ComparisonOrder::Partial, "PARTIAL"),
            (ComparisonOrder::Total, "TOTAL"),
            (ComparisonOrder::Weak, "WEAK"),
        ] {
            for (direction, direction_name) in [(Direction::Eq, "EQ"), (Direction::Lt, "LT")] {
                let mut ctx = Context::new();
                let input = TensorType::get(&ctx, element, vec![]).unwrap();
                let output = TensorType::get(&ctx, ElementType::Pred, vec![]).unwrap();
                let signature = FunctionType::get(&ctx, vec![input], vec![output]).into();
                let comp = ComputationOp::new(&mut ctx, Identifier::new("main"), signature);
                let block = BasicBlock::new(&mut ctx, None, vec![]);
                block.insert_at_back(comp.body(&ctx), &ctx);
                let param = ParameterOp::new(&mut ctx, input, vec![], Index(0));
                param.get_operation().insert_at_back(block, &ctx);
                let p = param.result(&ctx);
                let cmp = CompareOp::new(&mut ctx, output, vec![p, p], direction, order);
                cmp.get_operation().insert_at_back(block, &ctx);
                let value = cmp.result(&ctx);
                ReturnOp::new(&mut ctx, value)
                    .get_operation()
                    .insert_at_back(block, &ctx);
                let source = format!(
                    "HloModule m ENTRY main {{ p = {name}[] parameter(0) ROOT r = pred[] compare(p,p), direction={direction_name}, order={order_name} }}"
                );
                assert_eq!(
                    xlahlo::verification::verify(&ctx, comp.get_operation()).is_ok(),
                    native_accepts(&source),
                    "{source}"
                );
            }
        }
    }
}
