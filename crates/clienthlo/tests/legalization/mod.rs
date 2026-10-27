use clienthlo::{
    Diagnostic, RecoveryCategory,
    ops::*,
    passes::ChloToStablehloPass,
    pliron::{
        context::{Context, Ptr},
        irbuild::IRStatus,
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
        pass::{AnalysisManager, Pass, Passes},
        r#type::TypeHandle,
    },
    stablehlo::{
        self,
        ops::{FuncOp, ModuleOp, ReturnOp},
        types::{ElementType, function_type, tensor},
    },
};

mod complex;
mod evaluate;
mod math;
mod precision;
mod region_values;
mod regions;
mod tensors;

#[test]
fn lowering_rejects_a_chlo_root_without_erasing_it() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let (_module, fun) = function(&mut ctx, vec![ty], vec![ty]);
    let x = fun.arguments(&ctx)[0];
    let square = SquareOp::new_inferred(&mut ctx, x).unwrap();
    let pointer = square.get_operation();
    attach(&mut ctx, &fun, square);
    let error = run(&mut ctx, pointer).err().unwrap();
    assert!(matches!(
        Diagnostic::from_error(&error),
        Some(Diagnostic::UnsupportedLowering { .. })
    ));
    assert!(Operation::get_op::<SquareOp>(pointer, &ctx).is_some());
}

#[test]
fn unmapped_metadata_is_not_dropped() {
    use clienthlo::pliron::{
        builtin::attributes::BoolAttr, identifier::Identifier, printable::Printable,
    };
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
    let x = fun.arguments(&ctx)[0];
    let square = SquareOp::new_inferred(&mut ctx, x).unwrap();
    square
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("vendor_metadata"), BoolAttr::new(true));
    attach(&mut ctx, &fun, square);
    let before = module.disp(&ctx).to_string();
    let error = run(&mut ctx, module.get_operation()).err().unwrap();
    assert!(matches!(
        Diagnostic::from_error(&error),
        Some(Diagnostic::UnsupportedLowering { .. })
    ));
    assert_eq!(module.disp(&ctx).to_string(), before);
}

#[test]
fn pass_manager_invalidates_cached_analyses() {
    use clienthlo::pliron::pass::{Analysis, AnalysisManager};
    struct ChloCount(usize);
    impl Analysis for ChloCount {
        fn name(&self) -> &str {
            "chlo-count"
        }
        fn compute(
            root: Ptr<Operation>,
            ctx: &Context,
            _: &mut AnalysisManager,
        ) -> clienthlo::Result<Self> {
            let mut work = vec![root];
            let mut count = 0;
            while let Some(op) = work.pop() {
                if Operation::get_opid(op, ctx)
                    .to_string()
                    .starts_with("chlo.")
                {
                    count += 1;
                }
                for region in op.deref(ctx).regions() {
                    for block in region.deref(ctx).iter(ctx) {
                        work.extend(block.deref(ctx).iter(ctx));
                    }
                }
            }
            Ok(Self(count))
        }
    }
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
    let x = fun.arguments(&ctx)[0];
    let square = SquareOp::new_inferred(&mut ctx, x).unwrap();
    attach(&mut ctx, &fun, square);
    let mut analyses = AnalysisManager::default();
    assert_eq!(
        analyses
            .get_analysis::<ChloCount>(module.get_operation(), &ctx)
            .unwrap()
            .0,
        1
    );
    let mut passes = Passes::default();
    passes.add_pass(ChloToStablehloPass);
    passes
        .run(module.get_operation(), &mut ctx, &mut analyses)
        .unwrap();
    assert_eq!(
        analyses
            .get_analysis::<ChloCount>(module.get_operation(), &ctx)
            .unwrap()
            .0,
        0
    );
}

#[test]
fn upstream_permissive_element_domains_fail_before_rewriting() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::I32);
    let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
    let args = fun.arguments(&ctx);
    let atan2 = BroadcastAtan2Op::new_inferred(&mut ctx, args[0], args[1], None).unwrap();
    let pointer = atan2.get_operation();
    attach(&mut ctx, &fun, atan2);
    let error = run(&mut ctx, module.get_operation()).err().unwrap();
    assert!(matches!(
        Diagnostic::from_error(&error),
        Some(Diagnostic::UnsupportedLowering { .. })
    ));
    assert!(Operation::get_op::<BroadcastAtan2Op>(pointer, &ctx).is_some());
}

fn function(
    ctx: &mut Context,
    inputs: Vec<TypeHandle>,
    outputs: Vec<TypeHandle>,
) -> (ModuleOp, FuncOp) {
    let module = ModuleOp::new(ctx);
    let signature = function_type(ctx, inputs, outputs);
    let fun = FuncOp::new(ctx, "main".into(), signature).unwrap();
    fun.get_operation()
        .insert_at_back(module.body_block(ctx).unwrap(), ctx);
    (module, fun)
}

fn attach(ctx: &mut Context, fun: &FuncOp, op: impl Op) {
    let pointer = op.get_operation();
    let values = pointer.deref(ctx).results().collect();
    pointer.insert_at_back(fun.entry_block(ctx).unwrap(), ctx);
    ReturnOp::new_many(ctx, values)
        .get_operation()
        .insert_at_back(fun.entry_block(ctx).unwrap(), ctx);
}

fn run(
    ctx: &mut Context,
    root: Ptr<Operation>,
) -> clienthlo::Result<clienthlo::pliron::pass::PassResult> {
    let mut passes = Passes::default();
    passes.add_pass(ChloToStablehloPass);
    passes.run(root, ctx, &mut AnalysisManager::default())
}

fn check(ctx: &mut Context, module: &ModuleOp) {
    assert_eq!(
        run(ctx, module.get_operation()).unwrap().ir_changed,
        IRStatus::Changed
    );
    stablehlo::verification::verify_module_fast(ctx, module.get_operation()).unwrap();
    assert_eq!(
        run(ctx, module.get_operation()).unwrap().ir_changed,
        IRStatus::Unchanged
    );
    #[cfg(feature = "mlir-interop")]
    {
        let text = stablehlo::mlir::export(ctx, module).unwrap();
        assert!(!text.contains("\"chlo."));
        native_verify(&text);
    }
}

#[cfg(feature = "mlir-interop")]
fn native_verify(source: &str) {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let Some(tool) = std::env::var_os("STABLEHLO_OPT") else {
        return;
    };
    let mut child = Command::new(tool)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{source}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn all_basic_broadcast_patterns() {
    macro_rules! cases {
        ($element:expr; $($op:ident),* $(,)?) => {$({
            let mut ctx = Context::new();
            let a = tensor(&ctx, &[2, 1], $element);
            let b = tensor(&ctx, &[3], $element);
            let result = tensor(&ctx, &[2, 3], $element);
            let (module, fun) = function(&mut ctx, vec![a,b], vec![result]);
            let args = fun.arguments(&ctx);
            let op = $op::new_inferred(&mut ctx, args[0], args[1], None).unwrap();
            attach(&mut ctx, &fun, op);
            check(&mut ctx, &module);
        })*};
    }
    cases!(ElementType::F32;
        BroadcastAddOp, BroadcastAtan2Op, BroadcastDivideOp, BroadcastMaximumOp,
        BroadcastMinimumOp, BroadcastMultiplyOp, BroadcastPowerOp, BroadcastRemainderOp, BroadcastSubtractOp, BroadcastNextAfterOp,
    );
    cases!(ElementType::I32;
        BroadcastAndOp, BroadcastOrOp, BroadcastXorOp, BroadcastShiftLeftOp,
        BroadcastShiftRightArithmeticOp, BroadcastShiftRightLogicalOp,
    );
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let complex = tensor(&ctx, &[2], ElementType::C64);
    let (module, fun) = function(&mut ctx, vec![ty, ty], vec![complex]);
    let args = fun.arguments(&ctx);
    let op = BroadcastComplexOp::new_inferred(&mut ctx, args[0], args[1], None).unwrap();
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
}

#[test]
fn dynamic_and_explicit_axis_broadcasts() {
    for (a, b, output, axes) in [
        (vec![2, -1], vec![-1], vec![2, -1], None),
        (vec![-1], vec![-1], vec![-1], None),
        (vec![0, -1], vec![1, -1], vec![0, -1], None),
        (vec![2], vec![2, 3], vec![2, 3], Some(vec![0])),
        (vec![-1], vec![-1, 3], vec![-1, 3], Some(vec![0])),
    ] {
        let mut ctx = Context::new();
        let a = tensor(&ctx, &a, ElementType::F32);
        let b = tensor(&ctx, &b, ElementType::F32);
        let out = tensor(&ctx, &output, ElementType::F32);
        let (module, fun) = function(&mut ctx, vec![a, b], vec![out]);
        let args = fun.arguments(&ctx);
        let op = BroadcastAddOp::new(&mut ctx, args[0], args[1], out, axes);
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
    }
}

#[test]
fn compare_preserves_relation_and_comparison_kind() {
    use clienthlo::attributes::{ComparisonDirection, ComparisonKind};
    for kind in [
        None,
        Some(ComparisonKind::NoType),
        Some(ComparisonKind::TotalOrder),
    ] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[-1, 2], ElementType::F32);
        let scalar = tensor(&ctx, &[], ElementType::F32);
        let predicate = tensor(&ctx, &[-1, 2], ElementType::I1);
        let (module, fun) = function(&mut ctx, vec![ty, scalar], vec![predicate]);
        let args = fun.arguments(&ctx);
        let op = BroadcastCompareOp::new_inferred(
            &mut ctx,
            args[0],
            args[1],
            BroadcastCompareOptions {
                broadcast_dimensions: None,
                direction: ComparisonDirection::Ge,
                comparison_kind: kind,
            },
        )
        .unwrap();
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
        let block = fun.entry_block(&ctx).unwrap();
        let cmp = block
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<stablehlo::ops::CompareOp>(op, &ctx))
            .find(|cmp| cmp.direction(&ctx) == Some(stablehlo::attributes::ComparisonDirection::Ge))
            .unwrap();
        assert_eq!(
            cmp.direction(&ctx),
            Some(stablehlo::attributes::ComparisonDirection::Ge)
        );
        assert_eq!(
            cmp.comparison_kind(&ctx),
            kind.filter(|&k| k != ComparisonKind::NoType)
                .map(Into::into)
        );
    }
}

#[test]
fn broadcasts_all_three_select_operands() {
    for dynamic in [false, true] {
        let mut ctx = Context::new();
        let size = if dynamic { -1 } else { 2 };
        let condition = tensor(&ctx, &[size, 1], ElementType::I1);
        let yes = tensor(&ctx, &[3], ElementType::F32);
        let no = tensor(&ctx, &[size, 3], ElementType::F32);
        let (module, fun) = function(&mut ctx, vec![condition, yes, no], vec![no]);
        let args = fun.arguments(&ctx);
        let op = BroadcastSelectOp::new_inferred(&mut ctx, args[0], args[1], args[2]).unwrap();
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
    }
}

#[test]
fn constants_and_dynamic_constant_like() {
    for shape in [vec![], vec![2], vec![0], vec![-1, 2]] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &shape, ElementType::F32);
        let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
        let x = fun.arguments(&ctx)[0];
        let op = ConstantLikeOp::new_inferred(
            &mut ctx,
            vec![1.0f32.to_bits() as u64],
            ElementType::F32,
            x,
        )
        .unwrap();
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
    }
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let (module, fun) = function(&mut ctx, vec![], vec![ty]);
    let op = ConstantOp::new(&mut ctx, ty, vec![0x80000000, 0x7fc00001], false);
    attach(&mut ctx, &fun, op);
    check(&mut ctx, &module);
}

#[test]
fn elemental_decompositions() {
    macro_rules! cases {
        ($element:expr; $($op:ident),* $(,)?) => {$({
            let mut ctx = Context::new();
            let ty = tensor(&ctx, &[-1, 2], $element);
            let out = if stringify!($op).starts_with("Is") { tensor(&ctx, &[-1,2], ElementType::I1) } else { ty };
            let (module, fun) = function(&mut ctx, vec![ty], vec![out]);
            let x = fun.arguments(&ctx)[0];
            let op = $op::new_inferred(&mut ctx, x).unwrap();
            attach(&mut ctx, &fun, op);
            check(&mut ctx, &module);
        })*};
    }
    cases!(ElementType::F32; AtanOp, AtanhOp, SquareOp, ConjOp, TanOp, IsInfOp, IsPosInfOp, IsNegInfOp);
    cases!(ElementType::C64; ConjOp, SquareOp, TanOp);
}

#[test]
fn high_integer_products() {
    for element in [
        ElementType::I2,
        ElementType::Ui4,
        ElementType::I8,
        ElementType::Ui16,
        ElementType::I32,
        ElementType::Ui32,
        ElementType::I64,
        ElementType::Ui64,
    ] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[-1], element);
        let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
        let args = fun.arguments(&ctx);
        let op = MulhiOp::new_inferred(&mut ctx, args[0], args[1]).unwrap();
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
    }
}

#[test]
fn next_after_handles_supported_float_widths() {
    for element in [
        ElementType::F16,
        ElementType::Bf16,
        ElementType::F32,
        ElementType::F64,
    ] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[-1], element);
        let (module, fun) = function(&mut ctx, vec![ty, ty], vec![ty]);
        let args = fun.arguments(&ctx);
        let op = NextAfterOp::new_inferred(&mut ctx, args[0], args[1]).unwrap();
        attach(&mut ctx, &fun, op);
        check(&mut ctx, &module);
    }
}

#[test]
fn top_k_preserves_stability_and_both_results() {
    for stable in [true, false] {
        for k in [0, 2, 4] {
            let mut ctx = Context::new();
            let ty = tensor(&ctx, &[2, 4], ElementType::F32);
            let out = tensor(&ctx, &[2, k], ElementType::F32);
            let indices = tensor(&ctx, &[2, k], ElementType::I32);
            let (module, fun) = function(&mut ctx, vec![ty], vec![out, indices]);
            let x = fun.arguments(&ctx)[0];
            let op = TopKOp::new_inferred(
                &mut ctx,
                x,
                TopKOptions {
                    k,
                    is_stable: stable,
                },
            )
            .unwrap();
            attach(&mut ctx, &fun, op);
            check(&mut ctx, &module);
        }
    }
}

#[test]
fn complex_top_k_is_rejected_before_rewriting() {
    use clienthlo::pliron::printable::Printable;
    let mut ctx = Context::new();
    let input = tensor(&ctx, &[4], ElementType::C64);
    let values = tensor(&ctx, &[2], ElementType::C64);
    let indices = tensor(&ctx, &[2], ElementType::I32);
    let (module, fun) = function(&mut ctx, vec![input], vec![values, indices]);
    let x = fun.arguments(&ctx)[0];
    let square = SquareOp::new_inferred(&mut ctx, x).unwrap();
    square
        .get_operation()
        .insert_at_back(fun.entry_block(&ctx).unwrap(), &ctx);
    let top_k = TopKOp::new_inferred(
        &mut ctx,
        x,
        TopKOptions {
            k: 2,
            is_stable: true,
        },
    )
    .unwrap();
    attach(&mut ctx, &fun, top_k);
    let before = module.disp(&ctx).to_string();
    let error = run(&mut ctx, module.get_operation()).err().unwrap();
    assert!(matches!(
        Diagnostic::from_error(&error),
        Some(Diagnostic::UnsupportedLowering { operation, .. }) if operation == "chlo.top_k"
    ));
    assert_eq!(module.disp(&ctx).to_string(), before);
}

#[test]
fn unsupported_preflight_is_explicit_and_does_not_rewrite_supported_neighbors() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let (module, fun) = function(&mut ctx, vec![ty], vec![ty]);
    let x = fun.arguments(&ctx)[0];
    let square = SquareOp::new_inferred(&mut ctx, x).unwrap();
    square
        .get_operation()
        .insert_at_back(fun.entry_block(&ctx).unwrap(), &ctx);
    let erf = ErfOp::new_inferred(&mut ctx, x).unwrap();
    erf.get_operation().deref_mut(&ctx).attributes.set(
        clienthlo::pliron::identifier::Identifier::new("vendor_metadata"),
        clienthlo::pliron::builtin::attributes::BoolAttr::new(true),
    );
    attach(&mut ctx, &fun, erf);
    let error = run(&mut ctx, module.get_operation()).err().unwrap();
    let diagnostic = Diagnostic::from_error(&error).unwrap();
    assert_eq!(
        diagnostic.recovery_category(),
        RecoveryCategory::Unsupported
    );
    assert!(
        matches!(diagnostic, Diagnostic::UnsupportedLowering { operation, .. } if operation == "chlo.erf")
    );
    assert!(Operation::get_op::<SquareOp>(square.get_operation(), &ctx).is_some());
}

#[test]
fn nested_regions_are_legalized() {
    use clienthlo::pliron::basic_block::BasicBlock;
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let pred = tensor(&ctx, &[], ElementType::I1);
    let (module, fun) = function(&mut ctx, vec![pred, ty], vec![ty]);
    let args = fun.arguments(&ctx);
    let conditional = stablehlo::ops::IfOp::new(&mut ctx, args[0], vec![ty]);
    let regions: Vec<_> = conditional.get_operation().deref(&ctx).regions().collect();
    for region in regions {
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(region, &ctx);
        let square = SquareOp::new_inferred(&mut ctx, args[1]).unwrap();
        let result = square.get_operation().deref(&ctx).get_result(0);
        square.get_operation().insert_at_back(block, &ctx);
        ReturnOp::new(&mut ctx, result)
            .get_operation()
            .insert_at_back(block, &ctx);
    }
    attach(&mut ctx, &fun, conditional);
    check(&mut ctx, &module);
}
