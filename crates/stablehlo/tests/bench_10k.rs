use pliron::{basic_block::BasicBlock, context::Context, op::Op, operation::Operation};
use stablehlo::{ops::*, types::*, verification::verify_module_fast};
use std::time::Instant;

#[test]
fn bench_10k() {
    let mut ctx = Context::default();
    let module = FuncOp::new(&mut ctx);
    let region = module.region(&ctx);
    let block = BasicBlock::new(&mut ctx, None, vec![]);
    block.insert_at_back(region, &ctx);
    let scalar_ty = tensor(&ctx, &[], ElementType::F32);

    let scalar = ConstantOp::new(&mut ctx, scalar_ty, vec![0], true);
    let scalar_value = scalar.get_operation().deref(&ctx).get_result(0);
    scalar.get_operation().insert_at_back(block, &ctx);
    let broadcast = BroadcastInDimOp::new_inferred(&mut ctx, scalar_value, &[4], vec![]).unwrap();
    let mut current = broadcast.get_operation().deref(&ctx).get_result(0);
    broadcast.get_operation().insert_at_back(block, &ctx);

    let tensor_ty = tensor(&ctx, &[4], ElementType::F32);
    let mut arithmetic = Vec::with_capacity(10_000);
    let started = Instant::now();
    let build_start = started;
    for i in 0..10_000 {
        let next = if i % 2 == 0 {
            AddOp::new(&mut ctx, current, current, tensor_ty).get_operation()
        } else {
            MultiplyOp::new(&mut ctx, current, current, tensor_ty).get_operation()
        };
        current = next.deref(&ctx).get_result(0);
        next.insert_at_back(block, &ctx);
        arithmetic.push(next);
    }
    let ret = ReturnOp::new(&mut ctx, current);
    ret.get_operation().insert_at_back(block, &ctx);

    let build_elapsed = build_start.elapsed();
    let mutation_start = Instant::now();
    let replacement = broadcast.get_operation().deref(&ctx).get_result(0);
    for ptr in arithmetic.into_iter().take(1_000) {
        let result = ptr.deref(&ctx).get_result(0);
        result.replace_all_uses_with(&ctx, &replacement);
        Operation::erase(ptr, &mut ctx);
    }
    let mutation_elapsed = mutation_start.elapsed();
    let verify_start = Instant::now();
    verify_module_fast(&ctx, module.get_operation())
        .expect("fast verifier must accept the generated function");
    let verify_elapsed = verify_start.elapsed();
    let elapsed = started.elapsed();
    println!(
        "bench_10k build={:.3}ms mutate={:.3}ms verify={:.3}ms total={:.3}ms",
        build_elapsed.as_secs_f64() * 1_000.0,
        mutation_elapsed.as_secs_f64() * 1_000.0,
        verify_elapsed.as_secs_f64() * 1_000.0,
        elapsed.as_secs_f64() * 1_000.0
    );
    if cfg!(not(debug_assertions)) {
        assert!(
            elapsed <= std::time::Duration::from_millis(15),
            "bench_10k took {elapsed:?}, expected <= 15 ms"
        );
    }
}
