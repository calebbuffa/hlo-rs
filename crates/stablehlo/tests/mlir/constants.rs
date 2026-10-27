use stablehlo::mlir::{export, import};

use stablehlo::{
    ops::ConstantOp,
    pliron::{context::Context, linked_list::ContainsLinkedList, operation::Operation},
};

#[test]
fn dense_and_sparse_quantized_constants_keep_storage_parameters_and_encodings() {
    for quantized in [
        "!quant.uniform<i8:f32, 0.25:3>",
        "!quant.uniform<i8:f32:0, {0.25:3, 0.5:4}>",
    ] {
        let result = format!("tensor<2x{quantized}, #stablehlo.type_extensions<bounds = [?]>>");
        let storage = "tensor<2xi8, #stablehlo.type_extensions<bounds = [?]>>";
        for value in ["dense<[3, 4]>", "sparse<[[1]], [4]>"] {
            let source = format!(
                r#"module {{ func.func @main() -> {result} {{
                    %c = "stablehlo.constant"() {{value = {value} : {storage}}} : () -> {result}
                    return %c : {result}
                }} }}"#
            );
            let mut ctx = Context::new();
            let module =
                import(&mut ctx, &source).unwrap_or_else(|error| panic!("{source}\n{error}"));
            let printed = export(&ctx, &module).unwrap();
            assert_eq!(
                printed.matches("#stablehlo.type_extensions").count(),
                4,
                "{printed}"
            );
            assert!(printed.contains("!quant.uniform"));
            let mut fresh = Context::new();
            let reparsed = import(&mut fresh, &printed).unwrap();
            assert_eq!(export(&fresh, &reparsed).unwrap(), printed);
            let bad = source.replace(
                &format!("{value} : {storage}"),
                &format!("{value} : tensor<2xi8>"),
            );
            assert!(import(&mut ctx, &bad).is_err(), "{bad}");
        }
    }
}

#[test]
fn packed_complex_payloads_have_component_order_and_exact_nan_bits() {
    let source = r#"module { func.func @main() -> tensor<complex<f64>> {
        %c = stablehlo.constant dense<"0x010000000000F07F0000000000000080"> : tensor<complex<f64>>
        return %c : tensor<complex<f64>>
    } }"#;
    let mut ctx = Context::new();
    let module = import(&mut ctx, source).unwrap();
    let function = module
        .body_block(&ctx)
        .unwrap()
        .deref(&ctx)
        .iter(&ctx)
        .next()
        .unwrap();
    let region = function.deref(&ctx).get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    let ptr = block.deref(&ctx).iter(&ctx).next().unwrap();
    let op = Operation::get_op_dyn(ptr, &ctx);
    let constant = op.downcast_ref::<ConstantOp>().unwrap();
    assert_eq!(
        constant.bits(&ctx).unwrap(),
        [0x7ff0_0000_0000_0001, 0x8000_0000_0000_0000]
    );
    assert!(constant.is_splat(&ctx).unwrap());
    let printed = export(&ctx, &module).unwrap();
    let mut fresh = Context::new();
    let reparsed = import(&mut fresh, &printed).unwrap();
    assert_eq!(export(&fresh, &reparsed).unwrap(), printed);
}
