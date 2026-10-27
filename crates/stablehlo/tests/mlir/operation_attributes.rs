use stablehlo::mlir::{export, import};

use stablehlo::{
    attributes::{ComparisonDirection, ComparisonKind},
    ops::{CompareOp, ModuleOp, TransposeOp},
    pliron::{
        builtin::attributes::StringAttr,
        context::{Context, Ptr},
        identifier::Identifier,
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
    },
};

fn first_operation(ctx: &Context, module: &ModuleOp) -> Ptr<Operation> {
    let body = module.body_block(ctx).unwrap();
    let function = body.deref(ctx).iter(ctx).next().unwrap();
    let region = function.deref(ctx).get_region(0);
    let block = region.deref(ctx).iter(ctx).next().unwrap();
    block.deref(ctx).iter(ctx).next().unwrap()
}

#[test]
fn native_readers_observe_exact_source_attribute_semantics() {
    let mut ctx = Context::new();
    let module = import(
        &mut ctx,
        r#"module {
        func.func @main(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {
          %r = "stablehlo.transpose"(%a) {permutation = array<i64: 1, 0>}
              : (tensor<2x3xf32>) -> tensor<3x2xf32>
          return %r : tensor<3x2xf32>
        }
    }"#,
    )
    .unwrap();
    let ptr = first_operation(&ctx, &module);
    let op = Operation::get_op_dyn(ptr, &ctx);
    assert_eq!(
        op.downcast_ref::<TransposeOp>()
            .unwrap()
            .permutation(&ctx)
            .unwrap(),
        [1, 0]
    );
    for (compare_type, expected) in [
        ("", None),
        (
            ", compare_type = #stablehlo.comparison_type<SIGNED>",
            Some(ComparisonKind::Signed),
        ),
    ] {
        let module = import(
            &mut ctx,
            &format!(
                r#"module {{
            func.func @main(%a: tensor<2xi32>) -> tensor<2xi1> {{
              %r = "stablehlo.compare"(%a, %a)
                  {{comparison_direction = #stablehlo.comparison_direction<LT>{compare_type}}}
                  : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
              return %r : tensor<2xi1>
            }}
        }}"#
            ),
        )
        .unwrap();
        let ptr = first_operation(&ctx, &module);
        let op = Operation::get_op_dyn(ptr, &ctx);
        let compare = op.downcast_ref::<CompareOp>().unwrap();
        assert_eq!(compare.direction(&ctx), Some(ComparisonDirection::Lt));
        assert_eq!(compare.comparison_kind(&ctx), expected);
    }
}

#[test]
fn dynamic_broadcast_optional_information_and_set_dimension_size_roundtrip() {
    for source in [
        r#"module {
            func.func @main(%a: tensor<2xf32>, %shape: tensor<2xi64>) -> tensor<3x2xf32> {
              %r = "stablehlo.dynamic_broadcast_in_dim"(%a, %shape) {
                  broadcast_dimensions = array<i64: 1>,
                  known_nonexpanding_dimensions = array<i64: 0>
              } : (tensor<2xf32>, tensor<2xi64>) -> tensor<3x2xf32>
              return %r : tensor<3x2xf32>
            }
        }"#,
        r#"module {
            func.func @main(%a: tensor<2xf32>, %size: tensor<i32>) -> tensor<?xf32> {
              %r = "stablehlo.set_dimension_size"(%a, %size) {dimension = 0 : i64}
                  : (tensor<2xf32>, tensor<i32>) -> tensor<?xf32>
              return %r : tensor<?xf32>
            }
        }"#,
    ] {
        let mut ctx = Context::new();
        let module = import(&mut ctx, source).unwrap_or_else(|error| panic!("{source}\n{error}"));
        let printed = export(&ctx, &module).unwrap();
        let mut fresh = Context::new();
        let reparsed = import(&mut fresh, &printed).unwrap();
        assert_eq!(export(&fresh, &reparsed).unwrap(), printed);
        if source.contains("known_nonexpanding_dimensions") {
            assert!(printed.contains("known_nonexpanding_dimensions"));
            assert!(!printed.contains("known_expanding_dimensions"));
        }
    }
}

#[test]
fn attribute_widths_classes_required_fields_and_permutations_are_not_coerced() {
    let mut ctx = Context::new();
    for attributes in [
        "{}",
        "{permutation = array<i32: 1, 0>}",
        "{permutation = [1, 0]}",
        "{permutation = array<i64: 0, 0>}",
        "{permutation = array<i64: 1, 0>, unknown = unit}",
        "{permutation = array<i64: 1, 0>, stablehlo_transpose_permutation = array<i64: 1, 0>}",
    ] {
        let source = format!(
            r#"module {{
            func.func @main(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {{
              %r = "stablehlo.transpose"(%a) {attributes} : (tensor<2x3xf32>) -> tensor<3x2xf32>
              return %r : tensor<3x2xf32>
            }}
        }}"#
        );
        assert!(import(&mut ctx, &source).is_err(), "{attributes}");
    }
    import(&mut ctx, "module { func.func @recovered() { return } }").unwrap();
}

#[test]
fn printer_rejects_unmapped_native_attributes_without_emitting_native_syntax() {
    let mut ctx = Context::new();
    let module = import(&mut ctx, "module { func.func @main() { return } }").unwrap();
    module.get_operation().deref_mut(&ctx).attributes.0.insert(
        Identifier::new("unmapped"),
        Box::new(StringAttr::from("keep")),
    );
    assert!(export(&ctx, &module).is_err());
    assert_eq!(module.get_operation().deref(&ctx).attributes.0.len(), 1);
}
