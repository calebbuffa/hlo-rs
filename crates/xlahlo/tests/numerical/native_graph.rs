use super::super::*;
use xlahlo::Diagnostic;
use xlahlo::literal::Literal;
use xlahlo::ops::control::CustomCallOp;
use xlahlo::pliron::{
    basic_block::BasicBlock,
    builtin::{attributes::StringAttr, ops::ModuleOp, types::FunctionType},
    context::Ptr,
    irfmt::parsers::spaced,
    op::verify_op,
    operation::Operation,
};
use xlahlo::verification::{verify, verify_with_options};

fn import_type(source: &Context, target: &mut Context, ty: TypeHandle) -> TypeHandle {
    parse_from_str(TypeHandle::parser(()), target, &ty.disp(source).to_string()).unwrap()
}

// Build isolated native computations so contract tests exercise symbol resolution
// and the supported graph verifier rather than internal family dispatch.
pub(crate) fn verify_callees(
    source: &Context,
    original: Ptr<Operation>,
    opcode: O,
    options: VerifyOptions,
    signatures: &[&Signature],
) -> Result<()> {
    let mut ctx = context();
    let original = original.deref(source);
    let operands = original
        .operand_types(source)
        .map(|ty| import_type(source, &mut ctx, ty))
        .collect::<Vec<_>>();
    let result = import_type(source, &mut ctx, original.get_type(0));
    let names = &original
        .attributes
        .get::<Callees>(&Identifier::new("hlo_callees"))
        .unwrap()
        .0;
    assert!(
        signatures.len() <= names.len(),
        "too many test callee signatures"
    );
    let (module, container) = module(&mut ctx);
    let (main, args) = computation(&mut ctx, container, "main", operands, result);
    let op = match opcode {
        O::Reduce => ReduceOp::new(
            &mut ctx,
            result,
            args,
            original
                .attributes
                .get::<Axes>(&Identifier::new("hlo_dimensions"))
                .unwrap()
                .clone(),
            Callees(names.clone()),
        )
        .get_operation(),
        O::ReduceWindow => ReduceWindowOp::new(
            &mut ctx,
            result,
            args,
            original
                .attributes
                .get::<Window>(&Identifier::new("hlo_window"))
                .unwrap()
                .clone(),
            Callees(names.clone()),
        )
        .get_operation(),
        O::Map => MapOp::new(&mut ctx, result, args, Callees(names.clone())).get_operation(),
        O::Sort => SortOp::new(
            &mut ctx,
            result,
            args,
            original
                .attributes
                .get::<Axes>(&Identifier::new("hlo_dimensions"))
                .unwrap()
                .clone(),
            Callees(names.clone()),
        )
        .get_operation(),
        O::Scan => ScanOp::new(
            &mut ctx,
            result,
            args,
            original
                .attributes
                .get::<Axes>(&Identifier::new("hlo_dimensions"))
                .unwrap()
                .clone(),
            *original
                .attributes
                .get::<NumericalInteger>(&Identifier::new("hlo_num_carries"))
                .unwrap(),
            *original
                .attributes
                .get::<TriState>(&Identifier::new("hlo_is_associative"))
                .unwrap(),
            Callees(names.clone()),
        )
        .get_operation(),
        O::SelectAndScatter => SelectAndScatterOp::new(
            &mut ctx,
            result,
            args,
            original
                .attributes
                .get::<Window>(&Identifier::new("hlo_window"))
                .unwrap()
                .clone(),
            Callees(names.clone()),
        )
        .get_operation(),
        _ => panic!("not a callable numerical opcode: {opcode}"),
    };
    op.insert_at_back(main, &ctx);
    let value = op.deref(&ctx).get_result(0);
    ret(&mut ctx, main, value);
    for (name, signature) in names.iter().zip(signatures) {
        let parameters = signature
            .0
            .iter()
            .map(|&ty| import_type(source, &mut ctx, ty))
            .collect();
        let result = import_type(source, &mut ctx, signature.1);
        let (body, _) = computation(&mut ctx, container, name.as_ref(), parameters, result);
        let root = CustomCallOp::new(
            &mut ctx,
            result,
            vec![],
            StringAttr::new("numerical_test_result".into()),
        );
        root.get_operation().insert_at_back(body, &ctx);
        let value = root.result(&ctx);
        ret(&mut ctx, body, value);
    }
    verify_with_options(&ctx, module.get_operation(), options)
}

fn roundtrip_graph(ctx: &Context, root: Ptr<Operation>) {
    let assembly = root.disp(ctx).to_string();
    let mut parsed = context();
    let root = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed,
        &assembly,
    )
    .unwrap_or_else(|error| panic!("{assembly}\n{error}"));
    verify(&parsed, root).unwrap();
    verify_op(Operation::get_op_dyn(root, &parsed).as_ref(), &parsed).unwrap();
}

fn module(ctx: &mut Context) -> (ModuleOp, Ptr<BasicBlock>) {
    let module = ModuleOp::new(ctx, Identifier::new("numerical"));
    xlahlo::ir::set_entry(ctx, module.get_operation(), Identifier::new("main")).unwrap();
    let block = module
        .get_operation()
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block()
        .unwrap();
    (module, block)
}

fn computation(
    ctx: &mut Context,
    container: Ptr<BasicBlock>,
    name: &str,
    inputs: Vec<TypeHandle>,
    output: TypeHandle,
) -> (Ptr<BasicBlock>, Vec<Value>) {
    let signature = FunctionType::get(ctx, inputs.clone(), vec![output]).into();
    let op = ComputationOp::new(ctx, name.try_into().unwrap(), signature);
    op.get_operation().insert_at_back(container, ctx);
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(op.body(ctx), ctx);
    let parameters = inputs
        .into_iter()
        .enumerate()
        .map(|(index, ty)| {
            let op = ParameterOp::new(ctx, ty, vec![], Index(index as u32));
            op.get_operation().insert_at_back(block, ctx);
            op.result(ctx)
        })
        .collect();
    (block, parameters)
}

fn ret(ctx: &mut Context, block: Ptr<BasicBlock>, value: Value) {
    ReturnOp::new(ctx, value)
        .get_operation()
        .insert_at_back(block, ctx);
}

#[test]
fn graph_errors_preserve_typed_numerical_cause_and_operation_context() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 4]);
    let rhs = a(&ctx, P::F32, &[4, 3]);
    let result = a(&ctx, P::F32, &[2, 3]);
    let (module, container) = module(&mut ctx);
    let (main, args) = computation(&mut ctx, container, "main", vec![lhs, rhs], result);
    let dot = DotOp::new(
        &mut ctx,
        result,
        args,
        dot_dims(&[1], &[0], &[], &[]),
        precision(),
    );
    dot.get_operation().insert_at_back(main, &ctx);
    let value = dot.result(&ctx);
    ret(&mut ctx, main, value);
    dot.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_precision"));
    let error = verify(&ctx, module.get_operation()).unwrap_err();
    let Some(Diagnostic::Operation { operation, source }) = error.err.downcast_ref::<Diagnostic>()
    else {
        panic!("expected operation context, got {error:?}");
    };
    assert!(operation.contains("hlo.dot"), "{operation}");
    assert!(matches!(
        source.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "hlo_precision",
            ..
        }),
    ));
    dot.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_precision"), precision());
    verify(&ctx, module.get_operation()).unwrap();
}

#[test]
fn indexed_graph_resolves_heterogeneous_comparator_without_strict_fp_equality() {
    let mut ctx = context();
    let keys = a(&ctx, P::F32, &[2, 3]);
    let vals = a(&ctx, P::S32, &[2, 3]);
    let result = Shape::tuple(&ctx, vec![keys, vals]).unwrap();
    let (module, container) = module(&mut ctx);
    let (main, params) = computation(&mut ctx, container, "main", vec![keys, vals], result);
    let sort = SortOp::new(
        &mut ctx,
        result,
        params,
        Axes(vec![1]),
        Callees(vec![Identifier::new("cmp")]),
    );
    sort.get_operation().insert_at_back(main, &ctx);
    let value = sort.result(&ctx);
    ret(&mut ctx, main, value);
    let pred = s(&ctx, P::Pred);
    let parameter_types = vec![
        s(&ctx, P::F64),
        s(&ctx, P::F16),
        s(&ctx, P::S32),
        s(&ctx, P::S32),
    ];
    let (cmp, _) = computation(&mut ctx, container, "cmp", parameter_types, pred);
    let literal = Literal::from_bits(&ctx, pred, &[1]).unwrap();
    let truth = ConstantOp::new(&mut ctx, pred, vec![], literal);
    truth.get_operation().insert_at_back(cmp, &ctx);
    let value = truth.result(&ctx);
    ret(&mut ctx, cmp, value);
    verify(&ctx, module.get_operation()).unwrap();
    roundtrip_graph(&ctx, module.get_operation());
    sort.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_callees"),
        Callees(vec![Identifier::new("missing")]),
    );
    assert!(verify(&ctx, module.get_operation()).is_err());
}

#[test]
fn indexed_graph_verifies_variadic_reducer_tuple_and_missing_callee() {
    let mut ctx = context();
    let f = s(&ctx, P::F32);
    let i = s(&ctx, P::S32);
    let fa = a(&ctx, P::F32, &[2, 3]);
    let ia = a(&ctx, P::S32, &[2, 3]);
    let result = Shape::tuple(&ctx, vec![a(&ctx, P::F32, &[3]), a(&ctx, P::S32, &[3])]).unwrap();
    let root = Shape::tuple(&ctx, vec![f, i]).unwrap();
    let (module, container) = module(&mut ctx);
    let (sum, params) = computation(&mut ctx, container, "sum", vec![f, i, f, i], root);
    let tuple = TupleOp::new(&mut ctx, root, vec![params[0], params[1]]);
    tuple.get_operation().insert_at_back(sum, &ctx);
    let value = tuple.result(&ctx);
    ret(&mut ctx, sum, value);
    let (main, params) = computation(&mut ctx, container, "main", vec![fa, ia, f, i], result);
    let reduce = ReduceOp::new(
        &mut ctx,
        result,
        params,
        Axes(vec![0]),
        Callees(vec![Identifier::new("sum")]),
    );
    reduce.get_operation().insert_at_back(main, &ctx);
    let value = reduce.result(&ctx);
    ret(&mut ctx, main, value);
    verify(&ctx, module.get_operation()).unwrap();
    roundtrip_graph(&ctx, module.get_operation());
    reduce
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_callees"), Callees(vec![]));
    assert!(verify(&ctx, module.get_operation()).is_err());
}

#[test]
fn indexed_graph_checks_scan_slices_carries_and_dynamic_layout_policy() {
    let mut ctx = context();
    let slice = a(&ctx, P::F32, &[3]);
    let input = dynamic(&ctx, P::F32, &[D::Static(4), D::Bounded(3)]);
    let array = a(&ctx, P::F32, &[4, 3]);
    let result = Shape::tuple(&ctx, vec![array, slice]).unwrap();
    let root = Shape::tuple(&ctx, vec![slice, slice]).unwrap();
    let (module, container) = module(&mut ctx);
    let (body, params) = computation(&mut ctx, container, "body", vec![slice, slice], root);
    let tuple = TupleOp::new(&mut ctx, root, params);
    tuple.get_operation().insert_at_back(body, &ctx);
    let value = tuple.result(&ctx);
    ret(&mut ctx, body, value);
    let (main, params) = computation(&mut ctx, container, "main", vec![input, slice], result);
    let scan = ScanOp::new(
        &mut ctx,
        result,
        params,
        Axes(vec![0]),
        NumericalInteger(1),
        TriState::True,
        Callees(vec![Identifier::new("body")]),
    );
    scan.get_operation().insert_at_back(main, &ctx);
    let value = scan.result(&ctx);
    ret(&mut ctx, main, value);
    verify(&ctx, module.get_operation()).unwrap();
    roundtrip_graph(&ctx, module.get_operation());
    assert!(
        xlahlo::verification::verify_with_options(
            &ctx,
            module.get_operation(),
            VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn native_assembly_round_trips_every_noncallable_numerical_opcode() {
    for opcode in [
        O::Dot,
        O::RaggedDot,
        O::ScaledDot,
        O::Convolution,
        O::TopK,
        O::Fft,
        O::TriangularSolve,
        O::Cholesky,
        O::BatchNormTraining,
        O::BatchNormInference,
        O::BatchNormGrad,
    ] {
        let mut ctx = context();
        let lhs = a(&ctx, P::F32, &[2, 4]);
        let rhs = a(&ctx, P::F32, &[4, 3]);
        let output = a(&ctx, P::F32, &[2, 3]);
        let vector = a(&ctx, P::F32, &[3]);
        let stats = Shape::tuple(&ctx, vec![output, vector, vector]).unwrap();
        let matrix = a(&ctx, P::F32, &[3, 3]);
        let (inputs, result) = match opcode {
            O::Dot | O::Convolution => (vec![lhs, rhs], output),
            O::RaggedDot => (
                vec![lhs, a(&ctx, P::F32, &[5, 4, 3]), a(&ctx, P::S32, &[5])],
                output,
            ),
            O::ScaledDot => (
                vec![lhs, rhs, a(&ctx, P::F16, &[1, 2]), a(&ctx, P::F16, &[2, 1])],
                output,
            ),
            O::TopK => (
                vec![lhs],
                Shape::tuple(&ctx, vec![output, a(&ctx, P::S32, &[2, 3])]).unwrap(),
            ),
            O::Fft => (vec![a(&ctx, P::F32, &[6])], a(&ctx, P::C64, &[4])),
            O::TriangularSolve => (vec![matrix, rhs], rhs),
            O::Cholesky => (vec![matrix], matrix),
            O::BatchNormTraining => (vec![output, vector, vector], stats),
            O::BatchNormInference => (vec![output, vector, vector, vector, vector], output),
            O::BatchNormGrad => (vec![output, vector, vector, vector, output], stats),
            _ => unreachable!(),
        };
        let (module, container) = module(&mut ctx);
        let (main, params) = computation(&mut ctx, container, "main", inputs, result);
        let bn = BatchNormOptions {
            epsilon_bits: 0.01f32.to_bits(),
            feature_index: 1,
        };
        let op = match opcode {
            O::Dot => DotOp::new(
                &mut ctx,
                result,
                params,
                dot_dims(&[1], &[0], &[], &[]),
                PrecisionConfig {
                    operand_precision: OperandPrecision(vec![Precision::High, Precision::Highest]),
                    algorithm: Algorithm::Unset,
                },
            )
            .get_operation(),
            O::RaggedDot => RaggedDotOp::new(
                &mut ctx,
                result,
                params,
                RaggedDotDimensionNumbers {
                    dot_dimension_numbers: dot_dims(&[1], &[1], &[], &[]),
                    lhs_ragged_dimensions: Axes(vec![0]),
                    rhs_group_dimensions: Axes(vec![0]),
                },
                precision(),
            )
            .get_operation(),
            O::ScaledDot => ScaledDotOp::new(
                &mut ctx,
                result,
                params,
                dot_dims(&[1], &[0], &[], &[]),
                precision(),
            )
            .get_operation(),
            O::Convolution => ConvolutionOp::new(
                &mut ctx,
                result,
                params,
                ConvolutionDimensionNumbers {
                    input_batch_dimension: 0,
                    input_feature_dimension: 1,
                    input_spatial_dimensions: Axes(vec![]),
                    kernel_input_feature_dimension: 0,
                    kernel_output_feature_dimension: 1,
                    kernel_spatial_dimensions: Axes(vec![]),
                    output_batch_dimension: 0,
                    output_feature_dimension: 1,
                    output_spatial_dimensions: Axes(vec![]),
                },
                Window(vec![]),
                NumericalInteger(1),
                NumericalInteger(1),
                ConvolutionKind::Fprop,
                precision(),
            )
            .get_operation(),
            O::TopK => TopKOp::new(
                &mut ctx,
                result,
                params,
                NumericalInteger(3),
                SortOptions { descending: true },
            )
            .get_operation(),
            O::Fft => {
                FftOp::new(&mut ctx, result, params, FftType::RFFT, Axes(vec![6])).get_operation()
            }
            O::TriangularSolve => TriangularSolveOp::new(
                &mut ctx,
                result,
                params,
                TriangularSolveOptions {
                    left_side: false,
                    lower: true,
                    unit_diagonal: true,
                    transpose_a: Transpose::Adjoint,
                },
            )
            .get_operation(),
            O::Cholesky => {
                CholeskyOp::new(&mut ctx, result, params, CholeskyOptions { lower: true })
                    .get_operation()
            }
            O::BatchNormTraining => {
                BatchNormTrainingOp::new(&mut ctx, result, params, bn).get_operation()
            }
            O::BatchNormInference => {
                BatchNormInferenceOp::new(&mut ctx, result, params, bn).get_operation()
            }
            O::BatchNormGrad => BatchNormGradOp::new(&mut ctx, result, params, bn).get_operation(),
            _ => unreachable!(),
        };
        op.insert_at_back(main, &ctx);
        let value = op.deref(&ctx).get_result(0);
        ret(&mut ctx, main, value);
        verify(&ctx, module.get_operation()).unwrap();
        roundtrip_graph(&ctx, module.get_operation());
    }
}

#[test]
fn native_assembly_round_trips_map_window_reduce_and_select_and_scatter_contracts() {
    for opcode in [O::Map, O::ReduceWindow, O::SelectAndScatter] {
        let mut ctx = context();
        let scalar = s(&ctx, P::F32);
        let input = a(&ctx, P::F32, &[5]);
        let windowed = a(&ctx, P::F32, &[3]);
        let (inputs, result) = match opcode {
            O::Map => (vec![input], input),
            O::ReduceWindow => (vec![input, scalar], windowed),
            _ => (vec![input, windowed, scalar], input),
        };
        let (module, container) = module(&mut ctx);
        let (body, params) = computation(
            &mut ctx,
            container,
            "body",
            vec![scalar; if opcode == O::Map { 1 } else { 2 }],
            scalar,
        );
        ret(&mut ctx, body, params[0]);
        if opcode == O::SelectAndScatter {
            let pred = s(&ctx, P::Pred);
            let (select, _) = computation(&mut ctx, container, "select", vec![scalar; 2], pred);
            let literal = Literal::from_bits(&ctx, pred, &[1]).unwrap();
            let truth = ConstantOp::new(&mut ctx, pred, vec![], literal);
            truth.get_operation().insert_at_back(select, &ctx);
            let value = truth.result(&ctx);
            ret(&mut ctx, select, value);
        }
        let (main, params) = computation(&mut ctx, container, "main", inputs, result);
        let op = match opcode {
            O::Map => MapOp::new(
                &mut ctx,
                result,
                params,
                Callees(vec![Identifier::new("body")]),
            )
            .get_operation(),
            O::ReduceWindow => ReduceWindowOp::new(
                &mut ctx,
                result,
                params,
                super::windows::window(3, 1, 0, 0, 1, 1),
                Callees(vec![Identifier::new("body")]),
            )
            .get_operation(),
            _ => SelectAndScatterOp::new(
                &mut ctx,
                result,
                params,
                super::windows::window(3, 1, 0, 0, 1, 1),
                Callees(vec![Identifier::new("select"), Identifier::new("body")]),
            )
            .get_operation(),
        };
        op.insert_at_back(main, &ctx);
        let value = op.deref(&ctx).get_result(0);
        ret(&mut ctx, main, value);
        verify(&ctx, module.get_operation()).unwrap();
        roundtrip_graph(&ctx, module.get_operation());
    }
}
