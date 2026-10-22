use std::collections::BTreeSet;
use xlahlo::{
    opcode::{CallContext, Opcode},
    ops::{ComputationOp, HloOpInterface, ReturnOp, elementwise::NegateOp},
    pliron::{
        basic_block::BasicBlock,
        builtin::types::FunctionType,
        context::Context,
        irfmt::parsers::spaced,
        linked_list::ContainsLinkedList,
        op::{Op, op_cast},
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
    },
    types::{ElementType, TensorType},
};

#[test]
fn opcode_names_and_arities_match_catalog() {
    let catalog = include_str!("../data/opcode_catalog.tsv");
    let mut expected = BTreeSet::new();
    for line in catalog.lines().filter(|line| !line.starts_with('#')) {
        let mut fields = line.split('\t');
        let variant = fields.next().expect("upstream variant");
        let name = fields.next().expect("upstream name").trim();
        let arity = fields.next().expect("upstream arity").trim();
        let opcode: Opcode = name.parse().unwrap();
        assert_eq!(format!("k{opcode:?}"), variant, "{name}");
        assert!(expected.insert(name));
        let arity = if arity == "*" {
            None
        } else {
            Some(arity.parse::<usize>().unwrap())
        };
        assert_eq!(opcode.arity(), arity, "{name}");
    }
    assert_eq!(expected.len(), 137);
    assert_eq!(Opcode::ALL.len(), expected.len());
    for opcode in Opcode::ALL {
        assert!(expected.remove(opcode.name()), "{opcode}");
        assert_eq!(opcode.to_string(), opcode.name());
    }
    assert!(expected.is_empty());
    assert!("unrecognized".parse::<Opcode>().is_err());
}

#[test]
fn result_accuracy_support_matches_opcode_contract() {
    let supported = [
        Opcode::Acos,
        Opcode::Acosh,
        Opcode::Asin,
        Opcode::Asinh,
        Opcode::Atanh,
        Opcode::Cbrt,
        Opcode::Cos,
        Opcode::Cosh,
        Opcode::Erf,
        Opcode::Exp,
        Opcode::Exp2,
        Opcode::Expm1,
        Opcode::Log,
        Opcode::Log1p,
        Opcode::Log2,
        Opcode::Logistic,
        Opcode::Rsqrt,
        Opcode::Sin,
        Opcode::Sinh,
        Opcode::Sqrt,
        Opcode::Tan,
        Opcode::Tanh,
    ];
    for &opcode in Opcode::ALL {
        assert_eq!(
            opcode.supports_result_accuracy(),
            supported.contains(&opcode),
            "{opcode}"
        );
    }
}

#[test]
fn opcode_capabilities_match_pinned_hlo_opcode_helpers() {
    // hlo_opcode.h and hlo_opcode.cc at 941335fe721cdc5fad5b6c37a0c2e3a7a36fea24.
    let control_flow = [
        Opcode::Call,
        Opcode::Conditional,
        Opcode::While,
        Opcode::AsyncStart,
        Opcode::AsyncUpdate,
        Opcode::AsyncDone,
    ];
    let embedded = [
        Opcode::AllReduce,
        Opcode::ReduceScatter,
        Opcode::CollectiveReduce,
        Opcode::AllReduceStart,
        Opcode::Map,
        Opcode::Reduce,
        Opcode::ReduceWindow,
        Opcode::Scan,
        Opcode::Scatter,
        Opcode::SelectAndScatter,
        Opcode::Sort,
        Opcode::Fusion,
        Opcode::CustomCall,
    ];
    let asynchronous = [Opcode::AsyncStart, Opcode::AsyncUpdate, Opcode::AsyncDone];
    let commutative = [
        Opcode::Add,
        Opcode::Multiply,
        Opcode::Mulhi,
        Opcode::Maximum,
        Opcode::Minimum,
        Opcode::And,
        Opcode::Or,
        Opcode::Xor,
    ];
    for &opcode in Opcode::ALL {
        assert_eq!(
            opcode.is_async(),
            asynchronous.contains(&opcode),
            "{opcode}"
        );
        assert_eq!(
            opcode.is_binary_commutative(),
            commutative.contains(&opcode),
            "{opcode}"
        );
        let expected = if control_flow.contains(&opcode) {
            CallContext::ControlFlow
        } else if embedded.contains(&opcode) {
            CallContext::Embedded
        } else {
            CallContext::None
        };
        assert_eq!(opcode.call_context(), expected, "{opcode}");
    }
    for (context, spelling) in [
        (CallContext::Embedded, "kEmbedded"),
        (CallContext::ControlFlow, "kControlFlow"),
        (CallContext::Both, "kBoth"),
        (CallContext::None, "kNone"),
    ] {
        assert_eq!(context.to_string(), spelling);
    }
}

#[test]
fn every_catalog_opcode_has_a_registered_native_hlo_interface() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let scalar = TensorType::get(&ctx, ElementType::F32, vec![]).unwrap();
    let signature = FunctionType::get(&ctx, vec![], vec![scalar]).into();
    let computation = ComputationOp::new(&mut ctx, "catalog", signature);
    let block = BasicBlock::new(&mut ctx, None, vec![]);
    block.insert_at_back(computation.body(&ctx), &ctx);
    let negate = NegateOp::new(&mut ctx, scalar, vec![]);
    negate.get_operation().insert_at_back(block, &ctx);
    let value = negate.result(&ctx);
    ReturnOp::new(&mut ctx, value)
        .get_operation()
        .insert_at_back(block, &ctx);
    let template = computation.disp(&ctx).to_string();
    assert!(template.contains("xlahlo.computation"));
    assert!(template.contains("xlahlo.negate"));
    assert!(template.contains("xlahlo.return"));
    assert!(template.contains("xlahlo.tensor"));
    for opcode in Opcode::ALL {
        let spelling = match opcode {
            Opcode::OptimizationBarrier => "optimization_barrier".to_owned(),
            Opcode::TopK => "top_k".to_owned(),
            _ => opcode.name().replace('-', "_"),
        };
        let name = format!("xlahlo.{spelling}");
        let assembly = template.replacen("xlahlo.negate", &name, 1);
        let mut parsed = Context::new();
        xlahlo::register(&mut parsed);
        let operation = parse_from_str(
            spaced(Operation::top_level_parser()),
            &mut parsed,
            &assembly,
        )
        .unwrap_or_else(|error| panic!("{opcode} must have a registered native parser: {error}"));
        let block = operation
            .deref(&parsed)
            .get_region(0)
            .deref(&parsed)
            .iter(&parsed)
            .next()
            .unwrap();
        let instruction = block.deref(&parsed).iter(&parsed).next().unwrap();
        let dynamic = Operation::get_op_dyn(instruction, &parsed);
        let hlo = op_cast::<dyn HloOpInterface>(dynamic.as_ref())
            .unwrap_or_else(|| panic!("{opcode} must implement HloOpInterface"));
        assert_eq!(hlo.opcode(), *opcode, "{name}");
        assert_eq!(Operation::get_opid(instruction, &parsed).to_string(), name);
    }
}
