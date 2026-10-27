use stablehlo::{
    attributes::*,
    mlir::{
        self,
        assembly::{self, MlirPrinter},
    },
    ops::*,
    pliron::{
        attribute::{AttrObj, Attribute},
        basic_block::BasicBlock,
        builtin::{
            attributes::{BoolAttr, DictAttr, IntegerAttr, StringAttr, VecAttr},
            types::{IntegerType, Signedness},
        },
        common_traits::Named,
        context::{Context, Ptr},
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, Typed},
        utils::apint::{APInt, bw},
        value::Value,
    },
    types::*,
};
use std::{
    collections::HashSet,
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn every_native_operation_has_an_authored_fixture() {
    let fixtures = include_str!("export.rs");
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("ops");
    for file in std::fs::read_dir(directory).unwrap() {
        let path = file.unwrap().path();
        if path.extension().is_none_or(|extension| extension != "rs") {
            continue;
        }
        let source = std::fs::read_to_string(path).unwrap();
        for declaration in source.split("#[pliron_op(").skip(1) {
            let (_, body) = declaration.split_once("pub struct ").unwrap();
            let wrapper = body
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .next()
                .unwrap();
            assert!(
                fixtures
                    .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .any(|token| token == wrapper),
                "missing authored export fixture for {wrapper}"
            );
        }
    }
}

fn native_verify(source: &str) {
    let Some(verifier) = std::env::var_os("STABLEHLO_OPT") else {
        return;
    };
    let mut child = Command::new(verifier)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch native MLIR verifier");
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

fn check(ctx: &Context, op: impl Op) -> String {
    let op = op.get_operation();
    let text = String::try_from(MlirPrinter::new(ctx, &op)).unwrap();
    let mut inside = HashSet::new();
    let mut used = Vec::new();
    fn walk(ctx: &Context, op: Ptr<Operation>, inside: &mut HashSet<Value>, used: &mut Vec<Value>) {
        let raw = op.deref(ctx);
        inside.extend(raw.results());
        used.extend(raw.operands());
        for region in raw.regions() {
            for block in region.deref(ctx).iter(ctx) {
                inside.extend(block.deref(ctx).arguments());
                for child in block.deref(ctx).iter(ctx) {
                    walk(ctx, child, inside, used);
                }
            }
        }
    }
    walk(ctx, op, &mut inside, &mut used);
    let mut seen = HashSet::new();
    let args = used
        .into_iter()
        .filter(|value| !inside.contains(value) && seen.insert(*value))
        .map(|value| {
            format!(
                "%{}: {}",
                value.unique_name(ctx),
                assembly::print_type(ctx, value.get_type(ctx)).unwrap()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    native_verify(&format!(
        "module {{ func.func @test({args}) {{\n{text}\nreturn\n}} }}"
    ));
    text
}

fn input(ctx: &mut Context, shape: &[i64], element: ElementType) -> Value {
    let ty = tensor(ctx, shape, element);
    BasicBlock::new(ctx, None, vec![ty])
        .deref(ctx)
        .get_argument(0)
}

fn groups(ctx: &Context) -> VecAttr {
    let ty = IntegerType::get(ctx, 64, Signedness::Signless);
    VecAttr::new(vec![Box::new(VecAttr::new(
        [0, 1]
            .into_iter()
            .map(|value| Box::new(IntegerAttr::new(ty, APInt::from_i64(value, bw(64)))) as AttrObj)
            .collect(),
    ))])
}

macro_rules! check_op {
    ($ctx:ident, $op:expr) => {{
        let op = $op;
        check(&$ctx, op)
    }};
}

fn yielding_region(
    ctx: &mut Context,
    op: impl Op,
    index: usize,
    args: Vec<TypeHandle>,
    value: Value,
) {
    let region = op.get_operation().deref(ctx).get_region(index);
    let block = BasicBlock::new(ctx, None, args);
    block.insert_at_back(region, ctx);
    ReturnOp::new(ctx, value)
        .get_operation()
        .insert_at_back(block, ctx);
}

fn add_reducer(ctx: &mut Context, op: impl Op, scalar: TypeHandle) {
    let region = op.get_operation().deref(ctx).get_region(0);
    let block = BasicBlock::new(ctx, None, vec![scalar, scalar]);
    block.insert_at_back(region, ctx);
    let lhs = block.deref(ctx).get_argument(0);
    let rhs = block.deref(ctx).get_argument(1);
    let sum = AddOp::new(ctx, lhs, rhs, scalar);
    sum.get_operation().insert_at_back(block, ctx);
    let result = sum.get_operation().deref(ctx).get_result(0);
    ReturnOp::new(ctx, result)
        .get_operation()
        .insert_at_back(block, ctx);
}

#[test]
fn reductions_and_collective_reducers() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let init = input(&mut ctx, &[], ElementType::F32);
    let scalar = init.get_type(&ctx);
    let op = ReduceOp::new(&mut ctx, vec![x], vec![init], vec![scalar], vec![0]);
    add_reducer(&mut ctx, op, scalar);
    check(&ctx, op);
    let op = ReduceWindowOp::new(&mut ctx, vec![x], vec![init], vec![ty], vec![1]);
    add_reducer(&mut ctx, op, scalar);
    check(&ctx, op);
    let replicas = groups(&ctx);
    let op = AllReduceOp::new(&mut ctx, vec![x], vec![ty], replicas);
    add_reducer(&mut ctx, op, scalar);
    check(&ctx, op);
    let replicas = groups(&ctx);
    let narrow = tensor(&ctx, &[1], ElementType::F32);
    let op = ReduceScatterOp::new(&mut ctx, vec![x], vec![narrow], 0, replicas);
    add_reducer(&mut ctx, op, scalar);
    check(&ctx, op);
}

#[test]
fn collectives_without_reducers() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let replicas = groups(&ctx);
    check_op!(
        ctx,
        AllToAllOp::new(&mut ctx, vec![x], vec![ty], 0, 0, 2, replicas)
    );
    let replicas = groups(&ctx);
    check_op!(
        ctx,
        CollectiveBroadcastOp::new(&mut ctx, vec![x], vec![ty], false, replicas)
    );
    let pairs = groups(&ctx)
        .0
        .remove(0)
        .downcast_ref::<VecAttr>()
        .unwrap()
        .clone();
    check_op!(ctx, CollectivePermuteOp::new(&mut ctx, x, ty, pairs));
}

#[test]
fn legacy_cross_replica_sum() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let replicas = groups(&ctx);
    check_op!(ctx, CrossReplicaSumOp::new(&mut ctx, x, ty, replicas));
}

#[test]
fn collective_reduce() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let scalar = tensor(&ctx, &[], ElementType::F32);
    let replicas = groups(&ctx);
    let op = CollectiveReduceOp::new(&mut ctx, vec![x], vec![ty], replicas);
    add_reducer(&mut ctx, op, scalar);
    check(&ctx, op);
}

#[test]
fn conditional_tuple_and_token_operations() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let predicate = input(&mut ctx, &[], ElementType::I1);
    let selector = input(&mut ctx, &[], ElementType::I32);
    let op = IfOp::new(&mut ctx, predicate, vec![ty]);
    yielding_region(&mut ctx, op, 0, vec![], x);
    yielding_region(&mut ctx, op, 1, vec![], x);
    check(&ctx, op);
    let op = CaseOp::new(&mut ctx, selector, vec![ty], 2);
    yielding_region(&mut ctx, op, 0, vec![], x);
    yielding_region(&mut ctx, op, 1, vec![], x);
    check(&ctx, op);
    let tuple = TupleOp::new(&mut ctx, vec![x, x]);
    let tuple_value = tuple.get_operation().deref(&ctx).get_result(0);
    check(&ctx, tuple);
    check_op!(ctx, GetTupleElementOp::new(&mut ctx, tuple_value, 1, ty));
    check_op!(ctx, OptimizationBarrierOp::new(&mut ctx, vec![x]));
    let token = CreateTokenOp::new(&mut ctx);
    let value = token.get_operation().deref(&ctx).get_result(0);
    check(&ctx, token);
    check_op!(ctx, AfterAllOp::new(&mut ctx, vec![value]));
    check_op!(ctx, ReplicaIdOp::new(&mut ctx));
    check_op!(ctx, PartitionIdOp::new(&mut ctx));
}

#[test]
fn side_effect_operations() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let token = CreateTokenOp::new(&mut ctx)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let layout = ArrayOfLayoutAttr {
        layouts: vec![LayoutAttr {
            minor_to_major: vec![0],
        }],
    };
    check_op!(ctx, InfeedOp::new(&mut ctx, token, ty, "".into(), layout));
    check_op!(ctx, OutfeedOp::new(&mut ctx, x, token, "".into()));
    check_op!(
        ctx,
        SendOp::new(
            &mut ctx,
            x,
            token,
            ChannelHandleAttr {
                handle: 1,
                channel_kind: ChannelKind::DeviceToDevice
            },
            false
        )
    );
    check_op!(
        ctx,
        RecvOp::new(
            &mut ctx,
            token,
            ty,
            ChannelHandleAttr {
                handle: 1,
                channel_kind: ChannelKind::DeviceToDevice
            },
            false
        )
    );
}

#[test]
fn flat_variadic_transfers_and_source_target_pairs() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let y = input(&mut ctx, &[], ElementType::I32);
    let types = [x.get_type(&ctx), y.get_type(&ctx)];
    let token = CreateTokenOp::new(&mut ctx)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let channel = ChannelHandleAttr {
        handle: 1,
        channel_kind: ChannelKind::DeviceToDevice,
    };
    for count in [0, 1, 2] {
        let layouts = ArrayOfLayoutAttr {
            layouts: [vec![0], vec![]]
                .into_iter()
                .take(count)
                .map(|minor_to_major| LayoutAttr { minor_to_major })
                .collect(),
        };
        let infeed =
            InfeedOp::new_variadic(&mut ctx, token, types[..count].to_vec(), "".into(), layouts);
        check(&ctx, infeed);
        let values = [x, y][..count].to_vec();
        let outfeed = OutfeedOp::new_variadic(&mut ctx, values.clone(), token, "".into());
        check(&ctx, outfeed);
        let send = SendOp::new_variadic(&mut ctx, values, token, channel.clone(), false);
        let pairs =
            assembly::parse_attribute(&ctx, "dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>").unwrap();
        let pair_type = IntegerType::get(&ctx, 64, Signedness::Signless);
        let pairs = pairs
            .downcast_ref::<DenseElementsAttr>()
            .unwrap()
            .values
            .0
            .iter()
            .map(|value| {
                let bits = value.downcast_ref::<ScalarBitsAttr>().unwrap().words[0];
                Box::new(IntegerAttr::new(pair_type, APInt::from_u64(bits, bw(64)))) as AttrObj
            })
            .collect::<Vec<_>>();
        send.set_attr_stablehlo_send_source_target_pairs(&ctx, VecAttr::new(pairs.clone()));
        assert!(check(&ctx, send).contains("source_target_pairs = dense<[[0, 1], [1, 0]]>"));
        let recv = RecvOp::new_variadic(
            &mut ctx,
            token,
            types[..count].to_vec(),
            channel.clone(),
            false,
        );
        recv.set_attr_stablehlo_recv_source_target_pairs(&ctx, VecAttr::new(pairs));
        assert!(check(&ctx, recv).contains("source_target_pairs = dense<[[0, 1], [1, 0]]>"));
    }
}

fn mesh_groups() -> ReplicaGroupMeshAxesAttr {
    ReplicaGroupMeshAxesAttr {
        mesh: MeshReferenceAttr {
            inline: vec![MeshAttr {
                axes: vec![MeshAxisAttr {
                    name: StringAttr::new("x".into()),
                    size: 2,
                }],
                device_ids: vec![],
            }],
            symbol: vec![],
        },
        axes: vec![AxisRefAttr {
            name: StringAttr::new("x".into()),
            sub_axis_info: vec![],
        }],
    }
}

#[test]
fn every_mesh_collective_exports() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let wide = tensor(&ctx, &[4], ElementType::F32);
    let narrow = tensor(&ctx, &[1], ElementType::F32);
    let scalar = tensor(&ctx, &[], ElementType::F32);
    macro_rules! mesh {
        ($operation:expr) => {{
            let op = $operation;
            op.set_replica_group_mesh_axes(&mut ctx, mesh_groups());
            assert!(op.replica_group_mesh_axes(&ctx).is_some());
            let text = check(&ctx, op);
            assert!(text.contains("replica_groups = #stablehlo.replica_group_mesh_axes"));
            assert!(text.contains("#stablehlo.axis_ref<name = \"x\">"));
        }};
    }
    let explicit = groups(&ctx);
    mesh!(AllGatherOp::new(&mut ctx, vec![x], vec![wide], 0, explicit));
    let explicit = groups(&ctx);
    mesh!(AllToAllOp::new(
        &mut ctx,
        vec![x],
        vec![ty],
        0,
        0,
        2,
        explicit
    ));
    let explicit = groups(&ctx);
    mesh!(CollectiveBroadcastOp::new(
        &mut ctx,
        vec![x],
        vec![ty],
        false,
        explicit
    ));
    let explicit = groups(&ctx);
    let all_reduce = AllReduceOp::new(&mut ctx, vec![x], vec![ty], explicit);
    add_reducer(&mut ctx, all_reduce, scalar);
    mesh!(all_reduce);
    let explicit = groups(&ctx);
    let scatter = ReduceScatterOp::new(&mut ctx, vec![x], vec![narrow], 0, explicit);
    add_reducer(&mut ctx, scatter, scalar);
    mesh!(scatter);
    let explicit = groups(&ctx);
    let reduce = CollectiveReduceOp::new(&mut ctx, vec![x], vec![ty], explicit);
    add_reducer(&mut ctx, reduce, scalar);
    mesh!(reduce);
}

#[test]
fn split_mesh_axis_collectives_export() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let wide = tensor(&ctx, &[4], ElementType::F32);
    let explicit = groups(&ctx);
    let gather = AllGatherOp::new(&mut ctx, vec![x], vec![wide], 0, explicit);
    let mut groups = mesh_groups();
    groups.mesh.inline[0].axes[0].size = 4;
    groups.axes[0].sub_axis_info = vec![SubAxisInfoAttr {
        pre_size: 2,
        size: 2,
    }];
    gather.set_replica_group_mesh_axes(&mut ctx, groups);
    let text = check(&ctx, gather);
    assert!(text.contains("#stablehlo<sub_axis_info(2)2>"));
}

#[test]
fn symbolic_mesh_groups_resolve_in_exported_modules() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx);
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let wide = tensor(&ctx, &[4], ElementType::F32);
    let signature = function_type(&ctx, vec![ty], vec![wide]);
    let function = FuncOp::new(&mut ctx, "gather".into(), signature).unwrap();
    function
        .get_operation()
        .insert_at_back(module.body_block(&ctx).unwrap(), &ctx);
    let argument = function.arguments(&ctx)[0];
    let explicit = groups(&ctx);
    let gather = AllGatherOp::new(&mut ctx, vec![argument], vec![wide], 0, explicit);
    let mut groups = mesh_groups();
    groups.mesh = MeshReferenceAttr {
        inline: vec![],
        symbol: vec![StringAttr::new("mesh".into())],
    };
    gather.set_replica_group_mesh_axes(&mut ctx, groups);
    gather
        .get_operation()
        .insert_at_back(function.entry_block(&ctx).unwrap(), &ctx);
    let result = gather.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, result)
        .get_operation()
        .insert_at_back(function.entry_block(&ctx).unwrap(), &ctx);
    assert!(
        mlir::export(&ctx, &module)
            .unwrap_err()
            .to_string()
            .contains("unresolved mesh symbol")
    );
    let signature = function_type(&ctx, vec![], vec![]);
    let mesh = FuncOp::new(&mut ctx, "mesh".into(), signature).unwrap();
    mesh.get_operation()
        .insert_at_back(module.body_block(&ctx).unwrap(), &ctx);
    ReturnOp::new_many(&mut ctx, vec![])
        .get_operation()
        .insert_at_back(mesh.entry_block(&ctx).unwrap(), &ctx);
    native_verify(&mlir::export(&ctx, &module).unwrap());
}

#[test]
fn native_only_structures_are_explicit_export_errors() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let composite = CompositeOp::new(
        &mut ctx,
        "example.identity".into(),
        1,
        "identity".into(),
        vec![x],
        vec![ty],
        1,
    );
    let error = String::try_from(MlirPrinter::new(&ctx, &composite.get_operation())).unwrap_err();
    assert!(
        error.to_string().contains("cannot contain regions"),
        "{error}"
    );
    let tuple = TupleOp::new(&mut ctx, vec![x]);
    let token = CreateTokenOp::new(&mut ctx)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let value = tuple.get_operation().deref(&ctx).get_result(0);
    let outfeed = OutfeedOp::new(&mut ctx, value, token, "".into());
    let error = String::try_from(MlirPrinter::new(&ctx, &outfeed.get_operation())).unwrap_err();
    assert!(
        error.to_string().contains("flat tensor/token values"),
        "{error}"
    );
}

#[test]
fn elementwise_options_and_iotas() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let pred = input(&mut ctx, &[2], ElementType::I1);
    let bool_ty = pred.get_type(&ctx);
    let shape = input(&mut ctx, &[1], ElementType::I64);
    check_op!(ctx, SelectOp::new(&mut ctx, pred, x, x, ty));
    check_op!(
        ctx,
        CompareOp::new(
            &mut ctx,
            x,
            x,
            bool_ty,
            ComparisonDirection::Lt,
            ComparisonKind::Float
        )
    );
    check_op!(ctx, ClampOp::new(&mut ctx, x, x, x, ty));
    check_op!(ctx, IotaOp::new(&mut ctx, ty, 0));
    check_op!(ctx, DynamicIotaOp::new(&mut ctx, shape, ty, 0));
    check_op!(ctx, ReducePrecisionOp::new(&mut ctx, x, ty, 8, 23));
}

#[test]
fn dot_fft_cholesky() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2, 2], ElementType::F32);
    let ty = x.get_type(&ctx);
    check_op!(
        ctx,
        DotOp::new(&mut ctx, x, x, ty, vec![Precision::Default; 2])
    );
    let dims = DotDimensionNumbersAttr {
        lhs_batching_dimensions: vec![],
        rhs_batching_dimensions: vec![],
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![0],
    };
    check_op!(ctx, DotGeneralOp::new(&mut ctx, x, x, ty, dims));
    check_op!(ctx, CholeskyOp::new(&mut ctx, x, ty, true));
    let c = input(&mut ctx, &[2], ElementType::C64);
    let cty = c.get_type(&ctx);
    check_op!(ctx, FftOp::new(&mut ctx, c, cty, FftKind::Fft, vec![2]));
}

#[test]
fn legacy_einsums() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2, 2], ElementType::F32);
    let ty = x.get_type(&ctx);
    check_op!(ctx, EinsumOp::new(&mut ctx, x, x, ty, "ij,jk->ik".into()));
    check_op!(ctx, UnaryEinsumOp::new(&mut ctx, x, ty, "ij->ji".into()));
}

#[test]
fn async_operations() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let start = AsyncStartOp::new(&mut ctx, vec![x], vec![ty]);
    let block = BasicBlock::new(&mut ctx, None, vec![ty]);
    block.insert_at_back(start.body_region(&ctx), &ctx);
    let argument = block.deref(&ctx).get_argument(0);
    let slice = SliceOp::new(&mut ctx, argument, ty, vec![0], vec![2], vec![1]);
    slice.get_operation().insert_at_back(block, &ctx);
    let value = slice.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, value)
        .get_operation()
        .insert_at_back(block, &ctx);
    check(&ctx, start);
    let future = start.get_operation().deref(&ctx).get_result(0);
    check_op!(ctx, AsyncDoneOp::new(&mut ctx, future, vec![ty]));
}

#[test]
fn indexing_operations() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[4], ElementType::F32);
    let indices = input(&mut ctx, &[2, 1], ElementType::I32);
    let updates = input(&mut ctx, &[2], ElementType::F32);
    let result = updates.get_type(&ctx);
    let dims = GatherDimensionNumbersAttr {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        operand_batching_dims: vec![],
        start_indices_batching_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
    };
    let gather = GatherOp::new(&mut ctx, x, indices, result, dims.clone(), vec![1]);
    check(&ctx, gather);
    gather.set_attr_stablehlo_gather_indices_are_sorted(&ctx, BoolAttr::new(true));
    assert!(check(&ctx, gather).contains("indices_are_sorted = true"));
    let shape = input(&mut ctx, &[1], ElementType::I64);
    let gather = DynamicGatherOp::new(&mut ctx, x, indices, shape, result, dims);
    check(&ctx, gather);
    gather.set_attr_stablehlo_dynamic_gather_indices_are_sorted(&ctx, BoolAttr::new(true));
    assert!(check(&ctx, gather).contains("indices_are_sorted = true"));
    let dims = ScatterDimensionNumbersAttr {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        input_batching_dims: vec![],
        scatter_indices_batching_dims: vec![],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let scatter = ScatterOp::new(&mut ctx, vec![x], indices, vec![updates], dims);
    let scalar = tensor(&ctx, &[], ElementType::F32);
    add_reducer(&mut ctx, scatter, scalar);
    check(&ctx, scatter);
    scatter.set_attr_stablehlo_scatter_indices_are_sorted(&ctx, BoolAttr::new(true));
    scatter.set_attr_stablehlo_scatter_unique_indices(&ctx, BoolAttr::new(true));
    let text = check(&ctx, scatter);
    assert!(text.contains("indices_are_sorted = true") && text.contains("unique_indices = true"));
}

#[test]
fn legacy_torch_index_select() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[4], ElementType::F32);
    let indices = input(&mut ctx, &[2], ElementType::I32);
    let ty = tensor(&ctx, &[2], ElementType::F32);
    check_op!(ctx, TorchIndexSelectOp::new(&mut ctx, x, indices, ty, 0));
}

#[test]
fn map_sort_select_and_scatter() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let scalar = tensor(&ctx, &[], ElementType::F32);
    let map = MapOp::new(&mut ctx, vec![x], ty, vec![0]);
    let block = BasicBlock::new(&mut ctx, None, vec![scalar]);
    block.insert_at_back(map.get_operation().deref(&ctx).get_region(0), &ctx);
    let arg = block.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, arg)
        .get_operation()
        .insert_at_back(block, &ctx);
    check(&ctx, map);
    let sort = SortOp::new(&mut ctx, vec![x], vec![ty], 0, true);
    let block = BasicBlock::new(&mut ctx, None, vec![scalar, scalar]);
    block.insert_at_back(sort.get_operation().deref(&ctx).get_region(0), &ctx);
    let lhs = block.deref(&ctx).get_argument(0);
    let rhs = block.deref(&ctx).get_argument(1);
    let boolean = tensor(&ctx, &[], ElementType::I1);
    let cmp = CompareOp::new(
        &mut ctx,
        lhs,
        rhs,
        boolean,
        ComparisonDirection::Lt,
        ComparisonKind::Float,
    );
    cmp.get_operation().insert_at_back(block, &ctx);
    let pred = cmp.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, pred)
        .get_operation()
        .insert_at_back(block, &ctx);
    check(&ctx, sort);
    check(&ctx, sort.with_dimension(&ctx, -1));
    sort.get_operation().deref_mut(&ctx).attributes.0.remove(
        &stablehlo::pliron::identifier::Identifier::new("stablehlo_reduction_sort_dimension"),
    );
    check(&ctx, sort);
    let init = input(&mut ctx, &[], ElementType::F32);
    let op = SelectAndScatterOp::new(&mut ctx, x, x, init, ty, vec![1]);
    let block = BasicBlock::new(&mut ctx, None, vec![scalar, scalar]);
    block.insert_at_back(op.get_operation().deref(&ctx).get_region(0), &ctx);
    let lhs = block.deref(&ctx).get_argument(0);
    let rhs = block.deref(&ctx).get_argument(1);
    let cmp = CompareOp::new(
        &mut ctx,
        lhs,
        rhs,
        boolean,
        ComparisonDirection::Ge,
        ComparisonKind::Float,
    );
    cmp.get_operation().insert_at_back(block, &ctx);
    let pred = cmp.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, pred)
        .get_operation()
        .insert_at_back(block, &ctx);
    yielding_region(&mut ctx, op, 1, vec![scalar, scalar], init);
    check(&ctx, op);
}

#[test]
fn quantization_operations() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let quantized = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: None,
            storage_max: None,
            expressed_type: ElementType::F32,
            quantization_dimension: None,
            scales: vec![0.5],
            zero_points: vec![0],
        },
    );
    let quantized = TensorType::from_element(&ctx, quantized, vec![2]);
    let op = UniformQuantizeOp::new(&mut ctx, x, quantized);
    check(&ctx, op);
    let value = op.get_operation().deref(&ctx).get_result(0);
    check_op!(ctx, UniformDequantizeOp::new(&mut ctx, value, ty));
}

#[test]
fn quantized_dense_and_sparse_constants_export_storage_typed_values() {
    let mut ctx = Context::new();
    for per_axis in [false, true] {
        let element = QuantizedElementType::intern(
            &ctx,
            UniformQuantizedSpec {
                storage_type: ElementType::I8,
                storage_min: None,
                storage_max: None,
                expressed_type: ElementType::F32,
                quantization_dimension: per_axis.then_some(0),
                scales: if per_axis { vec![0.5, 0.25] } else { vec![0.5] },
                zero_points: if per_axis { vec![0, 1] } else { vec![0] },
            },
        );
        let ty = TensorType::from_element(&ctx, element, vec![2]);
        let constant = ConstantOp::new(&mut ctx, ty, vec![1, 2], false);
        let text = check(&ctx, constant);
        assert!(text.contains("value = dense<[1, 2]> : tensor<2xi8>"));
        let value = Box::new(ScalarBitsAttr {
            ty: ElementType::I8.native(&ctx),
            words: vec![2],
        });
        let constant = ConstantOp::new_sparse(&mut ctx, ty, vec![vec![1]], vec![value]);
        assert!(check(&ctx, constant).contains("value = sparse<"));
    }
}

#[test]
fn custom_call_supports_every_upstream_api_version() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    for version in 0..=4 {
        let op = CustomCallOp::new(&mut ctx, "target".into(), vec![x], vec![ty], version);
        let text = check(&ctx, op);
        assert!(text.contains(&format!("api_version = {version} : i32")));
    }
}

#[test]
fn scalar_select_and_scatter_accepts_omitted_window_dimensions() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[], ElementType::F32);
    let ty = x.get_type(&ctx);
    let predicate = input(&mut ctx, &[], ElementType::I1);
    let op = SelectAndScatterOp::new(&mut ctx, x, x, x, ty, vec![]);
    yielding_region(&mut ctx, op, 0, vec![ty, ty], predicate);
    yielding_region(&mut ctx, op, 1, vec![ty, ty], x);
    for name in [
        "stablehlo_reduction_select_window",
        "stablehlo_reduction_select_strides",
        "stablehlo_reduction_select_padding",
    ] {
        op.get_operation()
            .deref_mut(&ctx)
            .attributes
            .0
            .remove(&stablehlo::pliron::identifier::Identifier::new(name));
    }
    check(&ctx, op);
}

#[test]
fn random_generators() {
    let mut ctx = Context::new();
    let bound = input(&mut ctx, &[], ElementType::F32);
    let shape = input(&mut ctx, &[1], ElementType::I64);
    let ty = tensor(&ctx, &[2], ElementType::F32);
    check_op!(
        ctx,
        RngOp::new(
            &mut ctx,
            bound,
            bound,
            shape,
            ty,
            RngDistributionAttr::Uniform
        )
    );
    let state = input(&mut ctx, &[2], ElementType::Ui64);
    let state_ty = state.get_type(&ctx);
    let bits = tensor(&ctx, &[2], ElementType::Ui32);
    check_op!(
        ctx,
        RngBitGeneratorOp::new(
            &mut ctx,
            state,
            vec![state_ty, bits],
            RngAlgorithmAttr::ThreeFry
        )
    );
}

#[test]
fn batch_norm_training_and_grad() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2, 3], ElementType::F32);
    let scale = input(&mut ctx, &[3], ElementType::F32);
    let ty = x.get_type(&ctx);
    let feature = scale.get_type(&ctx);
    let results = vec![ty, feature, feature];
    check_op!(
        ctx,
        BatchNormTrainingOp::new(
            &mut ctx,
            x,
            scale,
            scale,
            results.clone(),
            BatchNormOptions {
                feature_index: 1,
                epsilon: 0.001
            }
        )
    );
    check_op!(
        ctx,
        BatchNormGradOp::new(
            &mut ctx,
            BatchNormGradOperands {
                x,
                scale,
                mean: scale,
                variance: scale,
                grad: x
            },
            results,
            BatchNormOptions {
                feature_index: 1,
                epsilon: 0.001
            }
        )
    );
}

#[test]
fn dynamic_convolution() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[1, 4, 2], ElementType::F32);
    let kernel = input(&mut ctx, &[3, 2, 3], ElementType::F32);
    let padding = input(&mut ctx, &[1, 2], ElementType::I64);
    let ty = tensor(&ctx, &[1, -1, 3], ElementType::F32);
    let dims = ConvDimensionNumbersAttr {
        input_batch_dimension: 0,
        input_feature_dimension: 2,
        input_spatial_dimensions: vec![1],
        kernel_input_feature_dimension: 1,
        kernel_output_feature_dimension: 2,
        kernel_spatial_dimensions: vec![0],
        output_batch_dimension: 0,
        output_feature_dimension: 2,
        output_spatial_dimensions: vec![1],
    };
    let defaults = DynamicConvOp::new(&mut ctx, x, kernel, padding, ty, dims.clone());
    check(&ctx, defaults);
    let op = DynamicConvOp::new(&mut ctx, x, kernel, padding, ty, dims).with_window_options(
        &ctx,
        DynamicConvolutionOptions {
            strides: vec![1],
            lhs_dilation: vec![1],
            rhs_dilation: vec![1],
            feature_group_count: 1,
            batch_group_count: 1,
            window_reversal: vec![false],
            precision: vec![Precision::Default; 2],
        },
    );
    check(&ctx, op);
}

#[test]
fn calls_and_composite_symbols() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let module = ModuleOp::new(&mut ctx);
    let signature = function_type(&ctx, vec![ty], vec![ty]);
    let identity = FuncOp::new(&mut ctx, "identity".into(), signature).unwrap();
    identity
        .get_operation()
        .insert_at_back(module.body_block(&ctx).unwrap(), &ctx);
    let argument = identity.arguments(&ctx)[0];
    ReturnOp::new(&mut ctx, argument)
        .get_operation()
        .insert_at_back(identity.entry_block(&ctx).unwrap(), &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".into(), signature).unwrap();
    caller
        .get_operation()
        .insert_at_back(module.body_block(&ctx).unwrap(), &ctx);
    let block = caller.entry_block(&ctx).unwrap();
    let argument = caller.arguments(&ctx)[0];
    let call = CallOp::new(&mut ctx, "identity".into(), vec![argument], vec![ty]);
    call.get_operation().insert_at_back(block, &ctx);
    let argument = call.get_operation().deref(&ctx).get_result(0);
    let composite = CompositeOp::new(
        &mut ctx,
        "example.identity".into(),
        1,
        "identity".into(),
        vec![argument],
        vec![ty],
        0,
    );
    composite.get_operation().insert_at_back(block, &ctx);
    let result = composite.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, result)
        .get_operation()
        .insert_at_back(block, &ctx);
    native_verify(&mlir::export(&ctx, &module).unwrap());
}

#[test]
fn sparse_encodings_preserve_maps_flags_widths_and_slices() {
    use stablehlo::sparse::{
        SparseLevelAttr, SparseLevelFormat, SparseTensorDimSlice, SparseTensorEncoding,
    };
    let ctx = Context::new();
    let map = AffineMap {
        dimensions: 2,
        symbols: 0,
        results: vec![AffineExpr::dimension(1), AffineExpr::dimension(0)],
    };
    let mut compressed = SparseLevelAttr::new(SparseLevelFormat::Compressed);
    compressed.nonunique = true;
    compressed.nonordered = true;
    let mut encoding = SparseTensorEncoding::new(
        &ctx,
        vec![SparseLevelAttr::new(SparseLevelFormat::Dense), compressed],
        map,
    )
    .unwrap();
    encoding.position_width = 32;
    encoding.coordinate_width = 64;
    encoding.dimension_slices = vec![
        SparseTensorDimSlice {
            offset: vec![0],
            size: vec![2],
            stride: vec![1],
        },
        SparseTensorDimSlice {
            offset: vec![],
            size: vec![],
            stride: vec![1],
        },
    ];
    let text = assembly::print_attribute(&ctx, &encoding).unwrap();
    assert!(text.contains("d1 : dense, d0 : compressed(nonunique, nonordered)"));
    assert!(text.contains(
        "map = (d0 : #sparse_tensor<slice(0, 2, 1)>, d1 : #sparse_tensor<slice(?, ?, 1)>) ->"
    ));
    assert!(!text.contains("dimSlices"));
    native_verify(&format!(
        "module attributes {{test.encoding = {text}}} {{}}"
    ));
    let structured = SparseTensorEncoding::new(
        &ctx,
        vec![SparseLevelAttr::new(SparseLevelFormat::Structured {
            n: 2,
            m: 4,
        })],
        AffineMap {
            dimensions: 1,
            symbols: 0,
            results: vec![AffineExpr::dimension(0)],
        },
    )
    .unwrap();
    let text = assembly::print_attribute(&ctx, &structured).unwrap();
    assert!(text.contains("d0 : structured[2, 4]"));
    native_verify(&format!(
        "module attributes {{test.encoding = {text}}} {{}}"
    ));
    let symbolic = SparseTensorEncoding::new(
        &ctx,
        vec![SparseLevelAttr::new(SparseLevelFormat::Dense)],
        AffineMap {
            dimensions: 1,
            symbols: 1,
            results: vec![AffineExpr::dimension(0)],
        },
    )
    .unwrap();
    let text = assembly::print_attribute(&ctx, &symbolic).unwrap();
    assert!(text.contains("map = [s0](d0) ->"));
    native_verify(&format!(
        "module attributes {{test.encoding = {text}}} {{}}"
    ));
}

#[test]
fn invalid_locations_are_rejected_instead_of_clamped() {
    use stablehlo::pliron::{
        combine::stream::position::SourcePosition,
        location::{Location, Source},
    };
    let mut ctx = Context::new();
    let op = CreateTokenOp::new(&mut ctx);
    op.get_operation()
        .deref_mut(&ctx)
        .set_loc(Location::SrcPos {
            src: Source::InMemory,
            pos: SourcePosition {
                line: -1,
                column: 0,
            },
        });
    assert!(String::try_from(MlirPrinter::new(&ctx, &op.get_operation())).is_err());
}

#[test]
fn authored_module_exports_nested_regions_and_contextual_returns() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let module = ModuleOp::new(&mut ctx);
    let signature = function_type(&ctx, vec![ty], vec![ty]);
    let function = FuncOp::new(&mut ctx, "loop".into(), signature).unwrap();
    function
        .get_operation()
        .insert_at_back(module.body_block(&ctx).unwrap(), &ctx);
    let entry = function.entry_block(&ctx).unwrap();
    let arg = function.arguments(&ctx)[0];
    let loop_op = WhileOp::new(&mut ctx, vec![arg], vec![ty]);
    loop_op.get_operation().insert_at_back(entry, &ctx);
    let cond = BasicBlock::new(&mut ctx, None, vec![ty]);
    cond.insert_at_back(loop_op.condition_region(&ctx), &ctx);
    let pred_ty = tensor(&ctx, &[], ElementType::I1);
    let pred = ConstantOp::new(&mut ctx, pred_ty, vec![0], true);
    pred.get_operation().insert_at_back(cond, &ctx);
    let pred_value = pred.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, pred_value)
        .get_operation()
        .insert_at_back(cond, &ctx);
    let body = BasicBlock::new(&mut ctx, None, vec![ty]);
    body.insert_at_back(loop_op.body_region(&ctx), &ctx);
    let body_value = body.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, body_value)
        .get_operation()
        .insert_at_back(body, &ctx);
    let result_value = loop_op.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, result_value)
        .get_operation()
        .insert_at_back(entry, &ctx);
    let text = mlir::export(&ctx, &module).unwrap();
    assert_eq!(text.matches("\"stablehlo.return\"").count(), 2);
    assert_eq!(text.matches("\"func.return\"").count(), 1);
    assert!(text.contains("\"stablehlo.while\""));
    assert_eq!(
        text,
        String::try_from(MlirPrinter::new(&ctx, &module.get_operation())).unwrap()
    );
    native_verify(&text);
}

#[test]
fn authored_compute_options_have_mlir_names_and_storage() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[1, 4, 4, 2], ElementType::F32);
    let kernel = input(&mut ctx, &[3, 3, 2, 3], ElementType::F32);
    let result = tensor(&ctx, &[1, 2, 2, 3], ElementType::F32);
    let conv = ConvolutionOp::new(
        &mut ctx,
        x,
        kernel,
        result,
        ConvolutionOptions {
            dimensions: ConvDimensionNumbersAttr {
                input_batch_dimension: 0,
                input_feature_dimension: 3,
                input_spatial_dimensions: vec![1, 2],
                kernel_input_feature_dimension: 2,
                kernel_output_feature_dimension: 3,
                kernel_spatial_dimensions: vec![0, 1],
                output_batch_dimension: 0,
                output_feature_dimension: 3,
                output_spatial_dimensions: vec![1, 2],
            },
            strides: vec![1, 1],
            padding: vec![0; 4],
            lhs_dilation: vec![1, 1],
            rhs_dilation: vec![1, 1],
            feature_group_count: 1,
            batch_group_count: 1,
            window_reversal: vec![false, true],
            precision: vec![Precision::Default, Precision::High],
        },
    );
    let text = check(&ctx, conv);
    assert!(text.contains("feature_group_count = 1 : i64"), "{text}");
    assert!(
        text.contains("window_reversal = array<i1: false, true>"),
        "{text}"
    );
    assert!(!text.contains("stablehlo_compute"));
    let a = input(&mut ctx, &[2, 2], ElementType::F32);
    let b = input(&mut ctx, &[2, 1], ElementType::F32);
    let ty = tensor(&ctx, &[2, 1], ElementType::F32);
    let solve = TriangularSolveOp::new(
        &mut ctx,
        a,
        b,
        ty,
        TriangularSolveOptions {
            left_side: true,
            lower: true,
            unit_diagonal: false,
            transpose: TransposeMode::Adjoint,
        },
    );
    assert!(check(&ctx, solve).contains("#stablehlo<transpose ADJOINT>"));
    let x = input(&mut ctx, &[2, 3], ElementType::F32);
    let scale = input(&mut ctx, &[3], ElementType::F32);
    let ty = tensor(&ctx, &[2, 3], ElementType::F32);
    let op = BatchNormInferenceOp::new(
        &mut ctx,
        BatchNormInferenceOperands {
            x,
            scale,
            offset: scale,
            mean: scale,
            variance: scale,
        },
        ty,
        BatchNormOptions {
            feature_index: 1,
            epsilon: 0.001,
        },
    );
    assert!(check(&ctx, op).contains("epsilon = "));
}

#[test]
fn authored_custom_call_and_layouts_export() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2, 3], ElementType::F32);
    let ty = x.get_type(&ctx);
    let op = CustomCallOp::new(&mut ctx, "target".into(), vec![x], vec![ty], 4);
    op.set_backend_config(&mut ctx, Box::new(DictAttr::new(vec![])));
    op.set_attr_stablehlo_custom_call_operand_layouts(
        &ctx,
        ArrayOfLayoutAttr {
            layouts: vec![LayoutAttr {
                minor_to_major: vec![1, 0],
            }],
        },
    );
    op.set_attr_stablehlo_custom_call_result_layouts(
        &ctx,
        ArrayOfLayoutAttr {
            layouts: vec![LayoutAttr {
                minor_to_major: vec![1, 0],
            }],
        },
    );
    op.set_attr_stablehlo_custom_call_result_tilings(
        &ctx,
        ArrayOfTilingAttr {
            tilings: vec![TilingAttr {
                tiles: vec![
                    IndexTensorAttr { values: vec![2, 1] },
                    IndexTensorAttr {
                        values: vec![-1, 1],
                    },
                ],
            }],
        },
    );
    let text = check(&ctx, op);
    assert!(text.contains("call_target_name = \"target\""));
    assert!(text.contains("tensor<2xindex>"));
    assert!(text.contains(
        "result_tilings = [[dense<[2, 1]> : tensor<2xindex>, dense<[-1, 1]> : tensor<2xindex>]]"
    ));
}

#[test]
fn authored_collectives_use_dense_groups_and_unit_flags() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let result = tensor(&ctx, &[4], ElementType::F32);
    let replica_groups = groups(&ctx);
    let op = AllGatherOp::new(&mut ctx, vec![x], vec![result], 0, replica_groups);
    op.set_attr_stablehlo_all_gather_use_global_device_ids(
        &ctx,
        stablehlo::pliron::builtin::attributes::BoolAttr::new(true),
    );
    op.set_attr_stablehlo_all_gather_channel_handle(
        &ctx,
        ChannelHandleAttr {
            handle: 1,
            channel_kind: ChannelKind::DeviceToDevice,
        },
    );
    let text = check(&ctx, op);
    assert!(
        text.contains("replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>"),
        "{text}"
    );
    assert!(text.contains("use_global_device_ids = unit"), "{text}");
}

#[test]
fn all_to_all_native_only_flag_is_not_silently_discarded() {
    use stablehlo::pliron::builtin::attributes::BoolAttr;
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let replicas = groups(&ctx);
    let op = AllToAllOp::new(&mut ctx, vec![x], vec![ty], 0, 0, 2, replicas);
    op.set_attr_stablehlo_all_to_all_channel_handle(
        &ctx,
        ChannelHandleAttr {
            handle: 1,
            channel_kind: ChannelKind::DeviceToDevice,
        },
    );
    op.set_attr_stablehlo_all_to_all_use_global_device_ids(&ctx, BoolAttr::new(true));
    let error = String::try_from(MlirPrinter::new(&ctx, &op.get_operation())).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("no MLIR use_global_device_ids field"),
        "{error}"
    );
    op.set_attr_stablehlo_all_to_all_use_global_device_ids(&ctx, BoolAttr::new(false));
    let text = check(&ctx, op);
    assert!(!text.contains("use_global_device_ids"));
}

#[test]
fn framework_interfaces_cover_primitive_types_and_dialect_attributes() {
    let ctx = Context::new();
    for element in [
        ElementType::F32,
        ElementType::I64,
        ElementType::C64,
        ElementType::F4E2M1Fn,
    ] {
        let ty = tensor(&ctx, &[-1, 4], element);
        assert_eq!(
            String::try_from(MlirPrinter::new(&ctx, &ty)).unwrap(),
            assembly::print_type(&ctx, ty).unwrap()
        );
    }

    let attr = DotDimensionNumbersAttr {
        lhs_batching_dimensions: vec![],
        rhs_batching_dimensions: vec![],
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![0],
    };
    let attr: &dyn Attribute = &attr;
    assert_eq!(
        String::try_from(MlirPrinter::new(&ctx, attr)).unwrap(),
        assembly::print_attribute(&ctx, attr).unwrap()
    );
    let ty: TypeHandle = FutureType::get(&ctx, vec![tensor(&ctx, &[2], ElementType::F32)]).into();
    assert_eq!(
        assembly::parse_type(&ctx, &assembly::print_type(&ctx, ty).unwrap()).unwrap(),
        ty
    );
}

macro_rules! unary_exports {
    ($input:ident => $result:ident; $($test:ident: $op:ident),* $(,)?) => {$(
        #[test]
        fn $test() {
            let mut ctx = Context::new();
            let value = input(&mut ctx, &[2], ElementType::$input);
            let ty = tensor(&ctx, &[2], ElementType::$result);
            let op = $op::new(&mut ctx, value, ty);
            check(&ctx, op);
        }
    )*};
}

macro_rules! binary_exports {
    ($input:ident => $result:ident; $($test:ident: $op:ident),* $(,)?) => {$(
        #[test]
        fn $test() {
            let mut ctx = Context::new();
            let value = input(&mut ctx, &[2], ElementType::$input);
            let ty = tensor(&ctx, &[2], ElementType::$result);
            let op = $op::new(&mut ctx, value, value, ty);
            check(&ctx, op);
        }
    )*};
}

unary_exports!(F32 => F32;
    abs: AbsOp, cbrt: CbrtOp, ceil: CeilOp, cosine: CosineOp,
    exponential: ExponentialOp, exponential_minus_one: ExponentialMinusOneOp,
    exp2: Exp2Op, floor: FloorOp, log: LogOp, log_plus_one: LogPlusOneOp,
    log2: Log2Op, logistic: LogisticOp, negate: NegateOp,
    round_nearest_afz: RoundNearestAfzOp, round_nearest_even: RoundNearestEvenOp,
    rsqrt: RsqrtOp, sign: SignOp, sine: SineOp, sqrt: SqrtOp, tan: TanOp,
    tanh: TanhOp, convert: ConvertOp, bitcast_convert: BitcastConvertOp,
);
unary_exports!(I32 => I32;
    count_leading_zeros: CountLeadingZerosOp, not: NotOp, population_count: PopulationCountOp,
);
unary_exports!(C64 => F32; real: RealOp, imag: ImagOp);
unary_exports!(F32 => I1; is_finite: IsFiniteOp);
binary_exports!(F32 => F32;
    add: AddOp, atan2: Atan2Op, divide: DivideOp, maximum: MaximumOp,
    minimum: MinimumOp, multiply: MultiplyOp, power: PowerOp,
    remainder: RemainderOp, subtract: SubtractOp,
);
binary_exports!(I32 => I32;
    and: AndOp, or: OrOp, xor: XorOp, shift_left: ShiftLeftOp,
    shift_right_arithmetic: ShiftRightArithmeticOp, shift_right_logical: ShiftRightLogicalOp,
);
binary_exports!(F32 => C64; complex: ComplexOp);

macro_rules! constant_exports {
    ($($test:ident: $element:ident => $words:expr),* $(,)?) => {$(
        #[test]
        fn $test() {
            let mut ctx = Context::new();
            let ty = tensor(&ctx, &[2], ElementType::$element);
            let op = ConstantOp::new(&mut ctx, ty, $words, true);
            check(&ctx, op);
        }
    )*};
}

constant_exports!(
    constant_i1: I1 => vec![1],
    constant_i8: I8 => vec![0xff],
    constant_i64: I64 => vec![u64::MAX],
    constant_ui64: Ui64 => vec![u64::MAX],
    constant_f16: F16 => vec![0x8001],
    constant_bf16: Bf16 => vec![0x7f81],
    constant_f32: F32 => vec![0x7f80_0001],
    constant_f64: F64 => vec![0x7ff0_0000_0000_0001],
    constant_c64: C64 => vec![0x8000_0000_7f80_0001],
    constant_c128: C128 => vec![0x7ff0_0000_0000_0001, 0x8000_0000_0000_0000],
    constant_f4: F4E2M1Fn => vec![0x0f],
);

#[test]
fn authored_shape_catalog_exports() {
    let mut ctx = Context::new();
    let x = input(&mut ctx, &[2], ElementType::F32);
    let ty = x.get_type(&ctx);
    let scalar = input(&mut ctx, &[], ElementType::F32);
    let index = input(&mut ctx, &[], ElementType::I32);
    let shape = input(&mut ctx, &[1], ElementType::I64);
    let scalar_i32 = tensor(&ctx, &[], ElementType::I32);
    let wide = tensor(&ctx, &[4], ElementType::F32);
    macro_rules! check_op {
        ($op:expr) => {{
            let op = $op;
            check(&ctx, op);
        }};
    }
    check_op!(ReshapeOp::new(&mut ctx, x, ty));
    check_op!(TransposeOp::new(&mut ctx, x, ty, vec![0]));
    check_op!(ReverseOp::new(&mut ctx, x, ty, vec![0]));
    check_op!(BroadcastInDimOp::new(&mut ctx, scalar, ty, vec![]));
    check_op!(BroadcastOp::new(&mut ctx, scalar, ty, vec![2]));
    check_op!(DynamicBroadcastInDimOp::new(
        &mut ctx,
        x,
        shape,
        ty,
        vec![0],
        vec![],
        vec![]
    ));
    check_op!(DynamicReshapeOp::new(&mut ctx, x, shape, ty));
    check_op!(GetDimensionSizeOp::new(&mut ctx, x, scalar_i32, 0));
    check_op!(SetDimensionSizeOp::new(&mut ctx, x, index, ty, 0));
    check_op!(SliceOp::new(&mut ctx, x, ty, vec![0], vec![2], vec![1]));
    check_op!(ConcatenateOp::new(&mut ctx, vec![x, x], wide, 0));
    check_op!(DynamicSliceOp::new(&mut ctx, x, vec![index], ty, vec![2]));
    check_op!(DynamicUpdateSliceOp::new(&mut ctx, x, x, vec![index], ty));
    check_op!(RealDynamicSliceOp::new(
        &mut ctx, x, shape, shape, shape, ty
    ));
    check_op!(PadOp::new(
        &mut ctx,
        x,
        scalar,
        ty,
        vec![0],
        vec![0],
        vec![0]
    ));
    check_op!(DynamicPadOp::new(
        &mut ctx, x, scalar, shape, shape, shape, ty
    ));
}
