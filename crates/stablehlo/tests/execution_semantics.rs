use stablehlo::text::{parse_module, print_module};

#[path = "upstream_support/execution.rs"]
mod execution;
mod upstream_support;
use execution::Tensor;
#[path = "upstream_support/structure.rs"]
mod structure;

struct Case {
    name: String,
    source: String,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
}

fn cases() -> Vec<Case> {
    let mut cases = vec![Case {
        name: "arithmetic/reshape/transpose/slice".into(),
        source: "func.func @main(%a:tensor<4xf32>,%b:tensor<4xf32>)->tensor<2x1xf32> {
            %sum=stablehlo.add %a,%b:tensor<4xf32>
            %product=stablehlo.multiply %sum,%a:tensor<4xf32>
            %matrix=stablehlo.reshape %product:(tensor<4xf32>)->tensor<2x2xf32>
            %transpose=stablehlo.transpose %matrix,dims=[1,0]:(tensor<2x2xf32>)->tensor<2x2xf32>
            %slice=stablehlo.slice %transpose [0:2,0:1]:(tensor<2x2xf32>)->tensor<2x1xf32>
            return %slice:tensor<2x1xf32> }"
            .into(),
        inputs: vec![
            Tensor::f32(&[4], &[1., 2., 3., 4.]),
            Tensor::f32(&[4], &[4., 3., 2., 1.]),
        ],
        outputs: vec![Tensor::f32(&[2, 1], &[5., 10.])],
    }];
    for start in [-5_i32, 0, 3, 9] {
        let clamped = start.clamp(0, 3);
        cases.push(Case {
            name:format!("dynamic_slice/{start}"),
            source: "func.func @main(%x:tensor<6xi32>,%s:tensor<i32>)->tensor<3xi32> {
                %r=\"stablehlo.dynamic_slice\"(%x,%s){slice_sizes=array<i64:3>}:(tensor<6xi32>,tensor<i32>)->tensor<3xi32>
                return %r:tensor<3xi32> }".into(),
            inputs:vec![Tensor::i32(&[6],&[0,1,2,3,4,5]),Tensor::i32(&[],&[start])],
            outputs:vec![Tensor::i32(&[3],&[clamped,clamped+1,clamped+2])],
        });
    }
    for start in [-1_i32, 3, 8] {
        let offset = start.clamp(0, 4) as usize;
        let mut output = vec![0, 1, 2, 3, 4, 5];
        output[offset..offset + 2].copy_from_slice(&[10, 11]);
        cases.push(Case {
            name:format!("dynamic_update_slice/{start}"),
            source: "func.func @main(%x:tensor<6xi32>,%u:tensor<2xi32>,%s:tensor<i32>)->tensor<6xi32> {
                %r=\"stablehlo.dynamic_update_slice\"(%x,%u,%s):(tensor<6xi32>,tensor<2xi32>,tensor<i32>)->tensor<6xi32>
                return %r:tensor<6xi32> }".into(),
            inputs:vec![Tensor::i32(&[6],&[0,1,2,3,4,5]),Tensor::i32(&[2],&[10,11]),Tensor::i32(&[],&[start])],
            outputs:vec![Tensor::i32(&[6],&output)],
        });
    }
    let lhs = [1_f32, 2., 3., 4., 5., 6.];
    let rhs = [7_f32, 8., 9., 10., 11., 12.];
    let dot = (0..2)
        .flat_map(|row| {
            (0..2).map(move |column| {
                (0..3)
                    .map(|axis| lhs[row * 3 + axis] * rhs[axis * 2 + column])
                    .sum()
            })
        })
        .collect::<Vec<f32>>();
    cases.push(Case {
        name: "matrix dot".into(),
        source: "func.func @main(%a:tensor<2x3xf32>,%b:tensor<3x2xf32>)->tensor<2x2xf32> {
            %r=stablehlo.dot %a,%b:(tensor<2x3xf32>,tensor<3x2xf32>)->tensor<2x2xf32>
            return %r:tensor<2x2xf32> }"
            .into(),
        inputs: vec![Tensor::f32(&[2, 3], &lhs), Tensor::f32(&[3, 2], &rhs)],
        outputs: vec![Tensor::f32(&[2, 2], &dot)],
    });
    cases.push(Case {
        name: "reduce with runtime initial value".into(),
        source: "func.func @main(%a:tensor<2x3xi32>,%init:tensor<i32>)->tensor<2xi32> {
            %r=\"stablehlo.reduce\"(%a,%init)({
                ^bb0(%a0:tensor<i32>,%b0:tensor<i32>):
                %sum=stablehlo.add %a0,%b0:tensor<i32>
                stablehlo.return %sum:tensor<i32>
            }){dimensions=array<i64:1>}:(tensor<2x3xi32>,tensor<i32>)->tensor<2xi32>
            return %r:tensor<2xi32> }"
            .into(),
        inputs: vec![
            Tensor::i32(&[2, 3], &[1, 2, 3, 4, 5, 6]),
            Tensor::i32(&[], &[10]),
        ],
        outputs: vec![Tensor::i32(&[2], &[16, 25])],
    });
    cases.push(Case {
        name:"padding and reversal".into(),
        source:"func.func @main(%a:tensor<3xi32>,%v:tensor<i32>)->tensor<6xi32> {
            %pad=\"stablehlo.pad\"(%a,%v){edge_padding_low=array<i64:1>,edge_padding_high=array<i64:0>,interior_padding=array<i64:1>}:(tensor<3xi32>,tensor<i32>)->tensor<6xi32>
            %r=\"stablehlo.reverse\"(%pad){dimensions=array<i64:0>}:(tensor<6xi32>)->tensor<6xi32>
            return %r:tensor<6xi32> }".into(),
        inputs:vec![Tensor::i32(&[3],&[1,2,3]),Tensor::i32(&[],&[-1])],
        outputs:vec![Tensor::i32(&[6],&[3,-1,2,-1,1,-1])],
    });
    cases.push(Case {
        name: "multi-result control-flow state".into(),
        source: "func.func @main(%limit:tensor<i32>)->(tensor<i32>,tensor<i32>) {
            %zero=stablehlo.constant dense<0>:tensor<i32>
            %one=stablehlo.constant dense<1>:tensor<i32>
            %r:2=\"stablehlo.while\"(%zero,%zero)({
                ^bb0(%i:tensor<i32>,%sum:tensor<i32>):
                %p=stablehlo.compare LT,%i,%limit:(tensor<i32>,tensor<i32>)->tensor<i1>
                stablehlo.return %p:tensor<i1>
            },{
                ^bb0(%i:tensor<i32>,%sum:tensor<i32>):
                %next=stablehlo.add %i,%one:tensor<i32>
                %total=stablehlo.add %sum,%next:tensor<i32>
                stablehlo.return %next,%total:tensor<i32>,tensor<i32>
            }):(tensor<i32>,tensor<i32>)->(tensor<i32>,tensor<i32>)
            return %r#0,%r#1:tensor<i32>,tensor<i32> }"
            .into(),
        inputs: vec![Tensor::i32(&[], &[5])],
        outputs: vec![Tensor::i32(&[], &[5]), Tensor::i32(&[], &[15])],
    });
    let image = (0..16).map(|value| value as f32).collect::<Vec<_>>();
    let convolution = (0..3)
        .flat_map(|row| {
            let image = &image;
            (0..3).map(move |column| {
                image[row * 4 + column]
                    + image[row * 4 + column + 1]
                    + image[(row + 1) * 4 + column]
                    + image[(row + 1) * 4 + column + 1]
            })
        })
        .collect::<Vec<_>>();
    cases.push(Case {
        name: "convolution".into(),
        source:
            "func.func @main(%x:tensor<1x1x4x4xf32>,%k:tensor<1x1x2x2xf32>)->tensor<1x1x3x3xf32> {
            %r=\"stablehlo.convolution\"(%x,%k){
                dimension_numbers=#stablehlo.conv<[b,f,0,1]x[o,i,0,1]->[b,f,0,1]>,
                feature_group_count=1:i64,batch_group_count=1:i64
            }:(tensor<1x1x4x4xf32>,tensor<1x1x2x2xf32>)->tensor<1x1x3x3xf32>
            return %r:tensor<1x1x3x3xf32> }"
                .into(),
        inputs: vec![
            Tensor::f32(&[1, 1, 4, 4], &image),
            Tensor::f32(&[1, 1, 2, 2], &[1., 1., 1., 1.]),
        ],
        outputs: vec![Tensor::f32(&[1, 1, 3, 3], &convolution)],
    });
    cases.push(Case {
        name:"gather rows".into(),
        source:"func.func @main(%x:tensor<4x3xi32>,%indices:tensor<2x1xi32>)->tensor<2x3xi32> {
            %r=\"stablehlo.gather\"(%x,%indices){
                dimension_numbers=#stablehlo.gather<offset_dims=[1],collapsed_slice_dims=[0],start_index_map=[0],index_vector_dim=1>,
                slice_sizes=array<i64:1,3>
            }:(tensor<4x3xi32>,tensor<2x1xi32>)->tensor<2x3xi32>
            return %r:tensor<2x3xi32> }".into(),
        inputs:vec![Tensor::i32(&[4,3],&(1..=12).collect::<Vec<_>>()),Tensor::i32(&[2,1],&[2,0])],
        outputs:vec![Tensor::i32(&[2,3],&[7,8,9,1,2,3])],
    });
    cases.push(Case {
        name:"scatter-add".into(),
        source:"func.func @main(%x:tensor<6xi32>,%indices:tensor<2x1xi32>,%updates:tensor<2xi32>)->tensor<6xi32> {
            %r=\"stablehlo.scatter\"(%x,%indices,%updates)({
                ^bb0(%old:tensor<i32>,%new:tensor<i32>):
                %sum=stablehlo.add %old,%new:tensor<i32>
                stablehlo.return %sum:tensor<i32>
            }){
                scatter_dimension_numbers=#stablehlo.scatter<update_window_dims=[],inserted_window_dims=[0],scatter_dims_to_operand_dims=[0],index_vector_dim=1>
            }:(tensor<6xi32>,tensor<2x1xi32>,tensor<2xi32>)->tensor<6xi32>
            return %r:tensor<6xi32> }".into(),
        inputs:vec![Tensor::i32(&[6],&[0,1,2,3,4,5]),Tensor::i32(&[2,1],&[1,4]),Tensor::i32(&[2],&[10,20])],
        outputs:vec![Tensor::i32(&[6],&[0,11,2,3,24,5])],
    });
    cases.push(Case {
        name:"key/payload sort".into(),
        source:"func.func @main(%keys:tensor<4xi32>,%payload:tensor<4xi32>)->(tensor<4xi32>,tensor<4xi32>) {
            %r:2=\"stablehlo.sort\"(%keys,%payload)({
                ^bb0(%a:tensor<i32>,%b:tensor<i32>,%pa:tensor<i32>,%pb:tensor<i32>):
                %less=stablehlo.compare LT,%a,%b:(tensor<i32>,tensor<i32>)->tensor<i1>
                stablehlo.return %less:tensor<i1>
            }){dimension=0:i64,is_stable=true}:(tensor<4xi32>,tensor<4xi32>)->(tensor<4xi32>,tensor<4xi32>)
            return %r#0,%r#1:tensor<4xi32>,tensor<4xi32> }".into(),
        inputs:vec![Tensor::i32(&[4],&[3,1,4,2]),Tensor::i32(&[4],&[30,10,40,20])],
        outputs:vec![Tensor::i32(&[4],&[1,2,3,4]),Tensor::i32(&[4],&[10,20,30,40])],
    });
    cases.push(Case {
        name: "scalar map".into(),
        source: "func.func @main(%x:tensor<4xi32>)->tensor<4xi32> {
            %r=\"stablehlo.map\"(%x)({
                ^bb0(%a:tensor<i32>):
                %square=stablehlo.multiply %a,%a:tensor<i32>
                stablehlo.return %square:tensor<i32>
            }){dimensions=array<i64:0>}:(tensor<4xi32>)->tensor<4xi32>
            return %r:tensor<4xi32> }"
            .into(),
        inputs: vec![Tensor::i32(&[4], &[-2, -1, 0, 3])],
        outputs: vec![Tensor::i32(&[4], &[4, 1, 0, 9])],
    });
    cases.push(Case {
        name:"real FFT projection".into(),
        source:"func.func @main(%x:tensor<2xf32>)->tensor<2xf32> {
            %fft=\"stablehlo.fft\"(%x){fft_type=#stablehlo<fft_type RFFT>,fft_length=array<i64:2>}:(tensor<2xf32>)->tensor<2xcomplex<f32>>
            %r=stablehlo.real %fft:(tensor<2xcomplex<f32>>)->tensor<2xf32>
            return %r:tensor<2xf32> }".into(),
        inputs:vec![Tensor::f32(&[2],&[1.,3.])],
        outputs:vec![Tensor::f32(&[2],&[4.,-2.])],
    });
    for selector in [-1, 0, 1, 2, 10] {
        let expected = match selector {
            0 => 7,
            1 => -7,
            _ => 8,
        };
        cases.push(Case {
            name: format!("case selector/{selector}"),
            source: "func.func @main(%selector:tensor<i32>,%x:tensor<i32>)->tensor<i32> {
                %r=\"stablehlo.case\"(%selector)({
                    stablehlo.return %x:tensor<i32>
                },{
                    %negative=stablehlo.negate %x:tensor<i32>
                    stablehlo.return %negative:tensor<i32>
                },{
                    %one=stablehlo.constant dense<1>:tensor<i32>
                    %next=stablehlo.add %x,%one:tensor<i32>
                    stablehlo.return %next:tensor<i32>
                }):(tensor<i32>)->tensor<i32>
                return %r:tensor<i32> }"
                .into(),
            inputs: vec![Tensor::i32(&[], &[selector]), Tensor::i32(&[], &[7])],
            outputs: vec![Tensor::i32(&[], &[expected])],
        });
    }
    let bits = [0x00000000_u32, 0x80000000, 0x7fc12345, 0x7f800000];
    cases.push(Case {
        name: "bitcast preserves signed zeros/infinity/NaN payload".into(),
        source: "func.func @main(%x:tensor<4xf32>)->tensor<4xi32> {
            %r=stablehlo.bitcast_convert %x:(tensor<4xf32>)->tensor<4xi32>
            return %r:tensor<4xi32> }"
            .into(),
        inputs: vec![Tensor::f32(&[4], &bits.map(f32::from_bits))],
        outputs: vec![Tensor::i32(&[4], &bits.map(|bits| bits as i32))],
    });
    cases.push(Case {
        name: "float-to-integer conversion rounds toward zero".into(),
        source: "func.func @main(%x:tensor<4xf32>)->tensor<4xi32> {
            %r=stablehlo.convert %x:(tensor<4xf32>)->tensor<4xi32>
            return %r:tensor<4xi32> }"
            .into(),
        inputs: vec![Tensor::f32(&[4], &[-2.9, 1.5, 0.9, 3.1])],
        outputs: vec![Tensor::i32(&[4], &[-2, 1, 0, 3])],
    });
    cases.push(Case {
        name:"broadcast in dimension".into(),
        source:"func.func @main(%x:tensor<2xi32>)->tensor<3x2xi32> {
            %r=\"stablehlo.broadcast_in_dim\"(%x){broadcast_dimensions=array<i64:1>}:(tensor<2xi32>)->tensor<3x2xi32>
            return %r:tensor<3x2xi32> }".into(),
        inputs:vec![Tensor::i32(&[2],&[5,7])],
        outputs:vec![Tensor::i32(&[3,2],&[5,7,5,7,5,7])],
    });
    cases.push(Case {
        name: "windowed reduction".into(),
        source: "func.func @main(%x:tensor<4xi32>,%init:tensor<i32>)->tensor<3xi32> {
            %r=\"stablehlo.reduce_window\"(%x,%init)({
                ^bb0(%a:tensor<i32>,%b:tensor<i32>):
                %sum=stablehlo.add %a,%b:tensor<i32>
                stablehlo.return %sum:tensor<i32>
            }){window_dimensions=array<i64:2>}:(tensor<4xi32>,tensor<i32>)->tensor<3xi32>
            return %r:tensor<3xi32> }"
            .into(),
        inputs: vec![Tensor::i32(&[4], &[1, 2, 3, 4]), Tensor::i32(&[], &[0])],
        outputs: vec![Tensor::i32(&[3], &[3, 5, 7])],
    });
    cases.push(Case {
        name:"clamp and predicate selection".into(),
        source:"func.func @main(%x:tensor<4xi32>,%low:tensor<i32>,%high:tensor<i32>)->tensor<4xi32> {
            %clamped=\"stablehlo.clamp\"(%low,%x,%high):(tensor<i32>,tensor<4xi32>,tensor<i32>)->tensor<4xi32>
            %zero=stablehlo.constant dense<0>:tensor<4xi32>
            %positive=stablehlo.compare GT,%x,%zero:(tensor<4xi32>,tensor<4xi32>)->tensor<4xi1>
            %negative=stablehlo.negate %clamped:tensor<4xi32>
            %r=\"stablehlo.select\"(%positive,%clamped,%negative):(tensor<4xi1>,tensor<4xi32>,tensor<4xi32>)->tensor<4xi32>
            return %r:tensor<4xi32> }".into(),
        inputs:vec![Tensor::i32(&[4],&[-4,-1,0,3]),Tensor::i32(&[],&[-2]),Tensor::i32(&[],&[2])],
        outputs:vec![Tensor::i32(&[4],&[2,1,0,2])],
    });
    cases.push(Case {
        name:"tuple construction and projection".into(),
        source:"func.func @main(%x:tensor<2xi32>,%y:tensor<i32>)->tensor<2xi32> {
            %tuple=\"stablehlo.tuple\"(%x,%y):(tensor<2xi32>,tensor<i32>)->tuple<tensor<2xi32>,tensor<i32>>
            %r=\"stablehlo.get_tuple_element\"(%tuple){index=0:i32}:(tuple<tensor<2xi32>,tensor<i32>>)->tensor<2xi32>
            return %r:tensor<2xi32> }".into(),
        inputs:vec![Tensor::i32(&[2],&[9,10]),Tensor::i32(&[],&[11])],
        outputs:vec![Tensor::i32(&[2],&[9,10])],
    });
    cases
}

#[test]
fn semantic_cases_verify_and_roundtrip_before_independent_execution() {
    let cases = cases();
    assert_eq!(cases.len(), 29);
    for case in cases {
        let module =
            parse_module(&case.source).unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert_eq!(
            parse_module(&print_module(&module).unwrap()).unwrap(),
            module,
            "{}",
            case.name
        );
        assert!(
            !case.inputs.is_empty(),
            "execution tests must use runtime inputs"
        );
        assert!(!case.outputs.is_empty());
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib and its CPU client"]
fn rust_emitted_modules_execute_with_runtime_inputs_in_independent_xla_cpu() {
    let cases = cases();
    let requests = cases
        .iter()
        .map(|case| {
            (
                print_module(&parse_module(&case.source).unwrap()).unwrap(),
                case.inputs.clone(),
            )
        })
        .collect::<Vec<_>>();
    for (case, outputs) in cases.iter().zip(execution::execute(&requests)) {
        assert_eq!(outputs, case.outputs, "{}", case.name);
    }
    let modules = requests
        .into_iter()
        .map(|(module, _)| module)
        .collect::<Vec<_>>();
    for result in upstream_support::verify(&modules) {
        assert!(matches!(
            result,
            upstream_support::UpstreamResult::Accepted(_)
        ));
    }
}

#[test]
fn runtime_semantics_cases_preserve_ir_through_rust_bytecode_for_all_versions() {
    use stablehlo::bytecode::{read_module, write_module};
    let cases = cases();
    assert_eq!(cases.len(), 29);
    for case in cases {
        let expected = parse_module(&case.source).unwrap();
        for version in 0..=6 {
            let bytes = write_module(&expected, version).unwrap();
            let actual = read_module(&bytes)
                .unwrap_or_else(|error| panic!("{} / v{version}: {error}", case.name));
            structure::assert_equivalent(&expected, &actual);
        }
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE and STABLEHLO_UPSTREAM_PYTHON with pinned CPU client"]
fn rust_written_bytecode_executes_with_independent_runtime_inputs_at_every_version() {
    use stablehlo::bytecode::write_module;
    use upstream_support::UpstreamResult;
    let cases = cases();
    let mut labels = Vec::new();
    let mut bytes = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let module = parse_module(&case.source).unwrap();
        for version in 0..=6 {
            labels.push((index, version));
            bytes.push(write_module(&module, version).unwrap());
        }
    }
    let requests = labels
        .iter()
        .zip(upstream_support::verify_native_bytecode(&bytes))
        .map(|(&(index, version), result)| {
            let assembly = match result {
                UpstreamResult::Accepted(assembly) => assembly,
                UpstreamResult::Rejected(error) => {
                    panic!("{} / v{version}: {error}", cases[index].name)
                }
            };
            (assembly, cases[index].inputs.clone())
        })
        .collect::<Vec<_>>();
    for (&(index, version), outputs) in labels.iter().zip(execution::execute(&requests)) {
        assert_eq!(
            outputs, cases[index].outputs,
            "{} / v{version}",
            cases[index].name
        );
    }
    eprintln!(
        "29 runtime-input modules written in Rust at seven MLIR bytecode versions (203 executions), read by native1.22 and executed by the independent XLA CPU client with exact dtype/shape/byte comparisons."
    );
}
