//! Small, independent tensor executor for the emitted scan/ragged-dot subset.
use super::*;
use clienthlo::pliron::{basic_block::BasicBlock, r#type::Typed, value::Value};
use stablehlo::{ops as hlo, types::TensorType};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f64>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            shape: shape.to_vec(),
            data,
        }
    }
    fn scalar(value: f64) -> Self {
        Self::new(&[], vec![value])
    }
    fn index(&self, coordinates: &[usize]) -> usize {
        self.shape
            .iter()
            .zip(coordinates)
            .fold(0, |offset, (extent, i)| offset * extent + i)
    }
    fn at(&self, coordinates: &[usize]) -> f64 {
        self.data[self.index(coordinates)]
    }
    fn integers(&self) -> Vec<usize> {
        self.data.iter().map(|&v| v as usize).collect()
    }
}

fn coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for (dim, &extent) in shape.iter().enumerate().rev() {
        coordinates[dim] = index % extent;
        index /= extent;
    }
    coordinates
}

fn materialize(shape: Vec<usize>, value: impl Fn(&[usize]) -> f64) -> Tensor {
    let data = (0..shape.iter().product())
        .map(|i| value(&coordinates(i, &shape)))
        .collect();
    Tensor { shape, data }
}

pub(super) fn execute(ctx: &Context, fun: &FuncOp, inputs: &[Tensor]) -> Vec<Tensor> {
    execute_block(ctx, fun.entry_block(ctx).unwrap(), &HashMap::new(), inputs)
}

fn execute_block(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    captures: &HashMap<Value, Tensor>,
    inputs: &[Tensor],
) -> Vec<Tensor> {
    let mut values = captures.clone();
    values.extend(block.deref(ctx).arguments().zip(inputs.iter().cloned()));
    for pointer in block.deref(ctx).iter(ctx) {
        let ir = pointer.deref(ctx);
        let name = Operation::get_opid(pointer, ctx).to_string();
        let operands: Vec<_> = ir.operands().map(|v| values[&v].clone()).collect();
        if name == "stablehlo.return" {
            return operands;
        }
        let result_type = ir.get_result(0).get_type(ctx);
        let object = result_type.deref(ctx);
        let ty = object.downcast_ref::<TensorType>().unwrap();
        let static_shape = || {
            ty.dimensions
                .iter()
                .map(|&d| usize::try_from(d).unwrap())
                .collect::<Vec<_>>()
        };
        let output = match name.as_str() {
            "stablehlo.constant" => {
                let op = hlo::ConstantOp::from_operation(pointer);
                let bits = op.bits(ctx).unwrap();
                let float =
                    stablehlo::types::primitive_element(ctx, ty.element) == Some(ElementType::F64);
                let data: Vec<_> = bits
                    .into_iter()
                    .map(|bits| {
                        if float {
                            f64::from_bits(bits)
                        } else {
                            bits as i64 as f64
                        }
                    })
                    .collect();
                let shape = static_shape();
                let data = if data.len() == 1 {
                    vec![data[0]; shape.iter().product()]
                } else {
                    data
                };
                vec![Tensor::new(&shape, data)]
            }
            "stablehlo.convert" => vec![operands[0].clone()],
            "stablehlo.get_dimension_size" => {
                let axis = hlo::GetDimensionSizeOp::from_operation(pointer)
                    .dimension(ctx)
                    .unwrap();
                vec![Tensor::scalar(operands[0].shape[axis] as f64)]
            }
            "stablehlo.reshape" | "stablehlo.dynamic_reshape" => {
                let shape = if operands.len() == 2 {
                    operands[1].integers()
                } else {
                    static_shape()
                };
                vec![Tensor::new(&shape, operands[0].data.clone())]
            }
            "stablehlo.concatenate" => {
                let axis = hlo::ConcatenateOp::from_operation(pointer)
                    .dimension(ctx)
                    .unwrap();
                let mut shape = operands[0].shape.clone();
                shape[axis] = operands.iter().map(|v| v.shape[axis]).sum();
                vec![materialize(shape, |coordinate| {
                    let mut coordinate = coordinate.to_vec();
                    for operand in &operands {
                        if coordinate[axis] < operand.shape[axis] {
                            return operand.at(&coordinate);
                        }
                        coordinate[axis] -= operand.shape[axis];
                    }
                    panic!("concatenation index is outside operands")
                })]
            }
            "stablehlo.broadcast_in_dim" | "stablehlo.dynamic_broadcast_in_dim" => {
                let (shape, axes) = if operands.len() == 2 {
                    (
                        operands[1].integers(),
                        hlo::DynamicBroadcastInDimOp::from_operation(pointer)
                            .get_attr_stablehlo_dynamic_broadcast_dimensions(ctx).unwrap().0.iter()
                            .map(|v| v.downcast_ref::<clienthlo::pliron::builtin::attributes::IntegerAttr>()
                                .unwrap().value().to_u64() as usize).collect(),
                    )
                } else {
                    (
                        static_shape(),
                        hlo::BroadcastInDimOp::from_operation(pointer)
                            .dimensions(ctx)
                            .unwrap(),
                    )
                };
                vec![materialize(shape, |coordinate| {
                    let source: Vec<_> = axes
                        .iter()
                        .enumerate()
                        .map(|(i, &axis)| {
                            if operands[0].shape[i] == 1 {
                                0
                            } else {
                                coordinate[axis]
                            }
                        })
                        .collect();
                    operands[0].at(&source)
                })]
            }
            "stablehlo.dynamic_iota" => {
                let axis = hlo::DynamicIotaOp::from_operation(pointer)
                    .dimension(ctx)
                    .unwrap();
                vec![materialize(operands[0].integers(), |coordinate| {
                    coordinate[axis] as f64
                })]
            }
            "stablehlo.real_dynamic_slice" => {
                let start = operands[1].integers();
                let limit = operands[2].integers();
                let stride = operands[3].integers();
                let shape = start
                    .iter()
                    .zip(&limit)
                    .zip(&stride)
                    .map(|((&a, &b), &s)| (b - a).div_ceil(s))
                    .collect();
                vec![materialize(shape, |coordinate| {
                    let source: Vec<_> = coordinate
                        .iter()
                        .enumerate()
                        .map(|(i, &c)| start[i] + stride[i] * c)
                        .collect();
                    operands[0].at(&source)
                })]
            }
            "stablehlo.dynamic_update_slice" => {
                let mut output = operands[0].clone();
                let start: Vec<_> = operands[2..].iter().map(|v| v.data[0] as usize).collect();
                for (i, &value) in operands[1].data.iter().enumerate() {
                    let coordinate: Vec<_> = coordinates(i, &operands[1].shape)
                        .iter()
                        .zip(&start)
                        .map(|(a, b)| a + b)
                        .collect();
                    let index = output.index(&coordinate);
                    output.data[index] = value;
                }
                vec![output]
            }
            "stablehlo.while" => {
                let op = hlo::WhileOp::from_operation(pointer);
                let cond = op
                    .condition_region(ctx)
                    .deref(ctx)
                    .iter(ctx)
                    .next()
                    .unwrap();
                let body = op.body_region(ctx).deref(ctx).iter(ctx).next().unwrap();
                let mut carried = operands;
                let mut iterations = 0;
                while execute_block(ctx, cond, &values, &carried)[0].data[0] != 0.0 {
                    carried = execute_block(ctx, body, &values, &carried);
                    iterations += 1;
                    assert!(iterations <= 100, "test loop did not converge");
                }
                carried
            }
            "stablehlo.dot_general" => vec![dot(ctx, pointer, &operands[0], &operands[1])],
            "stablehlo.select" => {
                let shape = operands[1].shape.clone();
                vec![materialize(shape, |c| {
                    if operands[0].at(if operands[0].shape.is_empty() { &[] } else { c }) != 0.0 {
                        operands[1].at(c)
                    } else {
                        operands[2].at(c)
                    }
                })]
            }
            "stablehlo.add" | "stablehlo.subtract" | "stablehlo.multiply" | "stablehlo.and"
            | "stablehlo.compare" => {
                vec![materialize(operands[0].shape.clone(), |c| {
                    let a = operands[0].at(c);
                    let b = operands[1].at(c);
                    match name.as_str() {
                        "stablehlo.add" => a + b,
                        "stablehlo.subtract" => a - b,
                        "stablehlo.multiply" => a * b,
                        "stablehlo.and" => f64::from(a != 0.0 && b != 0.0),
                        _ => {
                            use stablehlo::attributes::ComparisonDirection::*;
                            f64::from(
                                match hlo::CompareOp::from_operation(pointer)
                                    .direction(ctx)
                                    .unwrap()
                                {
                                    Eq => a == b,
                                    Ne => a != b,
                                    Lt => a < b,
                                    Le => a <= b,
                                    Gt => a > b,
                                    Ge => a >= b,
                                },
                            )
                        }
                    }
                })]
            }
            other => panic!("missing tensor test executor for {other}"),
        };
        assert_eq!(output.len(), ir.get_num_results());
        values.extend(ir.results().zip(output));
    }
    panic!("block has no return")
}

fn dot(ctx: &Context, pointer: Ptr<Operation>, lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let d = hlo::DotGeneralOp::from_operation(pointer)
        .dimension_numbers(ctx)
        .unwrap()
        .clone();
    let left_free: Vec<_> = (0..lhs.shape.len())
        .filter(|i| {
            !d.lhs_batching_dimensions.contains(&(*i as i64))
                && !d.lhs_contracting_dimensions.contains(&(*i as i64))
        })
        .collect();
    let right_free: Vec<_> = (0..rhs.shape.len())
        .filter(|i| {
            !d.rhs_batching_dimensions.contains(&(*i as i64))
                && !d.rhs_contracting_dimensions.contains(&(*i as i64))
        })
        .collect();
    let mut shape: Vec<_> = d
        .lhs_batching_dimensions
        .iter()
        .map(|&i| lhs.shape[i as usize])
        .collect();
    shape.extend(left_free.iter().map(|&i| lhs.shape[i]));
    shape.extend(right_free.iter().map(|&i| rhs.shape[i]));
    let contract_shape: Vec<_> = d
        .lhs_contracting_dimensions
        .iter()
        .map(|&i| lhs.shape[i as usize])
        .collect();
    materialize(shape, |coordinate| {
        let mut left = vec![0; lhs.shape.len()];
        let mut right = vec![0; rhs.shape.len()];
        for (i, (&a, &b)) in d
            .lhs_batching_dimensions
            .iter()
            .zip(&d.rhs_batching_dimensions)
            .enumerate()
        {
            left[a as usize] = coordinate[i];
            right[b as usize] = coordinate[i];
        }
        let offset = d.lhs_batching_dimensions.len();
        for (i, &axis) in left_free.iter().enumerate() {
            left[axis] = coordinate[offset + i];
        }
        let offset = offset + left_free.len();
        for (i, &axis) in right_free.iter().enumerate() {
            right[axis] = coordinate[offset + i];
        }
        (0..contract_shape.iter().product())
            .map(|i| {
                let indices = coordinates(i, &contract_shape);
                for (i, (&a, &b)) in d
                    .lhs_contracting_dimensions
                    .iter()
                    .zip(&d.rhs_contracting_dimensions)
                    .enumerate()
                {
                    left[a as usize] = indices[i];
                    right[b as usize] = indices[i];
                }
                lhs.at(&left) * rhs.at(&right)
            })
            .sum()
    })
}
