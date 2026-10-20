use crate::verification::invalid;
use crate::{attributes::*, types::*};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{BoolAttr, IntegerAttr, VecAttr},
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};

#[pliron_op(name="stablehlo.constant",format,interfaces=[pliron::builtin::op_interfaces::NOpdsInterface<0>,pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_constant_bits:VecAttr,stablehlo_constant_splat:BoolAttr,stablehlo_constant_sparse:SparseElementsAttr))]
pub struct ConstantOp;
impl ConstantOp {
    pub fn new(ctx: &mut Context, result: TypeHandle, bits: Vec<u64>, splat: bool) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![],
                vec![],
                0,
            ),
        };
        let packed = integer_list(ctx, bits);
        op.set_attr_stablehlo_constant_bits(ctx, packed);
        op.set_attr_stablehlo_constant_splat(ctx, BoolAttr::new(splat));
        op
    }
    pub fn new_sparse(
        ctx: &mut Context,
        result: TypeHandle,
        indices: Vec<Vec<i64>>,
        values: Vec<AttrObj>,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![],
                vec![],
                0,
            ),
        };
        let result_object = result.deref(ctx);
        let storage_type = if let Some(result_tensor) = result_object.downcast_ref::<TensorType>() {
            let storage_element = result_tensor
                .element
                .deref(ctx)
                .downcast_ref::<QuantizedElementType>()
                .map_or(result_tensor.element, |quantized| quantized.storage);
            let storage_type =
                TensorType::get(ctx, result_tensor.dimensions.clone(), storage_element);
            if let Some(encoding) = result_tensor.encoding().cloned() {
                storage_type.deref(ctx).with_encoding(ctx, encoding)
            } else {
                storage_type.into()
            }
        } else {
            result
        };
        op.set_attr_stablehlo_constant_sparse(
            ctx,
            SparseElementsAttr {
                ty: storage_type,
                indices: indices
                    .into_iter()
                    .map(|values| IndexTensorAttr { values })
                    .collect(),
                values: VecAttr::new(values),
            },
        );
        op
    }
    pub fn sparse_value(&self, ctx: &Context) -> Option<SparseElementsAttr> {
        self.get_attr_stablehlo_constant_sparse(ctx)
            .map(|attribute| (*attribute).clone())
    }
    pub fn bits(&self, ctx: &Context) -> Option<Vec<u64>> {
        read_integer_list(&*self.get_attr_stablehlo_constant_bits(ctx)?)
    }
    pub fn is_splat(&self, ctx: &Context) -> Option<bool> {
        self.get_attr_stablehlo_constant_splat(ctx)
            .map(|v| bool::from((*v).clone()))
    }
}
impl Verify for ConstantOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 0 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "constant arity mismatch");
        }
        let ty = ir.get_result(0).get_type(ctx).deref(ctx);
        let Some(tensor) = ty.downcast_ref::<TensorType>() else {
            return invalid(self, ctx, "constant result must be ranked tensor");
        };
        tensor.verify(ctx)?;
        if tensor.dimensions.iter().any(|&d| d < 0) {
            return invalid(self, ctx, "constant result must have static shape");
        }
        let Some(count) = element_count(&tensor.dimensions) else {
            return invalid(self, ctx, "constant element count overflow");
        };
        if !crate::verification::elementwise::valid_stablehlo_element(ctx, tensor.element) {
            return invalid(self, ctx, "constant element type is unsupported");
        }
        if let Some(sparse) = self.sparse_value(ctx) {
            if self.bits(ctx).is_some() || self.is_splat(ctx).is_some() {
                return invalid(
                    self,
                    ctx,
                    "constant cannot contain dense and sparse payloads",
                );
            }
            let storage_matches = if let Some(q) = tensor
                .element
                .deref(ctx)
                .downcast_ref::<QuantizedElementType>()
            {
                let storage = sparse.ty.deref(ctx);
                storage.downcast_ref::<TensorType>().is_some_and(|storage| {
                    storage.element == q.storage
                        && storage.dimensions == tensor.dimensions
                        && storage.encoding() == tensor.encoding()
                })
            } else {
                sparse.ty == ir.get_result(0).get_type(ctx)
            };
            if !storage_matches {
                return invalid(
                    self,
                    ctx,
                    "sparse constant payload type differs from result type",
                );
            }
            sparse.verify(ctx)?;
            return Ok(());
        }
        let Some(bits) = self.bits(ctx) else {
            return invalid(self, ctx, "constant requires typed dense element bits");
        };
        let Some(splat) = self.is_splat(ctx) else {
            return invalid(self, ctx, "constant requires splat flag");
        };
        let Some(storage_width) = tensor
            .element
            .deref(ctx)
            .downcast_ref::<QuantizedElementType>()
            .and_then(|q| q.storage_type(ctx))
            .or_else(|| primitive_element(ctx, tensor.element))
            .and_then(ElementType::bit_width)
        else {
            return invalid(self, ctx, "constant element width is unsupported");
        };
        let words_per_element = storage_width.div_ceil(64) as usize;
        let element_values = if splat { 1 } else { count };
        let Some(expected) = element_values.checked_mul(words_per_element) else {
            return invalid(self, ctx, "constant payload word count overflow");
        };
        if bits.len() != expected || (splat && bits.is_empty()) {
            return invalid(
                self,
                ctx,
                "dense constant element count does not match result shape",
            );
        }
        let used_bits = storage_width % 64;
        if used_bits != 0
            && bits
                .chunks(words_per_element)
                .any(|words| words.last().is_some_and(|word| word >> used_bits != 0))
        {
            return invalid(
                self,
                ctx,
                "constant payload has nonzero bits beyond element width",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.dynamic_iota",format,interfaces=[pliron::builtin::op_interfaces::NOpdsInterface<1>,pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_dynamic_iota_dimension:IntegerAttr))]
pub struct DynamicIotaOp;
impl DynamicIotaOp {
    pub fn dimension_size(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn new(
        ctx: &mut Context,
        dimension_size: Value,
        result: TypeHandle,
        dimension: usize,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![dimension_size],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_dynamic_iota_dimension(ctx, integer_attr(ctx, dimension as u64));
        op
    }
    pub fn dimension(&self, ctx: &Context) -> Option<usize> {
        usize::try_from(
            self.get_attr_stablehlo_dynamic_iota_dimension(ctx)?
                .value()
                .to_string_decimal(false)
                .parse::<u64>()
                .ok()?,
        )
        .ok()
    }
    pub fn new_inferred(
        ctx: &mut Context,
        dimension_size: Value,
        result_shape: &[i64],
        element: ElementType,
        dimension: usize,
    ) -> Result<Self> {
        Ok(Self::new(
            ctx,
            dimension_size,
            tensor(ctx, result_shape, element),
            dimension,
        ))
    }
}
impl Verify for DynamicIotaOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let dim = self.dimension(ctx);
        crate::verification::elementwise::verify_iota(self, ctx, dim, true)
    }
}
