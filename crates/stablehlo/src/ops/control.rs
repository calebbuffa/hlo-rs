use crate::verification::invalid;
use crate::{attributes::*, types::*};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{BoolAttr, DictAttr, IntegerAttr, StringAttr, VecAttr},
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};
use pliron::{
    builtin::attributes::TypeAttr as PlironTypeAttr, identifier::Identifier, region::Region,
};

#[pliron_op(name="stablehlo.return",format="operands(CharSpace(`,`))",interfaces=[pliron::builtin::op_interfaces::NResultsInterface<0>,pliron::builtin::op_interfaces::IsTerminatorInterface])]
pub struct ReturnOp;
impl ReturnOp {
    pub fn new(ctx: &mut Context, value: Value) -> Self {
        Self::new_many(ctx, vec![value])
    }
    pub fn new_many(ctx: &mut Context, values: Vec<Value>) -> Self {
        Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], values, vec![], 0),
        }
    }
}
impl Verify for ReturnOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_num_results() != 0 {
            return invalid(self, ctx, "return cannot produce SSA results");
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.module",format="`module`",interfaces=[pliron::builtin::op_interfaces::OneRegionInterface])]
pub struct ModuleOp;
impl ModuleOp {
    pub fn new(ctx: &mut Context) -> Self {
        Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1),
        }
    }
    pub fn region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}
impl Verify for ModuleOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 0 || ir.get_num_results() != 0 {
            return invalid(self, ctx, "module cannot have operands or results");
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.call",format="attr($stablehlo_call_callee, $StringAttr) operands(CharSpace(`,`))",attributes=(stablehlo_call_callee:StringAttr))]
pub struct CallOp;
impl CallOp {
    pub fn new(
        ctx: &mut Context,
        callee: String,
        args: Vec<Value>,
        results: Vec<TypeHandle>,
    ) -> Self {
        let op = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), results, args, vec![], 0),
        };
        op.set_attr_stablehlo_call_callee(ctx, StringAttr::new(callee));
        op
    }
}
impl Verify for CallOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_attr_stablehlo_call_callee(ctx).is_none() {
            return invalid(self, ctx, "call requires a symbol reference");
        }
        let ir = self.get_operation().deref(ctx);
        if ir
            .operands()
            .any(|v| tensor_parts(ctx, v.get_type(ctx)).is_none())
            || ir
                .results()
                .any(|v| tensor_parts(ctx, v.get_type(ctx)).is_none())
        {
            return invalid(
                self,
                ctx,
                "call operands and results must be supported ranked tensors",
            );
        }
        Ok(())
    }
}

#[pliron_op(
    name = "stablehlo.while",
    format = "operands(CharSpace(`,`)) regions(CharSpace(`,`))"
)]
pub struct WhileOp;
impl WhileOp {
    pub fn new(ctx: &mut Context, carried: Vec<Value>, result_types: Vec<TypeHandle>) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                result_types,
                carried,
                vec![],
                2,
            ),
        }
    }
    pub fn condition_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
    pub fn body_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(1)
    }
}
impl Verify for WhileOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != ir.get_num_results()
            || ir.operands().any(|v| !is_value_type(ctx, v.get_type(ctx)))
            || ir.results().any(|v| !is_value_type(ctx, v.get_type(ctx)))
        {
            return invalid(
                self,
                ctx,
                "while carried operand/result tensor types must agree",
            );
        }
        for i in 0..ir.get_num_results() {
            if ir.get_operand(i).get_type(ctx) != ir.get_result(i).get_type(ctx) {
                return invalid(self, ctx, "while carried operand/result types must match");
            }
        }
        if ir.regions().count() != 2 {
            return invalid(self, ctx, "while requires condition and body regions");
        }
        let carried: Vec<_> = ir.operands().map(|v| v.get_type(ctx)).collect();
        let condition_result = [tensor(ctx, &[], ElementType::I1)];
        if !region_signature(ir.get_region(0), ctx, &carried, &condition_result)
            || !region_signature(ir.get_region(1), ctx, &carried, &carried)
        {
            return invalid(
                self,
                ctx,
                "while regions must accept and return the carried types with a scalar i1 condition",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.func",format="`func`",interfaces=[pliron::builtin::op_interfaces::OneRegionInterface],attributes=(stablehlo_function_name: StringAttr, stablehlo_function_signature: PlironTypeAttr))]
pub struct FuncOp;
impl FuncOp {
    pub fn new(ctx: &mut Context) -> Self {
        Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1),
        }
    }
    pub fn named(ctx: &mut Context, name: String, signature: TypeHandle) -> Self {
        let op = Self::new(ctx);
        op.set_attr_stablehlo_function_name(ctx, StringAttr::new(name));
        op.set_attr_stablehlo_function_signature(ctx, PlironTypeAttr::new(signature));
        op
    }
    pub fn region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}
impl Verify for FuncOp {
    fn verify(&self, _: &Context) -> Result<()> {
        Ok(())
    }
}

#[pliron_op(
    name = "stablehlo.if",
    format = "operands(CharSpace(`,`)) regions(CharSpace(`,`))"
)]
pub struct IfOp;
impl IfOp {
    pub fn new(ctx: &mut Context, predicate: Value, results: Vec<TypeHandle>) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                vec![predicate],
                vec![],
                2,
            ),
        }
    }
    pub fn true_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
    pub fn false_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(1)
    }
}
impl Verify for IfOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1
            || !scalar_element_is(ctx, ir.get_operand(0).get_type(ctx), ElementType::I1)
        {
            return invalid(self, ctx, "if requires a scalar i1 predicate and results");
        }
        for region in ir.regions() {
            if !region_returns(
                region,
                ctx,
                0,
                ir.results()
                    .map(|v| v.get_type(ctx))
                    .collect::<Vec<_>>()
                    .as_slice(),
                false,
            ) {
                return invalid(
                    self,
                    ctx,
                    "if branch must have one argument-free block returning compatible result types",
                );
            }
        }
        if ir.regions().count() != 2 {
            return invalid(self, ctx, "if requires two regions");
        }
        Ok(())
    }
}

#[pliron_op(
    name = "stablehlo.case",
    format = "operands(CharSpace(`,`)) regions(CharSpace(`,`))"
)]
pub struct CaseOp;
impl CaseOp {
    pub fn new(
        ctx: &mut Context,
        selector: Value,
        results: Vec<TypeHandle>,
        branch_count: usize,
    ) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                vec![selector],
                vec![],
                branch_count,
            ),
        }
    }
}
impl Verify for CaseOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1
            || !scalar_element_is(ctx, ir.get_operand(0).get_type(ctx), ElementType::I32)
            || ir.regions().count() == 0
        {
            return invalid(self, ctx, "case requires scalar i32 selector and branches");
        }
        let expected: Vec<_> = ir.results().map(|v| v.get_type(ctx)).collect();
        for region in ir.regions() {
            if !region_returns(region, ctx, 0, &expected, false) {
                return invalid(self, ctx, "case branch must return compatible result types");
            }
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.tuple",format="operands(CharSpace(`,`))",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct TupleOp;
impl TupleOp {
    pub fn new(ctx: &mut Context, values: Vec<Value>) -> Self {
        let ty = TupleType::get(ctx, values.iter().map(|v| v.get_type(ctx)).collect()).into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty],
                values,
                vec![],
                0,
            ),
        }
    }
}
impl Verify for TupleOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_results() != 1 {
            return invalid(self, ctx, "tuple has one result");
        }
        let tuple_ty = ir.get_result(0).get_type(ctx);
        let tuple_obj = tuple_ty.deref(ctx);
        let Some(tuple) = tuple_obj.downcast_ref::<TupleType>() else {
            return invalid(self, ctx, "tuple result must have tuple type");
        };
        if tuple.elements.len() != ir.get_num_operands()
            || tuple
                .elements
                .iter()
                .zip(ir.operands())
                .any(|(a, b)| *a != b.get_type(ctx))
        {
            return invalid(self, ctx, "tuple element types must match operands");
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.get_tuple_element",format="$0 attr($stablehlo_get_tuple_element_index, $IntegerAttr)",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_get_tuple_element_index:IntegerAttr))]
pub struct GetTupleElementOp;
impl GetTupleElementOp {
    pub fn new(ctx: &mut Context, value: Value, index: u32, result: TypeHandle) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![value],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_get_tuple_element_index(
            ctx,
            typed_integer_attr(ctx, index as u64, 32),
        );
        op
    }
}
impl Verify for GetTupleElementOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        let Some(index) = self
            .get_attr_stablehlo_get_tuple_element_index(ctx)
            .and_then(|i| i.value().to_string_decimal(false).parse::<usize>().ok())
        else {
            return invalid(self, ctx, "tuple index required");
        };
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "tuple projection arity mismatch");
        }
        let operand_ty = ir.get_operand(0).get_type(ctx);
        let operand_obj = operand_ty.deref(ctx);
        let Some(tuple) = operand_obj.downcast_ref::<TupleType>() else {
            return invalid(self, ctx, "operand must be tuple");
        };
        if tuple.elements.get(index) != Some(&ir.get_result(0).get_type(ctx)) {
            return invalid(self, ctx, "tuple index or result type invalid");
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.composite",format="operands(CharSpace(`,`)) regions(CharSpace(`,`))",attributes=(stablehlo_composite_name:StringAttr,stablehlo_composite_version:IntegerAttr,stablehlo_composite_decomposition:StringAttr,stablehlo_composite_attributes:DictAttr))]
pub struct CompositeOp;
impl CompositeOp {
    pub fn new(
        ctx: &mut Context,
        name: String,
        version: u32,
        decomposition: String,
        operands: Vec<Value>,
        results: Vec<TypeHandle>,
        region_count: usize,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                operands,
                vec![],
                region_count,
            ),
        };
        op.set_attr_stablehlo_composite_name(ctx, StringAttr::new(name));
        op.set_attr_stablehlo_composite_version(ctx, typed_integer_attr(ctx, version as u64, 32));
        op.set_attr_stablehlo_composite_decomposition(ctx, StringAttr::new(decomposition));
        op.set_attr_stablehlo_composite_attributes(ctx, DictAttr::new(vec![]));
        op
    }
}
impl Verify for CompositeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_num_results() == 0
            || self
                .get_attr_stablehlo_composite_name(ctx)
                .is_none_or(|n| n.as_str().is_empty())
            || self
                .get_attr_stablehlo_composite_decomposition(ctx)
                .is_none_or(|n| n.as_str().is_empty())
        {
            return invalid(self, ctx, "composite requires name and results");
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.custom_call",format="operands(CharSpace(`,`))",attributes=(stablehlo_custom_call_target:StringAttr,stablehlo_custom_call_has_side_effect:BoolAttr,stablehlo_custom_call_api_version:IntegerAttr,stablehlo_custom_call_output_operand_aliases:VecAttr,stablehlo_custom_call_operand_layouts:ArrayOfLayoutAttr,stablehlo_custom_call_result_layouts:ArrayOfLayoutAttr))]
pub struct CustomCallOp;
static CUSTOM_CALL_BACKEND_CONFIG: std::sync::LazyLock<Identifier> =
    std::sync::LazyLock::new(|| Identifier::new("stablehlo_custom_call_backend_config"));
impl CustomCallOp {
    /// Set the native string/bytes (untyped FFI) or dictionary (typed FFI) configuration.
    pub fn set_backend_config(&self, ctx: &mut Context, config: AttrObj) {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .insert(CUSTOM_CALL_BACKEND_CONFIG.clone(), config);
    }
    pub fn backend_config<'a>(&self, ctx: &'a Context) -> Option<std::cell::Ref<'a, AttrObj>> {
        std::cell::Ref::filter_map(self.get_operation().deref(ctx), |op| {
            op.attributes.0.get(&*CUSTOM_CALL_BACKEND_CONFIG)
        })
        .ok()
    }
}

impl CustomCallOp {
    pub fn new(
        ctx: &mut Context,
        target: String,
        operands: Vec<Value>,
        results: Vec<TypeHandle>,
        api_version: u32,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                operands,
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_custom_call_target(ctx, StringAttr::new(target));
        op.set_attr_stablehlo_custom_call_has_side_effect(ctx, BoolAttr::new(false));
        op.set_attr_stablehlo_custom_call_api_version(
            ctx,
            typed_integer_attr(ctx, api_version as u64, 32),
        );
        op.set_attr_stablehlo_custom_call_output_operand_aliases(ctx, VecAttr::new(vec![]));
        op.set_attr_stablehlo_custom_call_operand_layouts(
            ctx,
            ArrayOfLayoutAttr { layouts: vec![] },
        );
        op.set_attr_stablehlo_custom_call_result_layouts(
            ctx,
            ArrayOfLayoutAttr { layouts: vec![] },
        );
        op
    }
}
impl Verify for CustomCallOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if self
            .get_attr_stablehlo_custom_call_target(ctx)
            .is_none_or(|x| x.as_str().is_empty())
        {
            return invalid(self, ctx, "custom_call requires nonempty target name");
        }
        let api = self
            .get_attr_stablehlo_custom_call_api_version(ctx)
            .and_then(|a| a.value().to_string_decimal(false).parse::<u32>().ok());
        if api.is_none_or(|v| v > 4) {
            return invalid(self, ctx, "custom_call api_version must be between 0 and 4");
        }
        if let Some(config) = self.backend_config(ctx) {
            let dictionary = config.downcast_ref::<DictAttr>().is_some();
            let string = config.downcast_ref::<StringAttr>().is_some()
                || config.downcast_ref::<BytesAttr>().is_some();
            if (api == Some(4) && !dictionary) || (api != Some(4) && !string) {
                return invalid(
                    self,
                    ctx,
                    "custom_call backend_config must be a dictionary for typed FFI and string/bytes for untyped FFI",
                );
            }
            crate::attributes::verify_attribute(ctx, &config)?;
        }
        let operand_layouts = self
            .get_attr_stablehlo_custom_call_operand_layouts(ctx)
            .map(|a| a.layouts.clone())
            .unwrap_or_default();
        let result_layouts = self
            .get_attr_stablehlo_custom_call_result_layouts(ctx)
            .map(|a| a.layouts.clone())
            .unwrap_or_default();
        if operand_layouts.is_empty() != result_layouts.is_empty() {
            return invalid(
                self,
                ctx,
                "custom_call operand and result layouts must be supplied together",
            );
        }
        if !operand_layouts.is_empty() {
            let result_types: Vec<_> = if ir.get_num_results() == 1 {
                let ty = ir.get_result(0).get_type(ctx);
                let obj = ty.deref(ctx);
                obj.downcast_ref::<TupleType>()
                    .map(|t| t.elements.clone())
                    .unwrap_or_else(|| vec![ty])
            } else {
                ir.results().map(|v| v.get_type(ctx)).collect()
            };
            if operand_layouts.len() != ir.get_num_operands()
                || result_layouts.len() != result_types.len()
            {
                return invalid(
                    self,
                    ctx,
                    "custom_call layout counts must match operands and results",
                );
            }
            for (ty, layout) in ir
                .operands()
                .map(|v| v.get_type(ctx))
                .zip(&operand_layouts)
                .chain(result_types.iter().copied().zip(&result_layouts))
            {
                let object = ty.deref(ctx);
                let rank = object
                    .downcast_ref::<TensorType>()
                    .map(|t| t.rank())
                    .or_else(|| object.downcast_ref::<TupleType>().map(|_| usize::MAX));
                let values = &layout.minor_to_major;
                if rank.is_none_or(|r| r != values.len()) || {
                    let mut seen = std::collections::HashSet::new();
                    values
                        .iter()
                        .any(|&v| v < 0 || v as usize >= values.len() || !seen.insert(v))
                } {
                    return invalid(
                        self,
                        ctx,
                        "custom_call layout must be a valid rank permutation",
                    );
                }
            }
        }
        if let Some(aliases) = self.get_attr_stablehlo_custom_call_output_operand_aliases(ctx) {
            let mut outputs = std::collections::HashSet::new();
            for item in &aliases.0 {
                let Some(alias) = item.downcast_ref::<OutputOperandAliasAttr>() else {
                    return invalid(
                        self,
                        ctx,
                        "output_operand_aliases must contain typed aliases",
                    );
                };
                let Some(input) = usize::try_from(alias.operand_index).ok().and_then(|i| {
                    tuple_path(
                        ctx,
                        ir.get_operand(i).get_type(ctx),
                        &alias.operand_tuple_indices,
                    )
                }) else {
                    return invalid(self, ctx, "output_operand_aliases operand path is invalid");
                };
                let output = if ir.get_num_results() == 1 {
                    tuple_path(
                        ctx,
                        ir.get_result(0).get_type(ctx),
                        &alias.output_tuple_indices,
                    )
                } else {
                    match alias.output_tuple_indices.split_first() {
                        Some((&n, path)) => usize::try_from(n)
                            .ok()
                            .and_then(|i| ir.results().nth(i))
                            .and_then(|v| tuple_path(ctx, v.get_type(ctx), path)),
                        None => None,
                    }
                };
                if output != Some(input) || !outputs.insert(alias.output_tuple_indices.clone()) {
                    return invalid(
                        self,
                        ctx,
                        "output_operand_aliases paths must be unique and types must agree",
                    );
                }
            }
        }
        Ok(())
    }
}

#[pliron_op(
    name = "stablehlo.optimization_barrier",
    format = "operands(CharSpace(`,`))"
)]
pub struct OptimizationBarrierOp;
impl OptimizationBarrierOp {
    pub fn new(ctx: &mut Context, values: Vec<Value>) -> Self {
        let types = values.iter().map(|v| v.get_type(ctx)).collect();
        Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), types, values, vec![], 0),
        }
    }
}
impl Verify for OptimizationBarrierOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != ir.get_num_results()
            || ir.operands().zip(ir.results()).any(|(a, b)| {
                a.get_type(ctx) != b.get_type(ctx) || !is_value_type(ctx, a.get_type(ctx))
            })
        {
            return invalid(
                self,
                ctx,
                "optimization barrier preserves value operand types",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.create_token",format="",interfaces=[pliron::builtin::op_interfaces::NOpdsInterface<0>,pliron::builtin::op_interfaces::OneResultInterface])]
pub struct CreateTokenOp;
impl CreateTokenOp {
    pub fn new(ctx: &mut Context) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![TokenType::get(ctx).into()],
                vec![],
                vec![],
                0,
            ),
        }
    }
}
impl Verify for CreateTokenOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_operation().deref(ctx).get_num_operands() != 0
            || !is_token(
                ctx,
                self.get_operation().deref(ctx).get_result(0).get_type(ctx),
            )
        {
            return invalid(self, ctx, "create_token produces one token");
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.after_all",format="operands(CharSpace(`,`))",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct AfterAllOp;
impl AfterAllOp {
    pub fn new(ctx: &mut Context, tokens: Vec<Value>) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![TokenType::get(ctx).into()],
                tokens,
                vec![],
                0,
            ),
        }
    }
}
impl Verify for AfterAllOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_results() != 1 {
            return invalid(self, ctx, "after_all produces one token");
        }
        crate::verification::control::verify_token_values(self, ctx)
    }
}

fn typed_integer_attr(ctx: &Context, value: u64, width: u32) -> IntegerAttr {
    let ty = IntegerType::get(ctx, width, Signedness::Signless);
    IntegerAttr::new(ty, APInt::from_u64(value, bw(width as usize)))
}
fn scalar_element_is(ctx: &Context, ty: TypeHandle, e: ElementType) -> bool {
    tensor_parts(ctx, ty).is_some_and(|(shape, element)| shape.is_empty() && element == e)
}
fn valid_transfer_channel(attr: Option<&ChannelHandleAttr>, expected: i64) -> bool {
    attr.is_some_and(|channel| channel.channel_type == expected)
}
fn is_token(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx).downcast_ref::<TokenType>().is_some()
}
fn region_returns(
    region: Ptr<Region>,
    ctx: &Context,
    args: usize,
    expected: &[TypeHandle],
    allow_empty: bool,
) -> bool {
    use pliron::linked_list::ContainsLinkedList;
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    if blocks.len() != 1 {
        return false;
    }
    let block = blocks[0].deref(ctx);
    if block.get_num_arguments() != args {
        return false;
    }
    let ops: Vec<_> = block.iter(ctx).collect();
    let Some(last) = ops.last() else {
        return allow_empty;
    };
    if Operation::get_op_dyn(*last, ctx).get_opid().to_string() != "stablehlo.return" {
        return false;
    }
    let ret = last.deref(ctx);
    ret.get_num_operands() == expected.len()
        && ret
            .operands()
            .zip(expected)
            .all(|(v, t)| compatible_type(ctx, v.get_type(ctx), *t))
}
fn compatible_type(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    if a == b {
        return true;
    }
    crate::inference::compatible_tensor_types(ctx, a, b)
}
fn tuple_path(ctx: &Context, mut ty: TypeHandle, path: &[i64]) -> Option<TypeHandle> {
    for &index in path {
        let obj = ty.deref(ctx);
        let tuple = obj.downcast_ref::<TupleType>()?;
        ty = *tuple.elements.get(usize::try_from(index).ok()?)?;
    }
    Some(ty)
}
fn region_signature(
    region: Ptr<Region>,
    ctx: &Context,
    args: &[TypeHandle],
    results: &[TypeHandle],
) -> bool {
    use pliron::linked_list::ContainsLinkedList;
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    if blocks.len() != 1 {
        return false;
    }
    let block = blocks[0].deref(ctx);
    let actual_args: Vec<_> = block.arguments().map(|v| v.get_type(ctx)).collect();
    actual_args == args && region_returns(region, ctx, args.len(), results, false)
}

#[pliron_op(name="stablehlo.replica_id",format="",interfaces=[pliron::builtin::op_interfaces::NOpdsInterface<0>,pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ReplicaIdOp;
impl ReplicaIdOp {
    pub fn new(ctx: &mut Context) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![tensor(ctx, &[], ElementType::I32)],
                vec![],
                vec![],
                0,
            ),
        }
    }
}
impl Verify for ReplicaIdOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if !scalar_element_is(
            ctx,
            self.get_operation().deref(ctx).get_result(0).get_type(ctx),
            ElementType::I32,
        ) {
            return invalid(self, ctx, "replica_id returns scalar i32");
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.partition_id",format="",interfaces=[pliron::builtin::op_interfaces::NOpdsInterface<0>,pliron::builtin::op_interfaces::OneResultInterface])]
pub struct PartitionIdOp;
impl PartitionIdOp {
    pub fn new(ctx: &mut Context) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![tensor(ctx, &[], ElementType::I32)],
                vec![],
                vec![],
                0,
            ),
        }
    }
}
impl Verify for PartitionIdOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if !scalar_element_is(
            ctx,
            self.get_operation().deref(ctx).get_result(0).get_type(ctx),
            ElementType::I32,
        ) {
            return invalid(self, ctx, "partition_id returns scalar i32");
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.infeed",format="$0 attr($stablehlo_infeed_config, $StringAttr) attr($stablehlo_infeed_layout, $ArrayOfLayoutAttr)",attributes=(stablehlo_infeed_config:StringAttr,stablehlo_infeed_layout:ArrayOfLayoutAttr))]
pub struct InfeedOp;
impl InfeedOp {
    pub fn new(
        ctx: &mut Context,
        token: Value,
        data: TypeHandle,
        config: String,
        layout: ArrayOfLayoutAttr,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![data, TokenType::get(ctx).into()],
                vec![token],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_infeed_config(ctx, StringAttr::new(config));
        op.set_attr_stablehlo_infeed_layout(ctx, layout);
        op
    }
}
impl Verify for InfeedOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1
            || ir.get_num_results() != 2
            || !is_token(ctx, ir.get_operand(0).get_type(ctx))
            || !is_token(ctx, ir.get_result(1).get_type(ctx))
            || !is_value_type(ctx, ir.get_result(0).get_type(ctx))
        {
            return invalid(self, ctx, "infeed consumes and returns token with value");
        }
        if let Some(layout) = self.get_attr_stablehlo_infeed_layout(ctx)
            && !layout.layouts.is_empty()
        {
            let ty = ir.get_result(0).get_type(ctx);
            let object = ty.deref(ctx);
            let types = object
                .downcast_ref::<TupleType>()
                .map(|tuple| tuple.elements.clone())
                .unwrap_or_else(|| vec![ty]);
            if types.len() != layout.layouts.len() {
                return invalid(self, ctx, "infeed layout count must match data results");
            }
            for (ty, layout) in types.into_iter().zip(&layout.layouts) {
                let object = ty.deref(ctx);
                let Some(tensor) = object.downcast_ref::<TensorType>() else {
                    return invalid(self, ctx, "infeed layouts require ranked tensors");
                };
                let values = &layout.minor_to_major;
                let mut seen = std::collections::HashSet::new();
                if values.len() != tensor.rank()
                    || values.iter().any(|&axis| {
                        axis < 0 || axis as usize >= values.len() || !seen.insert(axis)
                    })
                {
                    return invalid(self, ctx, "infeed layout must be a rank permutation");
                }
            }
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.outfeed",format="operands(CharSpace(`,`)) attr($stablehlo_outfeed_config, $StringAttr)",attributes=(stablehlo_outfeed_config:StringAttr))]
pub struct OutfeedOp;
impl OutfeedOp {
    pub fn new(ctx: &mut Context, value: Value, token: Value, config: String) -> Self {
        let ty = token.get_type(ctx);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty],
                vec![value, token],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_outfeed_config(ctx, StringAttr::new(config));
        op
    }
}
impl Verify for OutfeedOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2
            || ir.get_num_results() != 1
            || !is_value_type(ctx, ir.get_operand(0).get_type(ctx))
            || !is_token(ctx, ir.get_operand(1).get_type(ctx))
            || !is_token(ctx, ir.get_result(0).get_type(ctx))
        {
            return invalid(
                self,
                ctx,
                "outfeed requires data and token and returns token",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.send",format="operands(CharSpace(`,`)) attr($stablehlo_send_channel_handle, $ChannelHandleAttr) attr($stablehlo_send_is_host_transfer, $BoolAttr)",attributes=(stablehlo_send_channel_handle:ChannelHandleAttr,stablehlo_send_is_host_transfer:BoolAttr))]
pub struct SendOp;
impl SendOp {
    pub fn new(
        ctx: &mut Context,
        value: Value,
        token: Value,
        channel: ChannelHandleAttr,
        host: bool,
    ) -> Self {
        let ty = token.get_type(ctx);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty],
                vec![value, token],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_send_channel_handle(ctx, channel);
        op.set_attr_stablehlo_send_is_host_transfer(ctx, BoolAttr::new(host));
        op
    }
}
impl Verify for SendOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2
            || ir.get_num_results() != 1
            || !is_value_type(ctx, ir.get_operand(0).get_type(ctx))
            || !is_token(ctx, ir.get_operand(1).get_type(ctx))
            || !is_token(ctx, ir.get_result(0).get_type(ctx))
        {
            return invalid(self, ctx, "send requires data and token and returns token");
        }
        let host = self
            .get_attr_stablehlo_send_is_host_transfer(ctx)
            .is_some_and(|v| bool::from(v.clone()));
        if !valid_transfer_channel(
            self.get_attr_stablehlo_send_channel_handle(ctx).as_deref(),
            if host { 2 } else { 1 },
        ) {
            return invalid(
                self,
                ctx,
                "send channel handle type does not match host-transfer mode",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.recv",format="$0 attr($stablehlo_recv_channel_handle, $ChannelHandleAttr) attr($stablehlo_recv_is_host_transfer, $BoolAttr)",attributes=(stablehlo_recv_channel_handle:ChannelHandleAttr,stablehlo_recv_is_host_transfer:BoolAttr))]
pub struct RecvOp;
impl RecvOp {
    pub fn new(
        ctx: &mut Context,
        token: Value,
        data: TypeHandle,
        channel: ChannelHandleAttr,
        host: bool,
    ) -> Self {
        let tok = token.get_type(ctx);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![data, tok],
                vec![token],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_recv_channel_handle(ctx, channel);
        op.set_attr_stablehlo_recv_is_host_transfer(ctx, BoolAttr::new(host));
        op
    }
}
impl Verify for RecvOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1
            || ir.get_num_results() != 2
            || !is_token(ctx, ir.get_operand(0).get_type(ctx))
            || !is_value_type(ctx, ir.get_result(0).get_type(ctx))
            || !is_token(ctx, ir.get_result(1).get_type(ctx))
        {
            return invalid(self, ctx, "recv consumes token and returns data and token");
        }
        let host = self
            .get_attr_stablehlo_recv_is_host_transfer(ctx)
            .is_some_and(|v| bool::from(v.clone()));
        if !valid_transfer_channel(
            self.get_attr_stablehlo_recv_channel_handle(ctx).as_deref(),
            if host { 3 } else { 1 },
        ) {
            return invalid(
                self,
                ctx,
                "recv channel handle type does not match host-transfer mode",
            );
        }
        Ok(())
    }
}

#[pliron_op(
    name = "stablehlo.async_start",
    format = "operands(CharSpace(`,`)) regions(CharSpace(`,`))"
)]
pub struct AsyncStartOp;
impl AsyncStartOp {
    pub fn new(ctx: &mut Context, operands: Vec<Value>, future_results: Vec<TypeHandle>) -> Self {
        let state = FutureType::get(ctx, future_results).into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![state],
                operands,
                vec![],
                1,
            ),
        }
    }
    pub fn body_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}
impl Verify for AsyncStartOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_results() != 1 || ir.regions().count() != 1 {
            return invalid(
                self,
                ctx,
                "async_start has one future result and one region",
            );
        }
        let future = ir.get_result(0).get_type(ctx);
        let obj = future.deref(ctx);
        let Some(future) = obj.downcast_ref::<FutureType>() else {
            return invalid(self, ctx, "async_start result must be future");
        };
        let region = ir.get_region(0);
        let operands: Vec<_> = ir.operands().map(|v| v.get_type(ctx)).collect();
        if !region_signature(region, ctx, &operands, &future.elements) {
            return invalid(
                self,
                ctx,
                "async_start body arguments and return must match operands and future results",
            );
        }
        use pliron::linked_list::ContainsLinkedList;
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        let block = blocks[0];
        let body: Vec<_> = block.deref(ctx).iter(ctx).collect();
        let mut body_ops = 0;
        for ptr in body {
            let operation = ptr.deref(ctx);
            let name = Operation::get_op_dyn(ptr, ctx).get_opid().to_string();
            if name == "stablehlo.return" {
                continue;
            }
            if !matches!(
                name.as_str(),
                "stablehlo.slice"
                    | "stablehlo.dynamic_slice"
                    | "stablehlo.all_gather"
                    | "stablehlo.all_reduce"
                    | "stablehlo.reduce_scatter"
                    | "stablehlo.all_to_all"
                    | "stablehlo.collective_broadcast"
                    | "stablehlo.collective_reduce"
                    | "stablehlo.collective_permute"
                    | "stablehlo.cross_replica_sum"
            ) {
                return invalid(
                    self,
                    ctx,
                    "async_start body permits only slice and collective operations",
                );
            }
            body_ops += 1;
            for value in operation.operands() {
                let block_def = value.defining_block();
                let op_def = value
                    .defining_op()
                    .and_then(|p| p.deref(ctx).get_parent_block());
                if block_def.or(op_def) != Some(block) {
                    return invalid(
                        self,
                        ctx,
                        "async_start body cannot capture values from above",
                    );
                }
            }
        }
        if body_ops != 1 {
            return invalid(
                self,
                ctx,
                "async_start body must contain one async operation",
            );
        }
        Ok(())
    }
}

#[pliron_op(name="stablehlo.async_done",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct AsyncDoneOp;
impl AsyncDoneOp {
    pub fn new(ctx: &mut Context, state: Value, results: Vec<TypeHandle>) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                vec![state],
                vec![],
                0,
            ),
        }
    }
}
impl Verify for AsyncDoneOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "async_done requires one future and one result");
        }
        let ty = ir.get_operand(0).get_type(ctx);
        let obj = ty.deref(ctx);
        let Some(future) = obj.downcast_ref::<FutureType>() else {
            return invalid(self, ctx, "async_done operand must be future");
        };
        if future.elements.len() != 1
            || !crate::inference::compatible_tensor_types(
                ctx,
                future.elements[0],
                ir.get_result(0).get_type(ctx),
            )
        {
            return invalid(
                self,
                ctx,
                "async_done result must match its single future element",
            );
        }
        Ok(())
    }
}
