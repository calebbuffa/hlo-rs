use super::{property_for, untranslatable};
use crate::ops::{Im2colOp, MixedIndex};
use pliron::{
    common_traits::Named,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    printable::State,
    result::Result,
    r#type::Typed,
    value::Value,
};
use pliron_mlir_interop::printers::{print_op, print_op_location, print_type, print_type_list};
use std::{collections::BTreeMap, fmt};

pub(super) fn upstream_name(ctx: &Context, op: Ptr<Operation>) -> String {
    Operation::get_op_dyn(op, ctx)
        .get_opid()
        .to_string()
        .replace("iree_linalg_ext.winograd_", "iree_linalg_ext.winograd.")
}

fn values(ctx: &Context, operands: &[Value], f: &mut fmt::Formatter<'_>) -> Result<()> {
    for (i, operand) in operands.iter().enumerate() {
        if i != 0 {
            write!(f, ", ")?;
        }
        write!(f, "%{}", operand.unique_name(ctx))?;
    }
    Ok(())
}

fn typed(
    ctx: &Context,
    keyword: &str,
    operands: &[Value],
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    if operands.is_empty() {
        return Ok(());
    }
    write!(f, " {keyword}(")?;
    values(ctx, operands, f)?;
    write!(f, " : ")?;
    print_type_list(ctx, operands.iter().map(|v| v.get_type(ctx)), state, f)?;
    write!(f, ")")?;
    Ok(())
}

fn region(
    ctx: &Context,
    op: Ptr<Operation>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let raw = op.deref(ctx);
    for region in raw.regions() {
        write!(f, " {{")?;
        {
            let _indent = state.indent();
            for block in region.deref(ctx).iter(ctx) {
                write!(
                    f,
                    "\n{:width$}^{}(",
                    "",
                    block.deref(ctx).unique_name(ctx),
                    width = state.current_indent() as usize
                )?;
                for (i, arg) in block.deref(ctx).arguments().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}: ", arg.unique_name(ctx))?;
                    print_type(ctx, arg.get_type(ctx), state, f)?;
                }
                write!(f, "):")?;
                let _indent = state.indent();
                for nested in block.deref(ctx).iter(ctx) {
                    write!(
                        f,
                        "\n{:width$}",
                        "",
                        width = state.current_indent() as usize
                    )?;
                    print_op(ctx, nested, state, f)?;
                }
            }
        }
        write!(
            f,
            "\n{:width$}}}",
            "",
            width = state.current_indent() as usize
        )?;
    }
    Ok(())
}

fn mixed(ctx: &Context, list: &[MixedIndex], f: &mut fmt::Formatter<'_>) -> Result<()> {
    write!(f, "[")?;
    for (i, item) in list.iter().enumerate() {
        if i != 0 {
            write!(f, ", ")?;
        }
        match item {
            MixedIndex::Static(n) => write!(f, "{n}")?,
            MixedIndex::Dynamic(v) => write!(f, "%{}", v.unique_name(ctx))?,
        }
    }
    write!(f, "]")?;
    Ok(())
}

struct Properties(BTreeMap<&'static str, String>);

impl Properties {
    fn take(&mut self, name: &str) -> Result<String> {
        self.0
            .remove(name)
            .ok_or_else(|| untranslatable(format!("missing property {name}")))
    }

    fn parens(&mut self, name: &str, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let value = self.take(name)?;
        write!(f, " {name}({value})")?;
        Ok(())
    }

    fn dictionary(
        &mut self,
        names: &[&str],
        open: &str,
        close: &str,
        f: &mut fmt::Formatter<'_>,
    ) -> Result<()> {
        let mut entries = Vec::new();
        for &name in names {
            if let Some(value) = self.0.remove(name) {
                entries.push(if value == "unit" {
                    name.to_owned()
                } else {
                    format!("{name} = {value}")
                });
            }
        }
        if !entries.is_empty() {
            write!(f, "{open}{}{close}", entries.join(", "))?;
        }
        Ok(())
    }
}

pub(super) fn print(
    ctx: &Context,
    op: Ptr<Operation>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    Operation::get_op_dyn(op, ctx).verify(ctx)?;
    let name = upstream_name(ctx, op);
    let mnemonic = name
        .strip_prefix("iree_linalg_ext.")
        .ok_or_else(|| untranslatable(name.clone()))?;
    let raw = op.deref(ctx);
    let operands = raw.operands().collect::<Vec<_>>();
    let results = raw.results().collect::<Vec<_>>();
    let mut props = Properties(BTreeMap::new());
    let mut segments = None;
    for (key, attr) in &raw.attributes.0 {
        if key.as_ref() == "builtin_given_names" {
            continue;
        }
        let (property_name, text) = property_for(ctx, &name, key.as_ref(), &**attr)?
            .ok_or_else(|| untranslatable(format!("unmapped attribute {key} on {name}")))?;
        if property_name == "operandSegmentSizes" {
            segments = attr
                .downcast_ref::<pliron::builtin::attributes::OperandSegmentSizesAttr>()
                .map(|s| s.0.clone());
            continue;
        }
        if let Some(mut text) = text {
            // Declarative assembly knows the attribute types and elides them.
            if let Some(value) = attr.downcast_ref::<pliron_mlir_builtin::attributes::IndexAttr>() {
                text = value.value.to_string();
            } else if let Some(value) =
                attr.downcast_ref::<pliron_mlir_builtin::attributes::I64ArrayAttr>()
            {
                text = format!(
                    "[{}]",
                    value
                        .values
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            props.0.insert(property_name, text);
        }
    }
    values(ctx, &results, f)?;
    if !results.is_empty() {
        write!(f, " = ")?;
    }
    write!(f, "{name}")?;
    match mnemonic {
        "yield" => {
            if !operands.is_empty() {
                write!(f, " ")?;
                values(ctx, &operands, f)?;
                write!(f, " : ")?;
                print_type_list(ctx, operands.iter().map(|v| v.get_type(ctx)), state, f)?;
            }
        }
        "index" => {
            write!(f, " {} : ", props.take("dim")?)?;
            print_type_list(ctx, results.iter().map(|v| v.get_type(ctx)), state, f)?;
        }
        "map_load" | "map_store" => {
            write!(
                f,
                " %{} into %{}",
                operands[0].unique_name(ctx),
                operands[1].unique_name(ctx)
            )?;
            region(ctx, op, state, f)?;
            write!(f, " : ")?;
            print_type(ctx, operands[0].get_type(ctx), state, f)?;
            write!(f, " into ")?;
            print_type(ctx, operands[1].get_type(ctx), state, f)?;
            arrow(ctx, &results, " -> ", state, f)?;
        }
        "im2col" => {
            let concrete = Operation::get_op::<Im2colOp>(op, ctx)
                .ok_or_else(|| untranslatable("expected im2col operation".into()))?;
            for key in ["strides", "dilations"] {
                write!(f, " {key} = {}", props.take(key)?)?;
            }
            write!(f, " kernel_size = ")?;
            mixed(ctx, &concrete.mixed_kernel_size(ctx)?, f)?;
            write!(f, " offsets = ")?;
            mixed(ctx, &concrete.mixed_offsets(ctx)?, f)?;
            write!(f, " output_sizes = [")?;
            for (i, list) in concrete.mixed_output_sizes(ctx)?.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                mixed(ctx, list, f)?;
            }
            write!(f, "]")?;
            for key in ["batch_pos", "m_pos", "k_pos", "input_k_perm", "output_perm"] {
                write!(f, " {key} = {}", props.take(key)?)?;
            }
            let padding = concrete.padding(ctx)?;
            for (low_name, high_name, low, high) in [
                (
                    "input_pad_low",
                    "input_pad_high",
                    &padding.input_low,
                    &padding.input_high,
                ),
                (
                    "output_pad_low",
                    "output_pad_high",
                    &padding.output_low,
                    &padding.output_high,
                ),
            ] {
                if !low.is_empty() || !high.is_empty() {
                    write!(f, " {low_name} = ")?;
                    mixed(ctx, low, f)?;
                    write!(f, " {high_name} = ")?;
                    mixed(ctx, high, f)?;
                }
            }
            if let Some(value) = padding.value {
                typed(ctx, "pad_value", &[value], state, f)?;
            }
            for key in [
                "static_kernel_size",
                "static_offsets",
                "static_output_sizes",
                "static_input_pad_low",
                "static_input_pad_high",
                "static_output_pad_low",
                "static_output_pad_high",
            ] {
                props.0.remove(key);
            }
            typed(ctx, "ins", &operands[..1], state, f)?;
            typed(ctx, "outs", &operands[1..2], state, f)?;
            arrow(ctx, &results, " -> ", state, f)?;
        }
        _ => {
            match mnemonic {
                "sort" | "scan" | "topk" | "topk_v2" | "arg_compare" => {
                    props.parens("dimension", f)?;
                    if mnemonic == "scan" {
                        props.parens("inclusive", f)?;
                    }
                    if props.0.remove("is_sorted").is_some() {
                        write!(f, " is_sorted")?;
                    }
                }
                "scatter" | "gather" => {
                    write!(f, " dimension_map = {}", props.take("dimension_map")?)?;
                    if mnemonic == "scatter" {
                        props.parens("unique_indices", f)?;
                    }
                }
                "winograd.input_transform"
                | "winograd.filter_transform"
                | "winograd.output_transform" => {
                    props.parens("output_tile_size", f)?;
                    props.parens("kernel_size", f)?;
                    props.parens(
                        if mnemonic == "winograd.filter_transform" {
                            "kernel_dimensions"
                        } else {
                            "image_dimensions"
                        },
                        f,
                    )?;
                }
                "custom_op" | "exp_reduction" => {
                    let maps = props.take("indexing_maps")?;
                    let iters = props.take("iterator_types")?;
                    write!(f, " {{indexing_maps = {maps}, iterator_types = {iters}")?;
                    if mnemonic == "exp_reduction" {
                        write!(
                            f,
                            ", exp_reduced_operands = {}",
                            props.take("exp_reduced_operands")?
                        )?;
                    }
                    write!(f, "}}")?;
                }
                "attention" | "online_attention" | "quantize_affine" | "dequantize_affine"
                | "group_mmt4d" => {
                    props.dictionary(
                        &[
                            "indexing_maps",
                            "decomposition_config",
                            "quant_min",
                            "quant_max",
                            "storage_unsigned",
                            "input_unsigned",
                            "zp_unsigned",
                            "transposed",
                        ],
                        " <",
                        ">",
                        f,
                    )?;
                }
                "fft" | "group_matmul" => {}
                _ => return Err(untranslatable(format!("no custom printer for {name}"))),
            }
            let (inputs, output_end) = match mnemonic {
                "arg_compare" => {
                    let sizes = segments
                        .as_ref()
                        .ok_or_else(|| untranslatable("missing operand segments".into()))?;
                    let ins = (sizes[0] + sizes[1]) as usize;
                    (ins, ins + 2)
                }
                _ => (operands.len() - results.len(), operands.len()),
            };
            typed(ctx, "ins", &operands[..inputs], state, f)?;
            typed(ctx, "outs", &operands[inputs..output_end], state, f)?;
            if output_end < operands.len() {
                typed(ctx, "index_base", &operands[output_end..], state, f)?;
            }
            region(ctx, op, state, f)?;
            arrow(
                ctx,
                &results,
                if mnemonic == "fft" { " : " } else { " -> " },
                state,
                f,
            )?;
        }
    }
    if !props.0.is_empty() {
        return Err(untranslatable(format!(
            "unprinted properties on {name}: {:?}",
            props.0.keys()
        )));
    }
    print_op_location(ctx, raw.loc(), state, f)
}

fn arrow(
    ctx: &Context,
    results: &[Value],
    separator: &str,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    if !results.is_empty() {
        write!(f, "{separator}")?;
        print_type_list(ctx, results.iter().map(|v| v.get_type(ctx)), state, f)?;
    }
    Ok(())
}
