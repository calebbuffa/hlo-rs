use super::{TextError, TextErrorKind, cursor::Cursor, floats, printer::mlir_type};
use crate::{
    attributes::*,
    pliron::{
        attribute::{AttrObj, Attribute},
        context::Context,
    },
    types::ElementType,
};
use std::collections::{BTreeMap, HashSet};

impl Cursor<'_> {
    pub(super) fn stablehlo_attribute(
        &mut self,
        ctx: &Context,
        name: &str,
        depth: usize,
    ) -> Result<AttrObj, TextError> {
        self.depth(depth)?;
        macro_rules! dimensions {
            ($ty:ident, [$($list:ident),*] $(, $scalar:ident)?) => {{
                let mut value = $ty { $($list: Vec::new(),)* $($scalar: 0,)? };
                self.fields(|cursor, key| {
                    match key {
                        $(stringify!($list) => value.$list = cursor.dimension_list()?,)*
                        $(stringify!($scalar) => value.$scalar = cursor.attribute_i64()?,)?
                        _ => return Err(cursor.error(TextErrorKind::Expected("dimension-number field"))),
                    }
                    Ok(())
                })?;
                Box::new(value) as AttrObj
            }};
        }
        Ok(match name {
            "dot" => dimensions!(
                DotDimensionNumbersAttr,
                [
                    lhs_batching_dimensions,
                    rhs_batching_dimensions,
                    lhs_contracting_dimensions,
                    rhs_contracting_dimensions
                ]
            ),
            "gather" => dimensions!(
                GatherDimensionNumbersAttr,
                [
                    offset_dims,
                    collapsed_slice_dims,
                    operand_batching_dims,
                    start_indices_batching_dims,
                    start_index_map
                ],
                index_vector_dim
            ),
            "scatter" => dimensions!(
                ScatterDimensionNumbersAttr,
                [
                    update_window_dims,
                    inserted_window_dims,
                    input_batching_dims,
                    scatter_indices_batching_dims,
                    scatter_dims_to_operand_dims
                ],
                index_vector_dim
            ),
            "conv" => Box::new(self.convolution_attribute()?),
            "channel_handle" => {
                let mut value = ChannelHandleAttr {
                    handle: 0,
                    channel_kind: ChannelKind::DeviceToDevice,
                };
                let fields = self.fields(|cursor, key| {
                    match key {
                        "handle" => value.handle = cursor.attribute_i64()?,
                        "type" => {
                            value.channel_kind = match cursor.attribute_i64()? {
                                1 => ChannelKind::DeviceToDevice,
                                2 => ChannelKind::DeviceToHost,
                                3 => ChannelKind::HostToDevice,
                                _ => {
                                    return Err(
                                        cursor.error(TextErrorKind::Expected("channel type"))
                                    );
                                }
                            }
                        }
                        _ => return Err(cursor.error(TextErrorKind::Expected("handle or type"))),
                    }
                    Ok(())
                })?;
                for field in ["handle", "type"] {
                    if !fields.contains(field) {
                        return Err(self.error(TextErrorKind::Expected(field)));
                    }
                }
                Box::new(value)
            }
            "output_operand_alias" => {
                self.expect("<")?;
                self.expect("output_tuple_indices")?;
                self.expect("=")?;
                let output_tuple_indices = self.dimension_list()?;
                self.expect(",")?;
                self.expect("operand_index")?;
                self.expect("=")?;
                let operand_index = self.attribute_i64()?;
                self.expect(",")?;
                self.expect("operand_tuple_indices")?;
                self.expect("=")?;
                let operand_tuple_indices = self.dimension_list()?;
                self.expect(">")?;
                Box::new(OutputOperandAliasAttr {
                    output_tuple_indices,
                    operand_index,
                    operand_tuple_indices,
                })
            }
            "dot_algorithm" => {
                self.expect("<")?;
                self.expect("lhs_precision_type")?;
                self.expect("=")?;
                let lhs_precision_type = self.ty(ctx, depth + 1)?;
                self.expect(",")?;
                self.expect("rhs_precision_type")?;
                self.expect("=")?;
                let rhs_precision_type = self.ty(ctx, depth + 1)?;
                self.expect(",")?;
                self.expect("accumulation_type")?;
                self.expect("=")?;
                let accumulation_type = self.ty(ctx, depth + 1)?;
                self.expect(",")?;
                self.expect("lhs_component_count")?;
                self.expect("=")?;
                let lhs_component_count = self.attribute_i64()?;
                self.expect(",")?;
                self.expect("rhs_component_count")?;
                self.expect("=")?;
                let rhs_component_count = self.attribute_i64()?;
                self.expect(",")?;
                self.expect("num_primitive_operations")?;
                self.expect("=")?;
                let num_primitive_operations = self.attribute_i64()?;
                self.expect(",")?;
                self.expect("allow_imprecise_accumulation")?;
                self.expect("=")?;
                let allow_imprecise_accumulation = if self.keyword("true") {
                    true
                } else if self.keyword("false") {
                    false
                } else {
                    return Err(self.error(TextErrorKind::Expected("true or false")));
                };
                self.expect(">")?;
                Box::new(DotAlgorithmAttr {
                    lhs_precision_type,
                    rhs_precision_type,
                    accumulation_type,
                    lhs_component_count,
                    rhs_component_count,
                    num_primitive_operations,
                    allow_imprecise_accumulation,
                })
            }
            "result_accuracy" => {
                self.expect("<")?;
                let mut atol_bits = 0;
                let mut rtol_bits = 0;
                let mut ulps = 0;
                if self.keyword("atol") {
                    self.expect("=")?;
                    atol_bits = floats::parse(&self.number()?, ElementType::F64)
                        .map_err(|kind| self.error(kind))?;
                    self.expect(",")?;
                }
                if self.keyword("rtol") {
                    self.expect("=")?;
                    rtol_bits = floats::parse(&self.number()?, ElementType::F64)
                        .map_err(|kind| self.error(kind))?;
                    self.expect(",")?;
                }
                if self.keyword("ulps") {
                    self.expect("=")?;
                    ulps = self.attribute_i64()?;
                    self.expect(",")?;
                }
                self.expect("mode")?;
                self.expect("=")?;
                let mode = self.attribute(ctx, depth + 1)?;
                let mode = mode.downcast_ref::<ResultAccuracyMode>().ok_or_else(|| {
                    self.error(TextErrorKind::Expected("StableHLO result accuracy mode"))
                })?;
                self.expect(">")?;
                Box::new(ResultAccuracyAttr {
                    atol_bits,
                    rtol_bits,
                    ulps,
                    mode: *mode,
                })
            }
            _ => return Err(self.error(TextErrorKind::Unsupported("StableHLO attribute"))),
        })
    }

    fn attribute_i64(&mut self) -> Result<i64, TextError> {
        i64::try_from(self.integer()?).map_err(|_| self.error(TextErrorKind::InvalidNumber))
    }

    fn dimension_list(&mut self) -> Result<Vec<i64>, TextError> {
        self.expect("[")?;
        let mut values = Vec::new();
        if !self.take("]") {
            loop {
                values.push(self.attribute_i64()?);
                if self.take("]") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(values)
    }

    fn fields(
        &mut self,
        parse: impl FnMut(&mut Self, &str) -> Result<(), TextError>,
    ) -> Result<HashSet<String>, TextError> {
        self.expect("<")?;
        self.fields_body(parse)
    }

    fn fields_body(
        &mut self,
        mut parse: impl FnMut(&mut Self, &str) -> Result<(), TextError>,
    ) -> Result<HashSet<String>, TextError> {
        let mut fields = HashSet::new();
        if !self.take(">") {
            loop {
                let field = self.word()?.to_owned();
                if !fields.insert(field.clone()) {
                    return Err(self.error(TextErrorKind::DuplicateField(field)));
                }
                self.expect("=")?;
                parse(self, &field)?;
                if self.take(">") {
                    break;
                }
                self.expect(",")?;
                if self.take(">") {
                    break;
                }
            }
        }
        Ok(fields)
    }

    fn convolution_attribute(&mut self) -> Result<ConvDimensionNumbersAttr, TextError> {
        self.expect("<")?;
        let mut value = ConvDimensionNumbersAttr {
            input_batch_dimension: 0,
            input_feature_dimension: 0,
            input_spatial_dimensions: Vec::new(),
            kernel_input_feature_dimension: 0,
            kernel_output_feature_dimension: 0,
            kernel_spatial_dimensions: Vec::new(),
            output_batch_dimension: 0,
            output_feature_dimension: 0,
            output_spatial_dimensions: Vec::new(),
        };
        if self.keyword("raw") {
            self.fields_body(|cursor, key| {
                match key {
                    "input_batch_dimension" => {
                        value.input_batch_dimension = cursor.attribute_i64()?
                    }
                    "input_feature_dimension" => {
                        value.input_feature_dimension = cursor.attribute_i64()?
                    }
                    "input_spatial_dimensions" => {
                        value.input_spatial_dimensions = cursor.dimension_list()?
                    }
                    "kernel_input_feature_dimension" => {
                        value.kernel_input_feature_dimension = cursor.attribute_i64()?
                    }
                    "kernel_output_feature_dimension" => {
                        value.kernel_output_feature_dimension = cursor.attribute_i64()?
                    }
                    "kernel_spatial_dimensions" => {
                        value.kernel_spatial_dimensions = cursor.dimension_list()?
                    }
                    "output_batch_dimension" => {
                        value.output_batch_dimension = cursor.attribute_i64()?
                    }
                    "output_feature_dimension" => {
                        value.output_feature_dimension = cursor.attribute_i64()?
                    }
                    "output_spatial_dimensions" => {
                        value.output_spatial_dimensions = cursor.dimension_list()?
                    }
                    _ => {
                        return Err(
                            cursor.error(TextErrorKind::Expected("convolution dimension field"))
                        );
                    }
                }
                Ok(())
            })?;
        } else {
            (
                value.input_batch_dimension,
                value.input_feature_dimension,
                value.input_spatial_dimensions,
            ) = self.convolution_dimensions("b", "f")?;
            self.expect("x")?;
            (
                value.kernel_input_feature_dimension,
                value.kernel_output_feature_dimension,
                value.kernel_spatial_dimensions,
            ) = self.convolution_dimensions("i", "o")?;
            self.expect("->")?;
            (
                value.output_batch_dimension,
                value.output_feature_dimension,
                value.output_spatial_dimensions,
            ) = self.convolution_dimensions("b", "f")?;
            self.expect(">")?;
        }
        Ok(value)
    }

    fn convolution_dimensions(
        &mut self,
        first: &'static str,
        second: &'static str,
    ) -> Result<(i64, i64, Vec<i64>), TextError> {
        self.expect("[")?;
        let mut first_axis = None;
        let mut second_axis = None;
        let mut spatial = BTreeMap::new();
        let mut axis = 0_i64;
        loop {
            self.space();
            if self
                .input
                .as_bytes()
                .get(self.offset)
                .is_some_and(u8::is_ascii_digit)
            {
                if spatial.insert(self.attribute_i64()?, axis).is_some() {
                    return Err(self.error(TextErrorKind::Expected("unique spatial dimension")));
                }
            } else {
                let marker = self.word()?;
                let target = if marker == first {
                    &mut first_axis
                } else if marker == second {
                    &mut second_axis
                } else {
                    return Err(self.error(TextErrorKind::Expected("convolution dimension marker")));
                };
                if target.replace(axis).is_some() {
                    return Err(self.error(TextErrorKind::Expected("unique nonspatial dimension")));
                }
            }
            axis += 1;
            if self.take("]") {
                break;
            }
            self.expect(",")?;
        }
        for (expected, &actual) in spatial.keys().enumerate() {
            if i64::try_from(expected).ok() != Some(actual) {
                return Err(self.error(TextErrorKind::Expected(
                    "spatial dimensions numbered consecutively from zero",
                )));
            }
        }
        Ok((
            first_axis.ok_or_else(|| self.error(TextErrorKind::Expected(first)))?,
            second_axis.ok_or_else(|| self.error(TextErrorKind::Expected(second)))?,
            spatial.into_values().collect(),
        ))
    }
}

pub(super) fn print(
    ctx: &Context,
    attribute: &dyn Attribute,
) -> crate::pliron::result::Result<Option<String>> {
    macro_rules! dimensions {
        ($ty:ident, $name:literal, [$($list:ident),*] $(, $scalar:ident)?) => {
            if let Some(value) = attribute.downcast_ref::<$ty>() {
                let fields = vec![
                    $(format!("{} = {:?}", stringify!($list), value.$list),)*
                    $(format!("{} = {}", stringify!($scalar), value.$scalar),)?
                ];
                return Ok(Some(format!("#stablehlo.{}<{}>", $name, fields.join(", "))));
            }
        };
    }
    dimensions!(
        DotDimensionNumbersAttr,
        "dot",
        [
            lhs_batching_dimensions,
            rhs_batching_dimensions,
            lhs_contracting_dimensions,
            rhs_contracting_dimensions
        ]
    );
    dimensions!(
        GatherDimensionNumbersAttr,
        "gather",
        [
            offset_dims,
            collapsed_slice_dims,
            operand_batching_dims,
            start_indices_batching_dims,
            start_index_map
        ],
        index_vector_dim
    );
    dimensions!(
        ScatterDimensionNumbersAttr,
        "scatter",
        [
            update_window_dims,
            inserted_window_dims,
            input_batching_dims,
            scatter_indices_batching_dims,
            scatter_dims_to_operand_dims
        ],
        index_vector_dim
    );
    if let Some(value) = attribute.downcast_ref::<ChannelHandleAttr>() {
        return Ok(Some(format!(
            "#stablehlo.channel_handle<handle = {}, type = {}>",
            value.handle, value.channel_kind as i64
        )));
    }
    if let Some(value) = attribute.downcast_ref::<OutputOperandAliasAttr>() {
        return Ok(Some(format!(
            "#stablehlo.output_operand_alias<output_tuple_indices = {:?}, operand_index = {}, operand_tuple_indices = {:?}>",
            value.output_tuple_indices, value.operand_index, value.operand_tuple_indices
        )));
    }
    if let Some(value) = attribute.downcast_ref::<DotAlgorithmAttr>() {
        return Ok(Some(format!(
            "#stablehlo.dot_algorithm<lhs_precision_type = {}, rhs_precision_type = {}, accumulation_type = {}, lhs_component_count = {}, rhs_component_count = {}, num_primitive_operations = {}, allow_imprecise_accumulation = {}>",
            mlir_type(ctx, value.lhs_precision_type)?,
            mlir_type(ctx, value.rhs_precision_type)?,
            mlir_type(ctx, value.accumulation_type)?,
            value.lhs_component_count,
            value.rhs_component_count,
            value.num_primitive_operations,
            value.allow_imprecise_accumulation
        )));
    }
    if let Some(value) = attribute.downcast_ref::<ResultAccuracyAttr>() {
        let mode = super::printer::mlir_attribute(ctx, &value.mode)?;
        return Ok(Some(format!(
            "#stablehlo.result_accuracy<atol = 0x{:X}, rtol = 0x{:X}, ulps = {}, mode = {}>",
            value.atol_bits, value.rtol_bits, value.ulps, mode
        )));
    }
    if let Some(value) = attribute.downcast_ref::<ConvDimensionNumbersAttr>() {
        return Ok(Some(format!(
            "#stablehlo.conv<raw input_batch_dimension = {}, input_feature_dimension = {}, input_spatial_dimensions = {:?}, kernel_input_feature_dimension = {}, kernel_output_feature_dimension = {}, kernel_spatial_dimensions = {:?}, output_batch_dimension = {}, output_feature_dimension = {}, output_spatial_dimensions = {:?}>",
            value.input_batch_dimension,
            value.input_feature_dimension,
            value.input_spatial_dimensions,
            value.kernel_input_feature_dimension,
            value.kernel_output_feature_dimension,
            value.kernel_spatial_dimensions,
            value.output_batch_dimension,
            value.output_feature_dimension,
            value.output_spatial_dimensions
        )));
    }
    Ok(None)
}
