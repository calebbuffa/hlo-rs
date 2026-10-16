//! HLO assembly for the implemented core slice, not MLIR assembly.
//! Unimplemented grammar and unprintable metadata return explicit errors.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::sync::Mutex;

use protobuf::{Enum, EnumFull, MessageFull};

use crate::{
    Dimension, Error, Literal, Module, Opcode, PrimitiveType, Result, Shape,
    proto::{
        hlo::{
            HloComputationProto, HloInstructionProto, HloModuleProto,
            hlo_instruction_proto::SliceDimensions,
        },
        xla_data::{
            ConvolutionDimensionNumbers, DotDimensionNumbers, FftType, LayoutProto,
            ProgramShapeProto, Window, WindowDimension,
            precision_config::{Algorithm, Precision},
        },
    },
};

const MAX_DEPTH: usize = 100;
mod accuracy;
mod attributes;
pub use attributes::{
    parse_frontend_attributes, parse_layout, parse_metadata, parse_original_value,
    parse_padding_config, parse_parameter_replication, parse_shape_list, parse_statistics,
    print_frontend_attributes, print_layout, print_metadata, print_original_value,
    print_padding_config, print_parameter_replication, print_shape_list, print_statistics,
};
mod asynchronous;
mod callable;
pub(crate) mod collective;
pub use collective::{
    CollectiveDeviceList, parse_collective_device_list, parse_replica_groups,
    print_collective_device_list, print_replica_groups,
};
mod contexts;
mod dot;
mod floats;
mod inferred;
mod layout;
mod literals;
mod metadata;
mod module_attributes;
mod numbers;
mod operands;
mod original_value;
mod sharding;
mod stack_frames;
pub use sharding::{parse as parse_sharding, print as print_sharding};
mod strings;
static SHORTFORM_COUNTERS: Mutex<(u32, u8)> = Mutex::new((0, 0));

#[derive(Clone, Copy, Debug)]
pub struct ParseOptions {
    pub fill_missing_layouts: bool,
    pub fill_shortform_constants_with_random_values: bool,
    pub max_bytes: usize,
    pub max_literal_elements: usize,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            fill_missing_layouts: true,
            fill_shortform_constants_with_random_values: true,
            max_bytes: 64 * 1024 * 1024,
            max_literal_elements: 1_000_000,
        }
    }
}

#[derive(Clone, Debug)]
struct Token {
    value: String,
    offset: usize,
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
    source_len: usize,
    options: ParseOptions,
}

type InstructionOperands = Vec<(String, Vec<i64>, Option<Shape>)>;
type PendingOperands = Vec<(i64, InstructionOperands)>;
type PendingNames = Vec<(i64, Vec<String>)>;

struct InstructionHead {
    instruction: HloInstructionProto,
    outer_shape: Option<Shape>,
    shape: Option<Shape>,
    outer_opcode: Opcode,
    opcode: Opcode,
    wrapped: Option<Opcode>,
}

struct InstructionContext<'a> {
    parser: &'a mut Parser,
    module: &'a mut HloModuleProto,
    computation: &'a mut HloComputationProto,
    names: &'a mut BTreeMap<String, i64>,
    parsed_names: &'a mut crate::ir::names::NameUniquer,
    next_instruction: &'a mut i64,
    next_computation: &'a mut i64,
    sugar: &'a mut BTreeMap<i64, Opcode>,
    generated_computations: &'a mut BTreeSet<i64>,
    pending_callees: &'a mut PendingNames,
    pending_operands: &'a mut PendingOperands,
    pending_controls: &'a mut PendingNames,
    async_backend_attributes: &'a mut BTreeSet<i64>,
    channel_presence: &'a mut BTreeSet<i64>,
    broadcasts_without_dimensions: &'a mut BTreeSet<i64>,
}

impl Parser {
    fn new(source: &str, options: ParseOptions) -> Result<Self> {
        Self::new_with_root_context(source, options, None)
    }

    fn new_with_root_context(
        source: &str,
        options: ParseOptions,
        root_context: Option<&str>,
    ) -> Result<Self> {
        if source.len() > options.max_bytes {
            return Err(Error::new("resource-limit", "text byte limit exceeded"));
        }
        let mut tokens: Vec<Token> = Vec::new();
        let mut braces: Vec<Option<String>> = Vec::new();
        let bytes = source.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index].is_ascii_whitespace() {
                index += 1;
                continue;
            }
            if bytes[index..].starts_with(b"//") {
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
                continue;
            }
            if bytes[index..].starts_with(b"/*") {
                let end = bytes[index + 2..]
                    .windows(2)
                    .position(|pair| pair == b"*/")
                    .ok_or_else(|| Error::new("syntax", "unterminated block comment").at(index))?;
                index += end + 4;
                continue;
            }
            let start = index;
            let assignment = if tokens.last().is_some_and(|token| token.value == "=") {
                tokens
                    .get(tokens.len().saturating_sub(2))
                    .map(|token| token.value.as_str())
            } else {
                None
            };
            if bytes[index] == b'{'
                && (assignment == Some("backend_config")
                    || (assignment.is_some()
                        && braces.last().and_then(Option::as_deref) == Some("frontend_attributes")))
            {
                index = strings::dictionary_end(source, index)?;
            } else if bytes[index..].starts_with(b"->") || bytes[index..].starts_with(b"<=") {
                index += 2;
            } else if b"()[]{},:=<>#".contains(&bytes[index]) {
                if bytes[index] == b'{' {
                    braces.push(
                        assignment
                            .or_else(|| tokens.is_empty().then_some(root_context).flatten())
                            .map(str::to_owned),
                    );
                }
                if bytes[index] == b'}' {
                    braces.pop();
                }
                index += 1;
            } else if matches!(bytes[index], b'"' | b'\'') {
                let quote = bytes[index];
                index += 1;
                while index < bytes.len() && bytes[index] != quote {
                    if bytes[index] == b'\\' {
                        index += 1;
                    }
                    index += 1;
                }
                if index >= bytes.len() {
                    return Err(Error::new("syntax", "unterminated quoted string").at(start));
                }
                index += 1;
                strings::unquote(
                    source
                        .get(start..index)
                        .ok_or_else(|| Error::new("syntax", "invalid string boundary").at(start))?,
                )?;
            } else {
                while index < bytes.len()
                    && !bytes[index].is_ascii_whitespace()
                    && !b"()[]{},:=<>#\"'".contains(&bytes[index])
                    && !bytes[index..].starts_with(b"->")
                    && !bytes[index..].starts_with(b"//")
                    && !bytes[index..].starts_with(b"/*")
                {
                    index += 1;
                }
                if index == start {
                    return Err(Error::new("syntax", "unexpected character").at(index));
                }
            }
            let value = source
                .get(start..index)
                .ok_or_else(|| Error::new("syntax", "invalid token boundary").at(start))?;
            tokens.push(Token {
                value: value.into(),
                offset: start,
            });
        }
        Ok(Self {
            tokens,
            index: 0,
            source_len: source.len(),
            options,
        })
    }

    fn offset(&self) -> usize {
        self.tokens
            .get(self.index)
            .map_or(self.source_len, |t| t.offset)
    }
    fn error(&self, message: impl Into<String>) -> Error {
        Error::new("syntax", message).at(self.offset())
    }
    fn peek(&self) -> &str {
        self.tokens.get(self.index).map_or("", |t| t.value.as_str())
    }
    fn take(&mut self, expected: &str) -> bool {
        if self.peek() == expected {
            self.index += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, expected: &str) -> Result<()> {
        if self.take(expected) {
            Ok(())
        } else {
            Err(self.error(format!("expected {expected:?}, found {:?}", self.peek())))
        }
    }
    fn word(&mut self) -> Result<String> {
        let token = self
            .tokens
            .get(self.index)
            .ok_or_else(|| self.error("unexpected end of input"))?;
        if token.value.len() == 1 && "()[]{},:=<>".contains(&token.value) {
            return Err(self.error("expected identifier or number"));
        }
        let word = token.value.clone();
        self.index += 1;
        Ok(word)
    }
    fn name(&mut self) -> Result<String> {
        let offset = self.offset();
        let word = self.word()?;
        let name = word.strip_prefix('%').unwrap_or(&word);
        if !valid_name(name) || !word.starts_with('%') && reserved_name(name) {
            return Err(Error::new("syntax", "unsupported or invalid HLO identifier").at(offset));
        }
        Ok(name.into())
    }
    fn integer(&mut self) -> Result<i64> {
        let offset = self.offset();
        numbers::integer(&self.word()?)
            .map_err(|_| Error::new("syntax", "expected integer token").at(offset))
    }

    fn boolean(&mut self) -> Result<bool> {
        match self.word()?.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(self.error("expected boolean")),
        }
    }
    fn booleans(&mut self) -> Result<Vec<bool>> {
        if !self.take("{") {
            return Ok(vec![self.boolean()?]);
        }
        let mut values = Vec::new();
        if !self.take("}") {
            loop {
                values.push(self.boolean()?);
                if self.take("}") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(values)
    }
    fn byte_string(&mut self) -> Result<Vec<u8>> {
        let offset = self.offset();
        strings::unquote(&self.word()?).map_err(|error| error.at(offset))
    }
    fn string_or_dictionary(&mut self) -> Result<Vec<u8>> {
        if self.peek().starts_with('{') && self.peek().len() > 1 {
            Ok(self.word()?.into_bytes())
        } else {
            self.byte_string()
        }
    }
    fn finish(&self) -> Result<()> {
        if self.index != self.tokens.len() {
            Err(self.error("trailing input"))
        } else {
            Ok(())
        }
    }
    fn integers(&mut self, open: &str, close: &str) -> Result<Vec<i64>> {
        self.expect(open)?;
        let mut values = Vec::new();
        if !self.take(close) {
            loop {
                values.push(self.integer()?);
                if self.take(close) {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(values)
    }
    fn names(&mut self) -> Result<Vec<String>> {
        self.expect("{")?;
        let mut values = Vec::new();
        if !self.take("}") {
            loop {
                values.push(self.name()?);
                if self.take("}") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(values)
    }
    fn integer_lists(&mut self) -> Result<Vec<Vec<i64>>> {
        self.expect("{")?;
        let mut lists = Vec::new();
        if !self.take("}") {
            loop {
                lists.push(self.integers("{", "}")?);
                if self.take("}") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(lists)
    }
    fn window(&mut self, outer_braces: bool) -> Result<Window> {
        if outer_braces {
            self.expect("{")?;
        }
        let mut fields = BTreeMap::new();
        let mut padding = None;
        let mut seen = BTreeSet::new();
        while !self.peek().is_empty() && (!outer_braces || self.peek() != "}") {
            let field = self.word()?;
            if !seen.insert(field.clone()) {
                return Err(self.error("duplicate window field"));
            }
            if !matches!(
                field.as_str(),
                "size" | "stride" | "lhs_dilate" | "rhs_dilate" | "rhs_reversal" | "pad"
            ) {
                return Err(
                    Error::new("unsupported", format!("window field {field}")).at(self.offset())
                );
            }
            self.expect("=")?;
            let source = self.word()?;
            if field == "pad" {
                let mut dimensions = Vec::new();
                for axis in source.split('x') {
                    let values = axis
                        .split('_')
                        .map(|n| {
                            n.parse::<i64>()
                                .map_err(|_| self.error("invalid window padding"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    if values.len() != 2 {
                        return Err(self.error("window padding requires low_high"));
                    }
                    dimensions.push((values[0], values[1]));
                }
                padding = Some(dimensions);
            } else {
                fields.insert(
                    field,
                    source
                        .split('x')
                        .map(|n| {
                            n.parse::<i64>()
                                .map_err(|_| self.error("invalid window dimension list"))
                        })
                        .collect::<Result<Vec<_>>>()?,
                );
            }
        }
        if outer_braces {
            self.expect("}")?;
        }
        let sizes = fields.remove("size").unwrap_or_default();
        if fields.values().any(|v| v.len() != sizes.len())
            || padding.as_ref().is_some_and(|p| p.len() != sizes.len())
        {
            return Err(self.error("window field cardinalities differ"));
        }
        let get =
            |name: &str, index: usize, default| fields.get(name).map_or(default, |v| v[index]);
        Ok(Window {
            dimensions: sizes
                .into_iter()
                .enumerate()
                .map(|(axis, size)| {
                    let (low, high) = padding.as_ref().map_or((0, 0), |p| p[axis]);
                    WindowDimension {
                        size,
                        stride: get("stride", axis, 1),
                        padding_low: low,
                        padding_high: high,
                        base_dilation: get("lhs_dilate", axis, 1),
                        window_dilation: get("rhs_dilate", axis, 1),
                        window_reversal: get("rhs_reversal", axis, 0) == 1,
                        ..Default::default()
                    }
                })
                .collect(),
            ..Default::default()
        })
    }

    fn shape(&mut self, depth: usize) -> Result<Shape> {
        enum Frame {
            Tuple(Vec<Shape>),
            Buffer,
        }
        let mut frames = Vec::new();
        'parse: loop {
            if depth + frames.len() >= MAX_DEPTH {
                return Err(
                    Error::new("resource-limit", "shape parser nesting limit").at(self.offset())
                );
            }
            let shape = if self.take("(") {
                if self.take(")") {
                    Shape::tuple(Vec::new())?
                } else {
                    frames.push(Frame::Tuple(Vec::new()));
                    continue;
                }
            } else if self.take("b") {
                self.expect("(")?;
                frames.push(Frame::Buffer);
                continue;
            } else {
                let element = PrimitiveType::from_hlo_name(&self.word()?)?;
                self.expect("[")?;
                let mut dimensions = Vec::new();
                if !self.take("]") {
                    loop {
                        dimensions.push(if self.take("?") {
                            Dimension::Unbounded
                        } else if self.take("<=") {
                            Dimension::Bounded(self.integer()?)
                        } else {
                            Dimension::Static(self.integer()?)
                        });
                        if self.take("]") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                let shape = match element {
                    PrimitiveType::TOKEN if dimensions.is_empty() => Shape::token(),
                    PrimitiveType::OPAQUE_TYPE if dimensions.is_empty() => Shape::opaque(),
                    ty if ty.is_array_element() => Shape::dynamic_array(ty, dimensions)?,
                    _ => {
                        return Err(
                            self.error("unsupported shape category or non-array dimensions")
                        );
                    }
                };
                let layout_follows = self.peek() == "{"
                    && self
                        .tokens
                        .get(self.index + 1)
                        .is_some_and(|t| t.value == ":" || t.value.parse::<i64>().is_ok());
                if layout_follows {
                    shape.with_layout(self.layout()?)?
                } else if shape.is_array()
                    && (self.options.fill_missing_layouts || shape.dimensions().is_empty())
                {
                    shape.with_default_layout()?
                } else {
                    shape
                }
            };

            let mut shape = shape;
            loop {
                match frames.pop() {
                    Some(Frame::Tuple(mut elements)) => {
                        elements.push(shape);
                        if self.take(")") {
                            shape = Shape::tuple(elements)?;
                        } else {
                            self.expect(",")?;
                            frames.push(Frame::Tuple(elements));
                            continue 'parse;
                        }
                    }
                    Some(Frame::Buffer) => {
                        self.expect(")")?;
                        shape = Shape::buffer(shape)?;
                    }
                    None => return Ok(shape),
                }
            }
        }
    }

    fn literal_values(&mut self, shape: &Shape, axis: usize, values: &mut Vec<u128>) -> Result<()> {
        struct Frame {
            axis: usize,
            next_index: usize,
            size: usize,
        }
        let rank = shape.rank()?;
        let mut current_axis = axis;
        let mut frames = Vec::new();
        'visit: loop {
            if current_axis >= MAX_DEPTH {
                return Err(Error::new("resource-limit", "literal nesting limit"));
            }
            if current_axis == rank {
                let offset = self.offset();
                values.push(self.scalar(shape.element_type()).map_err(|error| {
                    if error.offset.is_none() {
                        error.at(offset)
                    } else {
                        error
                    }
                })?);
            } else {
                self.expect("{")?;
                let size = usize::try_from(shape.dimensions()[current_axis])
                    .map_err(|_| self.error("dynamic literal assembly not implemented"))?;
                if size == 0 {
                    self.expect("}")?;
                } else {
                    frames.push(Frame {
                        axis: current_axis,
                        next_index: 0,
                        size,
                    });
                    current_axis += 1;
                    continue;
                }
            }

            loop {
                let Some(mut frame) = frames.pop() else {
                    return Ok(());
                };
                frame.next_index += 1;
                if frame.next_index < frame.size {
                    self.expect(",")?;
                    current_axis = frame.axis + 1;
                    frames.push(frame);
                    continue 'visit;
                }
                self.expect("}")?;
            }
        }
    }
}

fn valid_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b"_-.".contains(&b))
}

fn reserved_name(name: &str) -> bool {
    PrimitiveType::from_hlo_name(name)
        .is_ok_and(|ty| !matches!(ty, PrimitiveType::TUPLE | PrimitiveType::BUFFER))
        || matches!(
            name,
            "nan"
                | "true"
                | "false"
                | "inf"
                | "HloModule"
                | "ENTRY"
                | "ROOT"
                | "FileLocations"
                | "FileNames"
                | "FunctionNames"
                | "StackFrames"
                | "maximal"
                | "replicated"
                | "manual"
                | "last_tile_dim_replicate"
                | "shard_as"
                | "shard_like"
                | "unknown"
                | "unreduced"
        )
}

fn valid_instruction_name(name: &str) -> bool {
    valid_name(name)
        && !PrimitiveType::from_hlo_name(name)
            .is_ok_and(|ty| !matches!(ty, PrimitiveType::TUPLE | PrimitiveType::BUFFER))
        && (!name.starts_with("__") || name.starts_with("__xla_"))
}

pub fn parse_convolution_dimensions(source: &str) -> Result<ConvolutionDimensionNumbers> {
    fn component(source: &str, first: char, second: char) -> Result<(i64, i64, Vec<i64>)> {
        let count = source
            .chars()
            .filter(|&c| c != first && c != second && c != '?')
            .count();
        let mut spatial = vec![-1; count];
        let (mut a, mut b) = (0, 0);
        let mut seen = std::collections::BTreeSet::new();
        for (axis, label) in source.chars().enumerate() {
            if label == '?' {
                continue;
            }
            if !seen.insert(label) {
                return Err(Error::new(
                    "dimensions",
                    "convolution labels must be unique within each operand",
                ));
            }
            if label == first {
                a = axis as i64;
            } else if label == second {
                b = axis as i64;
            } else {
                let index = label
                    .to_digit(10)
                    .filter(|&index| (index as usize) < count)
                    .ok_or_else(|| Error::new("dimensions", "invalid convolution spatial label"))?;
                spatial[index as usize] = axis as i64;
            }
        }
        Ok((a, b, spatial))
    }
    let mut parser = Parser::new(source, Default::default())?;
    let inputs = parser.word()?;
    parser.expect("->")?;
    let output = parser.word()?;
    parser.finish()?;
    let (left, right) = inputs
        .split_once('_')
        .filter(|(_, right)| !right.contains('_'))
        .ok_or_else(|| {
            Error::new(
                "dimensions",
                "expected input_kernel->output convolution labels",
            )
        })?;
    let (input_batch_dimension, input_feature_dimension, input_spatial_dimensions) =
        component(left, 'b', 'f')?;
    let (
        kernel_input_feature_dimension,
        kernel_output_feature_dimension,
        kernel_spatial_dimensions,
    ) = component(right, 'i', 'o')?;
    let (output_batch_dimension, output_feature_dimension, output_spatial_dimensions) =
        component(&output, 'b', 'f')?;
    if input_spatial_dimensions.len() != kernel_spatial_dimensions.len()
        || input_spatial_dimensions.len() != output_spatial_dimensions.len()
    {
        return Err(Error::new(
            "dimensions",
            "convolution label spatial ranks differ",
        ));
    }
    Ok(ConvolutionDimensionNumbers {
        input_batch_dimension,
        input_feature_dimension,
        input_spatial_dimensions,
        kernel_input_feature_dimension,
        kernel_output_feature_dimension,
        kernel_spatial_dimensions,
        output_batch_dimension,
        output_feature_dimension,
        output_spatial_dimensions,
        ..Default::default()
    })
}

pub fn print_convolution_dimensions(dimensions: &ConvolutionDimensionNumbers) -> Result<String> {
    reject_fields(
        dimensions,
        &[
            "input_batch_dimension",
            "input_feature_dimension",
            "input_spatial_dimensions",
            "kernel_input_feature_dimension",
            "kernel_output_feature_dimension",
            "kernel_spatial_dimensions",
            "output_batch_dimension",
            "output_feature_dimension",
            "output_spatial_dimensions",
        ],
    )?;
    fn component(first: i64, second: i64, spatial: &[i64], labels: [char; 2]) -> Result<String> {
        let maximum = spatial
            .iter()
            .copied()
            .chain([first, second])
            .max()
            .ok_or_else(|| Error::new("dimensions", "empty convolution labels"))?;
        if first < 0
            || second < 0
            || spatial.iter().any(|&axis| axis < 0)
            || maximum >= ParseOptions::default().max_bytes as i64
        {
            return Err(Error::new(
                "dimensions",
                "convolution label axis is negative or exceeds text resource limit",
            ));
        }
        let mut result = vec!['?'; maximum as usize + 1];
        result[first as usize] = labels[0];
        result[second as usize] = labels[1];
        for (index, &axis) in spatial.iter().enumerate() {
            result[axis as usize] = char::from_digit(
                u32::try_from(index)
                    .map_err(|_| Error::new("overflow", "convolution spatial index overflow"))?,
                10,
            )
            .ok_or_else(|| {
                Error::new("unsupported", "text convolution spatial labels beyond nine")
            })?;
        }
        Ok(result.into_iter().collect())
    }
    let left = component(
        dimensions.input_batch_dimension,
        dimensions.input_feature_dimension,
        &dimensions.input_spatial_dimensions,
        ['b', 'f'],
    )?;
    let right = component(
        dimensions.kernel_input_feature_dimension,
        dimensions.kernel_output_feature_dimension,
        &dimensions.kernel_spatial_dimensions,
        ['i', 'o'],
    )?;
    let output = component(
        dimensions.output_batch_dimension,
        dimensions.output_feature_dimension,
        &dimensions.output_spatial_dimensions,
        ['b', 'f'],
    )?;
    let result = format!("{left}_{right}->{output}");
    if parse_convolution_dimensions(&result)? != *dimensions {
        return Err(Error::new(
            "unsupported",
            "convolution dimension labels cannot preserve these dimension numbers",
        ));
    }
    Ok(result)
}

pub fn parse_window(source: &str) -> Result<Window> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let window = parser.window(false)?;
    parser.finish()?;
    Ok(window)
}

pub fn print_window(window: &Window) -> Result<String> {
    reject_fields(window, &["dimensions"])?;
    if window.dimensions.is_empty() {
        return Ok(String::new());
    }
    for dimension in &window.dimensions {
        reject_fields(
            dimension,
            &[
                "size",
                "stride",
                "padding_low",
                "padding_high",
                "window_dilation",
                "base_dilation",
                "window_reversal",
            ],
        )?;
    }
    let values = |extract: fn(&WindowDimension) -> String| {
        window
            .dimensions
            .iter()
            .map(extract)
            .collect::<Vec<_>>()
            .join("x")
    };
    Ok(format!(
        "size={} stride={} pad={} lhs_dilate={} rhs_dilate={} rhs_reversal={}",
        values(|d| d.size.to_string()),
        values(|d| d.stride.to_string()),
        values(|d| format!("{}_{}", d.padding_low, d.padding_high)),
        values(|d| d.base_dilation.to_string()),
        values(|d| d.window_dilation.to_string()),
        values(|d| i32::from(d.window_reversal).to_string())
    ))
}

pub fn parse_shape(source: &str) -> Result<Shape> {
    parse_shape_with_options(source, ParseOptions::default())
}

pub fn parse_shape_with_options(source: &str, options: ParseOptions) -> Result<Shape> {
    let mut parser = Parser::new(source, options)?;
    let shape = parser.shape(0)?;
    parser.finish()?;
    Ok(shape)
}

pub fn print_shape(shape: &Shape) -> Result<String> {
    enum Work {
        Shape(Shape),
        Tuple(usize),
        Buffer,
    }
    let mut work = vec![Work::Shape(shape.clone())];
    let mut output: Vec<String> = Vec::new();
    while let Some(item) = work.pop() {
        match item {
            Work::Tuple(count) => {
                let start = output
                    .len()
                    .checked_sub(count)
                    .ok_or_else(|| Error::new("shape", "tuple printer stack underflow"))?;
                let children = output.split_off(start);
                output.push(format!("({})", children.join(", ")));
            }
            Work::Buffer => {
                let child = output
                    .pop()
                    .ok_or_else(|| Error::new("shape", "buffer printer stack underflow"))?;
                output.push(format!("b({child})"));
            }
            Work::Shape(shape) if shape.element_type() == PrimitiveType::BUFFER => {
                reject_fields(shape.as_proto(), &["element_type", "tuple_shapes"])?;
                let child = Shape::from_proto(shape.as_proto().tuple_shapes[0].clone())?;
                work.push(Work::Buffer);
                work.push(Work::Shape(child));
            }
            Work::Shape(shape) if shape.is_tuple() => {
                reject_fields(shape.as_proto(), &["element_type", "tuple_shapes"])?;
                let count = shape.as_proto().tuple_shapes.len();
                work.push(Work::Tuple(count));
                for child in shape.as_proto().tuple_shapes.iter().rev() {
                    work.push(Work::Shape(Shape::from_proto(child.clone())?));
                }
            }
            Work::Shape(shape) if !shape.is_array() => {
                reject_fields(shape.as_proto(), &["element_type"])?;
                match shape.element_type() {
                    PrimitiveType::TOKEN | PrimitiveType::OPAQUE_TYPE => {
                        output.push(format!("{}[]", shape.element_type().hlo_name()));
                    }
                    _ => return Err(Error::new("unsupported", "buffer shape text")),
                }
            }
            Work::Shape(shape) => {
                // A structural field whitelist prevents future schema additions from
                // being silently dropped by this intentionally partial printer.
                reject_fields(
                    shape.as_proto(),
                    &[
                        "element_type",
                        "dimensions",
                        "is_dynamic_dimension",
                        "layout",
                    ],
                )?;
                let dimensions = (0..shape.rank()?)
                    .map(|axis| match shape.dimension(axis)? {
                        Dimension::Static(n) => Ok(n.to_string()),
                        Dimension::Bounded(n) => Ok(format!("<={n}")),
                        Dimension::Unbounded => Ok("?".into()),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut result = format!(
                    "{}[{}]",
                    shape.element_type().hlo_name(),
                    dimensions.join(",")
                );
                if let Some(layout) = shape.layout() {
                    let layout = layout::print(layout)?;
                    if !shape.dimensions().is_empty() || layout != "{}" {
                        result.push_str(&layout);
                    }
                }
                output.push(result);
            }
        }
    }
    if output.len() != 1 {
        return Err(Error::new(
            "shape",
            "shape printer did not produce one result",
        ));
    }
    output
        .pop()
        .ok_or_else(|| Error::new("shape", "shape printer did not produce one result"))
}

/// Parses structure without requiring complete semantic verification, like
/// upstream ParseAndReturnUnverifiedModule.
pub fn parse_module(source: &str) -> Result<Module> {
    parse_module_with_options(source, ParseOptions::default())
}

pub fn parse_module_with_options(source: &str, options: ParseOptions) -> Result<Module> {
    let mut remaining_bytes = options.max_bytes;
    parse_module_nested(source, options, 0, &mut remaining_bytes)
}

/// Parses one native HLO instruction into a module with the implicit
/// shape-annotated operands materialized as entry parameters.
pub fn parse_instruction(source: &str) -> Result<Module> {
    parse_instruction_with_options(source, ParseOptions::default())
}

#[derive(Default)]
struct SingleInstructionBuilder {
    parameters: Vec<(String, Shape)>,
    next_parameter: usize,
    statements: Vec<String>,
    used_names: BTreeSet<String>,
}

impl SingleInstructionBuilder {
    fn unique_name(&mut self, base: &str) -> String {
        let mut name = base.to_owned();
        let mut suffix = 1_usize;
        while self.used_names.contains(&name) {
            name = format!("{base}.{suffix}");
            suffix += 1;
        }
        self.used_names.insert(name.clone());
        name
    }

    fn add_parameter(&mut self, requested_name: Option<String>, shape: Shape) -> Result<String> {
        if let Some(name) = requested_name.as_ref()
            && self.parameters.iter().any(|(existing, _)| existing == name)
        {
            return Ok(name.clone());
        }
        let requested_name = requested_name.unwrap_or_else(|| format!("_{}", self.next_parameter));
        let name = if self.used_names.contains(&requested_name) {
            self.unique_name(&requested_name)
        } else {
            self.used_names.insert(requested_name.clone());
            requested_name
        };
        let number = self.parameters.len();
        self.next_parameter = self.next_parameter.checked_add(1).ok_or_else(|| {
            Error::new(
                "resource-limit",
                "single-instruction parameter count overflow",
            )
        })?;
        self.parameters.push((name.clone(), shape.clone()));
        self.statements.push(format!(
            "%{name} = {} parameter({number})",
            print_shape(&shape)?
        ));
        Ok(name)
    }

    fn operation(
        &mut self,
        parser: &mut Parser,
        shape: Option<Shape>,
        opcode: String,
        instruction_name: String,
        root: bool,
        depth: usize,
    ) -> Result<String> {
        struct Frame {
            shape: Option<Shape>,
            opcode: String,
            instruction_name: String,
            root: bool,
            depth: usize,
            operands: Vec<String>,
            started: bool,
            after_nested: bool,
        }
        let mut frames = vec![Frame {
            shape,
            opcode,
            instruction_name,
            root,
            depth,
            operands: Vec::new(),
            started: false,
            after_nested: false,
        }];
        loop {
            let mut complete = false;
            {
                let frame = frames
                    .last_mut()
                    .ok_or_else(|| parser.error("single-instruction parser stack empty"))?;
                if frame.depth >= MAX_DEPTH {
                    return Err(parser.error("single-instruction nesting depth limit"));
                }
                frame
                    .opcode
                    .parse::<Opcode>()
                    .map_err(|_| parser.error(format!("unknown opcode {}", frame.opcode)))?;
                if !frame.started {
                    parser.expect("(")?;
                    frame.started = true;
                    complete = parser.take(")");
                } else if frame.after_nested {
                    frame.after_nested = false;
                    if parser.take(")") {
                        complete = true;
                    } else {
                        parser.expect(",")?;
                    }
                }
            }

            if !complete && !frames.last().is_some_and(|frame| frame.after_nested) {
                let operand_shape = if looks_like_shape(parser) {
                    Some(parser.shape(0)?)
                } else {
                    None
                };
                if matches!(parser.peek(), ")" | ",") {
                    let shape = operand_shape.clone().ok_or_else(|| {
                        parser.error("single-instruction operand requires a shape")
                    })?;
                    let name = self.add_parameter(None, shape)?;
                    frames
                        .last_mut()
                        .ok_or_else(|| parser.error("single-instruction parser stack empty"))?
                        .operands
                        .push(format!("%{name}"));
                    if parser.take(")") {
                        complete = true;
                    } else {
                        parser.expect(",")?;
                    }
                } else if parser
                    .tokens
                    .get(parser.index + 1)
                    .is_some_and(|token| token.value == "(")
                {
                    let name_or_opcode = parser.word()?;
                    if let Some(shape) = operand_shape.clone() {
                        self.add_parameter(Some(name_or_opcode.clone()), shape)?;
                    }
                    let nested_name = self.unique_name(&format!("{name_or_opcode}.anon"));
                    let nested_depth = frames
                        .last()
                        .map(|frame| frame.depth + 1)
                        .ok_or_else(|| parser.error("single-instruction parser stack empty"))?;
                    frames
                        .last_mut()
                        .ok_or_else(|| parser.error("single-instruction parser stack empty"))?
                        .after_nested = true;
                    frames.push(Frame {
                        shape: operand_shape,
                        opcode: name_or_opcode,
                        instruction_name: nested_name,
                        root: false,
                        depth: nested_depth,
                        operands: Vec::new(),
                        started: false,
                        after_nested: false,
                    });
                } else {
                    let name_or_opcode = parser.name()?;
                    let parameter_name = if let Some((existing, _)) = self
                        .parameters
                        .iter()
                        .find(|(existing, _)| existing == &name_or_opcode)
                    {
                        existing.clone()
                    } else {
                        let shape = operand_shape.clone().ok_or_else(|| {
                            parser.error(format!(
                                "operand {name_or_opcode} has no definition or explicit shape"
                            ))
                        })?;
                        self.add_parameter(Some(name_or_opcode.clone()), shape)?
                    };
                    let operand = if let Some(shape) = operand_shape {
                        format!("{} %{parameter_name}", print_shape(&shape)?)
                    } else {
                        format!("%{parameter_name}")
                    };
                    frames
                        .last_mut()
                        .ok_or_else(|| parser.error("single-instruction parser stack empty"))?
                        .operands
                        .push(operand);
                    if parser.take(")") {
                        complete = true;
                    } else {
                        parser.expect(",")?;
                    }
                }
            }

            if complete {
                let frame = frames
                    .pop()
                    .ok_or_else(|| parser.error("single-instruction parser stack empty"))?;
                let mut instruction = String::new();
                if let Some(shape) = frame.shape {
                    write!(instruction, "{} ", print_shape(&shape)?)
                        .map_err(|error| Error::new("print", error.to_string()))?;
                }
                write!(
                    instruction,
                    "{}({})",
                    frame.opcode,
                    frame.operands.join(", ")
                )
                .map_err(|error| Error::new("print", error.to_string()))?;
                let result = if frame.root {
                    if self.used_names.contains(&frame.instruction_name) {
                        return Err(parser.error(format!(
                            "duplicate instruction name {}",
                            frame.instruction_name
                        )));
                    }
                    self.used_names.insert(frame.instruction_name.clone());
                    format!("ROOT %{} = {instruction}", frame.instruction_name)
                } else {
                    self.statements
                        .push(format!("%{} = {instruction}", frame.instruction_name));
                    format!("%{}", frame.instruction_name)
                };
                if let Some(parent) = frames.last_mut() {
                    parent.operands.push(result);
                } else {
                    return Ok(result);
                }
            }
        }
    }
}

/// Parses one native HLO instruction with explicit text and literal limits.
pub fn parse_instruction_with_options(source: &str, options: ParseOptions) -> Result<Module> {
    let mut parser = Parser::new(source, options)?;
    let marked_root = parser.take("ROOT");
    let mut root_name = None;
    if !looks_like_shape(&parser) {
        let name = parser.name()?;
        parser.expect("=")?;
        root_name = Some(name);
    } else if marked_root {
        return Err(parser.error("ROOT requires a named instruction"));
    }
    let root_shape = if looks_like_shape(&parser) {
        Some(parser.shape(0)?)
    } else {
        None
    };
    let opcode = parser.word()?;
    let mut builder = SingleInstructionBuilder::default();

    let root = builder.operation(
        &mut parser,
        root_shape,
        opcode,
        root_name.unwrap_or_else(|| "_".into()),
        true,
        0,
    )?;
    let attributes = parser.tokens[parser.index..]
        .iter()
        .map(|token| token.value.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    parser.index = parser.tokens.len();
    parser.finish()?;
    builder.statements.push(root);
    let mut body = builder.statements.join(" ");
    if !attributes.is_empty() {
        write!(body, " {attributes}").map_err(|error| Error::new("print", error.to_string()))?;
    }
    let wrapped = format!("HloModule _ ENTRY _ {{ {body} }}");
    let mut module = parse_module_with_options(&wrapped, options)?;
    module.wire.name = "_".into();
    let mut schedule = crate::proto::hlo::HloScheduleProto::new();
    for computation in &module.wire.computations {
        if !computation.is_fusion_computation {
            schedule.sequences.insert(
                computation.id,
                crate::proto::hlo::hlo_schedule_proto::InstructionSequence {
                    instruction_ids: computation.instructions.iter().map(|i| i.id).collect(),
                    ..Default::default()
                },
            );
        }
    }
    module.set_schedule(Some(schedule))?;
    Ok(module)
}

fn parse_module_nested(
    source: &str,
    options: ParseOptions,
    module_depth: usize,
    remaining_bytes: &mut usize,
) -> Result<Module> {
    struct Frame {
        proto: HloModuleProto,
        config: crate::ModuleConfig,
        channel_presence: BTreeSet<i64>,
        pending: Vec<(Option<usize>, String)>,
        next: usize,
        depth: usize,
        parent_entry: Option<usize>,
    }

    let ParsedModuleShell {
        module,
        pending_recovery_modules,
    } = parse_module_shell(source, options, module_depth, remaining_bytes)?;
    let mut frames = vec![Frame {
        proto: module.to_proto_with_program_shapes()?,
        config: module.config(),
        channel_presence: module.channel_presence.clone(),
        pending: pending_recovery_modules,
        next: 0,
        depth: module_depth,
        parent_entry: None,
    }];
    loop {
        let next_module = frames.last_mut().and_then(|frame| {
            let next = frame.pending.get(frame.next).cloned();
            if next.is_some() {
                frame.next += 1;
            }
            next.map(|(entry, source)| (entry, source, frame.depth + 1))
        });
        if let Some((entry, source, depth)) = next_module {
            let ParsedModuleShell {
                module,
                pending_recovery_modules,
            } = parse_module_shell(&source, options, depth, remaining_bytes)?;
            if module.config() != crate::ModuleConfig::default() {
                return Err(Error::new(
                    "unsupported",
                    "nested recovery module configuration is not representable in HloModuleProto",
                ));
            }
            frames.push(Frame {
                proto: module.to_proto_with_program_shapes()?,
                config: module.config(),
                channel_presence: module.channel_presence.clone(),
                pending: pending_recovery_modules,
                next: 0,
                depth,
                parent_entry: entry,
            });
            continue;
        }

        let completed = frames
            .pop()
            .ok_or_else(|| Error::new("module", "recovery parser stack underflow"))?;
        if let Some(parent) = frames.last_mut() {
            if completed
                .proto
                .computations
                .iter()
                .flat_map(|c| &c.instructions)
                .any(|instruction| {
                    instruction.channel_id <= 0
                        && completed.channel_presence.contains(&instruction.id)
                })
            {
                return Err(Error::new(
                    "unsupported",
                    "nested recovery module cannot preserve nonpositive channel presence",
                ));
            }
            if let Some(entry) = completed.parent_entry {
                let recovery = parent
                    .proto
                    .original_value_recovery_table
                    .as_mut()
                    .and_then(|table| table.entries.get_mut(entry))
                    .ok_or_else(|| Error::new("module", "recovery entry index out of range"))?;
                recovery.recovery_module = Some(completed.proto).into();
            }
        } else {
            let mut module = Module::from_proto(completed.proto)?;
            module.set_config(completed.config);
            module.channel_presence = completed.channel_presence;
            return Ok(module);
        }
    }
}

struct ParsedModuleShell {
    module: Module,
    pending_recovery_modules: Vec<(Option<usize>, String)>,
}

fn parse_module_shell(
    source: &str,
    options: ParseOptions,
    module_depth: usize,
    remaining_bytes: &mut usize,
) -> Result<ParsedModuleShell> {
    if module_depth >= MAX_DEPTH {
        return Err(Error::new(
            "resource-limit",
            "recovery module nesting limit",
        ));
    }
    *remaining_bytes = remaining_bytes
        .checked_sub(source.len())
        .ok_or_else(|| Error::new("resource-limit", "aggregate module text byte limit"))?;
    let mut parser = Parser::new(source, options)?;
    let headerless = !parser.take("HloModule");
    if headerless
        && parser.peek() != "ENTRY"
        && parser
            .tokens
            .get(parser.index + 1)
            .is_none_or(|token| token.value != "{")
    {
        return Err(parser.error("expected HloModule or a computation body"));
    }
    let mut module = HloModuleProto {
        name: if headerless {
            String::new()
        } else {
            parser.name()?
        },
        frontend_attributes: Some(crate::proto::xla_data::FrontendAttributes::new()).into(),
        ..Default::default()
    };
    let mut pending_recovery_modules = Vec::new();
    let mut header_attributes = BTreeSet::new();
    let mut is_scheduled = false;
    let mut config = crate::ModuleConfig::default();
    while !headerless && parser.take(",") {
        let attribute = parser.word()?;
        parser.expect("=")?;
        if !header_attributes.insert(attribute.clone()) {
            return Err(parser.error("duplicate module attribute"));
        }
        match attribute.as_str() {
            "origin_recovery_table" => {
                let recovery = module_attributes::parse_recovery(&mut parser)?;
                module.original_value_recovery_table = recovery.table.into();
                pending_recovery_modules.extend(recovery.pending_modules);
                continue;
            }
            "frontend_attributes" => {
                let attributes = callable::parse_frontend(&mut parser)?;
                module.frontend_attributes = Some(attributes).into();
                continue;
            }
            "backend_config" => {
                module.backend_config = parser.string_or_dictionary()?;
                continue;
            }
            "replica_count" => {
                config.replica_count = parser.integer()?;
                continue;
            }
            "alias_passthrough_params" => {
                config.alias_passthrough_params = parser.boolean()?;
                continue;
            }
            "allow_spmd_sharding_propagation_to_parameters" => {
                let values = parser.booleans()?;
                if !values.is_empty() {
                    config.allow_spmd_sharding_propagation_to_parameters = values;
                }
                continue;
            }
            "allow_spmd_sharding_propagation_to_output" => {
                let values = parser.booleans()?;
                if !values.is_empty() {
                    config.allow_spmd_sharding_propagation_to_output = values;
                }
                continue;
            }
            "num_partitions" => {
                config.num_partitions = parser.integer()?;
                config.use_spmd_partitioning = config.num_partitions != 1;
                continue;
            }
            "debug_attributes" => {
                module.debug_attributes = module_attributes::parse_debug_attributes(&mut parser)?;
                continue;
            }
            "is_scheduled" => {
                is_scheduled = parser.boolean()?;
                continue;
            }
            "input_output_alias" => {
                module.input_output_alias =
                    Some(module_attributes::parse_aliases(&mut parser)?).into();
                continue;
            }
            "buffer_donor" => {
                module.buffer_donor = Some(module_attributes::parse_donors(&mut parser)?).into();
                continue;
            }
            _ => {}
        }
        if attribute != "entry_computation_layout" {
            return Err(
                Error::new("unsupported", format!("module text attribute {attribute}"))
                    .at(parser.offset()),
            );
        }
        parser.expect("{")?;
        parser.expect("(")?;
        let mut program = ProgramShapeProto::new();
        if !parser.take(")") {
            loop {
                program.parameters.push(parser.shape(0)?.into_proto());
                if parser.take(")") {
                    break;
                }
                parser.expect(",")?;
            }
        }
        parser.expect("->")?;
        program.result = Some(parser.shape(0)?.into_proto()).into();
        parser.expect("}")?;
        module.host_program_shape = Some(program).into();
    }
    module.stack_frame_index = stack_frames::parse(&mut parser)?.into();
    let mut next_instruction = 0_i64;
    let mut parsed_names = crate::ir::names::NameUniquer::default();
    let mut next_computation = 0_i64;
    let mut sugar = BTreeMap::new();
    let mut generated_computations = BTreeSet::new();
    let mut pending_callees: Vec<(i64, Vec<String>)> = Vec::new();
    let mut async_backend_attributes = BTreeSet::new();
    let mut channel_presence = BTreeSet::new();
    let mut entry_seen = false;
    while !parser.peek().is_empty() {
        let entry = parser.take("ENTRY");
        if entry && std::mem::replace(&mut entry_seen, true) {
            return Err(parser.error("multiple entry computations"));
        }
        let name = parser.name()?;
        let id = next_computation;
        next_computation += 1;
        if entry {
            module.entry_computation_name = name.clone();
            module.entry_computation_id = id;
        }
        let mut declared_result = None;
        if parser.take("(") {
            if !parser.take(")") {
                loop {
                    parser.name()?;
                    parser.expect(":")?;
                    parser.shape(0)?;
                    if parser.take(")") {
                        break;
                    }
                    parser.expect(",")?;
                }
            }
            parser.expect("->")?;
            declared_result = Some(parser.shape(0)?);
        }
        parser.expect("{")?;
        let mut computation = HloComputationProto {
            name,
            id,
            root_id: -1,
            ..Default::default()
        };
        let mut names = BTreeMap::new();
        let mut pending_operands = Vec::new();
        let mut pending_controls = Vec::new();
        let mut broadcasts_without_dimensions = BTreeSet::new();
        struct InstructionFrame {
            depth: usize,
            head: Box<InstructionHead>,
            operands: InstructionOperands,
            started: bool,
            after_nested: bool,
        }
        enum OperandProgress {
            Complete,
            Nested,
        }
        fn parse_instruction(context: &mut InstructionContext<'_>, depth: usize) -> Result<i64> {
            if depth >= MAX_DEPTH {
                return Err(context.parser.error("nested instruction depth limit"));
            }
            let head = parse_instruction_head(context, depth)?;
            let mut frames = vec![InstructionFrame {
                depth,
                head,
                operands: Vec::new(),
                started: false,
                after_nested: false,
            }];
            loop {
                let progress = {
                    let frame = frames
                        .last_mut()
                        .ok_or_else(|| context.parser.error("instruction parser stack empty"))?;
                    parse_instruction_operands(
                        context,
                        frame.depth,
                        &mut frame.head,
                        &mut frame.operands,
                        &mut frame.started,
                        &mut frame.after_nested,
                    )?
                };
                match progress {
                    OperandProgress::Nested => {
                        let depth =
                            frames.last().map(|frame| frame.depth + 1).ok_or_else(|| {
                                context.parser.error("instruction parser stack empty")
                            })?;
                        if depth >= MAX_DEPTH {
                            return Err(context.parser.error("nested instruction depth limit"));
                        }
                        frames.push(InstructionFrame {
                            depth,
                            head: parse_instruction_head(context, depth)?,
                            operands: Vec::new(),
                            started: false,
                            after_nested: false,
                        });
                    }
                    OperandProgress::Complete => {
                        let mut frame = frames.pop().ok_or_else(|| {
                            context.parser.error("instruction parser stack empty")
                        })?;
                        let id = finish_instruction(
                            context,
                            frame.depth,
                            &mut frame.head,
                            frame.operands,
                        )?;
                        if let Some(parent) = frames.last_mut() {
                            parent
                                .operands
                                .push((format!("#nested{id}"), Vec::new(), None));
                        } else {
                            return Ok(id);
                        }
                    }
                }
            }
        }
        fn parse_instruction_head(
            context: &mut InstructionContext<'_>,
            depth: usize,
        ) -> Result<Box<InstructionHead>> {
            let parser = &mut *context.parser;
            let computation = &mut *context.computation;
            let names = &mut *context.names;
            let next_instruction = &mut *context.next_instruction;
            let sugar = &mut *context.sugar;
            if parser.peek().is_empty() {
                return Err(parser.error("unterminated computation"));
            }
            let root = depth == 0 && parser.take("ROOT");
            if root && computation.root_id >= 0 {
                return Err(parser.error("multiple roots"));
            }
            let name = if depth == 0 {
                parser.name()?
            } else {
                format!("#nested{next_instruction}")
            };
            if depth == 0 && !valid_instruction_name(&name) {
                return Err(
                    parser.error("instruction name would be changed by native sanitization")
                );
            }
            if names.insert(name.clone(), *next_instruction).is_some() {
                return Err(parser.error("duplicate instruction name"));
            }
            if depth == 0 {
                parser.expect("=")?;
            }
            let outer_shape = if looks_like_shape(parser) {
                Some(parser.shape(0)?)
            } else {
                None
            };
            let (outer_opcode, wrapped) = asynchronous::parse_opcode(&parser.word()?)?;
            if outer_shape.is_none() && !inferred::can_omit_shape(outer_opcode) {
                return Err(parser.error("opcode requires an explicit result shape"));
            }
            if wrapped.is_some() && matches!(outer_opcode, Opcode::AsyncStart | Opcode::AsyncUpdate)
            {
                asynchronous::context_shapes(inferred::explicit(&outer_shape)?)?;
            }
            let opcode = if outer_opcode == Opcode::AsyncStart {
                wrapped.unwrap_or(outer_opcode)
            } else {
                outer_opcode
            };
            if let Some(wrapped) = wrapped {
                sugar.insert(*next_instruction, wrapped);
            }
            let shape = if wrapped.is_some() && outer_opcode == Opcode::AsyncStart {
                Some(asynchronous::context_shapes(inferred::explicit(&outer_shape)?)?.1)
            } else {
                outer_shape.clone()
            };
            let mut instruction = HloInstructionProto {
                name: if depth == 0 {
                    name
                } else {
                    format!("{}.anon", outer_opcode.name())
                },
                id: *next_instruction,
                opcode: opcode.name().into(),
                shape: shape.as_ref().map(|shape| shape.as_proto().clone()).into(),
                ..Default::default()
            };
            match opcode {
                Opcode::TopK => {
                    instruction.largest = true;
                    instruction.is_stable = true;
                }
                Opcode::Cholesky => {
                    instruction.cholesky_options.mut_or_insert_default();
                }
                Opcode::TriangularSolve => {
                    instruction.triangular_solve_options.mut_or_insert_default();
                }
                Opcode::Convolution => {
                    instruction.feature_group_count = 1;
                    instruction.batch_group_count = 1;
                    instruction.window.mut_or_insert_default();
                }
                Opcode::CustomCall => {
                    instruction.feature_group_count = 1;
                    instruction.batch_group_count = 1;
                    instruction.custom_call_api_version =
                        crate::proto::hlo::CustomCallApiVersion::API_VERSION_ORIGINAL.into();
                }
                Opcode::SelectAndScatter | Opcode::ReduceWindow => {
                    instruction.window.mut_or_insert_default();
                }
                Opcode::Scan => {
                    instruction.is_associative =
                        crate::proto::hlo::TriState::TRI_STATE_FALSE.into();
                }
                _ => {}
            }
            *next_instruction = next_instruction
                .checked_add(1)
                .ok_or_else(|| parser.error("instruction count overflow"))?;
            if root {
                computation.root_id = instruction.id;
            }
            Ok(Box::new(InstructionHead {
                instruction,
                outer_shape,
                shape,
                outer_opcode,
                opcode,
                wrapped,
            }))
        }
        fn parse_instruction_operands(
            context: &mut InstructionContext<'_>,
            _depth: usize,
            head: &mut InstructionHead,
            operands: &mut InstructionOperands,
            started: &mut bool,
            after_nested: &mut bool,
        ) -> Result<OperandProgress> {
            let parser = &mut *context.parser;
            let options = parser.options;
            let opcode = head.opcode;
            let outer_opcode = head.outer_opcode;
            let wrapped = head.wrapped;
            let shape = &head.shape;
            let instruction = &mut head.instruction;
            let mut operands_closed = false;
            if !*started {
                *started = true;
                parser.expect("(")?;
            } else if *after_nested {
                *after_nested = false;
                operands_closed = parser.take(")");
                if !operands_closed {
                    parser.expect(",")?;
                }
            }
            if !operands_closed {
                if opcode == Opcode::Parameter && wrapped.is_none() {
                    instruction.parameter_number = parser.integer()?;
                    parser.expect(")")?;
                    return Ok(OperandProgress::Complete);
                } else if opcode == Opcode::Constant && wrapped.is_none() {
                    parser.parse_constant(shape, instruction, options.max_literal_elements)?;
                    parser.expect(")")?;
                    return Ok(OperandProgress::Complete);
                } else if !parser.take(")") {
                    loop {
                        let start = parser.index;
                        let written_shape = if looks_like_shape(parser) {
                            Some(parser.shape(0)?)
                        } else {
                            None
                        };
                        let name = parser.name()?;
                        if parser.peek() == "(" {
                            parser.index = start;
                            *after_nested = true;
                            return Ok(OperandProgress::Nested);
                        } else {
                            let mut indices = Vec::new();
                            while parser.take("#") {
                                indices.push(parser.integer()?);
                                if indices.len() > MAX_DEPTH {
                                    return Err(
                                        parser.error("compact tuple projection depth limit")
                                    );
                                }
                            }
                            operands.push((name, indices, written_shape));
                        }
                        if parser.take(")") {
                            break;
                        }
                        parser.expect(",")?;
                    }
                }
            }
            if wrapped.is_some() && outer_opcode == Opcode::AsyncStart {
                if opcode == Opcode::Parameter {
                    parser.expect("(")?;
                    instruction.parameter_number = parser.integer()?;
                    parser.expect(")")?;
                } else if opcode == Opcode::Constant {
                    parser.expect("(")?;
                    parser.parse_constant(shape, instruction, options.max_literal_elements)?;
                    parser.expect(")")?;
                }
            }
            Ok(OperandProgress::Complete)
        }
        fn finish_instruction(
            context: &mut InstructionContext<'_>,
            depth: usize,
            head: &mut InstructionHead,
            operands: InstructionOperands,
        ) -> Result<i64> {
            let parser = &mut *context.parser;
            let options = parser.options;
            let module = &mut *context.module;
            let computation = &mut *context.computation;
            let names = &mut *context.names;
            let parsed_names = &mut *context.parsed_names;
            let next_instruction = &mut *context.next_instruction;
            let next_computation = &mut *context.next_computation;
            let generated_computations = &mut *context.generated_computations;
            let pending_callees = &mut *context.pending_callees;
            let pending_operands = &mut *context.pending_operands;
            let pending_controls = &mut *context.pending_controls;
            let async_backend_attributes = &mut *context.async_backend_attributes;
            let channel_presence = &mut *context.channel_presence;
            let broadcasts_without_dimensions = &mut *context.broadcasts_without_dimensions;
            let opcode = head.opcode;
            let outer_opcode = head.outer_opcode;
            let wrapped = head.wrapped;
            let shape = head.shape.take();
            let outer_shape = head.outer_shape.take();
            let mut instruction = std::mem::take(&mut head.instruction);
            let mut seen_attributes = BTreeSet::new();
            let mut callees = Vec::new();
            let mut named_callees = BTreeMap::new();
            let mut controls = Vec::new();
            while depth == 0 && parser.take(",") {
                let attribute = parser.word()?;
                if wrapped.is_some()
                    && outer_opcode != Opcode::AsyncStart
                    && !matches!(
                        attribute.as_str(),
                        "sharding"
                            | "metadata"
                            | "statistics"
                            | "parameter_replication"
                            | "origin"
                            | "backend_config"
                            | "frontend_attributes"
                            | "control-predecessors"
                            | "async_execution_thread"
                            | "output_to_operand_aliasing"
                    )
                {
                    return Err(parser.error("attribute does not apply to async update/done sugar"));
                }
                if !seen_attributes.insert(attribute.clone()) {
                    return Err(parser.error("duplicate instruction attribute"));
                }
                parser.expect("=")?;
                match attribute.as_str() {
                    "dimensions" => instruction.dimensions = parser.integers("{", "}")?,
                    "sharding" => instruction.sharding = Some(parser.sharding()?).into(),
                    "domain" if opcode == Opcode::Domain => parser.domain(&mut instruction)?,
                    "metadata" => instruction.metadata = Some(parser.metadata()?).into(),
                    "statistics" => instruction.statistics_viz = Some(parser.statistics()?).into(),
                    "parameter_replication" => {
                        if opcode != Opcode::Parameter {
                            return Err(parser.error("parameter replication requires a parameter"));
                        }
                        let replication = parser.parameter_replication()?;
                        if replication.replicated_at_leaf_buffers.len()
                            != inferred::explicit(&shape)?.leaf_count()
                        {
                            return Err(parser.error(
                                "parameter replication count differs from shape leaf count",
                            ));
                        }
                        instruction.parameter_replication = Some(replication).into();
                    }
                    "origin" => instruction.original_value = Some(parser.original_value()?).into(),
                    "result_accuracy" if opcode.supports_result_accuracy() => {
                        instruction.result_accuracy = Some(parser.accuracy()?).into()
                    }
                    "backend_config" => {
                        instruction.backend_config = parser.string_or_dictionary()?
                    }
                    "channel_id" => {
                        if !crate::ir::channels::supported(instruction.opcode.as_str()) {
                            return Err(parser.error("opcode does not carry a channel"));
                        }
                        instruction.channel_id = parser.integer()?;
                        channel_presence.insert(instruction.id);
                    }
                    "replica_groups" => collective::parse_groups(parser, &mut instruction)?,
                    "constrain_layout" => instruction.constrain_layout = parser.boolean()?,
                    "use_global_device_ids" => {
                        instruction.use_global_device_ids = parser.boolean()?
                    }
                    "has_dynamic_root" => instruction.has_dynamic_root = parser.boolean()?,
                    "is_host_transfer" => instruction.is_host_transfer = parser.boolean()?,
                    "cross_program_prefetch_index" => instruction.set_cross_program_prefetch_index(
                        i32::try_from(parser.integer()?).map_err(|_| {
                            parser.error("cross-program prefetch index exceeds i32")
                        })?,
                    ),
                    "frontend_attributes" => {
                        instruction.frontend_attributes =
                            Some(callable::parse_frontend(parser)?).into()
                    }
                    "custom_call_target" => {
                        instruction.custom_call_target = String::from_utf8(parser.byte_string()?)
                            .map_err(|_| parser.error("custom-call target must be valid UTF-8"))?
                    }
                    "literal" if opcode == Opcode::CustomCall => {
                        let mut remaining = options.max_literal_elements;
                        instruction.literal =
                            Some(parser.typed_literal(0, &mut remaining)?.into_proto()).into();
                    }
                    "custom_call_has_side_effect" => {
                        instruction.custom_call_has_side_effect = parser.boolean()?
                    }
                    "is_composite" => instruction.is_composite = parser.boolean()?,
                    "kind" => {
                        instruction.fusion_kind = parser.word()?;
                        if !matches!(
                            instruction.fusion_kind.as_str(),
                            "kLoop" | "kInput" | "kOutput" | "kCustom"
                        ) {
                            return Err(parser.error("unknown fusion kind"));
                        }
                    }
                    "output_to_operand_aliasing" => {
                        if !matches!(
                            outer_opcode,
                            Opcode::Fusion
                                | Opcode::CustomCall
                                | Opcode::AsyncStart
                                | Opcode::AsyncUpdate
                        ) {
                            return Err(
                                parser.error("opcode does not support output aliases in text")
                            );
                        }
                        instruction.output_operand_aliasing = callable::parse_aliases(parser)?
                    }
                    "async_execution_thread" => {
                        if !matches!(
                            outer_opcode,
                            Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone
                        ) {
                            return Err(parser.error(
                                "async_execution_thread requires a generic async instruction",
                            ));
                        }
                        instruction.async_execution_thread =
                            String::from_utf8(parser.byte_string()?).map_err(|_| {
                                parser.error("async execution thread must be valid UTF-8")
                            })?;
                        if instruction.async_execution_thread == "main" {
                            instruction.async_execution_thread.clear();
                        }
                    }
                    "operand_layout_constraints" => {
                        instruction.constrain_layout = true;
                        instruction.operand_shapes_with_layout = callable::parse_layouts(parser)?;
                    }
                    "api_version" => {
                        instruction.custom_call_api_version = callable::api_version(parser)?.into()
                    }
                    "schedule" => {
                        instruction.custom_call_schedule = callable::schedule(parser)?.into()
                    }
                    "window" => instruction.window = Some(parser.window(true)?).into(),
                    "is_stable" => instruction.is_stable = parser.boolean()?,
                    "largest" => instruction.largest = parser.boolean()?,
                    "k" => instruction.k = parser.integer()?,
                    "infeed_config" => instruction.infeed_config = parser.byte_string()?,
                    "outfeed_config" => instruction.outfeed_config = parser.byte_string()?,
                    "outfeed_shape" => {
                        instruction.outfeed_shape = Some(parser.shape(0)?.into_proto()).into()
                    }
                    "distribution" => {
                        instruction.distribution =
                            crate::proto::xla_data::RandomDistribution::from_str(
                                &parser.word()?.to_ascii_uppercase(),
                            )
                            .ok_or_else(|| parser.error("unknown random distribution"))?
                            .into()
                    }
                    "delta" => instruction.delta = parser.integer()?,
                    "num_carries" => instruction.num_carries = parser.integer()?,
                    "is_reverse" => instruction.is_reverse = parser.boolean()?,
                    "is_associative" => {
                        instruction.is_associative = if parser.boolean()? {
                            crate::proto::hlo::TriState::TRI_STATE_TRUE
                        } else {
                            crate::proto::hlo::TriState::TRI_STATE_FALSE
                        }
                        .into()
                    }
                    "mode" => {
                        if matches!(
                            opcode,
                            Opcode::AllReduce | Opcode::AllReduceStart | Opcode::ReduceScatter
                        ) {
                            if !matches!(
                                parser.word()?.as_str(),
                                "cross_replica"
                                    | "cross_partition"
                                    | "cross_replica_and_partition"
                                    | "flattened_id"
                            ) {
                                return Err(parser.error("unknown collective group mode"));
                            }
                            continue;
                        }
                        if !parser.word()?.eq_ignore_ascii_case("rotate") {
                            return Err(parser.error("unknown shuffle mode"));
                        }
                        instruction
                            .shuffle_mode
                            .mut_or_insert_default()
                            .mut_rotate();
                    }
                    "shifts" => {
                        instruction
                            .shuffle_mode
                            .mut_or_insert_default()
                            .mut_rotate()
                            .shifts = parser.integers("{", "}")?
                    }
                    "feature_group_count" => instruction.feature_group_count = parser.integer()?,
                    "batch_group_count" => instruction.batch_group_count = parser.integer()?,
                    "dim_labels" => {
                        let inputs = parser.word()?;
                        parser.expect("->")?;
                        let output = parser.word()?;
                        instruction.convolution_dimension_numbers = Some(
                            parse_convolution_dimensions(&format!("{inputs}->{output}"))?,
                        )
                        .into();
                    }
                    "convolution_kind" => {
                        instruction.conv_kind = crate::proto::hlo::ConvolutionKind::from_str(
                            &format!("CONVOLUTION_KIND_{}", parser.word()?.to_ascii_uppercase()),
                        )
                        .ok_or_else(|| parser.error("unknown convolution kind"))?
                        .into()
                    }
                    "indices_are_sorted" => instruction.indices_are_sorted = parser.boolean()?,
                    "unique_indices" => instruction.unique_indices = parser.boolean()?,
                    "index_vector_dim" => {
                        let value = parser.integer()?;
                        if opcode == Opcode::Gather {
                            instruction
                                .gather_dimension_numbers
                                .mut_or_insert_default()
                                .index_vector_dim = value;
                        } else {
                            instruction
                                .scatter_dimension_numbers
                                .mut_or_insert_default()
                                .index_vector_dim = value;
                        }
                    }
                    "slice_sizes"
                        if matches!(
                            opcode,
                            Opcode::CollectivePermute | Opcode::CollectivePermuteStart
                        ) =>
                    {
                        instruction.dynamic_slice_sizes =
                            parser.integer_lists()?.into_iter().flatten().collect();
                        if instruction.dynamic_slice_sizes.is_empty() {
                            return Err(Error::new(
                                "unsupported",
                                "empty in-place slice lists have no presence in the pinned wire schema",
                            ));
                        }
                    }
                    "slice_sizes" => instruction.gather_slice_sizes = parser.integers("{", "}")?,
                    "source_target_pairs" => {
                        instruction.source_target_pairs = parser
                            .integer_lists()?
                            .into_iter()
                            .map(|pair| {
                                let [source, target] = pair.as_slice() else {
                                    return Err(parser.error("source_target_pairs requires pairs"));
                                };
                                Ok(crate::proto::xla_data::SourceTarget {
                                    source: *source,
                                    target: *target,
                                    ..Default::default()
                                })
                            })
                            .collect::<Result<_>>()?;
                    }
                    "offset_dims"
                    | "collapsed_slice_dims"
                    | "start_index_map"
                    | "operand_batching_dims"
                    | "start_indices_batching_dims" => {
                        let values = parser.integers("{", "}")?;
                        let dimensions =
                            instruction.gather_dimension_numbers.mut_or_insert_default();
                        match attribute.as_str() {
                            "offset_dims" => dimensions.offset_dims = values,
                            "collapsed_slice_dims" => dimensions.collapsed_slice_dims = values,
                            "start_index_map" => dimensions.start_index_map = values,
                            "operand_batching_dims" => dimensions.operand_batching_dims = values,
                            _ => dimensions.start_indices_batching_dims = values,
                        }
                    }
                    "update_window_dims"
                    | "inserted_window_dims"
                    | "scatter_dims_to_operand_dims"
                    | "input_batching_dims"
                    | "scatter_indices_batching_dims" => {
                        let values = parser.integers("{", "}")?;
                        let dimensions = instruction
                            .scatter_dimension_numbers
                            .mut_or_insert_default();
                        match attribute.as_str() {
                            "update_window_dims" => dimensions.update_window_dims = values,
                            "inserted_window_dims" => dimensions.inserted_window_dims = values,
                            "scatter_dims_to_operand_dims" => {
                                dimensions.scatter_dims_to_operand_dims = values
                            }
                            "input_batching_dims" => dimensions.input_batching_dims = values,
                            _ => dimensions.scatter_indices_batching_dims = values,
                        }
                    }
                    "feature_index" => instruction.feature_index = parser.integer()?,
                    "epsilon" => {
                        let value = parser.number()?;
                        if value > f64::from(f32::MAX) || value < f64::from(f32::MIN) {
                            return Err(parser.error("epsilon exceeds float range"));
                        }
                        instruction.epsilon = value as f32;
                    }
                    "left_side" => {
                        instruction
                            .triangular_solve_options
                            .mut_or_insert_default()
                            .left_side = parser.boolean()?
                    }
                    "unit_diagonal" => {
                        instruction
                            .triangular_solve_options
                            .mut_or_insert_default()
                            .unit_diagonal = parser.boolean()?
                    }
                    "lower" => {
                        let value = parser.boolean()?;
                        if opcode == Opcode::Cholesky {
                            instruction.cholesky_options.mut_or_insert_default().lower = value;
                        } else {
                            instruction
                                .triangular_solve_options
                                .mut_or_insert_default()
                                .lower = value;
                        }
                    }
                    "transpose_a" => {
                        instruction
                            .triangular_solve_options
                            .mut_or_insert_default()
                            .transpose_a =
                            crate::proto::xla_data::triangular_solve_options::Transpose::from_str(
                                &parser.word()?,
                            )
                            .ok_or_else(|| parser.error("unknown triangular solve transpose"))?
                            .into()
                    }
                    "fft_type" => {
                        instruction.fft_type = FftType::from_str(&parser.word()?)
                            .ok_or_else(|| parser.error("unknown FFT type"))?
                            .into()
                    }
                    "fft_length" => instruction.fft_length = parser.integers("{", "}")?,
                    "lhs_contracting_dims"
                    | "rhs_contracting_dims"
                    | "lhs_batch_dims"
                    | "rhs_batch_dims" => {
                        let values = parser.integers("{", "}")?;
                        let dimensions = if opcode == Opcode::RaggedDot {
                            instruction
                                .ragged_dot_dimension_numbers
                                .mut_or_insert_default()
                                .dot_dimension_numbers
                                .mut_or_insert_default()
                        } else {
                            instruction.dot_dimension_numbers.mut_or_insert_default()
                        };
                        match attribute.as_str() {
                            "lhs_contracting_dims" => {
                                dimensions.lhs_contracting_dimensions = values
                            }
                            "rhs_contracting_dims" => {
                                dimensions.rhs_contracting_dimensions = values
                            }
                            "lhs_batch_dims" => dimensions.lhs_batch_dimensions = values,
                            _ => dimensions.rhs_batch_dimensions = values,
                        }
                    }
                    "operand_precision" => {
                        let values = parser
                            .names()?
                            .into_iter()
                            .map(|name| {
                                Precision::from_str(&name.to_ascii_uppercase())
                                    .map(Into::into)
                                    .ok_or_else(|| parser.error("unknown operand precision"))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        instruction
                            .precision_config
                            .mut_or_insert_default()
                            .operand_precision = values;
                    }
                    "algorithm" => {
                        if opcode == Opcode::RaggedDot {
                            return Err(
                                parser.error("ragged-dot does not support a precision algorithm")
                            );
                        }
                        let value = parser.word()?.to_ascii_uppercase();
                        if opcode == Opcode::RngBitGenerator {
                            instruction.rng_algorithm =
                                crate::proto::xla_data::RandomAlgorithm::from_str(&value)
                                    .ok_or_else(|| parser.error("unknown random algorithm"))?
                                    .into();
                        } else {
                            instruction
                                .precision_config
                                .mut_or_insert_default()
                                .algorithm = Algorithm::from_str(&format!("ALG_{value}"))
                                .ok_or_else(|| parser.error("unknown precision algorithm"))?
                                .into();
                        }
                    }
                    "index" => instruction.tuple_index = parser.integer()?,
                    "to_apply" | "calls" => callees.push(parser.name()?),
                    "body" | "condition" | "true_computation" | "false_computation" | "select"
                    | "scatter" => {
                        named_callees.insert(attribute.clone(), parser.name()?);
                    }
                    "branch_computations" | "called_computations" => {
                        callees.extend(parser.names()?)
                    }
                    "control-predecessors" => controls = parser.names()?,
                    "dynamic_slice_sizes" => {
                        instruction.dynamic_slice_sizes = parser.integers("{", "}")?
                    }
                    "direction" => instruction.comparison_direction = parser.word()?,
                    "exponent_bits" => {
                        instruction.exponent_bits = i32::try_from(parser.integer()?)
                            .map_err(|_| parser.error("exponent_bits exceeds i32 range"))?
                    }
                    "mantissa_bits" => {
                        instruction.mantissa_bits = i32::try_from(parser.integer()?)
                            .map_err(|_| parser.error("mantissa_bits exceeds i32 range"))?
                    }
                    "sparsity_config" => {
                        if !matches!(opcode, Opcode::Dot | Opcode::Convolution) {
                            return Err(parser.error("sparsity_config requires dot or convolution"));
                        }
                        instruction.sparsity_config = Some(dot::parse_sparsity(parser)?).into();
                    }
                    "block_scaling_config" => {
                        if !matches!(opcode, Opcode::Dot | Opcode::Convolution) {
                            return Err(
                                parser.error("block_scaling_config requires dot or convolution")
                            );
                        }
                        instruction.block_scaling_config =
                            Some(dot::parse_block_scaling(parser)?).into();
                    }
                    "lhs_ragged_dims" | "rhs_group_dims" => {
                        if opcode != Opcode::RaggedDot {
                            return Err(
                                parser.error("ragged dimension attributes require ragged-dot")
                            );
                        }
                        let values = parser.integers("{", "}")?;
                        let dimensions = instruction
                            .ragged_dot_dimension_numbers
                            .mut_or_insert_default();
                        if attribute == "lhs_ragged_dims" {
                            dimensions.lhs_ragged_dimensions = values;
                        } else {
                            dimensions.rhs_group_dimensions = values;
                        }
                    }
                    "order" => instruction.comparison_order = parser.word()?,
                    "type" => instruction.comparison_type = parser.word()?,
                    "padding" => instruction.padding_config = Some(parser.padding_config()?).into(),
                    "iota_dimension" => {
                        if opcode != Opcode::Iota {
                            return Err(parser.error("iota_dimension requires iota"));
                        }
                        instruction.dimensions = vec![parser.integer()?]
                    }
                    "slice" => {
                        parser.expect("{")?;
                        if !parser.take("}") {
                            loop {
                                parser.expect("[")?;
                                let start = parser.integer()?;
                                parser.expect(":")?;
                                let limit = parser.integer()?;
                                let stride = if parser.take(":") {
                                    parser.integer()?
                                } else {
                                    1
                                };
                                parser.expect("]")?;
                                instruction.slice_dimensions.push(SliceDimensions {
                                    start,
                                    limit,
                                    stride,
                                    ..Default::default()
                                });
                                if parser.take("}") {
                                    break;
                                }
                                parser.expect(",")?;
                            }
                        }
                    }
                    _ => {
                        return Err(Error::new(
                            "unsupported",
                            format!("instruction text attribute {attribute}"),
                        )
                        .at(parser.offset()));
                    }
                }
            }
            let required: &[&str] = match opcode {
                Opcode::Domain => &["domain"],
                Opcode::Fft => &["fft_type", "fft_length"],
                Opcode::TopK => &["k"],
                Opcode::Sort => &["dimensions", "to_apply"],
                Opcode::Gather => &[
                    "offset_dims",
                    "collapsed_slice_dims",
                    "start_index_map",
                    "index_vector_dim",
                    "slice_sizes",
                ],
                Opcode::Scatter => &[
                    "update_window_dims",
                    "inserted_window_dims",
                    "scatter_dims_to_operand_dims",
                    "index_vector_dim",
                    "to_apply",
                ],
                Opcode::BatchNormTraining | Opcode::BatchNormInference | Opcode::BatchNormGrad => {
                    &["epsilon", "feature_index"]
                }
                Opcode::Convolution => &["dim_labels"],
                Opcode::Rng => &["distribution"],
                Opcode::RngBitGenerator => &["algorithm"],
                Opcode::RngGetAndUpdateState => &["delta"],
                Opcode::Shuffle => &["dimensions", "mode", "shifts"],
                Opcode::SelectAndScatter => &["select", "scatter"],
                Opcode::Scan => &["dimensions", "to_apply", "num_carries"],
                Opcode::Fusion => &["kind", "calls"],
                Opcode::AsyncStart => &["calls"],
                Opcode::CustomCall => &["custom_call_target"],
                Opcode::AllGather | Opcode::AllGatherStart => &["dimensions"],
                Opcode::AllReduce | Opcode::AllReduceStart | Opcode::CollectiveReduce => {
                    &["to_apply"]
                }
                Opcode::ReduceScatter => &["dimensions", "to_apply"],
                Opcode::CollectivePermute | Opcode::CollectivePermuteStart => {
                    &["source_target_pairs"]
                }
                _ => &[],
            };
            if opcode == Opcode::Broadcast
                && wrapped.is_none()
                && !seen_attributes.contains("dimensions")
            {
                broadcasts_without_dimensions.insert(instruction.id);
            }
            for attribute in required {
                if !seen_attributes.contains(*attribute) {
                    return Err(parser.error(format!(
                        "missing required {attribute} attribute for {opcode}"
                    )));
                }
            }
            if collective::has_groups(opcode) && instruction.replica_group_list.is_none() {
                instruction.set_collective_device_list(
                    crate::proto::xla_data::CollectiveDeviceListProto::new(),
                );
            }
            if matches!(
                opcode,
                Opcode::CollectivePermute | Opcode::CollectivePermuteStart
            ) && seen_attributes.contains("slice_sizes")
                && operands.len() != 4
            {
                return Err(parser.error("in-place collective-permute requires four operands"));
            }
            if matches!(opcode, Opcode::GetDimensionSize | Opcode::SetDimensionSize)
                && instruction.dimensions.len() != 1
            {
                return Err(
                    parser.error("dimension-size instruction requires exactly one dimension")
                );
            }
            if matches!(
                opcode,
                Opcode::AllGather | Opcode::AllGatherStart | Opcode::ReduceScatter
            ) && instruction.dimensions.len() != 1
                || opcode == Opcode::AllToAll && instruction.dimensions.len() > 1
            {
                return Err(parser.error("collective requires exactly one supplied dimension"));
            }
            if matches!(
                opcode,
                Opcode::AllReduce | Opcode::AllReduceStart | Opcode::ReduceScatter
            ) && !seen_attributes.contains("mode")
                && instruction.use_global_device_ids
                && !seen_attributes.contains("channel_id")
            {
                return Err(parser.error("global device IDs require a channel"));
            }
            if opcode == Opcode::Infeed {
                let shape = inferred::explicit(&shape)?;
                let data = shape
                    .as_proto()
                    .tuple_shapes
                    .first()
                    .filter(|_| shape.is_tuple())
                    .ok_or_else(|| parser.error("infeed requires a nonempty tuple result"))?;
                instruction.shape = Some(
                    Shape::tuple(vec![Shape::from_proto(data.clone())?, Shape::token()])?
                        .into_proto(),
                )
                .into();
            }
            if opcode == Opcode::Outfeed {
                instruction.shape = Some(Shape::token().into_proto()).into();
            }
            if opcode == Opcode::CustomCall {
                if seen_attributes.contains("to_apply")
                    && seen_attributes.contains("called_computations")
                {
                    return Err(parser.error(
                        "custom-call cannot specify both to_apply and called_computations",
                    ));
                }
                if instruction.constrain_layout {
                    callees.clear();
                }
            }
            if matches!(
                opcode,
                Opcode::Dot
                    | Opcode::Convolution
                    | Opcode::CustomCall
                    | Opcode::RaggedDot
                    | Opcode::ScaledDot
            ) {
                if matches!(opcode, Opcode::Dot | Opcode::ScaledDot)
                    && instruction.dot_dimension_numbers.is_none()
                {
                    instruction.dot_dimension_numbers = Some(DotDimensionNumbers::new()).into();
                }
                if opcode == Opcode::RaggedDot {
                    instruction
                        .ragged_dot_dimension_numbers
                        .mut_or_insert_default()
                        .dot_dimension_numbers
                        .mut_or_insert_default();
                }
                let precision = instruction.precision_config.mut_or_insert_default();
                if !seen_attributes.contains("operand_precision") {
                    let count = if matches!(opcode, Opcode::RaggedDot | Opcode::ScaledDot) {
                        2
                    } else if wrapped.is_some() && outer_opcode == Opcode::AsyncStart {
                        asynchronous::parameter_count(opcode, inferred::explicit(&outer_shape)?)?
                    } else {
                        operands.len()
                    };
                    precision.operand_precision = vec![Precision::DEFAULT.into(); count];
                }
            }
            if matches!(opcode, Opcode::RaggedDot | Opcode::ScaledDot)
                && operands.len() != if opcode == Opcode::RaggedDot { 3 } else { 4 }
            {
                return Err(parser.error("wrong operand count for ragged/scaled dot"));
            }
            if !named_callees.is_empty() {
                let required = match opcode {
                    Opcode::While => ["body", "condition"],
                    Opcode::Conditional => ["true_computation", "false_computation"],
                    Opcode::SelectAndScatter => ["select", "scatter"],
                    _ => {
                        return Err(
                            parser.error("named branch/body attributes on non-control-flow opcode")
                        );
                    }
                };
                if !callees.is_empty() || named_callees.len() != 2 {
                    return Err(
                        parser.error("incomplete or conflicting called-computation attributes")
                    );
                }
                for key in required {
                    callees.push(
                        named_callees
                            .remove(key)
                            .ok_or_else(|| parser.error(format!("missing {key} computation")))?,
                    );
                }
            }
            if wrapped.is_some() && outer_opcode == Opcode::AsyncStart {
                let wrapper = *next_computation;
                *next_computation += 1;
                let (outer, body) = asynchronous::wrap(
                    instruction,
                    inferred::explicit(&outer_shape)?.clone(),
                    wrapper,
                    next_instruction,
                    channel_presence,
                )?;
                pending_callees.push((body.root_id, callees));
                callees = Vec::new();
                generated_computations.insert(wrapper);
                module.computations.push(body);
                instruction = outer;
            }
            if depth > 0 {
                instruction.name = parsed_names.unique(&format!("{}.anon", instruction.opcode))?;
                if names
                    .insert(instruction.name.clone(), instruction.id)
                    .is_some()
                {
                    return Err(parser.error("duplicate nested instruction name"));
                }
            } else {
                parsed_names.unique(&instruction.name)?;
            }
            pending_operands.push((instruction.id, operands));
            if matches!(
                outer_opcode,
                Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone
            ) && seen_attributes.contains("backend_config")
            {
                async_backend_attributes.insert(instruction.id);
            }
            pending_controls.push((instruction.id, controls));
            pending_callees.push((instruction.id, callees));
            let id = instruction.id;
            computation.instructions.push(instruction);
            Ok(id)
        }
        while !parser.take("}") {
            parse_instruction(
                &mut InstructionContext {
                    parser: &mut parser,
                    module: &mut module,
                    computation: &mut computation,
                    names: &mut names,
                    parsed_names: &mut parsed_names,
                    next_instruction: &mut next_instruction,
                    next_computation: &mut next_computation,
                    sugar: &mut sugar,
                    generated_computations: &mut generated_computations,
                    pending_callees: &mut pending_callees,
                    pending_operands: &mut pending_operands,
                    pending_controls: &mut pending_controls,
                    async_backend_attributes: &mut async_backend_attributes,
                    channel_presence: &mut channel_presence,
                    broadcasts_without_dimensions: &mut broadcasts_without_dimensions,
                },
                0,
            )?;
        }
        while parser.take(",") {
            let attribute = parser.word()?;
            parser.expect("=")?;
            match attribute.as_str() {
                "execution_thread" => {
                    computation.execution_thread = String::from_utf8(parser.byte_string()?)
                        .map_err(|_| {
                            parser.error("computation execution thread must be valid UTF-8")
                        })?;
                    if computation.execution_thread == "main" {
                        computation.execution_thread.clear();
                    }
                }
                _ => {
                    return Err(
                        parser.error(format!("unsupported computation attribute {attribute}"))
                    );
                }
            }
        }
        if computation.root_id < 0 {
            computation.root_id = computation
                .instructions
                .last()
                .ok_or_else(|| parser.error("empty computation"))?
                .id;
        }
        let mut written_operands = Vec::new();
        for (id, operands) in pending_operands {
            let mut ids = Vec::new();
            for (name, indices, written) in operands {
                let operand_id = operands::resolve(
                    &mut computation,
                    &mut names,
                    &name,
                    &indices,
                    id,
                    &mut next_instruction,
                )?;
                if let Some(written) = written {
                    written_operands.push((operand_id, written));
                }
                ids.push(operand_id);
            }
            let index = computation
                .instructions
                .iter()
                .position(|i| i.id == id)
                .ok_or_else(|| parser.error("missing instruction"))?;
            computation.instructions[index].operand_ids = ids;
        }
        for (id, controls) in pending_controls {
            let predecessors = controls
                .iter()
                .map(|name| {
                    names.get(name).copied().ok_or_else(|| {
                        parser.error(format!("undefined control predecessor {name}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            computation
                .instructions
                .iter_mut()
                .find(|i| i.id == id)
                .ok_or_else(|| parser.error("missing instruction"))?
                .control_predecessor_ids = predecessors;
        }
        contexts::normalize_with_inference(
            &mut computation,
            &channel_presence,
            |instruction, instructions| {
                if broadcasts_without_dimensions.contains(&instruction.id) {
                    let input = instruction
                        .operand_ids
                        .first()
                        .and_then(|id| instructions.iter().find(|i| i.id == *id))
                        .and_then(|instruction| instruction.shape.as_ref())
                        .ok_or_else(|| parser.error("broadcast requires a shaped operand"))?;
                    if Shape::from_proto(input.clone())?.array_view()?.rank()? != 0 {
                        return Err(
                            parser.error("nonscalar broadcast requires dimensions attribute")
                        );
                    }
                }
                if instruction.shape.is_some() {
                    return Ok(None);
                }
                let callees = pending_callees.iter().find(|(id, _)| *id == instruction.id)
                .into_iter().flat_map(|(_, names)| names).map(|name| {
                    let callee = module.computations.iter().find(|callee| callee.name == *name)
                        .ok_or_else(|| parser.error(format!("inferred result requires a previously defined computation {name}")))?;
                    crate::ir::computation_program_shape(callee)
                }).collect::<Result<Vec<_>>>()?;
                Ok(Some(inferred::infer(instruction, instructions, &callees)?))
            },
        )?;
        for (id, written) in written_operands {
            let actual = computation
                .instructions
                .iter()
                .find(|i| i.id == id)
                .and_then(|i| i.shape.as_ref())
                .ok_or_else(|| parser.error("missing operand shape"))?;
            if !written.compatible(&Shape::from_proto(actual.clone())?) {
                return Err(parser.error("written operand shape mismatch"));
            }
        }
        if let Some(declared) = declared_result {
            let root = computation
                .instructions
                .iter()
                .find(|instruction| instruction.id == computation.root_id)
                .and_then(|instruction| instruction.shape.as_ref())
                .ok_or_else(|| parser.error("missing computation root shape"))?;
            if !declared.compatible(&Shape::from_proto(root.clone())?) {
                return Err(parser.error("computation result annotation is incompatible with root"));
            }
        }
        for instruction in &computation.instructions {
            if instruction.opcode == "custom-call" {
                callable::validate_constraints(instruction, &computation.instructions)?;
            }
        }
        module.computations.push(computation);
    }
    if !entry_seen {
        let computation = module
            .computations
            .last()
            .ok_or_else(|| parser.error("no computations"))?;
        module.entry_computation_id = computation.id;
        module.entry_computation_name = computation.name.clone();
    }
    if headerless {
        module.name = format!("module_{}", module.entry_computation_name);
    }
    let mut computation_uniquer = crate::ir::names::NameUniquer::default();
    for computation in &module.computations {
        if !generated_computations.contains(&computation.id) {
            computation_uniquer.unique(&computation.name)?;
        }
    }
    for computation in &mut module.computations {
        if generated_computations.contains(&computation.id) {
            computation.name = computation_uniquer.unique("async_wrapped")?;
        }
    }
    let computation_names: BTreeMap<_, _> = module
        .computations
        .iter()
        .map(|c| (c.name.clone(), c.id))
        .collect();
    let callee_names: BTreeMap<_, _> = pending_callees.into_iter().collect();
    for computation in &mut module.computations {
        for instruction in &mut computation.instructions {
            for name in callee_names.get(&instruction.id).into_iter().flatten() {
                instruction.called_computation_ids.push(
                    *computation_names
                        .get(name)
                        .ok_or_else(|| parser.error(format!("undefined computation {name}")))?,
                );
            }
        }
    }
    parser.finish()?;
    let mut names = crate::ir::names::NameUniquer::default();
    for instruction in module
        .computations
        .iter_mut()
        .flat_map(|c| &mut c.instructions)
    {
        instruction.name = names.unique(&instruction.name)?;
    }
    asynchronous::bind_called_signatures(
        &mut module,
        &generated_computations,
        &mut next_instruction,
        &channel_presence,
    )?;
    let mut module = Module::from_proto(module)?;
    module.channel_presence = channel_presence;
    let module = asynchronous::bind_sugar(module, &sugar)?;
    let mut module = asynchronous::normalize(module, &async_backend_attributes)?;
    stack_frames::warn(stack_frames::normalize(&mut module.wire)?);
    if is_scheduled {
        let mut schedule = crate::proto::hlo::HloScheduleProto::new();
        for computation in &module.as_proto().computations {
            if !computation.is_fusion_computation {
                schedule.sequences.insert(
                    computation.id,
                    crate::proto::hlo::hlo_schedule_proto::InstructionSequence {
                        instruction_ids: computation.instructions.iter().map(|i| i.id).collect(),
                        ..Default::default()
                    },
                );
            }
        }
        module.set_schedule(Some(schedule))?;
    }
    module.set_config(config);
    Ok(ParsedModuleShell {
        module,
        pending_recovery_modules,
    })
}

fn looks_like_shape(parser: &Parser) -> bool {
    parser.peek() == "("
        || (parser.peek() == "b"
            && parser
                .tokens
                .get(parser.index + 1)
                .is_some_and(|token| token.value == "("))
        || (PrimitiveType::from_hlo_name(parser.peek()).is_ok()
            && parser
                .tokens
                .get(parser.index + 1)
                .is_some_and(|token| token.value == "["))
}

fn fill_shortform_bytes(bytes: &mut [u8], counters: &mut (u32, u8), predicate: bool) {
    let mask = if predicate { 0x01010101 } else { u32::MAX };
    for word in bytes.chunks_exact_mut(4) {
        word.copy_from_slice(&(counters.0 & mask).to_le_bytes());
        counters.0 = counters.0.wrapping_add(1);
    }
    let words = bytes.len() / 4;
    // The pinned C++ parser indexes the tail by the word count, not byte count.
    for index in 0..bytes.len() % 4 {
        bytes[words + index] = (u32::from(counters.1) & mask) as u8;
        counters.1 = counters.1.wrapping_add(1);
    }
}

fn reject_fields<M: MessageFull>(message: &M, allowed: &[&str]) -> Result<()> {
    if message
        .special_fields()
        .unknown_fields()
        .iter()
        .next()
        .is_some()
    {
        return Err(Error::new(
            "unsupported",
            "unknown wire fields cannot be represented in HLO text",
        ));
    }
    for field in M::descriptor().fields() {
        if !allowed.contains(&field.name()) && field.has_field(message) {
            return Err(Error::new(
                "unsupported",
                format!("HLO text for field {}", field.name()),
            ));
        }
    }
    Ok(())
}

/// Full-fidelity for its supported fields. It refuses to omit metadata or
/// payloads it cannot represent; use protobuf for complete wire coverage.
pub fn print_module(module: &Module) -> Result<String> {
    print_module_nested(module, 0)
}

fn print_module_nested(module: &Module, module_depth: usize) -> Result<String> {
    struct Frame {
        module: Module,
        depth: usize,
        children: Vec<Option<HloModuleProto>>,
        child_texts: Vec<Option<String>>,
        next_child: usize,
        parent_index: Option<usize>,
    }
    fn recovery_children(module: &Module) -> Vec<Option<HloModuleProto>> {
        module
            .as_proto()
            .original_value_recovery_table
            .as_ref()
            .map(|table| {
                table
                    .entries
                    .iter()
                    .map(|entry| entry.recovery_module.as_ref().cloned())
                    .collect()
            })
            .unwrap_or_default()
    }
    let owned_root = module.clone_for_printing();
    let mut frames = vec![Frame {
        children: recovery_children(module),
        child_texts: vec![
            None;
            module
                .as_proto()
                .original_value_recovery_table
                .as_ref()
                .map_or(0, |table| table.entries.len())
        ],
        module: owned_root,
        depth: module_depth,
        next_child: 0,
        parent_index: None,
    }];
    loop {
        let child = frames.last_mut().and_then(|frame| {
            while frame.next_child < frame.children.len() {
                let index = frame.next_child;
                frame.next_child += 1;
                if let Some(wire) = frame.children[index].take() {
                    return Some((index, wire, frame.depth + 1));
                }
            }
            None
        });
        if let Some((index, mut child_wire, depth)) = child {
            if depth >= MAX_DEPTH {
                return Err(Error::new(
                    "resource-limit",
                    "recovery module printing nesting limit",
                ));
            }
            if let Some(program) = child_wire.host_program_shape.as_mut() {
                if program.parameter_names
                    != (0..program.parameters.len())
                        .map(|index| format!("p{index}"))
                        .collect::<Vec<_>>()
                {
                    return Err(Error::new(
                        "unsupported",
                        "recovery host parameter names cannot be reconstructed",
                    ));
                }
                program.parameter_names.clear();
            }
            let child = Module::from_proto(child_wire)?;
            let child_count = child
                .as_proto()
                .original_value_recovery_table
                .as_ref()
                .map_or(0, |table| table.entries.len());
            frames.push(Frame {
                children: recovery_children(&child),
                child_texts: vec![None; child_count],
                module: child,
                depth,
                next_child: 0,
                parent_index: Some(index),
            });
            continue;
        }
        let frame = frames
            .pop()
            .ok_or_else(|| Error::new("print", "recovery module printer stack underflow"))?;
        let text = print_module_body(&frame.module, frame.depth, &frame.child_texts)?;
        if let Some(parent) = frames.last_mut() {
            let index = frame
                .parent_index
                .ok_or_else(|| Error::new("print", "missing recovery parent index"))?;
            let slot = parent
                .child_texts
                .get_mut(index)
                .ok_or_else(|| Error::new("print", "recovery child index out of range"))?;
            *slot = Some(text);
        } else {
            return Ok(text);
        }
    }
}

fn print_module_body(
    module: &Module,
    module_depth: usize,
    recovery_modules: &[Option<String>],
) -> Result<String> {
    if module_depth >= MAX_DEPTH {
        return Err(Error::new(
            "resource-limit",
            "recovery module printing nesting limit",
        ));
    }
    let mut wire = module.to_proto()?;
    if let Some(schedule) = wire.schedule.as_ref() {
        reject_fields(schedule, &["sequences"])?;
        let source: BTreeMap<_, _> = module
            .as_proto()
            .computations
            .iter()
            .map(|c| (c.id, c))
            .collect();
        if schedule.sequences.len()
            != wire
                .computations
                .iter()
                .filter(|c| !c.is_fusion_computation)
                .count()
        {
            return Err(Error::new(
                "unsupported",
                "module text requires schedules for exactly the nonfusion computations",
            ));
        }
        for computation in &mut wire.computations {
            if computation.is_fusion_computation {
                continue;
            }
            let sequence = schedule.sequences.get(&computation.id).ok_or_else(|| {
                Error::new(
                    "unsupported",
                    "module text cannot represent a partial schedule",
                )
            })?;
            reject_fields(sequence, &["instruction_ids"])?;
            if source[&computation.id]
                .instructions
                .iter()
                .map(|i| &i.id)
                .ne(sequence.instruction_ids.iter())
            {
                return Err(Error::new(
                    "unsupported",
                    "module text cannot preserve distinct backing and schedule instruction orders",
                ));
            }
            let positions: BTreeMap<_, _> = sequence
                .instruction_ids
                .iter()
                .enumerate()
                .map(|(position, &id)| (id, position))
                .collect();
            computation.instructions.sort_by_key(|i| positions[&i.id]);
        }
    }
    let empty_async_backends = asynchronous::print_policy(module, &wire)?;
    reject_fields(
        &wire,
        &[
            "name",
            "entry_computation_name",
            "entry_computation_id",
            "computations",
            "id",
            "host_program_shape",
            "input_output_alias",
            "buffer_donor",
            "schedule",
            "stack_frame_index",
            "debug_attributes",
            "original_value_recovery_table",
            "frontend_attributes",
            "backend_config",
        ],
    )?;
    if wire.id != 0 {
        return Err(Error::new(
            "unsupported",
            "module wire ID is not preserved by this HLO printer",
        ));
    }
    let mut result = format!(
        "HloModule {}{}",
        if reserved_name(&wire.name) { "%" } else { "" },
        wire.name
    );
    if wire.schedule.is_some() {
        result.push_str(", is_scheduled=true");
    }
    let config = module.config();
    if config.alias_passthrough_params {
        result.push_str(", alias_passthrough_params=true");
    }
    for (name, values) in [
        (
            "allow_spmd_sharding_propagation_to_parameters",
            &config.allow_spmd_sharding_propagation_to_parameters,
        ),
        (
            "allow_spmd_sharding_propagation_to_output",
            &config.allow_spmd_sharding_propagation_to_output,
        ),
    ] {
        if values.is_empty() {
            return Err(Error::new(
                "unsupported",
                "empty SPMD propagation configuration cannot round-trip through HLO headers",
            ));
        }
        if values != &[false] {
            write!(
                result,
                ", {name}={{{}}}",
                values
                    .iter()
                    .map(bool::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .map_err(|e| Error::new("print", e.to_string()))?;
        }
    }
    if config.use_spmd_partitioning != (config.num_partitions != 1) {
        return Err(Error::new(
            "unsupported",
            "SPMD mode cannot round-trip through HLO count attributes",
        ));
    }
    if config.replica_count != 1 {
        write!(result, ", replica_count={}", config.replica_count)
            .map_err(|e| Error::new("print", e.to_string()))?;
    }
    if config.num_partitions != 1 {
        write!(result, ", num_partitions={}", config.num_partitions)
            .map_err(|e| Error::new("print", e.to_string()))?;
    }
    if !valid_name(&wire.name) {
        return Err(Error::new("name", "unprintable module name"));
    }
    if let Some(program) = wire.host_program_shape.as_ref() {
        reject_fields(program, &["parameters", "result", "parameter_names"])?;
        let default_parameter_names = (0..program.parameters.len())
            .map(|index| format!("p{index}"))
            .collect::<Vec<_>>();
        if !program.parameter_names.is_empty() && program.parameter_names != default_parameter_names
        {
            return Err(Error::new("unsupported", "host program parameter names"));
        }
        let parameters = program
            .parameters
            .iter()
            .map(|s| print_shape(&Shape::from_proto(s.clone())?))
            .collect::<Result<Vec<_>>>()?;
        let output = Shape::from_proto(
            program
                .result
                .as_ref()
                .ok_or_else(|| Error::new("shape", "missing program result"))?
                .clone(),
        )?;
        write!(
            result,
            ", entry_computation_layout={{({})->{}}}",
            parameters.join(", "),
            print_shape(&output)?
        )
        .map_err(|e| Error::new("print", e.to_string()))?;
    }
    module_attributes::print(&wire, &mut result)?;
    module_attributes::print_recovery(&wire, &mut result, module_depth, recovery_modules)?;
    if let Some(attributes) = wire.frontend_attributes.as_ref() {
        let printed = callable::print_frontend(attributes)?;
        if !attributes.map.is_empty() {
            write!(result, ", frontend_attributes={printed}")
                .map_err(|e| Error::new("print", e.to_string()))?;
        }
    }
    if !wire.backend_config.is_empty() {
        write!(
            result,
            ", backend_config={}",
            strings::quote(&wire.backend_config)?
        )
        .map_err(|e| Error::new("print", e.to_string()))?;
    }
    result.push_str("\n\n");
    stack_frames::print(&wire, &mut result)?;
    let computation_names: BTreeMap<_, _> =
        wire.computations.iter().map(|c| (c.id, &c.name)).collect();
    for computation in &wire.computations {
        reject_fields(
            computation,
            &[
                "name",
                "id",
                "root_id",
                "instructions",
                "program_shape",
                "is_fusion_computation",
                "execution_thread",
            ],
        )?;
        if !valid_name(&computation.name) {
            return Err(Error::new("name", "unprintable computation name"));
        }
        if computation.id == wire.entry_computation_id {
            result.push_str("ENTRY ");
        }
        write!(result, "%{}", computation.name).map_err(|e| Error::new("print", e.to_string()))?;
        if let Some(program) = computation.program_shape.as_ref() {
            reject_fields(program, &["parameters", "result", "parameter_names"])?;
            let actual = module.program_shape(module.computation_handle(computation.id))?;
            if program.parameters != actual.parameters
                || program.result != actual.result
                || program.parameter_names != actual.parameter_names
            {
                return Err(Error::new(
                    "unsupported",
                    "computation signature differs from its instruction-derived shape",
                ));
            }
            if program.parameter_names.len() != program.parameters.len() {
                return Err(Error::new(
                    "unsupported",
                    "signature parameter names missing",
                ));
            }
            let mut parameters = Vec::new();
            for (name, shape) in program.parameter_names.iter().zip(&program.parameters) {
                if !valid_name(name) {
                    return Err(Error::new("name", "unprintable parameter name"));
                }
                parameters.push(format!(
                    "{name}: {}",
                    print_shape(&Shape::from_proto(shape.clone())?)?
                ));
            }
            write!(
                result,
                " ({}) -> {}",
                parameters.join(", "),
                print_shape(&Shape::from_proto(
                    program
                        .result
                        .as_ref()
                        .ok_or_else(|| Error::new("signature", "missing result"))?
                        .clone()
                )?)?
            )
            .map_err(|e| Error::new("print", e.to_string()))?;
        }
        result.push_str(" {\n");
        let names: BTreeMap<_, _> = computation
            .instructions
            .iter()
            .map(|i| (i.id, &i.name))
            .collect();
        for instruction in &computation.instructions {
            let opcode: Opcode = instruction.opcode.parse()?;
            let mut allowed = vec![
                "name",
                "id",
                "opcode",
                "shape",
                "operand_ids",
                "control_predecessor_ids",
                "backend_config",
                "frontend_attributes",
                "sharding",
                "metadata",
                "statistics_viz",
                "original_value",
            ];
            if opcode.supports_result_accuracy() {
                allowed.push("result_accuracy");
            }
            match opcode {
                Opcode::Domain => allowed.extend(["domain_entry_sharding", "domain_exit_sharding"]),
                Opcode::Parameter => allowed.extend(["parameter_number", "parameter_replication"]),
                Opcode::Constant => allowed.push("literal"),
                Opcode::CopyStart => allowed.push("cross_program_prefetch_index"),
                Opcode::AsyncStart | Opcode::AsyncUpdate => allowed.extend([
                    "called_computation_ids",
                    "async_execution_thread",
                    "output_operand_aliasing",
                ]),
                Opcode::AsyncDone => {
                    allowed.extend(["called_computation_ids", "async_execution_thread"])
                }
                Opcode::Send | Opcode::SendDone | Opcode::Recv | Opcode::RecvDone => {
                    allowed.extend(["channel_id", "is_host_transfer"])
                }
                Opcode::GetTupleElement => allowed.push("tuple_index"),
                Opcode::Slice => allowed.push("slice_dimensions"),
                Opcode::Broadcast
                | Opcode::Transpose
                | Opcode::Reverse
                | Opcode::Iota
                | Opcode::GetDimensionSize
                | Opcode::SetDimensionSize
                | Opcode::Concatenate => allowed.push("dimensions"),
                Opcode::Call => allowed.extend([
                    "called_computation_ids",
                    "is_composite",
                    "output_operand_aliasing",
                ]),
                Opcode::Fusion => allowed.extend([
                    "called_computation_ids",
                    "fusion_kind",
                    "output_operand_aliasing",
                ]),
                Opcode::CustomCall => allowed.extend([
                    "literal",
                    "window",
                    "convolution_dimension_numbers",
                    "called_computation_ids",
                    "custom_call_target",
                    "custom_call_has_side_effect",
                    "custom_call_api_version",
                    "custom_call_schedule",
                    "constrain_layout",
                    "operand_shapes_with_layout",
                    "output_operand_aliasing",
                    "feature_group_count",
                    "batch_group_count",
                    "precision_config",
                ]),
                Opcode::While | Opcode::Conditional => allowed.push("called_computation_ids"),
                Opcode::Reduce | Opcode::Map => {
                    allowed.extend(["dimensions", "called_computation_ids"])
                }
                Opcode::ReduceWindow => allowed.extend(["window", "called_computation_ids"]),
                Opcode::Sort => {
                    allowed.extend(["dimensions", "called_computation_ids", "is_stable"])
                }
                Opcode::Fft => allowed.extend(["fft_type", "fft_length"]),
                Opcode::Scan => allowed.extend([
                    "dimensions",
                    "called_computation_ids",
                    "num_carries",
                    "is_reverse",
                    "is_associative",
                ]),
                Opcode::Shuffle => allowed.extend(["dimensions", "shuffle_mode"]),
                Opcode::SelectAndScatter => allowed.extend(["window", "called_computation_ids"]),
                Opcode::Dot => allowed.extend([
                    "dot_dimension_numbers",
                    "precision_config",
                    "sparsity_config",
                    "block_scaling_config",
                ]),
                Opcode::RaggedDot => {
                    allowed.extend(["ragged_dot_dimension_numbers", "precision_config"])
                }
                Opcode::ScaledDot => allowed.extend(["dot_dimension_numbers", "precision_config"]),
                Opcode::Infeed => allowed.push("infeed_config"),
                Opcode::Outfeed => allowed.extend(["outfeed_config", "outfeed_shape"]),
                Opcode::Rng => allowed.push("distribution"),
                Opcode::RngBitGenerator => allowed.push("rng_algorithm"),
                Opcode::RngGetAndUpdateState => allowed.push("delta"),
                Opcode::Convolution => allowed.extend([
                    "convolution_dimension_numbers",
                    "window",
                    "feature_group_count",
                    "batch_group_count",
                    "precision_config",
                    "conv_kind",
                    "sparsity_config",
                    "block_scaling_config",
                ]),
                Opcode::Gather => allowed.extend([
                    "gather_dimension_numbers",
                    "gather_slice_sizes",
                    "indices_are_sorted",
                ]),
                Opcode::Scatter => allowed.extend([
                    "scatter_dimension_numbers",
                    "indices_are_sorted",
                    "unique_indices",
                    "called_computation_ids",
                ]),
                Opcode::TopK => allowed.extend(["k", "largest", "is_stable"]),
                Opcode::TriangularSolve => allowed.push("triangular_solve_options"),
                Opcode::Cholesky => allowed.push("cholesky_options"),
                Opcode::BatchNormTraining | Opcode::BatchNormInference | Opcode::BatchNormGrad => {
                    allowed.extend(["epsilon", "feature_index"])
                }
                Opcode::DynamicSlice => allowed.push("dynamic_slice_sizes"),
                Opcode::Pad => allowed.push("padding_config"),
                Opcode::ReducePrecision => allowed.extend(["exponent_bits", "mantissa_bits"]),
                Opcode::Compare => allowed.extend([
                    "comparison_direction",
                    "comparison_order",
                    "comparison_type",
                ]),
                Opcode::CollectivePermute | Opcode::CollectivePermuteStart => {
                    allowed.extend(["source_target_pairs", "dynamic_slice_sizes", "channel_id"])
                }
                opcode if collective::supported(opcode) => {
                    let opcode = collective::attribute_opcode(opcode);
                    allowed.extend([
                        "replica_groups",
                        "collective_device_list",
                        "iota_collective_device_list",
                        "mesh_axes_replica_group_list",
                        "channel_id",
                    ]);
                    if matches!(
                        opcode,
                        Opcode::AllGather
                            | Opcode::AllReduce
                            | Opcode::ReduceScatter
                            | Opcode::AllToAll
                            | Opcode::CollectiveReduce
                    ) {
                        allowed.push("constrain_layout");
                    }
                    if matches!(
                        opcode,
                        Opcode::AllGather
                            | Opcode::AllReduce
                            | Opcode::ReduceScatter
                            | Opcode::CollectiveReduce
                    ) {
                        allowed.push("use_global_device_ids");
                    }
                    if matches!(
                        opcode,
                        Opcode::AllGather | Opcode::AllToAll | Opcode::ReduceScatter
                    ) {
                        allowed.push("dimensions");
                    }
                    if matches!(
                        opcode,
                        Opcode::AllReduce | Opcode::ReduceScatter | Opcode::CollectiveReduce
                    ) {
                        allowed.push("called_computation_ids");
                    }
                    if matches!(
                        opcode,
                        Opcode::CollectiveReduce | Opcode::CollectiveBroadcast
                    ) {
                        allowed.push("has_dynamic_root");
                    }
                }
                _ => {}
            }
            reject_fields(instruction, &allowed)?;
            contexts::check_print(
                instruction,
                &computation.instructions,
                &module.channel_presence,
            )?;
            if !valid_instruction_name(&instruction.name) {
                return Err(Error::new("name", "unprintable instruction name"));
            }
            let shape = Shape::from_proto(
                instruction
                    .shape
                    .as_ref()
                    .ok_or_else(|| Error::new("shape", "missing shape"))?
                    .clone(),
            )?;
            write!(
                result,
                "  {}%{} = {} {}(",
                if instruction.id == computation.root_id {
                    "ROOT "
                } else {
                    ""
                },
                instruction.name,
                print_shape(&shape)?,
                opcode
            )
            .map_err(|e| Error::new("print", e.to_string()))?;
            if opcode == Opcode::Parameter {
                write!(result, "{}", instruction.parameter_number)
                    .map_err(|e| Error::new("print", e.to_string()))?;
            } else if opcode == Opcode::Constant {
                let literal = Literal::from_proto(
                    instruction
                        .literal
                        .as_ref()
                        .ok_or_else(|| Error::new("literal", "missing literal"))?
                        .clone(),
                )?;
                literals::print(&literal, &shape, &mut result)?;
            } else {
                let operands = instruction
                    .operand_ids
                    .iter()
                    .map(|id| {
                        names
                            .get(id)
                            .map(|name| format!("%{name}"))
                            .ok_or_else(|| Error::new("reference", "missing operand name"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                result.push_str(&operands.join(", "));
            }
            result.push(')');
            if matches!(opcode, Opcode::GetDimensionSize | Opcode::SetDimensionSize)
                && instruction.dimensions.len() != 1
            {
                return Err(Error::new("dimension", "opcode requires one dimension"));
            }
            if opcode == Opcode::GetTupleElement {
                write!(result, ", index={}", instruction.tuple_index)
                    .map_err(|e| Error::new("print", e.to_string()))?;
            }
            if opcode == Opcode::Iota {
                if instruction.dimensions.len() != 1 {
                    return Err(Error::new("dimension", "opcode requires one dimension"));
                }
                write!(result, ", iota_dimension={}", instruction.dimensions[0])
                    .map_err(|e| Error::new("print", e.to_string()))?;
            } else if !instruction.dimensions.is_empty()
                || matches!(
                    opcode,
                    Opcode::Broadcast | Opcode::Transpose | Opcode::Reverse | Opcode::Reduce
                )
            {
                write!(
                    result,
                    ", dimensions={{{}}}",
                    instruction
                        .dimensions
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                )
                .map_err(|e| Error::new("print", e.to_string()))?;
            }
            if opcode == Opcode::Slice {
                write!(
                    result,
                    ", slice={{{}}}",
                    instruction
                        .slice_dimensions
                        .iter()
                        .map(|d| format!("[{}:{}:{}]", d.start, d.limit, d.stride))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
                .map_err(|e| Error::new("print", e.to_string()))?;
            }
            if opcode == Opcode::DynamicSlice {
                write!(
                    result,
                    ", dynamic_slice_sizes={{{}}}",
                    instruction
                        .dynamic_slice_sizes
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                )
                .map_err(|e| Error::new("print", e.to_string()))?;
            }
            if opcode == Opcode::Compare {
                if !matches!(
                    instruction.comparison_direction.as_str(),
                    "EQ" | "NE" | "GE" | "GT" | "LE" | "LT"
                ) || !matches!(
                    instruction.comparison_order.as_str(),
                    "" | "TOTAL" | "WEAK" | "PARTIAL"
                ) || !matches!(
                    instruction.comparison_type.as_str(),
                    "" | "FLOAT" | "TOTALORDER" | "SIGNED" | "UNSIGNED"
                ) {
                    return Err(Error::new(
                        "comparison",
                        "unprintable comparison attributes",
                    ));
                }
                write!(result, ", direction={}", instruction.comparison_direction)
                    .map_err(|e| Error::new("print", e.to_string()))?;
                if !instruction.comparison_order.is_empty() {
                    write!(result, ", order={}", instruction.comparison_order)
                        .map_err(|e| Error::new("print", e.to_string()))?;
                }
                if !instruction.comparison_type.is_empty() {
                    write!(result, ", type={}", instruction.comparison_type)
                        .map_err(|e| Error::new("print", e.to_string()))?;
                }
            }
            if opcode == Opcode::ReducePrecision {
                write!(
                    result,
                    ", exponent_bits={}, mantissa_bits={}",
                    instruction.exponent_bits, instruction.mantissa_bits
                )
                .map_err(|e| Error::new("print", e.to_string()))?;
            }
            if opcode == Opcode::Pad {
                let config = instruction
                    .padding_config
                    .as_ref()
                    .ok_or_else(|| Error::new("padding", "missing padding configuration"))?;
                write!(result, ", padding={}", print_padding_config(config)?)
                    .map_err(|e| Error::new("print", e.to_string()))?;
            }
            let callees = instruction
                .called_computation_ids
                .iter()
                .map(|id| {
                    computation_names
                        .get(id)
                        .map(|name| format!("%{name}"))
                        .ok_or_else(|| Error::new("reference", "missing callee name"))
                })
                .collect::<Result<Vec<_>>>()?;
            match opcode {
                Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone => {
                    if opcode == Opcode::AsyncStart && callees.len() != 1 || callees.len() > 1 {
                        return Err(Error::new(
                            "callee",
                            "invalid async wrapped computation count",
                        ));
                    }
                    if let Some(callee) = callees.first() {
                        write!(result, ", calls={callee}")
                            .map_err(|e| Error::new("print", e.to_string()))?;
                    }
                    if !instruction.async_execution_thread.is_empty() {
                        write!(
                            result,
                            ", async_execution_thread={}",
                            strings::quote(instruction.async_execution_thread.as_bytes())?
                        )
                        .map_err(|e| Error::new("print", e.to_string()))?;
                    }
                }
                Opcode::Fusion => {
                    if callees.len() != 1 {
                        return Err(Error::new("callee", "fusion requires one callee"));
                    }
                    write!(result, ", calls={}", callees[0])
                        .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::CustomCall if !callees.is_empty() => {
                    write!(result, ", called_computations={{{}}}", callees.join(", "))
                        .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::Call
                | Opcode::Map
                | Opcode::Reduce
                | Opcode::ReduceWindow
                | Opcode::Sort
                | Opcode::Scan
                | Opcode::Scatter
                | Opcode::AllReduce
                | Opcode::AllReduceStart
                | Opcode::ReduceScatter
                | Opcode::CollectiveReduce => {
                    if callees.len() != 1 {
                        return Err(Error::new("callee", "call requires one callee"));
                    }
                    write!(result, ", to_apply={}", callees[0])
                        .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::SelectAndScatter => {
                    if callees.len() != 2 {
                        return Err(Error::new(
                            "callee",
                            "select-and-scatter requires two callees",
                        ));
                    }
                    write!(result, ", select={}, scatter={}", callees[0], callees[1])
                        .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::While => {
                    if callees.len() != 2 {
                        return Err(Error::new("callee", "while requires body and condition"));
                    }
                    write!(result, ", body={}, condition={}", callees[0], callees[1])
                        .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::Conditional => {
                    let selector = instruction
                        .operand_ids
                        .first()
                        .and_then(|id| computation.instructions.iter().find(|i| i.id == *id))
                        .and_then(|i| i.shape.as_ref())
                        .ok_or_else(|| Error::new("callee", "conditional missing selector"))?;
                    if selector.element_type.enum_value_or_default() == PrimitiveType::PRED {
                        if callees.len() != 2 {
                            return Err(Error::new(
                                "callee",
                                "predicate conditional requires two branches",
                            ));
                        }
                        write!(
                            result,
                            ", true_computation={}, false_computation={}",
                            callees[0], callees[1]
                        )
                        .map_err(|e| Error::new("print", e.to_string()))?;
                    } else {
                        write!(result, ", branch_computations={{{}}}", callees.join(", "))
                            .map_err(|e| Error::new("print", e.to_string()))?;
                    }
                }
                _ => {}
            }
            if !instruction.control_predecessor_ids.is_empty() {
                let controls = instruction
                    .control_predecessor_ids
                    .iter()
                    .map(|id| {
                        names.get(id).map(|name| format!("%{name}")).ok_or_else(|| {
                            Error::new("reference", "missing control predecessor name")
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                write!(result, ", control-predecessors={{{}}}", controls.join(", "))
                    .map_err(|e| Error::new("print", e.to_string()))?;
            }
            if matches!(
                opcode,
                Opcode::ReduceWindow | Opcode::Convolution | Opcode::SelectAndScatter
            ) || (opcode == Opcode::CustomCall && instruction.window.is_some())
            {
                let window = instruction
                    .window
                    .as_ref()
                    .ok_or_else(|| Error::new("window", "missing window configuration"))?;
                write!(result, ", window={{{}}}", print_window(window)?)
                    .map_err(|e| Error::new("print", e.to_string()))?;
            }
            if matches!(opcode, Opcode::Sort | Opcode::TopK) {
                write!(result, ", is_stable={}", instruction.is_stable)
                    .map_err(|e| Error::new("print", e.to_string()))?;
            }
            match opcode {
                Opcode::Scan => {
                    let associative = match instruction.is_associative.enum_value() {
                        Ok(crate::proto::hlo::TriState::TRI_STATE_TRUE) => true,
                        Ok(crate::proto::hlo::TriState::TRI_STATE_FALSE) => false,
                        _ => {
                            return Err(Error::new(
                                "unsupported",
                                "scan with unspecified/unknown associativity cannot be preserved in text",
                            ));
                        }
                    };
                    write!(
                        result,
                        ", num_carries={}, is_reverse={}, is_associative={}",
                        instruction.num_carries, instruction.is_reverse, associative
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::Shuffle => {
                    let mode = instruction
                        .shuffle_mode
                        .as_ref()
                        .ok_or_else(|| Error::new("shuffle", "missing shuffle mode"))?;
                    reject_fields(mode, &["rotate"])?;
                    if !mode.has_rotate() {
                        return Err(Error::new("shuffle", "missing rotate mode"));
                    }
                    reject_fields(mode.rotate(), &["shifts"])?;
                    write!(
                        result,
                        ", mode=rotate, shifts={{{}}}",
                        mode.rotate()
                            .shifts
                            .iter()
                            .map(i64::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::Infeed => {
                    write!(
                        result,
                        ", infeed_config={}",
                        strings::quote(&instruction.infeed_config)?
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::Outfeed => {
                    let shape = instruction.outfeed_shape.as_ref().ok_or_else(|| {
                        Error::new(
                            "unsupported",
                            "outfeed without host shape cannot be preserved in text",
                        )
                    })?;
                    write!(
                        result,
                        ", outfeed_config={}, outfeed_shape={}",
                        strings::quote(&instruction.outfeed_config)?,
                        print_shape(&Shape::from_proto(shape.clone())?)?
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::Rng => {
                    let distribution = instruction
                        .distribution
                        .enum_value()
                        .map_err(|_| Error::new("rng", "unknown random distribution"))?;
                    write!(
                        result,
                        ", distribution={}",
                        distribution.descriptor().name().to_ascii_lowercase()
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::RngBitGenerator => {
                    let algorithm = instruction
                        .rng_algorithm
                        .enum_value()
                        .map_err(|_| Error::new("rng", "unknown random algorithm"))?;
                    write!(
                        result,
                        ", algorithm={}",
                        algorithm.descriptor().name().to_ascii_lowercase()
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::RngGetAndUpdateState => write!(result, ", delta={}", instruction.delta)
                    .map_err(|e| Error::new("print", e.to_string()))?,
                Opcode::Gather => {
                    let dims = instruction
                        .gather_dimension_numbers
                        .as_ref()
                        .ok_or_else(|| Error::new("dimensions", "missing gather dimensions"))?;
                    reject_fields(
                        dims,
                        &[
                            "offset_dims",
                            "collapsed_slice_dims",
                            "start_index_map",
                            "operand_batching_dims",
                            "start_indices_batching_dims",
                            "index_vector_dim",
                        ],
                    )?;
                    for (name, values) in [
                        ("offset_dims", &dims.offset_dims),
                        ("collapsed_slice_dims", &dims.collapsed_slice_dims),
                        ("start_index_map", &dims.start_index_map),
                        ("operand_batching_dims", &dims.operand_batching_dims),
                        (
                            "start_indices_batching_dims",
                            &dims.start_indices_batching_dims,
                        ),
                        ("slice_sizes", &instruction.gather_slice_sizes),
                    ] {
                        write!(
                            result,
                            ", {name}={{{}}}",
                            values
                                .iter()
                                .map(i64::to_string)
                                .collect::<Vec<_>>()
                                .join(",")
                        )
                        .map_err(|e| Error::new("print", e.to_string()))?;
                    }
                    write!(
                        result,
                        ", index_vector_dim={}, indices_are_sorted={}",
                        dims.index_vector_dim, instruction.indices_are_sorted
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::Scatter => {
                    let dims = instruction
                        .scatter_dimension_numbers
                        .as_ref()
                        .ok_or_else(|| Error::new("dimensions", "missing scatter dimensions"))?;
                    reject_fields(
                        dims,
                        &[
                            "update_window_dims",
                            "inserted_window_dims",
                            "scatter_dims_to_operand_dims",
                            "input_batching_dims",
                            "scatter_indices_batching_dims",
                            "index_vector_dim",
                        ],
                    )?;
                    for (name, values) in [
                        ("update_window_dims", &dims.update_window_dims),
                        ("inserted_window_dims", &dims.inserted_window_dims),
                        (
                            "scatter_dims_to_operand_dims",
                            &dims.scatter_dims_to_operand_dims,
                        ),
                        ("input_batching_dims", &dims.input_batching_dims),
                        (
                            "scatter_indices_batching_dims",
                            &dims.scatter_indices_batching_dims,
                        ),
                    ] {
                        write!(
                            result,
                            ", {name}={{{}}}",
                            values
                                .iter()
                                .map(i64::to_string)
                                .collect::<Vec<_>>()
                                .join(",")
                        )
                        .map_err(|e| Error::new("print", e.to_string()))?;
                    }
                    write!(
                        result,
                        ", index_vector_dim={}, indices_are_sorted={}, unique_indices={}",
                        dims.index_vector_dim,
                        instruction.indices_are_sorted,
                        instruction.unique_indices
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::TopK => write!(
                    result,
                    ", k={}, largest={}",
                    instruction.k, instruction.largest
                )
                .map_err(|e| Error::new("print", e.to_string()))?,
                Opcode::TriangularSolve => {
                    let options =
                        instruction
                            .triangular_solve_options
                            .as_ref()
                            .ok_or_else(|| {
                                Error::new(
                                    "unsupported",
                                    "triangular solve without options cannot be preserved in text",
                                )
                            })?;
                    reject_fields(
                        options,
                        &["left_side", "lower", "unit_diagonal", "transpose_a"],
                    )?;
                    let transpose = options.transpose_a.enum_value().map_err(|_| {
                        Error::new("transpose", "unknown triangular solve transpose")
                    })?;
                    write!(
                        result,
                        ", left_side={}, lower={}, unit_diagonal={}, transpose_a={}",
                        options.left_side,
                        options.lower,
                        options.unit_diagonal,
                        transpose.descriptor().name()
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::Convolution => {
                    let dimensions = instruction
                        .convolution_dimension_numbers
                        .as_ref()
                        .ok_or_else(|| {
                            Error::new("dimensions", "missing convolution dimensions")
                        })?;
                    let kind = instruction
                        .conv_kind
                        .enum_value()
                        .map_err(|_| Error::new("convolution", "unknown convolution kind"))?;
                    write!(result,", dim_labels={}, feature_group_count={}, batch_group_count={}, convolution_kind={}",
                        print_convolution_dimensions(dimensions)?,instruction.feature_group_count,instruction.batch_group_count,
                        kind.descriptor().name().strip_prefix("CONVOLUTION_KIND_")
                            .ok_or_else(||Error::new("convolution","invalid convolution kind"))?.to_ascii_lowercase())
                        .map_err(|e|Error::new("print",e.to_string()))?;
                }
                Opcode::Cholesky => {
                    let options = instruction.cholesky_options.as_ref().ok_or_else(|| {
                        Error::new(
                            "unsupported",
                            "Cholesky without options cannot be preserved in text",
                        )
                    })?;
                    reject_fields(options, &["lower"])?;
                    write!(result, ", lower={}", options.lower)
                        .map_err(|e| Error::new("print", e.to_string()))?;
                }
                Opcode::BatchNormTraining | Opcode::BatchNormInference | Opcode::BatchNormGrad => {
                    let epsilon = f64::from(instruction.epsilon);
                    if epsilon.is_infinite()
                        || (epsilon as f32).to_bits() != instruction.epsilon.to_bits()
                    {
                        return Err(Error::new(
                            "unsupported",
                            "epsilon cannot round-trip through upstream's float attribute parser",
                        ));
                    }
                    write!(
                        result,
                        ", epsilon={}, feature_index={}",
                        numbers::print(u128::from(epsilon.to_bits()), PrimitiveType::F64)?,
                        instruction.feature_index
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?
                }
                _ => {}
            }
            if opcode == Opcode::Fft {
                let kind = instruction
                    .fft_type
                    .enum_value()
                    .map_err(|_| Error::new("fft", "unknown FFT type"))?;
                write!(
                    result,
                    ", fft_type={}, fft_length={{{}}}",
                    kind.descriptor().name(),
                    instruction
                        .fft_length
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                )
                .map_err(|e| Error::new("print", e.to_string()))?;
            }
            if matches!(opcode, Opcode::Dot | Opcode::RaggedDot | Opcode::ScaledDot) {
                let dimensions = if opcode == Opcode::RaggedDot {
                    instruction
                        .ragged_dot_dimension_numbers
                        .as_ref()
                        .and_then(|r| r.dot_dimension_numbers.as_ref())
                } else {
                    instruction.dot_dimension_numbers.as_ref()
                }
                .ok_or_else(|| Error::new("dot", "missing dot dimension numbers"))?;
                reject_fields(
                    dimensions,
                    &[
                        "lhs_contracting_dimensions",
                        "rhs_contracting_dimensions",
                        "lhs_batch_dimensions",
                        "rhs_batch_dimensions",
                    ],
                )?;
                for (name, axes) in [
                    (
                        "lhs_contracting_dims",
                        &dimensions.lhs_contracting_dimensions,
                    ),
                    (
                        "rhs_contracting_dims",
                        &dimensions.rhs_contracting_dimensions,
                    ),
                    ("lhs_batch_dims", &dimensions.lhs_batch_dimensions),
                    ("rhs_batch_dims", &dimensions.rhs_batch_dimensions),
                ] {
                    write!(
                        result,
                        ", {name}={{{}}}",
                        axes.iter()
                            .map(i64::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                }
            }
            if matches!(
                opcode,
                Opcode::Dot
                    | Opcode::Convolution
                    | Opcode::CustomCall
                    | Opcode::RaggedDot
                    | Opcode::ScaledDot
            ) {
                let precision = instruction.precision_config.as_ref().ok_or_else(|| {
                    Error::new(
                        "unsupported",
                        "dot/convolution without precision configuration cannot be preserved in text",
                    )
                })?;
                reject_fields(precision, &["operand_precision", "algorithm"])?;
                let values = precision
                    .operand_precision
                    .iter()
                    .map(|value| {
                        value
                            .enum_value()
                            .map(|v| v.descriptor().name().to_ascii_lowercase())
                            .map_err(|_| Error::new("precision", "unknown operand precision"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let algorithm = precision
                    .algorithm
                    .enum_value()
                    .map_err(|_| Error::new("precision", "unknown precision algorithm"))?
                    .descriptor()
                    .name()
                    .to_owned();
                write!(result, ", operand_precision={{{}}}", values.join(","))
                    .map_err(|e| Error::new("print", e.to_string()))?;
                if !matches!(opcode, Opcode::CustomCall | Opcode::RaggedDot) {
                    write!(
                        result,
                        ", algorithm={}",
                        algorithm
                            .strip_prefix("ALG_")
                            .ok_or_else(|| Error::new("precision", "invalid algorithm name"))?
                            .to_ascii_lowercase()
                    )
                    .map_err(|e| Error::new("print", e.to_string()))?;
                } else if precision.algorithm.enum_value_or_default() != Algorithm::ALG_UNSET {
                    return Err(Error::new(
                        "unsupported",
                        "custom-call precision algorithm cannot be preserved in text",
                    ));
                }
            }
            callable::print_fields(instruction, &mut result)?;
            dot::print_fields(instruction, &mut result)?;
            sharding::print_fields(instruction, &mut result)?;
            if let Some(value) = instruction.original_value.as_ref() {
                result.push_str(&format!(", origin={}", original_value::print(value)?));
            }
            if let Some(statistics) = instruction.statistics_viz.as_ref() {
                result.push_str(&format!(
                    ", statistics={}",
                    metadata::print_statistics(statistics)?
                ));
            }
            if let Some(replication) = instruction.parameter_replication.as_ref() {
                result.push_str(&format!(
                    ", parameter_replication={}",
                    metadata::print_parameter_replication(replication, &shape)?
                ));
            }
            if let Some(accuracy) = instruction.result_accuracy.as_ref() {
                result.push_str(&format!(", result_accuracy={}", accuracy::print(accuracy)?));
            }
            if empty_async_backends.contains(&instruction.id) {
                result.push_str(", backend_config=\"\"");
            }
            let channel =
                module.channel_id(module.instruction_handle(computation.id, instruction.id))?;
            if instruction.channel_id != 0 && channel != Some(instruction.channel_id) {
                return Err(Error::new(
                    "unsupported",
                    "raw channel field cannot preserve imported channel absence in text",
                ));
            }
            collective::print_fields(instruction, &computation.instructions, channel, &mut result)?;
            if matches!(
                opcode,
                Opcode::Send | Opcode::SendDone | Opcode::Recv | Opcode::RecvDone
            ) {
                if let Some(channel) = channel {
                    write!(result, ", channel_id={channel}")
                        .map_err(|e| Error::new("print", e.to_string()))?;
                }
                write!(
                    result,
                    ", is_host_transfer={}",
                    instruction.is_host_transfer
                )
                .map_err(|e| Error::new("print", e.to_string()))?;
            }
            if opcode == Opcode::CopyStart && instruction.has_cross_program_prefetch_index() {
                write!(
                    result,
                    ", cross_program_prefetch_index={}",
                    instruction.cross_program_prefetch_index()
                )
                .map_err(|e| Error::new("print", e.to_string()))?;
            }
            result.push('\n');
        }
        result.push('}');
        if !computation.execution_thread.is_empty() {
            write!(
                result,
                ", execution_thread={}",
                strings::quote(computation.execution_thread.as_bytes())?
            )
            .map_err(|e| Error::new("print", e.to_string()))?;
        }
        result.push_str("\n\n");
    }
    Ok(result)
}

fn literal_field(element: PrimitiveType) -> Result<&'static str> {
    use PrimitiveType::*;
    Ok(match element {
        PRED => "preds",
        S1 => "s1s",
        S2 => "s2s",
        S4 => "s4s",
        S8 => "s8s",
        S16 => "s16s",
        S32 => "s32s",
        S64 => "s64s",
        U1 => "u1s",
        U2 => "u2s",
        U4 => "u4s",
        U8 => "u8s",
        U16 => "u16s",
        U32 => "u32s",
        U64 => "u64s",
        F32 => "f32s",
        F64 => "f64s",
        F16 => "f16s",
        BF16 => "bf16s",
        F8E3M4 => "f8e3m4s",
        F8E4M3 => "f8e4m3s",
        F8E5M2 => "f8e5m2s",
        F8E4M3FN => "f8e4m3fns",
        F8E4M3FNUZ => "f8e4m3fnuzs",
        F8E4M3B11FNUZ => "f8e4m3b11fnuzs",
        F8E5M2FNUZ => "f8e5m2fnuzs",
        F8E8M0FNU => "f8e8m0fnus",
        F4E2M1FN => "f4e2m1fns",
        F6E2M3FN => "f6e2m3fns",
        F6E3M2FN => "f6e3m2fns",
        C64 => "c64s",
        C128 => "c128s",
        _ => return Err(Error::new("unsupported", "literal text type")),
    })
}

fn physical_indices(shape: &Shape) -> Result<Vec<usize>> {
    if shape.is_dynamic() {
        return Err(Error::new(
            "unsupported",
            "dynamic literal layout conversion",
        ));
    }
    let count = usize::try_from(shape.element_count()?)
        .map_err(|_| Error::new("overflow", "literal count"))?;
    let layout = shape
        .layout()
        .map(|l| l.minor_to_major.clone())
        .unwrap_or_else(|| (0..shape.dimensions().len() as i64).rev().collect());
    if let Some(layout) = shape.layout() {
        reject_fields(
            layout,
            &["minor_to_major", "tail_padding_alignment_in_elements"],
        )?;
        if layout.minor_to_major.len() != shape.rank()? {
            return Err(Error::new("unsupported", "AUTO literal layout"));
        }
        if !matches!(layout.tail_padding_alignment_in_elements, 0 | 1) {
            return Err(Error::new("unsupported", "padded literal layout"));
        }
    }

    let mut result = Vec::with_capacity(count);
    for logical in 0..count {
        let mut remainder = logical;
        let mut coordinates = vec![0; shape.dimensions().len()];
        for axis in (0..coordinates.len()).rev() {
            let size = shape.dimensions()[axis] as usize;
            coordinates[axis] = remainder % size;
            remainder /= size;
        }
        let mut physical = 0;
        let mut stride = 1;
        for &axis in &layout {
            let axis = axis as usize;
            physical += coordinates[axis] * stride;
            stride *= shape.dimensions()[axis] as usize;
        }
        result.push(physical);
    }
    Ok(result)
}

fn logical_to_physical(shape: &Shape, logical: &[u128]) -> Result<Vec<u128>> {
    let mut physical = vec![0; logical.len()];
    for (value, index) in logical.iter().zip(physical_indices(shape)?) {
        physical[index] = *value;
    }
    Ok(physical)
}

fn physical_to_logical(shape: &Shape, physical: &[u128]) -> Result<Vec<u128>> {
    physical_indices(shape)?
        .into_iter()
        .map(|index| {
            physical
                .get(index)
                .copied()
                .ok_or_else(|| Error::new("literal", "storage index out of range"))
        })
        .collect()
}

fn print_literal(
    shape: &Shape,
    values: &[u128],
    axis: usize,
    offset: &mut usize,
    output: &mut String,
) -> Result<()> {
    struct Frame {
        axis: usize,
        next_index: i64,
        size: i64,
    }
    let rank = shape.rank()?;
    let mut current_axis = axis;
    let mut frames = Vec::new();
    'visit: loop {
        if current_axis >= MAX_DEPTH {
            return Err(Error::new("resource-limit", "literal print nesting limit"));
        }
        if current_axis == rank {
            let bits = *values
                .get(*offset)
                .ok_or_else(|| Error::new("literal", "missing element"))?;
            *offset += 1;
            let ty = shape.element_type();
            let value = if ty == PrimitiveType::PRED {
                if bits == 0 {
                    "false".into()
                } else {
                    "true".into()
                }
            } else if ty.is_signed_integral() {
                let width = ty
                    .bit_width()
                    .ok_or_else(|| Error::new("literal", "missing bit width"))?;
                (((bits << (128 - width)) as i128) >> (128 - width)).to_string()
            } else if ty.is_integral() {
                bits.to_string()
            } else {
                numbers::print(bits, ty)?
            };
            output.push_str(&value);
        } else {
            let size = shape.dimensions()[current_axis];
            output.push('{');
            if size == 0 {
                output.push('}');
            } else {
                frames.push(Frame {
                    axis: current_axis,
                    next_index: 0,
                    size,
                });
                current_axis += 1;
                continue;
            }
        }

        loop {
            let Some(mut frame) = frames.pop() else {
                return Ok(());
            };
            frame.next_index += 1;
            if frame.next_index < frame.size {
                output.push_str(", ");
                current_axis = frame.axis + 1;
                frames.push(frame);
                continue 'visit;
            }
            output.push('}');
        }
    }
}

#[cfg(test)]
mod shortform_tests {
    use super::fill_shortform_bytes;

    #[test]
    fn source_counter_sequence_and_tail_offsets_are_exact() {
        let mut counters = (0, 0);
        let mut first = [0; 8];
        fill_shortform_bytes(&mut first, &mut counters, false);
        assert_eq!(first, [0, 0, 0, 0, 1, 0, 0, 0]);
        let mut next = [0; 7];
        fill_shortform_bytes(&mut next, &mut counters, false);
        assert_eq!(next, [2, 0, 1, 2, 0, 0, 0]);
        assert_eq!(counters, (3, 3));
    }

    #[test]
    fn predicate_mask_and_unsigned_wraparound_follow_upstream() {
        let mut counters = (u32::MAX, u8::MAX);
        let mut bytes = [0; 5];
        fill_shortform_bytes(&mut bytes, &mut counters, true);
        assert_eq!(bytes, [1, 1, 1, 1, 0]);
        assert_eq!(counters, (0, 0));
    }
}
