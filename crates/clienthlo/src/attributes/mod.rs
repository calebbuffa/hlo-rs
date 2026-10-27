//! Attributes and enums used by CHLO operations.
//!
//! Enum attributes carry valid Rust enum values, so their standalone verification
//! is infallible. Operation verification supplies the contextual checks, such as
//! precision-list length and whether dimension indices fit the operand ranks.
//! Constructors preserve CHLO's wire names and do not rename dialect attributes
//! to match Rust terminology.

use pliron::{
    builtin::{
        attributes::{IntegerAttr, VecAttr},
        types::{IntegerType, Signedness},
    },
    combine::{
        Parser, attempt, between, choice, parser::char::spaces, parser::char::string, token,
    },
    common_traits::Verify,
    context::Context,
    derive::{def_attribute, pliron_attr},
    irfmt::parsers::{delimited_list_parser, number_as_string_parser},
    location::Location,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream, StreamLocation},
    printable::{self, Printable},
    result::Result,
    utils::apint::{APInt, bw},
    verify_err,
};

mod enums;
pub use enums::{EnumParseError, EnumValueError};

/// Elementwise comparison relation, with CHLO's stable numeric encoding.
///
/// Text uses uppercase symbols such as `EQ` and `LT`. Conversion from `u64`
/// rejects undefined encodings with [`EnumValueError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ComparisonDirection {
    /// Equality.
    Eq = 0,
    /// Inequality.
    Ne = 1,
    /// Greater than or equal.
    Ge = 2,
    /// Greater than.
    Gt = 3,
    /// Less than or equal.
    Le = 4,
    /// Less than.
    Lt = 5,
}

impl From<ComparisonDirection> for u64 {
    fn from(value: ComparisonDirection) -> Self {
        value as u64
    }
}

impl TryFrom<u64> for ComparisonDirection {
    type Error = EnumValueError;

    fn try_from(value: u64) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Eq),
            1 => Ok(Self::Ne),
            2 => Ok(Self::Ge),
            3 => Ok(Self::Gt),
            4 => Ok(Self::Le),
            5 => Ok(Self::Lt),
            _ => Err(EnumValueError {
                enum_name: Self::NAME,
                value,
            }),
        }
    }
}

impl From<stablehlo::attributes::ComparisonDirection> for ComparisonDirection {
    fn from(value: stablehlo::attributes::ComparisonDirection) -> Self {
        match value {
            stablehlo::attributes::ComparisonDirection::Eq => Self::Eq,
            stablehlo::attributes::ComparisonDirection::Ne => Self::Ne,
            stablehlo::attributes::ComparisonDirection::Ge => Self::Ge,
            stablehlo::attributes::ComparisonDirection::Gt => Self::Gt,
            stablehlo::attributes::ComparisonDirection::Le => Self::Le,
            stablehlo::attributes::ComparisonDirection::Lt => Self::Lt,
        }
    }
}

impl From<ComparisonDirection> for stablehlo::attributes::ComparisonDirection {
    fn from(value: ComparisonDirection) -> Self {
        match value {
            ComparisonDirection::Eq => Self::Eq,
            ComparisonDirection::Ne => Self::Ne,
            ComparisonDirection::Ge => Self::Ge,
            ComparisonDirection::Gt => Self::Gt,
            ComparisonDirection::Le => Self::Le,
            ComparisonDirection::Lt => Self::Lt,
        }
    }
}

/// Interpretation of the operands of a comparison.
///
/// The Rust name avoids confusing a comparison mode with an IR type. The CHLO
/// wire name remains `comparison_type`, including the `NOTYPE` spelling.
/// Undefined numeric encodings are rejected with [`EnumValueError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ComparisonKind {
    /// No explicitly selected comparison semantics.
    NoType = 0,
    /// Floating-point comparison semantics.
    Float = 1,
    /// Total ordering, including exceptional floating-point values.
    TotalOrder = 2,
    /// Signed integer comparison semantics.
    Signed = 3,
    /// Unsigned integer comparison semantics.
    Unsigned = 4,
    /// Weak ordering semantics.
    WeakOrder = 5,
}

impl From<ComparisonKind> for u64 {
    fn from(value: ComparisonKind) -> Self {
        value as u64
    }
}

impl TryFrom<u64> for ComparisonKind {
    type Error = EnumValueError;

    fn try_from(value: u64) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::NoType),
            1 => Ok(Self::Float),
            2 => Ok(Self::TotalOrder),
            3 => Ok(Self::Signed),
            4 => Ok(Self::Unsigned),
            5 => Ok(Self::WeakOrder),
            _ => Err(EnumValueError {
                enum_name: Self::NAME,
                value,
            }),
        }
    }
}

impl From<stablehlo::attributes::ComparisonKind> for ComparisonKind {
    fn from(value: stablehlo::attributes::ComparisonKind) -> Self {
        match value {
            stablehlo::attributes::ComparisonKind::NoType => Self::NoType,
            stablehlo::attributes::ComparisonKind::Float => Self::Float,
            stablehlo::attributes::ComparisonKind::TotalOrder => Self::TotalOrder,
            stablehlo::attributes::ComparisonKind::Signed => Self::Signed,
            stablehlo::attributes::ComparisonKind::Unsigned => Self::Unsigned,
            stablehlo::attributes::ComparisonKind::WeakOrder => Self::WeakOrder,
        }
    }
}

impl From<ComparisonKind> for stablehlo::attributes::ComparisonKind {
    fn from(value: ComparisonKind) -> Self {
        match value {
            ComparisonKind::NoType => Self::NoType,
            ComparisonKind::Float => Self::Float,
            ComparisonKind::TotalOrder => Self::TotalOrder,
            ComparisonKind::Signed => Self::Signed,
            ComparisonKind::Unsigned => Self::Unsigned,
            ComparisonKind::WeakOrder => Self::WeakOrder,
        }
    }
}

/// Requested implementation precision for a dot-product operand.
///
/// These are backend precision requests, not a change to the tensor element type.
/// Undefined numeric encodings are rejected with [`EnumValueError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Precision {
    /// Use the default precision.
    Default = 0,
    /// Request high precision.
    High = 1,
    /// Request the highest supported precision.
    Highest = 2,
}

impl From<Precision> for u64 {
    fn from(value: Precision) -> Self {
        value as u64
    }
}

impl TryFrom<u64> for Precision {
    type Error = EnumValueError;

    fn try_from(value: u64) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Default),
            1 => Ok(Self::High),
            2 => Ok(Self::Highest),
            _ => Err(EnumValueError {
                enum_name: Self::NAME,
                value,
            }),
        }
    }
}

/// Typed `chlo.comparison_direction` attribute storing a comparison relation.
#[pliron_attr(name = "chlo.comparison_direction", format = "$value")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComparisonDirectionAttr {
    /// Relation used to compare each pair of operand elements.
    pub value: ComparisonDirection,
}

impl ComparisonDirectionAttr {
    /// Creates an attribute for the given comparison relation.
    pub const fn new(value: ComparisonDirection) -> Self {
        Self { value }
    }

    /// Returns the numeric encoding used by CHLO.
    pub fn numeric_value(&self) -> u64 {
        self.value.into()
    }
}

impl Verify for ComparisonDirectionAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        Ok(())
    }
}

/// Typed comparison semantics attribute, printed as `chlo.comparison_type`.
#[pliron_attr(name = "chlo.comparison_type", format = "$value")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComparisonKindAttr {
    /// Interpretation of the compared elements.
    pub value: ComparisonKind,
}

impl ComparisonKindAttr {
    /// Creates an attribute for the given comparison semantics.
    pub const fn new(value: ComparisonKind) -> Self {
        Self { value }
    }

    /// Returns the stable CHLO numeric encoding of the comparison semantics.
    pub fn numeric_value(&self) -> u64 {
        self.value.into()
    }
}

impl Verify for ComparisonKindAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        Ok(())
    }
}

/// Typed `chlo.precision` attribute storing a backend precision request.
#[pliron_attr(name = "chlo.precision", format = "$value")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrecisionAttr {
    /// Precision requested for an operand of a dot product.
    pub value: Precision,
}

impl PrecisionAttr {
    /// Creates an attribute for the requested precision.
    pub const fn new(value: Precision) -> Self {
        Self { value }
    }

    /// Returns the stable CHLO numeric encoding of the precision request.
    pub fn numeric_value(&self) -> u64 {
        self.value.into()
    }
}

impl Verify for PrecisionAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        Ok(())
    }
}

/// Ordered precision requests stored as `chlo.precision_config`.
///
/// The containing operation checks the allowed number of entries; the attribute
/// itself accepts an empty list. Construct it with [`precision_config`].
#[pliron_attr(
    name = "chlo.precision_config",
    format = "`[` vec($values, CharSpace(`,`)) `]`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrecisionConfigAttr {
    /// Precision entries in operand order, not one entry per output element.
    pub values: Vec<PrecisionAttr>,
}

impl Verify for PrecisionConfigAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for precision in &self.values {
            precision.verify(ctx)?;
        }
        Ok(())
    }
}

/// Collects precision requests into a typed configuration, preserving operand order.
///
/// An empty iterator creates an empty configuration. Operation verification checks
/// whether its length is valid for the operation using it.
pub fn precision_config(values: impl IntoIterator<Item = Precision>) -> PrecisionConfigAttr {
    PrecisionConfigAttr {
        values: values.into_iter().map(PrecisionAttr::new).collect(),
    }
}

/// Copies the precision entries from a typed configuration in operand order.
///
/// This cannot fail: every entry already contains a valid [`Precision`].
/// An empty configuration produces an empty vector.
pub fn read_precision_config(attr: &PrecisionConfigAttr) -> Vec<Precision> {
    attr.values.iter().map(|attr| attr.value).collect()
}

/// Dimension mappings of `chlo.ragged_dot`, printed using named axis lists.
///
/// All axes are zero-based. Attribute verification rejects negative axes;
/// [`crate::ops::RaggedDotOp`] checks ranks, pairing, uniqueness, and geometry.
/// Omitted textual fields are empty lists; duplicate fields are rejected.
#[def_attribute("chlo.ragged_dot")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RaggedDotDimensionNumbersAttr {
    /// Left-hand batch axes, paired in order with right-hand batch axes.
    pub lhs_batching_dimensions: Vec<i64>,
    /// Right-hand batch axes, paired in order with left-hand batch axes.
    pub rhs_batching_dimensions: Vec<i64>,
    /// Left-hand axes reduced by the dot product, paired with right-hand axes.
    pub lhs_contracting_dimensions: Vec<i64>,
    /// Right-hand axes reduced by the dot product, paired with left-hand axes.
    pub rhs_contracting_dimensions: Vec<i64>,
    /// Left-hand ragged axis partitioned according to the group sizes.
    pub lhs_ragged_dimensions: Vec<i64>,
    /// Right-hand group axis selecting the group-specific dot-product operand.
    pub rhs_group_dimensions: Vec<i64>,
}

impl Printable for RaggedDotDimensionNumbersAttr {
    fn fmt(
        &self,
        _: &Context,
        _: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let fields = [
            ("lhs_batching_dimensions", &self.lhs_batching_dimensions),
            ("rhs_batching_dimensions", &self.rhs_batching_dimensions),
            (
                "lhs_contracting_dimensions",
                &self.lhs_contracting_dimensions,
            ),
            (
                "rhs_contracting_dimensions",
                &self.rhs_contracting_dimensions,
            ),
            ("lhs_ragged_dimensions", &self.lhs_ragged_dimensions),
            ("rhs_group_dimensions", &self.rhs_group_dimensions),
        ];
        f.write_str("<")?;
        let mut separator = "";
        for (name, values) in fields {
            if values.is_empty() {
                continue;
            }
            write!(f, "{separator}{name}=[")?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{value}")?;
            }
            f.write_str("]")?;
            separator = ",";
        }
        f.write_str(">")
    }
}

fn ragged_field_parser<'a>(
    index: usize,
    name: &'static str,
) -> impl Parser<StateStream<'a>, Output = (usize, Vec<i64>)> + 'a {
    string(name)
        .skip(spaces())
        .skip(token('='))
        .skip(spaces())
        .with(delimited_list_parser(
            '[',
            ']',
            ',',
            number_as_string_parser().and_then(|digits| digits.parse::<i64>()),
        ))
        .map(move |values| (index, values))
}

impl Parsable for RaggedDotDimensionNumbersAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let fields_parser = between(
            token('<').skip(spaces()),
            spaces().with(token('>')),
            pliron::combine::sep_by::<Vec<(usize, Vec<i64>)>, _, _, _>(
                choice((
                    attempt(ragged_field_parser(0, "lhs_batching_dimensions")),
                    attempt(ragged_field_parser(1, "rhs_batching_dimensions")),
                    attempt(ragged_field_parser(2, "lhs_contracting_dimensions")),
                    attempt(ragged_field_parser(3, "rhs_contracting_dimensions")),
                    attempt(ragged_field_parser(4, "lhs_ragged_dimensions")),
                    attempt(ragged_field_parser(5, "rhs_group_dimensions")),
                ))
                .skip(spaces()),
                token(',').skip(spaces()),
            ),
        );
        fields_parser
            .then(|fields| {
                pliron::combine::parser(move |state_stream: &mut StateStream<'a>| {
                    let mut parsed: [Option<Vec<i64>>; 6] = Default::default();
                    for (index, values) in &fields {
                        if parsed[*index].replace(values.clone()).is_some() {
                            return pliron::input_err!(
                                state_stream.loc(),
                                "duplicate CHLO ragged-dot dimension field"
                            )
                            .into_parse_result();
                        }
                    }
                    let mut parsed = parsed.into_iter().map(Option::unwrap_or_default);
                    Ok(Self {
                        lhs_batching_dimensions: parsed.next().unwrap_or_default(),
                        rhs_batching_dimensions: parsed.next().unwrap_or_default(),
                        lhs_contracting_dimensions: parsed.next().unwrap_or_default(),
                        rhs_contracting_dimensions: parsed.next().unwrap_or_default(),
                        lhs_ragged_dimensions: parsed.next().unwrap_or_default(),
                        rhs_group_dimensions: parsed.next().unwrap_or_default(),
                    })
                    .into_parse_result()
                })
            })
            .parse_stream(state_stream)
            .into_result()
    }
}

/// The six dimension groups that define CHLO ragged-dot geometry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RaggedDotDimensions {
    /// Paired batch axes of the lhs.
    pub lhs_batching_dimensions: Vec<i64>,
    /// Paired batch axes of the rhs.
    pub rhs_batching_dimensions: Vec<i64>,
    /// Contracting axes of the lhs.
    pub lhs_contracting_dimensions: Vec<i64>,
    /// Contracting axes of the rhs.
    pub rhs_contracting_dimensions: Vec<i64>,
    /// Ragged axes of the lhs.
    pub lhs_ragged_dimensions: Vec<i64>,
    /// Group axes of the rhs.
    pub rhs_group_dimensions: Vec<i64>,
}

impl RaggedDotDimensionNumbersAttr {
    /// Creates the attribute from named batching, contracting, and ragged dimensions.
    ///
    /// Takes ownership of the axis lists without validating them. Call `verify`
    /// to check nonnegative indices and verify the containing operation to check
    /// their relationship to the operand shapes.
    pub fn new(dimensions: RaggedDotDimensions) -> Self {
        Self {
            lhs_batching_dimensions: dimensions.lhs_batching_dimensions,
            rhs_batching_dimensions: dimensions.rhs_batching_dimensions,
            lhs_contracting_dimensions: dimensions.lhs_contracting_dimensions,
            rhs_contracting_dimensions: dimensions.rhs_contracting_dimensions,
            lhs_ragged_dimensions: dimensions.lhs_ragged_dimensions,
            rhs_group_dimensions: dimensions.rhs_group_dimensions,
        }
    }

    /// Returns the dimension groups as an owned value.
    pub fn dimensions(&self) -> RaggedDotDimensions {
        RaggedDotDimensions {
            lhs_batching_dimensions: self.lhs_batching_dimensions.clone(),
            rhs_batching_dimensions: self.rhs_batching_dimensions.clone(),
            lhs_contracting_dimensions: self.lhs_contracting_dimensions.clone(),
            rhs_contracting_dimensions: self.rhs_contracting_dimensions.clone(),
            lhs_ragged_dimensions: self.lhs_ragged_dimensions.clone(),
            rhs_group_dimensions: self.rhs_group_dimensions.clone(),
        }
    }
}

impl Verify for RaggedDotDimensionNumbersAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if [
            &self.lhs_batching_dimensions,
            &self.rhs_batching_dimensions,
            &self.lhs_contracting_dimensions,
            &self.rhs_contracting_dimensions,
            &self.lhs_ragged_dimensions,
            &self.rhs_group_dimensions,
        ]
        .iter()
        .any(|dimensions| dimensions.iter().any(|dimension| *dimension < 0))
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidOperation {
                    operation: "CHLO ragged-dot dimension numbers".to_owned(),
                    location: "unknown".to_owned(),
                    reason: "dimensions must be non-negative"
                }
            );
        }
        Ok(())
    }
}

pub(crate) fn signed_list(ctx: &Context, values: &[i64]) -> VecAttr {
    VecAttr::new(
        values
            .iter()
            .map(|value| {
                Box::new(IntegerAttr::new(
                    IntegerType::get(ctx, 64, Signedness::Signed),
                    APInt::from_i64(*value, bw(64)),
                )) as pliron::attribute::AttrObj
            })
            .collect(),
    )
}

pub(crate) fn read_signed_list(attr: &VecAttr) -> Option<Vec<i64>> {
    attr.0
        .iter()
        .map(|value| {
            value
                .downcast_ref::<IntegerAttr>()?
                .value()
                .to_string_decimal(true)
                .parse()
                .ok()
        })
        .collect()
}
