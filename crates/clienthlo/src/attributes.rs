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
    derive::{def_attribute, format_attribute, pliron_attr},
    irfmt::parsers::{delimited_list_parser, number_as_string_parser},
    location::{Located, Location},
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    result::Result,
    utils::apint::{APInt, bw},
    verify_err,
};

#[format_attribute]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum ComparisonDirection {
    EQ = 0,
    NE = 1,
    GE = 2,
    GT = 3,
    LE = 4,
    LT = 5,
}

impl ComparisonDirection {
    #[allow(non_upper_case_globals)]
    pub const Eq: Self = Self::EQ;
    #[allow(non_upper_case_globals)]
    pub const Ne: Self = Self::NE;
    #[allow(non_upper_case_globals)]
    pub const Ge: Self = Self::GE;
    #[allow(non_upper_case_globals)]
    pub const Gt: Self = Self::GT;
    #[allow(non_upper_case_globals)]
    pub const Le: Self = Self::LE;
    #[allow(non_upper_case_globals)]
    pub const Lt: Self = Self::LT;

    pub const fn as_u64(self) -> u64 {
        self as u64
    }

    pub const fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::EQ),
            1 => Some(Self::NE),
            2 => Some(Self::GE),
            3 => Some(Self::GT),
            4 => Some(Self::LE),
            5 => Some(Self::LT),
            _ => None,
        }
    }
}

impl From<stablehlo::attributes::ComparisonDirection> for ComparisonDirection {
    fn from(value: stablehlo::attributes::ComparisonDirection) -> Self {
        match value {
            stablehlo::attributes::ComparisonDirection::Eq => Self::EQ,
            stablehlo::attributes::ComparisonDirection::Ne => Self::NE,
            stablehlo::attributes::ComparisonDirection::Ge => Self::GE,
            stablehlo::attributes::ComparisonDirection::Gt => Self::GT,
            stablehlo::attributes::ComparisonDirection::Le => Self::LE,
            stablehlo::attributes::ComparisonDirection::Lt => Self::LT,
        }
    }
}

impl From<ComparisonDirection> for stablehlo::attributes::ComparisonDirection {
    fn from(value: ComparisonDirection) -> Self {
        match value {
            ComparisonDirection::EQ => Self::Eq,
            ComparisonDirection::NE => Self::Ne,
            ComparisonDirection::GE => Self::Ge,
            ComparisonDirection::GT => Self::Gt,
            ComparisonDirection::LE => Self::Le,
            ComparisonDirection::LT => Self::Lt,
        }
    }
}

#[format_attribute]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum ComparisonType {
    NOTYPE = 0,
    FLOAT = 1,
    TOTALORDER = 2,
    SIGNED = 3,
    UNSIGNED = 4,
    WEAKORDER = 5,
}

impl ComparisonType {
    #[allow(non_upper_case_globals)]
    pub const NoType: Self = Self::NOTYPE;
    #[allow(non_upper_case_globals)]
    pub const Float: Self = Self::FLOAT;
    #[allow(non_upper_case_globals)]
    pub const TotalOrder: Self = Self::TOTALORDER;
    #[allow(non_upper_case_globals)]
    pub const Signed: Self = Self::SIGNED;
    #[allow(non_upper_case_globals)]
    pub const Unsigned: Self = Self::UNSIGNED;
    #[allow(non_upper_case_globals)]
    pub const WeakOrder: Self = Self::WEAKORDER;

    pub const fn as_u64(self) -> u64 {
        self as u64
    }

    pub const fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::NOTYPE),
            1 => Some(Self::FLOAT),
            2 => Some(Self::TOTALORDER),
            3 => Some(Self::SIGNED),
            4 => Some(Self::UNSIGNED),
            5 => Some(Self::WEAKORDER),
            _ => None,
        }
    }
}

impl From<stablehlo::attributes::ComparisonType> for ComparisonType {
    fn from(value: stablehlo::attributes::ComparisonType) -> Self {
        match value {
            stablehlo::attributes::ComparisonType::NoType => Self::NOTYPE,
            stablehlo::attributes::ComparisonType::Float => Self::FLOAT,
            stablehlo::attributes::ComparisonType::TotalOrder => Self::TOTALORDER,
            stablehlo::attributes::ComparisonType::Signed => Self::SIGNED,
            stablehlo::attributes::ComparisonType::Unsigned => Self::UNSIGNED,
            stablehlo::attributes::ComparisonType::WeakOrder => Self::WEAKORDER,
        }
    }
}

impl From<ComparisonType> for stablehlo::attributes::ComparisonType {
    fn from(value: ComparisonType) -> Self {
        match value {
            ComparisonType::NOTYPE => Self::NoType,
            ComparisonType::FLOAT => Self::Float,
            ComparisonType::TOTALORDER => Self::TotalOrder,
            ComparisonType::SIGNED => Self::Signed,
            ComparisonType::UNSIGNED => Self::Unsigned,
            ComparisonType::WEAKORDER => Self::WeakOrder,
        }
    }
}

#[format_attribute]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum Precision {
    DEFAULT = 0,
    HIGH = 1,
    HIGHEST = 2,
}

impl Precision {
    #[allow(non_upper_case_globals)]
    pub const Default: Self = Self::DEFAULT;
    #[allow(non_upper_case_globals)]
    pub const High: Self = Self::HIGH;
    #[allow(non_upper_case_globals)]
    pub const Highest: Self = Self::HIGHEST;

    pub const fn as_u64(self) -> u64 {
        self as u64
    }

    pub const fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::DEFAULT),
            1 => Some(Self::HIGH),
            2 => Some(Self::HIGHEST),
            _ => None,
        }
    }
}

#[pliron_attr(name = "chlo.comparison_direction", format = "$value")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComparisonDirectionAttr {
    pub value: ComparisonDirection,
}

impl ComparisonDirectionAttr {
    pub const fn new(value: ComparisonDirection) -> Self {
        Self { value }
    }

    pub const fn numeric_value(&self) -> u64 {
        self.value.as_u64()
    }
}

impl Verify for ComparisonDirectionAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        Ok(())
    }
}

#[pliron_attr(name = "chlo.comparison_type", format = "$value")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComparisonTypeAttr {
    pub value: ComparisonType,
}

impl ComparisonTypeAttr {
    pub const fn new(value: ComparisonType) -> Self {
        Self { value }
    }

    pub const fn numeric_value(&self) -> u64 {
        self.value.as_u64()
    }
}

impl Verify for ComparisonTypeAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        Ok(())
    }
}

#[pliron_attr(name = "chlo.precision", format = "$value")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrecisionAttr {
    pub value: Precision,
}

impl PrecisionAttr {
    pub const fn new(value: Precision) -> Self {
        Self { value }
    }

    pub const fn numeric_value(&self) -> u64 {
        self.value.as_u64()
    }
}

impl Verify for PrecisionAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        Ok(())
    }
}

#[pliron_attr(
    name = "chlo.precision_config",
    format = "`[` vec($values, CharSpace(`,`)) `]`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrecisionConfigAttr {
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

pub fn precision_config(values: impl IntoIterator<Item = Precision>) -> PrecisionConfigAttr {
    PrecisionConfigAttr {
        values: values.into_iter().map(PrecisionAttr::new).collect(),
    }
}

pub fn read_precision_config(attr: &PrecisionConfigAttr) -> Option<Vec<Precision>> {
    Some(attr.values.iter().map(|attr| attr.value).collect())
}

#[def_attribute("chlo.ragged_dot")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RaggedDotDimensionNumbersAttr {
    pub lhs_batching_dimensions: Vec<i64>,
    pub rhs_batching_dimensions: Vec<i64>,
    pub lhs_contracting_dimensions: Vec<i64>,
    pub rhs_contracting_dimensions: Vec<i64>,
    pub lhs_ragged_dimensions: Vec<i64>,
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

pub type RaggedDotDimensions = (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>);

impl RaggedDotDimensionNumbersAttr {
    pub fn new(
        _: &Context,
        lhs_batching_dimensions: Vec<i64>,
        rhs_batching_dimensions: Vec<i64>,
        lhs_contracting_dimensions: Vec<i64>,
        rhs_contracting_dimensions: Vec<i64>,
        lhs_ragged_dimensions: Vec<i64>,
        rhs_group_dimensions: Vec<i64>,
    ) -> Self {
        Self {
            lhs_batching_dimensions,
            rhs_batching_dimensions,
            lhs_contracting_dimensions,
            rhs_contracting_dimensions,
            lhs_ragged_dimensions,
            rhs_group_dimensions,
        }
    }

    pub fn dimensions(&self) -> RaggedDotDimensions {
        (
            self.lhs_batching_dimensions.clone(),
            self.rhs_batching_dimensions.clone(),
            self.lhs_contracting_dimensions.clone(),
            self.rhs_contracting_dimensions.clone(),
            self.lhs_ragged_dimensions.clone(),
            self.rhs_group_dimensions.clone(),
        )
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
