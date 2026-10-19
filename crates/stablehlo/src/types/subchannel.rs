use super::*;
use pliron::{location::Location, verify_err};
#[pliron_type(
    name = "stablehlo.subchannel_quantized",
    format = "`<` $storage `,` $expressed `,` `[` vec($quantized_dimensions, CharSpace(`,`)) `]` `,` `[` vec($block_sizes, CharSpace(`,`)) `]` `,` `[` vec($parameter_shape, CharSpace(`,`)) `]` `,` `[` vec($scale_bits, CharSpace(`,`)) `]` `,` `[` vec($zero_point_bits, CharSpace(`,`)) `]` `>`",
    generate_get = false
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubchannelQuantizedType {
    pub storage: TypeHandle,
    pub expressed: TypeHandle,
    pub quantized_dimensions: Vec<i32>,
    pub block_sizes: Vec<i64>,
    pub parameter_shape: Vec<i64>,
    pub scale_bits: Vec<u64>,
    pub zero_point_bits: Vec<u64>,
}
/// Parameter data for an interned subchannel quantized type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubchannelQuantizationParameters {
    /// Quantized tensor dimensions.
    pub quantized_dimensions: Vec<i32>,
    /// Block size for each quantized dimension.
    pub block_sizes: Vec<i64>,
    /// Shape of the scale and zero-point tables.
    pub parameter_shape: Vec<i64>,
    /// Exact bit patterns of the scales.
    pub scale_bits: Vec<u64>,
    /// Exact bit patterns of the zero points.
    pub zero_point_bits: Vec<u64>,
}

/// User-facing construction data for a subchannel quantized type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubchannelQuantizedSpec {
    /// Integer storage type.
    pub storage: ElementType,
    /// Optional inclusive storage range.
    pub range: (Option<i128>, Option<i128>),
    /// Expressed floating-point type.
    pub expressed: ElementType,
    /// Quantized tensor dimensions.
    pub quantized_dimensions: Vec<i32>,
    /// Block size per quantized dimension.
    pub block_sizes: Vec<i64>,
    /// Shape of the parameter tables.
    pub parameter_shape: Vec<i64>,
    /// Scale bit patterns.
    pub scales: Vec<u64>,
    /// Zero-point bit patterns.
    pub zero_points: Vec<u64>,
}

impl SubchannelQuantizedType {
    /// Intern a subchannel quantized type from its storage, expressed type, and parameters.
    pub fn get(
        ctx: &Context,
        storage: TypeHandle,
        expressed: TypeHandle,
        parameters: SubchannelQuantizationParameters,
    ) -> TypedHandle<Self> {
        Self::instantiate(
            Self {
                storage,
                expressed,
                quantized_dimensions: parameters.quantized_dimensions,
                block_sizes: parameters.block_sizes,
                parameter_shape: parameters.parameter_shape,
                scale_bits: parameters.scale_bits,
                zero_point_bits: parameters.zero_point_bits,
            },
            ctx,
        )
    }
    /// Intern a normalized subchannel quantized type.
    pub fn intern(ctx: &Context, spec: SubchannelQuantizedSpec) -> TypeHandle {
        let SubchannelQuantizedSpec {
            storage,
            range,
            expressed,
            quantized_dimensions: axes,
            block_sizes: blocks,
            parameter_shape: shape,
            mut scales,
            mut zero_points,
        } = spec;
        let count = element_count(&shape);
        for values in [&mut scales, &mut zero_points] {
            if count == Some(values.len())
                && values
                    .first()
                    .is_some_and(|v| values.iter().all(|x| x == v))
            {
                values.truncate(1);
            }
        }
        Self::get(
            ctx,
            AnyQuantizedType::intern(ctx, storage, range.0, range.1, Some(expressed)),
            expressed.native(ctx),
            SubchannelQuantizationParameters {
                quantized_dimensions: axes,
                block_sizes: blocks,
                parameter_shape: shape,
                scale_bits: scales,
                zero_point_bits: zero_points,
            },
        )
        .into()
    }
    pub fn storage_type(&self, ctx: &Context) -> Option<ElementType> {
        self.storage
            .deref(ctx)
            .downcast_ref::<AnyQuantizedType>()?
            .storage_type(ctx)
    }
    pub fn expressed_type(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.expressed)
    }
    pub fn quantized_dimensions(&self) -> &[i32] {
        &self.quantized_dimensions
    }
    pub fn block_sizes(&self) -> &[i64] {
        &self.block_sizes
    }
    pub fn parameter_shape(&self) -> &[i64] {
        &self.parameter_shape
    }
    pub fn scales(&self) -> &[u64] {
        &self.scale_bits
    }
    pub fn zero_points(&self) -> &[u64] {
        &self.zero_point_bits
    }
}
impl Verify for SubchannelQuantizedType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let storage = self.storage.deref(ctx);
        let Some(storage) = storage.downcast_ref::<AnyQuantizedType>() else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "subchannel type",
                    reason: "subchannel storage metadata is not an any-quantized type"
                }
            );
        };
        storage.verify(ctx)?;
        if storage.expressed_type(ctx) != self.expressed_type(ctx)
            || self.quantized_dimensions.len() != self.block_sizes.len()
            || self.quantized_dimensions.iter().any(|&a| a < 0)
            || self.block_sizes.iter().any(|&b| b <= 0)
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "subchannel type",
                    reason: "invalid subchannel axis/block metadata"
                }
            );
        }
        let Some(count) = element_count(&self.parameter_shape) else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "subchannel type",
                    reason: "subchannel parameter shape overflows"
                }
            );
        };
        for (element, values) in [
            (self.expressed_type(ctx), &self.scale_bits),
            (self.storage_type(ctx), &self.zero_point_bits),
        ] {
            let Some(width) = element.and_then(ElementType::bit_width) else {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidType {
                        ty: "subchannel type",
                        reason: "invalid subchannel parameter element width"
                    }
                );
            };
            if (values.len() != count && !(count != 0 && values.len() == 1))
                || (width < 64 && values.iter().any(|v| v >> width != 0))
            {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidType {
                        ty: "subchannel type",
                        reason: "subchannel parameter table shape/bitwidth mismatch"
                    }
                );
            }
        }
        Ok(())
    }
}
