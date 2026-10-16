use super::{Parser, reject_fields};
use crate::{
    Error, Result,
    proto::xla_data::{
        ResultAccuracy,
        result_accuracy::{Mode, Specs, Tolerance},
    },
};
use protobuf::EnumFull;

impl Parser {
    pub(super) fn accuracy(&mut self) -> Result<ResultAccuracy> {
        self.expect("{")?;
        let name = self.word()?;
        self.expect("=")?;
        let specs = match name.as_str() {
            "mode" => Specs::Mode(
                match self.word()?.as_str() {
                    "default" => Mode::DEFAULT,
                    "highest" => Mode::HIGHEST,
                    _ => return Err(self.error("unknown result accuracy mode")),
                }
                .into(),
            ),
            "tolerance" => {
                self.expect("{")?;
                let mut tolerance = Tolerance::default();
                if !self.take("}") {
                    loop {
                        let name = self.word()?;
                        self.expect("=")?;
                        let value = self.number()?;
                        match name.as_str() {
                            "atol" => tolerance.atol = value,
                            "rtol" => tolerance.rtol = value,
                            "ulps" => {
                                if !value.is_finite()
                                    || value < i64::MIN as f64
                                    || value >= -(i64::MIN as f64)
                                {
                                    return Err(
                                        self.error("ULP tolerance exceeds i64 conversion range")
                                    );
                                }
                                tolerance.ulps = value as i64;
                            }
                            _ => return Err(self.error("unknown result accuracy tolerance")),
                        }
                        if self.take("}") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                Specs::Tolerance(tolerance)
            }
            _ => return Err(self.error("expected mode or tolerance result accuracy specification")),
        };
        self.expect("}")?;
        let wire = ResultAccuracy {
            specs: Some(specs),
            ..Default::default()
        };
        crate::accuracy::validate(&wire)?;
        Ok(wire)
    }
}

pub(super) fn print(wire: &ResultAccuracy) -> Result<String> {
    crate::accuracy::validate(wire)?;
    reject_fields(wire, &["mode", "tolerance"])?;
    Ok(match wire.specs.as_ref() {
        Some(Specs::Mode(mode)) => format!(
            "{{mode={}}}",
            mode.enum_value()
                .map_err(|_| Error::new("result-accuracy", "unknown mode"))?
                .descriptor()
                .name()
                .to_ascii_lowercase()
        ),
        Some(Specs::Tolerance(tolerance)) => {
            reject_fields(tolerance, &["atol", "rtol", "ulps"])?;
            if (tolerance.ulps as f64) as i128 != i128::from(tolerance.ulps) {
                return Err(Error::new(
                    "unsupported",
                    "ULP integer cannot round-trip through upstream's double-valued text parser",
                ));
            }
            format!(
                "{{tolerance={{atol={},rtol={},ulps={}}}}}",
                super::numbers::print(
                    u128::from(tolerance.atol.to_bits()),
                    crate::PrimitiveType::F64
                )?,
                super::numbers::print(
                    u128::from(tolerance.rtol.to_bits()),
                    crate::PrimitiveType::F64
                )?,
                tolerance.ulps
            )
        }
        None => {
            return Err(Error::new(
                "unsupported",
                "empty result accuracy has no explicit text specification",
            ));
        }
    })
}
