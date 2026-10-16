use crate::{Error, Result, proto::xla_data::ResultAccuracy};

pub(crate) fn validate(accuracy: &ResultAccuracy) -> Result<()> {
    if let Some(crate::proto::xla_data::result_accuracy::Specs::Mode(mode)) = &accuracy.specs {
        mode.enum_value()
            .map_err(|_| Error::new("result-accuracy", "unknown accuracy mode"))?;
    }
    let tolerance = accuracy.tolerance();
    if !(tolerance.rtol >= 0.0 && tolerance.atol >= 0.0 && tolerance.ulps >= 0) {
        return Err(Error::new(
            "result-accuracy",
            "accuracy tolerances must be nonnegative",
        ));
    }
    Ok(())
}
