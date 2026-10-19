/// StableHLO enum spellings.
pub fn values(name: &str) -> Option<&'static [&'static str]> {
    Some(match name {
        "precision" => &["DEFAULT", "HIGH", "HIGHEST"],
        "result_accuracy_mode" => &["DEFAULT", "HIGHEST", "TOLERANCE"],
        "fft_type" => &["FFT", "IFFT", "RFFT", "IRFFT"],
        "comparison_direction" => &["EQ", "NE", "GE", "GT", "LE", "LT"],
        "comparison_type" => &[
            "NOTYPE",
            "FLOAT",
            "TOTALORDER",
            "SIGNED",
            "UNSIGNED",
            "WEAKORDER",
        ],
        "transpose" => &["TRANSPOSE_INVALID", "NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"],
        "rng_distribution" => &["UNIFORM", "NORMAL"],
        "rng_algorithm" => &["DEFAULT", "THREE_FRY", "PHILOX"],
        _ => return None,
    })
}
