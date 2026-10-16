//! Portable-artifact version discovery. This does not deserialize the IR body.

use std::fmt;

use crate::{StableHloVersion, VersionError};

/// Failure to extract the version from a StableHLO portable-artifact header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactVersionError {
    HeaderTooShort,
    MissingProducerTerminator,
    InvalidMagic,
    MissingStableHloProducer,
    InvalidVersion(VersionError),
}

/// Extracts the producer's StableHLO version, following `Serialization.cpp`.
///
/// This only inspects the header, not bytecode sections, VHLO operations or the
/// compatibility window. A successful result does not establish that an artifact
/// is valid, supported or safe to deserialize.
pub fn portable_artifact_version(source: &[u8]) -> Result<StableHloVersion, ArtifactVersionError> {
    if source.len() < 19 {
        return Err(ArtifactVersionError::HeaderTooShort);
    }
    let end = source
        .iter()
        .position(|&byte| byte == 0)
        .ok_or(ArtifactVersionError::MissingProducerTerminator)?;
    let header = &source[..end];
    if !header.starts_with(b"ML\xefR") {
        return Err(ArtifactVersionError::InvalidMagic);
    }
    let prefix = b"StableHLO_v";
    let start = header
        .windows(prefix.len())
        .position(|bytes| bytes == prefix)
        .ok_or(ArtifactVersionError::MissingStableHloProducer)?
        + prefix.len();
    let version = std::str::from_utf8(&header[start..])
        .map_err(|_| ArtifactVersionError::InvalidVersion(VersionError::InvalidSyntax))?;
    version
        .parse()
        .map_err(ArtifactVersionError::InvalidVersion)
}

impl fmt::Display for ArtifactVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderTooShort => {
                f.write_str("portable-artifact header is shorter than 19 bytes")
            }
            Self::MissingProducerTerminator => {
                f.write_str("portable-artifact producer is not null-terminated")
            }
            Self::InvalidMagic => {
                f.write_str("portable artifact does not start with MLIR bytecode magic")
            }
            Self::MissingStableHloProducer => {
                f.write_str("bytecode producer does not identify a StableHLO version")
            }
            Self::InvalidVersion(error) => write!(f, "invalid portable-artifact version: {error}"),
        }
    }
}

impl std::error::Error for ArtifactVersionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidVersion(error) => Some(error),
            _ => None,
        }
    }
}
