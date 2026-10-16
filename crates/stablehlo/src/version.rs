use std::{fmt, str::FromStr};

/// StableHLO source revision required by the reference conformance audits.
pub const STABLEHLO_SOURCE_COMMIT: &str = "25605c9f48a9aec305876d60d26b028398d298d2";
/// Current portable-artifact version defined by this source revision.
pub const STABLEHLO_CURRENT_VERSION: &str = "1.22.0";
/// Minimum portable-artifact version supported by this source revision.
pub const STABLEHLO_MINIMUM_VERSION: &str = "0.9.0";

/// A nonnegative, signed-int64-range StableHLO dialect version.
///
/// Ordering is lexicographic, as in the reference `vhlo::Version`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableHloVersion {
    major: u64,
    minor: u64,
    patch: u64,
}

/// Serialization compatibility windows at the pinned source revision.
///
/// The selected releases are pinned upstream policy, not dates computed from
/// the local clock. Choosing a version does not serialize or convert a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompatibilityRequirement {
    None,
    FourWeeks,
    TwelveWeeks,
    Maximum,
}

/// Invalid syntax, numeric range or an unsupported compatibility target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VersionError {
    InvalidSyntax,
    ComponentOutOfRange,
    UnsupportedTarget { version: StableHloVersion },
}

impl StableHloVersion {
    pub const CURRENT: Self = Self {
        major: 1,
        minor: 22,
        patch: 0,
    };
    pub const MINIMUM: Self = Self {
        major: 0,
        minor: 9,
        patch: 0,
    };

    /// Constructs a version within the reference's signed-int64 numeric range.
    pub const fn new(major: u64, minor: u64, patch: u64) -> Result<Self, VersionError> {
        if major > i64::MAX as u64 || minor > i64::MAX as u64 || patch > i64::MAX as u64 {
            return Err(VersionError::ComponentOutOfRange);
        }
        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    pub const fn major(self) -> u64 {
        self.major
    }
    pub const fn minor(self) -> u64 {
        self.minor
    }
    pub const fn patch(self) -> u64 {
        self.patch
    }

    pub const fn for_compatibility(requirement: CompatibilityRequirement) -> Self {
        match requirement {
            CompatibilityRequirement::None => Self::CURRENT,
            CompatibilityRequirement::FourWeeks => Self {
                major: 1,
                minor: 18,
                patch: 0,
            },
            CompatibilityRequirement::TwelveWeeks => Self {
                major: 1,
                minor: 16,
                patch: 3,
            },
            CompatibilityRequirement::Maximum => Self::MINIMUM,
        }
    }

    /// Returns the MLIR bytecode format selected by the pinned `Version.cpp`.
    ///
    /// This is version policy only; it does not implement MLIR bytecode.
    pub fn bytecode_version(self) -> Result<u64, VersionError> {
        self.validate_target()?;
        Ok(
            if self
                < (Self {
                    major: 0,
                    minor: 10,
                    patch: 0,
                })
            {
                0
            } else if self
                < (Self {
                    major: 0,
                    minor: 12,
                    patch: 0,
                })
            {
                1
            } else if self
                < (Self {
                    major: 0,
                    minor: 14,
                    patch: 0,
                })
            {
                3
            } else if self
                < (Self {
                    major: 0,
                    minor: 15,
                    patch: 0,
                })
            {
                4
            } else {
                6
            },
        )
    }

    /// Validates a requested version, then normalizes its opset patch to zero.
    ///
    /// `VhloToVersion.cpp::validateTargetVersion` checks the original version
    /// against the compatibility window before discarding its patch.
    pub fn target_opset(self) -> Result<Self, VersionError> {
        self.validate_target()?;
        Ok(Self { patch: 0, ..self })
    }

    fn validate_target(self) -> Result<(), VersionError> {
        if self < Self::MINIMUM || self > Self::CURRENT {
            return Err(VersionError::UnsupportedTarget { version: self });
        }
        Ok(())
    }
}

impl FromStr for StableHloVersion {
    type Err = VersionError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let mut parts = source.split('.');
        let mut values = [0; 3];
        for value in &mut values {
            let component = parts.next().ok_or(VersionError::InvalidSyntax)?;
            if component.is_empty() || !component.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(VersionError::InvalidSyntax);
            }
            // Leading zeroes are legal and do not consume the numeric range.
            let digits = component.trim_start_matches('0');
            *value = if digits.is_empty() {
                0
            } else {
                digits
                    .parse::<u64>()
                    .map_err(|_| VersionError::ComponentOutOfRange)?
            };
        }
        if parts.next().is_some() {
            return Err(VersionError::InvalidSyntax);
        }
        Self::new(values[0], values[1], values[2])
    }
}

impl fmt::Display for StableHloVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl fmt::Display for VersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSyntax => f.write_str(
                "StableHLO version must have the form major.minor.patch with decimal components",
            ),
            Self::ComponentOutOfRange => {
                f.write_str("StableHLO version component exceeds signed int64 range")
            }
            Self::UnsupportedTarget { version } => write!(
                f,
                "StableHLO target {version} is outside the supported {}..={} compatibility window",
                StableHloVersion::MINIMUM,
                StableHloVersion::CURRENT
            ),
        }
    }
}

impl std::error::Error for VersionError {}
