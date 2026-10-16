use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

/// A recoverable error, with a stable category and contextual explanation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub category: &'static str,
    pub message: String,
    pub offset: Option<usize>,
}

impl Error {
    pub fn new(category: &'static str, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
            offset: None,
        }
    }

    pub(crate) fn at(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    pub(crate) fn context(mut self, context: impl fmt::Display) -> Self {
        self.message = format!("{context}: {}", self.message);
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.category, self.message)?;
        if let Some(offset) = self.offset {
            write!(f, " at byte {offset}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}
