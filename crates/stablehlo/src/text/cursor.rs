use super::{TextError, TextErrorKind};

pub(super) struct Cursor<'a> {
    pub(super) input: &'a str,
    pub(super) offset: usize,
    pub(super) aliases: super::aliases::Aliases,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(input: &'a str) -> Self {
        Self {
            input,
            offset: 0,
            aliases: Default::default(),
        }
    }

    pub(super) fn position(&self) -> usize {
        self.offset
    }

    pub(super) fn error(&self, kind: TextErrorKind) -> TextError {
        TextError {
            offset: self.offset,
            kind,
        }
    }

    pub(super) fn space(&mut self) {
        loop {
            while self
                .input
                .as_bytes()
                .get(self.offset)
                .is_some_and(u8::is_ascii_whitespace)
            {
                self.offset += 1;
            }
            if !self.input[self.offset..].starts_with("//") {
                break;
            }
            self.offset = self.input[self.offset..]
                .find('\n')
                .map_or(self.input.len(), |index| self.offset + index + 1);
        }
    }

    pub(super) fn take(&mut self, text: &str) -> bool {
        self.space();
        if self.input[self.offset..].starts_with(text) {
            self.offset += text.len();
            true
        } else {
            false
        }
    }

    pub(super) fn expect(&mut self, text: &'static str) -> Result<(), TextError> {
        if self.take(text) {
            Ok(())
        } else {
            Err(self.error(TextErrorKind::Expected(text)))
        }
    }

    pub(super) fn starts(&mut self, text: &str) -> bool {
        self.space();
        self.input[self.offset..].starts_with(text)
    }

    pub(super) fn word(&mut self) -> Result<&'a str, TextError> {
        self.space();
        let start = self.offset;
        if !self
            .input
            .as_bytes()
            .get(start)
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        {
            return Err(self.error(TextErrorKind::Expected("identifier")));
        }
        while self
            .input
            .as_bytes()
            .get(self.offset)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'$'))
        {
            self.offset += 1;
        }
        if start == self.offset {
            Err(self.error(TextErrorKind::Expected("identifier")))
        } else {
            Ok(&self.input[start..self.offset])
        }
    }

    pub(super) fn integer(&mut self) -> Result<i128, TextError> {
        self.space();
        let start = self.offset;
        if matches!(self.input.as_bytes().get(self.offset), Some(b'-' | b'+')) {
            self.offset += 1;
        }
        let hexadecimal = self.input[self.offset..].starts_with("0x");
        if hexadecimal {
            self.offset += 2;
        }
        let digits = self.offset;
        while self.input.as_bytes().get(self.offset).is_some_and(|byte| {
            if hexadecimal {
                byte.is_ascii_hexdigit()
            } else {
                byte.is_ascii_digit()
            }
        }) {
            self.offset += 1;
        }
        if digits == self.offset {
            return Err(self.error(TextErrorKind::InvalidNumber));
        }
        if hexadecimal {
            let magnitude = i128::from_str_radix(&self.input[digits..self.offset], 16)
                .map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
            Ok(if self.input.as_bytes()[start] == b'-' {
                -magnitude
            } else {
                magnitude
            })
        } else {
            self.input[start..self.offset]
                .parse()
                .map_err(|_| self.error(TextErrorKind::InvalidNumber))
        }
    }

    pub(super) fn i64(&mut self) -> Result<i64, TextError> {
        i64::try_from(self.integer()?).map_err(|_| self.error(TextErrorKind::InvalidNumber))
    }

    pub(super) fn float(&mut self) -> Result<f64, TextError> {
        let literal = self.number_token()?;
        if let Some(hex) = literal.strip_prefix("0x") {
            return u64::from_str_radix(hex, 16)
                .map(f64::from_bits)
                .map_err(|_| self.error(TextErrorKind::InvalidNumber));
        }
        literal
            .parse()
            .map_err(|_| self.error(TextErrorKind::InvalidNumber))
    }

    pub(super) fn number_token(&mut self) -> Result<&'a str, TextError> {
        self.space();
        let start = self.offset;
        while self
            .input
            .as_bytes()
            .get(self.offset)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'+' | b'-'))
        {
            self.offset += 1;
        }
        if start == self.offset {
            return Err(self.error(TextErrorKind::InvalidNumber));
        }
        Ok(&self.input[start..self.offset])
    }

    pub(super) fn finish(&mut self) -> Result<(), TextError> {
        self.space();
        if self.offset == self.input.len() {
            Ok(())
        } else {
            Err(self.error(TextErrorKind::TrailingInput))
        }
    }

    pub(super) fn depth(&self, depth: usize) -> Result<(), TextError> {
        if depth >= 128 {
            Err(self.error(TextErrorKind::NestingLimit))
        } else {
            Ok(())
        }
    }
}
