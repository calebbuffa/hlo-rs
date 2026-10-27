use super::{TextError, TextErrorKind};

pub(super) struct Cursor<'a> {
    pub(super) input: &'a str,
    pub(super) offset: usize,
}

impl<'a> Cursor<'a> {
    pub(super) fn depth(&self, depth: usize) -> Result<(), TextError> {
        if depth >= 128 {
            Err(self.error(TextErrorKind::NestingLimit))
        } else {
            Ok(())
        }
    }
    pub(super) fn new(input: &'a str) -> Self {
        Self { input, offset: 0 }
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
                .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0))
            {
                self.offset += 1;
            }
            if !self.input[self.offset..].starts_with("//") {
                break;
            }
            self.offset = self.input[self.offset..]
                .find(['\r', '\n'])
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

    pub(super) fn keyword(&mut self, keyword: &str) -> bool {
        self.space();
        let tail = &self.input[self.offset..];
        if !tail.starts_with(keyword)
            || tail.as_bytes().get(keyword.len()).is_some_and(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$' | b'.')
            })
        {
            return false;
        }
        self.offset += keyword.len();
        true
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
        Ok(&self.input[start..self.offset])
    }

    pub(super) fn integer(&mut self) -> Result<i128, TextError> {
        if self.keyword("true") {
            return Ok(-1);
        }
        if self.keyword("false") {
            return Ok(0);
        }
        let negative = self.take("-");
        self.space();
        let hexadecimal = self.input[self.offset..].starts_with("0x");
        if hexadecimal {
            self.offset += 2;
        }
        let start = self.offset;
        while self.input.as_bytes().get(self.offset).is_some_and(|byte| {
            if hexadecimal {
                byte.is_ascii_hexdigit()
            } else {
                byte.is_ascii_digit()
            }
        }) {
            self.offset += 1;
        }
        if start == self.offset {
            return Err(self.error(TextErrorKind::InvalidNumber));
        }
        let magnitude = u128::from_str_radix(
            &self.input[start..self.offset],
            if hexadecimal { 16 } else { 10 },
        )
        .map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
        if negative && magnitude == 1_u128 << 127 {
            Ok(i128::MIN)
        } else {
            let magnitude =
                i128::try_from(magnitude).map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
            Ok(if negative { -magnitude } else { magnitude })
        }
    }

    pub(super) fn dimension(&mut self) -> Result<Option<i64>, TextError> {
        self.space();
        if self.take("?") {
            return Ok(Some(-1));
        }
        let start = self.offset;
        while self
            .input
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_digit)
        {
            self.offset += 1;
        }
        if start == self.offset {
            return Ok(None);
        }
        self.input[start..self.offset]
            .parse()
            .map(Some)
            .map_err(|_| self.error(TextErrorKind::InvalidNumber))
    }

    pub(super) fn starts(&mut self, text: &str) -> bool {
        self.space();
        self.input[self.offset..].starts_with(text)
    }

    pub(super) fn number(&mut self) -> Result<String, TextError> {
        let negative = self.take("-");
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
        let magnitude = &self.input[start..self.offset];
        Ok(if negative {
            format!("-{magnitude}")
        } else {
            magnitude.to_owned()
        })
    }

    pub(super) fn string_bytes(&mut self) -> Result<Vec<u8>, TextError> {
        self.expect("\"")?;
        let mut bytes = Vec::new();
        loop {
            let Some(&byte) = self.input.as_bytes().get(self.offset) else {
                return Err(self.error(TextErrorKind::Expected("closing quote")));
            };
            self.offset += 1;
            match byte {
                b'"' => return Ok(bytes),
                b'\n' | b'\r' | 0 => {
                    return Err(self.error(TextErrorKind::Expected("string character")));
                }
                b'\\' => {
                    let Some(&escape) = self.input.as_bytes().get(self.offset) else {
                        return Err(self.error(TextErrorKind::Expected("escape sequence")));
                    };
                    self.offset += 1;
                    bytes.push(match escape {
                        b'n' => b'\n',
                        b't' => b'\t',
                        b'\\' => b'\\',
                        b'"' => b'"',
                        first if first.is_ascii_hexdigit() => {
                            let Some(&second) = self.input.as_bytes().get(self.offset) else {
                                return Err(self.error(TextErrorKind::Expected("two hex digits")));
                            };
                            if !second.is_ascii_hexdigit() {
                                return Err(self.error(TextErrorKind::Expected("two hex digits")));
                            }
                            self.offset += 1;
                            let hex = |byte: u8| {
                                if byte.is_ascii_digit() {
                                    byte - b'0'
                                } else {
                                    byte.to_ascii_lowercase() - b'a' + 10
                                }
                            };
                            hex(first) * 16 + hex(second)
                        }
                        _ => return Err(self.error(TextErrorKind::Expected("MLIR string escape"))),
                    });
                }
                _ => bytes.push(byte),
            }
        }
    }

    pub(super) fn symbol(&mut self) -> Result<String, TextError> {
        if self.starts("\"") {
            let bytes = self.string_bytes()?;
            String::from_utf8(bytes)
                .map_err(|_| self.error(TextErrorKind::Expected("UTF-8 identifier")))
        } else {
            Ok(self.word()?.to_owned())
        }
    }
    pub(super) fn finish(&mut self) -> Result<(), TextError> {
        self.space();
        if self.offset == self.input.len() {
            Ok(())
        } else {
            Err(self.error(TextErrorKind::TrailingInput))
        }
    }
}
