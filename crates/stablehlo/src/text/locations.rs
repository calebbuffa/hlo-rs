use super::{TextError, TextErrorKind, attributes, cursor::Cursor};
use crate::{Attribute, Location};

#[derive(Clone)]
pub(super) enum ParsedLocation {
    Direct(Box<Location>),
    Alias { offset: usize, depth: usize },
}

impl ParsedLocation {
    pub(super) fn resolve(self, cursor: &mut Cursor<'_>) -> Result<Location, TextError> {
        match self {
            Self::Direct(location) => Ok(*location),
            Self::Alias { offset, depth } => {
                let saved = cursor.offset;
                cursor.offset = offset;
                let result = cursor.attribute_alias(depth);
                cursor.offset = saved;
                match result? {
                    Attribute::Location(location) => Ok(*location),
                    _ => Err(TextError {
                        offset,
                        kind: TextErrorKind::Expected("location attribute alias"),
                    }),
                }
            }
        }
    }
}

impl Cursor<'_> {
    pub(super) fn trailing_location(
        &mut self,
        depth: usize,
    ) -> Result<Option<ParsedLocation>, TextError> {
        if !self.location_keyword("loc") {
            return Ok(None);
        }
        self.expect("(")?;
        let location = if self.starts("#") {
            self.space();
            let offset = self.position();
            self.expect("#")?;
            if self.alias_name()?.contains('.') {
                return Err(self.error(TextErrorKind::Expected("location alias name")));
            }
            ParsedLocation::Alias { offset, depth }
        } else {
            ParsedLocation::Direct(Box::new(self.location_instance(depth + 1)?))
        };
        self.expect(")")?;
        Ok(Some(location))
    }

    pub(super) fn location(&mut self, depth: usize) -> Result<Location, TextError> {
        self.depth(depth)?;
        self.expect("loc")?;
        self.expect("(")?;
        let location = self.location_instance(depth + 1)?;
        self.expect(")")?;
        Ok(location)
    }

    fn location_instance(&mut self, depth: usize) -> Result<Location, TextError> {
        enum Frame {
            Name(Vec<u8>),
            CallSiteCallee,
            CallSiteCaller(Box<Location>),
            Fused {
                locations: Vec<Location>,
                metadata: Option<Box<Attribute>>,
            },
        }

        let mut frames = Vec::new();
        let mut value = None;
        loop {
            if let Some(location) = value.take() {
                let Some(frame) = frames.pop() else {
                    return Ok(location);
                };
                match frame {
                    Frame::Name(name) => {
                        self.expect(")")?;
                        value = Some(Location::Name {
                            name,
                            child: Some(Box::new(location)),
                        });
                    }
                    Frame::CallSiteCallee => {
                        if !self.location_keyword("at") {
                            return Err(self.error(TextErrorKind::Expected("at")));
                        }
                        frames.push(Frame::CallSiteCaller(Box::new(location)));
                        self.depth(depth + frames.len())?;
                    }
                    Frame::CallSiteCaller(callee) => {
                        self.expect(")")?;
                        value = Some(Location::CallSite {
                            callee,
                            caller: Box::new(location),
                        });
                    }
                    Frame::Fused {
                        mut locations,
                        metadata,
                    } => {
                        locations.push(location);
                        if self.take("]") {
                            value = Some(Location::Fused {
                                locations,
                                metadata,
                            });
                        } else {
                            self.expect(",")?;
                            frames.push(Frame::Fused {
                                locations,
                                metadata,
                            });
                            self.depth(depth + frames.len())?;
                        }
                    }
                }
                continue;
            }

            let current_depth = depth + frames.len();
            self.depth(current_depth)?;
            if self.starts("#") {
                value = Some(match self.attribute_alias(current_depth)? {
                    Attribute::Location(location) => *location,
                    _ => {
                        return Err(self.error(TextErrorKind::Expected("location attribute alias")));
                    }
                });
            } else if self.starts("\"") {
                let name = self.string_bytes()?;
                if self.take(":") {
                    let line = self.location_number()?;
                    if !self.take(":") {
                        value = Some(Location::FileLine { file: name, line });
                    } else {
                        let column = self.location_number()?;
                        if !self.location_keyword("to") {
                            value = Some(Location::FileLineCol {
                                file: name,
                                line,
                                column,
                            });
                        } else {
                            let end_line = if self.starts(":") {
                                None
                            } else {
                                Some(self.location_number()?)
                            };
                            self.expect(":")?;
                            value = Some(Location::FileRange {
                                file: name,
                                start_line: line,
                                start_column: column,
                                end_line,
                                end_column: self.location_number()?,
                            });
                        }
                    }
                } else if self.take("(") {
                    frames.push(Frame::Name(name));
                    self.depth(depth + frames.len())?;
                } else {
                    value = Some(Location::Name { name, child: None });
                }
            } else if self.location_keyword("unknown") {
                value = Some(Location::Unknown);
            } else if self.location_keyword("callsite") {
                self.expect("(")?;
                frames.push(Frame::CallSiteCallee);
                self.depth(depth + frames.len())?;
            } else if self.location_keyword("fused") {
                let metadata = if self.take("<") {
                    let metadata = self.attribute(current_depth + 1)?;
                    self.expect(">")?;
                    Some(Box::new(metadata))
                } else {
                    None
                };
                self.expect("[")?;
                if self.take("]") {
                    value = Some(Location::Fused {
                        locations: Vec::new(),
                        metadata,
                    });
                } else {
                    frames.push(Frame::Fused {
                        locations: Vec::new(),
                        metadata,
                    });
                    self.depth(depth + frames.len())?;
                }
            } else {
                return Err(self.error(TextErrorKind::Expected("MLIR location instance")));
            }
        }
    }

    fn location_number(&mut self) -> Result<u32, TextError> {
        self.space();
        if !self
            .input
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_digit)
        {
            return Err(self.error(TextErrorKind::InvalidNumber));
        }
        u32::try_from(self.integer()?).map_err(|_| self.error(TextErrorKind::InvalidNumber))
    }

    fn location_keyword(&mut self, keyword: &str) -> bool {
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
}

pub(super) fn print(location: &Location, depth: usize) -> Result<String, TextError> {
    Ok(format!("loc({})", print_instance(location, depth + 1)?))
}

fn print_instance(location: &Location, depth: usize) -> Result<String, TextError> {
    enum Event<'a> {
        Location(&'a Location, usize),
        Text(String),
    }

    let mut events = vec![Event::Location(location, depth)];
    let mut output = String::new();
    while let Some(event) = events.pop() {
        match event {
            Event::Text(text) => output.push_str(&text),
            Event::Location(location, depth) => {
                if depth >= 128 {
                    return Err(TextError {
                        offset: 0,
                        kind: TextErrorKind::NestingLimit,
                    });
                }
                match location {
                    Location::Unknown => output.push_str("unknown"),
                    Location::FileLine { file, line } => {
                        output.push_str(&format!("{}:{line}", attributes::quote_bytes(file)));
                    }
                    Location::FileLineCol { file, line, column } => {
                        output.push_str(&format!(
                            "{}:{line}:{column}",
                            attributes::quote_bytes(file)
                        ));
                    }
                    Location::FileRange {
                        file,
                        start_line,
                        start_column,
                        end_line,
                        end_column,
                    } => {
                        output.push_str(&format!(
                            "{}:{start_line}:{start_column} to {}:{end_column}",
                            attributes::quote_bytes(file),
                            end_line.map_or_else(String::new, |line| line.to_string()),
                        ));
                    }
                    Location::Name { name, child } => {
                        if let Some(child) = child {
                            events.push(Event::Text(")".into()));
                            events.push(Event::Location(child, depth + 1));
                            events.push(Event::Text("(".into()));
                        }
                        events.push(Event::Text(attributes::quote_bytes(name)));
                    }
                    Location::CallSite { callee, caller } => {
                        events.push(Event::Text(")".into()));
                        events.push(Event::Location(caller, depth + 1));
                        events.push(Event::Text(" at ".into()));
                        events.push(Event::Location(callee, depth + 1));
                        events.push(Event::Text("callsite(".into()));
                    }
                    Location::Fused {
                        locations,
                        metadata,
                    } => {
                        let metadata = metadata
                            .as_ref()
                            .map(|metadata| {
                                attributes::print(metadata, depth + 1)
                                    .map(|text| format!("<{text}>"))
                            })
                            .transpose()?
                            .unwrap_or_default();
                        events.push(Event::Text("]".into()));
                        for (index, child) in locations.iter().enumerate().rev() {
                            events.push(Event::Location(child, depth + 1));
                            if index > 0 {
                                events.push(Event::Text(", ".into()));
                            }
                        }
                        events.push(Event::Text(format!("fused{metadata}[")));
                    }
                }
            }
        }
    }
    Ok(output)
}
