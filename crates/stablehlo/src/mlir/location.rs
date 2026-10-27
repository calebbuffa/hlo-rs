use super::{TextError, TextErrorKind, cursor::Cursor};
use crate::pliron::{
    combine::stream::position::SourcePosition,
    context::Context,
    location::{Location, Source},
};

pub(super) fn validate(ctx: &Context, location: &Location) -> crate::pliron::result::Result<()> {
    let mut pending = vec![location];
    while let Some(location) = pending.pop() {
        match location {
            Location::SrcPos { src, pos } => {
                if pos.line < 0 || pos.column < 0 {
                    return crate::pliron::input_err_noloc!(TextError {
                        offset: 0,
                        kind: TextErrorKind::InvalidIr("negative MLIR location coordinate".into())
                    });
                }
                if let Source::File(path) = src
                    && crate::pliron::uniqued_any::get(ctx, *path)
                        .to_str()
                        .is_none()
                {
                    return crate::pliron::input_err_noloc!(TextError {
                        offset: 0,
                        kind: TextErrorKind::InvalidIr("MLIR location path is not UTF-8".into())
                    });
                }
            }
            Location::Fused { locations, .. } => pending.extend(locations),
            Location::Named { child_loc, .. } => pending.push(child_loc),
            Location::CallSite { callee, caller } => pending.extend([&**callee, &**caller]),
            Location::Unknown => {}
        }
    }
    Ok(())
}

impl Cursor<'_> {
    pub(super) fn optional_location(&mut self, ctx: &Context) -> Result<Location, TextError> {
        if !self.keyword("loc") {
            return Ok(Location::Unknown);
        }
        self.expect("(")?;
        let location = self.location(ctx, 0)?;
        self.expect(")")?;
        Ok(location)
    }

    fn location(&mut self, ctx: &Context, depth: usize) -> Result<Location, TextError> {
        self.depth(depth)?;
        if self.keyword("unknown") {
            return Ok(Location::Unknown);
        }
        if self.keyword("callsite") {
            self.expect("(")?;
            let callee = Box::new(self.location(ctx, depth + 1)?);
            if !self.keyword("at") {
                return Err(self.error(TextErrorKind::Expected("at")));
            }
            let caller = Box::new(self.location(ctx, depth + 1)?);
            self.expect(")")?;
            return Ok(Location::CallSite { callee, caller });
        }
        if self.keyword("fused") {
            let metadata = if self.take("<") {
                let attribute = self.attribute(ctx, depth + 1)?;
                self.expect(">")?;
                Some(attribute)
            } else {
                None
            };
            self.expect("[")?;
            let mut locations = Vec::new();
            if !self.take("]") {
                loop {
                    locations.push(self.location(ctx, depth + 1)?);
                    if self.take("]") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            return Ok(Location::Fused {
                metadata,
                locations,
            });
        }
        let name = String::from_utf8(self.string_bytes()?)
            .map_err(|_| self.error(TextErrorKind::Unsupported("non-UTF8 location name")))?;
        if self.take(":") {
            let line = self.location_coordinate()?;
            self.expect(":")?;
            let column = self.location_coordinate()?;
            return Ok(Location::SrcPos {
                src: Source::new_from_file(ctx, name),
                pos: SourcePosition { line, column },
            });
        }
        let child_loc = if self.take("(") {
            let child = self.location(ctx, depth + 1)?;
            self.expect(")")?;
            child
        } else {
            Location::Unknown
        };
        Ok(Location::Named {
            name,
            child_loc: Box::new(child_loc),
        })
    }

    fn location_coordinate(&mut self) -> Result<i32, TextError> {
        let Some(value) = self.dimension()? else {
            return Err(self.error(TextErrorKind::Expected("location coordinate")));
        };
        i32::try_from(value)
            .ok()
            .filter(|&value| value >= 0)
            .ok_or_else(|| self.error(TextErrorKind::InvalidNumber))
    }
}
