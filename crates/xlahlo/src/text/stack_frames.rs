use super::*;
use crate::proto::hlo::{
    StackFrameIndexProto,
    stack_frame_index_proto::{FileLocation, StackFrame},
};
use std::sync::atomic::{AtomicUsize, Ordering};

static WARNINGS: AtomicUsize = AtomicUsize::new(0);

fn int32(parser: &mut Parser) -> Result<i32> {
    i32::try_from(parser.integer()?).map_err(|_| parser.error("stack-frame integer exceeds i32"))
}

fn attributes(parser: &mut Parser, required: &[&str]) -> Result<BTreeMap<String, i32>> {
    parser.expect("{")?;
    let mut values = BTreeMap::new();
    while !parser.take("}") {
        parser.take(",");
        let key = parser.word()?;
        if !required.contains(&key.as_str()) || values.contains_key(&key) {
            return Err(parser.error("unknown or duplicate stack-frame attribute"));
        }
        parser.expect("=")?;
        values.insert(key, int32(parser)?);
    }
    if values.len() != required.len() {
        return Err(parser.error("missing stack-frame attribute"));
    }
    Ok(values)
}

pub(super) fn parse(parser: &mut Parser) -> Result<Option<StackFrameIndexProto>> {
    if !parser.take("FileNames") {
        return Ok(None);
    }
    let mut index = StackFrameIndexProto::new();
    while parser.peek().parse::<i64>().is_ok() {
        parser.integer()?;
        index.file_names.push(
            String::from_utf8(parser.byte_string()?)
                .map_err(|_| parser.error("file name must be valid UTF-8"))?,
        );
    }
    parser.expect("FunctionNames")?;
    while parser.peek().parse::<i64>().is_ok() {
        parser.integer()?;
        index.function_names.push(
            String::from_utf8(parser.byte_string()?)
                .map_err(|_| parser.error("function name must be valid UTF-8"))?,
        );
    }
    parser.expect("FileLocations")?;
    while parser.peek().parse::<i64>().is_ok() {
        parser.integer()?;
        let fields = attributes(
            parser,
            &[
                "file_name_id",
                "function_name_id",
                "line",
                "end_line",
                "column",
                "end_column",
            ],
        )?;
        index.file_locations.push(FileLocation {
            file_name_id: fields["file_name_id"],
            function_name_id: fields["function_name_id"],
            line: fields["line"],
            end_line: fields["end_line"],
            column: fields["column"],
            end_column: fields["end_column"],
            ..Default::default()
        });
    }
    parser.expect("StackFrames")?;
    while parser.peek().parse::<i64>().is_ok() {
        parser.integer()?;
        let fields = attributes(parser, &["file_location_id", "parent_frame_id"])?;
        index.stack_frames.push(StackFrame {
            file_location_id: fields["file_location_id"],
            parent_frame_id: fields["parent_frame_id"]
                .checked_sub(1)
                .ok_or_else(|| parser.error("stack-frame parent subtraction overflows i32"))?,
            ..Default::default()
        });
    }
    Ok(Some(index))
}

fn at<T>(values: &[T], id: i32) -> Option<&T> {
    usize::try_from(id)
        .ok()?
        .checked_sub(1)
        .and_then(|i| values.get(i))
}

fn next_id(length: usize) -> Result<i32> {
    i32::try_from(length)
        .ok()
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| Error::new("resource-limit", "stack-frame index exceeds i32"))
}

fn intern_name(
    names: &mut Vec<String>,
    ids: &mut BTreeMap<String, i32>,
    name: &str,
) -> Result<i32> {
    if let Some(&id) = ids.get(name) {
        return Ok(id);
    }
    let id = next_id(names.len())?;
    ids.insert(name.into(), id);
    names.push(name.into());
    Ok(id)
}

#[derive(Default)]
struct Interner {
    index: StackFrameIndexProto,
    files: BTreeMap<String, i32>,
    functions: BTreeMap<String, i32>,
    locations: BTreeMap<(i32, i32, i32, i32, i32, i32), i32>,
    frames: BTreeMap<(i32, i32), i32>,
}

impl Interner {
    fn add(
        &mut self,
        file: &str,
        function: &str,
        location: &FileLocation,
        parent: i32,
    ) -> Result<i32> {
        let file = intern_name(&mut self.index.file_names, &mut self.files, file)?;
        let function = intern_name(
            &mut self.index.function_names,
            &mut self.functions,
            function,
        )?;
        let key = (
            file,
            function,
            location.line,
            location.column,
            location.end_line,
            location.end_column,
        );
        let location_id = if let Some(&id) = self.locations.get(&key) {
            id
        } else {
            let id = next_id(self.index.file_locations.len())?;
            self.locations.insert(key, id);
            self.index.file_locations.push(FileLocation {
                file_name_id: file,
                function_name_id: function,
                line: location.line,
                column: location.column,
                end_line: location.end_line,
                end_column: location.end_column,
                ..Default::default()
            });
            id
        };
        let key = (location_id, parent);
        if let Some(&id) = self.frames.get(&key) {
            return Ok(id);
        }
        let id = next_id(self.index.stack_frames.len())?;
        self.frames.insert(key, id);
        self.index.stack_frames.push(StackFrame {
            file_location_id: location_id,
            parent_frame_id: parent,
            ..Default::default()
        });
        Ok(id)
    }
}

pub(super) fn normalize(module: &mut HloModuleProto) -> Result<Vec<String>> {
    let source = module.stack_frame_index.take().unwrap_or_default();
    let mut output = Interner::default();
    let mut mapping = BTreeMap::new();
    // The pinned canonicalizer retains pending descendants after location errors.
    let mut path = Vec::new();
    let mut warnings = Vec::new();
    for instruction in module
        .computations
        .iter_mut()
        .flat_map(|c| &mut c.instructions)
    {
        let Some(metadata) = instruction.metadata.as_mut() else {
            continue;
        };
        let old = metadata.stack_frame_id;
        if old == 0 {
            continue;
        }
        let mut current = old;
        let mut visited = BTreeSet::new();
        let mut valid = true;
        while current != 0 {
            let Some(frame) = at(&source.stack_frames, current) else {
                warnings.push(format!("invalid stack_frame_id {current}"));
                valid = false;
                break;
            };
            if !visited.insert(current) {
                warnings.push(format!("cycle in stack_frame_id {current}"));
                valid = false;
                break;
            }
            if mapping.contains_key(&current) {
                break;
            }
            path.push(current);
            current = frame.parent_frame_id;
        }
        if !valid {
            path.clear();
            metadata.stack_frame_id = 0;
            continue;
        }
        let mut parent = if current == 0 {
            0
        } else {
            *mapping
                .get(&current)
                .ok_or_else(|| Error::new("stack-frame", "canonical parent is missing"))?
        };
        while let Some(old) = path.pop() {
            let frame = at(&source.stack_frames, old)
                .ok_or_else(|| Error::new("stack-frame", "collected frame is missing"))?;
            let Some(location) = at(&source.file_locations, frame.file_location_id) else {
                warnings.push(format!(
                    "invalid file_location_id {}",
                    frame.file_location_id
                ));
                parent = 0;
                break;
            };
            let (Some(file), Some(function)) = (
                at(&source.file_names, location.file_name_id),
                at(&source.function_names, location.function_name_id),
            ) else {
                warnings.push("invalid IDs in file_location".into());
                parent = 0;
                break;
            };
            parent = output.add(file, function, location, parent)?;
            mapping.insert(old, parent);
        }
        metadata.stack_frame_id = parent;
    }
    if !output.index.stack_frames.is_empty() {
        module.stack_frame_index = Some(output.index).into();
    }
    Ok(warnings)
}

pub(super) fn warn(warnings: Vec<String>) {
    for warning in warnings {
        if WARNINGS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < 10).then_some(n + 1)
            })
            .is_ok()
        {
            eprintln!("xlahlo: warning: {warning}");
        }
    }
}

pub(super) fn print(module: &HloModuleProto, text: &mut String) -> Result<()> {
    if module.stack_frame_index.is_none()
        && module
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .all(|i| i.metadata.as_ref().is_none_or(|m| m.stack_frame_id == 0))
    {
        return Ok(());
    }
    let mut normalized = module.clone();
    normalize(&mut normalized)?;
    let references = |module: &HloModuleProto| {
        module
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .map(|i| i.metadata.as_ref().map(|m| m.stack_frame_id))
            .collect::<Vec<_>>()
    };
    if normalized.stack_frame_index != module.stack_frame_index
        || references(&normalized) != references(module)
    {
        return Err(Error::new(
            "unsupported",
            "stack-frame tables or references would be normalized by the pinned parser",
        ));
    }
    let Some(index) = module.stack_frame_index.as_ref() else {
        return Ok(());
    };
    text.push_str("\nFileNames\n");
    for (id, name) in index.file_names.iter().enumerate() {
        writeln!(text, "{} {}", id + 1, strings::quote(name.as_bytes())?)
            .map_err(|e| Error::new("print", e.to_string()))?;
    }
    text.push_str("FunctionNames\n");
    for (id, name) in index.function_names.iter().enumerate() {
        writeln!(text, "{} {}", id + 1, strings::quote(name.as_bytes())?)
            .map_err(|e| Error::new("print", e.to_string()))?;
    }
    text.push_str("FileLocations\n");
    for (id, location) in index.file_locations.iter().enumerate() {
        writeln!(text, "{} {{file_name_id={} function_name_id={} line={} end_line={} column={} end_column={}}}",
            id + 1, location.file_name_id, location.function_name_id, location.line, location.end_line, location.column, location.end_column)
            .map_err(|e| Error::new("print", e.to_string()))?;
    }
    text.push_str("StackFrames\n");
    for (id, frame) in index.stack_frames.iter().enumerate() {
        let parent = frame.parent_frame_id.checked_add(1).ok_or_else(|| {
            Error::new(
                "unsupported",
                "stack-frame parent cannot be represented in text",
            )
        })?;
        writeln!(
            text,
            "{} {{file_location_id={} parent_frame_id={}}}",
            id + 1,
            frame.file_location_id,
            parent
        )
        .map_err(|e| Error::new("print", e.to_string()))?;
    }
    text.push('\n');
    Ok(())
}
