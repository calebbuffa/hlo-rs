//! Wire-level interchange, deliberately separate from semantic verification.

use protobuf::{Message, MessageFull};

use crate::{Error, Result};

/// Bounds allocation from encoded input. Decoding also uses the runtime's
/// recursion limit. Callers needing larger artifacts must opt in.
#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    pub max_bytes: usize,
    pub recursion_limit: u32,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024 * 1024,
            recursion_limit: 100,
        }
    }
}

pub fn decode<M: Message>(bytes: &[u8], options: DecodeOptions) -> Result<M> {
    if bytes.len() > options.max_bytes {
        return Err(Error::new("resource-limit", "protobuf byte limit exceeded"));
    }
    let mut input = protobuf::CodedInputStream::from_bytes(bytes);
    input.set_recursion_limit(options.recursion_limit);
    let mut message = M::new();
    message
        .merge_from(&mut input)
        .map_err(|e| Error::new("protobuf", e.to_string()))?;
    message
        .check_initialized()
        .map_err(|e| Error::new("protobuf", e.to_string()))?;
    input
        .check_eof()
        .map_err(|e| Error::new("protobuf", e.to_string()))?;
    Ok(message)
}

pub fn encode<M: Message>(message: &M) -> Result<Vec<u8>> {
    message
        .write_to_bytes()
        .map_err(|e| Error::new("protobuf", e.to_string()))
}

pub fn parse_textproto<M: MessageFull>(source: &str) -> Result<M> {
    protobuf::text_format::parse_from_str(source)
        .map_err(|e| Error::new("textproto", e.to_string()))
}

pub fn print_textproto<M: MessageFull>(message: &M) -> String {
    protobuf::text_format::print_to_string_pretty(message)
}

pub fn parse_json<M: MessageFull>(source: &str) -> Result<M> {
    protobuf_json_mapping::parse_from_str(source)
        .map_err(|e| Error::new("protojson", e.to_string()))
}

pub fn print_json<M: MessageFull>(message: &M) -> Result<String> {
    protobuf_json_mapping::print_to_string(message)
        .map_err(|e| Error::new("protojson", e.to_string()))
}
