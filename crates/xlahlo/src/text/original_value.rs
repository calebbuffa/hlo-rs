use super::{MAX_DEPTH, ParseOptions, Parser, strings};
use crate::{
    Error, OriginalValue, Result,
    proto::xla_data::{OriginalArrayProto, OriginalValueElementProto, OriginalValueProto},
};

impl Parser {
    pub(super) fn original_array(&mut self) -> Result<OriginalArrayProto> {
        self.expect("{")?;
        let mut array = OriginalArrayProto::new();
        if self.peek().starts_with('"') {
            array.instruction_name = self.utf8_string()?;
            if self.peek() == "{" {
                array.shape_index = self.integers("{", "}")?;
            }
        }
        self.expect("}")?;
        Ok(array)
    }

    pub(super) fn original_value(&mut self) -> Result<OriginalValueProto> {
        self.expect("{")?;
        let mut wire = OriginalValueProto::default();
        if self.take("[") {
            self.expect("synthetic_call")?;
            self.expect("]")?;
            self.expect("}")?;
            wire.call_hierarchy = Some(String::new());
            return Ok(OriginalValue::from_proto(wire)?.normalized_proto());
        }
        let mut index: Vec<i64> = Vec::new();
        while !self.take("}") {
            match self.peek() {
                "(" => {
                    self.expect("(")?;
                    if index.len() >= MAX_DEPTH {
                        return Err(self.error("original-value nesting limit exceeded"));
                    }
                    index.push(0);
                }
                ")" => {
                    self.expect(")")?;
                    index
                        .pop()
                        .ok_or_else(|| self.error("unmatched original-value tuple end"))?;
                }
                "," => {
                    self.expect(",")?;
                    if self.take("[") {
                        if wire.call_hierarchy.is_some() {
                            return Err(self.error("duplicate original-value call hierarchy"));
                        }
                        wire.call_hierarchy = Some(self.utf8_string()?);
                        self.expect("]")?;
                    } else if let Some(axis) = index.last_mut() {
                        *axis = axis.checked_add(1).ok_or_else(|| {
                            Error::new("overflow", "original-value tuple index exceeds i64")
                        })?;
                    }
                }
                "{" => {
                    let array = self.original_array()?;
                    let array = (!array.instruction_name.is_empty()).then_some(array);
                    wire.elements.push(OriginalValueElementProto {
                        shape_index: index.clone(),
                        original_array: array.into(),
                        ..Default::default()
                    });
                }
                _ => return Err(self.error("expected original-value tuple tree or call hierarchy")),
            }
        }
        Ok(OriginalValue::from_proto(wire)?.normalized_proto())
    }
}

pub(super) fn print(wire: &OriginalValueProto) -> Result<String> {
    let value = OriginalValue::from_proto(wire.clone())?;
    if wire != &value.normalized_proto() {
        return Err(Error::new(
            "unsupported",
            "noncanonical or unknown original-value fields cannot be preserved in HLO text",
        ));
    }
    if value.is_synthetic_call() {
        return Ok("{[synthetic_call]}".into());
    }
    let leaves: std::collections::BTreeMap<_, _> = value.leaves().collect();
    let mut result = String::from("{");
    fn node(
        index: &mut Vec<i64>,
        leaves: &std::collections::BTreeMap<&[i64], Option<&OriginalArrayProto>>,
        result: &mut String,
    ) -> Result<()> {
        if index.len() > MAX_DEPTH || result.len() > ParseOptions::default().max_bytes {
            return Err(Error::new("limit", "original-value text limit exceeded"));
        }
        if let Some(array) = leaves.get(index.as_slice()) {
            result.push('{');
            if let Some(array) = array {
                if array.instruction_name.is_empty() {
                    return Err(Error::new(
                        "unsupported",
                        "empty original-array name cannot be preserved",
                    ));
                }
                result.push_str(&strings::quote(array.instruction_name.as_bytes())?);
                if !array.shape_index.is_empty() {
                    result.push_str(" {");
                    result.push_str(
                        &array
                            .shape_index
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(","),
                    );
                    result.push('}');
                }
            }
            result.push('}');
        } else {
            let count = leaves
                .keys()
                .filter(|leaf| leaf.starts_with(index) && leaf.len() > index.len())
                .map(|leaf| leaf[index.len()])
                .max()
                .map_or(Some(0), |axis| axis.checked_add(1))
                .ok_or_else(|| Error::new("limit", "original-value tuple width exceeds i64"))?;
            if count > (ParseOptions::default().max_bytes / 3) as i64 {
                return Err(Error::new(
                    "limit",
                    "original-value tuple width exceeds text limit",
                ));
            }
            result.push('(');
            for axis in 0..count {
                if axis > 0 {
                    result.push(',');
                }
                index.push(axis);
                node(index, leaves, result)?;
                index.pop();
            }
            result.push(')');
        }
        Ok(())
    }
    node(&mut Vec::new(), &leaves, &mut result)?;
    if let Some(hierarchy) = wire.call_hierarchy.as_ref() {
        result.push_str(",[");
        result.push_str(&strings::quote(hierarchy.as_bytes())?);
        result.push(']');
    }
    result.push('}');
    Ok(result)
}
