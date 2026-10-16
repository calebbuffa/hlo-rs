use std::collections::{BTreeMap, BTreeSet};

use crate::{Error, PrimitiveType, Result, proto::hlo::HloModuleProto};

#[derive(Default)]
pub(crate) struct NameUniquer {
    roots: BTreeMap<String, (BTreeSet<i64>, i64)>,
}

fn split_suffix(name: &str) -> (&str, Option<i64>) {
    if let Some((root, suffix)) = name.rsplit_once('.')
        && !root.is_empty()
        && let Ok(id) = suffix.parse()
    {
        return (root, Some(id));
    }
    (name, None)
}

fn sanitize(name: &str) -> String {
    let mut name: String = name
        .bytes()
        .enumerate()
        .map(|(index, byte)| {
            char::from(
                if byte.is_ascii_alphabetic()
                    || byte == b'_'
                    || index != 0 && (byte.is_ascii_digit() || b".-".contains(&byte))
                {
                    byte
                } else {
                    b'_'
                },
            )
        })
        .collect();
    if PrimitiveType::from_hlo_name(&name)
        .is_ok_and(|ty| !matches!(ty, PrimitiveType::TUPLE | PrimitiveType::BUFFER))
    {
        name.push('_');
    }

    if name.starts_with("__") && !name.starts_with("__xla_") {
        name.replace_range(..1, "a");
    }
    name
}

impl NameUniquer {
    pub(crate) fn for_module(module: &HloModuleProto) -> Self {
        let mut names = Self::default();
        for instruction in module.computations.iter().flat_map(|c| &c.instructions) {
            let (root, suffix) = split_suffix(&instruction.name);
            names
                .roots
                .entry(root.to_owned())
                .or_default()
                .0
                .insert(suffix.unwrap_or(0));
        }
        names
    }

    pub(crate) fn unique(&mut self, name: &str) -> Result<String> {
        let name = sanitize(if name.is_empty() { "name" } else { name });
        let (root, suffix) = split_suffix(&name);
        let (used, next) = self.roots.entry(root.to_owned()).or_default();
        let mut id = suffix.unwrap_or(0);
        if !used.insert(id) {
            while used.contains(next) {
                *next = next
                    .checked_add(1)
                    .ok_or_else(|| Error::new("overflow", "name IDs exhausted"))?;
            }
            id = *next;
            used.insert(id);
            *next = next
                .checked_add(1)
                .ok_or_else(|| Error::new("overflow", "name IDs exhausted"))?;
        }
        Ok(if id == 0 && suffix.is_none() {
            root.to_owned()
        } else {
            format!("{root}.{id}")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_suffix_reservations_and_sanitization() {
        let mut names = NameUniquer::default();
        for (input, expected) in [
            ("p.01", "p.1"),
            ("p", "p"),
            ("p", "p.2"),
            ("p.0", "p.3"),
            ("p.-1", "p.-1"),
            ("p.-1", "p.4"),
            ("__reserved", "a_reserved"),
            ("__xla_valid", "__xla_valid"),
            ("f32", "f32_"),
            ("tuple", "tuple"),
            ("buffer", "buffer"),
            ("1bad /name", "_bad__name"),
            ("", "name"),
        ] {
            assert_eq!(names.unique(input).unwrap(), expected);
        }
    }
}
