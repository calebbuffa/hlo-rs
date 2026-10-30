//! Guards against dictionary keys declared by more than one operation, which
//! makes every `Context::new` panic.

use pliron::{context::get_dict_key_ids, op::Op};
use std::collections::BTreeMap;

#[test]
fn dictionary_keys_are_unique() {
    // Reference each dependency so its keys are linked into this binary.
    let _ = (
        pliron_iree_linalg_ext::ops::SortOp::get_opid_static,
        pliron_iree_linalg_ext::pliron_linalg::ops::MatmulOp::get_opid_static,
        pliron_iree_linalg_ext::pliron_linalg::pliron_math::ops::AbsFOp::get_opid_static,
        pliron_iree_linalg_ext::pliron_arith::ops::AddFOp::get_opid_static,
    );
    let mut seen: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for key in get_dict_key_ids() {
        seen.entry(key.id.to_string())
            .or_default()
            .push(format!("{}:{}", key.file, key.line));
    }
    let duplicates: Vec<_> = seen
        .into_iter()
        .filter(|(_, sites)| sites.len() > 1)
        .collect();
    assert!(duplicates.is_empty(), "{duplicates:#?}");
}
