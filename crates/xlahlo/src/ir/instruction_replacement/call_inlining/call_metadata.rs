//! OpenXLA CallInliner metadata rules from `propagate_call_metadata.cc`,
//! `call_inliner.cc` and `stack_frames.cc`.
use crate::ir::{ModuleMetadata, StackFrame, StackFrameIndex};
use crate::{
    attributes::OpMetadata,
    ir::OriginalValue,
    verification::{error, optional_attr, require},
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    operation::Operation,
    result::Result,
};
use std::collections::HashSet;

pub(super) fn update_op_name(metadata: &mut OpMetadata, prefix: &str) {
    if prefix.is_empty() {
        return;
    }
    let prefix = prefix.strip_suffix('/').unwrap_or(prefix);
    if prefix.is_empty() {
        return;
    }
    let name = metadata
        .op_name
        .strip_prefix('/')
        .unwrap_or(&metadata.op_name);
    let name = name.strip_suffix('/').unwrap_or(name);
    if name.starts_with(prefix) || prefix.len() + name.len() >= 1024 {
        return;
    }
    metadata.op_name = if name.is_empty() {
        prefix.into()
    } else {
        format!("{prefix}/{name}")
    };
}

fn chain(index: &StackFrameIndex, mut frame: i32) -> Result<Vec<StackFrame>> {
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    while frame != 0 {
        require(
            frame > 0 && seen.insert(frame),
            "invalid or cyclic stack frame chain",
        )?;
        let entry = usize::try_from(frame - 1)
            .ok()
            .and_then(|position| index.stack_frames.get(position))
            .ok_or_else(|| error("unknown stack frame"))?
            .clone();
        frame = entry.parent_frame_id;
        result.push(entry);
    }
    Ok(result)
}

pub(super) fn concatenate(index: &mut StackFrameIndex, prefix: i32, suffix: i32) -> Result<i32> {
    chain(index, prefix)?;
    let suffix_chain = chain(index, suffix)?;
    let mut current = prefix;
    for mut frame in suffix_chain.into_iter().rev() {
        frame.parent_frame_id = current;
        current =
            if let Some(position) = index.stack_frames.iter().position(|entry| *entry == frame) {
                i32::try_from(position + 1).map_err(|_| error("stack frame index overflow"))?
            } else {
                let id = i32::try_from(index.stack_frames.len() + 1)
                    .map_err(|_| error("stack frame index overflow"))?;
                index.stack_frames.push(frame);
                id
            };
    }
    Ok(current)
}

pub(super) fn propagate(
    ctx: &Context,
    module: Ptr<Operation>,
    instruction: Ptr<Operation>,
    prefix: &str,
    parent: i32,
) -> Result<()> {
    if prefix.is_empty() && parent == 0 {
        return Ok(());
    }
    let mut metadata = optional_attr::<OpMetadata>(ctx, instruction, "hlo_metadata")?
        .as_deref()
        .cloned()
        .unwrap_or_default();
    let previous = metadata.clone();
    update_op_name(&mut metadata, prefix);
    if parent != 0 {
        let mut module_metadata =
            optional_attr::<ModuleMetadata>(ctx, module, "hlo_module_metadata")?
                .as_deref()
                .cloned()
                .ok_or_else(|| error("stack propagation requires module metadata"))?;
        let frames = module_metadata
            .stack_frame_index
            .as_mut()
            .ok_or_else(|| error("stack propagation requires a stack frame index"))?;
        frames.verify(ctx)?;
        let suffix_chain = chain(frames, metadata.stack_frame_id)?;
        chain(frames, parent)?;
        let already_prefixed = metadata.stack_frame_id == parent
            || suffix_chain
                .iter()
                .any(|frame| frame.parent_frame_id == parent);
        if !already_prefixed {
            metadata.stack_frame_id = concatenate(frames, parent, metadata.stack_frame_id)?;
            module
                .deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_module_metadata"), module_metadata);
        }
    }
    if metadata != previous {
        instruction
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_metadata"), metadata);
    }
    Ok(())
}

pub(super) fn original_value(
    ctx: &Context,
    call: Ptr<Operation>,
    source: Ptr<Operation>,
) -> Result<Option<OriginalValue>> {
    let call = optional_attr::<OriginalValue>(ctx, call, "hlo_original_value")?;
    let Some(hierarchy) = call.as_ref().and_then(|value| value.call_hierarchy.first()) else {
        return Ok(None);
    };
    let mut value = optional_attr::<OriginalValue>(ctx, source, "hlo_original_value")?
        .as_deref()
        .cloned();
    if !hierarchy.is_empty()
        && let Some(value) = &mut value
    {
        for element in &mut value.elements {
            for array in &mut element.original_array {
                array.instruction_name = format!("{hierarchy}/{}", array.instruction_name);
            }
        }
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::super::inline_call;
    use super::super::tests::{attr, checked, fixture, name, roundtrip, source_root};
    use super::*;
    use crate::attributes::Callees;
    use crate::ir::FileLocation;
    use crate::ops::control::CallOp;
    use pliron::op::Op;

    #[test]
    fn names_and_stack_frames_match_prefix_length_and_interning_rules() {
        let mut metadata = OpMetadata {
            op_name: "/inner/".into(),
            ..Default::default()
        };
        update_op_name(&mut metadata, "outer/");
        assert_eq!(metadata.op_name, "outer/inner");
        update_op_name(&mut metadata, "outer/");
        assert_eq!(metadata.op_name, "outer/inner");
        metadata.op_name = "x".repeat(1023);
        update_op_name(&mut metadata, "p");
        assert_eq!(metadata.op_name.len(), 1023);
        let mut index = StackFrameIndex {
            file_names: vec!["file".into()],
            function_names: vec!["function".into()],
            file_locations: vec![FileLocation {
                file_name_id: 1,
                function_name_id: 1,
                ..Default::default()
            }],
            stack_frames: vec![
                StackFrame {
                    file_location_id: 1,
                    parent_frame_id: 0,
                },
                StackFrame {
                    file_location_id: 1,
                    parent_frame_id: 1,
                },
            ],
        };
        assert_eq!(concatenate(&mut index, 2, 1).unwrap(), 3);
        assert_eq!(concatenate(&mut index, 2, 1).unwrap(), 3);
        assert_eq!(index.stack_frames.len(), 3);
        assert_eq!(concatenate(&mut index, 0, 2).unwrap(), 2);
        assert!(concatenate(&mut index, 99, 1).is_err());
    }

    #[test]
    fn real_stack_frame_concatenation_interns_frames_and_repeated_inline_reuses_them() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, false);
        attr(
            ctx,
            f.module,
            "hlo_module_metadata",
            crate::ir::ModuleMetadata {
                stack_frame_index: Some(crate::ir::StackFrameIndex {
                    file_names: vec!["file".into()],
                    function_names: vec!["function".into()],
                    file_locations: vec![crate::ir::FileLocation {
                        file_name_id: 1,
                        function_name_id: 1,
                        ..Default::default()
                    }],
                    stack_frames: vec![
                        crate::ir::StackFrame {
                            file_location_id: 1,
                            parent_frame_id: 0,
                        },
                        crate::ir::StackFrame {
                            file_location_id: 1,
                            parent_frame_id: 1,
                        },
                    ],
                }),
                ..Default::default()
            },
        );
        let root = source_root(ctx, &f);
        attr(
            ctx,
            root,
            "hlo_metadata",
            OpMetadata {
                stack_frame_id: 1,
                ..Default::default()
            },
        );
        attr(
            ctx,
            f.call,
            "hlo_metadata",
            OpMetadata {
                stack_frame_id: 2,
                ..Default::default()
            },
        );
        let second = CallOp::new(ctx, f.ty, vec![f.parameter], Callees(vec!["callee".into()]))
            .get_operation();
        second.insert_before(ctx, f.ret);
        name(ctx, second, "second");
        attr(
            ctx,
            second,
            "hlo_metadata",
            OpMetadata {
                stack_frame_id: 2,
                ..Default::default()
            },
        );
        let first = checked(ctx, &f);
        let second =
            inline_call(ctx, second, Default::default(), Some(Default::default())).unwrap();
        for output in [first, second] {
            assert_eq!(
                optional_attr::<OpMetadata>(
                    ctx,
                    output.root.defining_op().unwrap(),
                    "hlo_metadata"
                )
                .unwrap()
                .unwrap()
                .stack_frame_id,
                3
            );
        }
        let metadata =
            optional_attr::<crate::ir::ModuleMetadata>(ctx, f.module, "hlo_module_metadata")
                .unwrap()
                .unwrap();
        assert_eq!(
            metadata
                .stack_frame_index
                .as_ref()
                .unwrap()
                .stack_frames
                .len(),
            3
        );
        drop(metadata);
        roundtrip(ctx, f.module);
    }
}
