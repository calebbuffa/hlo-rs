use xlahlo::{Module, PrimitiveType as P, VerifyOptions, text};

fn strict() -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    }
}
const SEND: &str = "HloModule m ENTRY main { p = f32[2] parameter(0) t = token[] after-all() s = () send(p,t), channel_id=1 ROOT d = token[] send-done(s), channel_id=1 }";
const RECV: &str = "HloModule m ENTRY main { t = token[] after-all() r = (f32[2]) recv(t), channel_id=2 ROOT d = s32[] recv-done(r), channel_id=2 }";

#[test]
fn channel_text_uses_constructor_shapes_not_written_context_shapes() {
    for source in [SEND, RECV] {
        let module = text::parse_module(source).unwrap();
        module.verify(strict()).unwrap();
        let start = module.as_proto().computations[0]
            .instructions
            .iter()
            .find(|i| matches!(i.opcode.as_str(), "send" | "recv"))
            .unwrap();
        let shape = start.shape.as_ref().unwrap();
        assert_eq!(shape.tuple_shapes.len(), 3);
        assert_eq!(
            shape.tuple_shapes[1].element_type.enum_value().unwrap(),
            P::U32
        );
        let printed = text::print_module(&module).unwrap();
        let reparsed = text::parse_module(&printed).unwrap();
        reparsed.verify(strict()).unwrap();
        assert_eq!(text::print_module(&reparsed).unwrap(), printed);
    }
    assert!(text::parse_module(&RECV.replace("(f32[2]) recv", "f32[2] recv")).is_err());
    assert!(
        text::parse_module(&SEND.replace(
            "d = token[] send-done(s), channel_id=1",
            "d = token[] send-done(s), channel_id=2"
        ))
        .is_err()
    );
}

#[test]
fn channel_pairing_checks_users_ids_flags_and_done_producers() {
    let module = text::parse_module(SEND).unwrap();
    for edit in 0..5 {
        let mut wire = module.to_proto().unwrap();
        let instructions = &mut wire.computations[0].instructions;
        match edit {
            0 => {
                instructions
                    .iter_mut()
                    .find(|i| i.opcode == "send-done")
                    .unwrap()
                    .channel_id = 2
            }
            1 => {
                instructions
                    .iter_mut()
                    .find(|i| i.opcode == "send-done")
                    .unwrap()
                    .is_host_transfer = true
            }
            2 => {
                let parameter = instructions
                    .iter()
                    .find(|i| i.opcode == "parameter")
                    .unwrap()
                    .id;
                instructions
                    .iter_mut()
                    .find(|i| i.opcode == "send-done")
                    .unwrap()
                    .operand_ids = vec![parameter];
            }
            3 => {
                let mut duplicate = instructions
                    .iter()
                    .find(|i| i.opcode == "send-done")
                    .unwrap()
                    .clone();
                duplicate.id = 99;
                duplicate.name = "extra".into();
                instructions.push(duplicate);
            }
            _ => {
                instructions
                    .iter_mut()
                    .find(|i| i.opcode == "send")
                    .unwrap()
                    .shape
                    .as_mut()
                    .unwrap()
                    .tuple_shapes[1]
                    .element_type = P::S32.into()
            }
        }
        assert!(
            Module::from_proto(wire)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let source = SEND.replace(
        "ROOT d =",
        "control = () custom-call(s), custom_call_target=\"control_dep\" ROOT d =",
    );
    assert!(
        text::parse_module(&source)
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    assert!(
        text::parse_module(&source.replace("channel_id=1", "channel_id=0"))
            .unwrap()
            .verify(strict())
            .is_err()
    );
    text::parse_module(&source.replace(", channel_id=1", ""))
        .unwrap()
        .verify(strict())
        .unwrap();
}

#[test]
fn host_transfers_sharing_a_channel_compare_operand_shapes_and_frontend_attributes() {
    let source = "HloModule m ENTRY main { a = f32[2] parameter(0) b = f32[2] parameter(1) t = token[] after-all() s0 = () send(a,t), channel_id=1, is_host_transfer=true send1 = () send(b,t), channel_id=1, is_host_transfer=true d0 = token[] send-done(s0), channel_id=1, is_host_transfer=true d1 = token[] send-done(send1), channel_id=1, is_host_transfer=true ROOT r = (token[],token[]) tuple(d0,d1) }";
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    for attribute in [
        ", frontend_attributes={key=\"different\"}",
        ", backend_config=\"different\"",
    ] {
        let changed = source.replace(
            "send1 = () send(b,t), channel_id=1, is_host_transfer=true",
            &format!("send1 = () send(b,t), channel_id=1, is_host_transfer=true{attribute}"),
        );
        assert_eq!(
            text::parse_module(&changed)
                .unwrap()
                .verify(strict())
                .unwrap_err()
                .category,
            "channel"
        );
    }
    let changed = source.replace("b = f32[2]", "b = f32[3]");
    assert_eq!(
        text::parse_module(&changed)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "channel"
    );
}

#[test]
fn pinned_default_does_not_invent_send_recv_token_or_absent_channel_checks() {
    let source = SEND.replace("send(p,t)", "send(p,p)");
    text::parse_module(&source)
        .unwrap()
        .verify(strict())
        .unwrap();
    let source = RECV.replace("t = token[] after-all()", "t = s32[] parameter(0)");
    text::parse_module(&source)
        .unwrap()
        .verify(strict())
        .unwrap();
    let mut wire = text::parse_module(SEND).unwrap().to_proto().unwrap();
    wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "send")
        .unwrap()
        .channel_id = 0;
    Module::from_proto(wire).unwrap().verify(strict()).unwrap();
}

#[test]
fn printing_rejects_context_shapes_that_pinned_constructors_would_change() {
    let mut wire = text::parse_module(SEND).unwrap().to_proto().unwrap();
    let start = wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "send")
        .unwrap();
    start.shape.as_mut().unwrap().tuple_shapes[0]
        .layout
        .as_mut()
        .unwrap()
        .memory_space = 5;
    let module = Module::from_proto(wire).unwrap();
    module.verify(strict()).unwrap();
    assert_eq!(
        text::print_module(&module).unwrap_err().category,
        "unsupported"
    );
}
