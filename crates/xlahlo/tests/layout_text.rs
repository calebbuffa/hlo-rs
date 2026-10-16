use xlahlo::{VerifyOptions, text};

#[test]
fn dense_layout_fields_roundtrip_in_upstream_order() {
    for source in [
        "f32[4,6]{1,0:T(2,2)(d,*)L(64)E(4)S(5)SC(0:1,2)(1:1)M(8)}",
        "f32[]{:T(8)L(32)E(16)S(1)M(8)}",
        "b(f32[4,6]{0,1:T(d,4)SC(0:1,2)})",
        "(f32[]{:S(5)}, f16[2]{0:E(16)})",
        "f32[2]{:S(1)}",
        "f32[2]{0:SC(0:)}",
    ] {
        let shape = text::parse_shape(source).unwrap();
        let printed = text::print_shape(&shape).unwrap();
        assert_eq!(text::parse_shape(&printed).unwrap(), shape, "{source}");
        assert_eq!(printed, source);
    }
}

#[test]
fn invalid_or_out_of_order_layout_fields_are_rejected() {
    for source in [
        "f32[2]{0:E(4)T(2)}",
        "f32[2]{0:L(2)L(4)}",
        "f32[2]{0:T()}",
        "f32[2]{0:T(0)}",
        "f32[2]{0:E(-1)}",
        "f32[2]{0:SC(0:2)}",
        "f32[2]{0:SC(1:1)}",
        "f32[2]{0:T(2)Z(3)}",
        "f32[2]{0,:T(2)}",
    ] {
        assert!(text::parse_shape(source).is_err(), "{source}");
    }
    let negative = text::parse_shape("f32[2]{0:M(-1)}").unwrap();
    assert_eq!(
        text::print_shape(&negative).unwrap_err().category,
        "unsupported"
    );
}

#[test]
fn module_layouts_roundtrip_without_bypassing_memory_or_buffer_contracts() {
    let source =
        "HloModule m ENTRY main { p = f32[2]{0:S(5)} parameter(0) ROOT b = u8[8]{0} bitcast(p) }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let strict = VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    };
    assert_eq!(module.verify(strict).unwrap_err().category, "bitcast");
    let source = source.replace("S(5)", "S(1)");
    let module = text::parse_module(&source).unwrap();
    module.verify(strict).unwrap();
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(strict)
        .unwrap();
    let buffer = "HloModule m ENTRY main { p = b(f32[2]{0:S(5)}) parameter(0) ROOT b = b(u8[8]{0}) bitcast(p) }";
    assert_eq!(
        text::parse_module(buffer)
            .unwrap()
            .verify(Default::default())
            .unwrap_err()
            .category,
        "buffer"
    );
}
