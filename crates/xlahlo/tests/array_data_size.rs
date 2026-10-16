use xlahlo::{Dimension, Module, PrimitiveType, Shape, VerifyOptions, text};

#[test]
fn data_size_distinguishes_outer_tile_gaps_from_trailing_padding() {
    for (shape, expected) in [
        ("f32[]", 4),
        ("u4[3]", 3),
        ("u4[3]{0:E(4)}", 2),
        ("f32[3]{0:T(8)}", 12),
        ("f32[3,3]{1,0}", 36),
        ("f32[3,3]{1,0:T(2,2)}", 52),
        ("f32[3,3]{1,0:T(2,2)(8,8)}", 52),
        ("f32[3,3]{1,0:T(2,2)L(64)}", 52),
        ("s4[64,128]{1,0:E(4)}", 4096),
        ("s4[9216,6144]{1,0:T(64,128)(8,1)E(4)}", 28311552),
        ("f32[16,256]{1,0:T(d,128)}", 16384),
        ("f32[3,256]{1,0:T(d,128)(2,1)}", 3584),
        ("b(f32[3,3]{1,0:T(2,2)})", 52),
    ] {
        assert_eq!(
            text::parse_shape(shape).unwrap().array_data_size().unwrap(),
            expected,
            "{shape}"
        );
    }
    for shape in ["f32[0]", "f32[0,3]", "f32[3,0]", "f32[2,0,3]"] {
        assert_eq!(
            text::parse_shape(shape).unwrap().array_data_size().unwrap(),
            0
        );
    }
    for shape in ["token[]", "(f32[],f32[])", "f32[?]", "f32[2,3]{:S(1)}"] {
        assert!(text::parse_shape(shape).unwrap().array_data_size().is_err());
    }
    assert!(
        Shape::array(PrimitiveType::F32, vec![2])
            .unwrap()
            .array_data_size()
            .is_err()
    );
}

#[test]
fn small_tiled_data_sizes_agree_with_block_and_in_block_coordinates() {
    for rows in 1_i64..=6 {
        for cols in 1..=6 {
            for tile_rows in 1..=4 {
                for tile_cols in 1..=4 {
                    for transpose in [false, true] {
                        let (major, minor) = if transpose {
                            (cols, rows)
                        } else {
                            (rows, cols)
                        };
                        let order = if transpose { "0,1" } else { "1,0" };
                        let shape = text::parse_shape(&format!(
                            "u8[{rows},{cols}]{{{order}:T({tile_rows},{tile_cols})}}"
                        ))
                        .unwrap();
                        let block = (major - 1) / tile_rows * ((minor + tile_cols - 1) / tile_cols)
                            + (minor - 1) / tile_cols;
                        let in_block =
                            (major - 1) % tile_rows * tile_cols + (minor - 1) % tile_cols;
                        assert_eq!(
                            shape.array_data_size().unwrap(),
                            block * tile_rows * tile_cols + in_block + 1
                        );
                    }
                }
            }
        }
    }
}

fn padded_size(shape: &Shape) -> xlahlo::Result<i64> {
    if shape
        .layout()
        .is_some_and(|layout| !layout.tiles.is_empty())
    {
        Ok(64)
    } else {
        shape.byte_size(-1)
    }
}

fn options(allow: bool) -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        allow_bitcast_to_have_different_size: allow,
        shape_size: Some(padded_size),
        ..Default::default()
    }
}

#[test]
fn different_bitcast_sizes_require_opt_in_static_shapes_and_equal_data_sizes() {
    let source = "HloModule m ENTRY main { p = f32[3,3]{1,0:T(2,2)} parameter(0) ROOT b = u8[52] bitcast(p) }";
    let module = text::parse_module(source).unwrap();
    assert_eq!(
        module.verify(options(false)).unwrap_err().category,
        "bitcast"
    );
    module.verify(options(true)).unwrap();
    for bad in [
        source.replace("u8[52]", "u8[36]"),
        source.replace("f32[3,3]", "f32[<=3,3]"),
        source.replace("u8[52]", "u8[<=52]"),
    ] {
        assert!(
            text::parse_module(&bad)
                .unwrap()
                .verify(options(true))
                .is_err()
        );
    }
    // The option and the size callback are irrelevant without layout sensitivity.
    module
        .verify(VerifyOptions {
            layout_sensitive: false,
            ..options(false)
        })
        .unwrap();
    let mut wire = module.to_proto().unwrap();
    let bitcast = wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "bitcast")
        .unwrap();
    bitcast.shape.as_mut().unwrap().dimensions = vec![64];
    let result_shape = bitcast.shape.as_ref().unwrap().clone();
    if let Some(program) = wire.host_program_shape.as_mut() {
        program.result = Some(result_shape.clone()).into();
    }
    for computation in &mut wire.computations {
        if let Some(program) = computation.program_shape.as_mut() {
            program.result = Some(result_shape.clone()).into();
        }
    }
    Module::from_proto(wire)
        .unwrap()
        .verify(options(false))
        .unwrap();
    let mut layout = text::parse_shape("f32[2,2]")
        .unwrap()
        .layout()
        .unwrap()
        .clone();
    layout.tiles.push(xlahlo::proto::xla_data::TileProto {
        dimensions: vec![i64::MAX, i64::MAX],
        ..Default::default()
    });
    assert!(
        Shape::array(PrimitiveType::F32, vec![2, 2])
            .unwrap()
            .with_layout(layout)
            .unwrap()
            .array_data_size()
            .is_err()
    );
    assert!(
        Shape::dynamic_array(PrimitiveType::F32, vec![Dimension::Unbounded])
            .unwrap()
            .array_data_size()
            .is_err()
    );
}
