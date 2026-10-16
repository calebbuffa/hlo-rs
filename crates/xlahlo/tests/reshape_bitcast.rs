use std::collections::BTreeMap;
use xlahlo::{PrimitiveType as P, Shape, VerifyOptions, proto::xla_data::TileProto, text};

fn permutations(values: &[i64]) -> Vec<Vec<i64>> {
    if values.is_empty() {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    for (i, &axis) in values.iter().enumerate() {
        let mut rest = values.to_vec();
        rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, axis);
            result.push(tail);
        }
    }
    result
}

fn physical_offsets(shape: &Shape) -> Vec<i64> {
    (0..shape.element_count().unwrap())
        .map(|logical| {
            let mut coordinates = vec![0; shape.dimensions().len()];
            let mut remaining = logical;
            for axis in (0..coordinates.len()).rev() {
                coordinates[axis] = remaining % shape.dimensions()[axis];
                remaining /= shape.dimensions()[axis];
            }
            let mut stride = 1;
            let mut offset = 0;
            for &axis in &shape.layout().unwrap().minor_to_major {
                offset += coordinates[axis as usize] * stride;
                stride *= shape.dimensions()[axis as usize];
            }
            offset
        })
        .collect()
}

#[test]
fn unit_index_criterion_matches_full_per_element_enumeration_for_small_shapes() {
    let mut groups: BTreeMap<i64, Vec<(Shape, Vec<i64>)>> = BTreeMap::new();
    for rank in 0..=3 {
        for mut code in 0..4_usize.pow(rank) {
            let dimensions = (0..rank)
                .map(|_| {
                    let size = (code % 4) as i64;
                    code /= 4;
                    size
                })
                .collect::<Vec<_>>();
            for order in permutations(&(0..i64::from(rank)).collect::<Vec<_>>()) {
                let mut wire = Shape::array(P::U8, dimensions.clone())
                    .unwrap()
                    .with_default_layout()
                    .unwrap()
                    .into_proto();
                wire.layout.as_mut().unwrap().minor_to_major = order;
                let shape = Shape::from_proto(wire).unwrap();
                let offsets = physical_offsets(&shape);
                groups
                    .entry(shape.element_count().unwrap())
                    .or_default()
                    .push((shape, offsets));
            }
        }
    }
    for shapes in groups.values() {
        for (input, input_offsets) in shapes {
            for (output, output_offsets) in shapes {
                assert_eq!(
                    input.reshape_is_bitcast(output, false).unwrap(),
                    input_offsets == output_offsets,
                    "{input:?} -> {output:?}"
                );
            }
        }
    }
}

#[test]
fn bitcast_reshape_type_count_layout_and_zero_element_contracts() {
    let input = text::parse_shape("f32[2,3]{1,0}").unwrap();
    let output = text::parse_shape("s32[6]{0}").unwrap();
    assert!(!input.reshape_is_bitcast(&output, false).unwrap());
    assert!(input.reshape_is_bitcast(&output, true).unwrap());
    assert!(
        !input
            .reshape_is_bitcast(&text::parse_shape("f32[5]").unwrap(), false)
            .unwrap()
    );
    assert!(
        text::parse_shape("f32[0,3]{0,1}")
            .unwrap()
            .reshape_is_bitcast(&text::parse_shape("f32[2,0]").unwrap(), false)
            .unwrap()
    );
    assert!(
        Shape::array(P::F32, vec![6])
            .unwrap()
            .reshape_is_bitcast(&input, false)
            .is_err()
    );
    assert!(Shape::token().reshape_is_bitcast(&input, false).is_err());
    assert!(
        text::parse_shape("f32[?]")
            .unwrap()
            .reshape_is_bitcast(&input, false)
            .is_err()
    );
    let buffer = text::parse_shape("b(f32[2,3])").unwrap();
    assert_eq!(buffer.array_view().unwrap(), input);
    assert!(
        buffer
            .reshape_is_bitcast(&text::parse_shape("f32[6]").unwrap(), false)
            .unwrap()
    );
    assert!(
        input
            .reshape_is_bitcast(&text::parse_shape("b(f32[6])").unwrap(), false)
            .unwrap()
    );
    assert!(Shape::token().array_view().is_err());
}

#[test]
fn pinned_dense_indexing_ignores_tiles_packing_memory_and_dynamic_flags() {
    let mut wire = text::parse_shape("f32[<=2,3]").unwrap().into_proto();
    let layout = wire.layout.as_mut().unwrap();
    layout.tiles = vec![TileProto {
        dimensions: vec![4, 4],
        ..Default::default()
    }];
    layout.memory_space = 5;
    layout.element_size_in_bits = 16;
    layout.tail_padding_alignment_in_elements = 64;
    let input = Shape::from_proto(wire).unwrap();
    assert!(
        input
            .reshape_is_bitcast(&text::parse_shape("f32[6]").unwrap(), false)
            .unwrap()
    );
}

#[test]
fn physical_reshape_policy_is_optional_and_exempts_fused_reshapes() {
    let source =
        "HloModule m ENTRY main { p = f32[2,3]{0,1} parameter(0) ROOT r = f32[6] reshape(p) }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let options = VerifyOptions {
        verify_reshape_is_bitcast: true,
        ..Default::default()
    };
    assert_eq!(module.verify(options).unwrap_err().category, "reshape");
    text::parse_module(&source.replace("{0,1}", "{1,0}"))
        .unwrap()
        .verify(options)
        .unwrap();
    let source = "HloModule m body { p = f32[2,3]{0,1} parameter(0) ROOT r = f32[6] reshape(p) } ENTRY main { p = f32[2,3]{0,1} parameter(0) ROOT r = f32[6] fusion(p), kind=kLoop, calls=body }";
    text::parse_module(source).unwrap().verify(options).unwrap();
}
