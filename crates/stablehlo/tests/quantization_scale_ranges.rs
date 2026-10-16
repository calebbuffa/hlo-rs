use stablehlo::text::{parse_module, parse_type, print_type};
use stablehlo::{ElementType, QuantizedElementType, QuantizedTensorType, Type, TypeError};

fn metadata(expressed: ElementType, axis: Option<usize>, scales: Vec<f64>) -> QuantizedElementType {
    let zeros = vec![0; scales.len()];
    QuantizedElementType::new(ElementType::I8, None, None, expressed, axis, scales, zeros)
}

#[test]
fn every_modeled_float_accepts_exact_smallest_and_largest_scale() {
    use ElementType::*;
    let ranges = [
        (F4E2M1Fn, 0.5, 6.0),
        (F6E2M3Fn, 0.125, 7.5),
        (F6E3M2Fn, 0.0625, 28.0),
        (F8E3M4, 0.015625, 15.5),
        (F8E4M3, 0.001953125, 240.0),
        (F8E4M3Fn, 0.001953125, 448.0),
        (F8E4M3FnUz, 0.0009765625, 240.0),
        (F8E4M3B11FnUz, 0.0001220703125, 30.0),
        (F8E5M2, 0.0000152587890625, 57344.0),
        (F8E5M2FnUz, 0.00000762939453125, 57344.0),
        (F8E8M0FnUz, 2.0_f64.powi(-127), 2.0_f64.powi(127)),
        (F16, 2.0_f64.powi(-24), 65504.0),
        (
            Bf16,
            2.0_f64.powi(-133),
            f64::from(f32::from_bits(0x7f7f_0000)),
        ),
        (
            Tf32,
            2.0_f64.powi(-136),
            f64::from(f32::from_bits(0x7f7f_e000)),
        ),
        (F32, f64::from(f32::from_bits(1)), f64::from(f32::MAX)),
        (F64, f64::from_bits(1), f64::MAX),
    ];
    for (expressed, smallest, largest) in ranges {
        for scale in [smallest, smallest.next_up(), largest.next_down(), largest] {
            let element = metadata(expressed, None, vec![scale]);
            assert_eq!(element.verify(), Ok(()), "{expressed:?}: {scale}");
            let tensor = Type::QuantizedTensor(QuantizedTensorType::from_element(
                vec![Some(1)],
                element.clone(),
            ));
            tensor.verify().unwrap();
            let standalone = Type::QuantizedElement(element);
            assert_eq!(
                parse_type(&print_type(&standalone).unwrap()).unwrap(),
                standalone
            );
            assert_eq!(parse_type(&print_type(&tensor).unwrap()).unwrap(), tensor);
        }
        for scale in [smallest.next_down(), largest.next_up()] {
            let element = metadata(expressed, None, vec![scale]);
            assert!(element.verify().is_err(), "{expressed:?}: {scale}");
            assert!(print_type(&Type::QuantizedElement(element.clone())).is_err());
            assert!(
                QuantizedTensorType::from_element(vec![Some(1)], element)
                    .verify()
                    .is_err()
            );
        }
    }
}

#[test]
fn every_per_axis_scale_is_checked_without_rounding() {
    for bad in [0.125_f64.next_down(), 7.5_f64.next_up()] {
        assert_eq!(
            metadata(ElementType::F6E2M3Fn, Some(0), vec![0.5, bad, 1.0]).verify(),
            Err(TypeError::QuantizationScaleOutOfRange(
                ElementType::F6E2M3Fn
            )),
        );
    }
    let scale = 0.5_f64.next_up();
    let element = metadata(ElementType::F16, None, vec![scale]);
    element.verify().unwrap();
    assert_eq!(element.scales().next().unwrap().to_bits(), scale.to_bits());
    for invalid in [0.0, -0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert_eq!(
            metadata(ElementType::F32, None, vec![invalid]).verify(),
            Err(TypeError::InvalidQuantizationScale)
        );
    }
}

#[test]
fn inline_and_aliased_scale_ranges_are_checked_at_all_entry_points() {
    for value in ["1.0e-100", "1.0e100"] {
        for source in [
            format!("!quant.uniform<i8:f32, {value}>"),
            format!("tensor<2x!quant.uniform<i8:f32, {value}>>"),
            format!("!quant.uniform<i8:f32:0, {{0.5, {value}}}>"),
        ] {
            assert!(parse_type(&source).is_err(), "{source}");
        }
        let source = format!(
            "!q = !quant.uniform<i8:f32, {value}> module {{ func.func private @f(tensor<2x!q>) }}"
        );
        assert!(parse_module(&source).is_err());
    }
}
