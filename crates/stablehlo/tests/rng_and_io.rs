use stablehlo::{
    Attribute, Attributes, ChannelHandle, ElementType, StableHloAttribute, StableHloOp, TensorType,
    Type,
};

mod common;
use common::computation;

fn tensor(element: ElementType, shape: &[u64]) -> Type {
    TensorType::static_shape(element, shape.iter().copied()).into()
}

#[test]
fn rng_checks_distribution_elements_and_nonconstant_shape_rank() {
    for (lhs, rhs, output, distribution, rank, accepted) in [
        (
            ElementType::F32,
            ElementType::F32,
            ElementType::F32,
            "NORMAL",
            2,
            true,
        ),
        (
            ElementType::I32,
            ElementType::I32,
            ElementType::I32,
            "UNIFORM",
            2,
            true,
        ),
        (
            ElementType::I32,
            ElementType::I32,
            ElementType::I32,
            "NORMAL",
            2,
            false,
        ),
        (
            ElementType::F32,
            ElementType::F64,
            ElementType::F64,
            "UNIFORM",
            2,
            false,
        ),
        (
            ElementType::F32,
            ElementType::F32,
            ElementType::F64,
            "NORMAL",
            2,
            false,
        ),
        (
            ElementType::F32,
            ElementType::F32,
            ElementType::F32,
            "NORMAL",
            1,
            false,
        ),
        (
            ElementType::F32,
            ElementType::F32,
            ElementType::F32,
            "NORMAL",
            3,
            false,
        ),
    ] {
        let attributes = [(
            "rng_distribution".into(),
            Attribute::Enum {
                dialect: "stablehlo".into(),
                name: "rng_distribution".into(),
                value: distribution.into(),
            },
        )]
        .into();
        assert_eq!(
            computation(
                StableHloOp::Rng,
                vec![
                    tensor(lhs, &[]),
                    tensor(rhs, &[]),
                    tensor(ElementType::I64, &[rank])
                ],
                vec![tensor(output, &[2, 3])],
                attributes,
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn recv_requires_the_channel_direction_for_device_and_host_transfers() {
    for host in [false, true] {
        for channel_type in 0..=4 {
            let attributes: Attributes = [
                (
                    "channel_handle".into(),
                    Attribute::StableHlo(Box::new(StableHloAttribute::ChannelHandle(
                        ChannelHandle {
                            handle: 1,
                            channel_type,
                        },
                    ))),
                ),
                ("is_host_transfer".into(), Attribute::Bool(host)),
            ]
            .into();
            assert_eq!(
                computation(
                    StableHloOp::Recv,
                    vec![Type::Token],
                    vec![tensor(ElementType::F32, &[2, 3]), Type::Token],
                    attributes,
                )
                .is_ok(),
                channel_type == if host { 3 } else { 1 }
            );
        }
    }
}

#[test]
fn send_requires_device_to_host_channels_only_for_host_transfers() {
    for host in [false, true] {
        for channel_type in 0..=4 {
            let attributes: Attributes = [
                (
                    "channel_handle".into(),
                    Attribute::StableHlo(Box::new(StableHloAttribute::ChannelHandle(
                        ChannelHandle {
                            handle: 1,
                            channel_type,
                        },
                    ))),
                ),
                ("is_host_transfer".into(), Attribute::Bool(host)),
            ]
            .into();
            assert_eq!(
                computation(
                    StableHloOp::Send,
                    vec![tensor(ElementType::F32, &[2, 3]), Type::Token],
                    vec![Type::Token],
                    attributes,
                )
                .is_ok(),
                channel_type == if host { 2 } else { 1 }
            );
        }
    }
}
