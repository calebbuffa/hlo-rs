use std::fmt::Write;

use super::upstream_support::{self, UpstreamResult};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub dtype: String,
    pub dimensions: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl Tensor {
    pub fn i32(dimensions: &[usize], values: &[i32]) -> Self {
        assert_eq!(dimensions.iter().product::<usize>(), values.len());
        Self {
            dtype: "<i4".into(),
            dimensions: dimensions.to_vec(),
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    pub fn f32(dimensions: &[usize], values: &[f32]) -> Self {
        assert_eq!(dimensions.iter().product::<usize>(), values.len());
        Self {
            dtype: "<f4".into(),
            dimensions: dimensions.to_vec(),
            bytes: values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        }
    }

    fn encode(&self) -> String {
        let mut hex = String::new();
        for byte in &self.bytes {
            write!(&mut hex, "{byte:02x}").unwrap();
        }
        format!(
            "{}\t{}\t{}",
            self.dtype,
            self.dimensions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
            hex
        )
    }

    fn decode(source: &str) -> Self {
        let fields = source.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 3, "malformed upstream execution tensor");
        let dimensions = if fields[1].is_empty() {
            vec![]
        } else {
            fields[1]
                .split(',')
                .map(|dimension| dimension.parse().unwrap())
                .collect()
        };
        assert!(
            fields[2].len().is_multiple_of(2),
            "truncated upstream tensor data"
        );
        let bytes = fields[2]
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        Self {
            dtype: fields[0].into(),
            dimensions,
            bytes,
        }
    }
}

pub fn execute(requests: &[(String, Vec<Tensor>)]) -> Vec<Vec<Tensor>> {
    let payloads = requests
        .iter()
        .map(|(module, arguments)| {
            let mut payload = format!("{}\n", arguments.len());
            for argument in arguments {
                payload.push_str(&argument.encode());
                payload.push('\n');
            }
            payload.push_str(module);
            payload
        })
        .collect::<Vec<_>>();
    upstream_support::invoke(&payloads, "--execute")
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            let result = match result {
                UpstreamResult::Accepted(result) => result,
                UpstreamResult::Rejected(error) => {
                    panic!("upstream execution of case {index} failed: {error}")
                }
            };
            result.lines().map(Tensor::decode).collect()
        })
        .collect()
}
