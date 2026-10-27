#[path = "ops/semantics.rs"]
mod semantics;
#[path = "ops/tensor_semantics.rs"]
mod tensor_semantics;

use clienthlo::pliron::{builtin::attributes::VecAttr, context::Context, r#type::TypeHandle};
use stablehlo::types::{TensorType, UnrankedTensorType};

struct TensorMetadata {
    dimensions: Option<Vec<i64>>,
    encoding: VecAttr,
    bounds: Option<Vec<i64>>,
}

fn tensor_metadata(ctx: &Context, ty: TypeHandle) -> TensorMetadata {
    let object = ty.deref(ctx);
    if let Some(tensor) = object.downcast_ref::<TensorType>() {
        TensorMetadata {
            dimensions: Some(tensor.dimensions.clone()),
            encoding: tensor.encoding.clone(),
            bounds: tensor.bounds().map(<[i64]>::to_vec),
        }
    } else {
        assert!(object.downcast_ref::<UnrankedTensorType>().is_some());
        TensorMetadata {
            dimensions: None,
            encoding: VecAttr::new(vec![]),
            bounds: None,
        }
    }
}
