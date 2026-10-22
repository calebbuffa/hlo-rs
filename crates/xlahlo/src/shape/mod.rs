//! Shape, dimension, and physical-layout operations for native HLO types.
//!
//! Shapes are interned `pliron::r#type::TypeHandle` values, not owned shape
//! objects. Construction and transformation functions return a handle in the
//! supplied context; transformations leave the original type unchanged.
//! Query functions borrow interned metadata where possible. All handles passed
//! to a function must belong to its context.
//!
//! ```
//! use xlahlo::{pliron::context::Context, shape, types::ElementType};
//!
//! let context = Context::new();
//! let array = shape::array(&context, ElementType::F32, vec![2, 3])?;
//! assert_eq!(&*shape::dimensions(&context, array)?, &[2, 3]);
//! assert_eq!(shape::element_count(&context, array)?, 6);
//! # Ok::<(), xlahlo::pliron::result::Error>(())
//! ```

mod comparison;
mod construction;
mod dimension;
mod dimension_ops;
mod layout;
mod query;
mod storage;

pub use comparison::{
    compatible, compatible_ignoring_fp_precision, equal_ignoring_layout,
    equal_ignoring_layout_and_fp_precision, equal_including_layout,
};
pub use construction::{array, buffer, dynamic_array, opaque, scalar, token, tuple};
pub use dimension::{Dimension, UNBOUNDED_SIZE};
pub use dimension_ops::{
    delete_dimension, delete_dimensions, filter_dimensions, layout, with_default_layout,
    with_dimension_bound, with_dynamic_dimension, with_element_type, with_layout, without_layout,
};
pub use layout::{
    COMBINE_DIMENSION, DimLevelKind, Layout, MATCH_DIMENSION, SplitConfig, Tile, layouts_equal,
    validate_layout,
};

pub use query::{
    array_view, children, dimension, dimensions, element_count, element_type, has_complete_layout,
    has_unbounded, is_array, is_array_or_buffer, is_dynamic, is_tuple, leaf_count, primitive_type,
    rank,
};
pub use storage::{array_data_size, byte_size, reshape_is_bitcast, subshape};
