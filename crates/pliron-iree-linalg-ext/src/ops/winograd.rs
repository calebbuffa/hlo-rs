//! Winograd convolution transforms.

use crate::support::*;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_mlir_builtin::attributes::I64ArrayAttr;
use pliron_mlir_builtin::attributes::IndexAttr;

/// Tile parameters shared by the three transforms.
struct Params {
    output_tile: i64,
    kernel: i64,
    dims: Vec<i64>,
}

impl Params {
    fn input_tile(&self) -> i128 {
        i128::from(self.output_tile) + i128::from(self.kernel) - 1
    }
}

/// A ranked tensor's dimensions and element type.
type Shape = (Vec<i64>, TypeHandle);

/// Reads the parameters and the input and output shapes and elements.
fn prepare(
    ctx: &Context,
    op: Ptr<Operation>,
    output_tile: Option<i64>,
    kernel: Option<i64>,
    dims: Option<Vec<i64>>,
) -> Result<(Params, Shape, Shape)> {
    no_regions(ctx, op)?;
    ensure(
        ctx,
        op,
        num_operands(ctx, op) == 2,
        "expected one input and one output operand",
    )?;
    verify_outputs(ctx, op, 1..2)?;
    let params = Params {
        output_tile: output_tile.ok_or_else(|| fail(ctx, op, "missing output_tile_size"))?,
        kernel: kernel.ok_or_else(|| fail(ctx, op, "missing kernel_size"))?,
        dims: dims.ok_or_else(|| fail(ctx, op, "missing dimension list"))?,
    };
    let types = operand_types(ctx, op);
    let input = ranked_operand(ctx, op, types[0])?;
    let output = ranked_operand(ctx, op, types[1])?;
    Ok((params, input, output))
}

/// Returns whether image dimensions are `[2, 3]` (NCHW), failing unless they
/// are `[1, 2]` (NHWC) or `[2, 3]`.
fn is_nchw(ctx: &Context, op: Ptr<Operation>, dims: &[i64]) -> Result<bool> {
    ensure(ctx, op, dims.len() == 2, "expected only 2 image dimensions")?;
    match dims {
        [1, 2] => Ok(false),
        [2, 3] => Ok(true),
        _ => Err(fail(
            ctx,
            op,
            "expect image dimensions to be either [1, 2] or [2, 3]",
        )),
    }
}

fn compatible_expected(expected: &[Option<i128>], actual: &[i64]) -> bool {
    expected.len() == actual.len()
        && expected.iter().zip(actual).all(|(&expected, &actual)| {
            actual < 0 || expected.is_none_or(|n| n == i128::from(actual))
        })
}

fn extent(d: i64) -> Option<i128> {
    (d >= 0).then_some(i128::from(d))
}

fn ceil_div(n: i128, d: i128) -> i128 {
    let (q, r) = (n / d, n % d);
    q + i128::from(r != 0 && (r > 0) == (d > 0))
}

macro_rules! winograd_op {
    (
        $(#[$doc:meta])*
        $struct:ident, $name:literal, $dims_name:ident,
        $tile:ident, $get_tile:ident, $set_tile:ident,
        $kernel:ident, $get_kernel:ident, $set_kernel:ident,
        $dims:ident, $get_dims:ident, $set_dims:ident
    ) => {
        $(#[$doc])*
        #[pliron_op(
            name = $name,
            format = "operands(CharSpace(`,`)) attr_dict `:` typesig"
        )]
        pub struct $struct;
        pliron_arith::shared_attr!($struct, super::$tile, IndexAttr, $get_tile, $set_tile);
        pliron_arith::shared_attr!($struct, super::$kernel, IndexAttr, $get_kernel, $set_kernel);
        pliron_arith::shared_attr!($struct, super::$dims, I64ArrayAttr, $get_dims, $set_dims);

        impl $struct {
            /// Allocates the operation without verifying it.
            pub fn new(
                ctx: &mut Context,
                input: Value,
                output: Value,
                output_tile_size: i64,
                kernel_size: i64,
                $dims_name: Vec<i64>,
            ) -> Self {
                let results = vec![output.get_type(ctx)];
                let op = new_plain(ctx, |op| Self { op }, results, vec![input, output]);
                op.$set_tile(ctx, IndexAttr { value: output_tile_size });
                op.$set_kernel(ctx, IndexAttr { value: kernel_size });
                op.$set_dims(ctx, I64ArrayAttr::new($dims_name));
                op
            }

            /// Builds and verifies the operation.
            ///
            /// # Errors
            ///
            /// Returns an error when verification fails.
            pub fn build(
                ctx: &mut Context,
                input: Value,
                output: Value,
                output_tile_size: i64,
                kernel_size: i64,
                $dims_name: Vec<i64>,
            ) -> Result<Self> {
                let op = Self::new(ctx, input, output, output_tile_size, kernel_size, $dims_name);
                op.verify(ctx)?;
                Ok(op)
            }

            /// Returns the output tile size `m`.
            pub fn output_tile_size(&self, ctx: &Context) -> Option<i64> {
                self.$get_tile(ctx).map(|a| a.value)
            }

            /// Returns the kernel size `r`.
            pub fn kernel_size(&self, ctx: &Context) -> Option<i64> {
                self.$get_kernel(ctx).map(|a| a.value)
            }

            /// Returns `m + r - 1`, or `None` for missing attributes or i64 overflow.
            pub fn input_tile_size(&self, ctx: &Context) -> Option<i64> {
                i64::try_from(i128::from(self.output_tile_size(ctx)?) + i128::from(self.kernel_size(ctx)?) - 1).ok()
            }

            #[doc = concat!("Returns the `", stringify!($dims_name), "` attribute.")]
            pub fn $dims_name(&self, ctx: &Context) -> Vec<i64> {
                self.$get_dims(ctx).map(|a| a.values.clone()).unwrap_or_default()
            }

            fn params(
                &self,
                ctx: &Context,
            ) -> Result<(Params, (Vec<i64>, TypeHandle), (Vec<i64>, TypeHandle))> {
                prepare(
                    ctx,
                    self.get_operation(),
                    self.output_tile_size(ctx),
                    self.kernel_size(ctx),
                    self.$get_dims(ctx).map(|a| a.values.clone()),
                )
            }
        }
    };
}

winograd_op!(
    /// Winograd input transform: tiles an image into `(m + r - 1)^2` input
    /// tiles and applies `B^T x B` to each.
    ///
    /// A rank-2 input is a single tile. A rank-4 NHWC (`image_dimensions =
    /// [1, 2]`) or NCHW (`[2, 3]`) image produces
    /// `[m + r - 1, m + r - 1, N, H', W', C]` with
    /// `H' = ceil((H - r + 1) / m)`.
    WinogradInputTransformOp, "iree_linalg_ext.winograd_input_transform", image_dimensions,
    WINOGRAD_INPUT_TILE,
    get_attr_iree_winograd_input_transform_output_tile_size,
    set_attr_iree_winograd_input_transform_output_tile_size,
    WINOGRAD_INPUT_KERNEL,
    get_attr_iree_winograd_input_transform_kernel_size,
    set_attr_iree_winograd_input_transform_kernel_size,
    WINOGRAD_INPUT_DIMS,
    get_attr_iree_winograd_input_transform_image_dimensions,
    set_attr_iree_winograd_input_transform_image_dimensions
);

winograd_op!(
    /// Winograd filter transform: applies `G g G^T` to each kernel.
    ///
    /// A rank-2 input is a single `r x r` kernel. A rank-4 HWCF
    /// (`kernel_dimensions = [0, 1]`) or FCHW (`[2, 3]`) filter produces
    /// `[m + r - 1, m + r - 1, C, F]`.
    WinogradFilterTransformOp, "iree_linalg_ext.winograd_filter_transform", kernel_dimensions,
    WINOGRAD_FILTER_TILE,
    get_attr_iree_winograd_filter_transform_output_tile_size,
    set_attr_iree_winograd_filter_transform_output_tile_size,
    WINOGRAD_FILTER_KERNEL,
    get_attr_iree_winograd_filter_transform_kernel_size,
    set_attr_iree_winograd_filter_transform_kernel_size,
    WINOGRAD_FILTER_DIMS,
    get_attr_iree_winograd_filter_transform_kernel_dimensions,
    set_attr_iree_winograd_filter_transform_kernel_dimensions
);

winograd_op!(
    /// Winograd output transform: applies `A^T y A` to each tile and
    /// reassembles the image.
    ///
    /// A rank-2 input is a single tile. A rank-6
    /// `[m + r - 1, m + r - 1, N, H', W', C]` input produces an NHWC
    /// (`image_dimensions = [1, 2]`) or NCHW (`[2, 3]`) image with
    /// `H = m * H'`.
    WinogradOutputTransformOp, "iree_linalg_ext.winograd_output_transform", image_dimensions,
    WINOGRAD_OUTPUT_TILE,
    get_attr_iree_winograd_output_transform_output_tile_size,
    set_attr_iree_winograd_output_transform_output_tile_size,
    WINOGRAD_OUTPUT_KERNEL,
    get_attr_iree_winograd_output_transform_kernel_size,
    set_attr_iree_winograd_output_transform_kernel_size,
    WINOGRAD_OUTPUT_DIMS,
    get_attr_iree_winograd_output_transform_image_dimensions,
    set_attr_iree_winograd_output_transform_image_dimensions
);

impl Verify for WinogradInputTransformOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let (params, (input, input_element), (output, output_element)) = self.params(ctx)?;
        ensure(
            ctx,
            op,
            input_element == output_element,
            "expected input/output element types to be identical",
        )?;
        let tile = params.input_tile();
        match input.len() {
            2 => {
                ensure(
                    ctx,
                    op,
                    output.len() == 2,
                    "expected output operand to have rank 2 if input is of rank 2",
                )?;
                ensure(
                    ctx,
                    op,
                    input.iter().all(|&d| d < 0 || i128::from(d) <= tile),
                    "expected input dims not greater than input tile size if input is of rank 2",
                )?;
                ensure(
                    ctx,
                    op,
                    compatible_expected(&[Some(tile), Some(tile)], &output),
                    "expected output dims equal to inputTileSize if input is of rank 2",
                )
            }
            4 => {
                ensure(
                    ctx,
                    op,
                    output.len() == 6,
                    "expected output rank to be equal to input rank + 2",
                )?;
                let nchw = is_nchw(ctx, op, &params.dims)?;
                let mut expected = vec![Some(tile), Some(tile)];
                for (i, &d) in input.iter().enumerate() {
                    if d < 0 || !params.dims.contains(&(i as i64)) {
                        expected.push(extent(d));
                    } else {
                        ensure(
                            ctx,
                            op,
                            params.output_tile != 0,
                            "cannot infer Winograd image tiling with a zero output_tile_size",
                        )?;
                        expected.push(Some(ceil_div(
                            i128::from(d) - i128::from(params.kernel) + 1,
                            i128::from(params.output_tile),
                        )));
                    }
                }
                if nchw {
                    expected[3..].rotate_left(1);
                }
                ensure(
                    ctx,
                    op,
                    compatible_expected(&expected, &output),
                    "incompatible output shape",
                )
            }
            _ => Err(fail(
                ctx,
                op,
                "expected input operand to have rank either 2 or 4",
            )),
        }
    }
}

impl Verify for WinogradFilterTransformOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let (params, (input, input_element), (output, output_element)) = self.params(ctx)?;
        ensure(
            ctx,
            op,
            input_element == output_element,
            "expected input/output element types to be identical",
        )?;
        let (tile, kernel) = (params.input_tile(), params.kernel);
        match input.len() {
            2 => {
                ensure(
                    ctx,
                    op,
                    output.len() == 2,
                    "expected output operand to have rank 2 if input is of rank 2",
                )?;
                ensure(
                    ctx,
                    op,
                    compatible_expected(&[Some(i128::from(kernel)); 2], &input),
                    "expected input dims to be equal to kernel size if input is of rank 2",
                )?;
                ensure(
                    ctx,
                    op,
                    compatible_expected(&[Some(tile); 2], &output),
                    "expected output dims equal to input tile size if input is of rank 2",
                )
            }
            4 => {
                ensure(
                    ctx,
                    op,
                    output.len() == 4,
                    "expected output rank to be equal to input rank",
                )?;
                ensure(
                    ctx,
                    op,
                    params.dims.len() == 2,
                    "expected only 2 kernel dimensions",
                )?;
                let fchw = match params.dims.as_slice() {
                    [0, 1] => false,
                    [2, 3] => true,
                    _ => {
                        return Err(fail(
                            ctx,
                            op,
                            "expect kernel dimensions to be either [0, 1] or [2, 3]",
                        ));
                    }
                };
                ensure(
                    ctx,
                    op,
                    params.dims.iter().all(|&d| input[d as usize] == kernel),
                    "expect all kernel dimensions to have the kernel size",
                )?;
                let mut expected = vec![Some(tile); 2];
                expected.extend(
                    input
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !params.dims.contains(&(*i as i64)))
                        .map(|(_, &d)| extent(d)),
                );
                if fchw {
                    expected.swap(2, 3);
                }
                ensure(
                    ctx,
                    op,
                    compatible_expected(&expected, &output),
                    "incompatible output shape",
                )
            }
            _ => Err(fail(
                ctx,
                op,
                "expected input operand to have rank either 2 or 4",
            )),
        }
    }
}

impl Verify for WinogradOutputTransformOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let (params, (input, input_element), (output, output_element)) = self.params(ctx)?;
        let tile = params.input_tile();
        match input.len() {
            2 => {
                ensure(
                    ctx,
                    op,
                    output.len() == 2,
                    "expected output operand to have rank 2 if input is of rank 2",
                )?;
                ensure(
                    ctx,
                    op,
                    compatible_expected(&[Some(tile); 2], &input),
                    "expected input dims to be equal to input tile size if input is of rank 2",
                )?;
                ensure(
                    ctx,
                    op,
                    compatible_expected(&[Some(i128::from(params.output_tile)); 2], &output),
                    "expected output dims equal to output tile size if input is of rank 2",
                )
            }
            6 => {
                ensure(
                    ctx,
                    op,
                    input_element == output_element,
                    "expected input/output element types to be identical",
                )?;
                ensure(
                    ctx,
                    op,
                    output.len() == 4,
                    "expected output rank to be equal to input rank - 2",
                )?;
                let nchw = is_nchw(ctx, op, &params.dims)?;
                let mut shape = input.clone();
                if nchw {
                    shape[3..].rotate_right(1);
                }
                let expected: Vec<Option<i128>> = shape[2..]
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| {
                        if d < 0 || !params.dims.contains(&(i as i64)) {
                            extent(d)
                        } else {
                            Some(i128::from(params.output_tile) * i128::from(d))
                        }
                    })
                    .collect();
                ensure(
                    ctx,
                    op,
                    compatible_expected(&expected, &output),
                    "incompatible output shape",
                )
            }
            _ => Err(fail(
                ctx,
                op,
                "expected input operand to have rank either 2 or 6",
            )),
        }
    }
}
