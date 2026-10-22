//! Shape and element-type inference for CHLO operations.
//!
//! Dimensions are zero-based axis sizes; a negative size denotes a dynamic
//! dimension. An unranked tensor is distinct from a rank-zero tensor (a scalar).
//! These helpers implement individual inference rules, not complete operation
//! verification. In particular, shape-only helpers do not check element domains.

use crate::Diagnostic;
use pliron::{
    builtin::{attributes::VecAttr, types::Signedness},
    context::Context,
    r#type::{Type as _, TypeHandle},
};
use stablehlo::{
    attributes::TypeExtensionsAttr,
    types::{ElementType, QuantizedElementType, TensorType, UnrankedTensorType},
};

#[derive(Clone)]
pub(crate) struct TensorInfo {
    pub dimensions: Vec<i64>,
}

pub(crate) fn tensor_info(ctx: &Context, ty: TypeHandle) -> Option<TensorInfo> {
    let value = ty.deref(ctx);
    let tensor = value.downcast_ref::<TensorType>()?;
    Some(TensorInfo {
        dimensions: tensor.dimensions.clone(),
    })
}

pub(crate) fn ty(ctx: &Context, shape: Vec<i64>, element: ElementType) -> TypeHandle {
    TensorType::ranked(ctx, shape, element)
}

fn merge(a: i64, b: i64) -> Option<i64> {
    if a >= 0 && b >= 0 {
        if a == b {
            Some(a)
        } else if a == 1 {
            Some(b)
        } else if b == 1 {
            Some(a)
        } else {
            None
        }
    } else if a > 1 {
        Some(a)
    } else if b > 1 || a == 1 {
        Some(b)
    } else if b == 1 {
        Some(a)
    } else {
        Some(-1)
    }
}

/// Tests equal rank and agreement of corresponding static dimension sizes.
///
/// Negative dimensions are wildcards. This does not apply broadcasting: a
/// dimension of size one is compatible only with one or a dynamic dimension.
pub fn compatible_shapes(lhs: &[i64], rhs: &[i64]) -> bool {
    lhs.len() == rhs.len() && lhs.iter().zip(rhs).all(|(a, b)| *a < 0 || *b < 0 || a == b)
}

/// Merges equal-rank shapes, choosing a known size over a dynamic size at each axis.
///
/// Returns `None` for no input shapes, different ranks, or conflicting static
/// dimensions. A list containing only a scalar shape produces `Some(vec![])`.
pub fn merge_compatible_shapes(shapes: &[&[i64]]) -> Option<Vec<i64>> {
    let mut result = shapes.first()?.to_vec();
    for shape in &shapes[1..] {
        if shape.len() != result.len() {
            return None;
        }
        for (out, next) in result.iter_mut().zip(*shape) {
            if *out < 0 {
                *out = *next;
            } else if *next >= 0 && *out != *next {
                return None;
            }
        }
    }
    Some(result)
}

/// Computes the shape obtained by trailing-axis NumPy broadcasting.
///
/// Missing leading axes have size one. Negative dimensions are dynamic; known
/// nonunit dimensions take precedence where possible. Returns `None` when
/// static dimensions conflict. No input shapes produce a scalar shape.
pub fn numpy_broadcast_shape(shapes: &[&[i64]]) -> Option<Vec<i64>> {
    let mut shape = Vec::new();
    for next in shapes {
        let rank = shape.len().max(next.len());
        let mut out = vec![1; rank];
        for i in 0..rank {
            let a = if i < rank - shape.len() {
                1
            } else {
                shape[i - (rank - shape.len())]
            };
            let b = if i < rank - next.len() {
                1
            } else {
                next[i - (rank - next.len())]
            };
            out[i] = merge(a, b)?;
        }
        shape = out;
    }
    Some(shape)
}

/// Computes a broadcast shape using an optional explicit axis mapping.
///
/// With `None`, uses [`numpy_broadcast_shape`]. Otherwise `dimensions` maps each
/// axis of the lower-rank operand into the higher-rank operand; the left operand
/// is mapped when ranks are equal. Returns `None` for a mapping with the wrong
/// length, a negative or out-of-range axis, or incompatible mapped sizes.
///
/// This helper does not validate axis uniqueness or ordering. Callers requiring
/// those constraints must check them separately.
pub fn infer_broadcast_shape(
    lhs: &[i64],
    rhs: &[i64],
    dimensions: Option<&[i64]>,
) -> Option<Vec<i64>> {
    let (large, small) = if lhs.len() > rhs.len() {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    if let Some(dims) = dimensions {
        if dims.len() != small.len() {
            return None;
        }
        let mut out = large.to_vec();
        for (j, &d) in dims.iter().enumerate() {
            let i = usize::try_from(d).ok()?;
            if i >= large.len() {
                return None;
            }
            out[i] = merge(small[j], large[i])?;
        }
        Some(out)
    } else {
        numpy_broadcast_shape(&[lhs, rhs])
    }
}

pub(crate) fn invalid(op: &'static str, reason: &'static str) -> pliron::result::Error {
    Diagnostic::InvalidOperation {
        operation: op.into(),
        location: "<verification>".into(),
        reason,
    }
    .into()
}

pub(crate) fn invalid_at(
    op: &'static str,
    location: String,
    reason: &'static str,
) -> pliron::result::Error {
    Diagnostic::InvalidOperation {
        operation: op.into(),
        location,
        reason,
    }
    .into()
}

/// Shape, element and encoding of a ranked or unranked tensor type.
#[derive(Clone)]
pub struct Shaped {
    /// Axis sizes, or `None` for unranked tensors; `Some(vec![])` denotes a scalar.
    pub dimensions: Option<Vec<i64>>,
    /// Native IR element type, including quantized element types.
    pub element: TypeHandle,
    /// Ranked tensor encoding attributes; empty for unranked tensors.
    pub encoding: VecAttr,
    /// Bounds from a `stablehlo.type_extensions` encoding.
    pub bounds: Option<Vec<i64>>,
}

impl Shaped {
    /// Returns the number of axes, or `None` for an unranked tensor.
    pub fn rank(&self) -> Option<usize> {
        self.dimensions.as_ref().map(Vec::len)
    }
    /// Tests whether a type-extensions bounds encoding is present.
    ///
    /// Presence does not imply that every axis has a finite bound.
    pub fn is_bounded(&self) -> bool {
        self.bounds.is_some()
    }
    fn bound(&self, dim: usize) -> i64 {
        self.bounds
            .as_ref()
            .and_then(|b| b.get(dim).copied())
            .unwrap_or(-1)
    }
}

/// Copies shape, element type, encoding, and bounds from an HLO tensor type.
///
/// Returns `None` for a nontensor type. The returned metadata is an owned snapshot,
/// not a live view into the context, and extracting it does not verify the type.
pub fn shaped(ctx: &Context, ty: TypeHandle) -> Option<Shaped> {
    let value = ty.deref(ctx);
    if let Some(tensor) = value.downcast_ref::<TensorType>() {
        return Some(Shaped {
            dimensions: Some(tensor.dimensions.clone()),
            element: tensor.element,
            encoding: tensor.encoding.clone(),
            bounds: tensor.bounds().map(<[i64]>::to_vec),
        });
    }
    let unranked = value.downcast_ref::<UnrankedTensorType>()?;
    Some(Shaped {
        dimensions: None,
        element: unranked.element,
        encoding: VecAttr::new(vec![]),
        bounds: None,
    })
}

/// Element-type families of the upstream HLO tensor constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// `HLO_Float` (no `tf32`).
    Float,
    /// `HLO_Complex` (`complex<f32>`, `complex<f64>`).
    Complex,
    /// Either a permitted floating-point element or a complex element.
    FloatOrComplex,
    /// `HLO_Int`: signless/unsigned 2/4/8/16/32/64-bit integers.
    Int,
    /// `HLO_Pred` (`i1`).
    Pred,
    /// A predicate or a supported signless/unsigned integer element.
    PredOrInt,
    /// `HLO_Tensor` elements: float, pred, int, complex and per-tensor quantized.
    Tensor,
    /// `HLO_AnyTensor` elements: `Tensor` plus per-axis quantized.
    Any,
}

/// Tests membership in the HLO floating-point family, excluding `tf32`.
pub fn hlo_float(element: ElementType) -> bool {
    element != ElementType::Tf32 && element.is_float()
}
/// Tests for a signless or unsigned integer of width 2, 4, 8, 16, 32, or 64.
///
/// Predicates (`i1`) and explicitly signed integer types are excluded.
pub fn hlo_int(element: ElementType) -> bool {
    matches!(
        element.integer_parts(),
        Some((
            2 | 4 | 8 | 16 | 32 | 64,
            Signedness::Signless | Signedness::Unsigned
        ))
    )
}
/// Tests for `complex<f32>` or `complex<f64>`.
pub fn hlo_complex(element: ElementType) -> bool {
    matches!(element, ElementType::C64 | ElementType::C128)
}

fn quantized(ctx: &Context, element: TypeHandle) -> Option<QuantizedElementType> {
    element
        .deref(ctx)
        .downcast_ref::<QuantizedElementType>()
        .cloned()
}

fn storage_limits(storage: ElementType) -> Option<(i128, i128)> {
    let (width, signedness) = storage.integer_parts()?;
    if !matches!(width, 2 | 4 | 8 | 16 | 32) {
        return None;
    }
    Some(match signedness {
        Signedness::Unsigned => (0, (1_i128 << width) - 1),
        Signedness::Signless => (-(1_i128 << (width - 1)), (1_i128 << (width - 1)) - 1),
        Signedness::Signed => return None,
    })
}

fn effective_storage_range(ctx: &Context, q: &QuantizedElementType) -> Option<(i128, i128)> {
    let (lo, hi) = storage_limits(q.storage_type(ctx)?)?;
    let (min, max) = q.storage_range();
    Some((min.unwrap_or(lo), max.unwrap_or(hi)))
}

/// Upstream `HLO_QuantizedInt`/`HLO_PerAxisQuantizedInt` element validity.
fn valid_quantized(ctx: &Context, q: &QuantizedElementType) -> bool {
    let Some(storage) = q.storage_type(ctx) else {
        return false;
    };
    let Some((lo, hi)) = storage_limits(storage) else {
        return false;
    };
    let (min, max) = effective_storage_range(ctx, q).unwrap_or((lo, hi));
    let scales = q.scale_bits.len();
    lo <= min
        && min < max
        && max <= hi
        && q.expressed_type(ctx).is_some_and(ElementType::is_float)
        && q.axis >= -1
        && scales > 0
        && scales == q.zero_points.len()
        && (q.axis >= 0 || scales == 1)
        && q.scales().all(|s| s.is_finite() && s > 0.0)
        && q.zero_points().all(|z| (min..=max).contains(&z))
}

/// Tests whether a native element type satisfies an HLO element domain.
///
/// Quantized types are accepted only by [`Domain::Tensor`] (per-tensor) or
/// [`Domain::Any`] (per-tensor or per-axis), and their storage range, scales, and
/// zero points must be valid. Tensor shape and quantized-axis size are not checked.
pub fn element_in(ctx: &Context, element: TypeHandle, domain: Domain) -> bool {
    if let Some(q) = quantized(ctx, element) {
        return match domain {
            Domain::Tensor => q.axis == -1 && valid_quantized(ctx, &q),
            Domain::Any => valid_quantized(ctx, &q),
            _ => false,
        };
    }
    let Some(e) = stablehlo::types::primitive_element(ctx, element) else {
        return false;
    };
    match domain {
        Domain::Float => hlo_float(e),
        Domain::Complex => hlo_complex(e),
        Domain::FloatOrComplex => hlo_float(e) || hlo_complex(e),
        Domain::Int => hlo_int(e),
        Domain::Pred => e == ElementType::I1,
        Domain::PredOrInt => e == ElementType::I1 || hlo_int(e),
        Domain::Tensor | Domain::Any => {
            hlo_float(e) || hlo_complex(e) || hlo_int(e) || e == ElementType::I1
        }
    }
}

/// Upstream `isValidQuantizedDimension`: per-axis quantized ranked tensors need
/// an in-range axis whose static size matches the scale count.
///
/// Dynamic sizes defer the size check. Returns `true` for nontensors, unranked
/// tensors, and types without a per-axis quantized element. This is not a general
/// tensor or quantization validity check.
pub fn valid_quantized_dimension(ctx: &Context, ty: TypeHandle) -> bool {
    let Some(s) = shaped(ctx, ty) else {
        return true;
    };
    let Some(dims) = s.dimensions else {
        return true;
    };
    let Some(q) = quantized(ctx, s.element) else {
        return true;
    };
    let Ok(axis) = usize::try_from(q.axis) else {
        return true;
    };
    axis < dims.len()
        && (dims[axis] < 0 || usize::try_from(dims[axis]).is_ok_and(|d| d == q.scale_bits.len()))
}

/// Tests tensor membership, element-domain validity, and quantized-axis validity.
///
/// `ranked_only` requires a known rank, not a fully static shape. When `false`,
/// both ranked and unranked HLO tensors are accepted.
pub fn hlo_tensor_type(ctx: &Context, ty: TypeHandle, domain: Domain, ranked_only: bool) -> bool {
    let Some(s) = shaped(ctx, ty) else {
        return false;
    };
    (!ranked_only || s.dimensions.is_some())
        && element_in(ctx, s.element, domain)
        && valid_quantized_dimension(ctx, ty)
}

/// Tests HLO inference compatibility of element types or tensor element types.
///
/// Nonquantized elements must be identical. Quantized elements must share storage
/// and expressed types, effective storage ranges, and per-axis/per-tensor mode;
/// scale values, zero points, and the exact per-axis index are not compared.
pub fn compatible_elements(ctx: &Context, lhs: TypeHandle, rhs: TypeHandle) -> bool {
    let lhs = shaped(ctx, lhs).map_or(lhs, |s| s.element);
    let rhs = shaped(ctx, rhs).map_or(rhs, |s| s.element);
    match (quantized(ctx, lhs), quantized(ctx, rhs)) {
        (Some(a), Some(b)) => {
            a.storage == b.storage
                && a.expressed == b.expressed
                && effective_storage_range(ctx, &a) == effective_storage_range(ctx, &b)
                && (a.axis >= 0) == (b.axis >= 0)
        }
        (None, None) => lhs == rhs,
        _ => false,
    }
}

/// Tests shape compatibility, treating an unranked tensor as a wildcard.
///
/// Ranked tensors must have equal ranks and matching static sizes. Bounds,
/// encodings, and element types are ignored.
pub fn compatible_shape(lhs: &Shaped, rhs: &Shaped) -> bool {
    match (&lhs.dimensions, &rhs.dimensions) {
        (Some(a), Some(b)) => compatible_shapes(a, b),
        _ => true,
    }
}

/// Tests compatible ranks and sizes, including each tensor's dimension bounds.
///
/// A known size must not exceed the corresponding bound on the other tensor.
/// Unranked tensors remain compatible with every shape.
pub fn compatible_shapes_with_bounds(lhs: &Shaped, rhs: &Shaped) -> bool {
    if !compatible_shape(lhs, rhs) {
        return false;
    }
    let (Some(a), Some(b)) = (&lhs.dimensions, &rhs.dimensions) else {
        return true;
    };
    let fits = |dims: &[i64], other: &Shaped| {
        dims.iter()
            .enumerate()
            .all(|(i, &d)| other.bound(i) == -1 || d <= other.bound(i))
    };
    fits(a, rhs) && fits(b, lhs)
}

/// Tests HLO inference compatibility of two native types.
///
/// For two tensors, checks bounds-aware shape compatibility and element
/// compatibility. Otherwise checks only [`compatible_elements`]. This is a
/// compatibility relation, not type equality or complete type verification.
pub fn compatible_types(ctx: &Context, lhs: TypeHandle, rhs: TypeHandle) -> bool {
    match (shaped(ctx, lhs), shaped(ctx, rhs)) {
        (Some(a), Some(b)) => {
            compatible_shapes_with_bounds(&a, &b) && compatible_elements(ctx, a.element, b.element)
        }
        _ => compatible_elements(ctx, lhs, rhs),
    }
}

/// Creates a type-extensions encoding for per-axis bounds.
///
/// A bound of `-1` denotes no known bound. If every entry is `-1` (including an
/// empty list), returns an empty encoding instead. The caller must ensure the
/// bound count matches the ranked tensor and that bound values are valid.
pub fn bounds_encoding(bounds: Vec<i64>) -> VecAttr {
    if bounds.iter().all(|&b| b == -1) {
        VecAttr::new(vec![])
    } else {
        VecAttr::new(vec![Box::new(TypeExtensionsAttr { bounds })])
    }
}

/// Interns an unranked HLO tensor with the given native element type.
///
/// Does not validate the element type or attach a ranked tensor encoding.
pub fn unranked(ctx: &Context, element: TypeHandle) -> TypeHandle {
    UnrankedTensorType::instantiate(UnrankedTensorType { element }, ctx).into()
}

/// Ranked type with the given encoding, or unranked when `dimensions` is `None`.
///
/// Interns the type without verifying dimensions, bounds, or element validity.
/// Unranked types do not retain `encoding`.
pub fn shaped_type(
    ctx: &Context,
    dimensions: Option<Vec<i64>>,
    element: TypeHandle,
    encoding: VecAttr,
) -> TypeHandle {
    match dimensions {
        Some(dimensions) => TensorType::instantiate(
            TensorType {
                dimensions,
                element,
                encoding,
            },
            ctx,
        )
        .into(),
        None => unranked(ctx, element),
    }
}

/// Interns a tensor with the shape and encoding of `ty` and a new element type.
///
/// Preserves ranked/unranked status. Returns `None` if `ty` is not an HLO tensor;
/// does not verify the replacement element type.
pub fn same_shape(ctx: &Context, ty: TypeHandle, element: TypeHandle) -> Option<TypeHandle> {
    let s = shaped(ctx, ty)?;
    Some(shaped_type(ctx, s.dimensions, element, s.encoding))
}

/// Infers a tensor shape by combining the known dimensions and bounds of `types`.
///
/// Static sizes refine dynamic dimensions; dynamic bounds are intersected.
/// Unranked inputs do not constrain the rank. The element type and non-bounds
/// encoding come from the first ranked input. If all inputs are unranked, returns
/// the first input unchanged.
///
/// Returns `None` for empty input, nontensor types, conflicting ranks or static
/// sizes, or a known size exceeding a relevant bound. Callers must separately
/// establish element-type compatibility; this function does not check it.
pub fn most_specific(ctx: &Context, types: &[TypeHandle]) -> Option<TypeHandle> {
    let all = types
        .iter()
        .map(|&t| shaped(ctx, t))
        .collect::<Option<Vec<_>>>()?;
    let ranked: Vec<&Shaped> = all.iter().filter(|s| s.dimensions.is_some()).collect();
    let Some(first) = ranked.first() else {
        return types.first().copied();
    };
    let mut sizes = first.dimensions.clone()?;
    let rank = sizes.len();
    let mut bounds = first.bounds.clone().unwrap_or_else(|| vec![-1; rank]);
    bounds.resize(rank, -1);
    let any_bounds = first.bounds.is_some();
    for next in &ranked[1..] {
        let dims = next.dimensions.as_ref()?;
        if dims.len() != rank {
            return None;
        }
        for d in 0..rank {
            let (ls, rs, lb, rb) = (sizes[d], dims[d], bounds[d], next.bound(d));
            if ls >= 0 || rs >= 0 {
                if ls >= 0 && rs >= 0 && ls != rs {
                    return None;
                }
                let size = if ls >= 0 { ls } else { rs };
                let check = if lb >= 0 { lb } else { rb };
                if check >= 0 && size > check {
                    return None;
                }
                sizes[d] = size;
                bounds[d] = -1;
            } else {
                bounds[d] = match (lb >= 0, rb >= 0) {
                    (true, true) => lb.min(rb),
                    (true, false) => lb,
                    _ => rb,
                };
            }
        }
    }
    let encoding = if any_bounds {
        bounds_encoding(bounds)
    } else {
        first.encoding.clone()
    };
    Some(shaped_type(ctx, Some(sizes), first.element, encoding))
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Dim {
    size: i64,
    bounded: bool,
}

fn bounded_dims(s: &Shaped, dims: &[i64]) -> Vec<Dim> {
    dims.iter()
        .enumerate()
        .map(|(i, &d)| {
            if s.is_bounded() && d < 0 {
                Dim {
                    size: s.bound(i),
                    bounded: true,
                }
            } else {
                Dim {
                    size: d,
                    bounded: false,
                }
            }
        })
        .collect()
}

fn bounded_broadcast(a: &[Dim], b: &[Dim]) -> Option<Vec<Dim>> {
    let rank = a.len().max(b.len());
    let mut out = vec![
        Dim {
            size: 1,
            bounded: false
        };
        rank
    ];
    let one = Dim {
        size: 1,
        bounded: false,
    };
    for i in 1..=rank {
        let da = if i <= a.len() { a[a.len() - i] } else { one };
        let db = if i <= b.len() { b[b.len() - i] } else { one };
        out[rank - i] = if da.size == 1 {
            db
        } else if db.size == 1 {
            da
        } else if da.size != db.size || da.bounded != db.bounded {
            return None;
        } else {
            da
        };
    }
    Some(out)
}

/// Infers a tensor type using trailing-axis NumPy broadcasting and `element`.
///
/// Preserves inferred dynamic bounds when all input dimensions are static or
/// bounded. Returns an unranked type for empty input, any unranked operand, or
/// incompatible shapes; an illegal broadcast is not an error from this helper.
/// Does not check operand element types against the requested output element.
pub fn numpy_broadcast_type(
    ctx: &Context,
    operands: &[&Shaped],
    element: TypeHandle,
) -> TypeHandle {
    let mut ranked = Vec::with_capacity(operands.len());
    for s in operands {
        match &s.dimensions {
            Some(d) => ranked.push((*s, d.as_slice())),
            None => return unranked(ctx, element),
        }
    }
    if ranked.is_empty() {
        return unranked(ctx, element);
    }
    let static_or_bounded = ranked
        .iter()
        .all(|(s, d)| s.is_bounded() || d.iter().all(|&x| x >= 0));
    if static_or_bounded {
        let mut shape = bounded_dims(ranked[0].0, ranked[0].1);
        for (s, d) in &ranked[1..] {
            match bounded_broadcast(&shape, &bounded_dims(s, d)) {
                Some(next) => shape = next,
                None => return unranked(ctx, element),
            }
        }
        let dims = shape
            .iter()
            .map(|d| if d.bounded { -1 } else { d.size })
            .collect();
        let bounds = shape
            .iter()
            .map(|d| if d.bounded { d.size } else { -1 })
            .collect();
        return shaped_type(ctx, Some(dims), element, bounds_encoding(bounds));
    }
    let shapes: Vec<&[i64]> = ranked.iter().map(|(_, d)| *d).collect();
    match numpy_broadcast_shape(&shapes) {
        Some(shape) => shaped_type(ctx, Some(shape), element, VecAttr::new(vec![])),
        None => unranked(ctx, element),
    }
}

/// Infers a broadcast tensor type with an optional explicit axis mapping.
///
/// `None` delegates to [`numpy_broadcast_type`]. Explicit dimensions map the
/// lower-rank operand into the higher-rank operand (left into right on equal
/// ranks). The explicit-mapping path does not propagate encodings or bounds.
///
/// Returns an unranked type when either operand is unranked, the mapping has
/// invalid length or indices, or mapped sizes conflict. Mapping uniqueness and
/// ordering and operand element-type compatibility are not checked.
pub fn broadcast_type(
    ctx: &Context,
    lhs: &Shaped,
    rhs: &Shaped,
    dimensions: Option<&[i64]>,
    element: TypeHandle,
) -> TypeHandle {
    let Some(dims) = dimensions else {
        return numpy_broadcast_type(ctx, &[lhs, rhs], element);
    };
    let (Some(x), Some(y)) = (&lhs.dimensions, &rhs.dimensions) else {
        return unranked(ctx, element);
    };
    let large = if x.len() > y.len() { x } else { y };
    let small = if x.len() <= y.len() { x } else { y };
    if dims.len() != small.len() {
        return unranked(ctx, element);
    }
    let mut out = large.clone();
    for (j, &d) in dims.iter().enumerate() {
        let Some(i) = usize::try_from(d).ok().filter(|&i| i < large.len()) else {
            return unranked(ctx, element);
        };
        let Some(merged) = merge(small[j], large[i]) else {
            return unranked(ctx, element);
        };
        out[i] = merged;
    }
    shaped_type(ctx, Some(out), element, VecAttr::new(vec![]))
}
