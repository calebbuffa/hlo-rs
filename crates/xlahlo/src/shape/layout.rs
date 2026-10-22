use crate::shape;
use crate::{Diagnostic, types::ElementType};
use pliron::{context::Context, derive::pliron_attr, result::Result, r#type::TypeHandle};

/// Tile marker combining a logical dimension with its neighbor.
pub const COMBINE_DIMENSION: i64 = i64::MIN;
/// Tile marker matching the corresponding bound.
pub const MATCH_DIMENSION: i64 = i64::MIN + 1;

#[pliron_attr(
    name = "xlahlo.tile",
    format = "`[` vec($dimensions, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Physical tiling dimensions in minor-to-major order.
pub struct Tile {
    /// Tile extent for each tiled axis, including supported tile markers.
    pub dimensions: Vec<i64>,
}

#[pliron_attr(
    name = "xlahlo.split_config",
    format = "`<` $dimension `,` `[` vec($split_indices, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Split boundaries for one physical dimension.
pub struct SplitConfig {
    /// Physical dimension to split.
    pub dimension: i64,
    /// Ordered split boundaries along the selected physical dimension.
    pub split_indices: Vec<i64>,
}

#[pliron_attr(name = "xlahlo.dim_level_type", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Sparse level annotations retained independently from layout equality.
pub enum DimLevelKind {
    /// Stores every coordinate in the level.
    Dense,
    /// Stores nonzero coordinates in a compressed segment.
    Compressed,
    /// Stores at most one coordinate per parent position.
    Singleton,
    /// Uses compressed segments whose storage may include unused positions.
    LooseCompressed,
}

#[pliron_attr(
    name = "xlahlo.layout",
    format = "`<` `[` vec($minor_to_major, CharSpace(`,`)) `]` `,` `[` vec($tiles, CharSpace(`,`)) `]` `,` $tail_padding_alignment_in_elements `,` $element_size_in_bits `,` $memory_space `,` `[` vec($index_primitive_type, CharSpace(`,`)) `]` `,` `[` vec($pointer_primitive_type, CharSpace(`,`)) `]` `,` `[` vec($physical_shape, CharSpace(`,`)) `]` `,` `[` vec($split_configs, CharSpace(`,`)) `]` `,` $dynamic_shape_metadata_prefix_bytes `,` `[` vec($dim_level_kinds, CharSpace(`,`)) `]` `,` `[` vec($dim_unique, CharSpace(`,`)) `]` `,` `[` vec($dim_ordered, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Complete physical layout, including tiling, padding, memory space, and sparse metadata.
///
/// Public fields support explicit IR construction. Call [`validate_layout`]
/// before consuming an independently constructed record. The current validator
/// rejects sparse index, pointer, and physical-shape fields; retaining these
/// fields in the representation does not imply sparse execution support.
/// Semantic equality intentionally ignores runtime metadata prefixes and
/// obsolete sparse level annotations.
pub struct Layout {
    /// Logical axes ordered from fastest-varying to slowest-varying in storage.
    pub minor_to_major: Vec<i64>,
    /// Nested tiling descriptions in physical layout order.
    pub tiles: Vec<Tile>,
    /// Allocation alignment in elements; defaults to one.
    pub tail_padding_alignment_in_elements: i64,
    /// Explicit packed element width, or zero to use the element format's width.
    pub element_size_in_bits: i64,
    /// Backend-defined address-space identifier; zero denotes the default space.
    pub memory_space: i64,
    /// Optional singleton sparse-index element format.
    pub index_primitive_type: Vec<ElementType>,
    /// Optional singleton sparse-pointer element format.
    pub pointer_primitive_type: Vec<ElementType>,
    /// Optional singleton type describing the physical storage representation.
    pub physical_shape: Vec<TypeHandle>,
    /// Physical-axis split boundaries.
    pub split_configs: Vec<SplitConfig>,
    /// Number of bytes reserved before array storage for runtime shape metadata.
    pub dynamic_shape_metadata_prefix_bytes: i64,
    /// Per-level sparse storage categories retained independently of layout equality.
    pub dim_level_kinds: Vec<DimLevelKind>,
    /// Per-level guarantees that stored coordinates are unique.
    pub dim_unique: Vec<bool>,
    /// Per-level guarantees that stored coordinates are ordered.
    pub dim_ordered: Vec<bool>,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            minor_to_major: Vec::new(),
            tiles: Vec::new(),
            tail_padding_alignment_in_elements: 1,
            element_size_in_bits: 0,
            memory_space: 0,
            index_primitive_type: Vec::new(),
            pointer_primitive_type: Vec::new(),
            physical_shape: Vec::new(),
            split_configs: Vec::new(),
            dynamic_shape_metadata_prefix_bytes: 0,
            dim_level_kinds: Vec::new(),
            dim_unique: Vec::new(),
            dim_ordered: Vec::new(),
        }
    }
}

impl Layout {
    /// Construct the conventional dense descending minor-to-major order.
    pub fn descending(rank: usize) -> Self {
        Self {
            minor_to_major: (0..rank as i64).rev().collect(),
            ..Self::default()
        }
    }
    /// Delete a logical axis and renumber logical and physical split axes.
    ///
    /// Like OpenXLA `Layout::DeleteDimension`, this leaves tiles and obsolete
    /// sparse annotations unchanged. Invalid axes or permutations are errors
    /// and leave this layout unchanged.
    pub fn delete_dimension(&mut self, dimension: i64) -> Result<()> {
        let rank = self.minor_to_major.len();
        let mut seen = vec![false; rank];
        for &axis in &self.minor_to_major {
            let valid = usize::try_from(axis)
                .ok()
                .and_then(|axis| seen.get_mut(axis))
                .is_some_and(|slot| !std::mem::replace(slot, true));
            if !valid {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "dimension deletion requires a valid layout permutation".into(),
                }));
            }
        }
        let position = self
            .minor_to_major
            .iter()
            .position(|&axis| axis == dimension)
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: format!("layout dimension {dimension} out of range"),
                })
            })?;
        let physical_dimension = (rank - 1 - position) as i64;
        self.minor_to_major.remove(position);
        for axis in &mut self.minor_to_major {
            if *axis > dimension {
                *axis -= 1;
            }
        }
        self.split_configs.retain_mut(|split| {
            if split.dimension == physical_dimension {
                return false;
            }
            if split.dimension > physical_dimension {
                split.dimension -= 1;
            }
            true
        });
        Ok(())
    }
    pub(super) fn equal(&self, ctx: &Context, other: &Self) -> bool {
        self.minor_to_major == other.minor_to_major
            && self.tiles == other.tiles
            && self.tail_padding_alignment_in_elements.max(1)
                == other.tail_padding_alignment_in_elements.max(1)
            && self.element_size_in_bits == other.element_size_in_bits
            && self.memory_space == other.memory_space
            && self.index_primitive_type == other.index_primitive_type
            && self.pointer_primitive_type == other.pointer_primitive_type
            && self.split_configs == other.split_configs
            && self.physical_shape.len() == other.physical_shape.len()
            && self
                .physical_shape
                .iter()
                .zip(&other.physical_shape)
                .all(|(&a, &b)| shape::equal_including_layout(ctx, a, b))
    }
}

/// Compare layouts semantically, ignoring metadata prefixes and obsolete annotations.
pub fn layouts_equal(ctx: &Context, a: &Layout, b: &Layout) -> bool {
    a.equal(ctx, b)
}

/// Validates a layout against logical array dimensions.
///
/// Checks that the minor order is a permutation of the array axes, alignment
/// is positive, element width is nonnegative, tile markers are valid, and split
/// configurations identify distinct in-range physical axes. Sparse index, pointer, and
/// physical-shape representations are currently rejected.
///
/// An empty minor order in a nondefault memory space is accepted as a special
/// opaque-memory layout without applying the remaining dense-layout checks.
///
/// # Errors
///
/// Returns a typed invalid-shape diagnostic for the first violated constraint.
pub fn validate_layout(layout: &Layout, dimensions: &[i64]) -> Result<()> {
    if layout.minor_to_major.is_empty() && layout.memory_space != 0 {
        return Ok(());
    }
    if layout.minor_to_major.len() != dimensions.len() {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "layout rank mismatch".into(),
        }));
    }
    let mut seen = vec![false; dimensions.len()];
    for &axis in &layout.minor_to_major {
        let axis = usize::try_from(axis).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "negative layout axis".into(),
            })
        })?;
        let slot = seen.get_mut(axis).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "layout axis out of range".into(),
            })
        })?;
        if std::mem::replace(slot, true) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "duplicate layout axis".into(),
            }));
        }
    }
    if layout.tail_padding_alignment_in_elements <= 0 || layout.element_size_in_bits < 0 {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "nonpositive layout alignment or negative element size".into(),
        }));
    }
    if !layout.index_primitive_type.is_empty()
        || !layout.pointer_primitive_type.is_empty()
        || !layout.physical_shape.is_empty()
    {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "sparse physical/index/pointer fields rejected by pinned LayoutUtil".into(),
        }));
    }
    for tile in &layout.tiles {
        if tile.dimensions.is_empty()
            || tile
                .dimensions
                .iter()
                .any(|&n| n <= 0 && n != COMBINE_DIMENSION && n != MATCH_DIMENSION)
        {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "invalid tile dimension".into(),
            }));
        }
    }
    let mut split_axes = vec![false; dimensions.len()];
    for split in &layout.split_configs {
        let axis = usize::try_from(split.dimension).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "negative split axis".into(),
            })
        })?;
        let slot = split_axes.get_mut(axis).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "split axis out of range".into(),
            })
        })?;
        if std::mem::replace(slot, true) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "duplicate split axis".into(),
            }));
        }
        let logical_axis = layout.minor_to_major[dimensions.len() - 1 - axis] as usize;
        let mut previous = 0;
        for &index in &split.split_indices {
            if index <= previous || index >= dimensions[logical_axis] {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "split indices must increase within the physical dimension".into(),
                }));
            }
            previous = index;
        }
    }
    Ok(())
}
