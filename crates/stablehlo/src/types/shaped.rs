use super::*;
use pliron::{location::Location, verify_err};
#[pliron_type(
    name = "stablehlo.vector",
    format = "`<` $element `,` `[` vec($dimensions, CharSpace(`,`)) `]` `,` `[` vec($scalable, CharSpace(`,`)) `]` `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorType {
    pub element: TypeHandle,
    pub dimensions: Vec<i64>,
    pub scalable: Vec<bool>,
}
impl VectorType {
    pub fn fixed(ctx: &Context, element: ElementType, dimensions: Vec<i64>) -> TypeHandle {
        let scalable = vec![false; dimensions.len()];
        Self::get(ctx, element.native(ctx), dimensions, scalable).into()
    }
    pub fn new_scalable(
        ctx: &Context,
        element: ElementType,
        dimensions: Vec<i64>,
        scalable: Vec<bool>,
    ) -> TypeHandle {
        Self::get(ctx, element.native(ctx), dimensions, scalable).into()
    }
    pub fn element(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.element)
    }
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }
    pub fn scalable_dimensions(&self) -> &[bool] {
        &self.scalable
    }
    pub fn is_scalable(&self) -> bool {
        self.scalable.iter().any(|&v| v)
    }
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
}
impl Verify for VectorType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.dimensions.iter().any(|&d| d <= 0)
            || self.scalable.len() != self.dimensions.len()
            || self
                .element(ctx)
                .is_none_or(|e| e != ElementType::Index && !e.is_integer() && !e.is_float())
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "shaped type",
                    reason: "invalid vector shape, scalable flags, or element type"
                }
            );
        }
        Ok(())
    }
}
#[pliron_type(
    name = "stablehlo.buffer",
    format = "`<` $element `,` `[` vec($dimensions, CharSpace(`,`)) `]` `,` $layout `,` $memory_space `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferType {
    pub element: TypeHandle,
    pub dimensions: Vec<i64>,
    pub layout: VecAttr,
    pub memory_space: VecAttr,
}
impl BufferType {
    pub fn without_layout(ctx: &Context, element: TypeHandle, dimensions: Vec<i64>) -> TypeHandle {
        Self::get(
            ctx,
            element,
            dimensions,
            VecAttr::new(vec![]),
            VecAttr::new(vec![]),
        )
        .into()
    }
    pub fn element_type(&self) -> TypeHandle {
        self.element
    }
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }
    pub fn layout(&self) -> Option<&AttrObj> {
        self.layout.0.first()
    }
    pub fn memory_space(&self) -> Option<&AttrObj> {
        self.memory_space.0.first()
    }
    pub fn with_layout(&self, ctx: &Context, layout: AttrObj) -> TypeHandle {
        let identity = layout
            .downcast_ref::<crate::attributes::AffineMap>()
            .is_some_and(|map| {
                map.dimensions == self.dimensions.len()
                    && map.symbols == 0
                    && map.results.len() == map.dimensions
                    && map.results.iter().enumerate().all(|(index, result)| {
                        *result == crate::attributes::AffineExpr::Dimension(index)
                    })
            });
        Self::get(
            ctx,
            self.element,
            self.dimensions.clone(),
            VecAttr::new(if identity { vec![] } else { vec![layout] }),
            self.memory_space.clone(),
        )
        .into()
    }
    pub fn with_memory_space(&self, ctx: &Context, space: AttrObj) -> TypeHandle {
        let default = is_default_memory_space(ctx, &space);
        Self::get(
            ctx,
            self.element,
            self.dimensions.clone(),
            self.layout.clone(),
            VecAttr::new(if default { vec![] } else { vec![space] }),
        )
        .into()
    }
}
impl Verify for BufferType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_buffers(ctx, self.element, Some(self), None)
    }
}
#[pliron_type(
    name = "stablehlo.unranked_buffer",
    format = "`<` $element `,` $memory_space `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnrankedBufferType {
    pub element: TypeHandle,
    pub memory_space: VecAttr,
}
impl UnrankedBufferType {
    pub fn without_memory_space(ctx: &Context, element: TypeHandle) -> TypeHandle {
        Self::get(ctx, element, VecAttr::new(vec![])).into()
    }
    pub fn element_type(&self) -> TypeHandle {
        self.element
    }
    pub fn memory_space(&self) -> Option<&AttrObj> {
        self.memory_space.0.first()
    }
    pub fn with_memory_space(&self, ctx: &Context, space: AttrObj) -> TypeHandle {
        let default = is_default_memory_space(ctx, &space);
        Self::get(
            ctx,
            self.element,
            VecAttr::new(if default { vec![] } else { vec![space] }),
        )
        .into()
    }
    pub fn scalar_element(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.element)
    }
}

impl Verify for UnrankedBufferType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_buffers(ctx, self.element, None, Some(self))
    }
}
fn is_default_memory_space(ctx: &Context, space: &AttrObj) -> bool {
    if let Some(value) = space.downcast_ref::<BoolAttr>() {
        return !bool::from(value.clone());
    }
    if let Some(value) = space.downcast_ref::<IntegerAttr>() {
        return value.value().to_string_decimal(false) == "0";
    }
    if let Some(value) = space.downcast_ref::<crate::attributes::IndexAttr>() {
        return value.value == 0;
    }
    space
        .downcast_ref::<crate::attributes::ScalarBitsAttr>()
        .is_some_and(|value| {
            primitive_element(ctx, value.ty)
                .is_some_and(|element| element.is_integer() || element == ElementType::Index)
                && value.words.iter().all(|&word| word == 0)
        })
}
fn verify_buffers(
    ctx: &Context,
    mut element: TypeHandle,
    root: Option<&BufferType>,
    unranked: Option<&UnrankedBufferType>,
) -> Result<()> {
    fn local(
        ctx: &Context,
        shape: Option<&[i64]>,
        layout: Option<&VecAttr>,
        space: &VecAttr,
    ) -> Result<()> {
        if shape.is_some_and(|s| s.iter().any(|&d| d < -1))
            || layout.is_some_and(|l| l.0.len() > 1)
            || space.0.len() > 1
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "shaped type",
                    reason: "invalid buffer shape, layout or memory space count"
                }
            );
        }
        if let Some(layout) = layout {
            for attr in &layout.0 {
                attr.verify(ctx)?;
                if let Some(map) = attr.downcast_ref::<crate::attributes::AffineMap>() {
                    if shape.is_some_and(|shape| map.dimensions != shape.len()) {
                        return verify_err!(
                            Location::Unknown,
                            crate::Diagnostic::InvalidType {
                                ty: "shaped type",
                                reason: "affine buffer layout rank mismatch"
                            }
                        );
                    }
                } else if let Some(strided) =
                    attr.downcast_ref::<crate::attributes::StridedLayoutAttr>()
                {
                    if shape.is_some_and(|shape| strided.strides.len() != shape.len()) {
                        return verify_err!(
                            Location::Unknown,
                            crate::Diagnostic::InvalidType {
                                ty: "shaped type",
                                reason: "strided buffer layout rank mismatch"
                            }
                        );
                    }
                } else if attr
                    .downcast_ref::<crate::attributes::OpaqueAttr>()
                    .is_none()
                {
                    return verify_err!(
                        Location::Unknown,
                        crate::Diagnostic::InvalidType {
                            ty: "shaped type",
                            reason: "buffer layout must be affine, strided or opaque"
                        }
                    );
                }
            }
        }
        for attribute in &space.0 {
            crate::attributes::verify_attribute(ctx, attribute)?;
        }
        Ok(())
    }
    if let Some(buffer) = root {
        local(
            ctx,
            Some(&buffer.dimensions),
            Some(&buffer.layout),
            &buffer.memory_space,
        )?;
    }
    if let Some(buffer) = unranked {
        local(ctx, None, None, &buffer.memory_space)?;
    }
    loop {
        let ty = element.deref(ctx);
        if let Some(buffer) = ty.downcast_ref::<BufferType>() {
            local(
                ctx,
                Some(&buffer.dimensions),
                Some(&buffer.layout),
                &buffer.memory_space,
            )?;
            element = buffer.element;
        } else if let Some(buffer) = ty.downcast_ref::<UnrankedBufferType>() {
            local(ctx, None, None, &buffer.memory_space)?;
            element = buffer.element;
        } else if primitive_element(ctx, element).is_some()
            || ty.downcast_ref::<VectorType>().is_some()
        {
            return ty.verify(ctx);
        } else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "shaped type",
                    reason: "invalid buffer element type"
                }
            );
        }
    }
}
#[pliron_type(
    name = "stablehlo.future",
    format = "`<` vec($elements, CharSpace(`,`)) `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FutureType {
    pub elements: Vec<TypeHandle>,
}
impl Verify for FutureType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for &element in &self.elements {
            let ty = element.deref(ctx);
            if ty.downcast_ref::<TensorType>().is_none() {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidType {
                        ty: "shaped type",
                        reason: "future element must be a ranked tensor"
                    }
                );
            }
            ty.verify(ctx)?;
        }
        Ok(())
    }
}
macro_rules! unit_type {
    ($name:ident,$mnemonic:literal) => {
        #[pliron_type(name=$mnemonic,format,generate_get=true,verifier="succ")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name;
    };
}
unit_type!(StringType, "stablehlo.string");
unit_type!(NoneType, "stablehlo.none");
unit_type!(WitnessType, "stablehlo.witness");
