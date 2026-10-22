//! Sharding, metadata, frontend-attribute and original-value transfer.

use super::*;

#[derive(Clone)]
pub(super) struct Annotations {
    pub(super) metadata: Option<OpMetadata>,
    pub(super) frontend: FrontendAttributes,
    pub(super) sharding: Option<Sharding>,
    pub(super) original_value: Option<OriginalValue>,
}

impl Annotations {
    pub(super) fn read(ctx: &Context, value: Value) -> Result<Self> {
        let annotations = match value.defining_entity() {
            DefiningEntity::Op(operation) => Self {
                metadata: optional_attr::<OpMetadata>(ctx, operation, "hlo_metadata")?
                    .as_deref()
                    .cloned(),
                frontend: optional_attr::<FrontendAttributes>(
                    ctx,
                    operation,
                    "hlo_frontend_attributes",
                )?
                .as_deref()
                .cloned()
                .unwrap_or_default(),
                sharding: optional_attr::<Sharding>(ctx, operation, "hlo_sharding")?
                    .as_deref()
                    .cloned(),
                original_value: optional_attr::<OriginalValue>(
                    ctx,
                    operation,
                    "hlo_original_value",
                )?
                .as_deref()
                .cloned(),
            },
            DefiningEntity::Block(block) => {
                let parameters = fusion_parameters(ctx, block)?;
                let index = argument_index(ctx, block, value)?;
                let parameter = &parameters.0[index];
                Self {
                    metadata: parameter.metadata.clone(),
                    frontend: parameter.frontend_attributes.clone(),
                    sharding: parameter.sharding.clone(),
                    original_value: parameter.original_value.clone(),
                }
            }
        };
        annotations.validate(ctx, value)?;
        Ok(annotations)
    }

    pub(super) fn validate(&self, ctx: &Context, value: Value) -> Result<()> {
        self.frontend.verify(ctx)?;
        if let Some(metadata) = &self.metadata {
            metadata.verify(ctx)?;
        }
        if let Some(sharding) = &self.sharding {
            sharding.validate(ctx, value.get_type(ctx), None)?;
        }
        if let Some(provenance) = &self.original_value {
            provenance.verify_shape(ctx, value.get_type(ctx))?;
        }
        Ok(())
    }

    pub(super) fn write(self, ctx: &Context, value: Value) -> Result<()> {
        match value.defining_entity() {
            DefiningEntity::Op(operation) => {
                let mut ir = operation.deref_mut(ctx);
                if let Some(metadata) = self.metadata {
                    ir.attributes.set(Identifier::new("hlo_metadata"), metadata);
                }
                if !self.frontend.0.is_empty() {
                    ir.attributes
                        .set(Identifier::new("hlo_frontend_attributes"), self.frontend);
                }
                if let Some(sharding) = self.sharding {
                    ir.attributes.set(Identifier::new("hlo_sharding"), sharding);
                }
                if let Some(original) = self.original_value {
                    ir.attributes
                        .set(Identifier::new("hlo_original_value"), original);
                }
            }
            DefiningEntity::Block(block) => {
                let mut parameters = fusion_parameters(ctx, block)?;
                let index = argument_index(ctx, block, value)?;
                let parameter = &mut parameters.0[index];
                parameter.metadata = self.metadata;
                parameter.frontend_attributes = self.frontend;
                parameter.sharding = self.sharding;
                parameter.original_value = self.original_value;
                set_fusion_parameters(ctx, block, parameters)?;
            }
        }
        Ok(())
    }
}

pub(super) fn argument_index(ctx: &Context, block: Ptr<BasicBlock>, value: Value) -> Result<usize> {
    (0..block.deref(ctx).get_num_arguments())
        .find(|&index| block.deref(ctx).get_argument(index) == value)
        .ok_or_else(|| error("fusion argument does not belong to its body"))
}
