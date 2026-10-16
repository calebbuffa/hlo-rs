use crate::{Error, InstructionId, Module, Opcode, Result, Sharding, VerifyOptions};

/// Module configuration and verifier policy not stored in HloModuleProto.
#[derive(Clone, Copy, Debug)]
pub struct ShardingContext {
    pub use_spmd_partitioning: bool,
    pub verify_device_numbers: bool,
}

impl Default for ShardingContext {
    fn default() -> Self {
        Self {
            use_spmd_partitioning: false,
            verify_device_numbers: true,
        }
    }
}

fn normalized(module: &Module, id: InstructionId) -> Result<Option<Sharding>> {
    module
        .instruction(id)?
        .sharding
        .as_ref()
        .map(|wire| Sharding::from_proto(wire.clone())?.normalize_tuple(&module.shape(id)?))
        .transpose()
}

fn consistent(module: &Module, ids: &[InstructionId]) -> Result<()> {
    let mut previous: Option<Sharding> = None;
    for &id in ids {
        if let Some(sharding) = normalized(module, id)? {
            if let Some(previous) = &previous
                && !previous.equal_semantics(&sharding)?
            {
                return Err(Error::new("sharding", "inconsistent control-flow sharding")
                    .context(&module.instruction(id)?.name));
            }
            previous = Some(sharding);
        }
    }
    Ok(())
}

pub(super) fn verify(module: &Module, options: VerifyOptions) -> Result<()> {
    let context = options.sharding_context;
    let partitions = options.collective_context.partition_count;
    if partitions <= 0 {
        return Err(Error::new("sharding", "partition count must be positive"));
    }
    let num_devices =
        (context.use_spmd_partitioning && context.verify_device_numbers && partitions > 1)
            .then_some(partitions);
    for computation in module.computations() {
        for id in module.instructions(computation)? {
            let instruction = module.instruction(id)?;
            if let Some(sharding) = normalized(module, id)? {
                sharding
                    .validate(&module.shape(id)?, num_devices)
                    .map_err(|error| error.context(&instruction.name))?;
            }
            match instruction.opcode.parse::<Opcode>()? {
                Opcode::While => {
                    let callees = &instruction.called_computation_ids;
                    let body = module.computation_handle(callees[0]);
                    let condition = module.computation_handle(callees[1]);
                    let parameter = |computation| -> Result<InstructionId> {
                        for id in module.instructions(computation)? {
                            let instruction = module.instruction(id)?;
                            if instruction.opcode == "parameter"
                                && instruction.parameter_number == 0
                            {
                                return Ok(id);
                            }
                        }
                        Err(Error::new("sharding", "missing while parameter"))
                    };
                    consistent(
                        module,
                        &[
                            id,
                            module.root(body)?,
                            parameter(body)?,
                            parameter(condition)?,
                        ],
                    )?;
                }
                Opcode::Conditional => {
                    let mut related = vec![id];
                    for &callee in &instruction.called_computation_ids {
                        related.push(module.root(module.computation_handle(callee))?);
                    }
                    consistent(module, &related)?;
                }
                _ => (),
            }
        }
    }
    Ok(())
}
