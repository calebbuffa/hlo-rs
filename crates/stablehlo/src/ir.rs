use std::collections::HashMap;

use crate::{
    Attribute, Attributes, FunctionType, Location, Operation, StableHloOp, Type,
    verification::{BuildError, verify_computation},
};

/// Stable identity of a value produced within one computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

/// Index of a computation within its module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ComputationId(pub u32);

/// One typed SSA result.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperationResult {
    value: ValueId,
    ty: Type,
}

impl OperationResult {
    pub fn new(value: ValueId, ty: Type) -> Self {
        Self { value, ty }
    }

    pub fn value(&self) -> ValueId {
        self.value
    }

    pub fn ty(&self) -> &Type {
        &self.ty
    }
}

/// A block argument in a nested StableHLO region.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockArgument {
    value: ValueId,
    ty: Type,
    location: Option<Location>,
}

impl BlockArgument {
    pub fn new(value: ValueId, ty: Type) -> Self {
        Self {
            value,
            ty,
            location: None,
        }
    }

    pub fn with_location(mut self, location: Location) -> Self {
        self.location = Some(location);
        self
    }

    pub fn location(&self) -> Option<&Location> {
        self.location.as_ref()
    }

    pub fn value(&self) -> ValueId {
        self.value
    }

    pub fn ty(&self) -> &Type {
        &self.ty
    }
}

/// A block of operations. Values are identified in the enclosing computation;
/// operands may refer to values visible from lexically enclosing regions.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Block {
    arguments: Vec<BlockArgument>,
    instructions: Vec<Instruction>,
}

impl Block {
    pub fn new(arguments: Vec<BlockArgument>, instructions: Vec<Instruction>) -> Self {
        Self {
            arguments,
            instructions,
        }
    }

    pub fn arguments(&self) -> &[BlockArgument] {
        &self.arguments
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }
}

/// A possibly multi-block nested operation body.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Region {
    blocks: Vec<Block>,
}

impl Region {
    pub fn new(blocks: Vec<Block>) -> Self {
        Self { blocks }
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }
}

/// A typed SSA operation with zero or more results and nested regions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Instruction {
    pub(crate) operation: Operation,
    pub(crate) operands: Vec<ValueId>,
    pub(crate) called_computations: Vec<ComputationId>,
    pub(crate) results: Vec<OperationResult>,
    pub(crate) regions: Vec<Region>,
    pub(crate) location: Option<Location>,
}

impl Instruction {
    pub fn new(
        operation: Operation,
        operands: Vec<ValueId>,
        called_computations: Vec<ComputationId>,
        result: ValueId,
        ty: Type,
    ) -> Self {
        Self::with_results(
            operation,
            operands,
            called_computations,
            vec![OperationResult::new(result, ty)],
        )
    }

    pub fn with_results(
        operation: Operation,
        operands: Vec<ValueId>,
        called_computations: Vec<ComputationId>,
        results: Vec<OperationResult>,
    ) -> Self {
        Self {
            operation,
            operands,
            called_computations,
            results,
            regions: Vec::new(),
            location: None,
        }
    }

    pub fn with_regions(mut self, regions: Vec<Region>) -> Self {
        self.regions = regions;
        self
    }

    pub fn with_location(mut self, location: Location) -> Self {
        self.location = Some(location);
        self
    }

    pub fn location(&self) -> Option<&Location> {
        self.location.as_ref()
    }

    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn operands(&self) -> &[ValueId] {
        &self.operands
    }

    pub fn called_computations(&self) -> &[ComputationId] {
        &self.called_computations
    }

    /// Returns the sole result for single-result operations.
    ///
    /// Use [`Self::results`] for operations with zero or multiple results.
    pub fn result(&self) -> Option<ValueId> {
        (self.results.len() == 1).then(|| self.results[0].value())
    }

    /// Returns the sole result type for single-result operations.
    ///
    /// Use [`Self::results`] for operations with zero or multiple results.
    pub fn ty(&self) -> Option<&Type> {
        (self.results.len() == 1).then(|| self.results[0].ty())
    }

    pub fn results(&self) -> &[OperationResult] {
        &self.results
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }
}

/// A function definition with a single-entry SSA body, or an external declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Computation {
    pub(crate) name: String,
    pub(crate) attributes: Attributes,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) roots: Vec<ValueId>,
    pub(crate) declaration: Option<FunctionType>,
    pub(crate) location: Option<Location>,
    pub(crate) return_location: Option<Location>,
}

impl Computation {
    pub fn try_new(
        name: impl Into<String>,
        instructions: Vec<Instruction>,
        root: ValueId,
    ) -> Result<Self, BuildError> {
        Self::try_new_with_roots(name, instructions, vec![root])
    }

    pub fn try_new_with_roots(
        name: impl Into<String>,
        instructions: Vec<Instruction>,
        roots: Vec<ValueId>,
    ) -> Result<Self, BuildError> {
        let computation = Self {
            name: name.into(),
            attributes: Attributes::new(),
            instructions,
            roots,
            declaration: None,
            location: None,
            return_location: None,
        };
        verify_computation(&computation)?;
        Ok(computation)
    }

    /// Creates an external `func.func` declaration with private visibility.
    pub fn try_declare(
        name: impl Into<String>,
        signature: FunctionType,
    ) -> Result<Self, BuildError> {
        let computation = Self {
            name: name.into(),
            attributes: [("sym_visibility".into(), Attribute::String("private".into()))].into(),
            instructions: vec![],
            roots: vec![],
            declaration: Some(signature),
            location: None,
            return_location: None,
        };
        verify_computation(&computation)?;
        Ok(computation)
    }

    pub fn is_declaration(&self) -> bool {
        self.declaration.is_some()
    }

    pub fn location(&self) -> Option<&Location> {
        self.location.as_ref()
    }

    pub fn set_location(&mut self, location: Location) -> Result<(), BuildError> {
        crate::verification::verify_location(Some(&location))?;
        self.location = Some(location);
        Ok(())
    }

    pub fn return_location(&self) -> Option<&Location> {
        self.return_location.as_ref()
    }

    pub fn set_return_location(&mut self, location: Location) -> Result<(), BuildError> {
        if self.is_declaration() {
            return Err(BuildError::InvalidFunctionAttribute(
                "return location on an external declaration",
            ));
        }
        crate::verification::verify_location(Some(&location))?;
        self.return_location = Some(location);
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn attributes(&self) -> &Attributes {
        &self.attributes
    }

    /// Replaces function attributes after checking their types and signature metadata.
    pub fn set_attributes(&mut self, attributes: Attributes) -> Result<(), BuildError> {
        crate::verification::verify_function_attributes(self, &attributes)?;
        self.attributes = attributes;
        Ok(())
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn root(&self) -> Option<ValueId> {
        (self.roots.len() == 1).then(|| self.roots[0])
    }

    pub fn roots(&self) -> &[ValueId] {
        &self.roots
    }

    pub fn function_type(&self) -> FunctionType {
        if let Some(signature) = &self.declaration {
            return signature.clone();
        }
        let mut parameters = self
            .instructions
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                Operation::Parameter { number, .. } => {
                    instruction.ty().map(|ty| (*number, ty.clone()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        parameters.sort_unstable_by_key(|(number, _)| *number);
        FunctionType::new(
            parameters.into_iter().map(|(_, ty)| ty).collect::<Vec<_>>(),
            self.roots
                .iter()
                .filter_map(|root| self.value_type(*root).cloned())
                .collect::<Vec<_>>(),
        )
    }

    pub fn value_type(&self, id: ValueId) -> Option<&Type> {
        Self::find_value_type(&self.instructions, id)
    }

    fn find_value_type(instructions: &[Instruction], id: ValueId) -> Option<&Type> {
        for instruction in instructions {
            if let Some(result) = instruction.results.iter().find(|result| result.value == id) {
                return Some(&result.ty);
            }
            for region in &instruction.regions {
                for block in &region.blocks {
                    if let Some(argument) =
                        block.arguments.iter().find(|argument| argument.value == id)
                    {
                        return Some(&argument.ty);
                    }
                    if let Some(ty) = Self::find_value_type(&block.instructions, id) {
                        return Some(ty);
                    }
                }
            }
        }
        None
    }

    /// Replaces a computation body and root, rejecting structurally invalid SSA.
    pub fn replace_body(
        &mut self,
        instructions: Vec<Instruction>,
        root: ValueId,
    ) -> Result<(), BuildError> {
        self.replace_body_with_roots(instructions, vec![root])
    }

    pub fn replace_body_with_roots(
        &mut self,
        instructions: Vec<Instruction>,
        roots: Vec<ValueId>,
    ) -> Result<(), BuildError> {
        let declaration = self.declaration.take();
        let previous_instructions = std::mem::replace(&mut self.instructions, instructions);
        let previous_roots = std::mem::replace(&mut self.roots, roots);
        let verification = verify_computation(self).and_then(|()| {
            if declaration
                .as_ref()
                .is_some_and(|signature| *signature != self.function_type())
            {
                Err(BuildError::InvalidComputationSignature {
                    operation: "func.func",
                    name: self.name.clone(),
                })
            } else {
                Ok(())
            }
        });
        if let Err(error) = verification {
            self.instructions = previous_instructions;
            self.roots = previous_roots;
            self.declaration = declaration;
            return Err(error);
        }
        Ok(())
    }
}

/// A module containing functions and optional consumer-selected entry metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Module {
    name: String,
    attributes: Attributes,
    computations: Vec<Computation>,
    entry: Option<ComputationId>,
    location: Option<Location>,
}

impl Module {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: Attributes::new(),
            computations: Vec::new(),
            entry: None,
            location: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn location(&self) -> Option<&Location> {
        self.location.as_ref()
    }

    pub fn set_location(&mut self, location: Location) -> Result<(), BuildError> {
        crate::verification::verify_location(Some(&location))?;
        self.location = Some(location);
        Ok(())
    }

    pub fn attributes(&self) -> &Attributes {
        &self.attributes
    }

    pub fn set_attributes(&mut self, attributes: Attributes) -> Result<(), BuildError> {
        crate::verification::verify_module_attributes(&attributes)?;
        self.attributes = attributes;
        Ok(())
    }

    pub fn computations(&self) -> &[Computation] {
        &self.computations
    }

    pub fn entry(&self) -> Option<ComputationId> {
        self.entry
    }

    pub fn add_computation(&mut self, computation: Computation) -> ComputationId {
        let id = ComputationId(self.computations.len() as u32);
        self.computations.push(computation);
        id
    }

    pub fn set_entry(&mut self, id: impl Into<ComputationId>) -> Result<(), BuildError> {
        let id = id.into();
        if id.0 as usize >= self.computations.len() {
            return Err(BuildError::UnknownComputation(id));
        }
        self.entry = Some(id);
        Ok(())
    }

    pub fn computations_mut(&mut self) -> &mut [Computation] {
        &mut self.computations
    }

    /// Checks module entry, SSA structure, and nested-computation references.
    pub fn verify(&self) -> Result<(), BuildError> {
        crate::verification::verify_location(self.location())?;
        crate::verification::verify_module_attributes(&self.attributes)?;
        if let Some(entry) = self.entry
            && entry.0 as usize >= self.computations.len()
        {
            return Err(BuildError::UnknownComputation(entry));
        }
        let mut names = HashMap::new();
        for computation in &self.computations {
            if names.insert(computation.name(), computation).is_some() {
                return Err(BuildError::DuplicateComputationName(
                    computation.name().to_owned(),
                ));
            }
        }
        for computation in &self.computations {
            verify_computation(computation)?;
            verify_called_computations(&computation.instructions, self.computations.len())?;
            verify_computation_symbols(&computation.instructions, computation, &names)?;
        }
        Ok(())
    }
}

fn verify_computation_symbols(
    instructions: &[Instruction],
    enclosing: &Computation,
    names: &HashMap<&str, &Computation>,
) -> Result<(), BuildError> {
    for instruction in instructions {
        if let Some(Attribute::StableHlo(attribute)) = instruction
            .operation
            .attributes()
            .and_then(|attributes| attributes.get("replica_groups"))
            && let crate::StableHloAttribute::ReplicaGroupMeshAxes(groups) = attribute.as_ref()
            && let crate::MeshReference::Symbol(name) = &groups.mesh
        {
            let Some(symbol) = names.get(name.as_str()) else {
                return Err(BuildError::UnknownComputationName {
                    operation: instruction.operation.name(),
                    name: name.clone(),
                });
            };
            let Some(Attribute::StableHlo(attribute)) = symbol
                .attributes()
                .get("mesh")
                .or_else(|| symbol.attributes().get("stablehlo.mesh"))
            else {
                return Err(BuildError::InvalidMeshSymbol(name.clone()));
            };
            let crate::StableHloAttribute::Mesh(mesh) = attribute.as_ref() else {
                return Err(BuildError::InvalidMeshSymbol(name.clone()));
            };
            crate::verification::verify_resolved_mesh(instruction, enclosing, mesh)?;
        }
        let symbol_attribute = match &instruction.operation {
            Operation::Call { .. } => Some("callee"),
            Operation::StableHlo {
                opcode: StableHloOp::Composite,
                ..
            } => Some("decomposition"),
            _ => None,
        };
        if let Some(symbol_attribute) = symbol_attribute {
            let Some(Attribute::SymbolRef(path)) = instruction
                .operation
                .attributes()
                .and_then(|attributes| attributes.get(symbol_attribute))
            else {
                return Err(BuildError::MissingOrInvalidAttribute {
                    operation: instruction.operation.name(),
                    attribute: symbol_attribute,
                });
            };
            let [name] = path.as_slice() else {
                return Err(BuildError::InvalidSymbolReference {
                    operation: instruction.operation.name(),
                });
            };
            let Some(decomposition) = names.get(name.as_str()) else {
                return Err(BuildError::UnknownComputationName {
                    operation: instruction.operation.name(),
                    name: name.clone(),
                });
            };
            let signature = decomposition.function_type();
            let operands = instruction
                .operands()
                .iter()
                .map(|&value| enclosing.value_type(value).expect("verified SSA operand"));
            if !operands.eq(signature.inputs())
                || !instruction
                    .results()
                    .iter()
                    .map(|result| result.ty())
                    .eq(signature.outputs())
            {
                return Err(BuildError::InvalidComputationSignature {
                    operation: instruction.operation.name(),
                    name: name.clone(),
                });
            }
        }
        for region in instruction.regions() {
            for block in region.blocks() {
                verify_computation_symbols(block.instructions(), enclosing, names)?;
            }
        }
    }
    Ok(())
}

fn verify_called_computations(
    instructions: &[Instruction],
    computation_count: usize,
) -> Result<(), BuildError> {
    for instruction in instructions {
        for &called in &instruction.called_computations {
            if called.0 as usize >= computation_count {
                return Err(BuildError::UnknownComputation(called));
            }
        }
        for region in &instruction.regions {
            for block in &region.blocks {
                verify_called_computations(&block.instructions, computation_count)?;
            }
        }
    }
    Ok(())
}

impl From<u32> for ComputationId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
