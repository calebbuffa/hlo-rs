//! Concrete HLO operations. Constructors preserve the supplied result type;
//! `verify` checks it against inferred geometry before a graph is consumed.
use pliron::{
    builtin::attributes::TypeAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
    value::Value,
};

#[pliron::derive::op_interface]
/// Common semantic verification interface implemented by every HLO instruction.
pub trait HloOpInterface {
    /// Return the HLO opcode represented by this operation.
    fn opcode(&self) -> crate::opcode::Opcode;
    /// Validate local semantics under the supplied verifier policies.
    fn verify_with_options(
        &self,
        ctx: &Context,
        options: crate::verification::VerifyOptions,
    ) -> Result<()>;
    fn verify(_: &dyn Op, _: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

macro_rules! native_instruction {
    ($name:ident, $id:literal, $opcode:ident, $verifier:path, $( $key:ident : $ty:ty ),* $(,)?) => {
        #[pliron::derive::pliron_op(name = $id, format)]
        #[doc = concat!("The `", $id, "` instruction, represented by [`crate::opcode::Opcode::", stringify!($opcode), "`].")]
        ///
        /// This is a context-owned operation handle. Explicit construction
        /// records the declared result type, operand order, and attributes
        /// without validating them. Verify the operation before consuming it;
        /// whole-module verification additionally checks symbol references,
        /// ownership, and graph-wide constraints.
        pub struct $name;
        impl $name {
            /// The HLO opcode represented by this operation.
            pub const OPCODE: crate::opcode::Opcode = crate::opcode::Opcode::$opcode;
            /// Construct an instruction with explicit result type and typed attributes.
            ///
            /// Takes ownership of the operand list and attribute values.
            /// Operand counts, operand domains, result geometry, and attribute
            /// constraints are checked by the operation's verifier, not here.
            /// The new operation is detached until inserted into a block.
            #[inline]
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                ctx: &mut pliron::context::Context,
                result: pliron::r#type::TypeHandle,
                operands: Vec<pliron::value::Value>,
                $($key: $ty),*
            ) -> Self {
                let value = Self {
                    op: pliron::operation::Operation::new(
                        ctx,
                        <Self as pliron::op::Op>::get_concrete_op_info(),
                        vec![result],
                        operands,
                        vec![],
                        0,
                    ),
                };
                $(value.op.deref_mut(ctx).attributes.set(
                    pliron::identifier::Identifier::new(concat!("hlo_", stringify!($key))),
                    $key,
                );)*
                value
            }
            #[inline]
            /// Returns the instruction's single SSA result.
            ///
            /// The operation must still have the single result established by
            /// its constructor; raw mutation of its result list must preserve
            /// that invariant.
            pub fn result(&self, ctx: &pliron::context::Context) -> pliron::value::Value {
                self.op.deref(ctx).get_result(0)
            }
            $(#[doc = concat!("Borrow the `hlo_", stringify!($key), "` attribute, if correctly typed.")]
            pub fn $key<'a>(
                &self,
                ctx: &'a pliron::context::Context,
            ) -> Option<std::cell::Ref<'a, $ty>> {
                std::cell::Ref::filter_map(self.op.deref(ctx), |op| {
                    op.attributes.get::<$ty>(&pliron::identifier::Identifier::new(
                        concat!("hlo_", stringify!($key))
                    ))
                }).ok()
            })*
        }
        impl pliron::common_traits::Verify for $name {
            fn verify(&self, ctx: &pliron::context::Context) -> pliron::result::Result<()> {
                $verifier(ctx, self.op, Self::OPCODE, crate::verification::VerifyOptions::default())
            }
        }
        #[pliron::derive::op_interface_impl]
        impl crate::ops::HloOpInterface for $name {
            fn opcode(&self) -> crate::opcode::Opcode {
                Self::OPCODE
            }
            fn verify_with_options(
                &self,
                ctx: &pliron::context::Context,
                options: crate::verification::VerifyOptions,
            ) -> pliron::result::Result<()> {
                $verifier(ctx, self.op, Self::OPCODE, options)
            }
        }
    };
}

pub mod control;
pub mod elementwise;
pub mod movement;
pub mod numerical;

#[pliron_op(name = "xlahlo.computation", format, interfaces = [pliron::builtin::op_interfaces::IsolatedFromAboveInterface], attributes = (hlo_computation_name: crate::ir::ComputationName, hlo_computation_signature: TypeAttr))]
/// A named, isolated HLO computation with one block and a function signature.
pub struct ComputationOp;

impl ComputationOp {
    /// Construct an empty computation; callers must populate its body and return.
    pub fn new(
        ctx: &mut Context,
        name: impl Into<crate::ir::ComputationName>,
        signature: TypeHandle,
    ) -> Self {
        let value = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1),
        };
        value.set_attr_hlo_computation_name(ctx, name.into());
        value.set_attr_hlo_computation_signature(ctx, TypeAttr::new(signature));
        value
    }

    /// Return the computation's isolated body region.
    pub fn body(&self, ctx: &Context) -> Ptr<pliron::region::Region> {
        self.op.deref(ctx).get_region(0)
    }
}

impl Verify for ComputationOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::computation(ctx, self)
    }
}

#[pliron_op(name = "xlahlo.return", format, interfaces = [pliron::builtin::op_interfaces::IsTerminatorInterface])]
/// Terminator yielding the single result of a computation or fusion body.
pub struct ReturnOp;

impl ReturnOp {
    /// Construct a terminator returning an SSA value.
    pub fn new(ctx: &mut Context, value: Value) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![value],
                vec![],
                0,
            ),
        }
    }
}

impl Verify for ReturnOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.op.deref(ctx);
        crate::inference::require(
            op.get_num_operands() == 1
                && op.get_num_results() == 0
                && op.num_regions() == 0
                && op.get_num_successors() == 0,
            "return needs one value and no results, successors or regions",
        )
    }
}

pub(crate) fn register(ctx: &mut Context) {
    elementwise::register(ctx);
    movement::register(ctx);
    numerical::register(ctx);
    control::register(ctx);
    ComputationOp::register(ctx);
    ReturnOp::register(ctx);
}
