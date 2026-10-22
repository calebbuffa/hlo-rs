use super::{EvaluationError, failure};
use crate::{
    literal::{Literal, semantics as literal_semantics},
    shape,
    types::TokenType,
};
use pliron::{common_traits::Verify, context::Context, result::Result, r#type::TypeHandle};

/// An owned evaluation tree. Unknown arrays differ from unvisited arrays;
/// neither state allocates zero-filled stand-in data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluatedValue {
    shape: TypeHandle,
    state: ValueState,
}

/// Leaf knowledge and structural tuple/token values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueState {
    /// An array with a completely known, verified literal payload.
    Known(Literal),
    /// An array whose value cannot be determined from this request.
    Unknown,
    /// An array whose dependency has not yet been visited.
    Undetermined,
    /// Independently tracked children, which may mix known and unknown leaves.
    Tuple(Vec<EvaluatedValue>),
    /// A known sequencing token with no data payload.
    Token,
}

impl EvaluatedValue {
    /// Validate and own a literal, expanding tuples into independent children.
    pub fn known(ctx: &Context, literal: Literal) -> Result<Self> {
        literal.verify(ctx)?;
        let shape = literal.shape;
        let state = if shape::is_tuple(ctx, shape) {
            ValueState::Tuple(
                literal
                    .tuple_literals
                    .into_iter()
                    .map(|child| Self::known(ctx, child))
                    .collect::<Result<_>>()?,
            )
        } else if shape.deref(ctx).is::<TokenType>() {
            ValueState::Token
        } else if shape::is_array(ctx, shape) {
            ValueState::Known(literal)
        } else {
            return Err(failure(EvaluationError::Malformed {
                message: "evaluation values require array, tuple or token shapes".into(),
            }));
        };
        Ok(Self { shape, state })
    }

    /// Mark array leaves as unknown, retaining tuples and known token identity.
    pub fn unknown(ctx: &Context, shape: TypeHandle) -> Result<Self> {
        Self::empty(ctx, shape, false, 0)
    }

    /// Mark array leaves as not yet visited; tuple and token structure is owned.
    pub fn undetermined(ctx: &Context, shape: TypeHandle) -> Result<Self> {
        Self::empty(ctx, shape, true, 0)
    }

    fn empty(ctx: &Context, shape: TypeHandle, unvisited: bool, depth: usize) -> Result<Self> {
        if depth > 128 {
            return Err(failure(EvaluationError::Malformed {
                message: "evaluation tuple nesting exceeds 128".into(),
            }));
        }
        shape.deref(ctx).verify(ctx)?;
        let state = if shape::is_tuple(ctx, shape) {
            ValueState::Tuple(
                shape::children(ctx, shape)
                    .iter()
                    .map(|&child| Self::empty(ctx, child, unvisited, depth + 1))
                    .collect::<Result<_>>()?,
            )
        } else if shape.deref(ctx).is::<TokenType>() {
            ValueState::Token
        } else if shape::is_array(ctx, shape) {
            if unvisited {
                ValueState::Undetermined
            } else {
                ValueState::Unknown
            }
        } else {
            return Err(failure(EvaluationError::Malformed {
                message: "evaluation values require array, tuple or token shapes".into(),
            }));
        };
        Ok(Self { shape, state })
    }

    /// The declared value shape; literal leaves normalize missing/packed layouts.
    pub fn shape(&self) -> TypeHandle {
        self.shape
    }

    /// Borrows the knowledge state without copying literal or tuple storage.
    pub fn state(&self) -> &ValueState {
        &self.state
    }

    /// Returns whether every leaf can be materialized as a literal.
    ///
    /// Tokens and empty tuples are known; any unknown or unvisited array leaf
    /// makes the containing tree not fully known.
    pub fn is_known(&self) -> bool {
        match &self.state {
            ValueState::Known(_) | ValueState::Token => true,
            ValueState::Tuple(children) => children.iter().all(Self::is_known),
            ValueState::Unknown | ValueState::Undetermined => false,
        }
    }

    /// True for known or unknown leaves, false for unvisited leaves.
    pub fn is_determined(&self) -> bool {
        match &self.state {
            ValueState::Undetermined => false,
            ValueState::Tuple(children) => children.iter().all(Self::is_determined),
            ValueState::Known(_) | ValueState::Unknown | ValueState::Token => true,
        }
    }

    /// Borrows the subtree selected by a sequence of zero-based tuple indices.
    ///
    /// An empty path selects this value. Returns an error if an index is out
    /// of range or the path attempts to descend through a non-tuple state.
    pub fn subvalue(&self, path: &[usize]) -> Result<&Self> {
        let mut value = self;
        for &index in path {
            value = match &value.state {
                ValueState::Tuple(children) => children.get(index),
                _ => None,
            }
            .ok_or_else(|| {
                failure(EvaluationError::Malformed {
                    message: "evaluation tuple path is invalid".into(),
                })
            })?;
        }
        Ok(value)
    }

    /// Materialize only a completely known tree; unknown is a typed error.
    pub fn into_literal(self, ctx: &Context) -> Result<Literal> {
        match self.state {
            ValueState::Known(literal) => Ok(literal),
            ValueState::Token => Ok(Literal::token(ctx)),
            ValueState::Tuple(children) => {
                let children = children
                    .into_iter()
                    .map(|child| child.into_literal(ctx))
                    .collect::<Result<_>>()?;
                literal_semantics::relayout(ctx, &Literal::tuple(ctx, children)?, self.shape)
            }
            ValueState::Unknown | ValueState::Undetermined => {
                Err(failure(EvaluationError::UnknownValue))
            }
        }
    }

    pub(crate) fn tuple(ctx: &Context, shape: TypeHandle, children: Vec<Self>) -> Result<Self> {
        if !shape::is_tuple(ctx, shape) || shape::children(ctx, shape).len() != children.len() {
            return Err(failure(EvaluationError::Malformed {
                message: "evaluation tuple result has incompatible structure".into(),
            }));
        }
        let types = shape::children(ctx, shape).to_vec();
        let children = children
            .into_iter()
            .zip(types)
            .map(|(value, ty)| value.retarget(ctx, ty))
            .collect::<Result<_>>()?;
        Ok(Self {
            shape,
            state: ValueState::Tuple(children),
        })
    }

    pub(super) fn retarget(mut self, ctx: &Context, shape: TypeHandle) -> Result<Self> {
        if !shape::equal_ignoring_layout(ctx, self.shape, shape) {
            return Err(failure(EvaluationError::Malformed {
                message: "evaluation value changes element format, bounds or dynamic flags".into(),
            }));
        }
        match self.state {
            ValueState::Known(literal) => {
                self.state = ValueState::Known(literal_semantics::relayout(ctx, &literal, shape)?);
            }
            ValueState::Tuple(children) => return Self::tuple(ctx, shape, children),
            ValueState::Unknown | ValueState::Undetermined | ValueState::Token => {}
        }
        self.shape = shape;
        Ok(self)
    }
}
