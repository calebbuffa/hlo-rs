use super::*;

pub(super) fn constant_shape(shape: &Shape) -> Result<Shape> {
    let mut proto = shape.as_proto().clone();
    let mut pending = vec![&mut proto];
    while let Some(shape) = pending.pop() {
        if shape
            .element_type
            .enum_value()
            .is_ok_and(PrimitiveType::is_array_element)
        {
            if shape.dimensions.contains(&crate::UNBOUNDED_SIZE) {
                return Err(Error::new(
                    "unsupported",
                    "unbounded dimensions in text constants",
                ));
            }
            shape.is_dynamic_dimension = vec![false; shape.dimensions.len()];
        } else if shape.element_type.enum_value() == Ok(PrimitiveType::TUPLE) {
            pending.extend(&mut shape.tuple_shapes);
        }
    }
    Shape::from_proto(proto)
}

impl Parser {
    pub(super) fn parse_constant(
        &mut self,
        shape: &Option<Shape>,
        instruction: &mut HloInstructionProto,
        max_elements: usize,
    ) -> Result<()> {
        let shape = constant_shape(inferred::explicit(shape)?)?;
        let mut remaining = max_elements;
        let literal = self.literal(&shape, 0, &mut remaining)?;
        instruction.shape = Some(shape.into_proto()).into();
        instruction.literal = Some(literal.into_proto()).into();
        Ok(())
    }

    pub(super) fn typed_literal(&mut self, depth: usize, remaining: &mut usize) -> Result<Literal> {
        let mut current_depth = depth;
        let mut frames: Vec<Vec<Literal>> = Vec::new();
        'parse: loop {
            if current_depth >= MAX_DEPTH {
                return Err(Error::new("resource-limit", "typed literal nesting limit"));
            }
            let mut literal = if self.take("(") {
                if self.take(")") {
                    Literal::tuple(Vec::new())?
                } else {
                    frames.push(Vec::new());
                    current_depth += 1;
                    continue;
                }
            } else {
                let shape = self.shape(0)?;
                self.literal(&shape, current_depth, remaining)?
            };

            loop {
                let Some(children) = frames.last_mut() else {
                    return Ok(literal);
                };
                children.push(literal);
                if self.peek() == ")" {
                    self.expect(")")?;
                    let children = frames
                        .pop()
                        .ok_or_else(|| self.error("typed literal tuple stack underflow"))?;
                    literal = Literal::tuple(children)?;
                    current_depth -= 1;
                } else {
                    self.expect(",")?;
                    current_depth += 1;
                    continue 'parse;
                }
            }
        }
    }

    pub(super) fn literal(
        &mut self,
        shape: &Shape,
        depth: usize,
        remaining: &mut usize,
    ) -> Result<Literal> {
        struct Frame {
            shape: Shape,
            elements: Vec<Literal>,
            next_index: usize,
            depth: usize,
        }
        let mut current_shape = shape.clone();
        let mut current_depth = depth;
        let mut frames: Vec<Frame> = Vec::new();
        'parse: loop {
            if current_depth >= MAX_DEPTH {
                return Err(Error::new("resource-limit", "literal nesting limit"));
            }
            let literal = if current_shape.is_tuple() {
                self.expect("(")?;
                let count = current_shape.as_proto().tuple_shapes.len();
                if count == 0 {
                    self.expect(")")?;
                    Literal::tuple(Vec::new())?
                } else {
                    let child = current_shape.subshape(&[0])?;
                    frames.push(Frame {
                        shape: current_shape,
                        elements: Vec::with_capacity(count),
                        next_index: 1,
                        depth: current_depth,
                    });
                    current_shape = child;
                    current_depth += 1;
                    continue;
                }
            } else {
                let count = usize::try_from(current_shape.element_count()?)
                    .map_err(|_| self.error("literal count overflow"))?;
                *remaining = remaining.checked_sub(count).ok_or_else(|| {
                    Error::new("resource-limit", "literal element limit exceeded")
                })?;
                if self.peek() == "{"
                    && self
                        .tokens
                        .get(self.index + 1)
                        .is_some_and(|token| token.value == "...")
                {
                    self.expect("{")?;
                    self.expect("...")?;
                    self.expect("}")?;
                    if current_shape.rank()? == 0 {
                        return Err(self.error("elided constant must be an array of positive rank"));
                    }
                    let byte_width = current_shape
                        .element_type()
                        .bit_width()
                        .ok_or_else(|| self.error("constant requires array element type"))?
                        .div_ceil(8) as usize;
                    let mut bytes = vec![
                        0;
                        count.checked_mul(byte_width).ok_or_else(
                            || self.error("literal storage size overflow")
                        )?
                    ];
                    if self.options.fill_shortform_constants_with_random_values {
                        let mut counters = SHORTFORM_COUNTERS.lock().map_err(|_| {
                            Error::new("state", "short-form constant counter lock poisoned")
                        })?;
                        fill_shortform_bytes(
                            &mut bytes,
                            &mut counters,
                            current_shape.element_type() == PrimitiveType::PRED,
                        );
                    }
                    let mut physical = vec![0; bytes.len()];
                    for (logical, index) in
                        physical_indices(&current_shape)?.into_iter().enumerate()
                    {
                        physical[index * byte_width..(index + 1) * byte_width].copy_from_slice(
                            &bytes[logical * byte_width..(logical + 1) * byte_width],
                        );
                    }
                    Literal::from_storage_bytes(current_shape.clone(), &physical)?
                } else {
                    let mut values = Vec::with_capacity(count);
                    self.literal_values(&current_shape, 0, &mut values)?;
                    let physical = logical_to_physical(&current_shape, &values)?;
                    Literal::from_bits(current_shape.clone(), &physical)?
                }
            };

            let mut literal = literal;
            loop {
                let Some(frame) = frames.last_mut() else {
                    return Ok(literal);
                };
                frame.elements.push(literal);
                if frame.next_index < frame.shape.as_proto().tuple_shapes.len() {
                    self.expect(",")?;
                    let index = frame.next_index;
                    frame.next_index += 1;
                    current_shape = frame.shape.subshape(&[index])?;
                    current_depth = frame.depth + 1;
                    continue 'parse;
                }
                self.expect(")")?;
                let frame = frames
                    .pop()
                    .ok_or_else(|| self.error("literal parser tuple stack underflow"))?;
                literal = Literal::tuple(frame.elements)?;
            }
        }
    }
}

pub(super) fn print(literal: &Literal, shape: &Shape, output: &mut String) -> Result<()> {
    print_impl(literal, shape, output, false)
}

pub(super) fn print_typed(literal: &Literal, output: &mut String) -> Result<()> {
    print_impl(literal, literal.shape(), output, true)
}

fn print_impl(literal: &Literal, shape: &Shape, output: &mut String, typed: bool) -> Result<()> {
    fn with_defaults(shape: &Shape) -> Result<Shape> {
        enum Work {
            Shape(Shape),
            Tuple(usize),
        }
        let mut work = vec![Work::Shape(shape.clone())];
        let mut results = Vec::new();
        while let Some(item) = work.pop() {
            match item {
                Work::Tuple(count) => {
                    let start = results
                        .len()
                        .checked_sub(count)
                        .ok_or_else(|| Error::new("shape", "tuple default stack underflow"))?;
                    let children = results.split_off(start);
                    results.push(Shape::tuple(children)?);
                }
                Work::Shape(shape) if shape.is_tuple() => {
                    let count = shape.as_proto().tuple_shapes.len();
                    work.push(Work::Tuple(count));
                    for child in shape.as_proto().tuple_shapes.iter().rev() {
                        work.push(Work::Shape(Shape::from_proto(child.clone())?));
                    }
                }
                Work::Shape(shape) if shape.element_type() == PrimitiveType::TOKEN => {
                    return Err(Error::new("unsupported", "token literal text"));
                }
                Work::Shape(shape) if shape.layout().is_none() => {
                    results.push(shape.with_default_layout()?);
                }
                Work::Shape(shape) => results.push(shape),
            }
        }
        if results.len() != 1 {
            return Err(Error::new(
                "shape",
                "default layout traversal did not yield one shape",
            ));
        }
        results
            .pop()
            .ok_or_else(|| Error::new("shape", "default layout traversal did not yield one shape"))
    }

    enum Work {
        Literal(Box<Literal>, Shape),
        Text(&'static str),
    }
    let mut work = vec![Work::Literal(Box::new(literal.clone()), shape.clone())];
    while let Some(item) = work.pop() {
        let (literal, shape) = match item {
            Work::Text(text) => {
                output.push_str(text);
                continue;
            }
            Work::Literal(literal, shape) => (literal, shape),
        };
        if literal.shape().is_dynamic() {
            return Err(Error::new("unsupported", "dynamic literal text"));
        }
        if !literal.shape().compatible(&shape) {
            return Err(Error::new("literal", "constant shape mismatch"));
        }
        let expected = with_defaults(&shape)?;
        if print_shape(literal.shape())? != print_shape(&expected)? {
            return Err(Error::new(
                "unsupported",
                "constant instruction and literal have differing layouts or dynamic flags",
            ));
        }
        if shape.is_tuple() {
            reject_fields(literal.as_proto(), &["shape", "tuple_literals"])?;
            let count = literal.as_proto().tuple_literals.len();
            work.push(Work::Text(")"));
            for index in (0..count).rev() {
                let child = literal
                    .as_proto()
                    .tuple_literals
                    .get(index)
                    .ok_or_else(|| Error::new("literal", "missing tuple literal element"))?;
                let expected = literal.shape().subshape(&[index])?;
                if child.shape.as_ref() != Some(expected.as_proto()) {
                    return Err(Error::new(
                        "unsupported",
                        "tuple literal payload shape would be materialized or changed by text",
                    ));
                }
                work.push(Work::Literal(
                    Box::new(literal.sub_literal(&[index])?),
                    shape.subshape(&[index])?,
                ));
                if index > 0 {
                    work.push(Work::Text(", "));
                }
            }
            work.push(Work::Text("("));
            continue;
        }
        reject_fields(
            literal.as_proto(),
            &["shape", literal_field(shape.element_type())?],
        )?;
        let physical = literal.bits()?;
        let repacked = Literal::from_bits(literal.shape().clone(), &physical)?;
        if crate::interchange::encode(repacked.as_proto())?
            != crate::interchange::encode(literal.as_proto())?
        {
            return Err(Error::new(
                "unsupported",
                "literal contains storage bits not representable in HLO text",
            ));
        }
        let logical = physical_to_logical(literal.shape(), &physical)?;
        let mut offset = 0;
        if typed {
            write!(output, "{} ", print_shape(&shape)?)
                .map_err(|e| Error::new("print", e.to_string()))?;
        }
        print_literal(&shape, &logical, 0, &mut offset, output)?;
        if offset != logical.len() {
            return Err(Error::new(
                "literal",
                "literal printer left unconsumed elements",
            ));
        }
    }
    Ok(())
}
