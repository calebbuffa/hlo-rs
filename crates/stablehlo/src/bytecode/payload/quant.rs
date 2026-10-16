use super::*;

impl PayloadDecoder<'_, '_> {
    pub(super) fn quant_type(
        &self,
        cursor: &mut Cursor<'_>,
        depth: usize,
    ) -> Result<String, BytecodeError> {
        let tag = cursor.varint()?;
        if !matches!(tag, 4 | 5) {
            return Err(cursor.error(Kind::Unsupported("non-uniform Quant dialect type")));
        }
        let flags = cursor.varint()?;
        if flags > 1 {
            return Err(cursor.error(Kind::InvalidPayload("known quantization flags")));
        }
        let storage = Self::element(self.type_ref(cursor, depth)?, cursor)?;
        let width = storage
            .bit_width()
            .filter(|_| storage.is_integer())
            .ok_or_else(|| cursor.error(Kind::InvalidPayload("integer quantized storage type")))?;
        let storage = text::parse_type(&format!("{}i{width}", if flags == 0 { "u" } else { "" }))
            .map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))?;
        let storage = Self::element(storage, cursor)?;
        let expressed = Self::element(self.type_ref(cursor, depth)?, cursor)?;
        let (axis, minimum, maximum, scales, points) = if tag == 4 {
            let scale = f64::from_bits(Self::bits(cursor, 64)?);
            let point = i128::from(Self::signed(cursor)?);
            let minimum = i128::from(Self::signed(cursor)?);
            let maximum = i128::from(Self::signed(cursor)?);
            (None, minimum, maximum, vec![scale], vec![point])
        } else {
            let axis =
                usize::try_from(cursor.varint()?).map_err(|_| cursor.error(Kind::ResourceLimit))?;
            let minimum = i128::from(Self::signed(cursor)?);
            let maximum = i128::from(Self::signed(cursor)?);
            let scales = (0..cursor.count()?)
                .map(|_| Self::bits(cursor, 64).map(f64::from_bits))
                .collect::<Result<_, _>>()?;
            let points = Self::signed_list(cursor)?
                .into_iter()
                .map(i128::from)
                .collect();
            (Some(axis), minimum, maximum, scales, points)
        };
        let (default_min, default_max) = storage
            .integer_bounds()
            .ok_or_else(|| cursor.error(Kind::InvalidPayload("quantized storage bounds")))?;
        let element = crate::QuantizedElementType::new(
            storage,
            (minimum != default_min).then_some(minimum),
            (maximum != default_max).then_some(maximum),
            expressed,
            axis,
            scales,
            points,
        );
        element
            .verify()
            .map_err(|error| cursor.error(Kind::InvalidType(Box::new(error))))?;
        text::print_type(&Type::QuantizedElement(element))
            .map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))
    }
}
