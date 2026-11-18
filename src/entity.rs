use std::hash::Hasher;

use smallvec::SmallVec;
use twox_hash::XxHash64;

use crate::configuration::Configuration;

/// Number of entities of one field kept on the stack before spilling to the heap.
pub const SMALL_VECTOR_SIZE: usize = 8;

pub type Tokens<'a> = SmallVec<[&'a str; SMALL_VECTOR_SIZE]>;

/// Hash identifying an entity. Entities are identified by their name only, so the
/// same name appearing in two columns denotes the same entity.
#[inline(always)]
pub fn hash_entity(entity: &str) -> u64 {
    let mut hasher = XxHash64::default();
    hasher.write(entity.as_bytes());
    hasher.finish()
}

/// Why a line could not be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineError {
    /// Wrong number of fields for the declared columns.
    ColumnCount { expected: usize, found: usize },
    /// A field that should hold entities is empty.
    EmptyField,
    /// The weight field is not a finite, non-negative number.
    BadWeight,
    /// The line is not valid UTF-8.
    InvalidUtf8,
}

impl LineError {
    pub fn describe(&self) -> String {
        match self {
            LineError::ColumnCount { expected, found } => format!(
                "expected {} field(s) but found {} (fields are separated by a tab or, \
                 if the line has no tab, by a comma; entities inside a field by spaces)",
                expected, found
            ),
            LineError::EmptyField => "a field holds no entity".to_string(),
            LineError::BadWeight => "the weight is not a finite, non-negative number".to_string(),
            LineError::InvalidUtf8 => "the line is not valid UTF-8".to_string(),
        }
    }
}

/// A line split into its entity fields.
pub struct ParsedLine<'a> {
    /// Entities of every entity column, in column order.
    pub fields: SmallVec<[Tokens<'a>; 4]>,
    pub weight: f32,
}

/// Largest weight accepted for a single line / edge.
pub const MAX_WEIGHT: f32 = 1e9;

#[inline]
pub fn weight_is_valid(weight: f32) -> bool {
    weight.is_finite() && (0.0..=MAX_WEIGHT).contains(&weight)
}

/// Splits a line into fields and entities.
///
/// Returns `Ok(None)` for blank lines. Fields are separated by a tab; a line without
/// any tab is split on commas instead; a line with neither is a single field.
/// Entities within a field are separated by spaces. A non-`complex` field
/// contributes its first entity only.
pub fn parse_line<'a>(
    line: &'a str,
    config: &Configuration,
) -> Result<Option<ParsedLine<'a>>, LineError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let expected = config.columns.len();
    let separator = if trimmed.contains('\t') {
        Some('\t')
    } else if trimmed.contains(',') {
        Some(',')
    } else {
        None
    };

    let mut fields: SmallVec<[Tokens<'a>; 4]> = SmallVec::new();
    let mut weight = 1.0f32;
    let mut found = 0usize;

    let mut handle_field = |index: usize, field: &'a str| -> Result<(), LineError> {
        // Fields beyond the declared ones only matter for the error message.
        let Some(column) = config.columns.get(index) else {
            return Ok(());
        };
        if column.weight {
            weight = field
                .trim()
                .parse::<f32>()
                .ok()
                .filter(|w| weight_is_valid(*w))
                .ok_or(LineError::BadWeight)?;
            return Ok(());
        }
        let mut tokens: Tokens<'a> = SmallVec::new();
        for token in field.split(' ') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            tokens.push(token);
            if !column.complex {
                break;
            }
        }
        if tokens.is_empty() {
            return Err(LineError::EmptyField);
        }
        fields.push(tokens);
        Ok(())
    };

    let mut first_error = None;
    match separator {
        Some(sep) => {
            for field in trimmed.split(sep) {
                if let Err(e) = handle_field(found, field) {
                    first_error.get_or_insert(e);
                }
                found += 1;
            }
        }
        None => {
            if let Err(e) = handle_field(0, trimmed) {
                first_error.get_or_insert(e);
            }
            found = 1;
        }
    }

    if found != expected {
        return Err(LineError::ColumnCount { expected, found });
    }
    if let Some(e) = first_error {
        return Err(e);
    }
    Ok(Some(ParsedLine { fields, weight }))
}

/// Formats an integer identifier exactly like Python's `str(int)`, without allocating.
pub struct IntBuffer {
    bytes: [u8; 20],
    start: usize,
}

impl IntBuffer {
    #[inline]
    pub fn new(value: i64) -> IntBuffer {
        let mut bytes = [0u8; 20];
        let mut pos = bytes.len();
        let negative = value < 0;
        let mut rest = value.unsigned_abs();
        loop {
            pos -= 1;
            bytes[pos] = b'0' + (rest % 10) as u8;
            rest /= 10;
            if rest == 0 {
                break;
            }
        }
        if negative {
            pos -= 1;
            bytes[pos] = b'-';
        }
        IntBuffer { bytes, start: pos }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        // Only ASCII digits and '-' are ever written.
        std::str::from_utf8(&self.bytes[self.start..]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(columns: &str) -> Configuration {
        Configuration::new(columns, 16, Some(1)).unwrap()
    }

    fn fields(line: &str, columns: &str) -> Vec<Vec<String>> {
        parse_line(line, &config(columns))
            .unwrap()
            .unwrap()
            .fields
            .iter()
            .map(|f| f.iter().map(|t| t.to_string()).collect())
            .collect()
    }

    #[test]
    fn int_buffer_matches_display() {
        for v in [0i64, 7, -7, 10, 1234567890123, i64::MAX, i64::MIN, -1] {
            assert_eq!(IntBuffer::new(v).as_str(), v.to_string());
        }
    }

    #[test]
    fn single_reflexive_field() {
        assert_eq!(
            fields("a b  c ", "complex::reflexive::x"),
            vec![vec!["a", "b", "c"]]
        );
    }

    #[test]
    fn tab_and_comma_separated_fields() {
        assert_eq!(
            fields("u1\tp1 p2", "user complex::product"),
            vec![vec!["u1"], vec!["p1", "p2"]]
        );
        assert_eq!(
            fields("u1, p1 p2", "user complex::product"),
            vec![vec!["u1"], vec!["p1", "p2"]]
        );
        // a simple (non-complex) field keeps its first entity only
        assert_eq!(
            fields("u1 u9\tp1 p2", "user product"),
            vec![vec!["u1"], vec!["p1"]]
        );
    }

    #[test]
    fn weight_field() {
        let c = config("user product weight::w");
        let parsed = parse_line("u\tp\t2.5", &c).unwrap().unwrap();
        assert_eq!(parsed.weight, 2.5);
        assert_eq!(parsed.fields.len(), 2);
        assert_eq!(parse_line("u\tp\t-1", &c).err(), Some(LineError::BadWeight));
        assert_eq!(
            parse_line("u\tp\tnan", &c).err(),
            Some(LineError::BadWeight)
        );
        assert_eq!(
            parse_line("u\tp\tabc", &c).err(),
            Some(LineError::BadWeight)
        );
    }

    #[test]
    fn malformed_lines() {
        let c = config("user product");
        assert!(matches!(
            parse_line("u p", &c),
            Err(LineError::ColumnCount {
                expected: 2,
                found: 1
            })
        ));
        assert!(matches!(
            parse_line("u\tp\tx", &c),
            Err(LineError::ColumnCount {
                expected: 2,
                found: 3
            })
        ));
        assert_eq!(
            parse_line("u\t ", &c).err(),
            Some(LineError::ColumnCount {
                expected: 2,
                found: 1
            })
        );
        assert_eq!(
            parse_line("u\t \tp", &config("a b c")).err(),
            Some(LineError::EmptyField)
        );
        assert!(parse_line("   ", &c).unwrap().is_none());
        assert!(parse_line("", &c).unwrap().is_none());
    }
}
