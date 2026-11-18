use serde::{Deserialize, Serialize};

/// One column of the input, as declared in the `columns` string.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    /// The field holds several space-separated entities.
    pub complex: bool,
    /// The column is also related to itself (co-occurrence within the field).
    pub reflexive: bool,
    /// The field holds the (non-negative) weight of the whole line, not an entity.
    pub weight: bool,
}

/// A relation between two entity columns. `a == b` denotes a reflexive relation
/// (a column related to itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relation {
    pub a: u8,
    pub b: u8,
}

#[derive(Debug, Clone)]
pub struct Configuration {
    /// Every declared column, in input order (including the weight column, if any).
    pub columns: Vec<Column>,
    /// Names of the entity columns, in input order.
    pub entity_column_names: Vec<String>,
    /// For every declared column: its index among the entity columns (`None` for the weight column).
    pub entity_column_of: Vec<Option<u8>>,
    pub weight_column: Option<usize>,
    /// Relations between entity columns (indices into `entity_column_names`).
    pub relations: Vec<Relation>,
    pub hyperedge_trim_n: usize,
    pub num_workers: usize,
}

impl Configuration {
    pub fn new(
        columns: &str,
        hyperedge_trim_n: usize,
        num_workers: Option<usize>,
    ) -> Result<Configuration, String> {
        let columns = parse_fields(columns)?;

        let mut entity_column_names = Vec::new();
        let mut entity_column_of = Vec::with_capacity(columns.len());
        let mut weight_column = None;
        for (i, column) in columns.iter().enumerate() {
            if column.weight {
                weight_column = Some(i);
                entity_column_of.push(None);
            } else {
                entity_column_of.push(Some(entity_column_names.len() as u8));
                entity_column_names.push(column.name.clone());
            }
        }
        if entity_column_names.len() > u8::MAX as usize {
            return Err(format!(
                "Too many entity columns ({}); at most {} are supported",
                entity_column_names.len(),
                u8::MAX
            ));
        }

        // Same enumeration order as the original implementation:
        // for every column, first its reflexive relation, then the relations with later columns.
        let entity_columns: Vec<&Column> = columns.iter().filter(|c| !c.weight).collect();
        let mut relations = Vec::new();
        for i in 0..entity_columns.len() {
            if entity_columns[i].reflexive {
                relations.push(Relation {
                    a: i as u8,
                    b: i as u8,
                });
            }
            for j in (i + 1)..entity_columns.len() {
                relations.push(Relation {
                    a: i as u8,
                    b: j as u8,
                });
            }
        }
        if relations.is_empty() {
            return Err(
                "The columns definition must describe at least one relation: use two or more \
                 entity columns, or mark a column as 'complex::reflexive::<name>'."
                    .to_string(),
            );
        }
        if hyperedge_trim_n == 0 {
            return Err("hyperedge_trim_n must be at least 1".to_string());
        }

        Ok(Configuration {
            columns,
            entity_column_names,
            entity_column_of,
            weight_column,
            relations,
            hyperedge_trim_n,
            num_workers: resolve_num_workers(num_workers),
        })
    }
}

/// Number of worker threads to use: the requested number, or all available cores.
pub fn resolve_num_workers(num_workers: Option<usize>) -> usize {
    match num_workers {
        Some(n) if n > 0 => n,
        _ => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    }
}

pub fn parse_fields(columns: &str) -> Result<Vec<Column>, String> {
    let mut parsed: Vec<Column> = Vec::new();
    for col in columns.split_whitespace() {
        let parts: Vec<&str> = col.split("::").collect();
        let column_name = *parts.last().unwrap();
        if column_name.is_empty() {
            return Err(format!("Column definition '{}' has an empty name", col));
        }

        let mut column = Column {
            name: column_name.to_string(),
            ..Default::default()
        };
        for &part in &parts[..parts.len() - 1] {
            if part.eq_ignore_ascii_case("complex") {
                column.complex = true;
            } else if part.eq_ignore_ascii_case("reflexive") {
                column.reflexive = true;
            } else if part.eq_ignore_ascii_case("weight") {
                column.weight = true;
            } else {
                return Err(format!("Unrecognized column field modifier: {}", part));
            }
        }
        parsed.push(column);
    }

    if parsed.is_empty() {
        return Err("The columns definition is empty".to_string());
    }
    validate_column_modifiers(parsed)
}

fn validate_column_modifiers(cols: Vec<Column>) -> Result<Vec<Column>, String> {
    let mut weight_columns = 0;
    for col in &cols {
        if col.reflexive && !col.complex {
            return Err(format!(
                "A field cannot be REFLEXIVE but NOT COMPLEX. It does not make sense: {}",
                col.name
            ));
        }
        if col.weight {
            weight_columns += 1;
            if col.complex || col.reflexive {
                return Err(format!(
                    "A WEIGHT field holds a single number and cannot be COMPLEX or REFLEXIVE: {}",
                    col.name
                ));
            }
        }
    }
    if weight_columns > 1 {
        return Err("At most one column can be marked as 'weight::<name>'".to_string());
    }
    Ok(cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_reflexive_column() {
        let c = Configuration::new("complex::reflexive::product", 16, Some(1)).unwrap();
        assert_eq!(c.relations, vec![Relation { a: 0, b: 0 }]);
        assert_eq!(c.entity_column_names, vec!["product"]);
        assert_eq!(c.weight_column, None);
    }

    #[test]
    fn two_columns() {
        let c = Configuration::new("user complex::product", 16, Some(1)).unwrap();
        assert_eq!(c.relations, vec![Relation { a: 0, b: 1 }]);
    }

    #[test]
    fn several_relations_and_weight() {
        let c = Configuration::new(
            "user complex::reflexive::product weight::w brand",
            16,
            Some(1),
        )
        .unwrap();
        assert_eq!(c.entity_column_names, vec!["user", "product", "brand"]);
        assert_eq!(c.weight_column, Some(2));
        assert_eq!(c.entity_column_of, vec![Some(0), Some(1), None, Some(2)]);
        assert_eq!(
            c.relations,
            vec![
                Relation { a: 0, b: 1 },
                Relation { a: 0, b: 2 },
                Relation { a: 1, b: 1 },
                Relation { a: 1, b: 2 },
            ]
        );
    }

    #[test]
    fn rejects_bad_definitions() {
        assert!(Configuration::new("node", 16, None).is_err());
        assert!(Configuration::new("", 16, None).is_err());
        assert!(Configuration::new("reflexive::node", 16, None).is_err());
        assert!(Configuration::new("a b weight::x weight::y", 16, None).is_err());
        assert!(Configuration::new("a complex::weight::w", 16, None).is_err());
        assert!(Configuration::new("a bogus::b", 16, None).is_err());
        assert!(Configuration::new("a b", 0, None).is_err());
    }

    #[test]
    fn tolerates_repeated_whitespace() {
        let c = Configuration::new("  user   product ", 16, Some(2)).unwrap();
        assert_eq!(c.entity_column_names, vec!["user", "product"]);
        assert_eq!(c.num_workers, 2);
    }
}
