use crate::sparse_matrix::SparseMatrixDescriptor;

#[derive(Debug)]
pub struct Configuration {
    pub seed: Option<i64>,
    pub matrix_desc: SparseMatrixDescriptor,
    pub columns: Vec<Column>,
    pub hyperedge_trim_n: usize,
    pub num_workers_graph_building: usize,
}

#[derive(Debug, Default)]
pub struct Column {
    /// Name, header of the column
    pub name: String,

    /// The field is composite, containing multiple entity identifiers separated by space
    pub complex: bool,

    /// The field is reflexive, which means that it interacts with itself, additional output file is written for every such field
    pub reflexive: bool,
}

/// Extract columns config based on raw strings.
pub fn parse_fields(columns: &str) -> Result<Vec<Column>, String> {
    let cols: Vec<&str> = columns.split(' ').collect();

    let mut columns: Vec<Column> = Vec::new();
    for col in cols {
        let parts: Vec<&str> = col.split("::").collect();

        let column_name: &str;
        let mut complex = false;
        let mut reflexive = false;

        let parts_len = parts.len();
        if parts_len > 1 {
            column_name = *parts.last().unwrap();
            let column_name_idx = parts_len - 1;
            for &part in &parts[..column_name_idx] {
                if part.eq_ignore_ascii_case("complex") {
                    complex = true;
                } else if part.eq_ignore_ascii_case("reflexive") {
                    reflexive = true;
                } else {
                    let message = format!("Unrecognized column field modifier: {}", part);
                    return Err(message);
                }
            }
        } else {
            column_name = col;
        }
        let column = Column {
            name: column_name.to_string(),
            complex,
            reflexive,
        };
        columns.push(column);
    }

    let columns = validate_column_modifiers(columns)?;
    Ok(columns)
}

fn validate_column_modifiers(cols: Vec<Column>) -> Result<Vec<Column>, String> {
    for col in &cols {
        // transient::reflexive - this would generate no output
        // transient::reflexive::complex - this would generate no output
        if col.reflexive && !col.complex {
            let message = format!(
                "A field cannot be REFLEXIVE but NOT COMPLEX. It does not make sense: {}",
                col.name
            );
            return Err(message);
        }
    }
    Ok(cols)
}
