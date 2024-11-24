use crate::configuration::Column;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};

pub fn create_sparse_matrix_descriptor(
    colums: &Vec<Column>,
) -> Result<SparseMatrixDescriptor, &'static str> {
    let mut matrices_descs = create_sparse_matrices_descriptors(colums);
    if matrices_descs.len() != 1 {
        return Err("More than one relation! Adjust your columns so there is only one relation.");
    }
    Ok(matrices_descs.remove(0))
}

/// Creates combinations of column pairs as sparse matrices.
/// Let's say that we have such columns configuration: complex::a reflexive::complex::b c. This is provided
/// as `&[Column]` after parsing the config.
/// The allowed column modifiers are:
/// - transient - the field is virtual - it is considered during embedding process, no entity is written for the column,
/// - complex   - the field is composite, containing multiple entity identifiers separated by space,
/// - reflexive - the field is reflexive, which means that it interacts with itself, additional output file is written for every such field.
/// We create sparse matrix for every columns relations (based on column modifiers).
/// For our example we have:
/// - sparse matrix for column a and b,
/// - sparse matrix for column a and c,
/// - sparse matrix for column b and c,
/// - sparse matrix for column b and b (reflexive column).
/// Apart from column names in sparse matrix we provide indices for incoming data. We have 3 columns such as a, b and c
/// but column b is reflexive so we need to include this column. The result is: (a, b, c, b).
/// The rule is that every reflexive column is append with the order of occurrence to the end of constructed array.
pub fn create_sparse_matrices_descriptors(cols: &Vec<Column>) -> Vec<SparseMatrixDescriptor> {
    let mut sparse_matrix_builders: Vec<SparseMatrixDescriptor> = Vec::new();
    let num_fields = cols.len();
    let mut reflexive_count = 0;

    for i in 0..num_fields {
        for j in i..num_fields {
            let col_i = &cols[i];
            let col_j = &cols[j];
            if i < j {
                let sm = SparseMatrixDescriptor::new(
                    i as u8,
                    col_i.name.clone(),
                    j as u8,
                    col_j.name.clone(),
                );
                sparse_matrix_builders.push(sm);
            } else if i == j && col_i.reflexive {
                let new_j = num_fields + reflexive_count;
                reflexive_count += 1;
                let sm = SparseMatrixDescriptor::new(
                    i as u8,
                    col_i.name.clone(),
                    new_j as u8,
                    col_j.name.clone(),
                );
                sparse_matrix_builders.push(sm);
            }
        }
    }
    sparse_matrix_builders
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct SparseMatrixDescriptor {
    /// First column index for which we creates subgraph
    pub col_a_id: u8,

    /// First column name
    pub col_a_name: String,

    /// Second column index for which we creates subgraph
    pub col_b_id: u8,

    /// Second column name
    pub col_b_name: String,
}

#[pyclass(name = "SparseMatrix", module = "cleora")]
#[derive(Debug, Serialize, Deserialize)]
pub struct SparseMatrix {
    pub descriptor: SparseMatrixDescriptor,
    #[pyo3(get, set)]
    pub entity_ids: Vec<String>,
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
    /// Maps entities to its edges
    /// I-th slice represent edges going out of ith node
    /// Example:
    /// Given slices=[(0, 4), (4, 10), (10, 11)]
    /// edges[0..4] are outgoing edges for entity=0
    /// edges[4..10] are outgoing edges for entity=1
    /// edges[10..11] are outgoing edges for entity=2
    pub slices: Vec<(usize, usize)>,
    pub column_ids: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Entity {
    pub row_sum: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Edge {
    pub other_entity_ix: u32,
    pub left_markov_value: f32,
    pub symmetric_markov_value: f32,
}
