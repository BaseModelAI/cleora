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
    pub col_a_id: u8,
    pub col_a_name: String,
    pub col_b_id: u8,
    pub col_b_name: String,
}

#[pyclass(name = "SparseMatrix", module = "pycleora.pycleora")]
#[derive(Debug, Serialize, Deserialize)]
pub struct SparseMatrix {
    pub descriptor: SparseMatrixDescriptor,
    #[pyo3(get, set)]
    pub entity_ids: Vec<String>,
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
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
