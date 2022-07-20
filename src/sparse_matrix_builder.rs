use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use rustc_hash::FxHasher;

use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;

use crate::configuration::Column;
use crate::sparse_matrix::{Entry, Hash, SparseMatrix};

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
pub fn create_sparse_matrices_descriptors(cols: &[Column]) -> Vec<SparseMatrixDescriptor> {
    let mut sparse_matrix_builders: Vec<SparseMatrixDescriptor> = Vec::new();
    let num_fields = cols.len();
    let mut reflexive_count = 0;

    for i in 0..num_fields {
        for j in i..num_fields {
            let col_i = &cols[i];
            let col_j = &cols[j];
            if i < j && !(col_i.transient && col_j.transient) {
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

#[derive(Default, Debug, Clone, Copy)]
struct Entity {
    pub occurrence: u32,
    pub row_sum: f32,
    pub index: u32, // set in second stage
}

#[derive(Debug, Default, Clone)]
struct Edge {
    value: f32,
}

#[derive(Debug, Clone, Eq, PartialEq)]
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

impl SparseMatrixDescriptor {
    pub fn new(col_a_id: u8, col_a_name: String, col_b_id: u8, col_b_name: String) -> Self {
        Self {
            col_a_id,
            col_a_name,
            col_b_id,
            col_b_name,
        }
    }

    pub fn make_buffer(&self) -> SparseMatrixBuffer {
        SparseMatrixBuffer {
            descriptor: self.clone(),
            edge_count: 0,
            hash_2_row: Default::default(),
            hashes_2_edge: Default::default(),
        }
    }
}

#[derive(Debug)]
pub struct SparseMatrixBuffer {
    pub descriptor: SparseMatrixDescriptor,
    pub edge_count: u32,
    hash_2_row: HashMap<u64, Entity, BuildHasherDefault<FxHasher>>,
    hashes_2_edge: HashMap<(u64, u64), Edge, BuildHasherDefault<FxHasher>>,
}

impl SparseMatrixBuffer {
    pub fn handle_pair(&mut self, hashes: &[u64]) {
        let a = self.descriptor.col_a_id;
        let b = self.descriptor.col_b_id;
        self.add_pair_symmetric(
            hashes[(a + 1) as usize],
            hashes[(b + 1) as usize],
            hashes[0],
        );
    }

    fn add_pair_symmetric(&mut self, a_hash: u64, b_hash: u64, count: u64) {
        let value = 1f32 / (count as f32);

        self.update_row(a_hash, value);
        self.update_row(b_hash, value);

        self.edge_count += 1;

        self.update_edge(a_hash, b_hash, value);
        self.update_edge(b_hash, a_hash, value);
    }

    fn update_row(&mut self, hash: u64, val: f32) {
        let mut e = self.hash_2_row.entry(hash).or_default();
        e.occurrence += 1;
        e.row_sum += val;
    }

    fn update_edge(&mut self, a_hash: u64, b_hash: u64, val: f32) {
        let mut e = self.hashes_2_edge.entry((a_hash, b_hash)).or_default();
        e.value += val;
    }
}

#[derive(Debug)]
pub struct SparseMatrixBuffersReducer {
    descriptor: SparseMatrixDescriptor,
    buffers: Vec<SparseMatrixBuffer>,
}

impl SparseMatrixBuffersReducer {
    pub fn new(buffers: Vec<SparseMatrixBuffer>) -> Self {
        if buffers.is_empty() {
            panic!("Cannot reduce 0 buffers")
        }

        let descriptor = buffers[0].descriptor.clone();
        for buffer in &buffers {
            if descriptor != buffer.descriptor {
                panic!("Can only reduce buffers with the same sparse matrix description")
            }
        }

        Self {
            descriptor,
            buffers,
        }
    }

    pub fn reduce(self) -> SparseMatrix {
        let mut hash_2_row = self.reduce_row_maps();
        hash_2_row
            .iter_mut()
            .enumerate()
            .for_each(|(ix, (_, mut row))| row.index = ix as u32);

        let hashes_2_edge = self.reduce_edge_maps();

        let mut entries: Vec<_> = {
            hashes_2_edge
                .par_iter()
                .map(|((row_hash, col_hash), edge)| {
                    let row_entity = hash_2_row.get(row_hash).unwrap();
                    let col_entity = hash_2_row.get(col_hash).unwrap();

                    let normalized_edge_value = edge.value / row_entity.row_sum;

                    Entry {
                        row: row_entity.index,
                        col: col_entity.index,
                        value: normalized_edge_value,
                    }
                })
                .collect()
        };

        // Sort so we have better data locality in matrix propagation phase
        entries.par_sort_by_key(|e| (e.row, e.col));

        let hashes = hash_2_row
            .par_iter()
            .map(|(entity_hash, entity)| Hash {
                value: *entity_hash,
                occurrence: entity.occurrence,
            })
            .collect();

        SparseMatrix::new(self.descriptor, hashes, entries)
    }

    fn reduce_row_maps(&self) -> HashMap<u64, Entity, BuildHasherDefault<FxHasher>> {
        // There are duplicated keys. Empirically it works faster without handling dupes
        let row_keys: Vec<_> = self
            .buffers
            .iter()
            .flat_map(|b| b.hash_2_row.keys())
            .collect();

        row_keys
            .into_par_iter()
            .map(|hash| {
                let mut entity_agg = Entity::default();
                for b in self.buffers.iter() {
                    if let Some(entity) = b.hash_2_row.get(hash) {
                        entity_agg.occurrence += entity.occurrence;
                        entity_agg.row_sum += entity.row_sum;
                    }
                }
                (*hash, entity_agg)
            })
            .collect()
    }

    fn reduce_edge_maps(&self) -> HashMap<(u64, u64), Edge, BuildHasherDefault<FxHasher>> {
        // There are duplicated keys. Empirically it works faster without handling dupes
        let edges_keys: Vec<_> = self
            .buffers
            .iter()
            .flat_map(|b| b.hashes_2_edge.keys())
            .collect();

        edges_keys
            .into_par_iter()
            .map(|hash| {
                let mut edge_agg = Edge::default();
                for b in self.buffers.iter() {
                    if let Some(edge) = b.hashes_2_edge.get(hash) {
                        edge_agg.value += edge.value;
                    }
                }
                (*hash, edge_agg)
            })
            .collect()
    }
}
