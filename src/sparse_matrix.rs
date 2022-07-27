use crate::configuration::Column;

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

/// Represents graph based on incoming data.
/// It follows the sparse matrix coordinate format (COO). Its purpose is to save space by holding only
/// the coordinates and values of nonzero entities.
#[derive(Debug)]
pub struct SparseMatrix {
    pub descriptor: SparseMatrixDescriptor,
    // Graph nodes
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
    /// Maps entities to its edges
    pub slices: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub struct Edge {
    pub other_entity_ix: u32,
    pub value: f32,
}

impl SparseMatrix {
    pub fn new(
        descriptor: SparseMatrixDescriptor,
        entities: Vec<Entity>,
        edges: Vec<Edge>,
        slices: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            descriptor,
            entities,
            edges,
            slices,
        }
    }
}

/// Hash data
#[derive(Debug, Clone, Copy)]
pub struct Entity {
    /// Value of the hash
    pub hash_value: u64,

    /// Number of hash occurrences
    pub occurrence: u32,
}

/// Sparse matrix reader used in embedding process
pub trait SparseMatrixReader {
    /// Returns sparse matrix identifier
    fn get_id(&self) -> String;

    /// Returns total number of unique entities
    fn get_number_of_entities(&self) -> u32;

    /// Returns total number of entries
    fn get_number_of_entries(&self) -> u32;
}

pub struct CopyIter<'a, T: Copy>(std::slice::Iter<'a, T>);

impl<T: Copy> Iterator for CopyIter<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().copied()
    }
}

impl SparseMatrixReader for SparseMatrix {
    fn get_id(&self) -> String {
        format!("{}_{}", self.descriptor.col_a_id, self.descriptor.col_b_id)
    }

    fn get_number_of_entities(&self) -> u32 {
        self.entities.len() as u32
    }

    fn get_number_of_entries(&self) -> u32 {
        self.edges.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use crate::configuration::Column;
    use crate::sparse_matrix::{
        create_sparse_matrices_descriptors, SparseMatrixDescriptor, SparseMatrixReader,
    };
    use crate::sparse_matrix_builder::{NodeIndexer, SparseMatrixBuffersReducer};
    use rustc_hash::FxHasher;
    use std::collections::{HashMap, HashSet};
    use std::hash::Hasher;

    fn map_to_ids_and_names(
        sparse_matrices: &[SparseMatrixDescriptor],
    ) -> HashSet<(u8, &str, u8, &str)> {
        sparse_matrices
            .iter()
            .map(|desc| {
                (
                    desc.col_a_id,
                    desc.col_a_name.as_str(),
                    desc.col_b_id,
                    desc.col_b_name.as_str(),
                )
            })
            .collect()
    }

    #[test]
    fn create_sparse_matrices_if_no_columns_provided() {
        let sparse_matrices = create_sparse_matrices_descriptors(&[]);
        assert_eq!(true, sparse_matrices.is_empty())
    }

    #[test]
    fn create_sparse_matrices_if_transient_columns_provided() {
        let mut columns = vec![
            Column {
                name: String::from("a"),
                transient: true,
                ..Default::default()
            },
            Column {
                name: String::from("b"),
                transient: true,
                ..Default::default()
            },
        ];
        let sparse_matrices = create_sparse_matrices_descriptors(&columns);
        assert_eq!(true, sparse_matrices.is_empty());

        columns.push(Column {
            name: String::from("c"),
            complex: true,
            ..Default::default()
        });
        let sparse_matrices: Vec<_> = create_sparse_matrices_descriptors(&columns);
        let sparse_matrices: HashSet<_> = map_to_ids_and_names(&sparse_matrices);
        let expected_sparse_matrices: HashSet<_> = [(0, "a", 2, "c"), (1, "b", 2, "c")]
            .iter()
            .cloned()
            .collect();
        assert_eq!(expected_sparse_matrices, sparse_matrices)
    }

    #[test]
    fn create_sparse_matrices_if_reflexive_columns_provided() {
        let sparse_matrices = create_sparse_matrices_descriptors(&[
            Column {
                name: String::from("a"),
                ..Default::default()
            },
            Column {
                name: String::from("b"),
                transient: true,
                ..Default::default()
            },
            Column {
                name: String::from("c"),
                complex: true,
                reflexive: true,
                ..Default::default()
            },
            Column {
                name: String::from("d"),
                complex: true,
                ..Default::default()
            },
        ]);

        let sparse_matrices: HashSet<_> = map_to_ids_and_names(&sparse_matrices);
        let expected_sparse_matrices: HashSet<_> = [
            (0, "a", 1, "b"),
            (0, "a", 2, "c"),
            (0, "a", 3, "d"),
            (1, "b", 2, "c"),
            (1, "b", 3, "d"),
            (2, "c", 3, "d"),
            (2, "c", 4, "c"),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(expected_sparse_matrices, sparse_matrices)
    }
}
