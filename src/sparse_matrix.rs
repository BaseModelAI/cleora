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

    /// Maps id to hash value and occurrence
    id_2_hash: Vec<Hash>,

    /// Coordinates and values of nonzero entities
    entries: Vec<Entry>,
}

impl SparseMatrix {
    pub fn new(
        descriptor: SparseMatrixDescriptor,
        id_2_hash: Vec<Hash>,
        entries: Vec<Entry>,
    ) -> Self {
        Self {
            descriptor,
            id_2_hash,
            entries,
        }
    }
}

/// Hash data
#[derive(Debug, Clone, Copy)]
pub struct Hash {
    /// Value of the hash
    pub value: u64,

    /// Number of hash occurrences
    pub occurrence: u32,
}

/// Sparse matrix coordinate entry
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Entry {
    /// Matrix row
    pub row: u32,

    /// Matrix column
    pub col: u32,

    /// Matrix value
    pub value: f32,
}

/// Sparse matrix reader used in embedding process
pub trait SparseMatrixReader {
    /// Returns sparse matrix identifier
    fn get_id(&self) -> String;

    /// Returns total number of unique entities
    fn get_number_of_entities(&self) -> u32;

    /// Returns total number of entries
    fn get_number_of_entries(&self) -> u32;

    /// Returns iterator for hash data such as id and occurrence
    fn iter_hashes(&self) -> CopyIter<'_, Hash>;

    /// Returns iterator for entries
    fn iter_entries(&self) -> CopyIter<'_, Entry>;
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
        self.id_2_hash.len() as u32
    }

    fn get_number_of_entries(&self) -> u32 {
        self.entries.len() as u32
    }

    #[inline]
    fn iter_hashes(&self) -> CopyIter<'_, Hash> {
        CopyIter(self.id_2_hash.iter())
    }

    #[inline]
    fn iter_entries(&self) -> CopyIter<'_, Entry> {
        CopyIter(self.entries.iter())
    }
}

#[cfg(test)]
mod tests {
    use crate::configuration::Column;
    use crate::sparse_matrix::{
        create_sparse_matrices_descriptors, Entry, SparseMatrixDescriptor, SparseMatrixReader,
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

    fn prepare_entries(hash_2_id: HashMap<u64, u32>, edges: Vec<(&str, &str, f32)>) -> Vec<Entry> {
        let mut row_sum: Vec<f32> = Vec::with_capacity(hash_2_id.len() as usize);
        for _ in 0..hash_2_id.len() {
            row_sum.push(0.0);
        }

        let mut entries: Vec<_> = Vec::new();
        for (row, col, val) in edges {
            // undirected graph needs (row, col) and (col, row) edges
            let row = *hash_2_id.get(&hash(row)).unwrap();
            let col = *hash_2_id.get(&hash(col)).unwrap();
            let entry_row_col = Entry {
                row,
                col,
                value: val,
            };
            entries.push(entry_row_col);
            row_sum[entry_row_col.row as usize] += val;

            let entry_col_row = Entry {
                row: col,
                col: row,
                value: val,
            };
            entries.push(entry_col_row);
            row_sum[entry_col_row.row as usize] += val;
        }

        for mut entry in entries.iter_mut() {
            entry.value /= row_sum[entry.row as usize]
        }

        entries
    }

    fn hash(entity: &str) -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write(entity.as_bytes());
        hasher.finish()
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

    #[test]
    fn create_sparse_matrix_for_undirected_graph() {
        let sm_desc =
            SparseMatrixDescriptor::new(0u8, String::from("col_0"), 1u8, String::from("col_1"));

        let mut sm = sm_desc.make_buffer();

        // input line:
        // u1	p1 p2	b1 b2
        sm.handle_pair(&[4, hash("u1"), hash("p1"), hash("b1")]);
        sm.handle_pair(&[4, hash("u1"), hash("p1"), hash("b2")]);
        sm.handle_pair(&[4, hash("u1"), hash("p2"), hash("b1")]);
        sm.handle_pair(&[4, hash("u1"), hash("p2"), hash("b2")]);

        // input line:
        // u2	p2 p3 p4	b1
        sm.handle_pair(&[3, hash("u2"), hash("p2"), hash("b1")]);
        sm.handle_pair(&[3, hash("u2"), hash("p3"), hash("b1")]);
        sm.handle_pair(&[3, hash("u2"), hash("p4"), hash("b1")]);

        let node_indexer = {
            // [user, product] columns specified in sparse matrix description
            let nodes = vec!["u1", "u2", "p1", "p2", "p3", "p4"];
            let node_hashes: Vec<_> = nodes.iter().map(|n| hash(n)).collect();
            NodeIndexer {
                key_2_index: node_hashes
                    .iter()
                    .enumerate()
                    .map(|(ix, hash)| (*hash, ix))
                    .collect(),
                index_2_key: node_hashes,
            }
        };
        let sm = SparseMatrixBuffersReducer::new(node_indexer, vec![sm]).reduce();

        // number of unique entities
        assert_eq!(6, sm.get_number_of_entities());

        // number of edges for entities
        assert_eq!(10, sm.get_number_of_entries());

        let hash_2_id: HashMap<_, _> = sm
            .iter_hashes()
            .enumerate()
            .map(|id_and_hash| (id_and_hash.1.value, id_and_hash.0 as u32))
            .collect();
        // number of hashes
        assert_eq!(6, hash_2_id.len());

        // every relation for undirected graph is represented as two edges, for example:
        // (u1, p1, value) and (p1, u1, value)
        let edges = vec![
            ("u1", "p1", 1.0 / 2.0),
            ("u1", "p2", 1.0 / 2.0),
            ("u2", "p2", 1.0 / 3.0),
            ("u2", "p3", 1.0 / 3.0),
            ("u2", "p4", 1.0 / 3.0),
        ];
        let mut expected_entries = prepare_entries(hash_2_id, edges);
        let mut entries: Vec<_> = sm.iter_entries().collect();

        expected_entries.sort_by_key(|e| (e.row, e.col));
        entries.sort_by_key(|e| (e.row, e.col));

        assert_eq!(expected_entries, entries);
    }
}
