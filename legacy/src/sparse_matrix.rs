use crate::configuration::Column;
use log::info;
use rustc_hash::FxHashMap;
use std::collections::hash_map;
use std::mem;

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
pub fn create_sparse_matrices(cols: &[Column]) -> Vec<SparseMatrix> {
    let mut sparse_matrices: Vec<SparseMatrix> = Vec::new();
    let num_fields = cols.len();
    let mut reflexive_count = 0;

    for i in 0..num_fields {
        for j in i..num_fields {
            let col_i = &cols[i];
            let col_j = &cols[j];
            if i < j && !(col_i.transient && col_j.transient) {
                let sm =
                    SparseMatrix::new(i as u8, col_i.name.clone(), j as u8, col_j.name.clone());
                sparse_matrices.push(sm);
            } else if i == j && col_i.reflexive {
                let new_j = num_fields + reflexive_count;
                reflexive_count += 1;
                let sm =
                    SparseMatrix::new(i as u8, col_i.name.clone(), new_j as u8, col_j.name.clone());
                sparse_matrices.push(sm);
            }
        }
    }
    sparse_matrices
}

/// Represents graph based on incoming data.
/// It follows the sparse matrix coordinate format (COO). Its purpose is to save space by holding only
/// the coordinates and values of nonzero entities.
#[derive(Debug)]
pub struct SparseMatrix {
    /// First column index for which we creates subgraph
    pub col_a_id: u8,

    /// First column name
    pub col_a_name: String,

    /// Second column index for which we creates subgraph
    pub col_b_id: u8,

    /// Second column name
    pub col_b_name: String,

    /// Counts every occurrence of entity relationships from first and second column
    edge_count: u32,

    /// Maps entity hash to the id in such a way that each new hash gets another id (id + 1)
    hash_2_id: FxHashMap<u64, u32>,

    /// Maps id to hash value and occurrence
    id_2_hash: Vec<Hash>,

    /// Holds the sum of the values for each row
    row_sum: Vec<f32>,

    /// Maps a unique value (as combination of two numbers) to `entries` index
    pair_index: FxHashMap<u64, u32>,

    /// Coordinates and values of nonzero entities
    entries: Vec<Entry>,
}

/// Hash data
#[derive(Debug, Clone, Copy)]
pub struct Hash {
    /// Value of the hash
    pub value: u64,

    /// Number of hash occurrences
    pub occurrence: u32,
}

impl Hash {
    fn new(value: u64) -> Self {
        Self {
            value,
            occurrence: 1,
        }
    }
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

impl SparseMatrix {
    pub fn new(col_a_id: u8, col_a_name: String, col_b_id: u8, col_b_name: String) -> Self {
        Self {
            col_a_id,
            col_a_name,
            col_b_id,
            col_b_name,
            edge_count: 0,
            hash_2_id: FxHashMap::default(),
            id_2_hash: Vec::new(),
            row_sum: Vec::new(),
            pair_index: FxHashMap::default(),
            entries: Vec::new(),
        }
    }

    /// Handles hashes for one combination of incoming data. Let's say that input row looks like:
    /// userId1   | productId1, productId2  | brandId1, brandId2
    /// Note! To simplify explanation there is no any reflexive column so the result is:
    /// (userId1, productId1, brandId1),
    /// (userId1, productId1, brandId2),
    /// (userId1, productId2, brandId1),
    /// (userId1, productId2, brandId2)
    /// These cartesian products are provided as array of hashes. Sparse matrix has indices
    /// `col_a_id` and `col_b_id` (to corresponding columns) in order to read interesting hashes
    /// from provided slice. For one input row we actually call this function 4 times.
    pub fn handle_pair(&mut self, hashes: &[u64]) {
        let a = self.col_a_id;
        let b = self.col_b_id;
        self.add_pair_symmetric(
            hashes[(a + 1) as usize],
            hashes[(b + 1) as usize],
            hashes[0],
        );
    }

    /// It creates sparse matrix for two columns in the incoming data.
    /// Let's say that we have such columns:
    /// customers | products                | brands
    /// incoming data:
    /// userId1   | productId1, productId2  | brandId1, brandId2
    /// userId2   | productId1              | brandId3, brandId4, brandId5
    /// etc.
    /// One of the sparse matrices could represent customers and products relation (products and brands relation, customers and brands relation).
    /// This sparse matrix (customers and products relation) handles every combination in these columns according to
    /// total combinations in a row.
    /// The first row in the incoming data produces two combinations according to 4 total combinations:
    /// userId1, productId1 and userId1, productId2
    /// The second row produces one combination userId2, productId1 according to 3 total combinations.
    /// `a_hash` - hash of a entity for a column A
    /// `b_hash` - hash of a entity for a column B
    /// `count` - total number of combinations in a row
    fn add_pair_symmetric(&mut self, a_hash: u64, b_hash: u64, count: u64) {
        let a = self.update_hash_and_get_id(a_hash);
        let b = self.update_hash_and_get_id(b_hash);

        let value = 1f32 / (count as f32);

        self.edge_count += 1;

        self.add_or_update_entry(a, b, value);
        self.add_or_update_entry(b, a, value);

        self.update_row_sum(a, value);
        self.update_row_sum(b, value);
    }

    fn update_hash_and_get_id(&mut self, hash: u64) -> u32 {
        match self.hash_2_id.entry(hash) {
            hash_map::Entry::Vacant(entry) => {
                let id = self.id_2_hash.len() as u32;
                entry.insert(id);
                self.id_2_hash.push(Hash::new(hash));
                id
            }
            hash_map::Entry::Occupied(entry) => {
                let id = *entry.get();
                self.id_2_hash[id as usize].occurrence += 1;
                id
            }
        }
    }

    fn add_or_update_entry(&mut self, x: u32, y: u32, val: f32) {
        let magic = Self::magic_pair(x, y);
        let num_of_entries = self.entries.len() as u32;
        let position = *self.pair_index.entry(magic).or_insert(num_of_entries);

        if position < num_of_entries {
            self.entries[position as usize].value += val;
        } else {
            let entry = Entry {
                row: x,
                col: y,
                value: val,
            };
            self.entries.push(entry);
        }
    }

    /// Combining two numbers into a unique one: pairing functions.
    /// It uses "elegant pairing" (https://odino.org/combining-two-numbers-into-a-unique-one-pairing-functions/).
    fn magic_pair(a: u32, b: u32) -> u64 {
        let x = u64::from(a);
        let y = u64::from(b);
        if x >= y {
            x * x + x + y
        } else {
            y * y + x
        }
    }

    fn update_row_sum(&mut self, id: u32, val: f32) {
        let id = id as usize;
        if id < self.row_sum.len() {
            self.row_sum[id] += val;
        } else {
            self.row_sum.push(val);
        };
    }

    /// Normalization and other tasks after sparse matrix construction.
    pub fn finish(&mut self) {
        self.normalize();

        info!("Number of entities: {}", self.get_number_of_entities());
        info!("Number of edges: {}", self.edge_count);
        info!("Number of entries: {}", self.get_number_of_entries());

        let hash_2_id_mem_size = self.hash_2_id.capacity() * 12;
        let hash_mem_size = mem::size_of::<Hash>();
        let id_2_hash_mem_size = self.id_2_hash.capacity() * hash_mem_size;
        let row_sum_mem_size = self.row_sum.capacity() * 4;
        let pair_index_mem_size = self.pair_index.capacity() * 12;

        let entry_mem_size = mem::size_of::<Entry>();
        let entries_mem_size = self.entries.capacity() * entry_mem_size;

        let total_mem_size = hash_2_id_mem_size
            + id_2_hash_mem_size
            + row_sum_mem_size
            + pair_index_mem_size
            + entries_mem_size;

        info!(
            "Total memory usage by the struct ~ {} MB",
            (total_mem_size / 1048576)
        );
    }

    /// Normalize entries by dividing every entry value by row sum
    fn normalize(&mut self) {
        for entry in self.entries.iter_mut() {
            entry.value /= self.row_sum[entry.row as usize];
        }
    }
}

impl SparseMatrixReader for SparseMatrix {
    fn get_id(&self) -> String {
        format!("{}_{}", self.col_a_id, self.col_b_id)
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
    use crate::sparse_matrix::{create_sparse_matrices, Entry, SparseMatrix, SparseMatrixReader};
    use rustc_hash::FxHasher;
    use std::collections::{HashMap, HashSet};
    use std::hash::Hasher;

    fn map_to_ids_and_names(sparse_matrices: &[SparseMatrix]) -> HashSet<(u8, &str, u8, &str)> {
        sparse_matrices
            .iter()
            .map(|sm| {
                (
                    sm.col_a_id,
                    sm.col_a_name.as_str(),
                    sm.col_b_id,
                    sm.col_b_name.as_str(),
                )
            })
            .collect()
    }

    fn prepare_entries(hash_2_id: HashMap<u64, u32>, edges: Vec<(&str, &str, f32)>) -> Vec<Entry> {
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
            let entry_col_row = Entry {
                row: col,
                col: row,
                value: val,
            };
            entries.push(entry_row_col);
            entries.push(entry_col_row);
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
        let sparse_matrices = create_sparse_matrices(&[]);
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
        let sparse_matrices = create_sparse_matrices(&columns);
        assert_eq!(true, sparse_matrices.is_empty());

        columns.push(Column {
            name: String::from("c"),
            complex: true,
            ..Default::default()
        });
        let sparse_matrices = create_sparse_matrices(&columns);
        let sparse_matrices: HashSet<_> = map_to_ids_and_names(&sparse_matrices);
        let expected_sparse_matrices: HashSet<_> = [(0, "a", 2, "c"), (1, "b", 2, "c")]
            .iter()
            .cloned()
            .collect();
        assert_eq!(expected_sparse_matrices, sparse_matrices)
    }

    #[test]
    fn create_sparse_matrices_if_reflexive_columns_provided() {
        let sparse_matrices = create_sparse_matrices(&[
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
        let mut sm = SparseMatrix::new(0u8, String::from("col_0"), 1u8, String::from("col_1"));

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
        let expected_entries = prepare_entries(hash_2_id, edges);
        let entries: Vec<_> = sm.iter_entries().collect();
        assert_eq!(expected_entries, entries);
    }
}
