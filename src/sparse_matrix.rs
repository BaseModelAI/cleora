use crate::configuration::Column;
use crate::persistence::sparse_matrix::{
    Entry, InMemorySparseMatrixPersistor, SparseMatrixPersistor,
};

/// Creates combinations of column pairs as sparse matrices.
/// Let's say that we have such columns configuration: complex::a reflexive::complex::b c. This is provided
/// as Array[Column] after parsing the config.
/// The allowed column modifiers are:
/// - transient - the field is virtual - it is considered during embedding process, no output file is written for the field,
/// - complex   - the field is composite, containing multiple entity identifiers separated by space,
/// - reflexive - the field is reflexive, which means that it interacts with itself, additional output file is written for every such field.
/// We create sparse matrix for every columns relations (based on column modifiers).
/// For our example we have:
/// - sparse matrix for column a and b,
/// - sparse matrix for column a and c,
/// - sparse matrix for column b and c,
/// - sparse matrix for column b and b (reflexive column).
/// Apart from column names in sparse matrix we provide indices for incoming data. We have 3 columns such as a, b and c
/// but column b is reflexive so we need to include this column. The result is: Array(a, b, c, b).
/// The rule is that every reflexive column is append with the order of occurrence to the end of constructed array.
/// `dimension` - dimension for output embedding vector size
/// `cols` columns configuration with transient, reflexive, complex marks
/// return sparse matrices for columns configuration.
pub fn create_sparse_matrices(
    dimension: u16,
    cols: &[Column],
) -> Vec<SparseMatrix<InMemorySparseMatrixPersistor>> {
    let mut sparse_matrices: Vec<SparseMatrix<InMemorySparseMatrixPersistor>> = Vec::new();
    let num_fields = cols.len();
    let mut reflexive_count = 0;

    for i in 0..num_fields {
        for j in i..num_fields {
            let col_i = &cols[i];
            let col_j = &cols[j];
            if i < j && !(col_i.transient && col_j.transient) {
                let sm = SparseMatrix {
                    col_a_id: i as u8,
                    col_a_name: col_i.name.clone(),
                    col_b_id: j as u8,
                    col_b_name: col_j.name.clone(),
                    dimension,
                    sparse_matrix_persistor: InMemorySparseMatrixPersistor::default(),
                };
                sparse_matrices.push(sm);
            } else if i == j && col_i.reflexive {
                let new_j = num_fields + reflexive_count;
                reflexive_count += 1;
                let sm = SparseMatrix {
                    col_a_id: i as u8,
                    col_a_name: col_i.name.clone(),
                    col_b_id: new_j as u8,
                    col_b_name: col_j.name.clone(),
                    dimension,
                    sparse_matrix_persistor: InMemorySparseMatrixPersistor::default(),
                };
                sparse_matrices.push(sm);
            }
        }
    }
    sparse_matrices
}

#[derive(Debug)]
pub struct SparseMatrix<T: SparseMatrixPersistor + Sync> {
    pub col_a_id: u8,
    pub col_a_name: String,
    pub col_b_id: u8,
    pub col_b_name: String,
    pub dimension: u16,
    pub sparse_matrix_persistor: T,
}

impl<T> SparseMatrix<T>
where
    T: SparseMatrixPersistor + Sync,
{
    /// Handles hashes for one combination of incoming data. Let's say that input row looks like:
    /// userId1   | productId1, productId2  | brandId1, brandId2
    /// Note! To simplify explanation there is no any reflexive column so the result is:
    /// (userId1, productId1, brandId1),
    /// (userId1, productId1, brandId2),
    /// (userId1, productId2, brandId1),
    /// (userId1, productId2, brandId2)
    /// These cartesian products are provided as array of hashes (long values). Sparse matrix has indices (to corresponding columns)
    /// in order to read interesting hashes from provided array.
    /// For one input row we actually call this function 4 times.
    /// `hashes` - it contains array of hashes
    pub fn handle_pair(&mut self, hashes: &[u64]) {
        let a = self.col_a_id;
        let b = self.col_b_id;
        self.add_pair_symmetric(
            hashes[(a + 1) as usize],
            hashes[(b + 1) as usize],
            hashes[0],
        );
    }

    pub fn finish(&mut self) {
        self.sparse_matrix_persistor.finish();
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
        let a = self.get_or_create_id(a_hash);
        let b = self.get_or_create_id(b_hash);
        let value = 1f32 / (count as f32);

        self.sparse_matrix_persistor.increment_edge_counter();
        self.sparse_matrix_persistor
            .increment_hash_occurrence(a_hash);
        self.sparse_matrix_persistor
            .increment_hash_occurrence(b_hash);

        self.add_or_update_entry(a, b, value);
        self.add_or_update_entry(b, a, value);

        self.sparse_matrix_persistor.update_row_sum(a, value);
        self.sparse_matrix_persistor.update_row_sum(b, value);
    }

    fn get_or_create_id(&mut self, hash: u64) -> u32 {
        let res = self.sparse_matrix_persistor.get_id(hash);
        if res == -1 {
            let entity_count = self.sparse_matrix_persistor.get_entity_counter();
            self.sparse_matrix_persistor.add_hash_id(hash, entity_count);
            self.sparse_matrix_persistor
                .update_entity_counter(entity_count + 1);
            entity_count
        } else {
            res as u32
        }
    }

    fn add_or_update_entry(&mut self, x: u32, y: u32, val: f32) {
        let amount_of_data = self.sparse_matrix_persistor.get_amount_of_data();
        let position = self.get_or_put_position(x, y, amount_of_data);

        let entry = Entry {
            row: x,
            col: y,
            value: val,
        };
        if position != amount_of_data {
            self.sparse_matrix_persistor.update_entry(position, entry);
        } else {
            self.sparse_matrix_persistor.add_new_entry(position, entry);
        }
    }

    fn get_or_put_position(&mut self, a: u32, b: u32, tentative: u32) -> u32 {
        let magic = Self::magic_pair(a, b);
        let pos = self.sparse_matrix_persistor.get_pair_index(magic);
        if pos == -1 {
            self.sparse_matrix_persistor
                .add_pair_index(magic, tentative);
            tentative
        } else {
            pos as u32
        }
    }

    fn magic_pair(a: u32, b: u32) -> u64 {
        let x = u64::from(a);
        let y = u64::from(b);
        if x >= y {
            x * x + x + y
        } else {
            y * y + x
        }
    }

    pub fn normalize(&mut self) {
        let amount_of_data = self.sparse_matrix_persistor.get_amount_of_data();
        for i in 0..amount_of_data {
            let entry = self.sparse_matrix_persistor.get_entry(i);
            let new_value = entry.value / self.sparse_matrix_persistor.get_row_sum(entry.row);
            self.sparse_matrix_persistor.replace_entry(
                i,
                Entry {
                    row: entry.row,
                    col: entry.col,
                    value: new_value,
                },
            );
        }
    }

    pub fn get_id(&self) -> String {
        format!("{}_{}", self.col_a_id, self.col_b_id)
    }
}
