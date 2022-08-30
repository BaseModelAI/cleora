use crate::configuration::{Configuration, InitMethod};
use crate::persistence::embedding::EmbeddingPersistor;
use crate::persistence::entity::EntityMappingPersistor;
use crate::sparse_matrix::{Entry, SparseMatrix, SparseMatrixReader};
use log::{info, warn};
use memmap::MmapMut;
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use ndarray_linalg::lobpcg::{lobpcg, LobpcgResult};
use ndarray_linalg::TruncatedOrder;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs;
use std::fs::OpenOptions;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::sync::Arc;
use uuid::Uuid;

/// Number of broken entities (those with errors during writing to the file) which are logged.
/// There can be much more but log the first few.
const LOGGED_NUMBER_OF_BROKEN_ENTITIES: usize = 20;

/// Used during matrix initialization. No specific requirement (ca be lower as well).
const MAX_HASH_I64: i64 = 8 * 1024 * 1024;
const MAX_HASH_F32: f32 = MAX_HASH_I64 as f32;

/// Wrapper for different types of matrix structures such as 2-dim vectors or memory-mapped files
trait MatrixWrapper {
    /// Initializing a matrix with values from its dimensions and the hash values from the sparse matrix
    fn init_with_hashes<T: SparseMatrixReader + Sync + Send>(
        rows: usize,
        cols: usize,
        fixed_random_value: i64,
        sparse_matrix_reader: Arc<T>,
    ) -> Self;

    fn init_with_eigenvectors(
        rows: usize,
        cols: usize,
        sparse_matrix_reader: Arc<SparseMatrix>,
        order: TruncatedOrder,
    ) -> Self;

    /// Returns value for specific coordinates
    fn get_value(&self, row: usize, col: usize) -> f32;

    /// Normalizing a matrix by rows sum
    fn normalize(&mut self);

    /// Multiplies sparse matrix by the matrix
    fn multiply(sparse_matrix_reader: Arc<SparseMatrix>, other: Self) -> Self;
}

/// Two dimensional vectors as matrix representation
struct TwoDimVectorMatrix {
    rows: usize,
    cols: usize,
    matrix: Vec<Vec<f32>>,
}

impl MatrixWrapper for TwoDimVectorMatrix {
    fn init_with_hashes<T: SparseMatrixReader + Sync + Send>(
        rows: usize,
        cols: usize,
        fixed_random_value: i64,
        sparse_matrix_reader: Arc<T>,
    ) -> Self {
        let result: Vec<Vec<f32>> = (0..cols)
            .into_par_iter()
            .map(|i| {
                let mut col: Vec<f32> = Vec::with_capacity(rows);
                for hsh in sparse_matrix_reader.iter_hashes() {
                    let col_value = init_value(i, hsh.value, fixed_random_value);
                    col.push(col_value);
                }
                col
            })
            .collect();
        Self {
            rows,
            cols,
            matrix: result,
        }
    }

    fn init_with_eigenvectors(
        rows: usize,
        cols: usize,
        sparse_matrix_reader: Arc<SparseMatrix>,
        order: TruncatedOrder,
    ) -> Self {
        let matrix_transform = |x: ArrayView2<f32>| -> Array2<f32> {
            let shape = x.shape();
            let mut arr = Array2::zeros((shape[0], shape[1]));

            let mut sparse_laplacian_entries: Vec<Entry> = (&sparse_matrix_reader.entries)
                .into_par_iter()
                .map(|entry| Entry {
                    row: entry.row,
                    col: entry.col,
                    value: -entry.value,
                })
                .collect();

            for i in 0..rows {
                sparse_laplacian_entries.push(Entry {
                    row: i as u32,
                    col: i as u32,
                    value: sparse_matrix_reader.get_row_sum(i as u32),
                })
            }

            for entry in sparse_laplacian_entries {
                for i in 0..shape[0] {
                    arr[[entry.row as usize, i as usize]] +=
                        entry.value * x[[entry.col as usize, i as usize]];
                }
            }

            arr
        };

        let mut initial_guess: Array2<f32> = Array2::zeros((rows, cols));
        for i in 0..std::cmp::min(rows, cols) {
            initial_guess[[i, i]] = 1f32;
        }
        let empty_preconditioner = |_x: ArrayViewMut2<f32>| {};
        let constraints = Option::None;
        let tolerance = 0.1f32;
        let maxiter = 1000;

        match lobpcg(
            matrix_transform,
            initial_guess,
            empty_preconditioner,
            constraints,
            tolerance,
            maxiter,
            order,
        ) {
            LobpcgResult::Ok(_eigenvalues, eigenvectors, _norms) => {
                let mut matrix: Vec<Vec<f32>> = Vec::new();

                for i in 0..rows {
                    let mut row: Vec<f32> = Vec::new();

                    for j in 0..cols {
                        row.push(eigenvectors[[i, j]]);
                    }
                    matrix.push(row);
                }

                Self { rows, cols, matrix }
            }
            LobpcgResult::Err(_a, _b, _c, _err) => {
                panic!(
                    "Computing the eigenvectors of the Laplacian failed. {}",
                    _err
                )
            }
            LobpcgResult::NoResult(_err) => {
                panic!(
                    "Computing the eigenvectors of the Laplacian failed. {}",
                    _err
                )
            }
        }
    }

    #[inline]
    fn get_value(&self, row: usize, col: usize) -> f32 {
        let column: &Vec<f32> = self.matrix.get(col).unwrap();
        column[row]
    }

    fn normalize(&mut self) {
        let mut row_sum = vec![0f32; self.rows];

        for col in self.matrix.iter() {
            for (j, sum) in row_sum.iter_mut().enumerate() {
                let value = col[j];
                *sum += value.powi(2)
            }
        }

        let row_sum = Arc::new(row_sum);
        self.matrix.par_iter_mut().for_each(|col| {
            for (j, value) in col.iter_mut().enumerate() {
                let sum = row_sum[j];
                *value /= sum.sqrt();
            }
        });
    }

    fn multiply(sparse_matrix_reader: Arc<SparseMatrix>, other: Self) -> Self {
        let rnew = zero_2d(other.rows, other.cols);

        let result: Vec<Vec<f32>> = other
            .matrix
            .into_par_iter()
            .zip(rnew)
            .update(|data| {
                let (res_col, rnew_col) = data;
                for entry in &sparse_matrix_reader.entries {
                    let elem = rnew_col.get_mut(entry.row as usize).unwrap();
                    let value = res_col[entry.col as usize];
                    *elem += value * entry.value;
                }
            })
            .map(|data| data.1)
            .collect();

        Self {
            rows: other.rows,
            cols: other.cols,
            matrix: result,
        }
    }
}

fn init_value(col: usize, hsh: u64, fixed_random_value: i64) -> f32 {
    ((hash((hsh as i64) + (col as i64) + fixed_random_value) % MAX_HASH_I64) as f32) / MAX_HASH_F32
}

fn hash(num: i64) -> i64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_i64(num);
    hasher.finish() as i64
}

fn zero_2d(row: usize, col: usize) -> Vec<Vec<f32>> {
    let mut res: Vec<Vec<f32>> = Vec::with_capacity(col);
    for _i in 0..col {
        let col = vec![0f32; row];
        res.push(col);
    }
    res
}

/// Memory-mapped file as matrix representation. Every column of the matrix is placed side by side in the file.
struct MMapMatrix {
    rows: usize,
    cols: usize,
    file_name: String,
    matrix: MmapMut,
}

impl MatrixWrapper for MMapMatrix {
    fn init_with_hashes<T: SparseMatrixReader + Sync + Send>(
        rows: usize,
        cols: usize,
        fixed_random_value: i64,
        sparse_matrix_reader: Arc<T>,
    ) -> Self {
        let uuid = Uuid::new_v4();
        let file_name = format!("{}_matrix_{}", sparse_matrix_reader.get_id(), uuid);
        let mut mmap = create_mmap(rows, cols, file_name.as_str());

        mmap.par_chunks_mut(rows * 4)
            .enumerate()
            .for_each(|(i, chunk)| {
                // i - number of dimension
                // chunk - column/vector of bytes
                for (j, hsh) in sparse_matrix_reader.iter_hashes().enumerate() {
                    let col_value = init_value(i, hsh.value, fixed_random_value);
                    MMapMatrix::update_column(j, chunk, |value| unsafe { *value = col_value });
                }
            });

        mmap.flush()
            .expect("Can't flush memory map modifications to disk");

        Self {
            rows,
            cols,
            file_name,
            matrix: mmap,
        }
    }

    fn init_with_eigenvectors(
        rows: usize,
        cols: usize,
        sparse_matrix_reader: Arc<SparseMatrix>,
        order: TruncatedOrder,
    ) -> Self {
        let uuid = Uuid::new_v4();
        let file_name = format!("{}_matrix_{}", sparse_matrix_reader.get_id(), uuid);
        let mut mmap = create_mmap(rows, cols, file_name.as_str());

        let matrix_transform = |x: ArrayView2<f32>| -> Array2<f32> {
            let shape = x.shape();
            let mut arr = Array2::zeros((shape[0], shape[1]));

            let mut sparse_laplacian_entries: Vec<Entry> = (&sparse_matrix_reader.entries)
                .into_par_iter()
                .map(|entry| Entry {
                    row: entry.row,
                    col: entry.col,
                    value: -entry.value,
                })
                .collect();

            for i in 0..rows {
                sparse_laplacian_entries.push(Entry {
                    row: i as u32,
                    col: i as u32,
                    value: sparse_matrix_reader.get_row_sum(i as u32),
                })
            }

            for entry in sparse_laplacian_entries {
                for i in 0..shape[0] {
                    arr[[entry.row as usize, i as usize]] +=
                        entry.value * x[[entry.col as usize, i as usize]];
                }
            }

            arr
        };

        let mut initial_guess: Array2<f32> = Array2::zeros((rows, cols));
        for i in 0..std::cmp::min(rows, cols) {
            initial_guess[[i, i]] = 1f32;
        }
        let empty_preconditioner = |_x: ArrayViewMut2<f32>| {};
        let constraints = Option::None;
        let tolerance = 0.1f32;
        let maxiter = 1000;

        match lobpcg(
            matrix_transform,
            initial_guess,
            empty_preconditioner,
            constraints,
            tolerance,
            maxiter,
            order,
        ) {
            LobpcgResult::Ok(_eigenvalues, eigenvector, _norms) => {
                mmap.par_chunks_mut(rows * 4)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        for j in 0..cols {
                            let col_value = eigenvector[[i, j]];
                            MMapMatrix::update_column(j, chunk, |value| unsafe {
                                *value = col_value
                            });
                        }
                    });

                mmap.flush()
                    .expect("Can't flush memory map modifications to disk");

                Self {
                    rows,
                    cols,
                    file_name,
                    matrix: mmap,
                }
            }
            LobpcgResult::Err(_a, _b, _c, _err) => {
                panic!(
                    "Computing the eigenvectors of the Laplacian failed. {}",
                    _err
                )
            }
            LobpcgResult::NoResult(_err) => {
                panic!(
                    "Computing the eigenvectors of the Laplacian failed. {}",
                    _err
                )
            }
        }
    }

    #[inline]
    fn get_value(&self, row: usize, col: usize) -> f32 {
        let start_idx = ((col * self.rows) + row) * 4;
        let end_idx = start_idx + 4;
        let pointer: *const u8 = (&self.matrix[start_idx..end_idx]).as_ptr();
        unsafe {
            let value = pointer as *const f32;
            *value
        }
    }

    fn normalize(&mut self) {
        let entities_count = self.rows;
        let mut row_sum = vec![0f32; entities_count];

        for i in 0..(self.cols as usize) {
            for (j, sum) in row_sum.iter_mut().enumerate() {
                let value = self.get_value(j, i);
                *sum += value.powi(2)
            }
        }

        let row_sum = Arc::new(row_sum);
        self.matrix
            .par_chunks_mut(entities_count * 4)
            .enumerate()
            .for_each(|(_i, chunk)| {
                // i - number of dimension
                // chunk - column/vector of bytes
                for (j, &sum) in row_sum.iter().enumerate() {
                    MMapMatrix::update_column(j, chunk, |value| unsafe { *value /= sum.sqrt() });
                }
            });

        self.matrix
            .flush()
            .expect("Can't flush memory map modifications to disk");
    }

    fn multiply(sparse_matrix_reader: Arc<SparseMatrix>, other: Self) -> Self {
        let rows = other.rows;
        let cols = other.cols;

        let uuid = Uuid::new_v4();
        let file_name = format!("{}_matrix_{}", sparse_matrix_reader.get_id(), uuid);
        let mut mmap_output = create_mmap(rows, cols, file_name.as_str());

        let input = Arc::new(other);
        mmap_output
            .par_chunks_mut(rows * 4)
            .enumerate()
            .for_each_with(input, |input, (i, chunk)| {
                for entry in &sparse_matrix_reader.entries {
                    let input_value = input.get_value(entry.col as usize, i);
                    MMapMatrix::update_column(entry.row as usize, chunk, |value| unsafe {
                        *value += input_value * entry.value
                    });
                }
            });

        mmap_output
            .flush()
            .expect("Can't flush memory map modifications to disk");

        Self {
            rows,
            cols,
            file_name,
            matrix: mmap_output,
        }
    }
}

/// Creates memory-mapped file with allocated number of bytes
fn create_mmap(rows: usize, cols: usize, file_name: &str) -> MmapMut {
    let number_of_bytes = (rows * cols * 4) as u64;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(file_name)
        .expect("Can't create new set of options for memory mapped file");
    file.set_len(number_of_bytes).unwrap_or_else(|_| {
        panic!(
            "Can't update the size of {} file to {} bytes",
            file_name, number_of_bytes
        )
    });
    unsafe {
        MmapMut::map_mut(&file).unwrap_or_else(|_| {
            panic!(
                "Can't create memory mapped file for the underlying file {}",
                file_name
            )
        })
    }
}

/// Used to remove memory-mapped file after processing
impl Drop for MMapMatrix {
    fn drop(&mut self) {
        fs::remove_file(self.file_name.as_str()).unwrap_or_else(|_| {
            warn!(
                "File {} can't be removed after work. Remove the file in order to save disk space.",
                self.file_name.as_str()
            )
        });
    }
}

impl MMapMatrix {
    #[inline]
    fn update_column<F>(col: usize, chunk: &mut [u8], func: F)
    where
        F: Fn(*mut f32),
    {
        let start_idx = col * 4;
        let end_idx = start_idx + 4;
        let pointer: *mut u8 = (&mut chunk[start_idx..end_idx]).as_mut_ptr();
        let value = pointer as *mut f32;
        func(value);
    }
}

/// Calculate embeddings in memory.
pub fn calculate_embeddings<T2>(
    config: Arc<Configuration>,
    sparse_matrix_reader: Arc<SparseMatrix>,
    entity_mapping_persistor: Arc<T2>,
    embedding_persistor: &mut dyn EmbeddingPersistor,
) where
    T2: EntityMappingPersistor,
{
    let mult = MatrixMultiplicator::new(config.clone(), sparse_matrix_reader);
    let init: TwoDimVectorMatrix = mult.initialize();
    let res = mult.propagate(config.max_number_of_iteration, init);
    mult.persist(res, entity_mapping_persistor, embedding_persistor);

    info!("Finalizing embeddings calculations!")
}

/// Provides matrix multiplication based on sparse matrix data.
#[derive(Debug)]
struct MatrixMultiplicator<M: MatrixWrapper> {
    dimension: usize,
    number_of_entities: usize,
    fixed_random_value: i64,
    sparse_matrix_reader: Arc<SparseMatrix>,
    _marker: PhantomData<M>,
    init_method: InitMethod,
}

impl<M> MatrixMultiplicator<M>
where
    M: MatrixWrapper,
{
    fn new(config: Arc<Configuration>, sparse_matrix_reader: Arc<SparseMatrix>) -> Self {
        let rand_value = config.seed.map(hash).unwrap_or(0);
        Self {
            dimension: config.embeddings_dimension as usize,
            number_of_entities: sparse_matrix_reader.get_number_of_entities() as usize,
            fixed_random_value: rand_value,
            sparse_matrix_reader,
            _marker: PhantomData,
            init_method: config.init_method,
        }
    }

    /// Initialize a matrix
    fn initialize(&self) -> M {
        info!(
            "Start initialization. Dims: {}, entities: {}.",
            self.dimension, self.number_of_entities
        );

        let result = match self.init_method {
            InitMethod::Random => M::init_with_hashes(
                self.number_of_entities,
                self.dimension,
                self.fixed_random_value,
                self.sparse_matrix_reader.clone(),
            ),
            InitMethod::EigenvectorsSmallest | InitMethod::EigenvectorsLargest => {
                M::init_with_eigenvectors(
                    self.number_of_entities,
                    self.dimension,
                    self.sparse_matrix_reader.clone(),
                    match self.init_method {
                        InitMethod::EigenvectorsSmallest => TruncatedOrder::Smallest,
                        InitMethod::EigenvectorsLargest => TruncatedOrder::Largest,
                        InitMethod::Random => panic!("Issue in determining initialisation type."),
                    },
                )
            }
        };

        info!(
            "Done initializing. Dims: {}, entities: {}.",
            self.dimension, self.number_of_entities
        );
        result
    }

    /// The sparse matrix is multiplied by a freshly initialized matrix M.
    /// Multiplication is done against each column of matrix M in a separate thread.
    /// The obtained columns of the new matrix are subsequently merged into the full matrix.
    /// The matrix is L2-normalized, again in a multithreaded fashion across matrix columns.
    /// Finally, depending on the target iteration number, the matrix is either returned
    /// or fed for next iterations of multiplication against the sparse matrix.
    fn propagate(&self, max_iter: u8, res: M) -> M {
        info!("Start propagating. Number of iterations: {}.", max_iter);

        let mut new_res = res;
        for i in 0..max_iter {
            let mut next = M::multiply(self.sparse_matrix_reader.clone(), new_res);
            next.normalize();
            new_res = next;

            info!(
                "Done iter: {}. Dims: {}, entities: {}, num data points: {}.",
                i,
                self.dimension,
                self.number_of_entities,
                self.sparse_matrix_reader.get_number_of_entries()
            );
        }

        info!("Done propagating.");
        new_res
    }

    /// Saves results to output such as textfile, numpy etc
    fn persist<T1>(
        &self,
        res: M,
        entity_mapping_persistor: Arc<T1>,
        embedding_persistor: &mut dyn EmbeddingPersistor,
    ) where
        T1: EntityMappingPersistor,
    {
        info!("Start saving embeddings.");

        embedding_persistor
            .put_metadata(self.number_of_entities as u32, self.dimension as u16)
            .unwrap_or_else(|_| {
                // if can't write first data to the file, probably further is the same
                panic!(
                    "Can't write metadata. Entities: {}. Dimension: {}.",
                    self.number_of_entities, self.dimension
                )
            });

        // entities which can't be written to the file (error occurs)
        let mut broken_entities = HashSet::new();
        for (i, hash) in self.sparse_matrix_reader.iter_hashes().enumerate() {
            let entity_name_opt = entity_mapping_persistor.get_entity(hash.value);
            if let Some(entity_name) = entity_name_opt {
                let mut embedding: Vec<f32> = Vec::with_capacity(self.dimension);
                for j in 0..self.dimension {
                    let value = res.get_value(i, j);
                    embedding.push(value);
                }
                embedding_persistor
                    .put_data(&entity_name, hash.occurrence, embedding)
                    .unwrap_or_else(|_| {
                        broken_entities.insert(entity_name);
                    });
            };
        }

        if !broken_entities.is_empty() {
            log_broken_entities(broken_entities);
        }

        embedding_persistor
            .finish()
            .unwrap_or_else(|_| warn!("Can't finish writing to the file."));

        info!("Done saving embeddings.");
    }
}

fn log_broken_entities(broken_entities: HashSet<String>) {
    let num_of_broken_entities = broken_entities.len();
    let few_broken_entities: HashSet<_> = broken_entities
        .into_iter()
        .take(LOGGED_NUMBER_OF_BROKEN_ENTITIES)
        .collect();
    warn!(
        "Number of entities which can't be written to the file: {}. First {} broken entities: {:?}.",
        num_of_broken_entities, LOGGED_NUMBER_OF_BROKEN_ENTITIES, few_broken_entities
    );
}

/// Calculate embeddings with memory-mapped files.
pub fn calculate_embeddings_mmap<T2>(
    config: Arc<Configuration>,
    sparse_matrix_reader: Arc<SparseMatrix>,
    entity_mapping_persistor: Arc<T2>,
    embedding_persistor: &mut dyn EmbeddingPersistor,
) where
    T2: EntityMappingPersistor,
{
    let mult = MatrixMultiplicator::new(config.clone(), sparse_matrix_reader);
    let init: MMapMatrix = mult.initialize();
    let res = mult.propagate(config.max_number_of_iteration, init);
    mult.persist(res, entity_mapping_persistor, embedding_persistor);

    info!("Finalizing embeddings calculations!")
}