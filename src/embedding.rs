use crate::configuration::Configuration;
use crate::persistence::embedding::EmbeddingPersistor;
use crate::persistence::entity::EntityMappingPersistor;
use crate::sparse_matrix::SparseMatrixReader;
use log::{info, warn};
use memmap::MmapMut;
use rayon::prelude::*;
use rustc_hash::FxHasher;
use std::collections::HashSet;
use std::fs;
use std::fs::OpenOptions;
use std::hash::Hasher;
use std::sync::Arc;

/// Number of broken entities (those with errors during writing to the file) which are logged.
/// There can be much more but log the first few.
const LOGGED_NUMBER_OF_BROKEN_ENTITIES: usize = 20;

/// Calculate embeddings in memory.
pub fn calculate_embeddings<T1, T2>(
    config: Arc<Configuration>,
    sparse_matrix_reader: Arc<T1>,
    entity_mapping_persistor: Arc<T2>,
    embedding_persistor: &mut dyn EmbeddingPersistor,
) where
    T1: SparseMatrixReader + Sync + Send,
    T2: EntityMappingPersistor,
{
    let mult = MatrixMultiplicator {
        dimension: config.embeddings_dimension,
        sparse_matrix_reader,
    };
    let init = mult.initialize();
    let res = mult.propagate(config.max_number_of_iteration, init);
    mult.persist(res, entity_mapping_persistor, embedding_persistor);

    info!("Finalizing embeddings calculations!")
}

/// Provides matrix multiplication based on sparse matrix data.
#[derive(Debug)]
pub struct MatrixMultiplicator<T: SparseMatrixReader + Sync + Send> {
    pub dimension: u16,
    pub sparse_matrix_reader: Arc<T>,
}

impl<T> MatrixMultiplicator<T>
where
    T: SparseMatrixReader + Sync + Send,
{
    fn initialize(&self) -> Vec<Vec<f32>> {
        let entities_count = self.sparse_matrix_reader.get_number_of_entities();

        info!(
            "Start initialization. Dims: {}, entities: {}.",
            self.dimension, entities_count
        );

        // no specific requirement (ca be lower as well)
        let max_hash = 8 * 1024 * 1024;
        let max_hash_float = max_hash as f32;

        let result: Vec<Vec<f32>> = (0..self.dimension)
            .into_par_iter()
            .map(|i| {
                let mut col: Vec<f32> = Vec::with_capacity(entities_count as usize);
                for (j, hsh) in self.sparse_matrix_reader.iter_hashes().enumerate() {
                    let col_value = ((hash((hsh.value as i64) + (i as i64)) % max_hash) as f32)
                        / max_hash_float;
                    col.insert(j as usize, col_value);
                }
                col
            })
            .collect();

        info!(
            "Done initializing. Dims: {}, entities: {}.",
            self.dimension, entities_count
        );
        result
    }

    fn propagate(&self, max_iter: u8, res: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        info!("Start propagating. Number of iterations: {}.", max_iter);

        let entities_count = self.sparse_matrix_reader.get_number_of_entities();
        let mut new_res = res;
        for i in 0..max_iter {
            let next = self.next_power(new_res);
            new_res = self.normalize(next);
            info!(
                "Done iter: {}. Dims: {}, entities: {}, num data points: {}.",
                i,
                self.dimension,
                entities_count,
                self.sparse_matrix_reader.get_number_of_entries()
            );
        }
        info!("Done propagating.");
        new_res
    }

    fn next_power(&self, res: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let entities_count = self.sparse_matrix_reader.get_number_of_entities() as usize;
        let rnew = Self::zero_2d(entities_count, self.dimension as usize);

        let result: Vec<Vec<f32>> = res
            .into_par_iter()
            .zip(rnew)
            .update(|data| {
                let (res_col, rnew_col) = data;
                for entry in self.sparse_matrix_reader.iter_entries() {
                    let elem = rnew_col.get_mut(entry.row as usize).unwrap();
                    let value = res_col.get(entry.col as usize).unwrap();
                    *elem += *value * entry.value
                }
            })
            .map(|data| data.1)
            .collect();

        result
    }

    fn zero_2d(row: usize, col: usize) -> Vec<Vec<f32>> {
        let mut res: Vec<Vec<f32>> = Vec::with_capacity(col);
        for i in 0..col {
            let col = vec![0f32; row];
            res.insert(i, col);
        }
        res
    }

    fn normalize(&self, res: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let entities_count = self.sparse_matrix_reader.get_number_of_entities() as usize;
        let mut row_sum = vec![0f32; entities_count];

        for i in 0..(self.dimension as usize) {
            for j in 0..entities_count {
                let sum = row_sum.get_mut(j).unwrap();
                let col: &Vec<f32> = res.get(i).unwrap();
                let value = col.get(j).unwrap();
                *sum += value.powi(2)
            }
        }

        let row_sum = Arc::new(row_sum);
        let result: Vec<Vec<f32>> = res
            .into_par_iter()
            .update(|col| {
                for j in 0..entities_count {
                    let value = col.get_mut(j).unwrap();
                    let sum = row_sum.get(j).unwrap();
                    *value /= sum.sqrt();
                }
            })
            .collect();

        result
    }

    fn persist<T1>(
        &self,
        res: Vec<Vec<f32>>,
        entity_mapping_persistor: Arc<T1>,
        embedding_persistor: &mut dyn EmbeddingPersistor,
    ) where
        T1: EntityMappingPersistor,
    {
        info!("Start saving embeddings.");

        let entities_count = self.sparse_matrix_reader.get_number_of_entities();
        embedding_persistor
            .put_metadata(entities_count, self.dimension)
            .unwrap_or_else(|_| {
                // if can't write first data to the file, probably further is the same
                panic!(
                    "Can't write metadata. Entities: {}. Dimension: {}.",
                    entities_count, self.dimension
                )
            });

        // entities which can't be written to the file (error occurs)
        let mut broken_entities = HashSet::new();
        for (i, hash) in self.sparse_matrix_reader.iter_hashes().enumerate() {
            let entity_name_opt = entity_mapping_persistor.get_entity(hash.value);
            if let Some(entity_name) = entity_name_opt {
                let mut embedding: Vec<f32> = Vec::with_capacity(self.dimension as usize);
                for j in 0..(self.dimension as usize) {
                    let col: &Vec<f32> = res.get(j).unwrap();
                    let value = col.get(i as usize).unwrap();
                    embedding.insert(j, *value);
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

fn hash(num: i64) -> i64 {
    let mut hasher = FxHasher::default();
    hasher.write_i64(num);
    hasher.finish() as i64
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
pub fn calculate_embeddings_mmap<T1, T2>(
    config: Arc<Configuration>,
    sparse_matrix_reader: Arc<T1>,
    entity_mapping_persistor: Arc<T2>,
    embedding_persistor: &mut dyn EmbeddingPersistor,
) where
    T1: SparseMatrixReader + Sync + Send,
    T2: EntityMappingPersistor,
{
    let matrix_id = sparse_matrix_reader.get_id();

    let mult = MatrixMultiplicatorMMap {
        dimension: config.embeddings_dimension,
        sparse_matrix_reader,
    };
    let init = mult.initialize();
    let res = mult.propagate(config.max_number_of_iteration, init);
    mult.persist(res, entity_mapping_persistor, embedding_persistor);

    let work_file = format!("{}_matrix_{}", matrix_id, config.max_number_of_iteration);
    fs::remove_file(&work_file).unwrap_or_else(|_| {
        warn!(
            "File {} can't be removed after work. Remove the file in order to save disk space.",
            work_file
        )
    });

    info!("Finalizing embeddings calculations!")
}

/// Provides matrix multiplication based on sparse matrix data.
#[derive(Debug)]
pub struct MatrixMultiplicatorMMap<T: SparseMatrixReader + Sync + Send> {
    pub dimension: u16,
    pub sparse_matrix_reader: Arc<T>,
}

impl<T> MatrixMultiplicatorMMap<T>
where
    T: SparseMatrixReader + Sync + Send,
{
    fn initialize(&self) -> MmapMut {
        let entities_count = self.sparse_matrix_reader.get_number_of_entities();

        info!(
            "Start initialization. Dims: {}, entities: {}.",
            self.dimension, entities_count
        );

        let number_of_bytes = entities_count as u64 * self.dimension as u64 * 4;
        let file_name = format!("{}_matrix_0", self.sparse_matrix_reader.get_id());
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&file_name)
            .expect("Can't create new set of options for memory mapped file");
        file.set_len(number_of_bytes).unwrap_or_else(|_| {
            panic!(
                "Can't update the size of {} file to {} bytes",
                &file_name, number_of_bytes
            )
        });
        let mut mmap = unsafe {
            MmapMut::map_mut(&file).unwrap_or_else(|_| {
                panic!(
                    "Can't create memory mapped file for the underlying file {}",
                    file_name
                )
            })
        };

        // no specific requirement (ca be lower as well)
        let max_hash = 8 * 1024 * 1024;
        let max_hash_float = max_hash as f32;

        mmap.par_chunks_mut((entities_count * 4) as usize)
            .enumerate()
            .for_each(|(i, chunk)| {
                // i - number of dimension
                // chunk - column/vector of bytes
                for (j, hsh) in self.sparse_matrix_reader.iter_hashes().enumerate() {
                    let col_value = ((hash((hsh.value as i64) + (i as i64)) % max_hash) as f32)
                        / max_hash_float;

                    let start_idx = j * 4;
                    let end_idx = start_idx + 4;
                    let pointer: *mut u8 = (&mut chunk[start_idx..end_idx]).as_mut_ptr();
                    unsafe {
                        let value = pointer as *mut f32;
                        *value = col_value;
                    };
                }
            });

        info!(
            "Done initializing. Dims: {}, entities: {}.",
            self.dimension, entities_count
        );

        mmap.flush()
            .expect("Can't flush memory map modifications to disk");
        mmap
    }

    fn propagate(&self, max_iter: u8, res: MmapMut) -> MmapMut {
        info!("Start propagating. Number of iterations: {}.", max_iter);

        let entities_count = self.sparse_matrix_reader.get_number_of_entities();
        let mut new_res = res;
        for i in 0..max_iter {
            let next = self.next_power(i, new_res);
            new_res = self.normalize(next);

            let work_file = format!("{}_matrix_{}", self.sparse_matrix_reader.get_id(), i);
            fs::remove_file(&work_file).unwrap_or_else(|_| {
                warn!("File {} can't be removed after work. Remove the file in order to save disk space.", work_file)
            });

            info!(
                "Done iter: {}. Dims: {}, entities: {}, num data points: {}.",
                i,
                self.dimension,
                entities_count,
                self.sparse_matrix_reader.get_number_of_entries()
            );
        }
        info!("Done propagating.");
        new_res
    }

    fn next_power(&self, iteration: u8, res: MmapMut) -> MmapMut {
        let entities_count = self.sparse_matrix_reader.get_number_of_entities() as usize;

        let number_of_bytes = entities_count as u64 * self.dimension as u64 * 4;
        let file_name = format!(
            "{}_matrix_{}",
            self.sparse_matrix_reader.get_id(),
            iteration + 1
        );
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&file_name)
            .expect("Can't create new set of options for memory mapped file");
        file.set_len(number_of_bytes).unwrap_or_else(|_| {
            panic!(
                "Can't update the size of {} file to {} bytes",
                &file_name, number_of_bytes
            )
        });
        let mut mmap_output = unsafe {
            MmapMut::map_mut(&file).unwrap_or_else(|_| {
                panic!(
                    "Can't create memory mapped file for the underlying file {}",
                    file_name
                )
            })
        };

        let input = Arc::new(res);
        mmap_output
            .par_chunks_mut(entities_count * 4)
            .enumerate()
            .for_each_with(input, |input, (i, chunk)| {
                for entry in self.sparse_matrix_reader.iter_entries() {
                    let start_idx_input = ((i * entities_count) + entry.col as usize) * 4;
                    let end_idx_input = start_idx_input + 4;
                    let pointer: *const u8 = (&input[start_idx_input..end_idx_input]).as_ptr();
                    let input_value = unsafe {
                        let value = pointer as *const f32;
                        *value
                    };

                    let start_idx_output = entry.row as usize * 4;
                    let end_idx_output = start_idx_output + 4;
                    let pointer: *mut u8 =
                        (&mut chunk[start_idx_output..end_idx_output]).as_mut_ptr();
                    unsafe {
                        let value = pointer as *mut f32;
                        *value += input_value * entry.value;
                    };
                }
            });

        mmap_output
            .flush()
            .expect("Can't flush memory map modifications to disk");
        mmap_output
    }

    fn normalize(&self, mut res: MmapMut) -> MmapMut {
        let entities_count = self.sparse_matrix_reader.get_number_of_entities() as usize;
        let mut row_sum = vec![0f32; entities_count];

        for i in 0..(self.dimension as usize) {
            for j in 0..entities_count {
                let sum = row_sum.get_mut(j).unwrap();

                let start_idx = ((i * entities_count) + j) * 4;
                let end_idx = start_idx + 4;
                let pointer: *const u8 = (&res[start_idx..end_idx]).as_ptr();
                let value = unsafe {
                    let value = pointer as *const f32;
                    *value
                };

                *sum += value.powi(2)
            }
        }

        let row_sum = Arc::new(row_sum);
        res.par_chunks_mut(entities_count * 4)
            .enumerate()
            .for_each(|(_i, chunk)| {
                // i - number of dimension
                // chunk - column/vector of bytes
                for j in 0..entities_count {
                    let sum = *row_sum.get(j).unwrap();

                    let start_idx = j * 4;
                    let end_idx = start_idx + 4;
                    let pointer: *mut u8 = (&mut chunk[start_idx..end_idx]).as_mut_ptr();
                    unsafe {
                        let value = pointer as *mut f32;
                        *value /= sum.sqrt();
                    };
                }
            });

        res.flush()
            .expect("Can't flush memory map modifications to disk");
        res
    }

    fn persist<T1>(
        &self,
        res: MmapMut,
        entity_mapping_persistor: Arc<T1>,
        embedding_persistor: &mut dyn EmbeddingPersistor,
    ) where
        T1: EntityMappingPersistor,
    {
        info!("Start saving embeddings.");

        let entities_count = self.sparse_matrix_reader.get_number_of_entities();
        embedding_persistor
            .put_metadata(entities_count, self.dimension)
            .unwrap_or_else(|_| {
                // if can't write first data to the file, probably further is the same
                panic!(
                    "Can't write metadata. Entities: {}. Dimension: {}.",
                    entities_count, self.dimension
                )
            });

        // entities which can't be written to the file (error occurs)
        let mut broken_entities = HashSet::new();
        for (i, hash) in self.sparse_matrix_reader.iter_hashes().enumerate() {
            let entity_name_opt = entity_mapping_persistor.get_entity(hash.value);
            if let Some(entity_name) = entity_name_opt {
                let mut embedding: Vec<f32> = Vec::with_capacity(self.dimension as usize);
                for j in 0..(self.dimension as usize) {
                    let start_idx = ((j * entities_count as usize) + i as usize) * 4;
                    let end_idx = start_idx + 4;
                    let pointer: *const u8 = (&res[start_idx..end_idx]).as_ptr();
                    let value = unsafe {
                        let value = pointer as *const f32;
                        *value
                    };

                    embedding.insert(j, value);
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
