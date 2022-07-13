use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::Mutex;

use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use rustc_hash::FxHasher;

use crate::entity::Hyperedge;
use crate::sparse_matrix::{Entry, Hash, SparseMatrix, SparseMatrixDescriptor};

#[derive(Debug, Default)]
struct Row {
    occurrence: u32,
    row_sum: f32,
}

#[derive(Debug, Default)]
struct Edge {
    value: f32,
}

/// Data locality plays huge role in propagation phase performance
/// We want connected nodes to have similar indices, as they will get updated together.
/// NodeIndexer assigns successive indices to nodes connected via hyper-edges.
/// Such ordering yields significant performance boost in propagation phase.
#[derive(Debug)]
pub struct NodeIndexer {
    pub key_2_index: HashMap<u64, usize, BuildHasherDefault<FxHasher>>,
    pub index_2_key: Vec<u64>,
}

/// Non-blocking builder suitable for multi-thread usage
#[derive(Debug)]
pub struct NodeIndexerBuilder {
    col_a_id: u8,
    col_b_id: u8,
    key_2_index: DashMap<u64, usize, BuildHasherDefault<FxHasher>>,
    index_2_key: Mutex<Vec<u64>>,
    inserts_queue: SegQueue<u64>,
}

impl NodeIndexerBuilder {
    pub fn new(descriptor: &SparseMatrixDescriptor) -> Self {
        Self {
            col_a_id: descriptor.col_a_id,
            col_b_id: descriptor.col_b_id,
            key_2_index: Default::default(),
            index_2_key: Default::default(),
            inserts_queue: Default::default(),
        }
    }

    pub fn process(&self, hyperedge: &Hyperedge) {
        for hash in hyperedge.nodes(self.col_a_id as usize) {
            self.insert_nonblocking(&hash)
        }
        for hash in hyperedge.nodes(self.col_b_id as usize) {
            self.insert_nonblocking(&hash)
        }
    }

    pub fn finish(self) -> NodeIndexer {
        let mut index_to_key = self
            .index_2_key
            .try_lock()
            .expect("No one inserting at finish time")
            .to_owned();
        self.process_queue(&mut index_to_key);
        NodeIndexer {
            key_2_index: self.key_2_index.into_iter().collect(),
            index_2_key: index_to_key,
        }
    }

    fn insert_nonblocking(&self, key: &u64) {
        if self.key_2_index.contains_key(key) {
            return;
        }
        self.inserts_queue.push(*key);
        if let Ok(mut index_2_key) = self.index_2_key.try_lock() {
            // One worker that succeeds to lock it -> will process the queue.
            self.process_queue(&mut index_2_key);
        }
    }

    fn process_queue(&self, index_2_key: &mut Vec<u64>) {
        while let Some(key) = self.inserts_queue.pop() {
            if !self.key_2_index.contains_key(&key) {
                let next_id = index_2_key.len();
                index_2_key.push(key);
                self.key_2_index.insert(key, next_id);
            }
        }
    }
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
    hash_2_row: HashMap<u64, Row, BuildHasherDefault<FxHasher>>,
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
    node_indexer: NodeIndexer,
}

impl SparseMatrixBuffersReducer {
    pub fn new(node_indexer: NodeIndexer, buffers: Vec<SparseMatrixBuffer>) -> Self {
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
            node_indexer,
        }
    }

    pub fn reduce(self) -> SparseMatrix {
        let rows = self.reduce_row_maps();
        let hashes_2_edge = self.reduce_edge_maps();

        let mut entities: Vec<_> = {
            hashes_2_edge
                .par_iter()
                .map(|((row_hash, col_hash), edge)| {
                    let row_index = *self.node_indexer.key_2_index.get(row_hash).unwrap();
                    let col_index = *self.node_indexer.key_2_index.get(col_hash).unwrap();

                    let row_entity = rows.get(row_index).unwrap();
                    let normalized_edge_value = edge.value / row_entity.row_sum;

                    Entry {
                        row: row_index as u32,
                        col: col_index as u32,
                        value: normalized_edge_value,
                    }
                })
                .collect()
        };

        // Nodes are indexed in hyperedge batches in the order they are seen.
        // So nodes connected with a hyperedge should have close indices.
        //
        // `e.row + e.col` is a proxy to 'index of an edge between 'e.row'-th and 'e.col'-th entity'.
        // Sorting by it yields decent data locality. Edges operating on similar nodes are close.

        entities.par_sort_by_key(|e| e.row + e.col);

        let hashes: Vec<_> = self
            .node_indexer
            .index_2_key
            .par_iter()
            .zip(rows.par_iter())
            .map(|(hash, row)| Hash {
                value: *hash,
                occurrence: row.occurrence,
            })
            .collect();

        SparseMatrix::new(self.descriptor, hashes, entities)
    }

    fn reduce_row_maps(&self) -> Vec<Row> {
        self.node_indexer
            .index_2_key
            .par_iter()
            .map(|hash| {
                let mut entity_agg = Row::default();
                for b in self.buffers.iter() {
                    if let Some(entity) = b.hash_2_row.get(hash) {
                        entity_agg.occurrence += entity.occurrence;
                        entity_agg.row_sum += entity.row_sum;
                    }
                }
                entity_agg
            })
            .collect()
    }

    fn reduce_edge_maps(&self) -> HashMap<(u64, u64), Edge, BuildHasherDefault<FxHasher>> {
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
