use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::Mutex;

use dashmap::{DashMap, DashSet};
use itertools::Itertools;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelDrainFull;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use rustc_hash::FxHasher;

use crate::entity::Hyperedge;
use crate::sparse_matrix::{Edge, Entity, SparseMatrix, SparseMatrixDescriptor};

#[derive(Debug, Default)]
struct Row {
    occurrence: u32,
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
    unprocessed_keys: DashSet<u64, BuildHasherDefault<FxHasher>>,
    index_2_key: Mutex<Vec<u64>>,
}

impl NodeIndexerBuilder {
    pub fn new(descriptor: &SparseMatrixDescriptor) -> Self {
        Self {
            col_a_id: descriptor.col_a_id,
            col_b_id: descriptor.col_b_id,
            key_2_index: Default::default(),
            unprocessed_keys: Default::default(),
            index_2_key: Default::default(),
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
        self.process_queued(&mut index_to_key);
        NodeIndexer {
            key_2_index: self.key_2_index.into_iter().collect(),
            index_2_key: index_to_key,
        }
    }

    fn insert_nonblocking(&self, key: &u64) {
        // Rozrzucac po haszu do workerow?
        // ale na 1 workerze tez jest problem
        if self.key_2_index.contains_key(key) || self.unprocessed_keys.contains(key) {
            return;
        }
        self.unprocessed_keys.insert(*key);
        if let Ok(mut index_2_key) = self.index_2_key.try_lock() {
            // One worker that succeeds to lock it -> will process the queue.
            self.process_queued(&mut index_2_key);
        }
    }

    fn process_queued(&self, index_2_key: &mut Vec<u64>) {
        let keys: Vec<u64> = self.unprocessed_keys.iter().map(|k| *k).collect();
        for key in keys {
            if let Some(key) = self.unprocessed_keys.remove(&key) {
                if !self.key_2_index.contains_key(&key) {
                    let next_id = index_2_key.len();
                    index_2_key.push(key);
                    self.key_2_index.insert(key, next_id);
                }
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
    hashes_2_edge: HashMap<(u64, u64), f32, BuildHasherDefault<FxHasher>>,
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
    }

    fn update_edge(&mut self, a_hash: u64, b_hash: u64, val: f32) {
        let e = self.hashes_2_edge.entry((a_hash, b_hash)).or_default();
        *e += val;
    }
}

#[derive(Debug)]
pub struct SparseMatrixBuffersReducer {
    descriptor: SparseMatrixDescriptor,
    buffers: Vec<SparseMatrixBuffer>,
    node_indexer: NodeIndexer,
}

pub struct EdgeEntry {
    pub row: u32,
    pub col: u32,
    pub value: f32,
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
        let node_indexer = self.node_indexer;

        println!("DEBUG START REDUCE");

        // Extract buffers so their fields can be moved to reducing functions
        let (hash_2_row_maps, hashes_2_edge_map): (Vec<_>, Vec<_>) = self
            .buffers
            .into_iter()
            .map(|b| (b.hash_2_row, b.hashes_2_edge))
            .unzip();
        let entities = SparseMatrixBuffersReducer::reduce_to_entities(&node_indexer, hash_2_row_maps);
        println!("DEBUG entities REDUCE");
        let mut edges: Vec<_> = SparseMatrixBuffersReducer::reduce_to_edges(&node_indexer, hashes_2_edge_map);
        println!("DEBUG edges 1 REDUCE");
        edges.par_sort_by_key(|entry| (entry.row, entry.col));

        let slices: Vec<_> = edges
            .iter()
            .enumerate()
            .group_by(|(ix, entry)| entry.row)
            .into_iter()
            .map(|(_, mut group)| {
                let first = group.next().unwrap();
                let last = group.last().unwrap_or(first);
                (first.0, last.0 + 1)
            })
            .collect();
        println!("DEBUG SLICES DONE");

        let edges = edges
            .into_par_iter()
            .map(|entry| Edge {
                other_entity_ix: entry.col,
                value: entry.value,
            })
            .collect();
        println!("DEBUG edges 2 REDUCE");

        SparseMatrix::new(self.descriptor, entities, edges, slices)
    }

    fn reduce_to_entities(
        node_indexer: &NodeIndexer,
        entity_maps: Vec<HashMap<u64, Row, BuildHasherDefault<FxHasher>>>,
    ) -> Vec<Entity> {
        node_indexer
            .index_2_key
            .par_iter()
            .map(|hash| {
                let mut entity_agg = Entity {
                    hash_value: *hash,
                    occurrence: 0,
                };
                for entity_map in entity_maps.iter() {
                    if let Some(entity) = entity_map.get(hash) {
                        entity_agg.occurrence += entity.occurrence;
                    }
                }
                entity_agg
            })
            .collect()
    }

    fn reduce_to_edges(
        node_indexer: &NodeIndexer, edge_maps: Vec<HashMap<(u64, u64), f32, BuildHasherDefault<FxHasher>>>,
    ) -> Vec<EdgeEntry> {
        let reduced_edge_map: DashMap<(u64, u64), f32, BuildHasherDefault<FxHasher>> =
            Default::default();
        for mut edge_map in edge_maps.into_iter() {
            edge_map.par_drain().for_each(|(k, v)| {
                reduced_edge_map
                    .entry(k)
                    .and_modify(|rv| *rv += v)
                    .or_insert(v);
            })
        }
        reduced_edge_map
            .into_par_iter()
            .map(|((row_hash, col_hash), value)| {
                let row = *node_indexer.key_2_index.get(&row_hash).unwrap() as u32;
                let col = *node_indexer.key_2_index.get(&col_hash).unwrap() as u32;
                EdgeEntry {
                    row, col, value
                }
            })
            .collect()

    }
}
