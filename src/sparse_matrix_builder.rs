use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

use dashmap::DashMap;
use itertools::Itertools;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelDrainFull;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use rayon::ThreadPoolBuilder;
use rustc_hash::FxHasher;
use smallvec::SmallVec;

use crate::entity::{Hyperedge, SMALL_VECTOR_SIZE};
use crate::sparse_matrix::{Edge, Entity, SparseMatrix, SparseMatrixDescriptor};

#[derive(Debug, Default)]
struct Row {
    occurrence: u32,
    row_sum: f32,
}

/// Data locality plays huge role in propagation phase performance
/// We want connected nodes to have similar indices, as they will get updated together.
/// NodeIndexer assigns successive indices to nodes connected via hyper-edges.
/// Such ordering yields significant performance boost in propagation phase.
#[derive(Debug, Default)]
pub struct NodeIndexer {
    pub key_2_index: HashMap<u64, usize, BuildHasherDefault<FxHasher>>,
    pub index_2_key: Vec<u64>,
    pub index_2_entity_id: Vec<String>,
    pub index_2_column_id: Vec<u8>,
}

pub trait NodeIndexerBuilder {
    fn process(&self, key: u64, entity_id: &str, column_id: u8);
    fn finish(self) -> NodeIndexer;
}

#[derive(Debug)]
pub struct SyncNodeIndexerBuilder {
    node_indexer: RefCell<NodeIndexer>,
}

impl Default for SyncNodeIndexerBuilder {
    fn default() -> Self {
        SyncNodeIndexerBuilder {
            node_indexer: RefCell::new(NodeIndexer {
                key_2_index: Default::default(),
                index_2_key: vec![],
                index_2_column_id: vec![],
                index_2_entity_id: vec![],
            }),
        }
    }
}

impl NodeIndexerBuilder for SyncNodeIndexerBuilder {
    fn process(&self, key: u64, entity_id: &str, column_id: u8) {
        let mut node_indexer = self.node_indexer.borrow_mut();

        if node_indexer.key_2_index.contains_key(&key) {
            return;
        }
        let index = node_indexer.key_2_index.len();
        node_indexer.key_2_index.insert(key, index);
        node_indexer.index_2_key.push(key);
        node_indexer.index_2_entity_id.push(entity_id.to_string());
        node_indexer.index_2_column_id.push(column_id);
    }

    fn finish(self) -> NodeIndexer {
        self.node_indexer.into_inner()
    }
}

#[derive(Debug)]
pub struct IndexedEntity {
    index: usize,
    id: String,
    column_id: u8,
}

#[derive(Debug, Default)]
pub struct AsyncNodeIndexerBuilder {
    key_2_entity: DashMap<u64, IndexedEntity, BuildHasherDefault<FxHasher>>,
    next_index: AtomicUsize,
}

impl NodeIndexerBuilder for AsyncNodeIndexerBuilder {
    fn process(&self, key: u64, entity_id: &str, column_id: u8) {
        self.key_2_entity.entry(key).or_insert_with(|| {
            let index = self.next_index.fetch_add(1, Ordering::Relaxed);
            let id = entity_id.to_string();
            IndexedEntity {
                index,
                id,
                column_id,
            }
        });
    }

    fn finish(self) -> NodeIndexer {
        // Thin wrappers over pointer to make it Send/Sync
        // https://stackoverflow.com/a/70848420

        #[derive(Copy, Clone)]
        struct PointerU64(*mut u64);
        unsafe impl Send for PointerU64 {}
        unsafe impl Sync for PointerU64 {}

        #[derive(Copy, Clone)]
        struct PointerString(*mut String);
        unsafe impl Send for PointerString {}
        unsafe impl Sync for PointerString {}

        #[derive(Copy, Clone)]
        struct PointerU8(*mut u8);
        unsafe impl Send for PointerU8 {}
        unsafe impl Sync for PointerU8 {}

        let numel = self.next_index.into_inner();
        let mut index_2_key: Vec<u64> = vec![0; numel];
        let mut index_2_entity_id = vec![String::new(); numel];
        let mut index_2_column_id = vec![0; numel];

        let index_2_key_ptr = PointerU64(index_2_key.as_mut_ptr());
        let index_2_entity_id_ptr = PointerString(index_2_entity_id.as_mut_ptr());
        let index_2_column_id_ptr = PointerU8(index_2_column_id.as_mut_ptr());

        let key_2_index = self
            .key_2_entity
            .into_par_iter()
            .map(|(key, indexed_entity)| {
                let IndexedEntity {
                    index,
                    id: entity_id,
                    column_id,
                } = indexed_entity;
                unsafe {
                    ptr::write(index_2_key_ptr.0.add(index), key);
                    ptr::write(index_2_entity_id_ptr.0.add(index), entity_id);
                    ptr::write(index_2_column_id_ptr.0.add(index), column_id);
                }
                (key, index)
            })
            .collect();

        NodeIndexer {
            key_2_index,
            index_2_key,
            index_2_entity_id,
            index_2_column_id,
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

    pub fn make_buffer(&self, hyperedge_trim_n: usize) -> SparseMatrixBuffer {
        SparseMatrixBuffer {
            descriptor: self.clone(),
            edge_count: 0,
            hash_2_row: Default::default(),
            hashes_2_edge: Default::default(),
            hyperedge_trim_n,
        }
    }
}

#[derive(Debug)]
pub struct SparseMatrixBuffer {
    pub descriptor: SparseMatrixDescriptor,
    pub edge_count: u32,
    hash_2_row: HashMap<u64, Row, BuildHasherDefault<FxHasher>>,
    hashes_2_edge: HashMap<(u64, u64), f32, BuildHasherDefault<FxHasher>>,
    hyperedge_trim_n: usize,
}

impl SparseMatrixBuffer {
    pub fn handle_hyperedge(&mut self, hyperedge: &Hyperedge) {
        let SparseMatrixDescriptor {
            col_a_id, col_b_id, ..
        } = self.descriptor;
        let total_combinations = hyperedge.edges_num(col_a_id, col_b_id) as u32;

        let mut nodes_a = hyperedge.nodes(col_a_id as usize);
        let mut nodes_b = hyperedge.nodes(col_b_id as usize);

        for hash in &nodes_a {
            self.update_row(*hash, nodes_b.len() as u32);
        }
        for hash in &nodes_b {
            self.update_row(*hash, nodes_a.len() as u32);
        }

        let value = 1f32 / (total_combinations as f32);

        let (nodes_a_high, nodes_a_low) = self.get_high_low_nodes(&mut nodes_a);
        let (nodes_b_high, nodes_b_low) = self.get_high_low_nodes(&mut nodes_b);
        self.handle_combinations(nodes_a_high, nodes_b_high, value);
        self.handle_combinations(nodes_a_high, nodes_b_low, value);
        self.handle_combinations(nodes_a_low, nodes_b_high, value);
        // Ignore 'low-to-low' combinations
    }

    fn get_high_low_nodes<'a>(
        &self,
        nodes: &'a mut SmallVec<[u64; SMALL_VECTOR_SIZE]>,
    ) -> (&'a [u64], &'a [u64]) {
        if nodes.len() > self.hyperedge_trim_n {
            nodes.select_nth_unstable_by_key(self.hyperedge_trim_n, |h| {
                Reverse(self.hash_2_row.get(h).map_or(0, |r| r.occurrence))
            });
            nodes.split_at(self.hyperedge_trim_n)
        } else {
            (nodes, &[])
        }
    }

    fn handle_combinations(&mut self, a_hashes: &[u64], b_hashes: &[u64], value: f32) {
        for a_hash in a_hashes {
            for b_hash in b_hashes {
                self.add_pair_symmetric(*a_hash, *b_hash, value);
            }
        }
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
    fn add_pair_symmetric(&mut self, a_hash: u64, b_hash: u64, value: f32) {
        self.edge_count += 1;
        self.update_edge(a_hash, b_hash, value);
        self.update_edge(b_hash, a_hash, value);
    }

    fn update_row(&mut self, hash: u64, count: u32) {
        let val = 1f32 / (count as f32);
        let e = self.hash_2_row.entry(hash).or_default();
        e.occurrence += count;
        e.row_sum += val
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
    num_workers: usize,
}

pub struct EdgeEntry {
    pub row: u32,
    pub col: u32,
    pub value: f32,
}

impl SparseMatrixBuffersReducer {
    pub fn new(
        node_indexer: NodeIndexer,
        buffers: Vec<SparseMatrixBuffer>,
        num_workers: usize,
    ) -> Self {
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
            num_workers,
        }
    }

    pub fn reduce(self) -> SparseMatrix {
        ThreadPoolBuilder::new()
            .num_threads(self.num_workers)
            .build()
            .unwrap()
            .install(|| {
                let node_indexer = self.node_indexer;

                // Extract buffers so their fields can be moved to reducing functions
                let (hash_2_row_maps, hashes_2_edge_map): (Vec<_>, Vec<_>) = self
                    .buffers
                    .into_iter()
                    .map(|b| (b.hash_2_row, b.hashes_2_edge))
                    .unzip();
                let entities =
                    SparseMatrixBuffersReducer::reduce_to_entities(&node_indexer, hash_2_row_maps);
                let mut edges: Vec<_> =
                    SparseMatrixBuffersReducer::reduce_to_edges(&node_indexer, hashes_2_edge_map);
                edges.par_sort_by_key(|entry| (entry.row, entry.col));

                let slices: Vec<_> = edges
                    .iter()
                    .enumerate()
                    .group_by(|(_, entry)| entry.row)
                    .into_iter()
                    .map(|(_, mut group)| {
                        let first = group.next().expect("Group have at least one element");
                        let last = group.last().unwrap_or(first);
                        (first.0, last.0 + 1)
                    })
                    .collect();

                let mut edges: Vec<_> = edges
                    .into_par_iter()
                    .map(|entry| Edge {
                        other_entity_ix: entry.col,
                        // use this field for different purpose to avoid reallocation
                        left_markov_value: entry.value,
                        symmetric_markov_value: 0.0,
                    })
                    .collect();

                slices
                    .iter()
                    .enumerate()
                    .for_each(|(row_ix, (start_ix, end_ix))| {
                        let row_sum = entities[row_ix].row_sum;
                        let slice = &mut edges[(*start_ix)..(*end_ix)];
                        slice.iter_mut().for_each(|edge| {
                            let value = edge.left_markov_value;

                            let left_markov_normalization = row_sum;
                            let symmetric_markov_normalization = {
                                let col_sum = entities[edge.other_entity_ix as usize].row_sum;
                                (row_sum * col_sum).sqrt()
                            };
                            edge.left_markov_value = value / left_markov_normalization;
                            edge.symmetric_markov_value = value / symmetric_markov_normalization;
                        })
                    });

                SparseMatrix {
                    descriptor: self.descriptor,
                    entity_ids: node_indexer.index_2_entity_id,
                    column_ids: node_indexer.index_2_column_id,
                    entities,
                    edges,
                    slices,
                }
            })
    }

    fn reduce_to_entities(
        node_indexer: &NodeIndexer,
        entity_maps: Vec<HashMap<u64, Row, BuildHasherDefault<FxHasher>>>,
    ) -> Vec<Entity> {
        node_indexer
            .index_2_key
            .par_iter()
            .map(|hash| {
                let mut entity_agg = Entity { row_sum: 0.0 };
                for entity_map in entity_maps.iter() {
                    if let Some(entity) = entity_map.get(hash) {
                        entity_agg.row_sum += entity.row_sum;
                    }
                }
                entity_agg
            })
            .collect()
    }

    fn reduce_to_edges(
        node_indexer: &NodeIndexer,
        edge_maps: Vec<HashMap<(u64, u64), f32, BuildHasherDefault<FxHasher>>>,
    ) -> Vec<EdgeEntry> {
        // Dashmap to have concurrent write access with par_drain
        // par_drain is recommended to not increase peak memory usage
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
                let row = *node_indexer
                    .key_2_index
                    .get(&row_hash)
                    .expect("Hash value was indexed") as u32;
                let col = *node_indexer
                    .key_2_index
                    .get(&col_hash)
                    .expect("Hash value was indexed") as u32;
                EdgeEntry { row, col, value }
            })
            .collect()
    }
}
