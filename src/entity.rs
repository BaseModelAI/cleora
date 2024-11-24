use itertools::{Itertools, Product};
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use smallvec::{IntoIter, SmallVec};
use twox_hash::XxHash64;

use crate::configuration::Configuration;
use crate::sparse_matrix_builder::NodeIndexerBuilder;

/// Indicates how many elements in a vector can be placed on Stack (used by smallvec crate). The rest
/// of the vector is placed on Heap.
pub const SMALL_VECTOR_SIZE: usize = 8;

#[derive(Debug, Clone)]
pub struct Hyperedge {
    hashes: SmallVec<[u64; SMALL_VECTOR_SIZE]>,
    slices: [Range<u32>; 2],
}

impl Hyperedge {
    #[inline]
    pub fn nodes(&self, column_id: usize) -> SmallVec<[u64; SMALL_VECTOR_SIZE]> {
        let slice = self.slices.get(column_id).unwrap();
        let mut v = SmallVec::with_capacity(slice.len());
        for ix in slice.start..slice.end {
            v.push(self.hashes[ix as usize])
        }
        v
    }

    #[inline(always)]
    pub fn edges_iter(
        &self,
        col_id_a: u8,
        col_id_b: u8,
    ) -> Product<IntoIter<[u64; 8]>, IntoIter<[u64; 8]>> {
        let nodes_a = self.nodes(col_id_a as usize);
        let nodes_b = self.nodes(col_id_b as usize);
        nodes_a.into_iter().cartesian_product(nodes_b)
    }

    pub fn edges_num(&self, col_id_a: u8, col_id_b: u8) -> usize {
        self.slices[col_id_a as usize].len() * self.slices[col_id_b as usize].len()
    }
}

pub struct EntityProcessor<'a, S: NodeIndexerBuilder> {
    config: &'a Configuration,
    not_ignored_columns_count: u16,
    node_indexer: Arc<S>,
}

impl<'a, S: NodeIndexerBuilder> EntityProcessor<'a, S> {
    pub fn new(config: &'a Configuration, node_indexer: Arc<S>) -> EntityProcessor<'a, S> {
        let not_ignored_columns_count = config.columns.len() as u16;
        EntityProcessor {
            config,
            not_ignored_columns_count,
            node_indexer,
        }
    }

    /// Every row can create few combinations (cartesian products) which are hashed and provided for sparse matrix creation.
    /// `row` - array of strings such as: ("userId1", "productId1 productId2", "brandId1").
    pub fn process_row_and_get_edges(
        &self,
        row: &[SmallVec<[&str; SMALL_VECTOR_SIZE]>],
    ) -> Hyperedge {
        let mut hashes: SmallVec<[u64; SMALL_VECTOR_SIZE]> =
            SmallVec::with_capacity(self.not_ignored_columns_count as usize);
        let mut slices: [Range<u32>; 2] = [0..0, 0..0];
        let mut reflexive_count = 0;
        let mut current_offset = 0u32;

        for (i, column_entities) in row.iter().enumerate() {
            let column = &self.config.columns[i];
            let column_id = i as u8;
            if column.complex {
                for entity in column_entities {
                    let hash = hash_entity(entity);
                    hashes.push(hash);
                    self.node_indexer.process(hash, entity, column_id);
                }
                let length = column_entities.len() as u32;
                slices[i] = current_offset..(current_offset + length);
                if column.reflexive {
                    // put reflexive column data to the end of the buffers
                    let reflexive_id = (self.not_ignored_columns_count + reflexive_count) as usize;
                    slices[reflexive_id] = current_offset..(current_offset + length);
                    reflexive_count += 1;
                }
                current_offset += length;
            } else {
                let entity = column_entities.first().unwrap();
                let hash = hash_entity(entity);
                hashes.push(hash);
                self.node_indexer.process(hash, entity, column_id);
                let length = 1u32;
                slices[i] = current_offset..(current_offset + length);
                current_offset += length;
            }
        }
        Hyperedge { hashes, slices }
    }
}

#[inline(always)]
pub fn hash_entity(entity: &str) -> u64 {
    let mut hasher = XxHash64::default();
    hasher.write(entity.as_bytes());
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use smallvec::{smallvec, SmallVec};

    use crate::entity::{Hyperedge, SMALL_VECTOR_SIZE};

    #[test]
    fn generate_cartesian_product_hashes() {
        // hashes for entities in every column
        // column_1: 1 entity
        // column_2: 2 entities
        // column_3: 3 entities
        let slices = [0..2, 2..5];
        let hashes: SmallVec<[u64; SMALL_VECTOR_SIZE]> = smallvec![10, 20, 30, 40, 50];
        let hyperedge = Hyperedge { hashes, slices };
        let combinations: Vec<_> = hyperedge.edges_iter(0, 1).collect();
        assert_eq!((10, 30), *combinations.get(0).unwrap());
        assert_eq!((10, 40), *combinations.get(1).unwrap());
        assert_eq!((10, 50), *combinations.get(2).unwrap());
        assert_eq!((20, 30), *combinations.get(3).unwrap());
        assert_eq!((20, 40), *combinations.get(4).unwrap());
        assert_eq!((20, 50), *combinations.get(5).unwrap());
        assert_eq!(None, combinations.get(6));
    }
}
