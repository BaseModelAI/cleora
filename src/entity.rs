use crate::configuration::{Column, Configuration};
use crate::persistence::entity::EntityMappingPersistor;
use smallvec::{smallvec, SmallVec};
use std::hash::Hasher;
use std::sync::Arc;
use twox_hash::XxHash64;

/// Indicates how many elements in a vector can be placed on Stack (used by smallvec crate). The rest
/// of the vector is placed on Heap.
pub const SMALL_VECTOR_SIZE: usize = 8;

/// Marker for elements in a vector. Let's say that we have `vec![1, 2, 3, 4]`
/// and `LengthAndOffset { length: 2, offset : 1 }`. Offset points to the second element in the vector
/// and length tell us how many elements we should take (in that case 2 elements: 2 and 3).
#[derive(Clone, Copy)]
struct LengthAndOffset {
    length: u32,
    offset: u32,
}

struct CartesianProduct {
    has_next: bool,
    lengths_and_offsets: SmallVec<[LengthAndOffset; SMALL_VECTOR_SIZE]>,
    indices: SmallVec<[u32; SMALL_VECTOR_SIZE]>,
}

impl CartesianProduct {
    fn new(
        lengths_and_offsets: SmallVec<[LengthAndOffset; SMALL_VECTOR_SIZE]>,
    ) -> CartesianProduct {
        let indices: SmallVec<[u32; SMALL_VECTOR_SIZE]> = lengths_and_offsets
            .iter()
            .map(|length_and_offset| length_and_offset.offset)
            .collect();
        CartesianProduct {
            has_next: true,
            lengths_and_offsets,
            indices,
        }
    }
}

impl Iterator for CartesianProduct {
    /// The type of the elements being iterated over.
    type Item = SmallVec<[u32; SMALL_VECTOR_SIZE]>;

    /// Advances the iterator and returns the next value - cartesian product.
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next {
            return None;
        }

        let len = self.indices.len();
        let result: SmallVec<[u32; SMALL_VECTOR_SIZE]> = SmallVec::from_slice(&self.indices);
        for i in (0..len).rev() {
            let LengthAndOffset { length, offset } = self.lengths_and_offsets[i];
            let last_index = length + offset;
            if self.indices[i] == (last_index - 1) {
                self.indices[i] = offset;
                if i == 0 {
                    self.has_next = false;
                }
            } else {
                self.indices[i] += 1;
                break;
            }
        }
        Some(result)
    }
}

pub struct EntityProcessor<'a, T, F>
where
    T: EntityMappingPersistor,
    F: FnMut(SmallVec<[u64; SMALL_VECTOR_SIZE]>),
{
    config: &'a Configuration,
    field_hashes: SmallVec<[u64; SMALL_VECTOR_SIZE]>,
    not_ignored_columns_count: u16,
    columns_count: u16,
    entity_mapping_persistor: Arc<T>,
    hashes_handler: F,
}

impl<'a, T, F> EntityProcessor<'a, T, F>
where
    T: EntityMappingPersistor,
    F: FnMut(SmallVec<[u64; SMALL_VECTOR_SIZE]>),
{
    pub fn new(
        config: &'a Configuration,
        persistor: Arc<T>,
        hashes_handler: F,
    ) -> EntityProcessor<'a, T, F> {
        let columns = &config.columns;
        // hashes for column names are used to differentiate entities with the same name
        // from different columns
        let field_hashes_vec: Vec<u64> = columns.iter().map(|c| hash(&c.name)).collect();
        let field_hashes: SmallVec<[u64; SMALL_VECTOR_SIZE]> = SmallVec::from_vec(field_hashes_vec);
        let not_ignored_cols = config.not_ignored_columns();
        let mut not_ignored_columns_count = 0;
        let mut reflexive_columns_count = 0;
        for &col in &not_ignored_cols {
            not_ignored_columns_count += 1;
            if col.reflexive {
                reflexive_columns_count += 1
            };
        }

        let columns_count = not_ignored_columns_count + reflexive_columns_count;

        EntityProcessor {
            config,
            field_hashes,
            not_ignored_columns_count,
            columns_count,
            entity_mapping_persistor: persistor,
            hashes_handler,
        }
    }

    /// Every row can create few combinations (cartesian products) which are hashed and provided for sparse matrix creation.
    /// `row` - array of strings such as: ("userId1", "productId1 productId2", "brandId1").
    pub fn process_row<S: AsRef<str>>(&mut self, row: &[SmallVec<[S; SMALL_VECTOR_SIZE]>]) {
        let mut hashes: SmallVec<[u64; SMALL_VECTOR_SIZE]> =
            SmallVec::with_capacity(self.not_ignored_columns_count as usize);
        let mut lens_and_offsets: SmallVec<[LengthAndOffset; SMALL_VECTOR_SIZE]> =
            smallvec![LengthAndOffset{ length: 0, offset: 0}; self.columns_count as usize];
        let mut reflexive_count = 0;
        let mut current_offset = 0u32;

        let mut idx = 0;
        for (i, column_entities) in row.iter().enumerate() {
            let column = &self.config.columns[i];
            if !column.ignored {
                if column.complex {
                    for entity in column_entities {
                        let hash = self.field_hashes[i] ^ hash(entity.as_ref());
                        hashes.push(hash);
                        self.update_entity_mapping(entity.as_ref(), hash, column);
                    }
                    let length = column_entities.len() as u32;
                    lens_and_offsets[idx] = LengthAndOffset {
                        length,
                        offset: current_offset,
                    };
                    if column.reflexive {
                        // put reflexive column data to the end of the buffers
                        let reflexive_id =
                            (self.not_ignored_columns_count + reflexive_count) as usize;
                        lens_and_offsets[reflexive_id] = LengthAndOffset {
                            length,
                            offset: current_offset,
                        };
                        reflexive_count += 1;
                    }
                    current_offset += length;
                } else {
                    let entity = column_entities.get(0).unwrap().as_ref();
                    let hash = self.field_hashes[i] ^ hash(entity);
                    hashes.push(hash);
                    self.update_entity_mapping(entity, hash, column);
                    let length = 1u32;
                    lens_and_offsets[idx] = LengthAndOffset {
                        length,
                        offset: current_offset,
                    };
                    current_offset += length;
                }
                idx += 1;
            }
        }

        let hash_rows = self.generate_combinations_with_length(hashes, lens_and_offsets);
        for hash_row in hash_rows {
            (self.hashes_handler)(hash_row);
        }
    }

    #[inline(always)]
    fn update_entity_mapping(&mut self, entity: &str, hash: u64, column: &Column) {
        if !column.transient && !self.entity_mapping_persistor.contains(hash) {
            let entry = if self.config.prepend_field {
                let mut entry = column.name.clone();
                entry.push_str("__");
                entry.push_str(entity);
                entry
            } else {
                entity.to_string()
            };
            self.entity_mapping_persistor.put_data(hash, entry);
        }
    }

    /// It creates Cartesian Product for incoming data.
    /// Let's say that we have such columns:
    /// customers | products                | brands
    /// incoming data:
    /// userId1   | productId1, productId2  | brandId1, brandId2
    /// Total number of combinations is equal to 4 (1 * 2 * 2) based on:
    /// number of entities in customers column * number of entities in products column * number of entities in brands column
    /// Cartesian Products for our data:
    /// (userId1, productId1, brandId1), (userId1, productId1, brandId2), (userId1, productId2, brandId1), (userId1, productId2, brandId2)
    /// `hashes` - entity hashes
    /// `lens_and_offsets` - number of entities per column
    /// return entity hashes Cartesian Products. Size of the array (matrix) is equal to number of combinations x number of columns (including reflexive column)
    #[inline(always)]
    fn generate_combinations_with_length(
        &self,
        hashes: SmallVec<[u64; SMALL_VECTOR_SIZE]>,
        lens_and_offsets: SmallVec<[LengthAndOffset; SMALL_VECTOR_SIZE]>,
    ) -> impl Iterator<Item = SmallVec<[u64; SMALL_VECTOR_SIZE]>> {
        let row_length = lens_and_offsets.len();
        let mut total_combinations = 1;
        for len_and_offset in &lens_and_offsets {
            total_combinations *= len_and_offset.length;
        }

        let cartesian = CartesianProduct::new(lens_and_offsets);

        cartesian.map(move |indices| {
            let mut arr: SmallVec<[u64; SMALL_VECTOR_SIZE]> =
                SmallVec::with_capacity(row_length + 1);
            arr.push(total_combinations as u64);
            for i in indices {
                let value = hashes[i as usize];
                arr.push(value);
            }
            arr
        })
    }

    pub fn finish(&mut self) {
        let end_vec = vec![0u64];
        (self.hashes_handler)(SmallVec::from_vec(end_vec));
    }
}

#[inline(always)]
fn hash(entity: &str) -> u64 {
    let mut hasher = XxHash64::default();
    hasher.write(entity.as_bytes());
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use crate::configuration::{Column, Configuration};
    use crate::entity::{
        hash, CartesianProduct, EntityProcessor, LengthAndOffset, SMALL_VECTOR_SIZE,
    };
    use crate::persistence::entity::InMemoryEntityMappingPersistor;
    use smallvec::{smallvec, SmallVec};
    use std::sync::Arc;

    fn prepare_lengths_and_offsets(
        entities_per_column: &[u32],
    ) -> SmallVec<[LengthAndOffset; SMALL_VECTOR_SIZE]> {
        let mut lens_and_offsets: SmallVec<[LengthAndOffset; SMALL_VECTOR_SIZE]> =
            SmallVec::with_capacity(entities_per_column.len());
        let mut offset = 0;
        for &num_of_entities in entities_per_column {
            lens_and_offsets.push(LengthAndOffset {
                length: num_of_entities,
                offset,
            });
            offset += num_of_entities;
        }
        lens_and_offsets
    }

    fn prepare_hashes(
        total_combination: u64,
        entities: &[&str],
        field_hashes: &[u64],
    ) -> SmallVec<[u64; SMALL_VECTOR_SIZE]> {
        let mut hashes: SmallVec<[u64; SMALL_VECTOR_SIZE]> = SmallVec::new();
        hashes.push(total_combination);
        for (i, &entity) in entities.iter().enumerate() {
            let hash = field_hashes[i] ^ hash(entity);
            hashes.push(hash);
        }
        hashes
    }

    #[test]
    fn generate_cartesian_product_indices() {
        let lengths_and_offsets = prepare_lengths_and_offsets(&[2, 1, 3]);

        let cartesian_product = CartesianProduct::new(lengths_and_offsets);
        let mut iter = cartesian_product.into_iter();

        assert_eq!(Some(smallvec![0, 2, 3]), iter.next());
        assert_eq!(Some(smallvec![0, 2, 4]), iter.next());
        assert_eq!(Some(smallvec![0, 2, 5]), iter.next());
        assert_eq!(Some(smallvec![1, 2, 3]), iter.next());
        assert_eq!(Some(smallvec![1, 2, 4]), iter.next());
        assert_eq!(Some(smallvec![1, 2, 5]), iter.next());

        assert_eq!(None, iter.next());
    }

    #[test]
    fn generate_cartesian_product_hashes() {
        let dummy_config = Configuration::default(String::from(""), vec![]);

        // hashes for entities in every column
        // column_1: 1 entity
        // column_2: 2 entities
        // column_3: 3 entities
        let lengths_and_offsets = prepare_lengths_and_offsets(&[1, 2, 3]);
        let hashes: SmallVec<[u64; SMALL_VECTOR_SIZE]> = smallvec![10, 20, 30, 40, 50, 60];
        let mut total_combinations = 1u64;
        for len_and_offset in &lengths_and_offsets {
            total_combinations *= len_and_offset.length as u64;
        }

        let in_memory_entity_mapping_persistor = InMemoryEntityMappingPersistor::default();
        let in_memory_entity_mapping_persistor = Arc::new(in_memory_entity_mapping_persistor);
        let entity_processor = EntityProcessor::new(
            &dummy_config,
            in_memory_entity_mapping_persistor.clone(),
            |_hashes| {},
        );

        let combinations: Vec<_> = entity_processor
            .generate_combinations_with_length(hashes, lengths_and_offsets)
            .collect();
        assert_eq!(
            &SmallVec::from([total_combinations, 10, 20, 40]),
            combinations.get(0).unwrap()
        );
        assert_eq!(
            &SmallVec::from([total_combinations, 10, 20, 50]),
            combinations.get(1).unwrap()
        );
        assert_eq!(
            &SmallVec::from([total_combinations, 10, 20, 60]),
            combinations.get(2).unwrap()
        );
        assert_eq!(
            &SmallVec::from([total_combinations, 10, 30, 40]),
            combinations.get(3).unwrap()
        );
        assert_eq!(
            &SmallVec::from([total_combinations, 10, 30, 50]),
            combinations.get(4).unwrap()
        );
        assert_eq!(
            &SmallVec::from([total_combinations, 10, 30, 60]),
            combinations.get(5).unwrap()
        );
        assert_eq!(None, combinations.get(6));
    }

    #[test]
    fn process_row_and_handle_hashes() {
        let columns = vec![
            Column {
                name: String::from("column_1"),
                transient: false,
                complex: false,
                reflexive: false,
                ignored: true,
            },
            Column {
                name: String::from("column_2"),
                transient: true,
                complex: false,
                reflexive: false,
                ignored: false,
            },
            Column {
                name: String::from("column_3"),
                transient: false,
                complex: true,
                reflexive: true,
                ignored: false,
            },
            Column {
                name: String::from("column_4"),
                transient: false,
                complex: false,
                reflexive: false,
                ignored: false,
            },
        ];
        // columns configuration: ignored::column_1 transient::column_2 complex::reflexive::column3 column_4
        // first column is ignored - we don't process entities from that column
        // third column is reflexive so we put it at the end
        let column_names = vec![
            columns[1].name.clone(),
            columns[2].name.clone(),
            columns[3].name.clone(),
            columns[2].name.clone(),
        ];
        // hashes for column names are used to differentiate entities with the same name
        // from different columns
        let field_hashes: Vec<u64> = column_names.iter().map(|name| hash(name)).collect();

        // columns are most important, the rest can be omitted
        let dummy_config = Configuration::default(String::from(""), columns);

        let in_memory_entity_mapping_persistor = InMemoryEntityMappingPersistor::default();
        let in_memory_entity_mapping_persistor = Arc::new(in_memory_entity_mapping_persistor);
        let mut result: SmallVec<[SmallVec<[u64; SMALL_VECTOR_SIZE]>; SMALL_VECTOR_SIZE]> =
            SmallVec::new();
        let mut entity_processor = EntityProcessor::new(
            &dummy_config,
            in_memory_entity_mapping_persistor.clone(),
            |hashes| {
                result.push(hashes);
            },
        );

        let row = vec![
            smallvec!["a"],
            smallvec!["bb"],
            smallvec!["ccc", "ddd"],
            smallvec!["eeee"],
        ];
        entity_processor.process_row(&row);

        // first column is ignored, third one is reflexive so the entities go at the end
        // input: "bb", "ccc ddd", "eeee", "ccc ddd"
        // number of cartesian products from the above entities
        assert_eq!(4, result.len());
        assert_eq!(
            prepare_hashes(4, &["bb", "ccc", "eeee", "ccc"], &field_hashes),
            result[0]
        );
        assert_eq!(
            prepare_hashes(4, &["bb", "ccc", "eeee", "ddd"], &field_hashes),
            result[1]
        );
        assert_eq!(
            prepare_hashes(4, &["bb", "ddd", "eeee", "ccc"], &field_hashes),
            result[2]
        );
        assert_eq!(
            prepare_hashes(4, &["bb", "ddd", "eeee", "ddd"], &field_hashes),
            result[3]
        );
    }
}
