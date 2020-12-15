pub mod entity {
    use rustc_hash::FxHashMap;
    use std::sync::RwLock;

    pub trait EntityMappingPersistor {
        fn get_entity(&self, hash: u64) -> Option<String>;
        fn put_data(&self, hash: u64, entity: String);
        fn contains(&self, hash: u64) -> bool;
    }

    #[derive(Debug, Default)]
    pub struct InMemoryEntityMappingPersistor {
        entity_mappings: RwLock<FxHashMap<u64, String>>,
    }

    impl EntityMappingPersistor for InMemoryEntityMappingPersistor {
        fn get_entity(&self, hash: u64) -> Option<String> {
            let entity_mappings_read = self.entity_mappings.read().unwrap();
            entity_mappings_read.get(&hash).map(|s| s.to_string())
        }

        fn put_data(&self, hash: u64, entity: String) {
            let mut entity_mappings_write = self.entity_mappings.write().unwrap();
            entity_mappings_write.insert(hash, entity);
        }

        fn contains(&self, hash: u64) -> bool {
            let entity_mappings_read = self.entity_mappings.read().unwrap();
            entity_mappings_read.contains_key(&hash)
        }
    }
}

pub mod sparse_matrix {
    use log::info;
    use rustc_hash::FxHashMap;
    use std::collections::hash_map;
    use std::mem;

    #[derive(Debug, Clone, Copy)]
    pub struct Entry {
        pub row: u32,
        pub col: u32,
        pub value: f32,
    }

    pub struct EntryIter<'a>(std::slice::Iter<'a, Entry>);

    impl Iterator for EntryIter<'_> {
        type Item = Entry;

        #[inline(always)]
        fn next(&mut self) -> Option<Entry> {
            self.0.next().copied()
        }
    }

    pub trait SparseMatrixPersistor {
        fn increment_hash_occurrence(&mut self, hash: u64) -> u32;
        fn get_hash_occurrence(&self, hash: u64) -> u32;
        fn get_id(&self, hash: u64) -> i32;
        fn get_hash(&self, id: u32) -> i64;
        fn get_entity_counter(&self) -> u32;
        fn get_or_add_id_by_hash(&mut self, hash: u64) -> u32;
        fn increment_edge_counter(&mut self) -> u32;
        fn get_amount_of_data(&self) -> u32;
        fn get_or_add_pair_index(&mut self, magic: u64, pos: u32) -> u32;
        fn update_row_sum(&mut self, id: u32, val: f32);
        fn get_row_sum(&self, id: u32) -> f32;
        fn add_new_entry(&mut self, pos: u32, entry: Entry);
        fn update_entry(&mut self, pos: u32, entry: Entry);
        fn get_entry(&self, pos: u32) -> Entry;
        fn replace_entry(&mut self, pos: u32, entry: Entry);
        fn iter_entries(&self) -> EntryIter<'_>;
        fn finish(&self);
    }

    #[derive(Debug, Default)]
    pub struct InMemorySparseMatrixPersistor {
        edge_count: u32,
        hash_2_id: FxHashMap<u64, u32>,
        id_2_hash: Vec<u64>,
        hash_2_count: FxHashMap<u64, u32>,
        row_sum: Vec<f32>,
        pair_index: FxHashMap<u64, u32>,
        entries: Vec<Entry>,
    }

    impl SparseMatrixPersistor for InMemorySparseMatrixPersistor {
        fn increment_hash_occurrence(&mut self, hash: u64) -> u32 {
            let value = self.hash_2_count.entry(hash).or_insert(0);
            *value += 1;
            *value
        }

        fn get_hash_occurrence(&self, hash: u64) -> u32 {
            *self.hash_2_count.get(&hash).unwrap()
        }

        fn get_id(&self, hash: u64) -> i32 {
            match self.hash_2_id.get(&hash) {
                Some(value) => *value as i32,
                None => -1i32,
            }
        }

        fn get_hash(&self, id: u32) -> i64 {
            match self.id_2_hash.get(id as usize) {
                Some(value) => *value as i64,
                None => -1i64,
            }
        }

        fn get_entity_counter(&self) -> u32 {
            self.id_2_hash.len() as u32
        }

        fn get_or_add_id_by_hash(&mut self, hash: u64) -> u32 {
            match self.hash_2_id.entry(hash) {
                hash_map::Entry::Vacant(entry) => {
                    let id = self.id_2_hash.len() as u32;
                    entry.insert(id);
                    self.id_2_hash.push(hash);
                    id
                }
                hash_map::Entry::Occupied(entry) => *entry.get(),
            }
        }

        fn increment_edge_counter(&mut self) -> u32 {
            self.edge_count += 1;
            self.edge_count
        }

        fn get_amount_of_data(&self) -> u32 {
            self.entries.len() as u32
        }

        fn get_or_add_pair_index(&mut self, magic: u64, pos: u32) -> u32 {
            *self.pair_index.entry(magic).or_insert(pos)
        }

        fn update_row_sum(&mut self, id: u32, val: f32) {
            let id = id as usize;
            if self.row_sum.len() == id {
                self.row_sum.push(val);
            } else {
                self.row_sum[id] += val;
            };
        }

        fn get_row_sum(&self, id: u32) -> f32 {
            self.row_sum[id as usize]
        }

        fn add_new_entry(&mut self, _pos: u32, entry: Entry) {
            self.entries.push(entry);
        }

        fn update_entry(&mut self, pos: u32, entry: Entry) {
            self.entries[pos as usize].value += entry.value;
        }

        fn get_entry(&self, pos: u32) -> Entry {
            self.entries[pos as usize]
        }

        #[inline(always)]
        fn iter_entries(&self) -> EntryIter<'_> {
            EntryIter(self.entries.iter())
        }

        fn replace_entry(&mut self, pos: u32, entry: Entry) {
            self.entries[pos as usize] = entry
        }

        fn finish(&self) {
            info!("Number of entities: {}", self.get_entity_counter());
            info!("Number of edges: {}", self.edge_count);
            info!("Number of entries: {}", self.entries.len());

            let hash_2_id_mem_size = self.hash_2_id.capacity() * 12;
            let id_2_hash_mem_size = self.id_2_hash.capacity() * 8;
            let hash_2_count_mem_size = self.hash_2_count.capacity() * 12;
            let row_sum_mem_size = self.row_sum.capacity() * 4;
            let pair_index_mem_size = self.pair_index.capacity() * 12;

            let entry_mem_size = mem::size_of::<Entry>();
            let entries_mem_size = self.entries.capacity() * entry_mem_size;

            let total_mem_size = hash_2_id_mem_size
                + id_2_hash_mem_size
                + hash_2_count_mem_size
                + row_sum_mem_size
                + pair_index_mem_size
                + entries_mem_size;

            info!(
                "Total memory usage by the struct ~ {} MB",
                (total_mem_size / 1048576)
            );
        }
    }
}

pub mod embedding {
    use std::fs::File;
    use std::io;
    use std::io::{BufWriter, Write};

    pub trait EmbeddingPersistor {
        fn put_metadata(&mut self, entity_count: u32, dimension: u16) -> Result<(), io::Error>;
        fn put_data(
            &mut self,
            entity: &str,
            occur_count: u32,
            vector: Vec<f32>,
        ) -> Result<(), io::Error>;
        fn finish(&mut self) -> Result<(), io::Error>;
    }

    pub struct TextFileVectorPersistor {
        buf_writer: BufWriter<File>,
        produce_entity_occurrence_count: bool,
    }

    impl TextFileVectorPersistor {
        pub fn new(filename: String, produce_entity_occurrence_count: bool) -> Self {
            let msg = format!("Unable to create file: {}", filename);
            let file = File::create(filename).expect(&msg);
            TextFileVectorPersistor {
                buf_writer: BufWriter::new(file),
                produce_entity_occurrence_count,
            }
        }
    }

    impl EmbeddingPersistor for TextFileVectorPersistor {
        fn put_metadata(&mut self, entity_count: u32, dimension: u16) -> Result<(), io::Error> {
            let metadata = format!("{} {}", entity_count, dimension);
            self.buf_writer.write_all(metadata.as_bytes())?;
            Ok(())
        }

        fn put_data(
            &mut self,
            entity: &str,
            occur_count: u32,
            vector: Vec<f32>,
        ) -> Result<(), io::Error> {
            self.buf_writer.write_all(b"\n")?;
            self.buf_writer.write_all(entity.as_bytes())?;

            if self.produce_entity_occurrence_count {
                let occur = format!(" {}", occur_count);
                self.buf_writer.write_all(occur.as_bytes())?;
            }

            for &v in &vector {
                let vec = format!(" {}", v);
                self.buf_writer.write_all(vec.as_bytes())?;
            }

            Ok(())
        }

        fn finish(&mut self) -> Result<(), io::Error> {
            self.buf_writer.write_all(b"\n")?;
            Ok(())
        }
    }
}
