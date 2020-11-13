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

    impl InMemoryEntityMappingPersistor {
        pub fn new() -> Self {
            Self::default()
        }
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
    use std::mem;

    #[derive(Debug, Clone)]
    pub struct Entry {
        pub row: u32,
        pub col: u32,
        pub value: f32,
    }

    pub trait SparseMatrixPersistor {
        fn increment_hash_occurrence(&mut self, hash: u64) -> u32;
        fn get_hash_occurrence(&self, hash: u64) -> u32;
        fn get_id(&self, hash: u64) -> i32;
        fn get_hash(&self, id: u32) -> i64;
        fn get_entity_counter(&self) -> u32;
        fn update_entity_counter(&mut self, val: u32);
        fn add_hash_id(&mut self, hash: u64, id: u32);
        fn increment_edge_counter(&mut self) -> u32;
        fn get_amount_of_data(&self) -> u32;
        fn get_pair_index(&self, magic: u64) -> i32;
        fn add_pair_index(&mut self, magic: u64, pos: u32);
        fn update_row_sum(&mut self, id: u32, val: f32);
        fn get_row_sum(&self, id: u32) -> f32;
        fn add_new_entry(&mut self, pos: u32, entry: Entry);
        fn update_entry(&mut self, pos: u32, entry: Entry);
        fn get_entry(&self, pos: u32) -> Entry;
        fn replace_entry(&mut self, pos: u32, entry: Entry);
        fn finish(&self);
    }

    #[derive(Debug, Default)]
    pub struct InMemorySparseMatrixPersistor {
        entity_count: u32,
        edge_count: u32,
        hash_2_id: FxHashMap<u64, u32>,
        id_2_hash: FxHashMap<u32, u64>,
        hash_2_count: FxHashMap<u64, u32>,
        row_sum: Vec<f32>,
        pair_index: FxHashMap<u64, u32>,
        entries: Vec<Entry>,
    }

    impl InMemorySparseMatrixPersistor {
        pub fn new() -> Self {
            Self::default()
        }
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
            match self.id_2_hash.get(&id) {
                Some(value) => *value as i64,
                None => -1i64,
            }
        }

        fn get_entity_counter(&self) -> u32 {
            self.entity_count
        }

        fn update_entity_counter(&mut self, val: u32) {
            self.entity_count = val;
        }

        fn add_hash_id(&mut self, hash: u64, id: u32) {
            self.hash_2_id.insert(hash, id);
            self.id_2_hash.insert(id, hash);
        }

        fn increment_edge_counter(&mut self) -> u32 {
            self.edge_count += 1;
            self.edge_count
        }

        fn get_amount_of_data(&self) -> u32 {
            self.entries.len() as u32
        }

        fn get_pair_index(&self, magic: u64) -> i32 {
            match self.pair_index.get(&magic) {
                Some(value) => *value as i32,
                None => -1i32,
            }
        }

        fn add_pair_index(&mut self, magic: u64, pos: u32) {
            self.pair_index.insert(magic, pos);
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
            let entry = &self.entries[pos as usize];
            entry.clone()
        }

        fn replace_entry(&mut self, pos: u32, entry: Entry) {
            self.entries[pos as usize] = entry
        }

        fn finish(&self) {
            info!("Number of entities: {}", self.entity_count);
            info!("Number of edges: {}", self.edge_count);
            info!("Number of entries: {}", self.entries.len());

            let hash_2_id_mem_size = self.hash_2_id.capacity() * 12;
            let id_2_hash_mem_size = self.id_2_hash.capacity() * 12;
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
            let total_mem_size = total_mem_size + 8;

            info!(
                "Total memory usage by the struct ~ {} MB",
                (total_mem_size / 1048576)
            );
        }
    }
}

pub mod embedding {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    pub trait EmbeddingPersistor {
        fn put_metadata(&mut self, entity_count: u32, dimension: u16);
        fn put_data(&mut self, entity: String, occur_count: u32, vector: Vec<f32>);
        fn finish(&mut self);
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
        fn put_metadata(&mut self, entity_count: u32, dimension: u16) {
            let metadata = format!("{} {}", entity_count, dimension);
            self.buf_writer.write(metadata.as_bytes()).ok();
        }

        fn put_data(&mut self, entity: String, occur_count: u32, vector: Vec<f32>) {
            self.buf_writer.write(b"\n").ok();
            self.buf_writer.write(entity.as_bytes()).ok();

            if self.produce_entity_occurrence_count {
                let occur = format!(" {}", occur_count);
                self.buf_writer.write(occur.as_bytes()).ok();
            }

            for &v in &vector {
                let vec = format!(" {}", v);
                self.buf_writer.write(vec.as_bytes()).ok();
            }
        }

        fn finish(&mut self) {
            self.buf_writer.write(b"\n").ok();
        }
    }
}
