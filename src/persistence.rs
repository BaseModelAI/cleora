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
            write!(&mut self.buf_writer, "{} {}", entity_count, dimension)?;
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
                write!(&mut self.buf_writer, " {}", occur_count)?;
            }

            for &v in &vector {
                self.buf_writer.write_all(b" ")?;
                let mut buf = ryu::Buffer::new(); // cheap op
                self.buf_writer.write_all(buf.format_finite(v).as_bytes())?;
            }

            Ok(())
        }

        fn finish(&mut self) -> Result<(), io::Error> {
            self.buf_writer.write_all(b"\n")?;
            Ok(())
        }
    }
}
