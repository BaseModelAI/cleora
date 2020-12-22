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

    pub struct NpyPersistor {
        entities: Vec<String>,
        occurences: Vec<u32>,
        array: Option<ndarray::Array2<f32>>,
        occurences_buf: Option<BufWriter<File>>,
        array_buf: BufWriter<File>,
        entities_buf: BufWriter<File>,
    }

    impl NpyPersistor {
        pub fn new(filename: String, produce_entity_occurrence_count: bool) -> Self {
            let entities_filename = format!("{}.entities", &filename);
            let entities_buf = BufWriter::new(
                File::create(&entities_filename)
                    .unwrap_or_else(|_| panic!("Unable to create file: {}", &entities_filename)),
            );

            let occurences_filename = format!("{}.occurences", &filename);
            let occurences_buf = if produce_entity_occurrence_count {
                Some(BufWriter::new(
                    File::create(&occurences_filename).unwrap_or_else(|_| {
                        panic!("Unable to create file: {}", &occurences_filename)
                    }),
                ))
            } else {
                None
            };

            let array_filename = format!("{}.npy", &filename);
            let array_buf = BufWriter::new(
                File::create(&array_filename)
                    .unwrap_or_else(|_| panic!("Unable to create file: {}", &array_filename)),
            );

            Self {
                entities: vec![],
                occurences: vec![],
                array: None,
                occurences_buf,
                array_buf,
                entities_buf,
            }
        }
    }

    impl EmbeddingPersistor for NpyPersistor {
        fn put_metadata(&mut self, entity_count: u32, dimension: u16) -> Result<(), io::Error> {
            self.array = Some(ndarray::Array2::zeros((
                entity_count as usize,
                dimension as usize,
            )));
            Ok(())
        }

        fn put_data(
            &mut self,
            entity: &str,
            occur_count: u32,
            vector: Vec<f32>,
        ) -> Result<(), io::Error> {
            let array = self.array.as_mut().unwrap();
            array
                .row_mut(self.entities.len())
                .assign(&ndarray::ArrayView1::from(vector.as_slice()));
            self.entities.push(entity.to_owned());
            self.occurences.push(occur_count);

            Ok(())
        }

        fn finish(&mut self) -> Result<(), io::Error> {
            use ndarray::s;
            use ndarray_npy::WriteNpyExt;
            use std::io::{Error, ErrorKind};

            serde_json::to_writer_pretty(&mut self.entities_buf, &self.entities)?;

            let array = self.array.as_ref().expect("Called before put_metadata");
            // FIXME: This is workaround a bug that caused invalid entites_count being passed via
            // put_metadata call
            let array = array.slice(s![0..self.entities.len(), ..]);

            array.write_npy(&mut self.array_buf).map_err(|e| match e {
                ndarray_npy::WriteNpyError::Io(err) => err,
                other => Error::new(
                    ErrorKind::Other,
                    format!("Could not save embedding array: {}", other),
                ),
            })?;

            if let Some(occurences_buf) = self.occurences_buf.as_mut() {
                let occur = ndarray::ArrayView1::from(&self.occurences);
                occur.write_npy(occurences_buf).map_err(|e| {
                    Error::new(
                        ErrorKind::Other,
                        format!("Could not save occurences: {}", e),
                    )
                })?;
            }

            Ok(())
        }
    }
}
