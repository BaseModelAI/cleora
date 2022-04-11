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
    use memmap::MmapMut;
    use ndarray::{s, Array, ArrayViewMut1, ArrayViewMut2, Axis};
    use ndarray_npy::write_zeroed_npy;
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::io::{BufWriter, Error, ErrorKind, Write};

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

    pub struct NpyWriteContext<'a> {
        // Pointer needed to move state between put_metadata and put_data
        mmap_ptr: *mut MmapMut,
        mmap_data: ndarray::ArrayViewMut2<'a, f32>,
    }

    pub struct NpyPersistor<'a> {
        entities: Vec<String>,
        occurences: Vec<u32>,
        array_file_name: String,
        array_file: File,
        array_write_context: Option<NpyWriteContext<'a>>,
        occurences_buf: Option<BufWriter<File>>,
        entities_buf: BufWriter<File>,
    }

    impl NpyPersistor<'_> {
        pub fn new<'a>(filename: String, produce_entity_occurrence_count: bool) -> Self {
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

            let array_file_name = format!("{}.npy", &filename);
            let array_file = File::create(&array_file_name)
                .unwrap_or_else(|_| panic!("Unable to create file: {}", &array_file_name));

            Self {
                entities: vec![],
                occurences: vec![],
                array_file_name,
                array_file,
                array_write_context: None,
                occurences_buf,
                entities_buf,
            }
        }
    }

    impl<'a> EmbeddingPersistor for NpyPersistor<'a> {
        fn put_metadata(&mut self, entity_count: u32, dimension: u16) -> Result<(), io::Error> {
            use ndarray_npy::ViewMutNpyExt;
            write_zeroed_npy::<f32, _>(
                &self.array_file,
                [entity_count as usize, dimension as usize],
            );

            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.array_file_name)?;
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };
            let mut mmap = Box::new(mmap);
            let mut mmap = Box::leak(mmap);
            let mmap_ptr: *mut MmapMut = mmap as *mut _;

            let mut mmap_data = ArrayViewMut2::<'a, f32>::view_mut_npy(mmap)
                .map_err(|e| Error::new(ErrorKind::Other, format!("TODO rename")))?;

            self.array_write_context = Some(NpyWriteContext {
                mmap_ptr, // will be used to free memory
                mmap_data,
            });
            Ok(())
        }

        fn put_data(
            &mut self,
            entity: &str,
            occur_count: u32,
            vector: Vec<f32>,
        ) -> Result<(), io::Error> {
            let mut array = &mut self
                .array_write_context
                .as_mut()
                .expect("Should be defined. Was put_metadata not called?")
                .mmap_data;
            array
                .slice_mut(s![self.entities.len(), ..])
                .assign(&Array::from(vector));
            self.entities.push(entity.to_owned());
            self.occurences.push(occur_count);
            Ok(())
        }

        fn finish(&mut self) -> Result<(), io::Error> {
            use ndarray_npy::WriteNpyExt;
            use std::io::{Error, ErrorKind};

            let array_write_context = self
                .array_write_context
                .as_ref()
                .expect("Should be defined. Was put_metadata not called?");
            let recovered_mmap: Box<MmapMut> =
                unsafe { Box::from_raw(array_write_context.mmap_ptr) };
            recovered_mmap.flush()?;

            serde_json::to_writer_pretty(&mut self.entities_buf, &self.entities)?;

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
