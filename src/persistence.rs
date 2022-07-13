pub mod entity {
    use dashmap::DashMap;
    use rustc_hash::FxHasher;
    use std::hash::BuildHasherDefault;

    pub trait EntityMappingPersistor {
        fn get_entity(&self, hash: u64) -> Option<String>;
        fn put_data(&self, hash: u64, entity: String);
        fn contains(&self, hash: u64) -> bool;
    }

    #[derive(Debug, Default)]
    pub struct InMemoryEntityMappingPersistor {
        entity_mappings: DashMap<u64, String, BuildHasherDefault<FxHasher>>,
    }

    impl EntityMappingPersistor for InMemoryEntityMappingPersistor {
        fn get_entity(&self, hash: u64) -> Option<String> {
            self.entity_mappings.get(&hash).map(|s| s.to_string())
        }

        fn put_data(&self, hash: u64, entity: String) {
            self.entity_mappings.insert(hash, entity);
        }

        fn contains(&self, hash: u64) -> bool {
            self.entity_mappings.contains_key(&hash)
        }
    }
}

pub mod embedding {
    use crate::persistence::embedding::memmap::OwnedMmapArrayViewMut;
    use ndarray::{s, Array};
    use ndarray_npy::write_zeroed_npy;
    use std::fs::File;
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

    mod memmap {
        use memmap::MmapMut;
        use ndarray::ArrayViewMut2;
        use std::fs::OpenOptions;
        use std::io;
        use std::io::{Error, ErrorKind};
        use std::ptr::drop_in_place;

        pub struct OwnedMmapArrayViewMut {
            mmap_ptr: *mut MmapMut,
            mmap_data: Option<ndarray::ArrayViewMut2<'static, f32>>,
        }

        impl OwnedMmapArrayViewMut {
            pub fn new(filename: &str) -> Result<Self, io::Error> {
                use ndarray_npy::ViewMutNpyExt;

                let file = OpenOptions::new().read(true).write(true).open(filename)?;
                let mmap = unsafe { MmapMut::map_mut(&file)? };
                let mmap = Box::new(mmap);
                let mmap = Box::leak(mmap);
                let mmap_ptr: *mut MmapMut = mmap as *mut _;

                let mmap_data = ArrayViewMut2::<'static, f32>::view_mut_npy(mmap)
                    .map_err(|_| Error::new(ErrorKind::Other, "Mmap view error"))?;

                Ok(Self {
                    mmap_ptr,
                    mmap_data: Some(mmap_data),
                })
            }

            pub fn data_view<'a>(&'a mut self) -> &'a mut ArrayViewMut2<'a, f32> {
                let view = self
                    .mmap_data
                    .as_mut()
                    .expect("Should be always defined. None only used in Drop");

                // SAFETY: shortening lifetime from 'static to 'a is safe because underlying buffer won't be dropped until view is borrowed
                unsafe {
                    core::mem::transmute::<
                        &mut ArrayViewMut2<'static, f32>,
                        &mut ArrayViewMut2<'a, f32>,
                    >(view)
                }
            }
        }

        impl Drop for OwnedMmapArrayViewMut {
            fn drop(&mut self) {
                // Unwind references with reverse order.
                // First remove view that points to mmap_ptr
                self.mmap_data = None;
                // And now drop mmap_ptr
                // SAFETY: safe because pointer leaked in constructor.
                unsafe { drop_in_place(self.mmap_ptr) }
            }
        }
    }

    pub struct NpyPersistor {
        entities: Vec<String>,
        occurences: Vec<u32>,
        array_file_name: String,
        array_file: File,
        array_write_context: Option<OwnedMmapArrayViewMut>,
        occurences_buf: Option<BufWriter<File>>,
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

    impl EmbeddingPersistor for NpyPersistor {
        fn put_metadata(&mut self, entity_count: u32, dimension: u16) -> Result<(), io::Error> {
            write_zeroed_npy::<f32, _>(
                &self.array_file,
                [entity_count as usize, dimension as usize],
            )
            .map_err(|_| Error::new(ErrorKind::Other, "Write zeroed npy error"))?;
            self.array_write_context = Some(OwnedMmapArrayViewMut::new(&self.array_file_name)?);
            Ok(())
        }

        fn put_data(
            &mut self,
            entity: &str,
            occur_count: u32,
            vector: Vec<f32>,
        ) -> Result<(), io::Error> {
            let array = &mut self
                .array_write_context
                .as_mut()
                .expect("Should be defined. Was put_metadata not called?")
                .data_view();

            array
                .slice_mut(s![self.entities.len(), ..])
                .assign(&Array::from(vector));
            self.entities.push(entity.to_owned());
            self.occurences.push(occur_count);
            Ok(())
        }

        fn finish(&mut self) -> Result<(), io::Error> {
            use ndarray_npy::WriteNpyExt;

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
