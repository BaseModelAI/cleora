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
    use crate::persistence::embedding::memmap::OwnedMmapArrayViewMut;
    use ndarray::{s, Array};
    use ndarray_npy::write_zeroed_npy;
    use std::fs::File;
    use std::io;
    use std::io::{BufWriter, Error, ErrorKind, Write};

    use arrow::array::Array as ArrowArray;
    use arrow::array::{ArrayRef, Float32Array, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::collections::HashMap;
    use std::sync::Arc;

    pub trait EmbeddingPersistor {
        fn put_metadata(&mut self, entity_count: u32, dimension: u16) -> Result<(), io::Error>;
        fn put_data(
            &mut self,
            entity: &str,
            occur_count: u32,
            vector: Vec<f32>,
        ) -> Result<(), io::Error>;
        fn finish(&mut self) -> Result<(), io::Error>;
        fn close(self) -> Result<(), String>; 
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

        fn close(self) -> Result<(), String> {
            Ok(())
        }

    }

    pub struct ParquetVectorPersistor {
        filename: String,
        buf_writer: ArrowWriter<File>,
        produce_entity_occurrence_count: bool,
    }

    impl ParquetVectorPersistor {
        pub fn new(
            filename: String,
            dimension: u16,
            produce_entity_occurrence_count: bool,
        ) -> Self {
            let file = File::create("data.parquet").unwrap();

            // Default writer properties
            let props = WriterProperties::builder().build();

            //let mut metadata = HashMap::new();
            //metadata.insert("entity_count".to_string(), entity_count.to_string());
            //metadata.insert("dimension".to_string(), dimension.to_string());

            let mut fields: Vec<Field> = (0..dimension)
                .into_iter()
                .map(|x| Field::new(format!("f{}", x).as_str(), DataType::Float32, false))
                .collect();
            fields.push(Field::new("entity", DataType::Utf8, false));
            fields.push(Field::new("occur_count", DataType::UInt32, false));

            let schema = Schema::new(fields); //.with_metadata(metadata);

            let writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props)).unwrap();

            ParquetVectorPersistor {
                filename,
                buf_writer: writer,
                produce_entity_occurrence_count,
            }
        }
    }

    impl EmbeddingPersistor for ParquetVectorPersistor {
        fn put_metadata(&mut self, entity_count: u32, dimension: u16) -> Result<(), io::Error> {
            Ok(())
        }

        fn put_data(
            &mut self,
            entity: &str,
            occur_count: u32,
            vector: Vec<f32>,
        ) -> Result<(), io::Error> {
            let mut fields: Vec<(String, Arc<dyn ArrowArray>)> = (0..vector.len())
                .into_iter()
                .map(|x| {
                    (
                        format!("f{}", x),
                        Arc::new(Float32Array::from(vec![vector[x]])) as ArrayRef,
                    )
                })
                .collect();

            let e: ArrayRef = Arc::new(StringArray::from(vec![entity]));
            fields.push(("entity".to_string(), e));

            let e: ArrayRef = Arc::new(UInt32Array::from(vec![occur_count]));
            fields.push(("occur_count".to_string(), e));

            let batch = RecordBatch::try_from_iter(fields).unwrap();

            //println!("{:?}", batch.schema());

            self.buf_writer.write(&batch).expect("Writing batch");
            self.buf_writer.flush().expect("Flush");

            Ok(())
        }

        fn finish(&mut self) -> Result<(), io::Error> {
            Ok(())
        }

        fn close(self) -> Result<(), String> {
            self.buf_writer.close().unwrap();
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

        fn close(self) -> Result<(), String> {
            Ok(())
        }

    }
}
