use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::configuration::{Column, Configuration, FileType, OutputFormat};
use crate::embedding::{calculate_embeddings, calculate_embeddings_mmap};
use crate::entity::{EntityProcessor, SMALL_VECTOR_SIZE};
use crate::persistence::embedding::{EmbeddingPersistor, NpyPersistor, TextFileVectorPersistor};
use crate::persistence::entity::InMemoryEntityMappingPersistor;
use crate::sparse_matrix::{create_sparse_matrices, SparseMatrix};
use bus::Bus;
use log::{error, info};
use simdjson_rust::dom;
use smallvec::{smallvec, SmallVec};
use std::sync::Arc;
use std::thread;

/// Create SparseMatrix'es based on columns config. Every SparseMatrix operates in separate
/// thread. EntityProcessor reads data in main thread and broadcast cartesian products
/// to SparseMatrix'es.
pub fn build_graphs(
    config: &Configuration,
    in_memory_entity_mapping_persistor: Arc<InMemoryEntityMappingPersistor>,
) -> Vec<SparseMatrix> {
    let sparse_matrices = create_sparse_matrices(&config.columns);
    dbg!(&sparse_matrices);

    let mut bus: Bus<SmallVec<[u64; SMALL_VECTOR_SIZE]>> = Bus::new(128);
    let mut sparse_matrix_threads = Vec::new();
    for mut sparse_matrix in sparse_matrices {
        let rx = bus.add_rx();
        let handle = thread::spawn(move || {
            for received in rx {
                if received[0] != 0 {
                    sparse_matrix.handle_pair(&received);
                } else {
                    sparse_matrix.finish();
                    break;
                }
            }
            sparse_matrix
        });
        sparse_matrix_threads.push(handle);
    }

    let mut entity_processor =
        EntityProcessor::new(&config, in_memory_entity_mapping_persistor, |hashes| {
            bus.broadcast(hashes);
        });

    match &config.file_type {
        FileType::JSON => {
            let mut parser = dom::Parser::default();
            read_file(config, |line| {
                let row = parse_json_line(line, &mut parser, &config.columns);
                entity_processor.process_row(&row);
            });
        }
        FileType::TSV => {
            read_file(config, |line| {
                let row = parse_tsv_line(line);
                entity_processor.process_row(&row);
            });
        }
    }

    entity_processor.finish();

    let mut sparse_matrices = vec![];
    for join_handle in sparse_matrix_threads {
        let sparse_matrix = join_handle
            .join()
            .expect("Couldn't join on the associated thread");
        sparse_matrices.push(sparse_matrix);
    }

    sparse_matrices
}

/// Read file line by line. Pass every valid line to handler for parsing.
fn read_file<F>(config: &Configuration, mut line_handler: F)
where
    F: FnMut(&str),
{
    let input_file = File::open(&config.input).expect("Can't open file");
    let mut buffered = BufReader::new(input_file);

    let mut line_number = 1u64;
    let mut line = String::new();
    loop {
        match buffered.read_line(&mut line) {
            Ok(bytes_read) => {
                // EOF
                if bytes_read == 0 {
                    break;
                }

                line_handler(&line);
            }
            Err(err) => {
                error!("Can't read line number: {}. Error: {}.", line_number, err);
            }
        };

        // clear to reuse the buffer
        line.clear();

        if line_number % config.log_every_n as u64 == 0 {
            info!("Number of lines processed: {}", line_number);
        }

        line_number += 1;
    }
}

/// Parse a line of JSON and read its columns into a vector for processing.
fn parse_json_line(
    line: &str,
    parser: &mut dom::Parser,
    columns: &[Column],
) -> Vec<SmallVec<[String; SMALL_VECTOR_SIZE]>> {
    let parsed = parser.parse(&line).unwrap();
    columns
        .iter()
        .map(|c| {
            if !c.complex {
                let elem = parsed.at_key(&c.name).unwrap();
                let value = match elem.get_type() {
                    dom::element::ElementType::String => elem.get_string().unwrap(),
                    _ => elem.minify(),
                };
                smallvec![value]
            } else {
                parsed
                    .at_key(&c.name)
                    .unwrap()
                    .get_array()
                    .expect("Values for complex columns must be arrays")
                    .into_iter()
                    .map(|v| match v.get_type() {
                        dom::element::ElementType::String => v.get_string().unwrap(),
                        _ => v.minify(),
                    })
                    .collect()
            }
        })
        .collect()
}

/// Parse a line of TSV and read its columns into a vector for processing.
fn parse_tsv_line(line: &str) -> Vec<SmallVec<[&str; SMALL_VECTOR_SIZE]>> {
    let values = line.trim().split('\t');
    values.map(|c| c.split(' ').collect()).collect()
}

/// Train SparseMatrix'es (graphs) in separated threads.
pub fn train(
    config: Configuration,
    in_memory_entity_mapping_persistor: Arc<InMemoryEntityMappingPersistor>,
    sparse_matrices: Vec<SparseMatrix>,
) {
    let config = Arc::new(config);
    let mut embedding_threads = Vec::new();
    for sparse_matrix in sparse_matrices {
        let sparse_matrix = Arc::new(sparse_matrix);
        let config = config.clone();
        let in_memory_entity_mapping_persistor = in_memory_entity_mapping_persistor.clone();
        let handle = thread::spawn(move || {
            let directory = match config.output_dir.as_ref() {
                Some(out) => format!("{}/", out.clone()),
                None => String::from(""),
            };
            let ofp = format!(
                "{}{}__{}__{}.out",
                directory,
                config.relation_name,
                sparse_matrix.col_a_name.as_str(),
                sparse_matrix.col_b_name.as_str()
            );

            let mut persistor: Box<dyn EmbeddingPersistor> = match &config.output_format {
                OutputFormat::TextFile => Box::new(TextFileVectorPersistor::new(
                    ofp,
                    config.produce_entity_occurrence_count,
                )),
                OutputFormat::Numpy => Box::new(NpyPersistor::new(
                    ofp,
                    config.produce_entity_occurrence_count,
                )),
            };
            if config.in_memory_embedding_calculation {
                calculate_embeddings(
                    config.clone(),
                    sparse_matrix.clone(),
                    in_memory_entity_mapping_persistor,
                    persistor.as_mut(),
                );
            } else {
                calculate_embeddings_mmap(
                    config.clone(),
                    sparse_matrix.clone(),
                    in_memory_entity_mapping_persistor,
                    persistor.as_mut(),
                );
            }
        });
        embedding_threads.push(handle);
    }

    for join_handle in embedding_threads {
        let _ = join_handle
            .join()
            .expect("Couldn't join on the associated thread");
    }
}
