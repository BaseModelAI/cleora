use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::configuration::{Column, Configuration, FileType};
use crate::embedding::{calculate_embeddings, calculate_embeddings_mmap};
use crate::entity::{EntityProcessor, SMALL_VECTOR_SIZE};
use crate::persistence::embedding::TextFileVectorPersistor;
use crate::persistence::entity::InMemoryEntityMappingPersistor;
use crate::persistence::sparse_matrix::InMemorySparseMatrixPersistor;
use crate::sparse_matrix::{create_sparse_matrices, SparseMatrix};
use bus::Bus;
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
) -> Vec<SparseMatrix<InMemorySparseMatrixPersistor>> {
    let sparse_matrices = create_sparse_matrices(config.embeddings_dimension, &config.columns);
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

    let input_file = File::open(&config.input).expect("Can't open file"); // handle error
    let mut buffered = BufReader::new(input_file);

    let mut line = String::new();
    let mut parser = dom::Parser::default();
    match &config.file_type {
        FileType::JSON => {
            while buffered.read_line(&mut line).unwrap() > 0 {
                let row = read_json_columns(&line, &mut parser, &config.columns);
                entity_processor.process_row(&row);
                line.clear(); // clear to reuse the buffer
            }
        }
        FileType::TSV => {
            while buffered.read_line(&mut line).unwrap() > 0 {
                {
                    let values = line.trim().split('\t');
                    let row: Vec<_> = values.map(|c| c.split(' ').collect()).collect();
                    entity_processor.process_row(&row);
                }
                line.clear(); // clear to reuse the buffer
            }
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

/// Parse a line of JSON and read its columns into a vector for processing.
fn read_json_columns(
    line: &str,
    parser: &mut dom::Parser,
    columns: &[Column],
) -> Vec<SmallVec<[String; SMALL_VECTOR_SIZE]>> {
    let parsed = parser.parse(&line).unwrap();
    columns
        .iter()
        .map({
            |c| {
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
                        .expect("values for complex columns must be arrays")
                        .into_iter()
                        .map(|v| match v.get_type() {
                            dom::element::ElementType::String => v.get_string().unwrap(),
                            _ => v.minify(),
                        })
                        .collect()
                }
            }
        })
        .collect()
}

/// Train SparseMatrix'es (graphs) in separated threads.
pub fn train(
    config: Configuration,
    in_memory_entity_mapping_persistor: Arc<InMemoryEntityMappingPersistor>,
    sparse_matrices: Vec<SparseMatrix<InMemorySparseMatrixPersistor>>,
) {
    let config = Arc::new(config);
    let mut embedding_threads = Vec::new();
    for mut sparse_matrix in sparse_matrices {
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
            let mut text_file_embedding_persistor =
                TextFileVectorPersistor::new(ofp, config.produce_entity_occurrence_count);
            if config.in_memory_embedding_calculation {
                calculate_embeddings(
                    &mut sparse_matrix,
                    config.max_number_of_iteration,
                    in_memory_entity_mapping_persistor,
                    &mut text_file_embedding_persistor,
                );
            } else {
                calculate_embeddings_mmap(
                    &mut sparse_matrix,
                    config.max_number_of_iteration,
                    in_memory_entity_mapping_persistor,
                    &mut text_file_embedding_persistor,
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
