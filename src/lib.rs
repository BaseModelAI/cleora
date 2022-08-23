pub mod configuration;
pub mod embedding;
pub mod entity;
pub mod persistence;
pub mod pipeline;
pub mod sparse_matrix;
pub mod io;
use pyo3::prelude::*;

//pub use configuration;
pub use configuration::Configuration;
pub use configuration::OutputFormat;
use persistence::entity::InMemoryEntityMappingPersistor;
use pipeline::{build_graphs, train};
use std::sync::Arc;

#[pyfunction]
fn run(
    input: Vec<String>,
    type_name: Option<&str>,
    output_dir: Option<String>,
    dimension: u16,
    max_iter: u8,
    seed: Option<i64>,
    prepend_field: bool,
    log_every: u32,
    in_memory_embedding_calculation: bool,
    cols_str: String,
    output_format: &str,
    relation_name: String,
    chunk_size: usize,
) -> PyResult<String> {
    let file_type = match type_name {
        Some(type_name) => match type_name {
            "tsv" => configuration::FileType::Tsv,
            "json" => configuration::FileType::Json,
            _ => panic!("Invalid file type {}", type_name),
        },
        None => configuration::FileType::Tsv,
    };

    let cols_str_separated: Vec<&str> = cols_str.split(' ').collect();
    let columns = match configuration::extract_fields(cols_str_separated) {
        Ok(cols) => match configuration::validate_fields(cols) {
            Ok(validated_cols) => validated_cols,
            Err(msg) => panic!("Invalid column fields. Message: {}", msg),
        },
        Err(msg) => panic!("Parsing problem. Message: {}", msg),
    };

    let output_format_type = match output_format {
        "textfile" => OutputFormat::TextFile,
        "numpy" => OutputFormat::Numpy,
        "parquet" => OutputFormat::Parquet,
        _ => panic!("unsupported output format"),
    };

    let config = Configuration {
        produce_entity_occurrence_count: true,
        embeddings_dimension: dimension,
        max_number_of_iteration: max_iter,
        seed,
        prepend_field,
        log_every_n: log_every,
        in_memory_embedding_calculation,
        input,
        file_type,
        output_dir,
        output_format: output_format_type,
        relation_name,
        columns,
        chunk_size,
    };

    let in_memory_entity_mapping_persistor = InMemoryEntityMappingPersistor::default();
    let in_memory_entity_mapping_persistor = Arc::new(in_memory_entity_mapping_persistor);

    let sparse_matrices = build_graphs(&config, in_memory_entity_mapping_persistor.clone());

    train(config, in_memory_entity_mapping_persistor, sparse_matrices);

    Ok("OK".to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn cleora(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
