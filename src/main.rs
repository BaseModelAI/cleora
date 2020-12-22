use std::time::Instant;

use clap::{crate_authors, crate_description, crate_name, crate_version, App, Arg};
use cleora::configuration;
use cleora::configuration::Configuration;
use cleora::configuration::OutputFormat;
use cleora::persistence::entity::InMemoryEntityMappingPersistor;
use cleora::pipeline::{build_graphs, train};
use env_logger::Env;
use std::fs;
use std::sync::Arc;

#[macro_use]
extern crate log;

fn main() {
    let env = Env::default()
        .filter_or("MY_LOG_LEVEL", "info")
        .write_style_or("MY_LOG_STYLE", "always");
    env_logger::init_from_env(env);

    let now = Instant::now();

    let matches = App::new(crate_name!())
        .version(crate_version!())
        .author(crate_authors!())
        .about(crate_description!())
        .arg(
            Arg::with_name("input")
                .short("i")
                .long("input")
                .required(true)
                .help("Input file path")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("file-type")
                .short("t")
                .long("type")
                .possible_values(&["tsv", "json"])
                .help("Input file type")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output-dir")
                .short("o")
                .long("output-dir")
                .help("Output directory for files with embeddings")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("dimension")
                .short("d")
                .long("dimension")
                .required(true)
                .help("Embedding dimension size")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("number-of-iterations")
                .short("n")
                .long("number-of-iterations")
                .required(true)
                .help("Max number of iterations")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("columns")
                .short("c")
                .long("columns")
                .required(true)
                .help(
                    "Column names (max 12), with modifiers: [transient::, reflexive::, complex::]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("relation-name")
                .short("r")
                .long("relation-name")
                .default_value("emb")
                .help("Name of the relation, for output filename generation")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("prepend-field-name")
                .short("p")
                .long("prepend-field-name")
                .possible_values(&["0", "1"])
                .default_value("0")
                .help("Prepend field name to entity in output")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("log-every-n")
                .short("l")
                .long("log-every-n")
                .default_value("10000")
                .help("Log output every N lines")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("in-memory-embedding-calculation")
                .short("e")
                .long("in-memory-embedding-calculation")
                .possible_values(&["0", "1"])
                .default_value("1")
                .help("Calculate embeddings in memory or with memory-mapped files")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output-format")
                .short("f")
                .help("Output format. One of: textfile|numpy")
                .possible_values(&["textfile", "numpy"])
                .default_value("textfile")
                .takes_value(true),
        )
        .get_matches();

    info!("Reading args...");

    let input = matches.value_of("input").unwrap();
    let file_type = match matches.value_of("file-type") {
        Some(type_name) => match type_name {
            "tsv" => configuration::FileType::TSV,
            "json" => configuration::FileType::JSON,
            _ => panic!("Invalid file type {}", type_name),
        },
        None => configuration::FileType::TSV,
    };
    let output_dir = matches.value_of("output-dir").map(|s| s.to_string());
    // try to create output directory for files with embeddings
    if let Some(output_dir) = output_dir.as_ref() {
        fs::create_dir_all(output_dir).expect("Can't create output directory");
    }
    let dimension: u16 = matches.value_of("dimension").unwrap().parse().unwrap();
    let max_iter: u8 = matches
        .value_of("number-of-iterations")
        .unwrap()
        .parse()
        .unwrap();
    let relation_name = matches.value_of("relation-name").unwrap();
    let prepend_field_name = {
        let value: u8 = matches
            .value_of("prepend-field-name")
            .unwrap()
            .parse()
            .unwrap();
        value == 1
    };
    let log_every: u32 = matches.value_of("log-every-n").unwrap().parse().unwrap();
    let in_memory_embedding_calculation = {
        let value: u8 = matches
            .value_of("in-memory-embedding-calculation")
            .unwrap()
            .parse()
            .unwrap();
        value == 1
    };
    let columns = {
        let cols_str = matches.value_of("columns").unwrap();
        let cols_str_separated: Vec<&str> = cols_str.split(' ').collect();
        match configuration::extract_fields(cols_str_separated) {
            Ok(cols) => match configuration::validate_fields(cols) {
                Ok(validated_cols) => validated_cols,
                Err(msg) => panic!("Invalid column fields. Message: {}", msg),
            },
            Err(msg) => panic!("Parsing problem. Message: {}", msg),
        }
    };

    let output_format = match matches.value_of("output-format").unwrap() {
        "textfile" => OutputFormat::TextFile,
        "numpy" => OutputFormat::Numpy,
        _ => panic!("unsupported output format"),
    };

    let config = Configuration {
        produce_entity_occurrence_count: true,
        embeddings_dimension: dimension,
        max_number_of_iteration: max_iter,
        prepend_field: prepend_field_name,
        log_every_n: log_every,
        in_memory_embedding_calculation,
        input: input.to_string(),
        file_type,
        output_dir,
        output_format,
        relation_name: relation_name.to_string(),
        columns,
    };
    dbg!(&config);

    info!("Starting calculation...");
    let in_memory_entity_mapping_persistor = InMemoryEntityMappingPersistor::default();
    let in_memory_entity_mapping_persistor = Arc::new(in_memory_entity_mapping_persistor);

    let sparse_matrices = build_graphs(&config, in_memory_entity_mapping_persistor.clone());
    info!(
        "Finished Sparse Matrices calculation in {} sec",
        now.elapsed().as_secs()
    );

    train(config, in_memory_entity_mapping_persistor, sparse_matrices);
    info!("Finished in {} sec", now.elapsed().as_secs());
}
