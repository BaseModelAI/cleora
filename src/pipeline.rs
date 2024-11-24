use std::cmp::min;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::time::Instant;

use crossbeam::channel;
use crossbeam::channel::{Receiver, Sender};
use crossbeam::thread as cb_thread;
use crossbeam::thread::{Scope, ScopedJoinHandle};
use itertools::Itertools;
use log::{error, info, warn};
use smallvec::SmallVec;

use crate::configuration::Configuration;
use crate::entity::{EntityProcessor, Hyperedge, SMALL_VECTOR_SIZE};
use crate::sparse_matrix::SparseMatrix;
use crate::sparse_matrix_builder::NodeIndexerBuilder;
use crate::sparse_matrix_builder::{
    AsyncNodeIndexerBuilder, NodeIndexer, SparseMatrixBuffer, SparseMatrixBuffersReducer,
    SyncNodeIndexerBuilder,
};

pub fn build_graph_from_iterator<'a>(
    config: &Configuration,
    hyperedges: impl Iterator<Item = &'a str>,
) -> SparseMatrix {
    cb_thread::scope(|s| {
        let (hyperedges_s, hyperedges_r) = channel::bounded(64 * config.num_workers_graph_building);

        // Consumer first, producer second to avoid deadlock
        let matrix_buffer = make_consumer(hyperedges_r, config, s);
        let node_indexer = make_producer_from_iterator(config, hyperedges, hyperedges_s);

        let buffers = matrix_buffer
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect_vec();
        SparseMatrixBuffersReducer::new(node_indexer, buffers, config.num_workers_graph_building)
            .reduce()
    })
    .expect("All work in thread scope finished")
}

fn make_producer_from_iterator<'a>(
    config: &Configuration,
    hyperedges: impl Iterator<Item = &'a str>,
    hyperedges_s: Sender<Hyperedge>,
) -> NodeIndexer {
    let node_indexer_builder: Arc<SyncNodeIndexerBuilder> = Default::default();
    let entity_processor = EntityProcessor::new(config, node_indexer_builder.clone());
    for line in hyperedges {
        consume_line(config, &hyperedges_s, &entity_processor, line);
    }
    drop(entity_processor);
    let node_indexer_builder =
        Arc::try_unwrap(node_indexer_builder).expect("All other references should be dropped");
    node_indexer_builder.finish()
}

fn consume_line<S: NodeIndexerBuilder>(
    config: &Configuration,
    hyperedges_s: &Sender<Hyperedge>,
    entity_processor: &EntityProcessor<S>,
    line: &str,
) {
    let row = parse_tsv_line(line);
    let line_col_num = row.len();
    if line_col_num == config.columns.len() {
        let hyperedge = entity_processor.process_row_and_get_edges(&row);
        hyperedges_s.send(hyperedge).unwrap();
    } else {
        warn!(
            "Wrong number of columns (expected: {}, provided: {}). The line [{}] is skipped.",
            config.columns.len(),
            line_col_num,
            line
        );
    }
}

pub fn build_graph_from_files(config: &Configuration, input_files: Vec<String>) -> SparseMatrix {
    let processing_worker_num = config.num_workers_graph_building;
    cb_thread::scope(|s| {
        let (hyperedges_s, hyperedges_r) = channel::bounded(processing_worker_num * 64);

        // Consumer first, producer second to avoid deadlock
        let matrix_buffers: Vec<_> = make_consumer(hyperedges_r, config, s);
        let node_indexer = make_producer_from_files(config, &input_files, s, hyperedges_s);

        let buffers = matrix_buffers
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect_vec();

        let merging_start_time = Instant::now();
        let result =
            SparseMatrixBuffersReducer::new(node_indexer, buffers, processing_worker_num).reduce();
        info!(
            "Merging finished in {} sec",
            merging_start_time.elapsed().as_secs()
        );
        result
    })
    .expect("Threads finished work")
}

fn make_producer_from_files<'c: 'e, 'e: 's, 's>(
    config: &'c Configuration,
    input_files: &'c Vec<String>,
    s: &'s Scope<'e>,
    hyperedges_s: Sender<Hyperedge>,
) -> NodeIndexer {
    let (files_s, files_r) = channel::unbounded();

    for input in input_files {
        files_s.send(input).unwrap()
    }
    drop(files_s);

    let max_file_reading_worker_num = min(config.num_workers_graph_building, 4);
    let file_reading_worker_num = min(max_file_reading_worker_num, input_files.len());

    let log_every_n = 10000;

    if file_reading_worker_num == 1 {
        let node_indexer_builder: Arc<SyncNodeIndexerBuilder> = Default::default();
        let entity_processor = EntityProcessor::new(config, node_indexer_builder.clone());
        consume_files(config, hyperedges_s, files_r, log_every_n, entity_processor);
        let node_indexer_builder =
            Arc::try_unwrap(node_indexer_builder).expect("All other references should be dropped");
        node_indexer_builder.finish()
    } else {
        let node_indexer_builder: Arc<AsyncNodeIndexerBuilder> = Default::default();
        let producers = (0..file_reading_worker_num)
            .map(|_| {
                let hyperedges_s = hyperedges_s.clone();
                let files_r = files_r.clone();
                let entity_processor = EntityProcessor::new(config, node_indexer_builder.clone());

                s.spawn(move |_| {
                    consume_files(config, hyperedges_s, files_r, log_every_n, entity_processor);
                })
            })
            .collect_vec();
        drop(hyperedges_s); // hyperedges_s got distributed among producers, drop seed object
        drop(files_r);

        producers.into_iter().for_each(|h| h.join().unwrap());
        let node_indexer_builder =
            Arc::try_unwrap(node_indexer_builder).expect("All other references should be dropped");
        node_indexer_builder.finish()
    }
}

fn consume_files<S: NodeIndexerBuilder>(
    config: &Configuration,
    hyperedges_s: Sender<Hyperedge>,
    files_r: Receiver<&String>,
    log_every_n: u64,
    entity_processor: EntityProcessor<S>,
) {
    for input in files_r {
        read_file(input, log_every_n, |line| {
            consume_line(config, &hyperedges_s, &entity_processor, line);
        });
    }
}

fn make_consumer<'s, 'a: 'a>(
    hyperedges_r: Receiver<Hyperedge>,
    config: &'a Configuration,
    s: &'s Scope<'a>,
) -> Vec<ScopedJoinHandle<'s, SparseMatrixBuffer>> {
    (0..config.num_workers_graph_building)
        .map(|_| {
            let hyperedges_r = hyperedges_r.clone();
            let sparse_matrices = config.matrix_desc.clone();

            s.spawn(move |_| {
                let mut buffer = sparse_matrices.make_buffer(config.hyperedge_trim_n);
                for hyperedge in hyperedges_r {
                    buffer.handle_hyperedge(&hyperedge);
                }
                buffer
            })
        })
        .collect()
}

/// Read file line by line. Pass every valid line to handler for parsing.
fn read_file<F>(filepath: &str, log_every: u64, mut line_handler: F)
where
    F: FnMut(&str),
{
    let input_file = File::open(filepath).expect("Can't open file");
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

        if line_number % log_every == 0 {
            info!("Number of lines processed: {}", line_number);
        }

        line_number += 1;
    }
}

/// Parse a line of TSV and read its columns into a vector for processing.
fn parse_tsv_line(line: &str) -> Vec<SmallVec<[&str; SMALL_VECTOR_SIZE]>> {
    let values = line.trim().split('\t');
    values.map(|c| c.split(' ').collect()).collect()
}
