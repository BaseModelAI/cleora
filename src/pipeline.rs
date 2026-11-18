//! Drivers that feed hyperedges from the supported sources into the builder.

use std::fs::File;
use std::io::{BufRead, BufReader};

use smallvec::{smallvec, SmallVec};

use crate::configuration::Configuration;
use crate::embedding::thread_pool;
use crate::entity::{parse_line, weight_is_valid, IntBuffer, LineError, Tokens, MAX_WEIGHT};
use crate::sparse_matrix::SparseMatrix;
use crate::sparse_matrix_builder::{
    position, reduce, BuildStats, Indexer, WorkerBuffer, MAX_FILES, MAX_LINES,
};

/// Runs `work(first, last, buffer)` on `num_workers` threads over `0..len`, split
/// into contiguous ranges, and returns the per-thread buffers.
fn run_partitioned<F>(len: usize, num_workers: usize, work: F) -> Vec<WorkerBuffer>
where
    F: Fn(usize, usize, &mut WorkerBuffer) + Sync,
{
    let workers = num_workers.clamp(1, len.max(1));
    if workers == 1 {
        let mut buffer = WorkerBuffer::default();
        work(0, len, &mut buffer);
        return vec![buffer];
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let work = &work;
                let first = len * worker / workers;
                let last = len * (worker + 1) / workers;
                scope.spawn(move || {
                    let mut buffer = WorkerBuffer::default();
                    work(first, last, &mut buffer);
                    buffer
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("worker thread panicked"))
            .collect()
    })
}

fn finish(
    config: &Configuration,
    indexer: Indexer,
    buffers: Vec<WorkerBuffer>,
) -> Result<(SparseMatrix, BuildStats), String> {
    thread_pool(config.num_workers).install(|| reduce(config, indexer, buffers))
}

fn check_weights(weights: Option<&[f32]>, expected_len: usize) -> Result<(), String> {
    if let Some(weights) = weights {
        if weights.len() != expected_len {
            return Err(format!(
                "weights has {} element(s) but there are {} hyperedge(s)",
                weights.len(),
                expected_len
            ));
        }
        if let Some(index) = weights.iter().position(|w| !weight_is_valid(*w)) {
            return Err(format!(
                "weights[{}] = {} is not a finite number between 0 and {:e}",
                index, weights[index], MAX_WEIGHT
            ));
        }
    }
    Ok(())
}

/// Builds the matrix from lines held in memory. `weights`, when given, holds one
/// weight per line (and cannot be combined with a `weight::` column).
pub fn build_graph_from_lines<S: AsRef<str> + Sync>(
    config: &Configuration,
    lines: &[S],
    weights: Option<&[f32]>,
) -> Result<(SparseMatrix, BuildStats), String> {
    check_weights(weights, lines.len())?;
    if weights.is_some() && config.weight_column.is_some() {
        return Err(
            "Pass weights either as an argument or as a 'weight::' column, not both".to_string(),
        );
    }
    if lines.len() as u64 >= MAX_LINES {
        return Err(format!("Too many lines ({})", lines.len()));
    }

    let indexer = Indexer::default();
    let buffers = run_partitioned(lines.len(), config.num_workers, |first, last, buffer| {
        for index in first..last {
            let line = lines[index].as_ref();
            let pos = position(0, index as u64, 0);
            match parse_line(line, config) {
                Ok(Some(parsed)) => {
                    let weight = match weights {
                        Some(weights) => weights[index],
                        None => parsed.weight,
                    };
                    buffer.add_row(config, &indexer, &parsed.fields, weight, pos);
                }
                Ok(None) => {}
                Err(error) => buffer.skip_line(pos, line, error),
            }
        }
    });
    finish(config, indexer, buffers)
}

/// A chunk of a file: whole lines, newline-terminated.
struct Batch {
    file: u32,
    first_line: u64,
    data: Vec<u8>,
}

const BATCH_BYTES: usize = 1 << 20;

fn read_batches(
    file_index: u32,
    path: &str,
    sender: &crossbeam_channel::Sender<Batch>,
) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("Cannot open file '{}': {}", path, e))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut line_number = 0u64;
    loop {
        let mut data = Vec::with_capacity(BATCH_BYTES + 4096);
        let first_line = line_number;
        while data.len() < BATCH_BYTES {
            let read = reader
                .read_until(b'\n', &mut data)
                .map_err(|e| format!("Cannot read file '{}': {}", path, e))?;
            if read == 0 {
                break;
            }
            line_number += 1;
        }
        if data.is_empty() {
            return Ok(());
        }
        if line_number >= MAX_LINES {
            return Err(format!("File '{}' has too many lines", path));
        }
        if sender
            .send(Batch {
                file: file_index,
                first_line,
                data,
            })
            .is_err()
        {
            // the workers are gone; their error is reported instead
            return Ok(());
        }
    }
}

/// Builds the matrix from text files (one hyperedge per line).
pub fn build_graph_from_files(
    config: &Configuration,
    paths: &[String],
) -> Result<(SparseMatrix, BuildStats), String> {
    if paths.is_empty() {
        return Err("At least one file path is required".to_string());
    }
    if paths.len() > MAX_FILES {
        return Err(format!(
            "Too many files ({}); at most {} are supported",
            paths.len(),
            MAX_FILES
        ));
    }
    // Fail early, before any work, if a file cannot be opened.
    for path in paths {
        File::open(path).map_err(|e| format!("Cannot open file '{}': {}", path, e))?;
    }

    let indexer = Indexer::default();
    let workers = config.num_workers.max(1);
    let readers = workers.min(4).min(paths.len());
    let (batch_sender, batch_receiver) = crossbeam_channel::bounded::<Batch>(2 * workers);
    let (path_sender, path_receiver) = crossbeam_channel::unbounded::<(u32, &String)>();
    for (index, path) in paths.iter().enumerate() {
        path_sender.send((index as u32, path)).unwrap();
    }
    drop(path_sender);

    let (buffers, read_result) = std::thread::scope(|scope| {
        let worker_handles: Vec<_> = (0..workers)
            .map(|_| {
                let receiver = batch_receiver.clone();
                let indexer = &indexer;
                scope.spawn(move || {
                    let mut buffer = WorkerBuffer::default();
                    for batch in receiver {
                        process_batch(config, indexer, &batch, &mut buffer);
                    }
                    buffer
                })
            })
            .collect();
        drop(batch_receiver);

        let reader_handles: Vec<_> = (0..readers)
            .map(|_| {
                let sender = batch_sender.clone();
                let paths = path_receiver.clone();
                scope.spawn(move || -> Result<(), String> {
                    for (index, path) in paths {
                        read_batches(index, path, &sender)?;
                    }
                    Ok(())
                })
            })
            .collect();
        drop(batch_sender);
        drop(path_receiver);

        let mut read_result = Ok(());
        for handle in reader_handles {
            let result = handle.join().expect("reader thread panicked");
            if read_result.is_ok() {
                read_result = result;
            }
        }
        let buffers: Vec<WorkerBuffer> = worker_handles
            .into_iter()
            .map(|handle| handle.join().expect("worker thread panicked"))
            .collect();
        (buffers, read_result)
    });
    read_result?;
    finish(config, indexer, buffers)
}

fn process_batch(
    config: &Configuration,
    indexer: &Indexer,
    batch: &Batch,
    buffer: &mut WorkerBuffer,
) {
    let mut line_number = batch.first_line;
    for raw in batch.data.split_inclusive(|b| *b == b'\n') {
        let pos = position(batch.file, line_number, 0);
        line_number += 1;
        match std::str::from_utf8(raw) {
            Ok(line) => match parse_line(line, config) {
                Ok(Some(parsed)) => {
                    buffer.add_row(config, indexer, &parsed.fields, parsed.weight, pos)
                }
                Ok(None) => {}
                Err(error) => buffer.skip_line(pos, line.trim_end(), error),
            },
            Err(_) => buffer.skip_line(
                pos,
                String::from_utf8_lossy(raw).trim_end(),
                LineError::InvalidUtf8,
            ),
        }
    }
}

/// How the two endpoints of an edge map onto the declared columns.
#[derive(Clone, Copy)]
enum EdgeLayout {
    /// One reflexive column: the edge is the two-entity hyperedge `{source, target}`.
    SingleColumn,
    /// Two entity columns: `source` in the first, `target` in the second.
    TwoColumns,
}

fn edge_layout(config: &Configuration) -> Result<EdgeLayout, String> {
    if config.weight_column.is_some() {
        return Err(
            "Edge arrays take their weights from the 'weights' argument, not from a 'weight::' column"
                .to_string(),
        );
    }
    match config.entity_column_names.len() {
        1 => Ok(EdgeLayout::SingleColumn),
        2 => Ok(EdgeLayout::TwoColumns),
        n => Err(format!(
            "Edge arrays need a columns definition with one reflexive column or two columns, got {} columns",
            n
        )),
    }
}

fn add_edge(
    config: &Configuration,
    indexer: &Indexer,
    buffer: &mut WorkerBuffer,
    layout: EdgeLayout,
    source: &str,
    target: &str,
    weight: f32,
    pos: u64,
) {
    match layout {
        EdgeLayout::SingleColumn => {
            let fields: [Tokens<'_>; 1] = [smallvec![source, target]];
            buffer.add_row(config, indexer, &fields, weight, pos);
        }
        EdgeLayout::TwoColumns => {
            let fields: [Tokens<'_>; 2] = [smallvec![source], smallvec![target]];
            buffer.add_row(config, indexer, &fields, weight, pos);
        }
    }
}

/// Builds the matrix from parallel arrays of integer node identifiers. The result
/// is identical to feeding the lines `"<source> <target>"` (one reflexive column)
/// or `"<source>\t<target>"` (two columns).
pub fn build_graph_from_int_edges(
    config: &Configuration,
    sources: &[i64],
    targets: &[i64],
    weights: Option<&[f32]>,
) -> Result<(SparseMatrix, BuildStats), String> {
    let layout = edge_layout(config)?;
    if sources.len() != targets.len() {
        return Err(format!(
            "src has {} element(s) but dst has {}",
            sources.len(),
            targets.len()
        ));
    }
    check_weights(weights, sources.len())?;
    if sources.len() as u64 >= MAX_LINES {
        return Err(format!("Too many edges ({})", sources.len()));
    }

    let indexer = Indexer::default();
    let buffers = run_partitioned(sources.len(), config.num_workers, |first, last, buffer| {
        for index in first..last {
            let source = IntBuffer::new(sources[index]);
            let target = IntBuffer::new(targets[index]);
            let weight = weights.map_or(1.0, |w| w[index]);
            add_edge(
                config,
                &indexer,
                buffer,
                layout,
                source.as_str(),
                target.as_str(),
                weight,
                position(0, index as u64, 0),
            );
        }
    });
    finish(config, indexer, buffers)
}

/// Builds the matrix from parallel arrays of node names. Unlike text lines, names
/// may contain any character (spaces, tabs, commas).
pub fn build_graph_from_str_edges<S: AsRef<str> + Sync>(
    config: &Configuration,
    sources: &[S],
    targets: &[S],
    weights: Option<&[f32]>,
) -> Result<(SparseMatrix, BuildStats), String> {
    let layout = edge_layout(config)?;
    if sources.len() != targets.len() {
        return Err(format!(
            "src has {} element(s) but dst has {}",
            sources.len(),
            targets.len()
        ));
    }
    check_weights(weights, sources.len())?;
    if sources.len() as u64 >= MAX_LINES {
        return Err(format!("Too many edges ({})", sources.len()));
    }

    let indexer = Indexer::default();
    let buffers = run_partitioned(sources.len(), config.num_workers, |first, last, buffer| {
        for index in first..last {
            let weight = weights.map_or(1.0, |w| w[index]);
            add_edge(
                config,
                &indexer,
                buffer,
                layout,
                sources[index].as_ref(),
                targets[index].as_ref(),
                weight,
                position(0, index as u64, 0),
            );
        }
    });
    finish(config, indexer, buffers)
}

/// Builds the matrix from hyperedges given as lists of entity names per entity
/// column. Used by tests and by callers that already hold tokenised data.
pub fn build_graph_from_rows<S: AsRef<str> + Sync>(
    config: &Configuration,
    rows: &[Vec<Vec<S>>],
    weights: Option<&[f32]>,
) -> Result<(SparseMatrix, BuildStats), String> {
    check_weights(weights, rows.len())?;
    let columns = config.entity_column_names.len();
    if let Some(index) = rows
        .iter()
        .position(|row| row.len() != columns || row.iter().any(|field| field.is_empty()))
    {
        return Err(format!(
            "row {} does not hold a non-empty list of entities for each of the {} column(s)",
            index, columns
        ));
    }
    let indexer = Indexer::default();
    let buffers = run_partitioned(rows.len(), config.num_workers, |first, last, buffer| {
        for index in first..last {
            let fields: SmallVec<[Tokens<'_>; 4]> = rows[index]
                .iter()
                .zip(&config.columns)
                .map(|(field, _)| field.iter().map(|t| t.as_ref()).collect())
                .collect();
            let weight = weights.map_or(1.0, |w| w[index]);
            buffer.add_row(
                config,
                &indexer,
                &fields,
                weight,
                position(0, index as u64, 0),
            );
        }
    });
    finish(config, indexer, buffers)
}
