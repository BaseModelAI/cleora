# 4.0.0 (unreleased)

A correctness and performance release. **Embeddings computed by this version differ
from those of 3.x** (the initial vectors and the defaults of `embed()` changed, see
below), so embeddings stored from earlier versions should be recomputed rather than
mixed with new ones. Requires Python 3.10 or newer; tested on 3.10 to 3.15.

### Fixed
- **Initial vectors were almost collinear.** `initialize_deterministically` derived
  every value from a single multiplicative hash step, so the vectors of different
  entities were strongly correlated: at 256 dimensions the cosine between two
  random entities had a standard deviation of 0.45 (independent vectors give
  0.06) and the whole matrix had an effective rank of about 5, whatever the
  dimension. Values are now independent and uniform in [-1, 1), computed with
  SplitMix64 from the entity name, the column and the seed.
- **Graph construction depended on thread scheduling.** With more than one
  worker, edge weights were summed in a varying order and the entities kept
  when an oversized hyperedge is trimmed depended on which thread saw it
  first, so two runs on the same input could give different matrices. Weights
  are now accumulated exactly (fixed point), trimming uses global occurrence
  counts with a fixed tie-break, and entities are numbered by first appearance
  in the input. The matrix and the embeddings are bit-identical for any
  `num_workers`.
- **`embed_weighted` ignored its weights**: the weights cancelled in the
  row normalisation and the result equalled the unweighted embedding. Weights
  are now part of the matrix (`weights=` on `from_iterator` / `from_edges`, or a
  `weight::` column); an integer weight `k` equals repeating the line `k` times.
  `from_pandas(weight_col=...)`, `from_edge_list`, `from_networkx`,
  `from_scipy_sparse` and `from_numpy` pass weights through as well.
- **`embed_inductive` and `embed_streaming` moved every existing vector.** They
  re-ran the embedding, which with whitening rotates the whole space: the
  median cosine between an entity's vector before and after an update was
  0.04, and a classifier trained on the old vectors fell from 0.82 to 0.27
  accuracy. The default is now `mode="frozen"`: existing vectors do not change
  and new entities are placed at the weighted average of their neighbours.
  `mode="retrain"` keeps the previous behaviour; `embed_streaming(refresh_every=k)`
  periodically re-embeds and rotates the result back onto the old coordinates.
- **Bad input was accepted silently.** A missing file gave an empty graph, lines
  with the wrong number of columns were dropped without notice, and a file
  whose fields were separated by the wrong character gave an empty graph.
  Missing files now raise `FileNotFoundError`, skipped lines raise a
  `RuntimeWarning` naming the first offender, and input with no usable line
  raises `ValueError`. Files may have any extension.
- **Any call in a forked child process hung** once the parent had used the
  library (for example in a `multiprocessing` pool with the `fork` start method):
  the child waited for worker threads that only exist in the parent. Thread
  pools are now created per process.
- Fortran-ordered and strided arrays crashed `left_markov_propagate` /
  `symmetric_markov_propagate` with a Rust panic. Any layout is accepted now;
  wrong shapes and dtypes raise readable `TypeError` / `ValueError`.
- `get_entity_column_mask` returned all `False` for a reflexive column.
- **Whitening a graph with no more entities than dimensions erased it.** With
  `num_entities <= feature_dim + 1` the whitened vectors are, in general, all
  equidistant (every pairwise cosine equals -1/(N-1)), so the default `embed()`
  returned nothing usable for small graphs - including the quick start of the
  README. Whitening now warns about it, and it is no longer the default (see
  *Changed*).
- **`pycleora.datasets` returned synthetic graphs under the names of real
  datasets** (`cora`, `citeseer`, `pubmed`, `ppi`, `amazon_*`, `dblp`, `reddit`,
  `flickr`, `yelp`, ...). These loaders now download the real graphs from their
  original sources; `flickr`, `yelp`, `dblp` and `reddit`, for which there is no
  loader yet, raise an explanatory error. Every dataset reports `"synthetic"`
  and `"source"`. The old generators remain available, clearly labelled, through
  `load_synthetic()`.
- **Baseline algorithms** (`pycleora.algorithms`) were rewritten after their
  papers. ProNE and RandNE returned embeddings at chance level on real graphs.
  All baselines now start from the graph's
  adjacency matrix; they used Cleora's row-normalised matrix, self-loops
  included. DeepWalk and node2vec are documented as what they are here: random
  walks factorised with PPMI + SVD, not skip-gram.
- The wheels declared no dependencies although `import pycleora` needed NumPy
  and SciPy. NumPy is now declared; SciPy is optional (`pycleora[scipy]`,
  `pycleora[full]`) and imported only by the functions that need it.
- `pycleora.benchmark` measured memory with `tracemalloc`, which does not see
  allocations made by the Rust core (0.00 MB for an embedding of megabytes). The
  memory column is removed; the benchmark in `benchmarks/` measures the peak
  resident memory of a separate process per method. A failing algorithm is
  reported in its row with the error.
- **`find_most_similar(exclude_self=True)` returned the query itself** when
  `top_k` was at least the number of entities; **`predict_links` returned every
  pair twice** (`top_k=10` gave five distinct pairs); **`embed_directed`** gave
  entities without outgoing edges a zero vector, and the zeros spread upstream
  with every iteration. `embed(residual_weight=...)` silently ignored values
  outside [0, 1): they are an error now.
- **`mlp_classify` and `gcn_classify` chose their checkpoint on the test set** and
  reported the best test accuracy seen during training. The checkpoint is chosen
  on a validation part of the training entities; the test entities are scored
  once.
- **`pycleora.metrics`**: `average_precision` was not average precision,
  `clustering_scores` built its centroids from the true labels, and the ranking
  metrics looped over all nodes in Python. The module is rewritten on NumPy with
  the standard definitions and is tested against scikit-learn.
- `pycleora embed --algorithm node2vec` ran with p = q = 1, which is DeepWalk.
  `--p` and `--q` are passed on (defaults 1 and 0.5).
- `load_wikipedia_pos` dropped the edge weights of the published graph; it
  returns them as `"weights"`.
- `load_embeddings` no longer unpickles `.npz` content; the Planetoid loaders
  use a restricted unpickler.

### Changed
- **`embed()` runs classic Cleora with an automatic number of iterations by
  default**: `num_iterations="auto"`, `whiten=False`. The number of iterations
  comes from the graph (`SparseMatrix.suggest_iterations`): fewer on dense
  graphs, more for a reflexive column and for the symmetric normalisation.
  Up to 3.2 the default was 40 iterations with PCA whitening after each one;
  `embed(graph, num_iterations=40, whiten=True)` still does that. The change
  applies to every function that takes `whiten` and `num_iterations`, and to
  the command line (`pycleora embed --whiten` replaces the default; `-n` accepts
  `auto`). See the benchmark section of the README for the comparison.
- `SparseMatrix.entity_ids` is an immutable tuple that is built once. It used to
  be a new list on every access, which made loops over `graph.entity_ids[i]`
  quadratic. Assigning a new sequence of names still works.
- **Speed.** The propagation schedules work by matrix entries instead of rows,
  splits hub rows across threads, fuses the row normalisation into the product
  and prefetches; graph construction is parallel and deterministic. On an R-MAT
  graph with 0.6 million nodes and 12 million edges at 1024 dimensions
  (`benchmarks/kernel_speed.py`):

  | Machine | pycleora | Threads | Build | One iteration |
  |---|---|---:|---:|---:|
  | Intel Core Ultra 9 275HX (24 threads) | 3.2.1 | 24 | 17.8 s | 5.39 s |
  | | 4.0.0 | 24 | 1.11 s | 1.48 s |
  | | 4.0.0 | 1 | 6.38 s | 5.44 s |
  | Apple M4 (Mac mini, 10 cores) | 3.2.1 | 10 | 5.70 s | 4.02 s |
  | | 4.0.0 | 10 | 0.91 s | 1.29 s |
  | | 4.0.0 | 1 | 3.36 s | 3.09 s |

- The whitening path (`whiten=True`) computes the covariance and the projection
  in the Rust core, in place, with 2 instead of 6 embedding matrices of working
  memory. Results agree with the NumPy reference implementation up to floating
  point error; the sign of every whitened column is fixed instead of being left
  to the LAPACK build.
- The sparse matrix stores each weight once (CSR, normalised on the fly):
  8 instead of 12 bytes per non-zero entry.
- Entities are listed in order of first appearance in the input (the order
  used to depend on hashing and on the number of threads).
- Empty tokens are ignored: repeated spaces or a trailing space no longer create
  an entity with an empty name.
- Built with PyO3 0.30 as an `abi3` wheel: one wheel per platform for every
  Python from 3.10 on. Python 3.7-3.9 are no longer supported.
- The package version is defined in one place (`Cargo.toml`) and exposed as
  `pycleora.__version__`.

### Added
- `SparseMatrix.from_edges(src, dst, weights=None)` and
  `pycleora.io_utils.from_arrays`: build a graph from NumPy arrays of integer
  ids or from sequences of names without formatting and re-parsing strings.
- `embed(..., whiten=False)` runs classic Cleora entirely in the Rust core
  (`SparseMatrix.embed_fast`, `embed_fast_convergence`), with two embedding
  matrices of memory.
- `left_markov_propagate(x, out=..., rows=...)`: propagate into an existing
  buffer, or compute only selected rows of the result.
- `SparseMatrix.to_csr()`, `column_names`, `get_entity_indices`,
  `to_sparse_csr("raw")`, and module functions for the whitening kernels
  (`column_mean_cov`, `center_project`, `center_project_inplace`,
  `l2_normalize_inplace`).
- Real dataset loaders: Cora, CiteSeer and PubMed (Planetoid), Amazon Computers
  and Photo, PPI and Wikipedia POS (node2vec), PPI (GraphSAGE), email-Eu-core,
  dolphins, Les Miserables, football, ogbn-arxiv. Multi-label datasets return a
  label matrix.
- `benchmarks/`: the benchmark behind the numbers in the README - 26 public
  graphs, nine baselines run from their authors' code, node classification and
  link ranking with several probes and many metrics - replicable with one
  script (`benchmarks/replicate.sh`). See `benchmarks/README.md`.
- `SparseMatrix.from_cliques(cliques)`: a graph from an iterable of entity
  groups, without formatting strings; `SparseMatrix.isolated_entities`.
- `left_markov_propagate_normalized_inplace` /
  `symmetric_markov_propagate_normalized_inplace`: any number of iterations on
  an existing array, in place.
- `pycleora.metrics`: array-level functions for classification, multi-label,
  ranking, retrieval and clustering metrics.
- A Python test suite (`tests/python`) run by CI on Python 3.10 to 3.15 on Linux
  x86-64 and arm64 and on Apple silicon, together with the Rust tests, `cargo
  fmt` and `clippy`; wheels and the source distribution are installed and
  smoke-tested before upload; macOS x86_64 wheels. There is no 32-bit Linux
  wheel any more (NumPy has not published one since 1.22).
- `SparseMatrix.suggest_iterations(propagation, purpose)`: the rule behind
  `num_iterations="auto"`, with `purpose="similarity"` or `"classification"` for
  embeddings that serve one use only.

# 2.0.0 (Nov 24, 2024)

### New version released
- Python package with rust bindings
- Improved performance, memory usage and concurrency
- Allowing external embedding as seed
- Interoperable with numpy
- Not a standalone console application any more
- old version available in legacy/ folder will not be maintained

# 1.2.3 (June 29, 2022)

### Changed
- Bump libs ([#60]).

[#60]: https://github.com/Synerise/cleora/pull/60

### Fixed
- Check for malformed lines in input ([#59]).

[#59]: https://github.com/Synerise/cleora/pull/59


# 1.2.2 (June 24, 2022)

### Changed
- Allow cleora to accept multiple input files as positional args. Named argument 'input' is getting deprecated.

[#55]: https://github.com/Synerise/cleora/pull/55


# 1.2.1 (April 13, 2022)

### Changed
- Optimize "--output-format numpy" mode, so it doesn't require additional memory when writing output file ([#50]).
- Bump libs ([#52]).

[#50]: https://github.com/Synerise/cleora/pull/50
[#52]: https://github.com/Synerise/cleora/pull/52


# 1.2.0 (March 17, 2022)

### Added
- Use default hasher for vector init ([#47]).

[#47]: https://github.com/Synerise/cleora/pull/47


# 1.1.1 (May 14, 2021)

### Added
- Init embedding with seed during training ([#27]).

[#27]: https://github.com/Synerise/cleora/pull/27


# 1.1.0 (December 23, 2020)

### Changed
- Bumped `env_logger` to `0.8.2`, `smallvec` to `1.5.1`, removed `fnv` hasher ([#11]).

[#11]: https://github.com/Synerise/cleora/pull/11

### Added
- Tests (snapshots) for in-memory and memory-mapped files calculations of embeddings ([#12]).
- Support for `NumPy` output format (available via `--output-format` program argument) ([#15]).
- Jupyter notebooks with experiments ([#16]).

[#12]: https://github.com/Synerise/cleora/pull/12
[#15]: https://github.com/Synerise/cleora/pull/15
[#16]: https://github.com/Synerise/cleora/pull/16

### Improved
- Used `vector` for `hash_to_id` mappings, non-allocating cartesian product, `ryu` crate for faster write ([#13]).
- Sparse Matrix refactor (cleanup, simplification, using iter, speedup). Use Cargo.toml data for clap crate ([#17]).
- Unify and simplify embeddings calculation for in-memory and mmap matrices ([#18]).

[#13]: https://github.com/Synerise/cleora/pull/13
[#17]: https://github.com/Synerise/cleora/pull/17
[#18]: https://github.com/Synerise/cleora/pull/18


# 1.0.1 (November 23, 2020)

### Fixed
- Skip reading invalid UTF-8 line ([#8]).
- Fix clippy warnings ([#7]).

[#8]: https://github.com/Synerise/cleora/pull/8
[#7]: https://github.com/Synerise/cleora/pull/7

### Added
- JSON support ([#3]).
- Snapshot testing ([#5]).

[#3]: https://github.com/Synerise/cleora/pull/3
[#5]: https://github.com/Synerise/cleora/pull/5


# 1.0.0 (November 6, 2020)

- Initial release.