# 4.0.0 (unreleased)

A correctness release. **Embeddings computed by this version differ from those of
3.x** (the initial vectors changed, see below), so embeddings stored from earlier
versions should be recomputed rather than mixed with new ones. Requires
Python 3.10 or newer.

### Fixed
- **Initial vectors were almost collinear.** `initialize_deterministically` derived
  every value from a single multiplicative hash step, so the vectors of different
  entities were strongly correlated: at 256 dimensions the cosine between two
  random entities had a standard deviation of 0.45 (independent vectors give
  0.06) and the whole matrix had an effective rank of about 5, whatever the
  dimension. Values are now independent and uniform in [-1, 1), computed with
  SplitMix64 from the entity name, the column and the seed. Classic Cleora
  (8 iterations, 256 dimensions) gains from it: 0.804 -> 0.812 accuracy on Cora
  and 0.715 -> 0.753 on email-Eu-core. The whitened default is unaffected
  (0.814 -> 0.800-0.813 on Cora depending on the seed, 0.631 -> 0.630 on
  email-Eu-core; the spread between random train/test splits is 0.01-0.02).
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
- Fortran-ordered and strided arrays crashed `left_markov_propagate` /
  `symmetric_markov_propagate` with a Rust panic. Any layout is accepted now;
  wrong shapes and dtypes raise readable `TypeError` / `ValueError`.
- `get_entity_column_mask` returned all `False` for a reflexive column.
- **Whitening a graph with no more entities than dimensions erased it.** With
  `num_entities <= feature_dim + 1` the whitened vectors are, in general, all
  equidistant (every pairwise cosine equals -1/(N-1)), so the default `embed()`
  returned nothing usable for small graphs - including the quick start of the
  README. `embed()` and its variants now warn about it, and the README examples
  use settings that work.
- **`pycleora.datasets` returned synthetic graphs under the names of real
  datasets** (`cora`, `citeseer`, `pubmed`, `ppi`, `amazon_*`, `dblp`, `reddit`,
  `flickr`, `yelp`, ...). These loaders now download the real graphs from their
  original sources; `flickr`, `yelp`, `dblp` and `reddit`, for which there is no
  loader yet, raise an explanatory error. Every dataset reports `"synthetic"`
  and `"source"`. The old generators remain available, clearly labelled, through
  `load_synthetic()`.
- **Baseline algorithms** (`pycleora.algorithms`) were rewritten after their
  papers. ProNE and RandNE returned embeddings at chance level on real graphs
  (0.30 and 0.32 accuracy on Cora, where always predicting the largest class
  gives 0.30; now 0.79 and 0.71). All baselines now start from the graph's
  adjacency matrix; they used Cleora's row-normalised matrix, self-loops
  included. DeepWalk and node2vec are documented as what they are here: random
  walks factorised with PPMI + SVD, not skip-gram.
- The wheels declared no dependencies although `import pycleora` needed NumPy
  and SciPy. NumPy is now declared; SciPy is optional (`pycleora[scipy]`,
  `pycleora[full]`) and imported only by the functions that need it.
- `pycleora.benchmark` measured memory with `tracemalloc`, which does not see
  allocations made by the Rust core. It reports peak resident memory on Linux
  and says which method was used.
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
- The default `embed()` path (whitening after every iteration) computes the
  covariance and the projection in the Rust core, in place, on all cores. For
  100,000 nodes, 256 dimensions and 40 iterations on a 2-core machine: 32.0 s ->
  16.5 s, and 2 instead of 6 embedding matrices of working memory (589 MB ->
  197 MB). Results agree with the NumPy reference implementation up to floating
  point error; the sign of every whitened column is now fixed instead of being
  left to the LAPACK build.
- The sparse matrix stores each weight once (CSR, normalised on the fly):
  8 instead of 12 bytes per non-zero entry. On the same machine and graph
  (500,000 edges), building takes 0.34 s instead of 0.66 s, one propagation
  step 107 ms instead of 148 ms, classic Cleora with 40 iterations 3.7 s instead
  of 6.1 s, and `find_most_similar` 34 ms instead of 129 ms.
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
- `benchmarks/run_benchmarks.py`: the reproducible benchmark behind the numbers
  in the README.
- A Python test suite (`tests/python`) and a CI job that runs it together with
  the Rust tests; wheels are smoke-tested before upload; macOS x86_64 wheels.
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