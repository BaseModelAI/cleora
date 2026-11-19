import warnings

import numpy as np
from typing import Optional, List, Dict, Callable, Tuple, Union

from .pycleora import SparseMatrix
from . import pycleora as _core
from . import stats
from . import preprocess
from . import align
from . import ensemble
from . import search
from . import compress

__version__ = _core.__version__

DEFAULT_FEATURE_DIM = 256
# "auto": SparseMatrix.suggest_iterations - or WHITENED_ITERATIONS with whitening.
DEFAULT_NUM_ITERATIONS = "auto"
DEFAULT_WHITEN = False
# Whitening counteracts over-smoothing, so with it the number of iterations hardly
# matters once it is large enough. 40 is what pycleora 3.2 used.
WHITENED_ITERATIONS = 40

_NORMALIZATIONS = ("l2", "l1", "spectral", "none")
_ROW_CHUNK = 65536


def embed_using_baseline_cleora(graph, feature_dim: int, iter: int):
    embeddings = graph.initialize_deterministically(feature_dim)
    for i in range(iter):
        embeddings = graph.left_markov_propagate(embeddings)
        embeddings = _postprocess_iteration(embeddings, "l2", True)
    return embeddings


def _validate_propagation(propagation: str):
    if propagation not in ("left", "symmetric"):
        raise ValueError(f"Unknown propagation type: '{propagation}'. Use 'left' or 'symmetric'.")


def _validate_normalization(normalization: str):
    if normalization not in _NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization method: {normalization}. Use 'l2', 'l1', 'spectral', or 'none'."
        )


def _get_propagate_fn(graph: SparseMatrix, propagation: str):
    _validate_propagation(propagation)
    return (
        graph.symmetric_markov_propagate
        if propagation == "symmetric"
        else graph.left_markov_propagate
    )


def _to_scipy_sparse(graph: SparseMatrix, markov_type: str = "left"):
    """The graph as a scipy CSR matrix ('left', 'symmetric' or 'raw' weights)."""
    from scipy.sparse import csr_matrix
    indptr, indices, data = graph.to_csr(markov_type)
    n = graph.num_entities
    return csr_matrix((data, indices, indptr), shape=(n, n))


def _resolve_iterations(
    num_iterations: Union[int, str], graph: SparseMatrix, propagation: str, whiten: bool
) -> int:
    """The number of iterations as a non-negative int; ``"auto"`` is decided here."""
    if isinstance(num_iterations, str):
        if num_iterations != "auto":
            raise ValueError(f"num_iterations must be an int or 'auto', got '{num_iterations}'")
        return WHITENED_ITERATIONS if whiten else graph.suggest_iterations(propagation)
    if num_iterations < 0:
        raise ValueError(f"num_iterations must not be negative, got {num_iterations}")
    return num_iterations


def _warn_if_whitening_is_degenerate(num_entities: int, feature_dim: int) -> None:
    """Whitening N points in D >= N - 1 dimensions turns them into the corners of a
    regular simplex: every pair of vectors has the same cosine, -1 / (N - 1), and
    nothing of the graph is left in the embedding."""
    if 1 < num_entities <= feature_dim + 1:
        warnings.warn(
            f"Whitening cannot work here: the graph has {num_entities} entities and the embedding "
            f"{feature_dim} dimensions. With no more entities than dimensions + 1, whitening leaves "
            "nothing of the graph in the vectors (in general they all end up equidistant). "
            "Use a smaller feature_dim or whiten=False.",
            RuntimeWarning,
            stacklevel=3,
        )


def _as_float32_matrix(array, name: str = "embeddings") -> np.ndarray:
    """A private, C-contiguous float32 copy of a 2-D array."""
    matrix = np.array(array, dtype=np.float32, order="C")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-dimensional array, got {matrix.ndim} dimension(s)")
    return matrix


def embed(
    graph: SparseMatrix,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
    propagation: str = "left",
    normalization: str = "l2",
    seed: int = 0,
    initial_embeddings: Optional[np.ndarray] = None,
    num_workers: Optional[int] = None,
    callback: Optional[Callable[[int, np.ndarray], None]] = None,
    residual_weight: float = 0.0,
    convergence_threshold: float = 0.0,
    whiten: bool = DEFAULT_WHITEN,
) -> np.ndarray:
    """Embeds the entities of ``graph``.

    Every iteration propagates the vectors along the graph and normalises the rows.
    This is classic Cleora: it runs entirely in the Rust core and needs no more memory
    than two embedding matrices.

    ``num_iterations="auto"`` (the default) takes the number of iterations from the
    density of the graph: ``graph.suggest_iterations(propagation)``, a compromise
    that needs no labels. Similarity search is best with somewhat fewer iterations
    and classification with more: pass
    ``graph.suggest_iterations(propagation, purpose="similarity")`` or
    ``purpose="classification"`` when the use is known. When there is a validation
    set, choosing the number on it is cheap (``callback`` receives the vectors after
    every iteration) and usually worth another one or two per cent.

    ``whiten=True`` additionally PCA-whitens the columns after every iteration, which
    was the default of pycleora 3.2. It is 10-100 times slower and needs many more
    entities than dimensions: with ``num_entities <= feature_dim + 1`` all whitened
    vectors are equidistant and the result is useless (a ``RuntimeWarning`` says so).
    With whitening, ``"auto"`` means 40 iterations.
    """
    _validate_propagation(propagation)
    _validate_normalization(normalization)
    num_iterations = _resolve_iterations(num_iterations, graph, propagation, whiten)
    if whiten and num_iterations > 0:
        _warn_if_whitening_is_degenerate(graph.num_entities, feature_dim)

    use_fast_path = (
        initial_embeddings is None
        and callback is None
        and normalization == "l2"
        and not whiten
    )
    if use_fast_path:
        if convergence_threshold > 0:
            embeddings, _ = graph.embed_fast_convergence(
                feature_dim,
                num_iterations,
                propagation=propagation,
                seed=seed,
                residual_weight=residual_weight,
                convergence_threshold=convergence_threshold,
                num_workers=num_workers,
            )
        else:
            embeddings = graph.embed_fast(
                feature_dim,
                num_iterations,
                propagation=propagation,
                seed=seed,
                residual_weight=residual_weight,
                num_workers=num_workers,
            )
        return embeddings

    if initial_embeddings is not None:
        embeddings = _as_float32_matrix(initial_embeddings, "initial_embeddings")
        if embeddings.shape[0] != graph.num_entities:
            raise ValueError(
                f"initial_embeddings has {embeddings.shape[0]} rows but graph has {graph.num_entities} entities"
            )
    else:
        embeddings = graph.initialize_deterministically(feature_dim, seed)

    propagate_fn = _get_propagate_fn(graph, propagation)
    use_residual = 0.0 < residual_weight < 1.0

    if normalization == "l2" and not whiten and callback is None and not use_residual and convergence_threshold <= 0:
        # Nothing has to look at the vectors between iterations: the core runs them
        # all, with the normalisation fused into the propagation.
        iterate_inplace = (
            graph.symmetric_markov_propagate_normalized_inplace
            if propagation == "symmetric"
            else graph.left_markov_propagate_normalized_inplace
        )
        iterate_inplace(embeddings, num_iterations, num_workers)
        return embeddings

    if normalization == "spectral":
        # The SVD-based normalisation cannot work in place.
        for i in range(num_iterations):
            previous = embeddings
            embeddings = propagate_fn(embeddings, num_workers=num_workers)
            if use_residual:
                _mix_residual_inplace(embeddings, previous, residual_weight)
            embeddings = _postprocess_iteration(embeddings, normalization, whiten)
            if callback is not None:
                callback(i, embeddings)
            if convergence_threshold > 0 and i > 0 and embeddings.shape == previous.shape:
                if _compute_rmse(embeddings, previous) < convergence_threshold:
                    break
        return embeddings

    # Two buffers are reused for all iterations; every step works in place.
    buffer = np.empty_like(embeddings)
    for i in range(num_iterations):
        propagate_fn(embeddings, num_workers=num_workers, out=buffer)
        if use_residual:
            _mix_residual_inplace(buffer, embeddings, residual_weight)
        _postprocess_inplace(buffer, normalization, whiten, num_workers)
        converged = (
            convergence_threshold > 0
            and i > 0
            and _compute_rmse(buffer, embeddings) < convergence_threshold
        )
        embeddings, buffer = buffer, embeddings
        if callback is not None:
            # The buffers are recycled, so the callback gets its own copy.
            callback(i, embeddings.copy())
        if converged:
            break

    return embeddings


def whiten_embeddings(
    embeddings: np.ndarray,
    n_components: Optional[int] = None,
    num_workers: Optional[int] = None,
    inplace: bool = False,
) -> np.ndarray:
    """PCA-whitens the columns: zero mean, unit variance, no correlation.

    Columns come out ordered by decreasing variance of the input; the sign of every
    column is fixed so that the result does not depend on the LAPACK build.
    With ``inplace=True`` a C-contiguous float32 input is overwritten and returned
    (only possible when all components are kept).
    """
    source = np.asarray(embeddings)
    if source.ndim != 2:
        raise ValueError(f"embeddings must be a 2-dimensional array, got {source.ndim} dimension(s)")
    owned = False
    if source.dtype != np.float32 or not source.flags.c_contiguous:
        source = np.ascontiguousarray(source, dtype=np.float32)
        owned = True
    n, d = source.shape
    if n <= 1:
        return source if (inplace or owned) else source.copy()

    mean, covariance = _core.column_mean_cov(source, num_workers)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    # Eigenvectors are defined up to sign: make the largest component positive.
    pivot = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[pivot, np.arange(eigenvectors.shape[1])])
    signs[signs == 0] = 1.0
    eigenvectors = eigenvectors * signs

    scale = 1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10))
    transform = np.ascontiguousarray(eigenvectors * scale, dtype=np.float32)
    mean_f32 = mean.astype(np.float32)

    square = transform.shape[1] == d
    if square and (owned or (inplace and source.flags.writeable)):
        _core.center_project_inplace(source, mean_f32, transform, num_workers)
        return source
    return _core.center_project(source, mean_f32, transform, num_workers)


def embed_with_node_features(
    graph: SparseMatrix,
    node_features: Dict[str, np.ndarray],
    num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
    propagation: str = "left",
    normalization: str = "l2",
    feature_weight: float = 0.5,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    if not node_features:
        raise ValueError("node_features must be a non-empty dict of entity_id -> feature_vector")

    sample_feat = next(iter(node_features.values()))
    feat_dim = len(sample_feat)

    init_emb = graph.initialize_deterministically(feat_dim)

    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    for eid, feat in node_features.items():
        idx = index_map.get(eid)
        if idx is not None:
            feat_arr = np.array(feat, dtype=np.float32)
            if len(feat_arr) != feat_dim:
                raise ValueError(
                    f"Feature for '{eid}' has dimension {len(feat_arr)}, expected {feat_dim}"
                )
            init_emb[idx] = (1 - feature_weight) * init_emb[idx] + feature_weight * feat_arr

    return embed(
        graph,
        feature_dim=feat_dim,
        num_iterations=num_iterations,
        propagation=propagation,
        normalization=normalization,
        initial_embeddings=init_emb,
        num_workers=num_workers,
    )


def embed_with_attention(
    graph: SparseMatrix,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
    propagation: str = "left",
    normalization: str = "l2",
    attention_temperature: float = 1.0,
    seed: int = 0,
    num_workers: Optional[int] = None,
    callback: Optional[Callable[[int, np.ndarray], None]] = None,
    whiten: bool = DEFAULT_WHITEN,
) -> np.ndarray:
    _validate_propagation(propagation)

    if attention_temperature <= 0:
        raise ValueError(f"attention_temperature must be positive, got {attention_temperature}")

    num_iterations = _resolve_iterations(num_iterations, graph, propagation, whiten)
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")

    if whiten:
        _warn_if_whitening_is_degenerate(graph.num_entities, feature_dim)

    embeddings = graph.initialize_deterministically(feature_dim, seed)
    propagate_fn = _get_propagate_fn(graph, propagation)

    embeddings = propagate_fn(embeddings, num_workers=num_workers)
    embeddings = _postprocess_iteration(embeddings, normalization, whiten)

    if callback is not None:
        callback(0, embeddings)

    if num_iterations == 1:
        return embeddings

    from scipy.sparse import csr_matrix
    adj = _to_scipy_sparse(graph, propagation)
    n = adj.shape[0]
    indptr, adj_cols = adj.indptr, adj.indices
    row_lengths = np.diff(indptr)
    adj_rows = np.repeat(np.arange(n, dtype=adj_cols.dtype), row_lengths)
    nonempty = np.flatnonzero(row_lengths > 0)
    edge_chunk = max(1, (1 << 24) // max(feature_dim, 1))

    for i in range(1, num_iterations):
        norms = np.sqrt(np.einsum("ij,ij->i", embeddings, embeddings, dtype=np.float64))
        inv_norms = (1.0 / np.maximum(norms, 1e-10)).astype(np.float32)

        # cosine of every edge, computed in bounded chunks instead of materialising
        # two (edges x dim) arrays
        dot_products = np.empty(len(adj_cols), dtype=np.float64)
        for start in range(0, len(adj_cols), edge_chunk):
            stop = start + edge_chunk
            r, c = adj_rows[start:stop], adj_cols[start:stop]
            dot_products[start:stop] = (
                np.einsum("ij,ij->i", embeddings[r], embeddings[c], dtype=np.float64)
                * inv_norms[r] * inv_norms[c]
            )
        attention_scores = dot_products / attention_temperature

        row_max = np.zeros(n, dtype=np.float64)
        if len(nonempty):
            row_max[nonempty] = np.maximum.reduceat(attention_scores, indptr[:-1][nonempty])
        exp_scores = np.exp(attention_scores - row_max[adj_rows])

        # softmax over each row, applied on top of the graph weights and renormalised
        weights = exp_scores * adj.data
        row_sums = np.bincount(adj_rows, weights=weights, minlength=n)
        weights /= np.maximum(row_sums, 1e-10)[adj_rows]
        weighted_adj = csr_matrix((weights, adj_cols, indptr), shape=adj.shape)

        embeddings = (weighted_adj @ embeddings).astype(np.float32)
        embeddings = _postprocess_iteration(embeddings, normalization, whiten)

        if callback is not None:
            callback(i, embeddings)

    return embeddings


def embed_multiscale(
    graph: SparseMatrix,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    scales: List[int] = None,
    propagation: str = "left",
    normalization: str = "l2",
    seed: int = 0,
    num_workers: Optional[int] = None,
    whiten: bool = DEFAULT_WHITEN,
) -> np.ndarray:
    """Concatenates the embeddings after each of the given numbers of iterations.

    The block of scale ``k`` is exactly ``embed(graph, feature_dim, k, ...)``.
    """
    propagate_fn = _get_propagate_fn(graph, propagation)
    _validate_normalization(normalization)
    if whiten:
        _warn_if_whitening_is_degenerate(graph.num_entities, feature_dim)

    if scales is None:
        scales = [10, 20, 30, 40]

    if not scales or not all(isinstance(s, int) and s > 0 for s in scales):
        raise ValueError("scales must be a non-empty list of positive integers")

    embeddings = graph.initialize_deterministically(feature_dim, seed)
    all_embeddings = []
    in_place = normalization != "spectral"

    current_iter = 0
    for scale in sorted(scales):
        while current_iter < scale:
            embeddings = propagate_fn(embeddings, num_workers=num_workers)
            if in_place:
                _postprocess_inplace(embeddings, normalization, whiten, num_workers)
            else:
                embeddings = _postprocess_iteration(embeddings, normalization, whiten)
            current_iter += 1
        snapshot = embeddings.copy()
        all_embeddings.append(snapshot)

    return np.concatenate(all_embeddings, axis=1)


def embed_weighted(
    edges_with_weights: List[Tuple[str, float]],
    columns: str,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
    propagation: str = "left",
    normalization: str = "l2",
    seed: int = 0,
    hyperedge_trim_n: int = 16,
    num_workers: Optional[int] = None,
    whiten: bool = DEFAULT_WHITEN,
) -> Tuple[SparseMatrix, np.ndarray]:
    """Embeds a graph whose hyperedges carry weights.

    ``edges_with_weights`` holds ``(line, weight)`` pairs. A line with weight ``w``
    contributes ``w`` times the mass of an unweighted line, so an integer weight
    ``k`` is equivalent to repeating the line ``k`` times. Weights must be finite
    and non-negative; a zero weight removes the line.
    """
    lines = [line for line, _ in edges_with_weights]
    weights = np.fromiter((w for _, w in edges_with_weights), dtype=np.float64, count=len(lines))

    graph = SparseMatrix.from_iterator(
        iter(lines), columns, hyperedge_trim_n, num_workers, weights=weights
    )
    embeddings = embed(
        graph,
        feature_dim=feature_dim,
        num_iterations=num_iterations,
        propagation=propagation,
        normalization=normalization,
        seed=seed,
        num_workers=num_workers,
        whiten=whiten,
    )
    return graph, embeddings


def embed_directed(
    edges: List[str],
    columns: str,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
    normalization: str = "l2",
    seed: int = 0,
    hyperedge_trim_n: int = 16,
    num_workers: Optional[int] = None,
    whiten: bool = DEFAULT_WHITEN,
) -> Tuple[SparseMatrix, np.ndarray]:
    """Embeds a directed graph: in every line, earlier entities point to later ones.

    Each entity is propagated from the entities it points to only.
    """
    from scipy.sparse import csr_matrix, diags

    graph = SparseMatrix.from_iterator(iter(edges), columns, hyperedge_trim_n, num_workers)
    n = graph.num_entities
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}

    sources: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    pair_sources: List[int] = []
    pair_targets: List[int] = []
    for edge_str in edges:
        tokens = [index_map.get(token, -1) for token in edge_str.strip().split()]
        if len(tokens) == 2:
            if tokens[0] >= 0 and tokens[1] >= 0:
                pair_sources.append(tokens[0])
                pair_targets.append(tokens[1])
        elif len(tokens) > 2:
            ids = np.asarray(tokens, dtype=np.int64)
            first, second = np.triu_indices(len(ids), k=1)
            keep = (ids[first] >= 0) & (ids[second] >= 0)
            sources.append(ids[first][keep])
            targets.append(ids[second][keep])
    sources.append(np.asarray(pair_sources, dtype=np.int64))
    targets.append(np.asarray(pair_targets, dtype=np.int64))
    source_idx = np.concatenate(sources)
    target_idx = np.concatenate(targets)

    direction = csr_matrix(
        (np.ones(len(source_idx), dtype=np.float32), (source_idx, target_idx)), shape=(n, n)
    )
    direction.sum_duplicates()
    direction.data[:] = 1.0

    adj = _to_scipy_sparse(graph, "left").multiply(direction).tocsr().astype(np.float64)
    row_sums = np.asarray(adj.sum(axis=1)).ravel()
    row_sums = np.maximum(row_sums, 1e-10)
    adj = (diags(1.0 / row_sums) @ adj).tocsr()

    if whiten:
        _warn_if_whitening_is_degenerate(graph.num_entities, feature_dim)

    num_iterations = _resolve_iterations(num_iterations, graph, "left", whiten)
    embeddings = graph.initialize_deterministically(feature_dim, seed)
    for i in range(num_iterations):
        embeddings = (adj @ embeddings).astype(np.float32)
        embeddings = _postprocess_iteration(embeddings, normalization, whiten)

    return graph, embeddings


def supervised_refine(
    graph: SparseMatrix,
    embeddings: np.ndarray,
    positive_pairs: List[Tuple[str, str]],
    negative_pairs: Optional[List[Tuple[str, str]]] = None,
    learning_rate: float = 0.01,
    num_epochs: int = 50,
    margin: float = 0.5,
    num_negatives_per_positive: int = 5,
    callback: Optional[Callable[[int, float], None]] = None,
) -> np.ndarray:
    if embeddings.shape[0] != graph.num_entities:
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but graph has {graph.num_entities} entities"
        )

    refined = embeddings.copy().astype(np.float64)
    n_entities = graph.num_entities

    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}

    pos_indices = []
    for a, b in positive_pairs:
        ia = index_map.get(a)
        ib = index_map.get(b)
        if ia is None:
            raise ValueError(f"Entity '{a}' not found in graph")
        if ib is None:
            raise ValueError(f"Entity '{b}' not found in graph")
        pos_indices.append((ia, ib))

    neg_indices = []
    if negative_pairs is not None:
        for a, b in negative_pairs:
            ia = index_map.get(a)
            ib = index_map.get(b)
            if ia is None:
                raise ValueError(f"Entity '{a}' not found in graph")
            if ib is None:
                raise ValueError(f"Entity '{b}' not found in graph")
            neg_indices.append((ia, ib))

    rng = np.random.default_rng(42)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for ia, ib in pos_indices:
            va = refined[ia]
            vb = refined[ib]
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            if norm_a < 1e-10 or norm_b < 1e-10:
                continue
            cos_sim = np.dot(va, vb) / (norm_a * norm_b)
            loss_pos = max(0.0, 1.0 - cos_sim)
            total_loss += loss_pos

            if loss_pos > 0:
                grad_a = vb / (norm_a * norm_b) - va * cos_sim / (norm_a * norm_a)
                grad_b = va / (norm_a * norm_b) - vb * cos_sim / (norm_b * norm_b)
                refined[ia] += learning_rate * grad_a
                refined[ib] += learning_rate * grad_b

            if negative_pairs is not None:
                neg_pool = neg_indices
            else:
                neg_pool_indices = rng.choice(
                    n_entities,
                    size=min(num_negatives_per_positive, n_entities - 1),
                    replace=False,
                )
                neg_pool = [(ia, int(ni)) for ni in neg_pool_indices if ni != ia]

            for ia_neg, ib_neg in neg_pool[:num_negatives_per_positive]:
                va_neg = refined[ia_neg]
                vb_neg = refined[ib_neg]
                norm_an = np.linalg.norm(va_neg)
                norm_bn = np.linalg.norm(vb_neg)
                if norm_an < 1e-10 or norm_bn < 1e-10:
                    continue
                cos_neg = np.dot(va_neg, vb_neg) / (norm_an * norm_bn)
                loss_neg = max(0.0, cos_neg - margin)
                total_loss += loss_neg

                if loss_neg > 0:
                    grad_an = -vb_neg / (norm_an * norm_bn) + va_neg * cos_neg / (norm_an * norm_an)
                    grad_bn = -va_neg / (norm_an * norm_bn) + vb_neg * cos_neg / (norm_bn * norm_bn)
                    refined[ia_neg] += learning_rate * grad_an
                    refined[ib_neg] += learning_rate * grad_bn

        avg_loss = total_loss / max(len(pos_indices), 1)

        if callback is not None:
            callback(epoch, avg_loss)

        if avg_loss < 1e-6:
            break

    return _normalize(refined.astype(np.float32), "l2")


def update_graph(
    existing_edges: List[str],
    new_edges: List[str],
    columns: str,
    hyperedge_trim_n: int = 16,
    num_workers: Optional[int] = None,
) -> SparseMatrix:
    all_edges = list(existing_edges) + list(new_edges)
    return SparseMatrix.from_iterator(iter(all_edges), columns, hyperedge_trim_n, num_workers)


def remove_edges(
    existing_edges: List[str],
    edges_to_remove: List[str],
    columns: str,
    hyperedge_trim_n: int = 16,
    num_workers: Optional[int] = None,
) -> SparseMatrix:
    remove_set = set(edges_to_remove)
    remaining = [e for e in existing_edges if e not in remove_set]
    if not remaining:
        raise ValueError("Cannot remove all edges from the graph")
    return SparseMatrix.from_iterator(iter(remaining), columns, hyperedge_trim_n, num_workers)


def _rows_are_unit_norm(embeddings: np.ndarray) -> bool:
    sample = embeddings[: min(len(embeddings), 10000)]
    if len(sample) == 0:
        return False
    norms = np.sqrt(np.einsum("ij,ij->i", sample, sample, dtype=np.float64))
    return bool(np.all(np.abs(norms - 1.0) < 1e-3))


def _frozen_inductive_update(
    updated_graph: SparseMatrix,
    old_entity_ids,
    old_embeddings: np.ndarray,
    propagation: str = "left",
    refine_iterations: int = 3,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    """Embeddings for ``updated_graph`` that keep every already known entity where
    it is and place each new entity at the weighted average of its neighbours.

    Only the rows of the new entities are propagated, so the cost is proportional
    to the number of their connections, not to the size of the graph."""
    n = updated_graph.num_entities
    dim = old_embeddings.shape[1]
    old_index = {eid: i for i, eid in enumerate(old_entity_ids)}
    source = np.fromiter(
        (old_index.get(eid, -1) for eid in updated_graph.entity_ids), dtype=np.int64, count=n
    )
    known = source >= 0

    embeddings = np.zeros((n, dim), dtype=np.float32)
    embeddings[known] = old_embeddings[source[known]]
    new_rows = np.flatnonzero(~known)
    if len(new_rows) == 0:
        return embeddings

    propagate_fn = _get_propagate_fn(updated_graph, propagation)
    available = known.astype(np.float32).reshape(n, 1)
    unit_norm = _rows_are_unit_norm(old_embeddings)

    for _ in range(max(1, refine_iterations)):
        # Average over the neighbours that already have a vector. New entities that
        # are only reachable through other new entities are placed in later rounds.
        mass = propagate_fn(available, num_workers=num_workers, rows=new_rows).ravel()
        placed = mass > 1e-12
        estimate = propagate_fn(embeddings, num_workers=num_workers, rows=new_rows)
        estimate[placed] /= mass[placed, None]
        estimate[~placed] = 0.0
        if unit_norm:
            norms = np.sqrt(np.einsum("ij,ij->i", estimate, estimate, dtype=np.float64))
            estimate /= np.maximum(norms, 1e-10)[:, None].astype(np.float32)
        embeddings[new_rows] = estimate
        available[new_rows, 0] = placed

    return embeddings


def embed_inductive(
    trained_graph: SparseMatrix,
    trained_embeddings: np.ndarray,
    existing_edges: List[str],
    new_edges: List[str],
    columns: str,
    num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
    propagation: str = "left",
    normalization: str = "l2",
    hyperedge_trim_n: int = 16,
    num_workers: Optional[int] = None,
    mode: str = "frozen",
    refine_iterations: int = 3,
    whiten: bool = DEFAULT_WHITEN,
) -> Tuple[SparseMatrix, np.ndarray]:
    """Adds new edges (and the entities they bring) to an embedded graph.

    ``mode="frozen"`` (default): entities that were already embedded keep exactly
    the vector they had, so models trained on the old embeddings stay valid. Each
    new entity is placed at the weighted average of its neighbours' vectors
    (``refine_iterations`` rounds, so chains of new entities are reached too).
    A new entity with no path to a known one gets a zero vector.

    ``mode="retrain"``: the whole graph is embedded again, starting from the old
    vectors. All coordinates change (with whitening they are rotated arbitrarily),
    so the result is not comparable with the old embeddings.
    """
    if trained_embeddings.shape[0] != trained_graph.num_entities:
        raise ValueError(
            f"trained_embeddings has {trained_embeddings.shape[0]} rows but graph has {trained_graph.num_entities} entities"
        )
    if mode not in ("frozen", "retrain"):
        raise ValueError(f"Unknown mode '{mode}'. Use 'frozen' or 'retrain'.")
    _validate_propagation(propagation)

    updated_graph = update_graph(
        existing_edges, new_edges, columns, hyperedge_trim_n, num_workers
    )

    if mode == "frozen":
        updated_embeddings = _frozen_inductive_update(
            updated_graph,
            trained_graph.entity_ids,
            np.asarray(trained_embeddings, dtype=np.float32),
            propagation,
            refine_iterations,
            num_workers,
        )
        return updated_graph, updated_embeddings

    old_index_map = {eid: i for i, eid in enumerate(trained_graph.entity_ids)}
    dim = trained_embeddings.shape[1]
    # deterministic, small starting vectors for the entities that are new
    init_embeddings = updated_graph.initialize_deterministically(dim) * np.float32(0.01)
    source = np.fromiter(
        (old_index_map.get(eid, -1) for eid in updated_graph.entity_ids),
        dtype=np.int64,
        count=updated_graph.num_entities,
    )
    known = source >= 0
    init_embeddings[known] = trained_embeddings[source[known]]

    updated_embeddings = embed(
        updated_graph,
        feature_dim=dim,
        num_iterations=num_iterations,
        propagation=propagation,
        normalization=normalization,
        initial_embeddings=init_embeddings,
        num_workers=num_workers,
        whiten=whiten,
    )

    return updated_graph, updated_embeddings


def _align_to_previous(new_embeddings, new_entity_ids, old_embeddings, old_entity_ids):
    """Rotates freshly computed embeddings onto the coordinates of the previous ones
    (orthogonal Procrustes over the entities both have)."""
    old_index = {eid: i for i, eid in enumerate(old_entity_ids)}
    source = np.fromiter(
        (old_index.get(eid, -1) for eid in new_entity_ids), dtype=np.int64, count=len(new_entity_ids)
    )
    shared = np.flatnonzero(source >= 0)
    if len(shared) == 0 or new_embeddings.shape[1] != old_embeddings.shape[1]:
        return new_embeddings
    moving = new_embeddings[shared].astype(np.float64)
    target = old_embeddings[source[shared]].astype(np.float64)
    u, _, vt = np.linalg.svd(moving.T @ target)
    rotation = (u @ vt).astype(np.float32)
    return np.ascontiguousarray(new_embeddings @ rotation, dtype=np.float32)


def embed_streaming(
    edge_batches,
    columns: str,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
    propagation: str = "left",
    normalization: str = "l2",
    hyperedge_trim_n: int = 16,
    num_workers: Optional[int] = None,
    batch_callback: Optional[Callable[[int, SparseMatrix, np.ndarray], None]] = None,
    refresh_every: Optional[int] = None,
    whiten: bool = DEFAULT_WHITEN,
    seed: int = 0,
) -> Tuple[SparseMatrix, np.ndarray]:
    """Embeds a graph that arrives in batches of edges.

    The first batch is embedded in full. For every later batch the entities that
    are already embedded keep their vectors and new entities are placed next to
    their neighbours (see ``embed_inductive``), so the coordinates stay stable.

    With ``refresh_every=k`` every k-th batch re-embeds the whole graph and
    rotates the result onto the previous coordinates, which lets old entities
    move with the new evidence while keeping the space comparable.
    """
    if refresh_every is not None and refresh_every <= 0:
        raise ValueError(f"refresh_every must be positive, got {refresh_every}")

    all_edges = []
    graph = None
    embeddings = None
    entity_ids = None

    def full_embedding(current_graph):
        return embed(
            current_graph, feature_dim=feature_dim, num_iterations=num_iterations,
            propagation=propagation, normalization=normalization,
            num_workers=num_workers, whiten=whiten, seed=seed,
        )

    for batch_idx, batch in enumerate(edge_batches):
        all_edges.extend(batch)
        graph = SparseMatrix.from_iterator(
            iter(all_edges), columns, hyperedge_trim_n, num_workers
        )

        if embeddings is None:
            embeddings = full_embedding(graph)
        elif refresh_every is not None and batch_idx % refresh_every == 0:
            embeddings = _align_to_previous(
                full_embedding(graph), graph.entity_ids, embeddings, entity_ids
            )
        else:
            embeddings = _frozen_inductive_update(
                graph, entity_ids, embeddings, propagation, num_workers=num_workers
            )

        entity_ids = graph.entity_ids

        if batch_callback is not None:
            batch_callback(batch_idx, graph, embeddings)

    return graph, embeddings


def predict_links(
    graph: SparseMatrix,
    embeddings: np.ndarray,
    top_k: int = 10,
    exclude_existing: bool = True,
    source_entities: Optional[List[str]] = None,
) -> List[Dict]:
    """The ``top_k`` highest-cosine pairs (source, target) among the candidates of
    the given source entities (all entities by default)."""
    n = graph.num_entities
    if top_k <= 0 or n == 0:
        return []
    norms = np.sqrt(np.einsum("ij,ij->i", embeddings, embeddings, dtype=np.float64))
    normed = (embeddings / np.maximum(norms, 1e-10)[:, None]).astype(np.float32)

    if source_entities is not None:
        source_indices = np.asarray(graph.get_entity_indices(list(source_entities)), dtype=np.int64)
    else:
        source_indices = np.arange(n, dtype=np.int64)

    if exclude_existing:
        indptr, indices, _ = graph.to_csr()

    entity_ids = graph.entity_ids
    keep = min(top_k, n)
    block = max(1, (1 << 26) // max(n, 1))
    predictions = []
    for start in range(0, len(source_indices), block):
        sources = source_indices[start:start + block]
        sims = normed[sources] @ normed.T
        sims[np.arange(len(sources)), sources] = -2.0
        if exclude_existing:
            # the matrix is symmetric: the row of a source lists all its existing links
            for local, src in enumerate(sources):
                sims[local, indices[indptr[src]:indptr[src + 1]]] = -2.0
        candidates = np.argpartition(-sims, keep - 1, axis=1)[:, :keep]
        for local, src in enumerate(sources):
            row = candidates[local]
            row = row[np.argsort(-sims[local, row], kind="stable")]
            for tgt in row:
                score = float(sims[local, tgt])
                if score <= -2.0:
                    continue
                predictions.append({
                    "source": entity_ids[int(src)],
                    "target": entity_ids[int(tgt)],
                    "score": score,
                })

    predictions.sort(key=lambda x: x["score"], reverse=True)
    return predictions[:top_k]


def propagate_gpu(
    graph: SparseMatrix,
    embeddings: np.ndarray,
    num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
    propagation: str = "left",
    normalization: str = "l2",
    device: str = "cuda",
    callback: Optional[Callable[[int, np.ndarray], None]] = None,
    whiten: bool = DEFAULT_WHITEN,
) -> np.ndarray:
    _validate_propagation(propagation)
    num_iterations = _resolve_iterations(num_iterations, graph, propagation, whiten)

    if normalization not in ("l2", "l1", "none"):
        raise ValueError(f"GPU propagation supports 'l2', 'l1', or 'none' normalization. Got: '{normalization}'")

    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for GPU propagation. Install with: pip install torch"
        )

    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError(
            "CUDA is not available. Use device='cpu' for PyTorch CPU propagation, "
            "or use embed() for native Rust CPU propagation."
        )

    rows, cols, vals, n_rows, n_cols = graph.to_sparse_csr(propagation)
    rows_t = torch.from_numpy(np.array(rows, dtype=np.int64))
    cols_t = torch.from_numpy(np.array(cols, dtype=np.int64))
    vals_t = torch.from_numpy(np.array(vals, dtype=np.float32))

    indices = torch.stack([rows_t, cols_t])
    adj = torch.sparse_coo_tensor(indices, vals_t, size=(n_rows, n_cols)).to(device)
    adj = adj.coalesce()

    emb = torch.from_numpy(embeddings.astype(np.float32)).to(device)

    for i in range(num_iterations):
        emb = torch.sparse.mm(adj, emb)

        if normalization == "l2":
            norms = torch.norm(emb, dim=1, keepdim=True).clamp(min=1e-10)
            emb = emb / norms
        elif normalization == "l1":
            norms = torch.norm(emb, p=1, dim=1, keepdim=True).clamp(min=1e-10)
            emb = emb / norms

        if whiten:
            emb = _whiten_embeddings_torch(emb)

        if callback is not None:
            callback(i, emb.cpu().numpy())

    return emb.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def find_most_similar(
    graph: SparseMatrix,
    embeddings: np.ndarray,
    query_entity: str,
    top_k: int = 10,
    exclude_self: bool = True,
) -> List[Dict]:
    query_idx = graph.get_entity_index(query_entity)
    query_vec = embeddings[query_idx]

    norms = np.sqrt(np.einsum("ij,ij->i", embeddings, embeddings, dtype=np.float64))
    norms = np.maximum(norms, 1e-10)
    query_norm = max(float(norms[query_idx]), 1e-10)
    similarities = ((embeddings @ query_vec) / (norms * query_norm)).astype(np.float32)

    if exclude_self:
        similarities[query_idx] = -1.0

    keep = min(max(top_k, 0), len(similarities))
    if keep == 0:
        return []
    top_indices = np.argpartition(-similarities, keep - 1)[:keep]
    top_indices = top_indices[np.argsort(-similarities[top_indices], kind="stable")]
    entity_ids = graph.entity_ids
    results = []
    for idx in top_indices:
        results.append({
            "entity_id": entity_ids[int(idx)],
            "index": int(idx),
            "similarity": float(similarities[idx]),
        })
    return results


def embed_edge_features(
    graph: SparseMatrix,
    edge_features: Dict[str, np.ndarray],
    feature_dim: int = DEFAULT_FEATURE_DIM,
    num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
    propagation: str = "left",
    normalization: str = "l2",
    combine: str = "concat",
    num_workers: Optional[int] = None,
    whiten: bool = DEFAULT_WHITEN,
) -> np.ndarray:
    from scipy.sparse import csr_matrix, diags

    _validate_propagation(propagation)
    num_iterations = _resolve_iterations(num_iterations, graph, propagation, whiten)

    struct_emb = embed(
        graph, feature_dim=feature_dim, num_iterations=num_iterations,
        propagation=propagation, normalization=normalization, num_workers=num_workers,
        whiten=whiten,
    )

    if not edge_features:
        return struct_emb

    sample_feat = next(iter(edge_features.values()))
    edge_feat_dim = len(sample_feat)

    rows, cols, vals, n, _ = graph.to_sparse_csr(propagation)
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}

    node_feats = np.zeros((n, edge_feat_dim), dtype=np.float64)
    node_counts = np.zeros(n, dtype=np.float64)

    for edge_key, feat in edge_features.items():
        parts = edge_key.strip().split()
        if len(parts) == 2:
            ia = index_map.get(parts[0])
            ib = index_map.get(parts[1])
            if ia is not None and ib is not None:
                feat_arr = np.array(feat, dtype=np.float64)
                node_feats[ia] += feat_arr
                node_feats[ib] += feat_arr
                node_counts[ia] += 1
                node_counts[ib] += 1

    node_counts = np.maximum(node_counts, 1.0)
    node_feats /= node_counts[:, None]

    adj = csr_matrix(
        (vals.astype(np.float64), (rows.astype(np.int32), cols.astype(np.int32))),
        shape=(n, n),
    )

    H = node_feats
    for _ in range(num_iterations):
        H = (adj @ H)
        H = _postprocess_iteration(H.astype(np.float32), "l2", whiten).astype(np.float64)

    edge_emb = H.astype(np.float32)

    if combine == "concat":
        return np.concatenate([struct_emb, edge_emb], axis=1)
    elif combine == "mean":
        min_dim = min(struct_emb.shape[1], edge_emb.shape[1])
        return (struct_emb[:, :min_dim] + edge_emb[:, :min_dim]) / 2.0
    elif combine == "edge_only":
        return edge_emb
    else:
        raise ValueError(f"Unknown combine mode: '{combine}'. Use 'concat', 'mean', or 'edge_only'.")


class CleoraEmbedder:
    def __init__(
        self,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        num_iterations: Union[int, str] = DEFAULT_NUM_ITERATIONS,
        propagation: str = "left",
        normalization: str = "l2",
        columns: str = "complex::reflexive::node",
        seed: int = 0,
        hyperedge_trim_n: int = 16,
        num_workers: Optional[int] = None,
        whiten: bool = DEFAULT_WHITEN,
    ):
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.propagation = propagation
        self.normalization = normalization
        self.columns = columns
        self.seed = seed
        self.hyperedge_trim_n = hyperedge_trim_n
        self.num_workers = num_workers
        self.whiten = whiten
        self.graph_ = None
        self.embeddings_ = None
        self.entity_ids_ = None

    def fit(self, edges: List[str], y=None):
        self.graph_ = SparseMatrix.from_iterator(
            iter(edges), self.columns, self.hyperedge_trim_n, self.num_workers
        )
        self.embeddings_ = embed(
            self.graph_,
            feature_dim=self.feature_dim,
            num_iterations=self.num_iterations,
            propagation=self.propagation,
            normalization=self.normalization,
            seed=self.seed,
            num_workers=self.num_workers,
            whiten=self.whiten,
        )
        self.entity_ids_ = list(self.graph_.entity_ids)
        return self

    def transform(self, edges: Optional[List[str]] = None) -> np.ndarray:
        if self.embeddings_ is None:
            raise RuntimeError("Call fit() before transform()")
        if edges is None:
            return self.embeddings_
        index_map = {eid: i for i, eid in enumerate(self.entity_ids_)}
        seen = set()
        ordered_indices = []
        for edge in edges:
            for ent in edge.strip().split():
                if ent not in seen:
                    idx = index_map.get(ent)
                    if idx is not None:
                        seen.add(ent)
                        ordered_indices.append(idx)
        if not ordered_indices:
            raise ValueError("None of the entities in edges were found in the fitted graph")
        return self.embeddings_[ordered_indices]

    def fit_transform(self, edges: List[str], y=None) -> np.ndarray:
        return self.fit(edges, y).transform()

    def get_params(self, deep=True) -> Dict:
        return {
            "feature_dim": self.feature_dim,
            "num_iterations": self.num_iterations,
            "propagation": self.propagation,
            "normalization": self.normalization,
            "columns": self.columns,
            "seed": self.seed,
            "hyperedge_trim_n": self.hyperedge_trim_n,
            "num_workers": self.num_workers,
            "whiten": self.whiten,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


def _normalize(embeddings: np.ndarray, method: str) -> np.ndarray:
    if method == "l2":
        norms = np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms
    elif method == "l1":
        norms = np.linalg.norm(embeddings, ord=1, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms
    elif method == "spectral":
        norms = np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embeddings / norms
        u, s, vt = np.linalg.svd(normalized, full_matrices=False)
        return u * s
    elif method == "none":
        return embeddings
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'l2', 'l1', 'spectral', or 'none'.")


def _postprocess_iteration(
    embeddings: np.ndarray,
    normalization: str,
    whiten: bool,
) -> np.ndarray:
    embeddings = _normalize(embeddings, normalization)
    if whiten:
        embeddings = whiten_embeddings(embeddings)
    return embeddings


def _l1_normalize_inplace(embeddings: np.ndarray) -> None:
    for start in range(0, embeddings.shape[0], _ROW_CHUNK):
        block = embeddings[start:start + _ROW_CHUNK]
        norms = np.abs(block).sum(axis=1, keepdims=True)
        np.maximum(norms, 1e-10, out=norms)
        block /= norms


def _postprocess_inplace(
    embeddings: np.ndarray,
    normalization: str,
    whiten: bool,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    """Row normalisation and whitening of a C-contiguous float32 matrix, in place."""
    if normalization == "l2":
        _core.l2_normalize_inplace(embeddings, num_workers)
    elif normalization == "l1":
        _l1_normalize_inplace(embeddings)
    elif normalization != "none":
        raise ValueError(
            f"Unknown normalization method: {normalization}. Use 'l2', 'l1', 'spectral', or 'none'."
        )
    if whiten:
        whiten_embeddings(embeddings, num_workers=num_workers, inplace=True)
    return embeddings


def _mix_residual_inplace(target: np.ndarray, previous: np.ndarray, residual_weight: float) -> None:
    """target = (1 - w) * target + w * previous, without a full-size temporary."""
    weight = np.float32(residual_weight)
    keep = np.float32(1.0 - residual_weight)
    for start in range(0, target.shape[0], _ROW_CHUNK):
        block = target[start:start + _ROW_CHUNK]
        block *= keep
        block += weight * previous[start:start + _ROW_CHUNK]


def _compute_rmse(current: np.ndarray, previous: np.ndarray) -> float:
    total = 0.0
    for start in range(0, current.shape[0], _ROW_CHUNK):
        diff = current[start:start + _ROW_CHUNK].astype(np.float64) - previous[start:start + _ROW_CHUNK]
        total += float(np.einsum("ij,ij->", diff, diff))
    return float(np.sqrt(total / max(current.size, 1)))


def _whiten_embeddings_torch(embeddings):
    import torch

    n = embeddings.shape[0]
    if n <= 1:
        return embeddings.clone()

    mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean
    cov = centered.transpose(0, 1).matmul(centered) / max(n - 1, 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    scale = torch.rsqrt(torch.clamp(eigenvalues, min=1e-10))
    transform = eigenvectors * scale.unsqueeze(0)
    return centered.matmul(transform)
