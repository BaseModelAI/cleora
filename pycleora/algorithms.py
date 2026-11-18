"""Reference implementations of other graph-embedding methods.

They exist so that Cleora can be compared with well-known baselines without extra
dependencies (numpy and scipy only). They follow the published algorithms, but
they are compact, single-machine implementations: for production use of one of
these methods, prefer its authors' code.

All of them work on the plain weighted adjacency matrix of the graph (self-loops
removed), not on Cleora's Markov matrices.
"""

import numpy as np
from typing import Optional, Dict, List

try:
    from scipy.sparse import csr_matrix, diags, identity
except ImportError as error:  # SciPy is an optional dependency of pycleora
    raise ImportError(
        "pycleora.algorithms needs SciPy. Install it with: pip install 'pycleora[scipy]'"
    ) from error


def _graph_to_scipy(graph, markov_type: str = "left"):
    indptr, indices, data = graph.to_csr(markov_type)
    n = graph.num_entities
    return csr_matrix((data.astype(np.float64), indices, indptr), shape=(n, n))


def _graph_to_adjacency(graph, self_loops: bool = False):
    """Symmetric weighted adjacency matrix of the graph (float64 CSR)."""
    A = _graph_to_scipy(graph, "raw")
    if not self_loops:
        coo = A.tocoo()
        keep = coo.row != coo.col
        A = csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=A.shape)
    return A


def _degrees(A) -> np.ndarray:
    return np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1e-12)


def _row_normalized(A):
    return (diags(1.0 / _degrees(A)) @ A).tocsr()


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-10)).astype(np.float32)


def _pad_columns(matrix: np.ndarray, feature_dim: int) -> np.ndarray:
    if matrix.shape[1] >= feature_dim:
        return matrix[:, :feature_dim]
    pad = np.zeros((matrix.shape[0], feature_dim - matrix.shape[1]), dtype=matrix.dtype)
    return np.concatenate([matrix, pad], axis=1)


def _check_dense_size(n: int, max_nodes: Optional[int], method: str):
    if max_nodes is not None and n > max_nodes:
        gigabytes = n * n * 8 / 1e9
        raise ValueError(
            f"{method} builds dense {n} x {n} matrices (about {gigabytes:.0f} GB each), which is "
            f"refused above max_nodes={max_nodes}. Pass a larger max_nodes to run it anyway."
        )


def _randomized_svd(matrix, rank: int, seed: int = 0, n_iter: int = 5, oversample: int = 10):
    """Truncated SVD by randomized subspace iteration (Halko et al., 2011)."""
    n_rows, n_cols = matrix.shape
    rank = max(1, min(rank, n_rows, n_cols))
    size = min(rank + oversample, n_rows, n_cols)
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(matrix @ rng.standard_normal((n_cols, size)))
    for _ in range(n_iter):
        basis, _ = np.linalg.qr(matrix.T @ basis)
        basis, _ = np.linalg.qr(matrix @ basis)
    small = np.asarray((matrix.T @ basis).T)
    u_small, singular, vt = np.linalg.svd(small, full_matrices=False)
    return (basis @ u_small)[:, :rank], singular[:rank], vt[:rank]


def embed_prone(
    graph,
    feature_dim: int = 256,
    mu: float = 0.2,
    theta: float = 0.5,
    seed: int = 0,
    order: int = 10,
) -> np.ndarray:
    """ProNE (Zhang et al., IJCAI 2019).

    1. Sparse matrix factorisation: truncated SVD of the sparse matrix
       ``log P[i, j] - log(negative-sampling prior of j)`` defined on the edges.
    2. Spectral propagation: the result is filtered with a band-pass (Gaussian)
       graph filter, expanded in ``order`` Chebyshev terms.
    """
    from scipy.special import iv

    n = graph.num_entities
    A = _graph_to_adjacency(graph)
    if A.nnz == 0:
        return np.zeros((n, feature_dim), dtype=np.float32)

    # --- step 1: sparse matrix factorisation
    P = _row_normalized(A)
    prior = np.asarray(P.sum(axis=0)).ravel() ** 0.75
    prior = prior / prior.sum()
    F = P.copy()
    F.data = np.log(P.data) - np.log(np.maximum(prior[P.indices], 1e-300))
    u, singular, _ = _randomized_svd(F, feature_dim, seed=seed)
    embedding = _unit_rows(u * np.sqrt(singular)).astype(np.float64)

    # --- step 2: spectral propagation (Chebyshev expansion of a Gaussian filter)
    if order > 1:
        A_loop = (identity(n, format="csr") + A).tocsr()
        laplacian = identity(n, format="csr") - _row_normalized(A_loop)
        M = (laplacian - mu * identity(n, format="csr")).tocsr()

        term_prev = embedding
        term = M @ embedding
        term = 0.5 * (M @ term) - embedding
        filtered = iv(0, theta) * term_prev
        filtered -= 2 * iv(1, theta) * term
        for i in range(2, order):
            term_next = M @ term
            term_next = (M @ term_next - 2 * term) - term_prev
            if i % 2 == 0:
                filtered += 2 * iv(i, theta) * term_next
            else:
                filtered -= 2 * iv(i, theta) * term_next
            term_prev, term = term, term_next
        propagated = A_loop @ (embedding - filtered)
        u, singular, _ = np.linalg.svd(propagated, full_matrices=False)
        embedding = u * np.sqrt(singular)

    return _unit_rows(_pad_columns(embedding, feature_dim))


def embed_randne(
    graph,
    feature_dim: int = 256,
    num_iterations: int = 3,
    weights: Optional[List[float]] = None,
    seed: int = 0,
) -> np.ndarray:
    """RandNE (Zhang et al., ICDM 2018): iterative Gaussian random projection.

    ``U_0`` is an orthogonalised Gaussian matrix, ``U_i = S U_{i-1}`` with the
    symmetrically normalised adjacency ``S``, and the embedding is
    ``sum_i weights[i] * U_i``. ``num_iterations`` is the order ``q``; the default
    weights ``[1, 1e2, 1e4, 1e5]`` are the ones recommended by the authors for
    node classification (higher orders dominate).
    """
    n = graph.num_entities
    A = _graph_to_adjacency(graph)
    inv_sqrt = 1.0 / np.sqrt(_degrees(A))
    S = (diags(inv_sqrt) @ A @ diags(inv_sqrt)).tocsr()

    if weights is None:
        weights = [1.0, 1e2, 1e4, 1e5]
        if num_iterations + 1 > len(weights):
            weights = weights + [weights[-1]] * (num_iterations + 1 - len(weights))
    if len(weights) < num_iterations + 1:
        raise ValueError(
            f"weights needs {num_iterations + 1} values (orders 0..{num_iterations}), got {len(weights)}"
        )

    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((n, feature_dim)) / np.sqrt(feature_dim)
    if n >= feature_dim:
        projection, _ = np.linalg.qr(projection)

    current = projection
    embedding = weights[0] * current
    for i in range(1, num_iterations + 1):
        current = S @ current
        embedding = embedding + weights[i] * current

    return _unit_rows(embedding)


def embed_hope(
    graph,
    feature_dim: int = 256,
    beta: float = 0.5,
    max_nodes: Optional[int] = 20000,
) -> np.ndarray:
    """HOPE (Ou et al., KDD 2016) with the Katz proximity.

    Factorises ``S = (I - b A)^-1 (b A)`` where ``b = beta / spectral_radius(A)``,
    so ``beta`` in (0, 1) is the decay relative to the largest value for which the
    Katz series converges. Source and target halves are concatenated.
    Needs dense ``n x n`` matrices.
    """
    from scipy.sparse.linalg import eigsh

    if not 0 < beta < 1:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    n = graph.num_entities
    _check_dense_size(n, max_nodes, "HOPE")
    A = _graph_to_adjacency(graph)
    if n > 2 and A.nnz > 0:
        radius = float(abs(eigsh(A, k=1, which="LA", return_eigenvectors=False)[0]))
    else:
        radius = float(np.abs(A.toarray()).sum(axis=1).max()) if A.nnz else 1.0
    decay = beta / max(radius, 1e-12)

    dense = A.toarray()
    katz = np.linalg.solve(np.eye(n) - decay * dense, decay * dense)

    k = max(1, min(feature_dim // 2, n))
    u, singular, vt = np.linalg.svd(katz, full_matrices=False)
    sqrt_s = np.sqrt(singular[:k])
    result = np.concatenate([u[:, :k] * sqrt_s, vt[:k].T * sqrt_s], axis=1)
    return _unit_rows(_pad_columns(result, feature_dim))


def embed_netmf(
    graph,
    feature_dim: int = 256,
    window_size: int = 5,
    negative_samples: float = 1.0,
    max_nodes: Optional[int] = 20000,
) -> np.ndarray:
    """NetMF (Qiu et al., WSDM 2018), exact version for small windows.

    Factorises ``log max(M, 1)`` with
    ``M = vol(G) / (b T) * (sum_{r=1..T} P^r) D^-1``, the matrix DeepWalk
    implicitly factorises. Needs dense ``n x n`` matrices.
    """
    n = graph.num_entities
    _check_dense_size(n, max_nodes, "NetMF")
    A = _graph_to_adjacency(graph).toarray()
    degrees = np.maximum(A.sum(axis=1), 1e-12)
    volume = degrees.sum()

    P = A / degrees[:, None]
    power = np.eye(n)
    total = np.zeros((n, n))
    for _ in range(window_size):
        power = power @ P
        total += power

    M = (volume / (negative_samples * window_size)) * total / degrees[None, :]
    M = np.log(np.maximum(M, 1.0))

    # M is symmetric up to rounding; its SVD follows from the eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh((M + M.T) / 2)
    order = np.argsort(-np.abs(eigenvalues))[: min(feature_dim, n)]
    result = eigenvectors[:, order] * np.sqrt(np.abs(eigenvalues[order]))
    return _unit_rows(_pad_columns(result, feature_dim))


def embed_grarep(
    graph,
    feature_dim: int = 256,
    max_step: int = 4,
    max_nodes: Optional[int] = 20000,
) -> np.ndarray:
    """GraRep (Cao et al., CIKM 2015).

    For every step ``k = 1..max_step`` factorises the positive log-probability
    matrix of ``k``-step transitions, ``max(log(P^k[i, j] / column mean), 0)``, and
    concatenates the factors. Needs dense ``n x n`` matrices.
    """
    n = graph.num_entities
    _check_dense_size(n, max_nodes, "GraRep")
    A = _graph_to_adjacency(graph).toarray()
    P = A / np.maximum(A.sum(axis=1), 1e-12)[:, None]

    dim_per_step = max(feature_dim // max_step, 1)
    parts = []
    P_k = np.eye(n)
    for _ in range(max_step):
        P_k = P_k @ P
        column_mean = np.maximum(P_k.sum(axis=0, keepdims=True) / n, 1e-300)
        with np.errstate(divide="ignore"):
            M = np.log(P_k / column_mean)
        M[~np.isfinite(M)] = 0.0
        M = np.maximum(M, 0.0)
        u, singular, _ = np.linalg.svd(M, full_matrices=False)
        k = min(dim_per_step, n)
        parts.append(_unit_rows(u[:, :k] * np.sqrt(singular[:k])).astype(np.float64))

    return _unit_rows(_pad_columns(np.concatenate(parts, axis=1), feature_dim))


class _WalkSampler:
    """Vectorised random walks on a weighted graph (all walks advance together)."""

    def __init__(self, A, seed: int):
        A = A.tocsr()
        A.sort_indices()
        self.n = A.shape[0]
        self.indptr = A.indptr.astype(np.int64)
        self.indices = A.indices.astype(np.int64)
        self.degree = np.diff(self.indptr)
        rows = np.repeat(np.arange(self.n, dtype=np.int64), self.degree)
        row_totals = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1e-300)
        within = np.cumsum(A.data) - np.repeat(np.cumsum(A.data)[self.indptr[:-1] - 1] * (self.indptr[:-1] > 0), self.degree)
        # position k of row r holds r + (cumulative share of the row's weight), in (r, r + 1]
        self.cumulative = rows + np.minimum(within / row_totals[rows], 1.0)
        # (row, column) keys, sorted, for membership tests
        self.edge_keys = rows * self.n + self.indices
        self.rng = np.random.default_rng(seed)

    def step(self, current: np.ndarray) -> np.ndarray:
        """One weighted random step from every node in ``current``."""
        draw = current + self.rng.random(len(current)) * (1.0 - 1e-12)
        position = np.searchsorted(self.cumulative, draw, side="right")
        position = np.clip(position, self.indptr[current], self.indptr[current + 1] - 1)
        return self.indices[position]

    def connected(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        keys = a * self.n + b
        position = np.searchsorted(self.edge_keys, keys)
        position = np.minimum(position, len(self.edge_keys) - 1)
        return self.edge_keys[position] == keys

    def walks(self, num_walks: int, walk_length: int, p: float, q: float) -> np.ndarray:
        """Returns an array of shape (num_walks * active_nodes, walk_length)."""
        starts = np.flatnonzero(self.degree > 0)
        result = []
        biased = not (p == 1.0 and q == 1.0)
        upper = max(1.0, 1.0 / p, 1.0 / q)
        for _ in range(num_walks):
            walk = np.empty((len(starts), walk_length), dtype=np.int64)
            walk[:, 0] = starts
            current = starts
            previous = None
            for position in range(1, walk_length):
                if not biased or previous is None:
                    nxt = self.step(current)
                else:
                    # node2vec second-order walk by rejection sampling
                    nxt = np.empty_like(current)
                    pending = np.arange(len(current))
                    while len(pending):
                        candidate = self.step(current[pending])
                        back = candidate == previous[pending]
                        near = self.connected(previous[pending], candidate)
                        accept_probability = np.where(back, 1.0 / p, np.where(near, 1.0, 1.0 / q)) / upper
                        accepted = self.rng.random(len(pending)) < accept_probability
                        nxt[pending[accepted]] = candidate[accepted]
                        pending = pending[~accepted]
                walk[:, position] = nxt
                previous, current = current, nxt
            result.append(walk)
        return np.concatenate(result, axis=0) if result else np.empty((0, walk_length), dtype=np.int64)


def _walks_to_embeddings(walks: np.ndarray, n: int, feature_dim: int, window_size: int, seed: int) -> np.ndarray:
    """Truncated SVD of the positive PMI matrix of walk co-occurrences."""
    from scipy.sparse import coo_matrix

    counts = csr_matrix((n, n), dtype=np.float64)
    for offset in range(1, window_size + 1):
        if offset >= walks.shape[1]:
            break
        left = walks[:, :-offset].ravel()
        right = walks[:, offset:].ravel()
        pairs = coo_matrix((np.ones(len(left)), (left, right)), shape=(n, n)).tocsr()
        counts = counts + pairs + pairs.T
    counts = counts.tocoo()
    if counts.nnz == 0:
        return np.zeros((n, feature_dim), dtype=np.float32)

    total = counts.data.sum()
    row_sums = np.maximum(np.asarray(counts.tocsr().sum(axis=1)).ravel(), 1e-300)
    pmi = np.log(counts.data * total / (row_sums[counts.row] * row_sums[counts.col]))
    positive = pmi > 0
    ppmi = csr_matrix((pmi[positive], (counts.row[positive], counts.col[positive])), shape=(n, n))

    u, singular, _ = _randomized_svd(ppmi, feature_dim, seed=seed)
    return _unit_rows(_pad_columns(u * np.sqrt(singular), feature_dim))


def embed_deepwalk(
    graph,
    feature_dim: int = 256,
    num_walks: int = 10,
    walk_length: int = 80,
    window_size: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """DeepWalk-style embedding (Perozzi et al., KDD 2014), matrix-factorisation variant.

    Uniform (weight-proportional) random walks are sampled as in DeepWalk, but
    instead of training skip-gram with negative sampling, the positive PMI matrix
    of the walk co-occurrences is factorised with a truncated SVD - the objective
    skip-gram optimises implicitly (Levy & Goldberg, 2014). It is deterministic
    for a given seed and needs no training loop; it is not the original SGD-based
    implementation.
    """
    return embed_node2vec(graph, feature_dim, num_walks, walk_length, window_size, 1.0, 1.0, seed)


def embed_node2vec(
    graph,
    feature_dim: int = 256,
    num_walks: int = 10,
    walk_length: int = 80,
    window_size: int = 5,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """node2vec-style embedding (Grover & Leskovec, KDD 2016), matrix-factorisation variant.

    Biased second-order random walks with return parameter ``p`` and in-out
    parameter ``q``, followed by a truncated SVD of the positive PMI matrix of the
    walk co-occurrences (see :func:`embed_deepwalk`). With ``p = q = 1`` the walks
    are DeepWalk's and the two functions return the same embedding.
    """
    if p <= 0 or q <= 0:
        raise ValueError(f"p and q must be positive, got p={p}, q={q}")
    A = _graph_to_adjacency(graph)
    n = graph.num_entities
    if A.nnz == 0:
        return np.zeros((n, feature_dim), dtype=np.float32)
    sampler = _WalkSampler(A, seed)
    walks = sampler.walks(num_walks, walk_length, p, q)
    return _walks_to_embeddings(walks, n, feature_dim, window_size, seed)


def list_algorithms() -> List[Dict]:
    return [
        {"name": "prone", "function": "embed_prone",
         "description": "ProNE: sparse matrix factorisation followed by spectral propagation with Chebyshev polynomials."},
        {"name": "randne", "function": "embed_randne",
         "description": "RandNE: iterative Gaussian random projection of high-order proximities."},
        {"name": "hope", "function": "embed_hope",
         "description": "HOPE: factorisation of the Katz proximity matrix (dense, small graphs)."},
        {"name": "netmf", "function": "embed_netmf",
         "description": "NetMF: explicit factorisation of the matrix DeepWalk factorises implicitly (dense, small graphs)."},
        {"name": "grarep", "function": "embed_grarep",
         "description": "GraRep: factorisation of k-step transition matrices, concatenated (dense, small graphs)."},
        {"name": "deepwalk", "function": "embed_deepwalk",
         "description": "DeepWalk walks + PPMI/SVD factorisation (matrix-factorisation variant, not skip-gram)."},
        {"name": "node2vec", "function": "embed_node2vec",
         "description": "node2vec biased walks (p, q) + PPMI/SVD factorisation (matrix-factorisation variant, not skip-gram)."},
    ]
