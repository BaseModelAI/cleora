import numpy as np
from typing import Optional, Tuple, Dict, List
from scipy.sparse import csr_matrix, diags, eye


def _graph_to_scipy(graph, markov_type: str = "left"):
    rows, cols, vals, n, _ = graph.to_sparse_csr(markov_type)
    return csr_matrix(
        (vals.astype(np.float64), (rows.astype(np.int32), cols.astype(np.int32))),
        shape=(n, n),
    )


def _graph_to_adjacency(graph):
    rows, cols, vals, n, _ = graph.to_sparse_csr()
    A = csr_matrix(
        (vals.astype(np.float64), (rows.astype(np.int32), cols.astype(np.int32))),
        shape=(n, n),
    )
    return A


def embed_prone(
    graph,
    feature_dim: int = 128,
    mu: float = 0.2,
    theta: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    n = graph.num_entities
    A = _graph_to_adjacency(graph)

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-10)
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    L_norm = eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, feature_dim)).astype(np.float64)

    U = R.copy()
    chebyshev_prev = R.copy()
    chebyshev_curr = (L_norm @ R).toarray() if hasattr(L_norm @ R, 'toarray') else np.asarray(L_norm @ R)

    for k in range(2, min(10, n)):
        chebyshev_next = 2 * (L_norm @ chebyshev_curr) - chebyshev_prev
        if hasattr(chebyshev_next, 'toarray'):
            chebyshev_next = chebyshev_next.toarray()
        coeff = np.exp(-theta * k) * mu
        U += coeff * chebyshev_next
        chebyshev_prev = chebyshev_curr
        chebyshev_curr = chebyshev_next

    u_svd, s_svd, _ = np.linalg.svd(U, full_matrices=False)
    k = min(feature_dim, u_svd.shape[1])
    result = u_svd[:, :k] * np.sqrt(np.maximum(s_svd[:k], 0))

    if result.shape[1] < feature_dim:
        pad = np.zeros((n, feature_dim - result.shape[1]), dtype=np.float64)
        result = np.concatenate([result, pad], axis=1)

    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (result / norms).astype(np.float32)


def embed_randne(
    graph,
    feature_dim: int = 128,
    num_iterations: int = 3,
    weights: Optional[List[float]] = None,
    seed: int = 0,
) -> np.ndarray:
    n = graph.num_entities
    A = _graph_to_adjacency(graph)

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-10)
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    if weights is None:
        weights = [1.0 / (2 ** i) for i in range(num_iterations + 1)]

    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, feature_dim)).astype(np.float64)

    U = weights[0] * R
    current = R.copy()

    for i in range(num_iterations):
        current = A_norm @ current
        if hasattr(current, 'toarray'):
            current = current.toarray()
        w = weights[i + 1] if i + 1 < len(weights) else weights[-1]
        U += w * current

    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (U / norms).astype(np.float32)


def embed_hope(
    graph,
    feature_dim: int = 128,
    beta: float = 0.1,
) -> np.ndarray:
    n = graph.num_entities
    A = _graph_to_adjacency(graph)

    S = eye(n) - beta * A
    try:
        from scipy.sparse.linalg import inv as sparse_inv
        S_inv = sparse_inv(S.tocsc())
    except Exception:
        S_dense = S.toarray()
        S_inv = csr_matrix(np.linalg.inv(S_dense))

    M = S_inv - eye(n)

    k = min(feature_dim // 2, n - 1)
    try:
        from scipy.sparse.linalg import svds
        u, s, vt = svds(M, k=k)
        order = np.argsort(-s)
        u = u[:, order]
        s = s[order]
        vt = vt[order, :]
    except Exception:
        M_dense = M.toarray() if hasattr(M, 'toarray') else np.asarray(M)
        u, s, vt = np.linalg.svd(M_dense, full_matrices=False)
        u = u[:, :k]
        s = s[:k]
        vt = vt[:k, :]

    sqrt_s = np.sqrt(np.maximum(s, 0))
    source_emb = u * sqrt_s
    target_emb = vt.T * sqrt_s

    result = np.concatenate([source_emb, target_emb], axis=1)
    if result.shape[1] < feature_dim:
        pad = np.zeros((n, feature_dim - result.shape[1]), dtype=np.float64)
        result = np.concatenate([result, pad], axis=1)
    elif result.shape[1] > feature_dim:
        result = result[:, :feature_dim]

    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (result / norms).astype(np.float32)


def embed_netmf(
    graph,
    feature_dim: int = 128,
    window_size: int = 5,
    negative_samples: float = 1.0,
) -> np.ndarray:
    n = graph.num_entities
    A = _graph_to_adjacency(graph)

    degrees = np.array(A.sum(axis=1)).flatten()
    vol = degrees.sum()
    degrees = np.maximum(degrees, 1e-10)

    D_inv = diags(1.0 / degrees)
    P = D_inv @ A

    M_sum = csr_matrix((n, n), dtype=np.float64)
    P_power = eye(n, dtype=np.float64)

    for t in range(1, window_size + 1):
        P_power = P_power @ P
        M_sum = M_sum + P_power

    M_sum = M_sum / window_size

    D_diag = diags(degrees)
    M = (vol / negative_samples) * D_inv @ M_sum @ D_diag

    if hasattr(M, 'toarray'):
        M_dense = M.toarray()
    else:
        M_dense = np.asarray(M)

    M_dense = np.maximum(M_dense, 1.0)
    M_log = np.log(M_dense)

    u, s, vt = np.linalg.svd(M_log, full_matrices=False)
    k = min(feature_dim, n)
    result = u[:, :k] * np.sqrt(np.maximum(s[:k], 0))

    if result.shape[1] < feature_dim:
        pad = np.zeros((n, feature_dim - result.shape[1]), dtype=np.float64)
        result = np.concatenate([result, pad], axis=1)

    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (result / norms).astype(np.float32)


def embed_grarep(
    graph,
    feature_dim: int = 128,
    max_step: int = 4,
) -> np.ndarray:
    n = graph.num_entities
    A = _graph_to_adjacency(graph)

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-10)
    D_inv = diags(1.0 / degrees)
    P = D_inv @ A

    dim_per_step = max(feature_dim // max_step, 1)
    all_embs = []

    P_k = P.copy()
    for step in range(1, max_step + 1):
        if hasattr(P_k, 'toarray'):
            M = P_k.toarray()
        else:
            M = np.asarray(P_k)

        M = np.maximum(M, 1e-10)
        M_log = np.log(M) - np.log(1e-10)

        u, s, vt = np.linalg.svd(M_log, full_matrices=False)
        k = min(dim_per_step, n)
        step_emb = u[:, :k] * np.sqrt(np.maximum(s[:k], 0))
        all_embs.append(step_emb)

        if step < max_step:
            P_k = P_k @ P

    result = np.concatenate(all_embs, axis=1)
    if result.shape[1] > feature_dim:
        result = result[:, :feature_dim]
    elif result.shape[1] < feature_dim:
        n = result.shape[0]
        pad = np.zeros((n, feature_dim - result.shape[1]), dtype=np.float64)
        result = np.concatenate([result, pad], axis=1)

    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (result / norms).astype(np.float32)


def list_algorithms() -> List[Dict]:
    return [
        {"name": "prone", "function": "embed_prone",
         "description": "ProNE: Spectral propagation with Chebyshev polynomials. Fast and high quality."},
        {"name": "randne", "function": "embed_randne",
         "description": "RandNE: Random projection embedding. Extremely fast, good for very large graphs."},
        {"name": "hope", "function": "embed_hope",
         "description": "HOPE: High-Order Proximity Embedding. Asymmetric, good for directed graphs."},
        {"name": "netmf", "function": "embed_netmf",
         "description": "NetMF: Network Matrix Factorization. Theoretical generalization of DeepWalk."},
        {"name": "grarep", "function": "embed_grarep",
         "description": "GraRep: Multi-scale matrix factorization with k-step transitions."},
    ]
