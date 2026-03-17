import numpy as np
from typing import Optional, List, Dict, Callable, Tuple

from .pycleora import SparseMatrix


def embed_using_baseline_cleora(graph, feature_dim: int, iter: int):
    embeddings = graph.initialize_deterministically(feature_dim)
    for i in range(iter):
        embeddings = graph.left_markov_propagate(embeddings)
        embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    return embeddings


def _validate_propagation(propagation: str):
    if propagation not in ("left", "symmetric"):
        raise ValueError(f"Unknown propagation type: '{propagation}'. Use 'left' or 'symmetric'.")


def _get_propagate_fn(graph: SparseMatrix, propagation: str):
    _validate_propagation(propagation)
    return (
        graph.symmetric_markov_propagate
        if propagation == "symmetric"
        else graph.left_markov_propagate
    )


def _to_scipy_sparse(graph: SparseMatrix, markov_type: str = "left"):
    from scipy.sparse import csr_matrix
    rows, cols, vals, n_rows, n_cols = graph.to_sparse_csr(markov_type)
    return csr_matrix(
        (vals, (rows.astype(np.int32), cols.astype(np.int32))),
        shape=(n_rows, n_cols),
    )


def embed(
    graph: SparseMatrix,
    feature_dim: int = 128,
    num_iterations: int = 4,
    propagation: str = "left",
    normalization: str = "l2",
    seed: int = 0,
    initial_embeddings: Optional[np.ndarray] = None,
    num_workers: Optional[int] = None,
    callback: Optional[Callable[[int, np.ndarray], None]] = None,
) -> np.ndarray:
    propagate_fn = _get_propagate_fn(graph, propagation)

    if initial_embeddings is not None:
        embeddings = initial_embeddings.astype(np.float32)
        if embeddings.shape[0] != graph.num_entities:
            raise ValueError(
                f"initial_embeddings has {embeddings.shape[0]} rows but graph has {graph.num_entities} entities"
            )
    else:
        embeddings = graph.initialize_deterministically(feature_dim, seed)

    for i in range(num_iterations):
        embeddings = propagate_fn(embeddings, num_workers=num_workers)
        embeddings = _normalize(embeddings, normalization)

        if callback is not None:
            callback(i, embeddings)

    return embeddings


def embed_with_attention(
    graph: SparseMatrix,
    feature_dim: int = 128,
    num_iterations: int = 4,
    propagation: str = "left",
    normalization: str = "l2",
    attention_temperature: float = 1.0,
    seed: int = 0,
    num_workers: Optional[int] = None,
    callback: Optional[Callable[[int, np.ndarray], None]] = None,
) -> np.ndarray:
    _validate_propagation(propagation)

    if attention_temperature <= 0:
        raise ValueError(f"attention_temperature must be positive, got {attention_temperature}")

    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")

    embeddings = graph.initialize_deterministically(feature_dim, seed)
    propagate_fn = _get_propagate_fn(graph, propagation)

    embeddings = propagate_fn(embeddings, num_workers=num_workers)
    embeddings = _normalize(embeddings, normalization)

    if callback is not None:
        callback(0, embeddings)

    if num_iterations == 1:
        return embeddings

    from scipy.sparse import csr_matrix, diags
    adj = _to_scipy_sparse(graph, propagation)
    adj_rows, adj_cols = adj.nonzero()

    for i in range(1, num_iterations):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        emb_normed = embeddings / norms

        dot_products = np.sum(emb_normed[adj_rows] * emb_normed[adj_cols], axis=1)
        attention_scores = dot_products / attention_temperature

        row_max = np.full(adj.shape[0], -np.inf, dtype=np.float64)
        np.maximum.at(row_max, adj_rows, attention_scores)
        row_max[row_max == -np.inf] = 0.0

        shifted = attention_scores - row_max[adj_rows]
        exp_scores = np.exp(shifted)
        attn_exp = csr_matrix(
            (exp_scores, (adj_rows, adj_cols)),
            shape=adj.shape,
        )
        row_sums = np.array(attn_exp.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1e-10)
        inv_sums = 1.0 / row_sums
        norm_matrix = diags(inv_sums) @ attn_exp

        weighted_adj = norm_matrix.multiply(adj)
        row_sums_w = np.array(weighted_adj.sum(axis=1)).flatten()
        row_sums_w = np.maximum(row_sums_w, 1e-10)
        weighted_adj = diags(1.0 / row_sums_w) @ weighted_adj

        embeddings = (weighted_adj @ embeddings).astype(np.float32)
        embeddings = _normalize(embeddings, normalization)

        if callback is not None:
            callback(i, embeddings)

    return embeddings


def embed_multiscale(
    graph: SparseMatrix,
    feature_dim: int = 128,
    scales: List[int] = None,
    propagation: str = "left",
    normalization: str = "l2",
    seed: int = 0,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    propagate_fn = _get_propagate_fn(graph, propagation)

    if scales is None:
        scales = [1, 2, 4, 8]

    if not scales or not all(isinstance(s, int) and s > 0 for s in scales):
        raise ValueError("scales must be a non-empty list of positive integers")

    embeddings = graph.initialize_deterministically(feature_dim, seed)
    all_embeddings = []

    current_iter = 0
    for scale in sorted(scales):
        while current_iter < scale:
            embeddings = propagate_fn(embeddings, num_workers=num_workers)
            embeddings = _normalize(embeddings, normalization)
            current_iter += 1
        all_embeddings.append(embeddings.copy())

    return np.concatenate(all_embeddings, axis=1)


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


def embed_inductive(
    trained_graph: SparseMatrix,
    trained_embeddings: np.ndarray,
    existing_edges: List[str],
    new_edges: List[str],
    columns: str,
    num_iterations: int = 4,
    propagation: str = "left",
    normalization: str = "l2",
    hyperedge_trim_n: int = 16,
    num_workers: Optional[int] = None,
) -> Tuple[SparseMatrix, np.ndarray]:
    if trained_embeddings.shape[0] != trained_graph.num_entities:
        raise ValueError(
            f"trained_embeddings has {trained_embeddings.shape[0]} rows but graph has {trained_graph.num_entities} entities"
        )

    updated_graph = update_graph(
        existing_edges, new_edges, columns, hyperedge_trim_n, num_workers
    )

    old_index_map = {eid: i for i, eid in enumerate(trained_graph.entity_ids)}

    dim = trained_embeddings.shape[1]
    init_embeddings = np.random.randn(updated_graph.num_entities, dim).astype(np.float32) * 0.01

    for i, eid in enumerate(updated_graph.entity_ids):
        if eid in old_index_map:
            init_embeddings[i] = trained_embeddings[old_index_map[eid]]

    updated_embeddings = embed(
        updated_graph,
        feature_dim=dim,
        num_iterations=num_iterations,
        propagation=propagation,
        normalization=normalization,
        initial_embeddings=init_embeddings,
        num_workers=num_workers,
    )

    return updated_graph, updated_embeddings


def propagate_gpu(
    graph: SparseMatrix,
    embeddings: np.ndarray,
    num_iterations: int = 4,
    propagation: str = "left",
    normalization: str = "l2",
    device: str = "cuda",
    callback: Optional[Callable[[int, np.ndarray], None]] = None,
) -> np.ndarray:
    _validate_propagation(propagation)

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

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = embeddings / norms

    query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-10)
    similarities = normalized @ query_norm

    if exclude_self:
        similarities[query_idx] = -1.0

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "entity_id": graph.entity_ids[idx],
            "index": int(idx),
            "similarity": float(similarities[idx]),
        })
    return results


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
