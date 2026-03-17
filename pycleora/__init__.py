import numpy as np
from typing import Optional, List, Dict, Callable

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
