import numpy as np
from typing import List, Optional


def combine(
    embeddings_list: List[np.ndarray],
    method: str = "concat",
    weights: Optional[List[float]] = None,
    target_dim: Optional[int] = None,
) -> np.ndarray:
    """Combine multiple embedding matrices into a single matrix.

    Note: Entity alignment between different graphs is the caller's responsibility.
    Rows must correspond to the same entities across all matrices.

    Args:
        embeddings_list: List of embedding matrices. All must have the same
            number of rows (entities).
        method: Combination method. One of:
            - "concat": Horizontal concatenation.
            - "mean": Element-wise average (requires same dimensions).
            - "weighted": Weighted average (requires same dimensions and
              ``weights`` parameter).
            - "svd": Concatenation followed by SVD dimensionality reduction
              to ``target_dim``.
        weights: Weights for the "weighted" method. Must sum to a positive
            value and have the same length as ``embeddings_list``.
        target_dim: Target dimensionality for the "svd" method.

    Returns:
        Combined embedding matrix as a float32 numpy array.
    """
    if not embeddings_list:
        raise ValueError("embeddings_list must be non-empty")

    n_rows = embeddings_list[0].shape[0]
    for i, emb in enumerate(embeddings_list):
        if emb.ndim != 2:
            raise ValueError(f"Embedding at index {i} is not 2-dimensional")
        if emb.shape[0] != n_rows:
            raise ValueError(
                f"Embedding at index {i} has {emb.shape[0]} rows, "
                f"expected {n_rows}"
            )

    if method == "concat":
        return np.concatenate(embeddings_list, axis=1).astype(np.float32)

    elif method == "mean":
        _check_same_dims(embeddings_list)
        stacked = np.stack(embeddings_list, axis=0)
        return stacked.mean(axis=0).astype(np.float32)

    elif method == "weighted":
        _check_same_dims(embeddings_list)
        if weights is None:
            raise ValueError("weights parameter is required for method='weighted'")
        if len(weights) != len(embeddings_list):
            raise ValueError(
                f"weights has {len(weights)} elements but "
                f"embeddings_list has {len(embeddings_list)} elements"
            )
        w_sum = sum(weights)
        if w_sum <= 0:
            raise ValueError("weights must sum to a positive value")
        norm_weights = [w / w_sum for w in weights]
        result = np.zeros_like(embeddings_list[0], dtype=np.float64)
        for w, emb in zip(norm_weights, embeddings_list):
            result += w * emb
        return result.astype(np.float32)

    elif method == "svd":
        if target_dim is None:
            raise ValueError("target_dim parameter is required for method='svd'")
        if not isinstance(target_dim, int) or target_dim < 1:
            raise ValueError(f"target_dim must be a positive integer, got {target_dim}")
        concatenated = np.concatenate(embeddings_list, axis=1).astype(np.float64)
        mean = concatenated.mean(axis=0)
        centered = concatenated - mean
        U, S, _ = np.linalg.svd(centered, full_matrices=False)
        k = min(target_dim, U.shape[1])
        reduced = U[:, :k] * S[:k]
        if k < target_dim:
            pad = np.zeros((n_rows, target_dim - k), dtype=np.float64)
            reduced = np.concatenate([reduced, pad], axis=1)
        return reduced.astype(np.float32)

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Supported methods: 'concat', 'mean', 'weighted', 'svd'"
        )


def _check_same_dims(embeddings_list: List[np.ndarray]) -> None:
    dims = embeddings_list[0].shape[1]
    for i, emb in enumerate(embeddings_list):
        if emb.shape[1] != dims:
            raise ValueError(
                f"Embedding at index {i} has {emb.shape[1]} columns, "
                f"expected {dims}. All embeddings must have the same "
                f"dimensions for this method."
            )
