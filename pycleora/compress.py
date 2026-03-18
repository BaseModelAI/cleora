import numpy as np
from typing import Dict, Optional


def pca_compress(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    if target_dim > embeddings.shape[1]:
        raise ValueError(
            f"target_dim ({target_dim}) cannot exceed embedding dimension ({embeddings.shape[1]})"
        )

    centered = embeddings - np.mean(embeddings, axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    return (u[:, :target_dim] * s[:target_dim])


def random_projection(
    embeddings: np.ndarray,
    target_dim: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")

    rng = np.random.RandomState(seed)
    original_dim = embeddings.shape[1]
    projection_matrix = rng.randn(original_dim, target_dim) / np.sqrt(target_dim)
    return embeddings @ projection_matrix


class PQIndex:
    def __init__(
        self,
        codes: np.ndarray,
        codebooks: np.ndarray,
        num_subspaces: int,
        subspace_dim: int,
        original_shape: tuple,
    ):
        self._codes = codes
        self._codebooks = codebooks
        self._num_subspaces = num_subspaces
        self._subspace_dim = subspace_dim
        self._original_shape = original_shape

    def reconstruct(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        if indices is not None:
            codes = self._codes[indices]
        else:
            codes = self._codes

        n = codes.shape[0]
        dim = self._num_subspaces * self._subspace_dim
        result = np.empty((n, dim), dtype=np.float32)

        for m in range(self._num_subspaces):
            start = m * self._subspace_dim
            end = start + self._subspace_dim
            result[:, start:end] = self._codebooks[m, codes[:, m]]

        return result

    def search(self, query: np.ndarray, top_k: int = 10) -> Dict:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        q_norm = np.linalg.norm(query)
        if q_norm > 1e-10:
            query_normalized = query / q_norm
        else:
            query_normalized = query

        dist_tables = np.empty(
            (self._num_subspaces, self._codebooks.shape[1]), dtype=np.float32
        )
        for m in range(self._num_subspaces):
            start = m * self._subspace_dim
            end = start + self._subspace_dim
            q_sub = query_normalized[start:end]
            centroid_norms = np.linalg.norm(self._codebooks[m], axis=1, keepdims=True)
            centroid_norms = np.maximum(centroid_norms, 1e-10)
            normalized_centroids = self._codebooks[m] / centroid_norms
            dist_tables[m] = normalized_centroids @ q_sub

        n = self._codes.shape[0]
        scores = np.zeros(n, dtype=np.float32)
        for m in range(self._num_subspaces):
            scores += dist_tables[m, self._codes[:, m]]

        top_k_clamped = min(top_k, n)
        top_indices = np.argpartition(scores, -top_k_clamped)[-top_k_clamped:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return {
            "indices": top_indices,
            "scores": scores[top_indices],
        }


def product_quantize(
    embeddings: np.ndarray,
    num_subspaces: int = 8,
    num_centroids: int = 256,
    max_iter: int = 20,
    seed: Optional[int] = None,
) -> PQIndex:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    if num_subspaces <= 0:
        raise ValueError(f"num_subspaces must be positive, got {num_subspaces}")
    if num_centroids <= 0:
        raise ValueError(f"num_centroids must be positive, got {num_centroids}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")

    n, dim = embeddings.shape
    if dim % num_subspaces != 0:
        raise ValueError(
            f"Embedding dimension ({dim}) must be divisible by num_subspaces ({num_subspaces})"
        )

    subspace_dim = dim // num_subspaces
    rng = np.random.RandomState(seed)

    codebooks = np.empty(
        (num_subspaces, num_centroids, subspace_dim), dtype=np.float32
    )
    codes = np.empty((n, num_subspaces), dtype=np.uint8 if num_centroids <= 256 else np.uint16)

    for m in range(num_subspaces):
        start = m * subspace_dim
        end = start + subspace_dim
        sub_data = embeddings[:, start:end].astype(np.float32)

        init_indices = rng.choice(n, size=min(num_centroids, n), replace=False)
        centroids = sub_data[init_indices].copy()
        if num_centroids > n:
            extra = num_centroids - n
            centroids = np.vstack([
                centroids,
                sub_data[rng.choice(n, size=extra, replace=True)]
                + rng.randn(extra, subspace_dim).astype(np.float32) * 0.01,
            ])

        for _ in range(max_iter):
            dists = (
                np.sum(sub_data ** 2, axis=1, keepdims=True)
                - 2 * sub_data @ centroids.T
                + np.sum(centroids ** 2, axis=1)
            )
            assignments = np.argmin(dists, axis=1)

            new_centroids = np.empty_like(centroids)
            for c in range(num_centroids):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = sub_data[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        final_dists = (
            np.sum(sub_data ** 2, axis=1, keepdims=True)
            - 2 * sub_data @ centroids.T
            + np.sum(centroids ** 2, axis=1)
        )
        codes[:, m] = np.argmin(final_dists, axis=1)
        codebooks[m] = centroids

    return PQIndex(
        codes=codes,
        codebooks=codebooks,
        num_subspaces=num_subspaces,
        subspace_dim=subspace_dim,
        original_shape=embeddings.shape,
    )
