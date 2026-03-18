import numpy as np
from typing import Tuple


def procrustes(
    emb_source: np.ndarray,
    emb_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align source embeddings to target using orthogonal Procrustes analysis.

    Finds the orthogonal matrix R that minimizes ||emb_source @ R - emb_target||_F.
    Both embedding matrices must have the same shape (n_entities x dim).

    Note: Entity alignment between different graphs is the caller's responsibility.
    Rows must correspond to the same entities in both matrices.

    Args:
        emb_source: Source embedding matrix, shape (n, d).
        emb_target: Target embedding matrix, shape (n, d).

    Returns:
        A tuple of (aligned_embeddings, transformation_matrix) where
        aligned_embeddings = emb_source @ R and transformation_matrix = R.
    """
    if emb_source.shape != emb_target.shape:
        raise ValueError(
            f"emb_source shape {emb_source.shape} does not match "
            f"emb_target shape {emb_target.shape}"
        )
    if emb_source.ndim != 2:
        raise ValueError("Embeddings must be 2-dimensional arrays")

    M = emb_source.T @ emb_target
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    aligned = emb_source @ R
    return aligned.astype(np.float32), R.astype(np.float32)


def cca_align(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    n_components: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two embedding spaces using Canonical Correlation Analysis.

    Projects both sets of embeddings into a shared space that maximizes
    correlation between the two views.

    Note: Entity alignment between different graphs is the caller's responsibility.
    Rows must correspond to the same entities in both matrices.

    Args:
        emb_a: First embedding matrix, shape (n, d_a).
        emb_b: Second embedding matrix, shape (n, d_b).
        n_components: Number of canonical components. Defaults to
            min(d_a, d_b).

    Returns:
        A tuple of (aligned_a, aligned_b) projected into the shared space.
    """
    if emb_a.shape[0] != emb_b.shape[0]:
        raise ValueError(
            f"emb_a has {emb_a.shape[0]} rows but emb_b has {emb_b.shape[0]} rows"
        )
    if emb_a.ndim != 2 or emb_b.ndim != 2:
        raise ValueError("Embeddings must be 2-dimensional arrays")

    n = emb_a.shape[0]
    if n < 2:
        raise ValueError("CCA requires at least 2 samples (rows)")
    d_a = emb_a.shape[1]
    d_b = emb_b.shape[1]

    if n_components is None:
        n_components = min(d_a, d_b)
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError(f"n_components must be a positive integer, got {n_components}")
    if n_components > min(d_a, d_b):
        raise ValueError(
            f"n_components ({n_components}) cannot exceed min(d_a, d_b) = {min(d_a, d_b)}"
        )

    mean_a = emb_a.mean(axis=0)
    mean_b = emb_b.mean(axis=0)
    a_centered = emb_a - mean_a
    b_centered = emb_b - mean_b

    reg = 1e-8
    C_aa = (a_centered.T @ a_centered) / (n - 1) + reg * np.eye(d_a)
    C_bb = (b_centered.T @ b_centered) / (n - 1) + reg * np.eye(d_b)
    C_ab = (a_centered.T @ b_centered) / (n - 1)

    C_aa_inv_sqrt = _matrix_inv_sqrt(C_aa)
    C_bb_inv_sqrt = _matrix_inv_sqrt(C_bb)

    T = C_aa_inv_sqrt @ C_ab @ C_bb_inv_sqrt
    U, _, Vt = np.linalg.svd(T, full_matrices=False)

    U_k = U[:, :n_components]
    V_k = Vt[:n_components, :].T

    W_a = C_aa_inv_sqrt @ U_k
    W_b = C_bb_inv_sqrt @ V_k

    aligned_a = a_centered @ W_a
    aligned_b = b_centered @ W_b

    return aligned_a.astype(np.float32), aligned_b.astype(np.float32)


def alignment_score(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute a similarity metric between two embedding spaces.

    Aligns emb_a to emb_b using orthogonal Procrustes, then returns the
    mean cosine similarity between corresponding rows.

    Note: Entity alignment between different graphs is the caller's responsibility.
    Rows must correspond to the same entities in both matrices.

    Args:
        emb_a: First embedding matrix, shape (n, d).
        emb_b: Second embedding matrix, shape (n, d).

    Returns:
        Mean cosine similarity after alignment (float between -1 and 1).
    """
    aligned_a, _ = procrustes(emb_a, emb_b)

    norms_a = np.linalg.norm(aligned_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
    norms_a = np.maximum(norms_a, 1e-10)
    norms_b = np.maximum(norms_b, 1e-10)

    cos_sims = np.sum((aligned_a / norms_a) * (emb_b / norms_b), axis=1)
    return float(np.mean(cos_sims))


def _matrix_inv_sqrt(M: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
