import numpy as np
from typing import Optional, Dict, List


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    seed: int = 42,
) -> np.ndarray:
    if method == "tsne":
        return _tsne_reduce(embeddings, n_components, seed)
    elif method == "pca":
        return _pca_reduce(embeddings, n_components)
    elif method == "umap":
        return _umap_reduce(embeddings, n_components, seed)
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'tsne', 'pca', or 'umap'.")


def _pca_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    centered = embeddings - np.mean(embeddings, axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    return (u[:, :n_components] * s[:n_components])


def _tsne_reduce(embeddings: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    n = embeddings.shape[0]
    rng = np.random.default_rng(seed)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms
    sims = normed @ normed.T
    dists = np.maximum(1 - sims, 0)

    perplexity = min(30, n - 1)
    P = np.zeros((n, n))
    for i in range(n):
        beta = 1.0
        for _ in range(50):
            exp_d = np.exp(-dists[i] * beta)
            exp_d[i] = 0
            sum_exp = np.sum(exp_d) + 1e-10
            p_row = exp_d / sum_exp
            entropy = -np.sum(p_row * np.log(p_row + 1e-10))
            if abs(entropy - np.log(perplexity)) < 0.01:
                break
            if entropy > np.log(perplexity):
                beta *= 2
            else:
                beta /= 2
        P[i] = p_row

    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    Y = rng.normal(0, 0.01, size=(n, n_components)).astype(np.float64)
    lr = 200.0
    momentum = 0.5

    dY = np.zeros_like(Y)

    for t in range(300):
        sum_Y = np.sum(Y ** 2, axis=1)
        num = 1.0 / (1.0 + sum_Y[:, None] + sum_Y[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0)
        Q = num / (np.sum(num) + 1e-10)
        Q = np.maximum(Q, 1e-12)

        PQ = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ[i] * num[i])[:, None] * diff, axis=0)

        if t > 100:
            momentum = 0.8

        dY = momentum * dY - lr * grad
        Y += dY
        Y -= np.mean(Y, axis=0)

    return Y.astype(np.float32)


def _umap_reduce(embeddings: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=seed)
        return reducer.fit_transform(embeddings)
    except ImportError:
        return _pca_reduce(embeddings, n_components)


def plot_embeddings(
    embeddings_2d: np.ndarray,
    labels: Optional[np.ndarray] = None,
    entity_ids: Optional[List[str]] = None,
    title: str = "Graph Embeddings",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
    show_labels: bool = False,
    point_size: int = 50,
    colormap: str = "tab10",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap(colormap, len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[cmap(i)],
                label=f"Class {label}",
                s=point_size,
                alpha=0.7,
            )
        ax.legend()
    else:
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            s=point_size,
            alpha=0.7,
        )

    if show_labels and entity_ids is not None:
        for i, eid in enumerate(entity_ids):
            ax.annotate(eid, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=7, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    else:
        plt.close(fig)
        return fig


def visualize(
    graph,
    embeddings: np.ndarray,
    labels: Optional[Dict[str, int]] = None,
    method: str = "tsne",
    title: str = "Graph Embeddings",
    save_path: Optional[str] = None,
    show_labels: bool = True,
    figsize: tuple = (12, 10),
):
    emb_2d = reduce_dimensions(embeddings, method=method)

    label_arr = None
    if labels is not None:
        label_arr = np.zeros(graph.num_entities, dtype=np.int32)
        for eid, label in labels.items():
            try:
                idx = graph.get_entity_index(eid)
                label_arr[idx] = label
            except ValueError:
                pass

    return plot_embeddings(
        emb_2d,
        labels=label_arr,
        entity_ids=graph.entity_ids,
        title=title,
        save_path=save_path,
        show_labels=show_labels,
        figsize=figsize,
    )
