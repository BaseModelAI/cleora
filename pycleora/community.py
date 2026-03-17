import numpy as np
from typing import Dict, List, Optional, Tuple


def detect_communities_kmeans(
    graph,
    embeddings: np.ndarray,
    k: int,
    max_iterations: int = 100,
    seed: int = 42,
) -> Dict[str, int]:
    n = embeddings.shape[0]
    if k < 2:
        raise ValueError("k must be at least 2")
    if k > n:
        raise ValueError(f"k ({k}) cannot be larger than number of entities ({n})")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    centroids = normed[indices].copy()

    labels = np.zeros(n, dtype=np.int32)

    for _ in range(max_iterations):
        sims = normed @ centroids.T
        new_labels = np.argmax(sims, axis=1)

        if np.all(new_labels == labels):
            break
        labels = new_labels

        for i in range(k):
            mask = labels == i
            if np.sum(mask) > 0:
                centroid = np.mean(normed[mask], axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 1e-10:
                    centroids[i] = centroid / c_norm

    result = {}
    for i, eid in enumerate(graph.entity_ids):
        result[eid] = int(labels[i])

    return result


def detect_communities_spectral(
    graph,
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> Dict[str, int]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms

    u, s, vt = np.linalg.svd(normed, full_matrices=False)
    spectral_features = u[:, :k] * s[:k]

    s_norms = np.linalg.norm(spectral_features, axis=1, keepdims=True)
    s_norms = np.maximum(s_norms, 1e-10)
    spectral_normed = spectral_features / s_norms

    n = spectral_normed.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    centroids = spectral_normed[indices].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(100):
        sims = spectral_normed @ centroids.T
        new_labels = np.argmax(sims, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if np.sum(mask) > 0:
                centroid = np.mean(spectral_normed[mask], axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 1e-10:
                    centroids[i] = centroid / c_norm

    result = {}
    for i, eid in enumerate(graph.entity_ids):
        result[eid] = int(labels[i])

    return result


def detect_communities_louvain(
    graph,
    resolution: float = 1.0,
) -> Dict[str, int]:
    rows, cols, vals, n, _ = graph.to_sparse_csr()

    adj = {}
    degrees = np.zeros(n, dtype=np.float64)
    total_weight = 0.0

    for r, c, v in zip(rows, cols, vals):
        ri, ci = int(r), int(c)
        if ri == ci:
            continue
        w = 1.0
        if ri not in adj:
            adj[ri] = {}
        adj[ri][ci] = adj[ri].get(ci, 0.0) + w
        degrees[ri] += w
        total_weight += w

    if total_weight < 1e-10:
        return {eid: 0 for eid in graph.entity_ids}

    m = total_weight / 2.0
    community = list(range(n))
    sigma_tot = {i: degrees[i] for i in range(n)}

    improved = True
    max_passes = 50
    pass_count = 0

    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        for node in range(n):
            current_comm = community[node]
            ki = degrees[node]
            neighbors = adj.get(node, {})

            ki_in = {}
            for nb, w in neighbors.items():
                c = community[nb]
                ki_in[c] = ki_in.get(c, 0.0) + w

            sigma_tot[current_comm] -= ki

            best_comm = current_comm
            best_delta = 0.0

            ki_in_current = ki_in.get(current_comm, 0.0)
            delta_remove = ki_in_current / m - resolution * ki * sigma_tot.get(current_comm, 0.0) / (2.0 * m * m)

            for comm, ki_in_c in ki_in.items():
                if comm == current_comm:
                    continue
                sigma_c = sigma_tot.get(comm, 0.0)
                delta_add = ki_in_c / m - resolution * ki * sigma_c / (2.0 * m * m)
                delta = delta_add - delta_remove
                if delta > best_delta:
                    best_delta = delta
                    best_comm = comm

            if best_comm != current_comm:
                community[node] = best_comm
                sigma_tot[best_comm] = sigma_tot.get(best_comm, 0.0) + ki
                improved = True
            else:
                sigma_tot[current_comm] += ki

    label_map = {}
    next_label = 0
    for i in range(n):
        c = community[i]
        if c not in label_map:
            label_map[c] = next_label
            next_label += 1
        community[i] = label_map[c]

    result = {}
    for i, eid in enumerate(graph.entity_ids):
        result[eid] = community[i]

    return result


def modularity(graph, communities: Dict[str, int]) -> float:
    rows, cols, vals, n, _ = graph.to_sparse_csr()
    degrees = np.zeros(n, dtype=np.float64)
    total_weight = 0.0
    edge_list = []

    for r, c in zip(rows, cols):
        ri, ci = int(r), int(c)
        if ri == ci:
            continue
        edge_list.append((ri, ci))
        degrees[ri] += 1.0
        total_weight += 1.0

    if total_weight < 1e-10:
        return 0.0

    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    comm_arr = np.zeros(n, dtype=np.int32)
    for eid, comm in communities.items():
        idx = index_map.get(eid)
        if idx is not None:
            comm_arr[idx] = comm

    Q = 0.0
    for ri, ci in edge_list:
        if comm_arr[ri] == comm_arr[ci]:
            Q += 1.0 - degrees[ri] * degrees[ci] / total_weight

    return float(Q / total_weight)
