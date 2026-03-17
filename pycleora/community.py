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
    total_weight = 0.0
    degrees = np.zeros(n, dtype=np.float64)

    for r, c, v in zip(rows, cols, vals):
        r, c = int(r), int(c)
        v = float(v)
        if r not in adj:
            adj[r] = {}
        adj[r][c] = adj[r].get(c, 0) + v
        degrees[r] += v
        total_weight += v

    if total_weight < 1e-10:
        return {eid: 0 for eid in graph.entity_ids}

    m2 = total_weight

    community = list(range(n))

    improved = True
    while improved:
        improved = False
        for node in range(n):
            current_comm = community[node]
            best_comm = current_comm
            best_gain = 0.0

            neighbors = adj.get(node, {})
            neighbor_comms = set()
            for nb in neighbors:
                neighbor_comms.add(community[nb])
            neighbor_comms.add(current_comm)

            ki = degrees[node]

            for comm in neighbor_comms:
                sum_in = 0.0
                sum_tot = 0.0
                ki_in = 0.0

                for i in range(n):
                    if community[i] == comm:
                        sum_tot += degrees[i]
                        if i != node:
                            for nb, w in adj.get(i, {}).items():
                                if community[nb] == comm:
                                    sum_in += w

                for nb, w in neighbors.items():
                    if community[nb] == comm:
                        ki_in += w

                if comm == current_comm:
                    continue

                gain = (ki_in - resolution * ki * sum_tot / m2)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = comm

            if best_comm != current_comm:
                community[node] = best_comm
                improved = True

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
    total_weight = float(np.sum(vals))
    if total_weight < 1e-10:
        return 0.0

    degrees = np.zeros(n, dtype=np.float64)
    for r, v in zip(rows, vals):
        degrees[int(r)] += float(v)

    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    comm_arr = np.zeros(n, dtype=np.int32)
    for eid, comm in communities.items():
        idx = index_map.get(eid)
        if idx is not None:
            comm_arr[idx] = comm

    Q = 0.0
    for r, c, v in zip(rows, cols, vals):
        r, c = int(r), int(c)
        if comm_arr[r] == comm_arr[c]:
            Q += float(v) - degrees[r] * degrees[c] / total_weight

    return float(Q / total_weight)
