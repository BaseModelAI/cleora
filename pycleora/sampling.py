import numpy as np
from typing import List, Optional, Tuple, Dict, Set


def _build_adj(graph):
    rows, cols, vals, n, _ = graph.to_sparse_csr()
    adj = [[] for _ in range(n)]
    for r, c in zip(rows, cols):
        ri, ci = int(r), int(c)
        if ri != ci:
            adj[ri].append(ci)
    return adj, n


def sample_nodes(
    graph,
    num_nodes: int,
    seed: int = 42,
) -> List[str]:
    rng = np.random.default_rng(seed)
    n = graph.num_entities
    num_nodes = min(num_nodes, n)
    indices = rng.choice(n, size=num_nodes, replace=False)
    return [graph.entity_ids[i] for i in indices]


def sample_edges(
    graph,
    num_edges: int,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    rows, cols, vals, n, _ = graph.to_sparse_csr()
    edge_list = []
    seen = set()
    for r, c in zip(rows, cols):
        ri, ci = int(r), int(c)
        if ri < ci:
            edge_list.append((ri, ci))
            seen.add((ri, ci))
        elif ci < ri and (ci, ri) not in seen:
            edge_list.append((ci, ri))
            seen.add((ci, ri))

    rng = np.random.default_rng(seed)
    num_edges = min(num_edges, len(edge_list))
    indices = rng.choice(len(edge_list), size=num_edges, replace=False)
    return [(graph.entity_ids[edge_list[i][0]], graph.entity_ids[edge_list[i][1]]) for i in indices]


def sample_neighborhood(
    graph,
    seed_nodes: List[str],
    num_hops: int = 2,
    max_neighbors_per_hop: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    adj, n = _build_adj(graph)
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    rng = np.random.default_rng(seed)

    sampled_nodes = set()
    for eid in seed_nodes:
        idx = index_map.get(eid)
        if idx is not None:
            sampled_nodes.add(idx)

    frontier = set(sampled_nodes)
    for hop in range(num_hops):
        next_frontier = set()
        for node in frontier:
            neighbors = adj[node]
            if max_neighbors_per_hop and len(neighbors) > max_neighbors_per_hop:
                neighbors = rng.choice(neighbors, size=max_neighbors_per_hop, replace=False).tolist()
            for nb in neighbors:
                if nb not in sampled_nodes:
                    next_frontier.add(nb)
                    sampled_nodes.add(nb)
        frontier = next_frontier
        if not frontier:
            break

    sampled_edges = []
    for node in sampled_nodes:
        for nb in adj[node]:
            if nb in sampled_nodes:
                sampled_edges.append(f"{graph.entity_ids[node]} {graph.entity_ids[nb]}")

    return {
        "nodes": [graph.entity_ids[i] for i in sorted(sampled_nodes)],
        "edges": sampled_edges,
        "num_nodes": len(sampled_nodes),
        "num_edges": len(sampled_edges),
    }


def sample_subgraph(
    graph,
    num_nodes: int,
    method: str = "random_walk",
    walk_length: int = 100,
    seed: int = 42,
) -> Dict:
    adj, n = _build_adj(graph)
    rng = np.random.default_rng(seed)

    if method == "random_walk":
        sampled = set()
        start = int(rng.integers(0, n))
        curr = start
        for _ in range(walk_length * 10):
            sampled.add(curr)
            if len(sampled) >= num_nodes:
                break
            neighbors = adj[curr]
            if not neighbors:
                curr = int(rng.integers(0, n))
            else:
                curr = neighbors[int(rng.integers(len(neighbors)))]
    elif method == "random_node":
        num_nodes = min(num_nodes, n)
        sampled = set(rng.choice(n, size=num_nodes, replace=False).tolist())
    elif method == "bfs":
        sampled = set()
        start = int(rng.integers(0, n))
        queue = [start]
        sampled.add(start)
        qi = 0
        while qi < len(queue) and len(sampled) < num_nodes:
            curr = queue[qi]
            qi += 1
            for nb in adj[curr]:
                if nb not in sampled:
                    sampled.add(nb)
                    queue.append(nb)
                    if len(sampled) >= num_nodes:
                        break
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'random_walk', 'random_node', or 'bfs'.")

    edges = []
    for node in sampled:
        for nb in adj[node]:
            if nb in sampled:
                edges.append(f"{graph.entity_ids[node]} {graph.entity_ids[nb]}")

    return {
        "nodes": [graph.entity_ids[i] for i in sorted(sampled)],
        "edges": edges,
        "num_nodes": len(sampled),
        "num_edges": len(edges),
    }


def graphsaint_sample(
    graph,
    batch_size: int = 512,
    walk_length: int = 4,
    num_batches: int = 5,
    seed: int = 42,
) -> List[Dict]:
    adj, n = _build_adj(graph)
    rng = np.random.default_rng(seed)
    batches = []

    for b in range(num_batches):
        sampled = set()
        for _ in range(batch_size):
            start = int(rng.integers(0, n))
            curr = start
            for _ in range(walk_length):
                sampled.add(curr)
                neighbors = adj[curr]
                if neighbors:
                    curr = neighbors[int(rng.integers(len(neighbors)))]
                else:
                    break

        edges = []
        for node in sampled:
            for nb in adj[node]:
                if nb in sampled:
                    edges.append(f"{graph.entity_ids[node]} {graph.entity_ids[nb]}")

        batches.append({
            "batch_id": b,
            "nodes": [graph.entity_ids[i] for i in sorted(sampled)],
            "edges": edges,
            "num_nodes": len(sampled),
            "num_edges": len(edges),
        })

    return batches


def negative_sampling(
    graph,
    num_negatives: int = 1000,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    rows, cols, _, n, _ = graph.to_sparse_csr()
    existing = set()
    for r, c in zip(rows, cols):
        ri, ci = int(r), int(c)
        existing.add((min(ri, ci), max(ri, ci)))

    rng = np.random.default_rng(seed)
    negatives = []
    attempts = 0
    max_attempts = num_negatives * 20

    while len(negatives) < num_negatives and attempts < max_attempts:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j and (min(i, j), max(i, j)) not in existing:
            negatives.append((graph.entity_ids[i], graph.entity_ids[j]))
            existing.add((min(i, j), max(i, j)))
        attempts += 1

    return negatives


def train_test_split_edges(
    graph,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict:
    rows, cols, _, n, _ = graph.to_sparse_csr()
    edge_list = []
    seen = set()
    for r, c in zip(rows, cols):
        ri, ci = int(r), int(c)
        key = (min(ri, ci), max(ri, ci))
        if key not in seen and ri != ci:
            edge_list.append(key)
            seen.add(key)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edge_list))
    split = int(len(edge_list) * (1 - test_ratio))

    train_edges = [(graph.entity_ids[edge_list[i][0]], graph.entity_ids[edge_list[i][1]]) for i in perm[:split]]
    test_edges = [(graph.entity_ids[edge_list[i][0]], graph.entity_ids[edge_list[i][1]]) for i in perm[split:]]
    train_edge_strs = [f"{a} {b}" for a, b in train_edges]

    return {
        "train_edges": train_edges,
        "test_edges": test_edges,
        "train_edge_strings": train_edge_strs,
        "num_train": len(train_edges),
        "num_test": len(test_edges),
    }
