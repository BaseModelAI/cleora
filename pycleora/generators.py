import numpy as np
from typing import Dict, Optional, List


def erdos_renyi(
    num_nodes: int,
    p: float = 0.1,
    seed: int = 42,
    directed: bool = False,
) -> Dict:
    rng = np.random.default_rng(seed)
    edges = []
    edge_set = set()

    for i in range(num_nodes):
        jstart = 0 if directed else i + 1
        for j in range(jstart, num_nodes):
            if i == j:
                continue
            if rng.random() < p:
                edges.append(f"n{i} n{j}")
                edge_set.add((i, j))

    labels = {f"n{i}": 0 for i in range(num_nodes)}

    return {
        "name": f"Erdos-Renyi(n={num_nodes}, p={p})",
        "edges": edges,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": 1,
        "columns": "complex::reflexive::node",
        "model": "erdos_renyi",
    }


def barabasi_albert(
    num_nodes: int,
    m: int = 3,
    seed: int = 42,
) -> Dict:
    if num_nodes < 2:
        raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")
    if m < 1 or m >= num_nodes:
        raise ValueError(f"m must be >= 1 and < num_nodes ({num_nodes}), got {m}")

    rng = np.random.default_rng(seed)

    initial = max(m + 1, 2)
    initial = min(initial, num_nodes)
    adj_list = [set() for _ in range(num_nodes)]
    degrees = np.zeros(num_nodes, dtype=np.float64)

    for i in range(initial):
        for j in range(i + 1, initial):
            adj_list[i].add(j)
            adj_list[j].add(i)
            degrees[i] += 1
            degrees[j] += 1

    for new_node in range(initial, num_nodes):
        targets = set()
        deg_sum = degrees[:new_node].sum()
        if deg_sum < 1e-10:
            targets = set(rng.choice(new_node, size=min(m, new_node), replace=False))
        else:
            probs = degrees[:new_node] / deg_sum
            chosen = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
            targets = set(chosen.tolist())

        for t in targets:
            adj_list[new_node].add(t)
            adj_list[t].add(new_node)
            degrees[new_node] += 1
            degrees[t] += 1

    edges = []
    seen = set()
    for i in range(num_nodes):
        for j in adj_list[i]:
            key = (min(i, j), max(i, j))
            if key not in seen:
                edges.append(f"n{i} n{j}")
                seen.add(key)

    labels = {f"n{i}": 0 for i in range(num_nodes)}

    return {
        "name": f"Barabasi-Albert(n={num_nodes}, m={m})",
        "edges": edges,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": 1,
        "columns": "complex::reflexive::node",
        "model": "barabasi_albert",
    }


def stochastic_block_model(
    block_sizes: List[int],
    p_within: float = 0.3,
    p_between: float = 0.01,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    num_blocks = len(block_sizes)
    num_nodes = sum(block_sizes)

    block_assignment = []
    for block_id, size in enumerate(block_sizes):
        block_assignment.extend([block_id] * size)

    edges = []
    edge_set = set()

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = p_within if block_assignment[i] == block_assignment[j] else p_between
            if rng.random() < p:
                edges.append(f"n{i} n{j}")
                edge_set.add((i, j))

    labels = {f"n{i}": block_assignment[i] for i in range(num_nodes)}

    return {
        "name": f"SBM(blocks={block_sizes})",
        "edges": edges,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": num_blocks,
        "columns": "complex::reflexive::node",
        "model": "stochastic_block_model",
        "block_sizes": block_sizes,
    }


def planted_partition(
    num_communities: int = 4,
    community_size: int = 25,
    p_in: float = 0.3,
    p_out: float = 0.01,
    seed: int = 42,
) -> Dict:
    return stochastic_block_model(
        block_sizes=[community_size] * num_communities,
        p_within=p_in,
        p_between=p_out,
        seed=seed,
    )


def watts_strogatz(
    num_nodes: int,
    k: int = 6,
    beta: float = 0.3,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    edges_set = set()

    for i in range(num_nodes):
        for j in range(1, k // 2 + 1):
            nb = (i + j) % num_nodes
            edges_set.add((min(i, nb), max(i, nb)))

    rewired = set()
    for i in range(num_nodes):
        for j in range(1, k // 2 + 1):
            nb = (i + j) % num_nodes
            key = (min(i, nb), max(i, nb))
            if rng.random() < beta and key not in rewired:
                edges_set.discard(key)
                while True:
                    new_nb = int(rng.integers(0, num_nodes))
                    new_key = (min(i, new_nb), max(i, new_nb))
                    if new_nb != i and new_key not in edges_set:
                        edges_set.add(new_key)
                        rewired.add(new_key)
                        break

    edges = [f"n{i} n{j}" for i, j in edges_set]
    labels = {f"n{i}": i % 4 for i in range(num_nodes)}

    return {
        "name": f"Watts-Strogatz(n={num_nodes}, k={k}, beta={beta})",
        "edges": edges,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": 4,
        "columns": "complex::reflexive::node",
        "model": "watts_strogatz",
    }
