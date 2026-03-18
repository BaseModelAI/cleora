import numpy as np
from scipy.sparse import csr_matrix
from collections import deque


def _graph_to_adjacency(graph):
    rows, cols, vals, n, _ = graph.to_sparse_csr()
    A = csr_matrix(
        (vals.astype(np.float64), (rows.astype(np.int32), cols.astype(np.int32))),
        shape=(n, n),
    )
    return A


def _make_symmetric(A):
    S = ((A + A.T) > 0).astype(np.float64)
    S.setdiag(0)
    S.eliminate_zeros()
    return S


def clean_graph(edges, remove_self_loops=True, deduplicate=True, min_degree=None, max_degree=None):
    """Clean a list of edge strings. Returns a new list with optional self-loop removal,
    deduplication, and degree-based filtering applied."""
    result = []
    for edge in edges:
        parts = edge.strip().split()
        if remove_self_loops and len(parts) == 2 and parts[0] == parts[1]:
            continue
        result.append(edge.strip())

    if deduplicate:
        seen = set()
        deduped = []
        for edge in result:
            parts = tuple(sorted(edge.split()))
            if parts not in seen:
                seen.add(parts)
                deduped.append(edge)
        result = deduped

    if min_degree is not None or max_degree is not None:
        result = filter_by_degree_edges(result, min_degree=min_degree, max_degree=max_degree)

    return result


def filter_by_degree_edges(edges, min_degree=None, max_degree=None):
    from collections import Counter
    degree_count = Counter()
    for edge in edges:
        parts = edge.strip().split()
        for p in parts:
            degree_count[p] += 1

    valid_nodes = set()
    for node, deg in degree_count.items():
        if min_degree is not None and deg < min_degree:
            continue
        if max_degree is not None and deg > max_degree:
            continue
        valid_nodes.add(node)

    result = []
    for edge in edges:
        parts = edge.strip().split()
        if all(p in valid_nodes for p in parts):
            result.append(edge.strip())
    return result


def filter_by_degree(graph, min_degree=None, max_degree=None):
    """Return a list of edge strings containing only nodes whose degree falls
    within [min_degree, max_degree]. Operates on the SparseMatrix adjacency."""
    A = _make_symmetric(_graph_to_adjacency(graph))
    degrees = np.array(A.sum(axis=1)).flatten().astype(int)
    entity_ids = graph.entity_ids

    valid_nodes = set()
    for i, deg in enumerate(degrees):
        if min_degree is not None and deg < min_degree:
            continue
        if max_degree is not None and deg > max_degree:
            continue
        valid_nodes.add(entity_ids[i])

    rows_arr, cols_arr = A.nonzero()
    edges = []
    seen = set()
    for r, c in zip(rows_arr, cols_arr):
        if r >= c:
            continue
        src = entity_ids[r]
        dst = entity_ids[c]
        if src in valid_nodes and dst in valid_nodes:
            pair = (src, dst)
            if pair not in seen:
                seen.add(pair)
                edges.append(f"{src} {dst}")

    return edges


def largest_connected_component(graph, columns="complex::reflexive::node", hyperedge_trim_n=16, num_workers=None):
    """Extract the largest connected component as a new SparseMatrix.
    Pass the same columns and hyperedge_trim_n used to create the original graph
    to preserve its construction semantics."""
    from .pycleora import SparseMatrix

    A = _make_symmetric(_graph_to_adjacency(graph))
    n = A.shape[0]
    entity_ids = graph.entity_ids

    visited = np.zeros(n, dtype=bool)
    best_component = []

    for start in range(n):
        if visited[start]:
            continue
        component = []
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            component.append(node)
            neighbors = A[node].nonzero()[1]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        if len(component) > len(best_component):
            best_component = component

    if not best_component:
        raise ValueError("Graph has no nodes")

    component_set = set(best_component)
    rows_arr, cols_arr = A.nonzero()
    edges = []
    seen = set()
    for r, c in zip(rows_arr, cols_arr):
        if r >= c:
            continue
        if r in component_set and c in component_set:
            src = entity_ids[r]
            dst = entity_ids[c]
            pair = (src, dst)
            if pair not in seen:
                seen.add(pair)
                edges.append(f"{src} {dst}")

    if not edges:
        edges = [f"{entity_ids[best_component[0]]} {entity_ids[best_component[0]]}"]

    return SparseMatrix.from_iterator(
        iter(edges),
        columns,
        hyperedge_trim_n,
        num_workers,
    )
