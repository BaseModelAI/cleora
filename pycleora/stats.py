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


def degree_distribution(graph):
    """Return a list where index i holds the count of nodes with degree i."""
    A = _make_symmetric(_graph_to_adjacency(graph))
    degrees = np.array(A.sum(axis=1)).flatten().astype(int)
    max_deg = int(degrees.max()) if len(degrees) > 0 else 0
    hist = np.zeros(max_deg + 1, dtype=int)
    for d in degrees:
        hist[d] += 1
    return hist.tolist()


def clustering_coefficient(graph):
    """Return the average local clustering coefficient (0.0 to 1.0)."""
    A = _make_symmetric(_graph_to_adjacency(graph))
    n = A.shape[0]
    if n == 0:
        return 0.0

    A_bool = A.astype(bool).astype(np.float64)
    A2 = A_bool @ A_bool
    triangles_per_node = np.array(A_bool.multiply(A2).sum(axis=1)).flatten()
    degrees = np.array(A_bool.sum(axis=1)).flatten()

    total_cc = 0.0
    count = 0
    for i in range(n):
        d = int(degrees[i])
        if d >= 2:
            possible = d * (d - 1)
            total_cc += triangles_per_node[i] / possible
            count += 1

    return total_cc / count if count > 0 else 0.0


def connected_components(graph):
    """Return a list of components, each a list of integer node indices.
    Use graph.entity_ids[idx] to map indices back to entity ID strings."""
    A = _make_symmetric(_graph_to_adjacency(graph))
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if visited[start]:
            continue
        component = []
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            component.append(node)
            row = A[node]
            neighbors = row.nonzero()[1]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(component)

    return components


def diameter(graph):
    """Return the diameter of the largest connected component (longest shortest path)."""
    A = _make_symmetric(_graph_to_adjacency(graph))
    components = connected_components(graph)
    if not components:
        return 0

    largest = max(components, key=len)
    if len(largest) <= 1:
        return 0

    node_set = set(largest)
    n = A.shape[0]

    def bfs_eccentricity(start):
        dist = np.full(n, -1, dtype=int)
        dist[start] = 0
        queue = deque([start])
        max_dist = 0
        while queue:
            node = queue.popleft()
            neighbors = A[node].nonzero()[1]
            for nb in neighbors:
                if nb in node_set and dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    max_dist = max(max_dist, dist[nb])
                    queue.append(nb)
        return max_dist

    return max(bfs_eccentricity(node) for node in largest)


def betweenness_centrality(graph, top_k=10):
    """Return a dict of {entity_id: centrality_score} for the top-K nodes by betweenness."""
    A = _make_symmetric(_graph_to_adjacency(graph))
    n = A.shape[0]
    if n == 0:
        return {}

    centrality = np.zeros(n, dtype=np.float64)
    entity_ids = graph.entity_ids

    for s in range(n):
        stack = []
        predecessors = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)
        sigma[s] = 1.0
        dist = np.full(n, -1, dtype=int)
        dist[s] = 0
        queue = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            neighbors = A[v].nonzero()[1]
            for w in neighbors:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta = np.zeros(n, dtype=np.float64)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    centrality /= 2.0

    top_indices = np.argsort(centrality)[::-1][:top_k]
    return {entity_ids[i]: float(centrality[i]) for i in top_indices}


def pagerank(graph, top_k=10, damping=0.85, max_iter=100, tol=1e-6):
    """Return a dict of {entity_id: pagerank_score} for the top-K nodes."""
    A = _make_symmetric(_graph_to_adjacency(graph))
    n = A.shape[0]
    if n == 0:
        return {}

    entity_ids = graph.entity_ids
    out_degree = np.array(A.sum(axis=1)).flatten()
    dangling = (out_degree == 0)

    from scipy.sparse import diags
    safe_degree = out_degree.copy()
    safe_degree[dangling] = 1.0
    D_inv = diags(1.0 / safe_degree)
    M = (D_inv @ A).T

    pr = np.ones(n, dtype=np.float64) / n

    for _ in range(max_iter):
        dangling_mass = pr[dangling].sum()
        new_pr = (1 - damping + damping * dangling_mass) / n + damping * M @ pr
        if np.linalg.norm(new_pr - pr, ord=1) < tol:
            pr = new_pr
            break
        pr = new_pr

    top_indices = np.argsort(pr)[::-1][:top_k]
    return {entity_ids[i]: float(pr[i]) for i in top_indices}


def graph_summary(graph, top_k=10):
    """Return a dict with comprehensive graph statistics including num_nodes, num_edges,
    density, avg_degree, degree_distribution, clustering_coefficient,
    num_connected_components, diameter, betweenness_centrality (top-K), and pagerank (top-K)."""
    A = _make_symmetric(_graph_to_adjacency(graph))
    n = A.shape[0]
    num_edges_undirected = int(A.nnz / 2)
    degrees = np.array(A.sum(axis=1)).flatten()

    max_possible = n * (n - 1) if n > 1 else 1
    density = float(A.nnz) / max_possible if n > 1 else 0.0

    components = connected_components(graph)

    return {
        "num_nodes": n,
        "num_edges": num_edges_undirected,
        "density": density,
        "avg_degree": float(degrees.mean()) if n > 0 else 0.0,
        "degree_distribution": degree_distribution(graph),
        "clustering_coefficient": clustering_coefficient(graph),
        "num_connected_components": len(components),
        "diameter": diameter(graph),
        "betweenness_centrality": betweenness_centrality(graph, top_k=top_k),
        "pagerank": pagerank(graph, top_k=top_k),
    }
