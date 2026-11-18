import numpy as np
from typing import List, Optional, Tuple


def _upper_triangle(graph, markov_type: str = "left"):
    """One entry per undirected edge (self-loops included): (rows, cols, values)."""
    rows, cols, vals, _, _ = graph.to_sparse_csr(markov_type)
    keep = rows <= cols
    return rows[keep], cols[keep], vals[keep]


def to_networkx(graph, embeddings: Optional[np.ndarray] = None):
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for graph export. Install with: pip install networkx")

    G = nx.Graph()
    entity_ids = graph.entity_ids

    for i, eid in enumerate(entity_ids):
        attrs = {"index": i}
        if embeddings is not None:
            attrs["embedding"] = embeddings[i].tolist()
        G.add_node(eid, **attrs)

    rows, cols, vals = _upper_triangle(graph)
    G.add_weighted_edges_from(
        (entity_ids[r], entity_ids[c], v)
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist())
    )

    return G


def from_networkx(G, columns: str = "complex::reflexive::node", hyperedge_trim_n: int = 16, num_workers=None,
                  weight: Optional[str] = None):
    """Create a SparseMatrix from a networkx graph.

    Node names are converted with ``str``. ``weight`` optionally names the edge
    attribute that holds the edge weight (edges without it count as 1).
    """
    sources = []
    targets = []
    weights = [] if weight is not None else None
    for u, v, data in G.edges(data=True):
        sources.append(str(u))
        targets.append(str(v))
        if weights is not None:
            weights.append(float(data.get(weight, 1.0)))
    return from_arrays(sources, targets, weights=weights, columns=columns, num_workers=num_workers)


def to_pyg_data(graph, embeddings: np.ndarray):
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "PyTorch Geometric is required. Install with: pip install torch torch-geometric"
        )

    rows, cols, vals, _, _ = graph.to_sparse_csr()
    edge_index = torch.tensor(
        np.stack([rows.astype(np.int64), cols.astype(np.int64)]),
        dtype=torch.long,
    )
    edge_attr = torch.tensor(vals, dtype=torch.float)
    x = torch.tensor(embeddings, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def to_dgl_graph(graph, embeddings: np.ndarray):
    try:
        import dgl
        import torch
    except ImportError:
        raise ImportError("DGL is required. Install with: pip install dgl")

    rows, cols, vals, _, _ = graph.to_sparse_csr()
    src = torch.tensor(rows.astype(np.int64), dtype=torch.long)
    dst = torch.tensor(cols.astype(np.int64), dtype=torch.long)

    g = dgl.graph((src, dst))
    g.ndata["feat"] = torch.tensor(embeddings, dtype=torch.float)
    g.edata["weight"] = torch.tensor(vals, dtype=torch.float)
    return g


def save_embeddings(graph, embeddings: np.ndarray, filepath: str, format: str = "npz"):
    if format == "npz":
        np.savez(
            filepath,
            embeddings=embeddings,
            entity_ids=np.array(graph.entity_ids),
        )
    elif format == "csv":
        import csv
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["entity_id"] + [f"dim_{i}" for i in range(embeddings.shape[1])]
            writer.writerow(header)
            for i, eid in enumerate(graph.entity_ids):
                writer.writerow([eid] + embeddings[i].tolist())
    elif format == "tsv":
        with open(filepath, "w") as f:
            header = "entity_id\t" + "\t".join(f"dim_{i}" for i in range(embeddings.shape[1]))
            f.write(header + "\n")
            for i, eid in enumerate(graph.entity_ids):
                vals = "\t".join(f"{v:.6f}" for v in embeddings[i])
                f.write(f"{eid}\t{vals}\n")
    elif format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow is required for parquet export. Install with: pip install pyarrow")

        arrays = {"entity_id": list(graph.entity_ids)}
        for i in range(embeddings.shape[1]):
            arrays[f"dim_{i}"] = embeddings[:, i].tolist()
        table = pa.table(arrays)
        pq.write_table(table, filepath)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz', 'csv', 'tsv', or 'parquet'.")


def load_embeddings(filepath: str, format: str = "npz") -> Tuple[np.ndarray, List[str]]:
    if format == "npz":
        data = np.load(filepath, allow_pickle=False)
        return data["embeddings"], data["entity_ids"].tolist()
    elif format == "csv":
        import csv
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            entity_ids = []
            rows = []
            for row in reader:
                entity_ids.append(row[0])
                rows.append([float(v) for v in row[1:]])
        return np.array(rows, dtype=np.float32), entity_ids
    elif format == "tsv":
        entity_ids = []
        rows = []
        with open(filepath, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                entity_ids.append(parts[0])
                rows.append([float(v) for v in parts[1:]])
        return np.array(rows, dtype=np.float32), entity_ids
    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz', 'csv', or 'tsv'.")


def _as_id_array(values, name: str):
    """Node identifiers as an int64 array or a list of str."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got {array.ndim} dimension(s)")
    if array.dtype.kind in "iu":
        if array.dtype.kind == "u" and array.size and int(array.max()) > np.iinfo(np.int64).max:
            return [str(v) for v in array.tolist()]
        return np.ascontiguousarray(array, dtype=np.int64)
    if array.dtype.kind == "b":
        raise ValueError(f"{name} must hold node identifiers, got booleans")
    return [str(v) for v in array.tolist()]


def from_arrays(src, dst, weights=None, columns: str = "complex::reflexive::node", num_workers=None):
    """Create a SparseMatrix from two parallel arrays of node identifiers.

    ``src[i]`` and ``dst[i]`` are the two ends of edge ``i``. Integer arrays are
    consumed directly, without building Python strings (the entity name of node
    ``7`` is ``"7"``); anything else is converted with ``str``. ``weights``
    optionally holds one finite, non-negative weight per edge.

    With the default ``columns`` every edge is a two-entity hyperedge of a single
    reflexive column, exactly like the line ``f"{src} {dst}"``. Pass two column
    names (for example ``"user product"``) for a bipartite graph without
    self-loops, like the line ``f"{src}\t{dst}"``.
    """
    from .pycleora import SparseMatrix

    sources = _as_id_array(src, "src")
    targets = _as_id_array(dst, "dst")
    if len(sources) != len(targets):
        raise ValueError(f"src has {len(sources)} element(s) but dst has {len(targets)}")
    if isinstance(sources, list) != isinstance(targets, list):
        # mixed integer / non-integer identifiers: treat both as names
        if not isinstance(sources, list):
            sources = [str(v) for v in sources.tolist()]
        if not isinstance(targets, list):
            targets = [str(v) for v in targets.tolist()]
    if weights is not None:
        weights = np.ascontiguousarray(weights, dtype=np.float32)
    return SparseMatrix.from_edges(sources, targets, columns, weights, num_workers)


def from_pandas(df, source_col: str, target_col: str, weight_col: Optional[str] = None,
                 columns: str = "complex::reflexive::node",
                 hyperedge_trim_n: int = 16, num_workers=None):
    """Create a SparseMatrix from a pandas DataFrame.

    Each row becomes an edge between ``source_col`` and ``target_col``; rows with a
    missing endpoint are dropped. When ``weight_col`` is given its values are used
    as edge weights (rows with a missing or zero weight are dropped).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame import. Install with: pip install pandas")

    if source_col not in df.columns:
        raise ValueError(f"source_col '{source_col}' not found in DataFrame columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame columns: {list(df.columns)}")
    if weight_col is not None and weight_col not in df.columns:
        raise ValueError(f"weight_col '{weight_col}' not found in DataFrame columns: {list(df.columns)}")

    keep = df[source_col].notna() & df[target_col].notna()
    weights = None
    if weight_col is not None:
        weight_values = pd.to_numeric(df[weight_col], errors="coerce")
        keep &= weight_values.notna() & (weight_values != 0)
        weights = weight_values[keep].to_numpy(dtype=np.float32)
    if not bool(keep.any()):
        raise ValueError("No valid edges found in DataFrame (all rows may have NaN values)")

    return from_arrays(
        df.loc[keep, source_col].to_numpy(),
        df.loc[keep, target_col].to_numpy(),
        weights=weights,
        columns=columns,
        num_workers=num_workers,
    )


def from_scipy_sparse(matrix, entity_ids: Optional[List[str]] = None,
                      columns: str = "complex::reflexive::node",
                      hyperedge_trim_n: int = 16, num_workers=None,
                      use_weights: bool = False):
    """Create a SparseMatrix from a scipy sparse adjacency matrix.

    If entity_ids is None, uses stringified integer indices as node IDs.
    The adjacency is treated as undirected: an edge is created between nodes i and
    j if either (i, j) or (j, i) is stored, once. With ``use_weights=True`` the
    stored values become edge weights (the larger of the two directions).
    """
    try:
        import scipy.sparse
    except ImportError:
        raise ImportError("scipy is required for sparse matrix import. Install with: pip install scipy")

    if not scipy.sparse.issparse(matrix):
        raise ValueError("matrix must be a scipy sparse matrix")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")

    n = matrix.shape[0]
    if entity_ids is not None and len(entity_ids) != n:
        raise ValueError(f"entity_ids has {len(entity_ids)} elements but matrix has {n} rows")

    coo = matrix.tocoo()
    if coo.nnz == 0:
        raise ValueError("No edges found in the sparse matrix")
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    # One edge per unordered pair, in order of first appearance and with the
    # orientation of that first appearance.
    key = np.minimum(rows, cols) * n + np.maximum(rows, cols)
    unique_keys, first_index, inverse = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first_index, kind="stable")
    picked = first_index[order]
    weights = None
    if use_weights:
        strongest = np.zeros(len(unique_keys), dtype=np.float64)
        np.maximum.at(strongest, inverse, np.abs(np.asarray(coo.data, dtype=np.float64)))
        weights = strongest[order]

    return _from_index_edges(rows[picked], cols[picked], weights, entity_ids, columns, num_workers)


def _from_index_edges(low, high, weights, entity_ids, columns, num_workers):
    if weights is not None:
        keep = weights > 0
        low, high, weights = low[keep], high[keep], weights[keep]
        if len(low) == 0:
            raise ValueError("No edges with a positive weight found")
    if entity_ids is None:
        return from_arrays(low, high, weights=weights, columns=columns, num_workers=num_workers)
    names = np.asarray([str(eid) for eid in entity_ids], dtype=object)
    return from_arrays(names[low], names[high], weights=weights, columns=columns, num_workers=num_workers)


def from_edge_list(edges: List, columns: str = "complex::reflexive::node",
                   hyperedge_trim_n: int = 16, num_workers=None):
    """Create a SparseMatrix from a list of (source, target) or (source, target, weight) tuples.

    When the tuples carry weights they are used as edge weights (edges given as
    pairs count as 1).
    """
    if not edges:
        raise ValueError("edges list must not be empty")

    sources = []
    targets = []
    weights = []
    weighted = False
    for edge in edges:
        if len(edge) == 2:
            src, tgt = edge
            weight = 1.0
        elif len(edge) == 3:
            src, tgt, weight = edge
            weighted = True
        else:
            raise ValueError(f"Each edge must be a (source, target) or (source, target, weight) tuple, got length {len(edge)}")
        sources.append(str(src))
        targets.append(str(tgt))
        weights.append(float(weight))

    return from_arrays(
        sources, targets, weights=weights if weighted else None, columns=columns, num_workers=num_workers
    )


def from_numpy(adjacency_matrix, entity_ids: Optional[List[str]] = None,
               columns: str = "complex::reflexive::node",
               hyperedge_trim_n: int = 16, num_workers=None,
               use_weights: bool = False):
    """Create a SparseMatrix from a dense numpy adjacency matrix.

    If entity_ids is None, uses stringified integer indices as node IDs.
    The adjacency is treated as undirected: an edge is created between nodes i
    and j if either (i,j) or (j,i) is nonzero. With ``use_weights=True`` the
    values become edge weights (the larger magnitude of the two directions).
    """
    if not isinstance(adjacency_matrix, np.ndarray):
        raise ValueError("adjacency_matrix must be a numpy ndarray")

    if adjacency_matrix.ndim != 2:
        raise ValueError(f"adjacency_matrix must be 2-dimensional, got {adjacency_matrix.ndim} dimensions")

    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError(f"adjacency_matrix must be square, got shape {adjacency_matrix.shape}")

    n = adjacency_matrix.shape[0]
    if entity_ids is not None and len(entity_ids) != n:
        raise ValueError(f"entity_ids has {len(entity_ids)} elements but matrix has {n} rows")

    magnitude = np.abs(adjacency_matrix)
    magnitude = np.triu(np.maximum(magnitude, magnitude.T))
    low, high = np.nonzero(magnitude)
    if len(low) == 0:
        raise ValueError("No edges found in the adjacency matrix")
    weights = magnitude[low, high].astype(np.float64) if use_weights else None

    return _from_index_edges(low.astype(np.int64), high.astype(np.int64), weights, entity_ids, columns, num_workers)


def to_edge_list(graph) -> List[Tuple[str, str, float]]:
    """One (source, target, transition probability) triple per undirected edge."""
    rows, cols, vals = _upper_triangle(graph)
    entity_ids = graph.entity_ids
    return [
        (entity_ids[r], entity_ids[c], v)
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist())
    ]
