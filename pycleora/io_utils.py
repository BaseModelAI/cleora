import numpy as np
from typing import Dict, List, Optional, Tuple


def to_networkx(graph, embeddings: Optional[np.ndarray] = None):
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for graph export. Install with: pip install networkx")

    G = nx.Graph()

    for i, eid in enumerate(graph.entity_ids):
        attrs = {"index": i}
        if embeddings is not None:
            attrs["embedding"] = embeddings[i].tolist()
        G.add_node(eid, **attrs)

    rows, cols, vals, _, _ = graph.to_sparse_csr()
    seen = set()
    for r, c, v in zip(rows, cols, vals):
        r, c = int(r), int(c)
        edge_key = (min(r, c), max(r, c))
        if edge_key not in seen:
            seen.add(edge_key)
            G.add_edge(graph.entity_ids[r], graph.entity_ids[c], weight=float(v))

    return G


def from_networkx(G, columns: str = "complex::reflexive::node", hyperedge_trim_n: int = 16, num_workers=None):
    from .pycleora import SparseMatrix

    edges = []
    for u, v in G.edges():
        edges.append(f"{u} {v}")

    return SparseMatrix.from_iterator(iter(edges), columns, hyperedge_trim_n, num_workers)


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

        arrays = {"entity_id": graph.entity_ids}
        for i in range(embeddings.shape[1]):
            arrays[f"dim_{i}"] = embeddings[:, i].tolist()
        table = pa.table(arrays)
        pq.write_table(table, filepath)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz', 'csv', 'tsv', or 'parquet'.")


def load_embeddings(filepath: str, format: str = "npz") -> Tuple[np.ndarray, List[str]]:
    if format == "npz":
        data = np.load(filepath, allow_pickle=True)
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


def from_pandas(df, source_col: str, target_col: str, weight_col: Optional[str] = None,
                 columns: str = "complex::reflexive::node",
                 hyperedge_trim_n: int = 16, num_workers=None):
    """Create a SparseMatrix from a pandas DataFrame.

    Each row becomes an edge from source_col to target_col. When weight_col is
    provided, rows with NaN or zero weight are filtered out; the weight values
    themselves are not encoded into the SparseMatrix (use embed_weighted for
    weighted graph embedding).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame import. Install with: pip install pandas")

    from .pycleora import SparseMatrix

    if source_col not in df.columns:
        raise ValueError(f"source_col '{source_col}' not found in DataFrame columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame columns: {list(df.columns)}")
    if weight_col is not None and weight_col not in df.columns:
        raise ValueError(f"weight_col '{weight_col}' not found in DataFrame columns: {list(df.columns)}")

    edges = []
    for _, row in df.iterrows():
        src = row[source_col]
        tgt = row[target_col]
        if pd.isna(src) or pd.isna(tgt):
            continue
        if weight_col is not None:
            w = row[weight_col]
            if pd.isna(w) or float(w) == 0:
                continue
        edges.append(f"{src} {tgt}")

    if not edges:
        raise ValueError("No valid edges found in DataFrame (all rows may have NaN values)")

    return SparseMatrix.from_iterator(iter(edges), columns, hyperedge_trim_n, num_workers)


def from_scipy_sparse(matrix, entity_ids: Optional[List[str]] = None,
                      columns: str = "complex::reflexive::node",
                      hyperedge_trim_n: int = 16, num_workers=None):
    """Create a SparseMatrix from a scipy sparse adjacency matrix.

    If entity_ids is None, uses stringified integer indices as node IDs.
    The adjacency is treated as undirected: symmetric edge pairs are
    deduplicated so each edge appears once.
    """
    try:
        import scipy.sparse
    except ImportError:
        raise ImportError("scipy is required for sparse matrix import. Install with: pip install scipy")

    from .pycleora import SparseMatrix

    if not scipy.sparse.issparse(matrix):
        raise ValueError("matrix must be a scipy sparse matrix")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")

    n = matrix.shape[0]
    if entity_ids is not None:
        if len(entity_ids) != n:
            raise ValueError(f"entity_ids has {len(entity_ids)} elements but matrix has {n} rows")
        ids = [str(eid) for eid in entity_ids]
    else:
        ids = [str(i) for i in range(n)]

    coo = matrix.tocoo()
    seen = set()
    edges = []
    for r, c in zip(coo.row, coo.col):
        edge_key = (min(r, c), max(r, c))
        if edge_key not in seen:
            seen.add(edge_key)
            edges.append(f"{ids[r]} {ids[c]}")

    if not edges:
        raise ValueError("No edges found in the sparse matrix")

    return SparseMatrix.from_iterator(iter(edges), columns, hyperedge_trim_n, num_workers)


def from_edge_list(edges: List, columns: str = "complex::reflexive::node",
                   hyperedge_trim_n: int = 16, num_workers=None):
    """Create a SparseMatrix from a list of (source, target) or (source, target, weight) tuples.

    Weight values in 3-tuples are accepted but not encoded into the SparseMatrix
    (use embed_weighted for weighted graph embedding).
    """
    from .pycleora import SparseMatrix

    if not edges:
        raise ValueError("edges list must not be empty")

    edge_strs = []
    for edge in edges:
        if len(edge) == 2:
            src, tgt = edge
            edge_strs.append(f"{src} {tgt}")
        elif len(edge) == 3:
            src, tgt, _weight = edge
            edge_strs.append(f"{src} {tgt}")
        else:
            raise ValueError(f"Each edge must be a (source, target) or (source, target, weight) tuple, got length {len(edge)}")

    return SparseMatrix.from_iterator(iter(edge_strs), columns, hyperedge_trim_n, num_workers)


def from_numpy(adjacency_matrix, entity_ids: Optional[List[str]] = None,
               columns: str = "complex::reflexive::node",
               hyperedge_trim_n: int = 16, num_workers=None):
    """Create a SparseMatrix from a dense numpy adjacency matrix.

    If entity_ids is None, uses stringified integer indices as node IDs.
    The adjacency is treated as undirected: an edge is created between nodes i
    and j if either (i,j) or (j,i) is nonzero.
    """
    from .pycleora import SparseMatrix

    if not isinstance(adjacency_matrix, np.ndarray):
        raise ValueError("adjacency_matrix must be a numpy ndarray")

    if adjacency_matrix.ndim != 2:
        raise ValueError(f"adjacency_matrix must be 2-dimensional, got {adjacency_matrix.ndim} dimensions")

    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError(f"adjacency_matrix must be square, got shape {adjacency_matrix.shape}")

    n = adjacency_matrix.shape[0]
    if entity_ids is not None:
        if len(entity_ids) != n:
            raise ValueError(f"entity_ids has {len(entity_ids)} elements but matrix has {n} rows")
        ids = [str(eid) for eid in entity_ids]
    else:
        ids = [str(i) for i in range(n)]

    edges = []
    for i in range(n):
        for j in range(i, n):
            if adjacency_matrix[i, j] != 0 or adjacency_matrix[j, i] != 0:
                edges.append(f"{ids[i]} {ids[j]}")

    if not edges:
        raise ValueError("No edges found in the adjacency matrix")

    return SparseMatrix.from_iterator(iter(edges), columns, hyperedge_trim_n, num_workers)


def to_edge_list(graph) -> List[Tuple[str, str, float]]:
    rows, cols, vals, _, _ = graph.to_sparse_csr()
    seen = set()
    result = []
    for r, c, v in zip(rows, cols, vals):
        r, c = int(r), int(c)
        edge_key = (min(r, c), max(r, c))
        if edge_key not in seen:
            seen.add(edge_key)
            result.append((graph.entity_ids[r], graph.entity_ids[c], float(v)))
    return result
