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
