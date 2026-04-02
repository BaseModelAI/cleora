import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from .pycleora import SparseMatrix


class HeteroGraph:
    def __init__(self):
        self._node_types: Dict[str, Dict] = {}
        self._edge_types: Dict[str, Dict] = {}
        self._node_features: Dict[str, Dict[str, np.ndarray]] = {}

    def add_node_type(
        self,
        name: str,
        features: Optional[Dict[str, np.ndarray]] = None,
    ):
        self._node_types[name] = {"features": features or {}}
        if features:
            self._node_features[name] = features

    def add_edge_type(
        self,
        name: str,
        source_type: str,
        target_type: str,
        edges: List[Tuple[str, str]],
        weights: Optional[List[float]] = None,
    ):
        self._edge_types[name] = {
            "source_type": source_type,
            "target_type": target_type,
            "edges": edges,
            "weights": weights,
        }

    @property
    def node_types(self) -> List[str]:
        return list(self._node_types.keys())

    @property
    def edge_types(self) -> List[str]:
        return list(self._edge_types.keys())

    def num_nodes(self, node_type: Optional[str] = None) -> int:
        if node_type:
            features = self._node_types.get(node_type, {}).get("features", {})
            if features:
                return len(features)
            count = set()
            for et_info in self._edge_types.values():
                if et_info["source_type"] == node_type:
                    count.update(e[0] for e in et_info["edges"])
                if et_info["target_type"] == node_type:
                    count.update(e[1] for e in et_info["edges"])
            return len(count)
        total = set()
        for nt in self._node_types:
            features = self._node_types[nt].get("features", {})
            if features:
                total.update(f"{nt}_{k}" for k in features.keys())
        for et_info in self._edge_types.values():
            src_t = et_info["source_type"]
            tgt_t = et_info["target_type"]
            total.update(f"{src_t}_{e[0]}" for e in et_info["edges"])
            total.update(f"{tgt_t}_{e[1]}" for e in et_info["edges"])
        return len(total)

    def num_edges(self, edge_type: Optional[str] = None) -> int:
        if edge_type:
            return len(self._edge_types.get(edge_type, {}).get("edges", []))
        return sum(len(info["edges"]) for info in self._edge_types.values())

    def get_edges(self, edge_type: str) -> List[Tuple[str, str]]:
        if edge_type not in self._edge_types:
            raise ValueError(f"Unknown edge type: '{edge_type}'")
        return self._edge_types[edge_type]["edges"]

    def to_homogeneous_edges(self) -> List[str]:
        all_edges = []
        for et_info in self._edge_types.values():
            src_type = et_info["source_type"]
            tgt_type = et_info["target_type"]
            for src, tgt in et_info["edges"]:
                s_id = f"{src_type}_{src}" if len(self._node_types) > 1 else src
                t_id = f"{tgt_type}_{tgt}" if len(self._node_types) > 1 else tgt
                all_edges.append(f"{s_id} {t_id}")
        return all_edges

    def embed_per_relation(
        self,
        feature_dim: int = 256,
        num_iterations: int = 40,
        propagation: str = "left",
        normalization: str = "l2",
        combine: str = "concat",
        seed: int = 0,
        whiten: bool = True,
    ) -> Tuple[Dict[str, SparseMatrix], Dict[str, np.ndarray], Optional[np.ndarray]]:
        from . import embed

        graphs = {}
        embeddings = {}

        for et_name, et_info in self._edge_types.items():
            src_type = et_info["source_type"]
            tgt_type = et_info["target_type"]

            columns = f"complex::reflexive::node"

            edge_strs = []
            for src, tgt in et_info["edges"]:
                s_id = f"{src_type}_{src}"
                t_id = f"{tgt_type}_{tgt}"
                edge_strs.append(f"{s_id} {t_id}")

            graph = SparseMatrix.from_iterator(iter(edge_strs), columns)
            emb = embed(graph, feature_dim=feature_dim, num_iterations=num_iterations,
                       propagation=propagation, normalization=normalization, seed=seed,
                       whiten=whiten)

            graphs[et_name] = graph
            embeddings[et_name] = emb

        combined = None
        if combine == "concat" and len(embeddings) > 1:
            all_entities = set()
            for g in graphs.values():
                all_entities.update(g.entity_ids)
            all_entities = sorted(all_entities)
            entity_to_idx = {e: i for i, e in enumerate(all_entities)}

            parts = []
            for et_name in self._edge_types:
                g = graphs[et_name]
                emb = embeddings[et_name]
                part = np.zeros((len(all_entities), emb.shape[1]), dtype=np.float32)
                for i, eid in enumerate(g.entity_ids):
                    if eid in entity_to_idx:
                        part[entity_to_idx[eid]] = emb[i]
                parts.append(part)

            combined = np.concatenate(parts, axis=1)
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            combined = combined / norms

        elif combine == "mean" and len(embeddings) > 1:
            all_entities = set()
            for g in graphs.values():
                all_entities.update(g.entity_ids)
            all_entities = sorted(all_entities)
            entity_to_idx = {e: i for i, e in enumerate(all_entities)}

            dim = feature_dim
            combined = np.zeros((len(all_entities), dim), dtype=np.float64)
            counts = np.zeros(len(all_entities), dtype=np.float64)

            for et_name in self._edge_types:
                g = graphs[et_name]
                emb = embeddings[et_name]
                for i, eid in enumerate(g.entity_ids):
                    if eid in entity_to_idx:
                        idx = entity_to_idx[eid]
                        combined[idx] += emb[i].astype(np.float64)
                        counts[idx] += 1

            counts = np.maximum(counts, 1)
            combined = (combined / counts[:, None]).astype(np.float32)
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            combined = combined / norms

        return graphs, embeddings, combined

    def embed_metapath(
        self,
        metapath: List[str],
        feature_dim: int = 256,
        num_iterations: int = 40,
        normalization: str = "l2",
        seed: int = 0,
        whiten: bool = True,
    ) -> Tuple[SparseMatrix, np.ndarray]:
        from . import embed

        if len(metapath) < 2:
            raise ValueError("Metapath must have at least 2 edge types")

        for et in metapath:
            if et not in self._edge_types:
                raise ValueError(f"Unknown edge type in metapath: '{et}'")

        adjacencies = []
        for et_name in metapath:
            et_info = self._edge_types[et_name]
            src_type = et_info["source_type"]
            tgt_type = et_info["target_type"]
            adj = {}
            for src, tgt in et_info["edges"]:
                s_id = f"{src_type}_{src}"
                t_id = f"{tgt_type}_{tgt}"
                if s_id not in adj:
                    adj[s_id] = set()
                adj[s_id].add(t_id)
            adjacencies.append(adj)

        def compose_paths(adj_list):
            if len(adj_list) == 1:
                return adj_list[0]
            first = adj_list[0]
            rest = compose_paths(adj_list[1:])
            result = {}
            for src, mid_nodes in first.items():
                targets = set()
                for mid in mid_nodes:
                    if mid in rest:
                        targets.update(rest[mid])
                if targets:
                    result[src] = targets
            return result

        composed = compose_paths(adjacencies)

        edge_strs = []
        for src, targets in composed.items():
            for tgt in targets:
                if src != tgt:
                    edge_strs.append(f"{src} {tgt}")

        if not edge_strs:
            raise ValueError("Metapath produced no edges")

        columns = "complex::reflexive::node"

        graph = SparseMatrix.from_iterator(iter(edge_strs), columns)
        emb = embed(graph, feature_dim=feature_dim, num_iterations=num_iterations,
                   normalization=normalization, seed=seed, whiten=whiten)

        return graph, emb

    def summary(self) -> str:
        lines = ["HeteroGraph:"]
        lines.append(f"  Node types: {len(self._node_types)}")
        for nt in self._node_types:
            lines.append(f"    - {nt}: {self.num_nodes(nt)} nodes")
        lines.append(f"  Edge types: {len(self._edge_types)}")
        for et_name, et_info in self._edge_types.items():
            lines.append(f"    - {et_name} ({et_info['source_type']} -> {et_info['target_type']}): "
                        f"{len(et_info['edges'])} edges")
        lines.append(f"  Total nodes: {self.num_nodes()}")
        lines.append(f"  Total edges: {self.num_edges()}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"HeteroGraph(node_types={len(self._node_types)}, "
                f"edge_types={len(self._edge_types)}, "
                f"nodes={self.num_nodes()}, edges={self.num_edges()})")
