import numpy as np
from typing import List, Dict, Optional


class _BallTree:
    def __init__(self, data: np.ndarray):
        self._data = data
        self._n, self._dim = data.shape
        self._norms = np.linalg.norm(data, axis=1, keepdims=True)
        self._norms = np.maximum(self._norms, 1e-10)
        self._normalized = data / self._norms
        self._tree = self._build(np.arange(self._n))

    def _build(self, indices: np.ndarray):
        if len(indices) <= 32:
            return {"indices": indices, "leaf": True}

        points = self._normalized[indices]
        center = points.mean(axis=0)
        center_norm = np.linalg.norm(center)
        if center_norm > 1e-10:
            center = center / center_norm
        radius = np.max(np.linalg.norm(points - center, axis=1))

        spread_axis = np.argmax(np.var(points, axis=0))
        median_val = np.median(points[:, spread_axis])

        left_mask = points[:, spread_axis] <= median_val
        if left_mask.all() or (~left_mask).all():
            mid = len(indices) // 2
            left_mask[:] = False
            left_mask[:mid] = True

        return {
            "leaf": False,
            "center": center,
            "radius": radius,
            "left": self._build(indices[left_mask]),
            "right": self._build(indices[~left_mask]),
        }

    def query(self, query_vec: np.ndarray, top_k: int) -> tuple:
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 1e-10:
            query_normalized = query_vec / q_norm
        else:
            query_normalized = query_vec

        candidates = []
        self._search(self._tree, query_normalized, top_k, candidates)

        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:top_k]

        indices = np.array([c[0] for c in candidates], dtype=np.int64)
        distances = np.array([c[1] for c in candidates], dtype=np.float64)
        return indices, distances

    def _search(self, node, query_vec, top_k, candidates):
        if node["leaf"]:
            points = self._normalized[node["indices"]]
            sims = points @ query_vec
            for i, idx in enumerate(node["indices"]):
                self._insert_candidate(candidates, int(idx), float(sims[i]), top_k)
            return

        center_sim = np.dot(node["center"], query_vec)
        worst_sim = candidates[-1][1] if len(candidates) >= top_k else -2.0

        max_possible_sim = center_sim + node["radius"]
        if len(candidates) >= top_k and max_possible_sim < worst_sim:
            return

        left_center = node["left"].get("center")
        right_center = node["right"].get("center")

        if left_center is not None and right_center is not None:
            left_sim = np.dot(left_center, query_vec)
            right_sim = np.dot(right_center, query_vec)
            if left_sim >= right_sim:
                self._search(node["left"], query_vec, top_k, candidates)
                self._search(node["right"], query_vec, top_k, candidates)
            else:
                self._search(node["right"], query_vec, top_k, candidates)
                self._search(node["left"], query_vec, top_k, candidates)
        else:
            self._search(node["left"], query_vec, top_k, candidates)
            self._search(node["right"], query_vec, top_k, candidates)

    @staticmethod
    def _insert_candidate(candidates, idx, sim, top_k):
        if len(candidates) < top_k:
            candidates.append((idx, sim))
            if len(candidates) == top_k:
                candidates.sort(key=lambda x: -x[1])
        elif sim > candidates[-1][1]:
            candidates[-1] = (idx, sim)
            candidates.sort(key=lambda x: -x[1])


class ANNIndex:
    def __init__(self, graph, embeddings: np.ndarray, method: str = "hnsw"):
        if method not in ("hnsw", "brute"):
            raise ValueError(f"Unknown method: '{method}'. Use 'hnsw' or 'brute'.")

        self._graph = graph
        self._embeddings = embeddings
        self._method = method
        self._n, self._dim = embeddings.shape

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._normalized = embeddings / norms

        self._hnsw_index = None
        self._ball_tree = None

        if method == "hnsw":
            self._build_hnsw()
        elif method == "brute":
            pass

    def _build_hnsw(self):
        try:
            import hnswlib

            self._hnsw_index = hnswlib.Index(space="cosine", dim=self._dim)
            self._hnsw_index.init_index(
                max_elements=self._n, ef_construction=200, M=16
            )
            self._hnsw_index.add_items(self._normalized, np.arange(self._n))
            self._hnsw_index.set_ef(50)
        except ImportError:
            self._ball_tree = _BallTree(self._embeddings)

    def query(self, entity_id: str, top_k: int = 10, exclude_self: bool = True) -> List[Dict]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        idx = self._graph.get_entity_index(entity_id)
        query_vec = self._embeddings[idx]

        fetch_k = top_k + 1 if exclude_self else top_k
        results = self._query_internal(query_vec, fetch_k)

        if exclude_self:
            results = [r for r in results if r["entity_id"] != entity_id]
        return results[:top_k]

    def query_vector(self, vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        return self._query_internal(vector, top_k)

    def _query_internal(self, query_vec: np.ndarray, top_k: int) -> List[Dict]:
        if self._method == "brute":
            return self._brute_search(query_vec, top_k)
        elif self._hnsw_index is not None:
            return self._hnsw_search(query_vec, top_k)
        else:
            return self._ball_tree_search(query_vec, top_k)

    def _brute_search(self, query_vec: np.ndarray, top_k: int) -> List[Dict]:
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 1e-10:
            query_normalized = query_vec / q_norm
        else:
            query_normalized = query_vec

        similarities = self._normalized @ query_normalized
        top_k_clamped = min(top_k, self._n)
        top_indices = np.argpartition(similarities, -top_k_clamped)[-top_k_clamped:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return self._build_results(top_indices, similarities)

    def _hnsw_search(self, query_vec: np.ndarray, top_k: int) -> List[Dict]:
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 1e-10:
            query_normalized = query_vec / q_norm
        else:
            query_normalized = query_vec

        top_k_clamped = min(top_k, self._n)
        labels, distances = self._hnsw_index.knn_query(
            query_normalized.reshape(1, -1), k=top_k_clamped
        )

        indices = labels[0]
        cosine_sims = 1.0 - distances[0]

        return self._build_results(indices, cosine_sims, use_array_values=True)

    def _ball_tree_search(self, query_vec: np.ndarray, top_k: int) -> List[Dict]:
        top_k_clamped = min(top_k, self._n)
        indices, similarities = self._ball_tree.query(query_vec, top_k_clamped)

        return self._build_results(indices, similarities, use_array_values=True)

    def _build_results(
        self,
        indices: np.ndarray,
        similarities,
        use_array_values: bool = False,
    ) -> List[Dict]:
        results = []
        entity_ids = self._graph.entity_ids
        for i, idx in enumerate(indices):
            idx_int = int(idx)
            sim = float(similarities[i]) if use_array_values else float(similarities[idx_int])
            results.append({
                "entity_id": entity_ids[idx_int],
                "index": idx_int,
                "similarity": sim,
            })
        return results
