import numpy as np

from .pycleora import SparseMatrix

def embed_using_baseline_cleora(graph, feature_dim: int, iter: int):
    embeddings = graph.initialize_deterministically(feature_dim)
    for i in range(iter):
        embeddings = graph.left_markov_propagate(embeddings)
        embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    return embeddings