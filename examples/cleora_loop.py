import time

import numpy as np
from pycleora import SparseMatrix

start_time = time.time()

# graph = SparseMatrix.from_files(["zaba30_large_5m.tsv"], "basket complex::product", hyperedge_trim_n=16)
graph = SparseMatrix.from_files(["perf_inputs/0.tsv", "perf_inputs/1.tsv", "perf_inputs/2.tsv", "perf_inputs/3.tsv", "perf_inputs/4.tsv", "perf_inputs/5.tsv", "perf_inputs/6.tsv", "perf_inputs/7.tsv"], "complex::reflexive::name")

print("Entities n", len(graph.entity_ids))
# embeddings = np.random.randn(len(graph.entity_ids), 128).astype(np.float32)
embeddings = graph.initialize_deterministically(feature_dim=128, seed=0)

for i in range(3):
    embeddings = graph.left_markov_propagate(embeddings)
    # embeddings = graph.symmetric_markov_propagate(embeddings)

    embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    print(f"Iter {i} finished")

print(graph.entity_ids[:10])

print(f"Took {time.time() - start_time} seconds ")
