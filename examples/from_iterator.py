import time

import numpy as np
from pycleora import SparseMatrix

start_time = time.time()

def edges_iterator():
    lines = []

    files = ["perf_inputs/0.tsv", "perf_inputs/1.tsv", "perf_inputs/2.tsv", "perf_inputs/3.tsv", "perf_inputs/4.tsv", "perf_inputs/5.tsv", "perf_inputs/6.tsv", "perf_inputs/7.tsv"]
    for file in files:
        with open(file, 'rt') as f:
            lines.extend(f)

    iteration_start_time = time.time()
    for line in lines:
        yield line
    print(f"Iteration took {time.time() - iteration_start_time} seconds ")

graph = SparseMatrix.from_iterator(edges_iterator(), "complex::reflexive::product")

print("Entities n", len(graph.entity_ids))
print(graph.entity_ids[:10])

embeddings = np.random.randn(len(graph.entity_ids), 256).astype(np.float32)

for i in range(3):
    embeddings = graph.left_markov_propagate(embeddings)
    embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    print(f"Iter {i} finished")

print(f"Took {time.time() - start_time} seconds ")