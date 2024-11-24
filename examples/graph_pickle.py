import time

import numpy as np
from pycleora import SparseMatrix

import pickle

start_time = time.time()

graph = SparseMatrix.from_files(["perf_inputs/0.tsv"], "complex::reflexive::name")

print("Entities n", len(graph.entity_ids))
print(graph.entity_ids[:10])

with open('graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

with open('graph.pkl', 'rb') as f:
    graph_reread = pickle.load(f)

print(graph.entity_ids[:10])
print(graph_reread.entity_ids[:10])

embeddings = graph_reread.initialize_deterministically(feature_dim=128, seed=0)
embeddings = graph_reread.left_markov_propagate(embeddings)

print(embeddings)