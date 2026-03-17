"""
pycleora demo script - demonstrates the library working in Replit.
pycleora is a Python library for graph embeddings (no web server needed).
"""
import numpy as np
from pycleora import SparseMatrix, embed_using_baseline_cleora

print("=" * 50)
print("pycleora - Graph Embedding Library")
print("=" * 50)

print("\nBuilding graph from sample edge list...")
graph = SparseMatrix.from_files(
    ["files/samples/edgelist_1.tsv"],
    "complex::reflexive::product",
)
print(f"Entities in graph: {len(graph.entity_ids)}")

print("\nBuilding graph from inline edges (iterator)...")
edges = [
    "user1 item1 item2",
    "user2 item2 item3",
    "user3 item1 item3",
    "user1 item3 item4",
    "user4 item1 item4",
]
graph = SparseMatrix.from_iterator(iter(edges), "complex::reflexive::product")
print(f"Entities in graph: {len(graph.entity_ids)}")
print(f"Entity IDs: {graph.entity_ids}")

print("\nRunning embedding (3 iterations, dim=64)...")
embeddings = embed_using_baseline_cleora(graph, feature_dim=64, iter=3)
print(f"Embedding shape: {embeddings.shape}")
print(f"Sample embedding (first entity): {embeddings[0][:5]}...")

print("\npycleora is working correctly!")
print("\nSee examples/ folder for more usage examples.")
print("Press Ctrl+C to exit.")

import time
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass
