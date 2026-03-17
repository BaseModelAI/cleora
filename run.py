"""
pycleora demo - demonstrates all library features.
"""
import numpy as np
from pycleora import SparseMatrix, embed, embed_multiscale, find_most_similar, cosine_similarity
import pickle
import time

print("=" * 60)
print("  pycleora 2.1 - Graph Embedding Library (Enhanced)")
print("=" * 60)

edges = [
    "alice item_laptop item_mouse",
    "bob item_mouse item_keyboard item_monitor",
    "carol item_laptop item_keyboard",
    "dave item_monitor item_mouse item_laptop",
    "eve item_keyboard item_laptop item_mouse",
    "frank item_monitor item_keyboard",
    "grace item_laptop item_monitor",
    "heidi item_mouse item_monitor item_keyboard",
]

print("\n[1] Building graph from iterator...")
t0 = time.time()
graph = SparseMatrix.from_iterator(iter(edges), "complex::reflexive::product")
print(f"    {graph}")
print(f"    Built in {time.time()-t0:.4f}s")

print(f"\n[2] Graph properties:")
print(f"    Entities:    {graph.num_entities}")
print(f"    Edges:       {graph.num_edges}")
print(f"    Entity IDs:  {graph.entity_ids}")
print(f"    len(graph):  {len(graph)}")

print(f"\n[3] Entity lookup:")
idx = graph.get_entity_index("alice")
print(f"    alice -> index {idx}")
indices = graph.get_entity_indices(["bob", "carol", "dave"])
print(f"    bob, carol, dave -> indices {indices}")

print(f"\n[4] Standard embedding (left Markov, L2 norm, 4 iterations)...")
t0 = time.time()
emb = embed(graph, feature_dim=128, num_iterations=4, propagation="left")
print(f"    Shape: {emb.shape}, took {time.time()-t0:.4f}s")

print(f"\n[5] Symmetric Markov embedding...")
t0 = time.time()
emb_sym = embed(graph, feature_dim=128, num_iterations=4, propagation="symmetric")
print(f"    Shape: {emb_sym.shape}, took {time.time()-t0:.4f}s")

print(f"\n[6] Multi-scale embedding (scales=[1,2,4,8])...")
t0 = time.time()
emb_multi = embed_multiscale(graph, feature_dim=64, scales=[1, 2, 4, 8])
print(f"    Shape: {emb_multi.shape} (4 scales x 64 dim), took {time.time()-t0:.4f}s")

print(f"\n[7] Similarity search:")
similar = find_most_similar(graph, emb, "alice", top_k=5)
for r in similar:
    print(f"    {r['entity_id']:20s} similarity={r['similarity']:.4f}")

print(f"\n[8] Cosine similarity between alice and bob:")
alice_idx = graph.get_entity_index("alice")
bob_idx = graph.get_entity_index("bob")
sim = cosine_similarity(emb[alice_idx], emb[bob_idx])
print(f"    {sim:.4f}")

print(f"\n[9] Pickle serialization...")
data = pickle.dumps(graph)
graph2 = pickle.loads(data)
print(f"    Serialized size: {len(data)} bytes")
print(f"    Restored: {graph2}")

print(f"\n[10] Embedding with progress callback...")
def on_progress(iteration, embeddings):
    norm = np.linalg.norm(embeddings, axis=1).mean()
    print(f"     Iteration {iteration}: avg norm = {norm:.4f}")
emb_cb = embed(graph, feature_dim=64, num_iterations=3, callback=on_progress)

print(f"\n[11] Error handling (proper Python exceptions):")
try:
    graph.get_entity_index("nonexistent_entity")
except ValueError as e:
    print(f"    Caught ValueError: {e}")

try:
    SparseMatrix.from_files(["bad.xyz"], "product")
except ValueError as e:
    print(f"    Caught ValueError: {e}")

print("\n" + "=" * 60)
print("  All features working correctly!")
print("=" * 60)

print("\nPress Ctrl+C to exit.")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass
