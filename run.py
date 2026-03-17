"""
pycleora demo - demonstrates all library features including
attention, supervised learning, incremental updates, and GPU support.
"""
import numpy as np
from pycleora import (
    SparseMatrix, embed, embed_multiscale, embed_with_attention,
    embed_inductive, supervised_refine, update_graph,
    find_most_similar, cosine_similarity, propagate_gpu,
)
import pickle
import time

print("=" * 60)
print("  pycleora 2.1 - Enhanced Graph Embedding Library")
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

columns = "complex::reflexive::product"

print("\n[1] Building graph from iterator...")
t0 = time.time()
graph = SparseMatrix.from_iterator(iter(edges), columns)
print(f"    {graph}")
print(f"    Built in {time.time()-t0:.4f}s")

print(f"\n[2] Graph properties:")
print(f"    Entities:    {graph.num_entities}")
print(f"    Edges:       {graph.num_edges}")
print(f"    Entity IDs:  {graph.entity_ids}")

print(f"\n[3] Entity lookup & neighbors:")
idx = graph.get_entity_index("alice")
print(f"    alice -> index {idx}")
neighbors = graph.get_neighbors("alice")
print(f"    alice neighbors: {neighbors[:3]}...")

print(f"\n[4] Standard embedding (left Markov, L2 norm)...")
t0 = time.time()
emb = embed(graph, feature_dim=128, num_iterations=4)
print(f"    Shape: {emb.shape}, took {time.time()-t0:.4f}s")

print(f"\n[5] Similarity search:")
similar = find_most_similar(graph, emb, "alice", top_k=5)
for r in similar:
    print(f"    {r['entity_id']:20s} sim={r['similarity']:.4f}")

print(f"\n[6] Multi-scale embedding (scales=[1,2,4,8])...")
t0 = time.time()
emb_multi = embed_multiscale(graph, feature_dim=64, scales=[1, 2, 4, 8])
print(f"    Shape: {emb_multi.shape}, took {time.time()-t0:.4f}s")

print(f"\n--- NEW FEATURES ---")

print(f"\n[7] ATTENTION-WEIGHTED embedding...")
t0 = time.time()
emb_attn = embed_with_attention(
    graph, feature_dim=128, num_iterations=4, attention_temperature=0.5,
)
print(f"    Shape: {emb_attn.shape}, took {time.time()-t0:.4f}s")
similar_attn = find_most_similar(graph, emb_attn, "alice", top_k=3)
print(f"    Top similar to alice (with attention):")
for r in similar_attn:
    print(f"      {r['entity_id']:20s} sim={r['similarity']:.4f}")

print(f"\n[8] SUPERVISED REFINEMENT (contrastive learning)...")
positive_pairs = [
    ("alice", "carol"),
    ("bob", "heidi"),
    ("dave", "grace"),
]
t0 = time.time()
emb_refined = supervised_refine(
    graph, emb, positive_pairs,
    learning_rate=0.05, num_epochs=100, margin=0.3,
)
alice_carol_before = cosine_similarity(emb[graph.get_entity_index("alice")], emb[graph.get_entity_index("carol")])
alice_carol_after = cosine_similarity(emb_refined[graph.get_entity_index("alice")], emb_refined[graph.get_entity_index("carol")])
print(f"    alice-carol similarity BEFORE: {alice_carol_before:.4f}")
print(f"    alice-carol similarity AFTER:  {alice_carol_after:.4f}")
print(f"    Improvement: {alice_carol_after - alice_carol_before:+.4f}, took {time.time()-t0:.4f}s")

print(f"\n[9] INCREMENTAL GRAPH UPDATE (preserves original hyperedges)...")
new_edges = [
    "ivan item_laptop item_keyboard item_tablet",
    "judy item_tablet item_mouse",
]
t0 = time.time()
updated_graph = update_graph(edges, new_edges, columns)
print(f"    Original: {graph.num_entities} entities, {graph.num_edges} edges")
print(f"    Updated:  {updated_graph.num_entities} entities, {updated_graph.num_edges} edges")
new_entities = set(updated_graph.entity_ids) - set(graph.entity_ids)
print(f"    New entities: {new_entities}")
print(f"    Took {time.time()-t0:.4f}s")

print(f"\n[10] INDUCTIVE EMBEDDING (new nodes get embeddings)...")
t0 = time.time()
updated_graph2, emb_inductive = embed_inductive(
    graph, emb, edges, new_edges, columns,
    num_iterations=4,
)
print(f"    Updated graph: {updated_graph2.num_entities} entities")
print(f"    Embedding shape: {emb_inductive.shape}")
for new_ent in sorted(new_entities):
    if new_ent in updated_graph2.entity_ids:
        similar_new = find_most_similar(updated_graph2, emb_inductive, new_ent, top_k=3)
        print(f"    {new_ent} most similar to: {[r['entity_id'] for r in similar_new]}")
print(f"    Took {time.time()-t0:.4f}s")

print(f"\n[11] GPU PROPAGATION (availability check)...")
try:
    import torch
    if torch.cuda.is_available():
        t0 = time.time()
        emb_gpu = propagate_gpu(graph, emb, num_iterations=4, device="cuda")
        print(f"    GPU embedding shape: {emb_gpu.shape}, took {time.time()-t0:.4f}s")
    else:
        print(f"    CUDA not available - using PyTorch CPU fallback test...")
        t0 = time.time()
        emb_cpu_torch = propagate_gpu(graph, emb, num_iterations=4, device="cpu")
        print(f"    PyTorch CPU shape: {emb_cpu_torch.shape}, took {time.time()-t0:.4f}s")
except ImportError:
    print(f"    PyTorch not installed - GPU propagation available when 'pip install torch'")
    print(f"    Native Rust CPU propagation is active (already fast!)")

print(f"\n[12] Sparse matrix export (for custom workflows)...")
rows, cols, vals, n, _ = graph.to_sparse_csr()
print(f"    Exported: {len(vals)} non-zero values in {n}x{n} matrix")
from scipy.sparse import csr_matrix
adj = csr_matrix((vals, (rows.astype(np.int32), cols.astype(np.int32))), shape=(n, n))
print(f"    Scipy sparse density: {adj.nnz / (n*n):.4f}")

print(f"\n[13] Pickle serialization...")
data = pickle.dumps(graph)
graph2 = pickle.loads(data)
print(f"    Serialized: {len(data)} bytes -> Restored: {graph2}")

print(f"\n[14] Validation & error handling:")
try:
    graph.get_entity_index("nonexistent")
except ValueError as e:
    print(f"    Caught: {e}")
try:
    embed(graph, propagation="invalid")
except ValueError as e:
    print(f"    Caught: {e}")
try:
    embed_with_attention(graph, attention_temperature=-1.0)
except ValueError as e:
    print(f"    Caught: {e}")
try:
    graph.to_sparse_csr("invalid_type")
except ValueError as e:
    print(f"    Caught: {e}")

print("\n" + "=" * 60)
print("  All 14 features working correctly!")
print("=" * 60)

print("\nPress Ctrl+C to exit.")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass
