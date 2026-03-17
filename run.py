"""
pycleora 2.2 - Large-scale benchmark: 100,000 nodes
"""
import numpy as np
import time

from pycleora import (
    SparseMatrix, embed, embed_multiscale, embed_with_attention,
    find_most_similar, cosine_similarity, predict_links,
)
from pycleora.metrics import link_prediction_scores, node_classification_scores, clustering_scores
from pycleora.community import detect_communities_kmeans, modularity

def generate_large_graph(num_nodes=100_000, avg_edges_per_node=10, num_communities=20, seed=42):
    rng = np.random.default_rng(seed)
    community_assignments = rng.integers(0, num_communities, size=num_nodes)

    edges = []
    labels = {}

    for i in range(num_nodes):
        labels[f"n{i}"] = int(community_assignments[i])

    print(f"    Generating edges...")
    t0 = time.time()

    for i in range(num_nodes):
        comm = community_assignments[i]
        num_intra = int(avg_edges_per_node * 0.7)
        num_inter = avg_edges_per_node - num_intra

        same_comm = np.where(community_assignments == comm)[0]
        if len(same_comm) > 1:
            targets_intra = rng.choice(same_comm, size=min(num_intra, len(same_comm) - 1), replace=False)
            for t in targets_intra:
                if t != i:
                    edges.append(f"n{i} n{t}")

        targets_inter = rng.integers(0, num_nodes, size=num_inter)
        for t in targets_inter:
            if t != i:
                edges.append(f"n{i} n{t}")

    edges = list(set(edges))
    print(f"    Generated {len(edges)} unique edges in {time.time()-t0:.2f}s")
    return edges, labels


print("=" * 70)
print("  pycleora 2.2 - Large Scale Benchmark (100K nodes)")
print("=" * 70)

print(f"\n[1] Generating graph with 100,000 nodes, ~10 edges/node, 20 communities...")
edges, labels = generate_large_graph(num_nodes=100_000, avg_edges_per_node=10, num_communities=20)

print(f"\n[2] Building SparseMatrix...")
t0 = time.time()
graph = SparseMatrix.from_iterator(iter(edges), "complex::reflexive::node")
build_time = time.time() - t0
print(f"    {graph}")
print(f"    Build time: {build_time:.2f}s")

print(f"\n[3] Standard embedding (dim=128, 4 iterations)...")
t0 = time.time()
emb = embed(graph, feature_dim=128, num_iterations=4, propagation="left", normalization="l2")
embed_time = time.time() - t0
print(f"    Shape: {emb.shape}")
print(f"    Embedding time: {embed_time:.2f}s")
print(f"    Throughput: {graph.num_entities / embed_time:,.0f} nodes/sec")

print(f"\n[4] Similarity search (top 10 for n0)...")
t0 = time.time()
similar = find_most_similar(graph, emb, "n0", top_k=10)
search_time = time.time() - t0
print(f"    Search time: {search_time:.4f}s")
n0_comm = labels["n0"]
for r in similar[:5]:
    r_comm = labels.get(r['entity_id'], '?')
    same = "SAME" if r_comm == n0_comm else "diff"
    print(f"    {r['entity_id']:10s} sim={r['similarity']:.4f} comm={r_comm} ({same})")

print(f"\n[5] Multi-scale embedding (dim=64, scales=[1,2,4])...")
t0 = time.time()
emb_multi = embed_multiscale(graph, feature_dim=64, scales=[1, 2, 4])
ms_time = time.time() - t0
print(f"    Shape: {emb_multi.shape}, time: {ms_time:.2f}s")

print(f"\n[6] K-means community detection (k=20)...")
t0 = time.time()
comms = detect_communities_kmeans(graph, emb, k=20)
km_time = time.time() - t0
print(f"    Time: {km_time:.2f}s")
mod = modularity(graph, comms)
print(f"    Modularity: {mod:.4f}")

true_labels = np.array([labels[eid] for eid in graph.entity_ids])
pred_labels = np.array([comms[eid] for eid in graph.entity_ids])
cl_scores = clustering_scores(emb, true_labels)
print(f"    NMI: {cl_scores['nmi']:.4f}, Purity: {cl_scores['purity']:.4f}")

print(f"\n[7] Node classification (20 classes, 80/20 split)...")
t0 = time.time()
nc_scores = node_classification_scores(graph, emb, labels, train_ratio=0.8)
nc_time = time.time() - t0
print(f"    Accuracy: {nc_scores['accuracy']:.4f}")
print(f"    Macro-F1: {nc_scores['macro_f1']:.4f}")
print(f"    Time: {nc_time:.2f}s")

print(f"\n[8] Link prediction evaluation...")
rng = np.random.default_rng(123)
test_edge_indices = rng.choice(len(edges), size=min(500, len(edges)), replace=False)
test_edges_lp = []
for idx in test_edge_indices:
    parts = edges[idx].split()
    if len(parts) == 2:
        test_edges_lp.append((parts[0], parts[1]))
t0 = time.time()
lp_scores = link_prediction_scores(graph, emb, test_edges_lp[:200])
lp_time = time.time() - t0
print(f"    AUC: {lp_scores['auc']:.4f}")
print(f"    MRR: {lp_scores['mrr']:.4f}")
print(f"    Hits@10: {lp_scores['hits@10']:.4f}")
print(f"    Time: {lp_time:.2f}s")

print(f"\n[9] Link prediction (top 10 for n0)...")
t0 = time.time()
preds = predict_links(graph, emb, top_k=10, source_entities=["n0"])
lp2_time = time.time() - t0
for p in preds[:5]:
    t_comm = labels.get(p['target'], '?')
    same = "SAME" if t_comm == n0_comm else "diff"
    print(f"    n0 -> {p['target']:10s} score={p['score']:.4f} comm={t_comm} ({same})")
print(f"    Time: {lp2_time:.4f}s")

print(f"\n[10] Symmetric propagation comparison...")
t0 = time.time()
emb_sym = embed(graph, feature_dim=128, num_iterations=4, propagation="symmetric")
sym_time = time.time() - t0
print(f"    Symmetric embed time: {sym_time:.2f}s")
nc_sym = node_classification_scores(graph, emb_sym, labels, train_ratio=0.8)
print(f"    Symmetric accuracy: {nc_sym['accuracy']:.4f} vs Left accuracy: {nc_scores['accuracy']:.4f}")

print(f"\n[11] Sparse matrix export stats...")
rows, cols, vals, n, _ = graph.to_sparse_csr()
print(f"    Matrix size: {n}x{n}")
print(f"    Non-zeros: {len(vals):,}")
print(f"    Density: {len(vals) / (n * n) * 100:.4f}%")
print(f"    Memory (embeddings): {emb.nbytes / 1024 / 1024:.1f} MB")

print("\n" + "=" * 70)
print("  PERFORMANCE SUMMARY")
print("=" * 70)
print(f"    Graph build:        {build_time:8.2f}s")
print(f"    Embedding (128d):   {embed_time:8.2f}s  ({graph.num_entities/embed_time:,.0f} nodes/sec)")
print(f"    Multi-scale:        {ms_time:8.2f}s")
print(f"    Symmetric embed:    {sym_time:8.2f}s")
print(f"    K-means (k=20):     {km_time:8.2f}s")
print(f"    Node classif.:      {nc_time:8.2f}s")
print(f"    Link prediction:    {lp_time:8.2f}s")
print(f"    Similarity search:  {search_time:8.4f}s")
print(f"")
print(f"    Classification accuracy:  {nc_scores['accuracy']:.4f}")
print(f"    Clustering NMI:           {cl_scores['nmi']:.4f}")
print(f"    Link prediction AUC:      {lp_scores['auc']:.4f}")
print("=" * 70)

print("\nPress Ctrl+C to exit.")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass
