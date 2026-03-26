import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time

from pycleora import SparseMatrix, embed
from pycleora.algorithms import (
    embed_prone, embed_randne, embed_hope, embed_netmf,
    embed_grarep, embed_deepwalk, embed_node2vec,
)
from pycleora.datasets import load_dataset
from pycleora.community import detect_communities_louvain

DIM = 64
OUTPUT_DIR = "website/static/umap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGO_COLORS = {
    "Cleora": "#a78bfa",
    "ProNE": "#f59e0b",
    "RandNE": "#ef4444",
    "NetMF": "#3b82f6",
    "DeepWalk": "#f472b6",
    "HOPE": "#34d399",
    "GraRep": "#fb923c",
    "Node2Vec": "#22d3ee",
}

CLASS_PALETTES = {
    3: ["#e74c3c", "#3498db", "#2ecc71"],
    6: ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"],
    7: ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e91e63"],
}


def get_class_colors(num_classes):
    if num_classes in CLASS_PALETTES:
        return CLASS_PALETTES[num_classes]
    cmap = plt.cm.get_cmap("tab20", num_classes)
    return [matplotlib.colors.to_hex(cmap(i)) for i in range(num_classes)]


def make_algo_fn(algo_name, graph):
    if algo_name == "Cleora":
        return embed(graph, DIM, num_iterations=40, propagation="left", normalization="l2", whiten=True, seed=42)
    elif algo_name == "ProNE":
        return embed_prone(graph, DIM)
    elif algo_name == "RandNE":
        return embed_randne(graph, DIM)
    elif algo_name == "HOPE":
        return embed_hope(graph, DIM)
    elif algo_name == "NetMF":
        return embed_netmf(graph, DIM)
    elif algo_name == "GraRep":
        return embed_grarep(graph, DIM)
    elif algo_name == "DeepWalk":
        return embed_deepwalk(graph, DIM, num_walks=10, walk_length=20)
    elif algo_name == "Node2Vec":
        return embed_node2vec(graph, DIM, num_walks=10, walk_length=20, p=1.0, q=0.5)


def save_umap_plot(emb_2d, labels_arr, class_colors, algo_name, dataset_name, algo_color, num_classes):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')

    unique_labels = np.unique(labels_arr)
    for label in unique_labels:
        mask = labels_arr == label
        color = class_colors[int(label) % len(class_colors)]
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=color, s=3, alpha=0.6, edgecolors='none', rasterized=True
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(algo_name, color=algo_color, fontsize=14, fontweight='bold', pad=8)

    fname = f"{dataset_name.lower()}_{algo_name.lower()}.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(fpath, bbox_inches='tight', facecolor='#0a0a0f', edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved {fpath}")
    return fname


def run_dataset(ds_key, ds_display, algo_names):
    import umap

    print(f"\n{'='*60}")
    print(f"Dataset: {ds_display}")
    print(f"{'='*60}")

    ds = load_dataset(ds_key)
    graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
    labels = ds["labels"]
    num_classes = ds["num_classes"]

    if not labels or len(labels) < 4:
        print(f"  No labels, using Louvain communities...")
        labels = detect_communities_louvain(graph)
        num_classes = len(set(labels.values()))
        print(f"  Found {num_classes} communities")

    entity_ids = graph.entity_ids
    labels_arr = np.array([labels.get(eid, 0) for eid in entity_ids])

    unique_labels = np.unique(labels_arr)
    label_remap = {old: new for new, old in enumerate(unique_labels)}
    labels_arr = np.array([label_remap[l] for l in labels_arr])
    actual_classes = len(unique_labels)
    class_colors = get_class_colors(actual_classes)

    embeddings = {}
    for algo_name in algo_names:
        out_path = os.path.join(OUTPUT_DIR, f"{ds_display.lower()}_{algo_name.lower()}.png")
        if os.path.exists(out_path):
            print(f"  {algo_name}: already exists, skipping")
            continue
        print(f"  Running {algo_name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            emb = make_algo_fn(algo_name, graph)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.2f}s)")
            embeddings[algo_name] = emb
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.2f}s): {e}")

    if embeddings:
        print(f"\n  Running UMAP for {len(embeddings)} embeddings...")
        for algo_name, emb in embeddings.items():
            print(f"  UMAP {algo_name}...", end=" ", flush=True)
            t0 = time.time()
            try:
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                emb_2d = reducer.fit_transform(emb)
                elapsed = time.time() - t0
                print(f"done ({elapsed:.2f}s)")
                save_umap_plot(emb_2d, labels_arr, class_colors, algo_name, ds_display, ALGO_COLORS.get(algo_name, "#ffffff"), actual_classes)
            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED ({elapsed:.2f}s): {e}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "all"

    configs = {
        "cora": ("cora", "Cora", ["Cleora", "NetMF", "ProNE", "RandNE", "HOPE", "DeepWalk"]),
        "citeseer": ("citeseer", "CiteSeer", ["Cleora", "NetMF", "ProNE", "RandNE", "HOPE"]),
        "facebook": ("facebook", "Facebook", ["Cleora", "NetMF", "ProNE", "RandNE"]),
        "pubmed": ("pubmed", "PubMed", ["Cleora", "RandNE", "ProNE"]),
        "ppi": ("ppi", "PPI", ["Cleora", "RandNE", "ProNE"]),
    }

    if dataset == "all":
        for key in configs:
            run_dataset(*configs[key])
    elif dataset in configs:
        run_dataset(*configs[dataset])
    else:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {', '.join(configs.keys())}")
