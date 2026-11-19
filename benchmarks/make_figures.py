#!/usr/bin/env python
"""UMAP pictures of the embeddings: one figure per dataset, one panel per method.

    python benchmarks/make_figures.py --data ~/cleora-bench --out benchmarks/figures

Every panel is a 2-D UMAP (umap-learn, cosine metric, fixed seed) of the same random
sample of nodes, taken from the vectors the benchmark stored for the whole graph
(``sample.npz`` of seed 0). Points are coloured by node class - the eight largest
classes, the others in grey - or, for graphs without labels, by degree.

The pictures illustrate; they are not evidence. Whether one method separates classes
better than another is what the tables measure.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleora_bench import datasets, display  # noqa: E402

SURFACE = "#fcfcfb"
INK, MUTED = "#0b0b0b", "#52514e"
# categorical slots in their fixed order (a class keeps its colour in every panel)
CLASS_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
OTHER = "#b9b8b1"
DEGREE_RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # one hue, light to dark
MAX_POINTS = 15_000


def coordinates(sample_path, seed=0):
    """2-D UMAP of a stored sample (cached next to it)."""
    cache = os.path.join(os.path.dirname(sample_path), "umap.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import umap

    with np.load(sample_path) as archive:
        vectors = archive["vectors"][:MAX_POINTS].astype(np.float32)
    layout = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                       random_state=seed).fit_transform(vectors)
    np.save(cache, layout.astype(np.float32))
    return layout


def point_classes(graph, rows):
    """For every sampled node: the rank of its class by size (0 = largest; multi-label
    nodes take their most common label), or ``None`` when the graph has no labels."""
    if "labels" not in graph:
        return None, None
    labels = graph["labels"]
    if labels.ndim == 2:
        frequency = labels.sum(axis=0)
        chosen = np.where(labels[rows].any(axis=1), (labels[rows] * frequency).argmax(axis=1), -1)
    else:
        chosen = labels[rows]
    known = chosen[chosen >= 0]
    order = np.argsort(-np.bincount(known, minlength=int(known.max()) + 1), kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return np.where(chosen >= 0, rank[np.maximum(chosen, 0)], len(order)), int((np.bincount(known) > 0).sum())


def draw_dataset(name, panels, graph, path, columns=4):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    rows_of_nodes = panels[0][2]
    classes, num_classes = point_classes(graph, rows_of_nodes)
    if classes is None:
        degree = np.bincount(graph["edges"].ravel(), minlength=graph["meta"]["num_nodes"])[rows_of_nodes]
        colour_values = np.log10(np.maximum(degree, 1))
    grid_rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(grid_rows, columns, figsize=(2.9 * columns, 2.9 * grid_rows + 0.9), squeeze=False,
                                facecolor=SURFACE)
    point_size = float(np.clip(6000 / len(rows_of_nodes), 0.6, 6))
    for axis in axes.ravel():
        axis.set_facecolor(SURFACE)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis, (label, layout, _) in zip(axes.ravel(), panels):
        if classes is None:
            ramp = LinearSegmentedColormap.from_list("degree", DEGREE_RAMP)
            axis.scatter(layout[:, 0], layout[:, 1], c=colour_values, cmap=ramp, s=point_size, linewidths=0, alpha=0.8,
                         rasterized=True)
        else:
            colours = np.array(list(CLASS_COLOURS) + [OTHER])[np.minimum(classes, len(CLASS_COLOURS))]
            order = np.argsort(-np.minimum(classes, len(CLASS_COLOURS)), kind="stable")  # grey first, below the colours
            axis.scatter(layout[order, 0], layout[order, 1], c=colours[order], s=point_size, linewidths=0, alpha=0.8,
                         rasterized=True)
        axis.set_title(label, fontsize=10, color=INK, loc="left", pad=4)
    meta = graph["meta"]
    figure.suptitle(f"{display.DATASET_LABELS.get(name, name)}  ·  {meta['num_nodes']:,} nodes, {meta['num_edges']:,} edges",
                    x=0.01, y=0.995, ha="left", va="top", fontsize=12, color=INK)
    if classes is None:
        caption = "No node labels: colour = degree (light = low, dark = high)"
        figure.text(0.01, 0.012, caption, fontsize=9, color=MUTED, ha="left", va="bottom")
    else:
        shown = min(num_classes, len(CLASS_COLOURS))
        handles = [Line2D([], [], marker="o", linestyle="", markersize=6, color=CLASS_COLOURS[i],
                          label=f"class {i + 1}") for i in range(shown)]
        if num_classes > shown:
            handles.append(Line2D([], [], marker="o", linestyle="", markersize=6, color=OTHER,
                                  label=f"{num_classes - shown} smaller classes"))
        legend = figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=len(handles),
                               frameon=False, fontsize=9, handletextpad=0.2, columnspacing=1.0,
                               title="Classes by size" if shown > 1 else None, title_fontsize=9, alignment="left")
        for text in legend.get_texts():
            text.set_color(MUTED)
    figure.tight_layout(rect=(0, 0.05, 1, 0.975))
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="directory with datasets/ and results/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--dim", type=int, default=1024)
    args = parser.parse_args()
    args.data = os.path.abspath(os.path.expanduser(args.data))
    os.makedirs(args.out, exist_ok=True)
    results = os.path.join(args.data, "results")
    names = args.datasets.split(",") if args.datasets else sorted(os.listdir(results))
    for name in names:
        panels = []
        for row in display.FIGURE_METHODS:
            # the benchmark dimensionality, else (baselines on the largest graphs) that of the papers
            samples = [os.path.join(results, name, configuration, f"dim{dim}", "seed0", "sample.npz")
                       for dim in (args.dim, display.PAPER_DIM) for configuration in row["configurations"]]
            sample = next((path for path in samples if os.path.exists(path)), None)
            if sample is None:
                continue
            with np.load(sample) as archive:
                rows = archive["rows"][:MAX_POINTS]
            panels.append((row["label"], coordinates(sample), rows))
        if len(panels) < 2:
            continue
        graph = datasets.load(name, os.path.join(args.data, "datasets"))
        path = os.path.join(args.out, f"umap_{name}.png")
        draw_dataset(name, panels, graph, path)
        print(f"{name}: {len(panels)} panels -> {path}", flush=True)


if __name__ == "__main__":
    main()
