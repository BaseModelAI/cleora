#!/usr/bin/env python
"""The picture at the top of the README: three facts of the benchmark as bar charts.

    python benchmarks/make_summary_figure.py --data ~/cleora-bench --out benchmarks/figures

* on how many graphs each method is the best one in similarity search (the row
  ``recall@10`` of the table ``ranks_retrieval``);
* the time every method needs to embed one mid-sized graph, and
* the memory it needs on a larger one (tables ``time`` and ``memory``).

Every bar is read from the result files through the code that writes the tables. Two
files are written, ``summary_light.svg`` and ``summary_dark.svg``: the README shows the
one that fits the reader's colour scheme.
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_report  # noqa: E402
from cleora_bench import display  # noqa: E402

TIME_GRAPH, MEMORY_GRAPH = display.SUMMARY_TIME_GRAPH, display.SUMMARY_MEMORY_GRAPH
# ink, secondary ink, the method the page is about, every other method
THEMES = {
    "light": {"ink": "#0b0b0b", "secondary": "#52514e", "accent": "#2a78d6", "other": "#b9b8b1"},
    "dark": {"ink": "#f0efec", "secondary": "#c3c2b7", "accent": "#3987e5", "other": "#5f5e5a"},
}


def wins(results):
    """``[(label, graphs won, is Cleora)]`` in similarity search, and the number of graphs."""
    columns = ["cleora"] + list(display.BASELINES)
    summary = make_report.competition_summary(results.scenario_rows("retrieval", columns), columns)
    rows = [(display.METHODS[m]["label"], summary[m]["wins"], m == "cleora") for m in columns]
    return rows, summary["cleora"]["rows"]


def measured(results, dataset, quantity):
    """``[(label, median of the timing runs, is Cleora)]`` for the methods that ran on a graph."""
    return [(label, value, label == "Cleora") for label, value in make_report.timed(results, dataset, quantity).items()]


def panel(axis, rows, title, subtitle, text, theme, slots, more_is_better=False):
    """Horizontal bars, best first, every bar with its name and its value written out.
    ``slots``: bars the panel has room for, so that all panels draw them equally thick."""
    rows = sorted(rows, key=lambda row: (not row[2], -row[1] if more_is_better else row[1]))
    largest = max(value for _, value, _ in rows) or 1.0
    for position, (label, value, ours) in enumerate(rows):
        colour = theme["accent"] if ours else theme["other"]
        axis.barh(position, value, height=0.62, color=colour, linewidth=0)
        axis.text(-0.02 * largest, position, label, ha="right", va="center", fontsize=10.5,
                  color=theme["ink"] if ours else theme["secondary"], fontweight="bold" if ours else "normal")
        axis.text(value + 0.02 * largest, position, text(value), ha="left", va="center", fontsize=10.5,
                  color=theme["ink"] if ours else theme["secondary"], fontweight="bold" if ours else "normal")
    axis.set_ylim(slots - 0.4, -0.6)
    axis.set_xlim(0, largest * 1.22)
    axis.axis("off")
    axis.text(-0.3, 1.14, title, transform=axis.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold", color=theme["ink"])
    axis.text(-0.3, 1.05, subtitle, transform=axis.transAxes, ha="left", va="bottom", fontsize=10, color=theme["secondary"])


def draw(results, theme, path):
    import logging

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # the first family that exists is used
    plt.rcParams["font.family"] = ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "path"  # letters as outlines: the same picture on every machine
    plt.rcParams["svg.hashsalt"] = "cleora"  # and the same file for the same numbers
    won, graphs = wins(results)
    seconds, memory = measured(results, TIME_GRAPH, "seconds"), measured(results, MEMORY_GRAPH, "peak_rss_mb")
    size = lambda name: f"{results.meta[name]['num_nodes']:,} nodes, {results.meta[name]['num_edges'] / 1e6:.1f}M edges"
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.62})
    slots = max(len(won), len(seconds), len(memory))
    panel(axes[0], won, "Best in similarity search", f"number of graphs, out of {graphs}", str, theme, slots, more_is_better=True)
    panel(axes[1], seconds, "Time to embed", f"{display.DATASET_LABELS[TIME_GRAPH]}: {size(TIME_GRAPH)}",
          make_report.duration, theme, slots)
    panel(axes[2], memory, "Peak memory", f"{display.DATASET_LABELS[MEMORY_GRAPH]}: {size(MEMORY_GRAPH)}",
          lambda megabytes: f"{megabytes / 1024:.0f} GB", theme, slots)
    figure.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.02)
    figure.savefig(path, transparent=True, metadata={"Date": None} if path.endswith(".svg") else None)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="directory with datasets/, results/ and timing/")
    parser.add_argument("--out", required=True, help="directory for the two figures")
    args = parser.parse_args()
    results = make_report.Results(os.path.abspath(os.path.expanduser(args.data)))
    os.makedirs(args.out, exist_ok=True)
    for name, theme in THEMES.items():
        path = os.path.join(args.out, f"summary_{name}.svg")
        draw(results, theme, path)
        print(path)


if __name__ == "__main__":
    main()
