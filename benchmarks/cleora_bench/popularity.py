"""How the exponent of the popularity prior was chosen: what is measured, and its table.

Concern of this module: the record of the one design decision behind
``pycleora.DEFAULT_POPULARITY_EXPONENT``. ``benchmarks/popularity_exponent.py`` measures
the candidate exponents on *validation* edges and writes one file per graph; this module
names the candidates and the files and turns the files into a table.
"""

import glob
import json
import os

import numpy as np

EXPONENTS = (0.0, 0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5)
DIRECTORY = "popularity"
METRICS = ("mrr", "recall@10")  # against the sampled candidates / among all nodes


def path(data, dataset):
    return os.path.join(data, DIRECTORY, f"{dataset}.json")


def markdown(directory):
    """One row per exponent: mean validation scores over the measured graphs - of the
    prior applied everywhere, and of the benchmark's rule, which keeps the unit vectors
    (the cosine) on a graph where they validate better by MRR."""
    graphs = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as handle:
            graphs.append(json.load(handle))
    if not graphs:
        return ""
    lines = ["| Exponent | Graphs | MRR | recall@10 | MRR, prior or cosine | recall@10, prior or cosine | Graphs that take the prior |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for exponent in EXPONENTS:
        everywhere = [graph[str(exponent)] for graph in graphs]
        chosen = [graph[str(exponent)] if graph[str(exponent)]["mrr"] > graph["0.0"]["mrr"] else graph["0.0"] for graph in graphs]
        taken = sum(graph[str(exponent)]["mrr"] > graph["0.0"]["mrr"] for graph in graphs)
        rows.append([float(np.mean([scores[metric] for scores in table])) for table in (everywhere, chosen) for metric in METRICS] + [taken])
    best = [max(row[column] for row in rows) for column in range(4)]
    for exponent, row in zip(EXPONENTS, rows):
        cells = [f"**{value:.4f}**" if value == top else f"{value:.4f}" for value, top in zip(row[:4], best)]
        lines.append(f"| {exponent:g} | {len(graphs)} | " + " | ".join(cells) + f" | {row[4]} |")
    return "\n".join(lines)
