"""How the exponents of the popularity prior were chosen: what is measured, and its tables.

Concern of this module: the record of one design decision - which exponent
``pycleora.scale_by_popularity`` gets when the vectors are used to find links and when
they are features of a classifier. ``benchmarks/popularity_exponent.py`` measures the
candidate exponents on *validation* data and writes one file per task and graph; this
module names the candidates and the files and turns the files into tables.
"""

import glob
import json
import os

import numpy as np

EXPONENTS = (0.0, 0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25, 0.375, 0.5)
DIRECTORY = "popularity"
# what a file holds for every exponent, by task
METRICS = {"lp": ("mrr", "recall@10"),  # against the sampled candidates / among all nodes
           "nc": ("score",)}            # accuracy, or micro-F1 on multi-label graphs (logistic regression)


def path(data, task, dataset):
    return os.path.join(data, DIRECTORY, task, f"{dataset}.json")


def _graphs(data, task):
    graphs = []
    for name in sorted(glob.glob(path(data, task, "*"))):
        with open(name) as handle:
            graphs.append(json.load(handle))
    return graphs


def _bold_best(rows, columns):
    """The rows as strings, the best value of each of the first ``columns`` columns in bold."""
    best = [max(row[column] for row in rows) for column in range(columns)]
    return [[f"**{value:.4f}**" if value == top else f"{value:.4f}" for value, top in zip(row, best)] + [str(v) for v in row[columns:]]
            for row in rows]


def link_table(data):
    """One row per exponent: mean validation scores over the measured graphs - of the
    prior applied everywhere, and of the benchmark's rule, which keeps the unit vectors
    (the cosine) on a graph where they validate better by MRR."""
    graphs = _graphs(data, "lp")
    if not graphs:
        return ""
    rows = []
    for exponent in EXPONENTS:
        everywhere = [graph[str(exponent)] for graph in graphs]
        chosen = [graph[str(exponent)] if graph[str(exponent)]["mrr"] > graph["0.0"]["mrr"] else graph["0.0"] for graph in graphs]
        taken = sum(graph[str(exponent)]["mrr"] > graph["0.0"]["mrr"] for graph in graphs)
        rows.append([float(np.mean([scores[metric] for scores in table])) for table in (everywhere, chosen) for metric in METRICS["lp"]] + [taken])
    lines = ["| Exponent | Graphs | MRR | recall@10 | MRR, prior or cosine | recall@10, prior or cosine | Graphs that take the prior |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {exponent:g} | {len(graphs)} | " + " | ".join(cells) + " |" for exponent, cells in zip(EXPONENTS, _bold_best(rows, 4))]
    return "\n".join(lines)


def classification_table(data):
    """One row per exponent: the mean validation score of logistic regression over the
    measured graphs, and on how many of them it is above / below that of the unit vectors."""
    graphs = _graphs(data, "nc")
    if not graphs:
        return ""
    rows = []
    for exponent in EXPONENTS:
        scores = np.array([graph[str(exponent)]["score"] for graph in graphs])
        plain = np.array([graph["0.0"]["score"] for graph in graphs])
        rows.append([float(scores.mean()), int((scores > plain).sum()), int((scores < plain).sum())])
    lines = ["| Exponent | Graphs | Validation score | Graphs above the unit vectors | Graphs below |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {exponent:g} | {len(graphs)} | " + " | ".join(cells) + " |" for exponent, cells in zip(EXPONENTS, _bold_best(rows, 1))]
    return "\n".join(lines)
