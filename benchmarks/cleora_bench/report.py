"""Turns the job files of :mod:`runner` into tables.

Concern of this module: aggregation and presentation. No number is computed here that
was not measured by a job: the module averages over seeds and splits, chooses between
variants of a method on *validation* scores, ranks, and formats.

Variants. A row of a table is a *method*; a method may have several variants - a
baseline at its authors' dimensionality and at the benchmark's, Cleora at different
iteration counts. For every (dataset, task, probe) the variant with the best mean
validation score is reported, and the tables say which one it was.
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

# The headline metric of a probe, by kind of task.
HEADLINE = {"nc": "accuracy", "nc_multilabel": "micro_f1", "lp": "mrr"}


def load_jobs(results):
    """All job records under a results directory."""
    jobs = []
    for path in glob.glob(os.path.join(results, "*", "*", "dim*", "seed*", "*.json")):
        with open(path) as handle:
            jobs.append(json.load(handle))
    return jobs


def _probe_records(job):
    """Yields ``(probe, validation score, metrics dict)`` for every evaluation in a job:
    one per split for node classification, one for link prediction."""
    for probe, record in job.get("probes", {}).items():
        if job["task"] == "nc":
            for metrics, chosen in zip(record["metrics"], record["chosen"]):
                yield probe, chosen["validation_score"], metrics
        else:
            yield probe, record["chosen"]["validation_mrr"], record["metrics"]


def aggregate(jobs):
    """``{(dataset, task, method id, dim): entry}`` where ``entry`` holds, per probe, the
    list of validation scores and of metric dictionaries over seeds and splits, plus the
    embedding time / memory, the iteration counts Cleora reported, the mean degree of the
    graph that was embedded, and failures."""
    table = {}
    for job in jobs:
        key = (job["dataset"], job["task"], job["method"], job["dim"])
        entry = table.setdefault(key, {"probes": defaultdict(lambda: {"validation": [], "metrics": []}),
                                       "seconds": [], "peak_rss_mb": [], "iterations": [], "degree": None,
                                       "failures": []})
        embedding = job["embedding"]
        if embedding["status"] != "ok":
            entry["failures"].append(f"{embedding['status']}: {embedding['reason']}")
            continue
        entry["seconds"].append(embedding["seconds"])
        entry["peak_rss_mb"].append(embedding["peak_rss_mb"])
        entry["degree"] = 2.0 * embedding["edges"] / embedding["nodes"]  # of the graph that was embedded
        if "iterations" in embedding:
            entry["iterations"].append(embedding["iterations"])
        for probe, validation, metrics in _probe_records(job):
            entry["probes"][probe]["validation"].append(validation)
            entry["probes"][probe]["metrics"].append(metrics)
    return table


def select_variant(table, dataset, task, probe, variants):
    """Among ``variants`` (``(method id, dim)`` pairs) of one method: the one with the
    highest mean validation score for this probe. Returns ``(variant, entry)`` or
    ``(None, reasons)`` when none of them produced a result."""
    best, reasons = None, []
    for variant in variants:
        entry = table.get((dataset, task) + variant)
        if entry is None:
            continue
        runs = entry["probes"].get(probe)
        if not runs or not runs["validation"]:
            reasons.extend(entry["failures"])
            continue
        score = float(np.mean(runs["validation"]))
        if best is None or score > best[0]:
            best = (score, variant, entry)
    if best is None:
        return None, sorted(set(reasons))
    return best[1], best[2]


def cell(table, dataset, task, probe, metric, variants):
    """Mean and standard deviation of ``metric`` for the selected variant of a method.
    Returns a dict with ``mean``, ``std``, ``runs``, ``variant``; or ``failed`` / ``missing``."""
    variant, entry = select_variant(table, dataset, task, probe, variants)
    if variant is None:
        return {"failed": entry} if entry else {"missing": True}
    values = [m[metric] for m in entry["probes"][probe]["metrics"] if metric in m and m[metric] == m[metric]]
    if not values:
        return {"missing": True}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "runs": len(values),
            "variant": list(variant), "seconds": float(np.mean(entry["seconds"])),
            "peak_rss_mb": float(np.mean(entry["peak_rss_mb"]))}


def best_training_free(table, dataset, variants, metric="mrr"):
    """Dot product or cosine, whichever validates better (per method): the
    ``similarity`` pseudo-probe of the link tables."""
    candidates = []
    for probe in ("dot", "cosine"):
        variant, entry = select_variant(table, dataset, "lp", probe, variants)
        if variant is not None:
            candidates.append((float(np.mean(entry["probes"][probe]["validation"])), probe))
    if not candidates:
        return cell(table, dataset, "lp", "cosine", metric, variants)
    probe = max(candidates)[1]
    return dict(cell(table, dataset, "lp", probe, metric, variants), probe=probe)


def ranks(cells, higher_is_better=True):
    """Rank of every method within one row of a table (1 = best; ties share the mean
    rank). Methods without a result get the rank after the last one that has a result."""
    names = list(cells)
    values = np.array([cells[n].get("mean", np.nan) for n in names], dtype=np.float64)
    present = ~np.isnan(values)
    order = -values if higher_is_better else values
    result = {}
    for i, name in enumerate(names):
        if present[i]:
            result[name] = 1 + float((order[present] < order[i]).sum()) + 0.5 * float((order[present] == order[i]).sum() - 1)
        else:
            result[name] = float(present.sum() + 1)
    return result


def format_cell(value, best, second, digits=3, show_std=True):
    if "mean" not in value:
        if "failed" in value:
            reason = value["failed"][0] if value["failed"] else "failed"
            if "timeout" in reason:
                return "time limit"
            if "memory" in reason:
                return "out of memory"
            if "skipped" in reason:
                return "n/a (dense)"
            return "failed"
        return "–"
    text = f"{value['mean']:.{digits}f}"
    if show_std and value.get("runs", 1) > 1:
        text += f" ± {value['std']:.{digits}f}"
    if best is not None and abs(value["mean"] - best) < 0.5 * 10 ** -digits:
        return f"**{text}**"
    if second is not None and abs(value["mean"] - second) < 0.5 * 10 ** -digits:
        return f"<u>{text}</u>"
    return text


def markdown_table(rows, methods, labels, row_label="Dataset", digits=3, higher_is_better=True, show_std=True):
    """``rows``: ``{row name: {method: cell}}``. The best value of a row is bold, the
    second best underlined."""
    lines = ["| " + row_label + " | " + " | ".join(labels[m] for m in methods) + " |",
             "|---|" + "---:|" * len(methods)]
    for name, cells in rows.items():
        means = sorted({round(c["mean"], digits) for c in cells.values() if "mean" in c}, reverse=higher_is_better)
        best = means[0] if means else None
        second = means[1] if len(means) > 1 else None
        lines.append("| " + name + " | " + " | ".join(
            format_cell(cells.get(m, {"missing": True}), best, second, digits, show_std) for m in methods) + " |")
    return "\n".join(lines)


def summary_rows(rows, methods, higher_is_better=True):
    """Mean rank, number of wins (ties count) and number of rows with a result, per method."""
    rank_sum, wins, counted = defaultdict(float), defaultdict(int), defaultdict(int)
    used = 0
    for cells in rows.values():
        complete = {m: cells.get(m, {"missing": True}) for m in methods}
        if not any("mean" in c for c in complete.values()):
            continue
        used += 1
        row_ranks = ranks(complete, higher_is_better)
        top = min(row_ranks.values())
        for m in methods:
            rank_sum[m] += row_ranks[m]
            wins[m] += int(row_ranks[m] == top and "mean" in complete[m])
            counted[m] += int("mean" in complete[m])
    return {m: {"mean_rank": rank_sum[m] / max(used, 1), "wins": wins[m], "results": counted[m], "rows": used}
            for m in methods}
