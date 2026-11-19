#!/usr/bin/env python
"""Writes the result tables of the benchmark from its job files.

    python benchmarks/make_report.py --data ~/cleora-bench --out benchmarks/RESULTS.md --readme README.md

``--out`` receives every table. ``--readme`` names a file in which the text between
``<!-- benchmark:NAME -->`` and ``<!-- /benchmark:NAME -->`` is replaced by the table
of that name, so the README never holds a hand-typed number.

Every number comes from ``<data>/results`` (quality) or ``<data>/timing`` (time and
memory, measured separately on an idle machine). How a method's variant is chosen,
what the columns mean and how ranks are computed is documented in
``cleora_bench/report.py`` and ``cleora_bench/display.py``.
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleora_bench import configurations, datasets, display, iteration_fit, report, selection  # noqa: E402


class Results:
    """The aggregated job files plus what is needed to address them by table position."""

    def __init__(self, data):
        self.data = data
        self.table = report.aggregate(report.load_jobs(os.path.join(data, "results")))
        self.timing = report.aggregate(report.load_jobs(os.path.join(data, "timing")))
        self.selected = selection.load(data)
        present = {key[0] for key in self.table}
        self.datasets = [name for name in display.DATASET_ORDER if name in present]
        self.meta = {name: datasets.meta(name, os.path.join(data, "datasets")) for name in self.datasets}

    def variants(self, method, dataset, task):
        """The configurations a column stands for: exactly one for every method of the main
        tables; for "cleora_tuned" the four families with the suggested and with the
        validated iteration counts (the report then picks on validation)."""
        if method in display.SAME_METHOD:
            return [(display.cleora_configuration(display.METHODS[method]["setup"], task), display.MAIN_DIM)]
        if method == "cleora_tuned":
            suggested = configurations.automatic(display.TASK_PURPOSE[task])
            chosen = self.selected.get(dataset, {}).get(task, {}).values()
            return [(name, display.MAIN_DIM) for name in suggested + sorted(chosen)]
        return [(display.METHODS[method]["configuration"], display.MAIN_DIM)]

    def labelled(self, multilabel):
        return [name for name in self.datasets
                if self.meta[name]["num_classes"] > 0 and bool(self.meta[name].get("multilabel")) == multilabel]

    def with_task(self, task):
        return [name for name in self.datasets if any(key[0] == name and key[1] == task for key in self.table)]

    def rows(self, task, probe, metric, names, methods):
        """``{dataset label: {method: cell}}`` for one table."""
        result = {}
        for name in names:
            cells = {}
            for method in methods:
                variants = self.variants(method, name, task)
                if probe == "similarity":
                    cells[method] = report.best_training_free(self.table, name, variants, metric)
                else:
                    cells[method] = report.cell(self.table, name, task, probe, metric, variants)
            result[display.DATASET_LABELS[name]] = cells
        return result


def competition_summary(rows, methods, higher_is_better=True):
    """Mean rank and wins of every method. The Cleora columns are the same method fed in two
    ways and do not compete with each other: every baseline is ranked against the first of
    them, and each Cleora column against the baselines only."""
    same = [m for m in methods if m in display.SAME_METHOD]
    if len(same) < 2:
        return report.summary_rows(rows, methods, higher_is_better)
    others = [m for m in methods if m not in same]
    summary = report.summary_rows(rows, [same[0]] + others, higher_is_better)
    for method in same[1:]:
        summary[method] = report.summary_rows(rows, [method] + others, higher_is_better)[method]
    return summary


def table_with_summary(rows, methods, digits=3, higher_is_better=True, show_std=True):
    labels = {method: display.METHODS[method]["label"] for method in methods}
    text = report.markdown_table(rows, methods, labels, digits=digits, higher_is_better=higher_is_better,
                                 show_std=show_std)
    summary = competition_summary(rows, methods, higher_is_better)
    best_rank = min(s["mean_rank"] for s in summary.values())
    rank_cells = [f"**{summary[m]['mean_rank']:.2f}**" if summary[m]["mean_rank"] == best_rank else f"{summary[m]['mean_rank']:.2f}"
                  for m in methods]
    text += "\n| *mean rank* | " + " | ".join(rank_cells) + " |"
    text += "\n| *best on* | " + " | ".join(f"{summary[m]['wins']} / {summary[m]['results']}" for m in methods) + " |"
    return text


def rank_table(results, task, probes, names_by_metric, columns):
    """One row per probe: the mean rank of every method over the graphs of the task, and
    the number of graphs it is best on. ``names_by_metric``: ``{metric: datasets}``."""
    labels = [display.METHODS[method]["label"] for method in columns]
    lines = ["| Probe | Graphs | " + " | ".join(labels) + " |", "|---|---:|" + "---:|" * len(columns)]
    for probe, probe_label in probes.items():
        rows = {}
        for metric, names in names_by_metric.items():
            rows.update(results.rows(task, probe, metric, names, columns))
        summary = competition_summary(rows, columns)
        graphs = max(s["rows"] for s in summary.values())
        best = min(s["mean_rank"] for s in summary.values())
        cells = [f"{'**' if summary[m]['mean_rank'] == best else ''}{summary[m]['mean_rank']:.2f}"
                 f"{'**' if summary[m]['mean_rank'] == best else ''} ({summary[m]['wins']})" for m in columns]
        lines.append(f"| {probe_label} | {graphs} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dataset_table(results):
    lines = ["| Dataset | Nodes | Edges | Mean degree | Node labels | Seeds | Source |", "|---|---:|---:|---:|---|---:|---|"]
    for name in results.datasets:
        meta = results.meta[name]
        if meta["num_classes"] == 0:
            labels = "–"
        else:
            labels = f"{meta['num_classes']} {'labels (multi-label)' if meta.get('multilabel') else 'classes'}"
        runs = max((len(entry["seconds"]) for key, entry in results.table.items() if key[0] == name and key[1] == "lp"), default=0)
        notes = [note for note, flag in (("weighted", meta["weighted"]), ("bipartite", meta["bipartite"]),
                                         ("official node split", meta["official_node_split"]),
                                         ("official link split", meta["official_link_split"])) if flag]
        description = meta.get("citation", "") + ("; " + ", ".join(notes) if notes else "")
        lines.append(f"| {display.DATASET_LABELS[name]} | {meta['num_nodes']:,} | {meta['num_edges']:,} | "
                     f"{2 * meta['num_edges'] / meta['num_nodes']:.1f} | {labels} | {runs or '–'} | {description} |")
    return "\n".join(lines)


def families_table(results, task, probe, metric, names):
    """The four library-default runs, each on its own, next to the 3.2 default and the tuned row."""
    columns = list(display.FAMILIES) + ["tuned", "pycleora 3.2 default"]
    lines = ["| Dataset | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    purpose = display.TASK_PURPOSE[task]
    for name in names:
        cells = [report.cell(results.table, name, task, probe, metric,
                             [(configurations.AUTO[family + (purpose,)], display.MAIN_DIM)])
                 for family in display.FAMILIES.values()]
        cells.append(report.cell(results.table, name, task, probe, metric, results.variants("cleora_tuned", name, task)))
        cells.append(report.cell(results.table, name, task, probe, metric, results.variants("cleora_3_2", name, task)))
        means = sorted({round(c["mean"], 3) for c in cells if "mean" in c}, reverse=True)
        lines.append(f"| {display.DATASET_LABELS[name]} | " + " | ".join(
            report.format_cell(c, means[0] if means else None, means[1] if len(means) > 1 else None, 3, False)
            for c in cells) + " |")
    return "\n".join(lines)


def _iterations(entry):
    values = {record for record in entry.get("iterations", [])}
    return "/".join(str(v) for v in sorted(values)) if values else "–"


def iterations_table(results, task):
    """Iterations of the automatic rule next to those the sweep chose on validation, and
    what the difference costs (headline metric of the task, sweep probe)."""
    probe = selection.SWEEP_PROBE[task]
    purpose = display.TASK_PURPOSE[task]
    families = {f"{setup}/{'sym' if propagation == 'symmetric' else propagation}": configurations.AUTO[(setup, propagation, purpose)]
                for setup, propagation in display.FAMILIES.values()}
    header = [f"{label} ({kind})" for label in display.FAMILIES for kind in ("auto", "tuned", "loss")]
    lines = ["| Dataset | " + " | ".join(header) + " |", "|---|" + "---:|" * len(header)]
    losses = {family: [] for family in families}
    for name in results.with_task(task):
        metric = "mrr" if task == "lp" else ("micro_f1" if results.meta[name].get("multilabel") else "accuracy")
        cells = []
        for family, auto_name in families.items():
            auto = results.table.get((name, task, auto_name, display.MAIN_DIM))
            tuned_name = results.selected.get(name, {}).get(task, {}).get(family)
            tuned = results.table.get((name, task, tuned_name, display.MAIN_DIM)) if tuned_name else None
            auto_cell = report.cell(results.table, name, task, probe, metric, [(auto_name, display.MAIN_DIM)])
            tuned_cell = report.cell(results.table, name, task, probe, metric, [(tuned_name, display.MAIN_DIM)]) if tuned_name else {}
            cells.append(_iterations(auto) if auto else "–")
            cells.append(tuned_name.rsplit("_k", 1)[1] if tuned_name else "–")
            if "mean" in auto_cell and "mean" in tuned_cell:
                loss = 100 * (tuned_cell["mean"] - auto_cell["mean"])
                losses[family].append(loss)
                cells.append(f"{loss:+.1f}")
            else:
                cells.append("–")
        lines.append(f"| {display.DATASET_LABELS[name]} | " + " | ".join(cells) + " |")
    means = []
    for family in families:
        means += ["", "", f"{np.mean(losses[family]):+.1f}" if losses[family] else "–"]
    lines.append("| *mean loss* | " + " | ".join(means) + " |")
    return "\n".join(lines)


def _duration(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 100:
        return f"{seconds:.1f} s"
    if seconds < 6000:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def timing_table(results, quantity):
    """Median over the timing runs of every method (benchmark dimensionality)."""
    default = configurations.AUTO[("hyperedge", "left", "general")]  # what embed(graph) runs
    methods = [default, "cleora_whiten40"] + list(display.BASELINES)
    labels = {default: "Cleora", "cleora_whiten40": "pycleora 3.2 default",
              **{name: display.METHODS[name]["label"] for name in display.BASELINES}}
    names = [name for name in display.DATASET_ORDER if any(key[0] == name for key in results.timing)]
    lines = ["| Dataset | " + " | ".join(labels[m] for m in methods) + " |", "|---|" + "---:|" * len(methods)]
    for name in names:
        values, cells = {}, {}
        for method in methods:
            entry = results.timing.get((name, "time", method, display.MAIN_DIM))
            if entry is None:
                cells[method] = "–"
            elif not entry[quantity]:
                cells[method] = report.format_cell({"failed": entry["failures"]}, None, None)
            else:
                values[method] = float(np.median(entry[quantity]))
                cells[method] = _duration(values[method]) if quantity == "seconds" else f"{values[method] / 1024:.2f} GB"
        best = min(values.values()) if values else None
        lines.append(f"| {display.DATASET_LABELS[name]} | " + " | ".join(
            f"**{cells[m]}**" if m in values and values[m] == best else cells[m] for m in methods) + " |")
    return "\n".join(lines)


def families_validation_table(results):
    """What fixed the normalisation of each task: mean validation score of every family
    over the graphs on which all four ran (headline probe), and how often it validated best."""
    lines = ["| Task | Graphs | " + " | ".join(display.FAMILIES) + " |", "|---|---:|" + "---:|" * len(display.FAMILIES)]
    for task, probe, label in (("nc", "logistic", "node classification (logistic regression)"),
                               ("lp", "cosine", "link ranking (cosine)")):
        scores = {family: [] for family in display.FAMILIES}
        for name in results.with_task(task):
            row = {}
            for family, key in display.FAMILIES.items():
                entry = results.table.get((name, task, configurations.AUTO[key + (display.TASK_PURPOSE[task],)], display.MAIN_DIM))
                runs = entry["probes"].get(probe) if entry else None
                if runs and runs["validation"]:
                    row[family] = float(np.mean(runs["validation"]))
            if len(row) == len(display.FAMILIES):
                for family, value in row.items():
                    scores[family].append(value)
        graphs = len(next(iter(scores.values())))
        if graphs == 0:
            continue
        best = np.argmax(np.array([scores[family] for family in display.FAMILIES]), axis=0)
        cells = [f"{np.mean(scores[family]):.4f} ({int((best == index).sum())})" for index, family in enumerate(display.FAMILIES)]
        lines.append(f"| {label} | {graphs} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tuned_table(results, task, probe, names_by_metric):
    """What a validation set adds: Cleora as in the main tables, Cleora with the iteration
    count and the family chosen on validation, and the best baseline of the row."""
    lines = ["| Dataset | Cleora | Cleora, tuned | gain | Best baseline |", "|---|---:|---:|---:|---|"]
    gains = []
    for metric, names in names_by_metric.items():
        for name in names:
            cell = lambda method: (report.best_training_free(results.table, name, results.variants(method, name, task), metric)
                                   if probe == "similarity" else
                                   report.cell(results.table, name, task, probe, metric, results.variants(method, name, task)))
            plain, tuned = cell("cleora"), cell("cleora_tuned")
            rivals = {method: cell(method) for method in display.BASELINES}
            rivals = {method: value["mean"] for method, value in rivals.items() if "mean" in value}
            if "mean" not in plain or "mean" not in tuned:
                continue
            gains.append(100 * (tuned["mean"] - plain["mean"]))
            best = max(rivals, key=rivals.get) if rivals else None
            rival = f"{display.METHODS[best]['label']} {rivals[best]:.3f}" if best else "–"
            top = max([plain["mean"], tuned["mean"]] + list(rivals.values()))
            bold = lambda value: f"**{value:.3f}**" if value == top else f"{value:.3f}"
            lines.append(f"| {display.DATASET_LABELS[name]} | {bold(plain['mean'])} | {bold(tuned['mean'])} | "
                         f"{gains[-1]:+.1f} | {rival} |")
    if gains:
        lines.append(f"| *mean gain (points)* | | | {np.mean(gains):+.1f} | |")
    return "\n".join(lines)


def _best_baseline(summary):
    name = min(display.BASELINES, key=lambda method: summary[method]["mean_rank"])
    return f"{display.METHODS[name]['label']} ({summary[name]['mean_rank']:.2f})"


def _task_rows(results, task):
    """The rows of the headline probe of a task: logistic regression / dot product or cosine."""
    columns = list(display.MAIN_COLUMNS)
    if task == "lp":
        return results.rows("lp", "similarity", "mrr", results.with_task("lp"), columns)
    rows = results.rows("nc", "logistic", "accuracy", results.labelled(False), columns)
    rows.update(results.rows("nc", "logistic", "micro_f1", results.labelled(True), columns))
    return rows


def headline(results):
    """The summary at the top of the README: wins and mean ranks on the two headline probes."""
    lines = ["| | Graphs | Cleora: best on | mean rank | Cleora, two columns: best on | mean rank | Best baseline (mean rank) |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for task, label in (("lp", "Similarity search and link ranking (dot product / cosine, MRR)"),
                        ("nc", "Node classification (logistic regression, accuracy / micro-F1)")):
        summary = competition_summary(_task_rows(results, task), list(display.MAIN_COLUMNS))
        one, two = summary["cleora"], summary["cleora_pair"]
        lines.append(f"| {label} | {one['rows']} | **{one['wins']}** | {one['mean_rank']:.2f} | "
                     f"{two['wins']} | {two['mean_rank']:.2f} | {_best_baseline(summary)} |")
    return "\n".join(lines)


def caveats(results):
    """Where Cleora is not the best method: every graph on which a baseline, or the
    pycleora 3.2 default, is ahead of the Cleora column on a headline probe."""
    lines = []
    dataset_of = {display.DATASET_LABELS[d]: d for d in results.datasets}
    for task, label in (("lp", "Link ranking"), ("nc", "Node classification")):
        behind, previous = [], []
        for name, cells in _task_rows(results, task).items():
            if "mean" not in cells["cleora"]:
                continue
            ours = cells["cleora"]["mean"]
            rivals = {m: cells[m]["mean"] for m in display.BASELINES if "mean" in cells[m]}
            if rivals and max(rivals.values()) > ours:
                best = max(rivals, key=rivals.get)
                behind.append(f"{name} ({display.METHODS[best]['label']} {rivals[best]:.3f}, Cleora {ours:.3f})")
            dataset = dataset_of[name]
            probe = "cosine" if task == "lp" else "logistic"
            metric = "mrr" if task == "lp" else ("micro_f1" if results.meta[dataset].get("multilabel") else "accuracy")
            old = report.cell(results.table, dataset, task, probe, metric, results.variants("cleora_3_2", dataset, task))
            new = report.cell(results.table, dataset, task, probe, metric, results.variants("cleora", dataset, task))
            if "mean" in old and "mean" in new and old["mean"] > new["mean"]:
                previous.append(f"{name} ({old['mean']:.3f} against {new['mean']:.3f})")
        lines.append(f"- **{label}: a baseline is ahead of Cleora on** " + ("; ".join(behind) if behind else "no graph") + ".")
        lines.append(f"- **{label}: the pycleora 3.2 default (whitening, 40 iterations) is ahead of Cleora on** "
                     + ("; ".join(previous) if previous else "no graph") + ".")
    return "\n".join(lines)


def build(results):
    """``{name: markdown}`` of every table, in the order of the report."""
    columns = list(display.MAIN_COLUMNS)
    single, multi, link = results.labelled(False), results.labelled(True), results.with_task("lp")
    tables = {"datasets": dataset_table(results), "headline": headline(results), "caveats": caveats(results)}
    tables["ranks_nc"] = rank_table(results, "nc", display.NODE_PROBES, {"accuracy": single, "micro_f1": multi}, columns)
    tables["ranks_lp"] = rank_table(results, "lp", {"similarity": "dot product or cosine", **display.PAIR_PROBES},
                                    {"mrr": link}, columns)
    for probe in display.NODE_PROBES:
        tables[f"nc_accuracy_{probe}"] = table_with_summary(results.rows("nc", probe, "accuracy", single, columns), columns)
        tables[f"nc_micro_f1_{probe}"] = table_with_summary(results.rows("nc", probe, "micro_f1", multi, columns), columns)
    for metric in ("macro_f1", "balanced_accuracy", "mcc", "macro_roc_auc", "macro_average_precision", "top3_accuracy"):
        tables[f"nc_{metric}_logistic"] = table_with_summary(results.rows("nc", "logistic", metric, single, columns), columns, show_std=False)
    for metric in ("macro_f1", "micro_f1_known_count", "micro_average_precision", "label_ranking_average_precision",
                   "precision_at_1", "hamming_loss"):
        tables[f"nc_multilabel_{metric}_logistic"] = table_with_summary(results.rows("nc", "logistic", metric, multi, columns), columns, show_std=False, higher_is_better=metric != "hamming_loss")
    tables["lp_mrr_similarity"] = table_with_summary(results.rows("lp", "similarity", "mrr", link, columns), columns)
    for probe in display.PAIR_PROBES:
        tables[f"lp_mrr_{probe}"] = table_with_summary(results.rows("lp", probe, "mrr", link, columns), columns)
    for metric in ("hits@1", "hits@10", "hits@100", "roc_auc", "average_precision", "all_nodes_mrr", "recall@10",
                   "recall@50", "ndcg@10", "map@10"):
        tables[f"lp_{metric}_cosine"] = table_with_summary(results.rows("lp", "cosine", metric, link, columns), columns, show_std=False)
    tables["families_nc_accuracy"] = families_table(results, "nc", "logistic", "accuracy", single)
    tables["families_nc_micro_f1"] = families_table(results, "nc", "logistic", "micro_f1", multi)
    tables["families_lp_mrr"] = families_table(results, "lp", "cosine", "mrr", link)
    tables["iterations_nc"] = iterations_table(results, "nc")
    tables["iterations_lp"] = iterations_table(results, "lp")
    tables["families_validation"] = families_validation_table(results)
    tables["tuned_nc"] = tuned_table(results, "nc", "logistic", {"accuracy": single, "micro_f1": multi})
    tables["tuned_lp"] = tuned_table(results, "lp", "similarity", {"mrr": link})
    sweep = os.path.join(results.data, selection.SWEEP_DIRECTORY)
    if os.path.isdir(sweep):
        tables["iteration_rule_fit"] = iteration_fit.markdown(sweep)
        tables["iteration_rule_constants"] = iteration_fit.sensitivity_markdown(sweep)
    if results.timing:
        tables["time"] = timing_table(results, "seconds")
        tables["memory"] = timing_table(results, "peak_rss_mb")
    return tables


SECTIONS = (
    ("In short", "The two headline probes; the full tables follow.", ["headline"]),
    ("Where Cleora is not the best method", "", ["caveats"]),
    ("Datasets", "", ["datasets"]),
    ("Summary: mean rank of every method, by probe",
     "Mean rank over the graphs of the task (1 = best; a method without a result on a graph ranks after all that "
     "have one), and in parentheses the number of graphs the method is best on. The two Cleora columns are the "
     "same method fed in two ways and do not compete with each other: every baseline is ranked against Cleora "
     "with one column, and each Cleora column against the baselines only. Node classification ranks by accuracy "
     "(single-label graphs) or micro-F1 (multi-label), link ranking by MRR.",
     ["ranks_nc", "ranks_lp"]),
    ("Node classification, single-label graphs: accuracy",
     "10% of the nodes for training and 10% for validation (or the official split), three random splits per "
     "embedding; mean ± standard deviation over splits and seeds. One table per probe.",
     [f"nc_accuracy_{probe}" for probe in display.NODE_PROBES]),
    ("Node classification, single-label graphs: other metrics (logistic regression)", "",
     [f"nc_{metric}_logistic" for metric in ("macro_f1", "balanced_accuracy", "mcc", "macro_roc_auc",
                                             "macro_average_precision", "top3_accuracy")]),
    ("Node classification, multi-label graphs: micro-F1",
     "A label is predicted when its score is positive; no knowledge of the number of labels of a node is used.",
     [f"nc_micro_f1_{probe}" for probe in display.NODE_PROBES]),
    ("Node classification, multi-label graphs: other metrics (logistic regression)",
     "`micro_f1_known_count` is the protocol of the DeepWalk and node2vec papers: the number of labels of every "
     "test node is given to the classifier.",
     [f"nc_multilabel_{metric}_logistic" for metric in ("macro_f1", "micro_f1_known_count", "micro_average_precision",
                                                        "label_ranking_average_precision", "precision_at_1", "hamming_loss")]),
    ("Link ranking: MRR",
     "Every held-out edge is ranked against 500 sampled non-edges of the same source node. `similarity` is the better "
     "of dot product and cosine on validation; the other tables are one per probe.",
     ["lp_mrr_similarity"] + [f"lp_mrr_{probe}" for probe in display.PAIR_PROBES]),
    ("Link ranking: other metrics (cosine)",
     "`hits@k`, `roc_auc`, `average_precision`: against the 500 sampled non-edges. `all_nodes_mrr`, `recall@k`, "
     "`ndcg@k`, `map@k`: the held-out neighbours of a node ranked among all nodes, known neighbours filtered out.",
     [f"lp_{metric}_cosine" for metric in ("hits@1", "hits@10", "hits@100", "roc_auc", "average_precision",
                                           "all_nodes_mrr", "recall@10", "recall@50", "ndcg@10", "map@10")]),
    ("Cleora: the four ways of feeding and normalising a graph",
     "One column (`complex::reflexive::node`) or two (`src dst`), left or symmetric normalisation, each with the "
     "number of iterations the library suggests for the task; `tuned`: iteration count and family chosen on "
     "validation. Logistic regression for classification, cosine for link ranking.",
     ["families_nc_accuracy", "families_nc_micro_f1", "families_lp_mrr"]),
    ("Cleora: which normalisation for which task",
     "Mean validation score of each family over the graphs on which all four ran, and in parentheses the number "
     "of graphs on which it validated best. The main tables use the left normalisation for classification and the "
     "symmetric one for link ranking, for every graph, on the strength of this table.",
     ["families_validation"]),
    ("Cleora: what a validation set adds",
     "`Cleora`: as in the main tables. `Cleora, tuned`: the number of iterations (from a sweep over twelve values) "
     "and the family chosen on validation data. `gain` in points of the metric. Classification with logistic "
     "regression, link ranking with dot product or cosine.",
     ["tuned_nc", "tuned_lp"]),
    ("Number of iterations: automatic rule against the sweep",
     "`auto`: iterations of `SparseMatrix.suggest_iterations` for the task. `tuned`: best of the sweep on "
     "validation. `loss`: points of the metric (x 100) the automatic count is behind the tuned one; negative: "
     "it is ahead.",
     ["iterations_nc", "iterations_lp"]),
    ("Number of iterations: candidate rules, leave-one-graph-out",
     "From the iteration sweep (one seed; logistic regression / cosine). Loss: test score at the iteration count "
     "that validated best minus test score at the rule's count, in points of the metric (x 100).",
     ["iteration_rule_fit"]),
    ("Number of iterations: loss as a function of the constant c",
     "The form that ships, on all graphs of the sweep, for every candidate value of `c`.",
     ["iteration_rule_constants"]),
    ("Embedding time", "Median of three runs on an idle machine, whole graph, 1024 dimensions.", ["time"]),
    ("Peak memory", "Peak resident memory of the embedding process, interpreter and libraries included.", ["memory"]),
)


def write_report(tables, path):
    lines = ["# Benchmark results", "",
             "Generated by `benchmarks/make_report.py`; do not edit. The protocol is described in "
             "[`benchmarks/README.md`](README.md).", "",
             "**Bold**: best of the row. <u>Underlined</u>: second best. `n/a (size)`: the implementation builds "
             "dense n × n matrices or Python-level graph objects and was not run on a graph of this size. "
             "`time limit` / `out of memory`: the run was started and did not finish.", ""]
    for title, note, names in SECTIONS:
        present = [name for name in names if name in tables and tables[name].strip()]
        if not present:
            continue
        lines += [f"## {title}", ""] + ([note, ""] if note else [])
        for name in present:
            if len(names) > 1:
                lines += [f"### `{name}`", ""]
            lines += [tables[name], ""]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def inject(tables, path):
    """Replaces the marked blocks of a Markdown file with the tables they name."""
    with open(path) as handle:
        text = handle.read()

    def replace(match):
        name = match.group(1)
        if name not in tables:
            raise KeyError(f"{path} asks for the unknown table '{name}'")
        return f"<!-- benchmark:{name} -->\n{tables[name]}\n<!-- /benchmark:{name} -->"

    text, count = re.subn(r"<!-- benchmark:([\w@]+) -->.*?<!-- /benchmark:\1 -->", replace, text, flags=re.S)
    with open(path, "w") as handle:
        handle.write(text)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="directory with datasets/, results/ and timing/")
    parser.add_argument("--out", required=True, help="Markdown file that receives every table")
    parser.add_argument("--readme", action="append", default=[], help="Markdown file with marked blocks to fill")
    parser.add_argument("--dump", default=None, help="write every aggregated number to this .csv.gz file")
    args = parser.parse_args()
    results = Results(os.path.abspath(os.path.expanduser(args.data)))
    tables = build(results)
    if args.dump:
        report.dump({**results.table, **results.timing}, args.dump)
    write_report(tables, args.out)
    print(f"{len(tables)} tables -> {args.out}")
    for path in args.readme:
        print(f"{inject(tables, path)} blocks filled in {path}")


if __name__ == "__main__":
    main()
