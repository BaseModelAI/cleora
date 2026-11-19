"""How the rule behind ``num_iterations="auto"`` was found.

Concern of this module: fitting candidate *forms* of an iteration rule to the sweep and
measuring what each costs. The rule that ships is ``pycleora``'s
``SparseMatrix.suggest_iterations``; this module only justifies its form and constants.

A candidate maps a graph to a number of iterations ``K`` from two quick statistics:
``d``, the mean number of neighbours, and ``s``, the share of the weight on the diagonal
(1/2 for the one-column setup, 0 for two columns). Its *regret* on a sweep curve is the
test score at the iteration count that validated best minus the test score at ``K``, in
points of the metric. Parameters are fitted leave-one-graph-out, on validation scores
only: the regret reported for a graph never comes from parameters that saw it.
"""

import itertools
import os

import numpy as np

from . import configurations, report, selection

MAX_ITERATIONS = max(configurations.SWEEP_ITERATIONS)
SELF_SHARE = {"hyperedge": 0.5, "pair": 0.0}
TASK_LABELS = {"nc": "classification", "lp": "link ranking"}


def curves(sweep_results, task):
    """One record per (dataset, setup, normalisation) with a complete sweep: validation
    and test score for every iteration count, ``d`` and ``s`` of the graph embedded."""
    table = report.aggregate(report.load_jobs(sweep_results))
    probe = selection.SWEEP_PROBE[task]
    datasets = sorted({key[0] for key in table if key[1] == task})
    result = []
    for dataset in datasets:
        for setup in configurations.SETUPS:
            for propagation in ("left", "sym"):
                valid, test, degree = {}, {}, None
                for iterations, name in zip(configurations.SWEEP_ITERATIONS, configurations.sweep(setup, propagation)):
                    entry = table.get((dataset, task, name, 1024))
                    runs = entry["probes"].get(probe) if entry else None
                    if not runs or not runs["validation"]:
                        continue
                    metric = "mrr" if task == "lp" else ("accuracy" if "accuracy" in runs["metrics"][0] else "micro_f1")
                    valid[iterations] = float(np.mean(runs["validation"]))
                    test[iterations] = float(np.mean([m[metric] for m in runs["metrics"]]))
                    degree = entry["degree"]
                if len(valid) == len(configurations.SWEEP_ITERATIONS):
                    best = max(valid, key=valid.get)
                    result.append({"dataset": dataset, "setup": setup, "symmetric": propagation == "sym",
                                   "share": SELF_SHARE[setup], "degree": degree, "valid": valid, "test": test,
                                   "best_iterations": best, "best_test": test[best]})
    return result


def _score_at(curve, iterations):
    """Score at a number of iterations between the grid points: linear in log K."""
    grid = np.array(sorted(curve), dtype=np.float64)
    return float(np.interp(np.log(np.clip(iterations, grid[0], grid[-1])), np.log(grid), [curve[int(k)] for k in grid]))


def regret(record, iterations, which="test"):
    """Points of the metric (accuracy, micro-F1 or MRR, times 100) lost against the
    iteration count that validated best."""
    top = max(record["valid"].values()) if which == "valid" else record["best_test"]
    return 100.0 * (top - _score_at(record[which], iterations))


def _rounded(value):
    return float(np.clip(round(value), 1, MAX_ITERATIONS))


def _mean_regret(records, iterations_of):
    return float(np.mean([regret(r, _rounded(iterations_of(r)), "valid") for r in records]))


# ----------------------------------------------------------------------------- candidate forms
# Every form has shared structure parameters and one number of steps ``c`` per task.

STEPS = (6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48)


def _mixing(record, steps, symmetric_factor):
    factor = symmetric_factor if record["symmetric"] else 1.0
    return steps * factor / ((1.0 - record["share"]) * np.sqrt(max(record["degree"], 1.0)))


def _with_floors(record, steps, symmetric_factor, remainder, minimum):
    forget = np.log(remainder) / np.log(record["share"]) if record["share"] > 0 else 0.0
    return max(_mixing(record, steps, symmetric_factor), forget, minimum)


def _fit_constant(records):
    """The best single iteration count of every task, setup and normalisation."""
    best = {}
    for key in {(r["task"], r["setup"], r["symmetric"]) for r in records}:
        group = [r for r in records if (r["task"], r["setup"], r["symmetric"]) == key]
        best[key] = min(configurations.SWEEP_ITERATIONS, key=lambda k: _mean_regret(group, lambda r: k))
    return best


def _fit_form(records, form, structures):
    """Structure parameters shared by the tasks, the steps ``c`` chosen per task."""
    tasks = sorted({r["task"] for r in records})
    best = None
    for structure in structures:
        steps, total = {}, 0.0
        for task in tasks:
            group = [r for r in records if r["task"] == task]
            losses = {c: _mean_regret(group, lambda r: form(r, c, *structure)) for c in STEPS}
            steps[task] = min(losses, key=losses.get)
            total += losses[steps[task]]
        if best is None or total < best[0]:
            best = (total, steps, structure)
    return best[1], best[2]


FORMS = {
    "one constant per setup and normalisation": None,
    "mixing term only: c / ((1 - s) sqrt(d))": (_mixing, [(factor,) for factor in (1.0, 1.5, 2.0)]),
    "mixing term and floors (the form that ships)": (_with_floors, list(itertools.product(
        (1.0, 1.5, 2.0), (1 / 64, 1 / 256, 1 / 1024), (1, 2, 3)))),
}


def _fit(name, records):
    """``(parameters, iterations(record))`` of a form fitted to ``records``."""
    if FORMS[name] is None:
        constants = _fit_constant(records)
        fallback = int(np.median(list(constants.values())))
        return constants, lambda r: constants.get((r["task"], r["setup"], r["symmetric"]), fallback)
    form, structures = FORMS[name]
    steps, structure = _fit_form(records, form, structures)
    return (steps, structure), lambda r: form(r, steps[r["task"]], *structure)


def leave_one_out(records):
    """``{form: (held-out regret of every record, parameters fitted on all records)}``. The
    held-out regret of a record uses parameters fitted without any curve of its graph."""
    result = {}
    for name in FORMS:
        held = [None] * len(records)
        for dataset in sorted({r["dataset"] for r in records}):
            _, iterations_of = _fit(name, [r for r in records if r["dataset"] != dataset])
            for index, record in enumerate(records):
                if record["dataset"] == dataset:
                    held[index] = regret(record, _rounded(iterations_of(record)))
        result[name] = (held, _fit(name, records)[0])
    return result


def _describe(parameters):
    if isinstance(parameters, dict):
        return "; ".join(f"{TASK_LABELS[task]}, {setup}/{'sym' if symmetric else 'left'}: {k}"
                         for (task, setup, symmetric), k in sorted(parameters.items()))
    steps, structure = parameters
    text = ", ".join(f"c = {c} ({TASK_LABELS[task]})" for task, c in sorted(steps.items())) + f", symmetric x{structure[0]}"
    if len(structure) == 3:
        text += f", remainder 1/{round(1 / structure[1])}, minimum {structure[2]}"
    return text


def markdown(sweep_results):
    """The table of the methodology document: held-out loss of every form, per task."""
    records = [dict(record, task=task) for task in TASK_LABELS for record in curves(sweep_results, task)]
    lines = ["| Rule | Task | Graphs | Curves | Mean loss | Median loss | Worst loss |", "|---|---|---:|---:|---:|---:|---:|"]
    notes = []
    for name, (held, parameters) in leave_one_out(records).items():
        for task, label in TASK_LABELS.items():
            losses = [loss for loss, record in zip(held, records) if record["task"] == task]
            graphs = len({record["dataset"] for record in records if record["task"] == task})
            lines.append(f"| {name} | {label} | {graphs} | {len(losses)} | {np.mean(losses):.2f} | "
                         f"{np.median(losses):.2f} | {np.max(losses):.2f} |")
        notes.append(f"- *{name}*, fitted on all graphs: {_describe(parameters)}")
    return "\n".join(lines) + "\n\n" + "\n".join(notes)


def sensitivity_markdown(sweep_results):
    """Loss of the form that ships as a function of ``c`` (all graphs; the structure
    parameters are those fitted on all graphs): how flat the optimum is."""
    records = [dict(record, task=task) for task in TASK_LABELS for record in curves(sweep_results, task)]
    form, structures = FORMS["mixing term and floors (the form that ships)"]
    _, structure = _fit_form(records, form, structures)
    header = " | ".join(f"{label}: mean | median | worst" for label in TASK_LABELS.values())
    lines = [f"| c | {header} |", "|---:|" + "---:|" * (3 * len(TASK_LABELS))]
    for steps in STEPS:
        cells = []
        for task in TASK_LABELS:
            losses = [regret(r, _rounded(form(r, steps, *structure))) for r in records if r["task"] == task]
            cells += [f"{np.mean(losses):.2f}", f"{np.median(losses):.2f}", f"{np.max(losses):.2f}"]
        lines.append(f"| {steps} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
