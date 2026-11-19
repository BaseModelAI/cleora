"""Which fixed-iteration Cleora configurations make up the "tuned" rows.

Concern of this module: choosing the number of iterations from the sweep. For every
dataset, task and family (setup x normalisation) the iteration count with the best
*validation* score of the sweep is kept; test scores are never read here.

The sweep is a cheap pass (one seed, one probe per task: ``SWEEP_PROBE``); the chosen
configurations are then run like every other method, with all probes and seeds.
"""

import json
import os

import numpy as np

from . import configurations, report

SWEEP_PROBE = {"nc": "logistic", "lp": "cosine"}
SWEEP_DIRECTORY = "sweep"
SELECTION_FILE = "selected_iterations.json"
TUNED = "cleora_tuned"  # in a list of methods: "the configurations chosen for this dataset"


def choose(sweep_results):
    """``{dataset: {task: {family: configuration}}}`` from the job files of a sweep."""
    table = report.aggregate(report.load_jobs(sweep_results))
    chosen = {}
    for setup in configurations.SETUPS:
        for propagation in ("left", "sym"):
            family = f"{setup}/{propagation}"
            for name in configurations.sweep(setup, propagation):
                for (dataset, task, method, _dim), entry in table.items():
                    if method != name or task not in SWEEP_PROBE:
                        continue
                    scores = entry["probes"].get(SWEEP_PROBE[task], {}).get("validation", [])
                    if not scores:
                        continue
                    best = chosen.setdefault(dataset, {}).setdefault(task, {}).get(family)
                    score = float(np.mean(scores))
                    if best is None or score > best[1]:
                        chosen[dataset][task][family] = (name, score)
    return {dataset: {task: {family: name for family, (name, _) in families.items()} for task, families in tasks.items()}
            for dataset, tasks in chosen.items()}


def write(data):
    chosen = choose(os.path.join(data, SWEEP_DIRECTORY))
    with open(os.path.join(data, SELECTION_FILE), "w") as handle:
        json.dump(chosen, handle, indent=1, sort_keys=True)
    return chosen


def load(data):
    path = os.path.join(data, SELECTION_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def configurations_for(data, dataset, task):
    """Names of the configurations chosen for a dataset and task (one per family), sorted."""
    return sorted(set(load(data).get(dataset, {}).get(task, {}).values()))
