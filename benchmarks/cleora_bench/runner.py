"""Runs (dataset, method, dimension, seed, task) jobs and stores one JSON file per job.

Concern of this module: orchestration - which jobs exist, where their files live, when
a job is skipped, what is recorded when one fails. The protocol is in :mod:`tasks`, the
methods in :mod:`methods`.

Layout under the results directory::

    <dataset>/<method>/dim<D>/seed<S>/nc.json     node classification
    <dataset>/<method>/dim<D>/seed<S>/lp.json     link prediction / similarity ranking
    <dataset>/<method>/dim<D>/seed<S>/sample.npz  vectors of up to VISUALISATION_NODES nodes
                                                  (seed 0, whole graph), for the UMAP figures
    <dataset>/<method>/dim<D>/seed<S>/viz.json    record of the run that produced sample.npz
                                                  for a graph without node labels
    <dataset>/<method>/dim<D>/seed<S>/time.json   time and peak memory of embedding the whole
                                                  graph, measured on an otherwise idle machine

Every job is resumable: a finished job is never recomputed, a failed one records why
(time limit, out of memory, the error) and is retried only with ``retry_failed``.
"""

import json
import os
import shutil
import tempfile
import time

import numpy as np

from . import datasets, embed_worker, methods, tasks

VISUALISATION_NODES = 30_000


def job_directory(results, dataset, method_id, dim, seed):
    return os.path.join(results, dataset, method_id, f"dim{dim}", f"seed{seed}")


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump(payload, handle, indent=1, default=float)
    os.replace(path + ".tmp", path)


def _link_split(graph, seed, cache):
    """The link split of (dataset, seed): drawn once, then shared by every method."""
    path = os.path.join(cache, graph["meta"]["name"], f"link_split_seed{seed}.npz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        split = tasks.link_split(graph, seed)
        np.savez(path + ".tmp.npz", **split)
        os.replace(path + ".tmp.npz", path)
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _save_sample(directory, vectors):
    """A fixed random sample of the nodes with their vectors, for the figures."""
    rows = np.sort(np.random.default_rng(0).choice(len(vectors), min(len(vectors), VISUALISATION_NODES), replace=False))
    np.savez(os.path.join(directory, "sample.npz"), rows=rows, vectors=vectors[rows].astype(np.float16))


def _embed(method, params, edges, num_nodes, weights, dim, seed, threads, work, timeout):
    """Runs the method on the nodes of ``edges`` that have an edge. Returns
    ``(vectors in the numbering of the caller or None, record)``."""
    compact_edges, ids = tasks.compact(edges, num_nodes)
    graph_path = os.path.join(work, "graph.npz")
    extra = {} if weights is None else {"weights": weights}
    np.savez(graph_path, edges=compact_edges, num_nodes=len(ids), **extra)
    out, stats = os.path.join(work, "vectors.npy"), os.path.join(work, "stats.json")
    limit = methods.METHODS[method][3]
    if limit is not None and len(ids) > limit:
        return None, {"status": "skipped", "reason": f"needs dense n x n matrices; limited to {limit} nodes"}
    started = time.time()
    failure = embed_worker.run(method, graph_path, dim, seed, threads, out, stats, params=params, timeout=timeout)
    if failure is not None:
        return None, {"status": "failed", "reason": failure, "seconds_until_failure": time.time() - started}
    with open(stats) as handle:
        record = dict(json.load(handle), status="ok", nodes=int(len(ids)), edges=int(len(compact_edges)))
    return tasks.expand(np.load(out), ids, num_nodes), record


def run_job(dataset, method_id, method, params, dim, seed, task, *, data_root, results, cache, threads, timeout,
            device, node_probes=tasks.NODE_PROBES, pair_probes=tasks.PAIR_PROBES, split_seeds=(0, 1, 2),
            retry_failed=False):
    """Runs one job unless its result exists. Returns the result dictionary."""
    directory = job_directory(results, dataset, method_id, dim, seed)
    path = os.path.join(directory, f"{task}.json")
    if os.path.exists(path):
        with open(path) as handle:
            previous = json.load(handle)
        if previous["embedding"]["status"] != "failed" or not retry_failed:
            return previous

    graph = datasets.load(dataset, data_root)
    num_nodes = graph["meta"]["num_nodes"]
    result = {"dataset": dataset, "method": method_id, "implementation": method, "params": params, "dim": dim,
              "seed": seed, "task": task, "nodes": num_nodes, "edges": graph["meta"]["num_edges"]}
    os.makedirs(cache, exist_ok=True)
    os.makedirs(directory, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"{dataset}_{method_id}_", dir=cache)
    try:
        if task == "nc":
            if "labels" not in graph:
                raise ValueError(f"{dataset} has no node labels: it supports the link task only")
            vectors, result["embedding"] = _embed(method, params, graph["edges"], num_nodes, graph.get("weights"),
                                                  dim, seed, threads, work, timeout)
            if vectors is not None:
                result["probes"] = tasks.node_classification(vectors, graph, device, split_seeds, node_probes)
                if seed == 0:
                    _save_sample(directory, vectors)
        elif task in ("viz", "time"):
            # the whole graph is embedded and nothing is evaluated: "viz" keeps a sample of the
            # vectors of a graph without labels, "time" only the time and memory the method took
            vectors, result["embedding"] = _embed(method, params, graph["edges"], num_nodes, graph.get("weights"),
                                                  dim, seed, threads, work, timeout)
            if vectors is not None and task == "viz":
                _save_sample(directory, vectors)
        elif task == "lp":
            split = _link_split(graph, seed, cache)
            # edge weights are not carried into the link task: the split is over unweighted edges
            vectors, result["embedding"] = _embed(method, params, split["embed"], num_nodes, None,
                                                  dim, seed, threads, work, timeout)
            if vectors is not None:
                result["probes"] = tasks.link_prediction(vectors, split, graph, device, seed, pair_probes)
        else:
            raise ValueError(f"Unknown task '{task}'. Use 'nc', 'lp', 'viz' or 'time'.")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _write_json(path, result)
    return result
