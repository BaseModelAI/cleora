import threading
import time
import tracemalloc
import json
import os
import sys
from datetime import datetime

import numpy as np

DATASETS = [
    "facebook",
    "ppi_large",
    "flickr",
    "ogbn_arxiv",
    "yelp",
    "roadnet",
    "livejournal",
]

DATASET_DISPLAY_NAMES = {
    "facebook": "ego-Facebook",
    "ppi_large": "PPI-large",
    "flickr": "Flickr",
    "ogbn_arxiv": "ogbn-arxiv",
    "yelp": "Yelp",
    "roadnet": "roadNet-CA",
    "livejournal": "soc-LiveJournal1",
}

DATASET_NODE_COUNTS = {
    "facebook": 4_039,
    "ppi_large": 56_944,
    "flickr": 89_250,
    "ogbn_arxiv": 169_343,
    "yelp": 716_847,
    "roadnet": 1_965_206,
    "livejournal": 4_847_571,
}

DIM = 1024
LARGE_GRAPH_THRESHOLD = 50_000
HUGE_GRAPH_THRESHOLD = 3_000_000

_sync_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "current_dataset": None,
    "current_algorithm": None,
    "datasets_done": 0,
    "datasets_total": len(DATASETS),
    "percent": 0,
    "logs": [],
    "errors": [],
    "results": None,
    "skipped": [],
}

_lock = threading.Lock()


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    with _lock:
        _sync_state["logs"].append(entry)


def _make_algorithms(n_nodes):
    from pycleora import embed
    from pycleora.algorithms import (
        embed_prone, embed_randne, embed_netmf,
        embed_deepwalk, embed_node2vec,
    )

    algos = {}
    algos["Cleora"] = lambda g: embed(g, DIM, 4)
    algos["Cleora-sym"] = lambda g: embed(g, DIM, 4, propagation="symmetric")
    algos["ProNE"] = lambda g: embed_prone(g, DIM)
    algos["RandNE"] = lambda g: embed_randne(g, DIM)

    if n_nodes <= LARGE_GRAPH_THRESHOLD:
        algos["NetMF"] = lambda g: embed_netmf(g, DIM)

    if n_nodes <= 500:
        algos["DeepWalk"] = lambda g: embed_deepwalk(g, DIM, num_walks=20, walk_length=40)
        algos["Node2Vec"] = lambda g: embed_node2vec(g, DIM, num_walks=20, walk_length=40, p=1.0, q=0.5)
    elif n_nodes <= LARGE_GRAPH_THRESHOLD:
        algos["DeepWalk"] = lambda g: embed_deepwalk(g, DIM, num_walks=10, walk_length=20)
        algos["Node2Vec"] = lambda g: embed_node2vec(g, DIM, num_walks=10, walk_length=20, p=1.0, q=0.5)

    return algos


def _run_sync():
    from pycleora import SparseMatrix, embed
    from pycleora.metrics import node_classification_scores, cross_validate, silhouette_score
    from pycleora.community import detect_communities_louvain
    from pycleora.datasets import load_dataset

    all_results = {}

    try:
        for ds_idx, ds_name in enumerate(DATASETS):
            with _lock:
                _sync_state["current_dataset"] = DATASET_DISPLAY_NAMES.get(ds_name, ds_name)
                _sync_state["current_algorithm"] = None
                _sync_state["datasets_done"] = ds_idx
                _sync_state["percent"] = int((ds_idx / len(DATASETS)) * 100)

            expected_nodes = DATASET_NODE_COUNTS.get(ds_name, 0)
            if expected_nodes > HUGE_GRAPH_THRESHOLD:
                _log(f"Skipping {DATASET_DISPLAY_NAMES.get(ds_name, ds_name)} — {expected_nodes:,} nodes exceeds 3M threshold")
                with _lock:
                    _sync_state["skipped"].append({
                        "dataset": DATASET_DISPLAY_NAMES.get(ds_name, ds_name),
                        "reason": f"Too large ({expected_nodes:,} nodes > 3M threshold)",
                    })
                all_results[ds_name] = {"skipped": True, "num_nodes": expected_nodes}
                continue

            _log(f"Loading dataset: {DATASET_DISPLAY_NAMES.get(ds_name, ds_name)}")
            ds = load_dataset(ds_name)
            n_nodes = ds["num_nodes"]

            _log(f"Building graph: {n_nodes:,} nodes, {ds['num_edges']:,} edges")
            graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
            labels = ds["labels"]
            num_classes = ds["num_classes"]
            has_labels = len(labels) >= 4

            if not has_labels and n_nodes <= LARGE_GRAPH_THRESHOLD:
                _log("Generating community labels via Louvain...")
                labels = detect_communities_louvain(graph)
                num_classes = len(set(labels.values()))
                has_labels = len(labels) >= 4
                _log(f"Found {num_classes} communities")

            algos = _make_algorithms(n_nodes)
            ds_results = {}

            for algo_idx, (algo_name, algo_fn) in enumerate(algos.items()):
                with _lock:
                    _sync_state["current_algorithm"] = algo_name
                    base_pct = (ds_idx / len(DATASETS)) * 100
                    algo_pct = ((algo_idx + 1) / len(algos)) * (100 / len(DATASETS))
                    _sync_state["percent"] = int(base_pct + algo_pct)

                _log(f"  Running {algo_name} on {DATASET_DISPLAY_NAMES.get(ds_name, ds_name)}...")

                try:
                    tracemalloc.start()
                    t0 = time.time()
                    emb = algo_fn(graph)
                    elapsed = time.time() - t0
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    mem_mb = peak / 1024 / 1024

                    result_entry = {"time": elapsed, "memory_mb": mem_mb}

                    if has_labels:
                        nc = node_classification_scores(graph, emb, labels, seed=42)
                        result_entry["accuracy"] = nc["accuracy"]
                        result_entry["macro_f1"] = nc["macro_f1"]

                        true_arr = np.array([labels.get(eid, 0) for eid in graph.entity_ids])
                        result_entry["silhouette"] = float(silhouette_score(emb, true_arr))

                    ds_results[algo_name] = result_entry
                    _log(f"  {algo_name}: time={elapsed:.3f}s, mem={mem_mb:.1f}MB" +
                         (f", acc={result_entry.get('accuracy', 'N/A')}" if has_labels else ""))

                except Exception as e:
                    try:
                        tracemalloc.stop()
                    except Exception:
                        pass
                    ds_results[algo_name] = {"error": str(e)}
                    _log(f"  {algo_name}: ERROR — {str(e)[:100]}")
                    with _lock:
                        _sync_state["errors"].append(f"{algo_name} on {ds_name}: {str(e)[:200]}")

            all_results[ds_name] = ds_results

        with _lock:
            _sync_state["datasets_done"] = len(DATASETS)

        _log("Running cross-validation...")
        cv_results = {}
        for ds_name in DATASETS:
            expected_nodes = DATASET_NODE_COUNTS.get(ds_name, 0)
            if expected_nodes > LARGE_GRAPH_THRESHOLD:
                _log(f"  Skipping CV for {DATASET_DISPLAY_NAMES.get(ds_name, ds_name)} — too large ({expected_nodes:,} nodes)")
                continue

            ds = load_dataset(ds_name)

            _log(f"  Cross-validation on {DATASET_DISPLAY_NAMES.get(ds_name, ds_name)}...")
            with _lock:
                _sync_state["current_dataset"] = DATASET_DISPLAY_NAMES.get(ds_name, ds_name)
                _sync_state["current_algorithm"] = "Cross-validation"

            graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
            labels = ds["labels"]
            if len(labels) < 4:
                labels = detect_communities_louvain(graph)

            emb = embed(graph, DIM, 4)
            cv = cross_validate(graph, emb, labels, k_folds=5, seed=42)
            cv_results[ds_name] = {
                "mean_accuracy": cv["mean_accuracy"],
                "std_accuracy": cv["std_accuracy"],
                "mean_macro_f1": cv["mean_macro_f1"],
                "std_macro_f1": cv["std_macro_f1"],
            }
            _log(f"  {DATASET_DISPLAY_NAMES.get(ds_name, ds_name)}: acc={cv['mean_accuracy']:.4f}±{cv['std_accuracy']:.4f}")

        final_results = {
            "benchmarks": all_results,
            "cross_validation": cv_results,
            "timestamp": datetime.now().isoformat(),
        }

        with _lock:
            _sync_state["results"] = final_results
            _sync_state["percent"] = 100

        try:
            _write_benchmarks_js(final_results)
            _log("Benchmark sync complete! Results saved to benchmarks.js")
        except Exception as e:
            _log(f"WARNING: Failed to write benchmarks.js — {str(e)}")
            with _lock:
                _sync_state["errors"].append(f"File write failed: {str(e)}")
            _log("Benchmark computation complete but results file was not updated. Results are available via API.")

    except Exception as e:
        _log(f"FATAL ERROR: {str(e)}")
        with _lock:
            _sync_state["errors"].append(f"Fatal: {str(e)}")

    finally:
        with _lock:
            _sync_state["running"] = False
            _sync_state["finished_at"] = datetime.now().isoformat()
            _sync_state["current_dataset"] = None
            _sync_state["current_algorithm"] = None


def _write_benchmarks_js(results):
    benchmarks = results["benchmarks"]
    cv = results["cross_validation"]

    datasets_order = ["facebook", "ppi_large", "flickr", "ogbn_arxiv", "yelp"]
    dataset_labels = ["ego-Facebook", "PPI-large", "Flickr", "ogbn-arxiv", "Yelp"]

    all_algos = set()
    for ds_name in datasets_order:
        ds_r = benchmarks.get(ds_name, {})
        if isinstance(ds_r, dict) and "skipped" not in ds_r:
            all_algos.update(ds_r.keys())
    algo_order = ["Cleora", "Cleora-sym", "ProNE", "RandNE", "NetMF", "DeepWalk", "Node2Vec"]
    algorithms = [a for a in algo_order if a in all_algos]

    summary = {}
    for algo in algorithms:
        row = []
        for ds_name in datasets_order:
            ds_r = benchmarks.get(ds_name, {})
            if isinstance(ds_r, dict) and "skipped" not in ds_r:
                algo_r = ds_r.get(algo, {})
                acc = algo_r.get("accuracy")
                row.append(round(acc, 3) if acc is not None else None)
            else:
                row.append(None)
        summary[algo] = row

    speed_algos_order = ["Cleora", "Cleora-sym", "RandNE", "ProNE", "NetMF", "DeepWalk", "Node2Vec"]
    speed_algos = [a for a in speed_algos_order if a in all_algos]
    speed_keys = [
        ("facebook", "facebook"), ("ppi_large", "ppi_large"),
        ("flickr", "flickr"), ("ogbn_arxiv", "ogbn_arxiv"),
        ("yelp", "yelp"), ("roadnet", "roadnet"),
    ]
    speed_data = {"algorithms": speed_algos}
    js_key_map = {
        "facebook": "facebook", "ppi_large": "ppi_large",
        "flickr": "flickr", "ogbn_arxiv": "ogbn_arxiv",
        "yelp": "yelp", "roadnet": "roadnet",
    }
    for ds_name, js_key in speed_keys:
        ds_r = benchmarks.get(ds_name, {})
        row = []
        for algo in speed_algos:
            if isinstance(ds_r, dict) and "skipped" not in ds_r:
                algo_r = ds_r.get(algo, {})
                t = algo_r.get("time")
                row.append(round(t, 3) if t is not None else None)
            else:
                row.append(None)
        speed_data[js_key] = row

    memory_algos_order = ["Cleora", "Cleora-sym", "RandNE", "ProNE", "Node2Vec", "DeepWalk", "NetMF"]
    memory_algos = [a for a in memory_algos_order if a in all_algos]
    memory_data = {"algorithms": memory_algos}
    for ds_name, js_key in speed_keys:
        ds_r = benchmarks.get(ds_name, {})
        row = []
        for algo in memory_algos:
            if isinstance(ds_r, dict) and "skipped" not in ds_r:
                algo_r = ds_r.get(algo, {})
                m = algo_r.get("memory_mb")
                row.append(round(m, 2) if m is not None else None)
            else:
                row.append(None)
        memory_data[js_key] = row

    scatter = {}
    for ds_name in ["facebook", "ogbn_arxiv"]:
        ds_r = benchmarks.get(ds_name, {})
        if isinstance(ds_r, dict) and "skipped" not in ds_r:
            display = DATASET_DISPLAY_NAMES.get(ds_name, ds_name)
            scatter_entry = {}
            for algo, algo_r in ds_r.items():
                if "accuracy" in algo_r and "time" in algo_r:
                    scatter_entry[algo] = {
                        "acc": round(algo_r["accuracy"], 3),
                        "time": round(algo_r["time"], 3),
                    }
            if scatter_entry:
                scatter[display] = scatter_entry

    cv_datasets = []
    cv_mean_acc = []
    cv_std_acc = []
    cv_mean_f1 = []
    cv_std_f1 = []
    for ds_name, cv_r in cv.items():
        cv_datasets.append(DATASET_DISPLAY_NAMES.get(ds_name, ds_name))
        cv_mean_acc.append(round(cv_r["mean_accuracy"], 3))
        cv_std_acc.append(round(cv_r["std_accuracy"], 3))
        cv_mean_f1.append(round(cv_r["mean_macro_f1"], 3))
        cv_std_f1.append(round(cv_r["std_macro_f1"], 3))

    cv_data = {
        "datasets": cv_datasets,
        "meanAccuracy": cv_mean_acc,
        "stdAccuracy": cv_std_acc,
        "meanF1": cv_mean_f1,
        "stdF1": cv_std_f1,
    }

    benchmarks_js_path = os.path.join(os.path.dirname(__file__), "static", "benchmarks.js")

    with open(benchmarks_js_path, "r") as f:
        original = f.read()

    chart_code_start = original.find("function chartDefaults()")
    if chart_code_start == -1:
        chart_code_start = original.find("function buildAccuracyChart()")
    chart_code = original[chart_code_start:] if chart_code_start != -1 else ""

    def _js_val(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)

    def _js_array(arr):
        return "[" + ", ".join(_js_val(v) for v in arr) + "]"

    lines = []
    lines.append("const COLORS = {")
    lines.append("    accent: '#6c63ff',")
    lines.append("    accentBright: '#8b83ff',")
    lines.append("    green: '#34d399',")
    lines.append("    orange: '#f59e0b',")
    lines.append("    red: '#ef4444',")
    lines.append("    blue: '#3b82f6',")
    lines.append("    text: '#e4e4ef',")
    lines.append("    textMuted: '#8888a0',")
    lines.append("    textDim: '#5a5a70',")
    lines.append("    border: '#2a2a3a',")
    lines.append("    bgCard: '#12121a',")
    lines.append("    bg: '#0a0a0f',")
    lines.append("};")
    lines.append("")

    lines.append("const ALGO_COLORS = {")
    algo_colors = {
        "Cleora": "#6c63ff", "Cleora-sym": "#8b83ff", "ProNE": "#f59e0b",
        "RandNE": "#ef4444", "NetMF": "#3b82f6", "DeepWalk": "#f472b6", "Node2Vec": "#fb923c",
    }
    for algo in algorithms:
        lines.append(f"    '{algo}': '{algo_colors.get(algo, '#888888')}',")
    lines.append("};")
    lines.append("")

    lines.append(f"const DATASETS = {_js_array(dataset_labels)};")
    lines.append(f"const ALGORITHMS = {_js_array(algorithms)};")
    lines.append("")

    lines.append("const SUMMARY_DATA = {")
    for algo in algorithms:
        lines.append(f"    '{algo}': {_js_array(summary[algo])},")
    lines.append("};")
    lines.append("")

    lines.append("const SPEED_DATA = {")
    lines.append(f"    algorithms: {_js_array(speed_data['algorithms'])},")
    for ds_name, js_key in speed_keys:
        lines.append(f"    {js_key}: {_js_array(speed_data[js_key])},")
    lines.append("};")
    lines.append("")

    lines.append("const MEMORY_DATA = {")
    lines.append(f"    algorithms: {_js_array(memory_data['algorithms'])},")
    for ds_name, js_key in speed_keys:
        lines.append(f"    {js_key}: {_js_array(memory_data[js_key])},")
    lines.append("};")
    lines.append("")

    lines.append("const SCATTER_DATA = {")
    for ds_display, algos_data in scatter.items():
        lines.append(f"    '{ds_display}': {{")
        for algo, d in algos_data.items():
            lines.append(f"        '{algo}': {{ acc: {d['acc']}, time: {d['time']} }},")
        lines.append("    },")
    lines.append("};")
    lines.append("")

    lines.append("const CV_DATA = {")
    lines.append(f"    datasets: {_js_array(cv_data['datasets'])},")
    lines.append(f"    meanAccuracy: {_js_array(cv_data['meanAccuracy'])},")
    lines.append(f"    stdAccuracy: {_js_array(cv_data['stdAccuracy'])},")
    lines.append(f"    meanF1: {_js_array(cv_data['meanF1'])},")
    lines.append(f"    stdF1: {_js_array(cv_data['stdF1'])},")
    lines.append("};")
    lines.append("")

    if chart_code:
        lines.append(chart_code)

    new_content = "\n".join(lines)

    with open(benchmarks_js_path, "w") as f:
        f.write(new_content)

    _log("benchmarks.js updated with new results")


def start_sync():
    with _lock:
        if _sync_state["running"]:
            return False, "Sync is already running"

        _sync_state["running"] = True
        _sync_state["started_at"] = datetime.now().isoformat()
        _sync_state["finished_at"] = None
        _sync_state["current_dataset"] = None
        _sync_state["current_algorithm"] = None
        _sync_state["datasets_done"] = 0
        _sync_state["datasets_total"] = len(DATASETS)
        _sync_state["percent"] = 0
        _sync_state["logs"] = []
        _sync_state["errors"] = []
        _sync_state["results"] = None
        _sync_state["skipped"] = []

    thread = threading.Thread(target=_run_sync, daemon=True)
    thread.start()
    return True, "Sync started"


def get_status():
    with _lock:
        return {
            "running": _sync_state["running"],
            "started_at": _sync_state["started_at"],
            "finished_at": _sync_state["finished_at"],
            "current_dataset": _sync_state["current_dataset"],
            "current_algorithm": _sync_state["current_algorithm"],
            "datasets_done": _sync_state["datasets_done"],
            "datasets_total": _sync_state["datasets_total"],
            "percent": _sync_state["percent"],
            "logs": list(_sync_state["logs"]),
            "errors": list(_sync_state["errors"]),
            "skipped": list(_sync_state["skipped"]),
            "has_results": _sync_state["results"] is not None,
        }


def get_results():
    with _lock:
        return _sync_state["results"]
