import numpy as np
import time
from typing import Dict, List, Optional, Callable


def _error_entry(error: Exception) -> Dict:
    """The table row of an algorithm or dataset that failed: one failure must not
    cost the measurements of the others, and the row says what happened."""
    return {"error": f"{type(error).__name__}: {error}"}


def benchmark_algorithms(
    graph,
    labels: Dict[str, int],
    algorithms: Dict[str, Callable],
    metrics_fn: Optional[Callable] = None,
    num_runs: int = 1,
    seed: int = 42,
) -> Dict:
    """Time and classification scores of every algorithm on one graph.

    Memory is not measured: the peak of a call cannot be told apart from what the
    process already holds (allocators reuse freed memory), so a number taken here
    would say nothing about the algorithm. Measure it with one process per run.
    """
    from .metrics import node_classification_scores

    if metrics_fn is None:
        metrics_fn = lambda g, emb, lbls: node_classification_scores(g, emb, lbls, seed=seed)

    results = {}
    for name, algo_fn in algorithms.items():
        times = []
        scores_list = []

        for run in range(num_runs):
            try:
                t0 = time.perf_counter()
                emb = algo_fn(graph)
                elapsed = time.perf_counter() - t0

                scores = metrics_fn(graph, emb, labels)
                times.append(elapsed)
                scores_list.append(scores)
            except Exception as e:
                results[name] = _error_entry(e)
                break

        if name not in results:
            avg_scores = {}
            if scores_list:
                for key in scores_list[0]:
                    vals = [s[key] for s in scores_list if isinstance(s.get(key), (int, float))]
                    if vals:
                        avg_scores[key] = float(np.mean(vals))

            results[name] = {
                "avg_time": float(np.mean(times)),
                "std_time": float(np.std(times)) if len(times) > 1 else 0.0,
                "scores": avg_scores,
                "num_runs": num_runs,
            }

    return results


def benchmark_datasets(
    dataset_names: List[str],
    embed_fn: Callable,
    feature_dim: int = 256,
    seed: int = 42,
) -> Dict:
    from .datasets import load_dataset
    from .metrics import node_classification_scores
    from .pycleora import SparseMatrix

    results = {}
    for ds_name in dataset_names:
        try:
            ds = load_dataset(ds_name)
            t0 = time.perf_counter()
            graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
            emb = embed_fn(graph)
            elapsed = time.perf_counter() - t0

            scores = node_classification_scores(graph, emb, ds["labels"], seed=seed)
            results[ds_name] = {
                "num_nodes": ds["num_nodes"],
                "num_edges": ds["num_edges"],
                "num_classes": ds["num_classes"],
                "time": elapsed,
                "scores": scores,
            }
        except Exception as e:
            results[ds_name] = _error_entry(e)

    return results


def format_benchmark_table(results: Dict, metric: str = "accuracy") -> str:
    lines = []
    header = f"{'Algorithm':<15} {'Time (s)':<12} {metric.capitalize():<12}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, data in sorted(results.items()):
        if "error" in data:
            lines.append(f"{name:<15} ERROR: {data['error']}")
        else:
            t = data.get("avg_time", 0)
            s = data.get("scores", {}).get(metric, 0)
            lines.append(f"{name:<15} {t:<12.4f} {s:<12.4f}")

    return "\n".join(lines)


def format_dataset_table(results: Dict, metric: str = "accuracy") -> str:
    lines = []
    header = f"{'Dataset':<20} {'Nodes':<8} {'Edges':<10} {'Time (s)':<12} {metric.capitalize():<12}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, data in sorted(results.items()):
        if "error" in data:
            lines.append(f"{name:<20} ERROR: {data['error']}")
        else:
            n = data.get("num_nodes", 0)
            e = data.get("num_edges", 0)
            t = data.get("time", 0)
            s = data.get("scores", {}).get(metric, 0)
            lines.append(f"{name:<20} {n:<8} {e:<10} {t:<12.4f} {s:<12.4f}")

    return "\n".join(lines)
