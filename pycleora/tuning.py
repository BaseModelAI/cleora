import numpy as np
import time
from typing import Dict, List, Optional, Callable, Any
from itertools import product as iter_product


def grid_search(
    graph,
    labels: Dict[str, int],
    embed_fn: Callable,
    param_grid: Dict[str, List],
    eval_fn: Optional[Callable] = None,
    metric: str = "accuracy",
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    from .metrics import node_classification_scores

    if eval_fn is None:
        eval_fn = lambda g, emb, lbls: node_classification_scores(g, emb, lbls, seed=seed)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(iter_product(*values))

    results = []
    best_score = -1.0
    best_params = None
    best_embeddings = None

    for combo in combinations:
        params = dict(zip(keys, combo))
        t0 = time.time()

        try:
            emb = embed_fn(graph, **params)
            scores = eval_fn(graph, emb, labels)
            score = scores.get(metric, 0.0)
            elapsed = time.time() - t0

            result = {
                "params": params,
                "scores": scores,
                metric: score,
                "time": elapsed,
            }
            results.append(result)

            if verbose:
                print(f"  {params} -> {metric}={score:.4f} ({elapsed:.2f}s)")

            if score > best_score:
                best_score = score
                best_params = params
                best_embeddings = emb
        except Exception as e:
            if verbose:
                print(f"  {params} -> ERROR: {e}")
            results.append({"params": params, "error": str(e)})

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_embeddings": best_embeddings,
        "all_results": results,
        "num_combinations": len(combinations),
        "metric": metric,
    }


def random_search(
    graph,
    labels: Dict[str, int],
    embed_fn: Callable,
    param_distributions: Dict[str, Any],
    n_iter: int = 20,
    eval_fn: Optional[Callable] = None,
    metric: str = "accuracy",
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    from .metrics import node_classification_scores

    if eval_fn is None:
        eval_fn = lambda g, emb, lbls: node_classification_scores(g, emb, lbls, seed=seed)

    rng = np.random.default_rng(seed)
    results = []
    best_score = -1.0
    best_params = None
    best_embeddings = None

    for i in range(n_iter):
        params = {}
        for key, dist in param_distributions.items():
            if isinstance(dist, list):
                params[key] = dist[int(rng.integers(len(dist)))]
            elif isinstance(dist, tuple) and len(dist) == 2:
                low, high = dist
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = int(rng.integers(low, high + 1))
                else:
                    params[key] = float(rng.uniform(low, high))
            else:
                params[key] = dist

        t0 = time.time()
        try:
            emb = embed_fn(graph, **params)
            scores = eval_fn(graph, emb, labels)
            score = scores.get(metric, 0.0)
            elapsed = time.time() - t0

            result = {
                "params": params,
                "scores": scores,
                metric: score,
                "time": elapsed,
            }
            results.append(result)

            if verbose:
                print(f"  [{i+1}/{n_iter}] {params} -> {metric}={score:.4f} ({elapsed:.2f}s)")

            if score > best_score:
                best_score = score
                best_params = params
                best_embeddings = emb
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{n_iter}] {params} -> ERROR: {e}")
            results.append({"params": params, "error": str(e)})

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_embeddings": best_embeddings,
        "all_results": results,
        "n_iter": n_iter,
        "metric": metric,
    }
