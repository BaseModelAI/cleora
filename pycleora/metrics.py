"""Evaluation metrics for embeddings.

The module has two layers.

*Array-level metrics* take labels and scores and return numbers. They are the single
definition of every metric the library and its benchmark report:

* :func:`classification_metrics` - single-label node classification,
* :func:`multilabel_metrics` - multi-label node classification,
* :func:`ranking_metrics` - positions of the correct answer among candidates (MRR, Hits@k),
* :func:`retrieval_metrics` - top-k lists against sets of relevant items (Recall, NDCG, MAP),
* :func:`roc_auc`, :func:`average_precision` - binary scores,
* :func:`normalized_mutual_info`, :func:`adjusted_rand_index`, :func:`purity` - clusterings.

They need NumPy only and are tested against scikit-learn.

*Convenience functions* (:func:`link_prediction_scores`, :func:`node_classification_scores`,
:func:`clustering_scores`, :func:`map_at_k`, :func:`ndcg_at_k`, :func:`silhouette_score`,
:func:`cross_validate`) take a graph and its embeddings and apply those metrics to a
simple, parameter-free use of the vectors (cosine similarity, nearest centroid, k-means).
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ----------------------------------------------------------------------------- binary scores


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1; tied values share the mean of their ranks."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]]))
    ends = np.concatenate([starts[1:], [len(values)]])
    group_rank = (starts + ends + 1) / 2.0
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat(group_rank, ends - starts)
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve: the probability that a random positive scores higher
    than a random negative (ties count one half). ``nan`` if one class is missing."""
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = _average_ranks(scores)[labels].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: the mean of the precision at every positive, taken over the
    distinct score thresholds (the definition scikit-learn uses). ``nan`` without positives."""
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    # last position of every group of equal scores
    ends = np.flatnonzero(np.concatenate([ordered[1:] != ordered[:-1], [True]]))
    true_positives = np.cumsum(labels[order])[ends]
    precision = true_positives / (ends + 1.0)
    recall_steps = np.diff(np.concatenate([[0], true_positives])) / positives
    return float((precision * recall_steps).sum())


# ----------------------------------------------------------------------------- classification


def _f1_from_counts(true_positive, predicted, actual):
    denominator = predicted + actual
    return np.where(denominator > 0, 2.0 * true_positive / np.maximum(denominator, 1), 0.0)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, top_k: Sequence[int] = (3, 5)) -> Dict[str, float]:
    """Metrics of a single-label classifier.

    ``y_true`` holds class indices ``0..C-1`` and ``scores`` one column per class (any
    monotone score: probabilities, logits or margins).

    * ``accuracy`` - share of correct predictions (equal to micro-F1 for single-label data);
    * ``balanced_accuracy`` - mean recall over the classes that occur in ``y_true``;
    * ``macro_f1`` / ``weighted_f1`` - F1 per class, averaged over the classes that occur in
      the labels or the predictions, unweighted / weighted by class frequency;
    * ``mcc`` - Matthews correlation coefficient (multi-class form);
    * ``cohen_kappa`` - agreement beyond chance;
    * ``top{k}_accuracy`` - the true class is among the ``k`` highest scores;
    * ``macro_roc_auc`` / ``macro_average_precision`` - one-vs-rest, averaged over the
      classes that have both positive and negative examples.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores)
    count, num_classes = scores.shape
    predicted = scores.argmax(axis=1)
    confusion = np.bincount(y_true * num_classes + predicted, minlength=num_classes * num_classes)
    confusion = confusion.reshape(num_classes, num_classes).astype(np.float64)
    correct = np.diag(confusion)
    actual = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    occurring = (actual + predicted_count) > 0
    f1 = _f1_from_counts(correct, predicted_count, actual)

    accuracy = correct.sum() / count
    expected = (actual * predicted_count).sum() / count ** 2
    mcc_denominator = np.sqrt((count ** 2 - (predicted_count ** 2).sum()) * (count ** 2 - (actual ** 2).sum()))
    result = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float((correct[actual > 0] / actual[actual > 0]).mean()),
        "macro_f1": float(f1[occurring].mean()),
        "weighted_f1": float((f1 * actual).sum() / count),
        "mcc": float((correct.sum() * count - (actual * predicted_count).sum()) / mcc_denominator)
        if mcc_denominator > 0 else 0.0,
        "cohen_kappa": float((accuracy - expected) / (1 - expected)) if expected < 1 else 0.0,
    }
    true_score = scores[np.arange(count), y_true]
    better = (scores > true_score[:, None]).sum(axis=1)
    for k in top_k:
        if k < num_classes:
            result[f"top{k}_accuracy"] = float((better < k).mean())
    aucs, precisions = [], []
    for label in np.flatnonzero((actual > 0) & (actual < count)):
        aucs.append(roc_auc(y_true == label, scores[:, label]))
        precisions.append(average_precision(y_true == label, scores[:, label]))
    result["macro_roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    result["macro_average_precision"] = float(np.mean(precisions)) if precisions else float("nan")
    return result


def _multilabel_f1(y_true: np.ndarray, predicted: np.ndarray) -> Tuple[float, float, float]:
    """micro-, macro- and samples-F1 of 0/1 predictions."""
    true_positive = (y_true & predicted).sum(axis=0).astype(np.float64)
    micro = _f1_from_counts(true_positive.sum(), predicted.sum(), y_true.sum())
    macro = _f1_from_counts(true_positive, predicted.sum(axis=0), y_true.sum(axis=0)).mean()
    per_sample = _f1_from_counts((y_true & predicted).sum(axis=1), predicted.sum(axis=1), y_true.sum(axis=1))
    return float(micro), float(macro), float(per_sample.mean())


def label_ranking_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """For every true label of a sample: the share of true labels among the labels ranked
    at least as high; averaged over the labels of a sample, then over the samples that
    have at least one true and one false label (others count 1, as in scikit-learn)."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    total = 0.0
    for row in range(0, len(y_true), 4096):
        truth, block = y_true[row:row + 4096], scores[row:row + 4096]
        order = np.argsort(-block, axis=1, kind="stable")
        sorted_scores = np.take_along_axis(block, order, axis=1)
        sorted_truth = np.take_along_axis(truth, order, axis=1)
        # for position p: the last position holding the same score (ties rank last)
        last = np.empty(sorted_scores.shape, dtype=np.int64)
        last[:, -1] = sorted_scores.shape[1] - 1
        for position in range(sorted_scores.shape[1] - 2, -1, -1):
            same = sorted_scores[:, position] == sorted_scores[:, position + 1]
            last[:, position] = np.where(same, last[:, position + 1], position)
        true_until = np.take_along_axis(np.cumsum(sorted_truth, axis=1), last, axis=1)
        precision = np.where(sorted_truth, true_until / (last + 1.0), 0.0)
        labels_per_sample = truth.sum(axis=1)
        informative = (labels_per_sample > 0) & (labels_per_sample < truth.shape[1])
        total += np.where(informative, precision.sum(axis=1) / np.maximum(labels_per_sample, 1), 1.0).sum()
    return float(total / len(y_true))


def multilabel_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> Dict[str, float]:
    """Metrics of a multi-label classifier (``y_true`` and ``scores`` are ``[n, C]``).

    Decisions at the classifier's own threshold (``scores > threshold``):
    ``micro_f1``, ``macro_f1``, ``samples_f1``, ``subset_accuracy``, ``hamming_loss``.

    ``micro_f1_known_count`` / ``macro_f1_known_count`` follow the protocol of the DeepWalk
    and node2vec papers: every sample is assigned its ``k`` highest-scoring labels, where
    ``k`` is its true number of labels. They are reported for comparability with that
    literature; the number of labels is not available in practice.

    Threshold-free: ``micro_roc_auc``, ``macro_roc_auc``, ``micro_average_precision``,
    ``macro_average_precision`` (labels without positives, or without negatives for the
    AUC, are left out of the macro averages), ``label_ranking_average_precision`` and
    ``precision_at_1`` (the highest-scoring label is a true one).
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = scores > threshold
    micro, macro, samples = _multilabel_f1(y_true, predicted)

    order = np.argsort(-scores, axis=1, kind="stable")
    position = np.empty_like(order)
    np.put_along_axis(position, order, np.arange(scores.shape[1])[None, :], axis=1)
    known_count = position < y_true.sum(axis=1, keepdims=True)
    micro_known, macro_known, _ = _multilabel_f1(y_true, known_count)

    positives = y_true.sum(axis=0)
    with_both = np.flatnonzero((positives > 0) & (positives < len(y_true)))
    with_positives = np.flatnonzero(positives > 0)
    top_label = order[:, 0]
    return {
        "micro_f1": micro,
        "macro_f1": macro,
        "samples_f1": samples,
        "micro_f1_known_count": micro_known,
        "macro_f1_known_count": macro_known,
        "subset_accuracy": float((predicted == y_true).all(axis=1).mean()),
        "hamming_loss": float((predicted != y_true).mean()),
        "micro_roc_auc": roc_auc(y_true.ravel(), scores.ravel()),
        "macro_roc_auc": float(np.mean([roc_auc(y_true[:, j], scores[:, j]) for j in with_both]))
        if len(with_both) else float("nan"),
        "micro_average_precision": average_precision(y_true.ravel(), scores.ravel()),
        "macro_average_precision": float(np.mean([average_precision(y_true[:, j], scores[:, j]) for j in with_positives]))
        if len(with_positives) else float("nan"),
        "label_ranking_average_precision": label_ranking_average_precision(y_true, scores),
        "precision_at_1": float(y_true[np.arange(len(y_true)), top_label].mean()),
    }


# ----------------------------------------------------------------------------- ranking


def ranks_among(target_scores: np.ndarray, candidate_scores: np.ndarray) -> np.ndarray:
    """Rank (1 = best) of every target score among its own row of candidate scores.

    A candidate with the same score as the target counts as half a place, so that a
    constant scorer gets the middle rank instead of the best or the worst one.
    ``nan`` candidates are ignored (they stand for candidates that do not exist).
    """
    target = np.asarray(target_scores, dtype=np.float64)[:, None]
    candidates = np.asarray(candidate_scores, dtype=np.float64)
    return 1.0 + (candidates > target).sum(axis=1) + 0.5 * (candidates == target).sum(axis=1)


def ranking_metrics(ranks: np.ndarray, ks: Sequence[int] = (1, 3, 10, 50, 100)) -> Dict[str, float]:
    """Summary of the ranks of the correct answers (1 = best): mean reciprocal rank,
    mean and median rank, and Hits@k (the share of answers ranked within the top ``k``)."""
    ranks = np.asarray(ranks, dtype=np.float64)
    result = {
        "mrr": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
    }
    for k in ks:
        result[f"hits@{k}"] = float((ranks <= k).mean())
    return result


def retrieval_metrics(hits: np.ndarray, num_relevant: np.ndarray, ks: Sequence[int] = (10, 20, 50)) -> Dict[str, float]:
    """Quality of top-k lists, averaged over queries.

    ``hits[q, r]`` tells whether the item at rank ``r`` (0 = first) of query ``q`` is
    relevant and ``num_relevant[q]`` how many relevant items the query has in total.

    * ``recall@k`` - relevant items found in the top ``k`` / all relevant items;
    * ``precision@k`` - relevant items in the top ``k`` / ``k``;
    * ``hit_rate@k`` - at least one relevant item in the top ``k``;
    * ``ndcg@k`` - discounted gain ``1 / log2(rank + 1)`` relative to the ideal list;
    * ``map@k`` - mean of the precision at every relevant position, relative to
      ``min(num_relevant, k)``.
    """
    hits = np.asarray(hits).astype(bool)
    num_relevant = np.asarray(num_relevant, dtype=np.float64)
    width = hits.shape[1]
    discount = 1.0 / np.log2(np.arange(width) + 2.0)
    ideal = np.concatenate([[0.0], np.cumsum(discount)])
    found = np.cumsum(hits, axis=1)
    precision_at = found / (np.arange(width) + 1.0)
    result = {}
    for k in ks:
        if k > width:
            continue
        in_top = found[:, k - 1]
        attainable = np.minimum(num_relevant, k)
        result[f"recall@{k}"] = float((in_top / np.maximum(num_relevant, 1)).mean())
        result[f"precision@{k}"] = float((in_top / k).mean())
        result[f"hit_rate@{k}"] = float((in_top > 0).mean())
        gain = (hits[:, :k] * discount[:k]).sum(axis=1)
        result[f"ndcg@{k}"] = float((gain / np.maximum(ideal[attainable.astype(np.int64)], 1e-12)).mean())
        result[f"map@{k}"] = float(((precision_at[:, :k] * hits[:, :k]).sum(axis=1) / np.maximum(attainable, 1)).mean())
    return result


# ----------------------------------------------------------------------------- clusterings


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    width = int(b.max()) + 1
    return np.bincount(a * width + b, minlength=(int(a.max()) + 1) * width).reshape(-1, width).astype(np.float64)


def normalized_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Mutual information of two labelings divided by the mean of their entropies."""
    table = _contingency(np.asarray(labels_true), np.asarray(labels_pred))
    total = table.sum()
    rows, cols = table.sum(axis=1), table.sum(axis=0)
    nonzero = table > 0
    mutual = (table[nonzero] / total * np.log(table[nonzero] * total / np.outer(rows, cols)[nonzero])).sum()
    entropy = lambda counts: -(counts[counts > 0] / total * np.log(counts[counts > 0] / total)).sum()
    denominator = (entropy(rows) + entropy(cols)) / 2
    return float(mutual / denominator) if denominator > 1e-12 else 0.0


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = _contingency(np.asarray(labels_true), np.asarray(labels_pred))
    pairs = lambda counts: (counts * (counts - 1) / 2).sum()
    total_pairs = pairs(np.array([table.sum()]))
    both, in_true, in_pred = pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = in_true * in_pred / max(total_pairs, 1)
    denominator = (in_true + in_pred) / 2 - expected
    return float((both - expected) / denominator) if abs(denominator) > 1e-10 else 0.0


def purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Share of samples that belong to the most common true class of their cluster."""
    table = _contingency(np.asarray(labels_true), np.asarray(labels_pred))
    return float(table.max(axis=0).sum() / table.sum())


# ----------------------------------------------------------------------------- helpers for the convenience functions


def _unit_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-10)


def _index_map(graph) -> Dict[str, int]:
    return {entity_id: index for index, entity_id in enumerate(graph.entity_ids)}


def _pairs_to_indices(index_map: Dict[str, int], pairs) -> np.ndarray:
    """Index pairs of the edges whose two entities are both known."""
    rows = [(index_map[a], index_map[b]) for a, b in pairs if a in index_map and b in index_map]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def _spherical_kmeans(unit_vectors: np.ndarray, num_clusters: int, seed: int = 0, restarts: int = 5,
                      iterations: int = 50) -> np.ndarray:
    """k-means on the unit sphere (cosine similarity), k-means++ seeding, best of ``restarts``."""
    rng = np.random.default_rng(seed)
    count = len(unit_vectors)
    best_assignment, best_quality = None, -np.inf
    for _ in range(restarts):
        centres = [unit_vectors[rng.integers(count)]]
        distance = 1.0 - unit_vectors @ centres[0]
        for _ in range(1, num_clusters):
            weights = np.maximum(distance, 0) ** 2
            choice = rng.choice(count, p=weights / weights.sum()) if weights.sum() > 0 else rng.integers(count)
            centres.append(unit_vectors[choice])
            distance = np.minimum(distance, 1.0 - unit_vectors @ centres[-1])
        centres = np.array(centres)
        assignment = None
        for _ in range(iterations):
            similarity = unit_vectors @ centres.T
            new_assignment = similarity.argmax(axis=1)
            if assignment is not None and (new_assignment == assignment).all():
                break
            assignment = new_assignment
            sums = np.zeros_like(centres)
            np.add.at(sums, assignment, unit_vectors)
            empty = np.linalg.norm(sums, axis=1) < 1e-10
            sums[empty] = unit_vectors[rng.integers(count, size=int(empty.sum()))]
            centres = _unit_rows(sums)
        quality = float((unit_vectors @ centres.T).max(axis=1).sum())
        if quality > best_quality:
            best_assignment, best_quality = assignment, quality
    return best_assignment


def _nearest_centroid(train_x, train_y, test_x) -> np.ndarray:
    """Predicts, for every test vector, the class whose mean training vector is most
    similar (cosine)."""
    classes = np.unique(train_y)
    centroids = np.stack([train_x[train_y == c].mean(axis=0) for c in classes])
    return classes[(_unit_rows(test_x) @ _unit_rows(centroids).T).argmax(axis=1)]


def _label_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """accuracy / macro-F1 / weighted-F1 of hard predictions, through
    :func:`classification_metrics`."""
    classes, encoded = np.unique(np.concatenate([y_true, y_pred]), return_inverse=True)
    truth, predicted = encoded[:len(y_true)], encoded[len(y_true):]
    one_hot = np.zeros((len(truth), len(classes)))
    one_hot[np.arange(len(truth)), predicted] = 1.0
    return classification_metrics(truth, one_hot, top_k=())


# ----------------------------------------------------------------------------- convenience functions


def link_prediction_scores(
    graph,
    embeddings: np.ndarray,
    test_edges: List[Tuple[str, str]],
    negative_edges: Optional[List[Tuple[str, str]]] = None,
    num_negatives_per_positive: int = 50,
) -> Dict[str, float]:
    """Scores held-out edges against non-edges by cosine similarity.

    Without ``negative_edges``, ``num_negatives_per_positive`` random pairs per test edge
    are drawn (pairs of an entity with itself and edges of ``graph`` are excluded).
    ``mrr`` and ``hits@k`` rank every test edge among *all* negative pairs.
    """
    index_map = _index_map(graph)
    unit = _unit_rows(np.asarray(embeddings, dtype=np.float32))
    positives = _pairs_to_indices(index_map, test_edges)
    if len(positives) == 0:
        raise ValueError("No valid positive edges found")

    if negative_edges is not None:
        negatives = _pairs_to_indices(index_map, negative_edges)
    else:
        rng = np.random.default_rng(42)
        count = graph.num_entities
        indptr, indices, _ = graph.to_csr()
        edge_keys = np.repeat(np.arange(count, dtype=np.int64), np.diff(indptr)) * count + indices
        wanted = len(positives) * num_negatives_per_positive
        drawn = rng.integers(0, count, size=(2 * wanted + 16, 2))
        keep = (drawn[:, 0] != drawn[:, 1]) & ~np.isin(drawn[:, 0] * count + drawn[:, 1], edge_keys)
        negatives = drawn[keep][:wanted]
    if len(negatives) == 0:
        raise ValueError("No valid negative edges found")

    score = lambda pairs: np.einsum("ij,ij->i", unit[pairs[:, 0]], unit[pairs[:, 1]]).astype(np.float64)
    positive_scores, negative_scores = score(positives), score(negatives)
    labels = np.concatenate([np.ones(len(positive_scores), bool), np.zeros(len(negative_scores), bool)])
    all_scores = np.concatenate([positive_scores, negative_scores])
    ordered_negatives = np.sort(negative_scores)
    above = len(ordered_negatives) - np.searchsorted(ordered_negatives, positive_scores, side="right")
    equal = np.searchsorted(ordered_negatives, positive_scores, side="right") - \
        np.searchsorted(ordered_negatives, positive_scores, side="left")
    summary = ranking_metrics(1.0 + above + 0.5 * equal, ks=(1, 3, 10, 50))
    return {
        "auc": roc_auc(labels, all_scores),
        "mrr": summary["mrr"],
        "hits@1": summary["hits@1"],
        "hits@3": summary["hits@3"],
        "hits@10": summary["hits@10"],
        "hits@50": summary["hits@50"],
        "average_precision": average_precision(labels, all_scores),
        "num_positive": len(positive_scores),
        "num_negative": len(negative_scores),
        "mean_positive_score": float(positive_scores.mean()),
        "mean_negative_score": float(negative_scores.mean()),
    }


def node_classification_scores(
    graph,
    embeddings: np.ndarray,
    labels: Dict[str, int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, float]:
    """Nearest-centroid classification of the labelled entities on one random split.

    A quick, training-free check of how well the classes separate; see
    ``benchmarks/`` for the protocol with trained probes.
    """
    index_map = _index_map(graph)
    known = [(index_map[entity], label) for entity, label in labels.items() if entity in index_map]
    if len(known) < 4:
        raise ValueError(f"Need at least 4 labeled entities, got {len(known)}")
    features = np.asarray(embeddings)[[index for index, _ in known]]
    targets = np.array([label for _, label in known])

    order = np.random.default_rng(seed).permutation(len(targets))
    split = int(len(targets) * train_ratio)
    train, test = order[:split], order[split:]
    if len(test) == 0:
        raise ValueError("Test set is empty, reduce train_ratio")
    predicted = _nearest_centroid(features[train], targets[train], features[test])
    scores = _label_scores(targets[test], predicted)
    return {
        "accuracy": scores["accuracy"],
        "macro_f1": scores["macro_f1"],
        "weighted_f1": scores["weighted_f1"],
        "num_classes": len(np.unique(targets[train])),
        "train_size": len(train),
        "test_size": len(test),
    }


def clustering_scores(embeddings: np.ndarray, labels: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """Clusters the embeddings (spherical k-means, as many clusters as there are classes)
    and compares the clusters with the classes: ``nmi``, ``ari`` and ``purity``.

    ``avg_intra_cluster_similarity`` is the mean cosine similarity of pairs of vectors
    of the same class.
    """
    labels = np.asarray(labels)
    if len(labels) != embeddings.shape[0]:
        raise ValueError(f"embeddings has {embeddings.shape[0]} rows but labels has {len(labels)} entries")
    unit = _unit_rows(np.asarray(embeddings, dtype=np.float64))
    classes, encoded = np.unique(labels, return_inverse=True)
    clusters = _spherical_kmeans(unit, len(classes), seed=seed)

    # mean pairwise cosine within a class, from the norm of the class sum
    sums = np.zeros((len(classes), unit.shape[1]))
    np.add.at(sums, encoded, unit)
    sizes = np.bincount(encoded)
    several = sizes > 1
    within = ((sums[several] ** 2).sum(axis=1) - sizes[several]) / (sizes[several] * (sizes[several] - 1))
    return {
        "nmi": normalized_mutual_info(encoded, clusters),
        "ari": adjusted_rand_index(encoded, clusters),
        "purity": purity(encoded, clusters),
        "avg_intra_cluster_similarity": float(within.mean()) if several.any() else 0.0,
        "num_clusters": len(classes),
    }


def _top_k_hits(graph, embeddings: np.ndarray, test_edges, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """For every source entity of the test edges: which of its ``k`` nearest entities
    (cosine; the entity itself and its neighbours in ``graph`` excluded, unless they are
    test targets) are test targets, and how many test targets it has."""
    index_map = _index_map(graph)
    pairs = _pairs_to_indices(index_map, test_edges)
    if len(pairs) == 0:
        return np.zeros((0, k), dtype=bool), np.zeros(0)
    unit = _unit_rows(np.asarray(embeddings, dtype=np.float32))
    count = graph.num_entities
    indptr, indices, _ = graph.to_csr()
    sources, inverse = np.unique(pairs[:, 0], return_inverse=True)
    relevant = np.zeros((len(sources), count), dtype=bool)
    relevant[inverse, pairs[:, 1]] = True
    width = min(k, count)
    hits = np.zeros((len(sources), k), dtype=bool)
    block = max(1, (1 << 26) // max(count, 1))
    for start in range(0, len(sources), block):
        rows = sources[start:start + block]
        similarity = unit[rows] @ unit.T
        for local, source in enumerate(rows):
            similarity[local, indices[indptr[source]:indptr[source + 1]]] = -2.0
        similarity[np.arange(len(rows)), rows] = -2.0
        truth = relevant[start:start + block]
        similarity[truth] = (unit[rows] @ unit.T)[truth]
        top = np.argpartition(-similarity, width - 1, axis=1)[:, :width]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(similarity, top, axis=1), axis=1, kind="stable"), axis=1)
        hits[start:start + block, :width] = np.take_along_axis(truth, top, axis=1)
    return hits, relevant.sum(axis=1)


def map_at_k(graph, embeddings: np.ndarray, test_edges: List[Tuple[str, str]], k: int = 10) -> float:
    hits, num_relevant = _top_k_hits(graph, embeddings, test_edges, k)
    return retrieval_metrics(hits, num_relevant, ks=(k,))[f"map@{k}"] if len(hits) else 0.0


def ndcg_at_k(graph, embeddings: np.ndarray, test_edges: List[Tuple[str, str]], k: int = 10) -> float:
    hits, num_relevant = _top_k_hits(graph, embeddings, test_edges, k)
    return retrieval_metrics(hits, num_relevant, ks=(k,))[f"ndcg@{k}"] if len(hits) else 0.0


def silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient with the cosine distance.

    The mean distance of a vector to a group of unit vectors follows from the sum of
    the group, so no pairwise distance matrix is built."""
    labels = np.asarray(labels)
    count = len(labels)
    classes, encoded = np.unique(labels, return_inverse=True)
    if count < 2 or len(classes) < 2:
        return 0.0
    unit = _unit_rows(np.asarray(embeddings, dtype=np.float64))
    sums = np.zeros((len(classes), unit.shape[1]))
    np.add.at(sums, encoded, unit)
    sizes = np.bincount(encoded).astype(np.float64)
    similarity_sum = unit @ sums.T  # [n, classes]: sum of cosines to every member of the class
    own = similarity_sum[np.arange(count), encoded]
    own_size = sizes[encoded]
    with np.errstate(divide="ignore", invalid="ignore"):
        within = 1.0 - (own - 1.0) / (own_size - 1.0)
    other = 1.0 - similarity_sum / sizes[None, :]
    other[np.arange(count), encoded] = np.inf
    nearest = other.min(axis=1)
    silhouette = np.where(own_size > 1, (nearest - within) / np.maximum(np.maximum(within, nearest), 1e-10), 0.0)
    return float(silhouette.mean())


def cross_validate(
    graph,
    embeddings: np.ndarray,
    labels: Dict[str, int],
    k_folds: int = 5,
    eval_fn: Optional[Callable] = None,
    seed: int = 42,
) -> Dict[str, float]:
    index_map = _index_map(graph)
    valid_entities = [(entity, label) for entity, label in labels.items() if entity in index_map]

    if k_folds < 2:
        raise ValueError(f"k_folds must be >= 2, got {k_folds}")
    if len(valid_entities) < k_folds:
        raise ValueError(f"Not enough labeled entities ({len(valid_entities)}) for {k_folds}-fold CV")

    order = np.random.default_rng(seed).permutation(len(valid_entities))
    fold_size = len(valid_entities) // k_folds
    accuracies, f1s = [], []
    for fold in range(k_folds):
        start = fold * fold_size
        end = start + fold_size if fold < k_folds - 1 else len(valid_entities)
        test_positions = set(order[start:end].tolist())
        train_labels = {e: l for i, (e, l) in enumerate(valid_entities) if i not in test_positions}
        test_labels = {e: l for i, (e, l) in enumerate(valid_entities) if i in test_positions}
        evaluate = eval_fn or _simple_classify
        scores = evaluate(graph, embeddings, train_labels, test_labels)
        accuracies.append(scores.get("accuracy", 0.0))
        f1s.append(scores.get("macro_f1", 0.0))

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "fold_accuracies": accuracies,
        "k_folds": k_folds,
    }


def _simple_classify(graph, embeddings, train_labels, test_labels):
    index_map = _index_map(graph)
    embeddings = np.asarray(embeddings)
    train = [(index_map[e], l) for e, l in train_labels.items() if e in index_map]
    test = [(index_map[e], l) for e, l in test_labels.items() if e in index_map]
    if not train or not test:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    predicted = _nearest_centroid(embeddings[[i for i, _ in train]], np.array([l for _, l in train]),
                                  embeddings[[i for i, _ in test]])
    scores = _label_scores(np.array([l for _, l in test]), predicted)
    return {"accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"]}
