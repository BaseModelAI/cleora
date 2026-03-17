import numpy as np
from typing import List, Tuple, Optional, Dict, Callable


def link_prediction_scores(
    graph,
    embeddings: np.ndarray,
    test_edges: List[Tuple[str, str]],
    negative_edges: Optional[List[Tuple[str, str]]] = None,
    num_negatives_per_positive: int = 50,
) -> Dict[str, float]:
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    n = graph.num_entities

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms

    pos_scores = []
    for a, b in test_edges:
        ia, ib = index_map.get(a), index_map.get(b)
        if ia is None or ib is None:
            continue
        pos_scores.append(float(np.dot(normed[ia], normed[ib])))

    if not pos_scores:
        raise ValueError("No valid positive edges found")

    rng = np.random.default_rng(42)

    if negative_edges is not None:
        neg_scores = []
        for a, b in negative_edges:
            ia, ib = index_map.get(a), index_map.get(b)
            if ia is None or ib is None:
                continue
            neg_scores.append(float(np.dot(normed[ia], normed[ib])))
    else:
        neg_scores = []
        for _ in range(len(pos_scores) * num_negatives_per_positive):
            i, j = rng.integers(0, n, size=2)
            neg_scores.append(float(np.dot(normed[i], normed[j])))

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)

    all_scores = np.concatenate([pos_arr, neg_arr])
    all_labels = np.concatenate([np.ones(len(pos_arr)), np.zeros(len(neg_arr))])

    sorted_idx = np.argsort(-all_scores)
    sorted_labels = all_labels[sorted_idx]

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, 'trapezoid') else float(np.trapz(tpr, fpr))

    ranks = []
    for ps in pos_scores:
        rank = int(np.sum(neg_arr >= ps)) + 1
        ranks.append(rank)
    ranks = np.array(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks))

    hits_at_1 = float(np.mean(ranks <= 1))
    hits_at_3 = float(np.mean(ranks <= 3))
    hits_at_10 = float(np.mean(ranks <= 10))
    hits_at_50 = float(np.mean(ranks <= 50))

    ap = float(np.mean(pos_arr > np.median(neg_arr)))

    return {
        "auc": auc,
        "mrr": mrr,
        "hits@1": hits_at_1,
        "hits@3": hits_at_3,
        "hits@10": hits_at_10,
        "hits@50": hits_at_50,
        "average_precision": ap,
        "num_positive": len(pos_scores),
        "num_negative": len(neg_scores),
        "mean_positive_score": float(np.mean(pos_arr)),
        "mean_negative_score": float(np.mean(neg_arr)),
    }


def node_classification_scores(
    graph,
    embeddings: np.ndarray,
    labels: Dict[str, int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, float]:
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}

    indices = []
    y = []
    for entity_id, label in labels.items():
        idx = index_map.get(entity_id)
        if idx is not None:
            indices.append(idx)
            y.append(label)

    if len(indices) < 4:
        raise ValueError(f"Need at least 4 labeled entities, got {len(indices)}")

    X = embeddings[indices]
    y = np.array(y)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    split = int(len(y) * train_ratio)
    train_idx, test_idx = perm[:split], perm[split:]

    if len(test_idx) == 0:
        raise ValueError("Test set is empty, reduce train_ratio")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    classes = np.unique(y_train)
    centroids = {}
    for c in classes:
        mask = y_train == c
        centroids[c] = np.mean(X_train[mask], axis=0)

    norms_test = np.linalg.norm(X_test, axis=1, keepdims=True)
    norms_test = np.maximum(norms_test, 1e-10)
    X_test_normed = X_test / norms_test

    y_pred = []
    for i in range(len(X_test)):
        best_sim = -2.0
        best_class = classes[0]
        for c in classes:
            c_norm = np.linalg.norm(centroids[c])
            if c_norm < 1e-10:
                continue
            sim = float(np.dot(X_test_normed[i], centroids[c] / c_norm))
            if sim > best_sim:
                best_sim = sim
                best_class = c
        y_pred.append(best_class)

    y_pred = np.array(y_pred)
    accuracy = float(np.mean(y_pred == y_test))

    per_class_f1 = []
    for c in np.unique(y):
        tp = np.sum((y_pred == c) & (y_test == c))
        fp = np.sum((y_pred == c) & (y_test != c))
        fn = np.sum((y_pred != c) & (y_test == c))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))

    weights = []
    weighted_f1_sum = 0.0
    for i, c in enumerate(np.unique(y)):
        w = np.sum(y_test == c)
        weights.append(w)
        weighted_f1_sum += per_class_f1[i] * w
    weighted_f1 = float(weighted_f1_sum / max(sum(weights), 1))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_classes": len(classes),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
    }


def clustering_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    n = len(labels)
    if n != embeddings.shape[0]:
        raise ValueError(f"embeddings has {embeddings.shape[0]} rows but labels has {n} entries")

    unique_labels = np.unique(labels)
    k = len(unique_labels)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms

    label_map = {l: i for i, l in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])

    centroids = np.zeros((k, embeddings.shape[1]))
    for i in range(k):
        mask = mapped_labels == i
        if np.sum(mask) > 0:
            centroids[i] = np.mean(normed[mask], axis=0)

    sim_matrix = normed @ centroids.T
    predicted = np.argmax(sim_matrix, axis=1)

    contingency = np.zeros((k, k), dtype=np.int64)
    for i in range(n):
        contingency[mapped_labels[i], predicted[i]] += 1

    from itertools import permutations
    if k <= 10:
        best_acc = 0.0
        for perm in permutations(range(k)):
            acc = sum(contingency[i, perm[i]] for i in range(k)) / n
            if acc > best_acc:
                best_acc = acc
        purity = best_acc
    else:
        row_max = np.max(contingency, axis=1)
        purity = float(np.sum(row_max) / n)

    a = np.zeros(n, dtype=np.int64)
    b = np.zeros(n, dtype=np.int64)
    for i in range(n):
        a[i] = mapped_labels[i]
        b[i] = predicted[i]

    nmi = _normalized_mutual_info(a, b, k)

    intra_sim = 0.0
    count = 0
    for i in range(k):
        mask = mapped_labels == i
        cluster_vecs = normed[mask]
        if len(cluster_vecs) > 1:
            sims = cluster_vecs @ cluster_vecs.T
            n_c = len(cluster_vecs)
            intra_sim += (np.sum(sims) - n_c) / max(n_c * (n_c - 1), 1)
            count += 1
    avg_intra_similarity = float(intra_sim / max(count, 1))

    return {
        "nmi": nmi,
        "purity": purity,
        "avg_intra_cluster_similarity": avg_intra_similarity,
        "num_clusters": k,
    }


def map_at_k(
    graph,
    embeddings: np.ndarray,
    test_edges: List[Tuple[str, str]],
    k: int = 10,
) -> float:
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms

    rows, cols, _, _, _ = graph.to_sparse_csr()
    existing = set()
    for r, c in zip(rows, cols):
        existing.add((int(r), int(c)))

    aps = []
    queries = {}
    for a, b in test_edges:
        ia, ib = index_map.get(a), index_map.get(b)
        if ia is not None and ib is not None:
            queries.setdefault(ia, set()).add(ib)

    for src, true_targets in queries.items():
        sims = normed @ normed[src]
        sims[src] = -2.0
        for other in range(len(sims)):
            if (src, other) in existing and other not in true_targets:
                sims[other] = -2.0
        top_k = np.argsort(sims)[::-1][:k]

        hits = 0
        ap_sum = 0.0
        for rank, idx in enumerate(top_k):
            if idx in true_targets:
                hits += 1
                ap_sum += hits / (rank + 1)
        aps.append(ap_sum / min(len(true_targets), k))

    return float(np.mean(aps)) if aps else 0.0


def ndcg_at_k(
    graph,
    embeddings: np.ndarray,
    test_edges: List[Tuple[str, str]],
    k: int = 10,
) -> float:
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms

    queries = {}
    for a, b in test_edges:
        ia, ib = index_map.get(a), index_map.get(b)
        if ia is not None and ib is not None:
            queries.setdefault(ia, set()).add(ib)

    rows, cols, _, _, _ = graph.to_sparse_csr()
    existing = set()
    for r, c in zip(rows, cols):
        existing.add((int(r), int(c)))

    ndcgs = []
    for src, true_targets in queries.items():
        sims = normed @ normed[src]
        sims[src] = -2.0
        for other in range(len(sims)):
            if (src, other) in existing and other not in true_targets:
                sims[other] = -2.0
        top_k = np.argsort(sims)[::-1][:k]

        dcg = 0.0
        for rank, idx in enumerate(top_k):
            if idx in true_targets:
                dcg += 1.0 / np.log2(rank + 2)

        ideal_hits = min(len(true_targets), k)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))

        ndcgs.append(dcg / max(idcg, 1e-10))

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    n = len(labels_true)
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    contingency = np.zeros((len(classes_true), len(classes_pred)), dtype=np.int64)

    true_map = {c: i for i, c in enumerate(classes_true)}
    pred_map = {c: i for i, c in enumerate(classes_pred)}

    for i in range(n):
        contingency[true_map[labels_true[i]], pred_map[labels_pred[i]]] += 1

    sum_comb_c = sum(int(nij) * (int(nij) - 1) // 2 for row in contingency for nij in row)
    sum_comb_a = sum(int(ai) * (int(ai) - 1) // 2 for ai in contingency.sum(axis=1))
    sum_comb_b = sum(int(bj) * (int(bj) - 1) // 2 for bj in contingency.sum(axis=0))
    total_comb = n * (n - 1) // 2

    expected = sum_comb_a * sum_comb_b / max(total_comb, 1)
    max_index = (sum_comb_a + sum_comb_b) / 2
    denom = max_index - expected

    if abs(denom) < 1e-10:
        return 0.0
    return float((sum_comb_c - expected) / denom)


def silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    if n < 2:
        return 0.0

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms
    dist_matrix = 1.0 - normed @ normed.T

    silhouettes = []
    for i in range(n):
        own_label = labels[i]
        own_mask = labels == own_label
        own_count = np.sum(own_mask) - 1

        if own_count <= 0:
            silhouettes.append(0.0)
            continue

        a_i = np.sum(dist_matrix[i, own_mask]) / own_count

        b_i = np.inf
        for label in unique_labels:
            if label == own_label:
                continue
            other_mask = labels == label
            other_count = np.sum(other_mask)
            if other_count > 0:
                avg_dist = np.sum(dist_matrix[i, other_mask]) / other_count
                b_i = min(b_i, avg_dist)

        if b_i == np.inf:
            silhouettes.append(0.0)
        else:
            s_i = (b_i - a_i) / max(a_i, b_i, 1e-10)
            silhouettes.append(s_i)

    return float(np.mean(silhouettes))


def cross_validate(
    graph,
    embeddings: np.ndarray,
    labels: Dict[str, int],
    k_folds: int = 5,
    eval_fn: Optional[Callable] = None,
    seed: int = 42,
) -> Dict[str, float]:
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    valid_entities = [(eid, label) for eid, label in labels.items() if eid in index_map]

    if k_folds < 2:
        raise ValueError(f"k_folds must be >= 2, got {k_folds}")
    if len(valid_entities) < k_folds:
        raise ValueError(f"Not enough labeled entities ({len(valid_entities)}) for {k_folds}-fold CV")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(valid_entities))
    fold_size = len(valid_entities) // k_folds

    all_accuracies = []
    all_f1s = []

    for fold in range(k_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k_folds - 1 else len(valid_entities)
        test_indices = set(perm[test_start:test_end].tolist())

        train_labels = {}
        test_labels = {}
        for idx, (eid, label) in enumerate(valid_entities):
            if idx in test_indices:
                test_labels[eid] = label
            else:
                train_labels[eid] = label

        if eval_fn:
            scores = eval_fn(graph, embeddings, train_labels, test_labels)
        else:
            scores = _simple_classify(graph, embeddings, train_labels, test_labels)

        all_accuracies.append(scores.get("accuracy", 0.0))
        all_f1s.append(scores.get("macro_f1", 0.0))

    return {
        "mean_accuracy": float(np.mean(all_accuracies)),
        "std_accuracy": float(np.std(all_accuracies)),
        "mean_macro_f1": float(np.mean(all_f1s)),
        "std_macro_f1": float(np.std(all_f1s)),
        "fold_accuracies": all_accuracies,
        "k_folds": k_folds,
    }


def _simple_classify(graph, embeddings, train_labels, test_labels):
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    classes = sorted(set(train_labels.values()))
    centroids = {}
    for c in classes:
        vecs = [embeddings[index_map[eid]] for eid, label in train_labels.items() if label == c and eid in index_map]
        if vecs:
            centroids[c] = np.mean(vecs, axis=0)

    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for eid, true_label in test_labels.items():
        idx = index_map.get(eid)
        if idx is None:
            continue
        vec = embeddings[idx]
        norm_v = np.linalg.norm(vec)
        if norm_v < 1e-10:
            continue
        vec_n = vec / norm_v

        best_sim = -2
        best_class = classes[0]
        for c, centroid in centroids.items():
            cn = np.linalg.norm(centroid)
            if cn < 1e-10:
                continue
            sim = float(np.dot(vec_n, centroid / cn))
            if sim > best_sim:
                best_sim = sim
                best_class = c

        y_true.append(true_label)
        y_pred.append(best_class)
        total += 1
        if best_class == true_label:
            correct += 1

    accuracy = correct / max(total, 1)

    per_class_f1 = []
    all_classes = sorted(set(y_true + y_pred))
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    for c in all_classes:
        tp = np.sum((y_pred_arr == c) & (y_true_arr == c))
        fp = np.sum((y_pred_arr == c) & (y_true_arr != c))
        fn = np.sum((y_pred_arr != c) & (y_true_arr == c))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        per_class_f1.append(f1)

    return {"accuracy": accuracy, "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0}


def _normalized_mutual_info(a: np.ndarray, b: np.ndarray, k: int) -> float:
    n = len(a)
    contingency = np.zeros((k, k), dtype=np.float64)
    for i in range(n):
        contingency[a[i], b[i]] += 1

    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    mi = 0.0
    for i in range(k):
        for j in range(k):
            if contingency[i, j] > 0:
                mi += contingency[i, j] / n * np.log(
                    n * contingency[i, j] / max(row_sums[i] * col_sums[j], 1e-10)
                )

    h_a = -np.sum(row_sums / n * np.log(np.maximum(row_sums / n, 1e-10)))
    h_b = -np.sum(col_sums / n * np.log(np.maximum(col_sums / n, 1e-10)))

    denom = (h_a + h_b) / 2
    if denom < 1e-10:
        return 0.0
    return float(mi / denom)
