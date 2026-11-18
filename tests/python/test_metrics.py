"""pycleora.metrics against scikit-learn and against brute force."""

import numpy as np
import pytest

from pycleora import SparseMatrix, embed, metrics

sklearn_metrics = pytest.importorskip("sklearn.metrics")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_roc_auc_and_average_precision_match_sklearn_with_ties(rng):
    for round_to in (None, 1):  # continuous scores, and heavily tied ones
        labels = rng.random(500) < 0.3
        scores = rng.normal(size=500) + labels
        if round_to is not None:
            scores = scores.round(round_to)
        assert metrics.roc_auc(labels, scores) == pytest.approx(sklearn_metrics.roc_auc_score(labels, scores), abs=1e-12)
        assert metrics.average_precision(labels, scores) == pytest.approx(
            sklearn_metrics.average_precision_score(labels, scores), abs=1e-12)
    assert np.isnan(metrics.roc_auc(np.ones(5, bool), np.arange(5.0)))
    assert np.isnan(metrics.average_precision(np.zeros(5, bool), np.arange(5.0)))
    # a constant scorer is exactly at chance
    assert metrics.roc_auc(labels, np.zeros(500)) == 0.5


def test_classification_metrics_match_sklearn(rng):
    num_classes, count = 7, 900
    y = rng.integers(0, num_classes - 1, count)  # the last class never occurs
    scores = rng.normal(size=(count, num_classes))
    scores[np.arange(count), y] += 1.2
    predicted = scores.argmax(axis=1)
    result = metrics.classification_metrics(y, scores)

    assert result["accuracy"] == pytest.approx(sklearn_metrics.accuracy_score(y, predicted))
    assert result["balanced_accuracy"] == pytest.approx(sklearn_metrics.balanced_accuracy_score(y, predicted))
    assert result["macro_f1"] == pytest.approx(sklearn_metrics.f1_score(y, predicted, average="macro"))
    assert result["weighted_f1"] == pytest.approx(sklearn_metrics.f1_score(y, predicted, average="weighted"))
    assert result["mcc"] == pytest.approx(sklearn_metrics.matthews_corrcoef(y, predicted))
    assert result["cohen_kappa"] == pytest.approx(sklearn_metrics.cohen_kappa_score(y, predicted))
    labels = np.arange(num_classes)
    for k in (3, 5):
        assert result[f"top{k}_accuracy"] == pytest.approx(
            sklearn_metrics.top_k_accuracy_score(y, scores, k=k, labels=labels))
    present = np.unique(y)
    assert result["macro_roc_auc"] == pytest.approx(
        np.mean([sklearn_metrics.roc_auc_score(y == c, scores[:, c]) for c in present]))
    assert result["macro_average_precision"] == pytest.approx(
        np.mean([sklearn_metrics.average_precision_score(y == c, scores[:, c]) for c in present]))


def test_multilabel_metrics_match_sklearn(rng):
    count, num_labels = 600, 9
    y = rng.random((count, num_labels)) < 0.25
    y[:, -1] = False            # a label without positives
    y[:5] = False               # samples without labels
    y[5:8, :-1] = True          # samples with (almost) every label
    scores = rng.normal(size=(count, num_labels)) + 1.5 * y - 0.7
    scores = scores.round(1)    # ties
    predicted = scores > 0
    result = metrics.multilabel_metrics(y, scores)

    f1 = lambda average, prediction=predicted: sklearn_metrics.f1_score(y, prediction, average=average, zero_division=0)
    assert result["micro_f1"] == pytest.approx(f1("micro"))
    assert result["macro_f1"] == pytest.approx(f1("macro"))
    assert result["samples_f1"] == pytest.approx(f1("samples"))
    assert result["subset_accuracy"] == pytest.approx(sklearn_metrics.accuracy_score(y, predicted))
    assert result["hamming_loss"] == pytest.approx(sklearn_metrics.hamming_loss(y, predicted))
    with_positives = y.sum(axis=0) > 0
    assert result["micro_roc_auc"] == pytest.approx(sklearn_metrics.roc_auc_score(y.ravel(), scores.ravel()))
    assert result["macro_roc_auc"] == pytest.approx(
        sklearn_metrics.roc_auc_score(y[:, with_positives], scores[:, with_positives], average="macro"))
    assert result["micro_average_precision"] == pytest.approx(
        sklearn_metrics.average_precision_score(y.ravel(), scores.ravel()))
    assert result["macro_average_precision"] == pytest.approx(
        sklearn_metrics.average_precision_score(y[:, with_positives], scores[:, with_positives], average="macro"))
    assert result["label_ranking_average_precision"] == pytest.approx(
        sklearn_metrics.label_ranking_average_precision_score(y, scores))

    # the "known number of labels" protocol: top-k labels with k = true label count
    order = np.argsort(-scores, axis=1, kind="stable")
    known = np.zeros_like(y)
    for row, k in enumerate(y.sum(axis=1)):
        known[row, order[row, :k]] = True
    assert result["micro_f1_known_count"] == pytest.approx(f1("micro", known))
    assert result["macro_f1_known_count"] == pytest.approx(f1("macro", known))
    assert result["precision_at_1"] == pytest.approx(y[np.arange(count), order[:, 0]].mean())


def test_ranks_count_ties_as_half_a_place():
    targets = np.array([0.5, 0.5, 0.9])
    candidates = np.array([[0.1, 0.5, 0.7, np.nan], [0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.3, 0.4]])
    np.testing.assert_allclose(metrics.ranks_among(targets, candidates), [2.5, 3.0, 1.0])
    summary = metrics.ranking_metrics(np.array([1.0, 2.0, 4.0, 200.0]), ks=(1, 3, 100))
    assert summary["mrr"] == pytest.approx((1 + 0.5 + 0.25 + 0.005) / 4)
    assert (summary["hits@1"], summary["hits@3"], summary["hits@100"]) == (0.25, 0.5, 0.75)
    assert summary["median_rank"] == 3.0


def test_retrieval_metrics_match_brute_force(rng):
    queries, width = 40, 20
    hits = rng.random((queries, width)) < 0.2
    num_relevant = hits.sum(axis=1) + rng.integers(0, 5, queries)
    num_relevant[0] = 0  # a query without relevant items scores zero everywhere
    hits[0] = False
    result = metrics.retrieval_metrics(hits, num_relevant, ks=(5, 20))
    for k in (5, 20):
        recall, precision, hit_rate, ndcg, mean_ap = [], [], [], [], []
        for q in range(queries):
            top = hits[q, :k]
            recall.append(top.sum() / max(num_relevant[q], 1))
            precision.append(top.sum() / k)
            hit_rate.append(float(top.any()))
            dcg = sum(1 / np.log2(r + 2) for r in range(k) if top[r])
            ideal = sum(1 / np.log2(r + 2) for r in range(min(num_relevant[q], k)))
            ndcg.append(dcg / ideal if ideal else 0.0)
            found = 0
            total = 0.0
            for r in range(k):
                if top[r]:
                    found += 1
                    total += found / (r + 1)
            mean_ap.append(total / max(min(num_relevant[q], k), 1))
        assert result[f"recall@{k}"] == pytest.approx(np.mean(recall))
        assert result[f"precision@{k}"] == pytest.approx(np.mean(precision))
        assert result[f"hit_rate@{k}"] == pytest.approx(np.mean(hit_rate))
        assert result[f"ndcg@{k}"] == pytest.approx(np.mean(ndcg))
        assert result[f"map@{k}"] == pytest.approx(np.mean(mean_ap))


def test_clustering_comparisons_match_sklearn(rng):
    a = rng.integers(0, 6, 400)
    b = np.where(rng.random(400) < 0.7, a, rng.integers(0, 4, 400))
    assert metrics.normalized_mutual_info(a, b) == pytest.approx(sklearn_metrics.normalized_mutual_info_score(a, b))
    assert metrics.adjusted_rand_index(a, b) == pytest.approx(sklearn_metrics.adjusted_rand_score(a, b))
    table = sklearn_metrics.cluster.contingency_matrix(a, b)
    assert metrics.purity(a, b) == pytest.approx(table.max(axis=0).sum() / 400)


def test_silhouette_matches_sklearn(rng):
    labels = rng.integers(0, 4, 300)
    labels[0] = 9  # a class with a single member
    vectors = rng.normal(size=(300, 16)) + 2.0 * np.eye(16)[labels % 16]
    assert metrics.silhouette_score(vectors, labels) == pytest.approx(
        sklearn_metrics.silhouette_score(vectors, labels, metric="cosine"), abs=1e-9)


def test_clustering_scores_cluster_without_looking_at_the_labels(rng):
    labels = np.repeat(np.arange(4), 60)
    separated = rng.normal(scale=0.2, size=(240, 12)) + 3.0 * np.eye(12)[labels]
    found = metrics.clustering_scores(separated, labels)
    assert found["nmi"] > 0.95 and found["purity"] > 0.95 and found["ari"] > 0.95
    assert found["num_clusters"] == 4
    # Vectors that carry no information about the labels must not look clustered
    # (centroids computed from the true labels used to give a high score here).
    noise = metrics.clustering_scores(rng.normal(size=(240, 12)), labels)
    assert noise["nmi"] < 0.1 and noise["ari"] < 0.1


@pytest.fixture
def small_graph():
    rng = np.random.default_rng(3)
    blocks = np.repeat(np.arange(3), 40)
    edges = []
    for _ in range(1500):
        a, b = rng.integers(0, 120, 2)
        if a != b and (blocks[a] == blocks[b] or rng.random() < 0.05):
            edges.append((f"n{a}", f"n{b}"))
    edges = sorted(set(tuple(sorted(e)) for e in edges))
    order = rng.permutation(len(edges))
    test = [edges[i] for i in order[:60]]
    train = [edges[i] for i in order[60:]]
    graph = SparseMatrix.from_iterator(iter(f"{a} {b}" for a, b in train), "complex::reflexive::n")
    return graph, embed(graph, 32, 4, whiten=False), train, test, blocks


def test_link_prediction_scores_are_consistent(small_graph):
    graph, vectors, train, test, _ = small_graph
    ids = list(graph.entity_ids)
    rng = np.random.default_rng(1)
    negatives = [(ids[a], ids[b]) for a, b in rng.integers(0, len(ids), (400, 2)) if a != b]
    result = metrics.link_prediction_scores(graph, vectors, test, negative_edges=negatives)

    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    row = {name: i for i, name in enumerate(ids)}
    score = lambda pairs: np.array([unit[row[a]] @ unit[row[b]] for a, b in pairs if a in row and b in row], dtype=np.float64)
    positive, negative = score(test), score(negatives)
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    both = np.r_[positive, negative]
    assert result["auc"] == pytest.approx(sklearn_metrics.roc_auc_score(labels, both), abs=1e-9)
    assert result["average_precision"] == pytest.approx(sklearn_metrics.average_precision_score(labels, both), abs=1e-9)
    ranks = np.array([1 + (negative > p).sum() + 0.5 * (negative == p).sum() for p in positive])
    assert result["mrr"] == pytest.approx((1 / ranks).mean())
    assert result["hits@10"] == pytest.approx((ranks <= 10).mean())
    assert result["auc"] > 0.6  # clearly better than chance on a block graph

    # sampled negatives are never edges of the graph or self-pairs: on a complete graph none exist
    complete = SparseMatrix.from_iterator(iter(["a b", "a c", "b c"]), "complex::reflexive::n")
    with pytest.raises(ValueError):
        metrics.link_prediction_scores(complete, embed(complete, 4, 1, whiten=False), [("a", "b")])


def test_map_and_ndcg_match_brute_force(small_graph):
    graph, vectors, train, test, _ = small_graph
    ids = list(graph.entity_ids)
    row = {name: i for i, name in enumerate(ids)}
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    neighbours = {i: set() for i in range(len(ids))}
    for a, b in train:
        neighbours[row[a]].add(row[b])
        neighbours[row[b]].add(row[a])
    targets = {}
    for a, b in test:
        if a in row and b in row:
            targets.setdefault(row[a], set()).add(row[b])

    k = 10
    mean_ap, ndcg = [], []
    for source, wanted in targets.items():
        similarity = unit @ unit[source]
        candidates = [j for j in np.argsort(-similarity, kind="stable")
                      if j != source and (j not in neighbours[source] or j in wanted)][:k]
        found, total, dcg = 0, 0.0, 0.0
        for position, j in enumerate(candidates):
            if j in wanted:
                found += 1
                total += found / (position + 1)
                dcg += 1 / np.log2(position + 2)
        mean_ap.append(total / min(len(wanted), k))
        ndcg.append(dcg / sum(1 / np.log2(r + 2) for r in range(min(len(wanted), k))))
    assert metrics.map_at_k(graph, vectors, test, k=k) == pytest.approx(np.mean(mean_ap), abs=1e-6)
    assert metrics.ndcg_at_k(graph, vectors, test, k=k) == pytest.approx(np.mean(ndcg), abs=1e-6)


def test_node_classification_and_cross_validation(small_graph):
    graph, vectors, _, _, blocks = small_graph
    labels = {name: int(blocks[int(name[1:])]) for name in graph.entity_ids}
    scores = metrics.node_classification_scores(graph, vectors, labels, train_ratio=0.5, seed=1)
    assert scores["accuracy"] > 0.9 and scores["macro_f1"] > 0.9 and scores["weighted_f1"] > 0.9
    assert scores["train_size"] + scores["test_size"] == len(labels)
    folds = metrics.cross_validate(graph, vectors, labels, k_folds=4)
    assert folds["mean_accuracy"] > 0.9 and len(folds["fold_accuracies"]) == 4
