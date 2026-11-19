"""Evaluation tasks: what is predicted from the vectors, on which split, against which
candidates.

Concern of this module: the experimental protocol. It draws the splits and the negative
candidates (the same for every method), calls the probes and hands their scores to
``pycleora.metrics``. It does not know how the vectors were made.

Node classification (graphs with node labels)
    The vectors come from the whole graph (labels are never used to make them). Labelled
    nodes are split into train / validation / test - the dataset's official split when it
    has one, otherwise ``NODE_SPLIT`` stratified random splits. Every probe is fitted on
    train, tuned on validation and scored once on test.

Link prediction and similarity ranking (every graph)
    Edges are split at random into four disjoint sets (``LINK_SPLIT``):

    * ``embed``  - the only edges the embedding method sees;
    * ``probe``  - positives for training the pair probes. They are *not* in the embedding
                   graph: a probe trained on edges the vectors were fitted to would learn
                   to recognise memorised edges, not to predict missing ones;
    * ``valid``  - positives for model selection;
    * ``test``   - positives for the reported numbers.

    Every validation / test positive ``(u, v)`` is ranked, in both directions, against
    ``NEGATIVES_PER_POSITIVE`` random nodes that are not neighbours of the query (in
    the full graph). All methods and all probes are ranked against the same candidates.
    Training-free scores (dot product, cosine) are additionally ranked against *all*
    nodes ("retrieval"), which is what nearest-neighbour search over the vectors does.

    Datasets with an official link split (OGB) keep their official positives and
    negatives; only the probe positives are taken out of the official training edges.
    Retrieval among all nodes is measured on their official test positives as well.
"""

import numpy as np

from pycleora import metrics

from . import probes

NODE_SPLIT = {"train": 0.10, "valid": 0.10}          # the rest is test
LINK_SPLIT = {"probe": 0.10, "valid": 0.05, "test": 0.10}  # the rest is embed
LINK_CAPS = {"probe": 500_000, "valid": 20_000, "test": 50_000}  # positives, on large graphs
NEGATIVES_PER_POSITIVE = 500
PROBE_NEGATIVES_PER_POSITIVE = 3
SELECTION_ROWS, SELECTION_NEGATIVES = 1000, 100  # part of validation the trained pair probe is selected on
# Training pairs of the trained pair probe: their features (pairs x dimensions) are held in
# memory at once, and a linear model in the dimensions of the embedding needs no more.
PAIR_LINEAR_ROWS = 1_000_000
RETRIEVAL_QUERIES = 10_000
RETRIEVAL_DEPTH = 50
NODE_PROBES = ("logistic", "mlp")
PAIR_PROBES = ("dot", "cosine", "logistic")


# ----------------------------------------------------------------------------- graph helpers


def neighbour_lists(edges, num_nodes):
    """CSR adjacency (``indptr``, ``indices``; neighbours sorted) of an undirected edge list."""
    sources = np.concatenate([edges[:, 0], edges[:, 1]])
    targets = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.lexsort((targets, sources))
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(indptr, sources + 1, 1)
    return np.cumsum(indptr), targets[order]


def edge_keys(edges, num_nodes):
    """Sorted keys ``u * num_nodes + v`` of both directions of every edge."""
    keys = np.concatenate([edges[:, 0] * num_nodes + edges[:, 1], edges[:, 1] * num_nodes + edges[:, 0]])
    keys.sort()
    return keys


def _is_edge(keys, num_nodes, sources, targets):
    """Whether ``(sources[i], targets[i])`` is an edge, by binary search in ``edge_keys``."""
    wanted = sources.astype(np.int64) * num_nodes + targets
    position = np.minimum(np.searchsorted(keys, wanted), len(keys) - 1)
    return keys[position] == wanted


def compact(edges, num_nodes):
    """The edge list restricted to nodes that have an edge: ``(edges with new ids, ids)``,
    where ``ids[new] = old``. Embedding methods are only ever given such graphs."""
    present = np.zeros(num_nodes, dtype=bool)
    present[edges.ravel()] = True
    new_id = np.cumsum(present) - 1
    return new_id[edges], np.flatnonzero(present)


def expand(vectors, ids, num_nodes):
    """Vectors of a compacted graph back in the original numbering (zero rows for the
    nodes that had no edge)."""
    out = np.zeros((num_nodes, vectors.shape[1]), dtype=np.float32)
    out[ids] = vectors
    return out


# ----------------------------------------------------------------------------- node classification


def node_splits(graph, seeds):
    """Yields ``(train, valid, test)`` index arrays: the official split once, or one
    stratified random split per seed."""
    if "split_train" in graph:
        yield graph["split_train"], graph["split_valid"], graph["split_test"]
        return
    labels = graph["labels"]
    multilabel = labels.ndim == 2
    labelled = np.flatnonzero(labels.any(axis=1) if multilabel else labels >= 0)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        groups = [labelled] if multilabel else [labelled[labels[labelled] == c] for c in np.unique(labels[labelled])]
        parts = ([], [], [])
        for group in groups:
            group = rng.permutation(group)
            # every class keeps at least one training example
            train_end = max(1, int(round(NODE_SPLIT["train"] * len(group))))
            valid_end = train_end + int(round(NODE_SPLIT["valid"] * len(group)))
            for part, rows in zip(parts, (group[:train_end], group[train_end:valid_end], group[valid_end:])):
                part.append(rows)
        yield tuple(np.concatenate(part) for part in parts)


def _node_metrics(labels, scores):
    if labels.ndim == 2:
        return metrics.multilabel_metrics(labels, scores)
    return metrics.classification_metrics(labels, scores)


def node_classification(vectors, graph, device, seeds, which=NODE_PROBES):
    """Fits every node probe on every split. Returns
    ``{probe: {"metrics": [one dict per split], "chosen": [hyper-parameters per split]}}``."""
    labels = graph["labels"]
    multilabel = labels.ndim == 2
    num_classes = labels.shape[1] if multilabel else int(labels.max()) + 1
    results = {probe: {"metrics": [], "chosen": []} for probe in which}
    linear_device = probes.device_for(vectors.size, device)

    for split_number, (train, valid, test) in enumerate(node_splits(graph, seeds)):
        if not multilabel:  # official splits may contain unlabelled nodes
            train, valid, test = (rows[labels[rows] >= 0] for rows in (train, valid, test))
        y_train, y_valid, y_test = labels[train], labels[valid], labels[test]
        validate_on = lambda features: (lambda predict: probes.selection_score(predict(features), y_valid))

        def scaled(scaling):
            scale = probes.scaler(scaling, vectors[train])
            return scale(vectors[train]), scale(vectors[valid]), scale(vectors[test])

        best = {}
        if "logistic" in which:
            # scaling and decay chosen together on validation
            for scaling in probes.SCALINGS:
                x_train, x_valid, x_test = scaled(scaling)
                predict, score, info = probes.fit_logistic(x_train, y_train, num_classes, linear_device,
                                                           validate_on(x_valid))
                if "logistic" not in best or score > best["logistic"][0]:
                    best["logistic"] = (score, predict(x_test), dict(info, scaling=scaling, validation_score=score))
        if "mlp" in which:
            # the scaling the logistic probe preferred (standardisation if it was not run)
            scaling = best["logistic"][2]["scaling"] if "logistic" in best else "standard"
            x_train, x_valid, x_test = scaled(scaling)
            predict, score, info = probes.fit_mlp(x_train, y_train, num_classes, device, split_number,
                                                  validate_on(x_valid))
            best["mlp"] = (score, predict(x_test), dict(info, scaling=scaling, validation_score=score))
        for probe in which:
            results[probe]["metrics"].append(_node_metrics(y_test, best[probe][1]))
            results[probe]["chosen"].append({k: v for k, v in best[probe][2].items() if k != "validation"})
    return results


# ----------------------------------------------------------------------------- link prediction: the split


def _sample_negative_targets(rng, sources, keys, num_nodes, count, low, high):
    """``count`` random nodes from ``[low[i], high[i])`` per source that are neither the
    source nor one of its neighbours (-1 where none could be drawn in a few attempts)."""
    span = (high - low)[:, None]
    negatives = low[:, None] + (rng.random((len(sources), count)) * span).astype(np.int64)
    for _ in range(8):
        flat_sources = np.repeat(sources, count)
        bad = (_is_edge(keys, num_nodes, flat_sources, negatives.ravel()) |
               (negatives.ravel() == flat_sources)).reshape(negatives.shape)
        if not bad.any():
            break
        redraw = low[:, None] + (rng.random(negatives.shape) * span).astype(np.int64)
        negatives = np.where(bad, redraw, negatives)
    else:
        negatives[bad] = -1
    return negatives.astype(np.int32)


def _both_directions(pairs):
    """``(sources, targets)`` of every pair read in both directions."""
    return np.concatenate([pairs[:, 0], pairs[:, 1]]), np.concatenate([pairs[:, 1], pairs[:, 0]])


def _candidate_ranges(graph, targets):
    """The id range negatives for these targets are drawn from: all nodes, or - in a
    bipartite graph - the side the true target is on."""
    num_nodes = graph["meta"]["num_nodes"]
    if "num_left" not in graph:
        return np.zeros(len(targets), dtype=np.int64), np.full(len(targets), num_nodes, dtype=np.int64)
    num_left = int(graph["num_left"])
    on_left = targets < num_left
    return np.where(on_left, 0, num_left), np.where(on_left, num_left, num_nodes)


def link_split(graph, seed):
    """Draws the link-prediction split of a graph (deterministic in ``seed``).

    Returns a dict of arrays: ``embed`` (edges for the embedding method), ``probe_pairs``
    with ``probe_labels`` (training pairs of the pair probes), and for ``valid`` / ``test``
    either ``{part}_queries`` + ``{part}_targets`` + ``{part}_negatives`` (one row per
    ranking problem) or, for official splits with one shared list of negative pairs,
    ``{part}_pos`` + ``{part}_neg``.
    """
    rng = np.random.default_rng(seed)
    num_nodes = graph["meta"]["num_nodes"]
    official = graph["meta"]["official_link_split"]
    edges = graph["edges"]
    order = rng.permutation(len(edges))

    def take(name, pool_size):
        nonlocal order
        size = min(int(round(LINK_SPLIT[name] * pool_size)), LINK_CAPS[name])
        chosen, order = order[:size], order[size:]
        return edges[chosen]

    split = {"probe": take("probe", len(edges))}
    held_out = [split["probe"]]
    if official:
        # all true edges, for drawing negatives of the probe: train + official positives
        known = [edges] + [graph[f"lp_{part}_pos"] for part in ("valid", "test")]
    else:
        split["valid"], split["test"] = take("valid", len(edges)), take("test", len(edges))
        held_out += [split["valid"], split["test"]]
        known = [edges]
    split["embed"] = edges[order]
    keys = edge_keys(np.concatenate(known), num_nodes)

    # Nodes without an edge in the embedding graph get no vector: pairs that touch one
    # cannot be ranked by any method and are left out (the same pairs for every method).
    embedded = np.zeros(num_nodes, dtype=bool)
    embedded[split["embed"].ravel()] = True
    usable = lambda pairs: pairs[embedded[pairs].all(axis=1)]

    result = {"embed": split["embed"], "embedded": embedded}

    probe = usable(split["probe"])
    sources, targets = _both_directions(probe)
    low, high = _candidate_ranges(graph, targets)
    negatives = _sample_negative_targets(rng, sources, keys, num_nodes, PROBE_NEGATIVES_PER_POSITIVE, low, high)
    negative_sources = np.repeat(sources, PROBE_NEGATIVES_PER_POSITIVE)
    keep = (negatives.ravel() >= 0) & embedded[np.maximum(negatives.ravel(), 0)]
    result["probe_pairs"] = np.concatenate([
        np.stack([sources, targets], axis=1),
        np.stack([negative_sources[keep], negatives.ravel()[keep].astype(np.int64)], axis=1)])
    result["probe_labels"] = np.concatenate([np.ones(len(sources), np.float32), np.zeros(int(keep.sum()), np.float32)])

    for part in ("valid", "test"):
        if official and f"lp_{part}_neg" in graph:
            result[f"{part}_pos"] = usable(graph[f"lp_{part}_pos"])
            result[f"{part}_neg"] = usable(graph[f"lp_{part}_neg"])
            continue
        if official:  # ogbl-citation2: the official negatives of every (source, target)
            positives = graph[f"lp_{part}_pos"]
            keep = embedded[positives].all(axis=1)
            if keep.sum() > LINK_CAPS[part]:
                keep &= np.isin(np.arange(len(keep)), rng.choice(np.flatnonzero(keep), LINK_CAPS[part], replace=False))
            negatives = graph[f"lp_{part}_target_neg"][keep].astype(np.int64)
            negatives[~embedded[np.maximum(negatives, 0)]] = -1
            result[f"{part}_queries"], result[f"{part}_targets"] = positives[keep, 0], positives[keep, 1]
            result[f"{part}_negatives"] = negatives.astype(np.int32)
            continue
        sources, targets = _both_directions(usable(split[part]))
        low, high = _candidate_ranges(graph, targets)
        negatives = _sample_negative_targets(rng, sources, keys, num_nodes, NEGATIVES_PER_POSITIVE, low, high)
        negatives[~embedded[np.maximum(negatives, 0)]] = -1
        result[f"{part}_queries"], result[f"{part}_targets"], result[f"{part}_negatives"] = sources, targets, negatives
    return result


# ----------------------------------------------------------------------------- link prediction: the evaluation


def _candidate_ranks(score, queries, targets, negatives, rows=4096):
    """Rank of every target among its own negatives (absent ones, -1, are ignored)."""
    ranks = []
    for start in range(0, len(queries), rows):
        block = negatives[start:start + rows].astype(np.int64)
        block_queries = queries[start:start + rows]
        candidate_scores = score(np.repeat(block_queries, block.shape[1]), np.maximum(block, 0).ravel())
        candidate_scores = candidate_scores.astype(np.float64).reshape(block.shape)
        candidate_scores[block < 0] = np.nan
        ranks.append(metrics.ranks_among(score(block_queries, targets[start:start + rows]), candidate_scores))
    return np.concatenate(ranks) if ranks else np.zeros(0)


def _shared_negative_ranks(positive_scores, negative_scores):
    """Rank of every positive pair among one shared list of negative pairs."""
    ordered = np.sort(negative_scores.astype(np.float64))
    right = np.searchsorted(ordered, positive_scores, side="right")
    left = np.searchsorted(ordered, positive_scores, side="left")
    return 1.0 + (len(ordered) - right) + 0.5 * (right - left)


def _evaluate_scorer(score, split, part, limit=None):
    """Ranking and binary metrics of a pair scorer on ``part`` ('valid' or 'test')."""
    if f"{part}_pos" in split:
        positives, negatives = split[f"{part}_pos"], split[f"{part}_neg"]
        if limit is not None:  # an evenly spaced subset
            thin = lambda pairs, size: pairs[np.linspace(0, len(pairs) - 1, min(size, len(pairs))).astype(np.int64)]
            positives, negatives = thin(positives, limit[0]), thin(negatives, limit[0] * limit[1])
        positive_scores = score(positives[:, 0], positives[:, 1]).astype(np.float64)
        negative_scores = score(negatives[:, 0], negatives[:, 1]).astype(np.float64)
        ranks = _shared_negative_ranks(positive_scores, negative_scores)
        result = metrics.ranking_metrics(ranks, ks=(1, 10, 20, 50, 100))
    else:
        queries, targets, negatives = (split[f"{part}_{name}"] for name in ("queries", "targets", "negatives"))
        if limit is not None:
            queries, targets, negatives = queries[:limit[0]], targets[:limit[0]], negatives[:limit[0], :limit[1]]
        ranks = _candidate_ranks(score, queries, targets, negatives)
        result = metrics.ranking_metrics(ranks, ks=(1, 3, 10, 50, 100))
        # binary view: every positive against its first existing negative
        first = negatives[:, 0].astype(np.int64)
        exists = first >= 0
        positive_scores = score(queries[exists], targets[exists]).astype(np.float64)
        negative_scores = score(queries[exists], first[exists]).astype(np.float64)
    labels = np.concatenate([np.ones(len(positive_scores), bool), np.zeros(len(negative_scores), bool)])
    both = np.concatenate([positive_scores, negative_scores])
    result["roc_auc"] = metrics.roc_auc(labels, both)
    result["average_precision"] = metrics.average_precision(labels, both)
    result["num_ranked"] = int(len(ranks))
    return result


def _retrieval(vectors_on_device, split, graph, seed):
    """Ranks the test targets of sampled query nodes against *all* nodes of the graph
    with a training-free score (the rows of ``vectors_on_device`` are compared by dot
    product). Known neighbours other than test targets are removed from the candidates.

    The queries are those of the candidate ranking; a split that comes as one official
    list of positive pairs has every distinct pair queried from both of its ends.

    Returns MRR / Hits@k of the individual test edges and Recall / NDCG / MAP of the
    top-``RETRIEVAL_DEPTH`` list of every query node.
    """
    import torch

    num_nodes = graph["meta"]["num_nodes"]
    if "test_queries" in split:
        queries_all, targets_all = split["test_queries"], split["test_targets"]
    else:
        queries_all, targets_all = np.unique(np.stack(_both_directions(split["test_pos"]), axis=1), axis=0).T
    rng = np.random.default_rng(seed)
    nodes = np.unique(queries_all)
    if len(nodes) > RETRIEVAL_QUERIES:
        nodes = np.sort(rng.choice(nodes, RETRIEVAL_QUERIES, replace=False))
    selected = np.isin(queries_all, nodes)
    queries, targets = queries_all[selected], targets_all[selected]
    known = neighbour_lists(graph["edges"], num_nodes) if not graph["meta"]["official_link_split"] else \
        neighbour_lists(np.concatenate([graph["edges"], graph["lp_valid_pos"], graph["lp_test_pos"]]), num_nodes)
    indptr, indices = known
    device = vectors_on_device.device
    embedded = torch.as_tensor(split["embedded"], device=device)
    side = None
    if "num_left" in graph:
        side = torch.arange(num_nodes, device=device) < int(graph["num_left"])

    order = np.argsort(queries, kind="stable")
    queries, targets = queries[order], targets[order]
    starts = np.flatnonzero(np.concatenate([[True], queries[1:] != queries[:-1]]))
    ends = np.concatenate([starts[1:], [len(queries)]])
    ranks = np.zeros(len(queries))
    hits = np.zeros((len(starts), RETRIEVAL_DEPTH), dtype=bool)
    for first in range(0, len(starts), 256):
        group = slice(first, min(first + 256, len(starts)))
        nodes_here = queries[starts[group]]
        scores = vectors_on_device[torch.as_tensor(nodes_here, device=device)] @ vectors_on_device.T
        scores[:, ~embedded] = -float("inf")
        for local, node in enumerate(nodes_here):
            relevant = torch.as_tensor(targets[starts[first + local]:ends[first + local]], device=device)
            relevant_scores = scores[local, relevant].clone()
            scores[local, torch.as_tensor(indices[indptr[node]:indptr[node + 1]], device=device)] = -float("inf")
            scores[local, node] = -float("inf")
            if side is not None:  # only the other side of a bipartite graph can be recommended
                scores[local, side == side[node]] = -float("inf")
            # rank of every test target among the non-neighbours (ties: half a place)
            above = (scores[local][None, :] > relevant_scores[:, None]).sum(dim=1)
            equal = (scores[local][None, :] == relevant_scores[:, None]).sum(dim=1)
            ranks[starts[first + local]:ends[first + local]] = (1 + above + 0.5 * equal).cpu().numpy()
            scores[local, relevant] = relevant_scores
            top = scores[local].topk(RETRIEVAL_DEPTH).indices
            hits[first + local] = torch.isin(top, relevant).cpu().numpy()
    result = {f"all_nodes_{name}": value for name, value in metrics.ranking_metrics(ranks, ks=(1, 10, 100)).items()}
    result.update(metrics.retrieval_metrics(hits, ends - starts, ks=(10, 20, 50)))
    result["num_queries"] = int(len(starts))
    return result


def link_prediction(vectors, split, graph, device, seed, which=PAIR_PROBES):
    """Scores every pair probe on the test part of a link split. Returns
    ``{probe: {"metrics": {...}, "chosen": {...}}}``; the training-free probes also carry
    the all-nodes retrieval metrics."""
    # Scoring candidates means millions of vector products: everything in this task runs
    # on the accelerator when there is one.
    unit = probes.unit_vectors(vectors, device)
    selection = (SELECTION_ROWS, SELECTION_NEGATIVES)
    validate = lambda score: _evaluate_scorer(score, split, "valid", limit=selection)["mrr"]
    pairs, labels = split["probe_pairs"], split["probe_labels"]
    linear_rows = np.arange(len(pairs))
    if len(pairs) > PAIR_LINEAR_ROWS:
        linear_rows = np.sort(np.random.default_rng(seed).choice(len(pairs), PAIR_LINEAR_ROWS, replace=False))
    results = {}
    for probe in which:
        if probe in ("dot", "cosine"):
            matrix = probes.to_device(vectors, device) if probe == "dot" else unit
            score = probes.product_scorer(matrix)
            chosen = {"validation_mrr": _evaluate_scorer(score, split, "valid")["mrr"]}
            test = _evaluate_scorer(score, split, "test")
            test.update(_retrieval(matrix, split, graph, seed))
            del matrix, score
        elif probe == "logistic":
            score, validation, info = probes.fit_pair_logistic(unit, pairs[linear_rows], labels[linear_rows],
                                                               device, validate)
            chosen = {"decay": info["decay"], "validation_mrr": validation}
            test = _evaluate_scorer(score, split, "test")
        else:
            raise ValueError(f"Unknown pair probe '{probe}'")
        results[probe] = {"metrics": test, "chosen": chosen}
    return results
