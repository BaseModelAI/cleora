import numpy as np
from typing import Dict, Optional, Tuple, List


def label_propagation(
    graph,
    labels: Dict[str, int],
    num_iterations: int = 30,
    alpha: float = 0.5,
) -> Dict[str, int]:
    from scipy.sparse import csr_matrix, diags

    rows, cols, vals, n, _ = graph.to_sparse_csr()
    A = csr_matrix(
        (vals.astype(np.float64), (rows.astype(np.int32), cols.astype(np.int32))),
        shape=(n, n),
    )

    degrees = np.array(A.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-10)
    D_inv = diags(1.0 / degrees)
    S = D_inv @ A

    if not labels:
        raise ValueError("labels must be a non-empty dict")

    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    classes = sorted(set(labels.values()))
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    Y = np.zeros((n, num_classes), dtype=np.float64)
    labeled_mask = np.zeros(n, dtype=bool)

    for eid, label in labels.items():
        idx = index_map.get(eid)
        if idx is not None:
            Y[idx, class_to_idx[label]] = 1.0
            labeled_mask[idx] = True

    Y_fixed = Y.copy()
    F = Y.copy()

    for _ in range(num_iterations):
        F = alpha * (S @ F) + (1 - alpha) * Y_fixed
        F[labeled_mask] = Y_fixed[labeled_mask]

    predictions = {}
    for i, eid in enumerate(graph.entity_ids):
        pred_class_idx = int(np.argmax(F[i]))
        predictions[eid] = classes[pred_class_idx]

    return predictions


def mlp_classify(
    graph,
    embeddings: np.ndarray,
    labels: Dict[str, int],
    hidden_dim: int = 64,
    learning_rate: float = 0.01,
    num_epochs: int = 200,
    train_ratio: float = 0.8,
    seed: int = 42,
    l2_reg: float = 1e-4,
) -> Dict[str, float]:
    if not labels:
        raise ValueError("labels must be a non-empty dict")
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}

    indices = []
    y_list = []
    for eid, label in labels.items():
        idx = index_map.get(eid)
        if idx is not None:
            indices.append(idx)
            y_list.append(label)

    if len(indices) < 4:
        raise ValueError(f"Need at least 4 labeled entities, got {len(indices)}")

    X = embeddings[indices].astype(np.float64)
    y = np.array(y_list)
    classes = np.unique(y)
    num_classes = len(classes)
    class_map = {c: i for i, c in enumerate(classes)}
    y_mapped = np.array([class_map[c] for c in y])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    split = int(len(y) * train_ratio)
    train_idx, test_idx = perm[:split], perm[split:]

    if len(test_idx) == 0:
        raise ValueError("Test set is empty, reduce train_ratio")

    X_train, y_train = X[train_idx], y_mapped[train_idx]
    X_test, y_test = X[test_idx], y_mapped[test_idx]

    input_dim = X.shape[1]
    scale1 = np.sqrt(2.0 / input_dim)
    scale2 = np.sqrt(2.0 / hidden_dim)
    W1 = rng.standard_normal((input_dim, hidden_dim)) * scale1
    b1 = np.zeros(hidden_dim)
    W2 = rng.standard_normal((hidden_dim, num_classes)) * scale2
    b2 = np.zeros(num_classes)

    def relu(x):
        return np.maximum(x, 0)

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)

    def forward(X_batch):
        z1 = X_batch @ W1 + b1
        h1 = relu(z1)
        z2 = h1 @ W2 + b2
        probs = softmax(z2)
        return z1, h1, z2, probs

    best_acc = 0.0
    best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()

    batch_size = min(256, len(X_train))

    for epoch in range(num_epochs):
        perm_train = rng.permutation(len(X_train))
        for start in range(0, len(X_train), batch_size):
            batch_idx = perm_train[start:start + batch_size]
            X_b = X_train[batch_idx]
            y_b = y_train[batch_idx]
            bs = len(X_b)

            z1, h1, z2, probs = forward(X_b)

            one_hot = np.zeros((bs, num_classes))
            one_hot[np.arange(bs), y_b] = 1.0
            dz2 = (probs - one_hot) / bs

            dW2 = h1.T @ dz2 + l2_reg * W2
            db2 = np.sum(dz2, axis=0)

            dh1 = dz2 @ W2.T
            dz1 = dh1 * (z1 > 0).astype(np.float64)

            dW1 = X_b.T @ dz1 + l2_reg * W1
            db1 = np.sum(dz1, axis=0)

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            _, _, _, test_probs = forward(X_test)
            test_pred = np.argmax(test_probs, axis=1)
            acc = float(np.mean(test_pred == y_test))
            if acc > best_acc:
                best_acc = acc
                best_W1, best_b1 = W1.copy(), b1.copy()
                best_W2, best_b2 = W2.copy(), b2.copy()

    W1, b1, W2, b2 = best_W1, best_b1, best_W2, best_b2
    _, _, _, test_probs = forward(X_test)
    y_pred = np.argmax(test_probs, axis=1)
    accuracy = float(np.mean(y_pred == y_test))

    per_class_f1 = []
    for c_idx in range(num_classes):
        tp = np.sum((y_pred == c_idx) & (y_test == c_idx))
        fp = np.sum((y_pred == c_idx) & (y_test != c_idx))
        fn = np.sum((y_pred != c_idx) & (y_test == c_idx))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_classes": num_classes,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "num_epochs": num_epochs,
        "hidden_dim": hidden_dim,
    }


def label_propagation_predict(
    graph,
    embeddings: np.ndarray,
    labels: Dict[str, int],
    num_iterations: int = 30,
    alpha: float = 0.5,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, float]:
    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}

    labeled_entities = [eid for eid in labels if eid in index_map]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labeled_entities))
    split = int(len(labeled_entities) * train_ratio)

    train_labels = {}
    test_labels = {}
    for i in perm[:split]:
        eid = labeled_entities[i]
        train_labels[eid] = labels[eid]
    for i in perm[split:]:
        eid = labeled_entities[i]
        test_labels[eid] = labels[eid]

    predictions = label_propagation(graph, train_labels, num_iterations, alpha)

    correct = 0
    total = 0
    for eid, true_label in test_labels.items():
        pred_label = predictions.get(eid)
        if pred_label is not None:
            total += 1
            if pred_label == true_label:
                correct += 1

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "train_size": len(train_labels),
        "test_size": len(test_labels),
        "total_predictions": len(predictions),
    }


def gcn_classify(
    graph,
    embeddings: np.ndarray,
    labels: Dict[str, int],
    hidden_dim: int = 64,
    learning_rate: float = 0.01,
    num_epochs: int = 200,
    train_ratio: float = 0.8,
    seed: int = 42,
    l2_reg: float = 1e-4,
    num_layers: int = 2,
    dropout: float = 0.5,
) -> Dict[str, float]:
    from scipy.sparse import csr_matrix, diags, eye

    if not labels:
        raise ValueError("labels must be a non-empty dict")
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    index_map = {eid: i for i, eid in enumerate(graph.entity_ids)}
    n = graph.num_entities

    rows, cols, vals, _, _ = graph.to_sparse_csr()
    A = csr_matrix(
        (vals.astype(np.float64), (rows.astype(np.int32), cols.astype(np.int32))),
        shape=(n, n),
    )
    A_hat = A + eye(n, format="csr")
    degrees = np.array(A_hat.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    indices = []
    y_list = []
    for eid, label in labels.items():
        idx = index_map.get(eid)
        if idx is not None:
            indices.append(idx)
            y_list.append(label)

    if len(indices) < 4:
        raise ValueError(f"Need at least 4 labeled entities, got {len(indices)}")

    y = np.array(y_list)
    classes = np.unique(y)
    num_classes = len(classes)
    class_map = {c: i for i, c in enumerate(classes)}
    y_mapped = np.array([class_map[c] for c in y])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    split = int(len(y) * train_ratio)
    train_idx, test_idx = perm[:split], perm[split:]

    if len(test_idx) == 0:
        raise ValueError("Test set is empty, reduce train_ratio")

    node_indices = np.array(indices)
    train_nodes = node_indices[train_idx]
    test_nodes = node_indices[test_idx]
    y_train = y_mapped[train_idx]
    y_test = y_mapped[test_idx]

    X = embeddings.astype(np.float64)
    input_dim = X.shape[1]

    dims = [input_dim]
    for _ in range(num_layers - 1):
        dims.append(hidden_dim)
    dims.append(num_classes)

    weights = []
    for i in range(len(dims) - 1):
        scale = np.sqrt(2.0 / dims[i])
        W = rng.standard_normal((dims[i], dims[i + 1])) * scale
        weights.append(W)

    def relu(x):
        return np.maximum(x, 0)

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)

    def forward(X_in, training=False):
        H = X_in
        activations = [H]
        pre_activations = []
        for layer_idx, W in enumerate(weights):
            H = A_norm @ H
            Z = H @ W
            pre_activations.append(Z)
            if layer_idx < len(weights) - 1:
                H = relu(Z)
                if training and dropout > 0:
                    mask = (rng.random(H.shape) > dropout).astype(np.float64) / (1 - dropout)
                    H = H * mask
            else:
                H = softmax(Z)
            activations.append(H)
        return activations, pre_activations

    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_nodes] = True
    y_full = np.zeros(n, dtype=np.int64)
    for i, ni in enumerate(train_nodes):
        y_full[ni] = y_train[i]

    best_acc = 0.0
    best_weights = [w.copy() for w in weights]

    for epoch in range(num_epochs):
        activations, pre_activations = forward(X, training=True)
        output = activations[-1]

        grad_output = np.zeros_like(output)
        one_hot = np.zeros((n, num_classes))
        for i, ni in enumerate(train_nodes):
            one_hot[ni, y_train[i]] = 1.0
        grad_output = (output - one_hot) / len(train_nodes)
        grad_output[~train_mask] = 0.0

        for layer_idx in range(len(weights) - 1, -1, -1):
            H_prev = activations[layer_idx]
            H_prop = A_norm @ H_prev

            dW = H_prop.T @ grad_output + l2_reg * weights[layer_idx]
            weights[layer_idx] -= learning_rate * dW

            if layer_idx > 0:
                grad_H = grad_output @ weights[layer_idx].T
                grad_H = A_norm.T @ grad_H
                grad_H = grad_H * (pre_activations[layer_idx - 1] > 0).astype(np.float64)
                grad_output = grad_H

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            activations, _ = forward(X, training=False)
            preds = np.argmax(activations[-1][test_nodes], axis=1)
            acc = float(np.mean(preds == y_test))
            if acc > best_acc:
                best_acc = acc
                best_weights = [w.copy() for w in weights]

    weights = best_weights
    activations, _ = forward(X, training=False)
    y_pred = np.argmax(activations[-1][test_nodes], axis=1)
    accuracy = float(np.mean(y_pred == y_test))

    per_class_f1 = []
    for c_idx in range(num_classes):
        tp = np.sum((y_pred == c_idx) & (y_test == c_idx))
        fp = np.sum((y_pred == c_idx) & (y_test != c_idx))
        fn = np.sum((y_pred != c_idx) & (y_test == c_idx))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_classes": num_classes,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
    }
