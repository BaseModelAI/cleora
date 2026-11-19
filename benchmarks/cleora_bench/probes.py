"""Probes: small supervised models trained on frozen embeddings.

Concern of this module: fitting a probe and returning its scores. It knows nothing about
graphs, embedding methods or metrics.

Every probe sees the same inputs for every embedding method, and every hyper-parameter
is chosen on validation data only (the functions never receive test labels).

Node probes (one vector -> classes)
    ``logistic``  linear, logistic loss: multinomial (single-label) / one-vs-rest (multi-label)
    ``svm``       linear, squared hinge loss, one-vs-rest (the objective of LinearSVC)
    ``mlp``       one hidden layer, trained with AdamC
    ``knn``       cosine k-nearest-neighbours vote: no training at all

Pair probes (two vectors -> link score)
    ``dot`` / ``cosine``   no training
    ``bilinear``           x^T M y, M trained with AdamC
    ``mlp``                concatenation -> hidden layer -> sigmoid, trained with AdamC
    ``logistic`` / ``svm`` linear model on the element-wise product of the two vectors

Feature scaling. Embedding methods differ in what the scale of their coordinates means:
standardising every dimension helps some and destroys the spectrum of others (the
columns of an SVD-based embedding are ordered by importance). The scaling is therefore
a hyper-parameter of the probe, chosen on validation like the others, from ``SCALINGS``.

The linear probes minimise  mean_i loss_i + decay / 2 * ||W||^2  with full-batch L-BFGS,
which finds the optimum of these convex problems to high precision, so the result does
not depend on a learning-rate schedule. ``decay`` is chosen from ``DECAYS``.
"""

import math

import numpy as np
import torch

# strongest first: every fit starts from the solution of the previous (stronger) decay
DECAYS = (1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def pick_device():
    """The CUDA device to train the probes on - the least busy one, found with
    ``nvidia-smi`` so that no context is opened on the others (set
    ``CUDA_VISIBLE_DEVICES`` to choose yourself) - else Apple MPS, else the CPU."""
    import os
    import subprocess

    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1 or "CUDA_VISIBLE_DEVICES" in os.environ:
            return torch.device("cuda:0")
        try:
            listing = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30, check=True).stdout
            usage = [tuple(int(value) for value in line.split(",")) for line in listing.strip().splitlines()]
            return torch.device(f"cuda:{min(usage, key=lambda gpu: (gpu[1], gpu[2]))[0]}")
        except (OSError, subprocess.SubprocessError, ValueError):
            return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# A linear probe on a matrix with fewer elements than this is fitted on the CPU: its
# thousands of tiny L-BFGS steps are faster there than through a GPU. (The probes trained
# with AdamC multiply large matrices at every step and always use the GPU when there is one.)
SMALL_PROBLEM = 1 << 24


def device_for(elements, device):
    return torch.device("cpu") if elements <= SMALL_PROBLEM else device


SCALINGS = ("raw", "unit", "standard")


def scaler(kind, train_features):
    """Returns ``scale(features)`` for one of ``SCALINGS``:

    ``raw``       the vectors as the method produced them;
    ``unit``      every vector scaled to unit L2 norm;
    ``standard``  every dimension shifted and scaled to zero mean and unit variance,
                  with the statistics of the training rows.

    ``raw`` and ``unit`` are additionally multiplied by one number, chosen so that the
    training rows have a mean squared norm equal to the number of dimensions (as
    standardised features have). It changes nothing about the geometry, but it puts all
    three scalings - and all methods, whatever the magnitude of their vectors - on the
    same footing with respect to the regularisation grid.
    """
    def with_common_magnitude(transform):
        transformed = transform(train_features)
        mean_square = float(np.einsum("ij,ij->", transformed, transformed, dtype=np.float64)) / len(transformed)
        factor = np.float32(np.sqrt(transformed.shape[1] / max(mean_square, 1e-30)))
        return lambda features: transform(features) * factor

    if kind == "raw":
        return with_common_magnitude(lambda features: np.asarray(features, dtype=np.float32))
    if kind == "unit":
        return with_common_magnitude(
            lambda features: features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12))
    if kind == "standard":
        mean = train_features.mean(axis=0, dtype=np.float64)
        std = train_features.std(axis=0, dtype=np.float64)
        std[std < 1e-12] = 1.0
        mean, std = mean.astype(np.float32), std.astype(np.float32)
        return lambda features: (features - mean) / std
    raise ValueError(f"Unknown scaling '{kind}'. Use one of {SCALINGS}.")


def to_device(array, device, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)


def _targets(labels, num_classes, device):
    """Single-label targets as class indices, multi-label targets as a 0/1 matrix."""
    if labels.ndim == 2:
        return to_device(labels, device), True
    assert labels.min() >= 0 and labels.max() < num_classes
    return to_device(labels, device, torch.long), False


def _linear_loss(logits, targets, multilabel, kind):
    if kind == "logistic":
        if multilabel:
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / len(logits)
        return torch.nn.functional.cross_entropy(logits, targets)
    if kind == "svm":
        signs = targets * 2 - 1 if multilabel else torch.nn.functional.one_hot(targets, logits.shape[1]) * 2.0 - 1
        return torch.clamp(1 - signs * logits, min=0).pow(2).sum() / len(logits)
    raise ValueError(f"Unknown linear probe '{kind}'. Use 'logistic' or 'svm'.")


def selection_score(scores, labels):
    """What a node probe's hyper-parameters are selected by on the validation rows:
    accuracy (single-label) or micro-F1 at the model's own decision threshold (multi-label)."""
    if labels.ndim == 2:
        predicted = scores > 0
        true_positive = float((predicted & labels).sum())
        return 2 * true_positive / max(float(predicted.sum() + labels.sum()), 1.0)
    return float((scores.argmax(axis=1) == labels).mean())


def _batched(function, features, device, batch=1 << 16):
    out = []
    with torch.no_grad():
        for start in range(0, len(features), batch):
            out.append(function(to_device(features[start:start + batch], device)).float().cpu().numpy())
    return np.concatenate(out)


def fit_linear(train_x, train_y, kind, num_classes, device, validate, decays=DECAYS, max_iter=300, wrap=None):
    """Fits a linear probe (``kind``: 'logistic' or 'svm') for every decay and keeps the
    one for which ``validate(predict)`` is highest.

    ``predict(features) -> scores`` (numpy, one column per class); ``wrap(weight, bias)``
    replaces it by another function of the fitted parameters (used by the pair probes).
    Returns ``(predict, validation score, info)``.
    """
    x = train_x if torch.is_tensor(train_x) else to_device(train_x, device)
    y, multilabel = _targets(train_y, num_classes, device)
    weight = torch.zeros(num_classes, x.shape[1], device=device, requires_grad=True)
    bias = torch.zeros(num_classes, device=device, requires_grad=True)

    def predictor(w, b):
        if wrap is not None:
            return wrap(w, b)
        return lambda features: _batched(lambda block: block @ w.T + b, features, device)

    tried, best = {}, None
    for decay in decays:
        optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=max_iter, history_size=20,
                                      tolerance_grad=1e-6, tolerance_change=1e-10, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = _linear_loss(x @ weight.T + bias, y, multilabel, kind) + 0.5 * decay * weight.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        candidate = predictor(weight.detach().clone(), bias.detach().clone())
        tried[decay] = validate(candidate)
        if best is None or tried[decay] > best[0]:
            best = (tried[decay], decay, candidate)
    return best[2], best[0], {"decay": best[1], "validation": tried}


def _adamc(parameters, peak_lr, weight_decay, total_steps, warmup=0.05):
    """AdamC (Defazio, 2025: AdamW whose decay is scaled by lr / peak lr) from timm, with
    linear warm-up and cosine decay of the learning rate."""
    from timm.optim import create_optimizer_v2

    optimizer = create_optimizer_v2(parameters, opt="adamc", lr=peak_lr, weight_decay=weight_decay)
    warmup_steps = max(1, int(warmup * total_steps))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _train_with_adamc(make_model, step_loss, num_examples, validate_model, seed, epochs, batch_size, peak_lr,
                      weight_decays, device):
    """Shared training loop of the AdamC probes: for every weight decay, mini-batch
    training with a warm-up + cosine schedule; the model state of the best validation
    epoch over all decays is returned (with its score and what was tried)."""
    # small training sets get small batches: at least ~32 steps per epoch
    batch_size = int(min(batch_size, max(32, num_examples // 32)))
    steps_per_epoch = max(1, math.ceil(num_examples / batch_size))
    best, tried = None, {}
    for weight_decay in weight_decays:
        torch.manual_seed(seed)
        model = make_model().to(device)
        optimizer, scheduler = _adamc(model.parameters(), peak_lr, weight_decay, epochs * steps_per_epoch)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        best_here = None
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(num_examples, generator=generator).to(device)
            for start in range(0, num_examples, batch_size):
                loss = step_loss(model, order[start:start + batch_size])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            model.eval()
            score = validate_model(model)
            if best_here is None or score > best_here[0]:
                best_here = (score, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
        tried[weight_decay] = {"validation": best_here[0], "epoch": best_here[1]}
        if best is None or best_here[0] > best[0]:
            best = (best_here[0], weight_decay, best_here[2])
    model = make_model().to(device)
    model.load_state_dict(best[2])
    model.eval()
    return model, best[0], {"weight_decay": best[1], "validation": tried}


def fit_mlp(train_x, train_y, num_classes, device, seed, validate, hidden=512, dropout=0.5,
            epochs=100, batch_size=4096, peak_lr=1e-3, weight_decays=(1e-2, 1e-1)):
    """One-hidden-layer MLP (softmax for single-label, a sigmoid per label for multi-label
    targets) trained with AdamC; weight decay and epoch are chosen by ``validate(predict)``.

    Returns ``(predict, validation score, info)``.
    """
    x = to_device(train_x, device)
    y, multilabel = _targets(train_y, num_classes, device)

    def make_model():
        return torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                   torch.nn.Linear(hidden, num_classes))

    def predictor(model):
        return lambda features: _batched(model, features, device)

    model, score, info = _train_with_adamc(
        make_model, lambda model, rows: _linear_loss(model(x[rows]), y[rows], multilabel, "logistic"), len(x),
        lambda model: validate(predictor(model)), seed, epochs, batch_size, peak_lr, weight_decays, device)
    return predictor(model), score, info


def knn_votes(train_x, train_y, query_x, num_classes, neighbours, device, batch=1024):
    """Cosine k-nearest-neighbour vote for every ``k`` in ``neighbours``.

    Single-label targets: the score of a class is the summed similarity of the nearest
    training vectors of that class. Multi-label targets: the share of the nearest
    training vectors that carry the label, minus one half (so that 0 is the threshold).
    Returns ``{k: scores}``.
    """
    unit = lambda array: torch.nn.functional.normalize(to_device(array, device), dim=1)
    train = unit(train_x)
    targets, multilabel = _targets(train_y, num_classes, device)
    largest = min(max(neighbours), len(train))
    scores = {k: [] for k in neighbours}
    for start in range(0, len(query_x), batch):
        similarity, nearest = (unit(query_x[start:start + batch]) @ train.T).topk(largest, dim=1)
        for k in neighbours:
            k_used = min(k, largest)
            if multilabel:
                votes = targets[nearest[:, :k_used]].mean(dim=1) - 0.5
            else:
                votes = torch.zeros(len(nearest), num_classes, device=device)
                votes.scatter_add_(1, targets[nearest[:, :k_used]], similarity[:, :k_used])
            scores[k].append(votes.cpu().numpy())
    return {k: np.concatenate(parts) for k, parts in scores.items()}


# ----------------------------------------------------------------------------- pair probes
#
# A pair scorer is a function  score(sources, targets) -> numpy scores  over arrays of
# node indices. The trained ones learn from pairs with 0/1 labels (edges and non-edges)
# and are selected by ``validate(score)``, which the caller defines (a ranking metric).


class _Bilinear(torch.nn.Module):
    """x^T (I + delta) y * temperature + bias. The matrix starts at the identity (the
    plain dot product) and the weight decay acts on its difference from it."""

    def __init__(self, dim):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(dim, dim))
        self.temperature = torch.nn.Parameter(torch.tensor(10.0))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, left, right):
        return self.temperature * ((left + left @ self.delta) * right).sum(dim=1) + self.bias


class _PairMLP(torch.nn.Module):
    """concatenation -> hidden layer -> one logit (the sigmoid is in the loss). The graph
    is undirected, so a pair is scored in both orders and the logits are averaged."""

    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(2 * dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                          torch.nn.Linear(hidden, 1))

    def forward(self, left, right):
        forward = self.layers(torch.cat([left, right], dim=1))
        backward = self.layers(torch.cat([right, left], dim=1))
        return 0.5 * (forward + backward).squeeze(1)


def _pair_scorer(function, vectors, batch=1 << 15):
    """Wraps ``function(left vectors, right vectors) -> scores`` (torch) as a pair scorer."""
    def score(sources, targets):
        out = []
        with torch.no_grad():
            for start in range(0, len(sources), batch):
                left = vectors[torch.as_tensor(sources[start:start + batch], device=vectors.device)]
                right = vectors[torch.as_tensor(targets[start:start + batch], device=vectors.device)]
                out.append(function(left, right).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    return score


def unit_vectors(vectors, device):
    """The embedding matrix on ``device`` with unit-norm rows (what the pair probes see)."""
    return torch.nn.functional.normalize(to_device(vectors, device), dim=1)


def product_scorer(matrix):
    """Dot product of two rows of ``matrix`` (a tensor on the device): the plain dot
    product for raw vectors, the cosine for unit vectors."""
    return _pair_scorer(lambda left, right: (left * right).sum(dim=1), matrix)


def fit_pair_model(kind, unit, pairs, labels, device, seed, validate, hidden=512, dropout=0.2,
                   epochs=30, batch_size=8192, peak_lr=1e-3, weight_decays=(1e-2, 1e-1)):
    """Trains a ``'bilinear'`` or ``'mlp'`` pair model with AdamC on labelled pairs
    (binary cross-entropy on the sigmoid of the logit). Weight decay and epoch are chosen
    by ``validate(score)``. Returns ``(score, validation score, info)``."""
    dim = unit.shape[1]
    sources = torch.as_tensor(pairs[:, 0], device=device)
    targets = torch.as_tensor(pairs[:, 1], device=device)
    truth = to_device(labels, device)
    if kind == "bilinear":
        make_model, inputs = (lambda: _Bilinear(dim)), unit
    else:
        # coordinates of order one, as a freshly initialised layer expects
        make_model, inputs = (lambda: _PairMLP(dim, hidden, dropout)), unit * math.sqrt(dim)

    def step_loss(model, rows):
        logits = model(inputs[sources[rows]], inputs[targets[rows]])
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, truth[rows])

    model, score, info = _train_with_adamc(
        make_model, step_loss, len(pairs), lambda model: validate(_pair_scorer(model, inputs)),
        seed, epochs, batch_size, peak_lr, weight_decays, device)
    return _pair_scorer(model, inputs), score, info


def fit_pair_linear(kind, unit, pairs, labels, device, validate):
    """Linear model (``'logistic'`` or ``'svm'``) on the element-wise product of the two
    vectors of a pair - the edge feature node2vec found to work best. The decay is chosen
    by ``validate(score)``. Returns ``(score, validation score, info)``."""
    index = lambda column: torch.as_tensor(pairs[:, column], device=device)
    features = unit[index(0)] * unit[index(1)]
    targets = np.asarray(labels).reshape(-1, 1) > 0

    def as_pair_scorer(weight, bias):
        return _pair_scorer(lambda left, right: ((left * right) @ weight.T + bias).squeeze(1), unit)

    return fit_linear(features, targets, kind, 1, device, validate, wrap=as_pair_scorer)
