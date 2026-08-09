"""Shared helpers: data splitting, batching, scaling, gradient checking."""

import numpy as np


def train_test_split(X, y, test_size=0.2, seed=0):
    """Shuffle then slice. No stratification — kept deliberately simple."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def iterate_minibatches(X, y, batch_size=32, shuffle=True, seed=None):
    """Yield (X_batch, y_batch) tuples for one epoch."""
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, n, batch_size):
        chunk = idx[start : start + batch_size]
        yield X[chunk], y[chunk]


def standardize(X, mean=None, std=None):
    """Zero mean, unit variance per feature. Returns (X_scaled, mean, std).

    Fit the scaler on train only, then reuse mean/std on test — otherwise
    test statistics leak into training.
    """
    X = np.asarray(X, float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def one_hot(y, n_classes=None):
    y = np.asarray(y, int).ravel()
    if n_classes is None:
        n_classes = y.max() + 1
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1.0
    return out


def numerical_gradient(f, x, h=1e-5):
    """Central-difference gradient of scalar f at array x.

    This is the ground truth we check every analytic gradient against.
    O(2n) function evaluations, so it is only practical for tiny inputs —
    which is exactly what a unit test should use.
    """
    x = np.asarray(x, float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        original = x[i]
        x[i] = original + h
        f_plus = f(x)
        x[i] = original - h
        f_minus = f(x)
        x[i] = original
        grad[i] = (f_plus - f_minus) / (2 * h)
        it.iternext()
    return grad


def relative_error(a, b, eps=1e-12):
    """Scale-free comparison. Below 1e-7 means the gradients agree."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b)))
