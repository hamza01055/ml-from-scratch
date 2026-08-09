"""Loss functions and their gradients with respect to predictions."""

import numpy as np

EPS = 1e-12


def mse(y_true, y_pred):
    """Mean squared error, averaged over all elements."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_pred - y_true) ** 2))


def mse_grad(y_true, y_pred):
    """dL/dy_pred for mse. The 2/n factor comes from the mean and the square."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return 2.0 * (y_pred - y_true) / y_true.size


def binary_cross_entropy(y_true, y_prob):
    """BCE for probabilities in (0, 1). Clipped to avoid log(0) = -inf."""
    y_true = np.asarray(y_true, float)
    p = np.clip(np.asarray(y_prob, float), EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def binary_cross_entropy_grad(y_true, y_prob):
    y_true = np.asarray(y_true, float)
    p = np.clip(np.asarray(y_prob, float), EPS, 1.0 - EPS)
    return (p - y_true) / (p * (1 - p) * y_true.size)


def cross_entropy(y_true_onehot, y_prob):
    """Categorical cross entropy averaged over the batch."""
    y = np.asarray(y_true_onehot, float)
    p = np.clip(np.asarray(y_prob, float), EPS, 1.0)
    return float(-np.sum(y * np.log(p)) / y.shape[0])


def softmax_cross_entropy_grad(y_true_onehot, y_prob):
    """dL/dlogits when the last layer is softmax and the loss is cross entropy.

    The famous simplification: the softmax Jacobian and the cross-entropy
    gradient cancel, leaving just (p - y) / batch_size.
    """
    y = np.asarray(y_true_onehot, float)
    p = np.asarray(y_prob, float)
    return (p - y) / y.shape[0]
