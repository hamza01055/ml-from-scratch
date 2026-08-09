"""Activation functions and their derivatives.

Each activation is a pair: the forward function f(z), and its derivative
written in terms of whatever is cheapest to reuse from the forward pass.
"""

import numpy as np


def sigmoid(z):
    """Numerically stable logistic sigmoid.

    The naive 1/(1+exp(-z)) overflows for very negative z. We branch so the
    exponent is always <= 0.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_prime_from_output(a):
    """d/dz sigmoid(z), given a = sigmoid(z). Cheaper than recomputing."""
    return a * (1.0 - a)


def relu(z):
    return np.maximum(0.0, z)


def relu_prime(z):
    """Derivative of ReLU. The kink at 0 is defined to be 0 by convention."""
    return (np.asarray(z) > 0).astype(float)


def tanh(z):
    return np.tanh(z)


def tanh_prime_from_output(a):
    """d/dz tanh(z), given a = tanh(z)."""
    return 1.0 - a**2


def softmax(z, axis=-1):
    """Numerically stable softmax.

    Subtracting the row max leaves the result unchanged mathematically but
    keeps every exponent <= 0, so nothing overflows.
    """
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)
