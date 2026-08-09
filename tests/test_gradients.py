"""Every analytic gradient in this repo is checked against finite differences.

If a gradient is wrong, training still "works" — it just quietly converges to
the wrong thing. These tests are the cheapest insurance against that.
"""

import numpy as np
import pytest

from mlscratch.activations import sigmoid, softmax
from mlscratch.losses import binary_cross_entropy, binary_cross_entropy_grad, mse, mse_grad
from mlscratch.models import NeuralNetwork
from mlscratch.utils import numerical_gradient, one_hot, relative_error

TOL = 1e-6


def test_mse_gradient():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=5)
    y_pred = rng.normal(size=5)
    analytic = mse_grad(y_true, y_pred)
    numeric = numerical_gradient(lambda p: mse(y_true, p), y_pred.copy())
    assert relative_error(analytic, numeric) < TOL


def test_bce_gradient():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=6).astype(float)
    y_pred = rng.uniform(0.1, 0.9, size=6)
    analytic = binary_cross_entropy_grad(y_true, y_pred)
    numeric = numerical_gradient(lambda p: binary_cross_entropy(y_true, p), y_pred.copy())
    assert relative_error(analytic, numeric) < 1e-5


def test_sigmoid_is_stable_at_extremes():
    out = sigmoid(np.array([-1e4, 0.0, 1e4]))
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(2)
    z = rng.normal(scale=50, size=(4, 7))
    p = softmax(z, axis=1)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0)


def test_network_backprop_matches_finite_differences():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4, 3))
    y = one_hot(rng.integers(0, 2, size=4), 2)
    net = NeuralNetwork([3, 5, 2], seed=0)

    probs, cache = net.forward(X)
    grads = net.backward(y, cache)

    def loss_for(param_key, value):
        saved = net.params[param_key].copy()
        net.params[param_key] = value
        p, _ = net.forward(X)
        net.params[param_key] = saved
        return -np.sum(y * np.log(np.clip(p, 1e-12, 1.0))) / y.shape[0]

    for key in net.params:
        numeric = numerical_gradient(
            lambda v, k=key: loss_for(k, v), net.params[key].copy()
        )
        assert relative_error(grads[key], numeric) < 1e-5, f"bad gradient for {key}"
