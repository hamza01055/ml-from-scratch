"""Behavioural tests: do the models actually learn?"""

import numpy as np

from mlscratch.metrics import accuracy, r2_score
from mlscratch.models import LinearRegression, LogisticRegression, NeuralNetwork
from mlscratch.optim import Adam


def _linear_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + 1.0 + rng.normal(0, 0.1, n)
    return X, y, true_w


def test_normal_equation_recovers_true_weights():
    X, y, true_w = _linear_data()
    model = LinearRegression(method="normal").fit(X, y)
    assert np.allclose(model.coef_, true_w, atol=0.05)
    assert abs(model.intercept_ - 1.0) < 0.05


def test_gradient_descent_matches_closed_form():
    X, y, _ = _linear_data()
    closed = LinearRegression(method="normal").fit(X, y)
    gd = LinearRegression(lr=0.1, n_iters=2000).fit(X, y)
    assert np.allclose(closed.theta, gd.theta, atol=0.01)
    assert gd.history[-1] < gd.history[0]


def test_linear_regression_fits_well():
    X, y, _ = _linear_data()
    model = LinearRegression(method="normal").fit(X, y)
    assert r2_score(y, model.predict(X)) > 0.95


def test_logistic_regression_separates_linear_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(lr=0.5, n_iters=2000).fit(X, y)
    assert accuracy(y, model.predict(X)) > 0.95
    assert model.history[-1] < model.history[0]


def test_network_learns_xor_which_is_not_linearly_separable():
    X = np.array([[0.0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    net = NeuralNetwork([2, 16, 2], seed=1)
    net.fit(X, y, epochs=2000, batch_size=4, optimizer=Adam(lr=0.05))
    assert accuracy(y, net.predict(X)) == 1.0


def test_adam_reaches_lower_loss_than_plain_sgd():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)

    sgd_net = NeuralNetwork([4, 32, 2], lr=0.05, seed=0).fit(X, y, epochs=60)
    adam_net = NeuralNetwork([4, 32, 2], seed=0).fit(X, y, epochs=60, optimizer=Adam(lr=0.01))
    assert adam_net.history[-1] < sgd_net.history[-1]
