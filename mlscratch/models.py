"""Reference implementations of the models built in the notebooks.

The notebooks derive these step by step. This module is the cleaned-up
version you can import once you have understood the derivation.
"""

import numpy as np

from .activations import relu, relu_prime, sigmoid, softmax
from .losses import (
    binary_cross_entropy,
    cross_entropy,
    mse,
    softmax_cross_entropy_grad,
)
from .utils import iterate_minibatches, one_hot


def _add_bias(X):
    """Prepend a column of ones so the intercept is just another weight."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


class LinearRegression:
    """y = Xw + b, fit either in closed form or by gradient descent."""

    def __init__(self, lr=0.01, n_iters=1000, method="gd"):
        self.lr, self.n_iters, self.method = lr, n_iters, method
        self.theta = None
        self.history = []

    def fit(self, X, y):
        Xb = _add_bias(np.asarray(X, float))
        y = np.asarray(y, float).ravel()

        if self.method == "normal":
            # theta = (X'X)^-1 X'y — pinv handles singular X'X gracefully.
            self.theta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
            return self

        n, d = Xb.shape
        self.theta = np.zeros(d)
        for _ in range(self.n_iters):
            pred = Xb @ self.theta
            grad = (2.0 / n) * Xb.T @ (pred - y)
            self.theta -= self.lr * grad
            self.history.append(mse(y, pred))
        return self

    def predict(self, X):
        return _add_bias(np.asarray(X, float)) @ self.theta

    @property
    def coef_(self):
        return self.theta[1:]

    @property
    def intercept_(self):
        return self.theta[0]


class LogisticRegression:
    """Binary classifier: p = sigmoid(Xw + b), trained on cross entropy."""

    def __init__(self, lr=0.1, n_iters=1000, l2=0.0):
        self.lr, self.n_iters, self.l2 = lr, n_iters, l2
        self.theta = None
        self.history = []

    def fit(self, X, y):
        Xb = _add_bias(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        n, d = Xb.shape
        self.theta = np.zeros(d)

        for _ in range(self.n_iters):
            p = sigmoid(Xb @ self.theta)
            # The gradient of BCE w.r.t. theta collapses to X'(p - y)/n.
            grad = Xb.T @ (p - y) / n
            if self.l2:
                reg = self.l2 * self.theta
                reg[0] = 0.0  # never penalize the intercept
                grad += reg
            self.theta -= self.lr * grad
            self.history.append(binary_cross_entropy(y, p))
        return self

    def predict_proba(self, X):
        return sigmoid(_add_bias(np.asarray(X, float)) @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class NeuralNetwork:
    """A fully connected network with ReLU hidden layers and a softmax head.

    Weights use He initialisation, which keeps the variance of activations
    roughly constant as signal passes through ReLU layers.
    """

    def __init__(self, layer_sizes, lr=0.01, seed=0):
        self.layer_sizes = layer_sizes
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.params = {}
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            self.params[f"W{i}"] = rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            self.params[f"b{i}"] = np.zeros(fan_out)
        self.n_layers = len(layer_sizes) - 1
        self.history = []

    def forward(self, X):
        """Return output probabilities plus the cache needed for backprop."""
        cache = {"a0": X}
        a = X
        for i in range(self.n_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"z{i}"] = z
            a = softmax(z) if i == self.n_layers - 1 else relu(z)
            cache[f"a{i + 1}"] = a
        return a, cache

    def backward(self, y_onehot, cache):
        """Chain rule, walked backwards one layer at a time."""
        grads = {}
        # Softmax + cross entropy gives this clean starting delta.
        delta = softmax_cross_entropy_grad(y_onehot, cache[f"a{self.n_layers}"])
        for i in reversed(range(self.n_layers)):
            grads[f"W{i}"] = cache[f"a{i}"].T @ delta
            grads[f"b{i}"] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.params[f"W{i}"].T) * relu_prime(cache[f"z{i - 1}"])
        return grads

    def fit(self, X, y, epochs=20, batch_size=32, optimizer=None, seed=0):
        n_classes = self.layer_sizes[-1]
        Y = one_hot(y, n_classes)
        for epoch in range(epochs):
            for xb, yb in iterate_minibatches(X, Y, batch_size, seed=seed + epoch):
                probs, cache = self.forward(xb)
                grads = self.backward(yb, cache)
                if optimizer is not None:
                    self.params = optimizer.step(self.params, grads)
                else:
                    for k in self.params:
                        self.params[k] -= self.lr * grads[k]
            probs, _ = self.forward(X)
            self.history.append(cross_entropy(Y, probs))
        return self

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)
