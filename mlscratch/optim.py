"""Optimizers, written as small stateful objects.

Every optimizer answers one question: given the current parameters and their
gradients, what is the next set of parameters?
"""

import numpy as np


class SGD:
    """Stochastic gradient descent, optionally with classical momentum."""

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity = {}

    def step(self, params, grads):
        for key in params:
            if self.momentum:
                v = self._velocity.get(key, np.zeros_like(params[key]))
                v = self.momentum * v - self.lr * grads[key]
                self._velocity[key] = v
                params[key] = params[key] + v
            else:
                params[key] = params[key] - self.lr * grads[key]
        return params


class Adam:
    """Adam: per-parameter adaptive learning rates with bias correction.

    m is an exponential moving average of the gradient (first moment),
    v of the squared gradient (second moment). Both start at zero, which
    biases them low early on — hence the bias-correction divides.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self._m, self._v, self._t = {}, {}, 0

    def step(self, params, grads):
        self._t += 1
        for key in params:
            g = grads[key]
            m = self._m.get(key, np.zeros_like(g))
            v = self._v.get(key, np.zeros_like(g))

            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g**2)
            self._m[key], self._v[key] = m, v

            m_hat = m / (1 - self.beta1**self._t)
            v_hat = v / (1 - self.beta2**self._t)

            params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params
