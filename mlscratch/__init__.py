"""mlscratch — machine learning fundamentals implemented from scratch in NumPy.

Every algorithm in this package is written without scikit-learn, PyTorch, or
TensorFlow. The point is to see the math, not to hide it.
"""

__version__ = "0.1.0"

from . import activations, losses, metrics, optim, utils  # noqa: F401

__all__ = ["activations", "losses", "metrics", "optim", "utils"]
