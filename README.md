# Machine Learning from Scratch

**Every core ML algorithm, implemented in NumPy, derived step by step in Jupyter notebooks.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/hamza01055/ml-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/hamza01055/ml-from-scratch/actions)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamza01055/ml-from-scratch)

No `scikit-learn`. No PyTorch. No `model.fit()` that hides the interesting part.
Just NumPy, the math written out, and code you can read top to bottom.

---

## Why this repo exists

You can call `LogisticRegression().fit(X, y)` without knowing what a gradient is.
That works right up until the model does something strange and you have no idea why.

I'm learning this properly by building each algorithm myself, and writing down the
derivation as I go. This repo is that work, in public. If it's useful to you too,
that's a bonus.

**Every notebook follows the same shape:**

1. **The problem** — what breaks if we don't have this
2. **The math** — derived, not quoted
3. **The code** — NumPy, commented, no framework
4. **The check** — analytic gradients verified against finite differences
5. **Exercises** — the parts I deliberately left for you

---

## Chapters

| # | Notebook | What you'll build | Status |
|---|---|---|---|
| 00 | [Setup & NumPy refresher](notebooks/00_setup_and_numpy_refresher.ipynb) | Shapes, broadcasting, vectorisation, numerical stability | Complete |
| 01 | [Linear regression](notebooks/01_linear_regression.ipynb) | Normal equation + gradient descent, from the same loss | Complete |
| 02 | [Gradient descent in depth](notebooks/02_gradient_descent.ipynb) | SGD, momentum, Adam — with the optimisation paths drawn | Complete |
| 03 | [Logistic regression](notebooks/03_logistic_regression.ipynb) | Sigmoid, cross entropy, decision boundaries, why MSE fails | Complete |
| 04 | [Neural nets & backprop](notebooks/04_neural_network_backprop.ipynb) | A full MLP and backprop, gradient-checked | Complete |
| 05 | [Regularisation](notebooks/05_regularization.ipynb) | L1/L2, dropout, early stopping, bias-variance | In progress |
| 06 | [k-NN & k-Means](notebooks/06_knn_and_kmeans.ipynb) | Distance metrics, clustering, curse of dimensionality | In progress |
| 07 | [Decision trees](notebooks/07_decision_trees.ipynb) | Gini, information gain, bagging, random forests | In progress |
| 08 | [PCA](notebooks/08_pca_and_dimensionality.ipynb) | Eigendecomposition, SVD, explained variance | In progress |
| 09 | [Model evaluation](notebooks/09_model_evaluation.ipynb) | Cross-validation, ROC/AUC, data leakage | In progress |
| 10 | [Naive Bayes & SVM](notebooks/10_naive_bayes_and_svm.ipynb) | Hinge loss, margins, the kernel trick | In progress |
| 11 | [Tiny autodiff engine](notebooks/11_micrograd_autodiff.ipynb) | Reverse-mode autodiff in ~100 lines | In progress |

Chapters 00–04 are written out in full and run end to end. The rest have complete
outlines and working starter code, and get filled in as I work through them.

---

## Quick start

```bash
git clone https://github.com/hamza01055/ml-from-scratch.git
cd ml-from-scratch

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab notebooks/
```

Start at `00_setup_and_numpy_refresher.ipynb` and go in order. Each chapter
assumes the one before it.

**Prefer not to install anything?** Every notebook opens in
[Google Colab](https://colab.research.google.com/github/hamza01055/ml-from-scratch)
with zero setup.

---

## The `mlscratch` package

The notebooks derive each algorithm slowly. `mlscratch/` is the cleaned-up version
of the same code, importable once you've understood the derivation.

```python
from mlscratch.models import NeuralNetwork
from mlscratch.optim import Adam
from mlscratch.metrics import accuracy

net = NeuralNetwork([2, 32, 16, 2], seed=0)
net.fit(X_train, y_train, epochs=150, optimizer=Adam(lr=0.01))

print(accuracy(y_test, net.predict(X_test)))
```

| Module | Contents |
|---|---|
| `activations.py` | sigmoid, ReLU, tanh, softmax — all numerically stable |
| `losses.py` | MSE, binary & categorical cross entropy, with gradients |
| `models.py` | `LinearRegression`, `LogisticRegression`, `NeuralNetwork` |
| `optim.py` | `SGD` (with momentum), `Adam` (with bias correction) |
| `metrics.py` | accuracy, precision/recall/F1, confusion matrix, R² |
| `utils.py` | train/test split, mini-batching, standardisation, gradient checking |

---

## On correctness

A wrong gradient doesn't raise an exception. Training still runs, the loss still
goes down, and you end up with a model that quietly underperforms for reasons
you'll never find.

So every analytic gradient here is checked against central-difference finite
differences, and those checks run in CI:

```bash
pytest tests/ -v
```

```
tests/test_gradients.py::test_mse_gradient                                PASSED
tests/test_gradients.py::test_bce_gradient                                PASSED
tests/test_gradients.py::test_network_backprop_matches_finite_differences PASSED
tests/test_models.py::test_normal_equation_recovers_true_weights          PASSED
tests/test_models.py::test_network_learns_xor_which_is_not_...   PASSED
...
11 passed in 0.89s
```

The backprop check compares every weight and bias matrix against finite
differences and requires relative error below `1e-6`. Current worst case: `1.2e-8`.

---

## What you need to know first

- **Python** — comfortable with functions, classes, and NumPy basics
- **Calculus** — what a derivative is, and the chain rule. That's genuinely it;
  everything else is derived in place.
- **Linear algebra** — matrix multiplication and what a shape means

You do **not** need prior ML experience. That's the point.

---

## Contributing

Corrections, clearer explanations, and finished chapters are all welcome.
Issues are good for "this explanation didn't land" as well as for bugs — if
something was confusing, that's a defect in the writing.

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## References

Books and courses that shaped these notes. All are worth your time:

- Bishop, *Pattern Recognition and Machine Learning*
- Goodfellow, Bengio & Courville, [*Deep Learning*](https://www.deeplearningbook.org/) (free online)
- Hastie, Tibshirani & Friedman, [*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/) (free online)
- Andrej Karpathy, [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)
- Andrew Ng, [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)

---

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, teach from it.

If it helped, a star costs you nothing and helps someone else find it.
