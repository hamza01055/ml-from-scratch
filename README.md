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

| # | Notebook | What you'll build |
|---|---|---|
| 00 | [Setup & NumPy refresher](notebooks/00_setup_and_numpy_refresher.ipynb) | Shapes, broadcasting, vectorisation, numerical stability |
| 01 | [Linear regression](notebooks/01_linear_regression.ipynb) | Normal equation + gradient descent, from the same loss |
| 02 | [Gradient descent in depth](notebooks/02_gradient_descent.ipynb) | SGD, momentum, Adam — with the optimisation paths drawn |
| 03 | [Logistic regression](notebooks/03_logistic_regression.ipynb) | Sigmoid, cross entropy, decision boundaries, why MSE fails |
| 04 | [Neural nets & backprop](notebooks/04_neural_network_backprop.ipynb) | A full MLP and backprop, gradient-checked |
| 05 | [Regularisation](notebooks/05_regularization.ipynb) | Ridge, lasso via coordinate descent, dropout, early stopping, bias-variance measured empirically |
| 06 | [k-NN & k-Means](notebooks/06_knn_and_kmeans.ipynb) | Distance metrics, k-means++, silhouette, the curse of dimensionality |
| 07 | [Decision trees](notebooks/07_decision_trees.ipynb) | Gini, information gain, bagging, random forests, OOB error |
| 08 | [PCA](notebooks/08_pca_and_dimensionality.ipynb) | Eigendecomposition vs SVD, explained variance, denoising, where it fails |
| 09 | [Model evaluation](notebooks/09_model_evaluation.ipynb) | Stratified k-fold, data leakage, ROC/AUC, bootstrap confidence intervals |
| 10 | [Naive Bayes & SVM](notebooks/10_naive_bayes_and_svm.ipynb) | Log-space likelihoods, hinge loss, margins, the kernel trick |
| 11 | [Tiny autodiff engine](notebooks/11_micrograd_autodiff.ipynb) | Reverse-mode autodiff in ~100 lines, then a network trained on it |

**All twelve chapters are written out in full.** Every notebook runs end to end
in CI with its outputs committed, so what you see is what the code actually
produced.

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
| `neighbors.py` | `KNNClassifier`, `KNNRegressor`, vectorised distance metrics |
| `cluster.py` | `KMeans` with k-means++, silhouette score |
| `tree.py` | `DecisionTreeClassifier`, `RandomForestClassifier`, Gini/entropy |
| `decomposition.py` | `PCA` via SVD, with whitening and reconstruction |
| `naive_bayes.py` | `GaussianNB`, `MultinomialNB` with Laplace smoothing |
| `svm.py` | `LinearSVM`, `KernelSVM`, RBF and polynomial kernels |
| `autograd.py` | `Value`, `Neuron`, `Layer`, `MLP` — reverse-mode autodiff |
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
tests/test_gradients.py  ..........        analytic vs finite differences
tests/test_models.py     ......            do the models actually learn?
tests/test_classical.py  ...............   kNN, k-means, trees, PCA, NB, SVM
tests/test_autograd.py   .............     the autodiff engine
59 passed in 14.74s
```

Three checks worth calling out:

- **Backprop** (chapter 04) is compared against central-difference finite
  differences for every weight and bias matrix. Worst relative error: `1.2e-8`.
- **The autodiff engine** (chapter 11) is checked the same way across a whole
  network. Worst relative error: `3.1e-8`. Two completely independent
  implementations agreeing to eight digits is the strongest evidence available
  that both are right.
- **`test_kernel_svm_converges_across_C`** is a regression test. With a constant
  step size the subgradient method diverged for large `C` and silently collapsed
  to predicting one class. That bug is now pinned.

---

## What you need to know first

- **Python** — comfortable with functions, classes, and NumPy basics
- **Calculus** — what a derivative is, and the chain rule. That's genuinely it;
  everything else is derived in place.
- **Linear algebra** — matrix multiplication and what a shape means

You do **not** need prior ML experience. That's the point.

---

## A few things I got wrong along the way

Left in deliberately, because the mistakes are more instructive than the fixes:

- **Greedy trees can't do XOR.** I initially claimed the tree "found the two
  boundaries on its own." It didn't — every root split on XOR has near-zero
  information gain, so noise decides which one wins. Chapter 07 now shows the
  actual numbers.
- **Kernel SVMs need a decaying step size.** Hinge loss is non-smooth, so this is
  *subgradient* descent, which only converges with a diminishing step. A constant
  step made `b` drift without bound. Chapter 10 explains it.
- **PCA can make things worse.** I expected it to rescue k-NN from the curse of
  dimensionality. It didn't — PCA maximises variance and has no idea which
  directions carry the label. Chapter 08 shows the table where it *hurt*.
- **Leaking a scaler barely matters.** The classic "standardise before splitting"
  error produced a difference indistinguishable from zero on my data. Chapter 09
  says so rather than manufacturing a scary number.

## Contributing

Corrections, clearer explanations, and better exercises are all welcome.
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
