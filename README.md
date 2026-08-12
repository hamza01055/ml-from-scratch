<div align="center">


<h1>Machine Learning From Scratch</h1>

<p><strong>Understand machine learning by building it from the ground up.</strong></p>

<p>
12 core machine learning chapters implemented with <strong>Python + NumPy</strong> —<br>
derived mathematically, visualized experimentally, and verified numerically.
</p>

<p>
<code>No scikit-learn</code> &nbsp;·&nbsp;
<code>No PyTorch</code> &nbsp;·&nbsp;
<code>No black-box model.fit()</code>
</p>


<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-From%20Scratch-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
<br>
<img alt="Tests" src="https://img.shields.io/badge/Tests-59%20Passing-2EA44F?style=for-the-badge&logo=pytest&logoColor=white">
<img alt="Chapters" src="https://img.shields.io/badge/Chapters-12-orange?style=for-the-badge">
<img alt="License" src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge">
</p>

<p>
<a href="#-chapters"><strong>Chapters</strong></a> &nbsp;·&nbsp;
<a href="#-quick-start"><strong>Quick Start</strong></a> &nbsp;·&nbsp;
<a href="#-correctness-first"><strong>Correctness</strong></a> &nbsp;·&nbsp;
<a href="#-reusable-python-package"><strong>Package</strong></a> &nbsp;·&nbsp;
<a href="#-lessons-from-building-it"><strong>Lessons</strong></a> &nbsp;·&nbsp;
<a href="https://colab.research.google.com/github/hamza01055/ml-from-scratch"><strong>Open in Colab</strong></a>
</p>

<br>

</div>

---

<div align="center">
<em>Just the mathematics, the implementation, and the reasoning behind the algorithms.</em>
</div>

---

## ⚡ At a Glance

|  |  |
|---|---|
| **12** | Complete ML chapters |
| **59** | Automated tests |
| **NumPy** | Core implementation |
| **Jupyter** | Interactive learning |
| **Finite Differences** | Gradient verification |
| **MIT** | Open-source license |

### Core Topics

`Linear Regression` · `Logistic Regression` · `Neural Networks` · `Backpropagation` · `Adam` · `k-NN` · `k-Means` · `Decision Trees` · `Random Forests` · `PCA` · `Naive Bayes` · `SVM` · `Autodiff`

---

## 🎯 Why This Project?

Modern ML libraries make it incredibly easy to train a model:

```python
model.fit(X, y)
```

But that single line hides almost everything interesting.

This project takes a different approach:

```text
Mathematics
     ↓
Derivation
     ↓
NumPy implementation
     ↓
Visualization
     ↓
Numerical verification
     ↓
Reusable implementation
```

The goal isn't to reinvent production ML frameworks.

The goal is to understand **what happens underneath them**.

---

## 🧠 What You'll Learn

By the end of the repository, you'll have implemented:

- 🧮 **Linear Regression** — Normal equation and gradient descent
- 🎯 **Logistic Regression** — Sigmoid and cross-entropy
- 🧠 **Neural Networks** — Backpropagation from scratch
- ⚡ **Optimizers** — SGD, Momentum, and Adam
- 🛡️ **Regularization** — L1, L2, dropout, and early stopping
- 📍 **k-NN & k-Means** — Classification and clustering
- 🌳 **Decision Trees & Random Forests** — Gini, information gain, bagging, and OOB evaluation
- 📉 **PCA** — Dimensionality reduction using SVD
- 📊 **Model Evaluation** — Cross-validation, ROC/AUC, and data leakage
- 📐 **Naive Bayes** — Gaussian and Multinomial implementations
- 🔷 **SVMs** — Linear and kernel SVMs
- 🔄 **Automatic Differentiation** — A tiny reverse-mode autodiff engine

---

## 📚 Chapters

| # | Topic | Key Concepts |
|---|---|---|
| 00 | [NumPy Foundations](notebooks/00_setup_and_numpy_refresher.ipynb) | Shapes, broadcasting, vectorization |
| 01 | [Linear Regression](notebooks/01_linear_regression.ipynb) | Normal equation, gradient descent |
| 02 | [Optimization](notebooks/02_gradient_descent.ipynb) | SGD, Momentum, Adam |
| 03 | [Logistic Regression](notebooks/03_logistic_regression.ipynb) | Sigmoid, cross-entropy, boundaries |
| 04 | [Neural Networks](notebooks/04_neural_network_backprop.ipynb) | MLP, backpropagation |
| 05 | [Regularization](notebooks/05_regularization.ipynb) | Ridge, Lasso, dropout |
| 06 | [k-NN & k-Means](notebooks/06_knn_and_kmeans.ipynb) | Neighbors, clustering, k-means++ |
| 07 | [Decision Trees](notebooks/07_decision_trees.ipynb) | Gini, entropy, random forests |
| 08 | [PCA](notebooks/08_pca_and_dimensionality.ipynb) | SVD, variance, reconstruction |
| 09 | [Model Evaluation](notebooks/09_model_evaluation.ipynb) | CV, ROC/AUC, leakage |
| 10 | [Naive Bayes & SVM](notebooks/10_naive_bayes_and_svm.ipynb) | Bayes, hinge loss, kernels |
| 11 | [Tiny Autodiff](notebooks/11_micrograd_autodiff.ipynb) | Reverse-mode autodiff |

---

## 🏗️ Repository Structure

```text
ml-from-scratch/
│
├── notebooks/
│   ├── 00_setup_and_numpy_refresher.ipynb
│   ├── 01_linear_regression.ipynb
│   ├── 02_gradient_descent.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_neural_network_backprop.ipynb
│   ├── 05_regularization.ipynb
│   ├── 06_knn_and_kmeans.ipynb
│   ├── 07_decision_trees.ipynb
│   ├── 08_pca_and_dimensionality.ipynb
│   ├── 09_model_evaluation.ipynb
│   ├── 10_naive_bayes_and_svm.ipynb
│   └── 11_micrograd_autodiff.ipynb
│
├── mlscratch/
│   ├── activations.py
│   ├── losses.py
│   ├── models.py
│   ├── optim.py
│   ├── neighbors.py
│   ├── cluster.py
│   ├── tree.py
│   ├── decomposition.py
│   ├── naive_bayes.py
│   ├── svm.py
│   ├── autograd.py
│   ├── metrics.py
│   └── utils.py
│
├── tests/
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

---

## 🔬 Correctness First

One of the main goals of this project is not simply making the algorithms run.

It's making sure the mathematics is actually correct.

For gradient-based algorithms, analytical gradients are compared against central-difference numerical gradients.

### Current verification

```text
59 tests passed
```

Neural-network backpropagation:

```text
Worst relative error: 1.2e-8
```

Autodiff engine:

```text
Worst relative error: 3.1e-8
```

This catches a class of bugs that ordinary training tests can easily miss.

---

## 🧪 Testing

Run the complete suite:

```bash
pytest tests/ -v
```

The tests cover:

- 🧮 Gradient calculations
- 📈 Regression models
- 🎯 Classification models
- 🧠 Neural-network backpropagation
- 🔎 k-NN
- 🔵 k-Means
- 🌳 Decision trees
- 🌲 Random forests
- 📉 PCA
- 📊 Naive Bayes
- ⚡ SVM
- 🔗 Kernel behavior
- 🤖 Automatic differentiation
- 🛡️ Numerical edge cases

---

## 📦 Reusable Python Package

The notebooks are designed for learning.

The `mlscratch` package contains the cleaned-up implementations.

```python
from mlscratch.models import NeuralNetwork
from mlscratch.optim import Adam
from mlscratch.metrics import accuracy

model = NeuralNetwork(
    [2, 32, 16, 2],
    seed=0
)

model.fit(
    X_train,
    y_train,
    epochs=150,
    optimizer=Adam(lr=0.01)
)

predictions = model.predict(X_test)

print(accuracy(y_test, predictions))
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/hamza01055/ml-from-scratch.git

cd ml-from-scratch

python -m venv .venv
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the notebooks:

```bash
jupyter lab notebooks/
```

Start here:

```text
notebooks/00_setup_and_numpy_refresher.ipynb
```

---

## ☁️ Google Colab

Don't want to install anything?

Open the repository in Google Colab:

**[Open in Google Colab →](https://colab.research.google.com/github/hamza01055/ml-from-scratch)**

---

## 💡 Lessons From Building It

This repository also documents mistakes and failed assumptions.

### Decision Trees and XOR

Greedy decision trees don't automatically discover the correct XOR structure when individual splits provide little information gain.

### Kernel SVM

Hinge-loss optimization is non-smooth. Using an inappropriate constant step size caused instability for large values of `C`.

### PCA

PCA maximizes variance, not predictive power. Dimensionality reduction can therefore make classification worse.

### Data Leakage

Standardizing data before splitting is still leakage even when the resulting metric difference appears negligible.

These failures are part of the project because understanding **why something fails** is often more valuable than seeing only the successful implementation.

---

## 🛠️ Tech Stack

```text
Python
NumPy
Jupyter
Matplotlib
Pytest
GitHub Actions
```

The core ML algorithms are implemented without high-level ML frameworks.

---

## 🎓 Prerequisites

You only need:

**Python** — Functions, classes, and basic NumPy.

**Calculus** — Derivatives and the chain rule.

**Linear Algebra** — Vectors, matrices, multiplication, and shapes.

You don't need previous machine-learning experience.

---

## 🤝 Contributing

Found a bug?

Have a clearer explanation?

Want to improve an exercise?

Open an issue or submit a pull request.

If an explanation is confusing, consider that a documentation bug.

---

## 📖 References

- Christopher Bishop — *Pattern Recognition and Machine Learning*
- Goodfellow, Bengio & Courville — *Deep Learning*
- Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning*
- Andrej Karpathy — *Neural Networks: Zero to Hero*
- Andrew Ng — *Machine Learning Specialization*

---

## 👨‍💻 Author

### Hamza Shahzad

AI Engineer focused on building production-ready AI systems.

Interested in:

`AI` · `Machine Learning` · `LLMs` · `Agentic AI` · `RAG` · `AI Automation` · `Computer Vision` · `Python` · `FastAPI`

This repository represents the fundamentals behind that work:

> **Learn the mathematics. Build the algorithm. Verify the implementation.**

---

## ⭐ Support

If this project helped you understand an ML concept, consider giving it a ⭐.

It helps other developers discover the repository.

**GitHub:** [https://github.com/hamza01055/ml-from-scratch](https://github.com/hamza01055/ml-from-scratch)

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
