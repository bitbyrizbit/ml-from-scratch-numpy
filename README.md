![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

# ml-from-scratch-numpy

NumPy implementations of core supervised learning algorithms built
entirely from first principles — no scikit-learn for training, no
black-box abstractions. Every gradient is derived analytically, every
parameter update is implemented manually, and every result is validated
against a production-grade library.

---

## TL;DR

Six self-contained ML modules — EDA, Linear Regression, Logistic
Regression, Regularization, SGD, and Neural Network — each built from
mathematical definitions using only NumPy. Modular architecture, explicit
gradient derivation, convergence analysis, and sklearn validation
throughout. Built to understand what happens under the hood, not just to
get predictions out.

---

## Motivation

Most ML education skips the internals. You call `.fit()`, get predictions,
and move on.

This repository takes the opposite approach. Every algorithm is
reconstructed from its mathematical foundation — gradients derived by
hand, optimization implemented explicitly, and convergence behavior
analyzed experimentally.

The goal is not to outperform scikit-learn. The goal is to understand
exactly what scikit-learn is doing, and why it works.

---

## Architecture Decision

Each algorithm lives in its own self-contained module with a consistent
internal structure: model definition, optimizer, loss function,
preprocessing, metrics, and utilities are always separated into distinct
components.

This mirrors how production ML systems are organized — not because this
project needs that scale, but because the habit of separating concerns is
worth building from the start. It also means each module can be studied,
modified, or extended independently without touching anything else in the
repository.

Notebooks serve as walkthrough layers only. All analytical logic lives
in `src/`.

---

## Repository Structure

```
ml-from-scratch-numpy/
├── eda-engine/                       # Exploratory Data Analysis pipeline
├── linear-regression/                # Multivariate Linear Regression
├── logistic-regression/              # Binary Logistic Regression
├── regularized-logistic-regression/  # L1 & L2 Regularization
├── sgd-logistic-regression/          # Batch vs SGD vs Mini-batch
├── neural-network/                   # Two-layer feedforward network
└── .gitignore
```

Each module is self-contained and can be studied independently. Start
with `eda-engine` if approaching the repository sequentially.

---

## Modules

### EDA Engine
Modular exploratory data analysis pipeline built on the Red Wine Quality
dataset. Separates statistical analysis, visualization, and correlation
logic into reusable components rather than keeping everything inside a
notebook. Establishes the analytical foundation that informs preprocessing
and feature engineering decisions in subsequent modules.

→ [View Module](./eda-engine/README.md)

---

### Linear Regression
Multivariate Linear Regression implemented from scratch using batch
gradient descent on the Red Wine Quality dataset. Covers analytical
gradient derivation, feature standardization, convergence analysis across
learning rates, and full validation against scikit-learn. Treats
regression not as a function call but as an optimization problem solved
step by step.

→ [View Module](./linear-regression/README.md)

---

### Logistic Regression
Binary Logistic Regression built from first principles on the Breast
Cancer Wisconsin dataset. Covers sigmoid activation, cross-entropy loss
derivation, numerical stability handling, threshold analysis, and
validation against scikit-learn. Includes dedicated experiment notebooks
for learning rate and threshold sensitivity analysis.

→ [View Module](./logistic-regression/README.md)

---

### Regularized Logistic Regression
Extends the logistic regression implementation with L1 and L2
regularization. Covers penalty term derivation, gradient modification,
weight shrinkage behavior, and L1-induced sparsity. Includes a controlled
lambda sweep across seven values with full metrics — train accuracy, val
accuracy, weight norm, and train loss — recorded to CSV and visualized.

→ [View Module](./regularized-logistic-regression/README.md)

---

### SGD — Batch vs Mini-batch vs Stochastic
Isolates and compares three gradient descent strategies on the same model,
dataset, and initialization. Only the optimizer changes across runs —
everything else is held constant. Covers convergence speed, loss
trajectory characteristics, noise and variance in updates, and batch size
impact. Results are logged to CSV and visualized across controlled
experiments.

→ [View Module](./sgd-logistic-regression/README.md)

---

### Neural Network
Two-layer feedforward network (ReLU hidden → Sigmoid output) implemented
from scratch. Covers forward propagation, manual backpropagation, gradient
flow through layers, and the mathematical simplification that makes the
output gradient clean. Validated against scikit-learn on a classification
task, and tested on a nonlinear dataset to demonstrate what a hidden layer
actually enables over logistic regression.

→ [View Module](./neural-network/README.md)

---

## Datasets

| Module | Dataset | Source |
|---|---|---|
| EDA Engine | Red Wine Quality | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| Linear Regression | Red Wine Quality | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| Logistic Regression | Breast Cancer Wisconsin | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) |
| Regularized Logistic Regression | Synthetic (make_classification) | sklearn.datasets |
| SGD | Synthetic (make_classification) | sklearn.datasets |
| Neural Network | Synthetic (make_classification, make_moons) | sklearn.datasets |

---

## Setup

Clone the repository:

```bash
git clone https://github.com/bitbyrizbit/ml-from-scratch-numpy.git
cd ml-from-scratch-numpy
```

Navigate to any module and follow its README for environment setup and
run instructions. Each module contains its own `requirements.txt`.

---

## Roadmap

Modules are sequenced intentionally — each builds on concepts introduced
in the previous one.

- [x] EDA Engine — analytical foundation, feature inspection, correlation analysis
- [x] Linear Regression — gradient descent, MSE optimization, convergence behavior
- [x] Logistic Regression — probabilistic classification, cross-entropy, decision calibration
- [x] Regularized Logistic Regression — L1 and L2 penalties, sparsity, weight shrinkage, lambda sweep
- [x] SGD — batch vs mini-batch vs stochastic, convergence profiles, noise analysis
- [x] Neural Network — backpropagation, nonlinear boundaries, hidden layer mechanics
