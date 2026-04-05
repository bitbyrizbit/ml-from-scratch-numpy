![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

# ml-from-scratch-numpy

NumPy implementations of core supervised learning algorithms built entirely from first principles — no scikit-learn for training, no black-box abstractions. Every gradient is derived analytically, every parameter update is implemented manually, and every result is validated against a production-grade library.

---

## TL;DR

Three self-contained ML modules — EDA, Linear Regression, and Logistic Regression — each built from mathematical definitions using only NumPy. Modular architecture, explicit gradient derivation, convergence analysis, and sklearn validation throughout. Built to understand what happens under the hood, not just to get predictions out.

---

## Motivation

Most ML education skips the internals. You call `.fit()`, get predictions, and move on.

This repository takes the opposite approach. Every algorithm is reconstructed from its mathematical foundation — gradients derived by hand, optimization implemented explicitly, and convergence behavior analyzed experimentally.

The goal is not to outperform scikit-learn. The goal is to understand exactly what scikit-learn is doing, and why it works.

---

## Architecture Decision

Each algorithm lives in its own self-contained module with a consistent internal structure: model definition, optimizer, loss function, preprocessing, metrics, and utilities are always separated into distinct components.

This mirrors how production ML systems are organized — not because this project needs that scale, but because the habit of separating concerns is worth building from the start. It also means each module can be studied, modified, or extended independently without touching anything else in the repository.

Notebooks serve as walkthrough layers only. All analytical logic lives in `src/`.

---

## Repository Structure

```
ml-from-scratch-numpy/
├── eda-engine/                  # Exploratory Data Analysis pipeline
├── linear-regression/           # Multivariate Linear Regression
├── logistic-regression/         # Binary Logistic Regression
└── .gitignore
```

Each module is self-contained and can be studied independently. Start with `eda-engine` if approaching the repository sequentially.

---

## Modules

### EDA Engine
Modular exploratory data analysis pipeline built on the Red Wine Quality dataset. Separates statistical analysis, visualization, and correlation logic into reusable components rather than keeping everything inside a notebook. Establishes the analytical foundation that informs preprocessing and feature engineering decisions in subsequent modules.

→ [View Module](./eda-engine/README.md)

---

### Linear Regression
Multivariate Linear Regression implemented from scratch using batch gradient descent on the Red Wine Quality dataset. Covers analytical gradient derivation, feature standardization, convergence analysis across learning rates, and full validation against scikit-learn. Treats regression not as a function call but as an optimization problem solved step by step.

→ [View Module](./linear-regression/README.md)

---

### Logistic Regression
Binary Logistic Regression built from first principles on the Breast Cancer Wisconsin dataset. Covers sigmoid activation, cross-entropy loss derivation, numerical stability handling, threshold analysis, and validation against scikit-learn. Includes dedicated experiment notebooks for learning rate and threshold sensitivity analysis.

→ [View Module](./logistic-regression/README.md)

---

## Datasets

| Module | Dataset | Source |
|---|---|---|
| EDA Engine | Red Wine Quality | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| Linear Regression | Red Wine Quality | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| Logistic Regression | Breast Cancer Wisconsin | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) |

---

## Setup

Clone the repository:

```bash
git clone https://github.com/bitbyrizbit/ml-from-scratch-numpy.git
cd ml-from-scratch-numpy
```

Navigate to any module and follow its README for environment setup and run instructions.

---

## Roadmap

Modules are sequenced intentionally — each builds on concepts introduced in the previous one.

- [x] EDA Engine — analytical foundation, feature inspection, correlation analysis
- [x] Linear Regression — gradient descent, MSE optimization, convergence behavior
- [x] Logistic Regression — probabilistic classification, cross-entropy, decision calibration
- [ ] Ridge Regression — L2 regularization, bias-variance tradeoff
- [x] Stochastic Gradient Descent — mini-batch optimization, convergence comparison
- [ ] Neural Network — extending logistic regression to multiple layers from scratch
