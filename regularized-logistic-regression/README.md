# Regularized-Logistic-Regression

A from-scratch implementation of Logistic Regression with L1 and L2 regularization, built entirely using NumPy. No scikit-learn for training - only for validation and dataset generation.

The focus is not just correctness, but understanding how regularization changes optimization dynamics, weight behavior, and generalization - experimentally, not theoretically.

---

## TL;DR

Binary Logistic Regression with L1 and L2 regularization implemented from first principles. Three notebooks cover model validation, decision boundary visualization, and controlled lambda sweep experiments. Results validated against scikit-learn. Built to study what regularization actually does to weights, loss, and generalization.

---

## Motivation

Regularization is usually introduced as "add a penalty term." 
That explanation skips the interesting parts - how does it change the loss surface? How does it interact with gradient descent? Does more regularization always help?

This project answers those questions experimentally. The implementation is kept minimal so the behavior of regularization is the focus, not the engineering around it.

---

## Problem Setup

Binary classification using synthetic datasets generated via `sklearn.datasets.make_classification`. All features standardized before training. Scaler fit on training data only - no leakage.

| Notebook | Samples | Features | Split |
|---|---|---|---|
| 01 — Model Validation | 500 | 20 (10 informative) | 80/20 train-test |
| 02 — Decision Boundary | 300 | 2 (for visualization) | No split |
| 03 — Lambda Experiments | 500 | 200 (20 informative) | 80/20 train-val |

All experiments use `random_state=42` and `np.random.seed(42)`.

---

## Architecture

```
src/
├── model.py           # Forward pass, sigmoid, prediction
├── optimizer.py       # Full-batch gradient descent training loop
├── losses.py          # Binary cross-entropy loss
├── regularization.py  # L1 and L2 penalty terms and gradients
├── metrics.py         # Accuracy, precision, recall
└── __init__.py
```

Model and optimizer are deliberately separated. The model handles parameters and forward pass only - training logic lives entirely in the optimizer. Regularization is isolated into its own class so penalty and gradient logic are never mixed into the training loop directly.

---

## Mathematical Formulation

**Hypothesis**

`h(x) = σ(wᵀx + b)`  where  `σ(z) = 1 / (1 + e⁻ᶻ)`

**Loss — Binary Cross-Entropy**

`L = -(1/m) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]`

**L2 Regularization**

`L_total = L + (λ/2) ||w||²`  →  gradient:  `λw`

**L1 Regularization**

`L_total = L + λ ||w||₁`  →  gradient:  `λ sign(w)`

Bias is excluded from regularization in both cases.

---

## Notebooks

### `01_model_validation.ipynb` — Correctness Check

Validates the from-scratch implementation against scikit-learn on a 20-feature dataset.

| Model | Test Accuracy |
|---|---|
| Ours — No Regularization | 0.84 |
| Sklearn — No Regularization | 0.82 |
| Ours — L2 (λ = 0.1) | 0.83 |
| Sklearn — L2 (λ = 0.1) | 0.82 |

Loss curves converge monotonically for both variants, confirming stable optimization and correct gradient computation.

---

### `02_decision_boundary_analysis.ipynb` — Visual Effect of Regularization

Uses a 2D dataset to visualize how L1 and L2 change the decision boundary under identical conditions.

| Model | Weight Norm \|\|w\|\| |
|---|---|
| No Regularization | 2.56 |
| L2 (λ = 0.1) | 1.39 |
| L1 (λ = 0.1) | 1.61 |

L2 shrinks both weights proportionally. L1 drives `w[0]` to near zero (-0.0008), producing a near-horizontal boundary - feature suppression visible directly in 2D.

---

### `03_regularization_experiments.ipynb` — Lambda Sweep

Sweeps λ across seven values on a 200-feature dataset. Three controlled experiments: accuracy vs λ, weight norm vs λ, and L1 sparsity vs λ.

**Results — `outputs/metrics.csv`**

| λ | Train Acc | Val Acc | Weight Norm | Train Loss |
|---|---|---|---|---|
| 0.000 | 0.9225 | 0.68 | 2.309 | 0.2415 |
| 0.001 | 0.9225 | 0.68 | 2.289 | 0.2453 |
| 0.010 | 0.9225 | 0.68 | 2.125 | 0.2752 |
| 0.100 | 0.9000 | 0.68 | 1.259 | 0.4118 |
| 1.000 | 0.8375 | 0.68 | 0.357 | 0.5925 |
| 10.00 | 0.8050 | 0.63 | 0.054 | 0.6773 |
| 100.0 | 0.5125 | 0.48 | 0.006 | 0.6914 |

---

## Key Findings

**L2 Weight Shrinkage**
Weight norm decreases monotonically from 2.31 at λ=0 to 0.006 at λ=100.
L2 penalizes large coefficients smoothly — no weight reaches exactly zero.

**Bias-Variance Tradeoff**
Training accuracy drops consistently with λ. Validation accuracy holds at 0.68 through λ=1, then falls to 0.63 at λ=10 and 0.48 at λ=100. 
The model begins underfitting beyond λ=10 on this dataset.

**L1 Sparsity**
At λ = 0.1, 178 out of 200 weights fall below the near-zero threshold. At higher λ, sparsity does not increase monotonically in this setup - full-batch gradient descent with subgradient updates can cause weights to oscillate around zero instead of settling exactly at zero.

**Train vs Validation Loss**
Train loss rises steadily with λ. Validation loss initially decreases as regularization reduces overfitting, then rises again as the model becomes too constrained. The crossover is visible in the experiment plot.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/bitbyrizbit/ml-from-scratch-numpy.git
cd regularized-logistic-regression
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order:

```bash
jupyter notebook
```

1. `01_model_validation.ipynb` — correctness check against sklearn
2. `02_decision_boundary_analysis.ipynb` — visual effect of L1 vs L2
3. `03_regularization_experiments.ipynb` — lambda sweep, sparsity, weight norm

---

## Closing Note

This project moves beyond implementing logistic regression and into understanding how models are controlled.

Regularization is often treated as a hyperparameter to tune. In practice, it is a direct mechanism for shaping model behavior — influencing not just performance, but how a model distributes importance across features and responds to noise.

By isolating L1 and L2 under controlled conditions, this work shows that different penalties do not simply "improve generalization" - they impose fundamentally different constraints on the solution space.

Understanding these constraints is what allows models to be designed, not just trained.

---

## One-Line Takeaway

Regularization is not a fix for overfitting — it is a mechanism for shaping the solution space and controlling how a model learns.