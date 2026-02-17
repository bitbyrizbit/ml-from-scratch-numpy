# **Linear Regression From Scratch**

## Project Overview

This project rebuilds Multivariate Linear Regression entirely from first principles using NumPy.

No scikit-learn is used for training.

The objective is not merely to fit a model, but to reconstruct the entire learning pipeline - from analytical gradient derivation and batch gradient descent optimization to numerical stability considerations and validation against a trusted implementation. The dataset used is the Red Wine Quality dataset, where physicochemical properties are used to predict quality scores.

This repository focuses on mathematical transparency and engineering structure rather than black-box usage.

---

## Problem Formulation

Given a feature matrix
`X ∈ ℝⁿˣᵈ`

and a target vector
`y ∈ ℝⁿ`

we aim to learn parameters:
`w ∈ ℝᵈ`
`b ∈ ℝ`

such that:
`ŷ = Xw + b`

**The optimization objective is Mean Squared Error:**
`L(w, b) = (1/n) Σ (yᵢ − ŷᵢ)²`

Since this objective is convex, gradient descent is guaranteed to converge to the global minimum for a suitable learning rate.

---

## Dataset

The dataset consists of physicochemical measurements of red wine samples collected for quality assessment. Features include:

* Fixed acidity
* Volatile acidity
* Citric acid
* Residual sugar
* Chlorides
* Free sulfur dioxide
* Total sulfur dioxide
* Density
* pH
* Sulphates
* Alcohol

Target:
* Quality score (int rating)


## Architecture

The implementation follows a modular design resembling production ML systems:

```
src/
├── model.py          # Linear regression model
├── optimizer.py      # Batch gradient descent
├── loss.py           # Loss functions
├── metrics.py        # Evaluation metrics
├── preprocessing.py  # Feature scaling
└── utils.py          # Train-test split
```

### Design Decisions

The architecture separates the model definition from the optimization logic to preserve clarity and extensibility. Loss computation is isolated from training to ensure modularity, while preprocessing is carefully constrained to training data to prevent leakage.


---

## Optimization Strategy

Batch Gradient Descent is implemented manually.

Gradients:
`∂L/∂w = (2/n) Xᵀ(ŷ − y)`
`∂L/∂b = (2/n) Σ(ŷ − y)`

Parameter update:
`w ← w − α ∂L/∂w`
`b ← b − α ∂L/∂b`

Where `α` is the learning rate.

Loss is tracked per iteration to analyze convergence behavior.


## Preprocessing

Feature standardization is applied:

`X_scaled = (X − μ) / σ`

**Why:**
* Prevents scale-dominated gradients
* Improves convergence speed
* Enhances numerical stability

The scaler is fit only on training data to avoid data leakage.

---

## Evaluation

Model performance is assessed using:

* Mean Squared Error (MSE)
* R² Score
* Train vs Test comparison
* Residual analysis

Residual plots are used to verify linear assumptions and inspect bias patterns.

Implementation correctness is validated by comparing results with `scikit-learn`’s LinearRegression.

Alignment confirms gradient and update accuracy.

--- 

## Experimental Insights

Training dynamics revealed the importance of proper feature scaling. Without normalization, gradient updates were unstable and loss reduction was inconsistent due to features operating on different numerical ranges. After standardization, convergence became smooth and significantly faster, confirming the sensitivity of gradient descent to feature scale.

Learning rate selection proved critical. Higher values of α accelerated initial loss reduction but risked oscillation or divergence, while smaller values ensured stability at the cost of slower convergence. Empirical tuning demonstrated the expected trade-off between speed and stability, reinforcing theoretical expectations from convex optimization.

Residual analysis indicated that errors were reasonably symmetrically distributed around zero, suggesting that the linear assumption is appropriate for the dataset at this scale. No extreme bias patterns were observed, though minor variance differences across quality levels suggest that nonlinear extensions could potentially improve performance.

---


## Setup

Clone the repository:

```bash
cd linear-regression-from-scratch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
cd notebooks 
jupyter notebook linear_regression_walkthrough.ipynb
```

---

## Future Extensions

* Add L2 regularization (Ridge)
* Implement Stochastic Gradient Descent
* Add early stopping
* Implement closed-form normal equation
* Extend to Polynomial Regression
* Compare convergence profiles

