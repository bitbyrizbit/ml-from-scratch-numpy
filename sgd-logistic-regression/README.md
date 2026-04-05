# **Stochastic Gradient Descent vs Batch Optimization — From Scratch (Logistic Regression)**

This project implements Logistic Regression entirely from scratch using NumPy, with a focused investigation into **how different gradient descent strategies behave in practice**.

Rather than treating optimization as a black box, this work isolates and compares:

* **Batch Gradient Descent**
* **Stochastic Gradient Descent (SGD)**
* **Mini-batch Gradient Descent**

on the *same model, same dataset, same initialization*.

The goal is not just correctness—but understanding **training dynamics, variance, convergence behavior, and real-world trade-offs**.

---

## Motivation

Gradient descent is often taught as a single algorithm.

In practice, it is a **family of strategies** with fundamentally different behaviors.

Most implementations hide this behind APIs:

```python
model.fit(...)
```

This project reconstructs the full pipeline and asks:

> *How does the choice of optimization strategy change the way a model learns—even for a convex problem?*

---

## What This Project Demonstrates

* Logistic Regression implemented from first principles (NumPy)
* Clean separation between **model and optimizer design**
* Batch, SGD, and Mini-batch implementations with identical interfaces
* Controlled experiments isolating optimizer behavior
* Analysis of:

  * Convergence speed
  * Loss trajectory characteristics
  * Noise and variance in updates
  * Batch size impact
* Reproducible experiment tracking and logging

---

## Core Idea

All experiments follow a strict rule:

> **Only the optimizer changes. Everything else remains constant.**

* Same dataset
* Same initialization
* Same learning rate (unless explicitly varied)

This ensures differences arise purely from **optimization dynamics**, not confounding factors.

---

## Dataset

Synthetic classification dataset:

* 500 samples
* 20 features
* Standardized before training

This setup provides:

* Enough complexity to observe meaningful behavior
* Fast iteration for controlled experimentation
* A well-conditioned convex problem


Generated using `sklearn.datasets.make_classification` with a fixed random 
seed for full reproducibility across all experiments.

---

## Optimization Strategies Implemented

### 1. Batch Gradient Descent

* Uses entire dataset per update
* Low-variance gradients
* Smooth and stable convergence

---

### 2. Stochastic Gradient Descent (SGD)

* Updates using a single sample
* High-variance gradient estimates
* Noisy but frequent updates

---

### 3. Mini-batch Gradient Descent

* Uses small batches (16 / 32 / 64)
* Trade-off between stability and speed
* Industry-standard approach

---

## Experimental Design

The project is structured around **controlled experiments**:

### 1. Loss Trajectory Comparison

* Raw loss curves (noise visualization)
* Smoothed curves (fair comparison)

### 2. Convergence Speed

* Measured as:

  > *iterations required to reach a fixed loss threshold*

### 3. Noise Analysis

* SGD loss plotted at update level
* Highlights high-frequency oscillations

### 4. Batch Size Impact

* Batch sizes: `[1, 16, 32, 64, full]`
* Observes transition from SGD → Batch GD

### 5. Reproducibility

* Fixed random seed
* Identical initialization across runs

---

## Key Results

From `metrics.csv`:

```
Batch GD           - 50 iterations
SGD                - 99 iterations
MiniBatch (16)     - 295 iterations
MiniBatch (32)     - 320 iterations
MiniBatch (64)     - 160 iterations
```

---

## Observations & Insights

### 1. Batch GD Converged Fastest

Contrary to common intuition, **Batch Gradient Descent reached the threshold fastest** in this setup.

Reason:

* Low-variance, stable updates
* Well-conditioned convex problem

---

### 2. SGD Exhibits High Variance

* Loss curve shows strong oscillations
* Large spikes observed in early training
* Does not decrease monotonically

Yet:

* Still trends downward in expectation

> SGD uses *noisy but unbiased* gradient estimates

---

### 3. Learning Rate Sensitivity (Critical)

Reducing learning rate from `0.01 -> 0.005`:

* Reduced variance significantly
* Improved stability
* Preserved stochastic behavior

> SGD performance is highly dependent on learning rate tuning

---

### 4. Mini-batch Did Not Always Win

Mini-batch methods:

* Reduced noise compared to SGD
* But did **not outperform Batch GD** in convergence speed

This highlights:

> The commonly cited efficiency of mini-batch methods is **context-dependent**

---

### 5. Visualization Matters

Raw loss curves were misleading due to:

* Different update frequencies
* Scale imbalance (SGD vs Batch)

To address this:

* Smoothed curves were used
* Zoomed comparisons were introduced

> Proper visualization is essential for correct interpretation

---

## Core Takeaway

> **Optimization strategy is not universally optimal—even for convex problems.**

Performance depends on:

* Learning rate
* Batch size
* Problem conditioning
* Convergence criteria

In stable, well-conditioned settings:

> **Low-variance updates (Batch GD) can outperform stochastic methods.**

---

## Engineering Design

### Clean Architecture

```
model.py        - pure mathematical model
optimizers/     - training strategies
utils/          - batching, data, reproducibility
```

---

### Optimizer Abstraction

All optimizers follow a unified interface:

```python
optimizer.train(model, X, y)
```

This enables:

* Clean experiments
* Easy extension
* Consistent comparisons

---

### Numerical Stability

* Clipping in sigmoid and log loss
* Fully vectorized operations
* No loops over features

---

### Reproducibility

* Fixed random seeds
* Deterministic dataset generation
* Logged experiment results

---

## Project Structure

```
sgd-logistic-regression/
│
├── src/
│   ├── model.py
│   ├── metrics.py
│   ├── optimizers/
│   └── utils/
│
├── notebooks/
│   ├── 01_batch_vs_sgd.ipynb
│   ├── 02_minibatch_analysis.ipynb
│   └── 03_convergence_visualization.ipynb
│
├── experiments/
│   ├── loss_curves_comparison.png
│   ├── loss_comparison_smoothed.png
│   ├── loss_zoomed.png
│   ├── minibatch_batchsize_impact.png
│   ├── convergence_speed.png
│   └── noise_visualization.png
│
├── outputs/
│   └── metrics.csv
│
├── logs/
│   └── experiment_log.csv
│
├── requirements.txt
└── README.md
```

---

## How to Run

Clone the repo:
```bash
cd sgd-logistic-regression-from-scratch
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order:

1. `01_batch_vs_sgd.ipynb`
2. `02_minibatch_analysis.ipynb`
3. `03_convergence_visualization.ipynb`

---

## Closing Note

This project is not about implementing logistic regression.

It is about understanding:

* How optimization actually behaves
* Why training dynamics matter
* Why “faster” algorithms are not universally better

In practice, performance emerges from the interaction of:

> **algorithm × hyperparameters × data**

Understanding that interaction is what separates implementation from engineering.

