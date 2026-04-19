# Neural-Network

A two-layer neural network implemented from scratch using NumPy.
No machine learning libraries used for training — only for dataset generation and final validation.

The focus is on understanding how forward propagation, backpropagation, and gradient descent interact - implemented explicitly, not abstracted away behind an API.

---

## TL;DR

Two-layer feedforward network (ReLU hidden → Sigmoid output) built from first principles. Three notebooks cover structural validation, backpropagation correctness, and nonlinear decision boundary visualization. Validated against scikit-learn. Extends directly from the logistic regression module - the hidden layer is the only architectural
addition, and it is what enables nonlinear learning.

---

## Motivation

A neural network is often introduced as a stack of matrix multiplications and activation functions. That description is accurate but skips the part that matters — how does the error signal flow backward through those layers, and why does that cause the network to learn?

This project implements that process by hand. Every gradient is derived and applied explicitly. The goal is to understand what a hidden layer actually does, not just how to call `.fit()`.

---

## Architecture

```
src/
├── model.py        # NeuralNetwork — forward pass, backprop, training loop
├── layers.py       # DenseLayer — linear transform, stores input/Z for backprop
├── activations.py  # sigmoid, relu and their derivatives
├── losses.py       # BinaryCrossEntropy
├── metrics.py      # accuracy
└── __init__.py
```

Model and layers are separated deliberately. `DenseLayer` owns the linear transformation and its gradients. `NeuralNetwork` owns the activation functions, the loss, and the update logic. Each layer stores its input and pre-activation output during the forward pass — this is what makes backpropagation possible without recomputing intermediate values.

---

## Mathematical Formulation

**Forward Pass**

```
Z1 = X · W1 + b1        hidden pre-activation
A1 = ReLU(Z1)            hidden activation
Z2 = A1 · W2 + b2       output pre-activation
A2 = sigmoid(Z2)         output probability
```

**Loss — Binary Cross-Entropy**

`L = -(1/m) Σ [ y log(A2) + (1-y) log(1-A2) ]`

**Backpropagation**

```
dZ2 = A2 - y                    simplified (sigmoid + BCE cancellation)
dW2 = (1/m) A1ᵀ · dZ2
dA1 = dZ2 · W2ᵀ
dZ1 = dA1 * ReLU'(Z1)
dW1 = (1/m) Xᵀ · dZ1
```

`dZ2 = A2 - y` is the result of the sigmoid and cross-entropy gradients cancelling — the same simplification derived in the logistic regression module, now applied to the output layer of a deeper network.

---

## Notebooks

### `01_forward_pass_validation.ipynb` — Structural Sanity Check

Verifies the network is correctly wired before any training happens.

| Check | Expected | Result |
|---|---|---|
| A1 shape | (5, 4) | (5, 4) |
| A2 shape | (5, 1) | (5, 1) |
| Min probability | > 0 | ~0.5000 |
| Max probability | < 1 | ~0.5000 |
| Loss after 10 epochs | Decreasing | |

Initial probabilities cluster near 0.5 — expected with small random
weight initialization. The model has no signal yet and produces
near-uniform predictions. Loss decreases monotonically over 10
iterations, confirming gradients are flowing correctly end to end.

---

### `02_backpropagation_debugging.ipynb` — Correctness Validation

Trains on a real 20-feature classification dataset and validates against scikit-learn's LogisticRegression.

**Dataset:** 500 samples, 20 features (10 informative), 80/20 split, standardized. `random_state=42`.

| Model | Test Accuracy |
|---|---|
| Neural Network | 0.92 |
| Sklearn LogisticRegression | 0.85 |

Training loss: 0.6931 -> 0.1238 over 1000 epochs. Smooth monotonic descent — no oscillation, no plateaus.

The network outperforms logistic regression, confirming that backpropagation is computing gradients correctly and parameter updates are working as expected.

Results saved to `outputs/metrics.csv`.

---

### `03_nonlinear_decision_boundary.ipynb` — Why Hidden Layers Matter

Demonstrates the core capability that separates a neural network from logistic regression — nonlinear decision boundaries.

**Dataset:** `make_moons`, 500 samples, `noise=0.1` — two interleaving crescents that are not linearly separable. No train-test split; goal is visualization.

| Model | Accuracy |
|---|---|
| Neural Network | 0.8920 |
| Logistic Regression | 0.8720 |

NN training loss: 0.6932 -> 0.2098 over 10000 epochs.

Logistic regression produces a straight boundary — it cannot follow the crescent shape of the data regardless of how long it trains. The neural network learns a curved boundary that wraps around each class. The boundary plots show this directly: the NN probability surface bends where the LR surface stays flat.

This is what the hidden layer enables. Without it, the model is logistic regression.

---

## Key Findings

**Gradient simplification**
`dZ2 = A2 - y` is not a shortcut — it is the mathematically correct result of the sigmoid and cross-entropy derivatives cancelling. It keeps the output layer gradient clean and numerically stable.

**Hidden layers enable nonlinearity**
Logistic regression and a neural network with no hidden layer are the same model. The ReLU hidden layer allows the network to compose linear functions into a nonlinear one. NB03 makes this visible directly in the boundary plots — the NN boundary curves, the LR boundary cannot.

**Learning rate matters**
NB02 uses `lr=0.1`. The same architecture with `lr=0.01` barely moves — loss stays near 0.69 and the model fails to learn. The network configuration didn't change. The learning rate did.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/bitbyrizbit/ml-from-scratch-numpy.git
cd neural-network
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order:

```bash
jupyter notebook
```

1. `01_forward_pass_validation.ipynb` — structural check before training
2. `02_backpropagation_debugging.ipynb` — correctness validation against sklearn
3. `03_nonlinear_decision_boundary.ipynb` — nonlinear boundary visualization

---

## Closing Note

This project is where the shift from “models” to “learning systems” begins.

Logistic regression showed how optimization works. This neural network extends that idea into layered computation — where each layer transforms the representation before passing it forward. Backpropagation is not just an algorithm here; it is the mechanism that connects all layers through a shared objective.

By implementing every step manually, the focus stays on how information flows — forward as predictions, backward as gradients. The hidden layer is not just an addition to the architecture; it is what allows the model to reshape the input space into something separable.

Understanding this flow is what makes deeper models intuitive rather than abstract.

---

## One-Line Takeaway

A neural network is not a black box — it is a structured composition of simple functions, trained end-to-end through the chain rule.