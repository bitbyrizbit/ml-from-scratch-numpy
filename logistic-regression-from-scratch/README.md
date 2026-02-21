# **Logistic Regression From Scratch (NumPy)**

This repository contains a from-scratch implementation of Binary Logistic Regression built entirely from first principles using NumPy. No high-level machine learning libraries are used for training - only for final validation.

The goal of this project was not merely to reproduce a standard algorithm, but to understand in detail how probabilistic classification models behave under optimization, how their gradients simplify mathematically, and how engineering considerations such as numerical stability influence even simple models.

---

## Motivation

Logistic Regression is often introduced as a “simple classification algorithm.”  
In reality, it sits at a beautiful intersection of linear algebra, probability theory, and convex optimization.

Rather than calling a library function, this implementation reconstructs the algorithm step by step:

- Implementing the hypothesis and gradient equations manually
- Implementing cross-entropy loss manually
- Computing gradients explicitly
- Validating convergence behavior experimentally

Only after completing the implementation was it compared against scikit-learn to verify correctness.

---

## What This Project Demonstrates

- Manual implementation of logistic regression using NumPy
- Vectorized gradient descent optimization
- Numerical stability handling
- Experimental analysis of learning rates and thresholds
- Validation against a production-grade ML library

---

## Dataset

Experiments were conducted using the Breast Cancer Wisconsin dataset, a well-known binary classification dataset commonly used for evaluating machine learning algorithms. The dataset consists of 569 samples with 30 numerical features computed from digitized images of fine needle aspirates of breast masses. 

The dataset was split into training and testing subsets to evaluate generalization performance. Prior to training, features were standardized to zero mean and unit variance. This preprocessing step is essential for gradient-based optimization, as it ensures that all features contribute proportionally to parameter updates and improves convergence stability.

The choice of this dataset allows the implementation to be validated on a real-world medical classification task while keeping the focus on optimization behavior and correctness of the underlying algorithm.

--- 

## Mathematical Formulation

The model estimates the probability that a sample belongs to class 1 using:
**h(x) = σ(wᵀx + b)**

where the sigmoid function
**σ(z) = 1 / (1 + e⁻ᶻ)**

maps real-valued inputs into the interval (0, 1).

Logistic regression therefore produces linear decision boundaries, even though its output is probabilistic.



## Why Cross-Entropy and Not Mean Squared Error?

A common mistake is to use Mean Squared Error for classification.  
While technically possible, it leads to poor optimization behavior when paired with a sigmoid output.

When predictions saturate (very close to 0 or 1), MSE produces extremely small gradients, slowing learning significantly - even when the prediction is confidently wrong.

Cross-entropy avoids this problem. Its gradient remains strong for misclassified samples, leading to faster and more stable convergence.

This is not just a mathematical preference - it is an optimization necessity.

---

## Convexity and Convergence

One of the most important theoretical properties of logistic regression is that the cross-entropy loss is convex.

This guarantees that:

- There are no local minima
- Gradient descent converges to a global minimum (with an appropriate learning rate)

Understanding this property changes how one thinks about optimization - convergence behavior becomes predictable rather than mysterious.

---

## Gradient Simplification

A particularly elegant result appears during derivation.

When sigmoid activation is paired with cross-entropy loss, the gradient simplifies to:

**∂L/∂w = (1/m) Xᵀ(ŷ − y)**
**∂L/∂b = (1/m) Σ(ŷ − y)**
  
This simplification is one of the reasons logistic regression remains such an instructive model in machine learning education.

---

## Engineering Decisions

While the mathematics is clean, practical implementation requires care.

Large values passed into the exponential function can cause overflow. To address this, inputs to the sigmoid function are clipped to maintain numerical stability.

Similarly, predicted probabilities are clipped before applying logarithms in the loss function to avoid undefined log(0) behavior.

All computations are fully vectorized using matrix operations. This ensures both computational efficiency and conceptual alignment with how professional ML libraries are implemented internally.


### Reproducibility
The main training workflow can be reproduced entirely from the walkthrough notebook. Experiments are separated into dedicated notebooks to maintain clarity between implementation and analysis.

---

## Experimental Observations

During experimentation, varying the learning rate revealed expected but instructive behavior. A high learning rate caused oscillations in the loss curve, while an excessively small learning rate slowed convergence dramatically. A moderate learning rate produced stable convergence - visually reinforcing the theory of gradient-based optimization.

Different classification thresholds were tested to observe their effect on model predictions. Because logistic regression outputs probabilities, classification depends on a chosen threshold. Lower thresholds increased recall, while higher thresholds increased precision. This demonstrated that classification performance is influenced not only by learned parameters but also by decision calibration.

---

## Validation

Model performance was evaluated using accuracy and compared against scikit-learn's implementation.

After implementation, the learned parameters and predictions were compared against:

`sklearn.linear_model.LogisticRegression`

The results closely matched, confirming the correctness of gradient computation and optimization logic.

---


## Project Structure

```
logistic-regression-from-scratch/
│
├── src/
│   ├── loss.py
│   ├── metrics.py
│   ├── optimizer.py
│   ├── plotting.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── model.py
│
├── notebooks/
│   └── logistic_regression_walkthrough.ipynb
│
├── experiments/
│   ├── threshold_analysis.ipynb
│   └── learning_rate_analysis.ipynb
│
├── outputs/
│   ├── logs/
│   └── plots/
|
├── requirements.txt
└── README.md
```

---

## How to Run

Clone the repository

```bash
cd logistic-regression-from-scratch
pip install -r requirements.txt
jupyter notebook
```

---

## Closing Notes

This project was built as an exercise in understanding rather than abstraction. Implementing logistic regression from scratch highlights how much of modern machine learning relies on relatively simple mathematical structures executed with careful numerical engineering.

While production systems rely on optimized libraries, building the model manually clarifies what those abstractions are actually doing under the hood - and why they work.