# **EDA Engine – Red Wine Quality Dataset**

## **Overview**

The EDA Engine is a structured Exploratory Data Analysis module built as the foundational component of the `ml-from-scratch-numpy` repository.

Rather than performing exploratory analysis entirely inside a notebook, this module separates analytical logic into reusable Python components. The objective is to approach EDA not as a one-time activity, but as an engineered, modular process that supports future machine learning implementations.

The dataset used in this module is the Red Wine Quality dataset, a real-world regression problem involving physicochemical properties of wine samples.

---

## **Problem Statement**

The dataset contains multiple chemical attributes of red wine samples, such as:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Sulphates
- Alcohol
- pH
- Density

Each sample is assigned a quality score.

The learning objective is to model wine quality as a function of its physicochemical properties. This is treated as a supervised regression problem.

Future modules in the repository will implement regression algorithms from scratch using NumPy to predict wine quality.

---

## **Why This Dataset?**

The Red Wine Quality dataset was selected because:

- All features are numerical, making it ideal for algorithm implementation from first principles.
- It represents a realistic regression task.
- It contains meaningful correlations between chemical properties and quality.
- It allows exploration of feature scaling, multicollinearity, and outlier impact.
- It is widely recognized and benchmarked in machine learning literature.

This makes it well-suited for studying how gradient-based learning behaves on structured numerical data.

---

## **Engineering Approach**

Instead of keeping all logic inside a Jupyter notebook, the EDA process is modularized into reusable components.

This improves:

- Code clarity
- Reproducibility
- Separation of concerns
- Maintainability
- Scalability for future datasets

The notebook serves as a walkthrough layer, while the `src/` directory contains structured analytical logic.

---

## **Project Structure**

```
eda_engine/
├── data/
│   └── winequality-red.csv        # Raw dataset
│
├── notebooks/
│   └── eda_walkthrough.ipynb      # Interactive exploration
│
├── src/
│   ├── loader.py                  # Dataset loading utilities
│   ├── statistics.py              # Statistical summary functions
│   ├── visualization.py           # Plotting utilities
│   ├── correlation.py             # Correlation analysis tools
│   ├── analysis.py                # Orchestrates EDA workflow
│   └── __init__.py
│
└── README.md
```

---

## **Analytical Components**

### Data Loading (`loader.py`)
Responsible for reading and validating dataset structure.

### Statistical Analysis (`statistics.py`)
Implements:
- Descriptive statistics
- Measures of central tendency
- Dispersion metrics

### Visualization (`visualization.py`)
Generates:
- Histograms
- Boxplots
- Distribution plots
- Relationship visualizations

### Correlation Analysis (`correlation.py`)
- Correlation matrix computation
- Feature-to-target analysis
- Multicollinearity inspection

### Orchestration (`analysis.py`)
Coordinates the complete EDA workflow by integrating statistical and visualization modules.

---

## **Key Insights from Analysis**

- Alcohol content shows a positive relationship with wine quality.
- Volatile acidity tends to negatively correlate with quality.
- Certain features exhibit skewed distributions.
- Mild outliers are present in some chemical attributes.
- Feature scaling will likely be required before implementing gradient-based regression.
- Correlation structure suggests multicollinearity among specific acidity-related features.

These observations directly influence preprocessing decisions and optimization stability in future regression implementations.

---

## **How to Run**
1. Navigate to the module directory:

```bash
cd eda_engine
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. Install required dependencies:

```bash
pip install numpy matplotlib seaborn jupyter
```

4. Launch the notebook:

```bash
jupyter notebook
```

5. Open `notebooks/eda_walkthrough.ipynb` and run all cells.

---

## **Role Within the Repository**

The EDA Engine establishes a strong analytical foundation for implementing Machine Learning algorithms from scratch.

The insights gathered here directly influence:

- Feature normalization strategy
- Choice of loss function
- Learning rate sensitivity
- Convergence behavior of gradient descent
- Interpretation of regression coefficients

By engineering EDA as a modular system, this repository emphasizes clarity, reproducibility, and principled machine learning development from first principles.
