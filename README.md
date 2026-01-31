# American Express Credit Default Prediction

## Project Overview

This project implements an end-to-end **credit default prediction pipeline** using the American Express dataset.
The objective is to reproduce a rigorous machine-learning workflow for credit-risk modeling while emphasizing **reproducibility, interpretability, and computational feasibility** rather than purely maximizing model complexity.

The project follows a structured methodology: performance-aware data loading, customer-level feature engineering, systematic model comparison, and transparent handling of computational constraints.

---

## Problem Statement

Credit default prediction is a central problem in financial risk management.
Accurately ranking customers by default risk enables better exposure management, pricing strategies, and loss mitigation.

The American Express dataset is particularly challenging due to:
- large scale (raw data >15 GB),
- anonymized features,
- longitudinal (time-series) customer behavior.

This project treats these challenges as first-class design constraints.

---

## Dataset Description

- **Source:** American Express Default Prediction dataset
- **Structure:** anonymized customer-level time-series data
- **Target:** binary default indicator
- **Raw size:** >15 GB
- **Final working sample:** **30,000 customers**

Due to memory limitations, the full dataset was not loaded at once. Instead, a **performance-aware sampling strategy** was applied and the dataset was frozen early to ensure reproducibility.

---

## Methodology

The project follows a 15-step structured workflow:

1. Problem definition and metric selection
2. Dataset inspection
3. Performance-aware data loading
4. Customer sampling and dataset freezing
5. Data cleaning
6. Time-series aggregation to customer level
7. Final dataset assembly
8. Evaluation protocol definition
9. Baseline models
10. Hyperparameter tuning (controlled)
11. Class imbalance handling
12. Ensemble models
13. Model evaluation and comparison
14. Model interpretation
15. Final conclusions and future work

Each step is isolated to prevent data leakage and to allow fair comparison across models.

---

## Feature Engineering

Time-series customer data was aggregated into fixed-length feature vectors using statistics such as:

- mean
- standard deviation
- minimum / maximum
- count
- most recent value (`last`)

This captures:
- long-term behavior,
- recency effects,
- and extreme historical events.

The final dataset contains:
- **30,000 rows (customers)**
- **1,086 engineered features**

---

## Evaluation Protocol

Models were evaluated using a **repeated stratified holdout** strategy to reduce variance from a single split.

**Primary metrics:**
- **ROC-AUC** (ranking quality)
- **PR-AUC** (default detection quality)

**Secondary metrics:**
- precision
- recall
- F1-score (at fixed thresholds)

This protocol ensures stable and fair performance comparison.

---

## Models Evaluated

### Baseline models
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors

### Tuned models
- Logistic Regression (RandomizedSearchCV)
- Decision Tree (RandomizedSearchCV)

### Class imbalance strategies
- Class weighting
- Threshold tuning

### Ensemble models
- Random Forest (fixed configuration)
- HistGradientBoostingClassifier

> **Note:** Full Random Forest hyperparameter tuning was attempted but excluded due to excessive runtime, even under constrained search settings. This limitation is documented transparently.

---

## Final Model Selection

The **unweighted Logistic Regression model (LR_Unweighted)** was selected as the final model.

**Reasons:**
- Highest **ROC-AUC** and **PR-AUC**
- Stable performance across repeated splits
- No meaningful improvement from tuning
- High interpretability
- Low computational cost

This result indicates that the aggregated feature space is close to linearly separable and that **feature engineering was more impactful than model complexity**.

---

## Model Interpretation

Because Logistic Regression is inherently interpretable, model coefficients were analyzed and converted into **odds ratios**.

Key findings:
- Both **recent behavior** (`last`) and **long-term patterns** (`mean`, `min`, `max`) are important drivers of default risk.
- Influential features were stable across multiple training splits.
- Interpretation focuses on **relative influence**, as features are anonymized.

---

## Computational Constraints

- Full Random Forest tuning exceeded feasible runtime (> several hours).
- Parallelism and search space were reduced, but tuning remained impractical.
- The project proceeds with fixed-configuration ensembles and alternative models.

This reflects real-world data science practice, where resource constraints influence modeling decisions.

---

## Limitations

- Anonymized features prevent direct business interpretation.
- Sequence-based models (RNNs, transformers) were not explored.
- Cost-sensitive decision thresholds were not explicitly modeled.

---

## Future Work

Potential extensions include:
- sequence-aware models for raw time-series data,
- probability calibration analysis,
- cost-sensitive optimization,
- temporal validation on future cohorts.

---

## Key Takeaway

> With careful feature engineering and disciplined evaluation, simple and interpretable models can achieve state-of-the-art performance on complex credit-risk prediction tasks.

---

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_data_sampling.ipynb
│   ├── 02_preprocessing_and_aggregation.ipynb
│   ├── 03_modeling_and_evaluation.ipynb
├── data/
│   ├── amex_step7_X.parquet
│   └── amex_step7_y.parquet
├── results/
│   ├── step9_baseline_results.csv
│   ├── step10_tuned_results.csv
│   ├── step11_imbalance_results.csv
│   ├── step12_ensemble_results.csv
│   └── step13_model_comparison.csv
├── README.md
