# MSc Thesis Code – Oncogenic Mutation Prediction

This repository contains code and notebooks accompanying my MSc thesis.  
The work focuses on predicting oncogenic/pathogenic effects of somatic mutations using engineered protein features and machine learning models.

The repository is structured into two main studies:

- **`large_data_study/`**  
  - `ml_large_data/`: Python modules implementing models, training utilities, metrics, and plotting functions.  
  - `training_notebooks/`: Jupyter notebooks for feature selection, model development, and out-of-bag (OOB) evaluation.  

- **`EGFR_study/`**  
  - Two focused notebooks (`EGFR_EDA_and_feature_selection.ipynb`, `EGFR_model_development.ipynb`) studying mutations in the EGFR protein.

---

## Part 1: Preprocessing

The preprocessing stage includes:

- **Filtering & criteria for selection:**  
  Variants are filtered by data quality and mapping consistency across sources.  
- **Feature engineering:**  
  Deterministic sequence-context and structural features are computed.  
- **Hypothesis testing & refinement:**  
  Candidate features undergo correlation pruning, PLS-based dimensionality reduction, and bootstrapped stability selection to handle multicollinearity.  

Mathematically, given a dataset \( (X, y) \), we select a feature subset \( S \subseteq \{1, \ldots, p\} \) that maximizes predictive stability under repeated subsampling:
$$
S = \arg\max_{S'} \; \Pr_{b \sim \text{Bootstrap}} \big[ j \in S' \big],
$$
where stability is defined as the frequency with which a feature is retained across bootstrap replicates.

---

## Part 2: Model Development and Training

We evaluate a range of models implemented in [`ml_large_data`](./large_data_study/ml_large_data):

- **Baselines:** Logistic Regression, Naive Bayes, and a Bayesian Neural Network (Bayes by Backprop) [`baselines.py`](./large_data_study/ml_large_data/baselines.py).  
- **Tree boosting models:** LightGBM, XGBoost, and CatBoost with hyperparameters tuned using [Optuna](https://optuna.org/) [`tuning.py`](./large_data_study/ml_large_data/tuning.py).  
- **Bayesian Neural Network:** Implemented in [`bnn.py`](./large_data_study/ml_large_data/bnn.py), following Blundell et al. (2015), optimizing the evidence lower bound:
  \[
  \mathcal{L} = \mathbb{E}_{q(w)}\big[-\log p(y \mid x, w)\big] + \text{KL}\big(q(w)\,\|\,p(w)\big).
  \]

### Hyperparameter Optimization
For boosting models, we optimize trial-specific parameters \( \theta \) to maximize validation ROC AUC:
\[
\theta^* = \arg\max_{\theta \in \mathcal{H}} \; \text{ROC-AUC}(f_\theta(X_{\text{val}}), y_{\text{val}}).
\]

### Metrics
Performance is evaluated using:  
- **ROC AUC**, **MCC**, **F1 score**, **Balanced Accuracy**, **Precision**, **Recall**  
(implemented in [`metrics.py`](./large_data_study/ml_large_data/metrics.py)).
