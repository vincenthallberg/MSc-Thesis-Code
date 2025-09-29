import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score,
                             precision_score, recall_score, roc_curve, auc,
                             confusion_matrix, precision_recall_curve)

from scipy import sparse

import shap

from .utils import detect_steps, ensure_dense, transform_to_frame, get_feature_names, take_rows


def plot_roc_curves(all_model_preds, y_true, title="ROC Curve Comparison"):
    """
    Plot ROC curves for multiple models on a single chart.

    Parameters
    ----------
    all_model_preds : dict
        {model_name: {'y_pred_prob': array-like, 'y_pred': array-like}}
        Only 'y_pred_prob' is required here.
    y_true : array-like
        Ground-truth labels.
    title : str
        Figure title.
    """
    plt.figure(figsize=(10, 8))
    for name, data in all_model_preds.items():
        fpr, tpr, _ = roc_curve(y_true, data['y_pred_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name):
    """Generates and displays a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Benign (0)', 'Predicted Pathogenic (1)'],
                yticklabels=['Actual Benign (0)', 'Actual Pathogenic (1)'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

def plot_precision_recall_curves(all_model_preds, y_true):
    """Plots Precision-Recall curves for all models on one chart."""
    plt.figure(figsize=(10, 8))
    for name, data in all_model_preds.items():
        precision, recall, _ = precision_recall_curve(y_true, data['y_pred_prob'])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR AUC = {pr_auc:.3f})")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tree_importances(model_or_pipe, feature_names, top=30, title="Tree importances"):
    """
    Works with either a pipeline or a bare estimator that exposes .feature_importances_.
    """
    est = model_or_pipe
    if isinstance(model_or_pipe, Pipeline):
        _, clf_name = detect_steps(model_or_pipe)
        est = model_or_pipe.named_steps[clf_name]
    if not hasattr(est, "feature_importances_"):
        raise ValueError("Estimator has no feature_importances_.")
    imp = pd.Series(est.feature_importances_, index=feature_names).sort_values(ascending=False).head(top)
    plt.figure(figsize=(8, max(4, 0.3*len(imp))))
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.title(title)
    plt.xlabel("feature_importances_")
    plt.tight_layout()
    plt.show()
    return imp

def plot_abs_coefs(model_or_pipe, feature_names, top=30, title="|coef|"):
    """
    For linear models with .coef_ (LogisticRegression etc.). Works for pipeline or bare estimator.
    """
    est = model_or_pipe
    if isinstance(model_or_pipe, Pipeline):
        _, clf_name = detect_steps(model_or_pipe)
        est = model_or_pipe.named_steps[clf_name]
    if not hasattr(est, "coef_"):
        raise ValueError("Estimator has no coef_.")
    coefs = np.ravel(est.coef_)
    imp = pd.Series(np.abs(coefs), index=feature_names).sort_values(ascending=False).head(top)
    plt.figure(figsize=(8, max(4, 0.3*len(imp))))
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.title(title)
    plt.xlabel("|coefficient|")
    plt.tight_layout()
    plt.show()
    return imp


def permutation_importance_pipeline(pipe_or_est, X_raw_or_Xt, y, *,
                                    pre_name: str = None,
                                    clf_name: str = None,
                                    scoring="roc_auc", n_repeats=5, top=30,
                                    feature_names=None, title=None, random_state=42):
    """
    - If a Pipeline is passed: transform once with its preprocessor and run PI on the estimator only.
    - If a bare estimator is passed: assumes X_raw_or_Xt is already transformed correctly.
    Returns the importance Series.
    """
    est = pipe_or_est
    Xt = X_raw_or_Xt
    fn = feature_names

    if isinstance(pipe_or_est, Pipeline):
        pn, cn = detect_steps(pipe_or_est) if (pre_name is None or clf_name is None) else (pre_name, clf_name)
        pre = pipe_or_est.named_steps[pn] if pn is not None else None
        est = pipe_or_est.named_steps[cn]
        Xt = transform_to_frame(pre, X_raw_or_Xt) if pre is not None else X_raw_or_Xt
        if fn is None:
            fn = getattr(Xt, "columns", None)
            if fn is None:
                fn = get_feature_names(pre, X_ref=X_raw_or_Xt) or [f"f{i}" for i in range(np.asarray(Xt).shape[1])]
    else:
        # bare estimator
        if fn is None and hasattr(Xt, "columns"):
            fn = Xt.columns

    r = permutation_importance(est, Xt, y, scoring=scoring, n_repeats=n_repeats,
                               random_state=random_state, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=list(fn)).sort_values(ascending=False).head(top)

    plt.figure(figsize=(8, max(4, 0.3*len(imp))))
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.title(title or f"Permutation importance ({scoring})")
    plt.xlabel(f"Mean Δ{scoring}")
    plt.tight_layout()
    plt.show()
    return imp


def plot_baseline_importances(models_dict, X_test, y_test, *, top=30):
    """
    For the baselines dict {"Logistic Regression": pipeline, "Naive Bayes": estimator, "Bayesian NN": estimator}
    - If model has coef_ -> |coef| plot
    - Else -> permutation importance on X_test / y_test
    """
    fn = list(getattr(X_test, "columns", [f"f{i}" for i in range(np.asarray(X_test).shape[1])]))
    for name, model in models_dict.items():
        print(f"\n=== {name} ===")
        try:
            plot_abs_coefs(model, fn, top=top, title=f"{name} – |coef|")
        except Exception:
            permutation_importance_pipeline(model, X_test, y_test,
                                            scoring="roc_auc", n_repeats=5, top=top,
                                            feature_names=fn, title=f"{name} – Permutation importance")