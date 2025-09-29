import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef,
                             balanced_accuracy_score, precision_score, recall_score)
                             

def compute_all_metrics(y_true, y_pred, y_pred_prob):
    return {
        'ROC AUC': round(roc_auc_score(y_true, y_pred_prob), 4),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 4),
        'F1 Score': round(f1_score(y_true, y_pred), 4),
        'Balanced Accuracy': round(balanced_accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
    }

def find_best_threshold_mcc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    thr = np.r_[0.0, np.sort(np.unique(y_prob)), 1.0]
    best_t, best_m = 0.5, -1
    from sklearn.metrics import matthews_corrcoef
    for t in thr:
        y_hat = (y_prob >= t).astype(int)
        m = matthews_corrcoef(y_true, y_hat)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)