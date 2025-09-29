import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from scipy import sparse


def make_weights(y):
    """Return (class_weights dict for tree libs, sample_weight array, scale_pos_weight for XGB)."""
    y = np.asarray(y).astype(int)
    classes = np.array([0, 1])
    cw_vals = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = {0: float(cw_vals[0]), 1: float(cw_vals[1])}
    sample_weight = np.where(y == 1, class_weights[1], class_weights[0]).astype(float)
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = (neg / pos) if pos > 0 else 1.0
    return class_weights, sample_weight, float(spw)

def detect_steps(pipe: Pipeline) -> Tuple[str, str]:
    """
    Return (pre_name, clf_name) for a pipeline.
    Finds the first ColumnTransformer (or 'pre'), and the last step with predict_proba (or 'clf').
    """
    pre_name, clf_name = None, None
    for name, step in pipe.named_steps.items():
        if pre_name is None and isinstance(step, ColumnTransformer):
            pre_name = name
    if pre_name is None and "pre" in pipe.named_steps:
        pre_name = "pre"
    if pre_name is None and "preprocessor" in pipe.named_steps:
        pre_name = "preprocessor"

    # heuristic for classifier (last step with predict_proba or decision_function)
    for name, step in pipe.named_steps.items():
        if hasattr(step, "predict_proba") or hasattr(step, "decision_function"):
            clf_name = name
    if clf_name is None and "clf" in pipe.named_steps:
        clf_name = "clf"
    if clf_name is None and "classifier" in pipe.named_steps:
        clf_name = "classifier"

    if clf_name is None:
        # fall back to last step
        clf_name = list(pipe.named_steps)[-1]
    return pre_name, clf_name

def ensure_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)

def take_rows(X, rows):
    if hasattr(X, "iloc"):
        return X.iloc[rows]
    return X[rows]

def transform_to_frame(preprocessor, X_raw) -> pd.DataFrame:
    """
    Transform X_raw with 'preprocessor'. Always returns a DataFrame with proper column names.
    """
    Xt = preprocessor.transform(X_raw)
    if hasattr(Xt, "columns"):
        return Xt  # already a DataFrame
    Xt_dense = ensure_dense(Xt)
    # try to get names; otherwise reuse raw names or generic f{i}
    names = None
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = None
    if names is None:
        if hasattr(X_raw, "columns"):
            names = X_raw.columns
        else:
            names = [f"f{i}" for i in range(Xt_dense.shape[1])]
    return pd.DataFrame(Xt_dense, index=getattr(X_raw, "index", None), columns=list(names))

def get_feature_names(preprocessor, X_ref: Optional[pd.DataFrame]=None):
    """
    Robust feature-name getter.
    - If transformer exposes get_feature_names_out → use it.
    - Else, if X_ref is given, transform a tiny slice and read .columns.
    - Else, return None (callers can fall back to DataFrame columns they already have).
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return preprocessor.get_feature_names_out()
        except Exception:
            pass
    if X_ref is not None:
        try:
            # transform a tiny sample
            sl = X_ref.iloc[:2] if hasattr(X_ref, "iloc") else X_ref[:2]
            Xt = preprocessor.transform(sl)
            if hasattr(Xt, "columns"):
                return Xt.columns.to_numpy()
        except Exception:
            pass
    return None