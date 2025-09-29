from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from .bnn import BayesByBackpropClassifier

def make_baseline_models():
    """Return a dict of baseline models."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=10000,
                tol=1e-4,
                random_state=42,
            ),
        ),
        "Naive Bayes": GaussianNB(),
        "Bayesian NN": BayesByBackpropClassifier(
            hidden_sizes=(256, 128),
            prior_pi=0.5, prior_sigma1=1.0, prior_sigma2=0.1,
            epochs=200, batch_size=128, lr=1e-3, kl_anneal=True,
            n_mc_predict=100,
            class_weight="balanced",
            random_state=42, verbose=False,
        ),
    }

# convenience alias so you can call baseline_models() as you do in the notebook
baseline_models = make_baseline_models

__all__ = ["make_baseline_models", "baseline_models"]

if __name__ == "__main__":
    print(list(make_baseline_models().keys()))