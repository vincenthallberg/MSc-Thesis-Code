import math, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

def _log_gaussian(x, mu, sigma):
    # elementwise log N(x | mu, sigma^2)
    return -0.5*math.log(2*math.pi) - torch.log(sigma) - 0.5*((x - mu)/sigma)**2

def _log_scale_mixture_prior(w, pi, s1, s2):
    logN1 = _log_gaussian(w, 0.0, s1)
    logN2 = _log_gaussian(w, 0.0, s2)
    logpi, log1m = torch.log(pi+1e-12), torch.log(1-pi+1e-12)
    return torch.logsumexp(torch.stack([logpi+logN1, log1m+logN2], 0), dim=0)

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, *, prior_pi=0.5, prior_sigma1=1.0, prior_sigma2=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # variational params
        self.weight_mu  = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.05))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu    = nn.Parameter(torch.zeros(out_features))
        self.bias_rho   = nn.Parameter(torch.full((out_features,), -5.0))

        # fixed prior hyperparams
        self.register_buffer('prior_pi', torch.tensor(prior_pi, dtype=torch.float32))
        self.register_buffer('prior_sigma1', torch.tensor(prior_sigma1, dtype=torch.float32))
        self.register_buffer('prior_sigma2', torch.tensor(prior_sigma2, dtype=torch.float32))

    def _sigma(self, rho):
        # softplus to ensure positivity, numerically stable
        return F.softplus(rho)

    def forward(self, x, sample=True):
        # sample weights (reparameterization) once per forward
        if self.training or sample:
            eps_w = torch.randn_like(self.weight_mu)
            eps_b = torch.randn_like(self.bias_mu)
            weight_sigma = self._sigma(self.weight_rho)
            bias_sigma   = self._sigma(self.bias_rho)
            weight = self.weight_mu + weight_sigma * eps_w
            bias   = self.bias_mu   + bias_sigma   * eps_b
        else:
            weight = self.weight_mu
            bias   = self.bias_mu

        out = F.linear(x, weight, bias)

        # accumulate log q and log p for the KL term
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma   = self._sigma(self.bias_rho)

        log_q_w = _log_gaussian(weight, self.weight_mu, weight_sigma).sum()
        log_q_b = _log_gaussian(bias,   self.bias_mu,   bias_sigma).sum()
        log_q = log_q_w + log_q_b

        log_p_w = _log_scale_mixture_prior(weight, self.prior_pi, self.prior_sigma1, self.prior_sigma2).sum()
        log_p_b = _log_scale_mixture_prior(bias,   self.prior_pi, self.prior_sigma1, self.prior_sigma2).sum()
        log_p = log_p_w + log_p_b

        return out, log_q, log_p

class BayesMLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes=(128,128),
                 prior_pi=0.5, prior_sigma1=1.0, prior_sigma2=0.1):
        super().__init__()
        layers = []
        last = in_dim
        self.blayers = nn.ModuleList()
        for h in hidden_sizes:
            bl = BayesianLinear(last, h, prior_pi=prior_pi, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2)
            self.blayers.append(bl)
            last = h
        self.out = BayesianLinear(last, 1, prior_pi=prior_pi, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2)

    def forward(self, x, sample=True):
        log_q_total = 0.0
        log_p_total = 0.0
        for bl in self.blayers:
            x, lq, lp = bl(x, sample=sample)
            log_q_total = log_q_total + lq
            log_p_total = log_p_total + lp
            x = F.relu(x)
        logits, lq, lp = self.out(x, sample=sample)
        log_q_total = log_q_total + lq
        log_p_total = log_p_total + lp
        return logits.squeeze(-1), log_q_total, log_p_total  # logits, scalars

class BayesByBackpropClassifier(BaseEstimator, ClassifierMixin):
    """
    Binary classifier using Bayes by Backprop (Blundell et al., 2015).
    - Variational posterior: diagonal Gaussian per weight (mu, rho).
    - Prior: scale mixture of two zero-mean Gaussians (pi, sigma1, sigma2).
    - Loss: (log q - log p) reweighted per minibatch + BCE-with-logits likelihood.
    """
    def __init__(self,
                 hidden_sizes=(128, 128),
                 prior_pi=0.5, prior_sigma1=1.0, prior_sigma2=0.1,
                 epochs=150, batch_size=128, lr=1e-3, weight_decay=0.0,
                 kl_anneal=True, device=None, n_mc_predict=20,
                 class_weight='balanced', random_state=42, verbose=False):
        self.hidden_sizes = tuple(hidden_sizes)
        self.prior_pi = float(prior_pi)
        self.prior_sigma1 = float(prior_sigma1)
        self.prior_sigma2 = float(prior_sigma2)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.kl_anneal = bool(kl_anneal)
        self.n_mc_predict = int(n_mc_predict)
        self.class_weight = class_weight  # 'balanced' | None | {0:w0, 1:w1}
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.device = device  # 'cuda' | 'cpu' | None

        # placeholders
        self._net = None
        self._in_dim = None
        self._classes_ = np.array([0,1])

    def _resolve_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_weights(self, y):
        y = np.asarray(y).astype(int)
        if isinstance(self.class_weight, dict):
            cw = self.class_weight
        elif self.class_weight == 'balanced':
            neg, pos = (y==0).sum(), (y==1).sum()
            total = neg + pos
            w0 = total/(2.0*neg) if neg>0 else 0.0
            w1 = total/(2.0*pos) if pos>0 else 0.0
            cw = {0:w0, 1:w1}
        else:
            cw = {0:1.0, 1:1.0}
        sample_w = np.where(y==1, cw[1], cw[0]).astype(np.float32)
        return cw, sample_w

    def fit(self, X, y):
        rs = check_random_state(self.random_state)
        torch.manual_seed(self.random_state)
        self._in_dim = X.shape[1]
        dev = self._resolve_device()

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).ravel()
        class_w, sample_w = self._prepare_weights(y_np)

        # tensors
        X_tensor = torch.from_numpy(X_np)
        y_tensor = torch.from_numpy(y_np)
        w_tensor = torch.from_numpy(sample_w)

        dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self._net = BayesMLP(self._in_dim, self.hidden_sizes,
                             prior_pi=self.prior_pi, prior_sigma1=self.prior_sigma1, prior_sigma2=self.prior_sigma2).to(dev)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        num_batches = max(1, math.ceil(len(dataset)/self.batch_size))

        for epoch in range(self.epochs):
            self._net.train()
            total_loss = 0.0
            # simple linear KL annealing across epochs
            kl_scale_epoch = (epoch+1)/self.epochs if self.kl_anneal else 1.0

            for xb, yb, wb in loader:
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                wb = wb.to(dev, non_blocking=True)

                opt.zero_grad()
                logits, log_q, log_p = self._net(xb, sample=True)

                # BCE with logits, sample-weighted
                bce = F.binary_cross_entropy_with_logits(logits, yb, reduction='none')
                bce = (bce * wb).sum()

                # minibatch KL weighting (approx. Eq. 8/9, here 1/num_batches)
                kl = (log_q - log_p) * (kl_scale_epoch / num_batches)

                loss = bce + kl
                loss.backward()
                opt.step()

                total_loss += loss.item()

            if self.verbose and ((epoch+1) % max(1, self.epochs//10) == 0):
                print(f"[BNN] Epoch {epoch+1}/{self.epochs} - loss {total_loss/num_batches:.4f}")

        self.device_ = str(dev)
        return self

    def predict_proba(self, X):
        if self._net is None:
            raise RuntimeError("Model not fitted.")
        dev = self._resolve_device()
        self._net.eval()
        X_np = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X_np).to(dev)

        # MC average of predictive probabilities
        with torch.no_grad():
            probs = []
            for _ in range(self.n_mc_predict):
                logits, _, _ = self._net(X_tensor, sample=True)
                probs.append(torch.sigmoid(logits).cpu().numpy())
            p_mean = np.mean(probs, axis=0)
        p_mean = np.clip(p_mean, 1e-7, 1-1e-7)
        return np.column_stack([1.0 - p_mean, p_mean])

    def predict(self, X):
        proba = self.predict_proba(X)[:,1]
        return (proba >= 0.5).astype(int)

    # sklearn compatibility
    def get_params(self, deep=True):
        return {k:getattr(self,k) for k in (
            "hidden_sizes","prior_pi","prior_sigma1","prior_sigma2",
            "epochs","batch_size","lr","weight_decay","kl_anneal",
            "device","n_mc_predict","class_weight","random_state","verbose"
        )}

    def set_params(self, **params):
        for k,v in params.items():
            setattr(self,k,v)
        return self
