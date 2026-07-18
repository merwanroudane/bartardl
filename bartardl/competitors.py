"""The three competitor estimators benchmarked in the paper (Section 3).

Each estimator exposes the same small interface as :class:`~bartardl.bart.BART`
so they can be dropped into the ARDL wrapper and the model-comparison
tables interchangeably:

* ``fit(X, y)``           -> self
* ``predict(X)``          -> ndarray
* ``importance_``         -> non-negative array, one weight per predictor
  (normalised so it sums to one, mirroring BART's inclusion proportions).

Estimators
----------
* :class:`Lasso` -- L1 penalised regression, Tibshirani (1996)
  <doi:10.1111/j.2517-6161.1996.tb02080.x>, cross-validated ``lambda``.
* :class:`ElasticNet` -- L1/L2 compromise, Zou and Hastie (2005)
  <doi:10.1111/j.1467-9868.2005.00503.x>, cross-validated ``lambda`` and
  ``alpha``.
* :class:`BayesianNetwork` -- a stochastic-search variable-selection
  (SSVS) / spike-and-slab Gibbs sampler.  Following Ahelegbey, Billio and
  Casarin (2016) <doi:10.1002/jae.2443>, a non-zero coefficient is read as
  a present edge between a predictor and the response, so the posterior
  inclusion probabilities are the natural "network" importance measure.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler


class _LinearBase:
    """Shared plumbing: standardise X, fit an sklearn model, expose importance."""

    def __init__(self):
        self._scaler = StandardScaler()
        self.model = None
        self.importance_ = None
        self.coef_ = None
        self.intercept_ = 0.0

    def _standardise(self, X, fit=False):
        return (self._scaler.fit_transform(X) if fit
                else self._scaler.transform(X))

    def _set_importance(self, coef):
        imp = np.abs(coef)
        tot = imp.sum()
        self.importance_ = imp / tot if tot > 0 else np.zeros_like(imp)

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=float))


class Lasso(_LinearBase):
    """Cross-validated LASSO (Tibshirani, 1996)."""

    def __init__(self, cv: int = 5, random_state=None, max_iter: int = 20000):
        super().__init__()
        self.model = LassoCV(cv=cv, max_iter=max_iter,
                             random_state=random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self._set_importance(self.coef_)
        return self


class ElasticNet(_LinearBase):
    """Cross-validated Elastic Net (Zou and Hastie, 2005).

    ``l1_ratio`` (the paper's ``alpha``) is itself selected by CV over the
    supplied grid; the default grid leans toward the LASSO end, as
    recommended in the paper, to avoid the instabilities of highly
    correlated predictors.
    """

    def __init__(self, cv: int = 5, l1_ratio=None,
                 random_state=None, max_iter: int = 20000):
        super().__init__()
        if l1_ratio is None:
            l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        self.model = ElasticNetCV(cv=cv, l1_ratio=l1_ratio,
                                  max_iter=max_iter, random_state=random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.l1_ratio_ = self.model.l1_ratio_
        self._set_importance(self.coef_)
        return self


class BayesianNetwork:
    """Spike-and-slab (SSVS) linear regression -- the Bayesian-Network estimator.

    The model is

        y = X beta + u,      u ~ N(0, sigma^2 I),
        beta_j | gamma_j ~ gamma_j * N(0, c*tau^2) + (1 - gamma_j) * N(0, tau^2),
        gamma_j ~ Bernoulli(pi),

    where ``gamma_j = 1`` marks a present edge between predictor ``j`` and
    the response (Ahelegbey, Billio and Casarin, 2016).  A Gibbs sampler
    draws ``beta``, ``gamma`` and ``sigma^2``; the posterior mean of
    ``gamma`` gives the inclusion probabilities used for importance, and
    the posterior mean of ``beta`` is used for prediction.

    Predictors are standardised internally; predictions are mapped back to
    the original scale.
    """

    def __init__(self, n_iter: int = 2000, burn: int = 1000, pi: float = 0.5,
                 tau: float = 0.1, c: float = 100.0, random_state=None):
        self.n_iter = n_iter
        self.burn = burn
        self.pi = pi
        self.tau = tau
        self.c = c
        self._rng = np.random.default_rng(random_state)
        self._scaler = StandardScaler()

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        Xs = self._scaler.fit_transform(X)
        self._y_mean = y.mean()
        yc = y - self._y_mean
        n, p = Xs.shape

        beta = np.zeros(p)
        gamma = np.ones(p, dtype=bool)
        sigma2 = float(np.var(yc)) or 1.0
        a0, b0 = 2.0, 1.0                      # inverse-gamma prior on sigma^2
        tau2, ctau2 = self.tau ** 2, self.c * self.tau ** 2

        XtX = Xs.T @ Xs
        Xty = Xs.T @ yc

        gamma_sum = np.zeros(p)
        beta_sum = np.zeros(p)
        n_kept = 0

        for it in range(self.n_iter):
            # --- update each coefficient conditionally ------------------- #
            for j in range(p):
                prior_var = ctau2 if gamma[j] else tau2
                r_j = Xty[j] - XtX[j] @ beta + XtX[j, j] * beta[j]
                prec = XtX[j, j] / sigma2 + 1.0 / prior_var
                mean = (r_j / sigma2) / prec
                beta[j] = mean + self._rng.normal() / np.sqrt(prec)

                # --- update the inclusion indicator gamma_j ------------- #
                # log posterior odds of gamma_j = 1 vs 0 given beta_j.
                log_slab = -0.5 * np.log(ctau2) - beta[j] ** 2 / (2 * ctau2)
                log_spike = -0.5 * np.log(tau2) - beta[j] ** 2 / (2 * tau2)
                log_odds = (np.log(self.pi) - np.log1p(-self.pi)
                            + log_slab - log_spike)
                prob = 1.0 / (1.0 + np.exp(-log_odds))
                gamma[j] = self._rng.random() < prob

            # --- update sigma^2 ----------------------------------------- #
            resid = yc - Xs @ beta
            shape = a0 + 0.5 * n
            rate = b0 + 0.5 * float(resid @ resid)
            sigma2 = rate / self._rng.gamma(shape)

            if it >= self.burn:
                gamma_sum += gamma
                beta_sum += beta
                n_kept += 1

        self.inclusion_prob_ = gamma_sum / max(n_kept, 1)
        self.coef_ = beta_sum / max(n_kept, 1)
        tot = self.inclusion_prob_.sum()
        self.importance_ = (self.inclusion_prob_ / tot if tot > 0
                            else np.zeros(p))
        return self

    def predict(self, X):
        Xs = self._scaler.transform(np.asarray(X, dtype=float))
        return self._y_mean + Xs @ self.coef_


__all__ = ["Lasso", "ElasticNet", "BayesianNetwork"]
