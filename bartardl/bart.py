"""Bayesian Additive Regression Trees (BART) -- a self-contained sampler.

This module implements the sum-of-trees model of Chipman, George and
McCulloch (2010) <doi:10.1214/09-AOAS285> exactly as described in
Mahdavi, Ehsani, Ahelegbey and Mohammadpour (2024), *Measuring Causal
Effect with ARDL-BART* <doi:10.4236/am.2024.154018>.

The model is

    y = f(x) + e,   e ~ N(0, sigma^2),
    f(x) = sum_{l=1}^{m} g(x; T_l, M_l),

a sum of ``m`` regression trees.  Each tree ``T_l`` carries a set of leaf
parameters ``M_l``.  The priors follow the paper (Section 2):

* **Tree structure.** A node at depth ``d`` is non-terminal with
  probability ``alpha * (1 + d) ** (-beta)`` (paper Eq. 3), with the
  recommended defaults ``alpha = 0.95`` and ``beta = 2``.  This keeps
  every single tree shallow ("weak learner") and is the source of BART's
  regularisation.
* **Leaf parameters.** Each leaf value is ``mu ~ N(0, tau)`` with
  ``tau = (0.5 / (k * sqrt(m))) ** 2`` after the response has been shifted
  and scaled to ``[-0.5, 0.5]``.  ``k = 2`` by default (paper Section 4).
* **Error variance.** ``sigma^2 ~ InvGamma(nu/2, nu*lambda/2)`` with
  ``lambda`` calibrated so that there is a prior probability ``q`` that
  ``sigma`` is below the data-based estimate ``sigma_hat`` (the residual
  standard deviation of an OLS fit).  ``nu = 3`` and ``q = 0.9`` reproduce
  the paper's settings.

The ensemble is drawn with the Bayesian back-fitting MCMC of Chipman et
al.: each sweep proposes a single *grow* or *prune* move per tree via a
Metropolis-Hastings step, then re-draws the leaf values and ``sigma``.

Nothing here depends on an external BART library; the sampler is written
in NumPy so the whole package installs from PyPI with only ``numpy``,
``scipy`` and ``scikit-learn``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import chi2


# --------------------------------------------------------------------------- #
# Tree data structure
# --------------------------------------------------------------------------- #
class _Node:
    """A single node of a regression tree.

    Leaves carry a ``value`` (the leaf parameter ``mu``); internal nodes
    carry a splitting rule ``x[:, var] <= thr`` with ``left`` / ``right``
    children.  ``rows`` caches the training-row indices that reach the node
    so that grow / prune moves stay local and cheap.
    """

    __slots__ = ("depth", "is_leaf", "var", "thr", "left", "right",
                 "value", "rows")

    def __init__(self, depth: int, rows: np.ndarray):
        self.depth = depth
        self.is_leaf = True
        self.var: Optional[int] = None
        self.thr: Optional[float] = None
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.value: float = 0.0
        self.rows = rows


class _Tree:
    """A regression tree rooted at a single leaf covering all rows."""

    def __init__(self, n_rows: int):
        self.root = _Node(depth=0, rows=np.arange(n_rows))

    # -- structural queries ------------------------------------------------ #
    def leaves(self) -> list[_Node]:
        out: list[_Node] = []
        stack = [self.root]
        while stack:
            nd = stack.pop()
            if nd.is_leaf:
                out.append(nd)
            else:
                stack.append(nd.left)
                stack.append(nd.right)
        return out

    def nog_nodes(self) -> list[_Node]:
        """Internal nodes whose two children are both leaves (prunable)."""
        out: list[_Node] = []
        stack = [self.root]
        while stack:
            nd = stack.pop()
            if not nd.is_leaf:
                if nd.left.is_leaf and nd.right.is_leaf:
                    out.append(nd)
                else:
                    stack.append(nd.left)
                    stack.append(nd.right)
        return out

    def predict_rows(self, n_rows: int) -> np.ndarray:
        """Vector of fitted values on the training rows (uses cached rows)."""
        out = np.zeros(n_rows)
        for leaf in self.leaves():
            out[leaf.rows] = leaf.value
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fitted values for an arbitrary design matrix ``X``."""
        out = np.zeros(X.shape[0])
        idx = np.arange(X.shape[0])
        stack = [(self.root, idx)]
        while stack:
            nd, rows = stack.pop()
            if nd.is_leaf:
                out[rows] = nd.value
            else:
                go_left = X[rows, nd.var] <= nd.thr
                stack.append((nd.left, rows[go_left]))
                stack.append((nd.right, rows[~go_left]))
        return out


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class BARTConfig:
    """Hyper-parameters of the BART prior and sampler.

    The defaults reproduce the paper's simulation settings
    (``m = 200``, ``k = 2``, ``nu = 3``, ``q = 0.9``, ``alpha = 0.95``,
    ``beta = 2``).  ``n_draws`` / ``n_burn`` default to values that keep an
    example runnable in seconds; set them to ``20000`` / ``250`` to match
    the paper exactly.
    """

    n_trees: int = 200            # m
    k: float = 2.0                # leaf-prior width factor
    nu: float = 3.0               # sigma prior degrees of freedom
    q: float = 0.90               # sigma prior quantile
    alpha: float = 0.95           # tree-prior base (Eq. 3)
    beta: float = 2.0             # tree-prior growth penalty (Eq. 3)
    n_draws: int = 1000           # retained posterior draws
    n_burn: int = 1000            # burn-in iterations
    thin: int = 1                 # keep every ``thin``-th draw
    p_grow: float = 0.5           # proposal prob. of a grow move
    min_leaf: int = 1             # minimum rows in a leaf
    split_prob: Optional[np.ndarray] = None  # arbitrary discrete var. prior
    random_state: Optional[int] = None


# --------------------------------------------------------------------------- #
# The BART sampler
# --------------------------------------------------------------------------- #
class BART:
    """Bayesian Additive Regression Trees regressor.

    Parameters
    ----------
    config : BARTConfig, optional
        Prior and sampler settings.  Individual fields may also be passed
        as keyword arguments for convenience, e.g. ``BART(n_trees=50)``.

    Attributes populated after :meth:`fit`
    --------------------------------------
    yhat_train_ : ndarray, shape (n,)
        Posterior-mean fit on the training data (original scale).
    sigma_draws_ : ndarray, shape (n_kept,)
        Posterior draws of the residual standard deviation.
    var_counts_ : ndarray, shape (n_features,)
        Total number of times each predictor was used as a splitting
        variable across kept draws.
    inclusion_proportion_ : ndarray, shape (n_features,)
        ``var_counts_`` normalised to sum to one -- the "variable
        inclusion proportions" of Chipman et al. used for BART variable
        importance (paper Section 4.1.2).
    """

    def __init__(self, config: Optional[BARTConfig] = None, **kwargs):
        if config is None:
            config = BARTConfig()
        for key, val in kwargs.items():
            if not hasattr(config, key):
                raise TypeError(f"unknown BART option {key!r}")
            setattr(config, key, val)
        self.cfg = config
        self._rng = np.random.default_rng(config.random_state)

    # ------------------------------------------------------------------ #
    # Response scaling helpers
    # ------------------------------------------------------------------ #
    def _scale_y(self, y: np.ndarray) -> np.ndarray:
        self._y_min, self._y_max = float(y.min()), float(y.max())
        self._y_center = 0.5 * (self._y_min + self._y_max)
        self._y_range = self._y_max - self._y_min
        if self._y_range == 0:
            self._y_range = 1.0
        return (y - self._y_center) / self._y_range

    def _unscale_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self._y_range + self._y_center

    # ------------------------------------------------------------------ #
    # sigma prior calibration
    # ------------------------------------------------------------------ #
    def _calibrate_sigma_prior(self, X: np.ndarray, ys: np.ndarray):
        """Return (nu, lambda) with lambda set from the q-quantile rule."""
        n, p = X.shape
        # sigma_hat: residual SD of an OLS fit of the scaled response, with a
        # graceful fall-back to the marginal SD when X is rank-deficient.
        try:
            Xd = np.column_stack([np.ones(n), X])
            beta, *_ = np.linalg.lstsq(Xd, ys, rcond=None)
            resid = ys - Xd @ beta
            sigma_hat = resid.std(ddof=1)
        except np.linalg.LinAlgError:
            sigma_hat = ys.std(ddof=1)
        if not np.isfinite(sigma_hat) or sigma_hat <= 0:
            sigma_hat = ys.std(ddof=1) or 1.0
        nu = self.cfg.nu
        # nu*lambda / sigma^2 ~ chi^2_nu, so P(sigma^2 < sigma_hat^2) = q
        # <=> nu*lambda / sigma_hat^2 = chi2.ppf(1 - q, nu).
        lam = sigma_hat ** 2 * chi2.ppf(1.0 - self.cfg.q, nu) / nu
        return nu, lam

    # ------------------------------------------------------------------ #
    # Marginal log-likelihood of a node (leaf integrated out)
    # ------------------------------------------------------------------ #
    def _node_llik(self, n: int, s: float, sigma2: float, tau: float) -> float:
        """Log marginal likelihood of a node holding ``n`` residuals summing
        to ``s`` under leaf prior N(0, tau) and noise ``sigma2``.

        Terms constant across the parent/children partition (the
        ``sum r_i^2`` piece) cancel in every grow / prune ratio and are
        dropped here.
        """
        denom = sigma2 + n * tau
        return 0.5 * np.log(sigma2 / denom) + (tau * s * s) / (2.0 * sigma2 * denom)

    # ------------------------------------------------------------------ #
    # Grow / prune Metropolis-Hastings step for one tree
    # ------------------------------------------------------------------ #
    def _grow(self, tree: _Tree, X: np.ndarray, resid: np.ndarray,
              sigma2: float, tau: float) -> None:
        cfg = self.cfg
        leaves = tree.leaves()
        b = len(leaves)
        leaf = leaves[self._rng.integers(b)]
        rows = leaf.rows
        if rows.size < 2 * cfg.min_leaf:
            return
        # choose a splitting variable (uniform, or arbitrary discrete prior)
        if cfg.split_prob is not None:
            var = int(self._rng.choice(X.shape[1], p=cfg.split_prob))
        else:
            var = int(self._rng.integers(X.shape[1]))
        vals = X[rows, var]
        uniq = np.unique(vals)
        if uniq.size < 2:
            return
        # candidate thresholds are the interior unique values
        thr = float(self._rng.choice(uniq[:-1]))
        go_left = vals <= thr
        n_l = int(go_left.sum())
        n_r = rows.size - n_l
        if n_l < cfg.min_leaf or n_r < cfg.min_leaf:
            return

        s_p = float(resid[rows].sum())
        s_l = float(resid[rows[go_left]].sum())
        s_r = s_p - s_l

        llik = (self._node_llik(n_l, s_l, sigma2, tau)
                + self._node_llik(n_r, s_r, sigma2, tau)
                - self._node_llik(rows.size, s_p, sigma2, tau))

        d = leaf.depth
        p_d = cfg.alpha * (1.0 + d) ** (-cfg.beta)
        p_d1 = cfg.alpha * (2.0 + d) ** (-cfg.beta)
        # structure prior ratio (split-rule probability cancels the proposal)
        log_prior = np.log(p_d) + 2.0 * np.log(1.0 - p_d1) - np.log(1.0 - p_d)
        # transition ratio: w2 = number of nog nodes AFTER the grow
        w2 = len(tree.nog_nodes()) + 1
        log_trans = (np.log(1.0 - cfg.p_grow) - np.log(cfg.p_grow)
                     + np.log(b) - np.log(w2))

        if np.log(self._rng.random()) < log_prior + log_trans + llik:
            leaf.is_leaf = False
            leaf.var, leaf.thr = var, thr
            leaf.left = _Node(d + 1, rows[go_left])
            leaf.right = _Node(d + 1, rows[~go_left])

    def _prune(self, tree: _Tree, X: np.ndarray, resid: np.ndarray,
               sigma2: float, tau: float) -> None:
        cfg = self.cfg
        nogs = tree.nog_nodes()
        if not nogs:
            return
        w2 = len(nogs)
        node = nogs[self._rng.integers(w2)]
        rows = node.rows
        n_l = node.left.rows.size
        n_r = node.right.rows.size
        s_l = float(resid[node.left.rows].sum())
        s_r = float(resid[node.right.rows].sum())
        s_p = s_l + s_r

        llik = (self._node_llik(rows.size, s_p, sigma2, tau)
                - self._node_llik(n_l, s_l, sigma2, tau)
                - self._node_llik(n_r, s_r, sigma2, tau))

        d = node.depth
        p_d = cfg.alpha * (1.0 + d) ** (-cfg.beta)
        p_d1 = cfg.alpha * (2.0 + d) ** (-cfg.beta)
        log_prior = -(np.log(p_d) + 2.0 * np.log(1.0 - p_d1) - np.log(1.0 - p_d))
        b_new = len(tree.leaves()) - 1  # leaves after collapsing this node
        log_trans = (np.log(cfg.p_grow) - np.log(1.0 - cfg.p_grow)
                     + np.log(w2) - np.log(b_new))

        if np.log(self._rng.random()) < log_prior + log_trans + llik:
            node.is_leaf = True
            node.var = node.thr = None
            node.left = node.right = None

    def _draw_leaves(self, tree: _Tree, resid: np.ndarray,
                     sigma2: float, tau: float) -> None:
        """Draw each leaf value from its conjugate normal posterior."""
        for leaf in tree.leaves():
            rows = leaf.rows
            n = rows.size
            s = float(resid[rows].sum()) if n else 0.0
            prec = 1.0 / tau + n / sigma2
            mean = (s / sigma2) / prec
            leaf.value = mean + self._rng.normal() / np.sqrt(prec)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BART":
        """Run the back-fitting MCMC and store posterior summaries."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_ = p

        ys = self._scale_y(y)
        nu, lam = self._calibrate_sigma_prior(X, ys)
        self._nu, self._lambda = nu, lam
        m = self.cfg.n_trees
        tau = (0.5 / (self.cfg.k * np.sqrt(m))) ** 2

        # initialise ensemble: m single-leaf trees, fit = 0
        trees = [_Tree(n) for _ in range(m)]
        tree_fit = np.zeros((m, n))
        total_fit = np.zeros(n)
        sigma2 = float(np.var(ys)) or 1.0

        keep_yhat = []
        keep_sigma = []
        var_counts = np.zeros(p)
        n_kept = 0

        total_iter = self.cfg.n_burn + self.cfg.n_draws
        for it in range(total_iter):
            for j in range(m):
                # partial residual excluding tree j
                resid = ys - (total_fit - tree_fit[j])
                if self._rng.random() < self.cfg.p_grow:
                    self._grow(trees[j], X, resid, sigma2, tau)
                else:
                    self._prune(trees[j], X, resid, sigma2, tau)
                self._draw_leaves(trees[j], resid, sigma2, tau)
                new_fit = trees[j].predict_rows(n)
                total_fit += new_fit - tree_fit[j]
                tree_fit[j] = new_fit

            # draw sigma^2 from its inverse-gamma full conditional
            sse = float(np.sum((ys - total_fit) ** 2))
            shape = 0.5 * (nu + n)
            rate = 0.5 * (nu * lam + sse)
            sigma2 = rate / self._rng.gamma(shape)

            if it >= self.cfg.n_burn and (it - self.cfg.n_burn) % self.cfg.thin == 0:
                keep_yhat.append(total_fit.copy())
                keep_sigma.append(np.sqrt(sigma2))
                for tr in trees:
                    stack = [tr.root]
                    while stack:
                        nd = stack.pop()
                        if not nd.is_leaf:
                            var_counts[nd.var] += 1
                            stack.append(nd.left)
                            stack.append(nd.right)
                n_kept += 1

        self._trees = trees  # final state (used for out-of-sample predict)
        yhat_scaled = np.array(keep_yhat)
        self.yhat_train_draws_ = self._unscale_y(yhat_scaled)
        self.yhat_train_ = self.yhat_train_draws_.mean(axis=0)
        self.sigma_draws_ = np.array(keep_sigma) * self._y_range
        self.var_counts_ = var_counts
        tot = var_counts.sum()
        self.inclusion_proportion_ = (var_counts / tot if tot > 0
                                      else np.zeros(p))
        self.n_kept_ = n_kept
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Posterior-mean prediction for ``X`` (final ensemble state).

        Notes
        -----
        Out-of-sample prediction uses the *final* MCMC state of the
        ensemble, which is the standard cheap point predictor.  For fully
        Bayesian predictive intervals on the training data use
        :attr:`yhat_train_draws_`.
        """
        X = np.asarray(X, dtype=float)
        pred_scaled = np.zeros(X.shape[0])
        for tr in self._trees:
            pred_scaled += tr.predict(X)
        pred = self._unscale_y(pred_scaled)
        if return_std:
            return pred, float(self.sigma_draws_.mean())
        return pred


__all__ = ["BART", "BARTConfig"]
