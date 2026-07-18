"""The ARDL-BART model: a single-equation, multi-lag ARDL whose conditional
mean is estimated non-parametrically.

The paper (Section 1) treats an ARDL specification as "a VAR taken one
equation at a time": each target variable is regressed on ``p`` lags of
*every* variable in the system (including its own), with no cross-equation
error covariance.  The novelty is that the conditional mean

    y_t = f( y_{t-1}, ..., x_{k, t-1}, ... ) + e_t

is fitted with BART instead of OLS, so it can capture the non-linear and
interaction effects the paper argues dominate macroeconomic data.

This module provides:

* :func:`make_lag_matrix` -- build the ARDL design matrix and aligned
  target from a multivariate time series.
* :func:`transform_series` -- the stationarity transformations of the
  paper's Table 3 (FRED-MD transformation codes).
* :class:`ARDLBART` -- fit one ARDL equation for a chosen target with any
  estimator that follows the ``fit``/``predict``/``importance_`` protocol
  (BART by default; the competitors are drop-in replacements).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .bart import BART


# --------------------------------------------------------------------------- #
# Stationarity transformations (paper Table 3 / FRED-MD codes)
# --------------------------------------------------------------------------- #
def transform_series(x: np.ndarray, code: int) -> np.ndarray:
    """Apply a FRED-MD stationarity transformation.

    Codes (paper Table 3)::

        1 = no transformation          4 = log
        2 = first difference           5 = first difference of log
        3 = second difference          6 = second difference of log

    The returned array has the same length as ``x`` with leading ``Na`` for
    the observations lost to differencing.
    """
    x = np.asarray(x, dtype=float)
    if code == 1:
        return x
    if code == 2:
        return np.concatenate([[np.nan], np.diff(x)])
    if code == 3:
        return np.concatenate([[np.nan, np.nan], np.diff(x, n=2)])
    if code == 4:
        return np.log(x)
    if code == 5:
        return np.concatenate([[np.nan], np.diff(np.log(x))])
    if code == 6:
        return np.concatenate([[np.nan, np.nan], np.diff(np.log(x), n=2)])
    raise ValueError(f"unknown transformation code {code!r}")


def transform_frame(df: pd.DataFrame, codes: dict[str, int]) -> pd.DataFrame:
    """Transform every column of ``df`` by its code and drop the lost rows."""
    out = {c: transform_series(df[c].to_numpy(), codes[c]) for c in df.columns}
    res = pd.DataFrame(out, index=df.index)
    return res.dropna()


# --------------------------------------------------------------------------- #
# ARDL design matrix
# --------------------------------------------------------------------------- #
def make_lag_matrix(data: pd.DataFrame, target: str, n_lags: int,
                    include_target_lags: bool = True):
    """Build the ARDL design matrix for one target equation.

    Parameters
    ----------
    data : pandas.DataFrame
        The (already stationary) multivariate time series, one column per
        variable, ordered in time.
    target : str
        Column to be explained.
    n_lags : int
        Number of lags of every variable to include (the paper uses 4).
    include_target_lags : bool
        Whether lags of the target itself enter the regressors (they do in
        a genuine ARDL; set ``False`` for a pure distributed-lag model).

    Returns
    -------
    X : pandas.DataFrame
        Design matrix with columns named ``"<var>.lag<k>"``.
    y : pandas.Series
        The contemporaneous target, aligned to ``X``.
    """
    if target not in data.columns:
        raise KeyError(f"{target!r} is not a column of data")
    cols = []
    frames = []
    for var in data.columns:
        if var == target and not include_target_lags:
            continue
        for k in range(1, n_lags + 1):
            frames.append(data[var].shift(k).rename(f"{var}.lag{k}"))
            cols.append(f"{var}.lag{k}")
    X = pd.concat(frames, axis=1)
    y = data[target]
    both = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    return both[cols], both["__y__"].rename(target)


# --------------------------------------------------------------------------- #
# Fitted-model container
# --------------------------------------------------------------------------- #
@dataclass
class ARDLResult:
    """Result of fitting one ARDL equation."""

    target: str
    n_lags: int
    feature_names: list[str]
    estimator: object
    fitted: np.ndarray            # in-sample fitted values
    resid: np.ndarray             # in-sample residuals
    y_train: np.ndarray
    importance: np.ndarray        # per-feature importance / inclusion prop.

    def importance_frame(self) -> pd.DataFrame:
        """Importance as a tidy, sorted DataFrame (variable, lag, weight)."""
        recs = []
        for name, w in zip(self.feature_names, self.importance):
            var, lag = name.rsplit(".lag", 1)
            recs.append({"feature": name, "variable": var,
                         "lag": int(lag), "importance": float(w)})
        return (pd.DataFrame(recs)
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))

    def top_features(self, k: int = 5) -> pd.DataFrame:
        return self.importance_frame().head(k)


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class ARDLBART:
    """Single-equation ARDL with a pluggable (non-parametric) conditional mean.

    Parameters
    ----------
    n_lags : int
        Lag order ``p`` of the ARDL (paper default: 4).
    estimator : object or callable, optional
        The conditional-mean model.  Anything exposing
        ``fit(X, y)`` / ``predict(X)`` and an ``importance_`` (or BART's
        ``inclusion_proportion_``) attribute works.  A **callable** (e.g.
        the :class:`~bartardl.bart.BART` class) is instantiated per fit so
        each equation gets a fresh model; an **instance** is cloned via its
        constructor arguments where possible, else reused.  Defaults to
        :class:`~bartardl.bart.BART`.
    include_target_lags : bool
        Passed to :func:`make_lag_matrix`.
    estimator_kwargs : dict, optional
        Keyword arguments forwarded when instantiating ``estimator``.
    """

    def __init__(self, n_lags: int = 4, estimator=BART,
                 include_target_lags: bool = True,
                 estimator_kwargs: Optional[dict] = None):
        self.n_lags = n_lags
        self.estimator = estimator
        self.include_target_lags = include_target_lags
        self.estimator_kwargs = estimator_kwargs or {}

    # -- helpers ---------------------------------------------------------- #
    def _new_estimator(self):
        est = self.estimator
        if isinstance(est, type) or callable(est) and not hasattr(est, "fit"):
            return est(**self.estimator_kwargs)
        return est  # a ready instance (reused)

    @staticmethod
    def _get_importance(est, n_features):
        for attr in ("inclusion_proportion_", "importance_"):
            if hasattr(est, attr):
                return np.asarray(getattr(est, attr), dtype=float)
        return np.full(n_features, np.nan)

    # -- API -------------------------------------------------------------- #
    def fit(self, data: pd.DataFrame, target: str) -> ARDLResult:
        """Fit the ARDL equation for ``target`` on ``data``."""
        X, y = make_lag_matrix(data, target, self.n_lags,
                               self.include_target_lags)
        est = self._new_estimator()
        est.fit(X.to_numpy(), y.to_numpy())
        fitted = np.asarray(est.predict(X.to_numpy()), dtype=float)
        result = ARDLResult(
            target=target,
            n_lags=self.n_lags,
            feature_names=list(X.columns),
            estimator=est,
            fitted=fitted,
            resid=y.to_numpy() - fitted,
            y_train=y.to_numpy(),
            importance=self._get_importance(est, X.shape[1]),
        )
        self.result_ = result
        self._design_columns = list(X.columns)
        return result

    def predict(self, data: pd.DataFrame, target: str) -> pd.Series:
        """Predict ``target`` on new (or the same) data using the fit model."""
        if not hasattr(self, "result_"):
            raise RuntimeError("call fit() before predict()")
        X, y = make_lag_matrix(data, target, self.n_lags,
                               self.include_target_lags)
        X = X[self._design_columns]
        pred = self.result_.estimator.predict(X.to_numpy())
        return pd.Series(pred, index=y.index, name=f"{target}_hat")


__all__ = [
    "ARDLBART", "ARDLResult", "make_lag_matrix",
    "transform_series", "transform_frame",
]
