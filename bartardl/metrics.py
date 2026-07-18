"""Evaluation metrics and the paper's model-comparison tables.

The paper reports two kinds of tables:

* **Simulation tables (Tables 1 and 2)** -- the mean, median and the
  0.50 / 0.75 quantiles of the *relative* RMSE of each method across
  Monte-Carlo replications.  "Relative" RMSE divides a method's RMSE by
  the irreducible noise level ``sigma`` of the DGP, so a perfect model
  scores 1.0.  Reproduce with :func:`relative_rmse` +
  :func:`simulation_table`.

* **Empirical table (Table 4)** -- the out-of-sample forecast RMSE of each
  method for every target variable, together with a per-target rank.
  Reproduce with :func:`ranking_table`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    """Root-mean-square error."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_rmse(y_true, y_pred, sigma: float) -> float:
    """RMSE divided by the DGP noise level ``sigma`` (paper Tables 1-2)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return rmse(y_true, y_pred) / sigma


def simulation_table(rmse_draws: dict[str, np.ndarray],
                     quantiles=(0.50, 0.75)) -> pd.DataFrame:
    """Summarise Monte-Carlo (relative) RMSE draws as in Tables 1 and 2.

    Parameters
    ----------
    rmse_draws : dict[str, ndarray]
        Mapping ``method name -> array of relative-RMSE values`` (one per
        Monte-Carlo replication).
    quantiles : tuple of float
        Extra quantiles to report alongside the mean and median.

    Returns
    -------
    pandas.DataFrame
        Rows = methods, columns = ``mean``, ``median`` and one column per
        requested quantile (``Q0.50`` ...), matching the paper's layout.
    """
    rows = {}
    for name, draws in rmse_draws.items():
        draws = np.asarray(draws, dtype=float)
        row = {"mean": draws.mean(), "median": np.median(draws)}
        for q in quantiles:
            row[f"Q{q:.2f}"] = np.quantile(draws, q)
        rows[name] = row
    return pd.DataFrame(rows).T


def ranking_table(rmse_by_target: dict[str, dict[str, float]],
                  ascending: bool = True) -> pd.DataFrame:
    """Forecast-RMSE table with per-target ranks (paper Table 4).

    Parameters
    ----------
    rmse_by_target : dict[str, dict[str, float]]
        ``{target variable: {method: rmse}}``.
    ascending : bool
        If ``True`` (default) lower RMSE ranks better (rank 1 = best).

    Returns
    -------
    pandas.DataFrame
        A wide table with, for each method, an ``RMSE`` column and a
        ``Rank`` column, indexed by target variable.
    """
    targets = list(rmse_by_target)
    methods = list(next(iter(rmse_by_target.values())))
    data = {}
    for m in methods:
        data[(m, "RMSE")] = [rmse_by_target[t][m] for t in targets]
    df = pd.DataFrame(data, index=targets)
    # per-target ranks across methods
    rmse_only = pd.DataFrame(
        {m: [rmse_by_target[t][m] for t in targets] for m in methods},
        index=targets,
    )
    ranks = rmse_only.rank(axis=1, ascending=ascending, method="min").astype(int)
    for m in methods:
        df[(m, "Rank")] = ranks[m].values
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # order columns method-by-method: (m, RMSE), (m, Rank)
    ordered = []
    for m in methods:
        ordered.extend([(m, "RMSE"), (m, "Rank")])
    return df[ordered]


def best_method_counts(ranking: pd.DataFrame) -> pd.Series:
    """How many targets each method ranks #1 for (summary of Table 4)."""
    methods = ranking.columns.get_level_values(0).unique()
    counts = {}
    for m in methods:
        counts[m] = int((ranking[(m, "Rank")] == 1).sum())
    return pd.Series(counts, name="wins")


__all__ = [
    "rmse", "relative_rmse", "simulation_table",
    "ranking_table", "best_method_counts",
]
