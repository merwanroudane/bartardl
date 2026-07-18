"""bartardl -- Non-parametric ARDL modelling with Bayesian Additive Regression Trees.

A faithful Python implementation of

    Mahdavi, P., Ehsani, M. A., Ahelegbey, D. F. and Mohammadpour, M. (2024).
    "Measuring Causal Effect with ARDL-BART: A Macroeconomic Application."
    Applied Mathematics, 15, 292-312. <doi:10.4236/am.2024.154018>

The package fits an autoregressive distributed lag (ARDL) model whose
conditional mean is estimated non-parametrically with Bayesian Additive
Regression Trees (BART), and benchmarks it against LASSO, Elastic Net and a
Bayesian-Network (spike-and-slab) selector -- with publication-quality
figures and tables.

Quick start
-----------
>>> from bartardl import ARDLBART, load_macro
>>> data = load_macro()                      # stationary US macro panel
>>> model = ARDLBART(n_lags=4)               # ARDL(4) with a BART mean
>>> res = model.fit(data, target="GDP")
>>> res.top_features(5)                       # inclusion-proportion importance
"""

from .bart import BART, BARTConfig
from .ardl import (ARDLBART, ARDLResult, make_lag_matrix,
                   transform_series, transform_frame)
from .competitors import Lasso, ElasticNet, BayesianNetwork
from .simulate import friedman, SimData
from .metrics import (rmse, relative_rmse, simulation_table,
                      ranking_table, best_method_counts)
from .datasets import (load_macro, fetch_fred_macro, SERIES,
                       TRANSFORM_CODES, transform_table)
from . import viz, colors

__version__ = "0.1.0"

__all__ = [
    # core
    "BART", "BARTConfig", "ARDLBART", "ARDLResult",
    "make_lag_matrix", "transform_series", "transform_frame",
    # competitors
    "Lasso", "ElasticNet", "BayesianNetwork",
    # simulation & metrics
    "friedman", "SimData", "rmse", "relative_rmse",
    "simulation_table", "ranking_table", "best_method_counts",
    # data
    "load_macro", "fetch_fred_macro", "SERIES", "TRANSFORM_CODES",
    "transform_table",
    # viz / colours
    "viz", "colors",
    "__version__",
]
