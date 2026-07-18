"""Reproduce the paper's linear simulation (Table 1 / Figure 1).

Under a *linear* DGP the paper finds BART is competitive but NOT the best:
the linear selectors (LASSO / Elastic Net) have the lowest relative RMSE.
This script re-runs that Monte-Carlo experiment and prints the summary
table plus a box plot.

Run:  python examples/01_simulation_linear.py
"""
import numpy as np

import bartardl as ba
from bartardl import viz
import matplotlib.pyplot as plt

P = 10                 # try 100 to reproduce the high-dimensional panel
N = 100
SIGMA = 1.0
N_REP = 30             # paper uses 100; fewer keeps the demo quick

methods = {
    "BART": lambda: ba.BART(n_trees=50, n_burn=200, n_draws=200, random_state=0),
    "LASSO": ba.Lasso,
    "Elastic-Net": ba.ElasticNet,
    "Bayesian-Net": lambda: ba.BayesianNetwork(n_iter=300, burn=150, random_state=0),
}

rmse_draws = {m: [] for m in methods}
for rep in range(N_REP):
    train = ba.friedman(n=N, p=P, kind="linear", sigma=SIGMA, random_state=rep)
    test = ba.friedman(n=N, p=P, kind="linear", sigma=SIGMA, random_state=1000 + rep)
    for name, ctor in methods.items():
        est = ctor()
        est.fit(train.X, train.y)
        pred = est.predict(test.X)
        rmse_draws[name].append(ba.relative_rmse(test.y, pred, SIGMA))

table = ba.simulation_table(rmse_draws)
print(viz.journal_table(table, caption=f"Linear DGP (p={P}) -- relative RMSE",
                        bold_min_rows=False))

viz.rmse_boxplot(rmse_draws, title=f"Linear DGP (p={P})")
plt.tight_layout()
plt.savefig("sim_linear_boxplot.png")
print("\nSaved sim_linear_boxplot.png")
