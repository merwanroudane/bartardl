"""Reproduce the paper's non-linear simulation (Table 2 / Figures 4-6).

Under the classic Friedman *non-linear* DGP, BART attains the lowest
relative RMSE -- the paper's central simulation result.  This script
re-runs the experiment, prints the summary table, saves the RMSE box plot
and the BART variable-inclusion figure.

Run:  python examples/02_simulation_nonlinear.py
"""
import numpy as np

import bartardl as ba
from bartardl import viz
import matplotlib.pyplot as plt

P = 10
N = 100
SIGMA = 1.0
N_REP = 30

methods = {
    "BART": lambda: ba.BART(n_trees=50, n_burn=200, n_draws=200, random_state=0),
    "LASSO": ba.Lasso,
    "Elastic-Net": ba.ElasticNet,
    "Bayesian-Net": lambda: ba.BayesianNetwork(n_iter=300, burn=150, random_state=0),
}

rmse_draws = {m: [] for m in methods}
last_bart = None
for rep in range(N_REP):
    train = ba.friedman(n=N, p=P, kind="nonlinear", sigma=SIGMA, random_state=rep)
    test = ba.friedman(n=N, p=P, kind="nonlinear", sigma=SIGMA, random_state=1000 + rep)
    for name, ctor in methods.items():
        est = ctor()
        est.fit(train.X, train.y)
        pred = est.predict(test.X)
        rmse_draws[name].append(ba.relative_rmse(test.y, pred, SIGMA))
        if name == "BART":
            last_bart = est

table = ba.simulation_table(rmse_draws)
print(viz.journal_table(table, caption=f"Non-linear DGP (p={P}) -- relative RMSE"))

viz.rmse_boxplot(rmse_draws, title=f"Non-linear DGP (p={P})")
plt.tight_layout(); plt.savefig("sim_nonlinear_boxplot.png")

names = [f"x{i+1}" for i in range(P)]
viz.inclusion_plot(last_bart.inclusion_proportion_, names,
                   title="BART variable inclusion (non-linear DGP)")
plt.tight_layout(); plt.savefig("sim_nonlinear_importance.png")
print("\nSaved sim_nonlinear_boxplot.png and sim_nonlinear_importance.png")
