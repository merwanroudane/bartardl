"""Reproduce the empirical application (paper Section 5, Table 4 / Figure 7).

For each of the eight macro variables we fit a one-equation ARDL(4) with
every estimator, compare one-step-ahead forecast RMSE on a hold-out tail,
rank the methods per target (Table 4), and plot the BART inclusion
proportions for GDP (Figure 7).

By default this runs on the bundled offline panel.  To use the real FRED
data instead:  ``data = ba.load_macro(source="fred")``  (needs internet and
``pip install pandas-datareader``).

Run:  python examples/03_macro_forecasting.py
"""
import numpy as np

import bartardl as ba
from bartardl import viz, make_lag_matrix
import matplotlib.pyplot as plt

data = ba.load_macro()                       # stationary panel
targets = list(data.columns)
H = 40                                       # hold-out length (quarters)

methods = {
    "BART": (ba.BART, dict(n_trees=100, n_burn=300, n_draws=300, random_state=0)),
    "LASSO": (ba.Lasso, {}),
    "Elastic-Net": (ba.ElasticNet, {}),
    "Bayesian-Net": (ba.BayesianNetwork, dict(n_iter=400, burn=200, random_state=0)),
}

train, test = data.iloc[:-H], data
rmse_by_target = {}
bart_gdp = None
for tgt in targets:
    rmse_by_target[tgt] = {}
    for name, (est, kw) in methods.items():
        model = ba.ARDLBART(n_lags=4, estimator=est, estimator_kwargs=kw)
        model.fit(train, tgt)
        pred = model.predict(test, tgt).iloc[-H:]
        actual = data[tgt].loc[pred.index]
        rmse_by_target[tgt][name] = ba.rmse(actual, pred)
        if name == "BART" and tgt == "GDP":
            bart_gdp = model.result_

ranking = ba.ranking_table(rmse_by_target)
print(viz.journal_table(ranking, caption="Table 4 -- hold-out forecast RMSE by target",
                        bold_min_rows=False))
print("\nBART is best (rank 1) for", int(ba.best_method_counts(ranking)["BART"]),
      "of", len(targets), "targets")

# Figure 7: BART inclusion proportions for GDP
imp = bart_gdp.importance
viz.inclusion_plot(imp, bart_gdp.feature_names, top=12,
                   title="Figure 7 -- BART inclusion proportions for GDP")
plt.tight_layout(); plt.savefig("macro_gdp_importance.png")
print("Saved macro_gdp_importance.png")
