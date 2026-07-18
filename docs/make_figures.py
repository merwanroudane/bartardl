"""Generate the figures and tables embedded in the README / docs.

Writes PNGs to ``docs/figures/`` and prints the tables (relative-RMSE
simulation tables and the empirical ranking table) so their numbers can be
quoted in the documentation.  Everything runs on the bundled offline data.

Run:  python docs/make_figures.py
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bartardl as ba
from bartardl import viz

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def simulation(kind, p=10, n=100, sigma=1.0, n_rep=25):
    methods = {
        "BART": lambda: ba.BART(n_trees=50, n_burn=150, n_draws=150, random_state=0),
        "LASSO": ba.Lasso,
        "Elastic-Net": ba.ElasticNet,
        "Bayesian-Net": lambda: ba.BayesianNetwork(n_iter=250, burn=120, random_state=0),
    }
    draws = {m: [] for m in methods}
    last_bart = None
    for rep in range(n_rep):
        tr = ba.friedman(n=n, p=p, kind=kind, sigma=sigma, random_state=rep)
        te = ba.friedman(n=n, p=p, kind=kind, sigma=sigma, random_state=900 + rep)
        for name, ctor in methods.items():
            est = ctor(); est.fit(tr.X, tr.y)
            draws[name].append(ba.relative_rmse(te.y, est.predict(te.X), sigma))
            if name == "BART":
                last_bart = est
    return draws, last_bart


def main():
    # ---- non-linear simulation (BART wins) ------------------------------ #
    nl, bart_nl = simulation("nonlinear")
    print("=== Non-linear DGP (p=10) relative RMSE ===")
    print(ba.simulation_table(nl).round(3).to_string())
    viz.rmse_boxplot(nl, title="Non-linear DGP (p = 10)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "sim_nonlinear_box.png")); plt.close()

    viz.inclusion_plot(bart_nl.inclusion_proportion_,
                       [f"x{i+1}" for i in range(10)],
                       title="BART inclusion proportions (non-linear DGP)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "sim_nonlinear_imp.png")); plt.close()

    # ---- linear simulation (LASSO wins) --------------------------------- #
    lin, _ = simulation("linear")
    print("\n=== Linear DGP (p=10) relative RMSE ===")
    print(ba.simulation_table(lin).round(3).to_string())
    viz.rmse_boxplot(lin, title="Linear DGP (p = 10)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "sim_linear_box.png")); plt.close()

    # ---- empirical ranking (Table 4) ------------------------------------ #
    data = ba.load_macro()
    H = 40
    methods = {
        "BART": (ba.BART, dict(n_trees=100, n_burn=250, n_draws=250, random_state=0)),
        "LASSO": (ba.Lasso, {}),
        "Elastic-Net": (ba.ElasticNet, {}),
        "Bayesian-Net": (ba.BayesianNetwork, dict(n_iter=400, burn=200, random_state=0)),
    }
    train = data.iloc[:-H]
    rmse_by_target, bart_gdp = {}, None
    for tgt in data.columns:
        rmse_by_target[tgt] = {}
        for name, (est, kw) in methods.items():
            mdl = ba.ARDLBART(n_lags=4, estimator=est, estimator_kwargs=kw)
            mdl.fit(train, tgt)
            pred = mdl.predict(data, tgt).iloc[-H:]
            rmse_by_target[tgt][name] = ba.rmse(data[tgt].loc[pred.index], pred)
            if name == "BART" and tgt == "GDP":
                bart_gdp = mdl.result_
    ranking = ba.ranking_table(rmse_by_target)
    print("\n=== Empirical hold-out forecast RMSE (Table 4) ===")
    print(ranking.round(4).to_string())
    print("\nBART wins:", int(ba.best_method_counts(ranking)["BART"]),
          "of", len(data.columns))

    viz.inclusion_plot(bart_gdp.importance, bart_gdp.feature_names, top=12,
                       title="BART inclusion proportions for GDP")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "macro_gdp_imp.png")); plt.close()

    # actual vs fitted for GDP
    viz.forecast_plot(bart_gdp.y_train, bart_gdp.fitted,
                      title="ARDL-BART: actual vs fitted GDP growth")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "macro_gdp_fit.png")); plt.close()

    # parula swatch
    from bartardl.colors import parula_colors
    fig, ax = plt.subplots(figsize=(6, 1.1))
    cols = parula_colors(64)
    for i, c in enumerate(cols):
        ax.axvspan(i, i + 1, color=c)
    ax.set_xlim(0, 64); ax.set_yticks([]); ax.set_xticks([])
    ax.set_title("parula_colors(64)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "parula.png")); plt.close()

    print("\nFigures written to", FIG)


if __name__ == "__main__":
    main()
