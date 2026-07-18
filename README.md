# bartardl

**Non-parametric ARDL modelling with Bayesian Additive Regression Trees (BART).**

A faithful, self-contained Python implementation of

> Mahdavi, P., Ehsani, M. A., Ahelegbey, D. F. and Mohammadpour, M. (2024).
> *Measuring Causal Effect with ARDL-BART: A Macroeconomic Application.*
> **Applied Mathematics**, 15, 292–312.
> [doi:10.4236/am.2024.154018](https://doi.org/10.4236/am.2024.154018)

`bartardl` fits an **autoregressive distributed lag (ARDL)** model whose
conditional mean is estimated **non-parametrically with BART** instead of OLS,
so it can capture the non-linear and interaction effects the paper argues
dominate macroeconomic data. It benchmarks BART against **LASSO**, **Elastic
Net** and a **Bayesian-Network** (spike-and-slab) selector, and ships
publication-quality figures and tables.

The BART sampler is written from scratch in NumPy — no external BART engine —
so the package installs with only `numpy`, `scipy`, `pandas`, `scikit-learn`
and `matplotlib`.

---

## Highlights

- 🌲 **Faithful BART** — sum-of-trees with the exact priors of Chipman, George
  and McCulloch (2010): tree prior `α(1+d)^(−β)`, conjugate-normal leaves, and
  the `(ν, q)`-calibrated inverse-gamma error prior. Paper defaults
  (`m = 200, k = 2, ν = 3, q = 0.9, α = 0.95, β = 2`) are the library defaults.
- 🔁 **ARDL layer** — one equation per target, every variable entered at lags
  `1…p`, with the paper's Table 3 stationarity transforms built in.
- 🧮 **Competitors as drop-ins** — `Lasso`, `ElasticNet`, `BayesianNetwork`
  share the estimator protocol, so any of them can be the ARDL conditional mean.
- 📊 **Top-journal graphics** — RMSE box plots, variable-inclusion bars with
  95% whiskers, actual-vs-fitted overlays, LaTeX/HTML/console tables, and the
  MATLAB **Parula** palette (`parula_colors`, `turbo_colors`, …).
- 🧪 **Reproduces the paper** — the Friedman simulations and the eight-variable
  US-macro application, with an offline synthetic panel so every example runs
  without a network.

---

## Installation

```bash
git clone https://github.com/merwanroudane/bartardl.git
cd bartardl
pip install -e .

# optional: live FRED download for the real macro data
pip install -e ".[fred]"
```

Requires Python ≥ 3.9.

---

## Quick start

```python
import bartardl as ba

data  = ba.load_macro()                     # stationary US macro panel (8 vars)
model = ba.ARDLBART(                         # ARDL(4) with a BART conditional mean
    n_lags=4,
    estimator=ba.BART,
    estimator_kwargs=dict(n_trees=100, n_burn=500, n_draws=500, random_state=0),
)
res = model.fit(data, target="GDP")

print("In-sample RMSE:", ba.rmse(res.y_train, res.fitted))
print(res.top_features(5))                  # 5 most influential lagged regressors
```

```
In-sample RMSE: 0.0019

  feature variable  lag  importance
  PC.lag1       PC    1    0.0892
  IP.lag1       IP    1    0.0760
 GDP.lag1      GDP    1    0.0601
   U.lag3        U    3    0.0402
 INF.lag3      INF    3    0.0401
```

(this is exactly `python examples/quickstart.py`)

---

## The model

For a target `y` the ARDL-BART equation is

```
y_t = f( y_{t-1}, …, y_{t-p},  x_{1,t-1}, …, x_{k,t-p} ) + e_t ,   e_t ~ N(0, σ²)
```

and `f` is a **sum of `m` regression trees** (BART):

```
f(x) = Σ_{l=1}^{m} g(x ; T_l , M_l)
```

The priors (paper §2) regularise every tree to be a weak learner:

| Component | Prior | Default |
|---|---|---|
| Tree structure | split prob. at depth `d` = `α (1+d)^(−β)` | `α = 0.95`, `β = 2` |
| Leaf value | `μ ~ N(0, τ)`, `τ = (0.5 / (k√m))²` (y scaled to `[−0.5, 0.5]`) | `k = 2` |
| Error variance | `σ² ~ InvGamma(ν/2, νλ/2)`, `λ` from the `q`-quantile rule | `ν = 3`, `q = 0.9` |

The ensemble is drawn by **Bayesian back-fitting MCMC**: each sweep proposes a
single *grow* or *prune* move per tree (Metropolis–Hastings), then re-samples the
leaf values and `σ²`. Variable importance is the **inclusion proportion** — the
share of all splitting rules that use each predictor.

> **Faithfulness notes.** The paper's "ARDL" is the single-equation,
> distributed-lag regression sense (no bounds test / no error-correction term),
> and its "causal effect" framing is operationalised here as forecasting +
> variable importance, exactly as in the paper's own experiments. See
> [`docs/guide.md`](docs/guide.md) for a point-by-point reconciliation with the
> article.

---

## Reproducing the paper

### 1 · Simulations (Tables 1–2, Figures 1–6)

`python examples/01_simulation_linear.py` and
`python examples/02_simulation_nonlinear.py`.

Relative RMSE (RMSE ÷ noise σ; a perfect model scores 1.0), `p = 10`:

**Non-linear DGP — BART wins** (paper Table 2)

| method | mean | median | Q0.50 | Q0.75 |
|---|---|---|---|---|
| **BART** | **2.554** | **2.534** | **2.534** | **2.735** |
| LASSO | 2.750 | 2.741 | 2.741 | 2.951 |
| Elastic-Net | 2.753 | 2.743 | 2.743 | 2.951 |
| Bayesian-Net | 2.727 | 2.714 | 2.714 | 2.939 |

**Linear DGP — the linear selectors win** (paper Table 1)

| method | mean | median | Q0.50 | Q0.75 |
|---|---|---|---|---|
| BART | 1.474 | 1.418 | 1.418 | 1.530 |
| LASSO | 1.033 | 1.023 | 1.023 | 1.080 |
| Elastic-Net | 1.033 | 1.023 | 1.023 | 1.080 |
| **Bayesian-Net** | **1.023** | **1.009** | **1.009** | **1.072** |

<p align="center">
  <img src="docs/figures/sim_nonlinear_box.png" width="46%">
  <img src="docs/figures/sim_nonlinear_imp.png" width="46%">
</p>

### 2 · Empirical application (Table 4, Figure 7)

`python examples/03_macro_forecasting.py` — eight US macro variables,
ARDL(4), hold-out forecast RMSE with per-target ranks:

| target | BART | LASSO | Elastic-Net | Bayesian-Net | best |
|---|---|---|---|---|---|
| GDP | **0.0032** | 0.0057 | 0.0057 | 0.0103 | BART |
| INF | **0.0036** | 0.0037 | 0.0037 | 0.0075 | BART |
| FF  | 0.1216 | **0.0895** | 0.0905 | 0.0937 | LASSO |
| M2  | 0.0035 | **0.0031** | 0.0031 | 0.0076 | LASSO |
| PC  | **0.0039** | 0.0056 | 0.0056 | 0.0091 | BART |
| IP  | **0.0037** | 0.0065 | 0.0065 | 0.0099 | BART |
| U   | **0.1051** | 0.1251 | 0.1252 | 0.1367 | BART |
| INV | **0.0118** | 0.0149 | 0.0149 | 0.0162 | BART |

**BART ranks first for 6 of 8 targets**, with the linear selectors taking the
two (near-)linear series — the same qualitative pattern as the paper's Table 4.

<p align="center">
  <img src="docs/figures/macro_gdp_imp.png" width="46%">
  <img src="docs/figures/macro_gdp_fit.png" width="46%">
</p>

> The bundled panel is a **reproducible synthetic** data set (built so its
> Table-3 transforms carry genuine, economically-motivated non-linearity) so the
> example runs offline. For the real series use
> `ba.load_macro(source="fred")` (needs internet + `pandas-datareader`).

---

## API / syntax reference

### `ARDLBART(n_lags=4, estimator=BART, include_target_lags=True, estimator_kwargs=None)`

Single-equation ARDL with a pluggable conditional mean.

```python
model = ba.ARDLBART(
    n_lags=4,                                   # lag order p
    estimator=ba.BART,                          # any fit/predict/importance_ model
    estimator_kwargs=dict(n_trees=200, n_draws=1000, random_state=0),
)
res = model.fit(data, target="GDP")             # -> ARDLResult
pred = model.predict(new_data, target="GDP")    # -> pandas Series
```

`ARDLResult` exposes `.fitted`, `.resid`, `.y_train`, `.feature_names`,
`.importance`, `.importance_frame()` and `.top_features(k)`.

### `BART(**config)`

```python
m = ba.BART(
    n_trees=200,      # m           n_draws=1000,   # retained draws
    k=2.0,            #             n_burn=1000,    # burn-in
    nu=3.0, q=0.90,   # σ prior      alpha=0.95, beta=2.0,   # tree prior
    random_state=0,
).fit(X, y)

m.predict(Xnew)                 # posterior-mean prediction
m.predict(Xnew, return_std=True)
m.inclusion_proportion_         # variable importance (sums to 1)
m.sigma_draws_                  # posterior σ draws
m.yhat_train_draws_             # full posterior of the training fit
```

Set `n_draws=20000, n_burn=250` to match the paper exactly.

### Competitors (same protocol)

```python
ba.Lasso(cv=5)
ba.ElasticNet(cv=5, l1_ratio=[0.1, 0.5, 0.9, 0.99])
ba.BayesianNetwork(n_iter=2000, burn=1000, pi=0.5)   # spike-and-slab SSVS
```

Each has `.fit`, `.predict`, `.importance_` and `.coef_`.

### Simulation, metrics, data

```python
ba.friedman(n=100, p=10, kind="nonlinear", sigma=1.0, random_state=0)
ba.relative_rmse(y_true, y_pred, sigma)
ba.simulation_table(rmse_draws)                 # Tables 1–2
ba.ranking_table(rmse_by_target)                # Table 4
ba.load_macro(source="auto")                    # "bundled" | "synthetic" | "fred"
ba.transform_table()                            # paper Table 3
```

### Figures & colours

```python
from bartardl import viz
viz.rmse_boxplot(rmse_draws, title="…")
viz.inclusion_plot(importance, feature_names, ci=..., top=12)
viz.forecast_plot(y_true, y_pred)
viz.journal_table(df, fmt="latex", caption="…")   # "console" | "html" | "latex"

from bartardl.colors import parula_colors, get_cmap
parula_colors(8)                                  # 8 hex stops of MATLAB Parula
get_cmap("Parula")                                # a Matplotlib colormap
```

<p align="center"><img src="docs/figures/parula.png" width="70%"></p>

---

## Documentation

- **[`docs/guide.md`](docs/guide.md)** — a detailed, from-the-ground-up guide to
  *how the code is written*: the BART sampler internals (grow/prune acceptance
  ratios, back-fitting, σ calibration), the ARDL design, how the competitors and
  the SSVS sampler work, how to extend the package, and a full reconciliation
  with the paper.
- **[`examples/`](examples)** — runnable scripts for the quick start, both
  simulations, and the macro application.
- **`docs/make_figures.py`** — regenerates every figure and table in this README.

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Citing

If you use this software, please cite the paper and the package:

```bibtex
@article{mahdavi2024ardlbart,
  author  = {Mahdavi, Pegah and Ehsani, Mohammad Ali and
             Ahelegbey, Daniel Felix and Mohammadpour, Mehrnaz},
  title   = {Measuring Causal Effect with {ARDL-BART}: A Macroeconomic Application},
  journal = {Applied Mathematics},
  year    = {2024},
  volume  = {15},
  pages   = {292--312},
  doi     = {10.4236/am.2024.154018}
}

@software{roudane_bartardl,
  author = {Merwan Roudane},
  title  = {bartardl: Non-parametric ARDL modelling with BART},
  url    = {https://github.com/merwanroudane/bartardl}
}
```

## License

MIT © Merwan Roudane. See [LICENSE](LICENSE).
