# `bartardl` — a detailed developer & methodology guide

This guide explains **how the package is built**, function by function, so you
can read, verify, extend or port the code. It also reconciles the
implementation with the source paper.

> Mahdavi, Ehsani, Ahelegbey & Mohammadpour (2024), *Measuring Causal Effect
> with ARDL-BART*, Applied Mathematics 15, 292–312,
> [doi:10.4236/am.2024.154018](https://doi.org/10.4236/am.2024.154018).

Contents:

1. [Package map](#1-package-map)
2. [The BART sampler (`bart.py`)](#2-the-bart-sampler-bartpy)
3. [The ARDL layer (`ardl.py`)](#3-the-ardl-layer-ardlpy)
4. [Competitors (`competitors.py`)](#4-competitors-competitorspy)
5. [Simulation, metrics, data](#5-simulation-metrics-data)
6. [Visualisation & colours](#6-visualisation--colours)
7. [Reconciliation with the paper](#7-reconciliation-with-the-paper)
8. [Extending the package](#8-extending-the-package)

---

## 1 · Package map

```
bartardl/
  bart.py         BART sum-of-trees sampler (self-contained NumPy)
  ardl.py         ARDL lag construction + transforms + ARDLBART wrapper
  competitors.py  Lasso, ElasticNet, BayesianNetwork (SSVS)
  simulate.py     Friedman linear / non-linear DGPs
  metrics.py      RMSE, relative RMSE, simulation & ranking tables
  datasets.py     US-macro loader (bundled / synthetic / live FRED)
  colors.py       Parula & friends colour maps
  viz.py          box plots, inclusion plots, journal tables
  data/us_macro.csv   bundled offline panel
```

Everything is glued together by one protocol. A **conditional-mean estimator**
is any object with:

```python
est.fit(X, y)            # X: (n, p) ndarray, y: (n,) ndarray  -> self
est.predict(Xnew)        # -> (m,) ndarray
est.importance_          # (p,) non-negative weights (BART: inclusion_proportion_)
```

`BART`, `Lasso`, `ElasticNet` and `BayesianNetwork` all satisfy it, which is why
any of them can be dropped into `ARDLBART(estimator=…)`.

---

## 2 · The BART sampler (`bart.py`)

BART approximates the regression function by a **sum of `m` shallow trees**:

```
y = f(x) + e,  e ~ N(0, σ²),   f(x) = Σ_l g(x; T_l, M_l).
```

### 2.1 Data structures

- `_Node` — one tree node. A **leaf** holds `value` (the leaf parameter `μ`); an
  **internal** node holds a rule `x[:, var] <= thr` and two children. Each node
  caches `rows`, the training indices that reach it, so grow/prune stay *local*
  (no full re-traversal).
- `_Tree` — rooted at a single leaf covering all rows. Helpers: `leaves()`,
  `nog_nodes()` (internal nodes whose **both** children are leaves — the only
  prunable ones), `predict_rows()` (in-sample), `predict()` (out-of-sample).

### 2.2 Response scaling and priors

`fit` first shifts/scales `y` into `[−0.5, 0.5]`:

```
y_center = (min+max)/2,  y_range = max−min,  y_scaled = (y − y_center)/y_range.
```

This lets the **leaf prior** be the standard `μ ~ N(0, τ)` with

```
τ = (0.5 / (k·√m))².
```

Because `m` independent leaves sum to `f(x)`, the implied prior on `f` is roughly
`N(0, (0.5/k)²)`, so `±k` prior SDs cover the data range — that is the meaning of
`k` (default 2).

The **error-variance prior** `σ² ~ InvGamma(ν/2, νλ/2)` is calibrated by
`_calibrate_sigma_prior`. With `σ̂` = residual SD of an OLS fit of the scaled
response, `λ` is set so there is prior probability `q` that `σ < σ̂`:

```
νλ / σ̂² = χ²_ν quantile at (1−q)   ⇒   λ = σ̂²·χ²_{ν}(1−q)/ν.
```

`ν = 3, q = 0.9` (the paper) put most prior mass on models that beat OLS.

### 2.3 The marginal likelihood of a node

For grow/prune we integrate out the leaf value `μ`. A node holding `n` residuals
that sum to `s`, with leaf prior `N(0, τ)` and noise `σ²`, has log marginal
likelihood (dropping the `Σr_i²` term that cancels in every ratio):

```
LL(n, s) = ½·log( σ² / (σ² + nτ) ) + τ·s² / ( 2σ²(σ² + nτ) ).
```

This is `_node_llik`.

### 2.4 Grow / prune Metropolis–Hastings

Per sweep, per tree, we propose **grow** (prob. `p_grow`, default 0.5) or
**prune**. Choosing the split variable uniformly and the split value uniformly
makes the rule-selection probability **cancel** between the tree-structure prior
and the transition proposal, leaving a compact acceptance ratio.

**Grow** (split a randomly chosen leaf at depth `d`; `b` = #leaves before,
`w2` = #nog nodes after):

```
log A = log p(d) + 2 log(1−p(d+1)) − log(1−p(d))          # tree-structure prior
      + log(1−p_grow) − log(p_grow) + log(b) − log(w2)      # transition
      + LL(nL,sL) + LL(nR,sR) − LL(nP,sP)                   # marginal likelihood
```

with `p(d) = α(1+d)^(−β)`. **Prune** is the exact reverse. See `_grow` / `_prune`.
Proposals that would leave a child with fewer than `min_leaf` rows, or split a
constant column, are rejected up front.

### 2.5 Leaf draw and σ update

After the structure move, `_draw_leaves` samples each leaf from its
conjugate-normal full conditional:

```
precision = 1/τ + n/σ²,   mean = (s/σ²)/precision,   μ ~ N(mean, 1/precision).
```

Once all trees are updated, `σ²` is drawn from its inverse-gamma full
conditional given the full residuals `e = y_scaled − Σ_j f_j`:

```
σ² ~ InvGamma( (ν+n)/2 , (νλ + Σe²)/2 ).
```

### 2.6 Back-fitting loop and outputs

The sampler keeps a running `total_fit` and each tree's contribution `tree_fit[j]`,
so the **partial residual** for tree `j` is `y − (total_fit − tree_fit[j])` — an
`O(n)` update rather than recomputing the ensemble. After burn-in it records, per
kept draw: the training fit, `σ`, and every split variable (for inclusion
proportions). Exposed afterwards:

| attribute | meaning |
|---|---|
| `yhat_train_` | posterior-mean fit (original scale) |
| `yhat_train_draws_` | full posterior of the training fit |
| `sigma_draws_` | posterior σ draws (original scale) |
| `inclusion_proportion_` | variable importance, sums to 1 |
| `var_counts_` | raw split counts per predictor |

**Cost.** One sweep is `O(m · n)` plus the local moves. The library defaults
(`n_trees=200`, `n_burn=1000`, `n_draws=1000`) run a small macro equation in a
few seconds; the paper's `20000/250` setting is available but slower in pure
Python.

---

## 3 · The ARDL layer (`ardl.py`)

### 3.1 Stationarity transforms — paper Table 3

`transform_series(x, code)` implements the FRED-MD codes: `1` none, `2` Δ,
`3` Δ², `4` log, `5` Δlog, `6` Δ²log. `transform_frame(df, codes)` applies a
code per column and drops the rows lost to differencing.

### 3.2 The design matrix

`make_lag_matrix(data, target, n_lags, include_target_lags=True)` stacks lags
`1…p` of **every** column (the paper's "VAR one equation at a time"):

- columns are named `"<var>.lag<k>"`;
- with 8 variables and `p = 4` that is **32 regressors** — exactly the paper's
  empirical specification;
- rows with any NaN (from lagging) are dropped and `X`, `y` are aligned.

### 3.3 `ARDLBART`

`fit(data, target)` builds the design, instantiates a **fresh** estimator (so
every equation is independent — the paper fits equation by equation), fits it,
and packages an `ARDLResult` with fitted values, residuals, feature names and the
importance vector (read from `inclusion_proportion_` or `importance_`).
`predict(data, target)` rebuilds the design with the stored column order and
returns a date-indexed `Series`.

`ARDLResult.importance_frame()` splits each `"<var>.lag<k>"` name back into
`(variable, lag, importance)` and sorts — the tidy form used by the figures.

---

## 4 · Competitors (`competitors.py`)

`Lasso` and `ElasticNet` wrap scikit-learn's cross-validated `LassoCV` /
`ElasticNetCV` on standardised predictors; importance is `|coef|` normalised.

`BayesianNetwork` is a **stochastic-search variable-selection (SSVS)** /
spike-and-slab Gibbs sampler:

```
y = Xβ + u,  u ~ N(0, σ²I),
β_j | γ_j ~ γ_j·N(0, c·τ²) + (1−γ_j)·N(0, τ²),   γ_j ~ Bernoulli(π).
```

Each sweep: draw every `β_j` from its normal full conditional; flip `γ_j` from a
Bernoulli whose odds compare the slab vs spike density at the current `β_j`; draw
`σ²` from its inverse-gamma. Following Ahelegbey–Billio–Casarin (2016), `γ_j = 1`
is read as a **present edge** between predictor `j` and the response, so the
posterior mean of `γ` gives the inclusion probabilities used for importance, and
the posterior mean of `β` is used to predict. This is a faithful, dependency-free
stand-in for the paper's Bayesian-network selector.

---

## 5 · Simulation, metrics, data

- **`simulate.friedman`** — draws `x ~ U(0,1)^p`. Non-linear (Eq. 12):
  `10·sin(π x1 x2) + 20(x3−0.5)² + 10 x4 + 5 x5 + e`. Linear (Eq. 11): the
  additive linearisation `2 + 3(x1+…+x5) + e` (the published Eq. 11 is garbled in
  the PDF text layer). Only `x1…x5` are relevant.
- **`metrics`** — `rmse`, `relative_rmse` (÷ σ, so 1.0 is perfect),
  `simulation_table` (mean/median/quantiles → Tables 1–2), `ranking_table`
  (per-target RMSE + rank → Table 4), `best_method_counts`.
- **`datasets`** — `SERIES` documents the eight variables and their FRED codes
  (Table 3). `load_macro` returns the transformed panel from the bundled CSV, a
  fresh synthetic panel, or a **live FRED** pull. The synthetic generator builds
  each series in *transformed space* with smooth sinusoid-interaction and
  threshold dynamics, then reconstructs levels by inverting the transform — so
  the Table-3 transforms recover exactly the non-linear target a model must learn
  (which is why BART beats the linear selectors there, mirroring the paper).

---

## 6 · Visualisation & colours

`colors.parula_colors(n)` interpolates `n` hex stops across the **64 MATLAB
R2014b Parula control points** (the same `colorRampPalette` idea as the author's
R packages); `matlab_jet_colors`, `turbo_colors`, `bluered_colors`,
`sinha_colors`, `get_cmap` and `resolve_colorscale` follow suit.

`viz` provides `set_journal_style`, `rmse_boxplot` (Figs 1/4), `inclusion_plot`
(Figs 2/7, with optional 95% whiskers), `forecast_plot`, and `journal_table`
(console / LaTeX `booktabs` / HTML). Every plotting function returns the
Matplotlib `Axes` so you can compose or save.

---

## 7 · Reconciliation with the paper

The implementation is faithful to what the paper *does*; two labels in the paper
are looser than their names suggest, and the package is explicit about it.

| Paper term | What the paper actually does | This package |
|---|---|---|
| "ARDL" | single-equation distributed-lag regression on lags of all variables | `make_lag_matrix` + `ARDLBART` — **no** bounds test / ECM (none is in the paper) |
| "causal effect" | forecasting + variable-inclusion importance | same; treatment-effect machinery is **not** claimed |
| Tables 1–2 "relative RMSE" | RMSE relative to a baseline noise level | `relative_rmse(…, sigma)` |
| σ prior `v, q` | `ν ∈ [3,10]`, `q = 0.9` | `nu`, `q` in `BARTConfig` |
| Hyperparameters | `m=200, k=2, ν=3, q=0.9, α=0.95, β=2`, 20000 draws / 250 burn | library defaults; set `n_draws=20000, n_burn=250` to match exactly |

Verified numerically: the non-linear DGP has BART lowest (≈2.55 vs ≈2.75), the
linear DGP has the linear selectors lowest (BART worst ≈1.47), and the macro
application has BART ranking first on 6 of 8 targets — the paper's pattern.

---

## 8 · Extending the package

**Add an estimator.** Implement `fit(X, y)`, `predict(X)` and an `importance_`
array, then drop it in: `ARDLBART(estimator=MyModel, estimator_kwargs={…})`.

**Change the split-variable prior.** Pass `BART(split_prob=probs)` with a length-`p`
discrete distribution — the paper's relaxation of the uniform rule (§2.1).

**Predictive intervals.** Use `BART.yhat_train_draws_` (shape `n_draws × n`) for
posterior bands on the training fit, and `sigma_draws_` for the noise.

**Match the paper exactly.** `BART(n_trees=200, k=2, nu=3, q=0.9, alpha=0.95,
beta=2, n_draws=20000, n_burn=250)`.

**Real data.** `load_macro(source="fred")` (install `bartardl[fred]`) pulls the
eight FRED series and applies the Table-3 transforms.
