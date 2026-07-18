"""Quick start: fit an ARDL-BART equation and read off the drivers.

Run:  python examples/quickstart.py
"""
import bartardl as ba

# 1. A stationary, ready-to-model US macro panel (bundled offline data).
data = ba.load_macro()
print("Data:", data.shape, "->", list(data.columns))

# 2. Fit an ARDL(4) whose conditional mean is a BART ensemble.
model = ba.ARDLBART(
    n_lags=4,
    estimator=ba.BART,
    estimator_kwargs=dict(n_trees=100, n_burn=500, n_draws=500, random_state=0),
)
res = model.fit(data, target="GDP")

# 3. In-sample fit and the five most influential lagged regressors.
print("In-sample RMSE:", round(ba.rmse(res.y_train, res.fitted), 4))
print("\nTop-5 variable inclusion proportions for GDP:")
print(res.top_features(5).to_string(index=False))
