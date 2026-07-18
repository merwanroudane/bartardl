"""Tests for the ARDL layer, transforms and data loading."""
import numpy as np
import pandas as pd
import pytest

from bartardl import (ARDLBART, BART, Lasso, ElasticNet, BayesianNetwork,
                      load_macro, make_lag_matrix, transform_series)


def test_transform_codes():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(transform_series(x, 1), x)              # none
    np.testing.assert_allclose(transform_series(x, 2)[1:], [1, 2, 4])  # 1st diff
    logdiff = transform_series(x, 5)
    np.testing.assert_allclose(logdiff[1:], np.log([2, 2, 2]))         # dlog
    with pytest.raises(ValueError):
        transform_series(x, 99)


def test_make_lag_matrix_dimensions():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)),
                      columns=["a", "b", "c"])
    X, y = make_lag_matrix(df, "a", n_lags=4)
    assert X.shape[1] == 3 * 4                 # 3 vars x 4 lags
    assert len(X) == len(y) == 50 - 4          # rows lost to lagging
    assert "a.lag1" in X.columns and "c.lag4" in X.columns


def test_load_macro_shapes():
    raw = load_macro(source="bundled", transform=False)
    assert list(raw.columns) == ["GDP", "INF", "FF", "M2", "PC", "IP", "U", "INV"]
    stat = load_macro(source="bundled", transform=True)
    assert len(stat) < len(raw)                # rows dropped by differencing


@pytest.mark.parametrize("est,kw", [
    (BART, dict(n_trees=20, n_burn=60, n_draws=60, random_state=0)),
    (Lasso, {}),
    (ElasticNet, {}),
    (BayesianNetwork, dict(n_iter=150, burn=80, random_state=0)),
])
def test_ardl_fit_predict_each_estimator(est, kw):
    data = load_macro(source="bundled")
    model = ARDLBART(n_lags=4, estimator=est, estimator_kwargs=kw)
    res = model.fit(data, "GDP")
    assert len(res.feature_names) == 8 * 4
    assert res.fitted.shape == res.y_train.shape
    pred = model.predict(data, "GDP")
    assert len(pred) == len(res.y_train)
    imp = res.importance_frame()
    assert {"variable", "lag", "importance"}.issubset(imp.columns)


def test_top_features_sorted():
    data = load_macro(source="bundled")
    res = ARDLBART(n_lags=4, estimator=BART,
                   estimator_kwargs=dict(n_trees=25, n_burn=80, n_draws=80,
                                         random_state=0)).fit(data, "INV")
    top = res.top_features(5)
    assert len(top) == 5
    assert top["importance"].is_monotonic_decreasing
