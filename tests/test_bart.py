"""Tests for the BART sampler and its statistical behaviour."""
import numpy as np
import pytest

from bartardl import BART, friedman
from bartardl.metrics import rmse


def test_bart_recovers_nonlinear_signal():
    """BART should beat a mean predictor and estimate sigma near the truth."""
    tr = friedman(n=200, p=10, kind="nonlinear", sigma=1.0, random_state=1)
    te = friedman(n=200, p=10, kind="nonlinear", sigma=1.0, random_state=2)
    m = BART(n_trees=50, n_burn=300, n_draws=300, random_state=0).fit(tr.X, tr.y)
    pred = m.predict(te.X)
    assert rmse(te.y, pred) < rmse(te.y, np.full_like(te.y, tr.y.mean()))
    # posterior sigma should be in a sane neighbourhood of the truth (=1)
    assert 0.5 < m.sigma_draws_.mean() < 2.0


def test_inclusion_proportions_concentrate_on_relevant():
    tr = friedman(n=250, p=10, kind="nonlinear", sigma=1.0, random_state=3)
    m = BART(n_trees=50, n_burn=300, n_draws=300, random_state=0).fit(tr.X, tr.y)
    ip = m.inclusion_proportion_
    assert ip.shape == (10,)
    assert np.isclose(ip.sum(), 1.0)
    # the five relevant predictors should carry most of the mass
    assert ip[:5].sum() > ip[5:].sum()


def test_reproducible_with_seed():
    tr = friedman(n=120, p=8, kind="linear", random_state=5)
    a = BART(n_trees=25, n_burn=100, n_draws=100, random_state=42).fit(tr.X, tr.y)
    b = BART(n_trees=25, n_burn=100, n_draws=100, random_state=42).fit(tr.X, tr.y)
    np.testing.assert_allclose(a.yhat_train_, b.yhat_train_)


def test_predict_shape_and_std():
    tr = friedman(n=80, p=6, random_state=7)
    m = BART(n_trees=20, n_burn=80, n_draws=80, random_state=0).fit(tr.X, tr.y)
    pred, sd = m.predict(tr.X, return_std=True)
    assert pred.shape == (80,)
    assert sd > 0


def test_unknown_option_raises():
    with pytest.raises(TypeError):
        BART(not_a_real_option=1)
