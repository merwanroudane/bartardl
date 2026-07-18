"""Tests for metrics, tables and the colour maps."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from bartardl import (rmse, relative_rmse, simulation_table, ranking_table,
                      best_method_counts)
from bartardl.colors import parula_colors, get_cmap, resolve_colorscale
from bartardl import viz


def test_rmse_and_relative():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert relative_rmse(y, y + 1, sigma=2.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        relative_rmse(y, y, sigma=0.0)


def test_simulation_table_columns():
    draws = {"BART": np.random.rand(20) + 1, "LASSO": np.random.rand(20) + 1}
    tab = simulation_table(draws)
    assert list(tab.columns) == ["mean", "median", "Q0.50", "Q0.75"]
    assert set(tab.index) == {"BART", "LASSO"}


def test_ranking_table_and_wins():
    rmses = {
        "GDP": {"BART": 1.0, "LASSO": 1.5},
        "INF": {"BART": 2.0, "LASSO": 1.0},
    }
    r = ranking_table(rmses)
    assert r[("BART", "Rank")].loc["GDP"] == 1
    assert r[("LASSO", "Rank")].loc["INF"] == 1
    wins = best_method_counts(r)
    assert wins["BART"] == 1 and wins["LASSO"] == 1


def test_parula_colors():
    cols = parula_colors(8)
    assert len(cols) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in cols)
    assert parula_colors(1) == [parula_colors(1)[0]]


def test_colormaps():
    assert get_cmap("Parula").N == 256
    assert resolve_colorscale("Sinha").N == 256
    with pytest.raises(ValueError):
        get_cmap("not-a-map")


def test_journal_table_formats():
    df = pd.DataFrame({"a": [1.234, 2.345], "b": [3.456, 4.567]})
    assert "a" in viz.journal_table(df, fmt="console")
    assert "<" in viz.journal_table(df, fmt="html")
    assert "\\" in viz.journal_table(df, fmt="latex")
