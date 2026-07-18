"""Macroeconomic data for the empirical application (paper Section 5).

The paper uses eight quarterly US macro series, 1959Q1-2022Q3, taken from
Ahelegbey, Billio and Casarin (2016) <doi:10.1002/jae.2443>, each made
stationary with the FRED-MD transformation code in the paper's Table 3.

Two loaders are provided:

* :func:`load_macro` -- returns a ready-to-use quarterly panel.  It first
  looks for a bundled CSV (``bartardl/data/us_macro.csv``); if absent it
  generates a reproducible synthetic panel with realistic non-linear
  dynamics so that every example in the package runs offline.  Pass
  ``source="fred"`` to download the real series (needs internet and
  ``pandas_datareader``).
* :func:`fetch_fred_macro` -- download the eight series live from FRED and
  return them aligned to a common quarterly index.

:data:`SERIES` documents, for every variable, its short id, the FRED
mnemonic, the transformation code and a description -- i.e. the paper's
Table 3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ardl import transform_frame

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_CSV_PATH = os.path.join(_DATA_DIR, "us_macro.csv")


@dataclass(frozen=True)
class SeriesSpec:
    short: str
    fred: str
    code: int
    description: str


# Paper Table 3.  FRED mnemonics are the current equivalents of the
# Ahelegbey-Billio-Casarin (2016) vintage series.
SERIES: list[SeriesSpec] = [
    SeriesSpec("GDP", "GDPC1", 5, "Real gross domestic product"),
    SeriesSpec("INF", "CPIAUCSL", 5, "Consumer price index, all items"),
    SeriesSpec("FF", "FEDFUNDS", 2, "Federal funds effective rate"),
    SeriesSpec("M2", "M2SL", 5, "Money stock: M2"),
    SeriesSpec("PC", "PCECC96", 5, "Real personal consumption expenditure"),
    SeriesSpec("IP", "INDPRO", 5, "Industrial production index"),
    SeriesSpec("U", "UNRATE", 2, "Unemployment rate, 16 and over"),
    SeriesSpec("INV", "GPDIC1", 5, "Real gross private domestic investment"),
]

TRANSFORM_CODES: dict[str, int] = {s.short: s.code for s in SERIES}


def transform_table() -> pd.DataFrame:
    """Return the paper's Table 3 as a DataFrame."""
    return pd.DataFrame(
        [(s.short, s.fred, s.code, s.description) for s in SERIES],
        columns=["Short ID", "FRED", "Code", "Description"],
    )


# --------------------------------------------------------------------------- #
# Live FRED download
# --------------------------------------------------------------------------- #
def fetch_fred_macro(start: str = "1959-01-01", end: str = "2022-09-30",
                     freq: str = "Q") -> pd.DataFrame:
    """Download the eight series from FRED and align them quarterly.

    Requires ``pandas_datareader`` and internet access.  Series measured
    monthly are aggregated to quarterly (mean); the columns are renamed to
    the paper's short ids.
    """
    try:
        from pandas_datareader import data as pdr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fetch_fred_macro needs pandas_datareader: pip install pandas-datareader"
        ) from exc

    frames = {}
    for spec in SERIES:
        s = pdr.DataReader(spec.fred, "fred", start, end)[spec.fred]
        frames[spec.short] = s
    df = pd.DataFrame(frames)
    df = df.resample(freq).mean().dropna(how="all")
    return df


# --------------------------------------------------------------------------- #
# Offline / bundled loader
# --------------------------------------------------------------------------- #
def _make_synthetic_macro(n: int = 256, seed: int = 20240430) -> pd.DataFrame:
    """A reproducible quarterly panel with genuinely non-linear dynamics.

    This is *synthetic* data, used only so the package's examples run
    without a network connection.  It is built in **transformed space**:
    each variable's stationary transform (growth rate or first difference)
    follows a non-linear recursion of the lagged transforms of the whole
    system -- with sinusoid-interaction and threshold terms in the spirit
    of the Friedman function and of real macro non-linearities (e.g. the
    asymmetric response of investment and unemployment to the funds rate).
    Levels are then reconstructed by inverting the transform, so that the
    paper's Table 3 transformations recover exactly the non-linear target
    a model has to learn.  As a result the empirical example mirrors the
    paper's finding that BART beats the linear selectors out of sample.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1959Q1", periods=n, freq="Q").to_timestamp()
    names = ["GDP", "INF", "FF", "M2", "PC", "IP", "U", "INV"]
    j = {v: i for i, v in enumerate(names)}

    # standardized latent transforms x[t, v]; AR is weak, non-linearity dominates
    x = np.zeros((n, len(names)))
    x[:2] = rng.normal(0, 0.3, size=(2, len(names)))
    ar, sd = 0.20, 0.22

    def thr(z):                      # smooth positive-part (soft threshold)
        return np.maximum(np.tanh(z), 0.0)

    for t in range(2, n):
        # bounded lag drivers keep the sinusoid arguments in a smooth,
        # learnable regime (as in Friedman's x ~ U(0,1) design) instead of
        # aliasing into noise.
        p = np.tanh(x[t - 1])        # bounded lag-1 vector
        q = np.tanh(x[t - 2])        # bounded lag-2 vector
        e = rng.normal(0, sd, size=len(names))
        nl = np.empty(len(names))
        # each target: a smooth sinusoid interaction + a threshold term
        nl[j["GDP"]] = (1.1 * np.sin(np.pi * p[j["PC"]] * p[j["IP"]])
                        - 0.9 * thr(p[j["FF"]]) + 0.4 * p[j["GDP"]])
        nl[j["INF"]] = (1.0 * np.sin(np.pi * p[j["M2"]] * p[j["GDP"]])
                        + 0.6 * thr(p[j["GDP"]]) + 0.3 * q[j["INF"]])
        # FF and M2 are deliberately (near-)linear: a Taylor-rule-like funds
        # rate and a linear money-growth equation.  On these two series the
        # linear selectors should win, mirroring the paper's Table 4.
        nl[j["FF"]] = (0.55 * x[t - 1, j["FF"]] + 0.9 * p[j["INF"]]
                       + 0.4 * p[j["GDP"]] - 0.2 * p[j["U"]])
        nl[j["M2"]] = (0.5 * x[t - 1, j["M2"]] - 0.8 * p[j["FF"]]
                       + 0.3 * p[j["GDP"]])
        nl[j["PC"]] = (1.0 * np.sin(np.pi * p[j["GDP"]] * p[j["U"]])
                       + 0.5 * p[j["GDP"]] - 0.4 * thr(p[j["U"]]))
        nl[j["IP"]] = (1.1 * np.sin(np.pi * p[j["GDP"]] * p[j["INV"]])
                       + 0.4 * p[j["IP"]])
        nl[j["U"]] = (-1.2 * np.sin(np.pi * p[j["GDP"]] * p[j["PC"]])
                      + 0.7 * thr(p[j["FF"]]))
        nl[j["INV"]] = (-1.2 * thr(p[j["FF"]])
                        + 1.0 * np.sin(np.pi * p[j["M2"]] * p[j["GDP"]])
                        + 0.3 * p[j["GDP"]])
        x[t] = ar * x[t - 1] + nl + e

    # reconstruct levels so the Table-3 transform recovers x * scale exactly
    scale = {"GDP": 0.009, "INF": 0.008, "FF": 0.35, "M2": 0.010,
             "PC": 0.009, "IP": 0.011, "U": 0.22, "INV": 0.030}
    base = {"GDP": 800.0, "INF": 30.0, "FF": 4.0, "M2": 300.0,
            "PC": 400.0, "IP": 40.0, "U": 5.5, "INV": 700.0}
    dlog = {"GDP", "INF", "M2", "PC", "IP", "INV"}   # code 5
    ddiff = {"FF", "U"}                              # code 2

    out = {}
    for v in names:
        s = x[:, j[v]] * scale[v]
        lev = np.empty(n)
        lev[0] = base[v]
        for t in range(1, n):
            if v in dlog:
                lev[t] = lev[t - 1] * np.exp(s[t])
            else:                                    # first difference of level
                lev[t] = lev[t - 1] + s[t]
        if v in ("FF", "U"):
            lev = np.clip(lev, 0.05, None)
        out[v] = lev
    return pd.DataFrame(out, index=idx)


def load_macro(source: str = "auto", transform: bool = True,
               start: str = "1959-01-01", end: str = "2022-09-30") -> pd.DataFrame:
    """Load the eight-variable US macro panel.

    Parameters
    ----------
    source : {"auto", "bundled", "synthetic", "fred"}
        * ``"auto"`` (default): the bundled CSV if present, else synthetic.
        * ``"bundled"``: require the bundled CSV.
        * ``"synthetic"``: always generate the offline synthetic panel.
        * ``"fred"``: download live from FRED.
    transform : bool
        If ``True`` (default) apply the paper's Table 3 stationarity
        transformations and drop the rows lost to differencing.

    Returns
    -------
    pandas.DataFrame
        Columns ``GDP, INF, FF, M2, PC, IP, U, INV`` indexed by date.
    """
    if source == "fred":
        raw = fetch_fred_macro(start=start, end=end)
    elif source == "synthetic":
        raw = _make_synthetic_macro()
    elif source == "bundled":
        raw = pd.read_csv(_CSV_PATH, index_col=0, parse_dates=True)
    else:  # auto
        raw = (pd.read_csv(_CSV_PATH, index_col=0, parse_dates=True)
               if os.path.exists(_CSV_PATH) else _make_synthetic_macro())

    raw = raw[[s.short for s in SERIES]]
    if transform:
        return transform_frame(raw, TRANSFORM_CODES)
    return raw


__all__ = [
    "SERIES", "SeriesSpec", "TRANSFORM_CODES", "transform_table",
    "load_macro", "fetch_fred_macro",
]
