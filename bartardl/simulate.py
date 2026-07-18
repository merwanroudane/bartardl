"""Friedman (1991) data-generating processes used in the paper.

The paper (Section 4) evaluates BART against LASSO, Elastic Net and a
Bayesian-Network estimator on the two Friedman
<doi:10.1214/aos/1176347963> designs, both with ``p`` in ``{10, 100}``
predictors of which only the first five are relevant.

Linear DGP (paper Eq. 11)::

    y = 2 + 10*pi*x1*x2 ... (linearised)  -> here the additive linear form
    y = 2 + 3*x1 + 3*x2 + 3*x3 + 3*x4 + 3*x5 + e

Non-linear DGP (paper Eq. 12, the classic Friedman function)::

    y = 10*sin(pi*x1*x2) + 20*(x3 - 0.5)^2 + 10*x4 + 5*x5 + e

with ``x ~ Uniform(0, 1)^p`` and ``e ~ N(0, 1)``.

Because the paper's Eq. (11) is garbled in the published PDF text layer,
the linear DGP is implemented as the standard additive linearisation of
the Friedman function (the five relevant predictors enter linearly with
equal weight).  Use :func:`friedman` and select ``kind`` accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimData:
    """A simulated design: predictors ``X``, response ``y`` and the truth."""

    X: np.ndarray
    y: np.ndarray
    f: np.ndarray                 # noise-free signal f(x)
    relevant: np.ndarray          # indices of the truly relevant predictors
    kind: str


def friedman(n: int = 100, p: int = 10, kind: str = "nonlinear",
             sigma: float = 1.0, random_state=None) -> SimData:
    """Draw a Friedman data set.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of predictors (``>= 5``); predictors 6..p are pure noise.
    kind : {"nonlinear", "linear"}
        Which data-generating process (paper Eq. 12 or Eq. 11).
    sigma : float
        Standard deviation of the Gaussian error term.
    random_state : int or numpy Generator, optional
        Seed for reproducibility.

    Returns
    -------
    SimData
    """
    if p < 5:
        raise ValueError("Friedman DGP needs at least 5 predictors")
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0.0, 1.0, size=(n, p))
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    if kind == "nonlinear":
        f = (10.0 * np.sin(np.pi * x1 * x2)
             + 20.0 * (x3 - 0.5) ** 2
             + 10.0 * x4
             + 5.0 * x5)
    elif kind == "linear":
        f = 2.0 + 3.0 * (x1 + x2 + x3 + x4 + x5)
    else:
        raise ValueError("kind must be 'linear' or 'nonlinear'")

    y = f + rng.normal(0.0, sigma, size=n)
    return SimData(X=X, y=y, f=f, relevant=np.arange(5), kind=kind)


__all__ = ["friedman", "SimData"]
