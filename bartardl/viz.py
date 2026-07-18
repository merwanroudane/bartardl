"""Publication-quality figures and journal-style tables.

Reproduces the paper's graphics with a clean, top-journal aesthetic:

* :func:`rmse_boxplot` -- Figures 1 and 4: box plots of (relative) RMSE
  across Monte-Carlo replications for the competing methods.
* :func:`inclusion_plot` -- Figures 2, 3, 5, 6, 7 and A1-A7: variable
  inclusion-proportion / importance bars with 95% interval whiskers.
* :func:`forecast_plot` -- actual-vs-fitted overlay for an ARDL equation.
* :func:`journal_table` -- render a DataFrame as a LaTeX ``booktabs`` /
  HTML / console table styled like a journal exhibit.

All figures use the Parula palette by default and a restrained
serif-free theme.  Every function returns the Matplotlib ``Axes`` so
callers can further customise or save.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .colors import parula_colors


# --------------------------------------------------------------------------- #
# Global aesthetic
# --------------------------------------------------------------------------- #
def set_journal_style() -> None:
    """Apply a restrained, high-contrast theme suitable for journals."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


# --------------------------------------------------------------------------- #
# Figures 1 / 4: RMSE box plots
# --------------------------------------------------------------------------- #
def rmse_boxplot(rmse_draws: dict[str, Sequence[float]],
                 title: Optional[str] = None, ylabel: str = "Relative RMSE",
                 ax: Optional[Axes] = None, palette: str = "parula") -> Axes:
    """Box plots of RMSE draws per method (paper Figures 1 and 4)."""
    set_journal_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.2))
    names = list(rmse_draws)
    data = [np.asarray(rmse_draws[n], dtype=float) for n in names]
    colors = parula_colors(len(names)) if palette == "parula" else None

    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.4),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    if colors:
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor("0.25")
    for whisk in bp["whiskers"]:
        whisk.set_color("0.4")
    for cap in bp["caps"]:
        cap.set_color("0.4")

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


# --------------------------------------------------------------------------- #
# Figures 2 / 7: variable inclusion proportions
# --------------------------------------------------------------------------- #
def inclusion_plot(importance, feature_names: Sequence[str],
                   ci: Optional[np.ndarray] = None, top: Optional[int] = None,
                   title: Optional[str] = None, ax: Optional[Axes] = None,
                   color_index: int = 20) -> Axes:
    """Horizontal bar chart of inclusion proportions / importance.

    Parameters
    ----------
    importance : array-like
        One weight per feature.
    feature_names : sequence of str
        Labels, aligned with ``importance``.
    ci : ndarray of shape (n_features, 2), optional
        Lower/upper bounds drawn as the 95% interval whiskers atop each bar
        (the segments in the paper's figures).
    top : int, optional
        Show only the ``top`` most important features.
    color_index : int
        Which Parula stop to colour the bars with.
    """
    set_journal_style()
    importance = np.asarray(importance, dtype=float)
    names = np.asarray(feature_names)
    order = np.argsort(importance)  # ascending -> largest at top of barh
    if top is not None:
        order = order[-top:]
    imp = importance[order]
    labels = names[order]

    if ax is None:
        height = max(3.0, 0.32 * len(order) + 1)
        _, ax = plt.subplots(figsize=(6.4, height))
    color = parula_colors(40)[min(color_index, 39)]
    ax.barh(range(len(order)), imp, color=color, alpha=0.9,
            edgecolor="0.25", height=0.72)
    if ci is not None:
        ci = np.asarray(ci)[order]
        lo = imp - ci[:, 0]
        hi = ci[:, 1] - imp
        ax.errorbar(imp, range(len(order)), xerr=[lo, hi], fmt="none",
                    ecolor="0.15", elinewidth=1.1, capsize=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Inclusion proportion")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return ax


# --------------------------------------------------------------------------- #
# Actual vs fitted
# --------------------------------------------------------------------------- #
def forecast_plot(y_true, y_pred, index=None, title: Optional[str] = None,
                  ax: Optional[Axes] = None) -> Axes:
    """Overlay actual and fitted series for an ARDL equation."""
    set_journal_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 3.6))
    pal = parula_colors(40)
    x = index if index is not None else np.arange(len(y_true))
    ax.plot(x, y_true, color="0.2", linewidth=1.4, label="Actual")
    ax.plot(x, y_pred, color=pal[6], linewidth=1.8, label="Fitted",
            alpha=0.9)
    ax.legend(loc="best")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    return ax


# --------------------------------------------------------------------------- #
# Journal-style tables
# --------------------------------------------------------------------------- #
def journal_table(df: pd.DataFrame, fmt: str = "console", float_format="%.4f",
                  caption: Optional[str] = None, label: Optional[str] = None,
                  bold_min_rows: bool = False) -> str:
    """Render ``df`` as a journal-styled table.

    Parameters
    ----------
    fmt : {"console", "latex", "html"}
        Output flavour.  ``latex`` emits a ``booktabs`` table.
    bold_min_rows : bool
        If ``True``, bold the minimum value in each row (useful for RMSE
        tables where the best model per row should stand out).
    """
    disp = df.copy()

    if fmt == "latex":
        try:
            styler = disp.style.format(float_format.replace("%", "{:").replace("f", "f}"))
            if caption:
                styler = styler.set_caption(caption)
            body = styler.to_latex(hrules=True, position_float="centering",
                                   label=label)
            return body
        except Exception:
            return disp.to_latex(float_format=lambda v: float_format % v,
                                 caption=caption, label=label)

    if fmt == "html":
        sty = disp.style.format(lambda v: float_format % v
                                if isinstance(v, (int, float, np.floating)) else v)
        if bold_min_rows:
            sty = sty.highlight_min(axis=1,
                                    props="font-weight:bold;color:#08306b;")
        if caption:
            sty = sty.set_caption(caption)
        sty = sty.set_table_styles([
            {"selector": "caption",
             "props": "caption-side:top;font-weight:bold;padding:6px;"},
            {"selector": "th",
             "props": "border-bottom:1px solid #333;text-align:right;padding:4px 8px;"},
            {"selector": "td", "props": "text-align:right;padding:4px 8px;"},
        ])
        return sty.to_html()

    # console
    lines = []
    if caption:
        lines.append(caption)
        lines.append("=" * max(len(caption), 40))
    with pd.option_context("display.float_format", lambda v: float_format % v):
        lines.append(disp.to_string())
    return "\n".join(lines)


__all__ = [
    "set_journal_style", "rmse_boxplot", "inclusion_plot",
    "forecast_plot", "journal_table",
]
