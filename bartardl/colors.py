"""Colour maps for publication-quality figures.

MATLAB's default *Parula* colormap (R2014b) is reproduced from its 64 RGB
control points and offered through :func:`parula_colors`, which returns a
list of ``n`` hex colours interpolated across those stops -- the same
``colorRampPalette`` idea used in the author's R packages.  Companion maps
``matlab_jet_colors``, ``turbo_colors``, ``bluered_colors`` and
``sinha_colors`` follow the same interface, and :func:`get_cmap` returns a
Matplotlib ``ListedColormap`` for any of them.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex

# 64-stop MATLAB R2014b Parula control points (RGB in [0, 1]).
_PARULA = np.array([
    [0.2081, 0.1663, 0.5292], [0.2116, 0.1898, 0.5777], [0.2123, 0.2138, 0.6270],
    [0.2081, 0.2386, 0.6771], [0.1959, 0.2645, 0.7279], [0.1707, 0.2919, 0.7792],
    [0.1253, 0.3242, 0.8303], [0.0591, 0.3598, 0.8683], [0.0117, 0.3875, 0.8820],
    [0.0060, 0.4086, 0.8828], [0.0165, 0.4266, 0.8786], [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641], [0.0629, 0.4737, 0.8554], [0.0723, 0.4887, 0.8467],
    [0.0779, 0.5040, 0.8384], [0.0793, 0.5200, 0.8312], [0.0749, 0.5375, 0.8263],
    [0.0641, 0.5570, 0.8240], [0.0488, 0.5772, 0.8228], [0.0343, 0.5966, 0.8199],
    [0.0265, 0.6137, 0.8135], [0.0239, 0.6287, 0.8038], [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768], [0.0267, 0.6642, 0.7607], [0.0384, 0.6743, 0.7436],
    [0.0590, 0.6838, 0.7254], [0.0843, 0.6928, 0.7062], [0.1133, 0.7015, 0.6859],
    [0.1453, 0.7098, 0.6646], [0.1801, 0.7177, 0.6424], [0.2178, 0.7250, 0.6193],
    [0.2586, 0.7317, 0.5954], [0.3022, 0.7376, 0.5712], [0.3482, 0.7424, 0.5473],
    [0.3953, 0.7459, 0.5244], [0.4420, 0.7481, 0.5033], [0.4871, 0.7491, 0.4840],
    [0.5300, 0.7491, 0.4661], [0.5709, 0.7485, 0.4494], [0.6100, 0.7473, 0.4337],
    [0.6473, 0.7456, 0.4188], [0.6834, 0.7435, 0.4044], [0.7184, 0.7411, 0.3905],
    [0.7525, 0.7384, 0.3768], [0.7858, 0.7356, 0.3633], [0.8185, 0.7327, 0.3498],
    [0.8507, 0.7299, 0.3360], [0.8824, 0.7274, 0.3217], [0.9139, 0.7258, 0.3063],
    [0.9450, 0.7261, 0.2886], [0.9739, 0.7314, 0.2666], [0.9938, 0.7455, 0.2403],
    [0.9990, 0.7653, 0.2164], [0.9955, 0.7861, 0.1967], [0.9880, 0.8066, 0.1794],
    [0.9789, 0.8271, 0.1633], [0.9697, 0.8481, 0.1475], [0.9626, 0.8705, 0.1309],
    [0.9589, 0.8949, 0.1132], [0.9598, 0.9218, 0.0948], [0.9661, 0.9514, 0.0755],
    [0.9763, 0.9831, 0.0538],
])

# Anchor points for the remaining maps (interpolated with colorRampPalette).
_JET = np.array([
    [0, 0, 0.5], [0, 0, 1], [0, 0.5, 1], [0, 1, 1], [0.5, 1, 0.5],
    [1, 1, 0], [1, 0.5, 0], [1, 0, 0], [0.5, 0, 0],
])
_TURBO = np.array([
    [0.190, 0.072, 0.232], [0.275, 0.408, 0.859], [0.153, 0.681, 0.925],
    [0.107, 0.898, 0.716], [0.437, 0.991, 0.388], [0.808, 0.940, 0.223],
    [0.977, 0.741, 0.216], [0.941, 0.403, 0.100], [0.729, 0.114, 0.023],
    [0.480, 0.016, 0.011],
])
_BLUERED = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
# A perceptually smooth deep-blue -> teal -> gold -> red map in the spirit
# of the diverging maps used by Sinha et al. (2023) energy-economics plots.
_SINHA = np.array([
    [0.031, 0.188, 0.420], [0.129, 0.443, 0.710], [0.404, 0.663, 0.812],
    [0.741, 0.843, 0.906], [0.969, 0.969, 0.969], [0.992, 0.859, 0.780],
    [0.957, 0.647, 0.510], [0.839, 0.376, 0.302], [0.647, 0.058, 0.082],
])

_ANCHORS = {
    "parula": _PARULA, "jet": _JET, "turbo": _TURBO,
    "bluered": _BLUERED, "sinha": _SINHA,
}


def _ramp(anchors: np.ndarray, n: int) -> list[str]:
    """Interpolate ``n`` hex colours across the RGB ``anchors`` (colorRampPalette)."""
    if n < 1:
        return []
    if n == 1:
        return [to_hex(anchors[0])]
    src = np.linspace(0, 1, len(anchors))
    dst = np.linspace(0, 1, n)
    rgb = np.column_stack([np.interp(dst, src, anchors[:, c]) for c in range(3)])
    return [to_hex(c) for c in rgb]


def parula_colors(n: int) -> list[str]:
    """``n`` hex colours spanning the MATLAB R2014b Parula colormap."""
    return _ramp(_PARULA, n)


def matlab_jet_colors(n: int) -> list[str]:
    """``n`` hex colours spanning MATLAB's classic ``jet`` colormap."""
    return _ramp(_JET, n)


def turbo_colors(n: int) -> list[str]:
    """``n`` hex colours spanning Google's ``turbo`` colormap."""
    return _ramp(_TURBO, n)


def bluered_colors(n: int) -> list[str]:
    """``n`` diverging blue-white-red hex colours."""
    return _ramp(_BLUERED, n)


def sinha_colors(n: int) -> list[str]:
    """``n`` diverging hex colours in the Sinha et al. (2023) palette."""
    return _ramp(_SINHA, n)


def get_cmap(name: str = "Parula", n: int = 256):
    """Return a Matplotlib colormap for any named palette above."""
    key = name.lower()
    if key not in _ANCHORS:
        raise ValueError(f"unknown colormap {name!r}; "
                         f"choose from {sorted(_ANCHORS)}")
    return ListedColormap(_ramp(_ANCHORS[key], n), name=name)


def resolve_colorscale(name: str = "Parula", n: int = 256):
    """Return a continuous ``LinearSegmentedColormap`` for the named palette."""
    key = name.lower()
    if key not in _ANCHORS:
        raise ValueError(f"unknown colormap {name!r}")
    return LinearSegmentedColormap.from_list(name, _ANCHORS[key], N=n)


__all__ = [
    "parula_colors", "matlab_jet_colors", "turbo_colors",
    "bluered_colors", "sinha_colors", "get_cmap", "resolve_colorscale",
]
