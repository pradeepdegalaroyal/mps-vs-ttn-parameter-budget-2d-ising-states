"""Budget metrics: smallest parameter count that reaches a target fidelity."""
from __future__ import annotations

import numpy as np


def params_at_fidelity(params: np.ndarray, fid: np.ndarray, target: float) -> float:
    """Parameters needed to reach ``target`` fidelity, by log-log interpolation of infidelity.

    ``params`` must ascend with bond dimension. Infidelity is made monotone with a running
    minimum. Returns NaN if the target is never reached (right-censored).
    """
    params = np.asarray(params, dtype=float)
    eps = np.minimum.accumulate(np.clip(1.0 - np.asarray(fid, dtype=float), 1e-16, 1.0))
    goal = max(1.0 - target, 1e-16)
    if eps[0] <= goal:
        return float(params[0])
    hit = np.flatnonzero(eps <= goal)
    if hit.size == 0:
        return float("nan")
    i = int(hit[0])
    x0, x1 = np.log(params[i - 1]), np.log(params[i])
    y0, y1 = np.log(eps[i - 1]), np.log(eps[i])
    if y0 == y1:
        return float(params[i])
    x = x0 + (np.log(goal) - y0) * (x1 - x0) / (y1 - y0)
    return float(np.exp(x))
