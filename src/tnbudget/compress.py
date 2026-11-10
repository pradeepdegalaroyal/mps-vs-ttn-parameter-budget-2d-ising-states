"""Sequential-SVD compression of a pure state into an MPS or a binary tree tensor network.

Both formats are obtained by projecting the state, cut by cut, onto the top-``chi`` Schmidt
subspace (TT-SVD for the chain; the bottom-up hierarchical analogue for the tree). Parameter
counts are the number of tensor entries of the format with nominal bond dimensions.
"""
from __future__ import annotations

import numpy as np

from .lattice import Layout


def _project_cut(t: np.ndarray, lo: int, hi: int, chi: int) -> np.ndarray:
    """Rank-``chi`` truncation of the bipartition positions [lo, hi) versus the rest."""
    axes = list(range(lo, hi))
    m = np.moveaxis(t, axes, range(len(axes)))
    shape = m.shape
    mat = m.reshape(2 ** (hi - lo), -1)
    if min(mat.shape) <= chi:
        return t
    # Top-chi Schmidt projector from the smaller Gram matrix (eigh is much faster than a full SVD
    # and accurate to ~1e-16 in the discarded weight, below the infidelity floor used here).
    if mat.shape[0] <= mat.shape[1]:
        _, vec = np.linalg.eigh(mat @ mat.conj().T)
        u = vec[:, -chi:]
        approx = u @ (u.conj().T @ mat)
    else:
        _, vec = np.linalg.eigh(mat.conj().T @ mat)
        v = vec[:, -chi:]
        approx = (mat @ v) @ v.conj().T
    return np.moveaxis(approx.reshape(shape), range(len(axes)), axes)


def to_ordered_tensor(psi: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    """Reshape a flat state to an N-axis tensor with axes permuted into layout order."""
    n = len(order)
    return np.transpose(psi.reshape((2,) * n), order)


def truncate_mps(t: np.ndarray, chi: int) -> np.ndarray:
    """Left-to-right TT-SVD truncation of an ordered tensor to bond dimension ``chi``."""
    out = t
    for k in range(1, t.ndim):
        out = _project_cut(out, 0, k, chi)
    return out


def truncate_ttn(t: np.ndarray, layout: Layout, chi: int) -> np.ndarray:
    """Bottom-up hierarchical truncation to a binary tree with bond dimension ``chi``."""
    out = t
    for lo, hi in layout.cuts():
        out = _project_cut(out, lo, hi, chi)
    return out


def fidelity(psi_t: np.ndarray, approx_t: np.ndarray) -> float:
    """|<psi|approx>|^2 / (<psi|psi><approx|approx>) for tensors of equal shape."""
    a = psi_t.reshape(-1)
    b = approx_t.reshape(-1)
    ov = np.vdot(a, b)
    return float(abs(ov) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def _cap(size: int, n: int) -> int:
    return 2 ** min(size, n - size)


def mps_params(n: int, chi: int) -> int:
    """Entries of an open MPS with physical dimension 2 and bond cap ``chi``."""
    d = [1] + [min(chi, _cap(k, n)) for k in range(1, n)] + [1]
    return sum(d[k] * 2 * d[k + 1] for k in range(n))


def ttn_params(layout: Layout, chi: int) -> int:
    """Entries of a binary tree tensor network (physical legs on the leaves) with bond cap ``chi``."""
    n = layout.n

    def dim(a: int, b: int) -> int:
        return 1 if (a, b) == (0, n) else min(chi, _cap(b - a, n))

    total = 0
    for a, m, b in layout.internal:
        if (a, b) == (0, n):
            total += dim(a, m) * dim(m, b)
        else:
            total += dim(a, m) * dim(m, b) * dim(a, b)
    return total


def cut_entropy(t: np.ndarray, k: int) -> float:
    """Von Neumann entropy (bits) across the prefix cut of an ordered tensor."""
    s = np.linalg.svd(t.reshape(2**k, -1), compute_uv=False)
    p = s**2
    p = p[p > 1e-14]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())
