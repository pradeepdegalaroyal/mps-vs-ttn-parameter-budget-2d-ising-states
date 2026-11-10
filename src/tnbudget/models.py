"""Transverse-field Ising model: ground states and noisy Trotterised quench trajectories.

H = - sum_<ij> J_ij Z_i Z_j - h sum_i X_i. Qubit 0 is the most significant axis of the state tensor.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


def make_couplings(bonds: list[tuple[int, int]], disorder: float, rng: np.random.Generator) -> np.ndarray:
    """Bond couplings J_ij = 1 + disorder * N(0, 1), clipped to stay ferromagnetic (>= 0.1)."""
    j = 1.0 + disorder * rng.standard_normal(len(bonds))
    return np.clip(j, 0.1, None)


def zz_diagonal(n: int, bonds: list[tuple[int, int]], j: np.ndarray) -> np.ndarray:
    """Diagonal of sum_ij J_ij Z_i Z_j over the computational basis (length 2**n)."""
    idx = np.arange(2**n, dtype=np.int64)
    z = [1.0 - 2.0 * ((idx >> (n - 1 - q)) & 1) for q in range(n)]
    d = np.zeros(2**n)
    for (a, b), jab in zip(bonds, j):
        d += jab * z[a] * z[b]
    return d


def _flip_sum(v: np.ndarray, n: int) -> np.ndarray:
    t = v.reshape((2,) * n)
    out = np.zeros_like(t)
    for q in range(n):
        out += np.flip(t, axis=q)
    return out.reshape(-1)


def ground_state(n: int, bonds: list[tuple[int, int]], j: np.ndarray, h: float) -> tuple[np.ndarray, float]:
    """Ground state in the Z2-even sector via matrix-free Lanczos. Returns (psi, energy)."""
    d = zz_diagonal(n, bonds, j)

    def matvec(v: np.ndarray) -> np.ndarray:
        return -d * v - h * _flip_sum(v, n)

    op = LinearOperator((2**n, 2**n), matvec=matvec, dtype=np.float64)
    v0 = np.ones(2**n) / np.sqrt(2**n)  # even-parity start keeps Lanczos in the even sector
    w, v = eigsh(op, k=1, which="SA", v0=v0, tol=1e-10)
    psi = v[:, 0]
    return psi / np.linalg.norm(psi), float(w[0])


def _apply_pauli(t: np.ndarray, q: int, kind: int) -> np.ndarray:
    """Apply X (0), Y (1) or Z (2) on axis q of the state tensor."""
    n = t.ndim
    if kind == 0:
        return np.flip(t, axis=q).copy()
    shape = [1] * n
    shape[q] = 2
    sign = np.array([1.0, -1.0]).reshape(shape)
    if kind == 2:
        return t * sign
    return 1j * np.flip(t * sign, axis=q)  # Y = i X Z


def trotter_snapshots(
    n: int,
    bonds: list[tuple[int, int]],
    j: np.ndarray,
    h: float,
    dt: float,
    snapshot_steps: list[int],
    p_error: float,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """First-order Trotter quench from |0...0> with per-qubit depolarising Pauli errors.

    After every Trotter step each qubit independently suffers a uniformly random Pauli with
    probability ``p_error`` (one stochastic trajectory of the depolarising channel). Returns
    normalised flat states at the requested step counts.
    """
    d = zz_diagonal(n, bonds, j)
    phase = np.exp(1j * dt * d).reshape((2,) * n)
    c, s = np.cos(dt * h), np.sin(dt * h)
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    out: dict[int, np.ndarray] = {}
    for step in range(1, max(snapshot_steps) + 1):
        psi = psi * phase
        for q in range(n):
            psi = c * psi + 1j * s * np.flip(psi, axis=q)
        if p_error > 0:
            hit = rng.random(n) < p_error
            for q in np.flatnonzero(hit):
                psi = _apply_pauli(psi, int(q), int(rng.integers(3)))
        if step in snapshot_steps:
            out[step] = psi.reshape(-1) / np.linalg.norm(psi)
    return out
