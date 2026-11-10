"""Hardware-efficient ansatz (HEA) baseline: variational state preparation on the lattice graph.

Layers of single-qubit RY-RZ rotations, each followed by CZ gates on one edge-colour class of
the device graph. Fitted by Adam to maximise fidelity with a target state (exact statevector).
"""
from __future__ import annotations

import numpy as np
import torch


def edge_colouring(bonds: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Greedy proper edge colouring (no two edges of a colour share a qubit)."""
    classes: list[list[tuple[int, int]]] = []
    used: list[set[int]] = []
    for a, b in bonds:
        for cls, u in zip(classes, used):
            if a not in u and b not in u:
                cls.append((a, b))
                u.update((a, b))
                break
        else:
            classes.append([(a, b)])
            used.append({a, b})
    return classes


def _cz_sign(n: int, edges: list[tuple[int, int]], device: str) -> torch.Tensor:
    idx = torch.arange(2**n, device=device)
    bit = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
    parity = torch.zeros(2**n, dtype=torch.long, device=device)
    for a, b in edges:
        parity += bit[a] * bit[b]
    return (1 - 2 * (parity % 2)).to(torch.float32).reshape((2,) * n)


def hea_params(n: int, depth: int) -> int:
    """Number of real rotation angles: 2 per qubit per rotation layer, ``depth + 1`` layers."""
    return 2 * n * (depth + 1)


def cz_count(bonds: list[tuple[int, int]], depth: int) -> int:
    """Total CZ gates of the ansatz (colour classes cycled over ``depth`` entangling layers)."""
    cls = edge_colouring(bonds)
    return sum(len(cls[layer % len(cls)]) for layer in range(depth))


def _rot(angles: torch.Tensor) -> torch.Tensor:
    """RZ(b) RY(a) for angles of shape (n, 2) -> (n, 2, 2) complex."""
    a, b = angles[:, 0], angles[:, 1]
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    ry = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)
    phase = torch.stack([torch.exp(-0.5j * b), torch.exp(0.5j * b)], -1)  # (n, 2)
    return phase.unsqueeze(-1) * ry


def _apply_rot_layer(psi: torch.Tensor, gates: torch.Tensor, n: int) -> torch.Tensor:
    for q in range(n):
        psi = torch.movedim(torch.tensordot(gates[q], psi, dims=([1], [q])), 0, q)
    return psi


def fit_hea(
    target: np.ndarray,
    n: int,
    bonds: list[tuple[int, int]],
    depth: int,
    seed: int,
    steps: int = 250,
    lr: float = 0.06,
    device: str | None = None,
) -> float:
    """Return the best fidelity reached by an HEA of the given depth fitted to ``target``."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device="cpu").manual_seed(seed)
    tgt = torch.as_tensor(target.reshape((2,) * n), dtype=torch.complex64, device=dev)
    signs = [_cz_sign(n, c, dev) for c in edge_colouring(bonds)]
    theta = (0.3 * torch.randn(depth + 1, n, 2, generator=g)).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    zero = torch.zeros((2,) * n, dtype=torch.complex64, device=dev)
    zero[(0,) * n] = 1.0

    def forward() -> torch.Tensor:
        psi = zero
        for layer in range(depth + 1):
            psi = _apply_rot_layer(psi, _rot(theta[layer]), n)
            if layer < depth:
                psi = psi * signs[layer % len(signs)]
        return psi

    best = 0.0
    for _ in range(steps):
        opt.zero_grad()
        f = torch.abs(torch.sum(tgt.conj() * forward())) ** 2
        (1.0 - f).backward()
        opt.step()
        best = max(best, float(f.detach()))
    with torch.no_grad():
        best = max(best, float(torch.abs(torch.sum(tgt.conj() * forward())) ** 2))
    return best
