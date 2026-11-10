"""Lattice geometry and tensor-network layouts (leaf orderings and binary trees)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    """A leaf ordering plus (optionally) a binary tree over contiguous leaf ranges.

    Attributes:
        order: ``order[k]`` is the physical site placed at position ``k``.
        internal: internal nodes ``(a, m, b)`` in post-order (children first); the last
            entry is the root. Node ``(a, m, b)`` covers positions ``[a, b)`` and has
            children ``[a, m)`` and ``[m, b)``. Empty for a pure chain (MPS) layout.
        name: short label.
    """

    order: tuple[int, ...]
    internal: tuple[tuple[int, int, int], ...]
    name: str

    @property
    def n(self) -> int:
        return len(self.order)

    def cuts(self) -> list[tuple[int, int]]:
        """Position ranges of all non-root tree nodes (leaves included), children first."""
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        n = self.n
        for a, m, b in self.internal:
            for rng in ((a, m), (m, b)):
                if rng not in seen and rng != (0, n):
                    seen.add(rng)
                    out.append(rng)
        return out


def bonds_2d(lx: int, ly: int) -> list[tuple[int, int]]:
    """Nearest-neighbour bonds of an open ``lx`` x ``ly`` square lattice; site id = x * ly + y."""
    bonds = []
    for x in range(lx):
        for y in range(ly):
            s = x * ly + y
            if y + 1 < ly:
                bonds.append((s, s + 1))
            if x + 1 < lx:
                bonds.append((s, s + ly))
    return bonds


def bonds_chain(n: int) -> list[tuple[int, int]]:
    """Open chain bonds."""
    return [(i, i + 1) for i in range(n - 1)]


def snake_order(lx: int, ly: int) -> tuple[int, ...]:
    """Boustrophedon order sweeping along the longer side, so the MPS cut width is the shorter side."""
    if ly > lx:
        raise ValueError("expect lx >= ly")
    order = []
    for x in range(lx):
        ys = range(ly) if x % 2 == 0 else range(ly - 1, -1, -1)
        order.extend(x * ly + y for y in ys)
    return tuple(order)


def _internal_from_tree(tree, start: int, out: list) -> int:
    """Walk a nested-tuple tree, appending internal nodes in post-order; return end position."""
    if isinstance(tree, int):
        return start + 1
    left, right = tree
    mid = _internal_from_tree(left, start, out)
    end = _internal_from_tree(right, mid, out)
    out.append((start, mid, end))
    return end


def _leaves(tree) -> list[int]:
    return [tree] if isinstance(tree, int) else _leaves(tree[0]) + _leaves(tree[1])


def _geo_tree(x0: int, x1: int, y0: int, y1: int, ly: int):
    nx, ny = x1 - x0, y1 - y0
    if nx == 1 and ny == 1:
        return x0 * ly + y0
    if nx >= ny:
        xm = (x0 + x1) // 2
        return (_geo_tree(x0, xm, y0, y1, ly), _geo_tree(xm, x1, y0, y1, ly))
    ym = (y0 + y1) // 2
    return (_geo_tree(x0, x1, y0, ym, ly), _geo_tree(x0, x1, ym, y1, ly))


def _balanced_tree(items: list[int]):
    if len(items) == 1:
        return items[0]
    m = len(items) // 2
    return (_balanced_tree(items[:m]), _balanced_tree(items[m:]))


def layout_from_tree(tree, name: str) -> Layout:
    """Build a :class:`Layout` from a nested-tuple binary tree of site indices."""
    internal: list[tuple[int, int, int]] = []
    _internal_from_tree(tree, 0, internal)
    return Layout(tuple(_leaves(tree)), tuple(internal), name)


def geometric_tree_layout(lx: int, ly: int) -> Layout:
    """Binary tree from recursive bisection of the longer side of the lattice."""
    return layout_from_tree(_geo_tree(0, lx, 0, ly, ly), "ttn-geometric")


def chain_tree_layout(n: int) -> Layout:
    """Balanced binary tree over a 1D chain in natural order."""
    return layout_from_tree(_balanced_tree(list(range(n))), "ttn-chain")


def random_tree_layout(n: int, rng: np.random.Generator) -> Layout:
    """Balanced binary tree over a random leaf permutation (layout ablation)."""
    perm = [int(i) for i in rng.permutation(n)]
    return layout_from_tree(_balanced_tree(perm), "ttn-random")


def chain_layout(order: tuple[int, ...], name: str = "mps") -> Layout:
    """MPS layout: just a leaf order (the cuts are the N-1 prefix cuts)."""
    return Layout(tuple(order), (), name)
