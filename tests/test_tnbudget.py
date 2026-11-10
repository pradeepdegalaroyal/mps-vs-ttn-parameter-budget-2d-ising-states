"""Fast unit tests for the tnbudget package."""
import numpy as np

from tnbudget.compress import (
    fidelity,
    mps_params,
    to_ordered_tensor,
    truncate_mps,
    truncate_ttn,
    ttn_params,
)
from tnbudget.hea import edge_colouring, fit_hea, hea_params
from tnbudget.lattice import (
    bonds_2d,
    chain_layout,
    chain_tree_layout,
    geometric_tree_layout,
    random_tree_layout,
    snake_order,
)
from tnbudget.metrics import params_at_fidelity
from tnbudget.models import ground_state, make_couplings, trotter_snapshots


def _dense_ground_energy(n, bonds, j, h):
    dim = 2**n
    idx = np.arange(dim)
    e = np.zeros((dim, dim))
    for (a, b), jab in zip(bonds, j):
        za = 1 - 2 * ((idx >> (n - 1 - a)) & 1)
        zb = 1 - 2 * ((idx >> (n - 1 - b)) & 1)
        e[idx, idx] -= jab * za * zb
    for q in range(n):
        e[idx ^ (1 << (n - 1 - q)), idx] -= h
    return np.linalg.eigvalsh(e)[0]


def test_ground_state_energy_matches_dense_diagonalisation():
    bonds = bonds_2d(3, 2)
    j = make_couplings(bonds, 0.2, np.random.default_rng(0))
    psi, e = ground_state(6, bonds, j, 1.3)
    assert abs(e - _dense_ground_energy(6, bonds, j, 1.3)) < 1e-8
    assert abs(np.linalg.norm(psi) - 1) < 1e-10


def test_full_bond_dimension_is_lossless_and_chi1_matches_product_state():
    bonds = bonds_2d(3, 3)
    j = make_couplings(bonds, 0.1, np.random.default_rng(1))
    psi, _ = ground_state(9, bonds, j, 2.0)
    mps = chain_layout(snake_order(3, 3))
    ttn = geometric_tree_layout(3, 3)
    tm, tt = to_ordered_tensor(psi, mps.order), to_ordered_tensor(psi, ttn.order)
    assert fidelity(tm, truncate_mps(tm, 64)) > 1 - 1e-12
    assert fidelity(tt, truncate_ttn(tt, ttn, 64)) > 1 - 1e-12
    assert fidelity(tm, truncate_mps(tm, 1)) < 1.0
    prod = np.kron(np.kron([1.0, 0.0], [0.6, 0.8]), [0.0, 1.0])
    pt = to_ordered_tensor(prod, (0, 1, 2))
    assert fidelity(pt, truncate_mps(pt, 1)) > 1 - 1e-12


def test_fidelity_is_monotone_in_bond_dimension():
    bonds = bonds_2d(4, 3)
    j = make_couplings(bonds, 0.2, np.random.default_rng(2))
    psi, _ = ground_state(12, bonds, j, 3.0)
    ttn = geometric_tree_layout(4, 3)
    t = to_ordered_tensor(psi, ttn.order)
    f = [fidelity(t, truncate_ttn(t, ttn, c)) for c in (1, 2, 4, 8)]
    assert all(b >= a - 1e-12 for a, b in zip(f, f[1:]))


def test_parameter_counts_hand_checked():
    # 4-site chain, chi = 2: bond dimensions (1, 2, 2, 2, 1), physical dimension 2
    assert mps_params(4, 2) == 1 * 2 * 2 + 2 * 2 * 2 + 2 * 2 * 2 + 2 * 2 * 1
    # 4-leaf balanced tree, chi = 2: two leaf-pair nodes (2*2*2 each) and a root matrix (2*2)
    lay = chain_tree_layout(4)
    assert ttn_params(lay, 2) == 2 * (2 * 2 * 2) + 2 * 2
    assert ttn_params(lay, 1) == 2 * (1 * 1 * 1) + 1


def test_tree_layouts_are_valid_permutations_with_contiguous_cuts():
    lay = geometric_tree_layout(4, 4)
    assert sorted(lay.order) == list(range(16))
    assert all(0 <= a < b <= 16 for a, b in lay.cuts())
    rl = random_tree_layout(12, np.random.default_rng(0))
    assert sorted(rl.order) == list(range(12)) and len(rl.internal) == 11


def test_params_at_fidelity_interpolates_and_flags_censoring():
    p = np.array([10.0, 100.0, 1000.0])
    f = np.array([0.5, 0.9, 0.99])
    assert 100.0 < params_at_fidelity(p, f, 0.95) < 1000.0
    assert np.isnan(params_at_fidelity(p, f, 0.999))
    assert params_at_fidelity(p, f, 0.4) == 10.0


def test_noisy_trotter_trajectory_is_normalised():
    bonds = bonds_2d(3, 2)
    j = np.ones(len(bonds))
    snaps = trotter_snapshots(6, bonds, j, 2.0, 0.1, [3, 6], 0.2, np.random.default_rng(0))
    for v in snaps.values():
        assert abs(np.linalg.norm(v) - 1) < 1e-10


def test_hea_fit_improves_over_random_start():
    bonds = bonds_2d(2, 2)
    j = np.ones(len(bonds))
    psi, _ = ground_state(4, bonds, j, 1.0)
    assert len(edge_colouring(bonds)) >= 2 and hea_params(4, 2) == 24
    f = fit_hea(psi, 4, bonds, depth=3, seed=0, steps=80, lr=0.1, device="cpu")
    assert f > 0.9
