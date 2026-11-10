"""Reproduce every result and figure: ``python experiments/run_all.py`` (add ``--quick`` for a smoke run).

Writes raw results to ``results/`` and figures to ``figures/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tnbudget.compress import (  # noqa: E402
    cut_entropy,
    fidelity,
    mps_params,
    to_ordered_tensor,
    truncate_mps,
    truncate_ttn,
    ttn_params,
)
from tnbudget.hea import cz_count, fit_hea, hea_params  # noqa: E402
from tnbudget.lattice import (  # noqa: E402
    bonds_2d,
    bonds_chain,
    chain_layout,
    chain_tree_layout,
    geometric_tree_layout,
    random_tree_layout,
    snake_order,
)
from tnbudget.models import ground_state, make_couplings, trotter_snapshots  # noqa: E402

torch.set_num_threads(6)

CHIS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]
CHIS_QUENCH = CHIS + [48, 64, 96, 128]
TARGETS = [0.9, 0.95, 0.99, 0.999, 0.9999]


def lattice_setup(name: str):
    """Return (n, bonds, mps_layout, geometric ttn layout) for a named lattice."""
    if name.startswith("chain"):
        n = int(name[5:])
        return n, bonds_chain(n), chain_layout(tuple(range(n))), chain_tree_layout(n)
    lx, ly = (int(c) for c in name.split("x"))
    n = lx * ly
    return n, bonds_2d(lx, ly), chain_layout(snake_order(lx, ly)), geometric_tree_layout(lx, ly)


def sweep_state(psi: np.ndarray, n: int, mps, ttn, scale: int, chis: list[int] = CHIS) -> list[dict]:
    """Truncate one state with both formats over all bond dimensions; return long-format rows."""
    rows = []
    tm = to_ordered_tensor(psi, mps.order)
    tt = to_ordered_tensor(psi, ttn.order)
    for chi in chis:
        rows.append(dict(fmt=mps.name, chi=chi, params=scale * mps_params(n, chi),
                         fidelity=fidelity(tm, truncate_mps(tm, chi))))
        rows.append(dict(fmt=ttn.name, chi=chi, params=scale * ttn_params(ttn, chi),
                         fidelity=fidelity(tt, truncate_ttn(tt, ttn, chi))))
    return rows


def run_ground(cfg: dict) -> pd.DataFrame:
    """E1: ground states of the (disordered) transverse-field Ising model, MPS vs tree."""
    rows = []
    for lat in cfg["lattices"]:
        n, bonds, mps, ttn = lattice_setup(lat)
        for h in cfg["h_grid"]:
            for seed in range(cfg["seeds"]):
                rng = np.random.default_rng(1000 * seed + 17)
                j = make_couplings(bonds, cfg["disorder"], rng)
                psi, _ = ground_state(n, bonds, j, h)
                layouts = [(mps, ttn, "geometric")]
                if lat == cfg["ablation_lattice"] and h in cfg["ablation_h"]:
                    perm_rng = np.random.default_rng(5000 + seed)
                    rt = random_tree_layout(n, perm_rng)
                    rm = chain_layout(rt.order, "mps-random")
                    layouts.append((rm, rt, "random"))
                tm = to_ordered_tensor(psi, mps.order)
                ent = cut_entropy(tm, n // 2)
                for m_l, t_l, kind in layouts:
                    for r in sweep_state(psi, n, m_l, t_l, 1):
                        rows.append(dict(exp="ground", lattice=lat, n=n, layout=kind, h=h,
                                         disorder=cfg["disorder"], seed=seed, mid_entropy=ent, **r))
        print(f"  ground {lat} done", flush=True)
    # disorder-strength robustness at the critical field
    lat = cfg["robust_lattice"]
    n, bonds, mps, ttn = lattice_setup(lat)
    for w in cfg["robust_disorder"]:
        for seed in range(cfg["seeds"]):
            rng = np.random.default_rng(1000 * seed + 17)
            j = make_couplings(bonds, w, rng)
            psi, _ = ground_state(n, bonds, j, cfg["h_crit"])
            ent = cut_entropy(to_ordered_tensor(psi, mps.order), n // 2)
            for r in sweep_state(psi, n, mps, ttn, 1):
                rows.append(dict(exp="robust", lattice=lat, n=n, layout="geometric", h=cfg["h_crit"],
                                 disorder=w, seed=seed, mid_entropy=ent, **r))
    return pd.DataFrame(rows)


def run_quench(cfg: dict) -> pd.DataFrame:
    """E2: Trotterised quench trajectories with depolarising Pauli errors, MPS vs tree."""
    rows = []
    lat = cfg["quench_lattice"]
    n, bonds, mps, ttn = lattice_setup(lat)
    steps = cfg["quench_steps"]
    for p in cfg["quench_p"]:
        for seed in range(cfg["seeds"]):
            rng = np.random.default_rng(1000 * seed + 17)
            j = make_couplings(bonds, cfg["disorder"], rng)
            noise_rng = np.random.default_rng(777 * seed + int(p * 1e4) + 3)
            snaps = trotter_snapshots(n, bonds, j, cfg["h_crit"], cfg["dt"], steps, p, noise_rng)
            for st, psi in snaps.items():
                ent = cut_entropy(to_ordered_tensor(psi, mps.order), n // 2)
                for r in sweep_state(psi, n, mps, ttn, 2, CHIS_QUENCH):  # complex entries -> 2 real numbers
                    rows.append(dict(exp="quench", lattice=lat, n=n, p=p, t=st * cfg["dt"],
                                     seed=seed, mid_entropy=ent, **r))
        print(f"  quench p={p} done", flush=True)
    return pd.DataFrame(rows)


def run_hea(cfg: dict) -> pd.DataFrame:
    """E3: hardware-efficient-ansatz fidelity versus real parameters on the same targets."""
    rows = []
    lat = cfg["hea_lattice"]
    n, bonds, mps, ttn = lattice_setup(lat)
    for h in cfg["hea_h"]:
        for seed in range(cfg["hea_seeds"]):
            rng = np.random.default_rng(1000 * seed + 17)
            j = make_couplings(bonds, cfg["disorder"], rng)
            psi, _ = ground_state(n, bonds, j, h)
            for depth in cfg["hea_depths"]:
                best = max(fit_hea(psi, n, bonds, depth, 31 * seed + k, steps=cfg["hea_steps"], lr=0.1, device="cpu")
                           for k in range(cfg["hea_restarts"]))
                rows.append(dict(exp="hea", lattice=lat, n=n, h=h, seed=seed, depth=depth,
                                 params=hea_params(n, depth), cz=cz_count(bonds, depth), fidelity=best))
            print(f"  hea h={h} seed={seed} done", flush=True)
    return pd.DataFrame(rows)


CONFIGS = {
    "full": dict(
        lattices=["3x3", "4x3", "4x4", "chain16"], h_grid=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
        seeds=5, disorder=0.2, ablation_lattice="4x4", ablation_h=[1.0, 3.0, 5.0],
        robust_lattice="4x4", robust_disorder=[0.0, 0.1, 0.2, 0.4, 0.6], h_crit=3.0,
        quench_lattice="4x4", quench_steps=[5, 10, 15, 20], dt=0.1, quench_p=[0.0, 0.003, 0.01, 0.03],
        hea_lattice="4x3", hea_h=[1.0, 3.0, 5.0], hea_seeds=2, hea_depths=[1, 2, 4, 6, 8],
        hea_steps=120, hea_restarts=1,
    ),
    "quick": dict(
        lattices=["3x3", "chain9"], h_grid=[1.0, 3.0], seeds=2, disorder=0.2,
        ablation_lattice="3x3", ablation_h=[3.0], robust_lattice="3x3", robust_disorder=[0.0, 0.4],
        h_crit=3.0, quench_lattice="3x3", quench_steps=[5, 10], dt=0.1, quench_p=[0.0, 0.03],
        hea_lattice="3x3", hea_h=[3.0], hea_seeds=1, hea_depths=[1, 2], hea_steps=30, hea_restarts=1,
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run (< 1 minute)")
    ap.add_argument("--skip-hea", action="store_true")
    ap.add_argument("--reanalyse", action="store_true", help="rebuild tables and figures from saved CSVs")
    args = ap.parse_args()
    cfg = dict(CONFIGS["quick" if args.quick else "full"])
    out = ROOT / ("results_quick" if args.quick else "results")
    figs = ROOT / ("figures_quick" if args.quick else "figures")
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    t0 = time.time()
    if args.reanalyse:
        ground = pd.read_csv(out / "ground_truncation.csv")
        quench = pd.read_csv(out / "quench_truncation.csv")
        hea = pd.read_csv(out / "hea_baseline.csv")
        old = json.loads((out / "summary.json").read_text())
        import analysis  # noqa: E402

        summary = analysis.summarise(ground, quench, hea, out, figs)
        summary["runtime_seconds"] = old["runtime_seconds"]
        summary["config"] = old["config"]
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        return
    print("E1: ground states", flush=True)
    ground = run_ground(cfg)
    ground.to_csv(out / "ground_truncation.csv", index=False)
    t1 = time.time()
    print("E2: noisy Trotter quench", flush=True)
    quench = run_quench(cfg)
    quench.to_csv(out / "quench_truncation.csv", index=False)
    t2 = time.time()
    hea = pd.DataFrame()
    if not args.skip_hea:
        print("E3: hardware-efficient ansatz baseline", flush=True)
        hea = run_hea(cfg)
        hea.to_csv(out / "hea_baseline.csv", index=False)
    t3 = time.time()

    import analysis  # noqa: E402

    summary = analysis.summarise(ground, quench, hea, out, figs)
    summary["runtime_seconds"] = dict(ground=round(t1 - t0, 1), quench=round(t2 - t1, 1),
                                      hea=round(t3 - t2, 1), total=round(time.time() - t0, 1))
    summary["config"] = {k: v for k, v in cfg.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"done in {time.time() - t0:.0f} s; results in {out}, figures in {figs}")


if __name__ == "__main__":
    main()
