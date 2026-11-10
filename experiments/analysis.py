"""Post-processing: parameter budgets at target fidelities, summary statistics and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from tnbudget.metrics import params_at_fidelity  # noqa: E402

# muted Tableau-10 colours
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, GREY = (
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f", "#79706e")
TARGETS = [0.9, 0.95, 0.99, 0.999, 0.9999]
FINE_TARGETS = 1.0 - np.logspace(-1, -4, 13)
DPI = 200

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def _method(fmt: str) -> str:
    return "MPS" if fmt.startswith("mps") else "TTN"


def budget_table(df: pd.DataFrame, keys: list[str], targets: list[float]) -> pd.DataFrame:
    """Parameters needed per state, per method, per target fidelity (NaN when never reached)."""
    d = df.copy()
    d["method"] = d["fmt"].map(_method)
    rows = []
    for key, g in d.groupby(keys + ["method"], sort=False):
        g = g.sort_values("chi")
        for tg in targets:
            val = params_at_fidelity(g["params"].to_numpy(), g["fidelity"].to_numpy(), tg)
            rows.append(dict(zip(keys + ["method"], key if isinstance(key, tuple) else (key,))) |
                        dict(target=tg, params_needed=val))
    return pd.DataFrame(rows)


def paired_ratio(bt: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """log10(P_MPS / P_TTN) per state and target; positive means the tree is cheaper."""
    w = bt.pivot_table(index=keys + ["target"], columns="method", values="params_needed", dropna=False).reset_index()
    w["log10_ratio"] = np.log10(w["MPS"] / w["TTN"])
    return w


def _band(ax, x, groups, color, label, ls="-", marker="o"):
    m = np.array([np.nanmean(np.asarray(g, dtype=float)) if len(g) else np.nan for g in groups])
    s = np.array([np.nanstd(np.asarray(g, dtype=float)) if len(g) else np.nan for g in groups])
    ax.plot(x, m, ls, color=color, marker=marker, ms=4, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, lw=0)


def _save(fig, path: Path, top: float = 1.0) -> None:
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_budget_vs_field(bt: pd.DataFrame, figs: Path, lattices: list[str]) -> None:
    fig, axes = plt.subplots(1, len(lattices), figsize=(3.4 * len(lattices), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, lat in zip(axes, lattices):
        sub = bt[(bt.lattice == lat) & (bt.layout == "geometric") & (bt.target == 0.99)]
        hs = sorted(sub.h.unique())
        for method, c, ls, mk in (("MPS", BLUE, "-", "o"), ("TTN", ORANGE, "--", "s")):
            grp = [sub[(sub.h == h) & (sub.method == method)].params_needed.to_numpy() for h in hs]
            _band(ax, hs, grp, c, method, ls, mk)
        ax.set_yscale("log")
        ax.set_xlabel("transverse field h / J")
        ax.set_title(lat if lat.startswith("chain") else f"{lat} lattice")
    axes[0].set_ylabel("parameters for fidelity >= 0.99")
    axes[0].legend(frameon=False)
    fig.suptitle("Parameter budget of ground-state compression (mean ± std over 5 disorder seeds)", y=0.99)
    _save(fig, figs / "fig1_budget_vs_field.png", top=0.93)


def fig_ecdf(pr: pd.DataFrame, figs: Path, lattices: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cols = [BLUE, ORANGE, GREEN, RED]
    for lat, c, ls in zip(lattices, cols, ["-", "--", "-.", ":"]):
        v = pr[(pr.lattice == lat) & (pr.layout == "geometric") & (pr.target == 0.99)].log10_ratio.dropna().to_numpy()
        v = np.sort(v)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, ls=ls, lw=1.8, label=f"{lat} (n = {len(v)})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("log10 (parameters MPS / parameters TTN) at fidelity >= 0.99")
    ax.set_ylabel("empirical CDF over states")
    ax.set_title("Where the tree is cheaper (right of 0) or dearer (left of 0)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, figs / "fig2_ecdf_ratio.png")


def fig_phase(bt_fine: pd.DataFrame, figs: Path, lat: str) -> np.ndarray:
    sub = bt_fine[(bt_fine.lattice == lat) & (bt_fine.layout == "geometric")]
    w = paired_ratio(sub, ["lattice", "layout", "h", "seed"])
    g = w.groupby(["h", "target"]).log10_ratio.mean().reset_index()
    hs = np.sort(g.h.unique())
    tg = np.sort(g.target.unique())
    z = np.array([[g[(g.h == h) & (g.target == t)].log10_ratio.iloc[0] for h in hs] for t in tg])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    lim = max(abs(np.nanmin(z)), abs(np.nanmax(z)), 0.05)
    cs = ax.contourf(hs, 1 - tg, z, levels=np.linspace(-lim, lim, 15), cmap="PuOr_r")
    ax.contour(hs, 1 - tg, z, levels=[0], colors="k", linewidths=1.5)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("transverse field h / J")
    ax.set_ylabel("target infidelity 1 - F (log scale)")
    ax.set_title(f"{lat}: log10 (MPS / TTN parameters); black line = break-even")
    fig.colorbar(cs, ax=ax, label="log10 parameter ratio (positive: TTN cheaper)")
    _save(fig, figs / "fig3_phase_map.png")
    return z


def fig_quench(qb: pd.DataFrame, qpr: pd.DataFrame, q: pd.DataFrame, figs: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    ps = sorted(qb.p.unique())
    cols = [BLUE, GREEN, ORANGE, RED]
    ts = sorted(qb.t.unique())
    for p, c, mk in zip(ps, cols, ["o", "s", "^", "D"]):
        sub = qpr[(qpr.p == p) & (qpr.target == 0.99)]
        _band(axes[0], ts, [sub[sub.t == t].log10_ratio.dropna().to_numpy() for t in ts], c, f"p = {p}", marker=mk)
        e = q.drop_duplicates(["p", "t", "seed"])
        e = e[e.p == p]
        _band(axes[2], ts, [e[e.t == t].mid_entropy.to_numpy() for t in ts], c, f"p = {p}", marker=mk)
        sb = qb[(qb.p == p) & (qb.target == 0.99)]
        for method, ls in (("MPS", "-"), ("TTN", "--")):
            m = [sb[(sb.t == t) & (sb.method == method)].params_needed.mean() for t in ts]
            axes[1].plot(ts, m, ls, color=c, marker=mk, ms=3.5, label=f"{method}, p = {p}" if p in (ps[0], ps[-1]) else None)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("log10 (parameters MPS / TTN), F >= 0.99")
    axes[0].set_title("Tree advantage after the quench")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("real parameters for F >= 0.99")
    axes[1].set_title("Budget (solid MPS, dashed TTN)")
    axes[1].legend(frameon=False, fontsize=6.5)
    axes[2].set_ylabel("entanglement entropy, middle cut (bits)")
    axes[2].set_title("Entanglement of the noisy trajectory")
    for ax in axes:
        ax.set_xlabel("evolution time t (units of 1/J)")
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, figs / "fig4_quench_noise.png")


def fig_hea(hea: pd.DataFrame, ground: pd.DataFrame, figs: Path) -> None:
    hs = sorted(hea.h.unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(3.6 * len(hs), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    lat = hea.lattice.iloc[0]
    for ax, h in zip(axes, hs):
        g = ground[(ground.lattice == lat) & (ground.h == h) & (ground.layout == "geometric")]
        for fmt, c, ls, mk in (("mps", BLUE, "-", "o"), ("ttn-geometric", ORANGE, "--", "s")):
            sub = g[g.fmt == fmt]
            chis = sorted(sub.chi.unique())
            x = [sub[sub.chi == c_].params.mean() for c_ in chis]
            _band(ax, x, [1 - sub[sub.chi == c_].fidelity.to_numpy() for c_ in chis], c,
                  "MPS" if fmt == "mps" else "TTN", ls, mk)
        hh = hea[hea.h == h]
        ds = sorted(hh.depth.unique())
        _band(ax, [hh[hh.depth == d].params.iloc[0] for d in ds],
              [1 - hh[hh.depth == d].fidelity.to_numpy() for d in ds], GREEN, "HEA (fitted)", "-.", "^")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1.5)
        ax.set_xlabel("real parameters")
        ax.set_title(f"{lat}, h / J = {h}")
    axes[0].set_ylabel("infidelity 1 - F")
    axes[0].legend(frameon=False)
    _save(fig, figs / "fig5_hea_vs_tensor_networks.png")


def fig_robust_ablation(rb: pd.DataFrame, ab: pd.DataFrame, figs: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ws = sorted(rb.disorder.unique())
    for method, c, ls, mk in (("MPS", BLUE, "-", "o"), ("TTN", ORANGE, "--", "s")):
        _band(axes[0], ws, [rb[(rb.disorder == w) & (rb.method == method)].params_needed.to_numpy() for w in ws],
              c, method, ls, mk)
    axes[0].set_xlabel("bond-disorder strength w (std of J_ij / J)")
    axes[0].set_ylabel("parameters for fidelity >= 0.99")
    axes[0].set_title("Robustness to coupling disorder, h / J = 3")
    axes[0].legend(frameon=False)
    hs = sorted(ab.h.unique())
    width = 0.2
    variants = [("MPS", "geometric", BLUE, "MPS snake"), ("MPS", "random", "#a0cbe8", "MPS random order"),
                ("TTN", "geometric", ORANGE, "TTN geometric"), ("TTN", "random", "#ffbe7d", "TTN random leaves")]
    for k, (method, layout, c, lab) in enumerate(variants):
        m = [ab[(ab.h == h) & (ab.method == method) & (ab.layout == layout)].params_needed.mean() for h in hs]
        s = [ab[(ab.h == h) & (ab.method == method) & (ab.layout == layout)].params_needed.std() for h in hs]
        axes[1].bar(np.arange(len(hs)) + (k - 1.5) * width, m, width, yerr=s, color=c, label=lab, capsize=2)
    axes[1].set_xticks(np.arange(len(hs)))
    axes[1].set_xticklabels([f"h / J = {h}" for h in hs])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("parameters for fidelity >= 0.99")
    axes[1].set_title("Layout ablation (4x4)")
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, figs / "fig6_robustness_and_layout_ablation.png")


def summarise(ground: pd.DataFrame, quench: pd.DataFrame, hea: pd.DataFrame, out: Path, figs: Path) -> dict:
    """Write budget tables and figures; return the summary dictionary used in the README."""
    g = ground[ground.exp == "ground"]
    keys = ["lattice", "layout", "h", "seed"]
    bt = budget_table(g, keys, TARGETS)
    bt.to_csv(out / "ground_budget.csv", index=False)
    pr = paired_ratio(bt, keys)
    fine = budget_table(g[(g.layout == "geometric")], keys, list(FINE_TARGETS))
    rb = budget_table(ground[ground.exp == "robust"], ["lattice", "disorder", "seed"], [0.99])
    rb.to_csv(out / "robust_budget.csv", index=False)
    qkeys = ["p", "t", "seed"]
    qb = budget_table(quench, qkeys, TARGETS)
    qb.to_csv(out / "quench_budget.csv", index=False)
    qpr = paired_ratio(qb, qkeys)

    lats = list(dict.fromkeys(g.lattice))
    fig_budget_vs_field(bt, figs, lats[1:] if len(lats) > 3 else lats)
    fig_ecdf(pr, figs, lats)
    big2d = [x for x in lats if not x.startswith("chain")][-1]
    z = fig_phase(fine, figs, big2d)
    fig_quench(qb, qpr, quench, figs)
    if len(hea):
        fig_hea(hea, g, figs)
    ab = bt[(bt.lattice == big2d) & (bt.target == 0.99) & (bt.h.isin(sorted(g[g.layout == "random"].h.unique())))]
    fig_robust_ablation(rb, ab, figs)
    ab_out = ab.groupby(["h", "method", "layout"]).params_needed.agg(["mean", "std"]).reset_index()
    ab_out.to_csv(out / "layout_ablation.csv", index=False)

    s: dict = {"phase_map_lattice": big2d, "phase_map_min": float(np.nanmin(z)), "phase_map_max": float(np.nanmax(z))}
    s["ground"] = {}
    for lat in lats:
        sub = pr[(pr.lattice == lat) & (pr.layout == "geometric")]
        d = {}
        for tg in TARGETS:
            v = sub[sub.target == tg].log10_ratio.dropna()
            tot = len(sub[sub.target == tg])
            try:
                pval = float(wilcoxon(v).pvalue) if len(v) > 5 else float("nan")
            except ValueError:
                pval = float("nan")
            d[str(tg)] = dict(n_states=int(len(v)), censored=int(tot - len(v)), mean_log10_ratio=float(v.mean()),
                              median_ratio=float(10 ** v.median()), frac_tree_cheaper=float((v > 0).mean()),
                              wilcoxon_p=pval)
        pp = bt[(bt.lattice == lat) & (bt.layout == "geometric") & (bt.target == 0.99)]
        d["p99_by_h"] = {f"{h}": {m: [float(pp[(pp.h == h) & (pp.method == m)].params_needed.mean()),
                                      float(pp[(pp.h == h) & (pp.method == m)].params_needed.std())]
                                  for m in ("MPS", "TTN")} for h in sorted(pp.h.unique())}
        s["ground"][lat] = d
    s["layout_ablation"] = ab_out.to_dict(orient="records")
    s["robust"] = rb.groupby(["disorder", "method"]).params_needed.agg(["mean", "std"]).reset_index().to_dict(orient="records")
    qs = qpr[qpr.target == 0.99].groupby(["p", "t"]).log10_ratio.agg(["mean", "std", "count"]).reset_index()
    s["quench_log10_ratio"] = qs.to_dict(orient="records")
    qm = qb[qb.target == 0.99].groupby(["p", "t", "method"]).params_needed.agg(["mean", "std"]).reset_index()
    s["quench_p99"] = qm.to_dict(orient="records")
    qe = quench.drop_duplicates(["p", "t", "seed"]).groupby(["p", "t"]).mid_entropy.agg(["mean", "std"]).reset_index()
    s["quench_entropy"] = qe.to_dict(orient="records")
    qc = qb[qb.target == 0.99].groupby(["p", "t", "method"]).params_needed.apply(lambda x: int(x.isna().sum())).reset_index()
    s["quench_censored_at_0.99"] = qc[qc.params_needed > 0].to_dict(orient="records")
    if len(hea):
        hs = hea.groupby(["h", "depth"]).agg(params=("params", "first"), cz=("cz", "first"),
                                              f_mean=("fidelity", "mean"), f_std=("fidelity", "std")).reset_index()
        s["hea"] = hs.to_dict(orient="records")
        lat = hea.lattice.iloc[0]
        cmp_rows = []
        for h in sorted(hea.h.unique()):
            for method_fmt, label in (("mps", "MPS"), ("ttn-geometric", "TTN")):
                sub = g[(g.lattice == lat) & (g.h == h) & (g.layout == "geometric") & (g.fmt == method_fmt)]
                for chi in (1, 2, 3, 4, 6):
                    ss = sub[sub.chi == chi]
                    cmp_rows.append(dict(h=h, method=label, chi=chi, params=float(ss.params.mean()),
                                         f_mean=float(ss.fidelity.mean()), f_std=float(ss.fidelity.std())))
        s["tn_small_budget"] = cmp_rows
    return s
