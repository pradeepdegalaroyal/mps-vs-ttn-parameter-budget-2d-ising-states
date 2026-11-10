# Research gap: chain versus tree tensor networks at equal parameter budget

## State of the art (from the supplied literature)

The supplied Scopus set contains six papers, all of role *recent*; it holds no landmark, review or adjacent papers. The table below therefore uses every paper in the set (fewer than the usual ten, and from one role), and no other reference is cited.

| Paper (first author, year, DOI) | Role | What it does | What it does NOT cover |
|---|---|---|---|
| Willner et al., 2027, 10.1016/j.laa.2026.09.019 | recent | Quotient geometry and first/second-order Riemannian optimisation of tree tensor networks (TTNs); a digit-classification test shows a trade-off between two horizontal distributions. | Physical many-body states; any comparison with a chain format, the matrix product state (MPS) at equal parameter count; noisy circuit states. |
| Pang et al., 2027, 10.1007/978-981-92-6831-3_2 | recent | Replaces Choi matrices by MPS in quantum-protocol reachability analysis: 20 qubits instead of 12 within 16 GB, under depolarizing noise. | Which tensor-network topology is cheapest; two-dimensional geometry; fidelity-versus-parameters curves. |
| Wang, 2027, 10.1016/j.ijleo.2026.172946 | recent | Gaussian-enhanced chaotic light: nonclassicality and von Neumann entropy depend on the enhancement strength. | Many-body entanglement structure or its compressibility (used here only as motivation to report entropies). |
| Sobhani, 2027, 10.1016/j.ijleo.2026.172948 | recent | Photon-number quantum non-demolition measurement with two weak cross-Kerr interactions. | Simulation cost or tensor networks. |
| Afzali, 2027, 10.22075/ppam.2026.41410.1235 | recent | Teleportation fidelity from Cooper-pair spin entanglement in s- and d-wave superconductors. | Classical simulability of the states. |
| Shen et al., 2027, 10.1016/j.cap.2026.10.019 | recent | Spatially inhomogeneous electron-nuclear spin system in n-AlGaAs: multiple Larmor components from site-dependent nuclear fields. | Quantum simulation; motivates *disordered* couplings as a robustness axis. |

The two papers that matter technically are Willner et al. (TTN geometry and optimisation) and Pang et al. (MPS as the memory-saving workhorse under noise, with a 16 GB benchmark). They stop short of the same question: **for a given budget of stored numbers, is a tree layout or a chain layout the cheaper classical description of a two-dimensional quantum state, and how does hardware-like noise change that answer?**

## Gap

1. MPS dominates practice (Pang et al.), while TTNs are analysed mostly as optimisation objects (Willner et al.); no supplied paper measures the *parameter cost* of the two topologies on the *same* two-dimensional many-body state.
2. The decision between "classically storable" and "needs a quantum device" is a statement about a parameter budget at a target fidelity, yet it is rarely reported together with a gate-based variational baseline (a hardware-efficient ansatz, HEA) fitted to the same state.
3. Output states of noisy Trotterised circuits are what a quantum device would produce; whether stochastic Pauli errors make such states cheaper or dearer for chain and tree formats is not covered.

## Falsifiable hypotheses and success metric

Metric: P(F), the smallest number of stored real parameters whose sequential singular value decomposition (SVD) compression reaches fidelity F with the exact state (log-log interpolation over bond dimensions; right-censored values are counted and reported). The paired statistic is log10(P_MPS / P_TTN) per state; positive means the tree is cheaper.

* **H1 (two-dimensional advantage).** On two-dimensional (2D) transverse-field Ising ground states, a geometry-aware binary TTN needs fewer parameters than a snake-ordered MPS at F >= 0.99 for more than half of all states, and the advantage grows from 9 to 16 sites. *Falsified* if the median ratio is <= 1 or does not grow with size.
* **H2 (one-dimensional control).** On a one-dimensional (1D) chain the MPS is cheaper than the TTN (ratio < 1).
* **H3 (layout ablation).** Scrambling the leaf order costs both formats parameters.
* **H4 (noise).** Pauli-trajectory noise raises the middle-cut entanglement of a Trotter quench and the budgets of both formats; the sign of the tree advantage is unchanged.
* **H5 (quantum baseline).** At matched real-parameter count a fitted HEA reaches lower fidelity than either tensor network on the same states.

## Novelty (one sentence)

This project measures, with paired statistics, phase maps and a gate-based variational baseline, the parameter budget of chain versus tree tensor networks on two-dimensional transverse-field Ising ground states and on noisy Trotter-circuit states, a combination that the supplied papers do not report.

## Scale statement

All results are exact-statevector simulations with 9 to 16 qubits (laptop scale). They characterise budgets at small size and say nothing about asymptotics.
