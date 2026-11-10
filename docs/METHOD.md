# Method

## Target states

Transverse-field Ising Hamiltonian (TFIM) H = - sum_<ij> J_ij Z_i Z_j - h sum_i X_i on open 3x3, 4x3 and 4x4 square lattices and a 16-site chain (control). J_ij = 1 + w N(0,1), clipped at 0.1; each seed is one disorder realisation (w = 0.2 unless swept).

* **E1, ground states.** Exact lowest state in the Z2-even sector by matrix-free Lanczos, h / J in {0.5, ..., 6} (the clean two-dimensional critical point is near h / J = 3).
* **E2, noisy quench.** First-order Trotter circuit (time step 0.1) from |0...0> at h / J = 3 on 4x4. After each step every qubit suffers a random Pauli (X, Y or Z with equal chance) with probability p in {0, 0.003, 0.01, 0.03}: one stochastic trajectory of a depolarizing channel. Snapshots at t = 0.5, 1.0, 1.5, 2.0 (units of 1/J). Entries are complex, so each stored entry counts as 2 real parameters.
* **E3, quantum baseline.** The hardware-efficient ansatz is fitted to the 4x3 ground states at h / J in {1, 3, 5}, depths 1, 2, 4, 6, 8.

## Compared representations

* **Classical baseline A, matrix product state (MPS).** Snake order along the long side (cut width = short side), left-to-right sequential Schmidt truncation (tensor-train singular value decomposition, SVD, style) to bond dimension chi.
* **Classical baseline B / method, tree tensor network (TTN).** Binary tree built by recursive bisection of the longer lattice side (geometry-aware); bottom-up sequential truncation of every non-root node's bipartition to bond dimension chi. For the chain control the tree is balanced over the natural order.
* **Standard quantum baseline, hardware-efficient ansatz (HEA).** Layers of RY-RZ rotations on all qubits, each followed by controlled-Z (CZ) gates on one edge-colour class of the lattice graph; fitted by Adam (120 steps, learning rate 0.1) to maximise fidelity with the exact target (exact statevector, no shot noise). Real parameters = 2 n (depth + 1).

Parameter counts are tensor entries at nominal bond dimensions (physical dimension 2, caps 2^min(|S|, n - |S|)); gauge freedom and isometry constraints are ignored for both formats.

## Metrics

Fidelity F = |<psi|approx>|^2 / (<psi|psi><approx|approx>); budget P(F) as in `RESEARCH_GAP.md`; paired log10 ratio; fraction of states where the tree is cheaper; Wilcoxon signed-rank p-value on paired log-ratios (states at different h are not independent, so p-values are descriptive).

## Ablation, robustness, seeds

* **Ablation:** geometry-aware versus random leaf ordering for both formats (4x4, h / J in {1, 3, 5}).
* **Noise / robustness sweeps:** Pauli error rate p (E2) and coupling-disorder strength w in {0, 0.1, 0.2, 0.4, 0.6} at h / J = 3 (E1).
* **Seeds:** 5 disorder realisations for E1 and E2 (noise trajectories have their own seeded generators); 2 for the HEA baseline (the most expensive part).

## Known simplifications

Sequential truncation is not variationally optimal for either format; variational refinement would lower both budgets. The HEA is trained with a short fixed schedule, so its fidelities are lower bounds on what the ansatz can represent.
