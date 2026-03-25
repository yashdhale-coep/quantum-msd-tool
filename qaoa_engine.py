"""
qaoa_engine.py
All physics, QUBO, and QAOA logic for the MSD optimization tool.
"""

import numpy as np
from itertools import product as iterproduct


# ─────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────

def natural_frequency(k, m):
    return np.sqrt(k / m)

def damping_ratio(c, k, m):
    return c / (2 * m * natural_frequency(k, m))

def peak_amplitude(k, c, m=1.0, F0=1.0):
    omega_n = natural_frequency(k, m)
    zeta    = damping_ratio(c, k, m)
    if zeta >= 1.0 / np.sqrt(2):
        return float(F0 / k)
    omega_peak = omega_n * np.sqrt(max(1 - 2 * zeta**2, 0))
    denom = np.sqrt((k - m * omega_peak**2)**2 + (c * omega_peak)**2)
    return float(F0 / denom)

def frequency_response(k, c, m, F0, omega_max=None, n_points=500):
    if omega_max is None:
        omega_max = float(np.sqrt(k / m)) * 4
    omega = np.linspace(0.5, omega_max, n_points)
    X = F0 / np.sqrt((k - m * omega**2)**2 + (c * omega)**2)
    return omega.tolist(), X.tolist()


# ─────────────────────────────────────────────────────────────
# Brute Force
# ─────────────────────────────────────────────────────────────

def run_brute_force(m, F0, k_values, c_values):
    """
    Evaluate all k×c combinations.
    Returns structured results for frontend consumption.
    """
    results = []
    cost_matrix = []

    for k in k_values:
        row = []
        for c in c_values:
            xp   = peak_amplitude(k, c, m, F0)
            zeta = damping_ratio(c, k, m)
            row.append(xp)
            results.append({
                "k": k, "c": c,
                "x_peak": round(xp, 8),
                "zeta": round(zeta, 4),
                "damping_type": (
                    "overdamped" if zeta > 1 else
                    "critically" if abs(zeta - 1) < 0.01 else
                    "underdamped"
                )
            })
        cost_matrix.append(row)

    best = min(results, key=lambda r: r["x_peak"])

    # Frequency response for all combos (for overlay plot)
    omega_max = max(float(np.sqrt(k / m)) for k in k_values) * 4
    freq_curves = []
    for r in results:
        omega, X = frequency_response(r["k"], r["c"], m, F0, omega_max)
        freq_curves.append({
            "k": r["k"], "c": r["c"],
            "omega": omega, "X": X,
            "is_best": r["k"] == best["k"] and r["c"] == best["c"]
        })

    return {
        "results":      results,
        "cost_matrix":  cost_matrix,
        "k_values":     k_values,
        "c_values":     c_values,
        "best":         best,
        "freq_curves":  freq_curves,
        "n_combos":     len(results)
    }


# ─────────────────────────────────────────────────────────────
# QUBO Construction
# ─────────────────────────────────────────────────────────────

def build_qubo(k_values, c_values, m, F0, lam=3.0):
    N_k = len(k_values)
    N_c = len(c_values)
    N   = N_k + N_c

    # Normalised cost matrix
    cost = np.array([[peak_amplitude(k, c, m, F0) for c in c_values] for k in k_values])
    cost_min, cost_max = cost.min(), cost.max()
    span = cost_max - cost_min
    cost_norm = (cost - cost_min) / span if span > 1e-12 else cost * 0

    Q = np.zeros((N, N))

    # Cross-block objective terms  x[i] * x[N_k+j]
    for i in range(N_k):
        for j in range(N_c):
            Q[i, N_k + j] += cost_norm[i, j]

    # One-hot penalty k-block
    for i in range(N_k):
        Q[i, i] += lam * (1 - 2)
        for j in range(i + 1, N_k):
            Q[i, j] += 2 * lam

    # One-hot penalty c-block
    for i in range(N_k, N):
        Q[i, i] += lam * (1 - 2)
        for j in range(i + 1, N):
            Q[i, j] += 2 * lam

    return Q, cost_norm


def qubo_to_ising(Q):
    """
    Convert upper-triangular QUBO Q to Ising h, J.
    Substitution: x_i = (1 - z_i) / 2  where z_i in {-1, +1}
    QUBO: min x^T Q x
    Ising: min sum_i h_i z_i + sum_{i<j} J_ij z_i z_j  + offset
    """
    n     = Q.shape[0]
    # Symmetrize
    Qfull = Q + Q.T - np.diag(np.diag(Q))
    h      = np.zeros(n)
    J      = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        # Diagonal: Q_ii * x_i = Q_ii * (1-z_i)/2
        # contributes Q_ii/2 constant and -Q_ii/2 * z_i
        h[i]   -= Qfull[i, i] / 2.0
        offset += Qfull[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            # Q_ij * x_i * x_j = Q_ij * (1-z_i)/2 * (1-z_j)/2
            # = Q_ij/4 * (1 - z_i - z_j + z_i*z_j)
            J[i, j] += Qfull[i, j] / 4.0
            h[i]    -= Qfull[i, j] / 4.0
            h[j]    -= Qfull[i, j] / 4.0
            offset  += Qfull[i, j] / 4.0

    return h, J, offset


# ─────────────────────────────────────────────────────────────
# Statevector QAOA simulator (pure numpy — no Qiskit needed)
# Mathematically identical to Qiskit AerSimulator statevector
# ─────────────────────────────────────────────────────────────

def _ising_energy(bitstring, h, J):
    """Compute Ising energy for a given bitstring (list of 0/1)."""
    z = np.array([1 - 2 * b for b in bitstring], dtype=float)  # 0→+1, 1→-1
    energy = np.dot(h, z)
    n = len(z)
    for i in range(n):
        for j in range(i + 1, n):
            energy += J[i, j] * z[i] * z[j]
    return energy


def _build_energy_vector(n_qubits, h, J):
    """Precompute Ising energy for all 2^n bitstrings."""
    N = 2 ** n_qubits
    energies = np.zeros(N)
    for idx in range(N):
        bits = [(idx >> (n_qubits - 1 - b)) & 1 for b in range(n_qubits)]
        energies[idx] = _ising_energy(bits, h, J)
    return energies


def _apply_cost_unitary(state, energies, gamma):
    """e^{-i gamma H_cost} |state>  — diagonal in computational basis."""
    return state * np.exp(-1j * gamma * energies)


def _apply_mixer_unitary(state, beta, n_qubits):
    """e^{-i beta sum_i X_i} |state>  — product of single-qubit Rx rotations."""
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    cur = state.copy()
    for qubit in range(n_qubits):
        mask = 1 << (n_qubits - 1 - qubit)
        nxt  = np.empty_like(cur)
        for idx in range(len(cur)):
            partner = idx ^ mask
            nxt[idx] = cos_b * cur[idx] - 1j * sin_b * cur[partner]
        cur = nxt
    return cur


def _qaoa_state(params, n_qubits, energies, p):
    """Run p-layer QAOA and return final statevector."""
    N     = 2 ** n_qubits
    state = np.ones(N, dtype=complex) / np.sqrt(N)   # uniform superposition

    for layer in range(p):
        gamma = params[2 * layer]
        beta  = params[2 * layer + 1]
        state = _apply_cost_unitary(state, energies, gamma)
        state = _apply_mixer_unitary(state, beta, n_qubits)

    return state


def _expectation(params, n_qubits, energies, p):
    state = _qaoa_state(params, n_qubits, energies, p)
    probs = np.abs(state) ** 2
    return float(np.dot(probs, energies))


def _lbfgsb_minimize(func, x0, n_restarts=5, maxiter=300):
    """
    Multi-start L-BFGS-B optimization to escape local minima.
    Scales restarts with the number of parameters to handle higher p layers better.
    """
    from scipy.optimize import minimize

    best_val     = np.inf
    best_params  = x0.copy()
    best_history = []

    rng = np.random.default_rng(7)
    starts = [x0] + [rng.uniform(0, 2 * np.pi, size=len(x0)) for _ in range(n_restarts - 1)]

    bounds = [(0, 2 * np.pi)] * len(x0)

    for start in starts:
        energy_history = []

        def wrapped(x):
            val = func(x)
            energy_history.append(float(val))
            return val

        res = minimize(wrapped, start, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-7})

        if res.fun < best_val:
            best_val     = res.fun
            best_params  = res.x
            best_history = energy_history

    return best_params, best_history


def _parameter_transfer_minimize(n_qubits, energies, n_restarts=6, maxiter=300):
    """
    Optimise QAOA angles for p=1 (few parameters, reliable convergence).
    The returned angles are used by the caller as a warm-start seed for
    higher-p optimisation, implementing the parameter transfer heuristic.
    Returns (best_params, energy_history).
    """
    from scipy.optimize import minimize

    def objective_p1(params):
        return _expectation(params, n_qubits, energies, 1)

    best_params  = None
    best_history = []
    best_val     = np.inf
    bounds       = [(0, 2 * np.pi)] * 2
    rng          = np.random.default_rng(7)

    starts = [np.array([0.5, 0.5])] + [rng.uniform(0, 2 * np.pi, 2) for _ in range(n_restarts - 1)]
    for start in starts:
        hist = []

        def w1(x, _h=hist):
            v = objective_p1(x)
            _h.append(float(v))
            return v

        res = minimize(w1, start, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-7})
        if res.fun < best_val:
            best_val     = res.fun
            best_params  = res.x
            best_history = hist

    return best_params, best_history


def run_qaoa(m, F0, k_values, c_values, p_layers=1, lam=5.0, n_shots=4096):
    """
    Full QAOA pipeline. Returns results + convergence history + probability distribution.
    """
    N_k = len(k_values)
    N_c = len(c_values)
    n_qubits = N_k + N_c

    # Build QUBO and Ising
    Q, cost_norm = build_qubo(k_values, c_values, m, F0, lam)
    h, J, offset = qubo_to_ising(Q)

    # Precompute all 2^n energies
    energies = _build_energy_vector(n_qubits, h, J)

    # Optimize angles — use parameter transfer for p>1, L-BFGS-B throughout
    def objective(params):
        return _expectation(params, n_qubits, energies, p_layers)

    if p_layers == 1:
        x0 = np.array([0.5, 0.5])
        n_restarts = 6
        optimal_params, energy_history = _lbfgsb_minimize(objective, x0, n_restarts=n_restarts, maxiter=400)
    else:
        # Parameter transfer: converge p=1 first, then extend and re-optimise each layer
        rng = np.random.default_rng(7)
        # Get good p=1 seed via transfer helper (energy history from this phase is discarded;
        # we report only the final higher-p convergence for a cleaner visualization)
        seed_params, _ = _parameter_transfer_minimize(n_qubits, energies, n_restarts=6, maxiter=400)

        # Extend seed to full p_layers by repeating best angles
        x0 = np.tile(seed_params, p_layers)
        # Add noise to break symmetry for higher layers
        x0[2:] += rng.uniform(-0.3, 0.3, size=len(x0) - 2)
        x0 = np.clip(x0, 0, 2 * np.pi)

        # More restarts for higher p (more parameters = harder landscape)
        n_restarts = 4 + 2 * p_layers
        optimal_params, energy_history = _lbfgsb_minimize(objective, x0, n_restarts=n_restarts, maxiter=500)

    # Final state + measurement probabilities
    final_state = _qaoa_state(optimal_params, n_qubits, energies, p_layers)
    sv_probs    = np.abs(final_state) ** 2  # index i = bitstring with MSB-first ordering

    # Simulate shots using statevector probabilities
    rng     = np.random.default_rng(42)
    all_idx = np.arange(len(sv_probs))
    pnorm   = sv_probs / sv_probs.sum()
    indices = rng.choice(all_idx, size=n_shots, p=pnorm)
    counts  = np.bincount(indices, minlength=len(sv_probs))
    meas_probs = counts / n_shots

    # Decode bitstrings
    def idx_to_bits(idx):
        # MSB first: bit 0 = qubit 0
        return [(idx >> (n_qubits - 1 - b)) & 1 for b in range(n_qubits)]

    def decode(bits):
        k_bits = bits[:N_k]
        c_bits = bits[N_k:]
        if sum(k_bits) == 1 and sum(c_bits) == 1:
            k_val = k_values[k_bits.index(1)]
            c_val = c_values[c_bits.index(1)]
            return k_val, c_val, True
        return None, None, False

    # Build sorted probability list
    prob_list = []
    for idx in range(len(meas_probs)):
        if meas_probs[idx] < 1e-5:
            continue
        bits = idx_to_bits(idx)
        k_val, c_val, valid = decode(bits)
        bs = "".join(str(b) for b in bits)
        prob_list.append({
            "bitstring": bs,
            "prob": round(float(meas_probs[idx]), 5),
            "k": k_val,
            "c": c_val,
            "valid": valid,
            "x_peak": round(peak_amplitude(k_val, c_val, m, F0), 8) if valid else None
        })
    prob_list.sort(key=lambda x: x["prob"], reverse=True)

    # Best valid result
    qaoa_best = next((p for p in prob_list if p["valid"]), None)

    # Classical best for comparison
    all_combos = [(k, c, peak_amplitude(k, c, m, F0)) for k, c in iterproduct(k_values, c_values)]
    classical_best = min(all_combos, key=lambda x: x[2])

    match = (
        qaoa_best is not None and
        qaoa_best["k"] == classical_best[0] and
        qaoa_best["c"] == classical_best[1]
    )

    # Thin out energy history for frontend (max 100 points)
    step = max(1, len(energy_history) // 100)
    convergence = [{"iter": i * step, "energy": round(v, 6)}
                   for i, v in enumerate(energy_history[::step])]

    return {
        "qaoa_best":        qaoa_best,
        "classical_best":   {"k": classical_best[0], "c": classical_best[1], "x_peak": round(classical_best[2], 8)},
        "match":            match,
        "optimal_params":   [round(float(v), 4) for v in optimal_params],
        "final_energy":     round(float(objective(optimal_params)), 6),
        "offset":           round(float(offset), 6),
        "convergence":      convergence,
        "prob_list":        prob_list[:30],   # top 30 for histogram
        "n_qubits":         n_qubits,
        "p_layers":         p_layers,
        "n_shots":          n_shots,
        "n_iters":          len(energy_history)
    }
