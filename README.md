# Quantum MSD Tool

A full-stack **Quantum + Classical optimization tool** for selecting optimal parameters of a single-degree-of-freedom **Mass–Spring–Damper (MSD)** system.

It minimizes peak vibration amplitude by comparing:
- **Classical brute-force search** (ground truth), and
- **QAOA-based optimization** (statevector simulation, from scratch in NumPy).

This repository implements the project described in:
`Group-No-5_quantum-msd-tool_642403021-612310156-612310149-612310145-612310068_Report.pdf`.

---

## What this project does

Given:
- Mass `m`
- Forcing amplitude `F0`
- Candidate stiffness values `K = {k1, ..., kN}`
- Candidate damping values `C = {c1, ..., cM}`

The tool finds the pair `(k*, c*)` that minimizes peak steady-state vibration amplitude `X_peak`.

It exposes an interactive browser UI that lets you:
- run brute-force optimization,
- run QAOA with depth `p = 1..3`,
- tune QUBO penalty `λ`,
- inspect heatmaps, frequency-response overlays, convergence curves, and bitstring probabilities,
- compare QAOA and brute-force outputs side-by-side.

---

## Key features

- Flask backend with JSON APIs
- Pure NumPy/SciPy QAOA simulation (no external quantum SDK required)
- QUBO construction with one-hot constraints
- QUBO → Ising conversion
- L-BFGS-B multi-start parameter optimization
- Parameter-transfer warm start for deeper QAOA layers
- Brute-force baseline for correctness checks
- Rich frontend visualization with Chart.js
- Input validation and constraint limits for stable execution

---

## Tech stack

- **Backend:** Python, Flask, Flask-CORS
- **Numerics/Optimization:** NumPy, SciPy (L-BFGS-B)
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Charts:** Chart.js

---

## Repository structure

```text
quantum-msd-tool/
├── app.py                 # Flask server + API routes + input validation
├── qaoa_engine.py         # Physics model, QUBO/Ising, QAOA, brute-force solver
├── templates/
│   └── index.html         # Full frontend UI (HTML/CSS/JS)
├── requirements.txt       # Python dependencies
├── Group-No-5_..._Report.pdf
└── Presentation.pdf
```

---

## Optimization pipeline

1. Parse and validate user inputs.
2. Compute vibration costs `X_peak(k, c)` for all combinations.
3. Build normalized QUBO objective + one-hot penalties.
4. Convert QUBO to Ising parameters `(h, J)`.
5. Run QAOA statevector simulation (depth `p`).
6. Optimize variational angles (`γ`, `β`) using L-BFGS-B (multi-start).
7. Sample measurements, decode valid one-hot bitstrings.
8. Return best quantum result and compare against brute-force optimum.

---

## API endpoints

### `GET /`
Serves the web interface.

### `POST /api/brute-force`
Runs exhaustive search over all `(k, c)` pairs.

Example payload:
```json
{
  "m": 1.0,
  "F0": 1.0,
  "k_values": "500,1000,2000,4000",
  "c_values": "5,10,20,40"
}
```

### `POST /api/qaoa`
Runs QAOA optimization and returns convergence + measurement statistics.

Example payload:
```json
{
  "m": 1.0,
  "F0": 1.0,
  "k_values": "500,1000,2000,4000",
  "c_values": "5,10,20,40",
  "p_layers": 2,
  "lam": 3.0
}
```

### `POST /api/freq-response`
Returns frequency response for a specific `(k, c)` pair.

Example payload:
```json
{
  "m": 1.0,
  "F0": 1.0,
  "k": 4000,
  "c": 40
}
```

---

## Input limits and constraints

Implemented server-side in `app.py`:

- `m`: `(0, 1000]`
- `F0`: `(0, 1000]`
- `k_values`: 2 to 6 values, each in `[1, 1e6]`
- `c_values`: 2 to 6 values, each in `[0.01, 1e5]`
- Maximum combinations: `6 × 6 = 36`
- `p_layers`: `1..3`
- `λ (lam)`: `[0.5, 20]`

These limits keep runtime and memory practical for statevector simulation.

---

## Quick start

### 1) Clone and enter project
```bash
git clone https://github.com/yashdhale-coep/quantum-msd-tool.git
cd quantum-msd-tool
```

### 2) Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run server
```bash
python app.py
```

### 5) Open in browser
Go to: `http://127.0.0.1:5000`

---

## Default experiment (from project report)

Using:
- `m = 1.0`
- `F0 = 1.0`
- `K = {500, 1000, 2000, 4000}`
- `C = {5, 10, 20, 40}`
- `p = 2`, `λ = 3.0`

Expected behavior:
- Brute force and QAOA both identify **`k* = 4000`, `c* = 40`**
- `X_peak ≈ 4.167 × 10⁻⁴`
- QAOA quality improves with depth, with diminishing returns from `p=2` to `p=3`

---

## Scalability notes

This implementation uses full statevector simulation, so memory scales as **O(2ⁿ)**.
For practical local use, keep total qubits (`N_k + N_c`) small (the report recommends up to ~12 qubits for typical hardware).

---

## Limitations

- No real quantum hardware execution (simulation-only)
- No noise model / error-mitigation flow
- Single-DOF MSD formulation
- One-hot encoding can be qubit-inefficient for large candidate sets
- For small problem sizes, brute-force is usually faster than QAOA

---

## Future scope

- Integrate with real quantum backends (IBM/Rigetti/IonQ)
- Add realistic noise channels and mitigation techniques
- Explore alternate encodings (domain-wall/binary)
- Add QAOA variants and noise-robust optimizers (e.g., SPSA)
- Extend to multi-DOF mechanical systems

---

## Acknowledgement

Project developed by Group No. 5, COEP Technological University, Pune.
See the included report PDF for full methodology, experiments, and references.
