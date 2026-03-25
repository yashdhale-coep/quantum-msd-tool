"""
app.py — Flask backend for Quantum MSD Optimization Tool
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback
from qaoa_engine import run_brute_force, run_qaoa, frequency_response, peak_amplitude

app = Flask(__name__)
CORS(app)

# ── Limits ──────────────────────────────────────────────────
MAX_K_VALUES   = 6
MAX_C_VALUES   = 6
MAX_BRUTE_COMBOS = 36   # 6×6
MAX_P_LAYERS   = 3


def parse_value_list(raw, name, min_val=0.01, max_val=1e6, max_count=6):
    """Parse a comma-separated string of positive floats."""
    try:
        vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"{name}: must be comma-separated numbers")
    if len(vals) < 2:
        raise ValueError(f"{name}: need at least 2 values")
    if len(vals) > max_count:
        raise ValueError(f"{name}: max {max_count} values allowed")
    for v in vals:
        if v < min_val or v > max_val:
            raise ValueError(f"{name}: each value must be between {min_val} and {max_val}")
    return sorted(list(set(vals)))   # deduplicate + sort


def parse_inputs(data):
    m  = float(data.get("m",  1.0))
    F0 = float(data.get("F0", 1.0))
    if m  <= 0: raise ValueError("Mass m must be positive")
    if F0 <= 0: raise ValueError("Forcing F0 must be positive")
    if m  > 1000: raise ValueError("Mass m: max 1000 kg")
    if F0 > 1000: raise ValueError("Forcing F0: max 1000 N")

    k_values = parse_value_list(data.get("k_values", "500,1000,2000,4000"), "k values", 1, 1e6, MAX_K_VALUES)
    c_values = parse_value_list(data.get("c_values", "5,10,20,40"),          "c values", 0.01, 1e5, MAX_C_VALUES)

    n_combos = len(k_values) * len(c_values)
    if n_combos > MAX_BRUTE_COMBOS:
        raise ValueError(f"Too many combinations ({n_combos}). Max is {MAX_BRUTE_COMBOS} ({MAX_K_VALUES}×{MAX_C_VALUES})")

    return m, F0, k_values, c_values


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/brute-force", methods=["POST"])
def api_brute_force():
    try:
        data     = request.get_json()
        m, F0, k_values, c_values = parse_inputs(data)
        result   = run_brute_force(m, F0, k_values, c_values)
        return jsonify({"ok": True, "data": result})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Server error: " + traceback.format_exc()}), 500


@app.route("/api/qaoa", methods=["POST"])
def api_qaoa():
    try:
        data     = request.get_json()
        m, F0, k_values, c_values = parse_inputs(data)
        p_layers = int(data.get("p_layers", 1))
        lam      = float(data.get("lam", 3.0))

        if p_layers < 1 or p_layers > MAX_P_LAYERS:
            raise ValueError(f"p_layers must be 1–{MAX_P_LAYERS}")
        if lam < 0.5 or lam > 20:
            raise ValueError("Lambda must be between 0.5 and 20")

        result = run_qaoa(m, F0, k_values, c_values, p_layers=p_layers, lam=lam)
        return jsonify({"ok": True, "data": result})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Server error: " + traceback.format_exc()}), 500


@app.route("/api/freq-response", methods=["POST"])
def api_freq_response():
    """Return frequency response for a single (k, c) combination — for heatmap cell click."""
    try:
        data  = request.get_json()
        m     = float(data.get("m",  1.0))
        F0    = float(data.get("F0", 1.0))
        k     = float(data.get("k"))
        c     = float(data.get("c"))
        omega, X = frequency_response(k, c, m, F0)
        xp    = peak_amplitude(k, c, m, F0)
        return jsonify({"ok": True, "omega": omega, "X": X, "x_peak": xp})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
