"""Quantum Amplitude Estimation (QAE) applied to a real probability from
this project's own model, as a demonstration of *why* quantum computing is
relevant to Monte Carlo simulation specifically (not a generic "add quantum"
exercise) — the same principle behind quantum finance's Monte Carlo
option-pricing work (e.g. IBM/Goldman Sachs, Stamatopoulos et al. 2020).

The core idea: classical Monte Carlo estimates a probability p by sampling
M times, with error shrinking as O(1/sqrt(M)) — this is exactly what
src/dixon_coles.py's simulate_season does, 10,000 times, to get title/top4/
relegation percentages. Quantum Amplitude Estimation estimates the same p
using M *oracle queries* (not classical samples) with error shrinking as
O(1/M) — a quadratic reduction in the queries needed for the same accuracy.

What's real here and what's a simplification, stated plainly:
- The probability being estimated (p = P(Arsenal beat Man City), from
  src/dixon_coles.py's fitted, backtested model) is real, not a toy number.
- The amplitude is *loaded* onto a qubit via a single Ry rotation, not
  computed by the quantum circuit from first principles (that would mean
  quantum state preparation of the underlying Poisson goal distributions —
  a real, harder research problem; not attempted here). The quantum part
  being demonstrated is the *estimation* algorithm's query complexity, not
  a claim that goals were "simulated quantumly."
- Everything runs on Qiskit Aer's noiseless simulator. Today's real NISQ
  hardware doesn't have the coherence time for the deep QPE circuit QAE
  needs to actually realize this speedup — this demonstrates the
  algorithmic principle, not a claim of practical quantum advantage today.

Run: python -m src.quantum_estimation
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms import AmplitudeEstimation, EstimationProblem, IterativeAmplitudeEstimation


def amplitude_oracle(p: float) -> QuantumCircuit:
    """A single-qubit state |psi> = sqrt(1-p)|0> + sqrt(p)|1> — the
    'A' operator whose |1>-amplitude squared is exactly p. This is the
    quantity Quantum Amplitude Estimation is built to estimate."""
    qc = QuantumCircuit(1)
    theta = 2 * np.arcsin(np.sqrt(p))
    qc.ry(theta, 0)
    return qc


def qae_estimate(p_true: float, num_eval_qubits: int) -> tuple[float, int]:
    """Runs textbook QAE (quantum phase estimation on the Grover operator
    built from `amplitude_oracle`). Returns (estimate, oracle_queries),
    where oracle_queries = 2^num_eval_qubits - 1 Grover iterations, each
    one call to A (matching the M in classical Monte Carlo's error
    formula so the two are compared on equal footing)."""
    problem = EstimationProblem(
        state_preparation=amplitude_oracle(p_true),
        objective_qubits=[0],
    )
    ae = AmplitudeEstimation(num_eval_qubits=num_eval_qubits)
    result = ae.estimate(problem)
    queries = 2**num_eval_qubits - 1
    return result.estimation, queries


def iae_estimate(p_true: float, epsilon_target: float) -> tuple[float, int]:
    """Iterative Amplitude Estimation (Grinko et al. 2019) — the variant
    actually used in the quantum-finance Monte Carlo literature this
    module is modeled on. Adaptive number of Grover iterations rather than
    a fixed power-of-two grid, so its query count and error track each
    other more smoothly than canonical QPE-based AE."""
    problem = EstimationProblem(state_preparation=amplitude_oracle(p_true), objective_qubits=[0])
    iae = IterativeAmplitudeEstimation(epsilon_target=epsilon_target, alpha=0.05)
    result = iae.estimate(problem)
    return result.estimation, result.num_oracle_queries


def classical_mc_estimate(p_true: float, n_samples: int, rng: np.random.Generator) -> float:
    return rng.binomial(1, p_true, size=n_samples).mean()


def compare(p_true: float, max_eval_qubits: int = 8, mc_trials: int = 200, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(1, max_eval_qubits + 1):
        qae_est, queries = qae_estimate(p_true, k)
        qae_error = abs(qae_est - p_true)

        mc_errors = [abs(classical_mc_estimate(p_true, queries, rng) - p_true) for _ in range(mc_trials)]
        mc_error = float(np.mean(mc_errors))

        rows.append({
            "eval_qubits": k,
            "queries": queries,
            "qae_estimate": qae_est,
            "qae_error": qae_error,
            "classical_error": mc_error,
        })
    return rows


def compare_iae(
    p_true: float, epsilon_targets: list[float], iae_trials: int = 8, mc_trials: int = 200, seed: int = 42
) -> list[dict]:
    """Skips any epsilon target the adaptive IAE schedule satisfies with
    zero Grover rounds (i.e. the initial classical-only estimate already
    met that precision) — there's no query count to compare against
    classical sampling on in that degenerate case.

    Each IAE point is itself a probabilistic measurement outcome (even on
    a noiseless simulator, finite-shot sampling noise applies), so it's
    averaged over `iae_trials` independent runs — the same reason
    classical Monte Carlo below is averaged over `mc_trials`, not judged
    on a single draw."""
    rng = np.random.default_rng(seed)
    rows = []
    for eps in epsilon_targets:
        iae_ests, queries_list = [], []
        for _ in range(iae_trials):
            iae_est, queries = iae_estimate(p_true, eps)
            if queries == 0:
                continue
            iae_ests.append(iae_est)
            queries_list.append(queries)
        if not iae_ests:
            continue

        iae_error = float(np.mean([abs(e - p_true) for e in iae_ests]))
        queries = int(round(np.mean(queries_list)))

        mc_errors = [abs(classical_mc_estimate(p_true, queries, rng) - p_true) for _ in range(mc_trials)]
        mc_error = float(np.mean(mc_errors))

        rows.append({
            "epsilon_target": eps,
            "queries": queries,
            "iae_error": iae_error,
            "classical_error": mc_error,
        })
    return rows


def main() -> None:
    p_true = 0.4584  # Arsenal vs Man City, P(home win) — from src/dixon_coles.py's fitted, backtested model
    print(f"Estimating p = {p_true} (Arsenal vs Man City home-win probability, from the fitted Dixon-Coles model)\n")

    rows = compare(p_true)
    print(f"{'queries M':>10} {'QAE estimate':>13} {'QAE error':>11} {'Classical error':>16} {'QAE/Classical':>14}")
    for r in rows:
        ratio = r["qae_error"] / r["classical_error"] if r["classical_error"] else float("nan")
        print(f"{r['queries']:>10} {r['qae_estimate']:>13.5f} {r['qae_error']:>11.5f} {r['classical_error']:>16.5f} {ratio:>14.3f}")

    # empirical scaling exponents: error ~ M^-alpha, fit alpha via log-log slope
    queries = np.array([r["queries"] for r in rows], dtype=float)
    qae_err = np.array([r["qae_error"] for r in rows])
    cl_err = np.array([r["classical_error"] for r in rows])
    qae_slope = np.polyfit(np.log(queries), np.log(np.clip(qae_err, 1e-6, None)), 1)[0]
    cl_slope = np.polyfit(np.log(queries), np.log(np.clip(cl_err, 1e-6, None)), 1)[0]
    print(f"\nEmpirical error scaling (log-log slope): QAE ~ M^{qae_slope:.2f} (theory: -1), classical ~ M^{cl_slope:.2f} (theory: -0.5)")
    print("(Canonical QPE-based AE snaps estimates to a 2^k-point grid, which makes its")
    print(" empirical error curve stepped rather than smooth at low qubit counts — real")
    print(" behavior, not a bug. Iterative Amplitude Estimation below is the adaptive")
    print(" variant that avoids this and is what's actually used in practice.)")

    print("\n--- Iterative Amplitude Estimation (adaptive, smoother; each point averaged over 8 runs) ---")
    iae_rows = compare_iae(p_true, [0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    print(f"{'queries M':>10} {'IAE error (avg)':>16} {'Classical error':>16} {'IAE/Classical':>14}")
    for r in iae_rows:
        ratio = r["iae_error"] / r["classical_error"] if r["classical_error"] else float("nan")
        print(f"{r['queries']:>10} {r['iae_error']:>16.5f} {r['classical_error']:>16.5f} {ratio:>14.3f}")

    iq = np.array([r["queries"] for r in iae_rows], dtype=float)
    iae_err = np.array([r["iae_error"] for r in iae_rows])
    cl_err2 = np.array([r["classical_error"] for r in iae_rows])
    iae_slope = np.polyfit(np.log(iq), np.log(np.clip(iae_err, 1e-6, None)), 1)[0]
    cl_slope2 = np.polyfit(np.log(iq), np.log(np.clip(cl_err2, 1e-6, None)), 1)[0]
    print(f"\nEmpirical error scaling (log-log slope): IAE ~ M^{iae_slope:.2f} (theory: -1), classical ~ M^{cl_slope2:.2f} (theory: -0.5)")


if __name__ == "__main__":
    main()
