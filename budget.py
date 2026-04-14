"""Population state-driven evaluation budget allocation (Section 3.4)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .surrogate import SurrogateSet
from .utils import constraint_violation


def allocate_budget(
    pop_b_x: NDArray[np.floating],
    surrogates: SurrogateSet,
    F_true: NDArray[np.floating],
    G_true: NDArray[np.floating],
    k: int,
    w1: float = 2.0,
    w2: float = 2.0,
    w3: float = 2.0,
    b: float = -1.0,
) -> tuple[int, int]:
    """Split the per-iteration budget *k* between Task A and Task B.

    Parameters
    ----------
    pop_b_x : ndarray of shape (N, d)
        Decision variables of Task B population.
    surrogates : SurrogateSet
        Trained surrogates (provides e_obj, e_con, and constraint predictions).
    F_true : ndarray of shape (n_archive, m)
        True objective values in the archive (for normalization denominators).
    G_true : ndarray of shape (n_archive, q)
        True constraint values in the archive.
    k : int
        Total evaluations per iteration.
    w1, w2, w3, b : float
        Sigmoid weights and bias.

    Returns
    -------
    k_A, k_B : int
        Evaluations allocated to Task A and Task B.
    """
    # --- Feature 1: feasibility ratio ρ_f (Eq. 9–10) ---
    G_pred, _ = surrogates.predict_constraints(pop_b_x)
    cv = constraint_violation(G_pred)
    rho_f = float(np.mean(cv <= 0.0))

    # --- Feature 2–3: normalized surrogate errors (Eq. 11–12) ---
    # Denominator: average std of training responses
    sigma_f = float(np.mean(np.std(F_true, axis=0)))
    sigma_g = float(np.mean(np.std(G_true, axis=0)))

    e_obj_bar = surrogates.e_obj / max(sigma_f, 1e-12)
    e_con_bar = surrogates.e_con / max(sigma_g, 1e-12)

    # Clamp e_con_bar when σ_g ≈ 0 (paper Section 3.4)
    if sigma_g < 1e-8:
        e_con_bar = 1.0

    # --- Sigmoid allocation (Eq. 13) ---
    z = w1 * rho_f + w2 * e_obj_bar - w3 * e_con_bar + b
    alpha_a = 1.0 / (1.0 + np.exp(-z))

    # --- Budget split (Eq. 15) ---
    k_a = max(1, int(np.floor(alpha_a * k)))
    k_b = k - k_a

    return k_a, k_b
