"""UPF-guided directional transfer mechanism (Section 3.3)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .surrogate import SurrogateSet
from .utils import constraint_violation, normalize_objectives


def compute_transfer(
    pop_b_x: NDArray[np.floating],
    upf_x: NDArray[np.floating],
    surrogates: SurrogateSet,
    f_min: NDArray[np.floating],
    f_max: NDArray[np.floating],
    beta_max: float,
) -> tuple[NDArray[np.floating], float]:
    """Compute direction vectors and adaptive transfer intensity.

    Parameters
    ----------
    pop_b_x : ndarray of shape (N, d)
        Decision variables of Task B population.
    upf_x : ndarray of shape (N_U, d)
        Decision variables of approximate UPF solutions from Task A.
    surrogates : SurrogateSet
        Trained surrogate models.
    f_min, f_max : ndarray of shape (m,)
        Objective ranges from the archive.
    beta_max : float
        Upper bound on transfer intensity.

    Returns
    -------
    directions : ndarray of shape (N, d)
        Direction vectors d_p = x_pi(p)^U - x_p.
    beta : float
        Adaptive transfer intensity.
    """
    N = pop_b_x.shape[0]

    # --- Predict and normalize objectives for both populations ---
    F_b_mu, _ = surrogates.predict_objectives(pop_b_x)
    F_u_mu, _ = surrogates.predict_objectives(upf_x)

    F_b_norm = normalize_objectives(F_b_mu, f_min, f_max)
    F_u_norm = normalize_objectives(F_u_mu, f_min, f_max)

    # --- Compute adaptive beta ---
    beta = _compute_beta(
        pop_b_x, F_b_norm, F_u_norm, surrogates, f_min, f_max, beta_max
    )

    # --- Nearest-neighbor matching in normalized objective space (Eq. 12) ---
    # For each x_p in P_B, find nearest UPF point in objective space
    directions = np.zeros_like(pop_b_x)
    for p in range(N):
        diffs = F_u_norm - F_b_norm[p]  # (N_U, m)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        pi_p = np.argmin(dists)
        directions[p] = upf_x[pi_p] - pop_b_x[p]  # Eq. (13)

    return directions, beta


def _compute_beta(
    pop_b_x: NDArray[np.floating],
    F_b_norm: NDArray[np.floating],
    F_u_norm: NDArray[np.floating],
    surrogates: SurrogateSet,
    f_min: NDArray[np.floating],
    f_max: NDArray[np.floating],
    beta_max: float,
) -> float:
    """Compute adaptive transfer intensity based on GD between fronts.

    beta = beta_max * exp(-GD^2)
    Falls back to beta_max when no feasible solution exists.
    """
    # Identify surrogate-feasible non-dominated solutions in P_B
    G_b_mu, _ = surrogates.predict_constraints(pop_b_x)
    cv = constraint_violation(G_b_mu)
    feasible_mask = cv <= 0.0

    if not np.any(feasible_mask):
        # No feasible solution: maximum transfer (paper Section 3.3)
        return beta_max

    # Non-dominated feasible solutions
    feasible_F = F_b_norm[feasible_mask]
    nd_idx = _get_nondominated_indices(feasible_F)

    if len(nd_idx) == 0:
        return beta_max

    F_c = feasible_F[nd_idx]  # Approximate CPF in normalized space

    # Generational Distance (Eq. 16)
    gd = _generational_distance(F_c, F_u_norm)

    # Task similarity (Eq. 17) and transfer intensity (Eq. 18)
    s = np.exp(-gd ** 2)
    beta = beta_max * s
    return float(beta)


def _generational_distance(
    F_c: NDArray[np.floating],
    F_u: NDArray[np.floating],
) -> float:
    """GD(F_C, F_U) = (1/|F_C|) Σ min_{f' ∈ F_U} ||f - f'||₂."""
    total = 0.0
    for f in F_c:
        dists = np.sqrt(np.sum((F_u - f) ** 2, axis=1))
        total += np.min(dists)
    return total / len(F_c)


def _get_nondominated_indices(F: NDArray[np.floating]) -> list[int]:
    """Return indices of non-dominated solutions."""
    n = F.shape[0]
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        for j in range(n):
            if i == j or not is_nd[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_nd[i] = False
                break
    return list(np.where(is_nd)[0])
