"""Utility functions: LHS sampling, normalization, metrics, boundary handling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def latin_hypercube_sampling(
    n: int,
    d: int,
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Generate *n* samples in *d* dimensions via maximin LHS.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of dimensions.
    lb, ub : array-like of shape (d,)
        Lower and upper bounds for each dimension.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    X : ndarray of shape (n, d)
        Sample matrix scaled to [lb, ub].
    """
    if rng is None:
        rng = np.random.default_rng()
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)

    # Standard LHS: one random point per stratum in each dimension
    X = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n

    # Scale to [lb, ub]
    X = lb + X * (ub - lb)
    return X


# ---------------------------------------------------------------------------
# Objective normalization
# ---------------------------------------------------------------------------

def normalize_objectives(
    F: NDArray[np.floating],
    f_min: NDArray[np.floating],
    f_max: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Normalize objective values to [0, 1] using archive ranges.

    Parameters
    ----------
    F : ndarray of shape (n, m)
        Objective values.
    f_min, f_max : ndarray of shape (m,)
        Per-objective min/max from the archive.

    Returns
    -------
    F_norm : ndarray of shape (n, m)
        Normalized objective values.
    """
    ranges = f_max - f_min
    ranges = np.where(ranges < 1e-12, 1.0, ranges)  # avoid division by zero
    return (F - f_min) / ranges


# ---------------------------------------------------------------------------
# Boundary handling
# ---------------------------------------------------------------------------

def clip_to_bounds(
    X: NDArray[np.floating],
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Clip decision variables to [lb, ub]."""
    return np.clip(X, lb, ub)


# ---------------------------------------------------------------------------
# Constraint violation
# ---------------------------------------------------------------------------

def constraint_violation(G: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute total constraint violation for each solution.

    Parameters
    ----------
    G : ndarray of shape (n, q)
        Constraint values. g_j(x) <= 0 means feasible.

    Returns
    -------
    cv : ndarray of shape (n,)
        Total constraint violation (sum of max(0, g_j)).
    """
    return np.sum(np.maximum(0.0, G), axis=1)


def is_feasible(G: NDArray[np.floating], tol: float = 0.0) -> NDArray[np.bool_]:
    """Return boolean mask of feasible solutions."""
    return constraint_violation(G) <= tol


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def igd_plus(
    approx: NDArray[np.floating],
    reference: NDArray[np.floating],
) -> float:
    """Compute the Inverted Generational Distance Plus (IGD+).

    Parameters
    ----------
    approx : ndarray of shape (n, m)
        Obtained non-dominated feasible solution set in objective space.
    reference : ndarray of shape (|R|, m)
        Reference point set sampled from the true CPF.

    Returns
    -------
    igdp : float
        IGD+ value (lower is better).
    """
    if approx.shape[0] == 0:
        return np.inf
    # For each reference point, find the modified distance to the closest approx point
    # d+(r, a) = sqrt(sum_i max(a_i - r_i, 0)^2)
    # IGD+ = (1/|R|) * sum_r min_a d+(r, a)
    dists = np.zeros(reference.shape[0])
    for i, r in enumerate(reference):
        diff = np.maximum(approx - r, 0.0)  # (n, m)
        d_plus = np.sqrt(np.sum(diff ** 2, axis=1))  # (n,)
        dists[i] = np.min(d_plus)
    return float(np.mean(dists))


def hypervolume_2d(
    approx: NDArray[np.floating],
    ref_point: NDArray[np.floating],
) -> float:
    """Compute exact hypervolume for bi-objective problems.

    Parameters
    ----------
    approx : ndarray of shape (n, 2)
        Non-dominated set in objective space.
    ref_point : ndarray of shape (2,)
        Reference point (should dominate all approx points component-wise).

    Returns
    -------
    hv : float
        Hypervolume indicator.
    """
    if approx.shape[0] == 0:
        return 0.0
    # Filter points dominated by ref_point
    mask = np.all(approx < ref_point, axis=1)
    pts = approx[mask]
    if pts.shape[0] == 0:
        return 0.0
    # Sort by first objective
    idx = np.argsort(pts[:, 0])
    pts = pts[idx]
    hv = 0.0
    prev_f2 = ref_point[1]
    for p in pts:
        if p[1] < prev_f2:
            hv += (ref_point[0] - p[0]) * (prev_f2 - p[1])
            prev_f2 = p[1]
    return float(hv)
