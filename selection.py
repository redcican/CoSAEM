"""Selection operators: fast non-dominated sorting, crowding distance, CDP."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .utils import constraint_violation


# ---------------------------------------------------------------------------
# Fast non-dominated sorting (Deb et al., 2002)
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(F: NDArray[np.floating]) -> list[list[int]]:
    """Partition solutions into non-domination fronts.

    Parameters
    ----------
    F : ndarray of shape (n, m)
        Objective values (minimization assumed).

    Returns
    -------
    fronts : list of lists
        ``fronts[k]`` contains the indices of solutions on front *k*
        (0-indexed; front 0 is the best).
    """
    n = F.shape[0]
    domination_count = np.zeros(n, dtype=int)
    dominated_set: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(F[i], F[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _dominates(F[j], F[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

    fronts: list[list[int]] = []
    current_front = [i for i in range(n) if domination_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for i in current_front:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def _dominates(a: NDArray[np.floating], b: NDArray[np.floating]) -> bool:
    """Return True if *a* Pareto-dominates *b* (all <=, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


# ---------------------------------------------------------------------------
# Crowding distance
# ---------------------------------------------------------------------------

def crowding_distance(F: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute crowding distance for a set of solutions.

    Parameters
    ----------
    F : ndarray of shape (n, m)
        Objective values of solutions on the same front.

    Returns
    -------
    dist : ndarray of shape (n,)
        Crowding distances (higher = more isolated = preferred).
    """
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n, dtype=np.float64)
    for j in range(m):
        order = np.argsort(F[:, j])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        obj_range = F[order[-1], j] - F[order[0], j]
        if obj_range < 1e-12:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (F[order[k + 1], j] - F[order[k - 1], j]) / obj_range
    return dist


# ---------------------------------------------------------------------------
# Environmental selection by NDS + crowding distance
# ---------------------------------------------------------------------------

def nds_select(
    F: NDArray[np.floating],
    N: int,
) -> NDArray[np.intp]:
    """Select *N* individuals using non-dominated sorting + crowding distance.

    Parameters
    ----------
    F : ndarray of shape (n, m)
        Objective values of the combined population.
    N : int
        Number of individuals to retain.

    Returns
    -------
    selected : ndarray of shape (N,)
        Indices of selected individuals.
    """
    fronts = fast_non_dominated_sort(F)
    selected: list[int] = []

    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            # Fill remaining slots by crowding distance (descending)
            remaining = N - len(selected)
            cd = crowding_distance(F[front])
            order = np.argsort(-cd)
            selected.extend(np.array(front)[order[:remaining]].tolist())
            break

    return np.array(selected, dtype=np.intp)


# ---------------------------------------------------------------------------
# Constrained Domination Principle (CDP) selection
# ---------------------------------------------------------------------------

def cdp_select(
    F: NDArray[np.floating],
    G: NDArray[np.floating],
    N: int,
) -> NDArray[np.intp]:
    """Select *N* individuals using the Constrained Domination Principle.

    CDP ranking:
    1. Feasible solutions dominate infeasible ones.
    2. Among feasible solutions, use Pareto dominance.
    3. Among infeasible solutions, prefer lower constraint violation.

    Parameters
    ----------
    F : ndarray of shape (n, m)
        Objective values.
    G : ndarray of shape (n, q)
        Constraint values (g_j <= 0 is feasible).
    N : int
        Number to select.

    Returns
    -------
    selected : ndarray of shape (N,)
        Indices of selected individuals.
    """
    n = F.shape[0]
    cv = constraint_violation(G)
    feasible_mask = cv <= 0.0

    feasible_idx = np.where(feasible_mask)[0]
    infeasible_idx = np.where(~feasible_mask)[0]

    selected: list[int] = []

    # Phase 1: Select from feasible solutions using NDS + crowding distance
    if len(feasible_idx) > 0:
        if len(feasible_idx) <= N:
            selected.extend(feasible_idx.tolist())
        else:
            sub_sel = nds_select(F[feasible_idx], N)
            selected.extend(feasible_idx[sub_sel].tolist())

    # Phase 2: If more slots remain, fill from infeasible by ascending CV
    if len(selected) < N and len(infeasible_idx) > 0:
        remaining = N - len(selected)
        cv_inf = cv[infeasible_idx]
        order = np.argsort(cv_inf)
        selected.extend(infeasible_idx[order[:remaining]].tolist())

    return np.array(selected[:N], dtype=np.intp)
