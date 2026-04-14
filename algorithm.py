"""Main CoSAEM algorithm loop (Algorithm 1 in the paper)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from .budget import allocate_budget
from .operators import reproduce_task_a, reproduce_task_b
from .selection import cdp_select, fast_non_dominated_sort, nds_select
from .surrogate import SurrogateSet, build_surrogates
from .transfer import compute_transfer
from .utils import (
    clip_to_bounds,
    constraint_violation,
    latin_hypercube_sampling,
    normalize_objectives,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Problem interface
# ---------------------------------------------------------------------------

class Problem(Protocol):
    """Interface that any optimization problem must satisfy."""

    @property
    def n_obj(self) -> int: ...
    @property
    def n_con(self) -> int: ...
    @property
    def n_var(self) -> int: ...
    @property
    def lb(self) -> NDArray[np.floating]: ...
    @property
    def ub(self) -> NDArray[np.floating]: ...

    def evaluate(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Evaluate objectives and constraints.

        Parameters
        ----------
        X : ndarray of shape (n, d)

        Returns
        -------
        F : ndarray of shape (n, m)
            Objective values.
        G : ndarray of shape (n, q)
            Constraint values (g_j <= 0 is feasible).
        """
        ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CoSAEMResult:
    """Result of a CoSAEM optimization run."""

    X: NDArray[np.floating]       # decision variables of final non-dominated feasible set
    F: NDArray[np.floating]       # objective values
    G: NDArray[np.floating]       # constraint values
    n_evals: int                  # total function evaluations used
    archive_X: NDArray[np.floating]
    archive_F: NDArray[np.floating]
    archive_G: NDArray[np.floating]
    history: list[dict]           # per-iteration metadata


# ---------------------------------------------------------------------------
# CoSAEM main algorithm
# ---------------------------------------------------------------------------

@dataclass
class CoSAEMConfig:
    """Configuration parameters for CoSAEM."""

    N: int = 100               # population size
    T_max: int = 300           # total evaluation budget
    G_inner: int = 50          # inner surrogate-assisted generations
    k: int = 5                 # evaluations per outer iteration
    F_de: float = 0.5          # DE scaling factor
    CR: float = 0.9            # crossover rate
    beta_max: float = 1.0      # transfer intensity upper bound
    w1: float = 2.0            # sigmoid weight for ρ_f
    w2: float = 2.0            # sigmoid weight for ē_obj
    w3: float = 2.0            # sigmoid weight for ē_con
    b: float = -1.0            # sigmoid bias
    gp_restarts: int = 3       # GP hyperparameter restarts
    seed: int = 42             # random seed


def run_cosam(problem: Problem, config: CoSAEMConfig | None = None) -> CoSAEMResult:
    """Execute the CoSAEM algorithm on the given problem.

    Parameters
    ----------
    problem : Problem
        The expensive constrained multiobjective optimization problem.
    config : CoSAEMConfig, optional
        Algorithm configuration. Uses defaults if not provided.

    Returns
    -------
    result : CoSAEMResult
    """
    if config is None:
        config = CoSAEMConfig()

    rng = np.random.default_rng(config.seed)
    d = problem.n_var
    m = problem.n_obj
    q = problem.n_con
    lb = np.asarray(problem.lb, dtype=np.float64)
    ub = np.asarray(problem.ub, dtype=np.float64)

    history: list[dict] = []

    # =====================================================================
    # Step 1: Initialize archive via LHS (line 1)
    # =====================================================================
    N_init = 11 * d - 1
    X_init = latin_hypercube_sampling(N_init, d, lb, ub, rng)
    F_init, G_init = problem.evaluate(X_init)
    n_evals = N_init

    archive_X = X_init.copy()
    archive_F = F_init.copy()
    archive_G = G_init.copy()

    logger.info(
        "Initialized archive with %d solutions (%d FEs used)", N_init, n_evals
    )

    # =====================================================================
    # Step 2: Initialize populations as top-N by NDS (line 2)
    # =====================================================================
    if archive_X.shape[0] <= config.N:
        sel_idx = np.arange(archive_X.shape[0])
    else:
        sel_idx = nds_select(archive_F, config.N)

    pop_a_x = archive_X[sel_idx].copy()
    pop_b_x = archive_X[sel_idx].copy()

    # =====================================================================
    # Main loop (lines 3–12)
    # =====================================================================
    iteration = 0
    while n_evals < config.T_max:
        iteration += 1
        remaining = config.T_max - n_evals
        k_iter = min(config.k, remaining)
        if k_iter <= 0:
            break

        logger.info(
            "Iteration %d: %d FEs used, %d remaining", iteration, n_evals, remaining
        )

        # -----------------------------------------------------------------
        # Lines 4–5: Build surrogates from archive
        # -----------------------------------------------------------------
        surrogates = build_surrogates(
            archive_X, archive_F, archive_G,
            n_restarts=config.gp_restarts,
            rng=rng,
        )

        # Objective ranges for normalization
        f_min = np.min(archive_F, axis=0)
        f_max = np.max(archive_F, axis=0)

        # -----------------------------------------------------------------
        # Line 6: Evolve Task A (unconstrained) for G inner generations
        # -----------------------------------------------------------------
        pop_a_x = _evolve_task_a(
            pop_a_x, surrogates, lb, ub,
            config.G_inner, config.N, config.F_de, config.CR, rng,
        )

        # Extract approximate UPF: non-dominated solutions in P_A
        F_a_pred, _ = surrogates.predict_objectives(pop_a_x)
        fronts_a = fast_non_dominated_sort(F_a_pred)
        upf_idx = fronts_a[0]
        upf_x = pop_a_x[upf_idx]

        # -----------------------------------------------------------------
        # Line 7: Directional transfer
        # -----------------------------------------------------------------
        directions, beta = compute_transfer(
            pop_b_x, upf_x, surrogates, f_min, f_max, config.beta_max
        )

        # -----------------------------------------------------------------
        # Line 8: Evolve Task B (constrained) for G inner generations
        # -----------------------------------------------------------------
        pop_b_x = _evolve_task_b(
            pop_b_x, directions, beta, surrogates, lb, ub,
            config.G_inner, config.N, config.F_de, config.CR, rng,
        )

        # -----------------------------------------------------------------
        # Line 9: Budget allocation
        # -----------------------------------------------------------------
        k_a, k_b = allocate_budget(
            pop_b_x, surrogates, archive_F, archive_G, k_iter,
            config.w1, config.w2, config.w3, config.b,
        )

        # -----------------------------------------------------------------
        # Line 10: Select infill candidates by max aggregate uncertainty
        # -----------------------------------------------------------------
        cands_a = _select_infill(pop_a_x, archive_X, surrogates, k_a)
        cands_b = _select_infill(pop_b_x, archive_X, surrogates, k_b)

        # -----------------------------------------------------------------
        # Line 11: Evaluate and merge into archive
        # -----------------------------------------------------------------
        cands_all = np.vstack([cands_a, cands_b]) if cands_b.shape[0] > 0 else cands_a
        if cands_all.shape[0] == 0:
            continue

        F_new, G_new = problem.evaluate(cands_all)
        n_evals += cands_all.shape[0]

        archive_X = np.vstack([archive_X, cands_all])
        archive_F = np.vstack([archive_F, F_new])
        archive_G = np.vstack([archive_G, G_new])

        # -----------------------------------------------------------------
        # Line 12: Update populations
        # -----------------------------------------------------------------
        # Task A: merge + NDS
        n_a = cands_a.shape[0]
        if n_a > 0:
            merged_a_x = np.vstack([pop_a_x, cands_a])
            merged_a_f, _ = surrogates.predict_objectives(merged_a_x)
            real_f_a = np.vstack([
                F_a_pred,
                F_new[:n_a],
            ])
            # Use real values where available, surrogate elsewhere
            merged_a_f[-n_a:] = F_new[:n_a]
            sel_a = nds_select(merged_a_f, config.N)
            pop_a_x = merged_a_x[sel_a]

        # Task B: merge + CDP
        n_b = cands_b.shape[0]
        if n_b > 0:
            merged_b_x = np.vstack([pop_b_x, cands_b])
            merged_b_f, _ = surrogates.predict_objectives(merged_b_x)
            merged_b_g, _ = surrogates.predict_constraints(merged_b_x)
            # Overwrite with real values for newly evaluated candidates
            merged_b_f[-n_b:] = F_new[n_a:]
            merged_b_g[-n_b:] = G_new[n_a:]
            sel_b = cdp_select(merged_b_f, merged_b_g, config.N)
            pop_b_x = merged_b_x[sel_b]

        # Record iteration info
        history.append({
            "iteration": iteration,
            "n_evals": n_evals,
            "beta": beta,
            "k_a": k_a,
            "k_b": k_b,
            "e_obj": surrogates.e_obj,
            "e_con": surrogates.e_con,
        })

    # =====================================================================
    # Lines 13–14: Extract non-dominated feasible solutions from P_B
    # =====================================================================
    F_final, G_final = problem.evaluate(pop_b_x)

    cv = constraint_violation(G_final)
    feasible_mask = cv <= 0.0

    if np.any(feasible_mask):
        feasible_F = F_final[feasible_mask]
        feasible_X = pop_b_x[feasible_mask]
        feasible_G = G_final[feasible_mask]
        fronts = fast_non_dominated_sort(feasible_F)
        nd_idx = fronts[0]
        result_X = feasible_X[nd_idx]
        result_F = feasible_F[nd_idx]
        result_G = feasible_G[nd_idx]
    else:
        # No feasible solution found — return least infeasible
        order = np.argsort(cv)
        result_X = pop_b_x[order[:1]]
        result_F = F_final[order[:1]]
        result_G = G_final[order[:1]]
        logger.warning("No feasible solution found; returning least infeasible.")

    return CoSAEMResult(
        X=result_X,
        F=result_F,
        G=result_G,
        n_evals=n_evals,
        archive_X=archive_X,
        archive_F=archive_F,
        archive_G=archive_G,
        history=history,
    )


# ---------------------------------------------------------------------------
# Helper: inner evolution for Task A
# ---------------------------------------------------------------------------

def _evolve_task_a(
    pop_x: NDArray[np.floating],
    surrogates: SurrogateSet,
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
    G: int,
    N: int,
    F_de: float,
    CR: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Surrogate-assisted inner evolution for Task A (unconstrained).

    DE/rand/1 + NDS selection for G generations.
    """
    for _ in range(G):
        offspring = reproduce_task_a(pop_x, lb, ub, F_de, CR, rng)
        combined = np.vstack([pop_x, offspring])
        F_pred, _ = surrogates.predict_objectives(combined)
        sel = nds_select(F_pred, N)
        pop_x = combined[sel]
    return pop_x


# ---------------------------------------------------------------------------
# Helper: inner evolution for Task B
# ---------------------------------------------------------------------------

def _evolve_task_b(
    pop_x: NDArray[np.floating],
    directions: NDArray[np.floating],
    beta: float,
    surrogates: SurrogateSet,
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
    G: int,
    N: int,
    F_de: float,
    CR: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Surrogate-assisted inner evolution for Task B (constrained).

    DE/current-to-UPF/1 + CDP selection for G generations.
    Directions are recomputed from the initial matching and held constant
    across inner generations; beta is fixed per outer iteration.
    """
    for _ in range(G):
        # Pad directions if pop size changed due to selection
        if directions.shape[0] != pop_x.shape[0]:
            # Reuse first N directions (wrap-around)
            idx = np.arange(pop_x.shape[0]) % directions.shape[0]
            dirs = directions[idx]
        else:
            dirs = directions

        offspring = reproduce_task_b(pop_x, dirs, beta, lb, ub, F_de, CR, rng)
        combined = np.vstack([pop_x, offspring])
        F_pred, _ = surrogates.predict_objectives(combined)
        G_pred, _ = surrogates.predict_constraints(combined)
        sel = cdp_select(F_pred, G_pred, N)
        pop_x = combined[sel]
    return pop_x


# ---------------------------------------------------------------------------
# Helper: infill selection by maximum aggregate surrogate uncertainty
# ---------------------------------------------------------------------------

def _select_infill(
    pop_x: NDArray[np.floating],
    archive_X: NDArray[np.floating],
    surrogates: SurrogateSet,
    k: int,
) -> NDArray[np.floating]:
    """Select k candidates with highest aggregate GP variance, not in archive.

    Infill criterion: max Σ_i σ̂²_i(x)
    """
    if k <= 0:
        return np.empty((0, pop_x.shape[1]))

    # Filter candidates not already in archive (by Euclidean distance)
    mask = np.ones(pop_x.shape[0], dtype=bool)
    for i in range(pop_x.shape[0]):
        dists = np.sqrt(np.sum((archive_X - pop_x[i]) ** 2, axis=1))
        if np.min(dists) < 1e-8:
            mask[i] = False

    candidates = pop_x[mask]
    if candidates.shape[0] == 0:
        # All already in archive; return random subset from pop
        idx = np.arange(pop_x.shape[0])
        np.random.shuffle(idx)
        return pop_x[idx[:k]]

    # Compute aggregate variance
    _, F_var = surrogates.predict_objectives(candidates)
    agg_var = np.sum(F_var, axis=1)

    # Select top-k
    n_select = min(k, candidates.shape[0])
    top_idx = np.argsort(-agg_var)[:n_select]
    return candidates[top_idx]
