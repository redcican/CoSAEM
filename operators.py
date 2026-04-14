"""Differential evolution operators: mutation, crossover, boundary repair."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .utils import clip_to_bounds


# ---------------------------------------------------------------------------
# DE/rand/1 mutation (Task A)
# ---------------------------------------------------------------------------

def de_rand_1(
    population: NDArray[np.floating],
    F: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """DE/rand/1 mutation: v_i = x_r1 + F * (x_r2 - x_r3).

    Parameters
    ----------
    population : ndarray of shape (N, d)
        Current population.
    F : float
        Scaling factor.
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    mutants : ndarray of shape (N, d)
        Mutant vectors.
    """
    N, d = population.shape
    mutants = np.empty_like(population)

    for i in range(N):
        candidates = list(range(N))
        candidates.remove(i)
        r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
        mutants[i] = population[r1] + F * (population[r2] - population[r3])

    return mutants


# ---------------------------------------------------------------------------
# DE/current-to-UPF/1 mutation (Task B with directional transfer)
# ---------------------------------------------------------------------------

def de_current_to_upf(
    population: NDArray[np.floating],
    directions: NDArray[np.floating],
    beta: float,
    F: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """DE/current-to-UPF/1: v_p = x_p + beta * d_p + F * (x_r1 - x_r2).

    Parameters
    ----------
    population : ndarray of shape (N, d)
        Current Task B population.
    directions : ndarray of shape (N, d)
        Direction vectors d_p = x_pi(p)^U - x_p.
    beta : float
        Transfer intensity.
    F : float
        DE scaling factor.
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    mutants : ndarray of shape (N, d)
        Mutant vectors.
    """
    N, d = population.shape
    mutants = np.empty_like(population)

    for i in range(N):
        candidates = list(range(N))
        candidates.remove(i)
        r1, r2 = rng.choice(candidates, size=2, replace=False)
        mutants[i] = (
            population[i]
            + beta * directions[i]
            + F * (population[r1] - population[r2])
        )

    return mutants


# ---------------------------------------------------------------------------
# Binomial crossover
# ---------------------------------------------------------------------------

def binomial_crossover(
    targets: NDArray[np.floating],
    mutants: NDArray[np.floating],
    CR: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Binomial crossover producing trial vectors.

    Parameters
    ----------
    targets : ndarray of shape (N, d)
        Target (parent) vectors.
    mutants : ndarray of shape (N, d)
        Mutant vectors.
    CR : float
        Crossover rate in [0, 1].
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    trials : ndarray of shape (N, d)
        Trial vectors.
    """
    N, d = targets.shape
    trials = targets.copy()

    # Random mask
    rand_mask = rng.random((N, d)) <= CR
    # Ensure at least one dimension from mutant
    j_rand = rng.integers(0, d, size=N)
    for i in range(N):
        rand_mask[i, j_rand[i]] = True

    trials[rand_mask] = mutants[rand_mask]
    return trials


# ---------------------------------------------------------------------------
# Full reproduction pipeline
# ---------------------------------------------------------------------------

def reproduce_task_a(
    population: NDArray[np.floating],
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
    F: float,
    CR: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Generate offspring for Task A using DE/rand/1 + binomial crossover."""
    mutants = de_rand_1(population, F, rng)
    trials = binomial_crossover(population, mutants, CR, rng)
    return clip_to_bounds(trials, lb, ub)


def reproduce_task_b(
    population: NDArray[np.floating],
    directions: NDArray[np.floating],
    beta: float,
    lb: NDArray[np.floating],
    ub: NDArray[np.floating],
    F: float,
    CR: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Generate offspring for Task B using DE/current-to-UPF/1 + binomial crossover."""
    mutants = de_current_to_upf(population, directions, beta, F, rng)
    trials = binomial_crossover(population, mutants, CR, rng)
    return clip_to_bounds(trials, lb, ub)
