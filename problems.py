"""Test problem definitions for validating CoSAEM."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MW4:
    """MW4 test problem (Ma & Wang, 2019).

    Bi-objective with 2 constraints, d=15 decision variables.
    Minimization of both objectives; g_j(x) <= 0 is feasible.
    """

    def __init__(self, n_var: int = 15):
        self._n_var = n_var

    @property
    def n_obj(self) -> int:
        return 2

    @property
    def n_con(self) -> int:
        return 2

    @property
    def n_var(self) -> int:
        return self._n_var

    @property
    def lb(self) -> NDArray[np.floating]:
        return np.zeros(self._n_var)

    @property
    def ub(self) -> NDArray[np.floating]:
        return np.ones(self._n_var)

    def evaluate(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        X = np.atleast_2d(X)
        n, d = X.shape
        F = np.zeros((n, 2))
        G = np.zeros((n, 2))

        # Objectives
        g = 1.0 + np.sum(
            2.0 * (X[:, 2:] + (X[:, 1:2] - 0.5) ** 2 - 1.0) ** 2,
            axis=1,
        )
        F[:, 0] = g * X[:, 0]
        F[:, 1] = g * (1.0 - X[:, 0] ** 2)

        # Constraints
        l = np.sqrt(2.0) * F[:, 1]
        G[:, 0] = F[:, 0] ** 2 / (1.0 + 0.1 * np.cos(16.0 * np.arctan(F[:, 0] / (l + 1e-30)))) - F[:, 0] * l - 1.0
        G[:, 1] = (F[:, 0] - l) ** 2 + F[:, 0] * l - np.sqrt(2.0) * (F[:, 0] + l) + 1.0

        return F, G


class C1DTLZ1:
    """C1-DTLZ1 test problem (Jain & Deb, 2014).

    m-objective DTLZ1 with one constraint.
    """

    def __init__(self, n_var: int = 6, n_obj: int = 2):
        self._n_var = n_var
        self._n_obj = n_obj

    @property
    def n_obj(self) -> int:
        return self._n_obj

    @property
    def n_con(self) -> int:
        return 1

    @property
    def n_var(self) -> int:
        return self._n_var

    @property
    def lb(self) -> NDArray[np.floating]:
        return np.zeros(self._n_var)

    @property
    def ub(self) -> NDArray[np.floating]:
        return np.ones(self._n_var)

    def evaluate(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        X = np.atleast_2d(X)
        n, d = X.shape
        m = self._n_obj
        k = d - m + 1  # number of distance variables

        # Distance parameter g
        x_m = X[:, m - 1:]
        g = 100 * (
            k
            + np.sum(
                (x_m - 0.5) ** 2 - np.cos(20.0 * np.pi * (x_m - 0.5)),
                axis=1,
            )
        )

        # Objectives (DTLZ1)
        F = np.zeros((n, m))
        for i in range(m):
            F[:, i] = 0.5 * (1.0 + g)
            for j in range(m - 1 - i):
                F[:, i] *= X[:, j]
            if i > 0:
                F[:, i] *= 1.0 - X[:, m - 1 - i]

        # Constraint: sum of f_i / 0.5 - 1 >= c (hyperplane constraint)
        # C1-DTLZ1: g1 = 1 - f_m / 0.6 - sum_{i=1}^{m-1} f_i / 0.5 <= 0
        G = np.zeros((n, 1))
        G[:, 0] = 1.0 - F[:, -1] / 0.6 - np.sum(F[:, :-1] / 0.5, axis=1)

        return F, G


class ZDT1Constrained:
    """Simple constrained bi-objective problem for quick testing.

    ZDT1 with a single circular constraint.
    """

    def __init__(self, n_var: int = 10):
        self._n_var = n_var

    @property
    def n_obj(self) -> int:
        return 2

    @property
    def n_con(self) -> int:
        return 1

    @property
    def n_var(self) -> int:
        return self._n_var

    @property
    def lb(self) -> NDArray[np.floating]:
        return np.zeros(self._n_var)

    @property
    def ub(self) -> NDArray[np.floating]:
        return np.ones(self._n_var)

    def evaluate(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        X = np.atleast_2d(X)
        n, d = X.shape
        F = np.zeros((n, 2))
        G = np.zeros((n, 1))

        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))

        F[:, 0] = f1
        F[:, 1] = f2

        # Constraint: exclude a circular region
        G[:, 0] = 0.5 - (f1 - 0.5) ** 2 - (f2 - 0.5) ** 2

        return F, G
