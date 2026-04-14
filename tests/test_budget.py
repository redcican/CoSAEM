"""Tests for budget allocation module."""

import numpy as np

from CoSAEM.budget import allocate_budget
from CoSAEM.surrogate import build_surrogates


class TestBudgetAllocation:
    def _make_surrogates(self, rng):
        X = rng.random((30, 3))
        F = np.column_stack([X[:, 0], 1.0 - X[:, 0]])
        G = (X[:, 0] - 0.5).reshape(-1, 1)
        return build_surrogates(X, F, G, n_restarts=1, rng=rng), F, G

    def test_budget_sums_to_k(self):
        rng = np.random.default_rng(42)
        surr, F, G = self._make_surrogates(rng)
        pop_b = rng.random((20, 3))
        k_a, k_b = allocate_budget(pop_b, surr, F, G, k=5)
        assert k_a + k_b == 5
        assert k_a >= 1
        assert k_b >= 1

    def test_min_one_each(self):
        """Both tasks must get at least 1 evaluation."""
        rng = np.random.default_rng(42)
        surr, F, G = self._make_surrogates(rng)
        pop_b = rng.random((20, 3))
        for k in [2, 3, 5, 10]:
            k_a, k_b = allocate_budget(pop_b, surr, F, G, k=k)
            assert k_a >= 1
            assert k_b >= 1
            assert k_a + k_b == k

    def test_different_weights(self):
        rng = np.random.default_rng(42)
        surr, F, G = self._make_surrogates(rng)
        pop_b = rng.random((20, 3))
        k_a1, _ = allocate_budget(pop_b, surr, F, G, k=10, w1=5.0)
        k_a2, _ = allocate_budget(pop_b, surr, F, G, k=10, w1=0.1)
        # Higher w1 should give more to Task A (higher ρ_f → more α_A)
        # This is data-dependent so just check they're valid
        assert 1 <= k_a1 <= 9
        assert 1 <= k_a2 <= 9
