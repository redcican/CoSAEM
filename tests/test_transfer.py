"""Tests for transfer module."""

import numpy as np

from CoSAEM.surrogate import build_surrogates
from CoSAEM.transfer import compute_transfer


class TestTransfer:
    def _make_surrogates(self, rng):
        X = rng.random((30, 3))
        F = np.column_stack([X[:, 0], 1.0 - X[:, 0]])
        G = (X[:, 0] - 0.5).reshape(-1, 1)
        return build_surrogates(X, F, G, n_restarts=1, rng=rng)

    def test_directions_shape(self):
        rng = np.random.default_rng(42)
        surr = self._make_surrogates(rng)
        pop_b = rng.random((10, 3))
        upf_x = rng.random((5, 3))
        f_min = np.array([0.0, 0.0])
        f_max = np.array([1.0, 1.0])
        dirs, beta = compute_transfer(pop_b, upf_x, surr, f_min, f_max, 1.0)
        assert dirs.shape == (10, 3)
        assert 0.0 <= beta <= 1.0

    def test_beta_max_when_infeasible(self):
        """When no feasible solution, beta should be beta_max."""
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        F = np.column_stack([X[:, 0], 1.0 - X[:, 0]])
        # All constraints heavily violated
        G = np.ones((30, 1)) * 10.0
        surr = build_surrogates(X, F, G, n_restarts=1, rng=rng)
        pop_b = rng.random((10, 3))
        upf_x = rng.random((5, 3))
        f_min = np.array([0.0, 0.0])
        f_max = np.array([1.0, 1.0])
        _, beta = compute_transfer(pop_b, upf_x, surr, f_min, f_max, 1.0)
        assert beta == 1.0

    def test_beta_bounded(self):
        rng = np.random.default_rng(42)
        surr = self._make_surrogates(rng)
        pop_b = rng.random((15, 3))
        upf_x = rng.random((8, 3))
        f_min = np.array([0.0, 0.0])
        f_max = np.array([1.0, 1.0])
        _, beta = compute_transfer(pop_b, upf_x, surr, f_min, f_max, 0.8)
        assert 0.0 <= beta <= 0.8
