"""Tests for utils module."""

import numpy as np
import pytest

from CoSAEM.utils import (
    clip_to_bounds,
    constraint_violation,
    hypervolume_2d,
    igd_plus,
    is_feasible,
    latin_hypercube_sampling,
    normalize_objectives,
)


class TestLHS:
    def test_shape(self):
        X = latin_hypercube_sampling(50, 5, np.zeros(5), np.ones(5))
        assert X.shape == (50, 5)

    def test_bounds(self):
        lb = np.array([-1.0, 0.0, 2.0])
        ub = np.array([1.0, 5.0, 10.0])
        X = latin_hypercube_sampling(100, 3, lb, ub)
        assert np.all(X >= lb)
        assert np.all(X <= ub)

    def test_reproducibility(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        X1 = latin_hypercube_sampling(20, 3, np.zeros(3), np.ones(3), rng1)
        X2 = latin_hypercube_sampling(20, 3, np.zeros(3), np.ones(3), rng2)
        np.testing.assert_array_equal(X1, X2)


class TestNormalize:
    def test_basic(self):
        F = np.array([[1.0, 2.0], [3.0, 4.0]])
        f_min = np.array([1.0, 2.0])
        f_max = np.array([3.0, 4.0])
        F_norm = normalize_objectives(F, f_min, f_max)
        np.testing.assert_allclose(F_norm[0], [0.0, 0.0])
        np.testing.assert_allclose(F_norm[1], [1.0, 1.0])

    def test_zero_range(self):
        F = np.array([[5.0], [5.0]])
        result = normalize_objectives(F, np.array([5.0]), np.array([5.0]))
        assert np.all(np.isfinite(result))


class TestConstraintViolation:
    def test_feasible(self):
        G = np.array([[-1.0, -2.0], [-0.5, -0.1]])
        cv = constraint_violation(G)
        np.testing.assert_allclose(cv, [0.0, 0.0])

    def test_infeasible(self):
        G = np.array([[1.0, -1.0], [2.0, 3.0]])
        cv = constraint_violation(G)
        np.testing.assert_allclose(cv, [1.0, 5.0])

    def test_is_feasible(self):
        G = np.array([[-1.0, -2.0], [0.5, -0.1]])
        mask = is_feasible(G)
        assert mask[0] and not mask[1]


class TestIGDPlus:
    def test_perfect(self):
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        approx = ref.copy()
        assert igd_plus(approx, ref) == pytest.approx(0.0, abs=1e-10)

    def test_empty_approx(self):
        ref = np.array([[0.0, 1.0]])
        approx = np.empty((0, 2))
        assert igd_plus(approx, ref) == np.inf

    def test_positive(self):
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        approx = np.array([[0.5, 0.5]])
        val = igd_plus(approx, ref)
        assert val > 0.0


class TestHV2D:
    def test_single_point(self):
        approx = np.array([[1.0, 1.0]])
        ref = np.array([2.0, 2.0])
        assert hypervolume_2d(approx, ref) == pytest.approx(1.0)

    def test_two_points(self):
        approx = np.array([[0.0, 1.0], [1.0, 0.0]])
        ref = np.array([2.0, 2.0])
        hv = hypervolume_2d(approx, ref)
        assert hv == pytest.approx(3.0)

    def test_empty(self):
        approx = np.empty((0, 2))
        ref = np.array([2.0, 2.0])
        assert hypervolume_2d(approx, ref) == 0.0
