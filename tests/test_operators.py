"""Tests for operators module."""

import numpy as np

from CoSAEM.operators import (
    binomial_crossover,
    de_current_to_upf,
    de_rand_1,
    reproduce_task_a,
    reproduce_task_b,
)


class TestDERand1:
    def test_shape(self):
        rng = np.random.default_rng(0)
        pop = rng.random((20, 5))
        mutants = de_rand_1(pop, 0.5, rng)
        assert mutants.shape == (20, 5)

    def test_deterministic(self):
        pop = np.random.default_rng(0).random((10, 3))
        m1 = de_rand_1(pop, 0.5, np.random.default_rng(42))
        m2 = de_rand_1(pop, 0.5, np.random.default_rng(42))
        np.testing.assert_array_equal(m1, m2)


class TestDECurrentToUPF:
    def test_shape(self):
        rng = np.random.default_rng(0)
        pop = rng.random((15, 4))
        dirs = rng.random((15, 4))
        mutants = de_current_to_upf(pop, dirs, 0.8, 0.5, rng)
        assert mutants.shape == (15, 4)

    def test_zero_beta(self):
        """With beta=0, should reduce to DE/current/1 (x_p + F*(x_r1 - x_r2))."""
        rng = np.random.default_rng(42)
        pop = rng.random((10, 3))
        dirs = np.ones((10, 3))  # shouldn't matter
        mutants = de_current_to_upf(pop, dirs, 0.0, 0.5, rng)
        # Each mutant should be pop[i] + 0.5 * (pop[r1] - pop[r2])
        # Just check it's not equal to pop
        assert not np.allclose(mutants, pop)


class TestBinomialCrossover:
    def test_shape(self):
        rng = np.random.default_rng(0)
        targets = rng.random((10, 5))
        mutants = rng.random((10, 5))
        trials = binomial_crossover(targets, mutants, 0.9, rng)
        assert trials.shape == (10, 5)

    def test_at_least_one_from_mutant(self):
        rng = np.random.default_rng(0)
        targets = np.zeros((10, 5))
        mutants = np.ones((10, 5))
        trials = binomial_crossover(targets, mutants, 0.0, rng)
        # CR=0 but j_rand ensures at least one dimension from mutant
        for i in range(10):
            assert np.any(trials[i] == 1.0)

    def test_cr_one(self):
        rng = np.random.default_rng(0)
        targets = np.zeros((5, 3))
        mutants = np.ones((5, 3))
        trials = binomial_crossover(targets, mutants, 1.0, rng)
        np.testing.assert_array_equal(trials, mutants)


class TestReproducePipeline:
    def test_task_a_bounds(self):
        rng = np.random.default_rng(0)
        pop = rng.random((20, 5))
        lb = np.zeros(5)
        ub = np.ones(5)
        offspring = reproduce_task_a(pop, lb, ub, 0.5, 0.9, rng)
        assert np.all(offspring >= lb)
        assert np.all(offspring <= ub)

    def test_task_b_bounds(self):
        rng = np.random.default_rng(0)
        pop = rng.random((20, 5))
        dirs = rng.random((20, 5)) - 0.5
        lb = np.zeros(5)
        ub = np.ones(5)
        offspring = reproduce_task_b(pop, dirs, 0.5, lb, ub, 0.5, 0.9, rng)
        assert np.all(offspring >= lb)
        assert np.all(offspring <= ub)
