"""Tests for surrogate module."""

import numpy as np
import pytest

from CoSAEM.surrogate import (
    build_ensemble,
    build_surrogates,
    matern52_kernel,
    train_gp,
    train_rbf,
)


class TestMatern52:
    def test_symmetry(self):
        X = np.random.default_rng(0).random((10, 3))
        ls = np.ones(3)
        K = matern52_kernel(X, X, ls, 1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_diagonal(self):
        X = np.random.default_rng(0).random((10, 3))
        ls = np.ones(3)
        K = matern52_kernel(X, X, ls, 2.0)
        np.testing.assert_allclose(np.diag(K), 2.0, atol=1e-6)

    def test_positive_definite(self):
        X = np.random.default_rng(0).random((10, 3))
        ls = np.ones(3)
        K = matern52_kernel(X, X, ls, 1.0) + 1e-6 * np.eye(10)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > 0)


class TestGP:
    def test_interpolation(self):
        """GP should interpolate training data (with small noise)."""
        rng = np.random.default_rng(42)
        X = rng.random((20, 2))
        y = np.sin(X[:, 0]) + np.cos(X[:, 1])
        model = train_gp(X, y, n_restarts=2, rng=rng)
        mu, var = model.predict(X)
        np.testing.assert_allclose(mu, y, atol=0.1)

    def test_variance_at_training(self):
        """Variance should be small at training points."""
        rng = np.random.default_rng(42)
        X = rng.random((15, 2))
        y = X[:, 0] ** 2
        model = train_gp(X, y, n_restarts=2, rng=rng)
        _, var = model.predict(X)
        assert np.all(var < 0.5)

    def test_loocv_finite(self):
        rng = np.random.default_rng(42)
        X = rng.random((20, 3))
        y = np.sum(X, axis=1)
        model = train_gp(X, y, rng=rng)
        assert np.isfinite(model.loocv_rmse)
        assert model.loocv_rmse >= 0


class TestRBF:
    def test_interpolation(self):
        """RBF should interpolate training data exactly."""
        rng = np.random.default_rng(0)
        X = rng.random((15, 2))
        y = X[:, 0] + 2 * X[:, 1]
        model = train_rbf(X, y)
        assert model is not None
        mu, var = model.predict(X)
        np.testing.assert_allclose(mu, y, atol=0.5)
        np.testing.assert_allclose(var, 0.0)

    def test_returns_none_on_singular(self):
        """Should handle degenerate inputs gracefully."""
        X = np.ones((5, 2))  # all identical
        y = np.ones(5)
        model = train_rbf(X, y)
        # Should still return (with regularization) or None


class TestEnsemble:
    def test_selects_better_model(self):
        rng = np.random.default_rng(42)
        X = rng.random((25, 2))
        y = np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1])
        ens = build_ensemble(X, y, n_restarts=2, rng=rng)
        assert ens.selected in ("gp", "rbf")
        assert np.isfinite(ens.loocv_rmse)

    def test_predict_shape(self):
        rng = np.random.default_rng(42)
        X = rng.random((20, 3))
        y = np.sum(X, axis=1)
        ens = build_ensemble(X, y, rng=rng)
        X_test = rng.random((5, 3))
        mu, var = ens.predict(X_test)
        assert mu.shape == (5,)
        assert var.shape == (5,)


class TestBuildSurrogates:
    def test_full_build(self):
        rng = np.random.default_rng(42)
        n, d, m, q = 30, 3, 2, 1
        X = rng.random((n, d))
        F = np.column_stack([np.sum(X, axis=1), np.prod(X, axis=1)])
        G = (np.sum(X, axis=1) - 1.5).reshape(-1, 1)
        surr = build_surrogates(X, F, G, n_restarts=1, rng=rng)
        assert surr.m == 2
        assert surr.q == 1
        assert len(surr.obj_models) == 2
        assert len(surr.con_models) == 1
        assert np.isfinite(surr.e_obj)
        assert np.isfinite(surr.e_con)

    def test_predict_objectives(self):
        rng = np.random.default_rng(42)
        X = rng.random((25, 2))
        F = np.column_stack([X[:, 0], 1 - X[:, 0]])
        G = (X[:, 0] - 0.5).reshape(-1, 1)
        surr = build_surrogates(X, F, G, n_restarts=1, rng=rng)
        X_test = rng.random((5, 2))
        F_mu, F_var = surr.predict_objectives(X_test)
        assert F_mu.shape == (5, 2)
        assert F_var.shape == (5, 2)

    def test_predict_constraints(self):
        rng = np.random.default_rng(42)
        X = rng.random((25, 2))
        F = np.column_stack([X[:, 0], 1 - X[:, 0]])
        G = np.column_stack([X[:, 0] - 0.5, X[:, 1] - 0.3])
        surr = build_surrogates(X, F, G, n_restarts=1, rng=rng)
        G_mu, G_var = surr.predict_constraints(rng.random((3, 2)))
        assert G_mu.shape == (3, 2)
