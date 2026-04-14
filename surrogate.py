"""Surrogate models: GP with Matérn-5/2 + cubic RBF ensemble with LOOCV."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# Matérn-5/2 kernel
# ---------------------------------------------------------------------------

def matern52_kernel(
    X1: NDArray[np.floating],
    X2: NDArray[np.floating],
    length_scales: NDArray[np.floating],
    variance: float,
) -> NDArray[np.floating]:
    """Matérn-5/2 kernel with ARD length scales.

    k(x, x') = σ² (1 + √5 r + 5/3 r²) exp(-√5 r)
    where r = ||( x - x') / l||₂.
    """
    # Scale inputs
    X1s = X1 / length_scales
    X2s = X2 / length_scales
    # Pairwise distances
    diff = X1s[:, np.newaxis, :] - X2s[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    sqrt5_r = np.sqrt(5.0) * r
    K = variance * (1.0 + sqrt5_r + 5.0 / 3.0 * r ** 2) * np.exp(-sqrt5_r)
    return K


# ---------------------------------------------------------------------------
# Gaussian Process model
# ---------------------------------------------------------------------------

@dataclass
class GPModel:
    """Trained GP model with Matérn-5/2 kernel."""

    X_train: NDArray[np.floating]
    y_train: NDArray[np.floating]
    y_mean: float
    length_scales: NDArray[np.floating]
    variance: float
    noise: float
    L: NDArray[np.floating]        # Cholesky factor of K + σ²I
    alpha: NDArray[np.floating]    # K_inv @ (y - μ)
    K_inv_diag: NDArray[np.floating]  # diagonal of K_inv for LOOCV
    loocv_rmse: float

    def predict(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Predict mean and variance at new points.

        Returns
        -------
        mu : ndarray of shape (n,)
        var : ndarray of shape (n,)
        """
        K_star = matern52_kernel(X, self.X_train, self.length_scales, self.variance)
        mu = K_star @ self.alpha + self.y_mean

        # Variance: k(x*, x*) - k*^T K^{-1} k*
        v = solve_triangular(self.L, K_star.T, lower=True)
        k_ss = self.variance  # diagonal of k(x*, x*)
        var = np.maximum(k_ss - np.sum(v ** 2, axis=0), 1e-10)
        return mu, var


def train_gp(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    n_restarts: int = 3,
    rng: np.random.Generator | None = None,
) -> GPModel:
    """Train a GP model with Matérn-5/2 kernel via marginal likelihood optimization.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    y : ndarray of shape (n,)
    n_restarts : int
        Number of random restarts for hyperparameter optimization.
    rng : numpy Generator

    Returns
    -------
    model : GPModel
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = X.shape
    y_mean = np.mean(y)
    y_centered = y - y_mean

    best_nlml = np.inf
    best_params = None

    for _ in range(n_restarts):
        # Initial hyperparameters (log-space)
        log_ls = rng.uniform(-1.0, 1.0, size=d)
        log_var = rng.uniform(-1.0, 1.0)
        log_noise = np.log(1e-4 + rng.uniform(0, 0.01))
        theta0 = np.concatenate([log_ls, [log_var, log_noise]])

        try:
            result = minimize(
                _neg_log_marginal_likelihood,
                theta0,
                args=(X, y_centered),
                method="L-BFGS-B",
                bounds=[(-5.0, 5.0)] * d + [(-5.0, 5.0), (-10.0, 0.0)],
                options={"maxiter": 100},
            )
            if result.fun < best_nlml:
                best_nlml = result.fun
                best_params = result.x
        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_params is None:
        # Fallback: use default hyperparameters
        best_params = np.concatenate([np.zeros(d), [0.0, np.log(1e-3)]])

    length_scales = np.exp(best_params[:d])
    variance = np.exp(best_params[d])
    noise = np.exp(best_params[d + 1])

    K = matern52_kernel(X, X, length_scales, variance)
    K_tilde = K + noise * np.eye(n)

    try:
        L = np.linalg.cholesky(K_tilde)
    except np.linalg.LinAlgError:
        K_tilde += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K_tilde)

    alpha = cho_solve((L, True), y_centered)

    # LOOCV RMSE (closed-form)
    K_inv = cho_solve((L, True), np.eye(n))
    K_inv_diag = np.diag(K_inv)
    loocv_errors = alpha / np.maximum(K_inv_diag, 1e-12)
    loocv_rmse = float(np.sqrt(np.mean(loocv_errors ** 2)))

    return GPModel(
        X_train=X,
        y_train=y,
        y_mean=y_mean,
        length_scales=length_scales,
        variance=variance,
        noise=noise,
        L=L,
        alpha=alpha,
        K_inv_diag=K_inv_diag,
        loocv_rmse=loocv_rmse,
    )


def _neg_log_marginal_likelihood(
    theta: NDArray[np.floating],
    X: NDArray[np.floating],
    y: NDArray[np.floating],
) -> float:
    """Negative log marginal likelihood for GP hyperparameter optimization."""
    n, d = X.shape
    length_scales = np.exp(theta[:d])
    variance = np.exp(theta[d])
    noise = np.exp(theta[d + 1])

    K = matern52_kernel(X, X, length_scales, variance)
    K_tilde = K + noise * np.eye(n)

    try:
        L = np.linalg.cholesky(K_tilde)
    except np.linalg.LinAlgError:
        return 1e10

    alpha = cho_solve((L, True), y)
    nlml = (
        0.5 * y @ alpha
        + np.sum(np.log(np.diag(L)))
        + 0.5 * n * np.log(2 * np.pi)
    )
    return float(nlml)


# ---------------------------------------------------------------------------
# Cubic RBF model
# ---------------------------------------------------------------------------

@dataclass
class RBFModel:
    """Trained cubic RBF interpolant: φ(r) = r³."""

    X_train: NDArray[np.floating]
    y_train: NDArray[np.floating]
    weights: NDArray[np.floating]
    poly_coeffs: NDArray[np.floating]
    loocv_rmse: float

    def predict(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Predict mean at new points. Variance is not available (returns zeros).

        Returns
        -------
        mu : ndarray of shape (n,)
        var : ndarray of shape (n,)  — always zeros
        """
        Phi = _cubic_rbf_matrix(X, self.X_train)
        P = _polynomial_matrix(X)
        mu = Phi @ self.weights + P @ self.poly_coeffs
        var = np.zeros(X.shape[0])
        return mu, var


def train_rbf(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
) -> RBFModel | None:
    """Train a cubic RBF interpolant with linear polynomial tail.

    Returns None if the system is singular.
    """
    n, d = X.shape
    Phi = _cubic_rbf_matrix(X, X)
    P = _polynomial_matrix(X)  # (n, d+1)
    p = d + 1

    # Augmented system: [Phi P; P^T 0] [w; c] = [y; 0]
    A = np.zeros((n + p, n + p))
    A[:n, :n] = Phi
    A[:n, n:] = P
    A[n:, :n] = P.T

    rhs = np.zeros(n + p)
    rhs[:n] = y

    try:
        sol = np.linalg.solve(A + 1e-10 * np.eye(n + p), rhs)
    except np.linalg.LinAlgError:
        return None

    weights = sol[:n]
    poly_coeffs = sol[n:]

    # LOOCV via leave-one-out (approximated using the inverse diagonal)
    try:
        A_reg = A + 1e-10 * np.eye(n + p)
        A_inv = np.linalg.inv(A_reg)
        A_inv_diag = np.diag(A_inv)[:n]
        loocv_errors = sol[:n] / np.maximum(np.abs(A_inv_diag), 1e-12)
        # This is an approximation; for the RBF system the LOO formula
        # is w_i / A_inv[i,i]
        loocv_rmse = float(np.sqrt(np.mean(loocv_errors ** 2)))
    except np.linalg.LinAlgError:
        loocv_rmse = np.inf

    return RBFModel(
        X_train=X,
        y_train=y,
        weights=weights,
        poly_coeffs=poly_coeffs,
        loocv_rmse=loocv_rmse,
    )


def _cubic_rbf_matrix(
    X1: NDArray[np.floating], X2: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute cubic RBF kernel matrix: φ(r) = r³."""
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    return r ** 3


def _polynomial_matrix(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """Linear polynomial tail: [1, x_1, ..., x_d]."""
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])


# ---------------------------------------------------------------------------
# Ensemble surrogate (GP + RBF model selection)
# ---------------------------------------------------------------------------

@dataclass
class EnsembleSurrogate:
    """GP+RBF ensemble: selects the model with lower LOOCV RMSE."""

    gp: GPModel
    rbf: RBFModel | None
    selected: str  # "gp" or "rbf"
    loocv_rmse: float

    def predict(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Predict using the selected model. GP variance used as fallback."""
        if self.selected == "rbf" and self.rbf is not None:
            mu, _ = self.rbf.predict(X)
            # GP variance as fallback (paper Section 3.2)
            _, var = self.gp.predict(X)
            return mu, var
        else:
            return self.gp.predict(X)


def build_ensemble(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    n_restarts: int = 3,
    rng: np.random.Generator | None = None,
) -> EnsembleSurrogate:
    """Build a GP+RBF ensemble for one response.

    Trains both GP and RBF, selects the model with lower LOOCV RMSE.
    """
    gp = train_gp(X, y, n_restarts=n_restarts, rng=rng)
    rbf = train_rbf(X, y)

    if rbf is not None and rbf.loocv_rmse < gp.loocv_rmse:
        selected = "rbf"
        loocv_rmse = rbf.loocv_rmse
    else:
        selected = "gp"
        loocv_rmse = gp.loocv_rmse

    return EnsembleSurrogate(
        gp=gp, rbf=rbf, selected=selected, loocv_rmse=loocv_rmse
    )


# ---------------------------------------------------------------------------
# Build all surrogates for an iteration
# ---------------------------------------------------------------------------

@dataclass
class SurrogateSet:
    """Collection of surrogates for all objectives and constraints."""

    obj_models: list[EnsembleSurrogate]
    con_models: list[EnsembleSurrogate]
    e_obj: float   # mean LOOCV RMSE across objectives
    e_con: float   # mean LOOCV RMSE across constraints
    m: int
    q: int

    def predict_objectives(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Predict all objectives.

        Returns
        -------
        F_mu : ndarray of shape (n, m)
        F_var : ndarray of shape (n, m)
        """
        n = X.shape[0]
        F_mu = np.empty((n, self.m))
        F_var = np.empty((n, self.m))
        for i, model in enumerate(self.obj_models):
            F_mu[:, i], F_var[:, i] = model.predict(X)
        return F_mu, F_var

    def predict_constraints(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Predict all constraints.

        Returns
        -------
        G_mu : ndarray of shape (n, q)
        G_var : ndarray of shape (n, q)
        """
        n = X.shape[0]
        G_mu = np.empty((n, self.q))
        G_var = np.empty((n, self.q))
        for j, model in enumerate(self.con_models):
            G_mu[:, j], G_var[:, j] = model.predict(X)
        return G_mu, G_var


def build_surrogates(
    X: NDArray[np.floating],
    F_true: NDArray[np.floating],
    G_true: NDArray[np.floating],
    n_restarts: int = 3,
    rng: np.random.Generator | None = None,
) -> SurrogateSet:
    """Build all objective and constraint surrogates from the archive.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Decision variables in the archive.
    F_true : ndarray of shape (n, m)
        True objective values.
    G_true : ndarray of shape (n, q)
        True constraint values.
    n_restarts : int
        Restarts for GP hyperparameter optimization.
    rng : numpy Generator

    Returns
    -------
    surr : SurrogateSet
    """
    if rng is None:
        rng = np.random.default_rng()

    m = F_true.shape[1]
    q = G_true.shape[1]

    obj_models = [
        build_ensemble(X, F_true[:, i], n_restarts=n_restarts, rng=rng)
        for i in range(m)
    ]
    con_models = [
        build_ensemble(X, G_true[:, j], n_restarts=n_restarts, rng=rng)
        for j in range(q)
    ]

    e_obj = float(np.mean([mod.loocv_rmse for mod in obj_models]))
    e_con = float(np.mean([mod.loocv_rmse for mod in con_models]))

    return SurrogateSet(
        obj_models=obj_models,
        con_models=con_models,
        e_obj=e_obj,
        e_con=e_con,
        m=m,
        q=q,
    )
