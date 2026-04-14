"""Integration test for the full CoSAEM algorithm."""

import numpy as np
import pytest

from CoSAEM.algorithm import CoSAEMConfig, run_cosam
from CoSAEM.problems import ZDT1Constrained
from CoSAEM.utils import constraint_violation


class TestCoSAEMIntegration:
    """End-to-end test with reduced budget for speed."""

    @pytest.fixture
    def small_config(self):
        return CoSAEMConfig(
            N=20,
            T_max=80,
            G_inner=5,
            k=3,
            gp_restarts=1,
            seed=42,
        )

    def test_runs_without_error(self, small_config):
        problem = ZDT1Constrained(n_var=5)
        result = run_cosam(problem, small_config)
        assert result.X.shape[0] > 0
        assert result.F.shape[1] == 2
        assert result.G.shape[1] == 1
        assert result.n_evals <= small_config.T_max + small_config.k

    def test_archive_grows(self, small_config):
        problem = ZDT1Constrained(n_var=5)
        result = run_cosam(problem, small_config)
        n_init = 11 * 5 - 1  # 54
        assert result.archive_X.shape[0] >= n_init

    def test_output_nondominated(self, small_config):
        problem = ZDT1Constrained(n_var=5)
        result = run_cosam(problem, small_config)
        F = result.F
        # Check no solution dominates another in the output
        n = F.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert not (np.all(F[i] <= F[j]) and np.any(F[i] < F[j])), \
                        f"Solution {i} dominates {j}"

    def test_history_recorded(self, small_config):
        problem = ZDT1Constrained(n_var=5)
        result = run_cosam(problem, small_config)
        assert len(result.history) > 0
        for entry in result.history:
            assert "iteration" in entry
            assert "beta" in entry
            assert "k_a" in entry
