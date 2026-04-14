"""Tests for selection module."""

import numpy as np

from CoSAEM.selection import (
    cdp_select,
    crowding_distance,
    fast_non_dominated_sort,
    nds_select,
)


class TestNDS:
    def test_single_front(self):
        # All on the Pareto front
        F = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        fronts = fast_non_dominated_sort(F)
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2}

    def test_two_fronts(self):
        F = np.array([
            [0.0, 1.0],  # front 0
            [1.0, 0.0],  # front 0
            [0.5, 1.5],  # front 1 (dominated by [0.0, 1.0])
        ])
        fronts = fast_non_dominated_sort(F)
        assert len(fronts) == 2
        assert set(fronts[0]) == {0, 1}
        assert set(fronts[1]) == {2}

    def test_dominated(self):
        F = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        fronts = fast_non_dominated_sort(F)
        assert len(fronts) == 3


class TestCrowdingDistance:
    def test_boundary_infinite(self):
        F = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        cd = crowding_distance(F)
        assert cd[0] == np.inf
        assert cd[3] == np.inf

    def test_two_points(self):
        F = np.array([[0.0, 1.0], [1.0, 0.0]])
        cd = crowding_distance(F)
        assert np.all(cd == np.inf)


class TestNDSSelect:
    def test_select_fewer(self):
        F = np.array([
            [0.0, 1.0], [0.5, 0.5], [1.0, 0.0],  # front 0
            [0.5, 0.8], [0.8, 0.5],                # front 1
        ])
        sel = nds_select(F, 3)
        assert len(sel) == 3
        # All front-0 solutions should be selected
        assert set(sel) == {0, 1, 2}


class TestCDPSelect:
    def test_feasible_preferred(self):
        F = np.array([[0.5, 0.5], [0.1, 0.1]])
        G = np.array([[-1.0], [1.0]])  # first feasible, second infeasible
        sel = cdp_select(F, G, 1)
        assert sel[0] == 0  # feasible is selected

    def test_infeasible_by_cv(self):
        F = np.array([[0.1, 0.1], [0.5, 0.5]])
        G = np.array([[2.0], [0.5]])  # both infeasible
        sel = cdp_select(F, G, 1)
        assert sel[0] == 1  # lower CV preferred

    def test_all_feasible(self):
        F = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.3, 0.7]])
        G = np.full((4, 1), -1.0)
        sel = cdp_select(F, G, 3)
        assert len(sel) == 3
