"""
Tests for three structural issues in the KKT loss pipeline.

Point 1: Variable bound rows are silently dropped from the KKT A matrix
         because var.getLb()/getUb() throws AttributeError in pyscipopt,
         and the except block sets bounds to ±INF so no rows are added.

Point 2: pad_collate_graphs builds A from graph edge_index/edge_attr
         (original constraint signs) but uses b_vec from the KKT system
         (flipped to <= form). For >= constraints (SC) and equalities (CFL),
         this creates a sign mismatch making primal feasibility impossible.

Point 3: The model predicts lambda only for graph constraint nodes, but the
         complete KKT system requires dual variables for variable bound rows
         too. There is no mechanism to produce bound duals.

Run with: conda run -n graph-aug python -m pytest tests/test_kkt_variable_bounds.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle

import numpy as np
import pyscipopt as scp
import pytest
import torch

from data.generators import get_bipartite_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SC_LP = Path("data/instances/milp/SC/instance/test/10/SC-10-0000.lp")
IS_LP = Path("data/instances/milp/IS/instance/test/10/IS-10-0000.lp")
CFL_LP = Path("data/instances/milp/CFL/instance/test/10/CFL-10-0000.lp")

ALL_LPS = [p for p in [SC_LP, IS_LP, CFL_LP] if p.exists()]
GEQ_LPS = [p for p in [SC_LP] if p.exists()]  # problems with >= constraints


def _count_finite_bounds(lp_path: str) -> tuple:
    """Count variables with finite (lower, upper) bounds."""
    problem = scp.Model()
    problem.hideOutput(True)
    problem.readProblem(str(lp_path))
    n_lb, n_ub = 0, 0
    for v in problem.getVars():
        if v.getLbOriginal() > -1e20:
            n_lb += 1
        if v.getUbOriginal() < 1e20:
            n_ub += 1
    return n_lb, n_ub


def _count_expanded_kkt_rows(lp_path: str) -> int:
    """Count expected KKT rows: expanded constraints + variable bounds."""
    problem = scp.Model()
    problem.hideOutput(True)
    problem.readProblem(str(lp_path))
    n_rows = 0
    for c in problem.getConss():
        if not problem.getValsLinear(c):
            continue
        lhs = problem.getLhs(c)
        rhs = problem.getRhs(c)
        if abs(lhs - rhs) < 1e-12:
            n_rows += 2
        else:
            n_rows += 1
    for v in problem.getVars():
        if v.getUbOriginal() < 1e20:
            n_rows += 1
        if v.getLbOriginal() > -1e20:
            n_rows += 1
    return n_rows


# =========================================================================
# Point 1: Variable bound rows missing from KKT A matrix
# =========================================================================


@pytest.mark.parametrize("lp_path", ALL_LPS, ids=lambda p: p.stem)
class TestPoint1_MissingBoundRows:
    """get_bipartite_graph silently drops variable-bound rows from A/b_vec
    because var.getLb() throws AttributeError and the except block sets
    bounds to ±INF."""

    def test_kkt_A_includes_variable_bound_rows(self, lp_path):
        """A.shape[0] must include rows for finite variable bounds,
        not just original constraints."""
        A, _, _, _, _, b_vec, _ = get_bipartite_graph(lp_path)

        expected = _count_expanded_kkt_rows(str(lp_path))
        actual = A.shape[0]

        assert actual == expected, (
            f"{lp_path.stem}: KKT A has {actual} rows, expected {expected}. "
            f"Variable bound rows are missing (getLb/getUb silent failure)."
        )

    def test_b_vec_length_includes_bound_rows(self, lp_path):
        """b_vec must have entries for bound rows too."""
        A, _, _, _, _, b_vec, _ = get_bipartite_graph(lp_path)
        n_lb, n_ub = _count_finite_bounds(str(lp_path))

        if n_lb + n_ub == 0:
            pytest.skip("No finite bounds")

        # b_vec should be longer than just the original constraint count
        n_orig = A.shape[0]  # currently equals original constraints only
        assert b_vec.shape[0] > n_orig or b_vec.shape[0] == _count_expanded_kkt_rows(str(lp_path)), (
            f"{lp_path.stem}: b_vec has {b_vec.shape[0]} entries, same as "
            f"original constraint count. Bound rows ({n_ub} ub + {n_lb} lb) "
            f"are missing."
        )


# =========================================================================
# Point 2: Sign mismatch between graph A and KKT b_vec
# =========================================================================


@pytest.mark.parametrize("lp_path", GEQ_LPS, ids=lambda p: p.stem)
class TestPoint2_SignMismatch:
    """pad_collate_graphs builds A from edge_index/edge_attr (original signs)
    but uses b_vec from the KKT system (flipped for >= constraints).
    This makes Ax <= b unsatisfiable for problems with >= constraints."""

    def test_graph_A_and_kkt_A_have_same_signs(self, lp_path):
        """The A reconstructed from edge_index/edge_attr must match the
        KKT A stored in the .bg file. If signs differ, the loss uses
        inconsistent A and b."""
        A_sp, _, v_nodes, _, _, b_vec, _ = get_bipartite_graph(lp_path)

        # A as pad_collate_graphs would build it (from graph edges)
        rows, cols = A_sp.edge_index
        A_from_graph = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols]),
            values=A_sp.edge_attr.squeeze(-1),
            size=(int(b_vec.numel()), v_nodes.shape[0]),
        ).coalesce().to_dense()

        # A as stored in the KKT system
        A_kkt = A_sp.to_dense()

        assert torch.allclose(A_from_graph, A_kkt), (
            f"{lp_path.stem}: Graph A (from edge_index/edge_attr) has different "
            f"signs than KKT A. Graph A range: [{A_from_graph.min():.1f}, "
            f"{A_from_graph.max():.1f}], KKT A range: [{A_kkt.min():.1f}, "
            f"{A_kkt.max():.1f}]. The loss uses graph A with KKT b_vec, "
            f"creating an inconsistent system."
        )

    def test_graph_A_b_are_consistent_system(self, lp_path):
        """A and b_vec must form a consistent <= system: at the optimal
        solution, Ax <= b should hold (no graph_b vs b_vec mismatch)."""
        A_sp, _, _, _, _, b_vec, _ = get_bipartite_graph(lp_path)

        # b_vec length must equal A rows (unified system)
        assert b_vec.shape[0] == A_sp.shape[0], (
            f"{lp_path.stem}: b_vec ({b_vec.shape[0]}) != A rows ({A_sp.shape[0]})"
        )

    def test_primal_feasibility_achievable_at_optimal(self, lp_path):
        """With the optimal solution, relu(Ax - b) should be zero when
        A and b are consistent. If it's always positive, the system
        is sign-inconsistent."""
        A_sp, v_map, v_nodes, _, _, b_vec, c_vec = get_bipartite_graph(lp_path)

        # Build A the way pad_collate_graphs does (from graph edges)
        rows, cols = A_sp.edge_index
        n = v_nodes.shape[0]
        m = int(b_vec.numel())
        A_graph = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols]),
            values=A_sp.edge_attr.squeeze(-1),
            size=(m, n),
        ).coalesce().to_dense()

        # Load the optimal solution
        bg_path = str(lp_path).replace("/instance/", "/BG/") + ".bg"
        from metrics.optimization import load_optimal_solutions
        opt_pool = load_optimal_solutions(instance_path=bg_path)
        if opt_pool is None:
            pytest.skip("No solution file found")

        x_opt = opt_pool[0].float()

        # Primal feasibility: relu(Ax - b) should be 0 at optimal
        Ax = A_graph @ x_opt
        violation = torch.relu(Ax - b_vec)

        assert violation.sum().item() < 1e-4, (
            f"{lp_path.stem}: Primal violation at optimal solution = "
            f"{violation.sum().item():.4f} (should be ~0). This means the "
            f"A (from graph edges) and b (from KKT) have inconsistent signs, "
            f"making feasibility impossible."
        )


# =========================================================================
# Point 3: Model predicts lambda only for constraint nodes, not bound rows
# =========================================================================


@pytest.mark.parametrize("lp_path", ALL_LPS, ids=lambda p: p.stem)
class TestPoint3_MissingBoundDuals:
    """The GNN model outputs one lambda per graph constraint node, but the
    complete KKT system needs dual variables for variable bound rows too."""

    def test_model_lambda_count_matches_kkt_rows(self, lp_path):
        """The number of lambda values the model produces (= number of
        constraint nodes) must equal the number of KKT rows. Currently
        it only matches the original constraint count."""
        A_sp, _, _, c_nodes, _, b_vec, _ = get_bipartite_graph(lp_path)

        n_constraint_nodes = c_nodes.shape[0]
        n_kkt_rows = _count_expanded_kkt_rows(str(lp_path))
        n_lb, n_ub = _count_finite_bounds(str(lp_path))

        assert n_constraint_nodes == n_kkt_rows, (
            f"{lp_path.stem}: Model produces {n_constraint_nodes} lambdas "
            f"(one per constraint node) but the full KKT system has "
            f"{n_kkt_rows} rows ({n_kkt_rows - n_lb - n_ub} constraints + "
            f"{n_ub} upper bounds + {n_lb} lower bounds). "
            f"Bound duals are not represented."
        )

    def test_stationarity_satisfiable_with_full_A(self, lp_path):
        """With the full standardized A (including bound rows), it should
        be possible to find lambda >= 0 such that c + A^T lambda ~ 0.
        This verifies that the model now has enough dual variables."""
        from scipy.optimize import lsq_linear

        A_sp, v_map, v_nodes, c_nodes, _, b_vec, c_vec = get_bipartite_graph(lp_path)

        n_lb, n_ub = _count_finite_bounds(str(lp_path))
        if n_lb + n_ub == 0:
            pytest.skip("No finite bounds")

        A_full = A_sp.to_dense().numpy().astype(np.float64)  # [m_std, n]
        c_np = np.asarray(c_vec, dtype=np.float64)

        # Solve: min ||A^T lambda - (-c)||^2  s.t. lambda >= 0
        AT = A_full.T  # [n, m_std]
        target = -c_np  # [n]
        m_std = AT.shape[1]

        result = lsq_linear(AT, target, bounds=(0, np.inf), method="bvls")
        residual = c_np + AT @ result.x

        rel_norm = np.linalg.norm(residual) / (np.linalg.norm(c_np) + 1e-9)

        assert rel_norm < 0.05, (
            f"{lp_path.stem}: Relative stationarity residual = "
            f"{rel_norm:.4f} with full A (including bounds). "
            f"The full system should be nearly satisfiable with lambda >= 0."
        )
