"""
Subproblem Decomposition with Solving
======================================
Loads an .lp MILP problem as a bipartite graph, splits it into subgraphs
(each with at most ``max_subgraph_ratio`` fraction of the total nodes)
using a configurable strategy, and **solves** each subgraph — producing
``x_pred`` (primal) and ``lambda_pred`` (dual) per subproblem.

Supports two solving methods:
- ``gnn``: predict x and lambda via a trained GNNPolicy model
- ``gurobi``: reconstruct the sub-LP from the subgraph and solve with Gurobi

Configuration is read from configs/decomposition/config.yml and can be
overridden via CLI flags (e.g. --max_subgraph_ratio 0.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import configargparse
import torch

from data.datasets import BipartiteNodeData
from data.generators import get_bipartite_graph
from models.decomposition import (
    build_halo_subgraphs,
    compute_coupling_diagnostics,
    n_splits_for,
    split_bipartite_graph_metis,
    split_by_constraints,
    split_by_variables,
)
from models.gnn import GNNPolicy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

VALID_STRATEGIES = ("variables", "constraints", "metis")
VALID_SOLVE_METHODS = ("gnn", "gurobi")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SubproblemResult:
    """Result of solving a single subproblem."""

    x_pred: torch.Tensor  # [n_sub_vars]
    lambda_pred: torch.Tensor  # [n_sub_cons]
    orig_var_ids: torch.Tensor  # mapping back to master variable indices
    orig_cons_ids: torch.Tensor  # mapping back to master constraint indices


# ---------------------------------------------------------------------------
# Gurobi sub-LP solver
# ---------------------------------------------------------------------------


def _solve_subproblem_gurobi(
    sg: BipartiteNodeData,
    graph_b: torch.Tensor,
    graph_sense: torch.Tensor,
    c_vec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct a sub-LP from the subgraph data and solve with Gurobi.

    Parameters
    ----------
    sg : BipartiteNodeData
        Subgraph with ``edge_index`` (local), ``edge_attr`` (A_ij coefficients),
        ``orig_cons_ids``, and ``orig_var_ids``.
    graph_b : torch.Tensor
        [m_orig] RHS values for all original constraints.
    graph_sense : torch.Tensor
        [m_orig] Sense codes (0=<=, 1=>=, 2==).
    c_vec : torch.Tensor
        [n] Objective coefficients for all variables.

    Returns
    -------
    x : torch.Tensor [n_sub_vars]
    lam : torch.Tensor [n_sub_cons]
    """
    import gurobipy as gp
    from gurobipy import GRB

    n_sub_vars = sg.variable_features.size(0)
    n_sub_cons = sg.constraint_features.size(0)

    # Extract sub-problem data using original indices
    orig_cons = sg.orig_cons_ids.numpy()
    orig_vars = sg.orig_var_ids.numpy()
    b_sub = graph_b[orig_cons].numpy() if isinstance(graph_b, torch.Tensor) else graph_b[orig_cons]
    sense_sub = graph_sense[orig_cons].numpy() if isinstance(graph_sense, torch.Tensor) else graph_sense[orig_cons]
    c_sub = c_vec[orig_vars].numpy() if isinstance(c_vec, torch.Tensor) else c_vec[orig_vars]

    # Build sparse A_sub from edge_index and edge_attr
    local_cons = sg.edge_index[0].numpy()  # local constraint indices
    local_vars = sg.edge_index[1].numpy()  # local variable indices
    coeffs = sg.edge_attr[:, 0].numpy()  # A_ij values

    # Map sense codes to Gurobi sense constants
    sense_map = {0: GRB.LESS_EQUAL, 1: GRB.GREATER_EQUAL, 2: GRB.EQUAL}

    model = gp.Model("subproblem")
    model.Params.OutputFlag = 0  # suppress output

    # Add continuous variables (LP relaxation)
    x_vars = model.addVars(
        n_sub_vars, lb=0.0, ub=1.0, name="x",
    )
    model.update()
    # Set objective coefficients
    model.setObjective(
        gp.LinExpr(c_sub.tolist(), [x_vars[j] for j in range(n_sub_vars)]),
        GRB.MINIMIZE,
    )

    # Build constraint matrix row by row, tracking which local indices
    # were actually added so duals can be placed at the correct positions.
    added_cons_indices: List[int] = []
    for i in range(n_sub_cons):
        edge_mask = local_cons == i
        if not edge_mask.any():
            continue
        var_indices = local_vars[edge_mask]
        var_coeffs = coeffs[edge_mask]

        sense_code = int(sense_sub[i])
        grb_sense = sense_map.get(sense_code, GRB.LESS_EQUAL)

        expr = gp.LinExpr()
        for j, coeff in zip(var_indices, var_coeffs):
            expr.addTerms(float(coeff), x_vars[int(j)])

        model.addLConstr(expr, grb_sense, float(b_sub[i]))
        added_cons_indices.append(i)

    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        x_sol = torch.tensor([x_vars[j].X for j in range(n_sub_vars)], dtype=torch.float32)
        # Extract dual values and place at correct constraint positions
        constrs = model.getConstrs()
        lam_sol = torch.zeros(n_sub_cons, dtype=torch.float32)
        for idx, constr in zip(added_cons_indices, constrs):
            lam_sol[idx] = constr.Pi
    else:
        logger.warning(
            "Gurobi sub-LP did not solve to optimality (status=%d). "
            "Returning zeros.",
            model.Status,
        )
        x_sol = torch.zeros(n_sub_vars, dtype=torch.float32)
        lam_sol = torch.zeros(n_sub_cons, dtype=torch.float32)

    return x_sol, lam_sol


# ---------------------------------------------------------------------------
# GNN model construction
# ---------------------------------------------------------------------------


def _build_gnn_model(
    checkpoint_path: str, args: configargparse.Namespace, device: torch.device
) -> GNNPolicy:
    """Instantiate a full GNNPolicy and load weights from checkpoint."""
    model = GNNPolicy(args).to(device)

    pkg = torch.load(checkpoint_path, map_location="cpu")
    if "model" in pkg:
        model.load_state_dict(pkg["model"])
        logger.info("Loaded full GNNPolicy from %s (key='model')", checkpoint_path)
    elif "encoder" in pkg:
        model.encoder.load_state_dict(pkg["encoder"], strict=False)
        logger.info(
            "Loaded encoder weights from %s (key='encoder'). "
            "Heads use random weights.",
            checkpoint_path,
        )
    else:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} has neither 'model' nor 'encoder' key. "
            f"Available keys: {list(pkg.keys())}"
        )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        default_config_files=["configs/decomposition/config.yml"],
    )

    g = parser.add_argument_group("decomposition")
    g.add_argument(
        "--lp_master_path",
        type=str,
        required=True,
        help="Path to the .lp MILP problem file",
    )
    g.add_argument(
        "--max_subgraph_ratio",
        type=float,
        required=True,
        help="Max subgraph size as fraction of total nodes (e.g. 0.2 = 20%%)",
    )
    g.add_argument(
        "--split_strategy",
        type=str,
        choices=list(VALID_STRATEGIES),
        required=True,
        help="Splitting strategy: variables, constraints, or metis",
    )
    g.add_argument(
        "--solve_method",
        type=str,
        choices=list(VALID_SOLVE_METHODS),
        required=True,
        help="Solving method: gnn (model prediction) or gurobi (LP relaxation)",
    )
    g.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to GNNPolicy checkpoint (required for --solve_method gnn)",
    )
    g.add_argument(
        "--halo_hops",
        type=int,
        default=0,
        help="Number of BFS hops for halo expansion (0=none, 1, 2, 4 typical). "
        "Only used with --split_strategy metis.",
    )
    g.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda:N",
    )

    # Add GNN model args so checkpoint can be loaded with matching architecture
    GNNPolicy.add_args(parser)

    args, _ = parser.parse_known_args()
    return args


@torch.inference_mode()
def main() -> List[SubproblemResult]:
    args = _parse_args()

    lp_path = Path(args.lp_master_path)
    if not lp_path.exists():
        raise FileNotFoundError(f"LP file not found: {lp_path}")

    ratio = args.max_subgraph_ratio
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"max_subgraph_ratio must be in (0, 1], got {ratio}")

    split_strategy = args.split_strategy
    solve_method = args.solve_method
    halo_hops = args.halo_hops
    checkpoint_path = args.checkpoint_path or None
    device = torch.device(args.device)

    if solve_method == "gnn" and not checkpoint_path:
        raise ValueError("--checkpoint_path is required when --solve_method is 'gnn'")

    # --- Step 1: Load bipartite graph ---
    A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec = get_bipartite_graph(lp_path)
    edge_index = A.edge_index  # [2, E] over original constraints
    edge_attr = A.edge_attr  # [E, 1]
    graph_b = A.graph_b  # [m_orig] RHS values
    graph_sense = A.graph_sense  # [m_orig] sense codes (0=<=, 1=>=, 2==)

    n_cons = c_nodes.size(0)
    n_vars = v_nodes.size(0)

    if split_strategy == "constraints":
        max_subgraph_size = max(1, int(n_cons * ratio))
        ratio_target, ratio_count = "constraints", n_cons
    elif split_strategy == "variables":
        max_subgraph_size = max(1, int(n_vars * ratio))
        ratio_target, ratio_count = "variables", n_vars
    else:  # metis
        max_subgraph_size = max(1, int((n_cons + n_vars) * ratio))
        ratio_target, ratio_count = "total nodes", n_cons + n_vars

    logger.info("LP file            : %s", lp_path)
    logger.info("max_subgraph_ratio : %.2f", ratio)
    logger.info(
        "%d %s -> max_subgraph_size = %d",
        ratio_count,
        ratio_target,
        max_subgraph_size,
    )
    logger.info("split_strategy     : %s", split_strategy)
    logger.info("halo_hops          : %d", halo_hops)
    logger.info("solve_method       : %s", solve_method)
    logger.info("checkpoint         : %s", checkpoint_path or "(none)")
    logger.info("device             : %s", device)

    logger.info(
        "Loaded graph — %d constraints, %d variables, %d edges",
        c_nodes.size(0),
        v_nodes.size(0),
        edge_index.size(1),
    )

    # --- Step 2: Split ---
    if split_strategy == "constraints":
        if halo_hops > 0:
            logger.warning("halo_hops=%d ignored for split_strategy='constraints'", halo_hops)
        subgraphs = split_by_constraints(
            c_nodes, v_nodes, edge_index, edge_attr, max_subgraph_size
        )
    elif split_strategy == "variables":
        if halo_hops > 0:
            logger.warning("halo_hops=%d ignored for split_strategy='variables'", halo_hops)
        subgraphs = split_by_variables(
            c_nodes, v_nodes, edge_index, edge_attr, max_subgraph_size
        )
    else:  # metis
        num_parts = n_splits_for(n_cons + n_vars, max_subgraph_size)
        specs = split_bipartite_graph_metis(
            c_nodes, v_nodes, edge_index, edge_attr, num_parts=num_parts
        )
        subgraphs = build_halo_subgraphs(
            specs, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=halo_hops,
        )

    logger.info("Created %d subgraphs:", len(subgraphs))
    for i, sg in enumerate(subgraphs):
        n_c = sg.constraint_features.size(0)
        n_v = sg.variable_features.size(0)
        halo_info = ""
        if hasattr(sg, "owned_cons_mask"):
            owned_c = int(sg.owned_cons_mask.sum().item())
            owned_v = int(sg.owned_var_mask.sum().item())
            halo_info = f" (owned: {owned_c}c+{owned_v}v, halo: {n_c - owned_c}c+{n_v - owned_v}v)"
        logger.info(
            "  subgraph %d — %d constraints, %d variables, %d edges%s",
            i, n_c, n_v, sg.edge_index.size(1), halo_info,
        )

    # --- Coupling diagnostics ---
    diag = compute_coupling_diagnostics(subgraphs, n_cons, n_vars, edge_index)
    logger.info("--- Coupling diagnostics ---")
    logger.info(
        "  coupling constraints  : %d / %d (%.1f%%)",
        diag.n_coupling_constraints,
        diag.n_total_constraints,
        diag.coupling_fraction * 100,
    )
    logger.info("  avg blocks/constraint : %.2f", diag.avg_blocks_per_constraint)
    logger.info(
        "  edge cut count        : %d / %d (%.1f%%)",
        diag.edge_cut_count,
        diag.n_total_edges,
        diag.edge_cut_fraction * 100,
    )

    # --- Step 3: Solve ---
    if solve_method == "gnn":
        model = _build_gnn_model(checkpoint_path, args, device)

    results: List[SubproblemResult] = []
    for i, sg in enumerate(subgraphs):
        if solve_method == "gnn":
            c_feat = sg.constraint_features.to(device)
            v_feat = sg.variable_features.to(device)
            ei = sg.edge_index.to(device)
            ea = sg.edge_attr.to(device)

            x_pred, lambda_pred = model(c_feat, ei, ea, v_feat)
            x_pred = x_pred.cpu()
            lambda_pred = lambda_pred.cpu()
        else:  # gurobi
            x_pred, lambda_pred = _solve_subproblem_gurobi(
                sg, graph_b, graph_sense, c_vec
            )

        result = SubproblemResult(
            x_pred=x_pred,
            lambda_pred=lambda_pred,
            orig_var_ids=sg.orig_var_ids,
            orig_cons_ids=sg.orig_cons_ids,
        )
        results.append(result)

        logger.info(
            "  subgraph %d solved — x_pred: %s, lambda_pred: %s",
            i,
            list(x_pred.shape),
            list(lambda_pred.shape),
        )

    return results


if __name__ == "__main__":
    main()
