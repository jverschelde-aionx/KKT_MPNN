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

from data.common import CONS_PAD, VARS_PAD
from data.datasets import BipartiteNodeData, right_pad
from data.generators import get_bipartite_graph
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
# Splitting helpers (identical to master_problem_decomposition.py)
# ---------------------------------------------------------------------------


def _n_splits_for(n_total: int, max_size: int) -> int:
    """Compute minimum number of partitions so each has at most max_size nodes."""
    import math

    return max(1, math.ceil(n_total / max_size))


def _split_by_constraints(
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    max_subgraph_size: int,
) -> List[BipartiteNodeData]:
    """Partition constraint nodes so each chunk has at most max_subgraph_size
    constraints. Connected variables are induced and may exceed the cap."""
    n_cons = c_nodes.size(0)
    n_splits = _n_splits_for(n_cons, max_subgraph_size)
    chunk_sizes = _balanced_chunks(n_cons, n_splits)

    subgraphs: List[BipartiteNodeData] = []
    offset = 0
    for size in chunk_sizes:
        if size == 0:
            offset += size
            continue
        cons_ids = torch.arange(offset, offset + size)
        offset += size
        sg = _extract_subgraph_by_constraints(
            cons_ids, c_nodes, v_nodes, edge_index, edge_attr
        )
        subgraphs.append(sg)
    return subgraphs


def _split_by_variables(
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    max_subgraph_size: int,
) -> List[BipartiteNodeData]:
    """Partition variable nodes so each chunk has at most max_subgraph_size
    variables. Connected constraints are induced and may exceed the cap."""
    n_vars = v_nodes.size(0)
    n_splits = _n_splits_for(n_vars, max_subgraph_size)
    chunk_sizes = _balanced_chunks(n_vars, n_splits)

    subgraphs: List[BipartiteNodeData] = []
    offset = 0
    for size in chunk_sizes:
        if size == 0:
            offset += size
            continue
        var_ids = torch.arange(offset, offset + size)
        offset += size
        sg = _extract_subgraph_by_variables(
            var_ids, c_nodes, v_nodes, edge_index, edge_attr
        )
        subgraphs.append(sg)
    return subgraphs


def _split_by_metis(
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    max_subgraph_size: int,
) -> List[BipartiteNodeData]:
    """Use METIS graph partitioning so each subgraph has at most
    max_subgraph_size total nodes."""
    try:
        import pymetis
    except ImportError:
        raise ImportError(
            "pymetis is required for the 'metis' split strategy. "
            "Install it with: pip install pymetis"
        )

    n_cons = c_nodes.size(0)
    n_vars = v_nodes.size(0)
    n_total = n_cons + n_vars
    n_splits = _n_splits_for(n_total, max_subgraph_size)

    if n_splits <= 1:
        return [
            _extract_subgraph(
                torch.arange(n_cons),
                torch.arange(n_vars),
                c_nodes,
                v_nodes,
                edge_index,
                edge_attr,
            )
        ]

    # Build adjacency list for pymetis (undirected bipartite graph).
    # Nodes 0..n_cons-1 are constraints, n_cons..n_total-1 are variables.
    adjacency: List[List[int]] = [[] for _ in range(n_total)]
    cons_idx = edge_index[0]  # constraint indices
    var_idx = edge_index[1]  # variable indices

    for c, v in zip(cons_idx.tolist(), var_idx.tolist()):
        v_shifted = v + n_cons
        adjacency[c].append(v_shifted)
        adjacency[v_shifted].append(c)

    # Deduplicate adjacency lists
    adjacency = [sorted(set(nbrs)) for nbrs in adjacency]

    _, membership = pymetis.part_graph(n_splits, adjacency=adjacency)
    membership = torch.tensor(membership, dtype=torch.long)

    subgraphs: List[BipartiteNodeData] = []
    for part in range(n_splits):
        part_nodes = (membership == part).nonzero(as_tuple=False).view(-1)
        cons_ids = part_nodes[part_nodes < n_cons]
        var_ids = part_nodes[part_nodes >= n_cons] - n_cons

        # If a partition has no constraint or variable nodes, include all
        # connected ones to avoid empty subgraphs.
        if cons_ids.numel() == 0 and var_ids.numel() > 0:
            var_set = set(var_ids.tolist())
            mask = torch.tensor(
                [v.item() in var_set for v in edge_index[1]], dtype=torch.bool
            )
            cons_ids = edge_index[0][mask].unique()
        elif var_ids.numel() == 0 and cons_ids.numel() > 0:
            cons_set = set(cons_ids.tolist())
            mask = torch.tensor(
                [c.item() in cons_set for c in edge_index[0]], dtype=torch.bool
            )
            var_ids = edge_index[1][mask].unique()

        if cons_ids.numel() == 0 or var_ids.numel() == 0:
            continue

        sg = _extract_subgraph(
            cons_ids, var_ids, c_nodes, v_nodes, edge_index, edge_attr
        )
        subgraphs.append(sg)

    return subgraphs


# ---------------------------------------------------------------------------
# Extraction utilities
# ---------------------------------------------------------------------------


def _balanced_chunks(total: int, n: int) -> List[int]:
    """Split `total` items into `n` chunks as evenly as possible."""
    base, remainder = divmod(total, n)
    return [base + (1 if i < remainder else 0) for i in range(n)]


def _extract_subgraph_by_constraints(
    cons_ids: torch.Tensor,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> BipartiteNodeData:
    """Given a set of constraint node ids, extract the induced subgraph
    (all variables connected to those constraints)."""

    cons_mask = torch.zeros(c_nodes.size(0), dtype=torch.bool)
    cons_mask[cons_ids] = True
    edge_mask = cons_mask[edge_index[0]]  # [E] bool
    var_ids = edge_index[1][edge_mask].unique()

    return _extract_subgraph(cons_ids, var_ids, c_nodes, v_nodes, edge_index, edge_attr)


def _extract_subgraph_by_variables(
    var_ids: torch.Tensor,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> BipartiteNodeData:
    """Given a set of variable node ids, extract the induced subgraph
    (all constraints connected to those variables)."""
    var_mask = torch.zeros(v_nodes.size(0), dtype=torch.bool)
    var_mask[var_ids] = True
    edge_mask = var_mask[edge_index[1]]
    cons_ids = edge_index[0][edge_mask].unique()
    return _extract_subgraph(cons_ids, var_ids, c_nodes, v_nodes, edge_index, edge_attr)


def _extract_subgraph(
    cons_ids: torch.Tensor,
    var_ids: torch.Tensor,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> BipartiteNodeData:
    """Build a BipartiteNodeData from a subset of constraint and variable nodes."""
    cons_set = set(cons_ids.tolist())
    var_set = set(var_ids.tolist())

    # Build remapping dicts
    cons_remap = {old: new for new, old in enumerate(sorted(cons_set))}
    var_remap = {old: new for new, old in enumerate(sorted(var_set))}

    # Find edges that connect the selected constraints and variables
    cons_mask = torch.zeros(c_nodes.size(0), dtype=torch.bool)
    cons_mask[cons_ids] = True
    var_mask = torch.zeros(v_nodes.size(0), dtype=torch.bool)
    var_mask[var_ids] = True
    edge_mask = cons_mask[edge_index[0]] & var_mask[edge_index[1]]
    sub_edge_index = edge_index[:, edge_mask]
    sub_edge_attr = edge_attr[edge_mask]

    # Remap edge indices
    new_cons = torch.tensor(
        [cons_remap[c.item()] for c in sub_edge_index[0]], dtype=torch.long
    )
    new_vars = torch.tensor(
        [var_remap[v.item()] for v in sub_edge_index[1]], dtype=torch.long
    )
    sub_edge_index = torch.stack([new_cons, new_vars], dim=0)

    # Extract node features
    sorted_cons = sorted(cons_set)
    sorted_vars = sorted(var_set)
    sub_c_nodes = c_nodes[sorted_cons]
    sub_v_nodes = v_nodes[sorted_vars]

    # Pad features to expected dimensions
    sub_c_nodes = right_pad(sub_c_nodes, CONS_PAD)
    sub_v_nodes = right_pad(sub_v_nodes, VARS_PAD)

    sg = BipartiteNodeData(
        constraint_features=sub_c_nodes,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        variable_features=sub_v_nodes,
    )

    sg.orig_cons_ids = torch.tensor(sorted_cons, dtype=torch.long)
    sg.orig_var_ids = torch.tensor(sorted_vars, dtype=torch.long)
    return sg


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
    b_sub = graph_b[sg.orig_cons_ids].numpy()
    sense_sub = graph_sense[sg.orig_cons_ids].numpy()
    c_sub = c_vec[sg.orig_var_ids].numpy()

    # Build sparse A_sub from edge_index and edge_attr
    local_cons = sg.edge_index[0].numpy()  # local constraint indices
    local_vars = sg.edge_index[1].numpy()  # local variable indices
    coeffs = sg.edge_attr[:, 0].numpy()  # A_ij values

    # Map sense codes to Gurobi sense constants
    sense_map = {0: GRB.LESS_EQUAL, 1: GRB.GREATER_EQUAL, 2: GRB.EQUAL}

    model = gp.Model("subproblem")
    model.Params.OutputFlag = 0  # suppress output

    # Add continuous variables (LP relaxation)
    x_vars = model.addMVar(n_sub_vars, lb=0.0, ub=1.0, obj=c_sub, name="x")

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

        model.addConstr(expr, grb_sense, float(b_sub[i]))
        added_cons_indices.append(i)

    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        x_sol = torch.tensor([v.X for v in x_vars.tolist()], dtype=torch.float32)
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
        subgraphs = _split_by_constraints(
            c_nodes, v_nodes, edge_index, edge_attr, max_subgraph_size
        )
    elif split_strategy == "variables":
        subgraphs = _split_by_variables(
            c_nodes, v_nodes, edge_index, edge_attr, max_subgraph_size
        )
    else:  # metis
        subgraphs = _split_by_metis(
            c_nodes, v_nodes, edge_index, edge_attr, max_subgraph_size
        )

    logger.info("Created %d subgraphs:", len(subgraphs))
    for i, sg in enumerate(subgraphs):
        logger.info(
            "  subgraph %d — %d constraints, %d variables, %d edges",
            i,
            sg.constraint_features.size(0),
            sg.variable_features.size(0),
            sg.edge_index.size(1),
        )

    # --- Coupling diagnostics ---
    var_to_block = -torch.ones(n_vars, dtype=torch.long)
    for k, sg in enumerate(subgraphs):
        var_to_block[sg.orig_var_ids] = k

    # For each edge, look up which block its variable belongs to
    edge_blocks = var_to_block[edge_index[1]]  # [E]

    # For each constraint, find the set of blocks its variables belong to
    n_blocks = len(subgraphs)
    cons_block_pair = torch.stack([edge_index[0], edge_blocks], dim=0)  # [2, E]
    assigned_mask = edge_blocks >= 0
    cons_block_pair = cons_block_pair[:, assigned_mask]
    unique_pairs = cons_block_pair.T.unique(dim=0)  # [P, 2]
    cons_ids_in_pairs, blocks_per_cons_counts = unique_pairs[:, 0].unique(
        return_counts=True
    )
    blocks_per_constraint = torch.zeros(n_cons, dtype=torch.long)
    blocks_per_constraint[cons_ids_in_pairs] = blocks_per_cons_counts

    coupling_mask = blocks_per_constraint > 1
    n_coupling = coupling_mask.sum().item()
    frac_coupling = n_coupling / max(n_cons, 1)
    avg_blocks = blocks_per_constraint.float().mean().item()

    # Edge cut: edges whose constraint and variable belong to different blocks
    cons_to_block = -torch.ones(n_cons, dtype=torch.long)
    for k, sg in enumerate(subgraphs):
        cons_to_block[sg.orig_cons_ids] = k
    edge_cons_block = cons_to_block[edge_index[0]]
    edge_var_block = var_to_block[edge_index[1]]
    both_assigned = (edge_cons_block >= 0) & (edge_var_block >= 0)
    edge_cut_count = int(
        ((edge_cons_block != edge_var_block) & both_assigned).sum().item()
    )

    logger.info("--- Coupling diagnostics ---")
    logger.info(
        "  coupling constraints  : %d / %d (%.1f%%)",
        n_coupling,
        n_cons,
        frac_coupling * 100,
    )
    logger.info("  avg blocks/constraint : %.2f", avg_blocks)
    logger.info(
        "  edge cut count        : %d / %d (%.1f%%)",
        edge_cut_count,
        edge_index.size(1),
        edge_cut_count / max(edge_index.size(1), 1) * 100,
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
