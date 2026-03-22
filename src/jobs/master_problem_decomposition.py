"""
Master Problem Decomposition
=============================
Loads an .lp MILP problem as a bipartite graph, splits it into subgraphs
(each with at most ``max_subgraph_ratio`` fraction of the total nodes)
using a configurable strategy, and embeds each subgraph through a
pretrained GNN encoder.

Configuration is read from configs/decomposition/config.yml and can be
overridden via CLI flags (e.g. --max_subgraph_ratio 0.3).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import configargparse
import torch

from data.common import CONS_PAD, VARS_PAD
from data.datasets import BipartiteNodeData, right_pad
from data.generators import get_bipartite_graph
from models.gnn import GNNEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

VALID_STRATEGIES = ("variables", "constraints", "metis")


# ---------------------------------------------------------------------------
# Splitting helpers
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
# Encoder construction
# ---------------------------------------------------------------------------


def _infer_embedding_size(state_dict: dict) -> int:
    """Infer embedding_size from checkpoint state dict by inspecting layer shapes."""
    # conv_v_to_c.feature_module_left.0.weight has shape [embedding_size, embedding_size]
    key = "conv_v_to_c.feature_module_left.0.weight"
    if key in state_dict:
        return state_dict[key].shape[0]
    # Fallback: cons_proj last linear output
    for k, v in state_dict.items():
        if k.startswith("cons_proj") and k.endswith(".weight") and v.dim() == 2:
            return v.shape[0]
    return 64


def _infer_conv_type(state_dict: dict) -> str:
    """Infer bipartite_conv type from checkpoint keys."""
    keys = set(state_dict.keys())
    if any("conv_v_to_c.conv." in k for k in keys):
        # GATv2Conv or TransformerConv have a .conv sub-module
        if any("att_src" in k for k in keys):
            return "gatv2"
        return "transformer"
    return "gcn"


def _build_encoder(checkpoint_path: str | None, device: torch.device) -> GNNEncoder:
    """Instantiate a GNNEncoder and optionally load weights from checkpoint.

    When a checkpoint is provided, model hyperparameters (embedding_size, conv type)
    are inferred from the state dict to ensure compatibility.
    """
    embedding_size = 64
    bipartite_conv = "gcn"

    if checkpoint_path:
        pkg = torch.load(checkpoint_path, map_location="cpu")
        if "encoder" not in pkg:
            raise RuntimeError(
                f"Checkpoint at {checkpoint_path} has no 'encoder' key. "
                f"Available keys: {list(pkg.keys())}"
            )
        state_dict = pkg["encoder"]
        embedding_size = _infer_embedding_size(state_dict)
        bipartite_conv = _infer_conv_type(state_dict)
        logger.info(
            "Inferred from checkpoint: embedding_size=%d, bipartite_conv=%s",
            embedding_size,
            bipartite_conv,
        )

    encoder = GNNEncoder(
        cons_nfeats=CONS_PAD,
        var_nfeats=VARS_PAD,
        edge_nfeats=1,
        embedding_size=embedding_size,
        num_emb_type="periodic",
        num_emb_freqs=16,
        num_emb_bins=32,
        lejepa_embed_dim=128,
        dropout=0.0,
        bipartite_conv=bipartite_conv,
        attn_heads=4,
    )

    if checkpoint_path:
        encoder.load_state_dict(state_dict, strict=False)
        logger.info("Loaded encoder weights from %s", checkpoint_path)
    else:
        logger.warning("No checkpoint_path set — using random encoder weights.")

    encoder.to(device)
    encoder.eval()
    return encoder


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
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to encoder checkpoint; empty = random weights",
    )
    g.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda:N",
    )

    args, _ = parser.parse_known_args()
    return args


@torch.inference_mode()
def main() -> List[Tuple[torch.Tensor, torch.Tensor]]:
    args = _parse_args()

    lp_path = Path(args.lp_master_path)
    if not lp_path.exists():
        raise FileNotFoundError(f"LP file not found: {lp_path}")

    ratio = args.max_subgraph_ratio
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"max_subgraph_ratio must be in (0, 1], got {ratio}")

    split_strategy = args.split_strategy
    checkpoint_path = args.checkpoint_path or None
    device = torch.device(args.device)

    # --- Step 1: Load bipartite graph ---
    A, v_map, v_nodes, c_nodes, b_vars, b_vec, c_vec = get_bipartite_graph(lp_path)
    edge_index = A.edge_index  # [2, E] over original constraints
    edge_attr = A.edge_attr  # [E, 1]

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
    logger.info("checkpoint         : %s", checkpoint_path or "(random weights)")
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
    # Use scatter: build a (n_cons, n_blocks) indicator then count unique blocks
    n_blocks = len(subgraphs)
    # Sparse indicator: constraint i touches block b
    cons_block_pair = torch.stack([edge_index[0], edge_blocks], dim=0)  # [2, E]
    # Remove edges to unassigned variables (block == -1)
    assigned_mask = edge_blocks >= 0
    cons_block_pair = cons_block_pair[:, assigned_mask]
    # Unique (constraint, block) pairs
    unique_pairs = cons_block_pair.T.unique(dim=0)  # [P, 2]
    # Count how many distinct blocks each constraint touches
    cons_ids_in_pairs, blocks_per_cons_counts = unique_pairs[:, 0].unique(
        return_counts=True
    )
    # Constraints not appearing in any edge get 0 blocks
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

    # --- Step 3: Embed ---
    encoder = _build_encoder(checkpoint_path, device)

    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, sg in enumerate(subgraphs):
        c_feat = sg.constraint_features.to(device)
        v_feat = sg.variable_features.to(device)
        ei = sg.edge_index.to(device)
        ea = sg.edge_attr.to(device)

        c_emb, v_emb = encoder.encode_nodes(c_feat, ei, ea, v_feat)
        results.append((c_emb.cpu(), v_emb.cpu()))

        logger.info(
            "  subgraph %d embeddings — constraints: %s, variables: %s",
            i,
            list(c_emb.shape),
            list(v_emb.shape),
        )

    return results


if __name__ == "__main__":
    main()
