"""
Decomposition utilities for bipartite MILP graphs.
===================================================
Consolidates splitting, extraction, coupling diagnostics, and validation
logic previously duplicated across job scripts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from data.common import CONS_PAD, VARS_PAD
from data.datasets import BipartiteNodeData, right_pad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PartitionSpec:
    """Describes which constraints and variables belong to a single partition."""

    part_id: int
    owned_cons_ids: torch.Tensor  # [C_k] sorted original constraint indices
    owned_var_ids: torch.Tensor  # [V_k] sorted original variable indices


@dataclass
class CouplingDiagnostics:
    """Summary statistics about inter-partition coupling."""

    n_coupling_constraints: int
    n_total_constraints: int
    coupling_fraction: float
    avg_blocks_per_constraint: float
    edge_cut_count: int
    n_total_edges: int
    edge_cut_fraction: float
    n_boundary_cons: int
    n_boundary_vars: int
    boundary_cons_fraction: float
    boundary_vars_fraction: float
    n_total_vars: int


@dataclass
class BlockGraph:
    """The block interaction graph over partitions.

    Edge feature layout: ``[cut_edge_count, sum_abs_coefficients,
    n_boundary_cons, n_boundary_vars]``.
    """

    n_blocks: int
    block_edge_index: torch.Tensor  # [2, E_block] undirected pairs
    block_edge_attr: torch.Tensor  # [E_block, 4]
    block_features: Optional[torch.Tensor] = None  # [K, d_block]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def n_splits_for(n_total: int, max_size: int) -> int:
    """Minimum number of partitions so each has at most *max_size* nodes."""
    return max(1, math.ceil(n_total / max_size))


def balanced_chunks(total: int, n: int) -> List[int]:
    """Split *total* items into *n* chunks as evenly as possible."""
    base, remainder = divmod(total, n)
    return [base + (1 if i < remainder else 0) for i in range(n)]


# ---------------------------------------------------------------------------
# Vectorised subgraph extraction
# ---------------------------------------------------------------------------


def extract_subgraph(
    cons_ids: torch.Tensor,
    var_ids: torch.Tensor,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> BipartiteNodeData:
    """Build a :class:`BipartiteNodeData` from a subset of constraint and
    variable nodes.  Fully vectorised – no Python loops over edges."""

    sorted_cons = cons_ids.unique(sorted=True)
    sorted_vars = var_ids.unique(sorted=True)

    # Boolean masks for edge filtering
    cons_mask = torch.zeros(c_nodes.size(0), dtype=torch.bool)
    cons_mask[sorted_cons] = True
    var_mask = torch.zeros(v_nodes.size(0), dtype=torch.bool)
    var_mask[sorted_vars] = True
    edge_mask = cons_mask[edge_index[0]] & var_mask[edge_index[1]]

    sub_edge_index = edge_index[:, edge_mask]
    sub_edge_attr = edge_attr[edge_mask]

    # Dense remap tensors: remap[old_id] -> new_local_id
    cons_remap = torch.zeros(c_nodes.size(0), dtype=torch.long)
    cons_remap[sorted_cons] = torch.arange(sorted_cons.size(0), dtype=torch.long)
    var_remap = torch.zeros(v_nodes.size(0), dtype=torch.long)
    var_remap[sorted_vars] = torch.arange(sorted_vars.size(0), dtype=torch.long)

    sub_edge_index = torch.stack(
        [cons_remap[sub_edge_index[0]], var_remap[sub_edge_index[1]]], dim=0
    )

    # Extract and pad node features
    sub_c_nodes = right_pad(c_nodes[sorted_cons], CONS_PAD)
    sub_v_nodes = right_pad(v_nodes[sorted_vars], VARS_PAD)

    sg = BipartiteNodeData(
        constraint_features=sub_c_nodes,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        variable_features=sub_v_nodes,
    )
    sg.orig_cons_ids = sorted_cons
    sg.orig_var_ids = sorted_vars
    return sg


def extract_subgraph_by_constraints(
    cons_ids: torch.Tensor,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> BipartiteNodeData:
    """Extract the subgraph induced by the given constraint nodes (all
    connected variables are included)."""
    cons_mask = torch.zeros(c_nodes.size(0), dtype=torch.bool)
    cons_mask[cons_ids] = True
    var_ids = edge_index[1][cons_mask[edge_index[0]]].unique()
    return extract_subgraph(cons_ids, var_ids, c_nodes, v_nodes, edge_index, edge_attr)


def extract_subgraph_by_variables(
    var_ids: torch.Tensor,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> BipartiteNodeData:
    """Extract the subgraph induced by the given variable nodes (all
    connected constraints are included)."""
    var_mask = torch.zeros(v_nodes.size(0), dtype=torch.bool)
    var_mask[var_ids] = True
    cons_ids = edge_index[0][var_mask[edge_index[1]]].unique()
    return extract_subgraph(cons_ids, var_ids, c_nodes, v_nodes, edge_index, edge_attr)


# ---------------------------------------------------------------------------
# Halo expansion
# ---------------------------------------------------------------------------


def _bfs_expand_bipartite(
    seed_cons: torch.Tensor,
    seed_vars: torch.Tensor,
    edge_index: torch.Tensor,
    n_cons: int,
    n_vars: int,
    hops: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-source BFS on the bipartite graph for *hops* steps.

    Returns ``(all_cons, all_vars)`` — sorted unique constraint and variable
    IDs reached within *hops* steps from the seed nodes.
    """
    cons_visited = torch.zeros(n_cons, dtype=torch.bool)
    if seed_cons.numel() > 0:
        cons_visited[seed_cons] = True
    vars_visited = torch.zeros(n_vars, dtype=torch.bool)
    if seed_vars.numel() > 0:
        vars_visited[seed_vars] = True

    ei_cons = edge_index[0]  # [E] constraint endpoints
    ei_vars = edge_index[1]  # [E] variable endpoints

    for _ in range(hops):
        # Both expansions read from start-of-iteration state
        new_vars = ei_vars[cons_visited[ei_cons]]
        new_cons = ei_cons[vars_visited[ei_vars]]

        cons_visited[new_cons] = True
        vars_visited[new_vars] = True

    return (
        cons_visited.nonzero(as_tuple=False).view(-1),
        vars_visited.nonzero(as_tuple=False).view(-1),
    )


def build_halo_subgraph(
    part: PartitionSpec,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    halo_hops: int,
) -> BipartiteNodeData:
    """Build a halo-expanded subgraph for a partition.

    Expands the owned nodes by *halo_hops* in the bipartite graph, extracts
    the induced subgraph, and marks which local nodes are owned vs. halo.

    Returns a :class:`BipartiteNodeData` with extra attributes:

    - ``owned_cons_mask`` : bool ``[n_cons_local]``, True = owned
    - ``owned_var_mask``  : bool ``[n_vars_local]``, True = owned
    - ``halo_depth``      : scalar tensor
    """
    if halo_hops < 0:
        raise ValueError(f"halo_hops must be >= 0, got {halo_hops}")

    n_cons = c_nodes.size(0)
    n_vars = v_nodes.size(0)

    all_cons, all_vars = _bfs_expand_bipartite(
        part.owned_cons_ids, part.owned_var_ids,
        edge_index, n_cons, n_vars, halo_hops,
    )

    sg = extract_subgraph(all_cons, all_vars, c_nodes, v_nodes, edge_index, edge_attr)

    # Ownership masks in local indexing
    owned_cons_set = torch.zeros(n_cons, dtype=torch.bool)
    owned_cons_set[part.owned_cons_ids] = True
    sg.owned_cons_mask = owned_cons_set[sg.orig_cons_ids]

    owned_vars_set = torch.zeros(n_vars, dtype=torch.bool)
    owned_vars_set[part.owned_var_ids] = True
    sg.owned_var_mask = owned_vars_set[sg.orig_var_ids]

    sg.halo_depth = torch.tensor(halo_hops)

    return sg


def build_halo_subgraphs(
    specs: List[PartitionSpec],
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    halo_hops: int,
) -> List[BipartiteNodeData]:
    """Build halo subgraphs for all partitions."""
    return [
        build_halo_subgraph(
            part, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=halo_hops,
        )
        for part in specs
    ]


# ---------------------------------------------------------------------------
# Splitting functions
# ---------------------------------------------------------------------------


def split_by_constraints(
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    max_subgraph_size: int,
) -> List[BipartiteNodeData]:
    """Partition constraint nodes; connected variables are induced."""
    n_cons = c_nodes.size(0)
    n_parts = n_splits_for(n_cons, max_subgraph_size)
    chunk_sizes = balanced_chunks(n_cons, n_parts)

    subgraphs: List[BipartiteNodeData] = []
    offset = 0
    for size in chunk_sizes:
        if size == 0:
            offset += size
            continue
        cons_ids = torch.arange(offset, offset + size)
        offset += size
        sg = extract_subgraph_by_constraints(
            cons_ids, c_nodes, v_nodes, edge_index, edge_attr
        )
        subgraphs.append(sg)
    return subgraphs


def split_by_variables(
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    max_subgraph_size: int,
) -> List[BipartiteNodeData]:
    """Partition variable nodes; connected constraints are induced."""
    n_vars = v_nodes.size(0)
    n_parts = n_splits_for(n_vars, max_subgraph_size)
    chunk_sizes = balanced_chunks(n_vars, n_parts)

    subgraphs: List[BipartiteNodeData] = []
    offset = 0
    for size in chunk_sizes:
        if size == 0:
            offset += size
            continue
        var_ids = torch.arange(offset, offset + size)
        offset += size
        sg = extract_subgraph_by_variables(
            var_ids, c_nodes, v_nodes, edge_index, edge_attr
        )
        subgraphs.append(sg)
    return subgraphs


def split_bipartite_graph_metis(
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    num_parts: int,
) -> List[PartitionSpec]:
    """Partition a bipartite graph using METIS.

    Both constraint and variable nodes live in a single undirected graph
    (constraints ``0 .. n_cons-1``, variables ``n_cons .. n_total-1``).

    Returns a list of :class:`PartitionSpec` (one per non-empty partition).
    Call :func:`extract_subgraph` separately to build ``BipartiteNodeData``
    objects — this two-step design allows halo expansion between the steps.
    """
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

    if num_parts <= 1:
        return [
            PartitionSpec(
                part_id=0,
                owned_cons_ids=torch.arange(n_cons),
                owned_var_ids=torch.arange(n_vars),
            )
        ]

    # Build adjacency list for pymetis (undirected bipartite graph).
    adjacency: List[List[int]] = [[] for _ in range(n_total)]
    cons_idx = edge_index[0]
    var_idx = edge_index[1]

    for c, v in zip(cons_idx.tolist(), var_idx.tolist()):
        v_shifted = v + n_cons
        adjacency[c].append(v_shifted)
        adjacency[v_shifted].append(c)

    # Deduplicate adjacency lists (pymetis requirement)
    adjacency = [sorted(set(nbrs)) for nbrs in adjacency]

    _, membership = pymetis.part_graph(num_parts, adjacency=adjacency)
    membership = torch.tensor(membership, dtype=torch.long)

    specs: List[PartitionSpec] = []
    for part in range(num_parts):
        part_nodes = (membership == part).nonzero(as_tuple=False).view(-1)
        cons_ids = part_nodes[part_nodes < n_cons]
        var_ids = part_nodes[part_nodes >= n_cons] - n_cons

        # Fallback: if one side is empty, induce from the other (vectorised)
        if cons_ids.numel() == 0 and var_ids.numel() > 0:
            v_mask = torch.zeros(n_vars, dtype=torch.bool)
            v_mask[var_ids] = True
            cons_ids = edge_index[0][v_mask[edge_index[1]]].unique()
        elif var_ids.numel() == 0 and cons_ids.numel() > 0:
            c_mask = torch.zeros(n_cons, dtype=torch.bool)
            c_mask[cons_ids] = True
            var_ids = edge_index[1][c_mask[edge_index[0]]].unique()

        if cons_ids.numel() == 0 or var_ids.numel() == 0:
            continue

        specs.append(
            PartitionSpec(
                part_id=part,
                owned_cons_ids=cons_ids.sort().values,
                owned_var_ids=var_ids.sort().values,
            )
        )

    return specs


def split_and_extract_metis(
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    num_parts: int,
) -> List[BipartiteNodeData]:
    """Convenience wrapper: partition with METIS and immediately extract
    subgraphs."""
    specs = split_bipartite_graph_metis(
        c_nodes, v_nodes, edge_index, edge_attr, num_parts=num_parts
    )
    return [
        extract_subgraph(
            s.owned_cons_ids, s.owned_var_ids, c_nodes, v_nodes, edge_index, edge_attr
        )
        for s in specs
    ]


# ---------------------------------------------------------------------------
# Coupling diagnostics
# ---------------------------------------------------------------------------


def compute_coupling_diagnostics(
    subgraphs: List[BipartiteNodeData],
    n_cons: int,
    n_vars: int,
    edge_index: torch.Tensor,
) -> CouplingDiagnostics:
    """Compute coupling statistics for a set of subgraphs against the
    original graph.

    Delegates to :func:`compute_coupling_diagnostics_from_specs` internally.
    """
    specs = [
        PartitionSpec(
            part_id=k,
            owned_cons_ids=sg.orig_cons_ids,
            owned_var_ids=sg.orig_var_ids,
        )
        for k, sg in enumerate(subgraphs)
    ]
    return compute_coupling_diagnostics_from_specs(
        specs, n_cons, n_vars, edge_index,
    )


def compute_coupling_diagnostics_from_specs(
    specs: List[PartitionSpec],
    n_cons: int,
    n_vars: int,
    edge_index: torch.Tensor,
    *,
    cons_is_boundary: Optional[torch.Tensor] = None,
    vars_is_boundary: Optional[torch.Tensor] = None,
) -> CouplingDiagnostics:
    """Compute coupling statistics from partition specs.

    Parameters
    ----------
    specs : list[PartitionSpec]
        One per partition.
    n_cons, n_vars : int
        Total number of constraints / variables in the full graph.
    edge_index : [2, E]
        Full-graph edge index.
    cons_is_boundary, vars_is_boundary : optional bool tensors [n_cons] / [n_vars]
        Pre-computed boundary masks.  If *None*, boundary nodes are derived
        from the partition assignments (nodes with at least one cross-partition
        edge).
    """
    n_edges = edge_index.size(1)

    # Map every node to its owning partition
    var_to_block = torch.full((n_vars,), -1, dtype=torch.long)
    cons_to_block = torch.full((n_cons,), -1, dtype=torch.long)
    for spec in specs:
        cons_to_block[spec.owned_cons_ids] = spec.part_id
        var_to_block[spec.owned_var_ids] = spec.part_id

    # --- Coupling constraints (touch >1 block) ---
    edge_blocks = var_to_block[edge_index[1]]
    cons_block_pair = torch.stack([edge_index[0], edge_blocks], dim=0)
    assigned_mask = edge_blocks >= 0
    cons_block_pair = cons_block_pair[:, assigned_mask]
    unique_pairs = cons_block_pair.T.unique(dim=0)
    cons_ids_in_pairs, blocks_per_cons_counts = unique_pairs[:, 0].unique(
        return_counts=True,
    )
    blocks_per_constraint = torch.zeros(n_cons, dtype=torch.long)
    blocks_per_constraint[cons_ids_in_pairs] = blocks_per_cons_counts

    coupling_mask = blocks_per_constraint > 1
    n_coupling = int(coupling_mask.sum().item())
    frac_coupling = n_coupling / max(n_cons, 1)
    avg_blocks = float(blocks_per_constraint.float().mean().item())

    # --- Edge cut ---
    edge_cons_block = cons_to_block[edge_index[0]]
    edge_var_block = var_to_block[edge_index[1]]
    both_assigned = (edge_cons_block >= 0) & (edge_var_block >= 0)
    cross_mask = (edge_cons_block != edge_var_block) & both_assigned
    edge_cut_count = int(cross_mask.sum().item())

    # --- Boundary nodes ---
    if cons_is_boundary is not None and vars_is_boundary is not None:
        n_bnd_c = int(cons_is_boundary.sum().item())
        n_bnd_v = int(vars_is_boundary.sum().item())
    else:
        bnd_cons = torch.zeros(n_cons, dtype=torch.bool)
        bnd_vars = torch.zeros(n_vars, dtype=torch.bool)
        bnd_cons[edge_index[0][cross_mask]] = True
        bnd_vars[edge_index[1][cross_mask]] = True
        n_bnd_c = int(bnd_cons.sum().item())
        n_bnd_v = int(bnd_vars.sum().item())

    return CouplingDiagnostics(
        n_coupling_constraints=n_coupling,
        n_total_constraints=n_cons,
        coupling_fraction=frac_coupling,
        avg_blocks_per_constraint=avg_blocks,
        edge_cut_count=edge_cut_count,
        n_total_edges=n_edges,
        edge_cut_fraction=edge_cut_count / max(n_edges, 1),
        n_boundary_cons=n_bnd_c,
        n_boundary_vars=n_bnd_v,
        boundary_cons_fraction=n_bnd_c / max(n_cons, 1),
        boundary_vars_fraction=n_bnd_v / max(n_vars, 1),
        n_total_vars=n_vars,
    )


def compute_halo_expansion_ratio(
    n_owned_cons: int,
    n_owned_vars: int,
    n_halo_cons: int,
    n_halo_vars: int,
) -> float:
    """Halo expansion ratio: ``(owned + halo) / owned``.

    Returns 1.0 when halo is zero, ``inf`` if partition has 0 owned nodes.
    """
    owned = n_owned_cons + n_owned_vars
    if owned == 0:
        return float("inf")
    return (owned + n_halo_cons + n_halo_vars) / owned


# ---------------------------------------------------------------------------
# Block graph construction
# ---------------------------------------------------------------------------


def build_block_graph(
    partitions: List[PartitionSpec],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    n_cons: int,
    n_vars: int,
) -> BlockGraph:
    """Build the block interaction graph over partitions.

    Adds an undirected block edge ``(k, l)`` whenever the original graph has
    at least one edge crossing partitions *k* and *l*.

    Returns a :class:`BlockGraph` with ``block_features=None`` (call
    :func:`compute_block_features` to populate it).
    """
    K = len(partitions)

    if K <= 1:
        return BlockGraph(
            n_blocks=K,
            block_edge_index=torch.zeros(2, 0, dtype=torch.long),
            block_edge_attr=torch.zeros(0, 4),
        )

    # Remap part_id → contiguous 0..K-1
    id_to_idx = {p.part_id: i for i, p in enumerate(partitions)}

    cons_to_block = torch.full((n_cons,), -1, dtype=torch.long)
    var_to_block = torch.full((n_vars,), -1, dtype=torch.long)
    for p in partitions:
        idx = id_to_idx[p.part_id]
        cons_to_block[p.owned_cons_ids] = idx
        var_to_block[p.owned_var_ids] = idx

    # Per-edge block assignments
    edge_cons_block = cons_to_block[edge_index[0]]
    edge_var_block = var_to_block[edge_index[1]]

    both_assigned = (edge_cons_block >= 0) & (edge_var_block >= 0)
    cross_mask = (edge_cons_block != edge_var_block) & both_assigned

    if not cross_mask.any():
        return BlockGraph(
            n_blocks=K,
            block_edge_index=torch.zeros(2, 0, dtype=torch.long),
            block_edge_attr=torch.zeros(0, 4),
        )

    # Canonical (min, max) block pairs for cross edges
    cross_cons_b = edge_cons_block[cross_mask]
    cross_var_b = edge_var_block[cross_mask]
    pair_lo = torch.min(cross_cons_b, cross_var_b)
    pair_hi = torch.max(cross_cons_b, cross_var_b)
    pairs = torch.stack([pair_lo, pair_hi], dim=0)  # [2, n_cross]

    unique_pairs, inverse = torch.unique(pairs, dim=1, return_inverse=True)
    n_block_edges = unique_pairs.size(1)

    # Cross-edge indices into the original edge_index
    cross_indices = cross_mask.nonzero(as_tuple=False).view(-1)

    # Edge features per block pair
    feat_cut_count = torch.zeros(n_block_edges)
    feat_sum_abs_coeff = torch.zeros(n_block_edges)
    feat_bnd_cons = torch.zeros(n_block_edges)
    feat_bnd_vars = torch.zeros(n_block_edges)

    # Cut count: scatter_add ones
    feat_cut_count.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float))

    # Sum |coefficients|
    cross_attr = edge_attr[cross_indices]  # [n_cross, d_edge]
    abs_sum_per_edge = cross_attr.abs().sum(dim=-1)  # [n_cross]
    feat_sum_abs_coeff.scatter_add_(0, inverse, abs_sum_per_edge)

    # Boundary cons/vars per block pair (count unique nodes per pair)
    cross_cons_ids = edge_index[0][cross_indices]  # original cons IDs
    cross_var_ids = edge_index[1][cross_indices]  # original var IDs

    for pair_idx in range(n_block_edges):
        mask = inverse == pair_idx
        feat_bnd_cons[pair_idx] = cross_cons_ids[mask].unique().size(0)
        feat_bnd_vars[pair_idx] = cross_var_ids[mask].unique().size(0)

    block_edge_attr = torch.stack(
        [feat_cut_count, feat_sum_abs_coeff, feat_bnd_cons, feat_bnd_vars], dim=1
    )  # [E_block, 4]

    return BlockGraph(
        n_blocks=K,
        block_edge_index=unique_pairs,
        block_edge_attr=block_edge_attr,
    )


def compute_block_features(
    partitions: List[PartitionSpec],
    c_emb: torch.Tensor,
    v_emb: torch.Tensor,
    *,
    include_metadata: bool = False,
    halo_subgraphs: Optional[List["BipartiteNodeData"]] = None,
    cons_is_boundary: Optional[torch.Tensor] = None,
    vars_is_boundary: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute block features by pooling owned-node embeddings per partition.

    Parameters
    ----------
    partitions : list[PartitionSpec]
    c_emb : [n_cons, d]  global-indexed constraint embeddings
    v_emb : [n_vars, d]  global-indexed variable embeddings
    include_metadata : bool
        Append 4 scalar features: owned var count, owned cons count,
        halo expansion ratio, boundary fraction.
    halo_subgraphs : optional list of BipartiteNodeData
        Required when ``include_metadata=True`` (for halo expansion ratio).
    cons_is_boundary, vars_is_boundary : optional bool tensors
        Required when ``include_metadata=True`` (for boundary fraction).

    Returns
    -------
    block_features : [K, d_block]
        ``d_block = 4*d`` (or ``4*d + 4`` with metadata).
    """
    d = v_emb.size(1)
    rows = []

    for i, part in enumerate(partitions):
        v_owned = v_emb[part.owned_var_ids]
        c_owned = c_emb[part.owned_cons_ids]

        mean_v = v_owned.mean(dim=0)
        max_v = v_owned.max(dim=0).values
        mean_c = c_owned.mean(dim=0)
        max_c = c_owned.max(dim=0).values

        r_k = torch.cat([mean_v, max_v, mean_c, max_c])  # [4*d]

        if include_metadata:
            n_ov = float(part.owned_var_ids.numel())
            n_oc = float(part.owned_cons_ids.numel())

            # Halo expansion ratio
            if halo_subgraphs is not None:
                sg = halo_subgraphs[i]
                n_total_c = sg.constraint_features.size(0)
                n_total_v = sg.variable_features.size(0)
                exp_ratio = compute_halo_expansion_ratio(
                    int(n_oc), int(n_ov),
                    n_total_c - int(n_oc), n_total_v - int(n_ov),
                )
            else:
                exp_ratio = 1.0

            # Boundary fraction
            if cons_is_boundary is not None and vars_is_boundary is not None:
                n_bnd_c = cons_is_boundary[part.owned_cons_ids].sum().item()
                n_bnd_v = vars_is_boundary[part.owned_var_ids].sum().item()
                bnd_frac = (n_bnd_c + n_bnd_v) / max(n_oc + n_ov, 1.0)
            else:
                bnd_frac = 0.0

            meta = torch.tensor([n_ov, n_oc, exp_ratio, bnd_frac])
            r_k = torch.cat([r_k, meta])

        rows.append(r_k)

    return torch.stack(rows, dim=0)  # [K, d_block]


def log_block_graph_diagnostics(bg: BlockGraph) -> None:
    """Log quality metrics for a :class:`BlockGraph`."""
    n_edges = bg.block_edge_index.size(1)
    avg_deg = 2.0 * n_edges / max(bg.n_blocks, 1)

    logger.info("Block graph: %d nodes, %d edges, avg degree %.2f",
                bg.n_blocks, n_edges, avg_deg)

    if n_edges > 0:
        attr = bg.block_edge_attr  # [E, 4]
        names = ["cut_count", "sum_abs_coeff", "bnd_cons", "bnd_vars"]
        for j, name in enumerate(names):
            col = attr[:, j]
            logger.info(
                "  %s: min=%.2f  max=%.2f  mean=%.2f",
                name, col.min().item(), col.max().item(), col.mean().item(),
            )

    if bg.block_features is not None:
        logger.info("Block features shape: %s", list(bg.block_features.shape))


# ---------------------------------------------------------------------------
# Boundary identification
# ---------------------------------------------------------------------------


def identify_boundary_nodes(
    specs: List[PartitionSpec],
    edge_index: torch.Tensor,
    n_cons: int,
    n_vars: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identify boundary nodes — owned nodes with cross-partition edges.

    A node is a boundary node if, in the original full graph, it has at least
    one neighbour assigned to a different partition.

    Returns
    -------
    cons_is_boundary : [n_cons] bool
    vars_is_boundary : [n_vars] bool
    """
    cons_to_part = torch.full((n_cons,), -1, dtype=torch.long)
    vars_to_part = torch.full((n_vars,), -1, dtype=torch.long)

    for spec in specs:
        cons_to_part[spec.owned_cons_ids] = spec.part_id
        vars_to_part[spec.owned_var_ids] = spec.part_id

    cons_idx = edge_index[0]  # [E]
    vars_idx = edge_index[1]  # [E]

    cross_mask = cons_to_part[cons_idx] != vars_to_part[vars_idx]

    cons_is_boundary = torch.zeros(n_cons, dtype=torch.bool)
    vars_is_boundary = torch.zeros(n_vars, dtype=torch.bool)

    cons_is_boundary[cons_idx[cross_mask]] = True
    vars_is_boundary[vars_idx[cross_mask]] = True

    return cons_is_boundary, vars_is_boundary


def compute_boundary_features(
    specs: List[PartitionSpec],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    n_cons: int,
    n_vars: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-node boundary features for the composer model.

    For each node computes five features (count-like features are normalised
    with :func:`torch.log1p`):

    ====  ===========================  ===========
    idx   feature                      normalisation
    ====  ===========================  ===========
    0     total degree                 log1p
    1     number of cross-partition edges  log1p
    2     cut fraction                 raw [0, 1]
    3     sum |coeff| on cut edges     log1p
    4     is_boundary indicator        0 / 1
    ====  ===========================  ===========

    Returns
    -------
    cons_boundary : [n_cons, 5]
    vars_boundary : [n_vars, 5]
    """
    # Build partition-assignment maps
    cons_to_block = torch.full((n_cons,), -1, dtype=torch.long)
    var_to_block = torch.full((n_vars,), -1, dtype=torch.long)

    id_to_idx = {p.part_id: i for i, p in enumerate(specs)}
    for p in specs:
        idx = id_to_idx[p.part_id]
        cons_to_block[p.owned_cons_ids] = idx
        var_to_block[p.owned_var_ids] = idx

    cons_idx = edge_index[0]  # [E]
    vars_idx = edge_index[1]  # [E]

    both_assigned = (cons_to_block[cons_idx] >= 0) & (var_to_block[vars_idx] >= 0)
    cross_mask = (cons_to_block[cons_idx] != var_to_block[vars_idx]) & both_assigned

    ones = torch.ones(edge_index.size(1))

    # --- Constraint nodes ---
    cons_total_deg = torch.zeros(n_cons)
    cons_total_deg.scatter_add_(0, cons_idx, ones)

    cons_cross_deg = torch.zeros(n_cons)
    cons_cross_deg.scatter_add_(0, cons_idx[cross_mask], ones[cross_mask])

    cons_abs_coeff = torch.zeros(n_cons)
    cross_abs = edge_attr[cross_mask].abs().sum(dim=-1)  # [n_cross]
    cons_abs_coeff.scatter_add_(0, cons_idx[cross_mask], cross_abs)

    cons_cut_frac = cons_cross_deg / cons_total_deg.clamp(min=1.0)
    cons_is_bnd = (cons_cross_deg > 0).float()

    cons_boundary = torch.stack([
        cons_total_deg.log1p(),
        cons_cross_deg.log1p(),
        cons_cut_frac,
        cons_abs_coeff.log1p(),
        cons_is_bnd,
    ], dim=1)  # [n_cons, 5]

    # --- Variable nodes ---
    vars_total_deg = torch.zeros(n_vars)
    vars_total_deg.scatter_add_(0, vars_idx, ones)

    vars_cross_deg = torch.zeros(n_vars)
    vars_cross_deg.scatter_add_(0, vars_idx[cross_mask], ones[cross_mask])

    vars_abs_coeff = torch.zeros(n_vars)
    vars_abs_coeff.scatter_add_(0, vars_idx[cross_mask], cross_abs)

    vars_cut_frac = vars_cross_deg / vars_total_deg.clamp(min=1.0)
    vars_is_bnd = (vars_cross_deg > 0).float()

    vars_boundary = torch.stack([
        vars_total_deg.log1p(),
        vars_cross_deg.log1p(),
        vars_cut_frac,
        vars_abs_coeff.log1p(),
        vars_is_bnd,
    ], dim=1)  # [n_vars, 5]

    return cons_boundary, vars_boundary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_partition(
    specs: List[PartitionSpec],
    n_cons: int,
    n_vars: int,
) -> None:
    """Check disjointness and full coverage of the partition.

    Raises :class:`ValueError` if any constraint or variable is duplicated
    across partitions, missing from all partitions, or if any partition is
    empty.
    """
    if not specs:
        raise ValueError("Empty partition list")

    all_cons = torch.cat([s.owned_cons_ids for s in specs])
    all_vars = torch.cat([s.owned_var_ids for s in specs])

    # Disjointness
    if all_cons.unique().size(0) != all_cons.size(0):
        raise ValueError(
            f"Constraint IDs overlap across partitions "
            f"({all_cons.size(0)} total vs {all_cons.unique().size(0)} unique)"
        )
    if all_vars.unique().size(0) != all_vars.size(0):
        raise ValueError(
            f"Variable IDs overlap across partitions "
            f"({all_vars.size(0)} total vs {all_vars.unique().size(0)} unique)"
        )

    # Full coverage
    if all_cons.unique().size(0) != n_cons:
        missing = set(range(n_cons)) - set(all_cons.tolist())
        raise ValueError(
            f"Not all constraints covered: {len(missing)} missing "
            f"(expected {n_cons}, got {all_cons.unique().size(0)})"
        )
    if all_vars.unique().size(0) != n_vars:
        missing = set(range(n_vars)) - set(all_vars.tolist())
        raise ValueError(
            f"Not all variables covered: {len(missing)} missing "
            f"(expected {n_vars}, got {all_vars.unique().size(0)})"
        )

    # Non-empty partitions
    for s in specs:
        if s.owned_cons_ids.numel() == 0 or s.owned_var_ids.numel() == 0:
            raise ValueError(
                f"Partition {s.part_id} is empty "
                f"(cons={s.owned_cons_ids.numel()}, vars={s.owned_var_ids.numel()})"
            )
