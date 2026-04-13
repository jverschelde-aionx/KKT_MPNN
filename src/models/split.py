from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from data.common import CONS_PAD, VARS_PAD
from data.datasets import BipartiteNodeData, right_pad

logger = logging.getLogger(__name__)

from typing import Any, List, Optional, Tuple

from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected

from data.split import (
    SplitInstanceBatch,
    SplitInstanceData,
    SplitPartitionData,
    SplitViewMasks,
)
from models.base import LeJepaEncoderModule
from models.gnn import GNNEncoder, GNNPolicy

"""
Decomposition utilities for bipartite MILP graphs.
===================================================
Consolidates splitting, extraction, coupling diagnostics, and validation
logic previously duplicated across job scripts.
"""

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
        part.owned_cons_ids,
        part.owned_var_ids,
        edge_index,
        n_cons,
        n_vars,
        halo_hops,
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
            part,
            c_nodes,
            v_nodes,
            edge_index,
            edge_attr,
            halo_hops=halo_hops,
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

    # First pass: collect raw METIS assignments per partition.
    raw_cons: List[torch.Tensor] = []
    raw_vars: List[torch.Tensor] = []
    for part in range(num_parts):
        part_nodes = (membership == part).nonzero(as_tuple=False).view(-1)
        raw_cons.append(part_nodes[part_nodes < n_cons])
        raw_vars.append(part_nodes[part_nodes >= n_cons] - n_cons)

    # Second pass: merge single-sided partitions into the neighbour partition
    # that owns the most of their adjacent nodes on the missing side.
    for part in range(num_parts):
        cons_ids = raw_cons[part]
        var_ids = raw_vars[part]
        if cons_ids.numel() > 0 and var_ids.numel() > 0:
            continue  # both sides present, nothing to merge
        if cons_ids.numel() == 0 and var_ids.numel() == 0:
            continue  # empty partition

        # Find the partition that owns most of the neighbours on the missing side.
        nodes = cons_ids if cons_ids.numel() > 0 else (var_ids + n_cons)
        # Gather neighbour nodes from adjacency
        nbr_ids = []
        for n_id in nodes.tolist():
            nbr_ids.extend(adjacency[n_id])
        if not nbr_ids:
            continue
        nbr_parts = membership[torch.tensor(nbr_ids, dtype=torch.long)]
        # Pick the most frequent neighbouring partition (excluding self)
        counts = torch.zeros(num_parts, dtype=torch.long)
        for p in nbr_parts.tolist():
            if p != part:
                counts[p] += 1
        target = int(counts.argmax())
        if counts[target] == 0:
            continue  # all neighbours in same partition — skip

        # Merge: donate our nodes to the target partition
        raw_cons[target] = torch.cat([raw_cons[target], cons_ids])
        raw_vars[target] = torch.cat([raw_vars[target], var_ids])
        raw_cons[part] = torch.tensor([], dtype=torch.long)
        raw_vars[part] = torch.tensor([], dtype=torch.long)

    specs: List[PartitionSpec] = []
    for part in range(num_parts):
        cons_ids = raw_cons[part]
        var_ids = raw_vars[part]
        if cons_ids.numel() == 0 or var_ids.numel() == 0:
            continue
        specs.append(
            PartitionSpec(
                part_id=part,
                owned_cons_ids=cons_ids.unique().sort().values,
                owned_var_ids=var_ids.unique().sort().values,
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
        specs,
        n_cons,
        n_vars,
        edge_index,
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
                    int(n_oc),
                    int(n_ov),
                    n_total_c - int(n_oc),
                    n_total_v - int(n_ov),
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

    logger.info(
        "Block graph: %d nodes, %d edges, avg degree %.2f",
        bg.n_blocks,
        n_edges,
        avg_deg,
    )

    if n_edges > 0:
        attr = bg.block_edge_attr  # [E, 4]
        names = ["cut_count", "sum_abs_coeff", "bnd_cons", "bnd_vars"]
        for j, name in enumerate(names):
            col = attr[:, j]
            logger.info(
                "  %s: min=%.2f  max=%.2f  mean=%.2f",
                name,
                col.min().item(),
                col.max().item(),
                col.mean().item(),
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

    cons_boundary = torch.stack(
        [
            cons_total_deg.log1p(),
            cons_cross_deg.log1p(),
            cons_cut_frac,
            cons_abs_coeff.log1p(),
            cons_is_bnd,
        ],
        dim=1,
    )  # [n_cons, 5]

    # --- Variable nodes ---
    vars_total_deg = torch.zeros(n_vars)
    vars_total_deg.scatter_add_(0, vars_idx, ones)

    vars_cross_deg = torch.zeros(n_vars)
    vars_cross_deg.scatter_add_(0, vars_idx[cross_mask], ones[cross_mask])

    vars_abs_coeff = torch.zeros(n_vars)
    vars_abs_coeff.scatter_add_(0, vars_idx[cross_mask], cross_abs)

    vars_cut_frac = vars_cross_deg / vars_total_deg.clamp(min=1.0)
    vars_is_bnd = (vars_cross_deg > 0).float()

    vars_boundary = torch.stack(
        [
            vars_total_deg.log1p(),
            vars_cross_deg.log1p(),
            vars_cut_frac,
            vars_abs_coeff.log1p(),
            vars_is_bnd,
        ],
        dim=1,
    )  # [n_vars, 5]

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


class BlockGNN(nn.Module):
    """
    Small block-level GNN over the partition graph.

    block_features: [K, d_block]
    block_edge_index: [2, E_block]
    block_edge_attr: [E_block, d_edge]
    """

    def __init__(
        self,
        *,
        d_block: int,
        d_hidden: int = 128,
        d_z: int = 128,
        d_edge: int = 4,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if d_hidden % heads != 0:
            raise ValueError(f"d_hidden={d_hidden} must be divisible by heads={heads}")

        self.input_proj = nn.Sequential(
            nn.LayerNorm(d_block),
            nn.Linear(d_block, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(d_edge, d_hidden),
            nn.ReLU(),
        )

        self.conv1 = GATv2Conv(
            in_channels=d_hidden,
            out_channels=d_hidden // heads,
            heads=heads,
            concat=True,
            edge_dim=d_hidden,
            dropout=dropout,
            add_self_loops=False,
        )
        self.conv2 = GATv2Conv(
            in_channels=d_hidden,
            out_channels=d_hidden // heads,
            heads=heads,
            concat=True,
            edge_dim=d_hidden,
            dropout=dropout,
            add_self_loops=False,
        )

        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)

        self.output_proj = nn.Sequential(
            nn.Linear(d_hidden, d_z),
            nn.ReLU(),
            nn.Linear(d_z, d_z),
        )

    def forward(
        self,
        block_features: torch.Tensor,
        block_edge_index: torch.Tensor,
        block_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns z_blocks: [K, d_z]
        """
        x = self.input_proj(block_features)

        if block_edge_index.numel() == 0:
            return self.output_proj(x)

        # Make undirected explicitly
        ei, ea = to_undirected(block_edge_index, block_edge_attr, reduce="mean")
        ea = self.edge_proj(ea)

        h = self.conv1(x, ei, ea)
        h = self.norm1(h + x)

        h2 = self.conv2(h, ei, ea)
        h = self.norm2(h2 + h)

        return self.output_proj(h)


class ComposerMLP(nn.Module):
    """
    Per-node composer head:
      [h_sub ; z_block ; boundary_feat] -> h_hat
    """

    def __init__(
        self,
        *,
        d_sub: int,
        d_z: int,
        d_boundary: int,
        d_out: int,
        d_hidden: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        d_in = d_sub + d_z + d_boundary

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(
        self,
        h_sub: torch.Tensor,
        z_block: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([h_sub, z_block, boundary_feat], dim=-1)
        return self.net(x)


class BlockGNNComposer(nn.Module):
    """
    Supports three modes:
      - local MLP only: use_block_context=False, use_block_gnn=False
      - pooled block context: use_block_context=True, use_block_gnn=False
      - block GNN composer: use_block_context=True, use_block_gnn=True
    """

    def __init__(
        self,
        *,
        d_sub: int,
        d_block: int,
        d_z: int,
        d_boundary: int,
        d_mlp_hidden: int = 256,
        heads: int = 4,
        dropout: float = 0.05,
        use_block_context: bool = True,
        use_block_gnn: bool = True,
        d_edge: int = 4,
    ) -> None:
        super().__init__()

        self.use_block_context = use_block_context
        self.use_block_gnn = use_block_gnn
        self.d_z = d_z

        self.context_proj = None
        self.block_gnn = None

        if self.use_block_context and not self.use_block_gnn:
            self.context_proj = nn.Sequential(
                nn.LayerNorm(d_block),
                nn.Linear(d_block, d_z),
                nn.ReLU(),
                nn.Linear(d_z, d_z),
            )

        if self.use_block_context and self.use_block_gnn:
            self.block_gnn = BlockGNN(
                d_block=d_block,
                d_hidden=max(d_z, 128),
                d_z=d_z,
                d_edge=d_edge,
                heads=heads,
                dropout=dropout,
            )

        self.cons_composer = ComposerMLP(
            d_sub=d_sub,
            d_z=d_z,
            d_boundary=d_boundary,
            d_out=d_sub,
            d_hidden=d_mlp_hidden,
            dropout=dropout,
        )
        self.var_composer = ComposerMLP(
            d_sub=d_sub,
            d_z=d_z,
            d_boundary=d_boundary,
            d_out=d_sub,
            d_hidden=d_mlp_hidden,
            dropout=dropout,
        )

    def _block_context(
        self,
        block_features: torch.Tensor,
        block_edge_index: torch.Tensor,
        block_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        K = block_features.size(0)
        dev = block_features.device

        if not self.use_block_context:
            return torch.zeros((K, self.d_z), device=dev, dtype=block_features.dtype)

        if self.use_block_gnn:
            return self.block_gnn(block_features, block_edge_index, block_edge_attr)

        return self.context_proj(block_features)

    def forward(
        self,
        *,
        block_features: torch.Tensor,  # [K, d_block]
        block_edge_index: torch.Tensor,  # [2, E_block]
        block_edge_attr: torch.Tensor,  # [E_block, d_edge]
        cons_block_id: torch.Tensor,  # [n_cons]
        vars_block_id: torch.Tensor,  # [n_vars]
        c_sub: torch.Tensor,  # [n_cons, d_sub]
        v_sub: torch.Tensor,  # [n_vars, d_sub]
        cons_boundary_feat: torch.Tensor,  # [n_cons, d_boundary]
        vars_boundary_feat: torch.Tensor,  # [n_vars, d_boundary]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_blocks = self._block_context(
            block_features, block_edge_index, block_edge_attr
        )

        c_ctx = z_blocks[cons_block_id]
        v_ctx = z_blocks[vars_block_id]

        c_hat = self.cons_composer(c_sub, c_ctx, cons_boundary_feat)
        v_hat = self.var_composer(v_sub, v_ctx, vars_boundary_feat)
        return c_hat, v_hat


class SplitBlockBiJepaPolicy(LeJepaEncoderModule):
    """
    Self-contained split architecture:
      - input is already pre-split into halo subgraphs + block graph
      - no full graph is needed during training/inference
      - BIJEPA is applied to composed node embeddings
    """

    @staticmethod
    def add_args(parser):
        GNNPolicy.add_args(parser)

        group = parser.add_argument_group("split_bijepa")
        group.add_argument("--num_blocks", type=int, default=5)
        group.add_argument("--halo_hops", type=int, default=0)
        group.add_argument("--composer_d_z", type=int, default=128)
        group.add_argument("--composer_hidden", type=int, default=256)
        group.add_argument("--composer_heads", type=int, default=4)
        group.add_argument("--composer_dropout", type=float, default=0.05)
        group.add_argument("--use_block_context", type=int, default=1)
        group.add_argument("--use_block_gnn", type=int, default=1)
        group.add_argument("--skip_composer", type=int, default=0,
                           help="Skip composer entirely (raw split baseline). 1=skip, 0=use composer.")

    @staticmethod
    def name(args):
        name = "split_block_bijepa"
        name += f"-dim={args.embedding_size}"
        name += f"-blocks={args.num_blocks}"
        name += f"-halo={args.halo_hops}"
        name += f"-dz={args.composer_d_z}"
        name += f"-ctx={args.use_block_context}"
        name += f"-bg={args.use_block_gnn}"
        name += f"-lmask={args.lejepa_local_mask}"
        name += f"-gmask={args.lejepa_global_mask}"
        return name

    def __init__(self, args):
        super().__init__(args.sigreg_slices, args.sigreg_points)

        # JEPA view config
        self.n_global_views = args.lejepa_n_global_views
        self.n_local_views = args.lejepa_n_local_views
        self.local_mask = args.lejepa_local_mask
        self.global_mask = args.lejepa_global_mask
        self.local_edge_mask = getattr(args, "lejepa_local_edge_mask", 0.20)
        self.global_edge_mask = getattr(args, "lejepa_global_edge_mask", 0.05)

        # shared subgraph encoder
        self._encoder = GNNEncoder(
            cons_nfeats=args.cons_nfeats,
            var_nfeats=args.var_nfeats,
            edge_nfeats=args.edge_nfeats,
            embedding_size=args.embedding_size,
            num_emb_type=args.num_emb_type,
            num_emb_freqs=args.num_emb_freqs,
            num_emb_bins=args.num_emb_bins,
            lejepa_embed_dim=args.lejepa_embed_dim,
            dropout=args.dropout,
            bipartite_conv=args.bipartite_conv,
            attn_heads=args.attn_heads,
        )
        self.d_sub = args.embedding_size

        # composer
        self.skip_composer = bool(getattr(args, "skip_composer", 0))
        self.composer = BlockGNNComposer(
            d_sub=self.d_sub,
            d_block=4 * self.d_sub,
            d_z=args.composer_d_z,
            d_boundary=5,
            d_mlp_hidden=args.composer_hidden,
            heads=args.composer_heads,
            dropout=args.composer_dropout,
            use_block_context=bool(args.use_block_context),
            use_block_gnn=bool(args.use_block_gnn),
            d_edge=4,  # assumes build_block_graph returns 4 edge features
        )

        # heads for later fine-tuning
        d_out = self.d_sub
        self.var_head = nn.Sequential(
            nn.Linear(d_out, d_out), nn.ReLU(), nn.Linear(d_out, 1, bias=False)
        )
        self.cons_head = nn.Sequential(
            nn.Linear(d_out, d_out), nn.ReLU(), nn.Linear(d_out, 1, bias=False)
        )
        self.lambda_act = nn.Softplus()

    @property
    def encoder(self) -> GNNEncoder:
        return self._encoder

    @encoder.setter
    def encoder(self, module):
        self._encoder = module

    # ------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------

    def _encode_block_slot_batched(
        self,
        instances: List[SplitInstanceData],
        k: int,
        device: torch.device,
        block_masks: Optional[List[tuple]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Batch all block-k subgraphs across instances and encode in one forward pass.
        block_masks: optional list of (cons_mask, var_mask, edge_mask) per instance.
        Returns per-instance (c_emb, v_emb) pairs.
        """
        c_features_list = []
        v_features_list = []
        edge_index_list = []
        edge_attr_list = []
        c_mask_list = []
        v_mask_list = []
        e_mask_list = []
        c_counts = []
        v_counts = []

        c_offset = 0
        v_offset = 0
        for idx, inst in enumerate(instances):
            g = inst.partitions[k].graph
            nc = g.constraint_features.size(0)
            nv = g.variable_features.size(0)

            c_features_list.append(g.constraint_features)
            v_features_list.append(g.variable_features)
            ei = g.edge_index.clone()
            ei[0] += c_offset
            ei[1] += v_offset
            edge_index_list.append(ei)
            edge_attr_list.append(g.edge_attr)

            if block_masks is not None:
                cm, vm, em = block_masks[idx]
                c_mask_list.append(cm)
                v_mask_list.append(vm)
                e_mask_list.append(em)

            c_counts.append(nc)
            v_counts.append(nv)
            c_offset += nc
            v_offset += nv

        c_all = torch.cat(c_features_list, dim=0).to(device)
        v_all = torch.cat(v_features_list, dim=0).to(device)
        ei_all = torch.cat(edge_index_list, dim=1).to(device)
        ea_all = torch.cat(edge_attr_list, dim=0).to(device)

        if block_masks is not None:
            cm_all = torch.cat(c_mask_list, dim=0).to(device)
            vm_all = torch.cat(v_mask_list, dim=0).to(device)
            em_all = torch.cat(e_mask_list, dim=0).to(device)
        else:
            cm_all = None
            vm_all = None
            em_all = None

        c_enc, v_enc = self.encoder.encode_nodes(
            c_all, ei_all, ea_all, v_all,
            cons_mask=cm_all, var_mask=vm_all, edge_mask=em_all,
        )

        # unbatch
        results = []
        c_pos = 0
        v_pos = 0
        for nc, nv in zip(c_counts, v_counts):
            results.append((c_enc[c_pos : c_pos + nc], v_enc[v_pos : v_pos + nv]))
            c_pos += nc
            v_pos += nv
        return results

    def _scatter_owned_batched(
        self,
        instances: List[SplitInstanceData],
        all_partition_embs: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Scatter owned embeddings back to global node positions for each instance.
        all_partition_embs[k][i] = (c_emb, v_emb) for block k, instance i.
        """
        results = []
        K = len(all_partition_embs)

        for i, inst in enumerate(instances):
            c_global = torch.zeros((inst.n_cons, self.d_sub), device=device)
            v_global = torch.zeros((inst.n_vars, self.d_sub), device=device)

            for k in range(K):
                part = inst.partitions[k]
                c_loc, v_loc = all_partition_embs[k][i]

                owned_c_local = part.owned_cons_local.to(device)
                owned_v_local = part.owned_var_local.to(device)
                c_owned = c_loc[owned_c_local]
                v_owned = v_loc[owned_v_local]

                c_global.index_copy_(
                    0, part.orig_cons_ids[part.owned_cons_local].to(device), c_owned
                )
                v_global.index_copy_(
                    0, part.orig_var_ids[part.owned_var_local].to(device), v_owned
                )
            results.append((c_global, v_global))
        return results

    def _pool_block_features(
        self,
        c_sub: torch.Tensor,
        v_sub: torch.Tensor,
        cons_block_id: torch.Tensor,
        vars_block_id: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        d = self.d_sub
        dev = c_sub.device

        # scatter_mean / scatter_max via index
        c_block = cons_block_id.unsqueeze(1).expand(-1, d)  # [n_cons, d]
        v_block = vars_block_id.unsqueeze(1).expand(-1, d)  # [n_vars, d]

        c_mean = torch.zeros(K, d, device=dev).scatter_reduce(0, c_block, c_sub, reduce="mean", include_self=False)
        c_max = torch.full((K, d), float("-inf"), device=dev).scatter_reduce(0, c_block, c_sub, reduce="amax", include_self=False)
        c_max = c_max.clamp(min=0.0)  # empty blocks -> 0

        v_mean = torch.zeros(K, d, device=dev).scatter_reduce(0, v_block, v_sub, reduce="mean", include_self=False)
        v_max = torch.full((K, d), float("-inf"), device=dev).scatter_reduce(0, v_block, v_sub, reduce="amax", include_self=False)
        v_max = v_max.clamp(min=0.0)  # empty blocks -> 0

        return torch.cat([v_mean, v_max, c_mean, c_max], dim=1)  # [K, 4*d]

    def _compose_instances(
        self,
        instances: List[SplitInstanceData],
        device: torch.device,
        view_masks: Optional[SplitViewMasks] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Block-wise batched encoding + per-instance composition.
        Encoder forwards = K (one per block slot), not K × B.
        Requires all instances to have the same number of blocks.
        """
        block_counts = {inst.num_blocks for inst in instances}
        if len(block_counts) != 1:
            raise ValueError(
                f"Block-wise batching requires uniform num_blocks, "
                f"but batch contains {block_counts}. "
                f"Use fixed --num_blocks in precompute_splits or bucket by block count."
            )
        K = block_counts.pop()

        # Phase 1: block-wise batched encoding
        # all_partition_embs[k][i] = (c_emb, v_emb)
        all_partition_embs = []
        for k in range(K):
            block_masks = view_masks.masks[k] if view_masks is not None else None
            block_embs = self._encode_block_slot_batched(
                instances, k, device, block_masks=block_masks,
            )
            all_partition_embs.append(block_embs)

        # Phase 2: scatter owned embeddings per instance
        per_inst_sub = self._scatter_owned_batched(
            instances, all_partition_embs, device
        )

        if self.skip_composer:
            return per_inst_sub

        # Phase 3: composer per instance (cheap, no encoder)
        results = []
        for i, inst in enumerate(instances):
            c_sub, v_sub = per_inst_sub[i]
            cons_block_id = inst.cons_block_id.to(device)
            vars_block_id = inst.vars_block_id.to(device)

            block_features = self._pool_block_features(
                c_sub, v_sub, cons_block_id, vars_block_id, K
            )

            c_hat, v_hat = self.composer(
                block_features=block_features,
                block_edge_index=inst.block_edge_index.to(device),
                block_edge_attr=inst.block_edge_attr.to(device),
                cons_block_id=cons_block_id,
                vars_block_id=vars_block_id,
                c_sub=c_sub,
                v_sub=v_sub,
                cons_boundary_feat=inst.cons_boundary_feat.to(device),
                vars_boundary_feat=inst.vars_boundary_feat.to(device),
            )
            results.append((c_hat, v_hat))
        return results

    # ------------------------------------------------------------
    # BIJEPA interface
    # ------------------------------------------------------------

    def embed(
        self,
        inputs: List[Any],
        view_masks_list: Optional[List[Optional[SplitViewMasks]]] = None,
    ) -> Tuple[torch.Tensor]:
        device = next(self.parameters()).device
        outs = []

        for idx, item in enumerate(inputs):
            if isinstance(item, SplitInstanceBatch):
                instances = item.instances
            elif isinstance(item, SplitInstanceData):
                instances = [item]
            else:
                raise TypeError(
                    f"Expected SplitInstanceBatch or SplitInstanceData, got {type(item)}"
                )

            vm = view_masks_list[idx] if view_masks_list is not None else None
            composed = self._compose_instances(instances, device, view_masks=vm)
            c_cat = torch.cat([c for c, _ in composed], dim=0)
            v_cat = torch.cat([v for _, v in composed], dim=0)
            outs.append(torch.cat([c_cat, v_cat], dim=0))

        return tuple(outs)

    def lejepa_pred_loss(self, all_embeddings, global_embeddings, all_views):
        centers = torch.stack(global_embeddings, 0).mean(0)        # [N, D]
        all_emb = torch.stack(all_embeddings, 0)                   # [V, N, D]
        pred_loss = (all_emb - centers.unsqueeze(0)).pow(2).mean()  # scalar
        return pred_loss, pred_loss

    def lejepa_loss(
        self,
        input,
        precomputed_views: Tuple[List[SplitViewMasks], List[SplitViewMasks]],
        lambd: float,
        std_loss_weight=0.0,
    ):
        """
        Override: embed all views once using the original batch + lightweight masks.
        No graph cloning, no re-embedding globals.
        """
        global_view_masks, all_view_masks = precomputed_views
        n_global = len(global_view_masks)

        if isinstance(input, SplitInstanceBatch):
            instances = input.instances
        else:
            instances = [input]

        # embed each view = same instances + different masks
        # reuse the batch object for all views, only masks differ
        inputs_repeated = [input] * len(all_view_masks)
        all_embeddings = self.embed(inputs_repeated, view_masks_list=all_view_masks)
        global_embeddings = all_embeddings[:n_global]

        if self.training:
            jitter = 1e-3
            embeddings_for_reg = [
                emb + jitter * torch.randn_like(emb) for emb in all_embeddings
            ]
        else:
            embeddings_for_reg = all_embeddings

        z_cat = torch.cat(all_embeddings, dim=0)
        std = z_cat.std(dim=0, unbiased=False).clamp_min(1e-6)
        std_loss = torch.nn.functional.relu(1.0 - std).mean()

        pred_loss, pred_loss_masked = self.lejepa_pred_loss(
            all_embeddings, global_embeddings, all_view_masks
        )

        sigreg_loss = torch.stack([self.sigreg(z) for z in embeddings_for_reg]).mean()

        loss = (1 - lambd) * pred_loss + lambd * sigreg_loss + std_loss_weight * std_loss
        return loss, pred_loss, pred_loss_masked, sigreg_loss

    @staticmethod
    def _generate_view_masks(
        instances: List[SplitInstanceData],
        cons_mask_rate: float,
        var_mask_rate: float,
        edge_mask_rate: float,
    ) -> SplitViewMasks:
        """Generate lightweight boolean masks per block slot, no graph cloning."""
        K = instances[0].num_blocks
        masks = []  # [K][B] of (cons_mask, var_mask, edge_mask)
        for k in range(K):
            block_masks = []
            for inst in instances:
                g = inst.partitions[k].graph
                nc = g.constraint_features.size(0)
                nv = g.variable_features.size(0)
                ne = g.edge_index.size(1)
                cm = torch.rand(nc) < cons_mask_rate
                vm = torch.rand(nv) < var_mask_rate
                em = torch.rand(ne) < edge_mask_rate
                block_masks.append((cm, vm, em))
            masks.append(block_masks)
        return SplitViewMasks(masks=masks)

    def make_lejepa_views(
        self,
        input: SplitInstanceBatch,
        n_global_views: int | None = None,
        n_local_views: int | None = None,
        local_mask: float | None = None,
        global_mask: float | None = None,
    ):
        if not isinstance(input, SplitInstanceBatch):
            raise TypeError(f"Expected SplitInstanceBatch, got {type(input)}")

        n_global_views = (
            self.n_global_views if n_global_views is None else n_global_views
        )
        n_local_views = self.n_local_views if n_local_views is None else n_local_views
        local_mask = self.local_mask if local_mask is None else local_mask
        global_mask = self.global_mask if global_mask is None else global_mask

        instances = input.instances

        global_view_masks = [
            self._generate_view_masks(
                instances, global_mask, global_mask, self.global_edge_mask
            )
            for _ in range(n_global_views)
        ]

        local_view_masks = [
            self._generate_view_masks(
                instances, local_mask, local_mask, self.local_edge_mask
            )
            for _ in range(n_local_views)
        ]

        all_view_masks = global_view_masks + local_view_masks
        return global_view_masks, all_view_masks

    # ------------------------------------------------------------
    # Later downstream use
    # ------------------------------------------------------------

    def predict_instance(
        self, inst: SplitInstanceData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        composed = self._compose_instances([inst], device)
        c_hat, v_hat = composed[0]
        x = self.var_head(v_hat).squeeze(-1)
        lam = self.lambda_act(self.cons_head(c_hat).squeeze(-1))
        return x, lam
