"""
Block-level composition pipeline.
==================================
Orchestrates halo-subgraph encoding, scatter-back, block feature pooling,
and block interaction graph construction.

Also contains the trainable Block GNN Composer model (Phase 6).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from data.common import CONS_PAD, VARS_PAD
from data.datasets import BipartiteNodeData, right_pad
from models.decomposition import (
    BlockGraph,
    PartitionSpec,
    build_block_graph,
    build_halo_subgraphs,
    compute_block_features,
    log_block_graph_diagnostics,
)
from models.gnn import GNNEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _encode_subgraph(
    encoder: GNNEncoder,
    sg: BipartiteNodeData,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen encoder on a single halo subgraph.

    Returns (c_emb, v_emb) on CPU, shapes [n_cons_local, d] and
    [n_vars_local, d].
    """
    c_feat = sg.constraint_features.to(device)
    v_feat = sg.variable_features.to(device)
    ei = sg.edge_index.to(device)
    ea = sg.edge_attr.to(device)
    c_emb, v_emb = encoder.encode_nodes(c_feat, ei, ea, v_feat)
    return c_emb.cpu(), v_emb.cpu()


def scatter_owned_embeddings(
    partitions: List[PartitionSpec],
    halo_subgraphs: List[BipartiteNodeData],
    partition_embeddings: List[Tuple[torch.Tensor, torch.Tensor]],
    n_cons: int,
    n_vars: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter owned-node embeddings from halo subgraphs to global indices.

    Parameters
    ----------
    partitions : list[PartitionSpec]
    halo_subgraphs : list[BipartiteNodeData]
        Must have ``orig_cons_ids``, ``orig_var_ids``, ``owned_cons_mask``,
        ``owned_var_mask`` attributes.
    partition_embeddings : list of (c_emb_local, v_emb_local)
        Per-subgraph encoder outputs, one per partition.
    n_cons, n_vars : int
        Total node counts in the full graph.

    Returns
    -------
    c_emb_global : [n_cons, d]
    v_emb_global : [n_vars, d]
    """
    d = partition_embeddings[0][0].size(1)
    c_emb = torch.zeros(n_cons, d)
    v_emb = torch.zeros(n_vars, d)

    for sg, (c_local, v_local) in zip(halo_subgraphs, partition_embeddings):
        owned_c_mask = sg.owned_cons_mask.bool()
        owned_v_mask = sg.owned_var_mask.bool()

        global_c_ids = sg.orig_cons_ids[owned_c_mask]
        global_v_ids = sg.orig_var_ids[owned_v_mask]

        c_emb[global_c_ids] = c_local[owned_c_mask]
        v_emb[global_v_ids] = v_local[owned_v_mask]

    return c_emb, v_emb


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


@torch.inference_mode()
def compose_block_graph(
    partitions: List[PartitionSpec],
    encoder: GNNEncoder,
    c_nodes: torch.Tensor,
    v_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    halo_hops: int,
    device: torch.device = torch.device("cpu"),
    include_metadata: bool = False,
    cons_is_boundary: torch.Tensor | None = None,
    vars_is_boundary: torch.Tensor | None = None,
) -> BlockGraph:
    """End-to-end block interaction graph construction.

    1. Build halo subgraphs for each partition.
    2. Encode each halo subgraph independently (frozen encoder).
    3. Scatter owned-node embeddings back to global-indexed tensors.
    4. Compute pooled block features.
    5. Build the block interaction graph.
    6. Log diagnostics.

    Returns a :class:`BlockGraph` with ``block_features`` populated.
    """
    n_cons = c_nodes.size(0)
    n_vars = v_nodes.size(0)

    # 1. Halo subgraphs
    halo_sgs = build_halo_subgraphs(
        partitions, c_nodes, v_nodes, edge_index, edge_attr,
        halo_hops=halo_hops,
    )

    # 2. Encode each subgraph
    partition_embs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for sg in halo_sgs:
        c_emb, v_emb = _encode_subgraph(encoder, sg, device)
        partition_embs.append((c_emb, v_emb))

    # 3. Scatter back to global indices
    c_global, v_global = scatter_owned_embeddings(
        partitions, halo_sgs, partition_embs, n_cons, n_vars,
    )

    # 4. Block features
    block_features = compute_block_features(
        partitions, c_global, v_global,
        include_metadata=include_metadata,
        halo_subgraphs=halo_sgs,
        cons_is_boundary=cons_is_boundary,
        vars_is_boundary=vars_is_boundary,
    )

    # 5. Block graph
    bg = build_block_graph(
        partitions, edge_index, edge_attr,
        n_cons=n_cons, n_vars=n_vars,
    )
    bg.block_features = block_features

    # 6. Diagnostics
    log_block_graph_diagnostics(bg)

    return bg


# ---------------------------------------------------------------------------
# Neural network modules (Phase 6)
# ---------------------------------------------------------------------------


class BlockGNN(nn.Module):
    """2-layer GATv2 on the block interaction graph.

    Processes pooled block features and inter-block edge features to produce
    per-block context vectors.

    Parameters
    ----------
    d_block : int
        Input block feature dimension (typically 4 * embedding_size).
    d_hidden : int
        Hidden dimension for GATv2 layers.
    d_z : int
        Output dimension per block.
    d_edge : int
        Block edge feature dimension (default 4: cut_count, sum_abs_coeff,
        n_boundary_cons, n_boundary_vars).
    heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_block: int,
        d_hidden: int,
        d_z: int,
        d_edge: int = 4,
        heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
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
            d_hidden, d_hidden // heads,
            heads=heads,
            edge_dim=d_hidden,
            dropout=dropout,
            add_self_loops=False,
        )
        self.norm1 = nn.LayerNorm(d_hidden)
        self.conv2 = GATv2Conv(
            d_hidden, d_hidden // heads,
            heads=heads,
            edge_dim=d_hidden,
            dropout=dropout,
            add_self_loops=False,
        )
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
        """Forward pass on the block interaction graph.

        Parameters
        ----------
        block_features : [K, d_block]
        block_edge_index : [2, E_block]  canonical (min, max) pairs
        block_edge_attr : [E_block, d_edge]

        Returns
        -------
        z : [K, d_z]  per-block context vectors
        """
        x = self.input_proj(block_features)  # [K, d_hidden]
        e = self.edge_proj(block_edge_attr)  # [E_block, d_hidden]

        # Make undirected: add reverse edges
        ei = torch.cat([block_edge_index, block_edge_index.flip(0)], dim=1)
        e_full = torch.cat([e, e], dim=0)

        x = x + self.conv1(x, ei, edge_attr=e_full)
        x = self.norm1(x)
        x = x + self.conv2(x, ei, edge_attr=e_full)
        x = self.norm2(x)

        return self.output_proj(x)  # [K, d_z]


class ComposerMLP(nn.Module):
    """MLP that fuses subgraph embedding, block context, and boundary features.

    Reconstructs the full-graph teacher embedding for one node type.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : [N, d_in]  concatenated inputs

        Returns
        -------
        h_hat : [N, d_out]
        """
        return self.mlp(x)


class BlockGNNComposer(nn.Module):
    """Block-level GNN + per-node composer heads.

    Given block graph features, per-node subgraph embeddings, per-node block
    assignments, and per-node boundary features, produces refined embeddings
    that approximate full-graph teacher embeddings.

    Supports ablation via ``use_block_context`` and ``use_block_gnn`` flags:

    - ``use_block_context=False``: Local MLP baseline (no block info).
    - ``use_block_context=True, use_block_gnn=False``: Pooled block-context
      MLP baseline (block features projected, no GNN message passing).
    - ``use_block_context=True, use_block_gnn=True``: Full block GNN composer.

    Parameters
    ----------
    d_sub : int
        Subgraph embedding dimension (encoder embedding_size).
    d_block : int
        Block feature dimension (from ``compute_block_features``).
    d_z : int
        Block context vector dimension.
    d_boundary : int
        Boundary feature dimension.
    d_mlp_hidden : int
        Hidden dimension for composer MLPs.
    heads : int
        Attention heads for BlockGNN.
    dropout : float
        Dropout rate.
    use_block_context : bool
        If False, composer input is ``[h_sub, boundary_features]`` only.
    use_block_gnn : bool
        If False but ``use_block_context`` is True, block context is a linear
        projection of pooled block features (no GNN message passing).
    """

    def __init__(
        self,
        d_sub: int,
        d_block: int,
        d_z: int,
        d_boundary: int,
        d_mlp_hidden: int = 256,
        heads: int = 4,
        dropout: float = 0.05,
        use_block_context: bool = True,
        use_block_gnn: bool = True,
    ):
        super().__init__()
        self.d_sub = d_sub
        self.d_z = d_z
        self.d_boundary = d_boundary
        self.use_block_context = use_block_context
        self.use_block_gnn = use_block_gnn

        if use_block_context:
            if use_block_gnn:
                self.block_gnn = BlockGNN(
                    d_block=d_block, d_hidden=d_z, d_z=d_z,
                    d_edge=4, heads=heads, dropout=dropout,
                )
            else:
                # Simple projection baseline: pooled features → d_z
                self.block_proj = nn.Sequential(
                    nn.LayerNorm(d_block),
                    nn.Linear(d_block, d_z),
                    nn.ReLU(),
                )
            d_composer_in = d_sub + d_z + d_boundary
        else:
            d_composer_in = d_sub + d_boundary

        self.var_composer = ComposerMLP(
            d_in=d_composer_in, d_out=d_sub,
            d_hidden=d_mlp_hidden, dropout=dropout,
        )
        self.cons_composer = ComposerMLP(
            d_in=d_composer_in, d_out=d_sub,
            d_hidden=d_mlp_hidden, dropout=dropout,
        )

    def forward(
        self,
        block_features: torch.Tensor,
        block_edge_index: torch.Tensor,
        block_edge_attr: torch.Tensor,
        cons_block_id: torch.Tensor,
        vars_block_id: torch.Tensor,
        c_sub_owned: torch.Tensor,
        v_sub_owned: torch.Tensor,
        cons_boundary_feat: torch.Tensor,
        vars_boundary_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        block_features : [K, d_block]
        block_edge_index : [2, E_block]
        block_edge_attr : [E_block, 4]
        cons_block_id : [N_cons_owned] long — block index per owned constraint
        vars_block_id : [N_var_owned] long — block index per owned variable
        c_sub_owned : [N_cons_owned, d_sub] — subgraph embeddings for owned cons
        v_sub_owned : [N_var_owned, d_sub] — subgraph embeddings for owned vars
        cons_boundary_feat : [N_cons_owned, d_boundary]
        vars_boundary_feat : [N_var_owned, d_boundary]

        Returns
        -------
        c_hat : [N_cons_owned, d_sub]  reconstructed constraint embeddings
        v_hat : [N_var_owned, d_sub]  reconstructed variable embeddings
        """
        if self.use_block_context:
            # Compute block context vectors
            if self.use_block_gnn:
                z_blocks = self.block_gnn(
                    block_features, block_edge_index, block_edge_attr,
                )  # [K, d_z]
            else:
                z_blocks = self.block_proj(block_features)  # [K, d_z]

            # Scatter to per-node via block IDs
            z_cons = z_blocks[cons_block_id]  # [N_cons_owned, d_z]
            z_vars = z_blocks[vars_block_id]  # [N_var_owned, d_z]

            cons_input = torch.cat([c_sub_owned, z_cons, cons_boundary_feat], dim=-1)
            vars_input = torch.cat([v_sub_owned, z_vars, vars_boundary_feat], dim=-1)
        else:
            cons_input = torch.cat([c_sub_owned, cons_boundary_feat], dim=-1)
            vars_input = torch.cat([v_sub_owned, vars_boundary_feat], dim=-1)

        c_hat = self.cons_composer(cons_input)
        v_hat = self.var_composer(vars_input)

        return c_hat, v_hat


def composer_loss(
    c_hat: torch.Tensor,
    v_hat: torch.Tensor,
    c_teacher: torch.Tensor,
    v_teacher: torch.Tensor,
    cons_is_boundary: torch.Tensor,
    vars_is_boundary: torch.Tensor,
    alpha_mse: float = 1.0,
    beta_cos: float = 1.0,
    boundary_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined MSE + cosine loss for embedding reconstruction.

    Boundary nodes are upweighted by ``boundary_weight``.

    Parameters
    ----------
    c_hat, v_hat : predicted embeddings (owned nodes)
    c_teacher, v_teacher : teacher embeddings (owned nodes)
    cons_is_boundary, vars_is_boundary : bool masks
    alpha_mse : weight for MSE term
    beta_cos : weight for cosine term
    boundary_weight : upweight factor for boundary nodes

    Returns
    -------
    loss : scalar tensor
    metrics : dict with per-type and per-boundary breakdown
    """
    def _weighted_mse(pred, target, is_boundary):
        w = torch.where(is_boundary, boundary_weight, 1.0)
        per_node = ((pred - target) ** 2).mean(dim=-1)  # [N]
        return (per_node * w).sum() / w.sum()

    def _weighted_cos_loss(pred, target, is_boundary):
        cos_sim = F.cosine_similarity(pred, target, dim=-1)  # [N]
        per_node = 1.0 - cos_sim
        w = torch.where(is_boundary, boundary_weight, 1.0)
        return (per_node * w).sum() / w.sum(), cos_sim

    cons_mse = _weighted_mse(c_hat, c_teacher, cons_is_boundary)
    vars_mse = _weighted_mse(v_hat, v_teacher, vars_is_boundary)

    cons_cos_loss, cons_cos_sim = _weighted_cos_loss(c_hat, c_teacher, cons_is_boundary)
    vars_cos_loss, vars_cos_sim = _weighted_cos_loss(v_hat, v_teacher, vars_is_boundary)

    mse_loss = 0.5 * (cons_mse + vars_mse)
    cos_loss = 0.5 * (cons_cos_loss + vars_cos_loss)
    loss = alpha_mse * mse_loss + beta_cos * cos_loss

    # Detailed metrics
    with torch.no_grad():
        metrics: Dict[str, float] = {
            "loss": loss.item(),
            "cons_mse": cons_mse.item(),
            "vars_mse": vars_mse.item(),
            "cons_cos": cons_cos_sim.mean().item(),
            "vars_cos": vars_cos_sim.mean().item(),
        }
        if cons_is_boundary.any():
            metrics["boundary_cons_cos"] = cons_cos_sim[cons_is_boundary].mean().item()
        if vars_is_boundary.any():
            metrics["boundary_vars_cos"] = vars_cos_sim[vars_is_boundary].mean().item()
        interior_cons = ~cons_is_boundary
        interior_vars = ~vars_is_boundary
        if interior_cons.any():
            metrics["interior_cons_cos"] = cons_cos_sim[interior_cons].mean().item()
        if interior_vars.any():
            metrics["interior_vars_cos"] = vars_cos_sim[interior_vars].mean().item()

    return loss, metrics
