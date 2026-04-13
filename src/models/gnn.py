from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch_geometric
from configargparse import Namespace
from rtdl_num_embeddings import PeriodicEmbeddings, PiecewiseLinearEmbeddings
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATv2Conv,
    TransformerConv,
)

from models.base import LeJepaEncoderModule


class GNNEncoder(nn.Module):
    """
    Numeric embeddings + bipartite message passing + LeJEPA projectors.
    Returns node hidden states; exposes node/graph embeddings for LeJEPA.
    """

    def __init__(
        self,
        *,
        cons_nfeats: int,
        var_nfeats: int,
        edge_nfeats: int,
        embedding_size: int,
        num_emb_type: str,
        num_emb_freqs: int,
        num_emb_bins: int,
        lejepa_embed_dim: int = 128,
        dropout: float = 0.1,
        bipartite_conv: str = "gatv2",
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        d_out = embedding_size

        # Numeric embeddings + small MLPs
        self.cons_num_emb, cons_out_dim = _build_numeric_block(
            cons_nfeats, num_emb_type, num_emb_freqs, num_emb_bins
        )
        self.var_num_emb, var_out_dim = _build_numeric_block(
            var_nfeats, num_emb_type, num_emb_freqs, num_emb_bins
        )
        self.edge_num_emb, edge_out_dim = _build_numeric_block(
            edge_nfeats, num_emb_type, num_emb_freqs, num_emb_bins
        )

        self.cons_proj = nn.Sequential(
            nn.LayerNorm(cons_out_dim),
            nn.Linear(cons_out_dim, d_out),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.var_proj = nn.Sequential(
            nn.LayerNorm(var_out_dim),
            nn.Linear(var_out_dim, d_out),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(edge_out_dim),
            nn.Linear(edge_out_dim, d_out),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )

        # Message passing
        self.conv_v_to_c = make_bipartite_conv(
            bipartite_conv, d_out, heads=attn_heads, dropout=dropout
        )
        self.conv_c_to_v = make_bipartite_conv(
            bipartite_conv, d_out, heads=attn_heads, dropout=dropout
        )
        self.conv_v_to_c2 = make_bipartite_conv(
            bipartite_conv, d_out, heads=attn_heads, dropout=dropout
        )
        self.conv_c_to_v2 = make_bipartite_conv(
            bipartite_conv, d_out, heads=attn_heads, dropout=dropout
        )

        # LeJEPA projectors
        self.cons_lejepa_proj = nn.Sequential(
            nn.Linear(d_out, d_out // 2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(d_out // 2, lejepa_embed_dim),
        )
        self.var_lejepa_proj = nn.Sequential(
            nn.Linear(d_out, d_out // 2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(d_out // 2, lejepa_embed_dim),
        )

        self.cons_mask_token = nn.Parameter(torch.zeros(1, d_out))
        self.var_mask_token = nn.Parameter(torch.zeros(1, d_out))
        self.edge_mask_token = nn.Parameter(torch.zeros(1, d_out))

        nn.init.normal_(self.cons_mask_token, std=0.02)
        nn.init.normal_(self.var_mask_token, std=0.02)
        nn.init.normal_(self.edge_mask_token, std=0.02)

    # Node hidden features (for KKT heads)
    def encode_nodes(
        self,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor,
        cons_mask: Optional[torch.Tensor] = None,  # [num_cons] bool
        var_mask: Optional[torch.Tensor] = None,  # [num_vars] bool
        edge_mask: Optional[torch.Tensor] = None,  # [num_edges] bool
    ):
        rev_idx = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        c = self.cons_proj(self.cons_num_emb(constraint_features))
        v = self.var_proj(self.var_num_emb(variable_features))
        e = self.edge_proj(self.edge_num_emb(edge_features))

        if cons_mask is None:
            cons_mask = torch.zeros(c.size(0), dtype=torch.bool, device=c.device)
        else:
            cons_mask = cons_mask.to(device=c.device, dtype=torch.bool)

        if var_mask is None:
            var_mask = torch.zeros(v.size(0), dtype=torch.bool, device=v.device)
        else:
            var_mask = var_mask.to(device=v.device, dtype=torch.bool)
        if edge_mask is not None:
            edge_mask = edge_mask.to(device=e.device, dtype=torch.bool).view(-1)

            if edge_mask.numel() != e.size(0):
                raise ValueError(
                    f"edge_mask has {edge_mask.numel()} elems, but e has {e.size(0)} edges"
                )

            if edge_mask.any():  # optional speed win
                e = torch.where(
                    edge_mask.unsqueeze(-1),
                    self.edge_mask_token.expand_as(
                        e
                    ),  # safer than expand(e.size(0), -1)
                    e,
                )

        # Apply mask tokens only where mask=True
        if cons_mask.any():
            c = torch.where(
                cons_mask.unsqueeze(-1), self.cons_mask_token.expand_as(c), c
            )
        if var_mask.any():
            v = torch.where(var_mask.unsqueeze(-1), self.var_mask_token.expand_as(v), v)

        c = self.conv_v_to_c(v, rev_idx, e, c)
        v = self.conv_c_to_v(c, edge_indices, e, v)
        c = self.conv_v_to_c2(v, rev_idx, e, c)
        v = self.conv_c_to_v2(c, edge_indices, e, v)
        return c, v

    def embed_batch(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        cons_mask = getattr(batch, "cons_mask", None)
        var_mask = getattr(batch, "var_mask", None)
        edge_mask = getattr(batch, "edge_mask", None)

        c, v = self.encode_nodes(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
            cons_mask=cons_mask,
            var_mask=var_mask,
            edge_mask=edge_mask,
        )

        ce = self.cons_lejepa_proj(c)  # [sum_m, D]
        ve = self.var_lejepa_proj(v)  # [sum_n, D]

        z = torch.cat([ce, ve], dim=0)  # [sum_m + sum_n, D]
        return (z,)


class GNNPolicy(LeJepaEncoderModule):
    @staticmethod
    def add_args(parser):
        LeJepaEncoderModule.add_args(parser)
        group = parser.add_argument_group("gnn")

        group.add_argument(
            "--embedding_size",
            default=64,
            type=int,
            help="Size of all embeddings (node, edge, etc.)",
        )
        group.add_argument(
            "--cons_nfeats",
            default=9,
            type=int,
            help="Number of features for constraints (3 scalar + 6 one-hot row type)",
        )
        group.add_argument(
            "--edge_nfeats", default=1, type=int, help="Number of features for edges"
        )
        group.add_argument(
            "--var_nfeats",
            default=18,
            type=int,
            help="Number of features for variables",
        )
        group.add_argument(
            "--num_emb_type", choices=["periodic", "pwl", "linear"], default="periodic"
        )
        group.add_argument("--num_emb_bins", type=int, default=32)  # for PWL
        group.add_argument("--num_emb_freqs", type=int, default=16)  # for periodic
        group.add_argument(
            "--lejepa_local_mask",
            type=float,
            default=0.40,
            help="Local view masking ratio",
        )
        group.add_argument(
            "--lejepa_global_mask",
            type=float,
            default=0.1,
            help="Global view masking ratio",
        )
        group.add_argument(
            "--lejepa_local_edge_mask",
            type=float,
            default=0.20,
            help="Local view masking ratio",
        )
        group.add_argument(
            "--lejepa_global_edge_mask",
            type=float,
            default=0.05,
            help="Global view masking ratio",
        )
        group.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            help="Dropout rate for GNN Encoder",
        )
        group.add_argument(
            "--lejepa_embed_dim",
            default=128,
            type=int,
            help="Embedding dimension for LeJEPA projector",
        )
        group.add_argument(
            "--bipartite_conv",
            type=str,
            choices=["gcn", "transformer", "gatv2"],
            default="gcn",
            help="Bipartite conv type (gcn=your current MessagePassing block)",
        )
        group.add_argument(
            "--attn_heads",
            type=int,
            default=4,
            help="Num attention heads for transformer/gatv2 (embedding_size must be divisible)",
        )

    @staticmethod
    def name(args):
        name = "gnn_policy"
        name += LeJepaEncoderModule.name(args)
        name += f"-dim={args.embedding_size}"
        name += f"-l_mask={args.lejepa_local_mask}"
        name += f"-g_mask={args.lejepa_global_mask}"
        name += f"-l_e_mask={args.lejepa_local_edge_mask}"
        name += f"-g_e_mask={args.lejepa_global_edge_mask}"
        name += f"-dp={args.dropout}"
        name += f"-l_dim={args.lejepa_embed_dim}"
        name += f"-conv={args.bipartite_conv}"
        if args.bipartite_conv in ("transformer", "gatv2"):
            name += f"-heads={args.attn_heads}"

        return name

    def __init__(self, args: Namespace):
        super().__init__(args.sigreg_slices, args.sigreg_points)

        self.n_global_views = args.lejepa_n_global_views
        self.n_local_views = args.lejepa_n_local_views
        self.local_mask = args.lejepa_local_mask
        self.global_mask = args.lejepa_global_mask
        self.global_edge_mask = args.lejepa_global_edge_mask
        self.local_edge_mask = args.lejepa_local_edge_mask

        self.encoder = GNNEncoder(
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
        d_out = args.embedding_size
        # Constraint embedding
        # Heads
        self.var_head = nn.Sequential(
            nn.Linear(d_out, d_out), nn.ReLU(), nn.Linear(d_out, 1, bias=False)
        )
        self.cons_head = nn.Sequential(
            nn.Linear(d_out, d_out), nn.ReLU(), nn.Linear(d_out, 1, bias=False)
        )
        self.lambda_act = nn.Softplus()  # smooth ≥0

    @property
    def encoder(self) -> GNNEncoder:
        return self._encoder

    @encoder.setter
    def encoder(self, module):
        self._encoder = module

    def forward(
        self,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c, v = self.encoder.encode_nodes(
            constraint_features, edge_indices, edge_features, variable_features
        )
        x_all = self.var_head(v).squeeze(-1)
        lam_all = self.lambda_act(self.cons_head(c).squeeze(-1))
        return x_all, lam_all

    def embed(self, inputs: List[Batch]) -> Tuple[torch.Tensor]:
        embeddings = []
        for batch in inputs:
            if not isinstance(batch, Batch):
                raise TypeError(f"Expected Batch, got {type(batch)}")

            emb = self.encoder.embed_batch(batch)
            embeddings.extend(emb)

        return tuple(embeddings)

    @staticmethod
    def _sample_node_masks_for_batch(
        batch: Batch, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          cons_mask: [sum_m] bool
          var_mask:  [sum_n] bool
        Ensures at least one unmasked node per graph per type.
        """
        device = batch.constraint_features.device
        cons_mask = (
            torch.rand(batch.constraint_features.size(0), device=device) < mask_ratio
        )
        var_mask = (
            torch.rand(batch.variable_features.size(0), device=device) < mask_ratio
        )

        cb = batch.constraint_features_batch
        vb = batch.variable_features_batch
        B = int(batch.num_graphs)

        for g in range(B):
            idx_c = (cb == g).nonzero(as_tuple=False).view(-1)
            if idx_c.numel() > 0 and cons_mask[idx_c].all():
                j = idx_c[torch.randint(idx_c.numel(), (1,), device=device)]
                cons_mask[j] = False

            idx_v = (vb == g).nonzero(as_tuple=False).view(-1)
            if idx_v.numel() > 0 and var_mask[idx_v].all():
                j = idx_v[torch.randint(idx_v.numel(), (1,), device=device)]
                var_mask[j] = False

        return cons_mask, var_mask

    @staticmethod
    def _sample_edge_mask_for_batch(
        batch: Batch, mask_ratio: float, ensure_one_visible_edge_per_graph: bool = False
    ) -> torch.Tensor:
        device = batch.edge_attr.device
        E = batch.edge_attr.size(0)
        if E == 0:
            return torch.zeros((0,), dtype=torch.bool, device=device)

        edge_mask = torch.rand(E, device=device) < mask_ratio

        if ensure_one_visible_edge_per_graph:
            # Which graph each edge belongs to (constraints endpoint determines graph id)
            edge_batch = batch.constraint_features_batch[batch.edge_index[0]]
            B = int(batch.num_graphs)
            for g in range(B):
                idx_e = (edge_batch == g).nonzero(as_tuple=False).view(-1)
                if idx_e.numel() > 0 and edge_mask[idx_e].all():
                    edge_mask[
                        idx_e[torch.randint(idx_e.numel(), (1,), device=device)]
                    ] = False

        return edge_mask

    def make_lejepa_views(
        self,
        input: Batch,
    ) -> Tuple[List[Batch], List[Batch]]:
        if not isinstance(input, Batch):
            raise TypeError(f"Expected torch_geometric.data.Batch, got {type(input)}")

        global_views: List[Batch] = []
        local_views: List[Batch] = []

        for _ in range(self.n_global_views):
            view = copy.copy(input)  # shallow copy, shares all tensors
            view.cons_mask, view.var_mask = self._sample_node_masks_for_batch(
                input, self.global_mask
            )
            view.edge_mask = self._sample_edge_mask_for_batch(
                input, self.global_edge_mask
            )
            global_views.append(view)

        for _ in range(self.n_local_views):
            view = copy.copy(input)
            view.cons_mask, view.var_mask = self._sample_node_masks_for_batch(
                input, self.local_mask
            )
            view.edge_mask = self._sample_edge_mask_for_batch(
                input, self.local_edge_mask
            )
            local_views.append(view)

        all_views = global_views + local_views
        return global_views, all_views


@torch.inference_mode()
def masked_view(
    graph: Data,
    cons_mask_ratio: float,
    vars_mask_ratio: float,
    *,
    edge_drop_ratio: float = 0.0,
) -> Data:
    m = graph.constraint_features.size(0)
    n = graph.variable_features.size(0)
    device = graph.constraint_features.device

    # True = masked
    cons_mask = torch.rand(m, device=device) < cons_mask_ratio
    var_mask = torch.rand(n, device=device) < vars_mask_ratio

    # Ensure at least one visible node per type
    if m > 0 and cons_mask.all():
        cons_mask[torch.randint(m, (1,), device=device)] = False
    if n > 0 and var_mask.all():
        var_mask[torch.randint(n, (1,), device=device)] = False

    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    if edge_drop_ratio and edge_drop_ratio > 0.0:
        E = edge_index.size(1)
        keep = torch.rand(E, device=device) >= edge_drop_ratio
        edge_index = edge_index[:, keep]
        edge_attr = edge_attr[keep]

    new_graph = type(graph)(
        constraint_features=graph.constraint_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        variable_features=graph.variable_features,
    )

    # carry optional attrs
    if hasattr(graph, "b_vec"):
        new_graph.b_vec = graph.b_vec
    if hasattr(graph, "c_vec"):
        new_graph.c_vec = graph.c_vec

    # store masks (will be batched/concatenated by PyG)
    new_graph.cons_mask = cons_mask
    new_graph.var_mask = var_mask
    return new_graph


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, embedding_size: int):
        super().__init__("add")

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(embedding_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * embedding_size, embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """

        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output


def _build_numeric_block(
    in_feats: int, emb_type: str, n_freq: int, n_bins: int
) -> Tuple[nn.Module, int]:
    if emb_type == "periodic":
        emb_layer = PeriodicEmbeddings(
            n_features=in_feats, n_frequencies=n_freq, lite=True
        )
    elif emb_type == "pwl":
        emb_layer = PiecewiseLinearEmbeddings(n_features=in_feats, num_bins=n_bins)
    else:  # "linear"
        return nn.Identity(), in_feats

    with torch.no_grad():
        dummy = torch.zeros(1, in_feats)
        out_dim = emb_layer(dummy).numel()
    return nn.Sequential(emb_layer, nn.Flatten(start_dim=1)), out_dim


def make_bipartite_conv(
    kind: str, embedding_size: int, heads: int, dropout: float
) -> nn.Module:
    if kind == "gcn":
        return BipartiteGraphConvolution(embedding_size)
    if kind == "transformer":
        return BipartiteTransformerConvolution(
            embedding_size, heads=heads, dropout=dropout
        )
    if kind == "gatv2":
        return BipartiteGATv2Convolution(embedding_size, heads=heads, dropout=dropout)
    raise ValueError(f"Unknown bipartite conv kind: {kind}")


class BipartiteTransformerConvolution(nn.Module):
    def __init__(self, embedding_size: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embedding_size % heads != 0:
            raise ValueError(
                f"embedding_size={embedding_size} must be divisible by heads={heads}"
            )

        # Sparse attention over edges (O(E), not O(N^2))
        self.conv = TransformerConv(
            (embedding_size, embedding_size),
            out_channels=embedding_size // heads,
            heads=heads,
            edge_dim=embedding_size,
            dropout=dropout,
            beta=True,  # learn residual mixing
        )
        self.post = nn.LayerNorm(embedding_size)
        self.out = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        msg = self.conv(
            (left_features, right_features), edge_indices, edge_attr=edge_features
        )
        msg = self.post(msg)
        return self.out(torch.cat([msg, right_features], dim=-1))


class BipartiteGATv2Convolution(nn.Module):
    def __init__(self, embedding_size: int, *, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embedding_size % heads != 0:
            raise ValueError(
                f"embedding_size={embedding_size} must be divisible by heads={heads}"
            )

        self.conv = GATv2Conv(
            (embedding_size, embedding_size),
            out_channels=embedding_size // heads,
            heads=heads,
            edge_dim=embedding_size,
            dropout=dropout,
            add_self_loops=False,  # important for bipartite pair-indexed graphs
        )
        self.post = nn.LayerNorm(embedding_size)
        self.out = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        msg = self.conv(
            (left_features, right_features), edge_indices, edge_attr=edge_features
        )
        msg = self.post(msg)
        return self.out(torch.cat([msg, right_features], dim=-1))
