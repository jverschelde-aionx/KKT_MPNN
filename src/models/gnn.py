from __future__ import annotations

from typing import List, Tuple

import torch
import torch_geometric
from configargparse import Namespace
from rtdl_num_embeddings import PeriodicEmbeddings, PiecewiseLinearEmbeddings
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph

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
        self.conv_v_to_c = BipartiteGraphConvolution(d_out)
        self.conv_c_to_v = BipartiteGraphConvolution(d_out)
        self.conv_v_to_c2 = BipartiteGraphConvolution(d_out)
        self.conv_c_to_v2 = BipartiteGraphConvolution(d_out)

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
        # Graph-level projector (2D -> D)
        self.graph_proj = nn.Linear(2 * lejepa_embed_dim, lejepa_embed_dim)

    # Node hidden features (for KKT heads)
    def encode_nodes(
        self,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rev_idx = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        c = self.cons_proj(self.cons_num_emb(constraint_features))
        v = self.var_proj(self.var_num_emb(variable_features))
        e = self.edge_proj(self.edge_num_emb(edge_features))

        c = self.conv_v_to_c(v, rev_idx, e, c)
        v = self.conv_c_to_v(c, edge_indices, e, v)
        c = self.conv_v_to_c2(v, rev_idx, e, c)
        v = self.conv_c_to_v2(c, edge_indices, e, v)
        return c, v

    def embed_batch(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        batch_const_emb, batch_var_emb = self.encode_nodes(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
        )

        cb = batch.constraint_features_batch
        vb = batch.variable_features_batch

        # Pool per graph (mean)
        ce = self.cons_lejepa_proj(batch_const_emb)
        ve = self.var_lejepa_proj(batch_var_emb)
        c_pool = global_mean_pool(ce, cb)
        v_pool = global_mean_pool(ve, vb)

        g = torch.cat([c_pool, v_pool], dim=-1)  # [num_graphs, 2D]
        outs.append(self.graph_proj(g))  # [num_graphs, D]
        return tuple(outs)


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
            default=4,
            type=int,
            help="Number of features for constraints",
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
            "--dropout",
            default=0.1,
            type=float,
            help="Dropout rate for GNN Encoder",
        )

    @staticmethod
    def name(args):
        name = "gnn_policy"
        name += LeJepaEncoderModule.name(args)
        name += f"-embedding_size={args.embedding_size}"
        name += f"-cons_nfeats={args.cons_nfeats}"
        name += f"-edge_nfeats={args.edge_nfeats}"
        name += f"-var_nfeats={args.var_nfeats}"
        name += f"-num_emb_type={args.num_emb_type}"
        if args.num_emb_type == "pwl":
            name += f"-num_emb_bins={args.num_emb_bins}"
        elif args.num_emb_type == "periodic":
            name += f"-num_emb_freqs={args.num_emb_freqs}"
        name += f"-l-mask={args.lejepa_local_mask}"
        name += f"-g-mask={args.lejepa_global_mask}"
        name += f"-dropout={args.dropout}"
        return name

    def __init__(self, args: Namespace):
        super().__init__(args.sigreg_slices, args.sigreg_points)
        self.encoder = GNNEncoder(
            cons_nfeats=args.cons_nfeats,
            var_nfeats=args.var_nfeats,
            edge_nfeats=args.edge_nfeats,
            embedding_size=args.embedding_size,
            num_emb_type=args.num_emb_type,
            num_emb_freqs=args.num_emb_freqs,
            num_emb_bins=args.num_emb_bins,
            lejepa_embed_dim=128,
            dropout=args.dropout,
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

    def make_lejepa_views(
        self,
        input: Batch,
        n_global_views: int = 2,
        n_local_views: int = 2,
        local_mask: float = 0.40,
        global_mask: float = 0.1,
    ) -> Tuple[List[Batch], List[Batch]]:
        if not isinstance(input, Batch):
            raise TypeError(f"Expected torch_geometric.data.Batch, got {type(input)}")

        data_list = input.to_data_list()
        global_views: List[Batch] = []
        local_views: List[Batch] = []
        follow_keys = ["constraint_features", "variable_features"]

        # For each view index, build a Batch of all graphs with that augmentation
        for _ in range(n_global_views):
            graphs = [subgraph_view(g, global_mask, global_mask) for g in data_list]
            global_views.append(Batch.from_data_list(graphs, follow_batch=follow_keys))

        for _ in range(n_local_views):
            graphs = [subgraph_view(g, local_mask, local_mask) for g in data_list]
            local_views.append(Batch.from_data_list(graphs, follow_batch=follow_keys))

        all_views = global_views + local_views
        return global_views, all_views


def subgraph_view(graph: Data, cons_keep_ratio: float, vars_keep_ratio: float) -> Data:
    m = graph.constraint_features.size(0)
    n = graph.variable_features.size(0)
    device = graph.constraint_features.device

    n_cons_keep = max(1, int(m * cons_keep_ratio))
    n_vars_keep = max(1, int(n * vars_keep_ratio))

    cons_idx = torch.randperm(m, device=device)[:n_cons_keep]
    vars_idx = torch.randperm(n, device=device)[:n_vars_keep]

    cons_keep = torch.zeros(m, dtype=torch.bool, device=device)
    cons_keep[cons_idx] = True
    vars_keep = torch.zeros(n, dtype=torch.bool, device=device)
    vars_keep[vars_idx] = True

    # filter edges that connect kept constraints <-> kept variables
    edge_c = graph.edge_index[0]
    edge_v = graph.edge_index[1]
    edge_keep = cons_keep[edge_c] & vars_keep[edge_v]

    edge_c_old = edge_c[edge_keep]
    edge_v_old = edge_v[edge_keep]
    edge_attr = graph.edge_attr[edge_keep]

    # remap old indices -> new [0..n_cons_keep) and [0..n_vars_keep)
    cons_map = -torch.ones(m, dtype=torch.long, device=device)
    cons_map[cons_idx] = torch.arange(n_cons_keep, device=device)

    vars_map = -torch.ones(n, dtype=torch.long, device=device)
    vars_map[vars_idx] = torch.arange(n_vars_keep, device=device)

    new_edge_index = torch.stack([cons_map[edge_c_old], vars_map[edge_v_old]], dim=0)

    new_graph = type(graph)(
        constraint_features=graph.constraint_features[cons_idx],
        edge_index=new_edge_index,
        edge_attr=edge_attr,
        variable_features=graph.variable_features[vars_idx],
    )

    if hasattr(graph, "b_vec"):
        new_graph.b_vec = graph.b_vec[cons_idx]
    if hasattr(graph, "c_vec"):
        new_graph.c_vec = graph.c_vec[vars_idx]

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
