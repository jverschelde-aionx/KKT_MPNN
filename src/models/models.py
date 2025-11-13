from typing import Tuple

import torch
import torch_geometric
from rtdl_num_embeddings import PeriodicEmbeddings, PiecewiseLinearEmbeddings
from torch import nn


class KKTNetMLP(nn.Module):
    def __init__(self, m: int, n: int, hidden=256, jepa_embed_dim=128):
        super().__init__()
        self.m = m
        self.n = n
        D_in = m * n + m + n
        self.net = nn.Sequential(
            nn.Linear(D_in, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
        )
        # shared trunk -> split heads
        self.head_x = nn.Sequential(nn.Linear(hidden, 64), nn.SELU(), nn.Linear(64, n))
        self.head_lam = nn.Sequential(
            nn.Linear(hidden, 64), nn.SELU(), nn.Linear(64, m)
        )
        # lambdas must be >= 0 per dual feasibility; we'll ReLU at loss-time OR here:
        self.relu = nn.ReLU()

        # JEPA components: projector and predictor
        # Projector: hidden → embedding (applied to both online and target)
        self.jepa_proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, jepa_embed_dim)
        )
        # Predictor: embedding → prediction (online only)
        self.jepa_pred_net = nn.Sequential(
            nn.Linear(jepa_embed_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, jepa_embed_dim)
        )

    def encode_trunk(self, flat_input):
        """
        Encode input to hidden representation (before task heads).

        Args:
            flat_input: [B, m*n + m + n]

        Returns:
            z: Hidden representation [B, hidden_dim]
        """
        return self.net(flat_input)

    def forward(self, flat_input):
        """
        Standard forward pass for KKT prediction.

        Args:
            flat_input: [B, m*n + m + n]

        Returns:
            y_pred: [B, n+m] = [x_pred, lambda_pred]
        """
        z = self.encode_trunk(flat_input)
        x = self.head_x(z)  # [B, n]
        lam = self.head_lam(z)  # [B, m]
        lam = self.relu(lam)  # enforce non-negativity at the output
        return torch.cat([x, lam], dim=-1)

    def jepa_embed(self, flat_input):
        """
        JEPA embedding: encode → project → L2-normalize.

        This method is used for both online and target encoders in JEPA training.

        Args:
            flat_input: [B, m*n + m + n]

        Returns:
            z: L2-normalized embedding [B, jepa_embed_dim]
        """
        z = self.encode_trunk(flat_input)
        z_proj = self.jepa_proj(z)
        z_norm = torch.nn.functional.normalize(z_proj, dim=-1)  # L2-normalize
        return z_norm

    def jepa_pred(self, z):
        """
        JEPA prediction: predict → L2-normalize.

        This method is only used for the online encoder (not target).

        Args:
            z: Embedding from jepa_embed [B, jepa_embed_dim]

        Returns:
            p: L2-normalized prediction [B, jepa_embed_dim]
        """
        p = self.jepa_pred_net(z)
        p_norm = torch.nn.functional.normalize(p, dim=-1)  # L2-normalize
        return p_norm


class GNNPolicy(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("gnn_policy")

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

    @staticmethod
    def name(args):
        name = "gnn_policy"
        name += f"-embedding_size={args.embedding_size}"
        name += f"-cons_nfeats={args.cons_nfeats}"
        name += f"-edge_nfeats={args.edge_nfeats}"
        name += f"-var_nfeats={args.var_nfeats}"
        name += f"-num_emb_type={args.num_emb_type}"
        if args.num_emb_type == "pwl":
            name += f"-num_emb_bins={args.num_emb_bins}"
        elif args.num_emb_type == "periodic":
            name += f"-num_emb_freqs={args.num_emb_freqs}"
        return name

    def __init__(self, args):
        super().__init__()
        d_out = args.embedding_size
        # Constraint embedding
        self.cons_num_emb, cons_out_dim = self._build_numeric_block(
            in_feats=args.cons_nfeats,
            emb_type=args.num_emb_type,
            n_freq=args.num_emb_freqs,
            n_bins=args.num_emb_bins,
        )
        self.cons_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_out_dim),
            torch.nn.Linear(cons_out_dim, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
        )

        # Variable embedding
        self.var_num_emb, var_out_dim = self._build_numeric_block(
            in_feats=args.var_nfeats,
            emb_type=args.num_emb_type,
            n_freq=args.num_emb_freqs,
            n_bins=args.num_emb_bins,
        )
        self.var_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(var_out_dim),
            torch.nn.Linear(var_out_dim, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
        )

        # Edge embedding (A_ij coefficients)
        self.edge_num_emb, edge_out_dim = self._build_numeric_block(
            in_feats=args.edge_nfeats,
            emb_type=args.num_emb_type,
            n_freq=args.num_emb_freqs,
            n_bins=args.num_emb_bins,
        )
        self.edge_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_out_dim),
            torch.nn.Linear(edge_out_dim, d_out),
            torch.nn.ReLU(),
            torch.nn.Linear(d_out, d_out),
            torch.nn.ReLU(),
        )

        # Message‑passing layers
        self.conv_v_to_c = BipartiteGraphConvolution(args.embedding_size)
        self.conv_c_to_v = BipartiteGraphConvolution(args.embedding_size)
        self.conv_v_to_c2 = BipartiteGraphConvolution(args.embedding_size)
        self.conv_c_to_v2 = BipartiteGraphConvolution(args.embedding_size)

        # Heads
        self.var_head = nn.Sequential(
            nn.Linear(d_out, d_out), nn.ReLU(), nn.Linear(d_out, 1, bias=False)
        )
        self.cons_head = nn.Sequential(
            nn.Linear(d_out, d_out), nn.ReLU(), nn.Linear(d_out, 1, bias=False)
        )
        self.lambda_act = nn.Softplus()  # smooth ≥0

    def encode(
        self,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rev_idx = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        periodic_c = self.cons_num_emb(constraint_features)
        periodic_v = self.var_num_emb(variable_features)
        periodic_e = self.edge_num_emb(edge_features)

        # numeric embeddings + small MLP
        c = self.cons_proj(periodic_c)
        v = self.var_proj(periodic_v)
        e = self.edge_proj(periodic_e)

        # two rounds of bipartite convolution
        c = self.conv_v_to_c(v, rev_idx, e, c)
        v = self.conv_c_to_v(c, edge_indices, e, v)

        c = self.conv_v_to_c2(v, rev_idx, e, c)
        v = self.conv_c_to_v2(c, edge_indices, e, v)

        return c, v

    def forward(
        self,
        constraint_features,  # (n_c, cons_nfeats)
        edge_indices,  # (2, |E|)
        edge_features,  # (|E|, edge_nfeats)
        variable_features,
    ):  # (n_v, var_nfeats)
        c, v = self.encode(
            constraint_features, edge_indices, edge_features, variable_features
        )
        x_all = self.var_head(v).squeeze(-1)  # (sum_n,)
        lam_all = self.lambda_act(self.cons_head(c).squeeze(-1))  # (sum_m,)  ≥ 0
        return x_all, lam_all

    @staticmethod
    def _build_numeric_block(
        in_feats: int,
        emb_type: str,
        n_freq: int,
        n_bins: int,
    ) -> Tuple[torch.nn.Module, int]:
        # Build numeric embedding and return (module, output_dim).

        if emb_type == "periodic":
            emb_layer = PeriodicEmbeddings(
                n_features=in_feats,
                n_frequencies=n_freq,
                lite=True,  # keep your current setting
            )
        elif emb_type == "pwl":
            emb_layer = PiecewiseLinearEmbeddings(
                n_features=in_feats,
                num_bins=n_bins,
            )
        else:  # "linear"
            return torch.nn.Identity(), in_feats

        # Determine the true output width by a forward pass on a dummy tensor.
        with torch.no_grad():
            dummy: torch.Tensor = torch.zeros(1, in_feats)
            out_dim: int = emb_layer(dummy).numel()

        return torch.nn.Sequential(emb_layer, torch.nn.Flatten(start_dim=1)), out_dim


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
