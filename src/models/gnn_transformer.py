import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

import wandb
from graphtrans.modules.masked_transformer_encoder import MaskedOnlyTransformerEncoder
from graphtrans.modules.transformer_encoder import TransformerNodeEncoder
from graphtrans.modules.utils import pad_batch


class BaseModel(nn.Module):
    @staticmethod
    def need_deg():
        return False

    @staticmethod
    def add_args(parser):
        return

    @staticmethod
    def name(args):
        raise NotImplementedError

    def __init__(self):
        super().__init__()

    def forward(self, batched_data, perturb=None):
        raise NotImplementedError

    def epoch_callback(self, epoch):
        return


class GNNTransformer(BaseModel):
    @staticmethod
    def get_emb_dim(args):
        return args.gnn_emb_dim

    @staticmethod
    def add_args(parser):
        TransformerNodeEncoder.add_args(parser)
        MaskedOnlyTransformerEncoder.add_args(parser)
        group = parser.add_argument_group("GNNTransformer - Training Config")
        group.add_argument("--pos_encoder", default=False, action="store_true")
        group.add_argument(
            "--pretrained_gnn",
            type=str,
            default=None,
            help="pretrained gnn_node node embedding path",
        )
        group.add_argument(
            "--freeze_gnn",
            type=int,
            default=None,
            help="Freeze gnn_node weight from epoch `freeze_gnn`",
        )

    @staticmethod
    def name(args):
        name = f"{args.model_type}-pooling={args.graph_pooling}"
        name += "-norm_input" if args.transformer_norm_input else ""
        name += f"+{args.gnn_type}"
        name += "-virtual" if args.gnn_virtual_node else ""
        name += f"-JK={args.gnn_JK}"
        name += f"-enc_layer={args.num_encoder_layers}"
        name += f"-enc_layer_masked={args.num_encoder_layers_masked}"
        name += f"-d={args.d_model}"
        name += f"-act={args.transformer_activation}"
        name += f"-tdrop={args.transformer_dropout}"
        name += f"-gdrop={args.gnn_dropout}"
        name += "-pretrained_gnn" if args.pretrained_gnn else ""
        name += f"-freeze_gnn={args.freeze_gnn}" if args.freeze_gnn is not None else ""
        name += "-prenorm" if args.transformer_prenorm else "-postnorm"
        return name

    def __init__(
        self,
        args,
        gnn_node: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.gnn_node = gnn_node
        self.node_type_embed = nn.Embedding(2, args.d_model)  # var/constr

        # load or freeze GNN weights
        if args.pretrained_gnn:
            # logger.info(self.gnn_node)
            state_dict = torch.load(args.pretrained_gnn)
            state_dict = self._gnn_node_state(state_dict["model"])
            logger.info("Load GNN state from: {}", state_dict.keys())
            self.gnn_node.load_state_dict(state_dict)
        self.freeze_gnn = args.freeze_gnn

        # projection from GNN dim to transformer dim
        gnn_emb_dim = gnn_node.out_dim
        self.gnn2transformer = nn.Linear(gnn_emb_dim, args.d_model)
        self.pos_encoder = (
            PositionalEncoding(args.d_model, dropout=0) if args.pos_encoder else None
        )
        # transformer encoders
        self.transformer_encoder = TransformerNodeEncoder(args)
        self.masked_transformer_encoder = MaskedOnlyTransformerEncoder(args)
        self.num_encoder_layers = args.num_encoder_layers
        self.num_encoder_layers_masked = args.num_encoder_layers_masked

        self.head = VarConstHead(args.d_model)

    def forward(self, batched_data, perturb=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # GNN encoding
        h_node = self.gnn_node(batched_data, perturb)
        h_node = self.gnn2transformer(h_node)  # [s, b, d_model]

        # Pad in the front
        padded_h_node, src_padding_mask, num_nodes, mask, max_num_nodes = pad_batch(
            h_node,
            batched_data.batch,
            get_mask=True,
        )

        node_type = torch.cat(
            [batched_data.is_constr_node, batched_data.is_var_node], dim=0
        ).long()[~src_padding_mask]  # shape (N_tot,)

        # add node type embedding
        transformer_out = padded_h_node + self.node_type_embed(node_type).view_as(
            padded_h_node
        )
        if self.pos_encoder is not None:
            transformer_out = self.pos_encoder(transformer_out)

        # masked-only encoder
        if self.num_encoder_layers_masked > 0:
            adj_list = getattr(batched_data, "adj_list", None)
            if adj_list is None:
                logger.warning(
                    "num_encoder_layers_masked>0 but no 'adj_list' on batch; skipping masked encoder."
                )
            else:
                padded_adj_list = torch.zeros(
                    (len(adj_list), max_num_nodes, max_num_nodes), device=h_node.device
                )
                for idx, adj_list_item in enumerate(adj_list):
                    N, _ = adj_list_item.shape
                    padded_adj_list[idx, 0:N, 0:N] = torch.from_numpy(adj_list_item)
                transformer_out = self.masked_transformer_encoder(
                    transformer_out.transpose(0, 1),
                    attn_mask=padded_adj_list,
                    valid_input_mask=src_padding_mask,
                ).transpose(0, 1)
        # vanilla encoder layers
        if self.num_encoder_layers > 0:
            transformer_out, _ = self.transformer_encoder(
                transformer_out, src_padding_mask
            )  # [s, b, h], [b, s]

        # remove padding & flatten to (N_tot, d)
        # transformer_out: S, B, d  →  B, S, d
        out_B_S_d = transformer_out.transpose(0, 1)
        valid_h = out_B_S_d[~src_padding_mask]  # (N_tot, d)

        # build masks that align with valid_h
        var_mask = batched_data.is_var_node.to(valid_h.device)
        constr_mask = batched_data.is_constr_node.to(valid_h.device)

        # per‑node scalar predictions
        x_hat, lam_hat = self.head(valid_h, var_mask, constr_mask)
        return x_hat, lam_hat

    def epoch_callback(self, epoch):
        # TODO: maybe unfreeze the gnn at the end.
        if self.freeze_gnn is not None and epoch >= self.freeze_gnn:
            logger.info(f"Freeze GNN weight after epoch: {epoch}")
            for param in self.gnn_node.parameters():
                param.requires_grad = False

    def _gnn_node_state(self, state_dict):
        module_name = "gnn_node"
        new_state_dict = dict()
        for k, v in state_dict.items():
            if module_name in k:
                new_key = k.split(".")
                module_index = new_key.index(module_name)
                new_key = ".".join(new_key[module_index + 1 :])
                new_state_dict[new_key] = v
        return new_state_dict

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class VarConstHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.to_xhat = nn.Linear(in_dim, 1)  # x is free (continuous)
        self.to_lam = nn.Sequential(
            nn.Linear(in_dim, 1),  # λ ≥ 0 for all ≤ rows
            nn.Softplus(),
        )

    def forward(self, h, var_mask, constr_mask):
        x_hat = self.to_xhat(h[var_mask]).squeeze(-1)
        lam_hat = self.to_lam(h[constr_mask]).squeeze(-1)
        return x_hat, lam_hat
