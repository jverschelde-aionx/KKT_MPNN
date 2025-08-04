import types
from argparse import Namespace

import pytest
import torch
import torch_geometric

from models.gnn_transformer import GNNTransformer
from models.losses import KKTLoss
from models.policy_encoder import (
    BipartiteNodeData,
    PolicyEncoder,
    collate,
)


#  Helpers
def make_dummy_graph(m: int, n: int) -> BipartiteNodeData:
    """Create a tiny graph with m constraints, n variables, all coeff = 1."""
    # features
    c_feats = torch.zeros(m, 4)  # cons_nfeats = 4
    v_feats = torch.zeros(n, 6)  # var_nfeats  = 6
    # fully‑connected bipartite incidence
    rows, cols = torch.meshgrid(torch.arange(m), torch.arange(n), indexing="ij")
    edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)
    edge_attr = torch.ones(edge_index.size(1), 1)  # edge_nfeats = 1
    return BipartiteNodeData(c_feats, edge_index, edge_attr, v_feats)


def dummy_args(**kwargs):
    base = dict(
        embedding_size=64,
        cons_nfeats=4,
        edge_nfeats=1,
        var_nfeats=6,
        # Transformer
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        transformer_dropout=0.1,
        transformer_activation="relu",
        num_encoder_layers=2,
        num_encoder_layers_masked=0,
        transformer_prenorm=False,
        transformer_norm_input=False,
        pos_encoder=False,
        max_input_len=10000,
        graph_pooling="mean",
        # GNN
        gnn_type="gcn",
        gnn_virtual_node=False,
        gnn_dropout=0.0,
        gnn_num_layer=3,
        gnn_emb_dim=64,
        gnn_JK="last",
        gnn_residual=False,
        max_seq_len=None,
        pretrained_gnn=None,
        freeze_gnn=None,
        num_emb_type="periodic",
        num_emb_bins=32,
        num_emb_freqs=16,
    )
    base.update(kwargs)
    return Namespace(**base)


#  Tests
def test_zero_arg_constructor():
    """Batch must be able to build a template with no args."""
    tmpl = BipartiteNodeData()  # should not raise
    assert not hasattr(tmpl, "constraint_features")


def test_collate_shapes():
    g1 = make_dummy_graph(2, 3)
    g2 = make_dummy_graph(4, 1)
    batch = collate([g1, g2])

    (batch_graph, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes) = batch

    # masks length equals pad length
    assert b_pad.shape == b_mask.shape
    assert c_pad.shape == c_mask.shape
    # sparse list length equals batch size
    assert len(A_list) == 2
    # node‑type masks were concatenated
    assert batch_graph.is_var_node.size(0) == (2 + 3) + (4 + 1)


def test_model_forward_and_loss():
    g1 = make_dummy_graph(3, 5)
    g2 = make_dummy_graph(1, 2)
    batch = collate([g1, g2])
    (batch_graph, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes) = batch

    args = dummy_args()
    encoder = PolicyEncoder(args)
    model = GNNTransformer(num_tasks=999, args=args, gnn_node=encoder)

    # forward
    x_hat, lam_hat = model(batch_graph)

    # lengths must match masks
    assert x_hat.size(0) == batch_graph.is_var_node.sum()
    assert lam_hat.size(0) == batch_graph.is_constr_node.sum()

    # KKT loss must return finite scalar value
    loss_fn = KKTLoss(m=4, n=7)  # numbers are unused in new version
    loss = loss_fn(
        x_hat, lam_hat, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes
    )
    assert torch.isfinite(loss)


@pytest.mark.parametrize("m,n", [(2, 3), (10, 10), (50, 4)])
def test_policy_encoder_dim(m, n):
    """Embedding dim must be encoder.out_dim."""
    args = dummy_args()
    enc = PolicyEncoder(args)
    g = make_dummy_graph(m, n)
    h = enc(g)
    assert h.size(1) == enc.out_dim == args.embedding_size
