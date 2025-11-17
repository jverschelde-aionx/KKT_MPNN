"""
Simple unit test for LeJEPA implementation.

Run with: pytest src/tests/test_lejepa.py -v
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.models import KKTNetMLP, GNNPolicy
from models.losses import lejepa_loss_mlp, lejepa_loss_gnn
from models.jepa_utils import (
    SigRegWrapper,
    make_lp_lejepa_views,
    make_gnn_lejepa_views,
)
from torch_geometric.data import Data, Batch


def test_lejepa_mlp_forward():
    """Test LeJEPA MLP forward pass"""
    m, n = 10, 8
    batch_size = 4

    model = KKTNetMLP(m, n, hidden=64, jepa_embed_dim=32)
    x = torch.randn(batch_size, m * n + m + n)

    # Test KKT forward pass
    y = model.forward(x)
    assert y.shape == (batch_size, n + m), f"Expected {(batch_size, n + m)}, got {y.shape}"

    # Test LeJEPA embedding
    z = model.lejepa_embed(x)
    assert z.shape == (batch_size, 32), f"Expected {(batch_size, 32)}, got {z.shape}"

    print("✓ LeJEPA MLP forward pass works")


def test_lejepa_mlp_loss():
    """Test LeJEPA MLP loss computation"""
    m, n = 10, 8
    batch_size = 4

    model = KKTNetMLP(m, n, hidden=64, jepa_embed_dim=32)

    # Create sample data
    A = torch.randn(batch_size, m, n)
    b = torch.randn(batch_size, m)
    c = torch.randn(batch_size, n)
    mask_m = torch.ones((batch_size, m), dtype=torch.float32)
    mask_n = torch.ones((batch_size, n), dtype=torch.float32)

    # Create LeJEPA views
    x_globals, x_all = make_lp_lejepa_views(
        A, b, c, mask_m, mask_n,
        Vg=2, Vl=2,
        light=(0.10, 0.05, 0.05),
        heavy=(0.40, 0.20, 0.20),
    )

    # Create SIGReg regularizer
    sigreg = SigRegWrapper(num_slices=256, num_points=17)

    # Compute loss
    loss = lejepa_loss_mlp(model, x_globals, x_all, sigreg, lambd=0.05)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    print(f"✓ LeJEPA MLP loss computed: {loss.item():.4f}")


def test_lejepa_mlp_backward():
    """Test LeJEPA MLP backward pass"""
    m, n = 10, 8
    batch_size = 4

    model = KKTNetMLP(m, n, hidden=64, jepa_embed_dim=32)

    # Create sample data
    A = torch.randn(batch_size, m, n)
    b = torch.randn(batch_size, m)
    c = torch.randn(batch_size, n)
    mask_m = torch.ones((batch_size, m), dtype=torch.float32)
    mask_n = torch.ones((batch_size, n), dtype=torch.float32)

    # Create LeJEPA views
    x_globals, x_all = make_lp_lejepa_views(
        A, b, c, mask_m, mask_n,
        Vg=2, Vl=2,
    )

    # Create SIGReg regularizer
    sigreg = SigRegWrapper(num_slices=256, num_points=17)

    # Compute loss and backprop
    loss = lejepa_loss_mlp(model, x_globals, x_all, sigreg, lambd=0.05)
    loss.backward()

    # Check gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.parameters() if p.requires_grad)
    assert has_grads, "Model should have gradients after backward"

    print("✓ LeJEPA MLP backward pass works")


def test_lejepa_gnn_forward():
    """Test LeJEPA GNN forward pass"""
    class Args:
        embedding_size = 64
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 18
        num_emb_type = "periodic"
        num_emb_bins = 32
        num_emb_freqs = 16

    model = GNNPolicy(Args())

    # Create sample graph
    num_cons = 10
    num_vars = 8
    num_edges = 30

    cons_feat = torch.randn(num_cons, 4)
    var_feat = torch.randn(num_vars, 18)
    edge_index = torch.randint(0, min(num_cons, num_vars), (2, num_edges))
    edge_feat = torch.randn(num_edges, 1)

    # Test KKT forward pass
    x_pred, lam_pred = model.forward(cons_feat, edge_index, edge_feat, var_feat)
    assert x_pred.shape == (num_vars,), f"Expected {(num_vars,)}, got {x_pred.shape}"
    assert lam_pred.shape == (num_cons,), f"Expected {(num_cons,)}, got {lam_pred.shape}"

    # Test LeJEPA embedding
    cons_emb, var_emb = model.lejepa_embed_nodes(cons_feat, edge_index, edge_feat, var_feat)
    assert cons_emb.shape[0] == num_cons, f"Expected {num_cons} constraint embeddings"
    assert var_emb.shape[0] == num_vars, f"Expected {num_vars} variable embeddings"

    print("✓ LeJEPA GNN forward pass works")


def test_lejepa_gnn_loss():
    """Test LeJEPA GNN loss computation"""
    class Args:
        embedding_size = 32
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 18
        num_emb_type = "periodic"
        num_emb_bins = 32
        num_emb_freqs = 16

    model = GNNPolicy(Args())

    # Create sample graph
    num_cons = 10
    num_vars = 8
    num_edges = 30

    cons_feat = torch.randn(num_cons, 4)
    var_feat = torch.randn(num_vars, 18)
    edge_index = torch.randint(0, min(num_cons, num_vars), (2, num_edges))
    edge_feat = torch.randn(num_edges, 1)

    graph = Data(
        constraint_features=cons_feat,
        variable_features=var_feat,
        edge_index=edge_index,
        edge_attr=edge_feat,
    )
    batch_graph = Batch.from_data_list([graph])

    # Create LeJEPA views
    globals_, alls_ = make_gnn_lejepa_views(
        batch_graph, Vg=2, Vl=2, light_ratio=0.05, heavy_ratio=0.30
    )

    # Create SIGReg regularizer
    sigreg = SigRegWrapper(num_slices=256, num_points=17)

    # Compute loss
    loss = lejepa_loss_gnn(model, globals_, alls_, sigreg, lambd=0.05)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    print(f"✓ LeJEPA GNN loss computed: {loss.item():.4f}")


def test_lejepa_gnn_backward():
    """Test LeJEPA GNN backward pass"""
    class Args:
        embedding_size = 32
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 18
        num_emb_type = "periodic"
        num_emb_bins = 32
        num_emb_freqs = 16

    model = GNNPolicy(Args())

    # Create sample graph
    num_cons = 10
    num_vars = 8
    num_edges = 30

    cons_feat = torch.randn(num_cons, 4)
    var_feat = torch.randn(num_vars, 18)
    edge_index = torch.randint(0, min(num_cons, num_vars), (2, num_edges))
    edge_feat = torch.randn(num_edges, 1)

    graph = Data(
        constraint_features=cons_feat,
        variable_features=var_feat,
        edge_index=edge_index,
        edge_attr=edge_feat,
    )
    batch_graph = Batch.from_data_list([graph])

    # Create LeJEPA views
    globals_, alls_ = make_gnn_lejepa_views(batch_graph, Vg=2, Vl=2)

    # Create SIGReg regularizer
    sigreg = SigRegWrapper(num_slices=256, num_points=17)

    # Compute loss and backprop
    loss = lejepa_loss_gnn(model, globals_, alls_, sigreg, lambd=0.05)
    loss.backward()

    # Check gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.parameters() if p.requires_grad)
    assert has_grads, "Model should have gradients after backward"

    print("✓ LeJEPA GNN backward pass works")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing LeJEPA Implementation")
    print("="*60 + "\n")

    print("MLP Tests:")
    print("-" * 60)
    test_lejepa_mlp_forward()
    test_lejepa_mlp_loss()
    test_lejepa_mlp_backward()

    print("\nGNN Tests:")
    print("-" * 60)
    test_lejepa_gnn_forward()
    test_lejepa_gnn_loss()
    test_lejepa_gnn_backward()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
