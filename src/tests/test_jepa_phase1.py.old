"""
Unit tests for JEPA Phase 1 implementation.

Tests cover:
1. JEPA utility functions (jepa_utils.py)
2. KKTNetMLP JEPA extensions
3. GNNPolicy JEPA extensions
4. Integration tests

Run with: pytest src/tests/test_jepa_phase1.py -v
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch_geometric
from torch_geometric.data import Data, Batch

from models.jepa_utils import (
    ema_update,
    cosine_pred_loss,
    make_lp_jepa_views,
    make_gnn_views,
    jepa_loss_mlp,
    jepa_loss_gnn,
)
from models.models import KKTNetMLP, GNNPolicy


class TestJEPAUtilities:
    """Test JEPA utility functions from jepa_utils.py"""

    def test_ema_update_modifies_target(self):
        """Test that EMA update correctly modifies target model parameters"""
        # Create two simple models
        online = torch.nn.Linear(10, 5)
        target = torch.nn.Linear(10, 5)

        # Initialize with different weights
        torch.nn.init.ones_(online.weight)
        torch.nn.init.zeros_(target.weight)

        # Store original target weights
        target_weight_before = target.weight.data.clone()

        # Apply EMA update with m=0.5 for easy calculation
        m = 0.5
        ema_update(target, online, m=m)

        # Check that target has changed
        assert not torch.allclose(target.weight.data, target_weight_before), \
            "Target weights should have changed after EMA update"

        # Check formula: target = m * target + (1-m) * online
        # Expected: 0.5 * 0 + 0.5 * 1 = 0.5
        expected = m * target_weight_before + (1 - m) * online.weight.data
        assert torch.allclose(target.weight.data, expected, atol=1e-6), \
            f"EMA update formula incorrect. Expected {expected[0, 0]}, got {target.weight.data[0, 0]}"

    def test_ema_update_with_default_momentum(self):
        """Test EMA update with default momentum (0.996)"""
        online = torch.nn.Linear(5, 5)
        target = torch.nn.Linear(5, 5)

        torch.nn.init.constant_(online.weight, 1.0)
        torch.nn.init.constant_(target.weight, 0.0)

        ema_update(target, online)  # Default m=0.996

        # Target should change very slowly with high momentum
        # Expected: 0.996 * 0 + 0.004 * 1 = 0.004
        expected_value = 0.004
        assert torch.allclose(target.weight.data, torch.tensor(expected_value), atol=1e-5), \
            f"Expected ~{expected_value}, got {target.weight.data[0, 0]}"

    def test_cosine_pred_loss_identical_inputs(self):
        """Test cosine loss returns 0 for identical inputs"""
        pred = torch.randn(8, 128)
        pred = torch.nn.functional.normalize(pred, dim=-1)

        loss = cosine_pred_loss(pred, pred)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5), \
            f"Loss for identical inputs should be 0, got {loss.item()}"

    def test_cosine_pred_loss_opposite_directions(self):
        """Test cosine loss returns ~4 for opposite directions"""
        pred = torch.randn(8, 128)
        pred = torch.nn.functional.normalize(pred, dim=-1)
        target = -pred  # Opposite direction

        loss = cosine_pred_loss(pred, target)

        assert torch.allclose(loss, torch.tensor(4.0), atol=1e-5), \
            f"Loss for opposite directions should be 4, got {loss.item()}"

    def test_cosine_pred_loss_orthogonal(self):
        """Test cosine loss for orthogonal vectors"""
        # Create orthogonal vectors
        pred = torch.zeros(8, 128)
        target = torch.zeros(8, 128)
        pred[:, 0] = 1.0  # First dimension
        target[:, 1] = 1.0  # Second dimension

        loss = cosine_pred_loss(pred, target)

        # Orthogonal: cos = 0, loss = 2 - 2*0 = 2
        assert torch.allclose(loss, torch.tensor(2.0), atol=1e-5), \
            f"Loss for orthogonal vectors should be 2, got {loss.item()}"

    def test_lp_masking_shapes(self):
        """Test LP-aware masking creates correct shapes"""
        B, M, N = 4, 10, 8
        A = torch.randn(B, M, N)
        b = torch.randn(B, M)
        c = torch.randn(B, N)
        mask_m = torch.full((B,), M, dtype=torch.long)
        mask_n = torch.full((B,), N, dtype=torch.long)

        x_online, x_target = make_lp_jepa_views(
            A, b, c, mask_m, mask_n,
            r_entry_on=0.3, r_row_on=0.2, r_col_on=0.2,
            r_entry_tg=0.1, r_row_tg=0.05, r_col_tg=0.05
        )

        expected_shape = (B, M * N + M + N)
        assert x_online.shape == expected_shape, \
            f"Online view shape incorrect: expected {expected_shape}, got {x_online.shape}"
        assert x_target.shape == expected_shape, \
            f"Target view shape incorrect: expected {expected_shape}, got {x_target.shape}"

    def test_lp_masking_creates_different_views(self):
        """Test that online and target views are different"""
        B, M, N = 4, 10, 8
        A = torch.randn(B, M, N)
        b = torch.randn(B, M)
        c = torch.randn(B, N)
        mask_m = torch.full((B,), M, dtype=torch.long)
        mask_n = torch.full((B,), N, dtype=torch.long)

        x_online, x_target = make_lp_jepa_views(
            A, b, c, mask_m, mask_n,
            r_entry_on=0.4, r_row_on=0.2, r_col_on=0.2,
            r_entry_tg=0.1, r_row_tg=0.05, r_col_tg=0.05
        )

        # Views should be different due to different masking ratios
        assert not torch.allclose(x_online, x_target), \
            "Online and target views should be different with asymmetric masking"

    def test_lp_masking_respects_padding(self):
        """Test that masking only occurs within real region (respects mask_m, mask_n)"""
        B, M, N = 2, 10, 8
        A = torch.randn(B, M, N)
        b = torch.randn(B, M)
        c = torch.randn(B, N)

        # Sample 0: all real, Sample 1: only first 5 constraints and 4 variables are real
        mask_m = torch.tensor([M, 5], dtype=torch.long)
        mask_n = torch.tensor([N, 4], dtype=torch.long)

        # Fill padding region with sentinel values
        A[1, 5:, :] = 999.0  # Padding constraints
        A[1, :, 4:] = 999.0  # Padding variables
        b[1, 5:] = 999.0
        c[1, 4:] = 999.0

        x_online, x_target = make_lp_jepa_views(
            A, b, c, mask_m, mask_n,
            r_entry_on=0.5, r_row_on=0.3, r_col_on=0.3,
            r_entry_tg=0.0, r_row_tg=0.0, r_col_tg=0.0
        )

        # Extract sample 1 data
        A_flat = x_online[1, :M*N].reshape(M, N)
        b_flat = x_online[1, M*N:M*N+M]
        c_flat = x_online[1, M*N+M:]

        # Check that padding regions are unchanged
        assert torch.allclose(A_flat[5:, :], torch.tensor(999.0)), \
            "Padding constraint rows should not be masked"
        assert torch.allclose(A_flat[:, 4:], torch.tensor(999.0)), \
            "Padding variable columns should not be masked"
        assert torch.allclose(b_flat[5:], torch.tensor(999.0)), \
            "Padding b values should not be masked"
        assert torch.allclose(c_flat[4:], torch.tensor(999.0)), \
            "Padding c values should not be masked"

    def test_lp_masking_zero_ratios(self):
        """Test LP masking with zero ratios produces clean target"""
        B, M, N = 2, 10, 8
        A = torch.randn(B, M, N)
        b = torch.randn(B, M)
        c = torch.randn(B, N)
        mask_m = torch.full((B,), M, dtype=torch.long)
        mask_n = torch.full((B,), N, dtype=torch.long)

        # Create views with zero target masking
        x_online, x_target = make_lp_jepa_views(
            A, b, c, mask_m, mask_n,
            r_entry_on=0.4, r_row_on=0.2, r_col_on=0.2,
            r_entry_tg=0.0, r_row_tg=0.0, r_col_tg=0.0  # No target masking
        )

        # Reconstruct original input
        A_flat = A.flatten(start_dim=1)
        x_original = torch.cat([A_flat, b, c], dim=1)

        # Target should match original (no masking)
        assert torch.allclose(x_target, x_original), \
            "Target with zero masking ratios should match original input"

    def test_gnn_masking_zeros_features(self):
        """Test that GNN masking zeros out node features"""
        # Create a simple bipartite graph
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
            edge_attr=edge_feat  # Fixed: use edge_attr instead of edge_features
        )
        batch_graph = Batch.from_data_list([graph])

        ctx_graph, tgt_graph, mask_cons, mask_vars = make_gnn_views(batch_graph, mask_ratio=0.3)

        # Check that masked constraint nodes are zeroed
        masked_cons_indices = mask_cons.nonzero(as_tuple=True)[0]
        if len(masked_cons_indices) > 0:
            assert torch.allclose(
                ctx_graph.constraint_features[masked_cons_indices],
                torch.tensor(0.0)
            ), "Masked constraint features should be zero"

        # Check that masked variable nodes are zeroed
        masked_var_indices = mask_vars.nonzero(as_tuple=True)[0]
        if len(masked_var_indices) > 0:
            assert torch.allclose(
                ctx_graph.variable_features[masked_var_indices],
                torch.tensor(0.0)
            ), "Masked variable features should be zero"

    def test_gnn_masking_preserves_graph_structure(self):
        """Test that GNN masking preserves edge structure"""
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
            edge_attr=edge_feat  # Fixed: use edge_attr instead of edge_features
        )
        batch_graph = Batch.from_data_list([graph])

        ctx_graph, tgt_graph, mask_cons, mask_vars = make_gnn_views(batch_graph, mask_ratio=0.3)

        # Check that edge structure is preserved
        assert torch.equal(ctx_graph.edge_index, batch_graph.edge_index), \
            "Edge indices should be unchanged"
        assert torch.equal(ctx_graph.edge_attr, batch_graph.edge_attr), \
            "Edge features should be unchanged"  # Fixed: use edge_attr

    def test_jepa_loss_mlp_returns_scalar(self):
        """Test that MLP JEPA loss returns a scalar"""
        m, n = 10, 8
        batch_size = 4

        online_model = KKTNetMLP(m, n, hidden=64, jepa_embed_dim=32)
        target_model = KKTNetMLP(m, n, hidden=64, jepa_embed_dim=32)

        x_online = torch.randn(batch_size, m * n + m + n)
        x_target = torch.randn(batch_size, m * n + m + n)

        loss = jepa_loss_mlp(online_model, target_model, x_online, x_target, mode="ema")

        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert loss.item() <= 4, "Cosine loss should be at most 4"

    def test_jepa_loss_mlp_simsiam_mode(self):
        """Test MLP JEPA loss in SimSiam mode (shared encoder)"""
        m, n = 10, 8
        batch_size = 4

        online_model = KKTNetMLP(m, n, hidden=64, jepa_embed_dim=32)
        # In SimSiam, target_model is not used (encoder is shared)
        target_model = None

        x_online = torch.randn(batch_size, m * n + m + n)
        x_target = torch.randn(batch_size, m * n + m + n)

        loss = jepa_loss_mlp(online_model, online_model, x_online, x_target, mode="simsiam")

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestKKTNetMLPExtensions:
    """Test JEPA extensions to KKTNetMLP"""

    def test_encode_trunk_shape(self):
        """Test that encode_trunk returns correct shape"""
        m, n = 10, 8
        hidden = 128
        batch_size = 4

        model = KKTNetMLP(m, n, hidden=hidden)
        x = torch.randn(batch_size, m * n + m + n)

        z = model.encode_trunk(x)

        assert z.shape == (batch_size, hidden), \
            f"Expected shape {(batch_size, hidden)}, got {z.shape}"

    def test_jepa_embed_normalization(self):
        """Test that jepa_embed returns L2-normalized vectors"""
        m, n = 10, 8
        jepa_dim = 64
        batch_size = 4

        model = KKTNetMLP(m, n, jepa_embed_dim=jepa_dim)
        x = torch.randn(batch_size, m * n + m + n)

        z = model.jepa_embed(x)

        # Check shape
        assert z.shape == (batch_size, jepa_dim), \
            f"Expected shape {(batch_size, jepa_dim)}, got {z.shape}"

        # Check L2-normalization
        norms = torch.norm(z, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), \
            f"Embeddings should be L2-normalized (norm=1), got norms: {norms}"

    def test_jepa_pred_normalization(self):
        """Test that jepa_pred returns L2-normalized vectors"""
        m, n = 10, 8
        jepa_dim = 64
        batch_size = 4

        model = KKTNetMLP(m, n, jepa_embed_dim=jepa_dim)
        z = torch.randn(batch_size, jepa_dim)

        p = model.jepa_pred(z)

        # Check shape
        assert p.shape == (batch_size, jepa_dim), \
            f"Expected shape {(batch_size, jepa_dim)}, got {p.shape}"

        # Check L2-normalization
        norms = torch.norm(p, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), \
            f"Predictions should be L2-normalized (norm=1), got norms: {norms}"

    def test_forward_backward_compatibility(self):
        """Test that forward method still works correctly"""
        m, n = 10, 8
        batch_size = 4

        model = KKTNetMLP(m, n)
        x = torch.randn(batch_size, m * n + m + n)

        # Forward pass should work
        y = model.forward(x)

        # Check output shape
        assert y.shape == (batch_size, n + m), \
            f"Expected shape {(batch_size, n + m)}, got {y.shape}"

        # Check that lambda values are non-negative (ReLU applied)
        lam = y[:, n:]
        assert torch.all(lam >= 0), "Lambda values should be non-negative"

    def test_jepa_components_dont_affect_forward(self):
        """Test that JEPA components don't interfere with KKT prediction"""
        m, n = 10, 8
        batch_size = 4

        model = KKTNetMLP(m, n)
        x = torch.randn(batch_size, m * n + m + n)

        # Get KKT prediction
        y_kkt = model.forward(x)

        # Use JEPA components
        z_embed = model.jepa_embed(x)
        p_pred = model.jepa_pred(z_embed)

        # KKT prediction should be unchanged
        y_kkt_after = model.forward(x)
        assert torch.allclose(y_kkt, y_kkt_after), \
            "JEPA operations should not affect KKT predictions"


class TestGNNPolicyExtensions:
    """Test JEPA extensions to GNNPolicy"""

    @pytest.fixture
    def gnn_args(self):
        """Create minimal args object for GNNPolicy"""
        class Args:
            embedding_size = 64
            cons_nfeats = 4
            edge_nfeats = 1
            var_nfeats = 18
            num_emb_type = "periodic"
            num_emb_bins = 32
            num_emb_freqs = 16
        return Args()

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for testing"""
        num_cons = 10
        num_vars = 8
        num_edges = 30

        cons_feat = torch.randn(num_cons, 4)
        var_feat = torch.randn(num_vars, 18)
        edge_index = torch.randint(0, min(num_cons, num_vars), (2, num_edges))
        edge_feat = torch.randn(num_edges, 1)

        return cons_feat, edge_index, edge_feat, var_feat

    def test_jepa_embed_nodes_returns_tuple(self, gnn_args, sample_graph_data):
        """Test that jepa_embed_nodes returns tuple of embeddings"""
        model = GNNPolicy(gnn_args)
        cons_feat, edge_index, edge_feat, var_feat = sample_graph_data

        result = model.jepa_embed_nodes(cons_feat, edge_index, edge_feat, var_feat)

        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return (cons_emb, var_emb)"

    def test_jepa_embed_nodes_normalization(self, gnn_args, sample_graph_data):
        """Test that jepa_embed_nodes returns L2-normalized embeddings"""
        model = GNNPolicy(gnn_args)
        cons_feat, edge_index, edge_feat, var_feat = sample_graph_data

        cons_emb, var_emb = model.jepa_embed_nodes(cons_feat, edge_index, edge_feat, var_feat)

        # Check constraint embeddings
        cons_norms = torch.norm(cons_emb, dim=-1)
        assert torch.allclose(cons_norms, torch.ones(len(cons_feat)), atol=1e-5), \
            f"Constraint embeddings should be L2-normalized, got norms: {cons_norms}"

        # Check variable embeddings
        var_norms = torch.norm(var_emb, dim=-1)
        assert torch.allclose(var_norms, torch.ones(len(var_feat)), atol=1e-5), \
            f"Variable embeddings should be L2-normalized, got norms: {var_norms}"

    def test_jepa_embed_nodes_shapes(self, gnn_args, sample_graph_data):
        """Test that jepa_embed_nodes returns correct shapes"""
        model = GNNPolicy(gnn_args)
        cons_feat, edge_index, edge_feat, var_feat = sample_graph_data
        jepa_dim = 128  # Default from model definition

        cons_emb, var_emb = model.jepa_embed_nodes(cons_feat, edge_index, edge_feat, var_feat)

        assert cons_emb.shape == (len(cons_feat), jepa_dim), \
            f"Expected constraint shape {(len(cons_feat), jepa_dim)}, got {cons_emb.shape}"
        assert var_emb.shape == (len(var_feat), jepa_dim), \
            f"Expected variable shape {(len(var_feat), jepa_dim)}, got {var_emb.shape}"

    def test_forward_backward_compatibility(self, gnn_args, sample_graph_data):
        """Test that forward method still works correctly"""
        model = GNNPolicy(gnn_args)
        cons_feat, edge_index, edge_feat, var_feat = sample_graph_data

        # Forward pass should work
        x_pred, lam_pred = model.forward(cons_feat, edge_index, edge_feat, var_feat)

        # Check shapes
        assert x_pred.shape == (len(var_feat),), \
            f"Expected x shape {(len(var_feat),)}, got {x_pred.shape}"
        assert lam_pred.shape == (len(cons_feat),), \
            f"Expected lambda shape {(len(cons_feat),)}, got {lam_pred.shape}"

        # Check lambda non-negativity
        assert torch.all(lam_pred >= 0), "Lambda values should be non-negative"

    def test_jepa_components_dont_affect_forward(self, gnn_args, sample_graph_data):
        """Test that JEPA components don't interfere with KKT prediction"""
        model = GNNPolicy(gnn_args)
        cons_feat, edge_index, edge_feat, var_feat = sample_graph_data

        # Get KKT prediction
        x_pred1, lam_pred1 = model.forward(cons_feat, edge_index, edge_feat, var_feat)

        # Use JEPA components
        cons_emb, var_emb = model.jepa_embed_nodes(cons_feat, edge_index, edge_feat, var_feat)

        # KKT prediction should be unchanged
        x_pred2, lam_pred2 = model.forward(cons_feat, edge_index, edge_feat, var_feat)

        assert torch.allclose(x_pred1, x_pred2), \
            "JEPA operations should not affect x predictions"
        assert torch.allclose(lam_pred1, lam_pred2), \
            "JEPA operations should not affect lambda predictions"


class TestIntegration:
    """Integration tests for JEPA components"""

    def test_mlp_model_instantiation(self):
        """Test that KKTNetMLP can be instantiated with JEPA components"""
        m, n = 10, 8
        model = KKTNetMLP(m, n, hidden=128, jepa_embed_dim=64)

        # Check that JEPA components exist
        assert hasattr(model, 'jepa_proj'), "Model should have jepa_proj"
        assert hasattr(model, 'jepa_pred_net'), "Model should have jepa_pred_net"

    def test_gnn_model_instantiation(self):
        """Test that GNNPolicy can be instantiated with JEPA components"""
        class Args:
            embedding_size = 64
            cons_nfeats = 4
            edge_nfeats = 1
            var_nfeats = 18
            num_emb_type = "periodic"
            num_emb_bins = 32
            num_emb_freqs = 16

        model = GNNPolicy(Args())

        # Check that JEPA components exist
        assert hasattr(model, 'cons_jepa_proj'), "Model should have cons_jepa_proj"
        assert hasattr(model, 'cons_jepa_pred'), "Model should have cons_jepa_pred"
        assert hasattr(model, 'var_jepa_proj'), "Model should have var_jepa_proj"
        assert hasattr(model, 'var_jepa_pred'), "Model should have var_jepa_pred"

    def test_mlp_end_to_end_jepa_training_step(self):
        """Test complete MLP JEPA training step"""
        m, n = 10, 8
        batch_size = 4

        # Create models
        online_model = KKTNetMLP(m, n, hidden=64, jepa_embed_dim=32)
        target_model = KKTNetMLP(m, n, hidden=64, jepa_embed_dim=32)

        # Initialize target with online weights
        target_model.load_state_dict(online_model.state_dict())

        # Create sample data
        A = torch.randn(batch_size, m, n)
        b = torch.randn(batch_size, m)
        c = torch.randn(batch_size, n)
        mask_m = torch.full((batch_size,), m, dtype=torch.long)
        mask_n = torch.full((batch_size,), n, dtype=torch.long)

        # Create views
        x_online, x_target = make_lp_jepa_views(
            A, b, c, mask_m, mask_n,
            r_entry_on=0.3, r_row_on=0.2, r_col_on=0.2,
            r_entry_tg=0.1, r_row_tg=0.05, r_col_tg=0.05
        )

        # Compute loss
        loss = jepa_loss_mlp(online_model, target_model, x_online, x_target, mode="ema")

        # Should be able to backprop
        loss.backward()

        # Check that gradients exist for online model
        has_grads = any(p.grad is not None for p in online_model.parameters() if p.requires_grad)
        assert has_grads, "Online model should have gradients after backward"

    def test_gnn_end_to_end_jepa_training_step(self):
        """Test complete GNN JEPA training step"""
        class Args:
            embedding_size = 32  # Smaller for faster test
            cons_nfeats = 4
            edge_nfeats = 1
            var_nfeats = 18
            num_emb_type = "periodic"
            num_emb_bins = 32
            num_emb_freqs = 16

        # Create models
        online_model = GNNPolicy(Args())
        target_model = GNNPolicy(Args())

        # Initialize target with online weights
        target_model.load_state_dict(online_model.state_dict())

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
            edge_attr=edge_feat  # Fixed: use edge_attr instead of edge_features
        )
        batch_graph = Batch.from_data_list([graph])

        # Create views
        ctx_graph, tgt_graph, mask_cons, mask_vars = make_gnn_views(batch_graph, mask_ratio=0.3)

        # Compute loss
        loss = jepa_loss_gnn(
            online_model, target_model,
            ctx_graph, tgt_graph,
            mask_cons, mask_vars,
            mode="ema"
        )

        # Should be able to backprop
        loss.backward()

        # Check that gradients exist for online model
        has_grads = any(p.grad is not None for p in online_model.parameters() if p.requires_grad)
        assert has_grads, "Online model should have gradients after backward"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
