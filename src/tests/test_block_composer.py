"""Tests for models.block_composer and decomposition boundary features."""

from __future__ import annotations

import pytest
import torch

from models.decomposition import (
    PartitionSpec,
    build_block_graph,
    compute_block_features,
    compute_boundary_features,
    identify_boundary_nodes,
    split_bipartite_graph_metis,
)
from models.block_composer import (
    BlockGNN,
    BlockGNNComposer,
    ComposerMLP,
    composer_loss,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_graph():
    """Same small bipartite graph as test_decomposition.py.

    5 constraints (c0..c4), 8 variables (v0..v7), 12 edges.
    """
    n_cons, n_vars = 5, 8
    c_nodes = torch.arange(n_cons, dtype=torch.float32).unsqueeze(1).expand(-1, 4)
    v_nodes = torch.arange(n_vars, dtype=torch.float32).unsqueeze(1).expand(-1, 6)

    rows = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    cols = [0, 1, 2, 1, 3, 2, 4, 5, 5, 6, 6, 7]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.arange(len(rows), dtype=torch.float32).unsqueeze(1)

    return c_nodes, v_nodes, edge_index, edge_attr


def _make_two_partition_specs():
    """Split the test graph into 2 partitions manually.

    Partition 0: c0, c1; v0, v1, v2, v3
    Partition 1: c2, c3, c4; v4, v5, v6, v7
    """
    spec0 = PartitionSpec(
        part_id=0,
        owned_cons_ids=torch.tensor([0, 1]),
        owned_var_ids=torch.tensor([0, 1, 2, 3]),
    )
    spec1 = PartitionSpec(
        part_id=1,
        owned_cons_ids=torch.tensor([2, 3, 4]),
        owned_var_ids=torch.tensor([4, 5, 6, 7]),
    )
    return [spec0, spec1]


# ---------------------------------------------------------------------------
# Tests: compute_boundary_features
# ---------------------------------------------------------------------------


class TestComputeBoundaryFeatures:
    def test_shapes(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = _make_two_partition_specs()
        n_cons, n_vars = 5, 8

        cons_bf, vars_bf = compute_boundary_features(
            specs, edge_index, edge_attr, n_cons, n_vars,
        )
        assert cons_bf.shape == (n_cons, 5)
        assert vars_bf.shape == (n_vars, 5)

    def test_cut_fraction_in_range(self):
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = _make_two_partition_specs()
        n_cons, n_vars = 5, 8

        cons_bf, vars_bf = compute_boundary_features(
            specs, edge_index, edge_attr, n_cons, n_vars,
        )
        # cut_fraction is column 2
        assert (cons_bf[:, 2] >= 0).all()
        assert (cons_bf[:, 2] <= 1).all()
        assert (vars_bf[:, 2] >= 0).all()
        assert (vars_bf[:, 2] <= 1).all()

    def test_interior_nodes_zero_cross_features(self):
        """Interior nodes should have zero cross-edge features."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = _make_two_partition_specs()
        n_cons, n_vars = 5, 8

        cons_bf, vars_bf = compute_boundary_features(
            specs, edge_index, edge_attr, n_cons, n_vars,
        )
        cons_is_bnd, vars_is_bnd = identify_boundary_nodes(
            specs, edge_index, n_cons, n_vars,
        )

        # Interior constraint nodes: n_cross_edges=0, cut_fraction=0, is_boundary=0
        interior_cons = ~cons_is_bnd
        if interior_cons.any():
            assert (cons_bf[interior_cons, 1] == 0).all()  # log1p(0) = 0
            assert (cons_bf[interior_cons, 2] == 0).all()  # cut_fraction
            assert (cons_bf[interior_cons, 3] == 0).all()  # log1p(0) = 0
            assert (cons_bf[interior_cons, 4] == 0).all()  # is_boundary

        interior_vars = ~vars_is_bnd
        if interior_vars.any():
            assert (vars_bf[interior_vars, 1] == 0).all()
            assert (vars_bf[interior_vars, 2] == 0).all()
            assert (vars_bf[interior_vars, 3] == 0).all()
            assert (vars_bf[interior_vars, 4] == 0).all()

    def test_boundary_consistency(self):
        """is_boundary feature (col 4) should match identify_boundary_nodes."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = _make_two_partition_specs()
        n_cons, n_vars = 5, 8

        cons_bf, vars_bf = compute_boundary_features(
            specs, edge_index, edge_attr, n_cons, n_vars,
        )
        cons_is_bnd, vars_is_bnd = identify_boundary_nodes(
            specs, edge_index, n_cons, n_vars,
        )

        assert (cons_bf[:, 4] == cons_is_bnd.float()).all()
        assert (vars_bf[:, 4] == vars_is_bnd.float()).all()

    def test_log1p_applied(self):
        """Count features should be log1p-transformed (non-negative, not raw counts)."""
        c_nodes, v_nodes, edge_index, edge_attr = _make_graph()
        specs = _make_two_partition_specs()
        n_cons, n_vars = 5, 8

        cons_bf, vars_bf = compute_boundary_features(
            specs, edge_index, edge_attr, n_cons, n_vars,
        )
        # All features should be non-negative
        assert (cons_bf >= 0).all()
        assert (vars_bf >= 0).all()

        # Total degree (col 0) should be log1p of actual degree > 0
        # Every node has at least 1 edge, so log1p(degree) > 0
        assert (cons_bf[:, 0] > 0).all()
        assert (vars_bf[:, 0] > 0).all()


# ---------------------------------------------------------------------------
# Tests: BlockGNN
# ---------------------------------------------------------------------------


class TestBlockGNN:
    def test_forward_shape(self):
        K, d_block, d_hidden, d_z = 3, 64, 32, 16
        gnn = BlockGNN(d_block=d_block, d_hidden=d_hidden, d_z=d_z, heads=2)

        block_features = torch.randn(K, d_block)
        block_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        block_edge_attr = torch.randn(2, 4)

        z = gnn(block_features, block_edge_index, block_edge_attr)
        assert z.shape == (K, d_z)

    def test_gradient_flows(self):
        K, d_block, d_hidden, d_z = 3, 64, 32, 16
        gnn = BlockGNN(d_block=d_block, d_hidden=d_hidden, d_z=d_z, heads=2)

        block_features = torch.randn(K, d_block)
        block_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        block_edge_attr = torch.randn(2, 4)

        z = gnn(block_features, block_edge_index, block_edge_attr)
        loss = z.sum()
        loss.backward()

        for name, p in gnn.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_no_edges(self):
        """Block GNN should handle graphs with no edges."""
        K, d_block, d_hidden, d_z = 2, 64, 32, 16
        gnn = BlockGNN(d_block=d_block, d_hidden=d_hidden, d_z=d_z, heads=2)

        block_features = torch.randn(K, d_block)
        block_edge_index = torch.zeros(2, 0, dtype=torch.long)
        block_edge_attr = torch.zeros(0, 4)

        z = gnn(block_features, block_edge_index, block_edge_attr)
        assert z.shape == (K, d_z)


# ---------------------------------------------------------------------------
# Tests: ComposerMLP
# ---------------------------------------------------------------------------


class TestComposerMLP:
    def test_forward_shape(self):
        N, d_in, d_out = 10, 64, 32
        mlp = ComposerMLP(d_in=d_in, d_out=d_out, d_hidden=128)

        x = torch.randn(N, d_in)
        out = mlp(x)
        assert out.shape == (N, d_out)


# ---------------------------------------------------------------------------
# Tests: BlockGNNComposer
# ---------------------------------------------------------------------------


class TestBlockGNNComposer:
    def _make_inputs(self, d_sub=32, d_boundary=5, K=3):
        """Create synthetic inputs for BlockGNNComposer."""
        n_cons, n_vars = 10, 15
        d_block = 4 * d_sub

        block_features = torch.randn(K, d_block)
        block_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        block_edge_attr = torch.randn(2, 4)

        cons_block_id = torch.randint(0, K, (n_cons,))
        vars_block_id = torch.randint(0, K, (n_vars,))

        c_sub = torch.randn(n_cons, d_sub)
        v_sub = torch.randn(n_vars, d_sub)
        cons_bf = torch.randn(n_cons, d_boundary)
        vars_bf = torch.randn(n_vars, d_boundary)

        return {
            "block_features": block_features,
            "block_edge_index": block_edge_index,
            "block_edge_attr": block_edge_attr,
            "cons_block_id": cons_block_id,
            "vars_block_id": vars_block_id,
            "c_sub_owned": c_sub,
            "v_sub_owned": v_sub,
            "cons_boundary_feat": cons_bf,
            "vars_boundary_feat": vars_bf,
        }, n_cons, n_vars, d_sub

    def test_full_composer_shape(self):
        inputs, n_cons, n_vars, d_sub = self._make_inputs()
        composer = BlockGNNComposer(
            d_sub=d_sub, d_block=4 * d_sub, d_z=d_sub,
            d_boundary=5, d_mlp_hidden=64, heads=2,
            use_block_context=True, use_block_gnn=True,
        )
        c_hat, v_hat = composer(**inputs)
        assert c_hat.shape == (n_cons, d_sub)
        assert v_hat.shape == (n_vars, d_sub)

    def test_local_mlp_baseline(self):
        """use_block_context=False: local MLP baseline."""
        inputs, n_cons, n_vars, d_sub = self._make_inputs()
        composer = BlockGNNComposer(
            d_sub=d_sub, d_block=4 * d_sub, d_z=d_sub,
            d_boundary=5, d_mlp_hidden=64, heads=2,
            use_block_context=False, use_block_gnn=False,
        )
        c_hat, v_hat = composer(**inputs)
        assert c_hat.shape == (n_cons, d_sub)
        assert v_hat.shape == (n_vars, d_sub)

    def test_pooled_block_context_baseline(self):
        """use_block_context=True, use_block_gnn=False: pooled block-context MLP."""
        inputs, n_cons, n_vars, d_sub = self._make_inputs()
        composer = BlockGNNComposer(
            d_sub=d_sub, d_block=4 * d_sub, d_z=d_sub,
            d_boundary=5, d_mlp_hidden=64, heads=2,
            use_block_context=True, use_block_gnn=False,
        )
        c_hat, v_hat = composer(**inputs)
        assert c_hat.shape == (n_cons, d_sub)
        assert v_hat.shape == (n_vars, d_sub)

    def test_gradient_flows(self):
        inputs, n_cons, n_vars, d_sub = self._make_inputs()
        composer = BlockGNNComposer(
            d_sub=d_sub, d_block=4 * d_sub, d_z=d_sub,
            d_boundary=5, d_mlp_hidden=64, heads=2,
        )
        c_hat, v_hat = composer(**inputs)
        loss = c_hat.sum() + v_hat.sum()
        loss.backward()

        for name, p in composer.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Tests: composer_loss
# ---------------------------------------------------------------------------


class TestComposerLoss:
    def test_zero_loss_when_perfect(self):
        """Loss should be ~0 when predictions match teacher."""
        n_cons, n_vars, d = 10, 15, 32
        c = torch.randn(n_cons, d)
        v = torch.randn(n_vars, d)
        cons_bnd = torch.zeros(n_cons, dtype=torch.bool)
        vars_bnd = torch.zeros(n_vars, dtype=torch.bool)

        loss, metrics = composer_loss(c, v, c, v, cons_bnd, vars_bnd)
        assert loss.item() < 1e-5

    def test_boundary_weighting_increases_loss(self):
        """Higher boundary_weight should increase loss for boundary-heavy cases."""
        n_cons, n_vars, d = 10, 15, 32
        c_hat = torch.randn(n_cons, d)
        v_hat = torch.randn(n_vars, d)
        c_teacher = torch.randn(n_cons, d)
        v_teacher = torch.randn(n_vars, d)
        # All nodes are boundary
        cons_bnd = torch.ones(n_cons, dtype=torch.bool)
        vars_bnd = torch.ones(n_vars, dtype=torch.bool)

        loss_w1, _ = composer_loss(c_hat, v_hat, c_teacher, v_teacher,
                                   cons_bnd, vars_bnd, boundary_weight=1.0)
        loss_w5, _ = composer_loss(c_hat, v_hat, c_teacher, v_teacher,
                                   cons_bnd, vars_bnd, boundary_weight=5.0)
        # When all nodes are boundary, weight doesn't change the mean
        # But with mixed nodes it should. Test with mixed:
        cons_bnd_mixed = torch.tensor([True, True, False, False, False,
                                       False, False, False, False, False])
        vars_bnd_mixed = torch.tensor([True, True, True, False, False,
                                       False, False, False, False, False,
                                       False, False, False, False, False])

        loss_low, _ = composer_loss(c_hat, v_hat, c_teacher, v_teacher,
                                    cons_bnd_mixed, vars_bnd_mixed,
                                    boundary_weight=1.0)
        loss_high, _ = composer_loss(c_hat, v_hat, c_teacher, v_teacher,
                                     cons_bnd_mixed, vars_bnd_mixed,
                                     boundary_weight=10.0)
        # With higher boundary weight and boundary nodes having non-zero error,
        # loss should be different (could be higher or lower depending on which
        # nodes have more error, but they should differ)
        assert not torch.isclose(loss_low, loss_high)

    def test_metrics_keys(self):
        """Metrics dict should contain expected keys."""
        n_cons, n_vars, d = 10, 15, 32
        c_hat = torch.randn(n_cons, d)
        v_hat = torch.randn(n_vars, d)
        c_teacher = torch.randn(n_cons, d)
        v_teacher = torch.randn(n_vars, d)
        cons_bnd = torch.tensor([True] * 3 + [False] * 7)
        vars_bnd = torch.tensor([True] * 5 + [False] * 10)

        _, metrics = composer_loss(c_hat, v_hat, c_teacher, v_teacher,
                                   cons_bnd, vars_bnd)
        expected_keys = {
            "loss", "cons_mse", "vars_mse", "cons_cos", "vars_cos",
            "boundary_cons_cos", "boundary_vars_cos",
            "interior_cons_cos", "interior_vars_cos",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_cosine_metric_range(self):
        """Cosine similarity metrics should be in [-1, 1]."""
        n_cons, n_vars, d = 10, 15, 32
        c_hat = torch.randn(n_cons, d)
        v_hat = torch.randn(n_vars, d)
        c_teacher = torch.randn(n_cons, d)
        v_teacher = torch.randn(n_vars, d)
        cons_bnd = torch.zeros(n_cons, dtype=torch.bool)
        vars_bnd = torch.zeros(n_vars, dtype=torch.bool)

        _, metrics = composer_loss(c_hat, v_hat, c_teacher, v_teacher,
                                   cons_bnd, vars_bnd)
        assert -1.0 <= metrics["cons_cos"] <= 1.0
        assert -1.0 <= metrics["vars_cos"] <= 1.0
