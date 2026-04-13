"""Tests for split model modes: raw split, composer, and composer+pretrained.

Verifies that:
  - skip_composer=1 bypasses the composer and produces valid outputs
  - skip_composer=0 uses the composer and produces different outputs
  - pretrained encoder loading works with both modes
  - all three modes produce correct shapes and are differentiable
  - halo_hops=0/1/2 are accepted without error
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from configargparse import Namespace

from data.datasets import BipartiteNodeData
from data.split import (
    SplitInstanceBatch,
    SplitInstanceData,
    SplitPartitionData,
)
from models.split import SplitBlockBiJepaPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_SUB = 16
N_CONS_PER_BLOCK = 4
N_VARS_PER_BLOCK = 6
N_EDGES_PER_BLOCK = 8
K = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(skip_composer: int = 0, halo_hops: int = 0) -> Namespace:
    return Namespace(
        sigreg_slices=64,
        sigreg_points=5,
        lejepa_n_global_views=2,
        lejepa_n_local_views=2,
        lejepa_local_mask=0.4,
        lejepa_global_mask=0.1,
        lejepa_local_edge_mask=0.2,
        lejepa_global_edge_mask=0.05,
        lejepa_lambda=0.05,
        lejepa_std_loss_weight=0.0,
        cons_nfeats=9,
        var_nfeats=18,
        edge_nfeats=1,
        embedding_size=D_SUB,
        num_emb_type="linear",
        num_emb_freqs=4,
        num_emb_bins=4,
        lejepa_embed_dim=D_SUB,
        dropout=0.0,
        bipartite_conv="gcn",
        attn_heads=2,
        num_blocks=K,
        halo_hops=halo_hops,
        composer_d_z=D_SUB,
        composer_hidden=32,
        composer_heads=2,
        composer_dropout=0.0,
        use_block_context=1,
        use_block_gnn=1,
        skip_composer=skip_composer,
    )


def _make_partition(part_id, n_cons, n_vars, n_edges):
    c_feat = torch.randn(n_cons, 9)
    v_feat = torch.randn(n_vars, 18)
    rows = torch.randint(0, n_cons, (n_edges,))
    cols = torch.randint(0, n_vars, (n_edges,))
    edge_index = torch.stack([rows, cols], dim=0)
    edge_attr = torch.randn(n_edges, 1)

    graph = BipartiteNodeData(
        constraint_features=c_feat,
        edge_index=edge_index,
        edge_attr=edge_attr,
        variable_features=v_feat,
    )
    return SplitPartitionData(
        part_id=part_id,
        graph=graph,
        orig_cons_ids=torch.arange(n_cons),
        orig_var_ids=torch.arange(n_vars),
        owned_cons_local=torch.arange(n_cons),
        owned_var_local=torch.arange(n_vars),
    )


def _make_instance(name="test_inst"):
    n_cons = N_CONS_PER_BLOCK * K
    n_vars = N_VARS_PER_BLOCK * K

    partitions = []
    for k in range(K):
        c_off = k * N_CONS_PER_BLOCK
        v_off = k * N_VARS_PER_BLOCK
        part = _make_partition(k, N_CONS_PER_BLOCK, N_VARS_PER_BLOCK, N_EDGES_PER_BLOCK)
        part.orig_cons_ids = torch.arange(c_off, c_off + N_CONS_PER_BLOCK)
        part.orig_var_ids = torch.arange(v_off, v_off + N_VARS_PER_BLOCK)
        partitions.append(part)

    block_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    block_edge_attr = torch.randn(2, 4)

    return SplitInstanceData(
        name=name,
        partitions=partitions,
        block_edge_index=block_edge_index,
        block_edge_attr=block_edge_attr,
        cons_block_id=torch.cat([torch.full((N_CONS_PER_BLOCK,), k, dtype=torch.long) for k in range(K)]),
        vars_block_id=torch.cat([torch.full((N_VARS_PER_BLOCK,), k, dtype=torch.long) for k in range(K)]),
        cons_boundary_feat=torch.randn(n_cons, 5),
        vars_boundary_feat=torch.randn(n_vars, 5),
        cons_is_boundary=torch.zeros(n_cons, dtype=torch.bool),
        vars_is_boundary=torch.zeros(n_vars, dtype=torch.bool),
        n_cons=n_cons,
        n_vars=n_vars,
        A_dense=torch.randn(n_cons, n_vars),
        b_vec=torch.randn(n_cons),
        c_vec=torch.randn(n_vars),
    )


def _make_batch(n_instances=2):
    return SplitInstanceBatch(
        instances=[_make_instance(name=f"inst_{i}") for i in range(n_instances)]
    )


# ---------------------------------------------------------------------------
# Test: skip_composer flag is stored correctly
# ---------------------------------------------------------------------------


class TestSkipComposerFlag:

    def test_skip_composer_false_by_default(self):
        args = _make_args(skip_composer=0)
        model = SplitBlockBiJepaPolicy(args)
        assert model.skip_composer is False

    def test_skip_composer_true_when_set(self):
        args = _make_args(skip_composer=1)
        model = SplitBlockBiJepaPolicy(args)
        assert model.skip_composer is True


# ---------------------------------------------------------------------------
# Test: raw split mode (skip_composer=1) produces valid outputs
# ---------------------------------------------------------------------------


class TestRawSplitMode:

    @pytest.fixture
    def raw_model(self):
        args = _make_args(skip_composer=1)
        m = SplitBlockBiJepaPolicy(args)
        m.eval()
        return m

    @pytest.fixture
    def batch(self):
        return _make_batch(2)

    def test_predict_instance_shapes(self, raw_model, batch):
        inst = batch.instances[0]
        x, lam = raw_model.predict_instance(inst)
        assert x.shape == (inst.n_vars,)
        assert lam.shape == (inst.n_cons,)

    def test_lambda_nonnegative(self, raw_model, batch):
        inst = batch.instances[0]
        _, lam = raw_model.predict_instance(inst)
        assert (lam >= 0).all()

    def test_compose_instances_skips_composer(self, raw_model, batch):
        device = next(raw_model.parameters()).device
        composed = raw_model._compose_instances(batch.instances, device)
        assert len(composed) == len(batch.instances)
        for i, inst in enumerate(batch.instances):
            c_hat, v_hat = composed[i]
            assert c_hat.shape == (inst.n_cons, D_SUB)
            assert v_hat.shape == (inst.n_vars, D_SUB)

    def test_predict_batch_shapes(self, raw_model, batch):
        from jobs.finetune_split import _predict_batch

        device = next(raw_model.parameters()).device
        y, A, b, c, mask_m, mask_n, names = _predict_batch(raw_model, batch, device)
        B = len(batch.instances)
        n_max = max(inst.n_vars for inst in batch.instances)
        m_max = max(inst.n_cons for inst in batch.instances)
        assert y.shape == (B, n_max + m_max)
        assert A.shape == (B, m_max, n_max)

    def test_loss_is_differentiable(self, raw_model, batch):
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        raw_model.train()
        device = next(raw_model.parameters()).device
        y, A, b, c, mask_m, mask_n, _ = _predict_batch(raw_model, batch, device)
        loss, _ = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in raw_model.parameters()
        )
        assert has_grad

    def test_composer_params_have_no_grad(self, raw_model, batch):
        """When skip_composer=1, composer parameters should not receive gradients."""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        raw_model.train()
        device = next(raw_model.parameters()).device
        y, A, b, c, mask_m, mask_n, _ = _predict_batch(raw_model, batch, device)
        loss, _ = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        loss.backward()
        for name, p in raw_model.composer.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                f"Composer param '{name}' received gradient in raw split mode"
            )


# ---------------------------------------------------------------------------
# Test: composer mode (skip_composer=0) uses the composer
# ---------------------------------------------------------------------------


class TestComposerMode:

    @pytest.fixture
    def composer_model(self):
        args = _make_args(skip_composer=0)
        m = SplitBlockBiJepaPolicy(args)
        m.eval()
        return m

    @pytest.fixture
    def batch(self):
        return _make_batch(2)

    def test_predict_instance_shapes(self, composer_model, batch):
        inst = batch.instances[0]
        x, lam = composer_model.predict_instance(inst)
        assert x.shape == (inst.n_vars,)
        assert lam.shape == (inst.n_cons,)

    def test_compose_instances_uses_composer(self, composer_model, batch):
        device = next(composer_model.parameters()).device
        composed = composer_model._compose_instances(batch.instances, device)
        assert len(composed) == len(batch.instances)
        for i, inst in enumerate(batch.instances):
            c_hat, v_hat = composed[i]
            assert c_hat.shape == (inst.n_cons, D_SUB)
            assert v_hat.shape == (inst.n_vars, D_SUB)

    def test_loss_is_differentiable(self, composer_model, batch):
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        composer_model.train()
        device = next(composer_model.parameters()).device
        y, A, b, c, mask_m, mask_n, _ = _predict_batch(composer_model, batch, device)
        loss, _ = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in composer_model.parameters()
        )
        assert has_grad

    def test_composer_params_receive_grad(self, composer_model, batch):
        """When skip_composer=0, composer parameters should receive gradients."""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        composer_model.train()
        device = next(composer_model.parameters()).device
        y, A, b, c, mask_m, mask_n, _ = _predict_batch(composer_model, batch, device)
        loss, _ = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        loss.backward()
        has_composer_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in composer_model.composer.parameters()
        )
        assert has_composer_grad


# ---------------------------------------------------------------------------
# Test: raw vs composer produce different outputs (same weights)
# ---------------------------------------------------------------------------


class TestRawVsComposerOutputsDiffer:

    def test_outputs_differ_same_encoder(self):
        """With identical encoder weights, raw and composer modes should
        produce different predictions because the composer transforms
        the scattered embeddings."""
        torch.manual_seed(42)
        args_raw = _make_args(skip_composer=1)
        raw_model = SplitBlockBiJepaPolicy(args_raw)
        raw_model.eval()

        args_comp = _make_args(skip_composer=0)
        comp_model = SplitBlockBiJepaPolicy(args_comp)
        # Copy encoder weights from raw to composer model
        comp_model._encoder.load_state_dict(raw_model._encoder.state_dict())
        # Copy head weights
        comp_model.var_head.load_state_dict(raw_model.var_head.state_dict())
        comp_model.cons_head.load_state_dict(raw_model.cons_head.state_dict())
        comp_model.eval()

        torch.manual_seed(0)
        inst = _make_instance("compare")

        x_raw, lam_raw = raw_model.predict_instance(inst)
        x_comp, lam_comp = comp_model.predict_instance(inst)

        # They should differ because the composer transforms the embeddings
        assert not torch.allclose(x_raw, x_comp, atol=1e-6), (
            "Raw and composer outputs should differ"
        )


# ---------------------------------------------------------------------------
# Test: pretrained encoder loading works with both modes
# ---------------------------------------------------------------------------


class TestPretrainedEncoderLoading:

    def _save_encoder(self, model, path):
        model.save_encoder(str(path))

    def test_load_encoder_into_raw_model(self):
        args_src = _make_args(skip_composer=0)
        src_model = SplitBlockBiJepaPolicy(args_src)

        with tempfile.TemporaryDirectory() as tmpdir:
            enc_path = Path(tmpdir) / "encoder.pt"
            self._save_encoder(src_model, enc_path)

            args_raw = _make_args(skip_composer=1)
            raw_model = SplitBlockBiJepaPolicy(args_raw)
            raw_model.load_encoder(str(enc_path), strict=True)

            # Verify encoder weights match
            for (n1, p1), (n2, p2) in zip(
                src_model.encoder.named_parameters(),
                raw_model.encoder.named_parameters(),
            ):
                assert n1 == n2
                assert torch.equal(p1, p2), f"Encoder param {n1} mismatch after load"

    def test_load_encoder_into_composer_model(self):
        args_src = _make_args(skip_composer=0)
        src_model = SplitBlockBiJepaPolicy(args_src)

        with tempfile.TemporaryDirectory() as tmpdir:
            enc_path = Path(tmpdir) / "encoder.pt"
            self._save_encoder(src_model, enc_path)

            args_comp = _make_args(skip_composer=0)
            comp_model = SplitBlockBiJepaPolicy(args_comp)
            comp_model.load_encoder(str(enc_path), strict=True)

            for (n1, p1), (n2, p2) in zip(
                src_model.encoder.named_parameters(),
                comp_model.encoder.named_parameters(),
            ):
                assert n1 == n2
                assert torch.equal(p1, p2), f"Encoder param {n1} mismatch after load"

    def test_load_full_checkpoint_into_composer_model(self):
        """Simulates loading a pretrain_checkpoint (full model state)."""
        args = _make_args(skip_composer=0)
        src_model = SplitBlockBiJepaPolicy(args)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save({"model": src_model.state_dict()}, ckpt_path)

            dst_model = SplitBlockBiJepaPolicy(args)
            from jobs.finetune_split import _load_pretrain_checkpoint
            _load_pretrain_checkpoint(dst_model, str(ckpt_path))

            for (n1, p1), (n2, p2) in zip(
                src_model.named_parameters(),
                dst_model.named_parameters(),
            ):
                assert n1 == n2
                assert torch.equal(p1, p2), f"Param {n1} mismatch after checkpoint load"


# ---------------------------------------------------------------------------
# Test: halo_hops argument is accepted for all modes
# ---------------------------------------------------------------------------


class TestHaloHopsAccepted:

    @pytest.mark.parametrize("halo_hops", [0, 1, 2])
    @pytest.mark.parametrize("skip_composer", [0, 1])
    def test_model_constructs(self, halo_hops, skip_composer):
        args = _make_args(skip_composer=skip_composer, halo_hops=halo_hops)
        model = SplitBlockBiJepaPolicy(args)
        assert model.skip_composer == bool(skip_composer)

    @pytest.mark.parametrize("halo_hops", [0, 1, 2])
    @pytest.mark.parametrize("skip_composer", [0, 1])
    def test_forward_pass(self, halo_hops, skip_composer):
        args = _make_args(skip_composer=skip_composer, halo_hops=halo_hops)
        model = SplitBlockBiJepaPolicy(args)
        model.eval()
        inst = _make_instance()
        x, lam = model.predict_instance(inst)
        assert x.shape == (inst.n_vars,)
        assert lam.shape == (inst.n_cons,)


# ---------------------------------------------------------------------------
# Test: freeze / unfreeze encoder works with both modes
# ---------------------------------------------------------------------------


class TestFreezeUnfreezeWithModes:

    @pytest.mark.parametrize("skip_composer", [0, 1])
    def test_freeze_encoder(self, skip_composer):
        args = _make_args(skip_composer=skip_composer)
        model = SplitBlockBiJepaPolicy(args)
        model.freeze_encoder()
        for p in model.encoder.parameters():
            assert not p.requires_grad

    @pytest.mark.parametrize("skip_composer", [0, 1])
    def test_unfreeze_encoder(self, skip_composer):
        args = _make_args(skip_composer=skip_composer)
        model = SplitBlockBiJepaPolicy(args)
        model.freeze_encoder()
        model.unfreeze_encoder()
        for p in model.encoder.parameters():
            assert p.requires_grad

    def test_raw_mode_frozen_encoder_heads_still_train(self):
        """In raw mode with frozen encoder, heads should still get gradients."""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        args = _make_args(skip_composer=1)
        model = SplitBlockBiJepaPolicy(args)
        model.freeze_encoder()
        model.train()

        batch = _make_batch(2)
        device = next(model.parameters()).device
        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        loss, _ = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        loss.backward()

        head_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in list(model.var_head.parameters()) + list(model.cons_head.parameters())
        )
        assert head_has_grad, "Heads should receive gradients even with frozen encoder in raw mode"

        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        assert not encoder_has_grad, "Frozen encoder should not receive gradients"
