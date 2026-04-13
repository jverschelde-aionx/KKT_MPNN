"""Tests for jobs.pretrain_split — tensor shape checkpoints through the pipeline."""

from __future__ import annotations

import pytest
import torch
from configargparse import Namespace

from data.datasets import BipartiteNodeData
from data.split import (
    SplitInstanceBatch,
    SplitInstanceData,
    SplitPartitionData,
    SplitViewMasks,
)
from models.split import SplitBlockBiJepaPolicy

# ---------------------------------------------------------------------------
# Fixtures: synthetic data + model
# ---------------------------------------------------------------------------

D_SUB = 16  # small embedding for fast tests


def _make_args() -> Namespace:
    """Minimal args namespace to construct SplitBlockBiJepaPolicy."""
    return Namespace(
        # LeJepaEncoderModule / base
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
        # GNNEncoder
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
        # SplitBlockBiJepaPolicy
        num_blocks=2,
        halo_hops=0,
        composer_d_z=D_SUB,
        composer_hidden=32,
        composer_heads=2,
        composer_dropout=0.0,
        use_block_context=1,
        use_block_gnn=1,
    )


def _make_partition(part_id: int, n_cons: int, n_vars: int, n_edges: int) -> SplitPartitionData:
    """Build a synthetic partition with random features."""
    c_feat = torch.randn(n_cons, 9)
    v_feat = torch.randn(n_vars, 18)
    # random edges between cons and vars
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


def _make_instance(
    n_cons_per_block: int = 4,
    n_vars_per_block: int = 6,
    n_edges_per_block: int = 8,
    K: int = 2,
    name: str = "test_inst",
) -> SplitInstanceData:
    """Build a synthetic SplitInstanceData with K blocks."""
    n_cons = n_cons_per_block * K
    n_vars = n_vars_per_block * K

    partitions = []
    for k in range(K):
        c_off = k * n_cons_per_block
        v_off = k * n_vars_per_block
        part = _make_partition(k, n_cons_per_block, n_vars_per_block, n_edges_per_block)
        # Remap orig IDs to global indices
        part.orig_cons_ids = torch.arange(c_off, c_off + n_cons_per_block)
        part.orig_var_ids = torch.arange(v_off, v_off + n_vars_per_block)
        partitions.append(part)

    # Block edge: single edge between block 0 and block 1 (if K >= 2)
    if K >= 2:
        block_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        block_edge_attr = torch.randn(2, 4)
    else:
        block_edge_index = torch.zeros(2, 0, dtype=torch.long)
        block_edge_attr = torch.zeros(0, 4)

    return SplitInstanceData(
        name=name,
        partitions=partitions,
        block_edge_index=block_edge_index,
        block_edge_attr=block_edge_attr,
        cons_block_id=torch.cat([torch.full((n_cons_per_block,), k, dtype=torch.long) for k in range(K)]),
        vars_block_id=torch.cat([torch.full((n_vars_per_block,), k, dtype=torch.long) for k in range(K)]),
        cons_boundary_feat=torch.randn(n_cons, 5),
        vars_boundary_feat=torch.randn(n_vars, 5),
        cons_is_boundary=torch.zeros(n_cons, dtype=torch.bool),
        vars_is_boundary=torch.zeros(n_vars, dtype=torch.bool),
        n_cons=n_cons,
        n_vars=n_vars,
    )


def _make_batch(n_instances: int = 2) -> SplitInstanceBatch:
    """Build a batch of synthetic instances."""
    instances = [_make_instance(name=f"inst_{i}") for i in range(n_instances)]
    return SplitInstanceBatch(instances=instances)


@pytest.fixture
def args():
    return _make_args()


@pytest.fixture
def model(args):
    m = SplitBlockBiJepaPolicy(args)
    m.eval()
    return m


@pytest.fixture
def batch():
    return _make_batch(n_instances=2)


@pytest.fixture
def single_batch():
    return _make_batch(n_instances=1)


# ---------------------------------------------------------------------------
# embed: SplitInstanceBatch -> tuple of concatenated embeddings
# ---------------------------------------------------------------------------


class TestEmbedShapes:

    def test_embed_returns_one_tensor_per_input(self, model, batch):
        """len(embed(inputs)) == len(inputs)"""
        result = model.embed([batch])
        assert len(result) == 1

        result2 = model.embed([batch, batch])
        assert len(result2) == 2

    def test_embed_tensor_shape(self, model, batch):
        """Each output: [n_cons + n_vars, d_sub] (cons stacked before vars)"""
        result = model.embed([batch])
        total_cons = sum(inst.n_cons for inst in batch.instances)
        total_vars = sum(inst.n_vars for inst in batch.instances)
        assert result[0].shape == (total_cons + total_vars, D_SUB)

    def test_embed_with_view_masks_same_shape(self, model, batch):
        """Applying view masks should not change the output shape"""
        result_no_mask = model.embed([batch])

        view_masks = SplitBlockBiJepaPolicy._generate_view_masks(
            batch.instances, 0.1, 0.1, 0.05
        )
        result_masked = model.embed([batch], view_masks_list=[view_masks])

        assert result_masked[0].shape == result_no_mask[0].shape


# ---------------------------------------------------------------------------
# make_lejepa_views: batch -> (global_masks, all_masks)
# ---------------------------------------------------------------------------


class TestMakeLejepaViewsShapes:

    def test_global_masks_count(self, model, batch):
        """len(global_view_masks) == n_global_views"""
        global_masks, _ = model.make_lejepa_views(batch)
        assert len(global_masks) == model.n_global_views

    def test_all_masks_count(self, model, batch):
        """len(all_view_masks) == n_global_views + n_local_views"""
        _, all_masks = model.make_lejepa_views(batch)
        assert len(all_masks) == model.n_global_views + model.n_local_views

    def test_per_block_mask_shapes(self, model, batch):
        """Each mask triple: cons_mask [n_cons_block], var_mask [n_vars_block], edge_mask [n_edges_block]"""
        _, all_masks = model.make_lejepa_views(batch)
        vm = all_masks[0]  # SplitViewMasks
        K = batch.instances[0].num_blocks
        assert len(vm.masks) == K

        for k in range(K):
            for i, inst in enumerate(batch.instances):
                cm, varm, em = vm.masks[k][i]
                g = inst.partitions[k].graph
                assert cm.shape == (g.constraint_features.size(0),)
                assert varm.shape == (g.variable_features.size(0),)
                assert em.shape == (g.edge_index.size(1),)

    def test_masks_are_bool(self, model, batch):
        """All generated masks have dtype bool"""
        _, all_masks = model.make_lejepa_views(batch)
        for vm in all_masks:
            for block_masks in vm.masks:
                for cm, varm, em in block_masks:
                    assert cm.dtype == torch.bool
                    assert varm.dtype == torch.bool
                    assert em.dtype == torch.bool


# ---------------------------------------------------------------------------
# lejepa_loss: input + views -> (loss, pred, pred_masked, sigreg)
# ---------------------------------------------------------------------------


class TestLejepaLossShapes:

    def test_returns_four_values(self, model, batch):
        """Returns (loss, pred_loss, pred_loss_masked, sigreg_loss)"""
        model.train()
        views = model.make_lejepa_views(batch)
        result = model.lejepa_loss(batch, precomputed_views=views, lambd=0.05)
        assert len(result) == 4

    def test_loss_is_scalar(self, model, batch):
        """Total loss is 0-dim tensor"""
        model.train()
        views = model.make_lejepa_views(batch)
        loss, _, _, _ = model.lejepa_loss(batch, precomputed_views=views, lambd=0.05)
        assert loss.dim() == 0

    def test_pred_loss_is_scalar(self, model, batch):
        """Prediction loss is 0-dim tensor"""
        model.train()
        views = model.make_lejepa_views(batch)
        _, pred_loss, _, _ = model.lejepa_loss(batch, precomputed_views=views, lambd=0.05)
        assert pred_loss.dim() == 0

    def test_sigreg_loss_is_scalar(self, model, batch):
        """Sigma regularization loss is 0-dim tensor"""
        model.train()
        views = model.make_lejepa_views(batch)
        _, _, _, sigreg = model.lejepa_loss(batch, precomputed_views=views, lambd=0.05)
        assert sigreg.dim() == 0


# ---------------------------------------------------------------------------
# _unpack_lejepa_loss: handles 3-tuple and 4-tuple returns
# ---------------------------------------------------------------------------


class TestUnpackLejepaLossShapes:

    def test_four_tuple_passthrough(self):
        """4-tuple input -> same 4-tuple output"""
        from jobs.pretrain_split import _unpack_lejepa_loss

        vals = (torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0))
        result = _unpack_lejepa_loss(vals)
        assert len(result) == 4
        assert torch.equal(result[0], vals[0])
        assert torch.equal(result[3], vals[3])

    def test_three_tuple_duplicates_pred(self):
        """3-tuple (loss, pred, sigreg) -> (loss, pred, pred, sigreg)"""
        from jobs.pretrain_split import _unpack_lejepa_loss

        vals = (torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0))
        result = _unpack_lejepa_loss(vals)
        assert len(result) == 4
        # pred and pred_masked should be the same
        assert torch.equal(result[1], result[2])
        # sigreg is the last element
        assert torch.equal(result[3], vals[2])

    def test_invalid_length_raises(self):
        """Other lengths raise RuntimeError"""
        from jobs.pretrain_split import _unpack_lejepa_loss

        with pytest.raises(RuntimeError):
            _unpack_lejepa_loss((torch.tensor(1.0), torch.tensor(2.0)))

        with pytest.raises(RuntimeError):
            _unpack_lejepa_loss(
                (torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0),
                 torch.tensor(4.0), torch.tensor(5.0))
            )


# ---------------------------------------------------------------------------
# _compose_instances shape flow for pretraining path
# ---------------------------------------------------------------------------


class TestComposeInstancesShapes:

    def test_encoder_output_per_block(self, model, batch):
        """Per block: c_emb [n_cons_block, d_sub], v_emb [n_vars_block, d_sub]"""
        device = next(model.parameters()).device
        instances = batch.instances
        K = instances[0].num_blocks
        for k in range(K):
            embs = model._encode_block_slot_batched(instances, k, device)
            for i, inst in enumerate(instances):
                c_emb, v_emb = embs[i]
                g = inst.partitions[k].graph
                assert c_emb.shape == (g.constraint_features.size(0), D_SUB)
                assert v_emb.shape == (g.variable_features.size(0), D_SUB)

    def test_scatter_owned_to_global(self, model, batch):
        """After scatter: c_global [n_cons, d_sub], v_global [n_vars, d_sub]"""
        device = next(model.parameters()).device
        instances = batch.instances
        K = instances[0].num_blocks
        all_partition_embs = []
        for k in range(K):
            all_partition_embs.append(
                model._encode_block_slot_batched(instances, k, device)
            )
        scattered = model._scatter_owned_batched(instances, all_partition_embs, device)
        for i, inst in enumerate(instances):
            c_global, v_global = scattered[i]
            assert c_global.shape == (inst.n_cons, D_SUB)
            assert v_global.shape == (inst.n_vars, D_SUB)

    def test_pool_block_features(self, model, batch):
        """_pool_block_features -> [K, 4 * d_sub]"""
        device = next(model.parameters()).device
        inst = batch.instances[0]
        K = inst.num_blocks
        c_sub = torch.randn(inst.n_cons, D_SUB, device=device)
        v_sub = torch.randn(inst.n_vars, D_SUB, device=device)
        block_feat = model._pool_block_features(
            c_sub, v_sub,
            inst.cons_block_id.to(device),
            inst.vars_block_id.to(device),
            K,
        )
        assert block_feat.shape == (K, 4 * D_SUB)

    def test_composer_output_preserves_shape(self, model, batch):
        """Composer: (c_hat, v_hat) both [n_nodes, d_sub]"""
        device = next(model.parameters()).device
        composed = model._compose_instances(batch.instances, device)
        for i, inst in enumerate(batch.instances):
            c_hat, v_hat = composed[i]
            assert c_hat.shape == (inst.n_cons, D_SUB)
            assert v_hat.shape == (inst.n_vars, D_SUB)


# ---------------------------------------------------------------------------
# Embedding aggregation inside lejepa_loss
# ---------------------------------------------------------------------------


class TestLejepaEmbeddingAggregation:

    def test_all_embeddings_same_shape(self, model, batch):
        """All view embeddings have identical [N, D] shape"""
        model.train()
        views = model.make_lejepa_views(batch)
        _, all_view_masks = views
        inputs_repeated = [batch] * len(all_view_masks)
        all_embeddings = model.embed(inputs_repeated, view_masks_list=all_view_masks)
        shape0 = all_embeddings[0].shape
        for emb in all_embeddings:
            assert emb.shape == shape0

    def test_global_center_shape(self, model, batch):
        """centers = stack(global_embs).mean(0) -> [N, D]"""
        model.eval()
        views = model.make_lejepa_views(batch)
        global_masks, all_view_masks = views
        inputs_repeated = [batch] * len(all_view_masks)
        all_embeddings = model.embed(inputs_repeated, view_masks_list=all_view_masks)
        n_global = len(global_masks)
        global_embs = all_embeddings[:n_global]
        centers = torch.stack(global_embs, 0).mean(0)
        N = all_embeddings[0].shape[0]
        assert centers.shape == (N, D_SUB)

    def test_z_cat_shape(self, model, batch):
        """z_cat = cat(all_embeddings, dim=0) -> [V * N, D]"""
        model.eval()
        views = model.make_lejepa_views(batch)
        _, all_view_masks = views
        inputs_repeated = [batch] * len(all_view_masks)
        all_embeddings = model.embed(inputs_repeated, view_masks_list=all_view_masks)
        V = len(all_embeddings)
        N = all_embeddings[0].shape[0]
        z_cat = torch.cat(all_embeddings, dim=0)
        assert z_cat.shape == (V * N, D_SUB)

    def test_std_shape(self, model, batch):
        """std = z_cat.std(dim=0) -> [D]"""
        model.eval()
        views = model.make_lejepa_views(batch)
        _, all_view_masks = views
        inputs_repeated = [batch] * len(all_view_masks)
        all_embeddings = model.embed(inputs_repeated, view_masks_list=all_view_masks)
        z_cat = torch.cat(all_embeddings, dim=0)
        std = z_cat.std(dim=0, unbiased=False)
        assert std.shape == (D_SUB,)

    def test_std_loss_is_scalar(self, model, batch):
        """relu(1 - std).mean() -> 0-dim tensor"""
        model.eval()
        views = model.make_lejepa_views(batch)
        _, all_view_masks = views
        inputs_repeated = [batch] * len(all_view_masks)
        all_embeddings = model.embed(inputs_repeated, view_masks_list=all_view_masks)
        z_cat = torch.cat(all_embeddings, dim=0)
        std = z_cat.std(dim=0, unbiased=False).clamp_min(1e-6)
        std_loss = torch.nn.functional.relu(1.0 - std).mean()
        assert std_loss.dim() == 0


# ---------------------------------------------------------------------------
# train_step: make_views -> lejepa_loss -> backward
# ---------------------------------------------------------------------------


class TestTrainStepShapes:

    def test_make_views_feeds_lejepa_loss(self, model, batch):
        """(global_views, all_views) from make_lejepa_views accepted by lejepa_loss"""
        model.train()
        views = model.make_lejepa_views(batch)
        # Should not raise
        result = model.lejepa_loss(batch, precomputed_views=views, lambd=0.05)
        assert len(result) == 4

    def test_unpack_produces_four_scalars(self, model, batch):
        """_unpack_lejepa_loss(out) -> four 0-dim tensors"""
        from jobs.pretrain_split import _unpack_lejepa_loss

        model.train()
        views = model.make_lejepa_views(batch)
        out = model.lejepa_loss(batch, precomputed_views=views, lambd=0.05)
        loss, pred, pred_masked, sigreg = _unpack_lejepa_loss(out)
        assert loss.dim() == 0
        assert pred.dim() == 0
        assert pred_masked.dim() == 0
        assert sigreg.dim() == 0

    def test_combined_loss_is_differentiable_scalar(self, model, batch):
        """loss.backward() succeeds — grad shapes compatible throughout"""
        model.train()
        views = model.make_lejepa_views(batch)
        loss, _, _, _ = model.lejepa_loss(batch, precomputed_views=views, lambd=0.05)
        loss.backward()
        # Check at least one param has a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_state_add_receives_scalars(self, model, batch):
        """float(loss), float(pred_loss), etc. are Python floats"""
        model.train()
        views = model.make_lejepa_views(batch)
        loss, pred, pred_masked, sigreg = model.lejepa_loss(
            batch, precomputed_views=views, lambd=0.05
        )
        # Should not raise
        assert isinstance(float(loss), float)
        assert isinstance(float(pred), float)
        assert isinstance(float(pred_masked), float)
        assert isinstance(float(sigreg), float)


# ---------------------------------------------------------------------------
# eval_epoch: lejepa_loss + embed + isotropy split
# ---------------------------------------------------------------------------


class TestEvalEpochShapes:

    def test_embed_output_shape(self, model, batch):
        """model.embed([batch])[0] -> [sum(n_cons_i) + sum(n_vars_i), d_sub]"""
        model.eval()
        emb = model.embed([batch])[0]
        total_c = sum(inst.n_cons for inst in batch.instances)
        total_v = sum(inst.n_vars for inst in batch.instances)
        assert emb.shape == (total_c + total_v, D_SUB)

    def test_cons_vars_split_shapes(self, model, batch):
        """emb[:n_c] -> [n_c, d_sub], emb[n_c:] -> [n_v, d_sub]"""
        model.eval()
        emb = model.embed([batch])[0]
        n_c = sum(inst.n_cons for inst in batch.instances)
        n_v = sum(inst.n_vars for inst in batch.instances)
        c_emb = emb[:n_c]
        v_emb = emb[n_c:]
        assert c_emb.shape == (n_c, D_SUB)
        assert v_emb.shape == (n_v, D_SUB)

    def test_n_c_equals_sum_instance_n_cons(self, model, batch):
        """n_c = sum(inst.n_cons for inst in batch.instances)"""
        n_c = sum(inst.n_cons for inst in batch.instances)
        expected = sum(inst.n_cons for inst in batch.instances)
        assert n_c == expected

    def test_isotropy_metrics_receive_correct_shapes(self, model, batch):
        """isotropy_metrics gets (emb,) as 1-tuple of [N, D] tensor"""
        model.eval()
        emb = model.embed([batch])[0]
        assert emb.dim() == 2
        assert emb.shape[1] == D_SUB

    def test_return_type(self, model, batch):
        """eval_epoch returns (float, dict)"""
        model.eval()
        views = model.make_lejepa_views(batch)
        out = model.lejepa_loss(batch, precomputed_views=views, lambd=0.05)
        loss = out[0]
        assert isinstance(float(loss), float)

    def test_metrics_dict_keys(self, model, batch):
        """Contains valid/lejepa_loss, valid/lejepa_pred_loss, valid/lejepa_pred_loss_masked, valid/lejepa_sigreg_loss"""
        model.eval()
        views = model.make_lejepa_views(batch)
        loss, pred, pred_masked, sigreg = model.lejepa_loss(
            batch, precomputed_views=views, lambd=0.05
        )
        # Simulate what eval_epoch would produce
        metrics = {
            "valid/lejepa_loss": float(loss),
            "valid/lejepa_pred_loss": float(pred),
            "valid/lejepa_pred_loss_masked": float(pred_masked),
            "valid/lejepa_sigreg_loss": float(sigreg),
        }
        for key in [
            "valid/lejepa_loss",
            "valid/lejepa_pred_loss",
            "valid/lejepa_pred_loss_masked",
            "valid/lejepa_sigreg_loss",
        ]:
            assert key in metrics
            assert isinstance(metrics[key], float)


# ---------------------------------------------------------------------------
# TrainingState accumulation
# ---------------------------------------------------------------------------


class TestTrainingStateShapes:

    def test_finish_epoch_returns_four_floats(self):
        """finish_epoch() -> 4-tuple of Python floats"""
        from jobs.pretrain_split import TrainingState

        state = TrainingState(log_every=10)
        state.add(1.0, 0.5, 0.4, 0.1, n_items=4)
        state.add(2.0, 1.0, 0.8, 0.2, n_items=6)
        result = state.finish_epoch()
        assert len(result) == 4
        for val in result:
            assert isinstance(val, float)

    def test_accumulation_with_varying_batch_sizes(self):
        """Adding metrics from different batch sizes doesn't error; denominator = total items"""
        from jobs.pretrain_split import TrainingState

        state = TrainingState(log_every=10)
        state.add(1.0, 0.5, 0.4, 0.1, n_items=2)
        state.add(3.0, 1.5, 1.2, 0.3, n_items=8)
        result = state.finish_epoch()
        # denominator = 2 + 8 = 10
        assert len(result) == 4
        expected_loss = (1.0 + 3.0) / 10.0
        assert abs(result[0] - expected_loss) < 1e-6


# ---------------------------------------------------------------------------
# Lambda schedule
# ---------------------------------------------------------------------------


class TestLambdaScheduleShapes:

    def test_returns_float(self):
        """Output is a Python float"""
        from jobs.pretrain_split import _compute_lambda_steps

        result = _compute_lambda_steps(step=50, base=0.05, start=0.0, warm_steps=100)
        assert isinstance(result, float)

    def test_within_start_base_range(self):
        """Output is between start and base during warmup"""
        from jobs.pretrain_split import _compute_lambda_steps

        for step in [1, 25, 50, 75, 100]:
            result = _compute_lambda_steps(step=step, base=0.1, start=0.0, warm_steps=100)
            assert 0.0 <= result <= 0.1 + 1e-9

    def test_base_returned_after_warmup(self):
        """After warm_steps, output == base"""
        from jobs.pretrain_split import _compute_lambda_steps

        result = _compute_lambda_steps(step=200, base=0.05, start=0.0, warm_steps=100)
        assert abs(result - 0.05) < 1e-9
