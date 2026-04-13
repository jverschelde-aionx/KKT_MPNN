"""Tests for jobs.finetune_split — tensor shape checkpoints through the pipeline."""

from __future__ import annotations

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
# Fixtures: synthetic data + model
# ---------------------------------------------------------------------------

D_SUB = 16  # small embedding for fast tests
N_CONS_PER_BLOCK = 4
N_VARS_PER_BLOCK = 6
N_EDGES_PER_BLOCK = 8
K = 2  # number of blocks


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
        num_blocks=K,
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
    n_cons_per_block: int = N_CONS_PER_BLOCK,
    n_vars_per_block: int = N_VARS_PER_BLOCK,
    n_edges_per_block: int = N_EDGES_PER_BLOCK,
    num_blocks: int = K,
    name: str = "test_inst",
) -> SplitInstanceData:
    """Build a synthetic SplitInstanceData with KKT data attached."""
    n_cons = n_cons_per_block * num_blocks
    n_vars = n_vars_per_block * num_blocks

    partitions = []
    for k in range(num_blocks):
        c_off = k * n_cons_per_block
        v_off = k * n_vars_per_block
        part = _make_partition(k, n_cons_per_block, n_vars_per_block, n_edges_per_block)
        part.orig_cons_ids = torch.arange(c_off, c_off + n_cons_per_block)
        part.orig_var_ids = torch.arange(v_off, v_off + n_vars_per_block)
        partitions.append(part)

    if num_blocks >= 2:
        block_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        block_edge_attr = torch.randn(2, 4)
    else:
        block_edge_index = torch.zeros(2, 0, dtype=torch.long)
        block_edge_attr = torch.zeros(0, 4)

    # Attach KKT data (A, b, c)
    A_dense = torch.randn(n_cons, n_vars)
    b_vec = torch.randn(n_cons)
    c_vec = torch.randn(n_vars)

    return SplitInstanceData(
        name=name,
        partitions=partitions,
        block_edge_index=block_edge_index,
        block_edge_attr=block_edge_attr,
        cons_block_id=torch.cat([torch.full((n_cons_per_block,), k, dtype=torch.long) for k in range(num_blocks)]),
        vars_block_id=torch.cat([torch.full((n_vars_per_block,), k, dtype=torch.long) for k in range(num_blocks)]),
        cons_boundary_feat=torch.randn(n_cons, 5),
        vars_boundary_feat=torch.randn(n_vars, 5),
        cons_is_boundary=torch.zeros(n_cons, dtype=torch.bool),
        vars_is_boundary=torch.zeros(n_vars, dtype=torch.bool),
        n_cons=n_cons,
        n_vars=n_vars,
        A_dense=A_dense,
        b_vec=b_vec,
        c_vec=c_vec,
    )


def _make_batch(n_instances: int = 2) -> SplitInstanceBatch:
    return SplitInstanceBatch(
        instances=[_make_instance(name=f"inst_{i}") for i in range(n_instances)]
    )


def _make_heterogeneous_batch() -> SplitInstanceBatch:
    """Batch where instances have different n_vars / n_cons but same num_blocks."""
    inst1 = _make_instance(n_cons_per_block=3, n_vars_per_block=5, name="small")
    inst2 = _make_instance(n_cons_per_block=6, n_vars_per_block=8, name="large")
    return SplitInstanceBatch(instances=[inst1, inst2])


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


@pytest.fixture
def device(model):
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# _predict_batch: SplitInstanceBatch -> dense padded tensors
# ---------------------------------------------------------------------------


class TestPredictBatchShapes:

    def test_y_pred_shape_is_B_by_nmax_plus_mmax(self, model, batch, device):
        """y_pred: [B, n_max + m_max]"""
        from jobs.finetune_split import _predict_batch

        y, A, b, c, mask_m, mask_n, names = _predict_batch(model, batch, device)
        B = len(batch.instances)
        n_max = max(inst.n_vars for inst in batch.instances)
        m_max = max(inst.n_cons for inst in batch.instances)
        assert y.shape == (B, n_max + m_max)

    def test_A_shape_is_B_mmax_nmax(self, model, batch, device):
        """A: [B, m_max, n_max]"""
        from jobs.finetune_split import _predict_batch

        _, A, _, _, _, _, _ = _predict_batch(model, batch, device)
        B = len(batch.instances)
        n_max = max(inst.n_vars for inst in batch.instances)
        m_max = max(inst.n_cons for inst in batch.instances)
        assert A.shape == (B, m_max, n_max)

    def test_b_and_c_shapes(self, model, batch, device):
        """b: [B, m_max], c: [B, n_max]"""
        from jobs.finetune_split import _predict_batch

        _, _, b, c, _, _, _ = _predict_batch(model, batch, device)
        B = len(batch.instances)
        n_max = max(inst.n_vars for inst in batch.instances)
        m_max = max(inst.n_cons for inst in batch.instances)
        assert b.shape == (B, m_max)
        assert c.shape == (B, n_max)

    def test_masks_shape_and_dtype(self, model, batch, device):
        """mask_m: [B, m_max] bool, mask_n: [B, n_max] bool"""
        from jobs.finetune_split import _predict_batch

        _, _, _, _, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        B = len(batch.instances)
        n_max = max(inst.n_vars for inst in batch.instances)
        m_max = max(inst.n_cons for inst in batch.instances)
        assert mask_m.shape == (B, m_max)
        assert mask_n.shape == (B, n_max)
        assert mask_m.dtype == torch.bool
        assert mask_n.dtype == torch.bool

    def test_masks_sum_matches_instance_sizes(self, model, batch, device):
        """mask_m[i].sum() == n_cons_i, mask_n[i].sum() == n_vars_i"""
        from jobs.finetune_split import _predict_batch

        _, _, _, _, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        for i, inst in enumerate(batch.instances):
            assert mask_m[i].sum().item() == inst.n_cons
            assert mask_n[i].sum().item() == inst.n_vars

    def test_padding_regions_are_zero(self, model, batch, device):
        """A, b, c are zero where masks are False"""
        from jobs.finetune_split import _predict_batch

        _, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        for i, inst in enumerate(batch.instances):
            n_vars = inst.n_vars
            n_cons = inst.n_cons
            n_max = A.shape[2]
            m_max = A.shape[1]
            # Padding region for b
            if n_cons < m_max:
                assert b[i, n_cons:].abs().sum().item() == 0.0
            # Padding region for c
            if n_vars < n_max:
                assert c[i, n_vars:].abs().sum().item() == 0.0
            # Padding region for A (rows beyond n_cons, cols beyond n_vars)
            if n_cons < m_max:
                assert A[i, n_cons:, :].abs().sum().item() == 0.0
            if n_vars < n_max:
                assert A[i, :, n_vars:].abs().sum().item() == 0.0

    def test_x_lambda_slicing_shapes(self, model, batch, device):
        """x_pred = y[:, :n_max] -> [B, n_max], lambda = y[:, n_max:] -> [B, m_max]"""
        from jobs.finetune_split import _predict_batch

        y, A, _, _, _, _, _ = _predict_batch(model, batch, device)
        B = y.shape[0]
        n_max = A.shape[2]
        m_max = A.shape[1]
        x_pred = y[:, :n_max]
        lambda_pred = y[:, n_max:]
        assert x_pred.shape == (B, n_max)
        assert lambda_pred.shape == (B, m_max)


# ---------------------------------------------------------------------------
# predict_instance: SplitInstanceData -> (x, lambda)
# ---------------------------------------------------------------------------


class TestPredictInstanceShapes:

    def test_x_shape_matches_n_vars(self, model, batch):
        """x: [n_vars] (1-D, not [n_vars, 1])"""
        inst = batch.instances[0]
        x, _ = model.predict_instance(inst)
        assert x.shape == (inst.n_vars,)
        assert x.dim() == 1

    def test_lambda_shape_matches_n_cons(self, model, batch):
        """lambda: [n_cons]"""
        inst = batch.instances[0]
        _, lam = model.predict_instance(inst)
        assert lam.shape == (inst.n_cons,)
        assert lam.dim() == 1

    def test_lambda_nonnegative(self, model, batch):
        """Softplus on cons_head -> lambda >= 0"""
        inst = batch.instances[0]
        _, lam = model.predict_instance(inst)
        assert (lam >= 0).all()


# ---------------------------------------------------------------------------
# _compose_instances: encoder -> scatter -> composer shape flow
# ---------------------------------------------------------------------------


class TestComposeInstancesShapes:

    def test_encoder_output_per_block(self, model, batch, device):
        """Per block: c_emb [n_cons_block, d_sub], v_emb [n_vars_block, d_sub]"""
        instances = batch.instances
        for k_idx in range(K):
            embs = model._encode_block_slot_batched(instances, k_idx, device)
            for i, inst in enumerate(instances):
                c_emb, v_emb = embs[i]
                g = inst.partitions[k_idx].graph
                assert c_emb.shape == (g.constraint_features.size(0), D_SUB)
                assert v_emb.shape == (g.variable_features.size(0), D_SUB)

    def test_scatter_owned_to_global(self, model, batch, device):
        """After scatter: c_global [n_cons, d_sub], v_global [n_vars, d_sub]"""
        instances = batch.instances
        all_partition_embs = []
        for k_idx in range(K):
            all_partition_embs.append(
                model._encode_block_slot_batched(instances, k_idx, device)
            )
        scattered = model._scatter_owned_batched(instances, all_partition_embs, device)
        for i, inst in enumerate(instances):
            c_global, v_global = scattered[i]
            assert c_global.shape == (inst.n_cons, D_SUB)
            assert v_global.shape == (inst.n_vars, D_SUB)

    def test_pool_block_features(self, model, batch, device):
        """_pool_block_features -> [K, 4 * d_sub]"""
        inst = batch.instances[0]
        c_sub = torch.randn(inst.n_cons, D_SUB, device=device)
        v_sub = torch.randn(inst.n_vars, D_SUB, device=device)
        block_feat = model._pool_block_features(
            c_sub, v_sub,
            inst.cons_block_id.to(device),
            inst.vars_block_id.to(device),
            K,
        )
        assert block_feat.shape == (K, 4 * D_SUB)

    def test_composer_output_preserves_shape(self, model, batch, device):
        """Composer: (c_hat, v_hat) both [n_nodes, d_sub], same as input"""
        composed = model._compose_instances(batch.instances, device)
        for i, inst in enumerate(batch.instances):
            c_hat, v_hat = composed[i]
            assert c_hat.shape == (inst.n_cons, D_SUB)
            assert v_hat.shape == (inst.n_vars, D_SUB)

    def test_compose_list_length_matches_batch(self, model, batch, device):
        """Returns one (c_hat, v_hat) tuple per instance in the batch"""
        composed = model._compose_instances(batch.instances, device)
        assert len(composed) == len(batch.instances)


# ---------------------------------------------------------------------------
# train_step: _predict_batch -> kkt -> surrogate -> backward
# ---------------------------------------------------------------------------


class TestTrainStepShapes:

    def test_kkt_loss_is_scalar(self, model, batch, device):
        """kkt() returns 0-dim tensor before surrogate addition"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        model.train()
        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        loss, _ = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        assert loss.dim() == 0

    def test_surrogate_slicing_shape(self, model, batch, device):
        """y_pred[:, :n] produces [B, n] matching A.shape[2]"""
        from jobs.finetune_split import _predict_batch

        model.train()
        y, A, _, _, _, _, _ = _predict_batch(model, batch, device)
        n = A.shape[2]
        x_pred = y[:, :n]
        assert x_pred.shape == (A.shape[0], n)

    def test_combined_loss_is_scalar(self, model, batch, device):
        """loss = kkt_loss + surrogate_loss is still a 0-dim tensor"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt, surrogate_loss

        model.train()
        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        kkt_loss, _ = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        n = A.shape[2]
        surr_loss, _ = surrogate_loss(
            x_pred=y[:, :n], A=A, b=b, c=c,
            mask_m=mask_m, mask_n=mask_n,
            violation_weight=10.0, objective_weight=1.0,
        )
        combined = kkt_loss + surr_loss
        assert combined.dim() == 0

    def test_loss_is_differentiable(self, model, batch, device):
        """loss.backward() succeeds — grad shapes are compatible throughout"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        model.train()
        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        loss, _ = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_state_add_receives_scalars(self, model, batch, device):
        """metrics['kkt_loss'] * n_graphs is a Python float, not a tensor"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        model.eval()
        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        _, metrics = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        n_graphs = batch.num_graphs
        val = float(metrics["kkt_loss"]) * n_graphs
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# KKT loss input shape compatibility
# ---------------------------------------------------------------------------


class TestKKTShapeCompatibility:

    def test_A_times_x_bmm_compatible(self, model, batch, device):
        """bmm(A [B,m,n], x [B,n,1]) -> [B,m,1]"""
        from jobs.finetune_split import _predict_batch

        y, A, _, _, _, mask_n, _ = _predict_batch(model, batch, device)
        B, m, n = A.shape
        x = y[:, :n].unsqueeze(-1)  # [B, n, 1]
        result = torch.bmm(A, x)
        assert result.shape == (B, m, 1)

    def test_AT_times_lambda_bmm_compatible(self, model, batch, device):
        """bmm(A^T [B,n,m], lam [B,m,1]) -> [B,n,1]"""
        from jobs.finetune_split import _predict_batch

        y, A, _, _, _, _, _ = _predict_batch(model, batch, device)
        B, m, n = A.shape
        lam = y[:, n:n + m].unsqueeze(-1)  # [B, m, 1]
        result = torch.bmm(A.transpose(1, 2), lam)
        assert result.shape == (B, n, 1)

    def test_kkt_loss_is_scalar(self, model, batch, device):
        """kkt() returns 0-dim tensor"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        loss, _ = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        assert loss.dim() == 0

    def test_kkt_components_are_per_instance(self, model, batch, device):
        """primal, dual, stationarity, comp_slack each [B] before .mean()"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import (
            get_complementary_slackness,
            get_dual_feasibility,
            get_primal_feasibility,
            get_stationarity,
        )

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        B = y.shape[0]
        n = A.shape[2]
        x_pred = y[:, :n]
        lambda_pred = y[:, n:]

        primal = get_primal_feasibility(x_pred, A, b, mask_m)
        assert primal.shape == (B,)

        dual = get_dual_feasibility(lambda_pred, mask_m)
        assert dual.shape == (B,)

        stat = get_stationarity(lambda_pred, A, c, mask_n)
        assert stat.shape == (B,)

        comp = get_complementary_slackness(x_pred, lambda_pred, A, b, mask_m)
        assert comp.shape == (B,)


# ---------------------------------------------------------------------------
# Surrogate loss shape compatibility
# ---------------------------------------------------------------------------


class TestSurrogateLossShapes:

    def test_loss_is_scalar(self, model, batch, device):
        """Returns 0-dim tensor"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import surrogate_loss

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        n = A.shape[2]
        loss, _ = surrogate_loss(
            x_pred=y[:, :n], A=A, b=b, c=c,
            mask_m=mask_m, mask_n=mask_n,
            violation_weight=10.0, objective_weight=1.0,
        )
        assert loss.dim() == 0

    def test_metrics_are_per_instance(self, model, batch, device):
        """surrogate_viol, surrogate_obj, surrogate_int each [B]"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import surrogate_loss

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        B = y.shape[0]
        n = A.shape[2]
        _, metrics = surrogate_loss(
            x_pred=y[:, :n], A=A, b=b, c=c,
            mask_m=mask_m, mask_n=mask_n,
            violation_weight=10.0, objective_weight=1.0,
            integrality_weight=0.1,
        )
        for key in ["surrogate_viol", "surrogate_obj", "surrogate_int"]:
            assert metrics[key].shape == (B,)

    def test_Ax_intermediate_shape(self, model, batch, device):
        """bmm(A, x.unsqueeze(-1)).squeeze(-1) -> [B, m]"""
        from jobs.finetune_split import _predict_batch

        y, A, _, _, _, _, _ = _predict_batch(model, batch, device)
        B, m, n = A.shape
        x = y[:, :n]
        Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
        assert Ax.shape == (B, m)


# ---------------------------------------------------------------------------
# eval_epoch: per-batch shape flow
# ---------------------------------------------------------------------------


class TestEvalEpochShapes:

    def test_x_pred_lambda_pred_slicing(self, model, batch, device):
        """x_pred = y[:, :n_max] -> [B, n_max], lambda_pred = y[:, n_max:n_max+m_max] -> [B, m_max]"""
        from jobs.finetune_split import _predict_batch

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        n_max = c.shape[1]
        m_max = b.shape[1]
        x_pred = y[:, :n_max]
        lambda_pred = y[:, n_max:n_max + m_max]
        assert x_pred.shape == (y.shape[0], n_max)
        assert lambda_pred.shape == (y.shape[0], m_max)

    def test_individual_kkt_components_shape(self, model, batch, device):
        """get_primal_feasibility etc. each return [B]"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import (
            get_complementary_slackness,
            get_dual_feasibility,
            get_primal_feasibility,
            get_stationarity,
        )

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        B = y.shape[0]
        n_max = c.shape[1]
        m_max = b.shape[1]
        x_pred = y[:, :n_max]
        lambda_pred = y[:, n_max:n_max + m_max]

        assert get_primal_feasibility(x_pred, A, b, mask_m).shape == (B,)
        assert get_dual_feasibility(lambda_pred, mask_m).shape == (B,)
        assert get_stationarity(lambda_pred, A, c, mask_n).shape == (B,)
        assert get_complementary_slackness(x_pred, lambda_pred, A, b, mask_m).shape == (B,)

    def test_kkt_loss_per_instance_shape(self, model, batch, device):
        """weighted sum of components -> [B]"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import (
            get_complementary_slackness,
            get_dual_feasibility,
            get_primal_feasibility,
            get_stationarity,
        )

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        B = y.shape[0]
        n_max = c.shape[1]
        m_max = b.shape[1]
        x_pred = y[:, :n_max]
        lambda_pred = y[:, n_max:n_max + m_max]

        kkt_per = (
            0.1 * get_primal_feasibility(x_pred, A, b, mask_m)
            + 0.1 * get_dual_feasibility(lambda_pred, mask_m)
            + 0.6 * get_stationarity(lambda_pred, A, c, mask_n)
            + 0.2 * get_complementary_slackness(x_pred, lambda_pred, A, b, mask_m)
        )
        assert kkt_per.shape == (B,)

    def test_duality_gap_intermediates(self, model, batch, device):
        """primal_obj: [B], dual_obj: [B], opt_gap: [B]"""
        from jobs.finetune_split import _predict_batch

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        B = y.shape[0]
        n_max = c.shape[1]
        m_max = b.shape[1]
        x_pred = y[:, :n_max]
        lambda_pred = y[:, n_max:n_max + m_max]

        mask_n_f = mask_n.float()
        mask_m_f = mask_m.float()
        primal_obj = (x_pred * c * mask_n_f).sum(dim=1)
        dual_obj = -(lambda_pred * b * mask_m_f).sum(dim=1)
        opt_gap = (2.0 * (primal_obj - dual_obj).abs()) / (
            primal_obj.abs() + dual_obj.abs() + 1e-9
        )
        assert primal_obj.shape == (B,)
        assert dual_obj.shape == (B,)
        assert opt_gap.shape == (B,)

    def test_binary_feasibility_metrics_shapes(self, model, batch, device):
        """feasibility_rate, viol_sum, viol_max, penalised_obj each [B]"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import binary_feasibility_metrics

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        B = y.shape[0]
        n = A.shape[2]
        x_pred = y[:, :n]

        metrics = binary_feasibility_metrics(
            x_pred=x_pred, A=A, b=b, c=c,
            mask_m=mask_m, mask_n=mask_n, logits=False,
        )
        for key in ["feasibility_rate", "viol_sum", "viol_max", "penalised_obj"]:
            assert metrics[key].shape == (B,)

    def test_per_instance_unpadded_slicing(self, model, batch, device):
        """x_pred[i, :n_vars] -> [n_vars], c[i, :n_vars] -> [n_vars]"""
        from jobs.finetune_split import _predict_batch

        y, A, _, c, _, mask_n, _ = _predict_batch(model, batch, device)
        n_max = c.shape[1]
        x_pred = y[:, :n_max]

        for i, inst in enumerate(batch.instances):
            n_vars = inst.n_vars
            x_i = x_pred[i, :n_vars]
            c_i = c[i, :n_vars]
            assert x_i.shape == (n_vars,)
            assert c_i.shape == (n_vars,)

    def test_objective_gap_tensor_shape(self, model, batch, device):
        """torch.tensor(obj_gaps) passed to update_batch is 1-D"""
        obj_gaps = [0.1, 0.2, 0.05]
        t = torch.tensor(obj_gaps)
        assert t.dim() == 1
        assert t.shape == (3,)

    def test_output_dict_values_are_scalars(self, model, batch, device):
        """out['valid/...'] are Python floats, not tensors"""
        from jobs.finetune_split import _predict_batch
        from metrics.optimization import kkt

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, batch, device)
        loss, metrics = kkt(
            y_pred=y, A=A, b=b, c=c, mask_m=mask_m, mask_n=mask_n,
            primal_weight=0.1, dual_weight=0.1,
            stationarity_weight=0.6, complementary_slackness_weight=0.2,
        )
        # Simulate eval_epoch output
        out = {"valid/kkt_loss": float(loss)}
        for k, v in metrics.items():
            out[f"valid/{k}"] = float(v)
        for val in out.values():
            assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Batch padding consistency
# ---------------------------------------------------------------------------


class TestBatchPaddingConsistency:

    def test_heterogeneous_n_vars_padded_to_max(self, model, device):
        """All var-dim tensors use n_max = max(n_vars_i)"""
        from jobs.finetune_split import _predict_batch

        het_batch = _make_heterogeneous_batch()
        y, A, _, c, _, mask_n, _ = _predict_batch(model, het_batch, device)
        n_max = max(inst.n_vars for inst in het_batch.instances)
        assert c.shape[1] == n_max
        assert A.shape[2] == n_max
        assert mask_n.shape[1] == n_max

    def test_heterogeneous_n_cons_padded_to_max(self, model, device):
        """All cons-dim tensors use m_max = max(n_cons_i)"""
        from jobs.finetune_split import _predict_batch

        het_batch = _make_heterogeneous_batch()
        _, A, b, _, mask_m, _, _ = _predict_batch(model, het_batch, device)
        m_max = max(inst.n_cons for inst in het_batch.instances)
        assert b.shape[1] == m_max
        assert A.shape[1] == m_max
        assert mask_m.shape[1] == m_max

    def test_single_instance_no_extra_padding(self, model, single_batch, device):
        """B=1: n_max == n_vars, m_max == n_cons exactly"""
        from jobs.finetune_split import _predict_batch

        y, A, b, c, mask_m, mask_n, _ = _predict_batch(model, single_batch, device)
        inst = single_batch.instances[0]
        assert c.shape[1] == inst.n_vars
        assert b.shape[1] == inst.n_cons
        assert mask_n[0].sum().item() == inst.n_vars
        assert mask_m[0].sum().item() == inst.n_cons


# ---------------------------------------------------------------------------
# TrainingState accumulation
# ---------------------------------------------------------------------------


class TestTrainingStateShapes:

    def test_finish_round_returns_five_floats(self):
        """finish_round() -> 5-tuple of Python floats"""
        from jobs.finetune_split import TrainingState

        state = TrainingState(log_every=10)
        state.step(4)
        state.add(1.0, 0.5, 0.3, 0.8, 0.2)
        state.step(6)
        state.add(2.0, 1.0, 0.6, 1.6, 0.4)
        result = state.finish_round()
        assert len(result) == 5
        for val in result:
            assert isinstance(val, float)

    def test_accumulation_with_varying_batch_sizes(self):
        """Adding metrics from different batch sizes doesn't error; denominator = total items"""
        from jobs.finetune_split import TrainingState

        state = TrainingState(log_every=10)
        state.step(2)
        state.add(1.0, 0.5, 0.3, 0.8, 0.2)
        state.step(8)
        state.add(3.0, 1.5, 0.9, 2.4, 0.6)
        result = state.finish_round()
        # denominator = 2 + 8 = 10
        expected_kkt = (1.0 + 3.0) / 10.0
        assert abs(result[0] - expected_kkt) < 1e-6
