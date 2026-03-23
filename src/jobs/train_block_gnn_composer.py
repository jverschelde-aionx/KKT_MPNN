"""
Block GNN Composer Training
============================
Trains the BlockGNNComposer model using the downstream KKT loss.

Pipeline per instance:
1. Composer produces improved node embeddings: c_hat, v_hat
2. Apply frozen heads: x_hat = var_head(v_hat), lam_hat = lambda_act(cons_head(c_hat))
3. Compute kkt_loss(y_pred, A, b, c, mask_m, mask_n)

Pre-computed once (cached):
- METIS partition, halo subgraphs, frozen encoder outputs
- KKT matrices A, b, c from the LP
- Block graph and boundary features

Baselines evaluated:
1. Raw split/halo — original heads on raw subgraph embeddings
2. Local MLP — [h_sub, boundary_features], no block context
3. Pooled block-context MLP — [h_sub, r_block, boundary_features], no GNN
4. Block GNN composer — [h_sub, z_block, boundary_features], full GNN
5. Full unsplit GNNPolicy — forward() on the full graph
"""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import configargparse
import torch
from torch.nn.utils import clip_grad_norm_

from data.common import CONS_PAD, VARS_PAD
from data.datasets import right_pad
from data.generators import get_bipartite_graph
from models.block_composer import (
    BlockGNNComposer,
    _encode_subgraph,
    scatter_owned_embeddings,
)
from models.decomposition import (
    build_block_graph,
    build_halo_subgraphs,
    compute_block_features,
    compute_boundary_features,
    identify_boundary_nodes,
    n_splits_for,
    split_bipartite_graph_metis,
    validate_partition,
)
from models.gnn import GNNPolicy
from models.losses import kkt_loss
from models.optimizer import make_optimizer, make_scheduler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached instance data
# ---------------------------------------------------------------------------


@dataclass
class InstanceData:
    """Pre-computed data for one LP instance (all tensors on CPU)."""

    name: str

    # Subgraph embeddings (owned nodes only, global ordering)
    c_sub_global: torch.Tensor          # [N_cons_total, d]  (zeros for missing)
    v_sub_global: torch.Tensor          # [N_var_total, d]

    # Block graph
    block_features: torch.Tensor        # [K, d_block]
    block_edge_index: torch.Tensor      # [2, E_block]
    block_edge_attr: torch.Tensor       # [E_block, 4]

    # Per-node block assignment (global indexing)
    cons_block_id: torch.Tensor         # [N_cons_total] long
    vars_block_id: torch.Tensor         # [N_var_total] long

    # Boundary features (already log1p-normalised, global indexing)
    cons_boundary_feat: torch.Tensor    # [N_cons_total, d_boundary]
    vars_boundary_feat: torch.Tensor    # [N_var_total, d_boundary]

    # Boundary masks (global indexing)
    cons_is_boundary: torch.Tensor      # [N_cons_total] bool
    vars_is_boundary: torch.Tensor      # [N_var_total] bool

    # KKT loss data (B=1, pre-batched)
    A_dense: torch.Tensor               # [1, m_kkt, n]
    b_vec: torch.Tensor                 # [1, m_kkt]
    c_vec: torch.Tensor                 # [1, n]
    mask_m: torch.Tensor                # [1, m_kkt]
    mask_n: torch.Tensor                # [1, n]

    # Full graph features (for full-policy baseline eval)
    c_nodes_full: torch.Tensor          # [N_cons_total, d_feat]
    v_nodes_full: torch.Tensor          # [N_var_total, d_feat]
    edge_index_full: torch.Tensor       # [2, E]
    edge_attr_full: torch.Tensor        # [E, d_edge]


# ---------------------------------------------------------------------------
# Preprocessing (called once before training)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _prepare_instance(
    encoder: torch.nn.Module,
    lp_path: Path,
    halo_hops: int,
    max_subgraph_ratio: float,
    device: torch.device,
) -> InstanceData:
    """Pre-compute all frozen-encoder outputs for one LP instance.

    Returns an :class:`InstanceData` with all tensors on CPU.
    """
    # 1. Load bipartite graph
    A_sparse, _v_map, v_nodes, c_nodes, _b_vars, b_vec, c_vec = (
        get_bipartite_graph(lp_path)
    )
    edge_index = A_sparse.edge_index    # bipartite graph edges
    edge_attr = A_sparse.edge_attr
    n_cons = c_nodes.size(0)
    n_vars = v_nodes.size(0)

    # 2. KKT matrices (A in <= form, includes variable bounds)
    m_kkt = b_vec.size(0)
    A_dense = A_sparse.to_dense().unsqueeze(0)        # [1, m_kkt, n]
    b_kkt = b_vec.unsqueeze(0)                         # [1, m_kkt]
    c_kkt = c_vec.unsqueeze(0)                         # [1, n]
    mask_m = torch.ones(1, m_kkt)
    mask_n = torch.ones(1, n_vars)

    # 3. METIS partition
    max_subgraph_size = max(1, int((n_cons + n_vars) * max_subgraph_ratio))
    num_parts = n_splits_for(n_cons + n_vars, max_subgraph_size)
    specs = split_bipartite_graph_metis(
        c_nodes, v_nodes, edge_index, edge_attr, num_parts=num_parts,
    )
    validate_partition(specs, n_cons, n_vars)

    # Remap part_id → contiguous 0..K-1
    id_to_idx = {p.part_id: i for i, p in enumerate(specs)}

    # 4. Build halo subgraphs + encode each independently
    halo_sgs = build_halo_subgraphs(
        specs, c_nodes, v_nodes, edge_index, edge_attr, halo_hops=halo_hops,
    )
    partition_embs = []
    for sg in halo_sgs:
        c_emb, v_emb = _encode_subgraph(encoder, sg, device)
        partition_embs.append((c_emb, v_emb))

    # 5. Scatter to global indices
    c_sub_global, v_sub_global = scatter_owned_embeddings(
        specs, halo_sgs, partition_embs, n_cons, n_vars,
    )

    # 6. Per-node block assignment (global indexing)
    cons_block_id = torch.zeros(n_cons, dtype=torch.long)
    vars_block_id = torch.zeros(n_vars, dtype=torch.long)
    for p in specs:
        idx = id_to_idx[p.part_id]
        cons_block_id[p.owned_cons_ids] = idx
        vars_block_id[p.owned_var_ids] = idx

    # 7. Boundary features + masks (global indexing)
    cons_boundary_feat, vars_boundary_feat = compute_boundary_features(
        specs, edge_index, edge_attr, n_cons, n_vars,
    )
    cons_is_boundary, vars_is_boundary = identify_boundary_nodes(
        specs, edge_index, n_cons, n_vars,
    )

    # 8. Block features (owned-only pooling) + block graph
    block_features = compute_block_features(
        specs, c_sub_global, v_sub_global, include_metadata=False,
    )
    bg = build_block_graph(
        specs, edge_index, edge_attr, n_cons=n_cons, n_vars=n_vars,
    )

    return InstanceData(
        name=lp_path.stem,
        c_sub_global=c_sub_global,
        v_sub_global=v_sub_global,
        block_features=block_features,
        block_edge_index=bg.block_edge_index,
        block_edge_attr=bg.block_edge_attr,
        cons_block_id=cons_block_id,
        vars_block_id=vars_block_id,
        cons_boundary_feat=cons_boundary_feat,
        vars_boundary_feat=vars_boundary_feat,
        cons_is_boundary=cons_is_boundary,
        vars_is_boundary=vars_is_boundary,
        A_dense=A_dense,
        b_vec=b_kkt,
        c_vec=c_kkt,
        mask_m=mask_m,
        mask_n=mask_n,
        c_nodes_full=c_nodes,
        v_nodes_full=v_nodes,
        edge_index_full=edge_index,
        edge_attr_full=edge_attr,
    )


def preprocess_instances(
    encoder: torch.nn.Module,
    lp_paths: List[Path],
    halo_hops: int,
    max_subgraph_ratio: float,
    device: torch.device,
) -> List[InstanceData]:
    """Pre-compute all frozen-encoder outputs. Called once before training."""
    cached = []
    for i, lp_path in enumerate(lp_paths):
        logger.info(
            "Preprocessing [%d/%d]: %s", i + 1, len(lp_paths), lp_path.stem,
        )
        data = _prepare_instance(
            encoder, lp_path, halo_hops, max_subgraph_ratio, device,
        )
        cached.append(data)
    return cached


# ---------------------------------------------------------------------------
# Helpers: apply heads and compute KKT
# ---------------------------------------------------------------------------


def _apply_heads_and_kkt(
    policy: GNNPolicy,
    c_emb: torch.Tensor,
    v_emb: torch.Tensor,
    data: InstanceData,
    device: torch.device,
    args,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Apply frozen heads to embeddings, compute KKT loss.

    Parameters
    ----------
    c_emb : [n_cons, d] on device
    v_emb : [n_vars, d] on device

    Returns (loss, metrics_dict). Loss has grad if c_emb/v_emb have grad.
    """
    x_pred = policy.var_head(v_emb).squeeze(-1)               # [n_vars]
    lam_pred = policy.lambda_act(policy.cons_head(c_emb).squeeze(-1))  # [n_cons]

    # Pack to [1, n], [1, m_kkt] for kkt_loss
    y_pred = torch.cat([x_pred, lam_pred]).unsqueeze(0)  # [1, n + m_kkt]

    # NOTE: kkt_loss expects y_pred = [B, n + m] where first n cols are x, rest are lam.
    # But the KKT A matrix has m_kkt rows (expanded <= form) which != n_cons.
    # The heads produce lam of length n_cons, but KKT needs lam of length m_kkt.
    # We need to produce y_pred = [1, n_vars + m_kkt].
    # Since the heads produce n_cons lambdas (one per original constraint),
    # but KKT has m_kkt constraints (expanded <= form with variable bounds),
    # we pad lam_pred to m_kkt with zeros for the bound constraints.
    m_kkt = data.A_dense.size(1)
    n_cons = c_emb.size(0)

    if n_cons < m_kkt:
        lam_padded = torch.zeros(m_kkt, device=device)
        lam_padded[:n_cons] = lam_pred
    else:
        lam_padded = lam_pred[:m_kkt]

    y_pred = torch.cat([x_pred, lam_padded]).unsqueeze(0)  # [1, n_vars + m_kkt]

    loss, kkt_metrics = kkt_loss(
        y_pred=y_pred,
        A=data.A_dense.to(device),
        b=data.b_vec.to(device),
        c=data.c_vec.to(device),
        mask_m=data.mask_m.to(device),
        mask_n=data.mask_n.to(device),
        primal_weight=getattr(args, "primal_weight", 0.1),
        dual_weight=getattr(args, "dual_weight", 0.1),
        stationarity_weight=getattr(args, "stationarity_weight", 0.6),
        complementary_slackness_weight=getattr(args, "complementary_slackness_weight", 0.2),
    )

    metrics = {
        "kkt_loss": loss.item(),
        "primal": kkt_metrics["primal_feasibility"].item(),
        "dual": kkt_metrics["dual_feasibility"].item(),
        "stationarity": kkt_metrics["stationarity"].item(),
        "comp_slack": kkt_metrics["complementary_slackness"].item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------


def train_epoch(
    composer: BlockGNNComposer,
    policy: GNNPolicy,
    train_data: List[InstanceData],
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    args,
) -> Dict[str, float]:
    """One training epoch: iterate over cached instances."""
    composer.train()
    optimizer.zero_grad(set_to_none=True)

    accum: Dict[str, float] = {}
    n_instances = 0
    grad_accum = getattr(args, "grad_accum_steps", 1)

    for i, data in enumerate(train_data):
        # Composer forward (trainable)
        c_hat, v_hat = composer(
            block_features=data.block_features.to(device),
            block_edge_index=data.block_edge_index.to(device),
            block_edge_attr=data.block_edge_attr.to(device),
            cons_block_id=data.cons_block_id.to(device),
            vars_block_id=data.vars_block_id.to(device),
            c_sub_owned=data.c_sub_global.to(device),
            v_sub_owned=data.v_sub_global.to(device),
            cons_boundary_feat=data.cons_boundary_feat.to(device),
            vars_boundary_feat=data.vars_boundary_feat.to(device),
        )

        # Apply frozen heads + KKT loss (gradients flow through c_hat, v_hat)
        loss, metrics = _apply_heads_and_kkt(
            policy, c_hat, v_hat, data, device, args,
        )

        (loss / grad_accum).backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(train_data):
            max_norm = getattr(args, "max_grad_norm", 1.0)
            clip_grad_norm_(composer.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        n_instances += 1

    return {k: v / n_instances for k, v in accum.items()}


@torch.no_grad()
def eval_epoch(
    composer: BlockGNNComposer,
    policy: GNNPolicy,
    val_data: List[InstanceData],
    device: torch.device,
    args,
) -> Dict[str, float]:
    """Validation epoch with raw-halo and full-policy baselines."""
    composer.eval()

    accum: Dict[str, float] = {}
    n_instances = 0

    for data in val_data:
        # --- Composed embeddings ---
        c_hat, v_hat = composer(
            block_features=data.block_features.to(device),
            block_edge_index=data.block_edge_index.to(device),
            block_edge_attr=data.block_edge_attr.to(device),
            cons_block_id=data.cons_block_id.to(device),
            vars_block_id=data.vars_block_id.to(device),
            c_sub_owned=data.c_sub_global.to(device),
            v_sub_owned=data.v_sub_global.to(device),
            cons_boundary_feat=data.cons_boundary_feat.to(device),
            vars_boundary_feat=data.vars_boundary_feat.to(device),
        )
        _, composed_metrics = _apply_heads_and_kkt(
            policy, c_hat, v_hat, data, device, args,
        )
        metrics = {f"composed_{k}": v for k, v in composed_metrics.items()}

        # --- Raw split/halo baseline ---
        _, raw_metrics = _apply_heads_and_kkt(
            policy,
            data.c_sub_global.to(device),
            data.v_sub_global.to(device),
            data, device, args,
        )
        metrics.update({f"raw_{k}": v for k, v in raw_metrics.items()})

        # --- Full unsplit policy baseline ---
        c_padded = right_pad(data.c_nodes_full, CONS_PAD).to(device)
        v_padded = right_pad(data.v_nodes_full, VARS_PAD).to(device)
        ei = data.edge_index_full.to(device)
        ea = data.edge_attr_full.to(device)
        _, full_metrics = _apply_heads_and_kkt(
            policy,
            *policy.encoder.encode_nodes(c_padded, ei, ea, v_padded),
            data, device, args,
        )
        metrics.update({f"full_{k}": v for k, v in full_metrics.items()})

        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        n_instances += 1

    return {k: v / n_instances for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args():
    parser = configargparse.ArgParser(
        default_config_files=["configs/decomposition/train_block_composer.yml"],
    )
    parser.add_argument("-c", "--config", is_config_file=True, help="Config file path")

    g = parser.add_argument_group("block_composer")
    g.add_argument("--encoder_path", type=str, required=True,
                    help="Path to pretrained encoder checkpoint (best_encoder.pt)")
    g.add_argument("--model_path", type=str, required=True,
                    help="Path to full finetuned GNNPolicy checkpoint (best.pt)")
    g.add_argument("--lp_dir", type=str, required=True,
                    help="Directory of LP files for training")
    g.add_argument("--val_lp_dir", type=str, default="",
                    help="Directory of LP files for validation")
    g.add_argument("--halo_hops", type=int, default=0,
                    help="Halo expansion depth (default 0)")
    g.add_argument("--max_subgraph_ratio", type=float, default=0.2,
                    help="Max partition size as fraction of total nodes")
    g.add_argument("--d_z", type=int, default=128,
                    help="Block GNN output dimension")
    g.add_argument("--d_mlp_hidden", type=int, default=256,
                    help="Composer MLP hidden dimension")
    g.add_argument("--heads", type=int, default=4,
                    help="GATv2 attention heads")
    g.add_argument("--composer_dropout", type=float, default=0.05,
                    help="Dropout for composer model")
    g.add_argument("--use_block_context", type=int, default=1,
                    help="1=use block context, 0=local MLP baseline")
    g.add_argument("--use_block_gnn", type=int, default=1,
                    help="1=block GNN, 0=pooled block-context projection")

    # KKT loss weights (match finetune defaults)
    g.add_argument("--primal_weight", type=float, default=0.1)
    g.add_argument("--dual_weight", type=float, default=0.1)
    g.add_argument("--stationarity_weight", type=float, default=0.6)
    g.add_argument("--complementary_slackness_weight", type=float, default=0.2)

    # Training hyperparams
    g.add_argument("--optimizer", type=str, default="adam",
                    choices=["adam", "adamw"])
    g.add_argument("--weight_decay", type=float, default=0.0)
    g.add_argument("--lr", type=float, default=0.001)
    g.add_argument("--scheduler", type=str, default="cosine_warmup",
                    choices=["none", "cosine_warmup", "onecycle"])
    g.add_argument("--warmup_pct", type=float, default=0.05)
    g.add_argument("--min_lr_ratio", type=float, default=0.1)
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--grad_accum_steps", type=int, default=4)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--early_stop_patience", type=int, default=10)
    g.add_argument("--log_every", type=int, default=1)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default="cpu")
    g.add_argument("--save_dir", type=str, default="experiments/block_composer")
    g.add_argument("--output_csv", type=str, default="")

    # GNN model args for loading the checkpoint
    GNNPolicy.add_args(parser)

    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pretrained encoder (for subgraph encoding) ---
    logger.info("Loading pretrained encoder from %s", args.encoder_path)
    encoder_model = GNNPolicy(args).to(device)
    encoder_ckpt = torch.load(args.encoder_path, map_location="cpu")
    if "encoder" not in encoder_ckpt:
        raise RuntimeError(f"Missing 'encoder' key in {args.encoder_path}")
    encoder_model.encoder.load_state_dict(encoder_ckpt["encoder"])
    encoder_model.eval()
    for p in encoder_model.parameters():
        p.requires_grad_(False)
    encoder = encoder_model.encoder

    # --- Load full finetuned GNNPolicy (for heads + full baseline) ---
    logger.info("Loading finetuned GNNPolicy from %s", args.model_path)
    policy = GNNPolicy(args).to(device)
    ckpt = torch.load(args.model_path, map_location="cpu")
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    # --- Collect LP files ---
    train_dir = Path(args.lp_dir)
    train_paths = sorted(train_dir.glob("*.lp"))
    if not train_paths:
        raise ValueError(f"No .lp files found in {train_dir}")

    val_paths: List[Path] = []
    if args.val_lp_dir:
        val_dir = Path(args.val_lp_dir)
        val_paths = sorted(val_dir.glob("*.lp"))

    logger.info("Train instances: %d", len(train_paths))
    logger.info("Val instances:   %d", len(val_paths))
    logger.info("Halo hops:       %d", args.halo_hops)
    logger.info("Device:          %s", device)

    # --- Preprocess (cached, done once) ---
    logger.info("=== Preprocessing training instances ===")
    train_data = preprocess_instances(
        encoder, train_paths, args.halo_hops, args.max_subgraph_ratio, device,
    )

    val_data: List[InstanceData] = []
    if val_paths:
        logger.info("=== Preprocessing validation instances ===")
        val_data = preprocess_instances(
            encoder, val_paths, args.halo_hops, args.max_subgraph_ratio, device,
        )

    # --- Infer dimensions from first cached instance ---
    sample = train_data[0]
    d_sub = sample.c_sub_global.shape[-1]
    d_block = sample.block_features.shape[-1]
    d_boundary = sample.cons_boundary_feat.shape[-1]

    logger.info("d_sub=%d  d_block=%d  d_boundary=%d", d_sub, d_block, d_boundary)

    # --- Build composer model ---
    composer = BlockGNNComposer(
        d_sub=d_sub,
        d_block=d_block,
        d_z=args.d_z,
        d_boundary=d_boundary,
        d_mlp_hidden=args.d_mlp_hidden,
        heads=args.heads,
        dropout=args.composer_dropout,
        use_block_context=bool(args.use_block_context),
        use_block_gnn=bool(args.use_block_gnn),
    ).to(device)

    n_params = sum(p.numel() for p in composer.parameters() if p.requires_grad)
    logger.info("Composer trainable params: %d", n_params)
    logger.info("Model:\n%s", composer)

    # --- Optimizer / scheduler ---
    optimizer = make_optimizer(composer, args)
    steps_per_epoch = max(1, len(train_data) // max(1, args.grad_accum_steps))
    scheduler = make_scheduler(optimizer, args, steps_per_epoch)

    # --- Training loop ---
    best_val_score = float("inf")
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_data)

        train_metrics = train_epoch(
            composer, policy, train_data, optimizer, scheduler, device, args,
        )

        # Validation
        val_metrics: Dict[str, float] = {}
        val_score = float("inf")
        if val_data:
            val_metrics = eval_epoch(
                composer, policy, val_data, device, args,
            )
            val_score = val_metrics.get("composed_kkt_loss", float("inf"))

        # Logging
        if epoch % args.log_every == 0 or epoch == 1:
            msg = (
                f"Epoch {epoch:3d} | "
                f"train kkt={train_metrics.get('kkt_loss', 0):.6f}"
            )
            if val_metrics:
                msg += (
                    f" | val composed={val_metrics.get('composed_kkt_loss', 0):.6f}"
                    f" raw={val_metrics.get('raw_kkt_loss', 0):.6f}"
                    f" full={val_metrics.get('full_kkt_loss', 0):.6f}"
                )
            logger.info(msg)

        # Save history
        row: Dict[str, float] = {
            "epoch": float(epoch),
            **{f"train_{k}": v for k, v in train_metrics.items()},
        }
        if val_metrics:
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        # Early stopping + checkpointing (lower KKT loss is better)
        if val_data:
            if val_score < best_val_score:
                best_val_score = val_score
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model": composer.state_dict(),
                        "val_kkt_loss": val_score,
                    },
                    save_dir / "best_composer.pt",
                )
                logger.info(
                    "  Saved best_composer.pt (val_kkt=%.6f)", val_score,
                )
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        else:
            torch.save(
                {"epoch": epoch, "model": composer.state_dict()},
                save_dir / "last_composer.pt",
            )

    # --- Final evaluation ---
    logger.info("=== Final evaluation ===")

    best_path = save_dir / "best_composer.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location="cpu")
        composer.load_state_dict(best_ckpt["model"])
        logger.info("Loaded best_composer.pt from epoch %d", best_ckpt["epoch"])

    composer.eval()

    eval_data = val_data if val_data else train_data
    final_metrics = eval_epoch(composer, policy, eval_data, device, args)

    logger.info("--- KKT Loss Comparison ---")
    logger.info("  Full unsplit: %.6f", final_metrics.get("full_kkt_loss", 0))
    logger.info("  Composed:     %.6f", final_metrics.get("composed_kkt_loss", 0))
    logger.info("  Raw halo:     %.6f", final_metrics.get("raw_kkt_loss", 0))

    logger.info("--- KKT Components (composed) ---")
    logger.info("  primal=%.6f  dual=%.6f  stat=%.6f  comp=%.6f",
                final_metrics.get("composed_primal", 0),
                final_metrics.get("composed_dual", 0),
                final_metrics.get("composed_stationarity", 0),
                final_metrics.get("composed_comp_slack", 0))

    logger.info("--- KKT Components (raw halo) ---")
    logger.info("  primal=%.6f  dual=%.6f  stat=%.6f  comp=%.6f",
                final_metrics.get("raw_primal", 0),
                final_metrics.get("raw_dual", 0),
                final_metrics.get("raw_stationarity", 0),
                final_metrics.get("raw_comp_slack", 0))

    # Write CSV if requested
    if args.output_csv and history:
        csv_path = Path(args.output_csv)
        all_keys: List[str] = []
        for row in history:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(history)
        logger.info("Training history written to %s", csv_path)

    return final_metrics


if __name__ == "__main__":
    main()
