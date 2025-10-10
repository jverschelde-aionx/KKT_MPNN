from __future__ import annotations

import hashlib
import math
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.nn.functional import cosine_similarity, mse_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb
from instances.common import ProblemClass
from models.gnn_transformer import GNNTransformer
from models.losses import KKTLoss, kkt_metrics
from models.policy_encoder import (
    GNNPolicy,
    GraphDataset,
    PolicyEncoder,
    collate,
)
from models.utils import (
    get_dual_feasibility_violation,
    get_optimal_solution,
    get_optimality_gap,
    get_prediction_signature,
)


def is_better(val, best, min_delta=0.0):
    return (best - val) > min_delta


def apply_overrides(args, overrides: Mapping) -> None:
    if not overrides:
        return

    def apply_block(block: Mapping):
        for k, v in block.items():
            if isinstance(v, Mapping):
                apply_block(
                    v
                )  # recurse into nested groups (gnn_policy, transformer, training, data, etc.)
            else:
                if hasattr(args, k):
                    setattr(args, k, v)

    apply_block(overrides)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: KKTLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float | None = None,
    start_step: int = 0,
    log_every: int = 50,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[float, int]:
    model.train()
    total_loss, n_batches = 0.0, 0
    step = start_step

    for batch in loader:
        step_start_time = time.time()
        (
            batch_graph,
            sparse_A_matrices,
            b_padded,
            c_padded,
            b_mask,
            c_mask,
            num_constraints_per_graph,
            num_variables_per_graph,
            source_paths,
        ) = batch

        step += 1

        wandb.log({"training/step": step}, step=step)

        batch_graph = batch_graph.to(device)
        b_padded = b_padded.to(device)
        c_padded = c_padded.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler.is_enabled()):
            x_hat, lambda_hat = model(batch_graph)

        # --- DATA STATS (valid entries only) ---
        # Masked valid b, c
        device = x_hat.device

        # Ensure each sparse matrix lives on the right device (no-op if already there).
        sparse_A_matrices = [
            A if A.device == device else A.to(device, non_blocking=True)
            for A in sparse_A_matrices
        ]

        with autocast(enabled=scaler.is_enabled()):
            # matrix loss expects A_list; keep them on the right device
            sparse_A_matrices = [
                A if A.device == x_hat.device else A.to(x_hat.device, non_blocking=True)
                for A in sparse_A_matrices
            ]
            loss = criterion(
                x_hat,
                lambda_hat,
                sparse_A_matrices,
                b_padded,
                c_padded,
                num_constraints_per_graph,
                num_variables_per_graph,
            )

        # Term breakdown (extremely useful to see what diverges)
        if (step % log_every) == 0:
            kkt = kkt_metrics(
                x_hat,
                lambda_hat,
                sparse_A_matrices,
                b_padded,
                c_padded,
                num_constraints_per_graph,
                num_variables_per_graph,
            )
            wandb.log(
                {f"train_kkt/{name}": value for name, value in kkt.items()}, step=step
            )
            wandb.log({"train/loss": float(loss)}, step=step)

        # shape invariants
        assert x_hat.numel() == sum(num_variables_per_graph), (
            x_hat.shape,
            num_variables_per_graph,
        )
        assert lambda_hat.numel() == sum(num_constraints_per_graph), (
            lambda_hat.shape,
            num_constraints_per_graph,
        )
        if (step % log_every) == 0:
            with torch.no_grad():
                xhat = x_hat.detach().float()
                wandb.log(
                    {
                        "xhat/mean": float(xhat.mean()),
                        "xhat/std": float(xhat.std()),
                        "xhat/min": float(xhat.min()),
                        "xhat/max": float(xhat.max()),
                    },
                    step=step,
                )

        # ----- Backward -----
        scaler.scale(loss).backward()

        # Gradients (after unscale) + clipping
        scaler.unscale_(optimizer)
        if step % log_every == 0:
            head_grad = 0.0
            count = 0
            for n, p in model.named_parameters():
                if (("out" in n) or ("head" in n)) and (p.grad is not None):
                    head_grad += p.grad.detach().norm().item()
                    count += 1
            wandb.log(
                {
                    "debug/head_grad_l2_before": head_grad,
                    "debug/head_params_with_grad_before": count,
                },
                step=step,
            )
        total_gradient_l2 = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=math.inf
        )
        gradient_was_clipped = 0.0
        if grad_clip:
            gradient_was_clipped = float(total_gradient_l2 > grad_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        wandb.log(
            {
                "gradients/global_l2_norm": float(total_gradient_l2),
                "gradients/clip_threshold": float(grad_clip if grad_clip else 0.0),
                "gradients/was_clipped": gradient_was_clipped,
            },
            step=step,
        )

        # Optimizer + AMP
        previous_amp_scale = scaler.get_scale()

        if step % log_every == 0:
            head_grad = 0.0
            count = 0
            for n, p in model.named_parameters():
                if (("out" in n) or ("head" in n)) and (p.grad is not None):
                    head_grad += p.grad.detach().norm().item()
                    count += 1
            wandb.log(
                {
                    "debug/head_grad_l2_after": head_grad,
                    "debug/head_params_with_grad_after": count,
                },
                step=step,
            )

        scaler.step(optimizer)
        scaler.update()

        # Batch‑stepped schedulers
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        new_amp_scale = scaler.get_scale()
        log_dict = {}
        if new_amp_scale < previous_amp_scale:
            log_dict.update(
                {
                    "amp/scale": float(new_amp_scale),
                    "amp/scale_drop_ratio": float(
                        previous_amp_scale / max(new_amp_scale, 1e-12)
                    ),
                }
            )
        elif (step % log_every) == 0:
            log_dict.update({"amp/scale": float(new_amp_scale)})

        # Lightweight perf + memory every log_every
        if (step % log_every) == 0:
            log_dict.update(
                {
                    "optimizer/lr": optimizer.param_groups[0]["lr"],
                    "performance/step_time_ms": (time.time() - step_start_time)
                    * 1000.0,
                }
            )
            if torch.cuda.is_available():
                log_dict.update(
                    {
                        "cuda/memory_allocated_mb": torch.cuda.memory_allocated()
                        / (1024**2),
                        "cuda/memory_reserved_mb": torch.cuda.memory_reserved()
                        / (1024**2),
                        "cuda/max_memory_allocated_mb": torch.cuda.max_memory_allocated()
                        / (1024**2),
                    }
                )
        if log_dict:
            wandb.log(log_dict, step=step)

        total_loss += float(loss)
        n_batches += 1

    return total_loss / max(n_batches, 1), step


@torch.no_grad()
def eval_epoch(
    model, loader, criterion, device, amp=False
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    term_sums = {"primal": 0.0, "dual": 0.0, "stationarity": 0.0, "compl_slack": 0.0}

    objective_gap_sum = 0.0  # avg objective gap (relative to optimal)
    optimality_gap_sum = 0.0  # avg optimality gap (symmetric relative)
    # track dual feasibility violation to interpret dual bound
    dual_violation_l2_sum = 0.0
    dual_violation_linf_sum = 0.0

    cosine_similarity_sum = 0.0
    l2_distance_sum = 0.0
    total_items = 0

    for (
        batch_graph,
        A_list,
        b_pad,
        c_pad,
        b_mask,
        c_mask,
        m_sizes,
        n_sizes,
        sources,
    ) in loader:
        batch_graph = batch_graph.to(device)
        b_pad, c_pad = b_pad.to(device), c_pad.to(device)
        A_list = [A.to(device) if A.device != device else A for A in A_list]

        with autocast(enabled=amp):
            x_hat, lam_hat = model(batch_graph)

            loss = criterion(
                x_hat,
                lam_hat,
                A_list,
                b_pad,
                c_pad,
                m_sizes,
                n_sizes,
            )
            metrics = kkt_metrics(
                x_hat,
                lam_hat,
                A_list,
                b_pad,
                c_pad,
                m_sizes,
                n_sizes,
            )
        total_loss += loss.item()
        for k in term_sums:
            term_sums[k] += metrics[k]
        n_batches += 1

        # Move predictions to CPU for easy slicing
        x_hat_cpu = x_hat.detach().float().cpu()
        lambda_hat_cpu = lam_hat.detach().float().cpu()
        b_pad_cpu = b_pad.detach().float().cpu()
        c_pad_cpu = c_pad.detach().float().cpu()

        off_x = 0
        off_l = 0
        sigs = []
        items_in_batch = len(m_sizes)
        total_items += items_in_batch

        for i in range(items_in_batch):
            m_i, n_i = int(m_sizes[i]), int(n_sizes[i])
            x_i = x_hat_cpu[off_x : off_x + n_i]  # (n_i,)
            lambda_i = lambda_hat_cpu[off_l : off_l + m_i]  # (m_i,)
            off_x += n_i
            off_l += m_i
            A_i = A_list[i].cpu()
            c_i = c_pad_cpu[i, :n_i]  # (n_i,)
            b_i = b_pad_cpu[i, :m_i]  # (m_i,)

            sig = get_prediction_signature(x_i.numpy())
            sigs.append(sig)

            optimality_gap = get_optimality_gap(x_i, lambda_i, c_i, b_i)

            optimality_gap_sum += optimality_gap

            violation_l2, violation_linf = get_dual_feasibility_violation(
                A_i, lambda_i, c_i
            )
            dual_violation_l2_sum += violation_l2
            dual_violation_linf_sum += violation_linf

            bg_path = sources[i]

            optimal_solution, objective_gap = get_optimal_solution(
                bg_path=bg_path, x_i=x_i, c_i=c_i
            )

            objective_gap_sum += objective_gap
            cosine_similarity_sum += float(
                cosine_similarity(x_i, optimal_solution, dim=0).item()
            )
            l2_distance_sum += float(mse_loss(x_i, optimal_solution).item())

        if n_batches == 0:
            raise ValueError(
                "eval_epoch received an empty loader: no batches to evaluate."
            )

    uniq = len(set(sigs))
    wandb.log({"xhat/unique_signatures": uniq, "xhat/unique_frac": uniq / len(sigs)})

    validation_metrics = {k: term_sums[k] / n_batches for k in term_sums}
    validation_metrics["objective_gap"] = objective_gap_sum / float(total_items)
    validation_metrics["optimality_gap"] = optimality_gap_sum / float(total_items)
    validation_metrics["dual_violation_l2"] = dual_violation_l2_sum / float(total_items)
    validation_metrics["dual_violation_linf"] = dual_violation_linf_sum / float(
        total_items
    )
    validation_metrics["cosine_similarity"] = cosine_similarity_sum / float(total_items)
    validation_metrics["l2_distance"] = l2_distance_sum / float(total_items)

    validation_metrics = {f"valid/{k}": v for k, v in validation_metrics.items()}

    return total_loss / n_batches, validation_metrics


def train(overrides: Optional[Mapping] = None):
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description="[KKT] Train GNN-Transformer",
        default_config_files=["config.yml"],
    )

    # Add GNNPolicy specific arguments
    GNNPolicy.add_args(parser)

    # Add GNNTransformer specific arguments
    GNNTransformer.add_args(parser)

    # Training
    t = parser.add_argument_group("training")
    t.add_argument("--devices", type=str, default="0")
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--num_workers", type=int, default=0)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument(
        "--scheduler",
        choices=["plateau", "cosine", "onecycle", "none"],
        default="none",
    )
    t.add_argument("--pct_start", type=float, default=0.3)  # OneCycle
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (fp16/bf16)",
    )

    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--kkt_w_primal", type=float, default=0.1)
    t.add_argument("--kkt_w_dual", type=float, default=0.1)
    t.add_argument("--kkt_w_station", type=float, default=0.6)
    t.add_argument("--kkt_w_comp", type=float, default=0.2)
    t.add_argument("--max_lr", type=float, default=0.001)
    t.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Log lightweight scalars every N steps.",
    )
    t.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable epoch-level early stopping on valid/loss",
    )
    t.add_argument(
        "--es_patience",
        type=int,
        default=4,
        help="#epochs without improvement before stopping",
    )
    t.add_argument(
        "--es_min_delta",
        type=float,
        default=6,
        help="Minimum absolute improvement to reset patience",
    )
    t.add_argument(
        "--es_min_epochs",
        type=int,
        default=3,
        help="Do not consider early stop before this epoch",
    )
    t.add_argument(
        "--es_cooldown",
        type=int,
        default=1,
        help="#epochs to wait after an improvement before counting patience",
    )
    t.add_argument(
        "--es_restore_best",
        action="store_true",
        help="Load best.pt weights into the model on early stop",
    )
    t.add_argument(
        "--es_explosion_factor",
        type=float,
        default=None,
        help="If set, stop when valid/loss > factor * best_val",
    )

    # Data
    d = parser.add_argument_group("data")
    d.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=[ProblemClass.INDEPENDANT_SET, ProblemClass.COMBINATORIAL_AUCTION],
        help="Problem type",
    )
    d.add_argument(
        "--is_sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 200, 500, 1000],
    )
    d.add_argument(
        "--ca_sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 200, 500, 1000],
    )
    d.add_argument(
        "--n_instances", type=int, default=1000, help="Number of instances per size"
    )
    d.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Fraction of instances for the test set",
    )
    d.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of instances for the validation set",
    )
    d.add_argument("--data_root", type=str, default="../data")
    d.add_argument(
        "--solve", action="store_true", help="Run Gurobi to collect solution pools"
    )
    d.add_argument(
        "--gurobi_threads", type=int, default=4, help="Number of threads for Gurobi"
    )
    d.add_argument(
        "--n_jobs",
        type=int,
        default=32,
        help="Number of parallel jobs for data generation",
    )

    args, _ = parser.parse_known_args()

    if overrides is not None:
        apply_overrides(args, overrides)

    print("args:", args)
    if args.d_model > 200:
        args.batch_size = 16

    # Ensure reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and args.devices
        else torch.device("cpu")
    )

    # Initialize wandb
    run_name = (
        "kkt_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + GNNTransformer._get_name(args)
        + GNNPolicy._get_name(args)
    )

    if wandb.run is None:
        wandb.init(project="kkt_transformer", name=run_name, config=vars(args))

    wandb.define_metric("training/step")
    wandb.define_metric("*", step_metric="training/step")

    # Setup data
    train_files, valid_files = [], []

    for problem in args.problems:
        dir_bg = Path(args.data_root) / problem / "BG"
        for dir in (dir_bg / "train").iterdir():
            train_files.extend([str(dir / f) for f in os.listdir(dir)])
        for dir in (dir_bg / "val").iterdir():
            valid_files.extend([str(dir / f) for f in os.listdir(dir)])

    train_files = sorted(train_files)
    valid_files = sorted(valid_files)

    train_data = GraphDataset(train_files)
    valid_data = GraphDataset(valid_files)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    # Retrieve problem instance dimensions (m,n) from the first sample
    (batch_graph, A_list, _b, _c, _bm, _cm, m_sizes, n_sizes, sources) = next(
        iter(valid_loader)
    )
    m, n = m_sizes[0], n_sizes[0]
    try:
        # Model, loss, optimiser, scheduler
        node_encoder = PolicyEncoder(args)
        model = GNNTransformer(
            args=args,
            gnn_node=node_encoder,
        ).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            wandb.summary["status/oom_at"] = "model.to"
            logger.exception("CUDA OOM while moving model to device.")
            return  # finally block will release memory
        raise

    logger.info("Model parameters: {}", model.count_parameters())
    wandb.watch(model, log="gradients", log_graph=False)

    loss_fn = KKTLoss(
        m,
        n,
        w_primal=args.kkt_w_primal,
        w_dual=args.kkt_w_dual,
        w_stat=args.kkt_w_station,
        w_comp=args.kkt_w_comp,
    )

    criterion = loss_fn.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=args.amp)

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=(args.max_lr or args.lr),
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.pct_start,
        )
    else:
        scheduler = None

    # Create save directory
    save_dir = Path("exps") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    cooldown_left = 0

    logger.info(f"train size: {len(train_data)} | valid size: {len(valid_data)}")
    wandb.log({"data/train_size": len(train_data), "data/valid_size": len(valid_data)})

    # Train loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.epoch_callback(epoch)

        train_loss, global_step = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            start_step=global_step,
            log_every=args.log_every,
            scheduler=scheduler,
        )

        # Epoch-level schedulers
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        val_loss, validation_metrics = eval_epoch(
            model, valid_loader, criterion, device, args.amp
        )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        # logging
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **validation_metrics,
            },
            step=global_step,
        )

        # Model checkpointing
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")

        # track best
        improved = is_better(val_loss, best_val, args.es_min_delta)
        if improved:
            best_val = float(val_loss)
            best_epoch = epoch
            epochs_since_improve = 0
            cooldown_left = args.es_cooldown
            torch.save(ckpt, save_dir / "best.pt")
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_epoch"] = best_epoch
        else:
            if cooldown_left > 0:
                cooldown_left -= 1
            else:
                epochs_since_improve += 1

        # ---- early stopping explosion check ----
        if (
            args.early_stop
            and epoch >= args.es_min_epochs
            and args.es_explosion_factor
            and best_val < float("inf")
        ):
            if val_loss > args.es_explosion_factor * best_val:
                logger.info("Stopping early due to loss explosion")
                logger.info(val_loss, best_val)

                logger.info(
                    "Early stop due to loss explosion (val {:.4f} > {:.2f}× best {:.4f})",
                    val_loss,
                    args.es_explosion_factor,
                    best_val,
                )
                wandb.run.summary["early_stopped"] = True
                wandb.run.summary["early_stop_epoch"] = epoch
                break

        logger.info(
            "Epoch {:03d} | train {:.4f} | valid {:.4f} (best {:.4f})",
            epoch,
            train_loss,
            val_loss,
            best_val,
        )
        # ---- early stopping check ----
        if (
            args.early_stop
            and epoch >= args.es_min_epochs
            and epochs_since_improve >= args.es_patience
        ):
            logger.info(
                "Early stopping at epoch {:03d} (no improvement for {} epochs). Best {:.4f} @ {:03d}",
                epoch,
                epochs_since_improve,
                best_val,
                best_epoch,
            )
            wandb.run.summary["early_stopped"] = True
            wandb.run.summary["early_stop_epoch"] = epoch

            if args.es_restore_best and (save_dir / "best.pt").exists():
                state = torch.load(save_dir / "best.pt", map_location=device)
                model.load_state_dict(state["model"])
                logger.info("Restored model weights from best.pt")

            break  # exit training loop
    logger.info("Finished training. Best validation loss = {:.4f}", best_val)


if __name__ == "__main__":
    train()
