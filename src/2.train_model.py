from __future__ import annotations

import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb
from instances.common import COMBINATORIAL_AUCTION, INDEPENDANT_SET
from models.gnn_transformer import GNNTransformer
from models.losses import KKTLoss, kkt_metrics
from models.policy_encoder import GNNPolicy, GraphDataset, PolicyEncoder, collate


def register_nan_sentinel(model, log_first_only=True):
    tripped = {"flag": False}

    def hook(mod, inp, out):
        if tripped["flag"] and log_first_only:
            return

        def bad(x):
            return isinstance(x, torch.Tensor) and (
                torch.isnan(x).any() or torch.isinf(x).any()
            )

        bad_in = any(
            bad(t) for t in (inp if isinstance(inp, (tuple, list)) else (inp,))
        )
        bad_out = False
        if isinstance(out, (tuple, list)):
            bad_out = any(bad(t) for t in out)
        else:
            bad_out = bad(out)
        if bad_in or bad_out:
            tripped["flag"] = True
            wandb.log(
                {
                    "nan/module": 1,
                    "nan/mod_name": str(mod),
                    "nan/bad_input": float(bad_in),
                    "nan/bad_output": float(bad_out),
                }
            )

    for m in model.modules():
        m.register_forward_hook(hook)


def _finite_stats(t: torch.Tensor) -> dict:
    # Safely compute stats even if all values are non‑finite
    t_det = t.detach()
    finite = torch.isfinite(t_det)
    numel = t_det.numel()

    out = {
        "finite_frac": (finite.float().mean().item() if numel > 0 else 1.0),
        "has_nan": torch.isnan(t_det).any().item(),
        "has_inf": torch.isinf(t_det).any().item(),
    }
    if finite.any():
        t_f = t_det[finite]
        out.update(
            {
                "min": t_f.min().item(),
                "max": t_f.max().item(),
                "mean": t_f.mean().item(),
                "abs_max": t_f.abs().max().item(),
            }
        )
    else:
        out.update(
            {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "abs_max": float("nan"),
            }
        )
    return out


def log_tensor_stats(prefix: str, t: torch.Tensor, step: int, hist_every: int = 0):
    d = {f"{prefix}/{k}": v for k, v in _finite_stats(t).items()}
    wandb.log(d, step=step)
    if hist_every and (step % hist_every == 0):
        wandb.log(
            {f"{prefix}/hist": wandb.Histogram(t.detach().float().cpu().numpy())},
            step=step,
        )


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
        (
            batch_graph,
            A_list,
            b_pad,
            c_pad,
            b_mask,
            c_mask,
            m_sizes,
            n_sizes,
            sources,
        ) = batch

        step += 1

        batch_graph = batch_graph.to(device)
        b_pad = b_pad.to(device)
        c_pad = c_pad.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler.is_enabled()):
            x_hat, lam_hat = model(batch_graph)

        # --- DATA STATS (valid entries only) ---
        # Masked valid b, c
        device = x_hat.device

        # Ensure each sparse matrix lives on the right device (no-op if already there).
        A_list = [
            A if A.device == device else A.to(device, non_blocking=True) for A in A_list
        ]

        b_valid = b_pad[b_mask].detach()
        c_valid = c_pad[c_mask].detach()
        if (step % log_every) == 0:
            log_tensor_stats("data/b", b_valid, step)
            log_tensor_stats("data/c", c_valid, step)
            # Sparse A values across the micro‑batch
            A_vals = torch.cat([A.values() for A in A_list]).to(device=x_hat.device)
            log_tensor_stats("data/A_values", A_vals, step)

        # --- MODEL OUTPUT STATS ---
        log_tensor_stats("pred/x_hat", x_hat, step)
        log_tensor_stats("pred/lam_hat", lam_hat, step)
        if (step % log_every) == 0:
            # how many lambda are negative? (should be rare with the penalty)
            frac_neg = (lam_hat < 0).float().mean().item()
            wandb.log({"pred/lam_neg_frac": frac_neg}, step=step)

        with autocast(enabled=scaler.is_enabled()):
            loss = criterion(
                x_hat, lam_hat, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes
            )

        # shape invariants
        assert x_hat.numel() == sum(n_sizes), (x_hat.shape, n_sizes)
        assert lam_hat.numel() == sum(m_sizes), (lam_hat.shape, m_sizes)

        # If anything is non‑finite, dump extra info and skip/update
        if not torch.isfinite(loss):
            wandb.log({"alerts/nonfinite_loss": 1.0}, step=step)
            # Which part is broken?
            x_ok = torch.isfinite(x_hat).all().item()
            l_ok = torch.isfinite(lam_hat).all().item()
            b_ok = torch.isfinite(b_valid).all().item()
            c_ok = torch.isfinite(c_valid).all().item()
            wandb.log(
                {
                    "alerts/x_finite": float(x_ok),
                    "alerts/lam_finite": float(l_ok),
                    "alerts/b_finite": float(b_ok),
                    "alerts/c_finite": float(c_ok),
                },
                step=step,
            )
            wandb.log(
                {
                    "alerts/sources": wandb.Table(
                        data=[[s] for s in sources], columns=["file"]
                    )
                },
                step=step,
            )
            raise RuntimeError(f"Non‑finite loss at step {step}")

        scaler.scale(loss).backward()

        # --- GRADIENT STATS ---
        scaler.unscale_(optimizer)
        # total grad norm before clipping
        total_gnorm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=math.inf
        )
        wandb.log({"grad/total_l2": float(total_gnorm)}, step=step)

        if grad_clip:
            # your actual clip after unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Batch-wise schedulers like OneCycleLR must be stepped here:
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        new_scale = scaler.get_scale()
        if new_scale < prev_scale:
            # scaler skipped the step due to inf/nan in grads
            wandb.log(
                {
                    "amp/scale": float(new_scale),
                    "amp/scale_drop": float(prev_scale / max(new_scale, 1e-12)),
                },
                step=step,
            )
        elif (step % log_every) == 0:
            wandb.log({"amp/scale": float(new_scale)}, step=step)

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
                x_hat, lam_hat, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes
            )
        # metrics
        metrics = kkt_metrics(
            x_hat, lam_hat, A_list, b_pad, c_pad, b_mask, c_mask, m_sizes, n_sizes
        )

        total_loss += loss.item()
        for k in term_sums:
            term_sums[k] += metrics[k]
        n_batches += 1

    avg_metrics = {k: term_sums[k] / n_batches for k in term_sums}
    return total_loss / n_batches, avg_metrics


def train():
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description="[KKT] Train GNN-Transformer",
        default_config_files=["config.yml"],
    )
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    #     "garbage_collection_threshold:0.6,max_split_size_mb:128"
    # )
    # Add GNNPolicy specific arguments
    GNNPolicy.add_args(parser)

    # Add GNNTransformer specific arguments
    GNNTransformer.add_args(parser)

    # Model
    m = parser.add_argument_group("model")
    m.add_argument("--graph_pooling", type=str, default="mean")
    m.add_argument("--gnn_type", type=str, default="gcn")
    m.add_argument("--gnn_virtual_node", action="store_true")
    m.add_argument("--gnn_dropout", type=float, default=0.0)
    m.add_argument("--gnn_num_layer", type=int, default=5)
    m.add_argument("--gnn_emb_dim", type=int, default=300)
    m.add_argument("--gnn_JK", type=str, default="last")
    m.add_argument("--gnn_residual", action="store_true")
    m.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="maximum sequence length to predict (default: None)",
    )

    # Training
    t = parser.add_argument_group("training")
    t.add_argument("--devices", type=str, default="0")
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--eval_batch_size", type=int, default=4)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--num_workers", type=int, default=0)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--loss", choices=["kkt", "adaptive-kkt"], default="kkt")
    t.add_argument(
        "--scheduler", choices=["plateau", "cosine", "onecycle", "none"], default="none"
    )
    t.add_argument("--pct_start", type=float, default=0.3)  # OneCycle
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument(
        "--amp", action="store_true", help="Use automatic mixed precision (fp16/bf16)"
    )

    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--kkt_w_primal", type=float, default=0.1)
    t.add_argument("--kkt_w_dual", type=float, default=0.1)
    t.add_argument("--kkt_w_station", type=float, default=0.6)
    t.add_argument("--kkt_w_comp", type=float, default=0.2)

    # Data
    d = parser.add_argument_group("data")
    d.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=[INDEPENDANT_SET, COMBINATORIAL_AUCTION],
        help="Problem type",
    )
    d.add_argument(
        "--is_sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000],
    )
    d.add_argument(
        "--ca_sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000],
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
        f"{'-'.join(args.problems)}"
        f"+GNNT={args.gnn_type}"
        f"+emb={args.gnn_emb_dim}"
        f"+lr={args.lr}"
        f"+loss={args.loss}"
        f"+{datetime.now().strftime('%m%d-%H%M%S')}"
    )
    wandb.init(project="kkt_transformer", name=run_name, config=vars(args))

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

    eval_bs = args.batch_size if args.eval_batch_size is None else args.eval_batch_size

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print(args)

    # Retrieve problem instance dimensions (m,n) from the first sample
    (batch_graph, A_list, _b, _c, _bm, _cm, m_sizes, n_sizes, sources) = next(
        iter(valid_loader)
    )
    m, n = m_sizes[0], n_sizes[0]
    args.cons_nfeats = batch_graph.constraint_features.size(1)
    args.var_nfeats = batch_graph.variable_features.size(1)
    args.edge_nfeats = batch_graph.edge_attr.size(1)

    # Model, loss, optimiser, scheduler
    node_encoder = PolicyEncoder(args)
    model = GNNTransformer(
        args=args,
        gnn_node=node_encoder,
    ).to(device)

    if os.environ.get("NAN_SENTINEL", "0") == "1":
        register_nan_sentinel(model)

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
            max_lr=args.lr,
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

    # Train loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.epoch_callback(epoch)
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_clip=args.grad_clip,
            start_step=global_step,
            log_every=50,
            scheduler=scheduler,
        )

        # Epoch-level schedulers
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        val_loss, val_terms = eval_epoch(
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
                **{f"valid/{k}": v for k, v in val_terms.items()},
            }
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

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")
            wandb.run.summary["best_val_loss"] = best_val

        logger.info(
            "Epoch {:03d} | train {:.4f} | valid {:.4f} (best {:.4f})",
            epoch,
            train_loss,
            val_loss,
            best_val,
        )

    logger.info("Finished training. Best validation loss = {:.4f}", best_val)


if __name__ == "__main__":
    train()
