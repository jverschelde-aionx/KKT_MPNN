from __future__ import annotations

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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader

import wandb
from models.gnn_transformer import GNNTransformer
from models.losses import AdaptiveKKTLoss, KKTLoss
from models.policy_encoder import GraphDataset, PolicyEncoder, collate


@torch.no_grad()
def kkt_metrics(
    y_pred: torch.Tensor, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> Dict[str, float]:
    """
    Compute each KKT term separately for logging.
    Shapes: see KKTLoss.
    """
    B, m, n = A.size(0), A.size(1), A.size(2)
    x_hat, lam_hat = y_pred[:, :n], y_pred[:, n:]
    Ax_minus_b = torch.bmm(A, x_hat.unsqueeze(-1)).squeeze(-1) - b
    primal = torch.relu(Ax_minus_b).pow(2).mean(dim=1).mean().item()
    dual = torch.relu(-lam_hat).pow(2).mean(dim=1).mean().item()
    At_lambda = torch.bmm(A.transpose(1, 2), lam_hat.unsqueeze(-1)).squeeze(-1)
    c_exp = c if c.dim() == 2 else c.unsqueeze(0).expand(B, -1)
    station = (c_exp + At_lambda).pow(2).mean(dim=1).mean().item()
    comp = (lam_hat * Ax_minus_b).pow(2).mean(dim=1).mean().item()
    return {
        "primal": primal,
        "dual": dual,
        "stationarity": station,
        "compl_slack": comp,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: KKTLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch_graph, A, b, c in loader:
        batch_graph = batch_graph.to(device)
        A, b, c = A.to(device), b.to(device), c.to(device)

        optimizer.zero_grad()
        y_pred: torch.Tensor = model(batch_graph)  # (B, n+m)
        loss: torch.Tensor = criterion(y_pred, A, b, c)
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: KKTLoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    term_sums = {"primal": 0.0, "dual": 0.0, "stationarity": 0.0, "compl_slack": 0.0}

    for batch_graph, A, b, c in loader:
        batch_graph = batch_graph.to(device)
        A, b, c = A.to(device), b.to(device), c.to(device)

        y_pred = model(batch_graph)
        loss = criterion(y_pred, A, b, c)
        metrics = kkt_metrics(y_pred, A, b, c)

        total_loss += loss.item()
        for k in term_sums:
            term_sums[k] += metrics[k]
        n_batches += 1

    avg_metrics = {k: term_sums[k] / n_batches for k in term_sums}
    return total_loss / n_batches, avg_metrics


def main():
    parser = configargparse.ArgumentParser(
        allow_abbrev=False,
        description="[KKT] GNN‑Transformer",
    )

    parser.add_argument("--configs", is_config_file=True, required=False)

    # Model
    g = parser.add_argument_group("model")
    g.add_argument("--graph_pooling", type=str, default="mean")
    g.add_argument("--gnn_type", type=str, default="gcn")
    g.add_argument("--gnn_virtual_node", action="store_true")
    g.add_argument("--gnn_dropout", type=float, default=0.0)
    g.add_argument("--gnn_num_layer", type=int, default=5)
    g.add_argument("--gnn_emb_dim", type=int, default=300)
    g.add_argument("--gnn_JK", type=str, default="last")
    g.add_argument("--gnn_residual", action="store_true")

    # Training
    t = parser.add_argument_group("training")
    t.add_argument("--devices", type=str, default="0")
    t.add_argument("--batch_size", type=int, default=64)
    t.add_argument("--eval_batch_size", type=int, default=None)
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
    t.add_argument("--seed", type=int, default=0)

    # Data
    parser.add_argument("--dataset", type=str, default="IS")
    parser.add_argument("--data_dir", type=str, default="../data/dataset")

    # Loss
    parser.add_argument("--kkt_w_primal", type=float, default=0.1)
    parser.add_argument("--kkt_w_dual", type=float, default=0.1)
    parser.add_argument("--kkt_w_station", type=float, default=0.6)
    parser.add_argument("--kkt_w_comp", type=float, default=0.2)

    args = parser.parse_known_args()

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
        f"{args.dataset}"
        f"+GNNT={args.gnn_type}"
        f"+emb={args.gnn_emb_dim}"
        f"+lr={args.lr}"
        f"+loss={args.loss}"
        f"+{datetime.now().strftime('%m%d-%H%M%S')}"
    )
    wandb.init(project="kkt_transformer", name=run_name, config=vars(args))

    # Setup data
    dir_bg = Path(args.data_dir) / args.dataset / "BG"
    train_files = sorted((dir_bg / "train").iterdir())
    valid_files = sorted((dir_bg / "valid").iterdir())

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

    # Retrieve problem instance dimensions (m,n) from the first sample
    _, A0, b0, c0 = next(iter(valid_loader))
    m, n = A0.size(1), A0.size(2)

    # Model, loss, optimiser, scheduler
    node_encoder = PolicyEncoder(args)
    model = GNNTransformer(
        num_tasks=m + n,
        args=args,
        gnn_node=node_encoder,
    ).to(device)

    logger.info("Model parameters: {}", model.count_parameters())
    wandb.watch(model)

    loss_fn = (
        KKTLoss(
            m,
            n,
            w_primal=args.kkt_w_primal,
            w_dual=args.kkt_w_dual,
            w_station=args.kkt_w_station,
            w_comp=args.kkt_w_comp,
        )
        if args.loss == "kkt"
        else AdaptiveKKTLoss(m, n)
    )

    criterion = loss_fn.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
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
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=args.grad_clip,
        )

        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        val_loss, val_terms = eval_epoch(model, valid_loader, criterion, device)

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
    main()
