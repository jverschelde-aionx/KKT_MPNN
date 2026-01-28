import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import configargparse
import numpy as np
import torch
from loguru import logger
from sklearn import base
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

import wandb
from data.common import ProblemClass
from jobs.utils import (
    apply_overrides,
    build_dataloaders,
    compute_lambda,
    device_from_args,
    init_logging_and_dirs,
    set_all_seeds,
)
from metrics.isotropy import isotropy_metrics
from models.base import LeJepaEncoderModule
from models.gnn import GNNPolicy
from models.mlp import KKTNetMLP
from models.optimizer import make_optimizer, make_scheduler


class TrainingState:
    def __init__(self, log_every: int) -> None:
        self.log_every = log_every
        self.steps: int = 0
        self.trained_items: int = 0
        self.jepa_loss_sum: float = 0.0
        self.jepa_pred_loss_sum: float = 0.0
        self.jepa_sigreg_loss_sum: float = 0.0
        self.epoch: int = 0

    def get_step(self) -> int:
        return self.steps

    @property
    def should_log(self) -> bool:
        return self.steps % max(1, self.log_every) == 0

    def add_training_step(
        self, loss: float, loss_pred: float, loss_sigreg: float, num_items: int
    ) -> None:
        wandb.log({"training/step": self.steps}, step=self.steps)
        self.steps += 1
        self.trained_items += int(num_items)
        self.jepa_loss_sum += loss
        self.jepa_pred_loss_sum += loss_pred
        self.jepa_sigreg_loss_sum += loss_sigreg

    def finish_epoch(self) -> Tuple[float, float, float]:
        denom = max(1, self.trained_items)
        out = (
            self.jepa_loss_sum / denom,
            self.jepa_pred_loss_sum / denom,
            self.jepa_sigreg_loss_sum / denom,
        )
        self.trained_items = 0
        self.jepa_loss_sum = self.jepa_pred_loss_sum = self.jepa_sigreg_loss_sum = 0.0
        self.epoch += 1
        return out


def train(overrides: Optional[Mapping] = None) -> str:
    try:
        parser = configargparse.ArgumentParser(
            allow_abbrev=False,
            default_config_files=["configs/pretrain_node_gnn_gatv2_CA_10.yml"],
        )

        # Training
        t = parser.add_argument_group("training")
        t.add_argument("--devices", type=str, default="0")
        t.add_argument("--batch_size", type=int, default=8)
        t.add_argument("--use_bipartite_graphs", action="store_true")
        t.add_argument("--epochs", type=int, default=20)
        t.add_argument("--lr", type=float, default=1e-3)
        t.add_argument("--num_workers", type=int, default=0)
        t.add_argument("--seed", type=int, default=0)
        t.add_argument("--log_every", type=int, default=50)
        t.add_argument("--wandb_project", type=str, default="kkt_gnn_pretraining")
        t.add_argument("--experiments_dir", type=str, default="./experiments")

        t.add_argument(
            "--optimizer", type=str, choices=["adam", "adamw"], default="adam"
        )
        t.add_argument("--weight_decay", type=float, default=0.0)
        t.add_argument(
            "--scheduler",
            type=str,
            choices=["none", "cosine", "cosine_warmup", "onecycle"],
            default="none",
        )
        t.add_argument("--warmup_pct", type=float, default=0.0)
        t.add_argument("--min_lr_ratio", type=float, default=0.0)
        t.add_argument(
            "--early_stop_patience",
            type=int,
            default=1000,
            help="Stop if valid/lejepa_loss does not improve for this many epochs.",
        )
        t.add_argument(
            "--early_stop_min_delta",
            type=float,
            default=0.0,
            help="Minimum absolute improvement to reset patience.",
        )
        t.add_argument(
            "--lejepa_lambda_start",
            type=float,
            default=None,
            help="If set, linearly ramp LeJEPA λ from this value down/up to --lejepa_lambda.",
        )
        t.add_argument(
            "--lejepa_lambda_warm_epochs",
            type=int,
            default=10,
            help="Number of epochs to linearly warm-start λ (ignored if --lejepa_lambda_start is None).",
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
            "--sc_sizes",
            type=int,
            nargs="+",
            default=[10, 50, 100, 200, 500, 1000],
        )
        d.add_argument(
            "--cfl_sizes",
            type=int,
            nargs="+",
            default=[10, 50, 100, 200, 500, 1000],
        )

        d.add_argument(
            "--rnd_sizes",
            type=int,
            nargs="+",
            default=[10, 50, 100, 200, 500, 1000],
        )
        t.add_argument("--max_grad_norm", type=float, default=1.0)
        d.add_argument("--n_instances", type=int, default=35000)

        d.add_argument("--data_root", type=str, default="../data")
        d.add_argument(
            "--val_split",
            type=float,
            default=0.15,
            help="Validation split ratio (default: 0.15 for 70/15/15 train/val/test)",
        )

        args, _ = parser.parse_known_args()

        GNNPolicy.add_args(parser) if args.use_bipartite_graphs else KKTNetMLP.add_args(
            parser
        )

        args, _ = parser.parse_known_args()

        if overrides is not None:
            apply_overrides(args, overrides)

        if not (0.0 <= args.lejepa_lambda <= 1.0):
            raise ValueError(
                f"lejepa_lambda must be in [0, 1], got {args.lejepa_lambda}. "
                "Values > 1 flip the sign of L_pred and will cause divergence."
            )

        print("args:", args)

        set_all_seeds(args.seed)

        device = device_from_args(args)

        train_loader, valid_loader, test_loader, N_max, M_max = build_dataloaders(
            args, for_pretraining=True
        )

        train_data = train_loader.dataset
        valid_data = valid_loader.dataset

        model = (
            GNNPolicy(args).to(device)
            if args.use_bipartite_graphs
            else KKTNetMLP(args, M_max, N_max).to(device)
        )

        # Initialize wandb
        save_dir = init_logging_and_dirs(args, model)

        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

        logger.info(f"Trainable params: total={total}, encoder={enc}")

        optimizer = make_optimizer(model, args)
        steps_per_epoch = len(train_loader)
        scheduler = make_scheduler(optimizer, args, steps_per_epoch)

        training_state = TrainingState(log_every=args.log_every)

        logger.info(f"train size: {len(train_data)} | valid size: {len(valid_data)}")
        wandb.log(
            {"data/train_size": len(train_data), "data/valid_size": len(valid_data)}
        )

        best_val = np.inf

        # Train loop
        for epoch in range(1, args.epochs + 1):
            current_lambda = compute_lambda(
                epoch=epoch,
                base=args.lejepa_lambda,
                start=args.lejepa_lambda_start,
                warm_epochs=args.lejepa_lambda_warm_epochs,
            )
            train_jepa_loss, train_jepa_pred_loss, train_jepa_sigreg_loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                training_state=training_state,
                lejepa_lambda=current_lambda,
                std_loss_weight=args.lejepa_std_loss_weight,
                max_grad_norm=args.max_grad_norm,
            )

            val_loss, val_metrics = eval_epoch(
                model=model,
                loader=valid_loader,
                device=device,
                lejepa_lambda=current_lambda,
                std_loss_weight=args.lejepa_std_loss_weight,
            )

            log_dict = {
                "epoch": epoch,
                "train/lejepa_loss": train_jepa_loss,
                "train/lejepa_pred_loss": train_jepa_pred_loss,
                "train/lejepa_sigreg_loss": train_jepa_sigreg_loss,
                **val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
                "lejepa_lambda_used": current_lambda,
            }

            wandb.log(log_dict, step=training_state.steps)

            # Save checkpoints
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "last.pt")
            # Save encoder-only artifact for downstream B1/B2
            model.save_encoder(str(save_dir / "last_encoder.pt"))

            improved = val_loss < (best_val - args.early_stop_min_delta)

            if improved:
                best_val = float(val_loss)
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(ckpt, save_dir / "best.pt")
                model.save_encoder(str(save_dir / "best_encoder.pt"))
                wandb.run.summary["best_val_loss"] = best_val
                wandb.run.summary["best_epoch"] = best_epoch
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop_patience:
                    logger.info(
                        "Early stopping at epoch {:03d}: no val improvement for {} epochs "
                        "(best {:.6f} at epoch {:03d})",
                        epoch,
                        epochs_no_improve,
                        best_val,
                        best_epoch,
                    )
                    wandb.run.summary["early_stop"] = True
                    wandb.run.summary["early_stop_epoch"] = epoch
                    break

            logger.info(
                "Epoch {:03d} | train {:.4f} | valid {:.4f} (best {:.4f} @ {:03d})",
                epoch,
                train_jepa_loss,
                val_loss,
                best_val,
                best_epoch,
            )

        logger.info(
            "Finished pretraining. Best validation LeJEPA loss = {:.4f}", best_val
        )

    except Exception as e:
        logger.exception("Exception during pretraining: {}", e)
        raise
    finally:
        wandb.finish()

    return str(save_dir)


def train_epoch(
    model: LeJepaEncoderModule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    training_state: TrainingState,
    lejepa_lambda: float,
    std_loss_weight: float,
    max_grad_norm: Union[float, None] = None,
) -> Tuple[float, Optional[float]]:
    model.train()
    for batch in loader:
        base = batch["base"].to(device, non_blocking=True)

        global_views, all_views = model.make_lejepa_views(
            base,
        )

        loss, pred_loss, pred_loss_masked, sigreg_loss = model.lejepa_loss(
            input=base,
            precomputed_views=(global_views, all_views),
            lambd=lejepa_lambda,
            std_loss_weight=std_loss_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        training_state.add_training_step(
            float(loss),
            float(pred_loss),
            float(sigreg_loss),
            num_items=getattr(
                base,
                "num_graphs",
                getattr(base, "size", lambda: [1])()[0] if hasattr(base, "size") else 1,
            ),
        )

        # light, periodic logging only
        if training_state.should_log:
            wandb.log(
                {
                    "train/lejepa_loss_b": float(loss),
                    "train/lejepa_pred_loss_b": float(pred_loss),
                    "train/lejepa_pred_loss_masked_b": float(pred_loss_masked),
                    "train/lejepa_sigreg_loss_b": float(sigreg_loss),
                },
                step=training_state.get_step(),
            )

    return training_state.finish_epoch()


@torch.no_grad()
def eval_epoch(
    model: LeJepaEncoderModule,
    loader: DataLoader,
    device: str,
    lejepa_lambda: float,
    std_loss_weight: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss: float = 0.0
    total_pred_loss: float = 0.0
    total_pred_loss_masked: float = 0.0
    total_sigreg_loss: float = 0.0
    total_loss: float = 0.0
    n_batches: int = 0

    # Isotropy accumulation
    iso_sums: Dict[str, float] = {}
    iso_count: int = 0
    total_graphs: int = 0

    for batch in loader:
        n_batches += 1

        base = batch["base"].to(device, non_blocking=True)
        global_views, all_views = model.make_lejepa_views(
            base,
        )

        B = base.num_graphs
        total_graphs += B

        (
            loss_val,
            pred_loss_val,
            pred_loss_masked,
            sigreg_loss_val,
        ) = model.lejepa_loss(
            base,
            precomputed_views=(global_views, all_views),
            lambd=lejepa_lambda,
            std_loss_weight=std_loss_weight,
        )

        total_loss += float(loss_val) * B
        total_pred_loss += float(pred_loss_val) * B
        total_sigreg_loss += float(sigreg_loss_val) * B
        total_pred_loss_masked += float(pred_loss_masked) * B

        embeddings = model.embed([base])[0]

        n_c = base.constraint_features.size(0)  # sum of constraint nodes in this Batch
        constraint_embeddings = embeddings[:n_c]  # constraint embeddings [sum_m, D]
        variable_embeddings = embeddings[n_c:]  # variable embeddings   [sum_n, D]

        iso_all = isotropy_metrics((embeddings,), model.sigreg, prefix="valid/iso_all/")

        iso_constraints = {}
        iso_variables = {}

        # Safety: isotropy metrics usually need at least 2 samples
        if constraint_embeddings.size(0) >= 2:
            iso_constraints = isotropy_metrics(
                (constraint_embeddings,), model.sigreg, prefix="valid/iso_cons/"
            )
        if variable_embeddings.size(0) >= 2:
            iso_variables = isotropy_metrics(
                (variable_embeddings,), model.sigreg, prefix="valid/iso_var/"
            )

        # Merge + accumulate
        for k, v in {**iso_constraints, **iso_variables, **iso_all}.items():
            iso_sums[k] = iso_sums.get(k, 0.0) + float(v)

        iso_count += 1
    avg_loss = total_loss / max(1, total_graphs)
    avg_pred_loss = total_pred_loss / max(1, total_graphs)
    avg_sigreg_loss = total_sigreg_loss / max(1, total_graphs)
    avg_pred_loss_masked = total_pred_loss_masked / max(1, total_graphs)

    metrics: Dict[str, float] = {
        "valid/lejepa_loss": avg_loss,
        "valid/lejepa_pred_loss": avg_pred_loss,
        "valid/lejepa_pred_loss_masked": avg_pred_loss_masked,
        "valid/lejepa_sigreg_loss": avg_sigreg_loss,
    }
    if iso_count > 0:
        for k, v in iso_sums.items():
            metrics[k] = v / iso_count
    return avg_loss, metrics


if __name__ == "__main__":
    train()
